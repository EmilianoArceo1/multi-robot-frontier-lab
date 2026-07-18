"""
Simulation controller logic.

This is the main file to read for behavior. It contains the methods that
start/reset the simulation, assign goals/frontiers, request A*/Dijkstra
routes, update sensor mapping, run each simulation step, check robot-obstacle
and robot-robot safety, and compute metrics.

It is implemented as a mixin so the Qt MainWindow can keep UI construction
separate from simulation behavior without a risky rewrite of all state
references in one step.
"""

from __future__ import annotations

import inspect
import functools
import json
import logging
import math
import os
import time
import zlib
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QRunnable
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from robot import Robot

from robotics_sim.simulation.config import *
from robotics_sim.planning.exploration_planners import (
    DEFAULT_EXPLORATION_PLANNER,
    select_exploration_goal,
)
from robotics_sim.simulation.navigation_modes import (
    GOAL_SEEKING_PLANNER,
    is_goal_seeking_planner,
    is_exploration_planner,
)
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.simulation.runtime_robot_registry import RuntimeRobotRegistry
from robotics_sim.simulation.telemetry import TelemetryLogger
from robotics_sim.simulation.robot_trace import (
    MAX_OBSTACLE_SECTIONS,
    RobotTrace,
    group_obstacle_points_into_sections,
    slug_route_failure_reason,
)
from robotics_sim.environment.belief_map import FREE, OCCUPIED, UNKNOWN, BeliefMap
from robotics_sim.diagnostics.snapshot_export import (
    DEFAULT_AUTO_TARGET_ROWS,
    SnapshotExportError,
    export_navigation_snapshots_xlsx,
    select_navigation_snapshot_events,
)
from robotics_sim.planning.planning_costmap import apply_hazard_belief_to_planning_grid
from robotics_sim.planning.path_simplifier import line_of_sight_grid_safe
from robotics_sim.simulation.hazard_service import RuntimeHazardService
from robotics_sim.simulation.hazard_safety_runtime import HazardSafetyRuntime
from robotics_sim.simulation.perf_monitor import PerfMonitor
from robotics_sim.app.widgets import make_icon, SimulationMetricsWindow, SimulationConsoleWindow
from robotics_sim.app.render_perf import format_route_plan_perf_line
from robotics_interfaces.plugins import PluginMetadata, build_runtime_profile
from robotics_sim.planning.coordinated_frontier_planner import validate_multi_robot_corridor
from robotics_sim.simulation.coordination import (
    MultiRobotCoordinator,
    RobotCoordinationState,
    map_robot_commands_by_id,
    runtime_profile_for_strategy,
    select_runtime_control_source,
    select_runtime_path_source,
)
from robotics_sim.simulation.plugin_loader import PluginLoadError

_LOGGER = logging.getLogger(__name__)

try:
    from robotics_sim.planning.planner_registry import compute_planned_waypoints
except ImportError:
    compute_planned_waypoints = None

try:
    from robotics_sim.environment.collision_checker import (
        CollisionChecker,
        CollisionReport,
        RobotSnapshot,
    )
except ImportError:
    CollisionChecker = None
    CollisionReport = None
    RobotSnapshot = None

# New POO architecture — imported lazily inside methods to avoid circular deps.
# These imports are only used by the new build_observation / apply_navigation_decision
# / planner_services plumbing.  The existing simulation loop is unchanged.
try:
    from robotics_sim.simulation.observation import RobotObservation
    from robotics_sim.simulation.planner_services import PlannerServices
except ImportError:  # pragma: no cover
    RobotObservation = None  # type: ignore[assignment,misc]
    PlannerServices = None  # type: ignore[assignment]

from robotics_sim.navigation.navigation_supervisor import NavigationSupervisor
from robotics_sim.diagnostics.capture import (
    NavigationDebugCapture,
    PlanDebugCapture,
    clearance_terms_from_report,
)
from robotics_sim.diagnostics.event_log import (
    NavigationDebugEvent,
    NavigationDebugEventKind,
    NavigationDebugEventLog,
)
from robotics_sim.diagnostics.navigation_snapshot import (
    AgentStateDebug,
    BeliefMapDebug,
    ControllerDebug,
    FrontierDebug,
    HazardBeliefDebug,
    HazardDebug,
    HazardSourceDebug,
    Maybe,
    NavigationDebugSnapshot,
    PathDebug,
    PlanningGridDebug,
    Pose,
    PredictedMotionDebug,
    RouteValidationDebug,
    RuntimeMetricsDebug,
    SafetyDebug,
    SensorDebug,
)
from robotics_sim.environment.hazard_belief import HazardBeliefFrame
from robotics_sim.environment.hazard_field import FireSource

# Minimum wall-clock gap between pushing occupancy snapshots into the
# canvas's grid overlay. The belief grid can be large, and copying it more
# often than the overlay could ever visibly refresh just burns CPU for no
# visual benefit -- 10 Hz is already faster than a human can perceive a
# grid-color change, and well above typical GUI paint cadence.
GRID_OVERLAY_SNAPSHOT_INTERVAL_S = 0.1
# While the canvas has degraded the overlay to grid-lines-only (visible
# cells over MAX_GRID_OVERLAY_CELLS), no cell coloring is ever drawn from
# the snapshot -- pushing one at 10 Hz would just be copying the belief
# grid for nothing. Falls back to a much slower cadence instead of
# skipping entirely, so a snapshot is still available the moment the user
# zooms in enough to leave degraded mode.
GRID_OVERLAY_SNAPSHOT_DEGRADED_INTERVAL_S = 1.0


def occupancy_grid_snapshot_from_belief(belief_map) -> dict | None:
    """Build a read-only debug snapshot of *belief_map* for the canvas's
    optional "Show Grid" overlay only.

    Returns a plain dict (resolution, bounds, grid) rather than the
    BeliefMap itself, and copies the grid array, so the canvas can never
    hold a live reference to -- or accidentally mutate -- the real
    belief/occupancy state that planning, routing, and exploration read.
    """
    if belief_map is None:
        return None

    return {
        "resolution": float(belief_map.resolution),
        "bounds": tuple(belief_map.bounds),
        "grid": belief_map.grid.copy(),
    }


class PlannerWorkerSignals(QObject):
    route_ready = Signal(int, bool, str, list)


class PlannerWorker(QRunnable):
    """
    Compute A*/Dijkstra routes outside the GUI thread.

    Only immutable/simple data is passed into the worker. It must never touch
    Qt widgets or the live Robot object.
    """

    def __init__(
        self,
        request_id: int,
        planner_kwargs: dict,
        path_simplifier: str,
        debug_capture: "PlanDebugCapture | None" = None,
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.request_id = int(request_id)
        self.planner_kwargs = dict(planner_kwargs)
        # Sideband key for [PERF] logging only -- not a real planner
        # parameter, so it is popped out here and never reaches
        # compute_planned_waypoints(**self.planner_kwargs) below.
        self.perf_reason = str(self.planner_kwargs.pop("__perf_reason__", "route_replan"))
        self.path_simplifier = str(path_simplifier)
        self.signals = PlannerWorkerSignals()
        # Mutated during run() on the background thread; read back on the
        # GUI thread from on_async_route_ready() only after route_ready has
        # fired -- same safe handoff already used for route_plan_ms/
        # route_plan_perf_line below (see the comment a few lines down).
        self.debug_capture = debug_capture

    def run(self):
        if bool(self.planner_kwargs.get("__hold__", False)):
            self.signals.route_ready.emit(
                self.request_id,
                False,
                str(self.planner_kwargs.get("__hold_reason__", "holding position")),
                [],
            )
            return

        if compute_planned_waypoints is None:
            self.signals.route_ready.emit(
                self.request_id,
                False,
                "planner package is not available",
                [],
            )
            return

        # Timed strictly around this existing call boundary -- never inside
        # compute_planned_waypoints/A* itself. Stored as plain attributes,
        # not printed and not appended to any GUI widget: this runs on a
        # background QRunnable thread, and touching Qt widgets or the
        # terminal from here is exactly what must not happen. The GUI
        # thread's route_ready handler (on_async_route_ready) can read
        # these attributes off the worker after the signal fires, once
        # run() has already returned.
        perf_start = time.perf_counter()
        try:
            supports_simplifier = False
            try:
                supports_simplifier = "path_simplifier" in inspect.signature(compute_planned_waypoints).parameters
            except (TypeError, ValueError):
                supports_simplifier = False

            if supports_simplifier:
                success, reason, waypoints = compute_planned_waypoints(
                    **self.planner_kwargs,
                    path_simplifier=self.path_simplifier,
                    debug_capture=self.debug_capture,
                )
            else:
                success, reason, waypoints = compute_planned_waypoints(
                    **self.planner_kwargs, debug_capture=self.debug_capture
                )
        except Exception as exc:  # noqa: BLE001 - report planner failures to GUI safely.
            success = False
            reason = f"planner worker failed: {exc}"
            waypoints = []

        # Not printed and not emitted by default (see class docstring/task
        # notes): route_plan_ms/route_plan_perf_line are stored for optional
        # later inspection only. format_route_plan_perf_line is still the
        # formatter used to build the stored text, so it stays exercised
        # and testable even though nothing routes it to stdout or the GUI.
        self.route_plan_ms = (time.perf_counter() - perf_start) * 1000.0
        self.route_plan_result = "ok" if success else "fail"
        self.route_plan_perf_line = format_route_plan_perf_line(
            route_plan_ms=self.route_plan_ms,
            reason=self.perf_reason,
            grid_resolution=float(self.planner_kwargs.get("resolution", 0.0) or 0.0),
            mapped_obs=len(self.planner_kwargs.get("obstacle_points") or []),
            result=self.route_plan_result,
        )

        self.signals.route_ready.emit(
            self.request_id,
            bool(success),
            str(reason),
            [tuple(point) for point in waypoints],
        )



def current_route_repair_goal(agent) -> tuple[float, float] | None:
    """The goal a route-repair replan (route_affected / REPLAN_FOR_SAFETY)
    must preserve, or None if there is nothing active to preserve.

    route_affected and safety replans exist to fix the CURRENT route, not
    to pick a new destination -- unlike REQUEST_PLAN ("frontier reached" /
    initial plan) or PREFETCH_NEXT_TARGET, which legitimately choose a
    fresh target. Preferring active_path_goal_xy (the route actually being
    tracked) over exploration_target_xy (which can be set slightly ahead of
    active_path_goal_xy, e.g. mid-prefetch) keeps repair replans targeting
    exactly what the robot was already navigating to.

    A module-level function (not a method) so it can be unit-tested
    without instantiating the Qt-based simulation engine.
    """
    if agent is None:
        return None
    return agent.active_path_goal_xy or agent.exploration_target_xy


def _evaluate_route_first_segment(
    collision_checker,
    start_xy: tuple[float, float],
    target_xy: tuple[float, float] | None,
    obstacle_points: list[tuple[float, float]],
    robot_radius: float,
):
    """Run the same CollisionChecker.check_segment_points() rule
    build_observation() uses to compute active_segment_blocked, and return
    the full CollisionReport (never reduced to a bool) so callers can
    surface the real blocking point/reason for diagnostics instead of only
    the accept/reject decision. Returns None when there is nothing to check
    (no collision_checker or no target), matching route_first_segment_blocked's
    prior "not blocked" default for that case.
    """
    if collision_checker is None or target_xy is None:
        return None
    return collision_checker.check_segment_points(
        start=start_xy,
        end=target_xy,
        obstacle_points=list(obstacle_points),
        robot_radius=float(robot_radius),
    )


def route_first_segment_blocked(
    collision_checker,
    start_xy: tuple[float, float],
    target_xy: tuple[float, float] | None,
    obstacle_points: list[tuple[float, float]],
    robot_radius: float,
) -> bool:
    """True when the segment start_xy -> target_xy is unsafe by the same
    CollisionChecker.check_segment_points() rule build_observation() uses to
    compute active_segment_blocked.

    Used by apply_route_result() to reject a newly-planned route before it
    becomes the active path, instead of accepting it and letting the very
    next tick's safety check immediately trip REPLAN_FOR_SAFETY again for
    the route we just assigned.

    A module-level function (not a method) so it can be unit-tested with a
    plain CollisionChecker instance, without instantiating the Qt-based
    simulation engine. Thin wrapper around _evaluate_route_first_segment()
    -- identical computation/behavior, kept so every existing caller/test
    using this exact name and bool return type is unaffected.
    """
    report = _evaluate_route_first_segment(
        collision_checker, start_xy, target_xy, obstacle_points, robot_radius
    )
    return False if report is None else bool(report.collision)


def route_reaches_goal(
    waypoints: list[tuple[float, float]],
    goal: tuple[float, float] | None,
    tolerance: float,
) -> bool:
    """True when the route's final waypoint is within *tolerance* of *goal*.

    Thin backward-compatible delegate: the actual invariant now lives in
    NavigationSupervisor.validate_route_endpoint(), the single place this
    check is defined. Kept as a module-level function (rather than inlining
    the supervisor call at each of this module's call sites) so existing
    imports/tests referencing engine.route_reaches_goal keep working
    unchanged.
    """
    return NavigationSupervisor.validate_route_endpoint(waypoints, goal, tolerance)


def _navigation_debug_explanation(
    *,
    tracking_mode: str,
    decision_kind: str,
    decision_reason: str,
    controller: "ControllerDebug",
    rotate_threshold: "Maybe",
    safety: "SafetyDebug",
    predicted_motion: "PredictedMotionDebug",
    route: "RouteValidationDebug",
) -> str:
    """One-line human explanation of what the robot is doing right now,
    built only from fields already present on this same snapshot -- never a
    separate inference over engine/agent state. A module-level pure
    function so it is directly unit-testable without a real engine/robot.
    """
    if tracking_mode == "ROTATE" and not controller.heading_error.unavailable and not rotate_threshold.unavailable:
        return (
            f"ROTATE: heading error {math.degrees(abs(controller.heading_error.value)):.1f}° "
            f"exceeds {math.degrees(rotate_threshold.value):.1f}°."
        )

    if (
        not predicted_motion.collision.unavailable
        and predicted_motion.collision.value is not None
        and predicted_motion.collision.value.blocked
    ):
        terms = predicted_motion.collision.value
        d = "n/a" if terms.distance.unavailable else f"{terms.distance.value:.2f}"
        return f"STOP: predicted clearance {d} m is at/under required {terms.required_clearance:.2f} m."

    if (
        not safety.active_segment.unavailable
        and safety.active_segment.value is not None
        and safety.active_segment.value.blocked
    ):
        terms = safety.active_segment.value
        d = "n/a" if terms.distance.unavailable else f"{terms.distance.value:.2f}"
        return f"STOP: active segment blocked (clearance {d} m < required {terms.required_clearance:.2f} m)."

    if decision_kind == "REPLAN_FOR_SAFETY":
        return f"REPLAN: {decision_reason or 'a safety condition triggered a replan'}."

    if (
        not route.first_segment.unavailable
        and route.first_segment.value is not None
        and route.first_segment.value.blocked
    ):
        return "HOLD: planner rejected the first segment of the route."

    if decision_kind == "ACCEPT_PENDING_PATH":
        return "TARGET CHANGED: accepting the prefetched path for the next target."

    if tracking_mode == "TRACK":
        return "TRACK: heading aligned; moving toward the active waypoint."

    if decision_kind == "HOLD":
        return f"HOLD: {decision_reason}." if decision_reason else "HOLD: no active route to follow."

    if tracking_mode == "STOP":
        return "STOP: waypoint reached within goal tolerance."

    if tracking_mode == "IDLE":
        return "IDLE: no active target."

    return f"{decision_kind}: {decision_reason}." if decision_reason else f"{decision_kind}."


def _emit_robot_trace(engine, method: str, **kwargs) -> None:
    """Call RobotTrace.<method>(**kwargs) on engine.robot_trace if present.

    Real SimulationControllerMixin instances always have robot_trace (see
    ensure_robot_trace()); lightweight duck-typed engine fakes used by
    several tests do not, and must not be required to grow one just to
    exercise unrelated behavior. Never raises when robot_trace is absent,
    and RobotTrace's own trace_*() methods already no-op instantly unless
    their category is enabled -- so this is nearly free in both the
    "disabled" and "fake engine" cases.
    """
    trace = getattr(engine, "robot_trace", None)
    if trace is None:
        return
    getattr(trace, method)(**kwargs)


def _record_perf(engine, phase: str, duration_s: float) -> None:
    """Record a PerfMonitor timing sample for *phase* if the engine
    actually has ensure_perf_monitor() -- mirrors _emit_robot_trace()'s
    defensive pattern: lightweight duck-typed engine fakes used by many
    existing tests do not have it, and must not be required to grow one
    just to exercise unrelated behavior. Never raises.
    """
    monitor_getter = getattr(engine, "ensure_perf_monitor", None)
    if monitor_getter is None:
        return
    monitor_getter().record(phase, duration_s)


def _timed_method(phase: str):
    """Decorator: records a PerfMonitor timing sample for *phase* around
    the wrapped method's call (see _record_perf() for the defensive,
    fake-engine-safe recording).

    Deliberately a decorator rather than a rename-and-wrap: it preserves
    the wrapped method's exact name (functools.wraps), so existing tests
    that bind e.g. SimulationControllerMixin.apply_route_result.__get__(fake)
    directly onto a lightweight SimpleNamespace fake keep working
    unchanged -- no second, differently-named method needs to also be
    bound onto those fakes. Zero effect on the wrapped method's own
    behavior/return value/control flow.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            start = time.perf_counter()
            try:
                return func(self, *args, **kwargs)
            finally:
                _record_perf(self, phase, time.perf_counter() - start)
        return wrapper
    return decorator


def format_narrow_passage_diagnostic(
    *,
    path_goal: tuple[float, float] | None,
    route_affected_recent: int,
    first_segment_blocked: int,
    predicted_collision: int,
    min_clearance: float | None,
    action: str,
) -> str:
    """Pure formatter for the throttled [NARROW_DIAG] console line.

    Kept separate from the throttling/state logic (RobotAgent.
    route_affected_replan_allowed(), engine.py's per-tick speed-cap sync)
    so the exact line format is testable without a real CollisionChecker/
    BeliefMap. min_clearance is an approximation -- nearest mapped
    obstacle point distance from the robot's current position, not an
    exact segment-clearance computation -- deliberately cheap since this
    is a diagnostic, not a safety check.
    """
    goal_text = "None" if path_goal is None else f"({float(path_goal[0]):.2f},{float(path_goal[1]):.2f})"
    clearance_text = "n/a" if min_clearance is None else f"{float(min_clearance):.2f}"
    return (
        f"[NARROW_DIAG] path_goal={goal_text} route_affected_recent={int(route_affected_recent)} "
        f"first_segment_blocked={int(first_segment_blocked)} predicted_collision={int(predicted_collision)} "
        f"min_clearance={clearance_text} action={action}"
    )


def effective_planning_clearance(robot_radius: float, safety_radius: float) -> float:
    """The clearance radius used to inflate obstacles for planning/reachability.

    Semantics (confirmed against config.py's own clamp, main_window.py's
    radius-consistency enforcement, simulation_canvas.py's safety-radius
    circle rendering, and the GUI slider label itself, "Safety Radius r
    (m)"): config.safety_radius is the TOTAL clearance radius from the
    robot's center, not an extra margin layered on top of the robot's own
    body -- it is clamped to never be smaller than robot_radius (the
    physical body radius) so a misconfigured safety_radius can never shrink
    the robot's effective footprint below its own physical size.

    Deliberately max(...), not robot_radius + safety_radius -- adding them
    would double-count the body radius safety_radius already includes,
    over-inflating every obstacle and narrow corridor by an extra
    robot_radius beyond what the configured "r" value calls for.
    """
    return max(float(robot_radius), float(safety_radius))


def candidate_reachable_on_planning_grid(
    planning_grid,
    planner_type: str,
    start_xy: tuple[float, float],
    candidate_xy: tuple[float, float],
    *,
    bounds: tuple[float, float, float, float],
    resolution: float,
    robot_radius: float,
    goal_tolerance: float,
) -> bool:
    """True when compute_planned_waypoints() can find a route from start_xy
    to candidate_xy on the given, already-built planning_grid, AND that
    route's final waypoint actually reaches candidate_xy within
    goal_tolerance.

    Intended to be called with the SAME planning grid real single-robot
    navigation A* uses (see build_planning_grid_for_robot()), which --
    unlike the exploration planner's own internal scoring grid -- also
    inflates around dense mapped-obstacle-point samples. Used to filter
    exploration candidates the real planner would immediately reject with
    "no path found", before one is ever requested.

    The endpoint check matters just as much as "a path exists":
    compute_planned_waypoints() can return success=True after silently
    relocating an occupied goal cell to the nearest traversable one (see
    planner_registry._nearest_traversable_cell()) -- a route to THAT cell,
    not to candidate_xy. Without this check, a candidate could be accepted
    as "reachable" here and then be rejected moments later by the exact
    same endpoint rule apply_route_result()/on_prefetch_route_ready() apply
    via NavigationSupervisor.validate_route_endpoint() -- wasting a
    REQUEST_PLAN/PREFETCH cycle on a candidate that was never going to work.

    A module-level function (not a method) so it can be unit-tested with a
    plain OccupancyGrid, without instantiating the Qt-based simulation
    engine.
    """
    if compute_planned_waypoints is None or planning_grid is None:
        return True
    success, _reason, waypoints = compute_planned_waypoints(
        planner_type=planner_type,
        start_xy=start_xy,
        goal_xy=candidate_xy,
        bounds=bounds,
        resolution=resolution,
        robot_radius=robot_radius,
        planning_grid=planning_grid.copy(),
        unknown_is_traversable=True,
        obstacle_points=[],
    )
    if not success or not waypoints:
        return False
    return NavigationSupervisor.validate_route_endpoint(waypoints, candidate_xy, goal_tolerance)


# ============================================================
# METRICS WINDOW


class SimulationControllerMixin:
    ROUTE_STATE_ACTIVE = "ACTIVE"
    ROUTE_STATE_HOLD_NO_FRONTIER = "HOLD_NO_FRONTIER"
    ROUTE_STATE_STUCK_SAFETY = "STUCK_SAFETY"
    ROUTE_STATE_ESCAPE_LOCAL = "ESCAPE_LOCAL"
    # A route candidate exists but corridor validation rejected it -- this is
    # not "no frontier", it is "no safe route to the frontier we do have".
    ROUTE_STATE_HOLD_ROUTE_BLOCKED = "HOLD_ROUTE_BLOCKED"
    # Specifically a route_conflict_with_active_route rejection: the target
    # itself is fine, a teammate's active route is just in the way right now.
    ROUTE_STATE_WAITING_FOR_CORRIDOR = "WAITING_FOR_CORRIDOR"

    # How many candidate targets to try (1 initial attempt + retries) before
    # a corridor-blocked robot is allowed to fall back to HOLD/WAITING.
    MAX_ROUTE_RECOVERY_ATTEMPTS = 3

    # NAVIGATION MODE / ROBOT AGENT HELPERS
    # ========================================================

    def exploration_planner_name(self) -> str:
        return str(getattr(self.config, "exploration_planner", GOAL_SEEKING_PLANNER))

    def is_goal_seeking_mode(self) -> bool:
        return is_goal_seeking_planner(self.exploration_planner_name())

    def is_exploration_mode(self) -> bool:
        return is_exploration_planner(self.exploration_planner_name())

    def ensure_runtime_robot_registry(self) -> RuntimeRobotRegistry:
        if not hasattr(self, "runtime_robot_registry") or self.runtime_robot_registry is None:
            self.runtime_robot_registry = RuntimeRobotRegistry()
            self.robot_agents = self.runtime_robot_registry.agents
        return self.runtime_robot_registry

    def sync_runtime_robot_agents(self) -> None:
        registry = self.ensure_runtime_robot_registry()
        robots = list(getattr(self, "robots", []) or [])
        if not robots and getattr(self, "robot", None) is not None:
            robots = [self.robot]

        radii = [self.safety_radius_for_robot(robot) for robot in robots]
        registry.sync_from_robots(
            robots=robots,
            planner_mode=self.exploration_planner_name(),
            final_goal_xy=self.final_goal_xy() if hasattr(self, "config") else None,
            radii=radii,
        )
        self.robot_agents = registry.agents

    def runtime_agent(self, robot_index: int | None = None):
        self.sync_runtime_robot_agents()
        if not getattr(self, "robot_agents", None):
            return None
        if robot_index is None:
            return self.robot_agents[0]
        index = int(robot_index)
        if 0 <= index < len(self.robot_agents):
            return self.robot_agents[index]
        return None

    # BELIEF MAP
    # ========================================================

    def reset_belief_map(
        self,
        robot_count: int = 1,
        *,
        preserve_hazards: bool = False,
    ) -> None:
        """Create a fresh logical occupancy/belief map.

        This is the source of truth for exploration logic. The canvas pixmaps are
        rendering caches only.
        """
        preserved_fire_specs = []
        if preserve_hazards:
            previous_service = getattr(self, "hazard_service", None)
            if previous_service is not None:
                preserved_fire_specs = [
                    (source.position, source.intensity, source.radius)
                    for source in previous_service.sources()
                ]

        self.belief_map = BeliefMap(
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=max(float(self.config.grid_resolution), 0.10),
            robot_count=max(1, int(robot_count)),
        )
        self.explored_free_points = set()
        # Dense, visible obstacle-boundary samples. This is intentionally
        # separate from belief_map.grid OCCUPIED cells; the grid is logical,
        # while these samples preserve the obstacle contour for rendering and
        # local safety checks.
        self.mapped_obstacle_points = []
        self.mapped_obstacle_point_keys: set[tuple[float, float]] = set()

        # Dynamic hazards are a parallel layer aligned with the belief map.
        # Recreating the service here gives start/reset the required ephemeral
        # lifecycle without ever writing temporary fire cells into occupancy.
        self.hazard_service = RuntimeHazardService(
            bounds=self.belief_map.bounds,
            resolution=self.belief_map.resolution,
            robot_count=self.belief_map.robot_count,
            default_intensity=float(getattr(self.config, "default_fire_intensity", 1.0)),
            default_radius=float(getattr(self.config, "default_fire_radius", 2.0)),
            selection_radius=float(getattr(self.config, "fire_selection_radius", 0.6)),
            block_threshold=float(getattr(self.config, "hazard_block_threshold", 0.55)),
        )
        for position, intensity, radius in preserved_fire_specs:
            if self.hazard_service.field.in_bounds_world(position):
                self.hazard_service.field.add_fire(
                    position, intensity=intensity, radius=radius
                )

        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "set_hazard_snapshot"):
            canvas.set_hazard_snapshot(self.hazard_service.snapshot())
        # Push the fresh (empty) discovered-hazard frame once, then clear
        # dirty -- there is nothing pending to flush right after a reset.
        self.push_discovered_hazard_frame()
        self._discovered_hazard_render_dirty = False

    def ensure_belief_map(self) -> BeliefMap:
        """Return the active belief map, creating it if needed."""
        if not hasattr(self, "belief_map") or self.belief_map is None:
            count = len(getattr(self, "robots", [])) if getattr(self, "robots", None) else 1
            self.reset_belief_map(robot_count=max(1, count))
        return self.belief_map

    def ensure_hazard_service(self) -> RuntimeHazardService:
        """Return the runtime hazard service aligned with the active belief."""
        belief = self.ensure_belief_map()
        service = getattr(self, "hazard_service", None)
        if (
            service is None
            or service.field.shape != belief.grid.shape
            or abs(service.field.resolution - belief.resolution) > 1e-9
            or service.field.bounds != belief.bounds
            or service.belief.robot_count != belief.robot_count
        ):
            self.hazard_service = RuntimeHazardService(
                bounds=belief.bounds,
                resolution=belief.resolution,
                robot_count=belief.robot_count,
                default_intensity=float(getattr(self.config, "default_fire_intensity", 1.0)),
                default_radius=float(getattr(self.config, "default_fire_radius", 2.0)),
                selection_radius=float(getattr(self.config, "fire_selection_radius", 0.6)),
                block_threshold=float(getattr(self.config, "hazard_block_threshold", 0.55)),
            )
        return self.hazard_service

    def ensure_hazard_safety_runtime(self) -> HazardSafetyRuntime:
        """Return the cached HazardSafetyRuntime, (re)creating it whenever
        any hazard_cbf_* config parameter changes -- mirrors ensure_hazard_
        service()'s cache-invalidation-by-key pattern. One shared instance
        serves every robot: the team HazardBelief (and therefore the
        distance field built from it) is the same for all robots, only the
        per-robot state/limits/safety radius passed to filter_control()
        differ."""
        config = self.config
        key = (
            float(getattr(config, "hazard_cbf_margin", 0.20)),
            float(getattr(config, "hazard_cbf_activation_distance", 1.50)),
            float(getattr(config, "hazard_cbf_k1", 2.0)),
            float(getattr(config, "hazard_cbf_k2", 2.0)),
            int(getattr(config, "hazard_cbf_pyramid_levels", 1)),
            float(getattr(config, "hazard_cbf_sdf_smoothing_sigma_cells", 0.75)),
            float(getattr(config, "hazard_cbf_acceleration_weight", 1.0)),
            float(getattr(config, "hazard_cbf_angular_weight", 0.35)),
            float(getattr(config, "hazard_block_threshold", 0.55)),
        )
        runtime = getattr(self, "_hazard_safety_runtime", None)
        if runtime is None or getattr(self, "_hazard_safety_runtime_key", None) != key:
            runtime = HazardSafetyRuntime(
                block_threshold=key[8],
                margin=key[0],
                activation_distance=key[1],
                k1=key[2],
                k2=key[3],
                pyramid_levels=key[4],
                smoothing_sigma_cells=key[5],
                acceleration_weight=key[6],
                angular_weight=key[7],
            )
            self._hazard_safety_runtime = runtime
            self._hazard_safety_runtime_key = key
        return runtime

    def apply_hazard_safety_filter(self, robot, control: np.ndarray) -> np.ndarray:
        """Filter one robot's proposed control through the Observed Hazard
        OGM-HOCBF safety filter.

        Pure wiring: gathers the team's observed HazardBelief snapshot, the
        robot's own state/limits/safety radius, and delegates to
        HazardSafetyRuntime -- no SDF construction, no gradients/Hessians,
        no HOCBF equations, and no QP solving live here (see
        hazard_safety_runtime.py / hazard_hocbf_filter.py /
        hazard_distance_field.py).

        Must be called AFTER the nominal control is decided (nominal_
        control_safe() for single-robot, select_runtime_control_source() for
        multi-robot, so a CONTROL-owning plugin cannot skip it) and BEFORE
        predicted_motion_report() -- see the call sites in simulation_step()
        and simulation_step_multi() for the exact ordering.
        """
        if not bool(getattr(self.config, "hazard_cbf_enabled", True)):
            return control

        hazard_service = getattr(self, "hazard_service", None)
        if hazard_service is None or robot is None:
            return control

        runtime = self.ensure_hazard_safety_runtime()
        result = runtime.filter_control(
            belief_frame=hazard_service.belief.snapshot(),
            geometry=hazard_service.belief.geometry,
            state=robot.state,
            limits=robot.limits,
            nominal_control=control,
            safety_radius=self.safety_radius_for_robot(robot),
        )
        return result.control

    def push_hazard_snapshot(self) -> None:
        """Push the GROUND-TRUTH hazard field snapshot. Kept for legacy/
        potential editor use (see SimulationCanvas.draw_fires(), no longer
        called from the live paint loop) -- never confuse this with
        push_discovered_hazard_frame(), which is what runtime actually
        renders."""
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "set_hazard_snapshot"):
            canvas.set_hazard_snapshot(self.ensure_hazard_service().snapshot())

    def push_discovered_hazard_frame(self) -> None:
        """Push the team's DISCOVERED hazard belief -- the only hazard layer
        live simulation renders (see SimulationCanvas.draw_discovered_
        hazard()). Independent of push_hazard_snapshot() (ground truth):
        the canvas must never receive both bundled into one ambiguous
        payload, so this uses its own explicit setter/dict shape.

        Uses HazardBelief.snapshot() (one full-grid copy) -- O(robots*
        height*width), not the narrow read_cells()/blocked_cells() the
        sensor-update/planning hot paths use. Callers must not invoke this
        per robot per tick: see _flush_discovered_hazard_render(), the only
        caller besides reset_belief_map()'s own initial empty-frame push,
        which collapses however many robots' FoV updates happened this
        step into at most one push.
        """
        canvas = getattr(self, "canvas", None)
        if canvas is None or not hasattr(canvas, "set_discovered_hazard_frame"):
            return
        service = self.ensure_hazard_service()
        canvas.set_discovered_hazard_frame(
            {
                "frame": service.belief.snapshot(),
                "bounds": service.field.bounds,
                "resolution": service.field.resolution,
            }
        )

    def _flush_discovered_hazard_render(self) -> None:
        """Push at most one discovered-hazard render frame per simulation
        step, regardless of how many robots' FoV updates marked the render
        dirty this tick (see update_explored_free_points_from_polygon()).

        Called once, after every robot due for a sensor update this tick has
        already run it -- see simulation_step()/simulation_step_multi()'s
        own call sites, right after their sensor-update block/loop.
        Reads only the dirty flag and HazardBelief (via push_discovered_
        hazard_frame()) -- never HazardField/FireSource.
        """
        if not getattr(self, "_discovered_hazard_render_dirty", False):
            return
        self._discovered_hazard_render_dirty = False
        self.push_discovered_hazard_frame()

    def occupancy_grid_snapshot(self) -> dict | None:
        """Read-only snapshot of the current belief/occupancy grid, for the
        canvas's optional "Show Grid" overlay only.

        Purely visual/debug: never mutated, never fed back into planning,
        routing, or exploration, and does not create or rebuild the belief
        map -- returns None if none exists yet.
        """
        return occupancy_grid_snapshot_from_belief(getattr(self, "belief_map", None))

    def push_grid_overlay_snapshot_if_due(self) -> None:
        """Push a fresh occupancy snapshot into the canvas's grid overlay,
        but only when the overlay is enabled, and at a rate that depends on
        whether the overlay is currently degraded (grid-lines-only, no cell
        coloring drawn -- see MAX_GRID_OVERLAY_CELLS in simulation_canvas.py):
        GRID_OVERLAY_SNAPSHOT_INTERVAL_S (10 Hz) normally, or the much
        slower GRID_OVERLAY_SNAPSHOT_DEGRADED_INTERVAL_S (1 Hz) while
        degraded, since a degraded overlay never uses the snapshot's cell
        colors anyway. Automatically returns to the fast rate as soon as
        the canvas reports it is no longer degraded (e.g. the user zoomed
        in below the visible-cell cap).

        Purely visual/debug, read-only, and rate-limited: the overlay
        cannot refresh visibly faster than this anyway, so throttling here
        avoids copying the (potentially large) belief grid on every single
        simulation tick for no visible benefit.
        """
        canvas = getattr(self, "canvas", None)
        if canvas is None or not getattr(canvas, "grid_overlay_enabled", False):
            return

        is_degraded = getattr(canvas, "is_grid_overlay_degraded", None)
        degraded = bool(is_degraded()) if callable(is_degraded) else False
        interval = (
            GRID_OVERLAY_SNAPSHOT_DEGRADED_INTERVAL_S if degraded else GRID_OVERLAY_SNAPSHOT_INTERVAL_S
        )

        now = time.perf_counter()
        last_push = getattr(self, "_grid_overlay_snapshot_last_push_time", None)
        if last_push is not None and (now - last_push) < interval:
            return

        self._grid_overlay_snapshot_last_push_time = now
        canvas.set_grid_overlay_snapshot(self.occupancy_grid_snapshot())

    def sync_legacy_map_views_from_belief(self) -> None:
        """Update legacy views without destroying boundary obstacle samples.

        The logical FREE/UNKNOWN/OCCUPIED state lives in ``belief_map.grid``.
        However, the visual obstacle trace and local route safety checks need
        dense boundary samples, not just one center point per occupied cell.

        Therefore this method exports explored FREE cells from the belief map,
        but deliberately does *not* replace ``self.mapped_obstacle_points``.
        Those mapped obstacle points are maintained by ``update_sensed_obstacles``
        as visible boundary samples.
        """
        belief = self.ensure_belief_map()
        self.explored_free_points = belief.explored_points()
        if not hasattr(self, "mapped_obstacle_points"):
            self.mapped_obstacle_points = []
        if not hasattr(self, "mapped_obstacle_point_keys"):
            self.mapped_obstacle_point_keys = {
                (round(float(p[0]), 3), round(float(p[1]), 3))
                for p in self.mapped_obstacle_points
            }

    # CONFIG
    # ========================================================

    def read_config(self) -> SimulationConfig:
        return SimulationConfig(
            x=float(self.x_input.value()),
            y=float(self.y_input.value()),
            theta=float(self.theta_input.value()),
            v=float(self.v_slider.value()),
            vision=float(self.vision_slider.value()),
            body_radius=float(self.body_radius_slider.value()),
            safety_radius=max(float(self.safety_radius_slider.value()), float(self.body_radius_slider.value())),
            goal_x=float(self.goal_x_input.value()),
            goal_y=float(self.goal_y_input.value()),
            max_speed=float(self.max_speed_input.value()),
            max_acceleration=float(self.max_accel_input.value()),
            max_angular_speed=float(self.max_omega_input.value()),
            goal_tolerance=float(self.goal_tol_input.value()),
            acceleration_gain=float(self.accel_gain_input.value()),
            planner_type=self.planner_combo.currentText(),
            path_simplifier=self.path_simplifier_combo.currentText(),
            exploration_planner=self.exploration_planner_combo.currentText(),
            coordinator_type=self.coordinator_combo.currentText() if hasattr(self, "coordinator_combo") else self.config.coordinator_type,
            exploration_replan_cooldown=max(0.0, float(self.exploration_cooldown_input.value())),
            ipp_distance_penalty=max(0.0, float(self.ipp_lambda_input.value())),
            vision_model=self.vision_combo.currentText(),
            agent_mode=self.top_bar.mode_selector.currentText(),
            grid_resolution=max(0.10, float(self.grid_resolution_input.value())),
            obstacles=list(self.config.obstacles),
            show_goal_preview=self.preview_switch.isChecked(),
            show_path=True,
            show_vision=True,
            show_explored_area=self.explored_area_switch.isChecked(),
            show_obstacles=self.obstacles_switch.isChecked(),
            mapping_point_spacing=self.config.mapping_point_spacing,
            robot_count=max(1, min(8, int(round(float(self.robot_count_input.value()))))) if hasattr(self, "robot_count_input") else self.config.robot_count,
            selected_robot_index=int(getattr(self, "selected_robot_index", 0)),
            same_robot_configuration=self.same_config_switch.isChecked() if hasattr(self, "same_config_switch") else self.config.same_robot_configuration,
            robots=list(getattr(self, "multi_robot_configs", self.config.robots)),
        )

    def enforce_radius_consistency(self, *_):
        """
        Keep safety radius r physically valid.

        r is a clearance radius, so it cannot be smaller than the robot body.
        """
        body_radius = float(self.body_radius_slider.value())
        safety_radius = float(self.safety_radius_slider.value())
        if safety_radius < body_radius:
            self.safety_radius_slider.setValue(body_radius)

    def enforce_selected_multi_radius_consistency(self, *_):
        """Keep per-robot safety radius physically valid in multi config."""
        if not hasattr(self, "multi_body_radius_slider"):
            return
        body_radius = float(self.multi_body_radius_slider.value())
        safety_radius = float(self.multi_safety_radius_slider.value())
        if safety_radius < body_radius:
            self.multi_safety_radius_slider.setValue(body_radius)

    def update_preview(self):
        self.enforce_radius_consistency()
        self.enforce_selected_multi_radius_consistency()
        self.update_relevant_parameter_visibility()
        self.config = self.read_config()
        self.canvas.set_preview_config(self.config)

    def apply_config_to_widgets(self, config: SimulationConfig) -> None:
        """
        Push a loaded .sim configuration back into the GUI controls.
        """
        self.x_input.setValue(config.x)
        self.y_input.setValue(config.y)
        self.theta_input.setValue(config.theta)
        self.v_slider.setValue(config.v)
        self.vision_slider.setValue(config.vision)
        self.body_radius_slider.setValue(config.body_radius)
        self.safety_radius_slider.setValue(max(config.safety_radius, config.body_radius))
        self.goal_x_input.setValue(config.goal_x)
        self.goal_y_input.setValue(config.goal_y)
        self.max_speed_input.setValue(config.max_speed)
        self.max_omega_input.setValue(config.max_angular_speed)
        self.max_accel_input.setValue(config.max_acceleration)
        self.goal_tol_input.setValue(config.goal_tolerance)
        self.accel_gain_input.setValue(config.acceleration_gain)
        self.preview_switch.setChecked(config.show_goal_preview)
        self.obstacles_switch.setChecked(config.show_obstacles)
        self.explored_area_switch.setChecked(config.show_explored_area)
        self.planner_combo.setCurrentText(config.planner_type)
        self.path_simplifier_combo.setCurrentText(config.path_simplifier)
        self.exploration_planner_combo.setCurrentText(config.exploration_planner)
        if hasattr(self, "coordinator_combo"):
            self.coordinator_combo.setCurrentText(config.coordinator_type)
        self.exploration_cooldown_input.setValue(config.exploration_replan_cooldown)
        self.ipp_lambda_input.setValue(config.ipp_distance_penalty)
        self.grid_resolution_input.setValue(config.grid_resolution)
        self.vision_combo.setCurrentText(config.vision_model)
        self.top_bar.mode_selector.setCurrentText(config.agent_mode)

        self.multi_robot_configs = normalized_robot_start_configs(config)
        self.selected_robot_index = max(0, min(int(config.selected_robot_index), len(self.multi_robot_configs) - 1))
        if hasattr(self, "robot_count_input"):
            self.robot_count_input.setValue(max(1, min(8, int(config.robot_count))))
            self.same_config_switch.setChecked(bool(config.same_robot_configuration))
            self.load_selected_robot_into_panel()

        self.config = config
        self.spatial_index.rebuild(self.config.obstacles)
        self.update_relevant_parameter_visibility()
        self.set_configuration_locked(self.running or self.robot is not None)
        self.canvas.set_preview_config(self.config)
        self.canvas.set_planned_path([])
        self.canvas.set_exploration_target(None)

    def save_simulation_config(self) -> None:
        self.config = self.read_config()

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save simulation scenario",
            "scenario.sim",
            "Simulation files (*.sim);;JSON files (*.json);;All files (*)",
        )

        if not path:
            return

        if not path.lower().endswith((".sim", ".json")):
            path += ".sim"

        try:
            save_sim_file(path, self.config)
            self.canvas.set_status(f"Saved scenario: {os.path.basename(path)}")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def load_simulation_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load simulation scenario",
            "",
            "Simulation files (*.sim);;JSON files (*.json);;All files (*)",
        )

        if not path:
            return

        try:
            config = load_sim_file(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.reset_simulation()
        self.apply_config_to_widgets(config)
        self.canvas.set_status(f"Loaded scenario: {os.path.basename(path)}")

    def export_navigation_snapshots(self) -> None:
        """Export the immutable in-memory navigation history to one XLSX row
        per SELECTED snapshot.

        The exporter is dependency-free and copies the current ring-buffer
        contents before writing, so the workbook is internally consistent
        even if a live run continues producing new snapshots while the save
        dialog is open. The full in-memory history (navigation_debug_log's
        bounded ring buffer, the </> step buttons, rollback) is never
        reduced by any of this -- only which rows land in the exported
        workbook is selected, via select_navigation_snapshot_events() (see
        robotics_sim/diagnostics/snapshot_export.py), and that selection
        always runs BEFORE any row is flattened.
        """
        log = getattr(self, "navigation_debug_log", None)
        events = tuple(log.events()) if log is not None else ()
        if not events:
            QMessageBox.information(
                self,
                "No snapshots",
                "There are no navigation snapshots to export yet.",
            )
            return

        automatic_label = f"Automatic filtered (~{DEFAULT_AUTO_TARGET_ROWS} rows, recommended)"
        raw_label = f"Raw (all {len(events)} snapshots, slower/larger)"
        stride2_label = "Every 2nd routine snapshot"
        stride3_label = "Every 3rd routine snapshot"
        stride5_label = "Every 5th routine snapshot"
        custom_label = "Custom stride..."

        choice, accepted = QInputDialog.getItem(
            self,
            "Export navigation snapshots",
            "Choose how many snapshots to export:",
            [automatic_label, raw_label, stride2_label, stride3_label, stride5_label, custom_label],
            0,
            False,
        )
        if not accepted:
            return

        if choice == raw_label:
            selection = select_navigation_snapshot_events(events, mode="raw")
        elif choice == stride2_label:
            selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=2)
        elif choice == stride3_label:
            selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=3)
        elif choice == stride5_label:
            selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=5)
        elif choice == custom_label:
            stride, accepted = QInputDialog.getInt(
                self,
                "Custom stride",
                "Export every Nth routine snapshot:",
                2,
                2,
                100,
            )
            if not accepted:
                return
            selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=stride)
        else:
            selection = select_navigation_snapshot_events(
                events, mode="automatic_filtered", target_rows=DEFAULT_AUTO_TARGET_ROWS
            )

        stamp = time.strftime("%Y%m%d_%H%M%S")
        if selection.mode == "raw":
            suggested_name = f"navigation_snapshots_raw_{stamp}.xlsx"
        elif selection.mode == "custom_stride":
            suggested_name = f"navigation_snapshots_stride_{selection.routine_stride}_{stamp}.xlsx"
        else:
            suggested_name = f"navigation_snapshots_filtered_{stamp}.xlsx"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export navigation snapshots",
            suggested_name,
            "Excel workbooks (*.xlsx);;All files (*)",
        )
        if not path:
            return

        try:
            count = export_navigation_snapshots_xlsx(
                selection.events,
                path,
                source_indices=selection.source_indices,
                source_count=selection.source_count,
                export_mode=selection.mode,
                routine_stride=selection.routine_stride,
                target_rows=selection.target_rows,
                semantic_events_preserved=selection.semantic_events_preserved,
            )
        except SnapshotExportError as exc:
            QMessageBox.critical(self, "Snapshot export failed", str(exc))
            return

        final_path = path if path.lower().endswith(".xlsx") else f"{path}.xlsx"
        mode_label = {
            "raw": "raw",
            "automatic_filtered": "automatic filtered",
            "custom_stride": "custom stride",
        }.get(selection.mode, selection.mode)
        self.canvas.set_status(
            f"Exported {count:,} of {selection.source_count:,} navigation snapshots "
            f"({mode_label}, routine stride {selection.routine_stride}): {os.path.basename(final_path)}"
        )

    def final_goal_xy(self) -> tuple[float, float]:
        return (float(self.config.goal_x), float(self.config.goal_y))

    def _restore_final_goal_into_config_and_widgets(self, goal_xy: tuple[float, float]) -> None:
        """Sync config.goal_x/goal_y and the Goal X/Y widgets to *goal_xy*.

        Used by restore_navigation_debug_snapshot(): final_goal_xy() (read by
        select_navigation_goal()/every subsequent replan) returns config.
        goal_x/goal_y directly, never agent.final_goal_xy -- restoring only
        the agent field leaves config, and the GUI showing it, pointed at
        whatever goal the live run had since moved on to.

        Widgets are updated with signals blocked, same pattern as
        MainWindow.load_selected_robot_into_panel()/refresh_same_position_
        rows() -- a restore must not trigger update_preview()-style edit
        callbacks, replans, or additional navigation-debug snapshots.
        """
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        self.config.goal_x = gx
        self.config.goal_y = gy

        widgets = [
            widget
            for widget in (getattr(self, "goal_x_input", None), getattr(self, "goal_y_input", None))
            if widget is not None
        ]
        blocked = [widget.blockSignals(True) for widget in widgets]
        try:
            if hasattr(self, "goal_x_input"):
                self.goal_x_input.setValue(gx)
            if hasattr(self, "goal_y_input"):
                self.goal_y_input.setValue(gy)
        finally:
            for widget, was_blocked in zip(widgets, blocked):
                widget.blockSignals(was_blocked)

    def select_navigation_goal(self, start_xy: tuple[float, float]) -> tuple[tuple[float, float] | None, str]:
        """
        Select the current navigation target.

        Goal seeking:
            the GUI final goal G is executable.

        Exploration:
            G is only a visual/reference goal. The executable target must come
            from the exploration planner. If no exploration target exists, return
            None so the caller can enter HOLD instead of planning to G.
        """
        final_goal = self.final_goal_xy()
        planner_name = str(self.config.exploration_planner)

        if is_goal_seeking_planner(planner_name):
            self.current_exploration_target = None
            self.last_goal_selection_reason = "using final mission goal"
            self.canvas.set_exploration_target(None)
            agent = self.runtime_agent(None)
            if agent is not None:
                agent.exploration_target_xy = None
            return final_goal, self.last_goal_selection_reason

        belief = self.ensure_belief_map()
        agent = self.runtime_agent(None)
        current_target = agent.exploration_target_xy if agent is not None else self.current_exploration_target

        # Exclude this robot's own recently-failed targets so a target that
        # just failed to plan (see RobotAgent.invalidate_failed_exploration_route())
        # is not immediately re-selected here -- this call is independent
        # from ExplorationBehavior's own selection and must honor the same
        # blacklist, or the two can disagree and undo each other's recovery.
        #
        # Also exclude the robot's own current position and the
        # just-completed active_path_goal_xy, mirroring
        # ExplorationBehavior._pick_next_target(): this call is independent
        # from that one, so without the same exclusions it can re-propose a
        # target ExplorationBehavior already rejected as "already reached",
        # producing a near-zero-length ROUTE ok that gets "reached" again
        # within a tick or two.
        excluded_targets: list[tuple[float, float]] = [
            (float(start_xy[0]), float(start_xy[1])),
        ]
        if agent is not None:
            excluded_targets.extend(
                agent.recently_failed_exploration_targets(
                    current_time=float(self.simulation_time),
                    cooldown=ExplorationBehavior._FAILED_TARGET_EXCLUSION_WINDOW,
                )
            )
            if agent.active_path_goal_xy is not None:
                excluded_targets.append(agent.active_path_goal_xy)

        result = select_exploration_goal(
            planner_name,
            belief_map=belief,
            robot_xy=(float(start_xy[0]), float(start_xy[1])),
            robot_heading=float(getattr(self.robot, "theta", 0.0)) if self.robot is not None else 0.0,
            current_target=current_target,
            final_goal_xy=final_goal,
            robot_count=1,
            robot_radius=float(self.safety_radius()),
            sensor_range=float(self.config.vision),
            vision_model=str(self.config.vision_model),
            ipp_distance_penalty=float(self.config.ipp_distance_penalty),
            excluded_targets=excluded_targets,
            target_exclusion_radius=(
                max(float(self.config.grid_resolution), 2.0 * float(self.config.goal_tolerance))
                if excluded_targets
                else 0.0
            ),
        )

        selected_score = None
        if result.success and result.target is not None:
            selected_key = (round(float(result.target[0]), 3), round(float(result.target[1]), 3))
            for candidate in result.candidates:
                if (round(float(candidate.target[0]), 3), round(float(candidate.target[1]), 3)) == selected_key:
                    selected_score = candidate.score
                    break
        self.telemetry.report_frontier_selection(
            robot_label="R1",
            success=bool(result.success),
            selected=result.target if result.success else None,
            reason=str(result.reason),
            score=selected_score,
            candidate_count=len(result.candidates),
        )

        if not result.success or result.target is None:
            self.current_exploration_target = None
            self.last_goal_selection_reason = str(result.reason)
            self.canvas.set_exploration_target(None)
            if agent is not None:
                agent.exploration_target_xy = None
            return None, self.last_goal_selection_reason

        target = (float(result.target[0]), float(result.target[1]))

        # Belt-and-suspenders: reject a candidate at/near the robot's
        # current position outright, regardless of whether the exclusion
        # above was actually honored internally -- this is the guarantee
        # that stops a near-zero-length route from ever being requested,
        # independent of exploration-planner internals. Mirrors
        # ExplorationBehavior._pick_next_target()'s own hard check.
        if math.hypot(
            target[0] - float(start_xy[0]), target[1] - float(start_xy[1])
        ) <= float(self.config.goal_tolerance):
            self.current_exploration_target = None
            self.last_goal_selection_reason = (
                f"{result.reason}; rejected: candidate within goal_tolerance of robot position"
            )
            self.canvas.set_exploration_target(None)
            if agent is not None:
                agent.exploration_target_xy = None
            return None, self.last_goal_selection_reason

        self.current_exploration_target = target
        self.last_goal_selection_reason = str(result.reason)
        self.canvas.set_exploration_target(target)
        if agent is not None:
            agent.set_exploration_target(target, reason=self.last_goal_selection_reason)
        return target, self.last_goal_selection_reason

    def force_robot_pose_free_in_belief(self, robot_index: int | None = None) -> bool:
        """Ensure the active robot center is FREE in the logical map.

        A live robot pose must never be rejected by A*/Dijkstra as an occupied
        start cell. This fixes false deadlocks caused by obstacle-point
        quantization or by an obstacle boundary sample landing on the robot's
        current grid cell. Ground-truth collision checks still remain active.
        """
        belief = self.ensure_belief_map()
        if robot_index is None:
            robot = getattr(self, "robot", None)
            idx = None
        else:
            idx = int(robot_index)
            if not (0 <= idx < len(getattr(self, "robots", []) or [])):
                return False
            robot = self.robots[idx]
        if robot is None:
            return False
        changed = belief.force_free_point(
            (float(robot.x), float(robot.y)),
            robot_index=idx,
            time_s=float(getattr(self, "simulation_time", 0.0)),
        )
        if changed:
            self.sync_legacy_map_views_from_belief()
        return bool(changed)

    def force_all_robot_poses_free_in_belief(self) -> int:
        """Force every live robot center to FREE and return changed cells."""
        changed = 0
        robots = list(getattr(self, "robots", []) or [])
        if robots:
            for index in range(len(robots)):
                changed += int(self.force_robot_pose_free_in_belief(index))
        elif getattr(self, "robot", None) is not None:
            changed += int(self.force_robot_pose_free_in_belief(None))
        return changed

    def sanitize_planner_obstacle_points(
        self,
        obstacle_points: list[tuple[float, float]],
        *,
        start_xy: tuple[float, float],
        robot_radius: float,
        resolution: float,
    ) -> tuple[list[tuple[float, float]], int]:
        """Remove obstacle samples that falsely occupy the robot's own start cell.

        The planner inflates obstacle points by the robot safety radius. If a
        quantized mapped-obstacle point falls on the current robot cell, A* can
        reject the route before it even starts. We clear only a small disk around
        the robot's current center for the planner input. This does not remove
        ground-truth obstacle checks or teammate dynamic obstacles outside the
        start cell.
        """
        sx, sy = float(start_xy[0]), float(start_xy[1])
        # Enough to clear the start cell and immediate quantization error, but
        # not enough to open corridors through real obstacles.
        clear_radius = max(float(resolution) * 1.25, min(float(robot_radius) * 0.75, float(resolution) * 2.5))
        clear_radius = max(clear_radius, 1e-6)
        kept: list[tuple[float, float]] = []
        removed = 0
        for point in obstacle_points:
            px, py = float(point[0]), float(point[1])
            if math.hypot(px - sx, py - sy) <= clear_radius:
                removed += 1
                continue
            kept.append((px, py))
        return kept, removed

    def obstacle_points_for_segment_safety_check(
        self,
        start_xy: tuple[float, float],
        robot_radius: float,
    ) -> list[tuple[float, float]]:
        """Return self.mapped_obstacle_points with the SAME near-start
        sanitization sanitize_planner_obstacle_points() already applies
        before the planner ever sees them, for use by every post-hoc
        segment/motion safety check (route_first_segment_blocked(),
        active_segment_blocked in build_observation(), predicted_collision
        in predicted_motion_report()).

        Root cause this closes: the planner (A*/reachability) has always
        sanitized mapped_obstacle_points before building its planning grid
        -- clearing a small disk around the robot's own current position,
        because a boundary sample the robot's own sensor just placed a few
        centimeters from its center is expected, not a real obstacle, and
        would otherwise make A* reject the route before it starts (see
        sanitize_planner_obstacle_points()'s own docstring). The four
        safety/validation checks above were never given the same
        sanitization -- they used the raw mapped_obstacle_points list
        directly, so CollisionChecker.check_segment_points()'s "distance
        from any obstacle point to the segment" rule always found that same
        near-start sample sitting well inside robot_radius of start_xy
        (since t=0 of any segment starting AT the robot is exactly where
        that sample was recorded), and rejected the route's first segment
        as "blocked" regardless of which direction it actually pointed.
        This is exactly the reproducible Office.sim sequence: reachability
        approves a candidate, A* finds a path on the sanitized grid, the
        simplifier agrees, and then route validation -- and predicted
        collision, using the same unsanitized list -- immediately rejects
        it. Using the SAME sanitized list here makes every checker agree
        with what the planner already assumed about the robot's own
        immediate surroundings. clear_radius stays exactly what
        sanitize_planner_obstacle_points() already computes (~1.25x grid
        resolution) -- this does not touch robot_radius/safety_radius, does
        not relax any check away from the robot's current position, and
        does not change the disk beyond the one already used to plan the
        route.
        """
        points, _ = self.sanitize_planner_obstacle_points(
            list(self.mapped_obstacle_points),
            start_xy=start_xy,
            robot_radius=robot_radius,
            resolution=float(self.config.grid_resolution),
        )
        return points

    def build_planning_grid_for_robot(
        self,
        robot,
        *,
        obstacle_points: list[tuple[float, float]] | None = None,
        robot_radius: float | None = None,
    ):
        """Build a planning grid from BeliefMap plus dense mapped/dynamic samples.

        BeliefMap is the source of logical UNKNOWN/FREE/OCCUPIED state.
        Dense boundary samples remain useful for route safety and are added only
        to this derived planning grid, not to the belief map itself.
        """
        belief = self.ensure_belief_map()
        radius = self.safety_radius_for_robot(robot) if robot_radius is None else float(robot_radius)
        planning_grid = belief.to_planning_grid(
            unknown_is_traversable=True,
            inflate_radius=max(0.0, radius),
        )
        if obstacle_points:
            planning_grid.add_obstacle_points(obstacle_points, padding=max(0.0, radius))

        # Project the team's DISCOVERED hazard belief into this derived
        # binary planning grid -- never the omniscient ground-truth
        # HazardField. BeliefMap.grid is never modified either way.
        hazard_service = getattr(self, "hazard_service", None)
        if hazard_service is not None:
            apply_hazard_belief_to_planning_grid(
                planning_grid,
                hazard_service.belief,
                block_threshold=hazard_service.block_threshold,
                inflate_radius=max(0.0, radius),
            )
        return planning_grid

    def planner_accepts_path_simplifier(self) -> bool:
        """Return True when the installed planner registry supports path_simplifier."""
        if compute_planned_waypoints is None:
            return False
        try:
            return "path_simplifier" in inspect.signature(compute_planned_waypoints).parameters
        except (TypeError, ValueError):
            return False

    def call_compute_planned_waypoints(
        self,
        planner_kwargs: dict,
        *,
        path_simplifier: str | None = None,
        debug_capture: PlanDebugCapture | None = None,
    ) -> tuple[bool, str, list[tuple[float, float]]]:
        """Call the planner without spamming TypeError fallback messages.

        debug_capture: optional outparam forwarded straight through to
        compute_planned_waypoints(); None (the default, used by every
        caller that does not care) costs nothing extra.
        """
        if bool(planner_kwargs.get("__hold__", False)):
            return False, str(planner_kwargs.get("__hold_reason__", "holding position")), []

        if compute_planned_waypoints is None:
            return False, "planner package is not available", []
        if path_simplifier is not None and self.planner_accepts_path_simplifier():
            return compute_planned_waypoints(
                **planner_kwargs, path_simplifier=path_simplifier, debug_capture=debug_capture
            )
        return compute_planned_waypoints(**planner_kwargs, debug_capture=debug_capture)


    def build_planner_kwargs(self, start_xy: tuple[float, float]) -> dict:
        """
        Build an immutable input packet for synchronous or asynchronous planning.
        """
        self.force_robot_pose_free_in_belief(None)
        goal_xy, goal_reason = self.select_navigation_goal(start_xy)
        self.last_goal_selection_reason = goal_reason

        resolution = float(self.config.grid_resolution)
        robot_radius = float(self.safety_radius())

        if goal_xy is None:
            return dict(
                __hold__=True,
                __hold_reason__=goal_reason,
                planner_type=self.config.planner_type,
                start_xy=(float(start_xy[0]), float(start_xy[1])),
                goal_xy=(float(start_xy[0]), float(start_xy[1])),
                planning_grid=None,
                bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
                resolution=resolution,
                robot_radius=robot_radius,
                obstacle_points=[],
            )

        obstacle_points, removed = self.sanitize_planner_obstacle_points(
            list(self.mapped_obstacle_points),
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            robot_radius=robot_radius,
            resolution=resolution,
        )

        if removed:
            self.last_goal_selection_reason = f"{goal_reason}; ignored {removed} own-start obstacle sample(s) for planning"

        planning_grid = self.build_planning_grid_for_robot(
            self.robot,
            obstacle_points=obstacle_points,
            robot_radius=robot_radius,
        )

        return dict(
            planner_type=self.config.planner_type,
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            obstacles=[],
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=resolution,
            robot_radius=robot_radius,
            planning_grid=planning_grid,
            unknown_is_traversable=True,
            obstacle_points=[],
        )

    def build_planner_kwargs_for_goal(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        robot=None,
    ) -> dict:
        """
        Build an immutable planning input packet for a *known* goal.

        Unlike build_planner_kwargs() this method does not call
        select_navigation_goal(); the caller already knows where to go (e.g.
        a prefetched frontier target chosen by ExplorationBehavior).
        """
        robot = robot if robot is not None else self.robot
        resolution = float(self.config.grid_resolution)
        robot_radius = self.safety_radius_for_robot(robot)

        obstacle_points, _ = self.sanitize_planner_obstacle_points(
            list(self.mapped_obstacle_points),
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            robot_radius=robot_radius,
            resolution=resolution,
        )

        planning_grid = self.build_planning_grid_for_robot(
            robot,
            obstacle_points=obstacle_points,
            robot_radius=robot_radius,
        )

        return dict(
            planner_type=self.config.planner_type,
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            obstacles=[],
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=resolution,
            robot_radius=robot_radius,
            planning_grid=planning_grid,
            unknown_is_traversable=True,
            obstacle_points=[],
        )

    def ensure_multi_exploration_target_slots(self) -> None:
        """Keep one exploration target and blacklist slot per runtime robot."""
        count = len(self.robots)
        if len(self.multi_exploration_targets) < count:
            self.multi_exploration_targets.extend([None] * (count - len(self.multi_exploration_targets)))
        elif len(self.multi_exploration_targets) > count:
            self.multi_exploration_targets = self.multi_exploration_targets[:count]

        if not hasattr(self, "multi_invalidated_exploration_targets"):
            self.multi_invalidated_exploration_targets = []
        if len(self.multi_invalidated_exploration_targets) < count:
            self.multi_invalidated_exploration_targets.extend([[] for _ in range(count - len(self.multi_invalidated_exploration_targets))])
        elif len(self.multi_invalidated_exploration_targets) > count:
            self.multi_invalidated_exploration_targets = self.multi_invalidated_exploration_targets[:count]

    def multi_reserved_exploration_targets(self, exclude_robot_index: int) -> list[tuple[float, float]]:
        """Return frontier targets already assigned to other robots.

        This is the minimal coordination layer: a robot may share the map with
        the team, but it must not share the same local frontier target F.
        """
        self.ensure_multi_exploration_target_slots()
        reserved: list[tuple[float, float]] = []
        for index, target in enumerate(self.multi_exploration_targets):
            if index == int(exclude_robot_index) or target is None:
                continue
            reserved.append((float(target[0]), float(target[1])))
        return reserved

    def multi_frontier_exclusion_radius(self) -> float:
        """Minimum distance between two reserved frontier targets.

        A radius of roughly two grid cells avoids duplicate or nearly duplicate
        F markers without being so large that robots starve in small maps.
        """
        return max(0.75, 2.0 * float(self.config.grid_resolution))

    def multi_dynamic_target_margin(self) -> float:
        """Extra clearance used when assigning frontiers around teammates."""
        return max(0.25, 0.5 * float(self.config.grid_resolution))

    def dynamic_robot_obstacles_for_target_selection(
        self,
        robot_index: int,
    ) -> list[tuple[float, float, float]]:
        """Return other runtime robots as dynamic disks for frontier selection."""
        robot_index = int(robot_index)
        disks: list[tuple[float, float, float]] = []
        for other_index, other in enumerate(self.robots):
            if other_index == robot_index:
                continue
            disks.append(
                (
                    float(other.x),
                    float(other.y),
                    float(self.safety_radius_for_robot(other)),
                )
            )
        return disks

    def target_is_clear_of_dynamic_robots(
        self,
        robot_index: int,
        target: tuple[float, float],
    ) -> tuple[bool, str]:
        """Validate that a proposed F_i is not inside a teammate safety zone."""
        if not (0 <= int(robot_index) < len(self.robots)):
            return False, "invalid robot index"

        robot = self.robots[int(robot_index)]
        ego_radius = float(self.safety_radius_for_robot(robot))
        margin = self.multi_dynamic_target_margin()

        for other_index, other in enumerate(self.robots):
            if other_index == int(robot_index):
                continue
            required = ego_radius + float(self.safety_radius_for_robot(other)) + margin
            distance = math.hypot(float(target[0]) - float(other.x), float(target[1]) - float(other.y))
            if distance <= required:
                return (
                    False,
                    f"target too close to R{other_index + 1} "
                    f"({distance:.2f} m < {required:.2f} m)",
                )

        return True, "target clear of dynamic robots"

    def target_is_clear_of_reserved_frontiers(
        self,
        robot_index: int,
        target: tuple[float, float],
    ) -> tuple[bool, str]:
        """Validate that F_i is not a near-duplicate of another reserved F_j."""
        radius = self.multi_frontier_exclusion_radius()
        for other_index, other_target in enumerate(self.multi_exploration_targets):
            if other_index == int(robot_index) or other_target is None:
                continue
            distance = math.hypot(float(target[0]) - float(other_target[0]), float(target[1]) - float(other_target[1]))
            if distance <= radius:
                return (
                    False,
                    f"target too close to F{other_index + 1} "
                    f"({distance:.2f} m < {radius:.2f} m)",
                )
        return True, "target clear of reserved frontiers"

    def target_is_clear_of_other_active_routes(
        self,
        robot_index: int,
        target: tuple[float, float],
    ) -> tuple[bool, str]:
        """Avoid assigning a frontier directly on a teammate's active path."""
        if not (0 <= int(robot_index) < len(self.robots)):
            return False, "invalid robot index"

        ego_radius = self.safety_radius_for_robot(self.robots[int(robot_index)])
        margin = self.multi_dynamic_target_margin()
        for other_index, other in enumerate(self.robots):
            if other_index == int(robot_index):
                continue
            route = self.current_route_points_for_robot(other)
            if len(route) < 2:
                continue
            required = ego_radius + self.safety_radius_for_robot(other) + margin
            for start, end in zip(route[:-1], route[1:]):
                distance = self.distance_point_to_segment(target, start, end)
                if distance <= required:
                    return (
                        False,
                        f"target too close to R{other_index + 1} active route "
                        f"({distance:.2f} m < {required:.2f} m)",
                    )
        return True, "target clear of active teammate routes"

    def multi_exploration_target_is_valid(
        self,
        robot_index: int,
        target: tuple[float, float],
    ) -> tuple[bool, str]:
        """Full validation for an already assigned or newly proposed F_i."""
        checks = (
            self.target_is_clear_of_reserved_frontiers,
            self.target_is_clear_of_dynamic_robots,
            self.target_is_clear_of_other_active_routes,
        )
        for check in checks:
            ok, reason = check(robot_index, target)
            if not ok:
                return False, reason
        return True, "target valid"

    def temporary_separation_target_for_robot(self, robot_index: int) -> tuple[float, float] | None:
        """Create a short-range separation target when robots start too close together."""
        if not (0 <= int(robot_index) < len(self.robots)):
            return None

        robot = self.robots[int(robot_index)]
        others = [other for idx, other in enumerate(self.robots) if idx != int(robot_index)]
        if not others:
            return None

        own_radius = float(self.safety_radius_for_robot(robot))
        max_other_radius = max((float(self.safety_radius_for_robot(other)) for other in others), default=own_radius)
        required_clearance = own_radius + max_other_radius + self.multi_dynamic_target_margin()
        nearest_distance = min(
            math.hypot(float(robot.x) - float(other.x), float(robot.y) - float(other.y))
            for other in others
        )
        if nearest_distance > max(required_clearance * 1.75, 1.0):
            return None

        centroid_x = sum(float(other.x) for other in others) / len(others)
        centroid_y = sum(float(other.y) for other in others) / len(others)
        dx = float(robot.x) - centroid_x
        dy = float(robot.y) - centroid_y
        if abs(dx) < 1e-8 and abs(dy) < 1e-8:
            dx, dy = 1.0, 0.0
        norm = math.hypot(dx, dy)
        if norm < 1e-8:
            dx, dy = 1.0, 0.0
            norm = 1.0

        step = max(required_clearance * 1.25, 0.75)
        target = (
            float(robot.x) + dx / norm * step,
            float(robot.y) + dy / norm * step,
        )
        target = (
            min(max(target[0], WORLD_X_MIN), WORLD_X_MAX),
            min(max(target[1], WORLD_Y_MIN), WORLD_Y_MAX),
        )
        return target

    def invalidated_frontiers_for_robot(self, robot_index: int) -> list[tuple[float, float]]:
        self.ensure_multi_exploration_target_slots()
        if not (0 <= int(robot_index) < len(self.multi_invalidated_exploration_targets)):
            return []
        return list(self.multi_invalidated_exploration_targets[int(robot_index)])

    def invalidate_current_multi_frontier(self, robot_index: int, reason: str = "") -> None:
        """Blacklist the current F_i for this robot and clear its assignment."""
        self.ensure_multi_exploration_target_slots()
        robot_index = int(robot_index)
        if not (0 <= robot_index < len(self.multi_exploration_targets)):
            return
        target = self.multi_exploration_targets[robot_index]
        if target is not None:
            invalid = self.multi_invalidated_exploration_targets[robot_index]
            target_tuple = (float(target[0]), float(target[1]))
            if all(math.hypot(target_tuple[0] - old[0], target_tuple[1] - old[1]) > 1e-6 for old in invalid):
                invalid.append(target_tuple)
            # Keep the blacklist bounded so a robot is not starved forever.
            if len(invalid) > 12:
                self.multi_invalidated_exploration_targets[robot_index] = invalid[-12:]
        self.multi_exploration_targets[robot_index] = None
        self.publish_multi_exploration_targets()

    def publish_multi_exploration_targets(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.set_multi_exploration_targets(self.multi_exploration_targets)

    def ensure_multi_route_state_slots(self) -> None:
        """Create per-robot route-state storage.

        This separates real navigation states from route-assignment messages.
        A robot that has no frontier must not keep asking A* for a fake
        one-cell route to its current position; it should be in an explicit
        HOLD/ STUCK state until a useful frontier or escape maneuver exists.
        """
        count = len(getattr(self, "robots", []))

        if not hasattr(self, "multi_route_states"):
            self.multi_route_states = []
        if len(self.multi_route_states) < count:
            self.multi_route_states.extend([self.ROUTE_STATE_ACTIVE] * (count - len(self.multi_route_states)))
        elif len(self.multi_route_states) > count:
            self.multi_route_states = self.multi_route_states[:count]

        if not hasattr(self, "multi_route_state_reasons"):
            self.multi_route_state_reasons = []
        if len(self.multi_route_state_reasons) < count:
            self.multi_route_state_reasons.extend([""] * (count - len(self.multi_route_state_reasons)))
        elif len(self.multi_route_state_reasons) > count:
            self.multi_route_state_reasons = self.multi_route_state_reasons[:count]

        if not hasattr(self, "multi_last_route_state_log_times"):
            self.multi_last_route_state_log_times = []
        if len(self.multi_last_route_state_log_times) < count:
            self.multi_last_route_state_log_times.extend([-1.0e9] * (count - len(self.multi_last_route_state_log_times)))
        elif len(self.multi_last_route_state_log_times) > count:
            self.multi_last_route_state_log_times = self.multi_last_route_state_log_times[:count]

    def set_multi_route_state(self, robot_index: int, state: str, reason: str = "", *, force_log: bool = False) -> None:
        """Set and log route-state transitions without spamming every frame."""
        self.ensure_multi_route_state_slots()
        robot_index = int(robot_index)
        if not (0 <= robot_index < len(self.multi_route_states)):
            return

        previous_state = self.multi_route_states[robot_index]
        previous_reason = self.multi_route_state_reasons[robot_index]
        reason = str(reason or "").strip()
        self.multi_route_states[robot_index] = str(state)
        self.multi_route_state_reasons[robot_index] = reason

        now = float(getattr(self, "simulation_time", 0.0))
        elapsed = now - float(self.multi_last_route_state_log_times[robot_index])
        changed = previous_state != state or previous_reason != reason
        if force_log or changed or elapsed >= 5.0:
            self.multi_last_route_state_log_times[robot_index] = now
            message = f"R{robot_index + 1} state={state}"
            if reason:
                message += f"; reason={reason}"
            self.log_console_message(message)

    def multi_goal_selection_is_hold(self, start_xy, goal_xy, reason: str) -> bool:
        """Detect a planner request that is really a hold/no-frontier state."""
        text = str(reason or "").lower()
        if (
            "no valid frontier" in text
            or "holding position" in text
            or "assigned frontier invalid" in text
        ):
            return True
        try:
            return math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1])) <= max(1e-6, 0.10 * float(self.config.grid_resolution))
        except Exception:
            return False

    def dynamic_robot_obstacle_points_for_robot(
        self,
        robot_index: int,
        samples_per_robot: int = 16,
    ) -> list[tuple[float, float]]:
        """
        Approximate every *other* robot as a dynamic obstacle point cloud.

        The path planner already knows how to avoid mapped obstacle points by
        inflating them with the current robot radius. To make another robot
        behave like a disk obstacle with its own radius, we sample its safety
        boundary plus its center. When the current robot's radius is applied by
        the planner, this approximates the required pairwise clearance
        r_i + r_j.
        """
        if not self.robots:
            return []

        points: list[tuple[float, float]] = []
        robot_index = int(robot_index)
        samples = max(8, int(samples_per_robot))

        for other_index, other in enumerate(self.robots):
            if other_index == robot_index:
                continue

            cx = float(other.x)
            cy = float(other.y)
            radius = max(0.02, float(self.safety_radius_for_robot(other)))
            points.append((cx, cy))

            for k in range(samples):
                angle = 2.0 * math.pi * k / samples
                points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

        return points

    @staticmethod
    def distance_point_to_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        """Distance from a point to a finite 2D segment."""
        px, py = float(point[0]), float(point[1])
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = clamp(t, 0.0, 1.0)
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        return math.hypot(px - closest_x, py - closest_y)

    def segment_violates_other_robot_clearance(
        self,
        robot_index: int,
        start: tuple[float, float],
        end: tuple[float, float] | None,
    ) -> tuple[bool, str]:
        """
        Check whether a proposed local segment would pass through another robot.

        This is separate from the hard pairwise position check. It treats other
        robots as dynamic obstacles before the robot commits to the next control,
        so routes do not simply cross through another robot's body/safety zone.
        """
        if end is None or not (0 <= int(robot_index) < len(self.robots)):
            return False, ""

        robot_index = int(robot_index)
        ego = self.robots[robot_index]
        ego_radius = self.safety_radius_for_robot(ego)

        for other_index, other in enumerate(self.robots):
            if other_index == robot_index:
                continue
            other_xy = (float(other.x), float(other.y))
            required = ego_radius + self.safety_radius_for_robot(other)
            distance = self.distance_point_to_segment(other_xy, start, end)
            if distance <= required:
                return (
                    True,
                    f"ROBOT OBSTACLE: R{robot_index + 1} local segment crosses R{other_index + 1} "
                    f"safety zone ({distance:.2f} m < {required:.2f} m).",
                )
        return False, ""

    def coordinator_runtime_profile(self):
        """Return the selected coordinator plugin's runtime profile, memoized.

        This is read from the per-frame multi-robot loop (path/control source
        selection), so it is cached per coordinator_type instead of re-running
        plugin discovery every frame.
        """
        strategy = str(self.config.coordinator_type)
        if getattr(self, "_cached_runtime_profile_strategy", None) != strategy:
            try:
                self._cached_runtime_profile = runtime_profile_for_strategy(strategy)
            except PluginLoadError:
                self._cached_runtime_profile = build_runtime_profile(
                    PluginMetadata(name=strategy, version="", description="", capabilities=())
                )
            self._cached_runtime_profile_strategy = strategy
        return self._cached_runtime_profile

    def multi_robot_coordination_states(self) -> list[RobotCoordinationState]:
        """Return plain robot state packets for the coordinator."""
        states: list[RobotCoordinationState] = []
        for robot in self.robots:
            states.append(
                RobotCoordinationState(
                    xy=(float(robot.x), float(robot.y)),
                    safety_radius=float(self.safety_radius_for_robot(robot)),
                    sensor_range=float(getattr(robot, "vision", self.config.vision)),
                    vision_model=str(self.config.vision_model),
                    theta=float(robot.theta),
                )
            )
        return states

    def multi_active_route_points_by_robot(self) -> list[list[tuple[float, float]]]:
        """Return the current active route of every robot for coordination."""
        routes: list[list[tuple[float, float]]] = []
        for robot in self.robots:
            try:
                routes.append(self.current_route_points_for_robot(robot))
            except Exception:
                routes.append([(float(robot.x), float(robot.y))])
        return routes

    def synchronize_multi_frontier_targets(
        self,
        requesting_robot_index: int,
        force_new_target: bool = False,
    ) -> None:
        """Assign missing frontier targets using the selected coordinator.

        The map is shared by the team, but each robot must own an independent
        local target F_i. This method is the bridge between the engine and
        robotics_sim.simulation.coordination.MultiRobotCoordinator.
        """
        if self.is_goal_seeking_mode():
            return

        self.ensure_multi_exploration_target_slots()
        requesting_robot_index = int(requesting_robot_index)

        if force_new_target and 0 <= requesting_robot_index < len(self.multi_exploration_targets):
            self.multi_exploration_targets[requesting_robot_index] = None

        robots_to_assign: list[int] = []
        for index, target in enumerate(self.multi_exploration_targets):
            if target is None:
                robots_to_assign.append(index)

        if force_new_target and requesting_robot_index not in robots_to_assign:
            robots_to_assign.append(requesting_robot_index)

        if not robots_to_assign:
            return

        coordinator = MultiRobotCoordinator(self.config.coordinator_type)

        # Per-robot explored footprints are required by the coordinated frontier
        # planner to penalize duplicated sensing.  Passing only the shared map is
        # not enough: it makes teammate-overlap ratios collapse to zero because
        # the planner cannot distinguish who already observed each cell.
        explored_points_by_robot = [
            list(self.belief_map.robot_explored_points(index))
            for index in range(len(self.robots))
        ]

        result = coordinator.assign_frontiers(
            planner_name=str(self.config.exploration_planner),
            robot_states=self.multi_robot_coordination_states(),
            existing_targets=list(self.multi_exploration_targets),
            robots_to_assign=robots_to_assign,
            invalidated_targets_by_robot=list(self.multi_invalidated_exploration_targets),
            explored_points=list(self.explored_free_points),
            mapped_obstacle_points=list(self.mapped_obstacle_points),
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=float(self.config.grid_resolution),
            final_goal_xy=self.final_goal_xy(),
            ipp_distance_penalty=float(self.config.ipp_distance_penalty),
            target_exclusion_radius=self.multi_frontier_exclusion_radius(),
            dynamic_obstacle_margin=self.multi_dynamic_target_margin(),
            route_points_by_robot=self.multi_active_route_points_by_robot(),
            explored_points_by_robot=explored_points_by_robot,
            goal_tolerance=float(self.config.goal_tolerance),
        )

        if not hasattr(self, "multi_robot_commands_by_id"):
            self.multi_robot_commands_by_id = {}
        commands_by_id = map_robot_commands_by_id(result.commands)
        self.multi_robot_commands_by_id.update(commands_by_id)

        # Preference order: command.target (richer, plugin-authoritative) ->
        # result.targets[index] (plain legacy field) -> the target the robot
        # already had. The third tier matters because a plugin only returns an
        # entry for the robots it was asked to (re)assign this call;
        # result.targets is None for every other robot, and blindly assigning
        # list(result.targets) would wipe out targets that were not part of
        # this batch.
        previous_targets = list(self.multi_exploration_targets)
        updated_targets: list[tuple[float, float] | None] = []
        for index in range(len(result.targets)):
            command = commands_by_id.get(index)
            if command is not None and command.target is not None:
                updated_targets.append(command.target)
            elif result.targets[index] is not None:
                updated_targets.append(result.targets[index])
            elif index < len(previous_targets):
                updated_targets.append(previous_targets[index])
            else:
                updated_targets.append(None)
        self.multi_exploration_targets = updated_targets

        registry = self.ensure_runtime_robot_registry()
        registry.sync_exploration_targets_from_legacy_list(self.multi_exploration_targets)
        self.robot_agents = registry.agents
        if 0 <= requesting_robot_index < len(result.reasons):
            self.last_goal_selection_reason = (
                f"R{requesting_robot_index + 1}: {result.reasons[requesting_robot_index]} "
                f"[{result.strategy}]"
            )
        self.publish_multi_exploration_targets()

    def select_navigation_goal_for_multi_robot(
        self,
        robot_index: int,
        start_xy: tuple[float, float],
        force_new_target: bool = False,
    ) -> tuple[tuple[float, float], str]:
        """Select a navigation target for exactly one robot.

        Important multi-robot rule:
            each robot owns its own frontier target F.

        Replanning due to safety should usually keep the same F and only
        recompute the path to it. Selecting a new F should happen when the
        current F is reached or when the robot has no assigned frontier yet.
        """
        final_goal = self.final_goal_xy()
        planner_name = str(self.config.exploration_planner)
        robot_index = int(robot_index)
        self.ensure_multi_exploration_target_slots()

        if is_goal_seeking_planner(planner_name):
            if 0 <= robot_index < len(self.multi_exploration_targets):
                self.multi_exploration_targets[robot_index] = None
            self.publish_multi_exploration_targets()
            return final_goal, f"R{robot_index + 1}: using shared final mission goal"

        if force_new_target and 0 <= robot_index < len(self.multi_exploration_targets):
            self.multi_exploration_targets[robot_index] = None

        existing_target = None
        if 0 <= robot_index < len(self.multi_exploration_targets):
            existing_target = self.multi_exploration_targets[robot_index]

        if existing_target is not None and not force_new_target:
            target = (float(existing_target[0]), float(existing_target[1]))
            still_valid, validity_reason = self.multi_exploration_target_is_valid(robot_index, target)
            if still_valid:
                self.publish_multi_exploration_targets()
                return target, f"R{robot_index + 1}: keeping assigned frontier F{robot_index + 1}"

            # A teammate may have moved into this frontier or reserved a nearby
            # one. Clear only this robot's F_i; do not disturb the other robots.
            self.invalidate_current_multi_frontier(robot_index, validity_reason)

        self.synchronize_multi_frontier_targets(
            requesting_robot_index=robot_index,
            force_new_target=force_new_target,
        )

        target = None
        if 0 <= robot_index < len(self.multi_exploration_targets):
            target = self.multi_exploration_targets[robot_index]

        if target is None:
            recovery_target = self.temporary_separation_target_for_robot(robot_index)
            if recovery_target is not None:
                ok, reason = self.multi_exploration_target_is_valid(robot_index, recovery_target)
                if ok:
                    self.multi_exploration_targets[robot_index] = recovery_target
                    self.publish_multi_exploration_targets()
                    return recovery_target, (
                        f"R{robot_index + 1}: temporary separation target while waiting for frontier"
                    )

            # Do not fall back to G while an exploration planner is selected.
            # The robot should hold its current position until a unique frontier exists.
            return (float(start_xy[0]), float(start_xy[1])), (
                f"R{robot_index + 1}: no valid frontier assigned by "
                f"{self.config.coordinator_type}; holding position"
            )

        target = (float(target[0]), float(target[1]))
        target_valid, target_valid_reason = self.multi_exploration_target_is_valid(robot_index, target)
        if not target_valid:
            self.multi_exploration_targets[robot_index] = None
            self.publish_multi_exploration_targets()
            return (float(start_xy[0]), float(start_xy[1])), (
                f"R{robot_index + 1}: assigned frontier invalid after validation; "
                f"{target_valid_reason}"
            )

        return target, (
            f"R{robot_index + 1}: frontier assigned by "
            f"{self.config.coordinator_type}"
        )

    def build_planner_kwargs_for_multi_robot(
        self,
        robot_index: int,
        force_new_exploration_target: bool = False,
    ) -> tuple[dict, str]:
        """Build planner inputs for one robot, including other robots as obstacles."""
        robot_index = int(robot_index)
        robot = self.robots[robot_index]
        start_xy = (float(robot.x), float(robot.y))

        agent = self.runtime_agent(robot_index)
        if agent is not None:
            agent.set_position(start_xy)
            agent.set_heading(float(robot.theta))

        goal_xy, goal_reason = self.select_navigation_goal_for_multi_robot(
            robot_index,
            start_xy,
            force_new_target=force_new_exploration_target,
        )

        if self.is_exploration_mode() and self.multi_goal_selection_is_hold(start_xy, goal_xy, goal_reason):
            return dict(
                __hold__=True,
                __hold_reason__=goal_reason,
                planner_type=self.config.planner_type,
                start_xy=start_xy,
                goal_xy=start_xy,
                planning_grid=None,
                obstacles=[],
                bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
                resolution=float(self.config.grid_resolution),
                robot_radius=float(self.safety_radius_for_robot(robot)),
                obstacle_points=[],
            ), goal_reason

        self.force_robot_pose_free_in_belief(robot_index)
        dynamic_points = self.dynamic_robot_obstacle_points_for_robot(robot_index)
        resolution = float(self.config.grid_resolution)
        robot_radius = float(self.safety_radius_for_robot(robot))
        obstacle_points, removed = self.sanitize_planner_obstacle_points(
            list(self.mapped_obstacle_points) + dynamic_points,
            start_xy=start_xy,
            robot_radius=robot_radius,
            resolution=resolution,
        )

        if removed:
            goal_reason = f"{goal_reason}; ignored {removed} own-start obstacle sample(s) for planning"

        planning_grid = self.build_planning_grid_for_robot(
            robot,
            obstacle_points=obstacle_points,
            robot_radius=robot_radius,
        )

        kwargs = dict(
            planner_type=self.config.planner_type,
            start_xy=start_xy,
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            obstacles=[],
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=resolution,
            robot_radius=robot_radius,
            planning_grid=planning_grid,
            unknown_is_traversable=True,
            obstacle_points=[],
        )
        return kwargs, goal_reason

    def compute_route_for_multi_robot(
        self,
        robot_index: int,
        force_new_exploration_target: bool = False,
    ) -> tuple[bool, str, list[tuple[float, float]]]:
        """Compute one robot's route.

        If the selected coordinator plugin owns PATH_PLANNING and supplied a
        usable command.path for this robot, that path is authoritative and the
        external A*/Direct planner below is never invoked. Otherwise (this is
        the case for MMPF and NOIC legacy today, since neither declares
        PATH_PLANNING) the external planner runs exactly as before.
        """

        def _legacy_route() -> tuple[bool, str, list[tuple[float, float]]]:
            planner_kwargs, goal_reason = self.build_planner_kwargs_for_multi_robot(
                robot_index,
                force_new_exploration_target=force_new_exploration_target,
            )
            if bool(planner_kwargs.get("__hold__", False)):
                return False, goal_reason, []

            goal_xy = tuple(planner_kwargs["goal_xy"])

            if self.config.planner_type == "Direct":
                return True, f"direct route; {goal_reason}", [goal_xy]

            if compute_planned_waypoints is None:
                return False, "planner package is not available", []

            success, reason, waypoints = self.call_compute_planned_waypoints(
                planner_kwargs,
                path_simplifier=self.config.path_simplifier,
            )

            return success, f"{goal_reason}; {reason}", waypoints

        profile = self.coordinator_runtime_profile()
        command = getattr(self, "multi_robot_commands_by_id", {}).get(int(robot_index))
        success, reason, waypoints = select_runtime_path_source(profile, command, _legacy_route)
        if profile.owns_path_planning and "fallback" in reason:
            self.log_console_message(f"R{int(robot_index) + 1}: {reason}")
        return success, reason, waypoints

    def segment_clear_for_robot_against_points(
        self,
        robot,
        start: tuple[float, float],
        end: tuple[float, float],
        obstacle_points: list[tuple[float, float]],
    ) -> bool:
        collision_checker = getattr(self, "collision_checker", None)
        if collision_checker is None:
            return True
        report = collision_checker.check_segment_points(
            start=(float(start[0]), float(start[1])),
            end=(float(end[0]), float(end[1])),
            obstacle_points=list(obstacle_points),
            robot_radius=float(self.safety_radius_for_robot(robot)),
        )
        return not bool(report.collision)

    def planning_line_of_sight_clear_for_robot(
        self,
        robot,
        target: tuple[float, float],
        obstacle_points: list[tuple[float, float]],
    ) -> bool:
        """Check a current-pose shortcut against the same derived grid as A*/Dijkstra.

        Production uses BeliefMap occupancy, UNKNOWN policy, inflated mapped
        obstacles and the hazard layer. Lightweight test doubles that do not
        expose the grid builder fall back to the existing point-clearance
        check rather than inventing a second map representation.
        """
        start = (float(robot.x), float(robot.y))
        builder = getattr(self, "build_planning_grid_for_robot", None)
        if callable(builder):
            planning_grid = builder(
                robot,
                obstacle_points=list(obstacle_points),
                robot_radius=float(self.safety_radius_for_robot(robot)),
            )
            start_cell = planning_grid.world_to_grid(*start, clamp=True)
            target_cell = planning_grid.world_to_grid(
                float(target[0]), float(target[1]), clamp=True
            )
            if not planning_grid.in_bounds(start_cell) or not planning_grid.in_bounds(target_cell):
                return False
            # The physical robot already occupies the start position. A stale
            # quantized obstacle sample must not prevent a local visibility
            # query, matching planner_registry's start-cell handling.
            if not planning_grid.is_traversable(start_cell):
                planning_grid.set_value(start_cell, FREE)
            if not planning_grid.is_traversable(target_cell):
                return False
            return bool(line_of_sight_grid_safe(planning_grid, start_cell, target_cell))

        return SimulationControllerMixin.segment_clear_for_robot_against_points(
            self, robot, start, target, obstacle_points
        )

    def clean_waypoints_for_robot(
        self,
        robot,
        waypoints: list[tuple[float, float]],
        obstacle_points: list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, float]]:
        """Clean waypoints using a specific robot pose and obstacle context."""
        if robot is None or not waypoints:
            return [tuple(point) for point in waypoints]

        start = (float(robot.x), float(robot.y))
        raw_points = [tuple((float(point[0]), float(point[1]))) for point in waypoints]
        cleaned: list[tuple[float, float]] = []

        for point in raw_points:
            if math.hypot(point[0] - start[0], point[1] - start[1]) <= 1e-6:
                continue
            if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) <= 1e-6:
                continue
            cleaned.append(point)

        if not cleaned:
            return []

        points_for_clearance = list(self.mapped_obstacle_points if obstacle_points is None else obstacle_points)
        hazard_service = getattr(self, "hazard_service", None)
        if hazard_service is not None:
            points_for_clearance.extend(hazard_service.observed_blocked_world_points())

        # Runtime invariant: if the final target is directly safe from the
        # robot's CURRENT pose, never execute an older/grid-artifact detour.
        # Raw/simplified planner paths remain available in diagnostics; only
        # the executable route is shortened. This also fixes prefetched routes
        # whose start pose became stale before promotion.
        final_target = cleaned[-1]
        if (
            getattr(self, "collision_checker", None) is not None
            and SimulationControllerMixin.planning_line_of_sight_clear_for_robot(
                self, robot, final_target, points_for_clearance
            )
        ):
            return [final_target]

        if self.config.path_simplifier != "Line of sight grid-safe":
            return cleaned

        simplified: list[tuple[float, float]] = []
        current = start
        index = 0

        while index < len(cleaned):
            farthest_visible = index
            for candidate_index in range(len(cleaned) - 1, index - 1, -1):
                candidate = cleaned[candidate_index]
                if self.segment_clear_for_robot_against_points(robot, current, candidate, points_for_clearance):
                    farthest_visible = candidate_index
                    break
            next_point = cleaned[farthest_visible]
            simplified.append(next_point)
            current = next_point
            index = farthest_visible + 1

        return simplified

    def compute_route(self, start_xy: tuple[float, float]) -> tuple[bool, str, list[tuple[float, float]]]:
        """
        Ask the selected planner for world-coordinate waypoints.

        This synchronous version is still used for initial startup and explicit
        goal changes. Replanning during motion uses PlannerWorker so expensive
        A*/Dijkstra calls do not freeze the GUI thread.
        """
        planner_kwargs = self.build_planner_kwargs(start_xy)
        if bool(planner_kwargs.get("__hold__", False)):
            self._nav_debug_last_plan_capture = None
            return False, str(planner_kwargs.get("__hold_reason__", "holding position")), []
        goal_xy = tuple(planner_kwargs["goal_xy"])

        if self.config.planner_type == "Direct":
            self._nav_debug_last_plan_capture = None
            return True, f"direct route; {self.last_goal_selection_reason}", [goal_xy]

        if compute_planned_waypoints is None:
            self._nav_debug_last_plan_capture = None
            return False, "planner package is not available", []

        # Stashed on self (not returned/passed as a parameter) so this
        # method's call signature never changes -- several existing tests
        # replace compute_route()/apply_route_result() with fixed-arity
        # lambdas; apply_route_result() reads this back via getattr(). See
        # _finalize_navigation_debug_snapshot() docstring for the same
        # defensive getattr pattern.
        #
        # Always built now, regardless of navigation_debug_enabled: capture
        # happens on every tick unconditionally, so history already has
        # data by the time a user turns Navigation on mid-run (that switch
        # now only gates the UI's browse/inspect/restore affordances).
        debug_capture = PlanDebugCapture()
        result = self.call_compute_planned_waypoints(
            planner_kwargs,
            path_simplifier=self.config.path_simplifier,
            debug_capture=debug_capture,
        )
        self._nav_debug_last_plan_capture = debug_capture
        return result

    def planner_label(self) -> str:
        exploration = self.config.exploration_planner
        if self.config.planner_type == "Direct":
            return f"Direct + {exploration}"
        return f"{self.config.planner_type} / {self.config.path_simplifier} + {exploration}"

    def segment_clear_against_current_map(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        """
        Return True when a continuous segment is safe with respect to the
        robot's current partial map.

        This is intentionally checked against mapped_obstacle_points, not the
        ground-truth rectangles. The robot should be allowed to plan through
        unknown space; if a hidden obstacle is discovered later, the safety
        layer will trigger replanning. Hazard is held to the same rule: only
        observed_blocked_world_points() (discovered), never the omniscient
        ground-truth HazardField.
        """
        if self.collision_checker is None:
            return True

        obstacle_points = list(self.mapped_obstacle_points)
        hazard_service = getattr(self, "hazard_service", None)
        if hazard_service is not None:
            obstacle_points.extend(hazard_service.observed_blocked_world_points())

        report = self.collision_checker.check_segment_points(
            start=(float(start[0]), float(start[1])),
            end=(float(end[0]), float(end[1])),
            obstacle_points=obstacle_points,
            robot_radius=float(self.safety_radius()),
        )
        return not bool(report.collision)

    def clean_waypoints_for_current_start(
        self,
        waypoints: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Clean planner output using the robot's actual continuous pose.

        Why this exists:
            A*/Dijkstra work on cell centers. Even after an aggressive
            simplifier, the first returned waypoint can be an artificial cell
            center beside the robot. Visually this looks like a useless target
            between S and F/G, and dynamically it can make the unicycle turn in
            a way that was not intended.

        Policy:
            - Always remove near-duplicate consecutive points.
            - In Line-of-sight mode, greedily collapse waypoints using the
              continuous partial map. If the real segment robot -> F/G is safe,
              the route becomes exactly [F/G].
            - Do not apply this aggressive cleanup to conservative modes; those
              modes should preserve the grid route topology for comparison.
        """
        if self.robot is None or not waypoints:
            return [tuple(point) for point in waypoints]

        start = (float(self.robot.x), float(self.robot.y))
        raw_points = [tuple((float(point[0]), float(point[1]))) for point in waypoints]

        # Remove consecutive duplicates and points already essentially reached
        # from the real robot pose.
        cleaned: list[tuple[float, float]] = []
        for point in raw_points:
            if math.hypot(point[0] - start[0], point[1] - start[1]) <= 1e-6:
                continue
            if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) <= 1e-6:
                continue
            cleaned.append(point)

        if not cleaned:
            return []

        # The executable route must never preserve a huge detour when the
        # intended final target is directly safe from the CURRENT pose. This
        # is a post-planning invariant, not a replacement for A*/Dijkstra; the
        # raw and configured-simplifier paths stay visible in diagnostics.
        if (
            getattr(self, "collision_checker", None) is not None
            and SimulationControllerMixin.segment_clear_against_current_map(
                self, start, cleaned[-1]
            )
        ):
            return [cleaned[-1]]

        # For non-direct cases, preserve the user's selected simplifier.
        if self.config.path_simplifier != "Line of sight grid-safe":
            return cleaned

        simplified: list[tuple[float, float]] = []
        current = start
        index = 0

        while index < len(cleaned):
            farthest_visible = index

            for candidate_index in range(len(cleaned) - 1, index - 1, -1):
                candidate = cleaned[candidate_index]
                if self.segment_clear_against_current_map(current, candidate):
                    farthest_visible = candidate_index
                    break

            next_point = cleaned[farthest_visible]
            simplified.append(next_point)
            current = next_point
            index = farthest_visible + 1

        return simplified

    @_timed_method("route_result_handling")
    def apply_route_result(
        self,
        success: bool,
        reason: str,
        waypoints: list[tuple[float, float]],
    ) -> None:
        """Times route-result handling (see _timed_method()) regardless of
        whether this is called synchronously within simulation_step() or
        later, from a queued Qt-signal callback delivered on a separate
        event-loop turn (the common case for async planner results)."""
        if self.robot is None:
            return

        # Consumed unconditionally (not just on the success path below) so a
        # capture from one call never leaks into a later, unrelated
        # apply_route_result() call (e.g. a hold/failure result that never
        # reaches the success branch that would otherwise clear it).
        pending_plan_capture = getattr(self, "_nav_debug_last_plan_capture", None)
        self._nav_debug_last_plan_capture = None

        self.route_result_count += 1

        if success and waypoints:
            clean_waypoints = self.clean_waypoints_for_current_start(waypoints)

            if not clean_waypoints:
                if self.is_exploration_mode():
                    clean_waypoints = [(float(self.robot.x), float(self.robot.y))]
                else:
                    clean_waypoints = [self.final_goal_xy()]

            # Reject a route whose very first segment is already unsafe by the
            # same rule build_observation() uses for active_segment_blocked.
            # Accepting it anyway would let the very next tick's safety check
            # immediately trip REPLAN_FOR_SAFETY again for the route we just
            # assigned, producing a safety-replan loop instead of a working
            # route. Falls through to the shared failure-handling code below.
            #
            # Computed via _evaluate_route_first_segment() (not the bool-only
            # route_first_segment_blocked() wrapper) so the full CollisionReport
            # -- blocking point, distance, reason -- survives for the
            # navigation debug snapshot below instead of being reduced to a
            # bare bool. Same single computation either way.
            start_xy = (float(self.robot.x), float(self.robot.y))
            first_segment_report = _evaluate_route_first_segment(
                self.collision_checker,
                start_xy,
                clean_waypoints[0] if clean_waypoints else None,
                list(self.mapped_obstacle_points),
                self.safety_radius(),
            )
            blocked_on_arrival = bool(clean_waypoints) and bool(
                first_segment_report is not None and first_segment_report.collision
            )

            # Reject a route that claims success but whose final waypoint
            # does not actually reach the exploration target it was asked
            # to route to (e.g. compute_planned_waypoints() silently
            # relocated an occupied goal cell to the nearest traversable
            # one). Accepting it anyway would follow the route to a
            # different endpoint and then get stuck there: active_path_goal_xy
            # would still point at the original, unreached target, so
            # "frontier reached" never fires and STATE keeps showing a
            # stale target/path_goal forever.
            misses_intended_goal = False
            endpoint_reaches_goal_debug = None
            if not blocked_on_arrival and self.is_exploration_mode() and clean_waypoints:
                intended_goal = self.current_exploration_target
                if intended_goal is not None:
                    misses_intended_goal = not route_reaches_goal(
                        clean_waypoints, intended_goal, float(self.config.goal_tolerance)
                    )
                    endpoint_reaches_goal_debug = not misses_intended_goal

            # Navigation debug diagnostics -- observational, never affects
            # blocked_on_arrival/misses_intended_goal or any acceptance
            # decision below. Captured unconditionally now (see
            # _finalize_navigation_debug_snapshot()'s docstring): navigation_
            # debug_enabled gates the UI's browse/inspect/restore
            # affordances, not whether a tick gets recorded, so history is
            # already populated by the time a user turns Navigation on.
            nav_capture = NavigationDebugCapture(plan=pending_plan_capture)
            if first_segment_report is not None:
                nav_capture.first_segment = clearance_terms_from_report(
                    first_segment_report,
                    checker="check_segment_points",
                    required_clearance=self.safety_radius(),
                )
            nav_capture.endpoint_reaches_goal = endpoint_reaches_goal_debug
            # getattr-guarded (not a direct self._finalize_navigation_debug_
            # snapshot(...) call): many lightweight duck-typed engine fakes
            # elsewhere in the test suite bind only the handful of
            # SimulationControllerMixin methods relevant to what they
            # actually test, not this one -- capture being unconditional
            # now (see that method's docstring) means this call site is
            # reached unconditionally too, so it must not assume the method
            # is bound.
            _nav_debug_finalize = getattr(self, "_finalize_navigation_debug_snapshot", None)
            if callable(_nav_debug_finalize):
                _nav_debug_finalize(
                    agent=self.runtime_agent(None),
                    decision_kind="ROUTE_RESULT",
                    decision_reason=reason,
                    event_kind=(
                        NavigationDebugEventKind.ROUTE_REJECTED
                        if (blocked_on_arrival or misses_intended_goal)
                        else NavigationDebugEventKind.PLAN_ACCEPTED
                    ),
                    capture=nav_capture,
                )
            # Persisted so subsequent ticks' snapshots (which do not
            # recompute a plan) can still show which planner/simplifier
            # produced the route currently being executed, instead of
            # reporting "unavailable" on every tick except the one the
            # route was actually accepted on. Cleared on the next
            # accepted/rejected plan, and by the same reset_simulation_
            # state() points that clear navigation_debug_log.
            if pending_plan_capture is not None and not (blocked_on_arrival or misses_intended_goal):
                self._nav_debug_last_accepted_plan = pending_plan_capture

            if blocked_on_arrival:
                reason = f"{reason}; rejected: first segment blocked on arrival"
                diag_agent = self.runtime_agent(None)
                if diag_agent is not None:
                    diag_agent.first_segment_blocked_count += 1
            elif misses_intended_goal:
                reason = f"{reason}; rejected: final waypoint does not reach path goal"
            else:
                # Belt-and-suspenders: if the planner returned the target the robot
                # just reached (hysteresis slipped through), refuse to reassign the
                # same route.  This prevents the infinite REQUEST_PLAN loop when
                # exploration hysteresis returns "kind=current" with length=0.
                if self.is_exploration_mode() and self.robot is not None and clean_waypoints:
                    agent_check = self.runtime_agent(None)
                    new_goal = clean_waypoints[-1]
                    old_goal = getattr(agent_check, "active_path_goal_xy", None) if agent_check is not None else None
                    robot_xy = (float(self.robot.x), float(self.robot.y))
                    same_target_radius = max(
                        float(self.config.grid_resolution),
                        2.0 * float(self.config.goal_tolerance),
                    )
                    if (
                        old_goal is not None
                        and math.hypot(new_goal[0] - old_goal[0], new_goal[1] - old_goal[1]) <= same_target_radius
                        and math.hypot(robot_xy[0] - old_goal[0], robot_xy[1] - old_goal[1]) <= float(self.config.goal_tolerance) * 2.0
                    ):
                        self.log_console_message(
                            f"[NAV] apply_route_result: planner returned already-reached target "
                            f"{new_goal}; forcing re-search."
                        )
                        if agent_check is not None:
                            agent_check.exploration_target_xy = None
                            agent_check.invalidate_route(reason="planner returned completed target; forcing re-search")
                            self._invalidate_prefetch_request(0, reason="planner returned completed target; forcing re-search")
                        self.current_exploration_target = None
                        self.canvas.set_exploration_target(None)
                        return

                if hasattr(self.robot, "set_waypoints"):
                    self.robot.set_waypoints(clean_waypoints)
                elif hasattr(self.robot, "set_goal"):
                    self.robot.set_goal(clean_waypoints[-1])
                else:
                    self.robot.goal = np.array(clean_waypoints[-1], dtype=float)

                # Sync RobotAgent so agent.active_target() is non-None next frame.
                # Without this, agent.step() keeps emitting REQUEST_PLAN because
                # agent.waypoints is never populated.
                agent = self.runtime_agent(None)
                if agent is not None and clean_waypoints:
                    agent.assign_path(
                        target=clean_waypoints[-1],
                        waypoints=clean_waypoints,
                        planner_reason=reason,
                    )

                self.canvas.set_planned_path([(self.robot.x, self.robot.y)] + clean_waypoints)
                if self.is_exploration_mode() and clean_waypoints:
                    self.canvas.set_exploration_target(clean_waypoints[-1])
                # Verbose legacy detail: debug-only console line. Normal
                # mode gets the compact [ROUTE ok] line from
                # log_route_assignment() below instead.
                self.telemetry.debug(
                    f"Planner: {self.planner_label()}. {self.last_goal_selection_reason}. {reason}. "
                    f"Mapped points: {len(self.mapped_obstacle_points)}."
                )
                self.log_route_assignment(
                    None,
                    (float(self.robot.x), float(self.robot.y)),
                    clean_waypoints,
                    f"{self.last_goal_selection_reason}; {reason}",
                )
                return

        if self.is_exploration_mode():
            hold_xy = (float(self.robot.x), float(self.robot.y))
            if hasattr(self.robot, "set_waypoints"):
                self.robot.set_waypoints([hold_xy])
            elif hasattr(self.robot, "set_goal"):
                self.robot.set_goal(hold_xy)
            else:
                self.robot.goal = np.array(hold_xy, dtype=float)

            # Keep agent in sync: no path, no stale goal, and no stale
            # exploration target -- otherwise desired_target_from_mode()
            # keeps returning the target that just failed to plan, and the
            # agent immediately re-requests a plan for it next tick.
            agent = self.runtime_agent(None)
            attempted_target = agent.exploration_target_xy if agent is not None else None
            if agent is not None:
                agent.invalidate_failed_exploration_route(
                    reason=f"planner failed: {reason}",
                    current_time=float(self.simulation_time),
                    map_signature=len(self.mapped_obstacle_points),
                )
                self._invalidate_prefetch_request(0, reason=f"planner failed: {reason}")

            self.current_exploration_target = None
            self.canvas.set_exploration_target(None)
            self.canvas.set_planned_path([hold_xy])
            self.canvas.set_status(
                f"Planner failed in exploration mode: {reason}. Holding current position; not falling back to G."
            )
            self.route_failure_count = getattr(self, "route_failure_count", 0) + 1
            self.telemetry.report_route_failure(
                robot_label="R1",
                start_xy=hold_xy,
                attempted_target=attempted_target,
                reason=reason,
                planner_type=str(self.config.planner_type),
                mapped_obstacle_count=len(self.mapped_obstacle_points),
            )
            # Opt-in terminal trace only (ROBOT_TRACE=route).
            _emit_robot_trace(
                self,
                "trace_route",
                sim_time=float(getattr(self, "simulation_time", 0.0)),
                robot_label="R1",
                result="fail",
                start=hold_xy,
                goal=attempted_target,
                reason=slug_route_failure_reason(reason),
                mapped_obstacle_count=len(getattr(self, "mapped_obstacle_points", [])),
            )
            return

        # Goal-seeking failure: fall back to direct goal.
        goal_xy = self.final_goal_xy()
        if hasattr(self.robot, "set_goal"):
            self.robot.set_goal(goal_xy)
        else:
            self.robot.goal = np.array(goal_xy, dtype=float)

        # Sync agent to the fallback waypoint so it emits FOLLOW_PATH next
        # frame instead of re-requesting a plan and looping forever.
        agent = self.runtime_agent(None)
        if agent is not None:
            agent.assign_path(
                target=goal_xy,
                waypoints=[goal_xy],
                planner_reason=f"fallback direct: {reason}",
            )

        self.canvas.set_planned_path([(self.robot.x, self.robot.y), goal_xy])
        self.canvas.set_status(
            f"Planner failed: {reason}. Falling back to direct goal."
        )

    def _navigation_debug_event_kind_for_decision(
        self, decision, predicted_report
    ) -> NavigationDebugEventKind:
        """Map a per-tick NavigationDecision (+ this tick's predicted-collision
        report, if any) to the navigation-debug event vocabulary.

        Reuses the exact "exploration exhausted" substring check
        apply_navigation_decision() already uses for its own EXHAUSTION_DIAG
        logging (see the `kind == "HOLD" and "exploration exhausted" in
        str(decision.reason)` condition further down in this file) rather
        than inventing a second detection rule for the same condition.
        """
        if predicted_report is not None and getattr(predicted_report, "collision", False):
            return NavigationDebugEventKind.PREDICTED_COLLISION
        kind = str(getattr(decision, "kind", ""))
        reason = str(getattr(decision, "reason", ""))
        if kind == "REPLAN_FOR_SAFETY":
            return NavigationDebugEventKind.SAFETY_REPLAN
        if kind == "HOLD":
            if "exploration exhausted" in reason:
                return NavigationDebugEventKind.EXHAUSTED
            return NavigationDebugEventKind.HOLD
        return NavigationDebugEventKind.TICK

    def _navigation_debug_belief_frame(self) -> Maybe[BeliefMapDebug]:
        """Return a compact immutable map frame.

        Navigation history records every simulation tick. Copying a full grid on
        every tick would make the replay prohibitively large, so the expensive
        grid/explored_by_robot compression is cached and reused across ticks
        that share the same BeliefMap.revision.

        visit_count/last_seen are compressed fresh on every call instead:
        BeliefMap.revision explicitly does NOT bump for visit-count/last-seen-
        only changes (see its own docstring), so reusing the revision-keyed
        cache for them would silently restore an earlier tick's visit history
        under a later tick's snapshot. These two arrays are small (uint16/
        float32, one grid-sized array each), so compressing them every call
        is cheap relative to the grid/explored cache this preserves.
        """
        belief = getattr(self, "belief_map", None)
        if belief is None:
            return Maybe.missing()

        revision = int(getattr(belief, "revision", 0))
        cache_key = (id(belief), revision, int(getattr(belief, "robot_count", 1)))
        cached_key = getattr(self, "_nav_debug_belief_frame_key", None)
        cached_parts = getattr(self, "_nav_debug_belief_frame_cache", None)
        if cached_key == cache_key and cached_parts is not None:
            resolution, bounds, grid_shape, grid_zlib, explored_shape, explored_packbits_zlib = cached_parts
        else:
            grid = np.ascontiguousarray(belief.grid, dtype=np.int8)
            explored = np.ascontiguousarray(belief.explored_by_robot, dtype=np.uint8)
            if explored.shape[0] == 1 and not np.any(explored):
                known_cells = grid != UNKNOWN
                if np.any(known_cells):
                    explored = explored.copy()
                    explored[0] = known_cells.astype(np.uint8, copy=False)
            packed_explored = np.packbits(explored.reshape(-1), bitorder="little")
            resolution = float(belief.resolution)
            bounds = tuple(float(v) for v in belief.bounds)
            grid_shape = (int(grid.shape[0]), int(grid.shape[1]))
            grid_zlib = zlib.compress(grid.tobytes(order="C"), level=1)
            explored_shape = (
                int(explored.shape[0]),
                int(explored.shape[1]),
                int(explored.shape[2]),
            )
            explored_packbits_zlib = zlib.compress(packed_explored.tobytes(), level=1)
            cached_parts = (resolution, bounds, grid_shape, grid_zlib, explored_shape, explored_packbits_zlib)
            self._nav_debug_belief_frame_key = cache_key
            self._nav_debug_belief_frame_cache = cached_parts

        visit_count = np.ascontiguousarray(belief.visit_count, dtype=np.uint16)
        last_seen = np.ascontiguousarray(belief.last_seen, dtype=np.float32)
        frame = BeliefMapDebug(
            revision=revision,
            resolution=resolution,
            bounds=bounds,
            grid_shape=grid_shape,
            grid_zlib=grid_zlib,
            explored_shape=explored_shape,
            explored_packbits_zlib=explored_packbits_zlib,
            visit_count_zlib=zlib.compress(visit_count.tobytes(order="C"), level=1),
            last_seen_zlib=zlib.compress(last_seen.tobytes(order="C"), level=1),
        )
        return Maybe.of(frame)

    def _navigation_debug_hazard_frame(self) -> Maybe[HazardDebug]:
        """Freeze the current hazard-field state (FireSources + next_fire_id)
        for restore. Unlike the belief grid, fire counts are small enough
        that rebuilding this tuple fresh every tick is cheap -- no revision-
        keyed cache needed here."""
        service = getattr(self, "hazard_service", None)
        if service is None:
            return Maybe.missing()
        field = service.field
        sources = tuple(
            HazardSourceDebug(
                fire_id=int(source.fire_id),
                position=(float(source.position[0]), float(source.position[1])),
                intensity=float(source.intensity),
                radius=float(source.radius),
            )
            for source in field.sources()
        )
        return Maybe.of(
            HazardDebug(
                version=int(field.version),
                next_fire_id=int(field.next_fire_id),
                sources=sources,
            )
        )

    def _navigation_debug_hazard_belief_frame(self) -> Maybe[HazardBeliefDebug]:
        """Return a compact immutable Team HazardBelief frame, reusing it
        until the belief actually changes -- same revision-keyed cache
        pattern as _navigation_debug_belief_frame() (BeliefMapDebug).

        Never reads HazardField/FireSource -- only HazardBelief.snapshot(),
        deliberately separate from _navigation_debug_hazard_frame() above
        (ground truth). The cache key includes id(belief) and shape/
        robot_count (not just revision) so a reset that recreates the
        RuntimeHazardService/HazardBelief, or a grid-resolution/robot-count
        change, invalidates it even on the rare chance a fresh belief's
        revision coincides with the previous one's.
        """
        hazard_service = getattr(self, "hazard_service", None)
        if hazard_service is None:
            return Maybe.missing()
        belief = hazard_service.belief

        revision = int(belief.revision)
        shape = (int(belief.shape[0]), int(belief.shape[1]))
        robot_count = int(belief.robot_count)
        cache_key = (id(belief), revision, shape, robot_count)
        cached_key = getattr(self, "_nav_debug_hazard_belief_frame_key", None)
        cached_frame = getattr(self, "_nav_debug_hazard_belief_frame_cache", None)
        if cached_key == cache_key and cached_frame is not None:
            return Maybe.of(cached_frame)

        frame = belief.snapshot()
        values = np.ascontiguousarray(frame.values, dtype=np.float32)
        observed = np.ascontiguousarray(frame.observed, dtype=bool)
        observed_by_robot = np.ascontiguousarray(frame.observed_by_robot, dtype=bool)
        packed_observed = np.packbits(observed.reshape(-1), bitorder="little")
        packed_observed_by_robot = np.packbits(observed_by_robot.reshape(-1), bitorder="little")

        debug_frame = HazardBeliefDebug(
            shape=shape,
            robot_count=robot_count,
            revision=revision,
            values_zlib=zlib.compress(values.tobytes(order="C"), level=1),
            observed_packbits_zlib=zlib.compress(packed_observed.tobytes(), level=1),
            observed_by_robot_packbits_zlib=zlib.compress(packed_observed_by_robot.tobytes(), level=1),
        )
        self._nav_debug_hazard_belief_frame_key = cache_key
        self._nav_debug_hazard_belief_frame_cache = debug_frame
        return Maybe.of(debug_frame)

    def _navigation_debug_agent_state_frame(self, agent) -> Maybe[AgentStateDebug]:
        """Freeze the RobotAgent fields restore needs explicitly -- see
        AgentStateDebug's docstring for exactly what is and is not included
        and why."""
        if agent is None:
            return Maybe.missing()
        final_goal = getattr(agent, "final_goal_xy", None)
        exploration_target = getattr(agent, "exploration_target_xy", None)
        active_path_goal = getattr(agent, "active_path_goal_xy", None)
        return Maybe.of(
            AgentStateDebug(
                final_goal_xy=(float(final_goal[0]), float(final_goal[1])) if final_goal is not None else None,
                exploration_target_xy=(
                    (float(exploration_target[0]), float(exploration_target[1]))
                    if exploration_target is not None
                    else None
                ),
                active_path_goal_xy=(
                    (float(active_path_goal[0]), float(active_path_goal[1]))
                    if active_path_goal is not None
                    else None
                ),
                active_path_mode=getattr(agent, "active_path_mode", None),
                route_generation=int(getattr(agent, "route_generation", 0)),
                route_affected_replan_count=int(getattr(agent, "route_affected_replan_count", 0)),
                first_segment_blocked_count=int(getattr(agent, "first_segment_blocked_count", 0)),
                last_frontier_candidate_count=int(getattr(agent, "last_frontier_candidate_count", 0)),
                prefetch_success_count=int(getattr(agent, "prefetch_success_count", 0)),
                prefetch_fail_count=int(getattr(agent, "prefetch_fail_count", 0)),
                safety_replan_count=int(getattr(agent, "safety_replan_count", 0)),
                target_switch_count=int(getattr(agent, "target_switch_count", 0)),
            )
        )

    def _navigation_debug_metrics_frame(self) -> Maybe[RuntimeMetricsDebug]:
        """Freeze engine-level cumulative counters -- see RuntimeMetricsDebug's
        docstring for why these must travel with simulation_time."""
        return Maybe.of(
            RuntimeMetricsDebug(
                total_distance_traveled=float(getattr(self, "total_distance_traveled", 0.0)),
                route_request_count=int(getattr(self, "route_request_count", 0)),
                route_result_count=int(getattr(self, "route_result_count", 0)),
                route_failure_count=int(getattr(self, "route_failure_count", 0)),
                sensor_update_count=int(getattr(self, "sensor_update_count", 0)),
                mapping_update_count=int(getattr(self, "mapping_update_count", 0)),
                safety_replan_count=int(getattr(self, "safety_replan_count", 0)),
                exploration_replan_count=int(getattr(self, "exploration_replan_count", 0)),
                planner_jobs_started=int(getattr(self, "planner_jobs_started", 0)),
                planner_jobs_completed=int(getattr(self, "planner_jobs_completed", 0)),
            )
        )

    def _finalize_navigation_debug_snapshot(
        self,
        *,
        agent,
        decision_kind: str,
        decision_reason: str,
        event_kind: NavigationDebugEventKind,
        capture: NavigationDebugCapture | None = None,
    ) -> None:
        """Freeze `capture` into a NavigationDebugSnapshot using values
        already known on self/agent at call time -- never recomputed here --
        then push it to the bounded event log and (if present) the canvas.

        Runs unconditionally whenever a robot exists -- NOT gated on
        navigation_debug_enabled. History must already be populated by the
        time a user turns Navigation on mid-run, so every tick is recorded
        regardless of that switch; navigation_debug_enabled only gates the
        UI's browse/inspect/restore affordances (step_navigation_debug_
        history(), can_restore_navigation_debug_snapshot(), the overlay
        itself), never whether a tick gets recorded. Still safe to call from
        lightweight duck-typed engine fakes that never set self.robot at all
        (self.robot is None guards that, not getattr-guarded navigation_
        debug_enabled).
        """
        if self.robot is None:
            return

        capture = capture or NavigationDebugCapture()
        robot = self.robot
        robot_xy = (float(robot.x), float(robot.y))
        robot_radius = self.body_radius_for_robot(robot)
        safety_radius = self.safety_radius_for_robot(robot)

        # Sourced from self.active_target_xy() / robot.waypoints (the
        # physics Robot's own WaypointManager), NOT agent.active_target()/
        # agent.waypoints -- the RobotAgent's route bookkeeping and the
        # physics robot's own waypoint tracker are two separate
        # WaypointManager instances (RobotAgent decides the route; Robot.
        # update() -> advance_waypoint_if_needed() is what actually
        # advances current_index during motion). Using the agent's copy
        # here could show a segment pointing at a waypoint the robot's own
        # tracker already advanced past. self.active_target_xy() is the
        # same accessor build_observation()/simulation_step() already use
        # for the real active_segment_blocked check, so this now agrees
        # with what the controller is actually tracking this tick.
        active_target_xy = self.active_target_xy()

        waypoints_mgr = getattr(robot, "waypoints", None)
        if waypoints_mgr is None or not hasattr(waypoints_mgr, "waypoints"):
            waypoints_mgr = getattr(agent, "waypoints", None) if agent is not None else None
        active_path: tuple[tuple[float, float], ...] = ()
        active_waypoint_index = None
        if waypoints_mgr is not None and getattr(waypoints_mgr, "waypoints", None):
            active_path = tuple(
                (float(p[0]), float(p[1])) for p in waypoints_mgr.waypoints
            )
            active_waypoint_index = int(getattr(waypoints_mgr, "current_index", 0))

        pending_path_raw = getattr(agent, "pending_path", None) if agent is not None else None
        pending_path = tuple((float(p[0]), float(p[1])) for p in pending_path_raw) if pending_path_raw else ()

        # Falls back to the last ACCEPTED plan when this tick did not just
        # compute a fresh one (the overwhelmingly common case -- most ticks
        # are routine FOLLOW_PATH ticks, not route-result ticks) so
        # planner/simplifier/raw/simplified-path keep describing the route
        # currently being executed instead of reading "unavailable" on
        # every tick except the exact one a route was accepted on.
        plan = capture.plan or getattr(self, "_nav_debug_last_accepted_plan", None)
        path = PathDebug(
            raw_path=Maybe.of(plan.raw_world_path) if plan and plan.raw_world_path is not None else Maybe.missing(),
            simplified_path=(
                Maybe.of(plan.simplified_world_path)
                if plan and plan.simplified_world_path is not None
                else Maybe.missing()
            ),
            active_path=active_path,
            pending_path=pending_path,
            active_segment=(robot_xy, active_target_xy) if active_target_xy is not None else None,
            active_waypoint_index=active_waypoint_index,
            planner_name=Maybe.of(plan.planner_name) if plan and plan.planner_name else Maybe.missing(),
            simplifier_name=Maybe.of(plan.simplifier_name) if plan and plan.simplifier_name else Maybe.missing(),
        )

        route = RouteValidationDebug(
            first_segment=Maybe.of(capture.first_segment) if capture.first_segment is not None else Maybe.missing(),
            endpoint_reaches_goal=capture.endpoint_reaches_goal,
        )

        predicted_motion = PredictedMotionDebug(
            trajectory=(
                Maybe.of(capture.predicted_trajectory) if capture.predicted_trajectory is not None else Maybe.missing()
            ),
            collision=Maybe.of(capture.predicted_collision) if capture.predicted_collision is not None else Maybe.missing(),
        )

        safety = SafetyDebug(
            robot_radius=float(robot_radius),
            safety_radius=float(safety_radius),
            active_segment=Maybe.of(capture.active_segment) if capture.active_segment is not None else Maybe.missing(),
        )

        planning_grid = PlanningGridDebug(
            start_cell=Maybe.of(plan.start_cell) if plan and plan.start_cell is not None else Maybe.missing(),
            start_cell_world=Maybe.of(plan.start_cell_world) if plan and plan.start_cell_world is not None else Maybe.missing(),
            first_waypoint_cell=(
                Maybe.of(plan.first_waypoint_cell) if plan and plan.first_waypoint_cell is not None else Maybe.missing()
            ),
            first_waypoint_world=(
                Maybe.of(plan.first_waypoint_world) if plan and plan.first_waypoint_world is not None else Maybe.missing()
            ),
            unknown_is_traversable=(
                Maybe.of(plan.unknown_is_traversable) if plan and plan.unknown_is_traversable is not None else Maybe.missing()
            ),
            start_cell_cleared=(
                Maybe.of(plan.start_cell_cleared) if plan and plan.start_cell_cleared is not None else Maybe.missing()
            ),
        )

        last_control = np.asarray(getattr(self, "last_control", None), dtype=float).reshape(-1) if getattr(
            self, "last_control", None
        ) is not None else None
        controller = ControllerDebug(
            v=float(robot.v),
            omega=float(last_control[1]) if last_control is not None and last_control.size >= 2 else 0.0,
            acceleration=float(last_control[0]) if last_control is not None and last_control.size >= 1 else 0.0,
            heading_error=Maybe.of(capture.heading_error) if capture.heading_error is not None else Maybe.missing(),
            distance_to_goal=Maybe.of(capture.distance_to_goal) if capture.distance_to_goal is not None else Maybe.missing(),
            desired_heading=Maybe.of(capture.desired_heading) if capture.desired_heading is not None else Maybe.missing(),
            nominal_control=Maybe.of(capture.nominal_control) if capture.nominal_control is not None else Maybe.missing(),
            applied_control=Maybe.of(capture.applied_control) if capture.applied_control is not None else Maybe.missing(),
        )

        frontier = FrontierDebug(
            candidate_count=Maybe.missing(),
            selected_target=Maybe.missing(),
            selected_score=Maybe.missing(),
            reason=Maybe.missing(),
        )

        # tracking_mode/rotate_threshold: read directly off the robot's
        # already-updated TrackingStateMachine (Robot.update_state_machine()
        # runs earlier this tick, before nominal_control_safe() is called) --
        # not recomputed. Which threshold is "active" depends on hysteresis:
        # ROTATE watches the lower rotate_to_track_threshold to exit; TRACK
        # (and the initial IDLE evaluation) watches track_to_rotate_threshold
        # (IDLE actually uses rotate_to_track_threshold too -- see
        # TrackingStateMachine.update()); STOP/BLOCKED/FAILED have no active
        # rotate/track threshold.
        state_machine = getattr(robot, "state_machine", None)
        tracking_mode = ""
        rotate_threshold = Maybe.missing()
        if state_machine is not None:
            mode_obj = getattr(state_machine, "mode", None)
            tracking_mode = str(getattr(mode_obj, "value", mode_obj) or "")
            if tracking_mode == "ROTATE" or tracking_mode == "IDLE":
                rotate_threshold = Maybe.of(float(state_machine.rotate_to_track_threshold))
            elif tracking_mode == "TRACK":
                rotate_threshold = Maybe.of(float(state_machine.track_to_rotate_threshold))

        explanation = _navigation_debug_explanation(
            tracking_mode=tracking_mode,
            decision_kind=str(decision_kind),
            decision_reason=str(decision_reason),
            controller=controller,
            rotate_threshold=rotate_threshold,
            safety=safety,
            predicted_motion=predicted_motion,
            route=route,
        )

        sensor_polygon = sensor_visible_polygon_world(
            origin=robot_xy,
            theta=float(robot.theta),
            vision=float(robot.vision),
            vision_model=self.config.vision_model,
            obstacles=list(self.config.obstacles),
            ray_count=(
                SENSOR_DRAW_RAYS_CAMERA
                if "Camera" in self.config.vision_model
                else SENSOR_DRAW_RAYS_OMNI
            ),
        )
        sensor_polygon_array = np.ascontiguousarray(sensor_polygon, dtype=np.float32)
        sensor = SensorDebug(
            vision_range=float(robot.vision),
            visible_polygon_count=int(sensor_polygon_array.shape[0]),
            visible_polygon_f32_zlib=zlib.compress(
                sensor_polygon_array.tobytes(order="C"), level=1
            ),
        )
        belief_map_debug = self._navigation_debug_belief_frame()
        hazard_debug = self._navigation_debug_hazard_frame()
        hazard_belief_debug = self._navigation_debug_hazard_belief_frame()
        agent_state_debug = self._navigation_debug_agent_state_frame(agent)
        metrics_debug = self._navigation_debug_metrics_frame()

        self._nav_debug_seq = getattr(self, "_nav_debug_seq", 0) + 1
        snapshot = NavigationDebugSnapshot(
            snapshot_id=self._nav_debug_seq,
            simulation_time=float(getattr(self, "simulation_time", 0.0)),
            robot_id="R1",
            navigation_state=str(getattr(agent, "status", "idle")),
            decision_kind=str(decision_kind),
            decision_reason=str(decision_reason),
            robot_pose=Pose(x=robot_xy[0], y=robot_xy[1], theta=float(robot.theta), v=float(robot.v)),
            path=path,
            route=route,
            predicted_motion=predicted_motion,
            safety=safety,
            planning_grid=planning_grid,
            controller=controller,
            frontier=frontier,
            tracking_mode=tracking_mode,
            rotate_threshold=rotate_threshold,
            explanation=explanation,
            mapped_obstacle_points_count=len(getattr(self, "mapped_obstacle_points", ())),
            sensor=sensor,
            belief_map=belief_map_debug,
            hazard=hazard_debug,
            hazard_belief=hazard_belief_debug,
            agent_state=agent_state_debug,
            metrics=metrics_debug,
        )

        # The ring buffer records every tick (a full in-memory history for
        # the </> step buttons to scrub through) -- bounded (see
        # NavigationDebugEventLog's max_size) and cleared whenever a
        # simulation starts/resets (the 3 reset_simulation_state() call
        # sites), so it never persists across a restart and never grows
        # without bound. The canvas also gets every snapshot below so the
        # live overlay/HUD stays current while running; pausing just stops
        # these calls, which is what leaves the last snapshot intact for
        # inspection.
        canvas = getattr(self, "canvas", None)
        self._nav_debug_live_snapshot = snapshot

        log = getattr(self, "navigation_debug_log", None)
        if log is not None:
            log.record(event_kind, snapshot)

        if event_kind is not NavigationDebugEventKind.TICK:
            # Pushed separately from the always-current "live" snapshot
            # below so the HUD's "last relevant event" line stays a sparse,
            # meaningful signal (PLAN_ACCEPTED/ROUTE_REJECTED/SAFETY_REPLAN/
            # ...) even while routine ticks keep recording into the full
            # history above.
            if canvas is not None and hasattr(canvas, "set_navigation_debug_last_event"):
                canvas.set_navigation_debug_last_event(NavigationDebugEvent(event_kind, snapshot))

        if canvas is not None and hasattr(canvas, "set_navigation_debug_snapshot"):
            canvas.set_navigation_debug_snapshot(snapshot)

    def navigation_debug_history_length(self) -> int:
        log = getattr(self, "navigation_debug_log", None)
        return len(log) if log is not None else 0

    def reset_navigation_debug_run_state(self) -> None:
        """Clear the in-memory replay and put the panel back on the robot/live state."""
        self.navigation_debug_log = NavigationDebugEventLog()
        self._nav_debug_seq = 0
        self._nav_debug_history_index = None
        self._nav_debug_last_accepted_plan = None
        self._nav_debug_live_snapshot = None
        self._nav_debug_pending_plan_capture_by_robot = {}
        self._nav_debug_belief_frame_key = None
        self._nav_debug_belief_frame_cache = None

        stop_scrub = getattr(self, "stop_navigation_history_scrub", None)
        if callable(stop_scrub):
            stop_scrub()

        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            if hasattr(canvas, "set_navigation_debug_snapshot"):
                canvas.set_navigation_debug_snapshot(None)
            if hasattr(canvas, "set_navigation_debug_last_event"):
                canvas.set_navigation_debug_last_event(None)
            if hasattr(canvas, "set_navigation_debug_history_position"):
                canvas.set_navigation_debug_history_position(None, 0)

        updater = getattr(self, "update_navigation_debug_step_buttons", None)
        if callable(updater):
            updater()

    def _push_navigation_debug_history_view(self, index: int) -> None:
        """Push the snapshot/event at `index` in the bounded event log to
        the canvas as the displayed view -- used only while paused and
        stepping. Never recomputes anything: `index` selects an already-
        frozen NavigationDebugSnapshot the log already holds."""
        log = getattr(self, "navigation_debug_log", None)
        if log is None:
            return
        event = log.event_at(index)
        if event is None:
            return

        self._nav_debug_history_index = index
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        if hasattr(canvas, "set_navigation_debug_snapshot"):
            canvas.set_navigation_debug_snapshot(event.snapshot)
        if hasattr(canvas, "set_navigation_debug_last_event"):
            canvas.set_navigation_debug_last_event(event)
        if hasattr(canvas, "set_navigation_debug_history_position"):
            canvas.set_navigation_debug_history_position(index + 1, len(log))

    def step_navigation_debug_history(self, delta: int) -> None:
        """Navigate LIVE -> frozen snapshots while paused, then browse strictly
        within [0, length-1]. Both directions clamp at their border instead of
        wrapping: `<` stops at the oldest snapshot, `>` stops at the newest one
        and no longer auto-jumps back to LIVE. LIVE is only re-entered by
        unpausing (resume_navigation_debug_live_view() is called from
        toggle_pause()) or by restore_navigation_debug_snapshot() -- never by
        this method -- so the UI's `>` button can simply disable itself at the
        newest index instead of secretly behaving like a third "back to live"
        action.

        The first `<` press from LIVE skips straight to length-2 (not
        length-1): index length-1 is the same tick LIVE is already showing,
        so landing on it would look like a no-op click. With only one saved
        snapshot (length == 1), that same index is the only one available.
        """
        if not getattr(self, "navigation_debug_enabled", False) or not getattr(self, "paused", False):
            return
        length = self.navigation_debug_history_length()
        if length == 0 or int(delta) == 0:
            return

        current = getattr(self, "_nav_debug_history_index", None)
        if int(delta) < 0:
            new_index = max(0, length - 2) if current is None else max(0, current - 1)
            self._push_navigation_debug_history_view(new_index)
        else:
            if current is None:
                return
            self._push_navigation_debug_history_view(min(length - 1, current + 1))
        updater = getattr(self, "update_navigation_debug_step_buttons", None)
        if callable(updater):
            updater()

    def resume_navigation_debug_live_view(self) -> None:
        """Restore the last real live snapshot after history inspection."""
        self._nav_debug_history_index = None
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        live_snapshot = getattr(self, "_nav_debug_live_snapshot", None)
        if live_snapshot is not None and hasattr(canvas, "set_navigation_debug_snapshot"):
            canvas.set_navigation_debug_snapshot(live_snapshot)
        log = getattr(self, "navigation_debug_log", None)
        latest = log.latest() if log is not None else None
        if hasattr(canvas, "set_navigation_debug_last_event"):
            canvas.set_navigation_debug_last_event(latest)
        if hasattr(canvas, "set_navigation_debug_history_position"):
            canvas.set_navigation_debug_history_position(None, self.navigation_debug_history_length())
        updater = getattr(self, "update_navigation_debug_step_buttons", None)
        if callable(updater):
            updater()

    def assign_route_to_robot(self) -> None:
        if self.robot is None:
            return

        self.route_request_count += 1
        success, reason, waypoints = self.compute_route((self.robot.x, self.robot.y))
        self.apply_route_result(success, reason, waypoints)

    @_timed_method("planner_dispatch")
    def request_route_async(
        self,
        reason: str,
        *,
        target_override: tuple[float, float] | None = None,
    ) -> bool:
        """
        Start a background replan and keep the GUI responsive.

        Times the synchronous "kick off a planner request" portion (see
        _timed_method()) -- the actual A*/planner computation, if async,
        runs on a background PlannerWorker thread and is measured
        separately, in route_result_handling, once its result arrives.

        target_override: when given (and planner_type != "Direct"),
        skips select_navigation_goal()'s own independent target re-derivation
        and plans directly to this target instead. ExplorationBehavior
        already chose and validated this target (e.g. it is not within
        goal_tolerance of the robot) -- re-deriving a target here via
        select_navigation_goal() is a second, independent selection that
        can disagree with ExplorationBehavior's and reintroduce exactly the
        "already reached" target it just rejected.
        """
        if self.robot is None:
            return False

        self.route_request_count += 1

        # Direct mode has no expensive global path planner, but exploration still
        # has to update F. Compute and apply the new exploration target
        # synchronously so "Planner = Direct" means "drive straight to the
        # selected frontier", not "freeze the first frontier forever".
        if self.config.planner_type == "Direct":
            success, route_reason, waypoints = self.compute_route((self.robot.x, self.robot.y))
            self.apply_route_result(success, f"{reason} {route_reason}", waypoints)
            return bool(success and waypoints)

        if compute_planned_waypoints is None:
            return False

        if self.planning_in_progress:
            return True

        self.route_request_id += 1
        request_id = self.route_request_id
        start_xy = (float(self.robot.x), float(self.robot.y))

        if target_override is not None:
            goal_xy = (float(target_override[0]), float(target_override[1]))
            goal_reason = "using ExplorationBehavior-selected target"
            self.current_exploration_target = goal_xy
            self.last_goal_selection_reason = goal_reason
            self.canvas.set_exploration_target(goal_xy)
            override_agent = self.runtime_agent(None)
            if override_agent is not None:
                override_agent.set_exploration_target(goal_xy, reason=goal_reason)
            planner_kwargs = self.build_planner_kwargs_for_goal(start_xy, goal_xy, robot=self.robot)
        else:
            planner_kwargs = self.build_planner_kwargs(start_xy)

        # [PERF]-logging only (see PlannerWorker.__init__); popped before the
        # real compute_planned_waypoints(**planner_kwargs) call.
        planner_kwargs["__perf_reason__"] = str(reason)

        if bool(planner_kwargs.get("__hold__", False)):
            self.apply_route_result(False, str(planner_kwargs.get("__hold_reason__", "holding position")), [])
            return False

        worker = PlannerWorker(
            request_id=request_id,
            planner_kwargs=planner_kwargs,
            path_simplifier=self.config.path_simplifier,
            # Always built now -- see _finalize_navigation_debug_snapshot()'s
            # docstring: capture is unconditional, not gated on navigation_
            # debug_enabled.
            debug_capture=PlanDebugCapture(),
        )
        worker.signals.route_ready.connect(self.on_async_route_ready)
        self.active_planner_workers[request_id] = worker

        self.planning_in_progress = True
        self.last_control = self.brake_control_for_collision()
        self.canvas.set_last_control(self.last_control)
        self.canvas.set_status(f"{reason} Planning in background...")
        self.planner_jobs_started = getattr(self, "planner_jobs_started", 0) + 1
        self.thread_pool.start(worker)
        return True

    def on_async_route_ready(
        self,
        request_id: int,
        success: bool,
        reason: str,
        waypoints: list,
    ) -> None:
        worker = self.active_planner_workers.pop(int(request_id), None)

        if request_id != self.route_request_id:
            return

        self.planning_in_progress = False
        self.planner_jobs_completed = getattr(self, "planner_jobs_completed", 0) + 1
        # PlannerWorker filled worker.debug_capture (if any) on the
        # background thread during run(); safe to read now, only after the
        # route_ready signal has fired -- same handoff pattern already used
        # for route_plan_ms/route_plan_perf_line. Almost every route after
        # the first goes through this async path (replanning during motion
        # always does, for a non-Direct planner), not compute_route()'s
        # synchronous path -- without this, apply_route_result() below
        # would never see a plan to persist, and planner/simplifier would
        # read "unavailable" for the rest of the run after the first route.
        self._nav_debug_last_plan_capture = getattr(worker, "debug_capture", None)
        clean_waypoints = [tuple(point) for point in waypoints]
        self.apply_route_result(success, reason, clean_waypoints)


    # ========================================================
    # LIVE METRICS
    # ========================================================

    def open_metrics_window(self) -> None:
        if self.metrics_window is None:
            self.metrics_window = SimulationMetricsWindow(self)
            # Open near the main window, but as an independent movable window.
            self.metrics_window.move(self.geometry().right() - 560, self.geometry().top() + 90)

        self.metrics_window.show()
        self.metrics_window.raise_()
        self.metrics_window.activateWindow()

    def open_console_window(self) -> None:
        if getattr(self, "console_window", None) is None:
            self.console_window = SimulationConsoleWindow(self)
            # Open below the metrics area, but keep it independent and movable.
            self.console_window.move(self.geometry().right() - 860, self.geometry().top() + 130)

        self.console_window.show()
        self.console_window.raise_()
        self.console_window.activateWindow()

    def get_console_lines(self) -> list[str]:
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "status_history_lines"):
            return canvas.status_history_lines()
        message = getattr(canvas, "status_message", "") if canvas is not None else ""
        return [str(message)] if message else []

    def clear_console_messages(self) -> None:
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "clear_status_history"):
            canvas.clear_status_history()

    def ensure_telemetry(self) -> TelemetryLogger:
        """Return the shared TelemetryLogger, creating it if needed.

        engine.py only decides WHEN to report an event (calls report_*()
        every tick/sample/route event); TelemetryLogger decides HOW that is
        throttled, aggregated, and formatted. The sink is this engine's own
        log_console_message(), so telemetry lines land in the same console
        history as everything else, without telemetry.py importing Qt/canvas.
        """
        if not hasattr(self, "_telemetry") or self._telemetry is None:
            self._telemetry = TelemetryLogger(sink=self.log_console_message)
        return self._telemetry

    @property
    def telemetry(self) -> TelemetryLogger:
        return self.ensure_telemetry()

    def ensure_robot_trace(self) -> RobotTrace:
        """Return the shared RobotTrace, creating it if needed.

        Reads the ROBOT_TRACE/ROBOT_TRACE_POINTS/ROBOT_TRACE_STDOUT/
        ROBOT_TRACE_DIR/BELIEF_TRACE_ARTIFACTS environment variables once,
        at first use -- completely separate from telemetry.py (the
        GUI-console channel) and render_perf.py (paint-timing PERF
        diagnostics). ROBOT_TRACE only ever controls terminal [TRACE ...]
        printing; belief-trace artifact files are a separate concern (see
        start_belief_trace_run()) that this constructor does NOT create --
        only RobotTrace.start_run(), called explicitly when a run actually
        starts, does that.
        """
        if not hasattr(self, "_robot_trace") or self._robot_trace is None:
            self._robot_trace = RobotTrace()
        return self._robot_trace

    @property
    def robot_trace(self) -> RobotTrace:
        return self.ensure_robot_trace()

    def start_belief_trace_run(self) -> None:
        """Create a fresh belief-trace artifact directory for this
        simulation run.

        Called explicitly from start_simulation() and restart_simulation()
        -- never merely from ensure_robot_trace()/RobotTrace construction,
        and never from reset_simulation() alone (also used by "load
        scenario", which must not create a run directory by itself).
        Enabled by default; BELIEF_TRACE_ARTIFACTS=0 (or false/no/off) is
        the only thing that can disable it, entirely independent of
        ROBOT_TRACE (which only controls terminal trace printing).

        Best-effort: records one compact notification in the in-app console
        when a directory is created. It never writes this operational message
        to stdout; launching the GUI must keep the user's terminal clean.
        """
        trace = self.ensure_robot_trace()
        run_dir = trace.start_run()
        if run_dir is not None:
            message = f"[BELIEF TRACE] writing artifacts to {run_dir}"
            self.log_console_message(message)

    def ensure_perf_monitor(self) -> PerfMonitor:
        """Return the shared PerfMonitor, creating it if needed.

        Reads SIM_PERF_LOG once, at first use -- disabled (silent) unless
        explicitly set. Independent of ROBOT_TRACE/belief-trace artifacts
        and of render_perf.py's own paint_fps/paint_ms measurement (this
        is the engine-side sim_step/sensor/route_affected/trace-queue
        timing; see on_simulation_tick()).
        """
        if not hasattr(self, "_perf_monitor") or self._perf_monitor is None:
            self._perf_monitor = PerfMonitor()
        return self._perf_monitor

    def _compute_nav_state(self, agent) -> str:
        """Best-effort, diagnostics-only navigation-state label for the
        [PERF] summary line: running | recovering | exhausted |
        safety_replan_loop | idle.

        Read-only: only reads existing RobotAgent fields, never calls
        exploration_exhausted() or any other navigation/decision method,
        and never affects behavior -- purely a label for humans reading
        the [PERF] line. Distinguishes "exhausted" (latched exhaustion
        flag AND no active route -- genuinely nothing left to do) from
        "safety_replan_loop"/"recovering" (actively retrying/replanning,
        which can itself be the expensive phase -- see
        _should_skip_for_exhausted_hold()'s docstring for why conflating
        these caused route_check_ms to incorrectly read 0.0).
        """
        if agent is None:
            return "idle"
        has_active_route = getattr(agent, "active_path_goal_xy", None) is not None
        exhausted_latched = getattr(agent, "exploration_exhausted_map_signature", None) is not None
        failures = int(getattr(agent, "consecutive_exploration_failures", 0))
        repairing = getattr(agent, "route_repair_in_progress_for_goal", None) is not None

        if exhausted_latched and not has_active_route:
            return "exhausted"
        if repairing:
            return "safety_replan_loop"
        if failures > 0:
            return "recovering"
        if has_active_route:
            return "running"
        return "idle"

    def on_simulation_tick(self) -> None:
        """QTimer callback: times the whole simulation_step() call, then
        (at most once every couple of seconds, and only when SIM_PERF_LOG
        is enabled) logs a compact [PERF] summary combining sim_step
        timing, the render-side paint_ms render_perf.py already measures,
        and the belief-trace background queue's size/drop pressure.

        Kept as a thin wrapper around simulation_step() specifically so
        none of simulation_step()'s own control flow/indentation needs to
        change for this instrumentation -- zero effect on navigation
        behavior.
        """
        monitor = self.ensure_perf_monitor()
        hits_before = getattr(self, "exhausted_idle_fast_path_hits", 0)
        start = time.perf_counter()
        self.simulation_step()
        duration_s = time.perf_counter() - start
        monitor.record("sim_step", duration_s)
        # Also record the SAME duration under a category-specific phase so
        # PerfMonitor can report fast_path_avg_ms/full_pipeline_avg_ms
        # separately -- a fast-path-skip tick and a full-pipeline tick have
        # wildly different costs, and blending them into one "sim_step"
        # average hid that a tick's phase timings (recorded only on
        # full-pipeline ticks) were being averaged over a different,
        # smaller denominator than avg_sim_step_ms itself (see
        # PerfMonitor.per_tick_ms()'s docstring for the accounting bug this
        # fixes).
        if getattr(self, "exhausted_idle_fast_path_hits", 0) > hits_before:
            monitor.record("sim_step_fast_path", duration_s)
        else:
            monitor.record("sim_step_full_pipeline", duration_s)
        monitor.note_tick()

        canvas = getattr(self, "canvas", None)
        perf_status = getattr(canvas, "latest_perf_status", None) if canvas is not None else None
        render_ms = float(perf_status["paint_ms"]) if perf_status else 0.0

        trace = getattr(self, "_robot_trace", None)
        writer = getattr(trace, "writer", None) if trace is not None else None
        trace_queue_size = writer.queue_size if writer is not None else 0
        dropped_trace_events = writer.dropped_total if writer is not None else 0

        agent = self.runtime_agent(None)
        nav_state = self._compute_nav_state(agent)

        monitor.maybe_log_summary(
            render_ms=render_ms,
            trace_queue_size=trace_queue_size,
            dropped_trace_events=dropped_trace_events,
            mapped_obstacle_count=len(getattr(self, "mapped_obstacle_points", [])),
            explored_percent=self.estimated_explored_percent() if self.robot is not None else None,
            nav_state=nav_state,
            planner_jobs_started=getattr(self, "planner_jobs_started", 0),
            planner_jobs_completed=getattr(self, "planner_jobs_completed", 0),
            safety_replans=getattr(self, "safety_replan_count", 0),
            route_failures=getattr(self, "route_failure_count", 0),
            repeated_safety_replans=getattr(self, "repeated_safety_replan_count", 0),
            exhausted_idle_fast_path_hits=getattr(self, "exhausted_idle_fast_path_hits", 0),
            exhausted_idle_full_updates=getattr(self, "exhausted_idle_full_updates", 0),
            exhausted_idle_skipped_canvas_updates=getattr(self, "exhausted_idle_skipped_canvas_updates", 0),
            exhausted_idle_skipped_sensor_updates=getattr(self, "exhausted_idle_skipped_sensor_updates", 0),
            reachability_context_builds=getattr(self, "reachability_context_builds", 0),
            log=self.log_console_message,
        )

    def _should_skip_for_exhausted_hold(self, *, min_interval_s: float = 1.0) -> bool:
        """True when expensive per-tick work (route_affected check, forced
        canvas repaint, belief snapshot) should be SKIPPED this tick.

        Read-only with respect to navigation state: only reads
        RobotAgent.exploration_exhausted_map_signature and
        active_path_goal_xy, both persistent flags set/cleared entirely by
        ExplorationBehavior/RobotAgent's own existing logic elsewhere --
        never calls exploration_exhausted() itself (which has its own
        reset side effect) or any other navigation/decision method.

        BOTH exploration_exhausted_map_signature is not None AND
        active_path_goal_xy is None are required before this is
        considered "exhausted" for throttling purposes -- not just the
        flag alone. Root-cause fix: exploration_exhausted_map_signature
        can go stale. It is set once consecutive_exploration_failures
        reaches its budget, and is only ever CLEARED by
        exploration_exhausted() itself, which only runs from
        ExplorationBehavior's "no active path" branch. Once the agent
        gets a new active route again (e.g. after a route_affected repair
        or a safety replan succeeds), that branch is never reached again
        while the route stays active -- so the stale signature can keep
        reading "not None" for the ENTIRE remaining safety-replan-loop/
        recovering episode, even though the agent is now busy, not idle.
        Without the active_path_goal_xy check, that stale flag silently
        suppressed route_affected_check/belief_snapshot/canvas updates
        throughout exactly the period they were most needed (confirmed by
        route_check_ms staying 0.0 despite route_affected=yes events in
        the app log). Requiring "no active route" as well means this only
        ever throttles genuine idle-exhaustion, never an active
        replan/recovery episode.

        When exhausted, throttles to at most once every min_interval_s
        simulated seconds ("keep occasional low-rate updates ~1Hz so the
        UI never looks frozen") -- the FIRST call after the interval
        elapses returns False (this tick is the "due" one -- do not skip)
        and stamps the timestamp; every call in between returns True
        (skip). Call this AT MOST ONCE per tick and reuse the result for
        every gate that tick, so all the gated work (or none of it) goes
        together on the same "due" tick.
        """
        agent = self.runtime_agent(None)
        exhausted = (
            agent is not None
            and getattr(agent, "exploration_exhausted_map_signature", None) is not None
            and getattr(agent, "active_path_goal_xy", None) is None
        )
        if not exhausted:
            return False
        last = getattr(self, "_last_exhausted_low_rate_time", None)
        if last is not None and (float(self.simulation_time) - last) < min_interval_s:
            return True
        self._last_exhausted_low_rate_time = float(self.simulation_time)
        return False

    def _exhausted_idle_fast_path_ready(self, agent) -> bool:
        """True when the ENTIRE per-tick pipeline (sensor update, agent
        decision, motion integration, telemetry) can be skipped this tick
        without changing simulation state -- a stronger condition than
        _should_skip_for_exhausted_hold()'s own narrower gate (which only
        throttles route_affected/canvas/belief-snapshot work).

        Read-only with respect to navigation state -- never calls
        exploration_exhausted()/agent.step()/any decision method, only
        reads existing flags, mirroring _should_skip_for_exhausted_hold().
        ALL of the following must hold, or the caller must fall through
        to the unmodified normal path:
            - _compute_nav_state() reports "exhausted": the agent has
              latched exploration_exhausted_map_signature AND has no
              active_path_goal_xy -- genuinely nothing left to route to,
              not just momentarily between replans (see
              _should_skip_for_exhausted_hold()'s docstring for why both
              are required)
            - no planner job is in flight (planning_in_progress)
            - no pending path/target is awaiting acceptance
              (RobotAgent.pending_path/pending_target_xy) -- a planner
              result could arrive and change things at any tick
            - the robot's own speed is at/below its stop tolerance
              (robot.v vs. robot.stop_speed_tolerance) -- if it is still
              moving, its pose (and therefore what its sensor would see)
              is still changing
            - the ground-truth obstacle count (config.obstacles, NOT the
              robot's own mapped_obstacle_points) has not changed since
              the last full-pipeline tick -- new obstacles are exactly
              the kind of change that could make an "exhausted" agent's
              situation different

        With the robot provably stationary and nothing pending, a fresh
        sensor scan from the same pose against the same ground-truth
        obstacles is guaranteed to reproduce the same result as the last
        one, and the agent/motion pipeline is guaranteed to again decide
        "stay put" -- so skipping them changes no simulation state, only
        how often it is recomputed. Scoped to single-robot
        simulation_step(); simulation_step_multi() is untouched.
        """
        if agent is None or self.robot is None:
            return False
        if self._compute_nav_state(agent) != "exhausted":
            return False
        if self.planning_in_progress:
            return False
        if getattr(agent, "pending_path", None) is not None:
            return False
        if getattr(agent, "pending_target_xy", None) is not None:
            return False
        if abs(float(self.robot.v)) > float(self.robot.stop_speed_tolerance):
            return False

        obstacle_count = len(self.config.obstacles)
        baseline = getattr(self, "_exhausted_idle_obstacle_count", None)
        if baseline is None:
            self._exhausted_idle_obstacle_count = obstacle_count
            return True
        if obstacle_count != baseline:
            self._exhausted_idle_obstacle_count = obstacle_count
            return False
        return True

    def _timed_route_affected_check(self, newly_discovered) -> bool:
        """Thin timing wrapper around new_information_affects_current_route()
        -- the exact computation behind the app log's "route_affected=yes"
        line (see telemetry.report_map_update()'s call site right after
        this). Extracted into its own method (rather than an inline
        time.perf_counter() pair) specifically so this timing can be unit
        tested directly, driving the real route_affected=yes code path,
        without needing the full simulation_step()/sensor/collision-
        checker stack. Zero effect on the check's own result/behavior.
        """
        start = time.perf_counter()
        try:
            return self.new_information_affects_current_route(newly_discovered)
        finally:
            _record_perf(self, "route_affected_check", time.perf_counter() - start)

    def _build_belief_trace_snapshot(self) -> dict | None:
        """Best-effort belief_final.json (+ belief_grid_final.npz) payload
        for RobotTrace.maybe_snapshot_belief(). Only ever called lazily
        when a snapshot write is actually due (see the call site in
        simulation_step()) -- diagnostics/output only, never touches
        mapping/planning state, only reads it.

        Snapshot safety: the belief-trace writer now runs on a background
        thread (AsyncTraceWriter), so this snapshot dict can sit in a
        queue while the simulation thread keeps mutating live state.
        Every value here is already an independent plain float/int/str/
        list -- except the grid array below, which is explicitly .copy()'d
        so the background thread never reads/serializes the SAME array
        object the belief map continues to write into concurrently.
        """
        points = list(getattr(self, "mapped_obstacle_points", []))
        snapshot: dict = {
            "explored_percent": self.estimated_explored_percent(),
            "robot_pose": (float(self.robot.x), float(self.robot.y)) if self.robot is not None else None,
            "mapped_obstacle_points_count": len(points),
        }
        if points:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            snapshot["mapped_obstacle_bbox"] = [min(xs), min(ys), max(xs), max(ys)]
        else:
            snapshot["mapped_obstacle_bbox"] = None
        sections = group_obstacle_points_into_sections(points) if points else []
        snapshot["obstacle_sections_summary"] = [
            dict(orientation=s.axis, coord=s.coordinate, span_min=s.span_min, span_max=s.span_max, n_points=s.count)
            for s in sections[:MAX_OBSTACLE_SECTIONS]
        ]

        belief = getattr(self, "belief_map", None)
        if belief is not None and hasattr(belief, "grid"):
            snapshot["grid_resolution"] = float(belief.resolution)
            grid = belief.grid
            snapshot["known_free_count"] = int(np.count_nonzero(grid == FREE))
            snapshot["known_occupied_count"] = int(np.count_nonzero(grid == OCCUPIED))
            snapshot["unknown_count"] = int(np.count_nonzero(grid == UNKNOWN))
            # .copy(): never hand the background writer thread a live
            # reference to the belief map's own mutable array (see
            # docstring above).
            snapshot["_grid"] = grid.copy()
        return snapshot

    @_timed_method("console_log")
    def log_console_message(self, message: str, *, visible_status: bool = False) -> None:
        """Write a readable debugging message to the simulation console.
        Timed under PerfMonitor's "console_log" phase (see _timed_method()).

        visible_status=True also replaces the short status shown at the top of
        the canvas. Most detailed traces should keep visible_status=False so the
        canvas does not become noisy or truncated.
        """
        message = str(message).strip()
        if not message:
            return

        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return

        if visible_status and hasattr(canvas, "set_status"):
            canvas.set_status(message)
        elif hasattr(canvas, "append_console_message"):
            canvas.append_console_message(message)
        elif hasattr(canvas, "_append_status_history"):
            canvas._append_status_history(message)

    def _xy_text(self, point) -> str:
        if point is None:
            return "--"
        try:
            return f"({float(point[0]):.2f}, {float(point[1]):.2f})"
        except Exception:
            return str(point)

    def _control_text(self, control) -> str:
        try:
            arr = np.asarray(control, dtype=float).reshape(-1)
            if arr.size >= 2:
                return f"u=({arr[0]:.3f}, {arr[1]:.3f})"
            if arr.size == 1:
                return f"u=({arr[0]:.3f})"
        except Exception:
            pass
        return "u=--"

    def simulation_start_summary(self, *, multi: bool) -> str:
        """Return a multi-line, copyable summary of the exact run configuration."""
        cfg = self.config
        mode = "Multiple Robot Mode" if multi else "Single Robot Mode"
        try:
            profile = runtime_profile_for_strategy(cfg.coordinator_type)
        except PluginLoadError:
            profile = None

        if profile is not None and profile.owns_target_generation:
            exploration_lines = [
                f"Exploration source: {cfg.coordinator_type}",
                f"Legacy frontier service (fallback only): {cfg.exploration_planner}",
            ]
        else:
            exploration_lines = [f"Exploration planner: {cfg.exploration_planner}"]

        lines = [
            "=== Simulation started ===",
            f"Mode: {mode}",
            f"Planner: {cfg.planner_type}",
            f"Path simplifier: {cfg.path_simplifier}",
            *exploration_lines,
            f"Multi-robot coordinator: {cfg.coordinator_type}",
        ]
        if profile is not None:
            lines.append(
                "Algorithm runtime profile: "
                f"owns_target_generation={profile.owns_target_generation}, "
                f"owns_task_allocation={profile.owns_task_allocation}, "
                f"owns_path_planning={profile.owns_path_planning}, "
                f"owns_control={profile.owns_control}, "
                f"uses_legacy_frontier_service={profile.uses_legacy_frontier_service}, "
                f"uses_external_path_planner={profile.uses_external_path_planner}, "
                f"uses_external_motion_controller={profile.uses_external_motion_controller}"
            )
        lines += [
            f"Vision model: {cfg.vision_model}",
            f"Sensor range: {float(cfg.vision):.2f} m",
            f"Grid resolution: {float(cfg.grid_resolution):.2f} m/cell",
            f"Goal G: ({float(cfg.goal_x):.2f}, {float(cfg.goal_y):.2f})",
            f"Robot body radius: {float(cfg.body_radius):.2f} m",
            f"Safety radius r: {float(cfg.safety_radius):.2f} m",
            f"Max speed: {float(cfg.max_speed):.2f} m/s",
            f"Max acceleration: {float(cfg.max_acceleration):.2f} m/s²",
            f"Max angular speed: {float(cfg.max_angular_speed):.2f} rad/s",
            f"Goal tolerance: {float(cfg.goal_tolerance):.2f} m",
            f"IPP λ distance penalty: {float(cfg.ipp_distance_penalty):.2f}",
            f"Exploration replan cooldown: {float(cfg.exploration_replan_cooldown):.2f} s",
            f"Obstacles in scenario: {len(cfg.obstacles)}",
        ]

        if multi:
            lines.append(f"Robot count: {len(getattr(self, 'robots', []) or [])}")
            lines.append(f"Same robot configuration: {bool(getattr(cfg, 'same_robot_configuration', True))}")
            for index, robot in enumerate(getattr(self, "robots", []) or []):
                lines.append(
                    f"R{index + 1} start: pos=({float(robot.x):.2f}, {float(robot.y):.2f}), "
                    f"theta={float(robot.theta):.3f} rad, v={float(robot.v):.3f} m/s, "
                    f"vision={float(getattr(robot, '_sim_vision', cfg.vision)):.2f} m, "
                    f"r={float(self.safety_radius_for_robot(robot)):.2f} m"
                )
        else:
            lines.append(
                f"R1 start: pos=({float(cfg.x):.2f}, {float(cfg.y):.2f}), "
                f"theta={float(cfg.theta):.3f} rad, v={float(cfg.v):.3f} m/s"
            )

        return "\n".join(lines)

    def log_route_assignment(
        self,
        robot_index: int | None,
        start_xy: tuple[float, float],
        waypoints: list[tuple[float, float]],
        reason: str,
    ) -> None:
        label = f"R{int(robot_index) + 1}" if robot_index is not None else "R1"
        target = waypoints[-1] if waypoints else None
        length = 0.0
        previous = start_xy
        for point in waypoints:
            length += math.hypot(float(point[0]) - float(previous[0]), float(point[1]) - float(previous[1]))
            previous = point
        self.telemetry.report_route_success(
            robot_label=label,
            start_xy=start_xy,
            goal_xy=target,
            wp_count=len(waypoints),
            planner_type=str(self.config.planner_type),
            simplifier=str(self.config.path_simplifier),
            length=length,
            mapped_obstacle_count=len(self.mapped_obstacle_points),
        )
        # Opt-in terminal trace only (ROBOT_TRACE=route).
        _emit_robot_trace(
            self,
            "trace_route",
            sim_time=float(getattr(self, "simulation_time", 0.0)),
            robot_label=label,
            result="ok",
            start=start_xy,
            goal=target,
            waypoint_count=len(waypoints),
            length=length,
            mapped_obstacle_count=len(getattr(self, "mapped_obstacle_points", [])),
            planner=str(self.config.planner_type),
            simplifier=str(self.config.path_simplifier),
        )

    def log_robot_motion(
        self,
        robot,
        *,
        robot_index: int | None = None,
        control=None,
        target=None,
        force: bool = False,
    ) -> None:
        """Log throttled robot motion traces with coordinates and target."""
        if robot is None:
            return

        interval = 0.50
        now = float(getattr(self, "simulation_time", 0.0))
        if robot_index is None:
            last = float(getattr(self, "last_motion_log_time", -1.0e9))
            if (not force) and now - last < interval:
                return
            self.last_motion_log_time = now
            label = "R1"
        else:
            log_times = getattr(self, "multi_last_motion_log_times", None)
            if log_times is None:
                log_times = {}
                self.multi_last_motion_log_times = log_times
            last = float(log_times.get(int(robot_index), -1.0e9))
            if (not force) and now - last < interval:
                return
            log_times[int(robot_index)] = now
            label = f"R{int(robot_index) + 1}"

        if target is None:
            target = self.active_target_xy()

        telemetry = self.telemetry
        telemetry.report_move(
            sim_time=now,
            robot_label=label,
            pos=(float(robot.x), float(robot.y)),
            theta=float(robot.theta),
            v=float(robot.v),
            target=target,
            control_text=self._control_text(control),
        )

        agent = self.runtime_agent(robot_index)
        path_goal = getattr(agent, "active_path_goal_xy", None) if agent is not None else None
        wp_index, wp_total = self._waypoint_progress_for_robot(robot)

        # No active exploration route (path_goal is None): the robot may
        # still have a single "hold" waypoint pointing at its own current
        # position (see set_robot_goal_or_waypoints() in the HOLD/exhausted
        # decision handlers), but that is not a real destination -- do not
        # report it as `target` in [STATE], or an exhausted/holding robot
        # misleadingly looks like it is still driving somewhere.
        holding_without_route = path_goal is None
        state_target = None if holding_without_route else target
        hold_pos = (float(robot.x), float(robot.y)) if holding_without_route else None
        state_wp_index = 0 if holding_without_route else wp_index
        state_wp_total = 0 if holding_without_route else wp_total

        telemetry.report_state(
            sim_time=now,
            wall_time=time.perf_counter(),
            speed_multiplier=float(getattr(self, "simulation_speed", 1.0)),
            robot_label=label,
            pos=(float(robot.x), float(robot.y)),
            theta=float(robot.theta),
            v=float(robot.v),
            state=mode_name(robot),
            target=state_target,
            path_goal=path_goal,
            hold_pos=hold_pos,
            wp_index=state_wp_index,
            wp_total=state_wp_total,
            mapped_obstacle_count=len(self.mapped_obstacle_points),
            explored_percent=self.estimated_explored_percent(),
            force=force,
        )

    def _waypoint_progress_for_robot(self, robot) -> tuple[int, int]:
        """(current 1-based index, total) waypoints for *robot*'s own route.

        Mirrors remaining_waypoint_count()'s introspection, generalized to
        any robot (not just self.robot), for [STATE] snapshots.
        """
        waypoint_manager = getattr(robot, "waypoints", None)
        raw_waypoints = getattr(waypoint_manager, "waypoints", None)
        current_index = getattr(waypoint_manager, "current_index", None)
        if raw_waypoints is not None and isinstance(current_index, int):
            total = len(raw_waypoints)
            index = min(current_index + 1, total) if total else 0
            return index, total
        return (0, 0)

    def latest_decision_message(self) -> str:
        status = ""
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            status = str(getattr(canvas, "status_message", "") or "").strip()
        reason = str(getattr(self, "last_goal_selection_reason", "") or "").strip()

        if reason and status and reason not in status:
            return f"{reason}\nStatus: {status}"
        return reason or status or "--"

    def estimated_explored_percent(self) -> float:
        return self.ensure_belief_map().stats().coverage_percent

    def point_inside_ground_truth_obstacle(self, point: tuple[float, float]) -> bool:
        """Return True if a world point is inside a scenario obstacle.

        This is used only for evaluation metrics, not for planning decisions.
        The planner still receives only the partial belief map.
        """
        x, y = point
        for obstacle in self.config.obstacles:
            ox, oy, width, height = map(float, obstacle)
            x0, x1 = sorted((ox, ox + width))
            y0, y1 = sorted((oy, oy + height))
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    def ground_truth_free_cell_count(self) -> int:
        """Count traversable cells in the full scenario for metrics only.

        The denominator for exploration quality should not be the whole
        rectangle, because obstacle interiors are not traversable. This metric
        deliberately uses ground truth only in the dashboard/evaluation layer.
        """
        belief = self.ensure_belief_map()
        count = 0
        for row in range(belief.height):
            for col in range(belief.width):
                if not self.point_inside_ground_truth_obstacle(belief.cell_to_world((row, col))):
                    count += 1
        return count

    def estimated_free_space_coverage_percent(self) -> float:
        belief = self.ensure_belief_map()
        free_cells = belief.stats().free_cells
        traversable_cells = self.ground_truth_free_cell_count()
        return 100.0 * free_cells / max(1, traversable_cells)

    def remaining_waypoint_count(self) -> int:
        if self.robot is None:
            return 0
        waypoint_manager = getattr(self.robot, "waypoints", None)
        raw_waypoints = getattr(waypoint_manager, "waypoints", None)
        current_index = getattr(waypoint_manager, "current_index", None)
        if raw_waypoints is not None and isinstance(current_index, int):
            return max(0, len(raw_waypoints) - int(current_index))
        return 1 if self.active_target_xy() is not None else 0

    def get_metrics_snapshot(self) -> list[tuple[str, str]]:
        robot_state = "None" if self.robot is None else mode_name(self.robot)
        robot_xy = "--"
        robot_theta = "--"
        robot_v = "--"
        target_xy = "--"
        distance_to_target = "--"
        distance_to_goal = "--"

        if self.robot is not None:
            robot_xy = f"({self.robot.x:.2f}, {self.robot.y:.2f})"
            robot_theta = f"{self.robot.theta:.3f} rad"
            robot_v = f"{self.robot.v:.3f} m/s"
            target = self.active_target_xy()
            if target is not None:
                target_xy = f"({target[0]:.2f}, {target[1]:.2f})"
                distance_to_target = f"{math.hypot(float(self.robot.x) - target[0], float(self.robot.y) - target[1]):.3f} m"
            gx, gy = self.final_goal_xy()
            distance_to_goal = f"{math.hypot(float(self.robot.x) - gx, float(self.robot.y) - gy):.3f} m"

        exploration_target = "--"
        if self.current_exploration_target is not None:
            exploration_target = f"({self.current_exploration_target[0]:.2f}, {self.current_exploration_target[1]:.2f})"

        belief = self.ensure_belief_map()
        stats = belief.stats()
        metrics = [
            ("Running", "Yes" if self.running and not self.paused else "No"),
            ("Robot state", robot_state),
            ("FPS", f"{self.canvas.fps:.1f}"),
            ("Simulation time", f"{self.simulation_time:.2f} s"),
            ("Simulation speed", f"{self.simulation_speed:.2f}x"),
            ("Robot position", robot_xy),
            ("Robot theta", robot_theta),
            ("Robot velocity", robot_v),
            ("Active target", target_xy),
            ("Exploration target F", exploration_target),
            ("Distance to active target", distance_to_target),
            ("Distance to final goal", distance_to_goal),
            ("Total distance traveled", f"{self.total_distance_traveled:.2f} m"),
            ("Path planner", self.config.planner_type),
            ("Path simplifier", self.config.path_simplifier),
            ("Exploration planner", self.config.exploration_planner),
            ("Multi-robot coordinator", self.config.coordinator_type),
            ("UI coordinator selection", self.coordinator_combo.currentText() if hasattr(self, "coordinator_combo") else "--"),
            ("Coordinator synced", "Yes" if (not hasattr(self, "coordinator_combo") or self.coordinator_combo.currentText() == self.config.coordinator_type) else "No"),
            ("IPP distance penalty λ", f"{self.config.ipp_distance_penalty:.2f}"),
            ("Planner requests", str(self.route_request_count)),
            ("Planner results applied", str(self.route_result_count)),
            ("Exploration replans", str(self.exploration_replan_count)),
            ("Safety replans", str(self.safety_replan_count)),
            ("Planning in background", "Yes" if self.planning_in_progress else "No"),
            ("Remaining waypoints", str(self.remaining_waypoint_count())),
            ("Belief FREE cells", str(stats.free_cells)),
            ("Belief OCCUPIED cells", str(stats.occupied_cells)),
            ("Belief UNKNOWN cells", str(stats.unknown_cells)),
            ("Belief known cells", str(stats.known_cells)),
            ("Belief coverage of rectangle", f"{stats.coverage_percent:.1f}%"),
            ("Free-space coverage", f"{self.estimated_free_space_coverage_percent():.1f}%"),
            ("Revisited free cells", str(stats.revisited_cells)),
            ("Revisit ratio", f"{100.0 * stats.revisit_ratio:.1f}%"),
            ("Avg visits per free cell", f"{stats.average_visits_per_free_cell:.2f}"),
            ("Multi-robot overlap cells", str(stats.overlap_cells)),
            ("Multi-robot overlap ratio", f"{100.0 * stats.overlap_ratio:.1f}%"),
        ]

        if getattr(self, "robots", None):
            per_robot_counts = belief.per_robot_explored_counts()
            per_robot_overlap = belief.per_robot_overlap_counts()
            for index, count in enumerate(per_robot_counts):
                metrics.append((f"R{index + 1} free cells", str(count)))
            if len(per_robot_counts) > 1:
                for index, count in enumerate(per_robot_overlap):
                    metrics.append((f"R{index + 1} overlap cells", str(count)))

        metrics.extend([
            ("Sensor updates", str(self.sensor_update_count)),
            ("Mapping updates", str(self.mapping_update_count)),
        ])
        return metrics

    # ========================================================
    # SIMULATION CONTROLS
    # ========================================================

    def update_start_pause_button(self) -> None:
        """
        Keep the main action button stateful.

        Start Simulation creates a new run only when there is no active run.
        During a run, the same button pauses/resumes. Restart is handled by the
        separate Restart button so Start no longer behaves like an accidental
        reset.
        """
        if not self.running:
            self.start_button.setText("Start")
            self.start_button.setIcon(make_icon("play", "white"))
        elif self.paused:
            self.start_button.setText("Resume")
            self.start_button.setIcon(make_icon("play", "white"))
        else:
            self.start_button.setText("Pause")
            self.start_button.setIcon(make_icon("pause", "white"))

    def update_navigation_debug_step_buttons(self) -> None:
        """Reflect history boundaries in the navigation_snapshot_bar -- the
        single, sole control for stepping through navigation-debug history
        (see main_window._build_navigation_snapshot_bar()). Called from
        every place history state can change -- step/resume/restore/toggle/
        reset -- so none of those call sites needs to know about the bar
        directly.
        """
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        length = self.navigation_debug_history_length()
        enabled = bool(getattr(self, "navigation_debug_enabled", False))
        active = enabled and bool(getattr(self, "paused", False)) and length > 0
        current = getattr(self, "_nav_debug_history_index", None)
        back_enabled = active and (current is None or current > 0)
        # Clamped at the newest snapshot -- stepping forward past it no
        # longer auto-resumes LIVE (see step_navigation_debug_history()), so
        # the button disables itself at that border just like `<` does at 0.
        forward_enabled = active and current is not None and current < length - 1

        bar = getattr(self, "navigation_snapshot_bar", None)
        updater = getattr(bar, "update_state", None)
        if callable(updater):
            can_restore, restore_reason = self.can_restore_navigation_debug_snapshot()
            updater(
                navigation_enabled=enabled,
                position=current,
                total=length,
                back_enabled=back_enabled,
                forward_enabled=forward_enabled,
                multiplier=float(getattr(self, "_nav_history_scrub_current_multiplier", 1.0)),
                resume_enabled=can_restore,
                resume_reason=restore_reason,
            )

    def can_restore_navigation_debug_snapshot(self) -> tuple[bool, str]:
        """Whether 'Resume from snapshot' is actionable right now, and the
        user-facing reason when it is not (surfaced as the button's tooltip).

        Ordered so the most fundamental blocker wins: navigation off, then
        multi-robot (v1 is single-robot only -- see restore_navigation_debug_
        snapshot()'s docstring), then no robot, then LIVE (nothing selected
        to restore from).
        """
        if not getattr(self, "navigation_debug_enabled", False):
            return False, "Enable Navigation to use snapshot controls."
        if "Multiple" in str(getattr(self.config, "agent_mode", "")):
            return False, "Resume from snapshot supports single-robot mode only."
        if getattr(self, "robot", None) is None:
            return False, "Start the simulation to capture snapshots."
        if getattr(self, "_nav_debug_history_index", None) is None:
            return False, "Select a historical snapshot to resume from."
        return True, ""

    def _restore_empty_hazard_belief(self, hazard_service: RuntimeHazardService) -> None:
        """Reset hazard_service.belief to a deterministic empty state:
        revision EXACTLY 0, not whatever HazardBelief.clear() would leave
        behind.

        HazardBelief.clear() is a no-op (keeps the current revision) when
        the belief was already empty, and otherwise bumps the revision by
        exactly 1 -- so restoring the SAME historical snapshot from two
        different "future" live states could clear() into two different
        revisions. That would be a nondeterministic result for a restore
        that is supposed to reproduce one frozen moment exactly, and would
        confuse the (id(belief), revision, ...) cache in _navigation_debug_
        hazard_belief_frame() and anything else that trusts revision as a
        change signal.

        Used for every fallback case in restore_navigation_debug_snapshot()
        (missing HazardBeliefDebug, corrupt payload, shape/robot_count
        mismatch) -- explicit HazardBelief.restore() with all-zero arrays,
        never HazardBelief.clear(). Never derives state from HazardField,
        never re-runs a sensor; the RuntimeHazardService instance itself is
        left exactly as it is.
        """
        belief = hazard_service.belief
        height, width = belief.shape
        belief.restore(
            HazardBeliefFrame(
                values=np.zeros((height, width), dtype=np.float32),
                observed=np.zeros((height, width), dtype=bool),
                observed_by_robot=np.zeros((belief.robot_count, height, width), dtype=bool),
                revision=0,
            )
        )

    def restore_navigation_debug_snapshot(self, index: int | None = None) -> bool:
        """Roll the live simulation back to the frozen state at history
        `index` (defaults to the currently viewed HISTORY position), then
        truncate everything recorded after it and return to LIVE.

        This is a real restore, not a view change: simulation_time, robot
        pose/kinematics, the active route/waypoint index, belief_map
        (occupancy + explored-by-robot), hazards (FireSources + next_fire_
        id), explicit RobotAgent state (goals/active-path-mode/route_
        generation/counters -- see AgentStateDebug), engine-level cumulative
        metrics (see RuntimeMetricsDebug), and the append-only mapped_
        obstacle_points list are all overwritten from the snapshot.
        In-flight async planner work is invalidated (route_request_id bump +
        worker-dict clear, the same pattern start_simulation()/reset_
        simulation() use) and pending/prefetch route state is cleared
        unconditionally so a path computed in the discarded future can never
        be promoted later. The event log is truncated so nothing from that
        discarded future remains scrubbable.

        Single-robot only for this version -- guarded by can_restore_
        navigation_debug_snapshot(), which multi-robot mode fails. One piece
        of runtime state is deliberately NOT rolled back (see the
        NavigationDebugSnapshot / AgentStateDebug docstrings for why): the
        executed-path trail (path_points) has no authoritative source to
        rebuild from and resets to the single restored point. The visible
        explored-area *coverage*, by contrast, does not regress even though
        its bounded sensor-sweep polygon list (explored_area_polygons) is
        cleared -- the canvas is reseeded directly from the just-restored
        belief.explored_by_robot mask (see canvas.set_explored_area_seed()),
        so what the user sees stays consistent with the authoritative belief
        state instead of visually "forgetting" coverage that was never
        actually un-explored.
        """
        can_restore, _reason = self.can_restore_navigation_debug_snapshot()
        if not can_restore:
            return False

        if index is None:
            index = self._nav_debug_history_index
        log = self.navigation_debug_log
        event = log.event_at(int(index))
        if event is None:
            return False
        snapshot = event.snapshot
        if snapshot.belief_map.unavailable or snapshot.belief_map.value is None:
            return False
        frame = snapshot.belief_map.value

        belief = self.ensure_belief_map()
        grid = np.frombuffer(zlib.decompress(frame.grid_zlib), dtype=np.int8).reshape(frame.grid_shape).copy()
        explored_packed = np.frombuffer(zlib.decompress(frame.explored_packbits_zlib), dtype=np.uint8)
        explored_count = int(np.prod(frame.explored_shape))
        explored = np.unpackbits(
            explored_packed, bitorder="little", count=explored_count
        ).reshape(frame.explored_shape).astype(bool, copy=False)

        # Compatibility for snapshots captured before single-robot sensor
        # footprints were attributed to robot 0.  Those frames can contain a
        # correct occupancy grid but an all-false explored_by_robot mask.  For
        # one robot, every known cell necessarily came from that robot's sensor,
        # so rebuild the missing ownership mask from the known belief cells.
        if explored.shape[0] == 1 and not np.any(explored):
            known_cells = grid != UNKNOWN
            if np.any(known_cells):
                explored = explored.copy()
                explored[0] = known_cells

        # visit_count/last_seen -- restored exactly, never reconstructed from
        # grid/explored (a cell can be FREE with any visit count >= 1; only
        # the exact captured count is correct). Empty bytes means this frame
        # predates the field existing -- fall back to BeliefMap's own
        # zero-state defaults for that case rather than crashing.
        if frame.visit_count_zlib:
            visit_count = (
                np.frombuffer(zlib.decompress(frame.visit_count_zlib), dtype=np.uint16)
                .reshape(frame.grid_shape)
                .copy()
            )
        else:
            visit_count = np.zeros(frame.grid_shape, dtype=np.uint16)
        if frame.last_seen_zlib:
            last_seen = (
                np.frombuffer(zlib.decompress(frame.last_seen_zlib), dtype=np.float32)
                .reshape(frame.grid_shape)
                .copy()
            )
        else:
            last_seen = np.full(frame.grid_shape, -1.0, dtype=np.float32)

        if (
            grid.shape != belief.grid.shape
            or explored.shape != belief.explored_by_robot.shape
            or visit_count.shape != belief.visit_count.shape
            or last_seen.shape != belief.last_seen.shape
        ):
            # Geometry changed since capture (resolution/bounds/robot_count) --
            # cannot restore safely.
            return False

        # 1. Pause.
        self.paused = True

        # 2. Simulation clock.
        self.simulation_time = float(snapshot.simulation_time)
        self.last_time = time.perf_counter()

        # 3. Robot pose / kinematics.
        self.robot.x = float(snapshot.robot_pose.x)
        self.robot.y = float(snapshot.robot_pose.y)
        self.robot.theta = float(snapshot.robot_pose.theta)
        self.robot.v = float(snapshot.robot_pose.v)

        # 4. Belief map + explored area (authoritative -- see docstring).
        # visit_count/last_seen are restored alongside grid/explored_by_robot
        # so average_seen_penalty()/stats() (revisit_ratio, etc.) read the
        # exact historical values instead of whatever the live run had
        # accumulated past this point.
        belief.grid = grid
        belief.explored_by_robot = explored
        belief.visit_count = visit_count
        belief.last_seen = last_seen
        belief.revision += 1
        self._nav_debug_belief_frame_key = None
        self._nav_debug_belief_frame_cache = None
        self.sync_legacy_map_views_from_belief()

        # 4b. Hazards -- a layer fully separate from occupancy (see
        # HazardField's module docstring); this never touches belief.grid
        # above. Ground-truth FireSources and the team's discovered
        # HazardBelief are restored independently of each other (see
        # HazardBeliefDebug's own docstring) -- a fire that was never
        # observed before capture stays unobserved after restore, exactly
        # as it was; HazardBelief is never derived from FireSource/
        # HazardField here.
        hazard_service = self.ensure_hazard_service()
        if not snapshot.hazard.unavailable and snapshot.hazard.value is not None:
            hazard_frame = snapshot.hazard.value
            restored_sources = tuple(
                FireSource(
                    fire_id=source.fire_id,
                    position=source.position,
                    intensity=source.intensity,
                    radius=source.radius,
                )
                for source in hazard_frame.sources
            )
            hazard_service.field.restore_sources(restored_sources, next_fire_id=hazard_frame.next_fire_id)

        # HazardBelief -- discovered-only. A snapshot captured before this
        # field existed (hazard_belief.unavailable), a shape/robot_count
        # mismatch (geometry changed since capture), or a corrupt/empty
        # byte payload all fall back to the SAME safe, DETERMINISTIC result:
        # an empty HazardBelief at revision exactly 0 (see _restore_empty_
        # hazard_belief() -- never HazardBelief.clear(), whose resulting
        # revision depends on whatever state existed before the restore).
        # Never HazardField as a stand-in for "observed" -- that is exactly
        # the omniscience leak this whole feature exists to prevent. No
        # sensor re-run, no marking every ground-truth cell observed.
        hazard_belief_maybe = snapshot.hazard_belief
        restored_belief = False
        if not hazard_belief_maybe.unavailable and hazard_belief_maybe.value is not None:
            belief_debug = hazard_belief_maybe.value
            try:
                values = (
                    np.frombuffer(zlib.decompress(belief_debug.values_zlib), dtype=np.float32)
                    .reshape(belief_debug.shape)
                    .copy()
                )
                observed_packed = np.frombuffer(
                    zlib.decompress(belief_debug.observed_packbits_zlib), dtype=np.uint8
                )
                observed = np.unpackbits(
                    observed_packed, bitorder="little", count=int(np.prod(belief_debug.shape))
                ).reshape(belief_debug.shape).astype(bool, copy=False)
                observed_by_robot_shape = (belief_debug.robot_count,) + tuple(belief_debug.shape)
                observed_by_robot_packed = np.frombuffer(
                    zlib.decompress(belief_debug.observed_by_robot_packbits_zlib), dtype=np.uint8
                )
                observed_by_robot = np.unpackbits(
                    observed_by_robot_packed, bitorder="little", count=int(np.prod(observed_by_robot_shape))
                ).reshape(observed_by_robot_shape).astype(bool, copy=False)

                # HazardBelief.restore() itself validates shape/dtype
                # against this instance's own geometry/robot_count and
                # raises ValueError on mismatch -- caught below, same
                # fallback as a snapshot with no hazard_belief at all.
                hazard_service.belief.restore(
                    HazardBeliefFrame(
                        values=values,
                        observed=observed,
                        observed_by_robot=observed_by_robot,
                        revision=int(belief_debug.revision),
                    )
                )
                restored_belief = True
            except (ValueError, zlib.error):
                restored_belief = False

        if not restored_belief:
            self._restore_empty_hazard_belief(hazard_service)

        # Any cached hazard-belief debug frame keyed on the now-discarded
        # future is stale.
        self._nav_debug_hazard_belief_frame_key = None
        self._nav_debug_hazard_belief_frame_cache = None

        self.push_hazard_snapshot()
        # Push the just-restored belief to the canvas exactly once. Hazard
        # observation runs synchronously inside record_explored_area() (no
        # in-flight async worker can land later and overwrite this), so
        # there is nothing else that could race this push.
        self.push_discovered_hazard_frame()
        self._discovered_hazard_render_dirty = False

        # 5. mapped_obstacle_points is append-only at runtime (see
        # update_sensed_obstacles()), so truncating to the snapshot's own
        # count reproduces the exact boundary-sample set known at capture
        # time without needing to store the points themselves.
        count = max(0, int(snapshot.mapped_obstacle_points_count))
        self.mapped_obstacle_points = list(self.mapped_obstacle_points[:count])
        self.mapped_obstacle_point_keys = {
            (round(float(p[0]), 3), round(float(p[1]), 3)) for p in self.mapped_obstacle_points
        }

        # 6. Cosmetic trails. The bounded sensor-sweep polygon list itself
        # (self.explored_area_polygons, capped to EXPLORED_POLYGON_HISTORY_
        # LIMIT sweeps) is not part of the authoritative contract and is
        # cleared -- but the visible explored-area *coverage* must not
        # regress just because that list is gone: the canvas is seeded
        # directly from the just-restored belief.explored_by_robot mask
        # (the authoritative state, already rolled back above), using the
        # same per-cell rasterization the historical-replay view uses. New
        # sensor sweeps recorded after this point paint on top of the seed
        # as usual. The executed-path trail has no equivalent authoritative
        # source to reseed from, so it resets to the single restored point.
        self.explored_area_polygons = []
        self.canvas.set_explored_area_polygons(self.explored_area_polygons)
        self.canvas.set_explored_area_seed(explored, float(frame.resolution), belief.bounds)
        self.path_points = [(self.robot.x, self.robot.y)]

        # 7. Route / active waypoint index, then explicit agent state on top
        # -- never inferred from active_path[-1] (see AgentStateDebug).
        active_path = list(snapshot.path.active_path)
        agent = self.runtime_agent(None)
        if active_path:
            self.robot.set_waypoints(active_path)
            clamped_index = max(0, min(int(snapshot.path.active_waypoint_index or 0), len(active_path) - 1))
            self.robot.waypoints.current_index = clamped_index
            if agent is not None:
                agent.waypoints.set_waypoints(active_path)
                agent.waypoints.current_index = clamped_index
        else:
            self.robot.waypoints.clear()
            self.robot.state_machine.reset()
            if agent is not None:
                agent.waypoints.clear()

        # 7a. Tracking FSM mode -- both branches above reset it to IDLE
        # (set_waypoints() does this as a side effect; the else branch does
        # it explicitly), so this must run AFTER them to actually restore
        # the captured ROTATE/TRACK/etc. mode instead of losing it. Uses
        # the state machine's own public restore API rather than writing
        # robot.state_machine.mode directly from here.
        self.robot.state_machine.restore_mode(snapshot.tracking_mode)

        if agent is not None:
            # Pending/prefetch route state and route-repair bookkeeping are
            # always cleared regardless of branch above -- a path prefetched
            # in the discarded future must never be promoted after restore.
            agent.pending_path = None
            agent.pending_target_xy = None
            agent.pending_path_route_generation = None
            agent.pending_path_created_for_active_goal = None
            agent.route_repair_in_progress_for_goal = None

            agent_state = snapshot.agent_state
            if not agent_state.unavailable and agent_state.value is not None:
                state = agent_state.value
                agent.final_goal_xy = state.final_goal_xy
                if state.final_goal_xy is not None:
                    # final_goal_xy() (the authoritative source select_
                    # navigation_goal()/replans read) returns config.goal_x/
                    # goal_y directly, NOT agent.final_goal_xy -- restoring
                    # only the agent field above left config/the GUI still
                    # showing whatever goal the live run had moved on to. No
                    # coordinate to restore config/widgets to when the
                    # snapshot itself had no final goal (state.final_goal_xy
                    # is None) -- config.goal_x/y has no "unset" state, so it
                    # is left as-is rather than forced to some placeholder.
                    self._restore_final_goal_into_config_and_widgets(state.final_goal_xy)
                agent.exploration_target_xy = state.exploration_target_xy
                agent.active_path_goal_xy = state.active_path_goal_xy
                agent.active_path_mode = state.active_path_mode
                agent.route_generation = state.route_generation
                agent.route_affected_replan_count = state.route_affected_replan_count
                agent.first_segment_blocked_count = state.first_segment_blocked_count
                agent.last_frontier_candidate_count = state.last_frontier_candidate_count
                agent.prefetch_success_count = state.prefetch_success_count
                agent.prefetch_fail_count = state.prefetch_fail_count
                agent.safety_replan_count = state.safety_replan_count
                agent.target_switch_count = state.target_switch_count
            else:
                # Degraded/older snapshot without explicit agent state --
                # fall back to inferring active_path_goal_xy from the route
                # itself rather than leaving whatever the live agent had
                # before this restore (a value from the discarded future).
                agent.active_path_goal_xy = active_path[-1] if active_path else None

            agent.status = snapshot.navigation_state or agent.status
            self.current_exploration_target = agent.exploration_target_xy
        else:
            self.current_exploration_target = None
        self.canvas.set_exploration_target(self.current_exploration_target)

        # 7b. Engine-level cumulative metrics (see RuntimeMetricsDebug) --
        # restored so none of them reads ahead of the rewound simulation_time.
        metrics = snapshot.metrics
        if not metrics.unavailable and metrics.value is not None:
            m = metrics.value
            self.total_distance_traveled = m.total_distance_traveled
            self.route_request_count = m.route_request_count
            self.route_result_count = m.route_result_count
            self.route_failure_count = m.route_failure_count
            self.sensor_update_count = m.sensor_update_count
            self.mapping_update_count = m.mapping_update_count
            self.safety_replan_count = m.safety_replan_count
            self.exploration_replan_count = m.exploration_replan_count
            self.planner_jobs_started = m.planner_jobs_started
            self.planner_jobs_completed = m.planner_jobs_completed

        # 8. Invalidate in-flight async planner work for the truncated
        # future -- identical pattern to start_simulation()/reset_simulation().
        self.planning_in_progress = False
        self.route_request_id += 1
        self.active_planner_workers.clear()
        self._invalidate_all_prefetch_requests(reason="simulation reset/restore")
        self._nav_debug_pending_plan_capture_by_robot = {}
        self._nav_debug_last_plan_capture = None
        self._nav_debug_last_accepted_plan = None

        # 9. Truncate history at this point; continue numbering from here.
        log.truncate_after(int(index))
        self._nav_debug_seq = int(snapshot.snapshot_id)
        self._nav_debug_live_snapshot = snapshot

        # 10. Return the view to LIVE.
        self._nav_debug_history_index = None

        self.canvas.set_robot(self.robot)
        self.canvas.set_path(self.path_points)
        self.canvas.set_planned_path([(self.robot.x, self.robot.y)] + active_path)
        self.canvas.set_mapped_obstacle_points(self.mapped_obstacle_points)
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)
        self.canvas.set_navigation_debug_snapshot(snapshot)
        self.canvas.set_navigation_debug_last_event(log.latest())
        self.canvas.set_navigation_debug_history_position(None, len(log))
        self.canvas.set_status(
            f"Resumed simulation from snapshot #{snapshot.snapshot_id} (t={snapshot.simulation_time:.2f}s)."
        )

        self.update_start_pause_button()
        self.update_navigation_debug_step_buttons()
        return True

    def handle_start_pause_button(self) -> None:
        has_runtime_robot = self.robot is not None or bool(getattr(self, "robots", []))
        if not self.running or not has_runtime_robot:
            self.start_simulation()
            return

        self.toggle_pause()

    def cycle_simulation_speed(self) -> None:
        self.simulation_speed_index = (
            self.simulation_speed_index + 1
        ) % len(self.simulation_speed_options)
        self.simulation_speed = self.simulation_speed_options[self.simulation_speed_index]
        self.speed_button.setText(f"Speed {self.simulation_speed:.2f}x")
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)
        self.canvas.set_status(f"Simulation speed set to {self.simulation_speed:.2f}x.")

    def restart_simulation(self) -> None:
        """
        Reset the run and leave the simulator stopped.

        This button returns the simulator to the configured initial state. It
        must not auto-start; the user explicitly presses Start Simulation when
        ready to run again.
        """
        self.reset_simulation()
        self.start_belief_trace_run()
        self.canvas.set_status("Restart complete. Press Start Simulation to run.")

    # ========================================================
    # SIMULATION
    # ========================================================

    def create_robot_instance(self, start_cfg: RobotStartConfig):
        """Create one Robot from a per-robot start configuration."""
        body_radius = max(0.01, float(start_cfg.body_radius))
        safety_radius = max(float(start_cfg.safety_radius), body_radius)
        initial_goal = (
            (float(start_cfg.x), float(start_cfg.y))
            if self.is_exploration_mode()
            else (float(self.config.goal_x), float(self.config.goal_y))
        )

        robot_kwargs = dict(
            x=float(start_cfg.x),
            y=float(start_cfg.y),
            theta=float(start_cfg.theta),
            v=float(start_cfg.v),
            vision=float(start_cfg.vision),
            goal=initial_goal,
            max_speed=float(start_cfg.max_speed),
            max_acceleration=float(start_cfg.max_acceleration),
            max_angular_speed=float(start_cfg.max_angular_speed),
            goal_tolerance=float(start_cfg.goal_tolerance),
            robot_radius=body_radius,
        )

        try:
            robot = Robot(**robot_kwargs)
        except TypeError:
            robot_kwargs.pop("robot_radius", None)
            robot = Robot(**robot_kwargs)

            limits = getattr(robot, "limits", None)
            if limits is not None and hasattr(limits, "robot_radius"):
                limits.robot_radius = body_radius

        # Store simulator-side radii/dynamics because the Robot class may not
        # expose a dedicated safety-radius field. Collision checking and
        # drawing read these attributes when present.
        robot._sim_body_radius = body_radius
        robot._sim_safety_radius = safety_radius
        robot._sim_acceleration_gain = float(start_cfg.acceleration_gain)
        robot._sim_goal_tolerance = float(start_cfg.goal_tolerance)

        self.apply_controller_parameters(robot, acceleration_gain=float(start_cfg.acceleration_gain))
        return robot

    def set_robot_goal_or_waypoints(self, robot, waypoints: list[tuple[float, float]]) -> None:
        if not waypoints:
            waypoints = [self.final_goal_xy()]

        # Always give each robot its own waypoint list. Reusing the same list
        # object across robots can make debugging look like a robot is following
        # another robot's route.
        robot_waypoints = [(float(point[0]), float(point[1])) for point in waypoints]

        if hasattr(robot, "set_waypoints"):
            robot.set_waypoints(robot_waypoints)
        elif hasattr(robot, "set_goal"):
            robot.set_goal(robot_waypoints[-1])
        else:
            robot.goal = np.array(robot_waypoints[-1], dtype=float)

    def current_route_points_for_robot(self, robot) -> list[tuple[float, float]]:
        """
        Return the remaining route assigned to a specific runtime robot.

        This is the multi-robot equivalent of current_route_points(). It lets
        the safety/replanning logic check whether newly mapped obstacle points
        actually invalidate that robot's active route.
        """
        if robot is None:
            return []

        points: list[tuple[float, float]] = [(float(robot.x), float(robot.y))]
        waypoint_manager = getattr(robot, "waypoints", None)
        raw_waypoints = getattr(waypoint_manager, "waypoints", None)
        current_index = getattr(waypoint_manager, "current_index", None)

        if raw_waypoints is not None and current_index is not None:
            for waypoint in raw_waypoints[int(current_index):]:
                waypoint_array = np.asarray(waypoint, dtype=float).reshape(-1)
                if waypoint_array.size >= 2:
                    points.append((float(waypoint_array[0]), float(waypoint_array[1])))
        else:
            goal = getattr(robot, "goal", None)
            if goal is not None:
                goal_array = np.asarray(goal, dtype=float).reshape(-1)
                if goal_array.size >= 2:
                    points.append((float(goal_array[0]), float(goal_array[1])))

        cleaned: list[tuple[float, float]] = []
        for point in points:
            if not cleaned or math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-6:
                cleaned.append(point)
        return cleaned

    def ensure_multi_replan_guard_slots(self) -> None:
        """Create per-robot cooldown state for repeated replanning triggers.

        Safety checks can fire every frame while a robot is stopped in front of
        a known obstacle. Without a guard, the simulator accepts the same route,
        rejects the same first segment, and logs hundreds of identical route
        assignments. This guard throttles identical replans while keeping the
        robot braked.
        """
        count = len(getattr(self, "robots", []))

        if not hasattr(self, "multi_last_safety_replan_sim_times"):
            self.multi_last_safety_replan_sim_times = []
        if len(self.multi_last_safety_replan_sim_times) < count:
            self.multi_last_safety_replan_sim_times.extend([-1.0e9] * (count - len(self.multi_last_safety_replan_sim_times)))
        elif len(self.multi_last_safety_replan_sim_times) > count:
            self.multi_last_safety_replan_sim_times = self.multi_last_safety_replan_sim_times[:count]

        if not hasattr(self, "multi_last_safety_replan_signatures"):
            self.multi_last_safety_replan_signatures = []
        if len(self.multi_last_safety_replan_signatures) < count:
            self.multi_last_safety_replan_signatures.extend([None] * (count - len(self.multi_last_safety_replan_signatures)))
        elif len(self.multi_last_safety_replan_signatures) > count:
            self.multi_last_safety_replan_signatures = self.multi_last_safety_replan_signatures[:count]

        if not hasattr(self, "multi_last_exploration_replan_sim_times"):
            self.multi_last_exploration_replan_sim_times = []
        if len(self.multi_last_exploration_replan_sim_times) < count:
            self.multi_last_exploration_replan_sim_times.extend([-1.0e9] * (count - len(self.multi_last_exploration_replan_sim_times)))
        elif len(self.multi_last_exploration_replan_sim_times) > count:
            self.multi_last_exploration_replan_sim_times = self.multi_last_exploration_replan_sim_times[:count]

    def safety_replan_cooldown_seconds(self) -> float:
        """Shared cooldown formula for throttling identical safety replans.

        Used by both multi_safety_replan_allowed() (per-robot-index, engine
        state) and the single-robot REPLAN_FOR_SAFETY branch of
        apply_navigation_decision() (per-agent, RobotAgent.safety_replan_allowed()),
        so the two throttles behave consistently.
        """
        return max(0.35, 0.75 * max(0.1, float(self.config.exploration_replan_cooldown)))

    def route_affected_replan_cooldown_seconds(self) -> float:
        """Cooldown for throttling repeated route_affected repair replans
        for the same path_goal (see RobotAgent.route_affected_replan_allowed()).

        Deliberately longer than safety_replan_cooldown_seconds(): unlike a
        safety flag, route_affected fires on ordinary sensor-driven map
        growth -- near a narrow passage this can trigger on nearly every
        sensor-update tick as boundary samples accumulate, so it can
        afford to wait longer between repair attempts for the same
        target. A directly unsafe active segment always bypasses this
        cooldown regardless (see active_segment_unsafe).
        """
        return max(0.75, 1.5 * max(0.1, float(self.config.exploration_replan_cooldown)))

    def sync_narrow_passage_speed_cap(self, agent) -> None:
        """Apply or lift the transient narrow-passage speed cap.

        While agent.is_narrow_passage_slowdown_active() (armed by repeated,
        throttled route_affected replans for the same path_goal -- see
        RobotAgent.route_affected_replan_allowed()), commanded max speed is
        temporarily reduced via the robot's existing max_speed setter, NOT
        config.max_speed and NOT robot dynamics. Restored to the configured
        value the moment the window expires. Purely a runtime control hint;
        a no-op when the agent/robot are unavailable.
        """
        if agent is None or self.robot is None or not hasattr(self.robot, "max_speed"):
            return

        if agent.is_narrow_passage_slowdown_active(float(self.simulation_time)):
            self.robot.max_speed = min(
                float(self.config.max_speed), agent._NARROW_PASSAGE_SLOWDOWN_SPEED_CAP
            )
        else:
            self.robot.max_speed = float(self.config.max_speed)

    def multi_safety_replan_allowed(
        self,
        robot_index: int,
        reason: str,
        target: tuple[float, float] | None,
    ) -> bool:
        """Throttle identical safety replans for a robot.

        Returning False means: keep the robot stopped this frame and retry later,
        instead of logging the same rejected route again.
        """
        self.ensure_multi_replan_guard_slots()
        robot_index = int(robot_index)
        if not (0 <= robot_index < len(self.multi_last_safety_replan_sim_times)):
            return True

        cooldown = self.safety_replan_cooldown_seconds()
        target_key = None
        if target is not None:
            target_key = (round(float(target[0]), 2), round(float(target[1]), 2))
        signature = (str(reason), target_key)
        elapsed = float(self.simulation_time) - float(self.multi_last_safety_replan_sim_times[robot_index])
        same_signature = signature == self.multi_last_safety_replan_signatures[robot_index]
        if same_signature and elapsed < cooldown:
            return False

        self.multi_last_safety_replan_sim_times[robot_index] = float(self.simulation_time)
        self.multi_last_safety_replan_signatures[robot_index] = signature
        return True

    def multi_exploration_target_replan_allowed(self, robot_index: int) -> bool:
        """Per-robot cooldown for target-reached frontier replans."""
        self.ensure_multi_replan_guard_slots()
        robot_index = int(robot_index)
        if not (0 <= robot_index < len(self.multi_last_exploration_replan_sim_times)):
            return True
        cooldown = max(0.25, float(self.config.exploration_replan_cooldown))
        elapsed = float(self.simulation_time) - float(self.multi_last_exploration_replan_sim_times[robot_index])
        if elapsed < cooldown:
            return False
        self.multi_last_exploration_replan_sim_times[robot_index] = float(self.simulation_time)
        return True

    def route_points_intersect_new_map_information(
        self,
        route_points: list[tuple[float, float]],
        mapped_points: list[tuple[float, float]],
        robot_radius: float | None = None,
    ) -> bool:
        if self.collision_checker is None or len(route_points) < 2 or not mapped_points:
            return False

        robot_radius = self.safety_radius() if robot_radius is None else float(robot_radius)
        for start, end in zip(route_points[:-1], route_points[1:]):
            report = self.collision_checker.check_segment_points(
                start=start,
                end=end,
                obstacle_points=mapped_points,
                robot_radius=robot_radius,
            )
            if report.collision:
                return True
        return False

    def hold_multi_robot_position(
        self,
        robot_index: int,
        reason: str = "",
        *,
        state: str | None = None,
    ) -> bool:
        """Assign a zero-length hold target to one robot.

        This is critical in exploration mode: if no valid frontier exists or the
        path planner fails, the robot must *not* fall back to the shared final
        goal G. G is only a visual mission reference while a frontier planner is
        active.

        state lets a caller that already knows *why* it is holding (e.g. a
        corridor rejection, not a missing frontier) say so explicitly instead
        of relying on substring-sniffing the reason text below, which only
        covers the case where nothing more specific is known.
        """
        if not (0 <= int(robot_index) < len(self.robots)):
            return False

        robot_index = int(robot_index)
        robot = self.robots[robot_index]
        hold_xy = (float(robot.x), float(robot.y))

        self.set_robot_goal_or_waypoints(robot, [hold_xy])

        while len(self.multi_planned_path_points) <= robot_index:
            self.multi_planned_path_points.append([])
        self.multi_planned_path_points[robot_index] = [hold_xy]

        self.ensure_multi_exploration_target_slots()
        if 0 <= robot_index < len(self.multi_exploration_targets):
            self.multi_exploration_targets[robot_index] = None
        self.publish_multi_exploration_targets()

        if reason:
            self.last_goal_selection_reason = f"R{robot_index + 1}: holding position; {reason}"
        else:
            self.last_goal_selection_reason = f"R{robot_index + 1}: holding position"

        reason_text = str(reason or "")
        if state is not None:
            resolved_state = state
        else:
            resolved_state = self.ROUTE_STATE_HOLD_NO_FRONTIER
            if "collision" in reason_text.lower() or "blocked" in reason_text.lower() or "safety" in reason_text.lower():
                resolved_state = self.ROUTE_STATE_STUCK_SAFETY
        self.set_multi_route_state(robot_index, resolved_state, reason_text or "hold position")
        return True

    def assign_route_to_multi_robot(
        self,
        robot_index: int,
        reason: str = "",
        force_new_exploration_target: bool = False,
    ) -> bool:
        """
        Assign a route to one runtime robot using the shared planner selectors.

        Other robots are treated as dynamic obstacles during planning. In
        exploration modes, the shared final goal is ignored and each robot gets
        a frontier target instead.
        """
        return self._assign_route_to_multi_robot_with_corridor_validation(
            robot_index,
            reason=reason,
            force_new_exploration_target=force_new_exploration_target,
            remaining_corridor_retries=self.MAX_ROUTE_RECOVERY_ATTEMPTS - 1,
        )

    def compute_grid_safe_fallback_route_for_multi_robot(
        self,
        robot_index: int,
        force_new_exploration_target: bool = False,
    ) -> tuple[bool, str, list[tuple[float, float]]]:
        """One-off A* fallback used only when Direct's corridor is rejected.

        This does not change self.config.planner_type -- Direct stays the
        globally selected planner. It just asks the grid-safe planner for one
        alternate route to the same goal before the target itself is given
        up on, since a straight line can be blocked while a route around the
        obstruction is not.
        """
        if compute_planned_waypoints is None:
            return False, "planner package is not available", []

        planner_kwargs, goal_reason = self.build_planner_kwargs_for_multi_robot(
            robot_index,
            force_new_exploration_target=force_new_exploration_target,
        )
        if bool(planner_kwargs.get("__hold__", False)):
            return False, goal_reason, []

        fallback_kwargs = dict(planner_kwargs)
        fallback_kwargs["planner_type"] = "A*"

        success, reason, waypoints = self.call_compute_planned_waypoints(
            fallback_kwargs,
            path_simplifier=self.config.path_simplifier,
        )
        return bool(success), f"{goal_reason}; grid-safe fallback (A*): {reason}", waypoints

    def _activate_multi_robot_route(
        self,
        robot_index: int,
        robot,
        old_robot,
        waypoints: list[tuple[float, float]],
        route_reason: str,
        reason: str,
    ) -> bool:
        """Shared tail: commit a validated route as ACTIVE and restore self.robot."""
        self.set_robot_goal_or_waypoints(robot, waypoints)
        self.set_multi_route_state(robot_index, self.ROUTE_STATE_ACTIVE, route_reason)

        while len(self.multi_planned_path_points) <= robot_index:
            self.multi_planned_path_points.append([])
        self.multi_planned_path_points[robot_index] = [(float(robot.x), float(robot.y))] + list(waypoints)

        self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
        self.route_request_count += 1
        self.route_result_count += 1
        if reason:
            self.last_goal_selection_reason = f"R{robot_index + 1}: {reason}; {route_reason}"
        else:
            self.last_goal_selection_reason = route_reason
        self.log_route_assignment(
            robot_index,
            (float(robot.x), float(robot.y)),
            list(waypoints),
            self.last_goal_selection_reason,
        )
        self.publish_multi_exploration_targets()
        return True

    def _assign_route_to_multi_robot_with_corridor_validation(
        self,
        robot_index: int,
        *,
        reason: str,
        force_new_exploration_target: bool,
        remaining_corridor_retries: int,
    ) -> bool:
        if not (0 <= int(robot_index) < len(self.robots)):
            return False

        robot_index = int(robot_index)
        robot = self.robots[robot_index]
        old_robot = self.robot
        self.robot = robot

        success, route_reason, waypoints = self.compute_route_for_multi_robot(
            robot_index,
            force_new_exploration_target=force_new_exploration_target,
        )

        if (not success or not waypoints) and self.is_exploration_mode():
            held = self.hold_multi_robot_position(
                robot_index,
                f"no valid exploration route; {route_reason}",
            )
            self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
            return held

        if not success or not waypoints:
            waypoints = [self.final_goal_xy()]

        obstacle_points = list(self.mapped_obstacle_points) + self.dynamic_robot_obstacle_points_for_robot(robot_index)
        waypoints = self.clean_waypoints_for_robot(robot, waypoints, obstacle_points=obstacle_points)

        if not waypoints:
            if self.is_exploration_mode():
                held = self.hold_multi_robot_position(
                    robot_index,
                    f"target already reached or no safe frontier waypoint; {route_reason}",
                )
                self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
                return held
            waypoints = [self.final_goal_xy()]

        # Validate the FULL corridor against teammates before this route is
        # allowed to become ACTIVE. This runs earlier and stricter than the
        # per-frame movement safety veto (segment_violates_other_robot_clearance
        # in the movement loop), which only checks the immediate next segment
        # once a route is already active -- that veto remains as a final
        # backstop, it is just no longer the first place a conflict is caught.
        # Direct is included: a single-segment route is still a full corridor.
        if self.is_exploration_mode():
            corridor_check = validate_multi_robot_corridor(
                start=(float(robot.x), float(robot.y)),
                waypoints=waypoints,
                ego_safety_radius=float(self.safety_radius_for_robot(robot)),
                other_robot_disks=self.dynamic_robot_obstacles_for_target_selection(robot_index),
                other_routes=[
                    route
                    for j, route in enumerate(self.multi_active_route_points_by_robot())
                    if j != robot_index
                ],
                margin=self.multi_dynamic_target_margin(),
            )
            if not corridor_check.is_valid:
                self.log_console_message(
                    f"R{robot_index + 1}: route candidate rejected: reason={corridor_check.reason_code}; "
                    f"{corridor_check.detail}"
                )

                # A straight line can be blocked while a route around the
                # obstruction is not. Try the grid-safe planner once, for the
                # SAME target, before giving up on it -- this only applies
                # when Direct is the globally selected planner AND no plugin
                # owns PATH_PLANNING. A PATH_PLANNING-owning plugin's
                # command.path is authoritative (see compute_route_for_multi_
                # robot/select_runtime_path_source); this local A* fallback
                # must not silently override it just because the (now
                # disabled) planner combo still shows "Direct".
                if (
                    self.config.planner_type == "Direct"
                    and not self.coordinator_runtime_profile().owns_path_planning
                ):
                    self.log_console_message(
                        f"R{robot_index + 1}: Direct route rejected, trying A* fallback"
                    )
                    fb_success, fb_reason, fb_waypoints = self.compute_grid_safe_fallback_route_for_multi_robot(
                        robot_index,
                        force_new_exploration_target=False,
                    )
                    if fb_success and fb_waypoints:
                        fb_waypoints = self.clean_waypoints_for_robot(
                            robot, fb_waypoints, obstacle_points=obstacle_points
                        )
                    if fb_success and fb_waypoints:
                        fb_corridor_check = validate_multi_robot_corridor(
                            start=(float(robot.x), float(robot.y)),
                            waypoints=fb_waypoints,
                            ego_safety_radius=float(self.safety_radius_for_robot(robot)),
                            other_robot_disks=self.dynamic_robot_obstacles_for_target_selection(robot_index),
                            other_routes=[
                                route
                                for j, route in enumerate(self.multi_active_route_points_by_robot())
                                if j != robot_index
                            ],
                            margin=self.multi_dynamic_target_margin(),
                        )
                        if fb_corridor_check.is_valid:
                            _LOGGER.debug(
                                "R%d: route_accepted_after_corridor_validation (A* fallback)",
                                robot_index + 1,
                            )
                            return self._activate_multi_robot_route(
                                robot_index, robot, old_robot, fb_waypoints, fb_reason, reason
                            )

                self.invalidate_current_multi_frontier(robot_index, corridor_check.detail)
                self.log_console_message(
                    f"R{robot_index + 1}: target_blacklisted_after_route_rejection"
                )

                if remaining_corridor_retries > 0:
                    attempt_number = self.MAX_ROUTE_RECOVERY_ATTEMPTS - remaining_corridor_retries
                    self.log_console_message(
                        f"R{robot_index + 1}: route rejected, trying alternative target "
                        f"{attempt_number + 1}/{self.MAX_ROUTE_RECOVERY_ATTEMPTS}"
                    )
                    self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
                    retry_reason = f"retry after {corridor_check.reason_code}"
                    if reason:
                        retry_reason = f"{reason}; {retry_reason}"
                    return self._assign_route_to_multi_robot_with_corridor_validation(
                        robot_index,
                        reason=retry_reason,
                        force_new_exploration_target=True,
                        remaining_corridor_retries=remaining_corridor_retries - 1,
                    )

                # Candidates exhausted. A conflict with a teammate's active
                # route is transient (they are moving; the corridor may clear
                # on its own), so it waits rather than reporting a permanent
                # hold. Any other corridor conflict is reported as a blocked
                # route, not as "no frontier" -- a target/frontier did exist.
                if corridor_check.reason_code == "route_conflict_with_active_route":
                    self.log_console_message(
                        f"R{robot_index + 1}: waiting for corridor instead of HOLD_NO_FRONTIER"
                    )
                    held = self.hold_multi_robot_position(
                        robot_index,
                        f"waiting for corridor; {corridor_check.detail}",
                        state=self.ROUTE_STATE_WAITING_FOR_CORRIDOR,
                    )
                else:
                    self.log_console_message(
                        f"R{robot_index + 1}: HOLD_ROUTE_BLOCKED; candidates exhausted"
                    )
                    held = self.hold_multi_robot_position(
                        robot_index,
                        f"no safe corridor available after retry; {corridor_check.detail}",
                        state=self.ROUTE_STATE_HOLD_ROUTE_BLOCKED,
                    )
                self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
                return held

            _LOGGER.debug("R%d: route_accepted_after_corridor_validation", robot_index + 1)

        return self._activate_multi_robot_route(robot_index, robot, old_robot, waypoints, route_reason, reason)

    def replan_multi_robots_affected_by_points(
        self,
        newly_mapped: list[tuple[float, float]],
        reason: str,
    ) -> int:
        """
        Replan only the robots whose current routes are invalidated by new map data.

        This runs regardless of planner_type: a Direct route is a straight
        line to a target, which can still be crossed by a newly discovered
        obstacle. assign_route_to_multi_robot() resolves Direct/A*/Dijkstra/
        plugin-owned paths uniformly, so there is nothing Direct-specific to
        special-case here.
        """
        if not newly_mapped:
            return 0

        replanned = 0
        for index, robot in enumerate(self.robots):
            route_points = self.current_route_points_for_robot(robot)
            if self.route_points_intersect_new_map_information(
                route_points,
                newly_mapped,
                robot_radius=self.safety_radius_for_robot(robot),
            ):
                if self.assign_route_to_multi_robot(index, reason=reason):
                    replanned += 1
        if replanned:
            self.safety_replan_count += replanned
        return replanned

    def start_multi_robot_simulation(self):
        """
        Start the executable multi-robot baseline.

        Current policy:
            - all robots share the global Planner / Path Simplifier selectors;
            - each robot receives its own route from its own current position;
            - the map is shared, but sensing/explored-area layers stay colored
              per robot;
            - if Same Configuration is OFF, only pose/initial-v overrides are
              per robot for now. Per-robot planner selection is intentionally a
              later experiment, because it would complicate comparisons.
        """
        self.spatial_index.rebuild(self.config.obstacles)
        self.planning_in_progress = False
        self.route_request_id += 1
        self.active_planner_workers.clear()
        self._invalidate_all_prefetch_requests(reason="simulation reset/restore")

        self.ensure_multi_robot_configs()
        robot_starts = normalized_robot_start_configs(self.config)
        self.robots = [self.create_robot_instance(start_cfg) for start_cfg in robot_starts]
        self.robot = self.robots[0] if self.robots else None
        self.sync_runtime_robot_agents()

        # Reset shared mapping/metrics before the first routes are computed.
        # The previous version computed routes first and reset the map after,
        # which made multi-robot planning look like it ignored the selected
        # planner or used stale information from a previous run.
        self.known_obstacles = []
        self.explored_area_polygons = []
        self.reset_belief_map(robot_count=len(self.robots) if getattr(self, "robots", None) else 1, preserve_hazards=True)
        self.current_exploration_target = None
        self.multi_exploration_targets = []
        self.multi_invalidated_exploration_targets = []
        self.last_exploration_replan_sim_time = -1.0e9
        self.last_exploration_gate_message_time = -1.0e9
        self.last_goal_selection_reason = "multi-robot baseline using shared final goal"
        self.route_request_count = 0
        self.route_result_count = 0
        self.navigation_debug_log = NavigationDebugEventLog()
        self._nav_debug_seq = 0
        self._nav_debug_history_index = None
        self._nav_debug_last_accepted_plan = None
        self._nav_debug_live_snapshot = None
        self._nav_debug_pending_plan_capture_by_robot = {}
        self.sensor_update_count = 0
        self.mapping_update_count = 0
        self.safety_replan_count = 0
        self.exploration_replan_count = 0
        self.total_distance_traveled = 0.0
        self.last_explored_pose = None
        self.multi_last_explored_poses = {}
        self.last_sensor_update_time = 0.0
        self.last_sensor_update_pose = None
        self._exhausted_idle_obstacle_count = None

        self.multi_path_points = [[(float(robot.x), float(robot.y))] for robot in self.robots]
        self.multi_robot_commands_by_id = {}
        self.multi_exploration_targets = [None for _ in self.robots]
        self.multi_invalidated_exploration_targets = [[] for _ in self.robots]
        self.multi_planned_path_points = [[] for _ in self.robots]
        self.multi_last_controls = [np.array([[0.0], [0.0]], dtype=float) for _ in self.robots]
        self.multi_route_states = [self.ROUTE_STATE_ACTIVE for _ in self.robots]
        self.multi_route_state_reasons = [""] * len(self.robots)
        self.multi_last_route_state_log_times = [-1.0e9] * len(self.robots)
        self.path_points = self.multi_path_points[0] if self.multi_path_points else []
        self.last_control = self.multi_last_controls[0] if self.multi_last_controls else np.array([[0.0], [0.0]], dtype=float)
        self.last_motion_log_time = -1.0e9
        self.multi_last_motion_log_times = {}
        self.simulation_time = 0.0
        self.last_time = time.perf_counter()

        self.log_console_message(self.simulation_start_summary(multi=True))

        self.canvas.set_mapped_obstacle_points(self.mapped_obstacle_points)
        self.push_hazard_snapshot()
        self.canvas.set_explored_area_polygons(self.explored_area_polygons)
        # A previous run's restore must not leak its seeded explored-area
        # coverage into this fresh one.
        self.canvas.clear_explored_area_seed()
        self.canvas.set_known_obstacles(self.known_obstacles)
        self.canvas.set_planned_path([])
        self.canvas.set_exploration_target(None)
        self.canvas.set_multi_exploration_targets(self.multi_exploration_targets)
        self.canvas.invalidate_explored_area_cache()
        self.canvas.invalidate_sensor_cache()

        # Initialize the shared map from all robot sensors before assigning
        # routes. This lets frontier exploration and obstacle-aware A*/Dijkstra
        # start from the team observation instead of an empty or stale map.
        for robot_index, robot in enumerate(self.robots):
            old_robot = self.robot
            self.robot = robot
            self.record_explored_area(force=True, robot_index=robot_index)
            self.update_sensed_obstacles(force_status=False)
            self.force_robot_pose_free_in_belief(robot_index)
            self.robot = old_robot

        # Global planner applies to every robot. Each robot still gets its own
        # route because the start pose is different.
        for robot_index in range(len(self.robots)):
            self.assign_route_to_multi_robot(robot_index, reason="Initial multi-robot route")

        self.running = True
        self.paused = False
        self.canvas.set_simulation_running_for_perf(True)
        self.set_configuration_locked(True)
        self.update_start_pause_button()
        self.speed_button.setText(f"Speed {self.simulation_speed:.2f}x")
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)
        self.canvas.set_multi_robots(
            self.robots,
            self.multi_path_points,
            self.multi_last_controls,
            planned_path_points=self.multi_planned_path_points,
            exploration_targets=self.multi_exploration_targets,
        )
        self.canvas.set_status(
            f"Multi-robot simulation running with {len(self.robots)} robots. "
            f"Planner shared: {self.config.planner_type}."
        )
        for robot_index, robot in enumerate(self.robots):
            target = (
                self.multi_exploration_targets[robot_index]
                if robot_index < len(self.multi_exploration_targets)
                else self.final_goal_xy()
            )
            self.log_robot_motion(
                robot,
                robot_index=robot_index,
                control=self.multi_last_controls[robot_index] if robot_index < len(self.multi_last_controls) else None,
                target=target,
                force=True,
            )
        self.top_bar.set_status("running")

    def start_simulation(self):
        # Diagnostics/output only: a fresh belief-trace artifact directory
        # for this run (see start_belief_trace_run()), independent of
        # ROBOT_TRACE and covering both the single- and multi-robot paths
        # below since it runs before the mode branch.
        self.start_belief_trace_run()
        self.config = self.read_config()
        if "Multiple" in self.config.agent_mode:
            self.start_multi_robot_simulation()
            return

        self.robots = []
        self.multi_path_points = []
        self.multi_planned_path_points = []
        self.multi_last_controls = []
        self.spatial_index.rebuild(self.config.obstacles)
        self.planning_in_progress = False
        self.route_request_id += 1
        self.active_planner_workers.clear()
        self._invalidate_all_prefetch_requests(reason="simulation reset/restore")

        initial_goal = (
            (float(self.config.x), float(self.config.y))
            if self.is_exploration_mode()
            else (float(self.config.goal_x), float(self.config.goal_y))
        )

        robot_kwargs = dict(
            x=self.config.x,
            y=self.config.y,
            theta=self.config.theta,
            v=self.config.v,
            vision=self.config.vision,
            goal=initial_goal,
            max_speed=self.config.max_speed,
            max_acceleration=self.config.max_acceleration,
            max_angular_speed=self.config.max_angular_speed,
            goal_tolerance=self.config.goal_tolerance,
            robot_radius=self.config.body_radius,
        )

        try:
            self.robot = Robot(**robot_kwargs)
        except TypeError:
            robot_kwargs.pop("robot_radius", None)
            self.robot = Robot(**robot_kwargs)

            limits = getattr(self.robot, "limits", None)
            if limits is not None and hasattr(limits, "robot_radius"):
                limits.robot_radius = self.config.body_radius

        self.apply_controller_parameters()
        self.sync_runtime_robot_agents()

        self.known_obstacles = []
        self.explored_area_polygons = []
        self.reset_belief_map(robot_count=len(self.robots) if getattr(self, "robots", None) else 1, preserve_hazards=True)
        self.current_exploration_target = None
        self.multi_exploration_targets = []
        self.multi_invalidated_exploration_targets = []
        self.last_exploration_replan_sim_time = -1.0e9
        self.last_exploration_gate_message_time = -1.0e9
        self.last_goal_selection_reason = "using final mission goal"
        self.route_request_count = 0
        self.route_result_count = 0
        self.navigation_debug_log = NavigationDebugEventLog()
        self._nav_debug_seq = 0
        self._nav_debug_history_index = None
        self._nav_debug_last_accepted_plan = None
        self._nav_debug_live_snapshot = None
        self._nav_debug_pending_plan_capture_by_robot = {}
        self.sensor_update_count = 0
        self.mapping_update_count = 0
        self.safety_replan_count = 0
        self.exploration_replan_count = 0
        self.total_distance_traveled = 0.0
        self.last_explored_pose = None
        self.last_motion_log_time = -1.0e9
        self.multi_last_motion_log_times = {}
        self.log_console_message(self.simulation_start_summary(multi=False))
        self.canvas.set_mapped_obstacle_points(self.mapped_obstacle_points)
        self.push_hazard_snapshot()
        self.canvas.set_explored_area_polygons(self.explored_area_polygons)
        # A previous run's restore must not leak its seeded explored-area
        # coverage into this fresh one.
        self.canvas.clear_explored_area_seed()
        self.record_explored_area(force=True)
        self.update_sensed_obstacles(force_status=False)
        self.force_robot_pose_free_in_belief(None)
        self.assign_route_to_robot()

        self.running = True
        self.paused = False
        self.canvas.set_simulation_running_for_perf(True)
        self.set_configuration_locked(True)

        self.path_points = [(self.robot.x, self.robot.y)]
        self.last_control = np.array([[0.0], [0.0]], dtype=float)
        self.simulation_time = 0.0
        self.last_time = time.perf_counter()
        self.last_sensor_update_time = 0.0
        self.last_sensor_update_pose = None
        self._exhausted_idle_obstacle_count = None

        self.update_start_pause_button()
        self.speed_button.setText(f"Speed {self.simulation_speed:.2f}x")
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)

        self.canvas.set_robot(self.robot)
        self.canvas.set_path(self.path_points)
        self.canvas.set_known_obstacles(self.known_obstacles)
        self.canvas.set_last_control(self.last_control)
        if self.config.planner_type == "Direct":
            self.canvas.set_status("Simulation running with direct route.")

        self.log_robot_motion(
            self.robot,
            robot_index=None,
            control=self.last_control,
            target=self.active_target_xy(),
            force=True,
        )
        self.top_bar.set_status("running")

    def reset_simulation(self):
        self.robot = None
        self.ensure_runtime_robot_registry().reset()
        self.robot_agents = self.ensure_runtime_robot_registry().agents
        self.running = False
        self.paused = False
        self.canvas.set_simulation_running_for_perf(False)
        self.set_configuration_locked(False)

        self.collision_checker = CollisionChecker() if CollisionChecker is not None else None
        self.last_collision_report = None
        self.spatial_index.rebuild(self.config.obstacles)
        self.known_obstacles: list[tuple[float, float, float, float]] = []
        self.explored_area_polygons: list[list[tuple[float, float]]] = []
        self.reset_belief_map(robot_count=1)
        self.current_exploration_target: tuple[float, float] | None = None
        self.multi_exploration_targets = []
        self.multi_invalidated_exploration_targets = []
        self.last_exploration_replan_sim_time = -1.0e9
        self.last_exploration_gate_message_time = -1.0e9
        self.last_goal_selection_reason = "using final mission goal"
        self.route_request_count = 0
        self.route_result_count = 0
        self.reset_navigation_debug_run_state()
        self.sensor_update_count = 0
        self.mapping_update_count = 0
        self.safety_replan_count = 0
        self.exploration_replan_count = 0
        self.total_distance_traveled = 0.0
        self.last_explored_pose: tuple[float, float, float] | None = None
        self.last_sensor_update_time = 0.0
        self.last_sensor_update_pose = None
        self._exhausted_idle_obstacle_count = None
        self.last_motion_log_time = -1.0e9
        self.multi_last_motion_log_times = {}
        self.planning_in_progress = False
        self.route_request_id += 1
        self.active_planner_workers.clear()
        self._invalidate_all_prefetch_requests(reason="simulation reset/restore")

        self.path_points = []
        self.robots = []
        self.multi_path_points = []
        self.multi_planned_path_points = []
        self.multi_last_controls = []
        self.last_control = np.array([[0.0], [0.0]], dtype=float)
        self.simulation_time = 0.0
        self.last_time = time.perf_counter()

        self.update_start_pause_button()
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)

        self.canvas.set_robot(None)
        self.canvas.set_multi_robots([], [], [], exploration_targets=[])
        self.canvas.set_path([])
        self.canvas.set_planned_path([])
        self.canvas.set_exploration_target(None)
        self.canvas.set_known_obstacles(self.known_obstacles)
        self.canvas.set_mapped_obstacle_points(self.mapped_obstacle_points)
        self.push_hazard_snapshot()
        self.canvas.set_explored_area_polygons(self.explored_area_polygons)
        # A previous run's restore must not leak its seeded explored-area
        # coverage into this fresh one.
        self.canvas.clear_explored_area_seed()
        self.canvas.set_last_control(self.last_control)
        self.canvas.set_status("Reset complete. Press Start Simulation to run.")

        self.top_bar.set_status("ready")

    def toggle_pause(self):
        if not self.running:
            return

        self.paused = not self.paused

        if self.paused:
            self.canvas.set_status("Simulation paused.")
            self.top_bar.set_status("paused")
        else:
            self.last_time = time.perf_counter()
            self.canvas.set_status("Simulation running.")
            self.top_bar.set_status("running")
            self.resume_navigation_debug_live_view()

        self.update_start_pause_button()
        self.update_navigation_debug_step_buttons()

    def set_goal_from_canvas(self, gx: float, gy: float):
        self.goal_x_input.setValue(gx)
        self.goal_y_input.setValue(gy)
        self.config = self.read_config()

        self.sync_runtime_robot_agents()
        registry = self.ensure_runtime_robot_registry()
        registry.set_final_goal_for_all(self.final_goal_xy())

        # In exploration modes the mission goal remains visible as a reference,
        # but it must not overwrite frontier targets.
        if self.is_exploration_mode():
            self.canvas.set_status(
                "Final goal updated visually. Exploration mode is active, so robots keep following frontiers."
            )
            return

        # Goal seeking is the only mode where G is executable. Changing G must
        # immediately invalidate old routes and assign fresh routes.
        if self.robots:
            for robot_index, agent in enumerate(getattr(self, "robot_agents", []) or []):
                agent.invalidate_route(reason="manual goal changed in Goal seeking")
                self._invalidate_prefetch_request(robot_index, reason="manual goal changed in Goal seeking")
            for robot_index in range(len(self.robots)):
                self.assign_route_to_multi_robot(robot_index, reason="Shared final goal updated")
            self.canvas.set_status("Goal updated and routes reassigned for all robots.")
            return

        if self.robot is not None:
            agent = self.runtime_agent(None)
            if agent is not None:
                agent.invalidate_route(reason="manual goal changed in Goal seeking")
                self._invalidate_prefetch_request(0, reason="manual goal changed in Goal seeking")
            self.assign_route_to_robot()
            self.canvas.set_status("Goal updated by canvas click and route reassigned.")

    def _route_intersects_hazard_points(
        self,
        route_points: list[tuple[float, float]],
        hazard_points: tuple[tuple[float, float], ...],
        *,
        robot_radius: float,
    ) -> bool:
        """Return whether a current route crosses thresholded hazard cells."""
        if self.collision_checker is None or len(route_points) < 2 or not hazard_points:
            return False
        for start, end in zip(route_points[:-1], route_points[1:]):
            report = self.collision_checker.check_segment_points(
                start=start,
                end=end,
                obstacle_points=list(hazard_points),
                robot_radius=max(0.0, float(robot_radius)),
            )
            if report.collision:
                return True
        return False

    def _replan_routes_affected_by_hazard(self) -> None:
        """Replan only active routes that now cross a blocked, OBSERVED
        thermal cell -- never the omniscient ground-truth HazardField."""
        if not getattr(self, "running", False):
            return

        service = self.ensure_hazard_service()
        hazard_points = service.observed_blocked_world_points()
        if not hazard_points:
            return

        robots = list(getattr(self, "robots", []) or [])
        if robots and "Multiple" in str(getattr(self.config, "agent_mode", "")):
            old_robot = getattr(self, "robot", None)
            try:
                for robot_index, robot in enumerate(robots):
                    route_points = self.current_route_points_for_robot(robot)
                    if not self._route_intersects_hazard_points(
                        route_points,
                        hazard_points,
                        robot_radius=self.safety_radius_for_robot(robot),
                    ):
                        continue
                    self.robot = robot
                    if str(getattr(self.config, "planner_type", "")) == "Direct":
                        if hasattr(robot, "force_stop"):
                            robot.force_stop(reason="dynamic fire hazard blocks direct route")
                        agent = self.runtime_agent(robot_index)
                        if agent is not None:
                            agent.invalidate_route(
                                reason="dynamic fire hazard blocks direct route"
                            )
                        self._invalidate_prefetch_request(
                            robot_index, reason="dynamic fire hazard blocks direct route"
                        )
                        continue
                    self.assign_route_to_multi_robot(
                        robot_index,
                        reason="Dynamic fire hazard affects current route",
                        force_new_exploration_target=False,
                    )
            finally:
                self.robot = old_robot
            return

        if self.robot is None:
            return
        if self._route_intersects_hazard_points(
            self.current_route_points(),
            hazard_points,
            robot_radius=self.safety_radius_for_robot(self.robot),
        ):
            agent = self.runtime_agent(None)
            if agent is not None:
                agent.invalidate_pending_path(
                    reason="dynamic fire hazard affects current route"
                )
                self._invalidate_prefetch_request(0, reason="dynamic fire hazard affects current route")
            if str(getattr(self.config, "planner_type", "")) == "Direct":
                if hasattr(self.robot, "force_stop"):
                    self.robot.force_stop(reason="dynamic fire hazard blocks direct route")
                if agent is not None:
                    agent.invalidate_route(
                        reason="dynamic fire hazard blocks direct route"
                    )
                self.canvas.set_status(
                    "Fire blocks the direct route. Select A* or Dijkstra to route around it."
                )
                return
            self.replan_after_new_information(
                "Dynamic fire hazard affects current route."
            )

    def add_fire(self, x: float, y: float) -> bool:
        """Add a temporary globally-known fire without changing occupancy."""
        service = self.ensure_hazard_service()
        try:
            change = service.add_fire((float(x), float(y)))
        except ValueError as exc:
            self.canvas.set_status(str(exc))
            return False

        self.push_hazard_snapshot()
        source = change.source
        self.canvas.set_status(
            f"Fire placed at ({source.position[0]:.2f}, {source.position[1]:.2f}); "
            f"radius={source.radius:.2f}m."
        )
        # No replan here: creating a FireSource only changes ground truth.
        # Route repair is driven by _replan_routes_affected_by_hazard() being
        # called after a sensor update actually OBSERVES a newly-blocked
        # cell (see update_explored_free_points_from_polygon()) -- never by
        # the omniscient act of a fire existing somewhere on the map.
        return True

    def remove_fire_near(self, x: float, y: float) -> bool:
        """Remove the nearest fire source without freeing occupancy cells."""
        service = self.ensure_hazard_service()
        change = service.remove_fire_near((float(x), float(y)))
        if not change.changed or change.source is None:
            return False

        self.push_hazard_snapshot()
        source = change.source
        self.canvas.set_status(
            f"Fire removed at ({source.position[0]:.2f}, {source.position[1]:.2f})."
        )
        return True

    def on_fire_toggle_requested(self, x: float, y: float) -> None:
        """Exploration click: remove a nearby source or create a new one."""
        service = self.ensure_hazard_service()
        if not service.field.in_bounds_world((float(x), float(y))):
            self.canvas.set_status("Fire ignored: click is outside the map bounds.")
            return

        change = service.toggle_fire_at((float(x), float(y)))
        self.push_hazard_snapshot()
        if change.action == "removed" and change.source is not None:
            source = change.source
            self.canvas.set_status(
                f"Fire removed at ({source.position[0]:.2f}, {source.position[1]:.2f})."
            )
            return
        if change.action == "added" and change.source is not None:
            source = change.source
            self.canvas.set_status(
                f"Fire placed at ({source.position[0]:.2f}, {source.position[1]:.2f}); "
                f"radius={source.radius:.2f}m."
            )
            # No replan here either -- see add_fire()'s comment.

    def body_radius_for_robot(self, robot=None) -> float:
        """Return physical body radius for a runtime robot or the global config."""
        target_robot = self.robot if robot is None else robot
        if target_robot is not None:
            if hasattr(target_robot, "_sim_body_radius"):
                return float(target_robot._sim_body_radius)
            limits = getattr(target_robot, "limits", None)
            if limits is not None and hasattr(limits, "robot_radius"):
                return float(limits.robot_radius)
            if hasattr(target_robot, "robot_radius"):
                return float(target_robot.robot_radius)
        return float(self.config.body_radius)

    def safety_radius_for_robot(self, robot=None) -> float:
        """Return clearance radius r for a runtime robot or the global config."""
        target_robot = self.robot if robot is None else robot
        body = self.body_radius_for_robot(target_robot)
        if target_robot is not None and hasattr(target_robot, "_sim_safety_radius"):
            return effective_planning_clearance(body, float(target_robot._sim_safety_radius))
        return effective_planning_clearance(body, float(self.config.safety_radius))

    def body_radius(self) -> float:
        """Backward-compatible alias for the current robot body radius."""
        return self.body_radius_for_robot(self.robot)

    def safety_radius(self) -> float:
        """Backward-compatible alias for the current robot clearance radius."""
        return self.safety_radius_for_robot(self.robot)

    def robot_radius(self) -> float:
        """
        Backward-compatible alias for the safety radius used by old calls.
        """
        return self.safety_radius()

    def apply_controller_parameters(self, robot=None, acceleration_gain: float | None = None) -> None:
        """
        Push GUI/per-robot controller parameters into the robot when the
        implementation exposes a modular TrackingController.
        """
        target_robot = self.robot if robot is None else robot
        if target_robot is None:
            return
        controller = getattr(target_robot, "controller", None)
        gain = float(self.config.acceleration_gain if acceleration_gain is None else acceleration_gain)
        if controller is not None and hasattr(controller, "acceleration_gain"):
            controller.acceleration_gain = gain

    def active_target_xy(self) -> tuple[float, float] | None:
        """
        Return the local target the robot is currently trying to reach.

        For the modular robot, this is active_waypoint(). For older robot
        versions, this falls back to robot.goal.
        """
        if self.robot is None:
            return None

        if hasattr(self.robot, "active_waypoint"):
            target = self.robot.active_waypoint()
            if target is not None:
                target_array = np.asarray(target, dtype=float).reshape(-1)
                return float(target_array[0]), float(target_array[1])

        goal = getattr(self.robot, "goal", None)
        if goal is not None:
            goal_array = np.asarray(goal, dtype=float).reshape(-1)
            if goal_array.size >= 2:
                return float(goal_array[0]), float(goal_array[1])

        return self.config.goal_x, self.config.goal_y

    def robot_snapshot(self):
        """
        Create a minimal dynamic snapshot for short-horizon collision prediction.
        """
        if self.robot is None or RobotSnapshot is None:
            return None

        return RobotSnapshot(
            x=float(self.robot.x),
            y=float(self.robot.y),
            theta=float(self.robot.theta),
            v=float(self.robot.v),
            max_speed=float(getattr(self.robot, "max_speed", self.config.max_speed)),
            max_acceleration=float(
                getattr(self.robot, "max_acceleration", self.config.max_acceleration)
            ),
            max_angular_speed=float(
                getattr(self.robot, "max_angular_speed", self.config.max_angular_speed)
            ),
        )

    def brake_control_for_collision(self) -> np.ndarray:
        """
        Return a braking control compatible with the robot interface.
        """
        if self.robot is not None and hasattr(self.robot, "brake_control"):
            return self.robot.brake_control()

        max_acceleration = float(
            getattr(self.robot, "max_acceleration", self.config.max_acceleration)
        )
        return np.array([[-max_acceleration], [0.0]], dtype=float)

    def stop_for_collision(self, message: str) -> None:
        """
        Stop the simulation after detecting an unsafe condition.

        The robot state is preserved so the canvas shows where the safety logic
        intervened.
        """
        self.running = False
        self.paused = False
        self.canvas.set_simulation_running_for_perf(False)
        self.last_control = self.brake_control_for_collision()
        self.canvas.set_last_control(self.last_control)
        self.canvas.set_status(message)
        self.top_bar.set_status("paused")
        self.update_start_pause_button()

    def nominal_control_safe(self, blocked: bool = False, capture=None) -> np.ndarray:
        """
        Call the robot nominal controller while supporting old and new APIs.

        capture: optional NavigationDebugCapture, forwarded down to
        TrackingController.compute_control() so it can stash heading_error/
        distance_to_goal. Falls back to the capture-less/blocked-less call
        shapes for older Robot implementations that do not accept these
        kwargs yet.
        """
        try:
            return self.robot.nominal_control(blocked=blocked, capture=capture)
        except TypeError:
            pass
        try:
            return self.robot.nominal_control(blocked=blocked)
        except TypeError:
            return self.robot.nominal_control()

    @staticmethod
    def distance_point_to_rect(point, obstacle) -> float:
        px, py = point
        ox, oy, ow, oh = obstacle
        closest_x = clamp(px, ox, ox + ow)
        closest_y = clamp(py, oy, oy + oh)
        return math.hypot(px - closest_x, py - closest_y)

    @staticmethod
    def sample_obstacle_boundary_points(
        obstacle: tuple[float, float, float, float],
        spacing: float,
    ) -> list[tuple[float, float]]:
        """
        Approximate a rectangular obstacle boundary with sparse points.

        The robot does not reveal the full rectangle when it senses it. It only
        adds visible boundary samples to its internal map, which creates a more
        realistic incremental mapping effect.
        """
        ox, oy, ow, oh = obstacle
        spacing = max(float(spacing), 0.015)
        points: list[tuple[float, float]] = []

        nx = max(1, int(math.ceil(ow / spacing)))
        ny = max(1, int(math.ceil(oh / spacing)))

        for i in range(nx + 1):
            x = ox + ow * i / nx
            points.append((x, oy))
            points.append((x, oy + oh))

        for j in range(1, ny):
            y = oy + oh * j / ny
            points.append((ox, y))
            points.append((ox + ow, y))

        return points

    @staticmethod
    def quantize_map_point(point: tuple[float, float], resolution: float) -> tuple[float, float]:
        """
        Quantize mapped points to avoid storing hundreds of near-duplicates.
        """
        # Keep points on the actual sampled boundary. Coarse grid quantization
        # made some mapped points look shifted relative to the rectangles.
        return (round(float(point[0]), 3), round(float(point[1]), 3))

    def visible_candidate_obstacles(self) -> list[tuple[float, float, float, float]]:
        """
        Return only obstacles that can affect the current sensor footprint.
        """
        if self.robot is None:
            return list(self.config.obstacles)

        return self.spatial_index.query_circle(
            origin=(float(self.robot.x), float(self.robot.y)),
            radius=float(getattr(self.robot, "vision", self.config.vision)),
            padding=max(self.safety_radius(), self.config.mapping_point_spacing),
        )

    def should_run_sensor_update(self, now: float) -> bool:
        """
        Throttle expensive sensor/mapping work.

        Robot dynamics still runs at the GUI timer rate. Sensor mapping runs
        around 10 Hz or sooner if the robot moves enough to reveal new geometry.
        """
        if self.robot is None:
            return False

        pose = (float(self.robot.x), float(self.robot.y), float(self.robot.theta))
        if self.last_sensor_update_pose is None:
            self.last_sensor_update_pose = pose
            self.last_sensor_update_time = float(now)
            return True

        last_x, last_y, last_theta = self.last_sensor_update_pose
        moved = math.hypot(pose[0] - last_x, pose[1] - last_y)
        rotated = abs(wrapped_angle_error(pose[2], last_theta))
        elapsed = float(now) - float(self.last_sensor_update_time)

        if (
            elapsed >= SENSOR_UPDATE_PERIOD_SEC
            or moved >= MIN_SENSOR_UPDATE_DISTANCE
            or rotated >= MIN_SENSOR_UPDATE_ROTATION
        ):
            self.last_sensor_update_pose = pose
            self.last_sensor_update_time = float(now)
            return True

        return False

    def point_visible_from_robot(
        self,
        point: tuple[float, float],
        candidate_obstacles: list[tuple[float, float, float, float]] | None = None,
    ) -> bool:
        """
        Return whether a boundary point is visible from the robot sensor.

        Visibility has three conditions:
            1. the point is inside sensor range;
            2. the point is inside the sensor angular model;
            3. no closer obstacle boundary occludes it.

        This prevents the map from being painted behind an obstacle. To map a
        full object, the robot must observe it from multiple sides.
        """
        if self.robot is None:
            return False

        rx = float(self.robot.x)
        ry = float(self.robot.y)
        px, py = point
        sensor_range = float(getattr(self.robot, "vision", self.config.vision))

        dx = float(px) - rx
        dy = float(py) - ry
        point_distance = math.hypot(dx, dy)

        if point_distance > sensor_range:
            return False

        if point_distance <= 1e-9:
            return False

        point_angle = math.atan2(dy, dx)

        if not angle_is_inside_sensor_model(
            angle=point_angle,
            robot_theta=float(self.robot.theta),
            vision_model=self.config.vision_model,
        ):
            return False

        first_hit = first_ray_hit_distance(
            origin=(rx, ry),
            angle=point_angle,
            obstacles=candidate_obstacles if candidate_obstacles is not None else self.visible_candidate_obstacles(),
            max_range=sensor_range,
        )

        # A boundary point is visible if it lies on the first surface hit by the
        # ray. If another obstacle is closer, the point is occluded.
        return point_distance <= first_hit + max(0.018, self.config.mapping_point_spacing * 0.70)

    @staticmethod
    def point_inside_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        """Return True when a world point is inside a polygon."""
        x, y = point
        inside = False
        n = len(polygon)
        if n < 3:
            return False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-12 else 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i

        return inside

    def update_explored_free_points_from_polygon(
        self,
        polygon: list[tuple[float, float]],
        robot_index: int | None = None,
    ) -> None:
        """Rasterize one sensor footprint into the authoritative belief.

        ``robot_index=None`` is the canvas convention for the single-robot
        homogeneous blue layer.  It is *not* a valid ownership value for
        ``BeliefMap.explored_by_robot``: passing None there updates occupancy
        but leaves the per-robot explored mask empty.  Snapshots restore that
        mask, so the visual explored area would disappear after a restore even
        though ``belief.grid`` still contained the mapped FREE/OCCUPIED cells.

        Keep the two concerns separate: single-robot rendering still uses
        ``robot_index=None`` at the canvas call site, while the belief records
        those observations under robot 0.
        """
        belief = self.ensure_belief_map()
        belief_robot_index = robot_index
        if belief_robot_index is None and int(getattr(belief, "robot_count", 1)) == 1:
            belief_robot_index = 0

        belief.mark_visible_polygon(
            polygon,
            robot_index=belief_robot_index,
            time_s=float(getattr(self, "simulation_time", 0.0)),
        )
        self.explored_free_points = belief.explored_points()

        # Same sensor update, same polygon: fuse ground-truth hazard into
        # the team HazardBelief for exactly the cells this FoV covers. Never
        # pass robot_index=None here -- HazardBelief requires a concrete
        # attribution index (see belief_robot_index's own None->0 mapping
        # above for single-robot).
        hazard_service = getattr(self, "hazard_service", None)
        if hazard_service is not None and belief_robot_index is not None:
            observation = hazard_service.observe_visible_polygon(polygon, robot_index=belief_robot_index)
            # Route repair is gated on a real threshold CROSSING newly
            # discovered by this observation -- never on `changed` alone,
            # which is also True for e.g. a new robot merely attributing an
            # already-known, already-blocked cell, or a newly observed but
            # safe cell. Creating/removing a FireSource never reaches here
            # at all (see add_fire()/on_fire_toggle_requested()).
            if observation.newly_blocked_cells > 0:
                self._replan_routes_affected_by_hazard()
            # Mark the render dirty only on an actual VISUAL change -- new
            # cells becoming observed, or an already-observed cell's value
            # changing. Deliberately NOT observation.changed: that is also
            # True for a newly_attributed_cells-only change (another robot
            # attributing an already-known, already-blocked cell), which
            # never alters what the heatmap looks like. The actual push is
            # collapsed to at most one per simulation step by
            # _flush_discovered_hazard_render(), called once after every
            # robot due this tick has run its sensor update -- never here,
            # which runs once per robot.
            visual_changed = (
                observation.newly_observed_cells > 0
                or observation.changed_value_cells > 0
            )
            if visual_changed:
                self._discovered_hazard_render_dirty = True

    def record_explored_area(self, force: bool = False, robot_index: int | None = None) -> None:
        """
        Store the current occlusion-aware sensor footprint as explored area.

        The trace is a visualization of coverage, independent from Robot Orders.
        To keep the GUI responsive, a new polygon is recorded only after the
        robot moves or rotates enough to add visible information.
        """
        if self.robot is None:
            return

        x = float(self.robot.x)
        y = float(self.robot.y)
        theta = float(self.robot.theta)
        vision = float(self.robot.vision)

        last_pose = self.last_explored_pose if robot_index is None else self.multi_last_explored_poses.get(int(robot_index))
        if last_pose is not None and not force:
            last_x, last_y, last_theta = last_pose
            moved = math.hypot(x - last_x, y - last_y)
            rotated = abs(wrapped_angle_error(theta, last_theta))

            # For omnidirectional/LiDAR sensors, orientation does not change the
            # visible footprint much. For camera/FoV, rotation matters.
            min_move = max(0.12, min(0.35, vision * 0.06))
            min_turn = 0.18 if "Camera" in self.config.vision_model else math.inf

            if moved < min_move and rotated < min_turn:
                return

        polygon = sensor_visible_polygon_world(
            origin=(x, y),
            theta=theta,
            vision=vision,
            vision_model=self.config.vision_model,
            obstacles=self.visible_candidate_obstacles(),
            ray_count=EXPLORED_RAYS_CAMERA if "Camera" in self.config.vision_model else EXPLORED_RAYS_OMNI,
        )

        if len(polygon) < 3:
            return

        # Update the geometric explored map used by frontier planners. This is
        # independent from the optimized visual pixmap cache.
        self.update_explored_free_points_from_polygon(polygon, robot_index=robot_index)

        # The canvas stores the accumulated explored area in a pixmap. Keep
        # only a short Python-side history to avoid growing copy costs over
        # long runs. The visual explored layer remains complete.
        self.explored_area_polygons.append(polygon)
        if len(self.explored_area_polygons) > EXPLORED_POLYGON_HISTORY_LIMIT:
            self.explored_area_polygons = self.explored_area_polygons[-EXPLORED_POLYGON_HISTORY_LIMIT:]

        if robot_index is None:
            self.last_explored_pose = (x, y, theta)
            self.canvas.append_explored_area_polygon(polygon, robot_index=None)
        else:
            self.multi_last_explored_poses[int(robot_index)] = (x, y, theta)
            self.canvas.append_explored_area_polygon(polygon, robot_index=int(robot_index))

    def update_sensed_obstacles(self, force_status: bool = True) -> list[tuple[float, float]]:
        """Update the partial obstacle map with visible boundary samples.

        Two map representations are updated together but kept separate:

        1. ``mapped_obstacle_points`` stores dense boundary samples. These points
           are used for the pink/red visual trace and for local known-obstacle
           safety checks. They should remain dense enough to look like an edge.

        2. ``belief_map.grid`` stores OCCUPIED cells. This is used for logical
           coverage/frontier metrics. It may be coarse, so it must not replace
           the boundary samples.
        """
        newly_mapped: list[tuple[float, float]] = []
        belief = self.ensure_belief_map()

        if not hasattr(self, "mapped_obstacle_points"):
            self.mapped_obstacle_points = []
        if not hasattr(self, "mapped_obstacle_point_keys"):
            self.mapped_obstacle_point_keys = {
                (round(float(p[0]), 3), round(float(p[1]), 3))
                for p in self.mapped_obstacle_points
            }

        spacing = max(float(self.config.mapping_point_spacing), 0.015)
        quantization = spacing
        # Diagnosis-only: this call runs before obstacle_extract's own timer
        # starts, and update_sensed_obstacles() has no wrapping timer of its
        # own, so this is a genuine TOP-LEVEL gap (not nested inside any
        # already-measured section) -- previously folded into
        # unaccounted_ms. Reported under its own top-level section (see
        # perf_monitor.py's _UNACCOUNTED_SECTIONS) -- gated by
        # should_run_sensor_update() like the rest of this method, so it is
        # intermittent (not every tick) unlike planner_services_refresh_ms,
        # but per_tick_ms() already accounts for that correctly.
        _visible_candidate_obstacles_start = time.perf_counter()
        candidate_obstacles = self.visible_candidate_obstacles()
        _record_perf(
            self, "visible_candidate_obstacles", time.perf_counter() - _visible_candidate_obstacles_start
        )

        _obstacle_extract_perf_start = time.perf_counter()
        for obstacle in candidate_obstacles:
            for point in self.sample_obstacle_boundary_points(tuple(obstacle), spacing):
                if not self.point_visible_from_robot(point, candidate_obstacles):
                    continue

                # Keep the sampled boundary location, only rounded to a stable
                # key. Do not collapse it to the belief cell center; doing so
                # destroys the visible line and weakens route safety checks.
                mapped_point = self.quantize_map_point(point, quantization)
                key = (round(float(mapped_point[0]), 3), round(float(mapped_point[1]), 3))
                if key in self.mapped_obstacle_point_keys:
                    continue

                self.mapped_obstacle_point_keys.add(key)
                self.mapped_obstacle_points.append(mapped_point)
                newly_mapped.append(mapped_point)
        _record_perf(self, "obstacle_extract", time.perf_counter() - _obstacle_extract_perf_start)

        if newly_mapped:
            _belief_update_perf_start = time.perf_counter()
            changed_cells = belief.mark_occupied_points(
                newly_mapped,
                time_s=float(getattr(self, "simulation_time", 0.0)),
            )
            # A live robot center is always traversable for its own next plan.
            # This does not erase dense obstacle-boundary samples; it only fixes
            # the exact start cell in the logical grid.
            self.force_all_robot_poses_free_in_belief()
            self.sync_legacy_map_views_from_belief()
            _record_perf(self, "belief_update", time.perf_counter() - _belief_update_perf_start)
            self.canvas.append_mapped_obstacle_points(newly_mapped)
            if force_status:
                self.canvas.set_status(
                    f"Mapped {len(newly_mapped)} obstacle boundary sample(s); "
                    f"{changed_cells} occupied belief cell(s)."
                )

        return newly_mapped

    def current_route_points(self) -> list[tuple[float, float]]:
        """
        Return the remaining route currently assigned to the robot.

        The first point is always the robot's current position. The rest are the
        active waypoint and the future waypoints, when the modular robot exposes
        a WaypointManager. This route is used only to decide whether newly mapped
        obstacle points actually affect the current plan.
        """
        if self.robot is None:
            return []

        points: list[tuple[float, float]] = [(float(self.robot.x), float(self.robot.y))]

        waypoint_manager = getattr(self.robot, "waypoints", None)
        raw_waypoints = getattr(waypoint_manager, "waypoints", None)
        current_index = getattr(waypoint_manager, "current_index", None)

        if raw_waypoints is not None and current_index is not None:
            for waypoint in raw_waypoints[int(current_index):]:
                waypoint_array = np.asarray(waypoint, dtype=float).reshape(-1)
                if waypoint_array.size >= 2:
                    points.append((float(waypoint_array[0]), float(waypoint_array[1])))
        else:
            target = self.active_target_xy()
            if target is not None:
                points.append(target)

        # Remove near-duplicate consecutive points. They create zero-length
        # route segments that can look like false safety interventions.
        cleaned: list[tuple[float, float]] = []
        for point in points:
            if not cleaned or math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-6:
                cleaned.append(point)

        return cleaned

    def route_intersects_mapped_points(
        self,
        route_points: list[tuple[float, float]],
        mapped_points: list[tuple[float, float]],
    ) -> bool:
        """
        Return whether mapped obstacle points invalidate the current route.

        A newly sensed point should not trigger replanning just because it exists.
        It should trigger replanning only if it violates the safety radius around
        the current route segments.
        """
        if self.collision_checker is None:
            return False

        if len(route_points) < 2 or not mapped_points:
            return False

        robot_radius = self.safety_radius()

        for start, end in zip(route_points[:-1], route_points[1:]):
            report = self.collision_checker.check_segment_points(
                start=start,
                end=end,
                obstacle_points=mapped_points,
                robot_radius=robot_radius,
            )
            if report.collision:
                return True

        return False

    def new_information_affects_current_route(
        self,
        newly_mapped: list[tuple[float, float]],
    ) -> bool:
        """
        Decide whether new sensor information requires replanning.

        Mapping and replanning are intentionally separated:
            - mapping updates the partial map whenever the sensor sees something;
            - replanning happens only when the new information threatens the
              route that the robot is currently executing.

        This prevents irrelevant discoveries, such as a wall behind or beside the
        robot, from changing a perfectly valid route.
        """
        route_points = self.current_route_points()
        return self.route_intersects_mapped_points(route_points, newly_mapped)

    def exploration_replan_allowed(self) -> tuple[bool, float]:
        """
        Gate frontier-target replans so exploration does not constantly destroy
        an aggressive path simplification result.

        This cooldown applies only to exploration target changes. Safety replans
        caused by a newly mapped obstacle or a predicted collision bypass this
        gate.
        """
        cooldown = max(0.0, float(self.config.exploration_replan_cooldown))
        elapsed = float(self.simulation_time) - float(self.last_exploration_replan_sim_time)
        remaining = max(0.0, cooldown - elapsed)
        return remaining <= 1e-9, remaining

    def request_exploration_route_async(self, reason: str) -> bool:
        """
        Request a new frontier target only when the exploration cooldown allows it.
        """
        allowed, remaining = self.exploration_replan_allowed()
        if not allowed:
            # Avoid spamming the status text every frame while the robot waits
            # at a reached local frontier target.
            if float(self.simulation_time) - float(self.last_exploration_gate_message_time) >= 0.50:
                self.canvas.set_status(
                    f"{reason} Waiting {remaining:.2f}s before next exploration replan."
                )
                self.last_exploration_gate_message_time = float(self.simulation_time)
            return False

        requested = self.request_route_async(reason)
        if requested:
            self.exploration_replan_count += 1
            self.last_exploration_replan_sim_time = float(self.simulation_time)
        return requested

    def replan_after_new_information(self, reason: str) -> bool:
        """
        Recompute the route using the robot's current partial map.

        The robot should not stop permanently when a local segment is blocked. It
        should update its map and ask the selected planner for a new route from
        its current state.

        This is a REPAIR replan (route_affected / REPLAN_FOR_SAFETY), not a
        fresh target selection: it must preserve the goal the robot was
        already navigating to (current_route_repair_goal()) via
        target_override, the same mechanism REQUEST_PLAN decisions already
        use. Without this, request_route_async() would fall back to
        select_navigation_goal() -- an independent frontier re-selection
        that has no idea a route repair, not a new destination, was asked
        for -- and could switch the robot to a completely different
        frontier just because a newly-mapped obstacle grazed its current
        route. Only when there is nothing active to repair
        (current_route_repair_goal() returns None) does this fall back to
        that normal target-selection behavior, unchanged.
        """
        if self.robot is None:
            return False

        if self.config.planner_type == "Direct":
            return False

        self.safety_replan_count += 1
        repair_goal = current_route_repair_goal(self.runtime_agent(None))
        return self.request_route_async(
            f"{reason} Replanning with {len(self.mapped_obstacle_points)} mapped boundary sample(s).",
            target_override=repair_goal,
        )

    def inter_robot_clearance_violation(self) -> tuple[bool, str]:
        """
        Check pairwise robot-robot safety clearance.

        Each robot is modeled as a disk with its own safety radius r. A violation
        occurs when the distance between centers is smaller than r_i + r_j. This
        is the first multi-robot safety layer; later we can replace the hard stop
        with CBF-based avoidance.
        """
        if len(self.robots) < 2:
            return False, ""

        for i in range(len(self.robots)):
            ri = self.robots[i]
            xi, yi = float(ri.x), float(ri.y)
            radius_i = self.safety_radius_for_robot(ri)
            for j in range(i + 1, len(self.robots)):
                rj = self.robots[j]
                xj, yj = float(rj.x), float(rj.y)
                radius_j = self.safety_radius_for_robot(rj)
                distance = math.hypot(xi - xj, yi - yj)
                minimum_distance = radius_i + radius_j
                if distance <= minimum_distance:
                    return (
                        True,
                        f"ROBOT-ROBOT COLLISION: R{i + 1} and R{j + 1} are too close "
                        f"({distance:.2f} m < {minimum_distance:.2f} m).",
                    )

        return False, ""

    def predicted_motion_report(
        self,
        *,
        control: np.ndarray,
        dt: float,
        robot_radius: float,
        known_obstacle_points: list[tuple[float, float]] | None = None,
        use_ground_truth: bool = True,
        capture=None,
    ):
        """Check short-horizon motion before applying a control.

        Known mapped-obstacle points are checked first when the installed
        CollisionChecker supports point-cloud prediction. Ground-truth rectangles
        are also checked as a simulator integrity guard: the simulator should
        stop before a collision, not after the robot has already entered an
        obstacle safety region.

        capture: optional NavigationDebugCapture. When provided, stashes the
        predicted trajectory (predict_unicycle_points() already computes
        this internally per check call below, but never returns it -- an
        extra cheap 10-step kinematic rollout when capture is requested is
        the simplest way to surface it without changing the checker's
        public return type) and the ClearanceTerms for whichever check
        found a collision, if any. None (the default) costs nothing extra.
        """
        if self.collision_checker is None:
            return None
        snapshot = self.robot_snapshot()
        if snapshot is None:
            return None

        safe_dt = max(float(dt), 1e-3)
        steps = 10

        if capture is not None and hasattr(self.collision_checker, "predict_unicycle_points"):
            capture.predicted_trajectory = tuple(
                self.collision_checker.predict_unicycle_points(snapshot, control, safe_dt, steps)
            )

        if known_obstacle_points and hasattr(self.collision_checker, "check_predicted_motion_points"):
            report = self.collision_checker.check_predicted_motion_points(
                snapshot=snapshot,
                control=control,
                dt=safe_dt,
                steps=steps,
                obstacle_points=known_obstacle_points,
                robot_radius=robot_radius,
            )
            # Captured on BOTH outcomes (not only collision=True) -- a clear
            # result is real, informative data ("checked, nothing found"),
            # not the same as "this check never ran". Whichever check runs
            # last below overwrites this with its own outcome, matching
            # which check actually gated the return value.
            if capture is not None:
                capture.predicted_collision = clearance_terms_from_report(
                    report, checker="check_predicted_motion_points", required_clearance=robot_radius
                )
            if getattr(report, "collision", False):
                return report

        if use_ground_truth and hasattr(self.collision_checker, "check_predicted_motion"):
            report = self.collision_checker.check_predicted_motion(
                snapshot=snapshot,
                control=control,
                dt=safe_dt,
                steps=steps,
                obstacles=self.config.obstacles,
                robot_radius=robot_radius,
            )
            if capture is not None:
                capture.predicted_collision = clearance_terms_from_report(
                    report, checker="check_predicted_motion", required_clearance=robot_radius
                )
            if getattr(report, "collision", False):
                return report

        return None

    def simulation_step_multi(self, real_dt: float) -> None:
        if not self.running or self.paused or not self.robots:
            return

        dt = min(real_dt, 0.05) * float(self.simulation_speed)
        self.simulation_time += dt

        if self.collision_checker is None:
            self.canvas.set_status("Collision checker unavailable.")
            return

        violation, message = self.inter_robot_clearance_violation()
        if violation:
            self.stop_for_collision(message)
            return

        run_sensor_update = self.should_run_sensor_update(time.perf_counter())
        if run_sensor_update:
            self.sensor_update_count += 1
            old_robot = self.robot
            newly_discovered_all: list[tuple[float, float]] = []

            for robot_index, robot in enumerate(self.robots):
                self.robot = robot
                self.record_explored_area(force=True, robot_index=robot_index)
                newly = self.update_sensed_obstacles(force_status=False)
                newly_discovered_all.extend(newly)

            self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)

            # All robots due this tick have run their sensor update above --
            # collapse however many of them marked the discovered-hazard
            # render dirty into at most one push.
            self._flush_discovered_hazard_render()

            if newly_discovered_all:
                self.mapping_update_count += 1
                replanned = self.replan_multi_robots_affected_by_points(
                    newly_discovered_all,
                    reason="New mapped obstacle affects robot route",
                )
                if replanned:
                    self.canvas.set_status(
                        f"Multi-robot mapping: {len(newly_discovered_all)} new boundary sample(s). "
                        f"Replanned {replanned} robot route(s)."
                    )
                else:
                    self.canvas.set_status(
                        f"Multi-robot mapping: {len(newly_discovered_all)} new obstacle boundary sample(s)."
                    )

        new_controls: list[np.ndarray] = []

        for index, robot in enumerate(self.robots):
            robot_position = (float(robot.x), float(robot.y))
            robot_radius = self.safety_radius_for_robot(robot)

            current_collision = self.collision_checker.check_position(
                position=robot_position,
                obstacles=self.config.obstacles,
                robot_radius=robot_radius,
            )
            if current_collision.collision:
                self.last_collision_report = current_collision
                self.robot = robot
                self.stop_for_collision(f"COLLISION: robot {index + 1} is inside an obstacle safety region.")
                return

            self.robot = robot
            target = self.active_target_xy()
            if target is not None:
                dynamic_points = self.dynamic_robot_obstacle_points_for_robot(index)
                active_segment_report = self.collision_checker.check_segment_points(
                    start=robot_position,
                    end=target,
                    obstacle_points=list(self.mapped_obstacle_points) + dynamic_points,
                    robot_radius=robot_radius,
                )
                robot_obstacle_violation, robot_obstacle_message = self.segment_violates_other_robot_clearance(
                    index,
                    robot_position,
                    target,
                )
                if active_segment_report.collision or robot_obstacle_violation:
                    block_reason = robot_obstacle_message if robot_obstacle_violation else "Active segment blocked by known obstacle"
                    self.set_multi_route_state(index, self.ROUTE_STATE_STUCK_SAFETY, block_reason)

                    # If the currently assigned frontier produces an unsafe first
                    # segment, do not keep re-planning to that same frontier.
                    # Blacklist it for this robot and request a different target.
                    if self.is_exploration_mode():
                        self.invalidate_current_multi_frontier(index, block_reason)

                    # Re-target regardless of planner_type: assign_route_to_multi_robot
                    # already resolves Direct/A*/Dijkstra/plugin-owned paths
                    # uniformly (see compute_route_for_multi_robot) and falls back
                    # to HOLD_ROUTE_BLOCKED/WAITING_FOR_CORRIDOR on its own when no
                    # safe route exists -- gating this on planner_type == "Direct"
                    # used to skip straight to stop_for_collision() (halting the
                    # WHOLE simulation) on the very first blocked segment whenever
                    # Direct was selected, since Direct is the default planner.
                    if self.multi_safety_replan_allowed(index, block_reason, target):
                        if self.assign_route_to_multi_robot(
                            index,
                            reason=block_reason,
                            force_new_exploration_target=True,
                        ):
                            control = self.brake_control_for_collision()
                            new_controls.append(control)
                            continue

                    # During the cooldown, stay stopped instead of logging the
                    # same rejected route every frame.
                    control = self.brake_control_for_collision()
                    new_controls.append(control)
                    continue

            # nominal_control_safe() also advances the robot's state machine
            # (active waypoint, ARRIVED/BLOCKED mode), so it always runs even
            # when a plugin owns CONTROL -- only the resulting control vector
            # may be replaced below. The safety veto further down (predicted
            # collision check) still runs on whatever control is used here, so
            # a CONTROL-owning plugin cannot bypass it.
            legacy_control = self.nominal_control_safe(blocked=False)
            control_profile = self.coordinator_runtime_profile()
            robot_command = getattr(self, "multi_robot_commands_by_id", {}).get(index)
            control, control_reason = select_runtime_control_source(
                control_profile, robot_command, legacy_control
            )
            if control_profile.owns_control:
                _LOGGER.debug("R%d control source: %s", index + 1, control_reason)
            control = np.asarray(control, dtype=float).reshape(np.asarray(legacy_control).shape)
            control = self.apply_hazard_safety_filter(robot, control)

            prediction_report = self.predicted_motion_report(
                control=control,
                dt=dt,
                robot_radius=robot_radius,
                known_obstacle_points=list(self.mapped_obstacle_points) + self.dynamic_robot_obstacle_points_for_robot(index),
                use_ground_truth=True,
            )
            if prediction_report is not None and getattr(prediction_report, "collision", False):
                self.last_collision_report = prediction_report
                block_reason = "Predicted collision before motion update"
                self.set_multi_route_state(index, self.ROUTE_STATE_STUCK_SAFETY, block_reason)
                if self.is_exploration_mode():
                    self.invalidate_current_multi_frontier(index, block_reason)

                # Re-target regardless of planner_type -- see the matching
                # comment on the active-segment block above for why gating
                # this on planner_type == "Direct" was wrong.
                if self.multi_safety_replan_allowed(index, block_reason, target):
                    if self.assign_route_to_multi_robot(
                        index,
                        reason=block_reason,
                        force_new_exploration_target=True,
                    ):
                        control = self.brake_control_for_collision()
                        new_controls.append(control)
                        continue

                control = self.brake_control_for_collision()
                new_controls.append(control)
                continue

            robot.update(control, dt)
            new_controls.append(control)
            self.log_robot_motion(
                robot,
                robot_index=index,
                control=control,
                target=target,
            )

            post_position = (float(robot.x), float(robot.y))
            post_collision = self.collision_checker.check_position(
                position=post_position,
                obstacles=self.config.obstacles,
                robot_radius=robot_radius,
            )
            if post_collision.collision:
                self.last_collision_report = post_collision
                self.stop_for_collision(f"COLLISION: robot {index + 1} entered an obstacle safety region after update.")
                return

            violation, message = self.inter_robot_clearance_violation()
            if violation:
                self.stop_for_collision(message)
                return

            # If frontier exploration is active, each robot can request a new
            # target after reaching its current one. This is intentionally simple
            # assignment for now; duplicate-frontier avoidance is the next layer.
            if self.is_exploration_mode():
                target = self.active_target_xy()
                tolerance = max(float(getattr(robot, "_sim_goal_tolerance", self.config.goal_tolerance)), 0.25)
                if target is not None and math.hypot(float(robot.x) - target[0], float(robot.y) - target[1]) <= tolerance:
                    if self.multi_exploration_target_replan_allowed(index):
                        if self.assign_route_to_multi_robot(
                            index,
                            reason="Exploration target reached",
                            force_new_exploration_target=True,
                        ):
                            self.exploration_replan_count += 1

            while len(self.multi_path_points) <= index:
                self.multi_path_points.append([])
            path = self.multi_path_points[index]
            new_path_point = (float(robot.x), float(robot.y))
            if path:
                self.total_distance_traveled += math.hypot(
                    new_path_point[0] - float(path[-1][0]),
                    new_path_point[1] - float(path[-1][1]),
                )
            path.append(new_path_point)
            if len(path) > 900:
                self.multi_path_points[index] = path[-900:]

        selected = max(0, min(int(self.selected_robot_index), len(self.robots) - 1))
        self.robot = self.robots[selected]
        self.path_points = self.multi_path_points[selected] if selected < len(self.multi_path_points) else []
        self.multi_last_controls = new_controls
        self.last_control = new_controls[selected] if selected < len(new_controls) else np.array([[0.0], [0.0]], dtype=float)

        self.canvas.set_multi_runtime_state(
            robots=self.robots,
            path_points=self.multi_path_points,
            planned_path_points=self.multi_planned_path_points,
            exploration_targets=self.multi_exploration_targets,
            last_controls=self.multi_last_controls,
            simulation_time=self.simulation_time,
            simulation_speed=self.simulation_speed,
        )

    def _append_executed_path_point(self, new_path_point: tuple[float, float]) -> None:
        """Append one point to the single-robot executed trail
        (self.path_points), trimming back to EXECUTED_TRAIL_MAX_POINTS
        only once EXECUTED_TRAIL_TRIM_MARGIN extra points have
        accumulated past the cap -- not on the very next tick after the
        cap is first reached.

        Trimming replaces self.path_points with a NEW list object (see
        SimulationCanvas.draw_executed_path()'s docstring: its executed-
        trail pixmap cache uses object identity to detect exactly this,
        forcing a full pixmap rebuild whenever it happens). One point is
        appended every tick, so a naive "> EXECUTED_TRAIL_MAX_POINTS: trim
        to EXECUTED_TRAIL_MAX_POINTS" would replace the list object on
        EVERY tick forever after the cap is first reached -- permanently
        defeating that cache (real Office.sim evidence:
        executed_trail_build_ms climbing to 5-10ms+ per frame once
        executed_trail_points hit 1200, with route_path_ms spikes over
        100ms). The margin means that identity change -- and the rebuild
        it forces -- happens once every EXECUTED_TRAIL_TRIM_MARGIN ticks
        instead of every tick, matching the "grew in place" case the
        pixmap cache is actually optimized for.
        """
        if self.path_points:
            self.total_distance_traveled += math.hypot(
                new_path_point[0] - float(self.path_points[-1][0]),
                new_path_point[1] - float(self.path_points[-1][1]),
            )
        self.path_points.append(new_path_point)

        if len(self.path_points) > EXECUTED_TRAIL_MAX_POINTS + EXECUTED_TRAIL_TRIM_MARGIN:
            self.path_points = self.path_points[-EXECUTED_TRAIL_MAX_POINTS:]

    def simulation_step(self):
        now = time.perf_counter()
        real_dt = now - self.last_time
        self.last_time = now
        real_dt = min(real_dt, 0.05)

        if self.running and self.robots:
            self.simulation_step_multi(real_dt)
            return

        if not self.running or self.paused or self.robot is None:
            return

        dt = real_dt * float(self.simulation_speed)
        self.simulation_time += dt

        if self.collision_checker is None:
            self.canvas.set_status("Collision checker unavailable.")
            return

        if self.planning_in_progress:
            # Keep the robot still while a new global route is being computed,
            # but do not block the GUI thread.
            self.last_control = self.brake_control_for_collision()
            self.canvas.set_runtime_state(
                robot=self.robot,
                path_points=self.path_points,
                last_control=self.last_control,
                simulation_time=self.simulation_time,
                simulation_speed=self.simulation_speed,
            )
            return

        # Exploration-exhausted HOLD: gates route_affected checks, forced
        # canvas repaints, and belief snapshots below to an occasional
        # ~1Hz trickle instead of every tick, while the agent has nothing
        # left to route to anyway. Computed once and reused for every gate
        # below so they all skip (or all run) together on the same tick --
        # see _should_skip_for_exhausted_hold()'s docstring.
        skip_for_exhausted_hold = self._should_skip_for_exhausted_hold()

        # Exhausted-idle fast path: while the agent is latched exhausted
        # with nothing left to route to, the robot is stopped, and no
        # planner/path work is in flight, the ENTIRE per-tick pipeline
        # below (sensor update, agent decision, motion integration,
        # telemetry) is a guaranteed no-op -- the robot's pose cannot
        # change without a control update, and its pose not changing
        # means a fresh sensor scan against the same (unchanged)
        # ground-truth obstacles cannot discover anything new either. See
        # _exhausted_idle_fast_path_ready()'s docstring for the exact
        # conditions and why each is necessary. Reuses the same
        # skip_for_exhausted_hold flag/throttle as the gates below so the
        # low-rate ~1Hz heartbeat and this fast path agree on the same
        # "due" tick.
        if self._exhausted_idle_fast_path_ready(self.runtime_agent(None)):
            if skip_for_exhausted_hold:
                self.exhausted_idle_fast_path_hits = getattr(self, "exhausted_idle_fast_path_hits", 0) + 1
                self.exhausted_idle_skipped_canvas_updates = (
                    getattr(self, "exhausted_idle_skipped_canvas_updates", 0) + 1
                )
                self.exhausted_idle_skipped_sensor_updates = (
                    getattr(self, "exhausted_idle_skipped_sensor_updates", 0) + 1
                )
                return
            # Heartbeat due -- fall through to the normal pipeline this
            # tick to refresh canvas/telemetry/sensor state, then resume
            # fast-pathing on the next tick.
            self.exhausted_idle_full_updates = getattr(self, "exhausted_idle_full_updates", 0) + 1

        run_sensor_update = self.should_run_sensor_update(now)
        if run_sensor_update:
            self.sensor_update_count += 1
            _explored_update_perf_start = time.perf_counter()
            self.record_explored_area(force=False)
            _record_perf(self, "explored_update", time.perf_counter() - _explored_update_perf_start)
            # The single robot due this tick has run its sensor update
            # above -- collapse whatever it marked dirty into at most one
            # push (see simulation_step_multi()'s equivalent call site).
            self._flush_discovered_hazard_render()
            # update_sensed_obstacles() times its own obstacle_extract/
            # belief_update sub-sections internally (see its body) --
            # finer-grained than the old combined "sensor_update" figure,
            # which this replaces.
            newly_discovered = self.update_sensed_obstacles(force_status=False)
            if newly_discovered:
                self.mapping_update_count += 1

            if self.robot is not None:
                # Opt-in terminal trace only (ROBOT_TRACE=map,obstacles,...);
                # both no-op immediately when their category is disabled, so
                # this costs nothing in the default (disabled) case beyond
                # two cheap attribute checks.
                _emit_robot_trace(
                    self,
                    "trace_map",
                    sim_time=float(self.simulation_time),
                    robot_label="R1",
                    pose=(float(self.robot.x), float(self.robot.y)),
                    explored_percent=self.estimated_explored_percent(),
                    mapped_obstacle_samples=len(self.mapped_obstacle_points),
                )
                _emit_robot_trace(
                    self,
                    "trace_obstacles",
                    sim_time=float(self.simulation_time),
                    robot_label="R1",
                    points=list(newly_discovered) if newly_discovered else [],
                    explored_percent=self.estimated_explored_percent(),
                )
                # Best-effort periodic belief-map snapshot file (see
                # belief_trace_writer.py): a no-op unless ROBOT_TRACE's file
                # sink is active, and the (grid-scanning) snapshot dict is
                # only ever built lazily, on the tick a write is actually
                # due -- never on every sensor update. Also skipped
                # entirely while exploration-exhausted-HOLD's own ~1Hz
                # throttle isn't due yet -- a stationary, exhausted robot
                # has nothing new for the snapshot to capture anyway.
                if not skip_for_exhausted_hold:
                    _belief_snapshot_perf_start = time.perf_counter()
                    _emit_robot_trace(
                        self,
                        "maybe_snapshot_belief",
                        sim_time=float(self.simulation_time),
                        provider=self._build_belief_trace_snapshot,
                    )
                    _record_perf(self, "belief_snapshot", time.perf_counter() - _belief_snapshot_perf_start)

            if newly_discovered and self.config.planner_type != "Direct" and not skip_for_exhausted_hold:
                route_affected = self._timed_route_affected_check(newly_discovered)
                _telemetry_map_update_perf_start = time.perf_counter()
                self.telemetry.report_map_update(
                    sim_time=float(self.simulation_time),
                    new_points=newly_discovered,
                    total_count=len(self.mapped_obstacle_points),
                    route_affected=route_affected,
                    explored_percent=self.estimated_explored_percent(),
                )
                _record_perf(self, "telemetry", time.perf_counter() - _telemetry_map_update_perf_start)
                if route_affected:
                    route_affected_agent = self.runtime_agent(None)
                    robot_xy_now = (float(self.robot.x), float(self.robot.y))

                    # Diagnostics only: bbox of the specific new points that
                    # triggered this route_affected=yes occurrence, shared by
                    # both outcomes (throttled/allowed) below so
                    # route_affected_events.csv always has it regardless of
                    # which branch is taken.
                    new_obstacle_xs = [float(p[0]) for p in newly_discovered]
                    new_obstacle_ys = [float(p[1]) for p in newly_discovered]
                    new_obstacle_bbox = (
                        min(new_obstacle_xs), min(new_obstacle_ys),
                        max(new_obstacle_xs), max(new_obstacle_ys),
                    )

                    # No active_segment_unsafe bypass here (an earlier
                    # version had one): a genuinely urgent, imminent
                    # collision is REPLAN_FOR_SAFETY's job, driven
                    # independently by active_segment_blocked/
                    # predicted_collision in RobotAgent.step() with its own
                    # separate throttle -- this guard's only job is to
                    # stop routine map-growth repairs from storming the
                    # planner, and must apply unconditionally to do that.
                    allowed = True
                    if route_affected_agent is not None:
                        allowed = route_affected_agent.route_affected_replan_allowed(
                            path_goal=route_affected_agent.active_path_goal_xy,
                            current_time=float(self.simulation_time),
                            cooldown=self.route_affected_replan_cooldown_seconds(),
                        )

                    if not allowed:
                        # Throttled: either a repair for this exact goal is
                        # already in flight, or the same path_goal repaired
                        # too recently -- routine boundary-sample growth
                        # near a narrow passage must not become a
                        # background full-replan storm. Diagnostic only,
                        # DEBUG-level (never spams normal/quiet consoles);
                        # naturally rate-limited to at most once per
                        # cooldown per path_goal by the throttle itself.
                        nearby_distances = [
                            math.hypot(robot_xy_now[0] - p[0], robot_xy_now[1] - p[1])
                            for p in self.mapped_obstacle_points
                        ]
                        min_clearance = min(nearby_distances) if nearby_distances else None
                        self.telemetry.debug(
                            format_narrow_passage_diagnostic(
                                path_goal=route_affected_agent.active_path_goal_xy if route_affected_agent else None,
                                route_affected_recent=route_affected_agent.route_affected_replan_count if route_affected_agent else 0,
                                first_segment_blocked=route_affected_agent.first_segment_blocked_count if route_affected_agent else 0,
                                predicted_collision=route_affected_agent.safety_replan_count if route_affected_agent else 0,
                                min_clearance=min_clearance,
                                action="slowdown",
                            )
                        )
                        # Opt-in terminal trace only (ROBOT_TRACE=safety);
                        # never printed/GUI-consoled unless explicitly enabled.
                        _emit_robot_trace(
                            self,
                            "trace_safety",
                            sim_time=float(self.simulation_time),
                            robot_label="R1",
                            goal=route_affected_agent.active_path_goal_xy if route_affected_agent else None,
                            repair_status="throttled",
                            min_clearance=min_clearance,
                        )
                        # Belief-trace artifact completeness (file-only,
                        # independent of ROBOT_TRACE): every route_affected=yes
                        # occurrence, throttled or not, must be recorded so
                        # total_route_affected never silently undercounts.
                        _emit_robot_trace(
                            self,
                            "trace_route_affected",
                            sim_time=float(self.simulation_time),
                            robot_id="R1",
                            path_goal=route_affected_agent.active_path_goal_xy if route_affected_agent else None,
                            active=robot_xy_now,
                            mapped_obs=len(self.mapped_obstacle_points),
                            new_obstacle_count=len(newly_discovered),
                            bbox=new_obstacle_bbox,
                            action="repair_throttled",
                        )
                        return

                    # A route-repair replan is a stronger event than
                    # prefetch: discard any pending path computed under the
                    # OLD route context so it cannot be silently promoted
                    # via ACCEPT_PENDING_PATH once the repaired route is
                    # accepted (same reasoning as the REPLAN_FOR_SAFETY
                    # branch in apply_navigation_decision()). Does not
                    # touch the active route itself.
                    if route_affected_agent is not None:
                        route_affected_agent.invalidate_pending_path(
                            reason="route_affected: new obstacle affects current route"
                        )
                        self._invalidate_prefetch_request(
                            0, reason="route_affected: new obstacle affects current route"
                        )
                    # Belief-trace artifact completeness: this is the
                    # previously-missing case -- route_affected=yes AND the
                    # repair is actually allowed to proceed (not throttled).
                    # Without this call, total_route_affected only counted
                    # throttled occurrences, undercounting every run where
                    # repairs mostly succeed.
                    _emit_robot_trace(
                        self,
                        "trace_route_affected",
                        sim_time=float(self.simulation_time),
                        robot_id="R1",
                        path_goal=route_affected_agent.active_path_goal_xy if route_affected_agent else None,
                        active=robot_xy_now,
                        mapped_obs=len(self.mapped_obstacle_points),
                        new_obstacle_count=len(newly_discovered),
                        bbox=new_obstacle_bbox,
                        action="repair_requested",
                    )
                    self.replan_after_new_information("New obstacle affects current route.")
                    return

        _misc_perf_start = time.perf_counter()
        robot_position = (float(self.robot.x), float(self.robot.y))
        robot_radius = self.safety_radius()
        target = self.active_target_xy()

        current_collision = self.collision_checker.check_position(
            position=robot_position,
            obstacles=self.config.obstacles,
            robot_radius=robot_radius,
        )
        _record_perf(self, "misc", time.perf_counter() - _misc_perf_start)

        if current_collision.collision:
            self.last_collision_report = current_collision
            self.stop_for_collision(
                "COLLISION: robot is inside an obstacle safety region."
            )
            return

        # ── Phase 2A: agent-based navigation decision ──────────────────────
        # The agent owns navigation state and policy; the engine is the executor.
        # If the agent layer is not yet available (first frame, registry not
        # initialised) we fall back to the legacy code path so the sim never
        # stalls.
        agent = self.runtime_agent(None)
        old_mode = mode_name(self.robot)
        self.sync_narrow_passage_speed_cap(agent)

        if agent is not None and RobotObservation is not None:
            # ── New OOP flow ──────────────────────────────────────────────
            # build_observation pre-computes active_segment_blocked.
            #
            # nav_debug_capture is always built now (see _finalize_
            # navigation_debug_snapshot()'s docstring): every tick is
            # recorded unconditionally, so history is already populated by
            # the time a user turns Navigation on mid-run -- navigation_
            # debug_enabled only gates the UI's browse/inspect/restore
            # affordances, not whether a tick gets captured.
            nav_debug_capture = NavigationDebugCapture()
            _runtime_state_build_perf_start = time.perf_counter()
            obs = self.build_observation(self.robot, agent, None, capture=nav_debug_capture)
            _record_perf(self, "runtime_state_build", time.perf_counter() - _runtime_state_build_perf_start)

            # Compute nominal control first so predicted_motion_report() can
            # use it; pass the blocked flag so the controller can slow down.
            _controller_perf_start = time.perf_counter()
            self.last_control = self.nominal_control_safe(
                blocked=obs.active_segment_blocked, capture=nav_debug_capture
            )
            self.last_control = self.apply_hazard_safety_filter(self.robot, self.last_control)

            predicted_report = self.predicted_motion_report(
                control=self.last_control,
                dt=dt,
                robot_radius=robot_radius,
                known_obstacle_points=self.obstacle_points_for_segment_safety_check(robot_position, robot_radius),
                use_ground_truth=True,
                capture=nav_debug_capture,
            )
            _record_perf(self, "controller", time.perf_counter() - _controller_perf_start)
            if predicted_report is not None and getattr(predicted_report, "collision", False):
                self.last_collision_report = predicted_report
                obs.predicted_collision = True

            planner_services = self.ensure_planner_services()
            _nav_decision_perf_start = time.perf_counter()
            decision = agent.step(obs, planner_services, dt)
            _record_perf(self, "nav_decision", time.perf_counter() - _nav_decision_perf_start)
            _apply_decision_perf_start = time.perf_counter()
            should_brake = self.apply_navigation_decision(self.robot, agent, decision)
            _record_perf(self, "apply_decision", time.perf_counter() - _apply_decision_perf_start)

            if nav_debug_capture is not None:
                # getattr-guarded -- see the equivalent call site in
                # apply_route_result() for why.
                _nav_debug_finalize = getattr(self, "_finalize_navigation_debug_snapshot", None)
                if callable(_nav_debug_finalize):
                    _nav_debug_finalize(
                        agent=agent,
                        decision_kind=str(decision.kind),
                        decision_reason=str(decision.reason),
                        event_kind=self._navigation_debug_event_kind_for_decision(decision, predicted_report),
                        capture=nav_debug_capture,
                    )

            if should_brake:
                self.last_control = self.brake_control_for_collision()
                self.canvas.set_runtime_state(
                    robot=self.robot,
                    path_points=self.path_points,
                    last_control=self.last_control,
                    simulation_time=self.simulation_time,
                    simulation_speed=self.simulation_speed,
                )
                return

            target = self.active_target_xy()
            _motion_update_perf_start = time.perf_counter()
            self.robot.update(self.last_control, dt)
            _record_perf(self, "motion_update", time.perf_counter() - _motion_update_perf_start)
            _telemetry_perf_start = time.perf_counter()
            self.log_robot_motion(
                self.robot,
                robot_index=None,
                control=self.last_control,
                target=target,
            )
            _record_perf(self, "telemetry", time.perf_counter() - _telemetry_perf_start)

        else:
            # ── Legacy fallback (agent layer unavailable) ─────────────────
            # Segment blocking is checked against the robot's current map, not
            # against omniscient ground truth. If blocked, it requests
            # replanning instead of treating a sharp turn as a terminal failure.
            local_path_report = self.collision_checker.check_segment_points(
                start=robot_position,
                end=target,
                obstacle_points=self.mapped_obstacle_points,
                robot_radius=robot_radius,
            )

            if local_path_report.collision:
                self.last_collision_report = local_path_report
                if self.replan_after_new_information("Active segment blocked by known obstacle."):
                    return

                self.last_control = self.nominal_control_safe(blocked=True)
                self.canvas.set_last_control(self.last_control)
                self.stop_for_collision(
                    "BLOCKED: direct segment intersects a known obstacle and replanning is unavailable."
                )
                return

            self.last_control = self.nominal_control_safe(blocked=False)
            self.last_control = self.apply_hazard_safety_filter(self.robot, self.last_control)

            predicted_report = self.predicted_motion_report(
                control=self.last_control,
                dt=dt,
                robot_radius=robot_radius,
                known_obstacle_points=list(self.mapped_obstacle_points),
                use_ground_truth=True,
            )
            if predicted_report is not None and getattr(predicted_report, "collision", False):
                self.last_collision_report = predicted_report
                if self.replan_after_new_information("Predicted collision before motion update."):
                    return

                self.stop_for_collision(
                    "PREDICTED COLLISION: control would enter an obstacle safety region before next update."
                )
                return

            self.robot.update(self.last_control, dt)
            self.log_robot_motion(
                self.robot,
                robot_index=None,
                control=self.last_control,
                target=target,
            )

            # In exploration mode, reaching a frontier target should select the
            # next frontier instead of leaving the robot permanently DONE.
            if self.is_exploration_mode():
                target = self.active_target_xy()
                if target is not None and math.hypot(
                    float(self.robot.x) - target[0], float(self.robot.y) - target[1]
                ) <= max(self.config.goal_tolerance, 0.25):
                    if self.request_exploration_route_async("Exploration target reached."):
                        return

        # ── Shared post-step checks ───────────────────────────────────────
        _misc_post_perf_start = time.perf_counter()
        new_mode = mode_name(self.robot)
        if old_mode != new_mode:
            self.canvas.set_status(f"State transition: {old_mode} → {new_mode}")

        post_position = (float(self.robot.x), float(self.robot.y))
        post_collision = self.collision_checker.check_position(
            position=post_position,
            obstacles=self.config.obstacles,
            robot_radius=robot_radius,
        )
        _record_perf(self, "misc", time.perf_counter() - _misc_post_perf_start)

        if post_collision.collision:
            self.last_collision_report = post_collision
            self.stop_for_collision(
                "COLLISION: robot entered an obstacle safety region after update."
            )
            return

        _misc_path_perf_start = time.perf_counter()
        new_path_point = (float(self.robot.x), float(self.robot.y))
        self._append_executed_path_point(new_path_point)
        _record_perf(self, "misc", time.perf_counter() - _misc_path_perf_start)

        # Skip the forced per-tick canvas repaint while latched in an
        # exploration-exhausted HOLD and not yet due for the ~1Hz trickle
        # update (see skip_for_exhausted_hold above) -- the robot isn't
        # moving and has nothing left to route to, so nothing visually
        # meaningful changes between ticks anyway.
        if not skip_for_exhausted_hold:
            _canvas_update_perf_start = time.perf_counter()
            self.canvas.set_runtime_state(
                robot=self.robot,
                path_points=self.path_points,
                last_control=self.last_control,
                simulation_time=self.simulation_time,
                simulation_speed=self.simulation_speed,
            )
            _record_perf(self, "canvas_state_update", time.perf_counter() - _canvas_update_perf_start)

        _misc_tail_perf_start = time.perf_counter()
        self.push_grid_overlay_snapshot_if_due()
        _record_perf(self, "misc", time.perf_counter() - _misc_tail_perf_start)

    # ========================================================
    # NEW POO INTERFACE — gradual migration helpers
    #
    # These methods wrap the new RobotObservation / NavigationDecision /
    # PlannerServices layer.  The existing simulation_step / simulation_step_multi
    # loops are untouched; call these from the new architecture incrementally.
    # ========================================================

    def ensure_planner_services(self):
        """Return the shared PlannerServices instance, creating it if needed.

        Refreshes is_candidate_reachable on every call so exploration target
        selection always checks against the current robot pose and map --
        see make_exploration_reachability_check(). The callback itself is
        now LAZY (see that method's docstring): this call only captures a
        snapshot of robot pose/config, it does not sanitize obstacles or
        build a planning grid.

        planner_services_refresh_ms times this ENTIRE method as a single
        top-level section for the optional [PERF] summary -- diagnosis-only
        instrumentation, added because this call sits between the
        "controller" and "nav_decision" timers in simulation_step() without
        any timer of its own, so its cost was previously silently folded
        into unaccounted_ms. Now that the expensive work is lazy, this
        measures only the remaining cheap snapshot/closure-creation work --
        see make_exploration_reachability_check()'s own docstring for where
        the (now lazy) nested sub-timings are recorded instead. Reading/
        recording this timing never changes what is_candidate_reachable
        does or how often this method is called.
        """
        if PlannerServices is None:
            return None
        if not hasattr(self, "_planner_services") or self._planner_services is None:
            self._planner_services = PlannerServices()
        _refresh_start = time.perf_counter()
        self._planner_services.is_candidate_reachable = self.make_exploration_reachability_check(
            getattr(self, "robot", None)
        )
        _record_perf(self, "planner_services_refresh", time.perf_counter() - _refresh_start)
        return self._planner_services

    def make_exploration_reachability_check(self, robot):
        """Build an is_candidate_reachable(xy) callback for PlannerServices,
        backed by the SAME planning grid (belief + dense mapped-obstacle
        samples, robot-radius padded) real single-robot navigation A* uses.

        This is what lets FoV-aware target selection reject a candidate the
        real planner would immediately fail on with "no path found",
        without exploration_planners.py depending on engine.py. Returns
        None (no filtering, existing behavior) when there is no live robot
        or the planner package is unavailable.

        LAZY by design: creating the callback only captures a cheap
        snapshot of robot pose/radius/grid_resolution/planner_type/
        goal_tolerance -- it does NOT sanitize mapped_obstacle_points or
        build a planning grid. That expensive work happens only inside
        _is_reachable(), on its FIRST actual invocation, and the result is
        cached in the `context` closure cell so every further invocation of
        the SAME callback (i.e. within the same tick, since
        ensure_planner_services() builds a brand-new callback every tick)
        reuses it instead of rebuilding. If exploration target selection
        never calls is_candidate_reachable() at all this tick (e.g. no
        candidates needed a reachability check), the grid is never built.
        A later tick's callback is a fresh closure over that tick's own
        snapshot, so it naturally observes the current pose/map on ITS
        first invocation -- no cache persists across ticks and no
        invalidation policy is needed.

        Diagnosis-only instrumentation: reachability_context_build_ms times
        the context-building work (now inside _is_reachable()'s first call,
        nested inside whatever top-level section actually invokes the
        callback -- typically "nav_decision", since exploration target
        selection runs inside agent.step() -- never subtracted from
        unaccounted_ms a second time, see perf_monitor.py's
        _UNACCOUNTED_SECTIONS comment), and
        reachability_obstacle_prepare_ms/reachability_grid_build_ms further
        split that into obstacle sanitization vs. planning-grid
        construction. reachability_context_builds counts how many times the
        context was ACTUALLY built (i.e. is_candidate_reachable was
        genuinely invoked at least once), not how many callbacks were
        created -- showing how often this work happens relative to how many
        ticks even needed it.
        """
        if robot is None or compute_planned_waypoints is None:
            return None

        robot_radius = self.safety_radius_for_robot(robot)
        resolution = float(self.config.grid_resolution)
        start_xy = (float(robot.x), float(robot.y))
        planner_type = str(self.config.planner_type)
        bounds = (WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX)
        goal_tolerance = float(self.config.goal_tolerance)

        # Populated on first real invocation of _is_reachable() below, then
        # reused by every further invocation of THIS SAME callback object.
        context: dict = {}

        def _build_context() -> None:
            _context_build_start = time.perf_counter()

            _obstacle_prepare_start = time.perf_counter()
            obstacle_points, _ = self.sanitize_planner_obstacle_points(
                list(self.mapped_obstacle_points),
                start_xy=start_xy,
                robot_radius=robot_radius,
                resolution=resolution,
            )
            _record_perf(self, "reachability_obstacle_prepare", time.perf_counter() - _obstacle_prepare_start)

            _grid_build_start = time.perf_counter()
            planning_grid = self.build_planning_grid_for_robot(
                robot,
                obstacle_points=obstacle_points,
                robot_radius=robot_radius,
            )
            _record_perf(self, "reachability_grid_build", time.perf_counter() - _grid_build_start)

            context["planning_grid"] = planning_grid
            _record_perf(self, "reachability_context_build", time.perf_counter() - _context_build_start)
            self.reachability_context_builds = getattr(self, "reachability_context_builds", 0) + 1

        def _is_reachable(candidate_xy: tuple[float, float]) -> bool:
            if "planning_grid" not in context:
                _build_context()
            return candidate_reachable_on_planning_grid(
                context["planning_grid"],
                planner_type,
                start_xy,
                (float(candidate_xy[0]), float(candidate_xy[1])),
                bounds=bounds,
                resolution=resolution,
                robot_radius=robot_radius,
                goal_tolerance=goal_tolerance,
            )
        return _is_reachable

    def build_observation(self, robot, agent, robot_index=None, capture=None):
        """
        Build a RobotObservation snapshot for one robot.

        The engine pre-computes the two safety flags (active_segment_blocked,
        predicted_collision) so the agent's step() never touches engine internals.

        Parameters
        ----------
        robot:
            The live Robot physics object.
        agent:
            The RobotAgent for this robot.
        robot_index:
            Index in self.robots (None for single-robot mode).
        capture:
            Optional NavigationDebugCapture. When provided, stashes the
            CollisionReport already computed below (checker, blocking point,
            distance) instead of discarding it down to
            active_segment_blocked's bare bool. None (the default) costs
            nothing extra.
        """
        if RobotObservation is None:
            return None

        robot_xy = (float(robot.x), float(robot.y))
        robot_radius = self.safety_radius_for_robot(robot)
        sensor_range = float(getattr(robot, "vision", self.config.vision))

        # Pre-compute active segment blocked flag.
        active_segment_blocked = False
        if self.collision_checker is not None:
            target = self.active_target_xy()
            if target is not None:
                dynamic_pts = (
                    self.dynamic_robot_obstacle_points_for_robot(int(robot_index))
                    if robot_index is not None and self.robots
                    else []
                )
                report = self.collision_checker.check_segment_points(
                    start=robot_xy,
                    end=target,
                    obstacle_points=self.obstacle_points_for_segment_safety_check(robot_xy, robot_radius) + dynamic_pts,
                    robot_radius=robot_radius,
                )
                active_segment_blocked = bool(report.collision)
                if capture is not None:
                    capture.active_segment = clearance_terms_from_report(
                        report, checker="check_segment_points", required_clearance=robot_radius
                    )

        # Dynamic obstacles: other robots as (cx, cy, radius) disks.
        dynamic_obstacles: list[tuple[float, float, float]] = []
        if self.robots and robot_index is not None:
            for other_idx, other in enumerate(self.robots):
                if other_idx == int(robot_index):
                    continue
                dynamic_obstacles.append(
                    (float(other.x), float(other.y), self.safety_radius_for_robot(other))
                )

        # Excluded frontier targets: other robots' current frontiers.
        excluded: list[tuple[float, float]] = []
        if hasattr(self, "multi_exploration_targets") and self.multi_exploration_targets:
            for idx, t in enumerate(self.multi_exploration_targets):
                if idx != (robot_index if robot_index is not None else 0) and t is not None:
                    excluded.append((float(t[0]), float(t[1])))

        return RobotObservation(
            robot_xy=robot_xy,
            robot_heading=float(robot.theta),
            robot_radius=robot_radius,
            belief_map=self.ensure_belief_map(),
            planning_grid=None,  # built lazily by PlannerServices when needed
            mapped_obstacle_points=list(self.mapped_obstacle_points),
            dynamic_obstacles=dynamic_obstacles,
            active_segment_blocked=active_segment_blocked,
            predicted_collision=False,  # caller can set after nominal control
            current_time=float(self.simulation_time),
            grid_resolution=float(self.config.grid_resolution),
            goal_tolerance=float(
                getattr(robot, "_sim_goal_tolerance", self.config.goal_tolerance)
            ),
            sensor_range=sensor_range,
            final_goal_xy=self.final_goal_xy(),
            vision_model=str(self.config.vision_model),
            ipp_distance_penalty=float(self.config.ipp_distance_penalty),
            excluded_targets=excluded,
            route_points_by_robot=self.multi_active_route_points_by_robot()
            if self.robots
            else [],
        )

    def apply_navigation_decision(self, robot, agent, decision) -> bool:
        """
        Apply a NavigationDecision returned by agent.step() to robot and planner.

        Returns True when the engine should use brake control this frame.

        This method is the counterpart of build_observation().  Together they
        form the new "engine as executor" contract:

            observation = self.build_observation(robot, agent, idx)
            decision    = agent.step(observation, self.ensure_planner_services(), dt)
            should_brake = self.apply_navigation_decision(robot, agent, decision)

        Integration notes
        -----------------
        The existing simulation_step / simulation_step_multi loops are not yet
        replaced.  Wire these calls in incrementally; both paths can coexist.

        FOLLOW_PATH:
            Engine does nothing extra; robot follows its existing waypoints.

        BRAKE:
            Engine uses brake_control_for_collision() for this robot.

        HOLD:
            Engine sets robot target to current position.
            NEVER falls back to G while an exploration planner is active.

        REQUEST_PLAN:
            Engine asks the planner for a new route to decision.target.
            Uses async worker for non-Direct planners; brakes if decision.brake.

        PREFETCH_NEXT_TARGET:
            Engine stores decision.target in agent.pending_target_xy.
            TODO (next phase): kick off an async PlannerWorker for the prefetch
            and write the result into agent.pending_path when ready.

        ACCEPT_PENDING_PATH:
            Engine calls agent.accept_pending_path() and pushes the waypoints
            into the Robot object.

        REPLAN_FOR_SAFETY:
            Engine triggers a safety replan and brakes while computing.

            Future: a CBF (control barrier function) safety filter could
            replace this reactive replan-and-brake pattern with a
            continuous collision-avoidance constraint on the commanded
            control itself, instead of discrete "detect unsafe -> stop ->
            replan" cycles. Not implemented here -- narrow-passage
            stability in this round is handled entirely by throttling/
            speed-capping the existing reactive path (see
            RobotAgent.route_affected_replan_allowed()/
            narrow_passage_slowdown_until_time).
        """
        kind = decision.kind

        # NavigationSupervisor centralizes the two decision-level invariants
        # that used to be separate inline checks scattered through this
        # method: (1) REPLAN_FOR_SAFETY only makes sense with an active
        # route -- RobotAgent.step() emits it unconditionally whenever
        # active_segment_blocked/predicted_collision is set, even with
        # nothing to replan (e.g. right after exploration_exhausted() put
        # the agent into a stable HOLD), because that check runs before
        # ExplorationBehavior.update() is ever reached; (2) REQUEST_PLAN to
        # a target already within goal_tolerance produces a near-zero-length
        # route that gets "reached" again within a tick. Both normalize to
        # HOLD here, at the engine boundary, before telemetry logs the
        # decision and before any route-request/failure-marking logic runs
        # below. Scoped to single-robot mode (not self.robots) -- multi-robot
        # has its own multi_safety_replan_allowed()/per-index route state
        # and assign_route_to_multi_robot() dedup, untouched here.
        if not self.robots:
            original_kind = kind
            decision = NavigationSupervisor.normalize_decision(
                agent,
                SimpleNamespace(robot_xy=(float(robot.x), float(robot.y))) if robot is not None else None,
                decision,
                float(self.config.goal_tolerance),
                map_signature=len(self.mapped_obstacle_points),
            )
            kind = decision.kind
            if original_kind == "REQUEST_PLAN" and kind == "HOLD" and agent is not None:
                # normalize_decision() is pure and never mutates agent/canvas
                # state; this mirrors what the REQUEST_PLAN-specific inline
                # guard it replaces used to do here: an already-reached
                # target must not linger in exploration_target_xy, or
                # ExplorationBehavior step 6 (no active path) would propose
                # the exact same already-reached target again next tick.
                agent.exploration_target_xy = None
                self.current_exploration_target = None
                self.canvas.set_exploration_target(None)

        if kind != "FOLLOW_PATH":
            _telemetry_nav_decision_perf_start = time.perf_counter()
            self.telemetry.report_nav_decision(
                sim_time=float(self.simulation_time),
                robot_label="R1",
                kind=kind,
                reason=decision.reason,
                active_target=getattr(agent, "active_target", lambda: None)(),
                path_goal=getattr(agent, "active_path_goal_xy", None),
                pending_target=getattr(agent, "pending_target_xy", None),
            )
            _record_perf(self, "telemetry", time.perf_counter() - _telemetry_nav_decision_perf_start)

            # Opt-in terminal trace only (ROBOT_TRACE=decision); never
            # printed/GUI-consoled unless explicitly enabled.
            _emit_robot_trace(
                self,
                "trace_decision",
                sim_time=float(self.simulation_time),
                robot_label="R1",
                kind=kind,
                reason=str(decision.reason),
                active_target=getattr(agent, "active_target", lambda: None)(),
                path_goal=getattr(agent, "active_path_goal_xy", None),
                pending_target=getattr(agent, "pending_target_xy", None),
            )

            if (
                agent is not None
                and kind == "REQUEST_PLAN"
                and bool(getattr(decision, "force_new_target", False))
            ):
                _emit_robot_trace(
                    self,
                    "trace_frontier",
                    sim_time=float(self.simulation_time),
                    source="map-wide-fallback" if agent.last_map_wide_fallback_attempted else agent.planner_mode,
                    selected=decision.target,
                    generated=agent.last_frontier_candidate_count,
                )

        # Diagnostics only, DEBUG-level (never spams normal/quiet consoles):
        # logged once, exactly at the tick exploration_exhausted() actually
        # fires, using data already computed for this decision -- never a
        # new per-tick candidate-generation pass. Read-only check against
        # decision.reason's existing text; does not change what that text
        # is or how it is decided.
        if kind == "HOLD" and agent is not None and "exploration exhausted" in str(decision.reason):
            belief = getattr(self, "belief_map", None)
            unknown_cells = int(np.count_nonzero(belief.grid == UNKNOWN)) if belief is not None else -1
            self.telemetry.debug(
                f"[EXHAUSTION_DIAG] unknown_cells={unknown_cells} "
                f"recovery_candidates={len(agent.recovery_targets())} "
                f"map_wide_fallback_tried={bool(agent.last_map_wide_fallback_attempted)} "
                f"last_frontier_candidates={int(agent.last_frontier_candidate_count)} "
                f'last_frontier_reason="{agent.last_frontier_selection_reason}" '
                f'reason="{decision.reason}"'
            )
            # Opt-in terminal trace only (ROBOT_TRACE=frontier).
            _emit_robot_trace(
                self,
                "trace_frontier",
                sim_time=float(self.simulation_time),
                source="map-wide-fallback" if agent.last_map_wide_fallback_attempted else agent.planner_mode,
                selected=None,
                generated=agent.last_frontier_candidate_count,
            )

        if kind == "FOLLOW_PATH":
            return False

        if kind == "BRAKE":
            return True

        if kind == "HOLD":
            hold_xy = (float(robot.x), float(robot.y))
            self.set_robot_goal_or_waypoints(robot, [hold_xy])
            # Route invalidation alone does not stop the robot: brake_control()
            # only decelerates gradually, and the dynamics model advances
            # position using the velocity from BEFORE this tick's
            # deceleration is applied -- so residual velocity can still
            # carry the robot into a collision after navigation has already
            # decided to hold (see Robot.force_stop()). Every HOLD is
            # treated as a hard stop here: NavigationDecision draws no
            # distinction between a "normal" HOLD (no valid next frontier)
            # and a safety-driven one (predicted collision normalized by
            # NavigationSupervisor) -- both reach this exact branch.
            if hasattr(robot, "force_stop"):
                robot.force_stop(reason=decision.reason or "hold")
            agent.invalidate_route(reason=decision.reason or "hold")
            hold_robot_index = 0
            if getattr(self, "robots", None):
                hold_robot_index = next((i for i, candidate in enumerate(self.robots) if candidate is robot), 0)
            self._invalidate_prefetch_request(hold_robot_index, reason=decision.reason or "hold")
            return False

        if kind == "ACCEPT_PENDING_PATH":
            robot_index = 0
            dynamic_points: list[tuple[float, float]] = []
            if getattr(self, "robots", None):
                robot_index = next((i for i, candidate in enumerate(self.robots) if candidate is robot), 0)
                dynamic_points = self.dynamic_robot_obstacle_points_for_robot(robot_index)

            # A prefetch may have been planned from a pose that is already
            # stale by the time it is promoted. Normalize it against the
            # robot's CURRENT pose before RobotAgent installs the waypoints.
            # If the final frontier is directly safe, this collapses the old
            # detour to one segment instead of making the robot drive back to
            # the prefetch origin.
            if agent.pending_path:
                normalized_pending = SimulationControllerMixin.clean_waypoints_for_robot(
                    self,
                    robot,
                    list(agent.pending_path),
                    obstacle_points=list(self.mapped_obstacle_points) + dynamic_points,
                )
                if not normalized_pending:
                    agent.reject_pending_path("pending path obsolete at handoff")
                    self._invalidate_prefetch_request(robot_index, reason="pending path obsolete at handoff")
                    return False
                agent.pending_path = normalized_pending

            _pending_accept_perf_start = time.perf_counter()
            waypoints = agent.accept_pending_path()
            _record_perf(self, "pending_path_acceptance", time.perf_counter() - _pending_accept_perf_start)
            if waypoints:
                pending_captures = getattr(self, "_nav_debug_pending_plan_capture_by_robot", {})
                promoted_plan_capture = pending_captures.pop(robot_index, None)
                if promoted_plan_capture is not None:
                    self._nav_debug_last_accepted_plan = promoted_plan_capture

                start_xy = (float(robot.x), float(robot.y))
                self.set_robot_goal_or_waypoints(robot, waypoints)
                self.canvas.set_planned_path([start_xy] + list(waypoints))
                if self.is_exploration_mode():
                    self.canvas.set_exploration_target(waypoints[-1])
                # Belief-trace artifact completeness: a promoted prefetched
                # path is a real route assignment even though it bypasses
                # log_route_assignment()/report_route_success() (no planner
                # call happens here) -- record it too, so route_events.csv
                # covers every path the robot actually starts following.
                prefetch_length = 0.0
                previous = start_xy
                for point in waypoints:
                    prefetch_length += math.hypot(float(point[0]) - float(previous[0]), float(point[1]) - float(previous[1]))
                    previous = point
                _emit_robot_trace(
                    self,
                    "trace_route",
                    sim_time=float(self.simulation_time),
                    robot_label="R1",
                    result="ok",
                    start=start_xy,
                    goal=waypoints[-1],
                    reason="accepted pending path (prefetch)",
                    waypoint_count=len(waypoints),
                    length=prefetch_length,
                    mapped_obstacle_count=len(self.mapped_obstacle_points),
                    planner=str(self.config.planner_type),
                    simplifier=str(self.config.path_simplifier),
                )
            return False

        if kind == "PREFETCH_NEXT_TARGET":
            agent.last_prefetch_time = float(self.simulation_time)
            self.request_prefetch_route_async(robot, agent, decision)
            return False

        if kind == "REQUEST_PLAN":
            # Route through the existing planner infrastructure.
            if self.robots:
                robot_index = next(
                    (i for i, r in enumerate(self.robots) if r is robot), 0
                )
                self.assign_route_to_multi_robot(
                    robot_index,
                    reason=decision.reason or "agent requested plan",
                    force_new_exploration_target=True,
                )
            else:
                # NavigationSupervisor.normalize_decision() above already
                # guarantees that a REQUEST_PLAN reaching this point is not
                # for an already-reached target (it would have been
                # normalized to HOLD, which returns before this branch is
                # ever entered) -- no redundant re-check needed here.
                if decision.force_new_target and agent is not None:
                    # The frontier was just reached.  Clear exploration_target_xy
                    # so select_navigation_goal() inside request_route_async()
                    # also sees current_target=None and cannot return it by hysteresis.
                    agent.exploration_target_xy = None
                self.request_route_async(
                    decision.reason or "agent requested plan",
                    target_override=decision.target if self.is_exploration_mode() else None,
                )
            return bool(decision.brake)

        if kind == "REPLAN_FOR_SAFETY":
            if self.robots:
                robot_index = next(
                    (i for i, r in enumerate(self.robots) if r is robot), 0
                )
                if self.multi_safety_replan_allowed(robot_index, decision.reason, decision.target):
                    self.assign_route_to_multi_robot(
                        robot_index,
                        reason=f"safety replan: {decision.reason}",
                        force_new_exploration_target=bool(self.is_exploration_mode()),
                    )
            else:
                # By this point kind can only still be "REPLAN_FOR_SAFETY"
                # here if NavigationSupervisor.normalize_decision() above
                # left it unchanged -- i.e. the agent is guaranteed to have
                # an active route. (No redundant has_active_route check
                # here; see the supervisor call at the top of this method.)
                # A safety replan is a stronger event than prefetch: any
                # pending path (computed for a different, unrelated target
                # under the OLD route context) is no longer trustworthy and
                # must not be silently promoted later via
                # ACCEPT_PENDING_PATH once the safety route is accepted.
                # Does not touch the active route itself -- only the
                # planner request below (if allowed) or the
                # elif/invalidate_failed_exploration_route() branch further
                # down decide what happens to that.
                agent.invalidate_pending_path(reason=f"safety replan: {decision.reason}")
                self._invalidate_prefetch_request(0, reason=f"safety replan: {decision.reason}")

                # Mirror multi_safety_replan_allowed(): throttle identical
                # (reason, target) safety replans instead of launching a new
                # planner request every single tick the segment stays
                # blocked. Without this, a route that gets accepted but
                # still has its first segment blocked (e.g. by a
                # newly-mapped obstacle sample) re-triggers REPLAN_FOR_SAFETY
                # on the very next tick, forever.
                allowed = agent.safety_replan_allowed(
                    reason=decision.reason or "",
                    target=decision.target,
                    current_time=float(self.simulation_time),
                    cooldown=self.safety_replan_cooldown_seconds(),
                    route_generation=agent.route_generation,
                )
                if allowed:
                    self.replan_after_new_information(
                        f"safety replan: {decision.reason}"
                    )
                elif agent is not None and self.is_exploration_mode():
                    # Same blocked segment/target as last time, within the
                    # cooldown: stop retrying it forever. Hold at the
                    # current position and mark the exploration target
                    # failed so ExplorationBehavior's recovery path
                    # (cooldown + blacklist, see _pick_next_target()) picks
                    # a fresh target instead of re-requesting the same
                    # route that keeps ending up blocked.
                    hold_xy = (float(robot.x), float(robot.y))
                    attempted_target = agent.exploration_target_xy
                    self.set_robot_goal_or_waypoints(robot, [hold_xy])
                    # Same hard-stop reasoning as the generic HOLD branch
                    # above: this route is being invalidated specifically
                    # because it kept ending up blocked/unsafe, so residual
                    # velocity here is exactly the "coast into a collision
                    # after safety logic already decided to stop" scenario
                    # this fix exists for.
                    if hasattr(robot, "force_stop"):
                        robot.force_stop(reason=f"repeated safety replan: {decision.reason}")
                    agent.invalidate_failed_exploration_route(
                        reason=f"repeated safety replan: {decision.reason}",
                        current_time=float(self.simulation_time),
                        map_signature=len(self.mapped_obstacle_points),
                    )
                    self.current_exploration_target = None
                    self.canvas.set_exploration_target(None)
                    self.canvas.set_status(
                        f"Holding: repeated safety replan for the same target ({decision.reason}); "
                        "marking target as failed and re-selecting."
                    )
                    self.route_failure_count = getattr(self, "route_failure_count", 0) + 1
                    self.repeated_safety_replan_count = getattr(self, "repeated_safety_replan_count", 0) + 1
                    self.telemetry.report_route_failure(
                        robot_label="R1",
                        start_xy=hold_xy,
                        attempted_target=attempted_target,
                        reason=f"repeated safety replan: {decision.reason}",
                        planner_type=str(self.config.planner_type),
                        mapped_obstacle_count=len(self.mapped_obstacle_points),
                    )
                    # Opt-in terminal trace only (ROBOT_TRACE=route).
                    _emit_robot_trace(
                        self,
                        "trace_route",
                        sim_time=float(self.simulation_time),
                        robot_label="R1",
                        result="fail",
                        start=hold_xy,
                        goal=attempted_target,
                        reason=slug_route_failure_reason(f"repeated safety replan: {decision.reason}"),
                        mapped_obstacle_count=len(self.mapped_obstacle_points),
                    )
            return True  # always brake for safety replans

        return False

    def _invalidate_prefetch_request(self, robot_id: int, reason: str = "") -> None:
        """Retire whatever prefetch request is in flight for one robot slot.

        Every caller that discards pending state (route-affected repair,
        safety replan, a target/goal change, HOLD, ACCEPT_PENDING_PATH
        handoff, reset, snapshot restore, ...) must also call this -- pending
        state lives on the agent (pending_path/pending_target_xy), but the
        *request* backing it lives here in prefetch_workers/prefetch_
        request_ids/prefetch_targets, and nothing previously kept those two
        in sync. Without this, a still-running worker survives past the
        pending state it was computing, blocks request_prefetch_route_async()
        from launching a replacement for the same slot (see its "already
        running" guard), and its eventual on_prefetch_route_ready() callback
        can validate a stale route against whatever *new* request has since
        taken the slot.

        Safe to call with nothing in flight for robot_id -- a pure no-op.
        """
        idx = int(robot_id)

        if not hasattr(self, "prefetch_workers"):
            self.prefetch_workers = {}
        if not hasattr(self, "prefetch_request_ids"):
            self.prefetch_request_ids = {}
        if not hasattr(self, "prefetch_targets"):
            self.prefetch_targets = {}

        had_request = idx in self.prefetch_workers or idx in self.prefetch_request_ids

        # 1. Invalidate the current request id FIRST. This is what actually
        # makes a late callback harmless: on_prefetch_route_ready() ignores
        # any request_id that no longer matches prefetch_request_ids[idx],
        # regardless of whether step 3's cancel() below has any real effect.
        self.prefetch_request_ids.pop(idx, None)

        # 2. Retire the worker from its slot -- this also frees
        # request_prefetch_route_async()'s "already running" guard so a
        # replacement can be launched for this robot immediately.
        worker = self.prefetch_workers.pop(idx, None)

        # 3. Best-effort cancel. PlannerWorker (QRunnable) has no
        # cooperative-cancellation hook today, so this is a courtesy for a
        # future worker type, not something step 1 depends on.
        cancel = getattr(worker, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                pass

        # 4. Clean exclusively the pending state this request captured --
        # never agent.pending_path/pending_target_xy, which is the caller's
        # own responsibility (invalidate_route()/invalidate_pending_path()/
        # reject_pending_path()/direct assignment during restore).
        self.prefetch_targets.pop(idx, None)

        if had_request and reason:
            self.log_console_message(f"[PREFETCH] invalidated for robot {idx}: {reason}")

    def _invalidate_all_prefetch_requests(self, reason: str = "") -> None:
        """Invalidate every in-flight prefetch request -- reset/restore."""
        idxs = (
            set(getattr(self, "prefetch_workers", {}))
            | set(getattr(self, "prefetch_request_ids", {}))
            | set(getattr(self, "prefetch_targets", {}))
        )
        for idx in idxs:
            self._invalidate_prefetch_request(idx, reason=reason)

    def request_prefetch_route_async(
        self,
        robot,
        agent,
        decision,
        robot_index: int = 0,
    ) -> bool:
        """
        Launch a background planner for the *next* frontier without stopping
        the robot or touching planning_in_progress.

        The result lands in agent.pending_path via on_prefetch_route_ready().
        ExplorationBehavior decides when to promote it to the active path
        (ACCEPT_PENDING_PATH).
        """
        if robot is None or agent is None:
            return False

        target = (
            decision.target
            if decision.target is not None
            else agent.pending_target_xy
        )
        if target is None:
            return False

        idx = int(robot_index)

        # Avoid double-launching: if a worker is already running for this
        # robot, leave it alone.
        if idx in getattr(self, "prefetch_workers", {}):
            return False

        # "Direct" planner needs no A* — store the path immediately.
        if self.config.planner_type == "Direct":
            agent.mark_pending_path_requested(target)
            agent.pending_path = [target]
            agent.prefetch_success_count += 1
            self.log_console_message(f"[PREFETCH] direct route to target={target}")
            return True

        if compute_planned_waypoints is None:
            return False

        start_xy = (float(robot.x), float(robot.y))
        planner_kwargs = self.build_planner_kwargs_for_goal(
            start_xy, target, robot=robot
        )

        if not hasattr(self, "prefetch_request_counter"):
            self.prefetch_request_counter = 0
        self.prefetch_request_counter += 1
        request_id = self.prefetch_request_counter

        if not hasattr(self, "prefetch_request_ids"):
            self.prefetch_request_ids = {}
        self.prefetch_request_ids[idx] = request_id

        # Capture the target THIS request was launched for, independent of
        # agent.pending_target_xy -- on_prefetch_route_ready() validates
        # against this, never against the agent's live (possibly since-
        # changed) pending_target_xy. See _invalidate_prefetch_request()'s
        # docstring for why the two can otherwise diverge.
        if not hasattr(self, "prefetch_targets"):
            self.prefetch_targets = {}
        self.prefetch_targets[idx] = target

        worker = PlannerWorker(
            request_id=request_id,
            planner_kwargs=planner_kwargs,
            path_simplifier=self.config.path_simplifier,
            # Always built now -- see _finalize_navigation_debug_snapshot()'s
            # docstring: capture is unconditional, not gated on navigation_
            # debug_enabled.
            debug_capture=PlanDebugCapture(),
        )
        # Capture idx in the closure so stale callbacks go to the right robot.
        captured_idx = idx
        worker.signals.route_ready.connect(
            lambda rid, ok, rsn, wps: self.on_prefetch_route_ready(
                rid, captured_idx, ok, rsn, wps
            )
        )

        if not hasattr(self, "prefetch_workers"):
            self.prefetch_workers = {}
        self.prefetch_workers[idx] = worker

        # Store target now so agent.step() can track pending_target_xy, and
        # stamp the route context this prefetch was requested under (see
        # RobotAgent.mark_pending_path_requested()/accept_pending_path()'s
        # staleness check).
        agent.mark_pending_path_requested(target)

        self.planner_jobs_started = getattr(self, "planner_jobs_started", 0) + 1
        self.thread_pool.start(worker)
        self.log_console_message(f"[PREFETCH] requested target={target}")
        return True

    def on_prefetch_route_ready(
        self,
        request_id: int,
        robot_index: int,
        success: bool,
        reason: str,
        waypoints: list,
    ) -> None:
        """
        Callback fired when a prefetch PlannerWorker finishes.

        Never touches planning_in_progress, never brakes the robot, and never
        clears the current active path.  The agent decides when to switch via
        ACCEPT_PENDING_PATH.
        """
        idx = int(robot_index)

        # Stale result: a newer prefetch (or an explicit _invalidate_
        # prefetch_request() call) already replaced this request's slot.
        # Checked BEFORE touching prefetch_workers/prefetch_targets below --
        # popping first would rip out whatever CURRENTLY-live request has
        # since taken this slot instead of the stale one that just landed.
        stored_id = getattr(self, "prefetch_request_ids", {}).get(idx)
        if stored_id != int(request_id):
            return

        prefetch_worker = getattr(self, "prefetch_workers", {}).get(idx)
        pending_plan_capture = getattr(prefetch_worker, "debug_capture", None)
        # The target THIS request was launched for -- never agent.pending_
        # target_xy, which is live agent state that could (in principle)
        # belong to a different request by the time this callback runs. See
        # _invalidate_prefetch_request()'s docstring.
        captured_target = getattr(self, "prefetch_targets", {}).get(idx)
        self._invalidate_prefetch_request(idx)  # this request is resolved either way

        self.planner_jobs_completed = getattr(self, "planner_jobs_completed", 0) + 1

        agent = self.runtime_agent(None if robot_index == 0 else robot_index)
        if agent is None:
            return

        if success and waypoints:
            clean_waypoints = [(float(p[0]), float(p[1])) for p in waypoints]

            # Reject a "successful" prefetch route whose final waypoint does
            # not actually reach captured_target (the target THIS request
            # was launched for) -- accept_pending_path() sets active_path_
            # goal_xy from pending_target_xy directly, not from the route's
            # own endpoint, so a mismatch here means the robot would follow
            # the route to a different point and then sit stuck there
            # forever (STATE showing a stale, unreached path_goal). This is
            # exactly the bug this check exists for.
            if not route_reaches_goal(
                clean_waypoints, captured_target, float(self.config.goal_tolerance)
            ):
                rejected_target = captured_target
                agent.reject_pending_path(f"{reason}; final waypoint does not reach path goal")
                # reject_pending_path() only clears pending_path/pending_target_xy
                # -- it does not blacklist the target, so without this the exact
                # same unreachable target could be immediately re-proposed by the
                # very next prefetch/REQUEST_PLAN cycle. Use the same
                # failed-target memory _pick_next_target()/select_navigation_goal()
                # already consult, so the exclusion window applies here exactly
                # like it does for a REQUEST_PLAN endpoint-mismatch failure (see
                # apply_route_result() -> invalidate_failed_exploration_route()).
                if rejected_target is not None:
                    agent.mark_exploration_target_failed(
                        rejected_target, current_time=float(self.simulation_time)
                    )
                self.log_console_message(
                    f"[PREFETCH] rejected: final waypoint does not reach target; {reason}"
                )
                # Captured unconditionally now, getattr-guarded -- see the
                # call site in apply_route_result() for why.
                _nav_debug_finalize = getattr(self, "_finalize_navigation_debug_snapshot", None)
                if callable(_nav_debug_finalize):
                    _nav_debug_finalize(
                        agent=agent,
                        decision_kind="ROUTE_RESULT",
                        decision_reason=f"{reason}; rejected: final waypoint does not reach path goal",
                        event_kind=NavigationDebugEventKind.ROUTE_REJECTED,
                        capture=NavigationDebugCapture(plan=pending_plan_capture, endpoint_reaches_goal=False),
                    )
                return

            # Reject a prefetch whose first segment (FROM THE ROBOT'S
            # CURRENT position, not wherever it was when the prefetch was
            # requested) is already unsafe by the same rule
            # apply_route_result() uses for the main route-acceptance
            # path. Without this, a prefetch computed before new obstacle
            # samples appeared near the route could still be promoted via
            # ACCEPT_PENDING_PATH straight into a now-unsafe segment.
            #
            # Uses _evaluate_route_first_segment() (not the bool-only
            # route_first_segment_blocked() wrapper) so the full
            # CollisionReport survives for the navigation debug snapshot --
            # same single computation either way.
            robot_xy_now = (float(self.robot.x), float(self.robot.y)) if self.robot is not None else None
            first_segment_report = (
                _evaluate_route_first_segment(
                    self.collision_checker,
                    robot_xy_now,
                    clean_waypoints[0],
                    list(self.mapped_obstacle_points),
                    self.safety_radius(),
                )
                if robot_xy_now is not None
                else None
            )
            if first_segment_report is not None and first_segment_report.collision:
                rejected_target = captured_target
                agent.first_segment_blocked_count += 1
                agent.reject_pending_path(f"{reason}; first segment blocked on arrival")
                if rejected_target is not None:
                    agent.mark_exploration_target_failed(
                        rejected_target, current_time=float(self.simulation_time)
                    )
                self.log_console_message(
                    f"[PREFETCH] rejected: first segment blocked on arrival; {reason}"
                )
                # Captured unconditionally now, getattr-guarded -- see the
                # call site in apply_route_result() for why.
                nav_capture = NavigationDebugCapture(plan=pending_plan_capture)
                nav_capture.first_segment = clearance_terms_from_report(
                    first_segment_report,
                    checker="check_segment_points",
                    required_clearance=self.safety_radius(),
                )
                _nav_debug_finalize = getattr(self, "_finalize_navigation_debug_snapshot", None)
                if callable(_nav_debug_finalize):
                    _nav_debug_finalize(
                        agent=agent,
                        decision_kind="ROUTE_RESULT",
                        decision_reason=f"{reason}; rejected: first segment blocked on arrival",
                        event_kind=NavigationDebugEventKind.ROUTE_REJECTED,
                        capture=nav_capture,
                    )
                return

            agent.pending_path = clean_waypoints
            pending_captures = getattr(self, "_nav_debug_pending_plan_capture_by_robot", None)
            if pending_captures is None:
                self._nav_debug_pending_plan_capture_by_robot = {}
                pending_captures = self._nav_debug_pending_plan_capture_by_robot
            if pending_plan_capture is not None:
                pending_captures[idx] = pending_plan_capture
            # pending_target_xy was set when the worker launched; keep it.
            agent.prefetch_success_count += 1
            self.log_console_message(
                f"[PREFETCH] success waypoints={len(clean_waypoints)}"
            )
        else:
            agent.reject_pending_path(reason)
            self.log_console_message(
                f"[PREFETCH] failed; keeping current route — {reason}"
            )

    # ========================================================

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.handle_start_pause_button()
        elif event.key() == Qt.Key_R:
            self.restart_simulation()

    # ========================================================
