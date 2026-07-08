"""
RobotAgent — per-robot navigation state and single step-entry point.

Responsibilities:
    • Store all navigation state for one robot (position, heading, goal,
      active path, pending prefetch path, metrics counters).
    • Provide the step() method that produces a NavigationDecision each frame.
    • Delegate exploration policy to ExplorationBehavior (lazily instantiated).

Does NOT:
    • Modify Qt widgets or the canvas.
    • Know about MainWindow or the engine's internal state.
    • Run A*, Dijkstra, or frontier detection — delegates to PlannerServices.
    • Perform rendering of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import ClassVar, TYPE_CHECKING, Literal

from robotics_sim.core.geometry import distance
from robotics_sim.planning.waypoint_manager import WaypointManager

if TYPE_CHECKING:
    from robotics_sim.navigation.navigation_decision import NavigationDecision
    from robotics_sim.simulation.observation import RobotObservation
    from robotics_sim.simulation.planner_services import PlannerServices


RobotStatus = Literal[
    "idle",
    "planning",
    "moving",
    "finished",
    "blocked",
    "responding_event",
]


def _as_point(point) -> tuple[float, float]:
    return (float(point[0]), float(point[1]))


@dataclass
class RobotAgent:
    """
    Per-robot navigation state container and decision entry point.

    Key field distinctions
    ----------------------
    final_goal_xy:
        GUI mission goal G. Executable ONLY in "Goal seeking" mode.
        In any exploration mode G is a visual reference, not a target.

    exploration_target_xy:
        Current frontier target assigned by the exploration planner.
        Set to None when no frontier is available.

    active_path_goal_xy:
        Final waypoint of the route currently tracked by the robot.
        Used to detect when a replan is needed because the target changed.

    active_path_mode:
        Planner mode that generated the current active route. Routes
        generated in a different mode are always invalidated before use.

    pending_path / pending_target_xy:
        Next route computed ahead of time (prefetch). The agent switches
        to this path when it gets close enough to the current frontier.
        A pending path is NEVER promoted to the active path automatically
        — the engine reads ACCEPT_PENDING_PATH and calls accept_pending_path().
    """

    robot_id: int
    position: tuple[float, float]
    heading: float = 0.0
    radius: float = 0.20

    planner_mode: str = "FoV-aware directional frontier"
    active_path_mode: str | None = None

    final_goal_xy: tuple[float, float] | None = None
    exploration_target_xy: tuple[float, float] | None = None
    active_path_goal_xy: tuple[float, float] | None = None

    # Prefetch state — next path computed before the current target is reached.
    pending_path: list[tuple[float, float]] | None = None
    pending_target_xy: tuple[float, float] | None = None

    status: RobotStatus = "idle"
    waypoints: WaypointManager = field(default_factory=WaypointManager)

    last_plan_reason: str = ""
    last_exploration_reason: str = ""
    last_prefetch_time: float = field(default=-1.0e9)
    last_replan_time: float = field(default=-1.0e9)

    # Recently-failed exploration targets — (target, sim_time_of_failure).
    # Bounded to _FAILED_TARGET_RETENTION_S so this cannot grow unbounded
    # over a long run; the actual "still blacklisted" window used for
    # re-selection is caller-supplied and typically much shorter (see
    # ExplorationBehavior._FAILED_TARGET_EXCLUSION_WINDOW).
    failed_exploration_targets: list[tuple[tuple[float, float], float]] = field(default_factory=list)
    last_exploration_failure_time: float = field(default=-1.0e9)

    # Throttle for identical REPLAN_FOR_SAFETY requests, mirroring the
    # engine's multi_safety_replan_allowed() for the single-robot path.
    last_safety_replan_time: float = field(default=-1.0e9)
    last_safety_replan_signature: tuple[str, tuple[float, float] | None] | None = None

    _FAILED_TARGET_RETENTION_S: ClassVar[float] = 60.0

    # Metrics counters (not displayed yet; available for future dashboard).
    stop_count_exploration: int = 0
    prefetch_success_count: int = 0
    prefetch_fail_count: int = 0
    safety_replan_count: int = 0
    target_switch_count: int = 0

    def __post_init__(self) -> None:
        # _behavior is set lazily via the property to avoid a circular import
        # at module load time (ExplorationBehavior -> navigation_decision ->
        # robot_agent would form a cycle if imported at top level).
        self._behavior = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ behavior

    @property
    def behavior(self):
        """ExplorationBehavior instance, created on first access."""
        if self._behavior is None:
            from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
            self._behavior = ExplorationBehavior()
        return self._behavior

    # ------------------------------------------------------------------ pose

    def set_position(self, position) -> None:
        self.position = _as_point(position)

    def set_heading(self, heading: float) -> None:
        self.heading = float(heading)

    # ------------------------------------------------------------------ goals

    def set_final_goal(self, goal) -> bool:
        """
        Update the manual mission goal.

        Returns True when the goal changed enough to require replanning in
        Goal seeking mode.
        """
        new_goal = _as_point(goal)
        changed = self.final_goal_xy is None or distance(self.final_goal_xy, new_goal) > 1e-9
        self.final_goal_xy = new_goal
        if changed and self.planner_mode == "Goal seeking":
            self.invalidate_route(reason="manual goal changed")
        return changed

    def set_planner_mode(self, mode: str) -> bool:
        """
        Change planner mode and invalidate routes from the old mode.

        Prevents a "Goal seeking" route from being followed in an exploration
        mode, and vice versa.
        """
        mode = str(mode)
        changed = mode != self.planner_mode
        if changed:
            self.planner_mode = mode
            self.invalidate_route(reason="planner mode changed")
            self.exploration_target_xy = None
        return changed

    def set_exploration_target(self, target, reason: str = "") -> bool:
        """
        Store a frontier target selected by the exploration planner.

        Returns True when the target changed enough to require a new path.
        """
        new_target = _as_point(target)
        changed = (
            self.exploration_target_xy is None
            or distance(self.exploration_target_xy, new_target) > 1e-9
        )
        self.exploration_target_xy = new_target
        self.last_exploration_reason = reason
        if changed and self.planner_mode != "Goal seeking":
            self.invalidate_route(reason="exploration target changed")
        return changed

    def desired_target_from_mode(self) -> tuple[float, float] | None:
        """
        Executable target implied by the current planner mode.

        Goal seeking  → final_goal_xy
        Exploration   → exploration_target_xy  (may be None)

        RULE: In exploration mode, final_goal_xy is NEVER returned here.
        """
        if self.planner_mode == "Goal seeking":
            return self.final_goal_xy
        return self.exploration_target_xy

    # ------------------------------------------------------------------ route state

    def invalidate_route(self, reason: str = "") -> None:
        """Clear the active path and the prefetch buffer.

        Does NOT touch exploration_target_xy: several callers (e.g.
        set_exploration_target()) invalidate the route to discard a
        now-stale path while deliberately keeping -- or having just
        assigned -- the exploration target. Use
        invalidate_failed_exploration_route() when the target itself must
        be abandoned because planning to it failed.
        """
        self.waypoints.clear()
        self.active_path_goal_xy = None
        self.active_path_mode = None
        self.pending_path = None
        self.pending_target_xy = None
        self.status = "idle"
        if reason:
            self.last_plan_reason = reason

    def invalidate_failed_exploration_route(
        self,
        reason: str = "",
        *,
        current_time: float = 0.0,
    ) -> None:
        """Clear the active/pending route AND the exploration target that
        failed to produce a usable path.

        Use this instead of invalidate_route() specifically when a planner
        attempt for the current exploration_target_xy has failed. Without
        clearing exploration_target_xy here, desired_target_from_mode()
        keeps returning the same unreachable target, and the exploration
        loop immediately re-requests a plan for it -- producing a repeated
        planner-failure loop instead of falling back to HOLD and picking a
        fresh target on the next tick.

        The failed target is also remembered (see mark_exploration_target_failed())
        so the next target-selection attempt can exclude it and a short
        retry cooldown can gate how soon that next attempt happens.
        """
        failed_target = self.exploration_target_xy
        self.invalidate_route(reason=reason)
        self.exploration_target_xy = None
        if failed_target is not None:
            self.mark_exploration_target_failed(failed_target, current_time=current_time)

    # ------------------------------------------------------------------ exploration failure memory

    def mark_exploration_target_failed(self, target, *, current_time: float) -> None:
        """Remember that planning to *target* failed at *current_time*.

        Also resets the retry-cooldown clock (see exploration_retry_on_cooldown()).
        The list is pruned to _FAILED_TARGET_RETENTION_S so it cannot grow
        unbounded over a long-running simulation.
        """
        self.failed_exploration_targets.append((_as_point(target), float(current_time)))
        self.note_exploration_retry_attempt(current_time)
        cutoff = float(current_time) - self._FAILED_TARGET_RETENTION_S
        self.failed_exploration_targets = [
            (point, failed_at) for point, failed_at in self.failed_exploration_targets
            if failed_at >= cutoff
        ]

    def note_exploration_retry_attempt(self, current_time: float) -> None:
        """Reset the retry-cooldown clock without adding a blacklist entry.

        Used when a re-selection attempt itself finds no candidate, so the
        agent backs off before trying frontier detection again instead of
        re-running it every single tick.
        """
        self.last_exploration_failure_time = float(current_time)

    def recently_failed_exploration_targets(
        self,
        *,
        current_time: float,
        cooldown: float,
    ) -> list[tuple[float, float]]:
        """Targets that failed to plan within the last *cooldown* seconds."""
        return [
            point
            for point, failed_at in self.failed_exploration_targets
            if (float(current_time) - failed_at) <= float(cooldown)
        ]

    def exploration_retry_on_cooldown(self, *, current_time: float, cooldown: float) -> bool:
        """True when a recent failure/empty-retry means we should keep holding."""
        return (float(current_time) - self.last_exploration_failure_time) < float(cooldown)

    # ------------------------------------------------------------------ safety replan throttle

    def safety_replan_allowed(
        self,
        *,
        reason: str,
        target: tuple[float, float] | None,
        current_time: float,
        cooldown: float,
    ) -> bool:
        """Throttle identical REPLAN_FOR_SAFETY requests for this robot.

        Mirrors engine.multi_safety_replan_allowed(): a (reason, rounded
        target) signature identifies "the same blocked segment/target as
        last time". Returning False means the caller should brake and hold
        this frame instead of launching another planner request for a
        situation it just tried and failed to resolve.
        """
        target_key = None
        if target is not None:
            target_key = (round(float(target[0]), 2), round(float(target[1]), 2))
        signature = (str(reason), target_key)
        elapsed = float(current_time) - float(self.last_safety_replan_time)
        same_signature = signature == self.last_safety_replan_signature
        if same_signature and elapsed < float(cooldown):
            return False
        self.last_safety_replan_time = float(current_time)
        self.last_safety_replan_signature = signature
        return True

    def assign_path(
        self,
        *,
        target: tuple[float, float],
        waypoints,
        planner_reason: str = "",
    ) -> None:
        """Accept a newly computed path and start tracking it."""
        self.active_path_goal_xy = _as_point(target)
        self.active_path_mode = self.planner_mode
        self.waypoints.set_waypoints(waypoints)
        self.last_plan_reason = planner_reason
        self.status = "moving" if self.waypoints.has_path() else "finished"

    def clear_if_planning_failed(self, reason: str) -> None:
        self.last_plan_reason = reason
        if not self.waypoints.has_path():
            self.status = "blocked"

    def needs_replan_for_target(
        self,
        target: tuple[float, float] | None,
        *,
        tolerance: float,
    ) -> bool:
        """True when a new path must be computed for *target*."""
        if target is None:
            return False
        if self.active_path_mode != self.planner_mode:
            return True
        if self.active_path_goal_xy is None:
            return True
        if distance(self.active_path_goal_xy, target) > max(float(tolerance), 0.0):
            return True
        if not self.waypoints.has_path():
            return True
        if self.status in {"blocked"}:
            return True
        return False

    # ------------------------------------------------------------------ active target

    def active_target(self) -> tuple[float, float] | None:
        """The waypoint the robot is currently tracking, or None."""
        wp = self.waypoints.active_waypoint()
        if wp is None:
            return None
        return (float(wp[0]), float(wp[1]))

    def distance_to_active_target(self) -> float:
        """Euclidean distance from the robot position to the active waypoint."""
        target = self.active_target()
        if target is None:
            return float("inf")
        return distance(self.position, target)

    def distance_to_active_path_goal(self) -> float:
        """Euclidean distance from the robot position to the END of the current path.

        Use this for prefetch threshold checks: prefetching should start when
        the robot is close to the final frontier of its route, not just to any
        intermediate waypoint.
        """
        if self.active_path_goal_xy is None:
            return float("inf")
        return distance(self.position, self.active_path_goal_xy)

    def should_prefetch(self, threshold_distance: float) -> bool:
        """True when the robot is close enough to start computing the next path."""
        if self.pending_path is not None:
            return False  # already have a prefetch in flight
        return self.distance_to_active_target() <= float(threshold_distance)

    # ------------------------------------------------------------------ pending path

    def accept_pending_path(self) -> list[tuple[float, float]] | None:
        """
        Switch to the prefetched path.

        Returns the waypoint list that was accepted, or None if there was no
        pending path.  The engine must call set_robot_goal_or_waypoints() with
        the returned list to push the change into the Robot object.
        """
        if self.pending_path is None:
            return None
        waypoints = list(self.pending_path)
        self.waypoints.set_waypoints(waypoints)
        if self.pending_target_xy is not None:
            self.active_path_goal_xy = self.pending_target_xy
            self.exploration_target_xy = self.pending_target_xy
        self.pending_path = None
        self.pending_target_xy = None
        self.status = "moving"
        self.prefetch_success_count += 1
        return waypoints

    def reject_pending_path(self, reason: str = "") -> None:
        """Discard the prefetched path without touching the current active route."""
        self.pending_path = None
        self.pending_target_xy = None
        self.prefetch_fail_count += 1
        if reason:
            self.last_plan_reason = f"prefetch rejected: {reason}"

    # ------------------------------------------------------------------ waypoint progress (compat)

    def update_waypoint_progress(self, tolerance: float) -> bool:
        """Advance through waypoints as the robot moves (used by engine compat code)."""
        advanced = self.waypoints.advance_if_reached(self.position, tolerance)
        if self.waypoints.is_finished():
            self.status = "finished"
        return advanced

    def forward_point(self, distance_m: float) -> tuple[float, float]:
        """World point at *distance_m* ahead of the current heading."""
        d = max(float(distance_m), 0.0)
        return (
            self.position[0] + d * math.cos(self.heading),
            self.position[1] + d * math.sin(self.heading),
        )

    # ------------------------------------------------------------------ STEP

    def step(
        self,
        observation: "RobotObservation",
        planner_services: "PlannerServices",
        dt: float,
    ) -> "NavigationDecision":
        """
        Compute one navigation decision for the current simulation frame.

        The engine should:
          1. Build a RobotObservation (via engine.build_observation()).
          2. Call this method.
          3. Apply the returned NavigationDecision (via engine.apply_navigation_decision()).

        This method syncs position/heading from the observation, then
        dispatches to _step_goal_seeking or ExplorationBehavior.update()
        depending on the current planner_mode.
        """
        from robotics_sim.navigation.navigation_decision import (
            follow,
            hold,
            replan_for_safety,
            request_plan,
        )
        from robotics_sim.simulation.navigation_modes import is_goal_seeking_planner

        # Sync pose (engine is the ground truth for robot physics).
        self.set_position(observation.robot_xy)
        self.set_heading(observation.robot_heading)

        # Safety overrides all other decisions.
        if observation.active_segment_blocked or observation.predicted_collision:
            reason = (
                "predicted collision"
                if observation.predicted_collision
                else "active segment blocked"
            )
            self.safety_replan_count += 1
            self.last_replan_time = observation.current_time
            return replan_for_safety(self.desired_target_from_mode(), reason=reason)

        # Mode dispatch.
        if is_goal_seeking_planner(self.planner_mode):
            return self._step_goal_seeking(observation)

        return self.behavior.update(self, observation, planner_services)

    def _step_goal_seeking(
        self,
        observation: "RobotObservation",
    ) -> "NavigationDecision":
        """
        Goal-seeking mode: drive toward final_goal_xy.

        The robot follows its current route if one exists, otherwise requests a
        fresh plan to final_goal_xy.  The exploration target is irrelevant here.
        """
        from robotics_sim.navigation.navigation_decision import follow, hold, request_plan

        target = self.final_goal_xy
        if target is None:
            return hold(reason="no final goal set in Goal seeking mode")

        active = self.active_target()
        if active is not None:
            return follow(active, reason="following route to final goal")

        return request_plan(target, reason="no active path; requesting route to final goal")
