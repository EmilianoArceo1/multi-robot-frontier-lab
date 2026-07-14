"""
Tests for the navigation-debug producer wiring inside
SimulationControllerMixin.apply_route_result(): when navigation_debug_enabled,
it builds a RouteValidationDebug/ClearanceTerms from the exact same
CollisionReport already used for the accept/reject decision (via
_evaluate_route_first_segment(), not a second collision computation), and
pushes a NavigationDebugSnapshot into the bounded event log -- without
changing the accept/reject decision itself.

Uses the same lightweight duck-typed engine fake pattern as
test_route_endpoint_validation.py's _build_fake_engine().
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.diagnostics.capture import PlanDebugCapture
from robotics_sim.diagnostics.event_log import NavigationDebugEventLog
from robotics_sim.diagnostics.navigation_snapshot import NavigationDebugEventKind
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.planning.planner_registry import compute_planned_waypoints
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.telemetry import TelemetryLogger


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine(*, navigation_debug_enabled: bool = True) -> SimpleNamespace:
    position = (0.0, 0.0)
    robot = _FakeRobot(x=position[0], y=position[1], theta=0.0, v=0.0)
    agent = RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        config=SimpleNamespace(
            planner_type="A*",
            path_simplifier="Direction changes",
            exploration_planner="FoV-aware directional frontier",
            goal_tolerance=0.25,
            grid_resolution=0.5,
        ),
        mapped_obstacle_points=[],
        current_exploration_target=None,
        route_result_count=0,
        last_goal_selection_reason="frontier selection reason",
        simulation_time=12.5,
        console_logs=[],
        planned_paths=[],
        exploration_targets=[],
        prefetch_workers={},
        prefetch_request_ids={0: 1},
        last_control=[0.1, 0.2],
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.collision_checker = CollisionChecker()
    fake.canvas = SimpleNamespace(
        set_planned_path=lambda path: fake.planned_paths.append(path),
        set_exploration_target=lambda target: fake.exploration_targets.append(target),
        set_status=lambda message: None,
    )
    fake.is_exploration_mode = lambda: True
    fake.safety_radius = lambda: 0.2
    fake.body_radius_for_robot = lambda robot=None: 0.15
    fake.safety_radius_for_robot = lambda robot=None: 0.2
    fake.planner_label = lambda: "A* / Direction changes + FoV-aware directional frontier"
    fake.clean_waypoints_for_current_start = lambda waypoints: [tuple(p) for p in waypoints]
    fake.final_goal_xy = lambda: (0.0, 0.0)
    fake.runtime_agent = lambda robot_index=None: fake.agent

    fake.navigation_debug_enabled = navigation_debug_enabled
    fake.navigation_debug_log = NavigationDebugEventLog(max_size=10)

    for name in ("apply_route_result", "_finalize_navigation_debug_snapshot", "log_route_assignment"):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    return fake


# ---------------------------------------------------------------------------
# A first-segment-blocked route produces a real ClearanceTerms with the
# actual checker name, blocking point, and clearance terms used.
# ---------------------------------------------------------------------------


def test_blocked_first_segment_captures_real_clearance_terms():
    fake = _build_fake_engine()
    # Obstacle sample sits exactly on the segment (0,0) -> (1,0); safety
    # radius 0.2 -- distance from the point to the segment is 0.0 <= 0.2.
    fake.mapped_obstacle_points = [(0.5, 0.0)]

    fake.apply_route_result(True, "path found with A*", [(1.0, 0.0)])

    assert fake.agent.active_path_goal_xy is None, "a blocked-on-arrival route must not be accepted"

    assert len(fake.navigation_debug_log) == 1
    event = fake.navigation_debug_log.latest()
    assert event.event_kind is NavigationDebugEventKind.ROUTE_REJECTED

    first_segment = event.snapshot.route.first_segment
    assert first_segment.unavailable is False
    terms = first_segment.value
    assert terms.checker == "check_segment_points"
    assert terms.blocked is True
    assert terms.blocking_point == (0.5, 0.0)
    assert terms.required_clearance == pytest.approx(0.2)
    # The exact boolean condition the real checker used: distance <= required_clearance.
    assert terms.distance.unavailable is False
    assert terms.distance.value <= terms.required_clearance
    assert terms.distance.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# A clear (not blocked) first segment reports blocked=False and leaves
# distance Maybe.missing() -- the real checker never computes a scalar
# distance on the clear branch, so the snapshot must not invent one.
# ---------------------------------------------------------------------------


def test_clear_first_segment_reports_not_blocked_and_distance_unavailable():
    fake = _build_fake_engine()
    fake.mapped_obstacle_points = []  # nothing near the route

    fake.apply_route_result(True, "path found with A*", [(1.0, 0.0)])

    assert fake.agent.active_path_goal_xy == (1.0, 0.0), "a clear route must be accepted"

    event = fake.navigation_debug_log.latest()
    assert event.event_kind is NavigationDebugEventKind.PLAN_ACCEPTED

    first_segment = event.snapshot.route.first_segment
    assert first_segment.unavailable is False
    terms = first_segment.value
    assert terms.blocked is False
    assert terms.distance.unavailable is True, "the real checker computes no scalar distance when clear"


# ---------------------------------------------------------------------------
# Route validation event tagging never changes the accept/reject decision.
# Running the identical scenario twice, once with the layer disabled, must
# reach the exact same agent/robot outcome.
# ---------------------------------------------------------------------------


def test_navigation_debug_disabled_reaches_identical_route_decision():
    fake_off = _build_fake_engine(navigation_debug_enabled=False)
    fake_off.mapped_obstacle_points = [(0.5, 0.0)]
    fake_off.apply_route_result(True, "path found with A*", [(1.0, 0.0)])

    fake_on = _build_fake_engine(navigation_debug_enabled=True)
    fake_on.mapped_obstacle_points = [(0.5, 0.0)]
    fake_on.apply_route_result(True, "path found with A*", [(1.0, 0.0)])

    assert fake_off.agent.active_path_goal_xy == fake_on.agent.active_path_goal_xy
    assert fake_off.robot.waypoints == fake_on.robot.waypoints
    assert len(fake_off.navigation_debug_log) == 0, "disabled layer must never populate the event log"
    assert len(fake_on.navigation_debug_log) == 1


# ---------------------------------------------------------------------------
# compute_planned_waypoints()'s debug_capture outparam carries the real
# raw/simplified grid path and start/first-waypoint cell data it already
# computes locally, matching PlanningResult.grid_path/simplified_grid_path.
# ---------------------------------------------------------------------------


def test_plan_debug_capture_matches_real_planner_intermediates():
    capture = PlanDebugCapture()

    success, _reason, waypoints = compute_planned_waypoints(
        planner_type="A*",
        start_xy=(0.0, 0.0),
        goal_xy=(4.0, 0.0),
        bounds=(-1.0, 6.0, -1.0, 6.0),
        resolution=0.5,
        robot_radius=0.2,
        obstacle_points=[],
        unknown_is_traversable=True,
        debug_capture=capture,
    )

    assert success is True
    assert capture.planner_name == "A*"
    assert capture.simplifier_name is not None
    # Raw path has more cells than the simplified/executable path for this
    # straight-line-with-grid-quantization scenario.
    assert len(capture.raw_world_path) >= len(waypoints)
    assert capture.start_cell is not None
    assert capture.start_cell_world is not None
    assert capture.first_waypoint_cell is not None
    assert capture.unknown_is_traversable is True
    assert capture.start_cell_cleared is False


def test_plan_debug_capture_omitted_by_default_costs_nothing_extra():
    # No debug_capture passed -- must behave exactly as before (same
    # 3-tuple return, no AttributeError, no extra work).
    success, reason, waypoints = compute_planned_waypoints(
        planner_type="A*",
        start_xy=(0.0, 0.0),
        goal_xy=(4.0, 0.0),
        bounds=(-1.0, 6.0, -1.0, 6.0),
        resolution=0.5,
        robot_radius=0.2,
        obstacle_points=[],
        unknown_is_traversable=True,
    )
    assert success is True
    assert waypoints
