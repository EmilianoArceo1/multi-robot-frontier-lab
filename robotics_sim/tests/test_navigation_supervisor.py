"""
Tests for NavigationSupervisor, the small centralized-invariant layer
introduced on refactor/simple-navigation-control.

Context: repeated bugs in single-robot exploration (route-less
REPLAN_FOR_SAFETY, planning to an already-reached target, routes accepted
without reaching their goal, the engine re-deriving a target
ExplorationBehavior already chose) were each fixed in a different place
across RobotAgent.step(), ExplorationBehavior.update(), and three separate
spots in engine.py. NavigationSupervisor centralizes the underlying
invariants into one small, stateless, directly-testable class:

    - normalize_decision(): REPLAN_FOR_SAFETY without an active route, and
      REQUEST_PLAN to an already-reached target, both become HOLD.
    - should_request_route(): the "already reached" check normalize_decision
      uses internally, also independently testable/reusable.
    - validate_route_endpoint(): a route/prefetch's final waypoint must
      actually reach the goal it claims to satisfy.

Tests 1-5 exercise NavigationSupervisor directly (no engine, no Qt). Tests
6-7 exercise engine.apply_navigation_decision()'s wiring through a minimal
duck-typed engine fake, the same pattern used in
test_already_reached_runtime_target.py and test_route_endpoint_validation.py.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.navigation_decision import (
    follow,
    replan_for_safety,
    request_plan,
)
from robotics_sim.navigation.navigation_supervisor import NavigationSupervisor
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.telemetry import TelemetryLogger


ALREADY_REACHED_TARGET = (-4.25, -1.75)
ROBOT_NEAR_TARGET = (-4.24, -1.76)  # ~0.014 m from ALREADY_REACHED_TARGET
FAR_TARGET = (9.0, 9.0)


def _make_agent(position=(0.0, 0.0)) -> RobotAgent:
    return RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")


def _make_observation(**overrides) -> RobotObservation:
    defaults = dict(
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        robot_radius=0.2,
        belief_map=None,
        planning_grid=None,
        mapped_obstacle_points=[],
        dynamic_obstacles=[],
        active_segment_blocked=False,
        predicted_collision=False,
        current_time=0.0,
        grid_resolution=0.5,
        goal_tolerance=0.25,
        sensor_range=2.5,
        final_goal_xy=None,
    )
    defaults.update(overrides)
    return RobotObservation(**defaults)


# ---------------------------------------------------------------------------
# 1. REPLAN_FOR_SAFETY with no active route normalizes to HOLD.
# ---------------------------------------------------------------------------


def test_supervisor_blocks_safety_replan_without_active_route():
    agent = _make_agent(position=(1.0, 1.0))
    assert agent.active_path_goal_xy is None
    assert agent.active_target() is None

    decision = replan_for_safety(target=None, reason="predicted collision")
    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25)

    normalized = NavigationSupervisor.normalize_decision(
        agent, observation, decision, goal_tolerance=0.25, map_signature=0
    )

    assert normalized.kind == "HOLD"
    # A HOLD decision carries no target and must never be treated as a
    # route request by the caller.
    assert NavigationSupervisor.should_request_route(
        agent.position, normalized.target, 0.25
    ) is False


# ---------------------------------------------------------------------------
# 2. REQUEST_PLAN to an already-reached target is blocked.
# ---------------------------------------------------------------------------


def test_supervisor_blocks_already_reached_target():
    agent = _make_agent(position=ROBOT_NEAR_TARGET)
    decision = request_plan(ALREADY_REACHED_TARGET, reason="frontier reached; requesting next frontier")
    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25)

    normalized = NavigationSupervisor.normalize_decision(
        agent, observation, decision, goal_tolerance=0.25, map_signature=0
    )

    assert normalized.kind == "HOLD"
    assert NavigationSupervisor.should_request_route(
        ROBOT_NEAR_TARGET, ALREADY_REACHED_TARGET, 0.25
    ) is False


# ---------------------------------------------------------------------------
# 3. REQUEST_PLAN to a genuinely far target is allowed through unchanged.
# ---------------------------------------------------------------------------


def test_supervisor_allows_valid_request_plan_target():
    agent = _make_agent(position=(0.0, 0.0))
    decision = request_plan(FAR_TARGET, reason="no active path; requesting initial frontier plan")
    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25)

    normalized = NavigationSupervisor.normalize_decision(
        agent, observation, decision, goal_tolerance=0.25, map_signature=0
    )

    assert normalized.kind == "REQUEST_PLAN"
    assert normalized.target == FAR_TARGET
    assert NavigationSupervisor.should_request_route((0.0, 0.0), FAR_TARGET, 0.25) is True


# ---------------------------------------------------------------------------
# 4/5. Route endpoint validation.
# ---------------------------------------------------------------------------


def test_supervisor_rejects_route_endpoint_mismatch():
    assert NavigationSupervisor.validate_route_endpoint(
        [(3.25, -4.75)], (2.75, -4.75), tolerance=0.25
    ) is False


def test_supervisor_accepts_route_endpoint_match():
    assert NavigationSupervisor.validate_route_endpoint(
        [(2.80, -4.70)], (2.75, -4.75), tolerance=0.25
    ) is True


# ---------------------------------------------------------------------------
# Engine wiring: minimal duck-typed engine fake, matching the pattern in
# test_already_reached_runtime_target.py / test_route_endpoint_validation.py.
# ---------------------------------------------------------------------------


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine(*, position=(0.0, 0.0), goal_tolerance=0.25) -> SimpleNamespace:
    robot = _FakeRobot(x=position[0], y=position[1])
    agent = _make_agent(position=position)

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        config=SimpleNamespace(
            planner_type="A*",
            path_simplifier="Direction changes",
            exploration_planner="FoV-aware directional frontier",
            goal_tolerance=goal_tolerance,
            grid_resolution=0.5,
        ),
        mapped_obstacle_points=[],
        current_exploration_target=None,
        route_request_count=0,
        route_request_id=0,
        route_result_count=0,
        planning_in_progress=False,
        active_planner_workers={},
        simulation_time=0.0,
        console_logs=[],
        exploration_targets=[],
        select_navigation_goal_calls=[],
        build_planner_kwargs_for_goal_calls=[],
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.canvas = SimpleNamespace(
        set_planned_path=lambda path: None,
        set_exploration_target=lambda target: fake.exploration_targets.append(target),
        set_status=lambda message: None,
        set_last_control=lambda control: None,
    )
    fake.is_exploration_mode = lambda: True
    fake.runtime_agent = lambda robot_index=None: fake.agent
    fake.safety_replan_cooldown_seconds = lambda: 0.5
    fake.replan_after_new_information = lambda reason: None
    fake.set_robot_goal_or_waypoints = lambda robot_obj, waypoints: robot_obj.set_waypoints(
        waypoints or [(robot_obj.x, robot_obj.y)]
    )

    def _spy_select_navigation_goal(start_xy):
        fake.select_navigation_goal_calls.append(start_xy)
        return (None, "no target")

    fake.select_navigation_goal = _spy_select_navigation_goal

    def _spy_build_planner_kwargs_for_goal(start_xy, goal_xy, *, robot=None):
        fake.build_planner_kwargs_for_goal_calls.append((start_xy, goal_xy))
        return dict(__hold__=True, __hold_reason__="test stub: no real planner")

    fake.build_planner_kwargs_for_goal = _spy_build_planner_kwargs_for_goal

    for name in (
        "apply_navigation_decision",
        "apply_route_result",
        "request_route_async",
        "_invalidate_prefetch_request",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    return fake


# ---------------------------------------------------------------------------
# 6. The engine must not log/apply an invalid REPLAN_FOR_SAFETY; it holds.
# ---------------------------------------------------------------------------


def test_engine_uses_supervisor_before_navigation_dispatch():
    fake = _build_fake_engine(position=(1.0, 1.0))
    decision = replan_for_safety(target=None, reason="predicted collision")

    should_brake = SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    assert should_brake is False, "a normalized HOLD must not use REPLAN_FOR_SAFETY's brake contract"
    assert fake.robot.waypoints == [(1.0, 1.0)], "the robot must hold at its current position"
    assert not any("REPLAN_FOR_SAFETY" in str(line) for line in fake.console_logs), (
        "an invalid route-less safety replan must never be logged/applied as REPLAN_FOR_SAFETY"
    )


# ---------------------------------------------------------------------------
# 7. The engine must not re-derive a target when the decision already has one.
# ---------------------------------------------------------------------------


def test_engine_does_not_rederive_target_when_decision_target_is_valid():
    fake = _build_fake_engine(position=(0.0, 0.0))
    decision = request_plan(FAR_TARGET, reason="frontier reached; requesting next frontier", force_new_target=True)

    SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    assert fake.select_navigation_goal_calls == [], (
        "select_navigation_goal() must not be called again when the decision already carries a valid target"
    )
    assert fake.build_planner_kwargs_for_goal_calls == [((0.0, 0.0), FAR_TARGET)], (
        "the route request must plan directly to the decision's own target"
    )
