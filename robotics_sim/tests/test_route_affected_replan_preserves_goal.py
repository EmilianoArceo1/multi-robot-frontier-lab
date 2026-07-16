"""
Regression tests for route-repair replans (route_affected / REPLAN_FOR_SAFETY)
silently switching to a brand-new frontier instead of repairing the route
to the goal the robot was already navigating to.

Manual Office.sim telemetry:

    path_goal=(7.25,3.75)
    ...
    [MAP] route_affected=yes
    [FRONTIER] selected=(1.25,-0.75)
    New obstacle affects current route. Replanning...
    [ROUTE ok] goal=(1.25,-0.75)

The robot had an active route to (7.25,3.75). A newly-mapped obstacle
affected that route, which should mean "find a new path to the same
destination" -- instead the engine dropped (7.25,3.75) entirely and
started navigating to a completely different frontier, (1.25,-0.75),
selected fresh via frontier scoring.

Root cause: engine.replan_after_new_information() -- the single function
behind BOTH the route_affected branch in simulation_step() and the
REPLAN_FOR_SAFETY branch in apply_navigation_decision() -- called
request_route_async(reason) with no target_override. Without one,
request_route_async() falls back to build_planner_kwargs() ->
select_navigation_goal(), an independent frontier re-selection that knows
nothing about the route that was actually being repaired.

Contract:
    - REQUEST_PLAN / "frontier reached" -> MAY select a new frontier
      (ExplorationBehavior._pick_next_target() already handles this).
    - PREFETCH_NEXT_TARGET -> MAY select a future target.
    - route_affected / REPLAN_FOR_SAFETY -> MUST preserve the current
      active_path_goal_xy if one exists; only fall back to fresh target
      selection when there is truly nothing active to repair.

Fix: a small module-level helper, engine.current_route_repair_goal(agent)
(-> agent.active_path_goal_xy or agent.exploration_target_xy, or None),
and replan_after_new_information() now passes its result as
target_override to request_route_async() -- reusing the exact same
target_override mechanism already used for REQUEST_PLAN decisions (added a
few rounds ago), rather than inventing a new code path. When there is no
active goal to preserve, target_override is None and the existing
select_navigation_goal() fallback runs unchanged.

These tests exercise engine.replan_after_new_information() and
apply_navigation_decision()'s REPLAN_FOR_SAFETY handling via a minimal
duck-typed engine fake, matching the pattern in
test_already_reached_runtime_target.py / test_route_endpoint_validation.py.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.simulation.engine import SimulationControllerMixin, current_route_repair_goal
from robotics_sim.simulation.telemetry import TelemetryLogger


REPAIR_GOAL = (7.25, 3.75)  # agent.active_path_goal_xy -- must be preserved
NEW_FRONTIER = (1.25, -0.75)  # what select_navigation_goal() would (wrongly) pick instead


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine(*, position=(6.0, 3.0), goal_tolerance=0.25) -> SimpleNamespace:
    robot = _FakeRobot(x=position[0], y=position[1])
    agent = RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")

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
        safety_replan_count=0,
        simulation_time=20.0,
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
    fake.set_robot_goal_or_waypoints = lambda robot_obj, waypoints: robot_obj.set_waypoints(
        waypoints or [(robot_obj.x, robot_obj.y)]
    )
    fake.safety_replan_cooldown_seconds = lambda: 0.5
    fake.brake_control_for_collision = lambda: None

    def _spy_select_navigation_goal(start_xy):
        fake.select_navigation_goal_calls.append(start_xy)
        return (NEW_FRONTIER, "fake frontier selection")

    fake.select_navigation_goal = _spy_select_navigation_goal

    def _spy_build_planner_kwargs_for_goal(start_xy, goal_xy, *, robot=None):
        fake.build_planner_kwargs_for_goal_calls.append((start_xy, goal_xy))
        return dict(__hold__=True, __hold_reason__="test stub: planner did not find a route")

    fake.build_planner_kwargs_for_goal = _spy_build_planner_kwargs_for_goal

    def _spy_build_planner_kwargs(start_xy):
        goal_xy, _reason = fake.select_navigation_goal(start_xy)
        return dict(__hold__=True, __hold_reason__="test stub: no real planner")

    fake.build_planner_kwargs = _spy_build_planner_kwargs
    fake.clean_waypoints_for_current_start = lambda waypoints: [tuple(p) for p in waypoints]
    fake.collision_checker = CollisionChecker()  # no obstacle points -> never blocks
    fake.safety_radius = lambda: 0.2

    for name in (
        "apply_navigation_decision",
        "apply_route_result",
        "request_route_async",
        "replan_after_new_information",
        "_invalidate_prefetch_request",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    return fake


def _give_agent_active_route(fake, *, target):
    fake.agent.assign_path(target=target, waypoints=[(6.5, 3.25), target], planner_reason="initial route")


def _replan_for_safety_decision(target, reason="predicted collision"):
    return SimpleNamespace(kind="REPLAN_FOR_SAFETY", reason=reason, target=target, brake=True)


# ---------------------------------------------------------------------------
# current_route_repair_goal() direct tests.
# ---------------------------------------------------------------------------


def test_current_route_repair_goal_prefers_active_path_goal():
    agent = RobotAgent(robot_id=0, position=(0.0, 0.0), planner_mode="FoV-aware directional frontier")
    agent.assign_path(target=REPAIR_GOAL, waypoints=[REPAIR_GOAL], planner_reason="r")
    agent.exploration_target_xy = NEW_FRONTIER  # deliberately different/stale

    assert current_route_repair_goal(agent) == REPAIR_GOAL


def test_current_route_repair_goal_falls_back_to_exploration_target():
    agent = RobotAgent(robot_id=0, position=(0.0, 0.0), planner_mode="FoV-aware directional frontier")
    agent.exploration_target_xy = NEW_FRONTIER
    assert agent.active_path_goal_xy is None

    assert current_route_repair_goal(agent) == NEW_FRONTIER


def test_current_route_repair_goal_is_none_when_nothing_active():
    agent = RobotAgent(robot_id=0, position=(0.0, 0.0), planner_mode="FoV-aware directional frontier")
    assert current_route_repair_goal(agent) is None
    assert current_route_repair_goal(None) is None


# ---------------------------------------------------------------------------
# 1. route_affected replan must repair towards the current path_goal, not
#    a freshly-selected frontier.
# ---------------------------------------------------------------------------


def test_route_affected_replan_uses_current_path_goal_not_new_frontier():
    fake = _build_fake_engine()
    _give_agent_active_route(fake, target=REPAIR_GOAL)

    fake.replan_after_new_information("New obstacle affects current route.")

    assert fake.select_navigation_goal_calls == [], (
        "select_navigation_goal() must not be used to replace the target during route repair"
    )
    assert fake.build_planner_kwargs_for_goal_calls == [((6.0, 3.0), REPAIR_GOAL)], (
        "the repair route must be requested toward the existing active_path_goal_xy, not a new frontier"
    )


# ---------------------------------------------------------------------------
# 2. Same contract for a REPLAN_FOR_SAFETY decision (predicted collision).
# ---------------------------------------------------------------------------


def test_predicted_collision_replan_uses_current_path_goal_not_new_frontier():
    fake = _build_fake_engine()
    _give_agent_active_route(fake, target=REPAIR_GOAL)

    decision = _replan_for_safety_decision(REPAIR_GOAL)
    should_brake = SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    assert should_brake is True
    assert fake.select_navigation_goal_calls == []
    assert fake.build_planner_kwargs_for_goal_calls == [((6.0, 3.0), REPAIR_GOAL)]


# ---------------------------------------------------------------------------
# 3. With nothing active to repair, existing fallback target selection is
#    still allowed.
# ---------------------------------------------------------------------------


def test_replan_falls_back_to_target_selection_only_when_no_active_path_goal():
    fake = _build_fake_engine()
    assert fake.agent.active_path_goal_xy is None
    assert fake.agent.exploration_target_xy is None

    fake.replan_after_new_information("New obstacle affects current route.")

    assert fake.select_navigation_goal_calls == [(6.0, 3.0)], (
        "with nothing active to repair, falling back to normal target selection must still work"
    )
    assert fake.build_planner_kwargs_for_goal_calls == []


# ---------------------------------------------------------------------------
# 4. A repair that fails outright must mark the CURRENT path_goal as
#    failed, not some newly-selected frontier.
# ---------------------------------------------------------------------------


def test_route_affected_replan_failure_marks_current_path_goal_failed():
    fake = _build_fake_engine()
    _give_agent_active_route(fake, target=REPAIR_GOAL)

    # build_planner_kwargs_for_goal is stubbed to simulate the planner
    # failing outright (__hold__), which request_route_async() turns into
    # its own internal apply_route_result(False, ...) call.
    fake.replan_after_new_information("New obstacle affects current route.")

    failed = fake.agent.recently_failed_exploration_targets(current_time=fake.simulation_time, cooldown=999.0)
    assert any(
        math.hypot(REPAIR_GOAL[0] - p[0], REPAIR_GOAL[1] - p[1]) < 1e-6 for p in failed
    ), "the repair goal itself must be recorded as the failed target"
    assert not any(
        math.hypot(NEW_FRONTIER[0] - p[0], NEW_FRONTIER[1] - p[1]) < 1e-6 for p in failed
    ), "a frontier that was never actually attempted must not be marked failed"
    assert fake.agent.active_path_goal_xy is None


# ---------------------------------------------------------------------------
# 5. Endpoint validation still applies to the preserved current goal: a
#    route that "succeeds" without reaching REPAIR_GOAL must be rejected,
#    not silently accepted, and REPAIR_GOAL (not some other point) is what
#    gets marked failed.
# ---------------------------------------------------------------------------


def test_route_affected_replan_applies_endpoint_validation_to_current_goal():
    fake = _build_fake_engine()
    _give_agent_active_route(fake, target=REPAIR_GOAL)

    # Simulate what replan_after_new_information()'s target_override
    # handling already establishes before a real planner worker would run:
    # both current_exploration_target and agent.exploration_target_xy set
    # to the goal being repaired (see request_route_async()'s
    # target_override branch).
    fake.current_exploration_target = REPAIR_GOAL
    fake.agent.exploration_target_xy = REPAIR_GOAL

    endpoint_missing_goal = (2.0, 2.0)  # nowhere near REPAIR_GOAL
    fake.apply_route_result(
        True, "path found with A*; goal adjusted to nearest traversable cell", [endpoint_missing_goal]
    )

    assert fake.agent.active_path_goal_xy is None, "a route that misses the repair goal must not be accepted"
    assert fake.agent.active_path_goal_xy != endpoint_missing_goal
    failed = fake.agent.recently_failed_exploration_targets(current_time=fake.simulation_time, cooldown=999.0)
    assert any(math.hypot(REPAIR_GOAL[0] - p[0], REPAIR_GOAL[1] - p[1]) < 1e-6 for p in failed)
