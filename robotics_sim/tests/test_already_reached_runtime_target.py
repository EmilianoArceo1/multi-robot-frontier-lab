"""
Regression tests for the runtime re-selecting an already-reached
exploration target.

Manual Office.sim telemetry, after the endpoint-validation fix landed:

    [FRONTIER] selected=(-4.25,-1.75)
    frontier reached; requesting next frontier
    [ROUTE ok] start=(-4.24,-1.76) goal=(-4.25,-1.75) wp=1 length=0.02
    [FRONTIER] selected=(-4.25,-1.75)
    frontier reached; requesting next frontier
    [ROUTE ok] start=(-4.24,-1.76) goal=(-4.25,-1.75) wp=1 length=0.02
    ... (repeats, route length shrinking towards 0)

Root cause: ExplorationBehavior._pick_next_target() already rejects a
candidate within goal_tolerance of the robot (fixed two rounds ago), but
engine.apply_navigation_decision()'s single-robot REQUEST_PLAN handler
never actually uses decision.target -- it calls request_route_async(),
which calls build_planner_kwargs() -> engine.select_navigation_goal(), a
SECOND, INDEPENDENT call into the exploration planner. That call only
excluded this agent's recently-failed targets; it never excluded the
robot's own current position or the just-completed active_path_goal_xy.
So even when ExplorationBehavior correctly rejected an already-reached
candidate, this second, independent selection could still re-propose it,
and nothing downstream re-checked the result before launching a route
request and accepting it as "[ROUTE ok]".

Fix (three layers, each closing the same gap from a different angle):
    1. engine.select_navigation_goal() now excludes the robot's own current
       position and the just-completed active_path_goal_xy (mirroring
       ExplorationBehavior._pick_next_target()), plus a hard post-hoc
       "reject if within goal_tolerance of the robot" check on whatever
       the exploration planner returns -- the same belt-and-suspenders
       pattern already used in _pick_next_target().
    2. engine.request_route_async() gained an optional target_override
       parameter: when given (exploration mode only), it skips
       select_navigation_goal()'s independent re-derivation entirely and
       plans directly to the already-chosen, already-validated target.
       apply_navigation_decision()'s REQUEST_PLAN handler now passes
       decision.target as target_override, so ExplorationBehavior's
       selection is actually honored instead of being silently discarded
       and re-derived.
    3. apply_navigation_decision() also adds a final runtime boundary
       guard, immediately before any route request is launched: if
       decision.target is within goal_tolerance of the robot's current
       position, no route request is launched at all -- the decision is
       converted to a HOLD-equivalent instead. This is the last line of
       defense regardless of which upstream path produced the target.

These tests exercise select_navigation_goal() with a real BeliefMap (test
2, no Qt/canvas) and apply_navigation_decision()/request_route_async() via
a minimal duck-typed engine fake (tests 1, 3, 4), matching the pattern
already used in test_multi_robot_route_validation.py and
test_route_endpoint_validation.py.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation import engine as engine_module
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.telemetry import TelemetryLogger


ALREADY_REACHED_TARGET = (-4.25, -1.75)
ROBOT_NEAR_TARGET = (-4.24, -1.76)  # ~0.014 m from ALREADY_REACHED_TARGET
FAR_TARGET = (9.0, 9.0)


# ---------------------------------------------------------------------------
# Fixture shared by tests 1, 3, 4: a spy-friendly duck-typed engine.
# ---------------------------------------------------------------------------


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine(*, position=ROBOT_NEAR_TARGET, goal_tolerance=0.25) -> SimpleNamespace:
    robot = _FakeRobot(x=position[0], y=position[1])
    agent = RobotAgent(
        robot_id=0,
        position=position,
        planner_mode="FoV-aware directional frontier",
    )

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
        request_route_async_calls=[],
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

    # select_navigation_goal(): spy, returns whatever the test configures.
    fake._select_navigation_goal_result = (None, "no target")

    def _spy_select_navigation_goal(start_xy):
        fake.select_navigation_goal_calls.append(start_xy)
        return fake._select_navigation_goal_result

    fake.select_navigation_goal = _spy_select_navigation_goal

    # build_planner_kwargs_for_goal(): spy, returns a minimal, __hold__ dict
    # so request_route_async() stops right after recording the call instead
    # of needing a real planning grid/thread pool.
    def _spy_build_planner_kwargs_for_goal(start_xy, goal_xy, *, robot=None):
        fake.build_planner_kwargs_for_goal_calls.append((start_xy, goal_xy))
        return dict(__hold__=True, __hold_reason__="test stub: no real planner")

    fake.build_planner_kwargs_for_goal = _spy_build_planner_kwargs_for_goal

    # build_planner_kwargs(): spy for the non-override path, same shape.
    def _spy_build_planner_kwargs(start_xy):
        goal_xy, _reason = fake.select_navigation_goal(start_xy)
        if goal_xy is None:
            return dict(__hold__=True, __hold_reason__="no target")
        return dict(__hold__=True, __hold_reason__="test stub: no real planner")

    fake.build_planner_kwargs = _spy_build_planner_kwargs

    for name in (
        "apply_navigation_decision",
        "apply_route_result",
        "request_route_async",
        "_invalidate_prefetch_request",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    # Wrap the real, bound request_route_async with a call-recording spy so
    # tests can assert whether it was invoked at all.
    _real_request_route_async = fake.request_route_async

    def _spy_request_route_async(reason, *, target_override=None):
        fake.request_route_async_calls.append((reason, target_override))
        return _real_request_route_async(reason, target_override=target_override)

    fake.request_route_async = _spy_request_route_async

    return fake


def _request_plan_decision(target, *, force_new_target=False, reason="frontier reached; requesting next frontier"):
    return SimpleNamespace(
        kind="REQUEST_PLAN", reason=reason, target=target, brake=False, force_new_target=force_new_target
    )


# ---------------------------------------------------------------------------
# 1. Engine boundary guard: no route request for a target within
#    goal_tolerance of the robot.
# ---------------------------------------------------------------------------


def test_engine_does_not_request_route_to_target_within_goal_tolerance():
    fake = _build_fake_engine(position=ROBOT_NEAR_TARGET, goal_tolerance=0.25)
    assert math.hypot(
        ALREADY_REACHED_TARGET[0] - ROBOT_NEAR_TARGET[0], ALREADY_REACHED_TARGET[1] - ROBOT_NEAR_TARGET[1]
    ) <= 0.25

    decision = _request_plan_decision(ALREADY_REACHED_TARGET, force_new_target=True)

    should_brake = SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    assert fake.request_route_async_calls == [], (
        "no route request may be launched for a target already within goal_tolerance"
    )
    assert should_brake is False
    assert fake.robot.waypoints == [ROBOT_NEAR_TARGET], "the robot must hold at its current position"
    assert fake.agent.exploration_target_xy is None
    assert not any("[ROUTE ok]" in str(line) for line in fake.console_logs)


# ---------------------------------------------------------------------------
# 2. select_navigation_goal() excludes the robot's own current position.
#
# select_exploration_goal() (the real FoV-aware frontier scorer) is
# monkeypatched here rather than driven through a real BeliefMap: the
# scorer's own "forward candidate" mechanism (walks straight ahead until it
# hits an obstacle) makes it very hard to reliably construct a belief map
# where a real close candidate wins on merit -- it dominates almost any
# synthetic map with open space ahead of the robot. Patching the
# module-level select_exploration_goal (imported into engine.py) instead
# tests exactly what select_navigation_goal() controls: what it passes as
# excluded_targets, and how it reacts to whatever comes back -- without
# depending on frontier/forward-candidate scoring internals this task must
# not touch.
# ---------------------------------------------------------------------------


def test_engine_select_navigation_goal_excludes_current_robot_position(monkeypatch):
    fake = _build_fake_engine(position=(0.0, 0.0), goal_tolerance=0.25)
    fake.ensure_belief_map = lambda: object()
    fake.final_goal_xy = lambda: (0.0, 0.0)
    fake.safety_radius = lambda: 0.2
    fake.config.vision = 6.0
    fake.config.vision_model = "LiDAR"
    fake.config.ipp_distance_penalty = 0.2

    calls: list[dict] = []

    def _fake_select_exploration_goal(planner_name, **kwargs):
        calls.append(kwargs)
        # Simulates a planner that ignores the exclusion list and always
        # proposes a candidate exactly at the robot's own position -- the
        # scenario that produced the observed "length=0.02" ROUTE ok loop.
        return ExplorationPlannerResult(True, (0.0, 0.0), "fake: always proposes robot's own position")

    monkeypatch.setattr(engine_module, "select_exploration_goal", _fake_select_exploration_goal)

    real_select_navigation_goal = SimulationControllerMixin.select_navigation_goal.__get__(fake)
    target, reason = real_select_navigation_goal((0.0, 0.0))

    assert len(calls) == 1
    excluded = calls[0]["excluded_targets"]
    assert (0.0, 0.0) in excluded, (
        "select_navigation_goal() must pass the robot's own current position as an excluded target"
    )

    # Even though the fake planner ignored the exclusion and returned the
    # robot's own position anyway, select_navigation_goal() must still
    # reject it (belt-and-suspenders) rather than accept it as a real target.
    assert target is None, (
        f"select_navigation_goal() must not return a candidate at/near the robot's own "
        f"position; got target={target!r} reason={reason!r}"
    )


# ---------------------------------------------------------------------------
# 3. A valid ExplorationBehavior target is honored directly; the engine
#    does not re-derive a (possibly different) target via
#    select_navigation_goal().
# ---------------------------------------------------------------------------


def test_request_plan_uses_behavior_target_without_reselecting_if_target_is_valid():
    fake = _build_fake_engine(position=(0.0, 0.0), goal_tolerance=0.25)

    decision = _request_plan_decision(FAR_TARGET, force_new_target=True)

    SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    assert fake.select_navigation_goal_calls == [], (
        "select_navigation_goal() must not be called again when the decision already carries a valid target"
    )
    assert fake.build_planner_kwargs_for_goal_calls == [((0.0, 0.0), FAR_TARGET)], (
        "the route request must plan directly to the behavior-selected target"
    )
    # The target_override branch commits FAR_TARGET as the exploration
    # target before planning (build_planner_kwargs_for_goal is stubbed to
    # simulate a subsequent planner failure, which legitimately clears it
    # again afterward -- that tail is exercised by
    # test_route_endpoint_validation.py, not this test).
    assert FAR_TARGET in fake.exploration_targets
    assert fake.agent.last_exploration_reason == "using ExplorationBehavior-selected target"


def test_request_plan_falls_back_to_select_navigation_goal_when_target_is_none():
    """decision.target may legitimately be None (e.g. a future decision
    kind); in that case the engine must still fall back to
    select_navigation_goal()."""
    fake = _build_fake_engine(position=(0.0, 0.0), goal_tolerance=0.25)
    fake._select_navigation_goal_result = (FAR_TARGET, "fallback selection")

    decision = SimpleNamespace(
        kind="REQUEST_PLAN", reason="agent requested plan", target=None, brake=False, force_new_target=False
    )

    SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    assert fake.select_navigation_goal_calls == [(0.0, 0.0)]
    assert fake.build_planner_kwargs_for_goal_calls == []


# ---------------------------------------------------------------------------
# 4. Repeated "frontier reached" ticks for the same already-reached point
#    must never recreate the ROUTE ok / REQUEST_PLAN loop.
# ---------------------------------------------------------------------------


def test_frontier_reached_loop_not_recreated_by_engine_target_selection():
    fake = _build_fake_engine(position=ROBOT_NEAR_TARGET, goal_tolerance=0.25)

    for tick in range(5):
        fake.simulation_time = float(tick)
        decision = _request_plan_decision(ALREADY_REACHED_TARGET, force_new_target=True)
        SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    assert fake.request_route_async_calls == [], (
        "no tick in this streak may launch a route request for the already-reached target"
    )
    assert not any("[ROUTE ok]" in str(line) for line in fake.console_logs)
    assert fake.robot.waypoints == [ROBOT_NEAR_TARGET]
