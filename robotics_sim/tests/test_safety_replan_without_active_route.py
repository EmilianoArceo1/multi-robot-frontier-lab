"""
Regression tests for REPLAN_FOR_SAFETY being emitted/reported with no
active route.

Manual Office.sim telemetry, after exploration reached exhaustion:

    [NAV] R1 kind=HOLD reason="exploration exhausted: no reachable frontier candidates" active=None path_goal=None pending=None
    [NAV] R1 kind=REPLAN_FOR_SAFETY reason="predicted collision" active=None path_goal=None pending=None
    Holding: repeated safety replan for the same target (predicted collision); marking target as failed and re-selecting.
    [ROUTE fail] attempted=None reason=repeated_safety_replan

A prior fix added a has_active_route guard to ExplorationBehavior.update()
step 1, and a matching defense-in-depth guard inside
engine.apply_navigation_decision()'s REPLAN_FOR_SAFETY branch. Both were
individually correct and are still covered below (test 1, test 4) -- but
manual acceptance still showed the invalid "kind=REPLAN_FOR_SAFETY" console
line, and occasionally the repeated_safety_replan [ROUTE fail] line too.

Root cause of the remaining gap: RobotAgent.step() (robotics_sim/core/
robot_agent.py) has its OWN, separate safety check that runs
UNCONDITIONALLY, before it ever dispatches to
ExplorationBehavior.update():

    if observation.active_segment_blocked or observation.predicted_collision:
        ...
        return replan_for_safety(self.desired_target_from_mode(), reason=reason)
    ...
    return self.behavior.update(self, observation, planner_services)

So whenever predicted_collision is True, RobotAgent.step() returns
REPLAN_FOR_SAFETY directly -- ExplorationBehavior.update() (and its
has_active_route guard) is never even reached. Every previous test in this
file called ExplorationBehavior.update() or the engine's REPLAN_FOR_SAFETY
handler directly, in isolation -- neither exercised the actual
RobotAgent.step() entry point, so neither caught that RobotAgent.step()
itself is the thing generating the invalid decision in the real runtime
loop (engine.simulation_step() calls agent.step(), not behavior.update()
directly).

robot_agent.py is not in this round's touchable scope, so the fix instead
normalizes the decision at the engine boundary, in
apply_navigation_decision(), BEFORE telemetry.report_nav_decision() is
called and before any route-request/failure-marking logic runs: if
kind == "REPLAN_FOR_SAFETY" and the agent has no active route
(active_path_goal_xy is not None and active_target() is not None both
required -- do not trust active_target() alone), the decision kind is
rewritten to "HOLD" right there. This guarantees no console line ever
reports "kind=REPLAN_FOR_SAFETY" for a route-less decision, and no
replan/failure-marking logic downstream ever runs for it, regardless of
which upstream code path (RobotAgent.step()'s own check, or
ExplorationBehavior.update(), or anything else) produced the decision.

These tests exercise RobotAgent/ExplorationBehavior directly (tests 1-2, no
Qt/canvas/engine) and engine.apply_navigation_decision() via a minimal
duck-typed engine fake (tests 3-5), matching the pattern already used in
test_multi_robot_route_validation.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.telemetry import TelemetryLogger


ACTIVE_TARGET = (4.75, -4.25)
ACTIVE_PATH_GOAL = (7.25, 3.75)


@dataclass
class _FakePlannerServices:
    """Stand-in for PlannerServices.select_exploration_target()."""

    target: tuple[float, float] | None
    calls: list[dict] = field(default_factory=list)

    def select_exploration_target(self, **kwargs) -> ExplorationPlannerResult:
        self.calls.append(kwargs)
        if self.target is None:
            return ExplorationPlannerResult(False, None, "no valid frontier candidates found")
        return ExplorationPlannerResult(True, self.target, "fake planner: selected target")


def _make_agent(position=(0.0, 0.0)) -> RobotAgent:
    return RobotAgent(
        robot_id=0,
        position=position,
        planner_mode="FoV-aware directional frontier",
    )


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
# 1. No active route -> HOLD, not REPLAN_FOR_SAFETY, even with a predicted
#    collision -- exercising ExplorationBehavior.update() directly.
# ---------------------------------------------------------------------------


def test_predicted_collision_without_active_route_holds_instead_of_replanning():
    agent = _make_agent()
    assert agent.active_path_goal_xy is None
    assert agent.active_target() is None

    behavior = ExplorationBehavior()
    observation = _make_observation(predicted_collision=True, current_time=10.0)
    fake_services = _FakePlannerServices(target=None)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind != "REPLAN_FOR_SAFETY", (
        "a predicted collision with no active route must not trigger a safety replan"
    )
    assert decision.kind == "HOLD"


# ---------------------------------------------------------------------------
# 2. Exhausted exploration + predicted collision -> the exhaustion HOLD
#    wins; no safety replan -- exercising ExplorationBehavior.update()
#    directly (this is the branch that decides "HOLD before safety").
# ---------------------------------------------------------------------------


def test_exhaustion_holds_before_safety_even_if_predicted_collision():
    agent = _make_agent()
    map_signature = 42
    for _ in range(RobotAgent._EXPLORATION_FAILURE_BUDGET):
        agent.register_exploration_failure(map_signature=map_signature)
    assert agent.exploration_exhausted(map_signature=map_signature)
    assert agent.active_path_goal_xy is None
    assert agent.active_target() is None

    behavior = ExplorationBehavior()
    observation = _make_observation(
        predicted_collision=True,
        mapped_obstacle_points=[(0.0, 0.0)] * map_signature,
        current_time=10.0,
    )
    fake_services = _FakePlannerServices(target=(9.0, 9.0))

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "HOLD"
    assert "exploration exhausted" in decision.reason
    assert decision.kind != "REPLAN_FOR_SAFETY"
    assert len(fake_services.calls) == 0, (
        "an exhausted, route-less robot must not even attempt frontier re-selection "
        "just because a predicted collision flag is set"
    )


# ---------------------------------------------------------------------------
# Engine-level fixture shared by tests 3-5.
# ---------------------------------------------------------------------------


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = waypoints


def _build_fake_engine() -> SimpleNamespace:
    robot = _FakeRobot(x=1.0, y=2.0)
    fake = SimpleNamespace(
        robots=[],
        robot=robot,
        config=SimpleNamespace(
            planner_type="A*",
            path_simplifier="Direction changes",
            exploration_planner="FoV-aware directional frontier",
            exploration_replan_cooldown=1.0,
            goal_tolerance=0.25,
        ),
        mapped_obstacle_points=[],
        simulation_time=10.0,
        console_logs=[],
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.is_exploration_mode = lambda: True
    fake.replan_calls = []
    fake.replan_after_new_information = lambda reason: fake.replan_calls.append(reason) or False

    for name in ("set_robot_goal_or_waypoints", "safety_replan_cooldown_seconds"):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    return fake


def _replan_for_safety_decision(target, reason="predicted collision"):
    return SimpleNamespace(kind="REPLAN_FOR_SAFETY", reason=reason, target=target, brake=True)


# ---------------------------------------------------------------------------
# 3. Route-less REPLAN_FOR_SAFETY must never be reported as
#    "kind=REPLAN_FOR_SAFETY" in telemetry, regardless of what upstream
#    code produced the decision (RobotAgent.step()'s own unconditional
#    safety check, in the real runtime, is exactly such a producer).
# ---------------------------------------------------------------------------


def test_no_active_route_safety_decision_is_not_reported_as_nav_replan():
    fake = _build_fake_engine()
    agent = _make_agent(position=(1.0, 2.0))
    assert agent.active_path_goal_xy is None
    assert agent.active_target() is None

    decision = _replan_for_safety_decision(None)

    should_brake = SimulationControllerMixin.apply_navigation_decision(
        fake, fake.robot, agent, decision
    )

    # Normalized to a genuine HOLD, not a safety replan: should_brake
    # follows HOLD's own contract (False -- the robot just holds its
    # current position, it does not need an extra brake flag on top).
    assert should_brake is False
    assert not any("kind=REPLAN_FOR_SAFETY" in str(line) for line in fake.console_logs), (
        "no console line may report kind=REPLAN_FOR_SAFETY for a route-less decision"
    )
    assert any("kind=HOLD" in str(line) for line in fake.console_logs), (
        "the route-less safety decision must be reported as HOLD instead"
    )
    assert fake.replan_calls == [], "no replan request may be launched with no active route"
    assert agent.exploration_target_xy is None
    assert agent.failed_exploration_targets == [], (
        "no target may be marked as failed when there was no active route to begin with"
    )
    assert not any("repeated_safety_replan" in str(line) for line in fake.console_logs), (
        "no repeated_safety_replan [ROUTE fail] line may be emitted for a nonexistent target"
    )


# ---------------------------------------------------------------------------
# 4. With an active route, a predicted collision still triggers a real
#    safety replan (unchanged behavior) at the ExplorationBehavior level.
# ---------------------------------------------------------------------------


def test_safety_replan_still_allowed_with_active_route():
    agent = _make_agent(position=(5.0, 0.0))
    agent.set_exploration_target(ACTIVE_PATH_GOAL, reason="test target")
    agent.assign_path(
        target=ACTIVE_PATH_GOAL,
        waypoints=[ACTIVE_TARGET, ACTIVE_PATH_GOAL],
        planner_reason="test route",
    )
    assert agent.active_path_goal_xy == ACTIVE_PATH_GOAL
    assert agent.active_target() == ACTIVE_TARGET

    behavior = ExplorationBehavior()
    observation = _make_observation(predicted_collision=True, robot_xy=agent.position, current_time=10.0)
    fake_services = _FakePlannerServices(target=None)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REPLAN_FOR_SAFETY"
    assert decision.reason == "predicted collision"
    assert decision.target == ACTIVE_PATH_GOAL


def test_engine_still_replans_for_safety_with_active_route():
    """Engine-level counterpart of test 4: a REPLAN_FOR_SAFETY decision with
    a real active route must still be handled as a safety replan, not
    normalized away."""
    fake = _build_fake_engine()
    agent = _make_agent(position=(1.0, 2.0))
    agent.set_exploration_target(ACTIVE_PATH_GOAL, reason="test target")
    agent.assign_path(
        target=ACTIVE_PATH_GOAL,
        waypoints=[ACTIVE_TARGET, ACTIVE_PATH_GOAL],
        planner_reason="test route",
    )

    decision = _replan_for_safety_decision(ACTIVE_PATH_GOAL)

    should_brake = SimulationControllerMixin.apply_navigation_decision(
        fake, fake.robot, agent, decision
    )

    assert should_brake is True
    assert any("kind=REPLAN_FOR_SAFETY" in str(line) for line in fake.console_logs), (
        "a genuine safety replan with an active route must still be reported as such"
    )
    assert fake.replan_calls == ["safety replan: predicted collision"]


# ---------------------------------------------------------------------------
# 5. Multiple consecutive ticks of "no route, predicted_collision=True"
#    must never alternate into a REPLAN_FOR_SAFETY line, and must not spam
#    the console once NAV dedup kicks in.
# ---------------------------------------------------------------------------


def test_runtime_like_exhausted_hold_with_predicted_collision_does_not_alternate_replan_and_hold():
    fake = _build_fake_engine()
    agent = _make_agent(position=(1.0, 2.0))
    assert agent.active_path_goal_xy is None
    assert agent.active_target() is None

    for tick in range(5):
        fake.simulation_time = 10.0 + tick * 0.1
        decision = _replan_for_safety_decision(None)
        should_brake = SimulationControllerMixin.apply_navigation_decision(
            fake, fake.robot, agent, decision
        )
        # Normalized to HOLD each tick: HOLD's own contract, not brake=True.
        assert should_brake is False

    assert not any("kind=REPLAN_FOR_SAFETY" in str(line) for line in fake.console_logs), (
        "no tick in this route-less streak may be reported as kind=REPLAN_FOR_SAFETY"
    )
    assert fake.replan_calls == [], "no replan request may ever be launched across the streak"
    assert not any("repeated_safety_replan" in str(line) for line in fake.console_logs)
    # NAV dedup collapses the identical repeated HOLD signature: far fewer
    # console lines than ticks, and every one that was emitted says HOLD.
    assert 0 < len(fake.console_logs) <= 5
    assert all("kind=HOLD" in str(line) for line in fake.console_logs)
