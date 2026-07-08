"""
Regression tests for exploration recovery after a planner failure.

Prior fix (test_robot_agent_route_failure.py / commit "fix: clear failed
exploration target after planner failure") stopped the infinite REQUEST_PLAN
spam by clearing agent.exploration_target_xy on planner failure. That
surfaced a second problem: ExplorationBehavior.update() step 6 only ever
read agent.desired_target_from_mode() -- it never asked the frontier planner
for a *new* target when there wasn't one already assigned. So once
exploration_target_xy was cleared, the agent sat in
    HOLD reason='no target available in current exploration mode'
forever, with no path back to exploring.

This file tests the recovery contract:
    1. A failed target is remembered and excluded from re-selection for a
       while (RobotAgent.mark_exploration_target_failed() /
       recently_failed_exploration_targets()).
    2. Immediately after a failure, ExplorationBehavior.update() must not
       spam a new selection attempt every tick -- it holds until a short
       retry cooldown elapses (RobotAgent.exploration_retry_on_cooldown()).
    3. Once the cooldown elapses, ExplorationBehavior.update() asks the
       frontier planner for a fresh target, excluding the failed one. If a
       different reachable target exists, the agent resumes exploring it.
    4. If no candidates exist at all, the agent HOLDs with a reason
       distinguishable from both the generic cold-start message and A*'s
       "no path found".

These tests exercise RobotAgent and ExplorationBehavior directly (no Qt, no
canvas, no engine.py, no real BeliefMap/A*) using a fake PlannerServices that
records how it was called and returns a canned result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.observation import RobotObservation


FAILED_TARGET = (7.75, -4.75)
ALTERNATE_TARGET = (2.0, 3.0)


@dataclass
class _FakePlannerServices:
    """Stand-in for PlannerServices.select_exploration_target().

    Records every call's kwargs (so tests can assert on excluded_targets)
    and returns a canned ExplorationPlannerResult: `target=None` simulates
    "no reachable frontier candidates", any other value simulates a
    successfully selected alternative.
    """

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


def _fail_target(agent: RobotAgent, target, *, current_time: float) -> None:
    """Simulate the exact sequence engine.apply_route_result() runs on a
    planner failure in exploration mode."""
    agent.set_exploration_target(target, reason="frontier reached; requesting next frontier")
    agent.assign_path(target=target, waypoints=[target], planner_reason="initial route")
    agent.invalidate_failed_exploration_route(
        reason="planner failed: no path found",
        current_time=current_time,
    )


def test_failed_exploration_target_is_not_immediately_reselected():
    """Once the retry cooldown has elapsed, re-selection must exclude the
    target that just failed, and must pick a different reachable target
    (ALTERNATE_TARGET) when one exists."""
    agent = _make_agent()
    _fail_target(agent, FAILED_TARGET, current_time=0.0)

    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    behavior = ExplorationBehavior()

    # Past both the retry cooldown and comfortably within the exclusion window.
    observation = _make_observation(
        robot_xy=agent.position,
        current_time=behavior._FAILURE_RETRY_COOLDOWN + 0.1,
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN", f"expected REQUEST_PLAN, got {decision!r}"
    assert decision.target == ALTERNATE_TARGET, (
        f"expected the fresh target {ALTERNATE_TARGET!r}, got {decision.target!r}"
    )
    assert decision.target != FAILED_TARGET

    assert len(fake_services.calls) == 1, "expected exactly one selection attempt"
    excluded = fake_services.calls[0]["excluded_targets"]
    assert FAILED_TARGET in excluded, (
        f"the just-failed target {FAILED_TARGET!r} must be passed as excluded_targets "
        f"so the planner does not immediately re-select it; excluded={excluded!r}"
    )


def test_exploration_failure_recovers_with_fresh_target_after_cooldown():
    """Immediately after a failure there must be no spam (no selection
    attempt at all); after the cooldown elapses, a fresh target is
    requested."""
    agent = _make_agent()
    _fail_target(agent, FAILED_TARGET, current_time=0.0)

    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    behavior = ExplorationBehavior()

    # Immediately after the failure: still within the retry cooldown.
    observation = _make_observation(robot_xy=agent.position, current_time=0.05)
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "HOLD", f"expected HOLD during retry cooldown, got {decision!r}"
    assert decision.kind != "REQUEST_PLAN"
    assert len(fake_services.calls) == 0, (
        "no selection attempt should be made while the retry cooldown is active"
    )

    # After the cooldown has elapsed: a fresh target request is allowed.
    observation = _make_observation(
        robot_xy=agent.position,
        current_time=behavior._FAILURE_RETRY_COOLDOWN + 0.1,
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == ALTERNATE_TARGET
    assert decision.target != FAILED_TARGET
    assert len(fake_services.calls) == 1


def test_exploration_hold_reason_when_no_targets_remain():
    """When no candidates exist at all after the cooldown, the agent must
    HOLD with a reason distinguishable from both the generic cold-start
    message and A*'s "no path found", and must not busy-loop the frontier
    planner every tick while still holding."""
    agent = _make_agent()
    _fail_target(agent, FAILED_TARGET, current_time=0.0)

    fake_services = _FakePlannerServices(target=None)  # no reachable frontier candidates
    behavior = ExplorationBehavior()

    observation = _make_observation(
        robot_xy=agent.position,
        current_time=behavior._FAILURE_RETRY_COOLDOWN + 0.1,
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "HOLD"
    assert "no reachable frontier candidates" in decision.reason, (
        f"expected a HOLD reason distinguishable from the generic cold-start message "
        f"and from A*'s own 'no path found', got reason={decision.reason!r}"
    )
    assert decision.reason != "no target available in current exploration mode"
    assert "no path found" not in decision.reason
    assert len(fake_services.calls) == 1

    # The failed re-selection attempt must reset the retry-cooldown clock,
    # so the very next tick does not immediately try again (no busy-loop).
    observation = _make_observation(
        robot_xy=agent.position,
        current_time=behavior._FAILURE_RETRY_COOLDOWN + 0.15,
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "HOLD"
    assert len(fake_services.calls) == 1, (
        "an unsuccessful re-selection attempt must reset the cooldown clock, "
        "otherwise the agent re-runs frontier detection every single tick"
    )
