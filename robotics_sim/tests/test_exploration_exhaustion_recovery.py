"""
Regression tests for the periodic exploration-recovery loop.

Prior fixes (test_robot_agent_route_failure.py, test_exploration_target_recovery.py,
test_single_robot_safety_replan.py) stopped the *per-tick* REQUEST_PLAN and
REPLAN_FOR_SAFETY spam. A slower, still-broken loop remained in the manual
Office.sim run:

    Planner failed in exploration mode: no path found
    [NAV] kind=HOLD reason='recovering after planner failure; retry cooldown active'
    [NAV] kind=REQUEST_PLAN reason='recovered after planner failure; requesting fresh frontier'
    Planner failed in exploration mode: no path found
    ... (repeats every cooldown cycle, indefinitely)

Root cause: every planner failure was treated as a one-off, independently
recoverable event. RobotAgent had no memory of "we've already tried and
failed N times in a row without any new map information" -- so
ExplorationBehavior.update() step 6 kept re-running frontier selection
every _FAILURE_RETRY_COOLDOWN seconds forever, even when the map had not
changed at all since the previous failure (i.e. there was never any reason
to expect a different outcome).

Fix:
    - RobotAgent.consecutive_exploration_failures / register_exploration_failure()
      / exploration_exhausted() track consecutive recovery failures and,
      once a small budget is reached, remember the map signature (here:
      len(mapped_obstacle_points)) at which the agent gave up.
    - ExplorationBehavior.update() step 6 checks exploration_exhausted()
      first: while the map signature is unchanged, it holds with reason
      "exploration exhausted: no reachable frontier candidates" and does
      NOT call the frontier planner at all -- no REQUEST_PLAN, no spam,
      regardless of how many cooldown cycles pass.
    - exploration_exhausted() itself clears the exhausted state (and the
      failure counter) the moment the map signature changes, so recovery
      resumes automatically once new information arrives -- no permanent
      dead end, no user reset required.
    - RobotAgent.assign_path() / accept_pending_path() reset the counter on
      any successfully-committed route, so a normal exploration run never
      accumulates failures across unrelated successful legs.
    - engine.apply_route_result()'s exploration-failure branch and the
      repeated-safety-replan branch now thread map_signature through to
      RobotAgent.invalidate_failed_exploration_route(), and their log
      messages include attempted_target for debuggability.

These tests exercise RobotAgent and ExplorationBehavior directly (no Qt, no
canvas, no full engine/GUI, no real BeliefMap/A*) using the same
_FakePlannerServices stub pattern as test_exploration_target_recovery.py.
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
# 1. Repeated no-path recovery failures, with no new map information, must
#    settle into a stable exhausted hold instead of retrying forever.
# ---------------------------------------------------------------------------


def test_recovery_stops_after_repeated_no_path_failures_without_new_map_information():
    agent = _make_agent()
    behavior = ExplorationBehavior()
    map_points = [(1.0, 1.0)]  # constant throughout: no new map information

    # First failure: target A, mirrors the original "frontier reached ->
    # planner failed" sequence.
    agent.set_exploration_target(FAILED_TARGET, reason="frontier reached; requesting next frontier")
    agent.assign_path(target=FAILED_TARGET, waypoints=[FAILED_TARGET], planner_reason="initial route")
    t = 0.0
    agent.invalidate_failed_exploration_route(
        reason="planner failed: no path found",
        current_time=t,
        map_signature=len(map_points),
    )

    # Two more recovery cycles: each time a *different*, freshly-reselected
    # candidate is offered and it also fails -- this is genuinely "nothing
    # reachable", not "stuck retrying the same target" (that case is
    # covered by test_exploration_target_recovery.py).
    for candidate in [(2.0, 2.0), (3.0, 3.0)]:
        t += behavior._FAILURE_RETRY_COOLDOWN + 0.1
        fake_services = _FakePlannerServices(target=candidate)
        observation = _make_observation(
            robot_xy=agent.position, current_time=t, mapped_obstacle_points=list(map_points),
        )
        decision = behavior.update(agent, observation, fake_services)
        assert decision.kind == "REQUEST_PLAN"
        assert decision.target == candidate

        # Simulate the engine: select_navigation_goal() commits `candidate`
        # to exploration_target_xy before A* runs, then A* fails.
        agent.set_exploration_target(candidate, reason="recovered after planner failure; requesting fresh frontier")
        agent.invalidate_failed_exploration_route(
            reason="planner failed: no path found",
            current_time=t,
            map_signature=len(map_points),
        )

    assert agent.consecutive_exploration_failures == RobotAgent._EXPLORATION_FAILURE_BUDGET

    # Reaching the old fixed budget must not suppress an untried candidate.
    t += behavior._FAILURE_RETRY_COOLDOWN + 0.1
    fake_services = _FakePlannerServices(target=(4.0, 4.0))
    observation = _make_observation(
        robot_xy=agent.position, current_time=t, mapped_obstacle_points=list(map_points),
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == (4.0, 4.0)
    assert len(fake_services.calls) == 1

    agent.set_exploration_target((4.0, 4.0), reason="probe after precheck failures")
    agent.invalidate_failed_exploration_route(
        reason="planner failed: no path found", current_time=t, map_signature=len(map_points)
    )
    t += behavior._FAILURE_RETRY_COOLDOWN + 0.1
    empty_services = _FakePlannerServices(target=None)
    observation = _make_observation(
        robot_xy=agent.position, current_time=t, mapped_obstacle_points=list(map_points),
    )
    decision = behavior.update(agent, observation, empty_services)
    assert decision.kind == "HOLD"
    assert decision.reason == "exploration exhausted: no reachable frontier candidates"
    calls_after_confirmation = len(empty_services.calls)

    # And it must stay exhausted indefinitely -- not just for one tick --
    # across many more cooldown cycles, as long as the map does not change.
    for _ in range(5):
        t += behavior._FAILURE_RETRY_COOLDOWN + 0.1
        observation = _make_observation(
            robot_xy=agent.position, current_time=t, mapped_obstacle_points=list(map_points),
        )
        decision = behavior.update(agent, observation, empty_services)
        assert decision.kind == "HOLD"
        assert decision.reason == "exploration exhausted: no reachable frontier candidates"

    assert len(empty_services.calls) == calls_after_confirmation


# ---------------------------------------------------------------------------
# 2. A recovery failure must record the attempted target (and count toward
#    the exhaustion budget) even when active_target() was None the whole
#    time -- i.e. before any route/waypoints were ever assigned.
# ---------------------------------------------------------------------------


def test_recovery_failure_records_attempted_target_even_when_active_target_is_none():
    agent = _make_agent()

    # Recovery decision emitted with no active waypoints yet -- mirrors
    # ExplorationBehavior step 6's REQUEST_PLAN before the engine has
    # computed any route for this tick.
    assert agent.active_target() is None
    assert agent.exploration_target_xy is None

    # engine.select_navigation_goal() chooses target B and, on success,
    # calls agent.set_exploration_target(B, ...) synchronously -- *before*
    # the async A* worker even runs.
    agent.set_exploration_target(ALTERNATE_TARGET, reason="FoV-aware directional frontier: selected best target")
    assert agent.active_target() is None  # still no waypoints -- A* has not run yet

    # A* then fails; engine.apply_route_result() calls this on the
    # exploration-failure branch.
    agent.invalidate_failed_exploration_route(
        reason="planner failed: no path found",
        current_time=5.0,
        map_signature=7,
    )

    assert ALTERNATE_TARGET in agent.recently_failed_exploration_targets(current_time=5.0, cooldown=5.0), (
        "the attempted target B must be recorded even though agent.active_target() "
        "was None throughout (no route had ever been assigned)"
    )
    assert agent.consecutive_exploration_failures == 1, (
        "a failure for a target chosen during recovery (active_target()=None) must "
        "still count toward the exhaustion budget"
    )

    # A subsequent fresh selection attempt must exclude B.
    fake_services = _FakePlannerServices(target=(9.0, 9.0))
    behavior = ExplorationBehavior()
    observation = _make_observation(
        robot_xy=agent.position,
        current_time=5.0 + behavior._FAILURE_RETRY_COOLDOWN + 0.1,
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    excluded = fake_services.calls[0]["excluded_targets"]
    assert ALTERNATE_TARGET in excluded


# ---------------------------------------------------------------------------
# 3. New map information must lift a stable "exploration exhausted" hold.
# ---------------------------------------------------------------------------


def test_exploration_recovery_resumes_after_new_map_information():
    agent = _make_agent()
    behavior = ExplorationBehavior()
    old_map_points = [(1.0, 1.0)]

    # Drive the agent directly into the exhausted state (equivalent to the
    # end state of test 1's failure loop, but without repeating it here).
    for _ in range(RobotAgent._EXPLORATION_FAILURE_BUDGET):
        agent.register_exploration_failure(map_signature=len(old_map_points))
    agent.exploration_exhaustion_confirmed_empty = True
    assert agent.exploration_exhausted(map_signature=len(old_map_points))

    # While the map is unchanged, recovery stays exhausted -- no REQUEST_PLAN.
    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    observation = _make_observation(
        robot_xy=agent.position, current_time=100.0, mapped_obstacle_points=list(old_map_points),
    )
    decision = behavior.update(agent, observation, fake_services)
    assert decision.kind == "HOLD"
    assert decision.reason == "exploration exhausted: no reachable frontier candidates"
    assert len(fake_services.calls) == 0

    # New map information appears (more mapped obstacle samples, e.g. from
    # continued sensing while holding): recovery must resume automatically,
    # without requiring a user reset.
    new_map_points = old_map_points + [(2.0, 2.0), (3.0, 3.0)]
    observation = _make_observation(
        robot_xy=agent.position, current_time=100.1, mapped_obstacle_points=list(new_map_points),
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN", (
        "new map information must clear the exhausted state and let recovery resume"
    )
    assert decision.target == ALTERNATE_TARGET
    assert agent.consecutive_exploration_failures == 0
    assert agent.exploration_exhausted_map_signature is None
