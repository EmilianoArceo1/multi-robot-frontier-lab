"""
Regression tests for the "already-reached target reselected" loop.

Manual Office.sim telemetry showed:

    [NAV] R1 kind=REQUEST_PLAN reason="frontier reached; requesting next frontier" ...
    [FRONTIER] R1 ... selected=(2.75,-3.75) ...
    [ROUTE ok] R1 ... goal=(2.75,-3.75) ... length=0.23 ...
    [NAV] R1 kind=REQUEST_PLAN reason="frontier reached; requesting next frontier" ...
    ... (repeats many times in the same timestamp window)

Root cause: ExplorationBehavior._pick_next_target() never excluded either
(a) the robot's own current position, or (b) the exploration target that
was just completed (agent.active_path_goal_xy) -- so once step 3 ("frontier
reached") cleared exploration_target_xy and asked for a fresh target,
FoV-aware frontier detection was free to immediately re-propose the exact
same nearby point (or one still within goal_tolerance of the robot). The
resulting route was near-zero length, "reached" again within a tick or
two, and the cycle repeated.

Fix: _pick_next_target() now
    (1) adds observation.robot_xy and agent.active_path_goal_xy (when set)
        to the excluded_targets passed to the planner, reusing the
        existing excluded_targets/target_exclusion_radius mechanism, and
    (2) hard-rejects (returns None for) any candidate the planner still
        returns that is within observation.goal_tolerance of the robot's
        current position, regardless of how the exclusion above was
        honored internally -- this is the actual guarantee, since it does
        not depend on the exploration planner respecting exclusions.

These tests exercise RobotAgent and ExplorationBehavior directly (no Qt, no
canvas, no full engine/GUI, no real BeliefMap/A*) using the same
_FakePlannerServices stub pattern as the other exploration-behavior
regression tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.observation import RobotObservation


REACHED_TARGET = (2.75, -3.75)
FAR_TARGET = (9.0, 9.0)


@dataclass
class _FakePlannerServices:
    """Stand-in for PlannerServices.select_exploration_target().

    Behaves like a real reachability-respecting planner: if `target` falls
    within `exclusion_radius` of anything in the caller's excluded_targets,
    it reports failure instead of returning that point -- this lets tests
    verify the CALLER (ExplorationBehavior) actually threads the exclusion
    through, not just that some independent hard filter catches it.
    """

    target: tuple[float, float] | None
    exclusion_radius: float = 0.3
    calls: list[dict] = field(default_factory=list)

    def select_exploration_target(self, **kwargs) -> ExplorationPlannerResult:
        self.calls.append(kwargs)
        if self.target is None:
            return ExplorationPlannerResult(False, None, "no valid frontier candidates found")

        excluded = kwargs.get("excluded_targets") or []
        for point in excluded:
            if math.hypot(self.target[0] - point[0], self.target[1] - point[1]) <= self.exclusion_radius:
                return ExplorationPlannerResult(
                    False, None, "no reachable frontier candidates: target excluded"
                )
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
# 1. A candidate at/near the robot's current position must be rejected,
#    regardless of whether the planner itself honored the exclusion.
# ---------------------------------------------------------------------------


def test_frontier_recovery_does_not_select_current_position_as_next_target():
    agent = _make_agent(position=(2.75, -3.75))
    behavior = ExplorationBehavior()
    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25, current_time=1.0)

    # The fake planner ignores exclusions entirely here (exclusion_radius=0)
    # and always returns a point exactly at the robot's own position --
    # simulating a planner/scoring quirk that still proposes an
    # already-reached candidate despite the exclusion list.
    fake_services = _FakePlannerServices(target=(2.75, -3.75), exclusion_radius=0.0)

    result = behavior._pick_next_target(agent, observation, fake_services)

    assert result is None, (
        "a candidate within goal_tolerance of the robot's current position "
        "must never be accepted as the next exploration target"
    )


def test_frontier_recovery_selects_farther_candidate_when_current_position_would_be_chosen():
    """When the excluded current position is properly honored by the
    planner, a farther, genuinely different candidate must still be
    selectable -- rejecting "already reached" must not turn into rejecting
    everything."""
    agent = _make_agent(position=(2.75, -3.75))
    behavior = ExplorationBehavior()
    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25, current_time=1.0)

    fake_services = _FakePlannerServices(target=FAR_TARGET, exclusion_radius=0.3)

    result = behavior._pick_next_target(agent, observation, fake_services)

    assert result == FAR_TARGET
    excluded = fake_services.calls[0]["excluded_targets"]
    assert agent.position in excluded or observation.robot_xy in excluded, (
        "the robot's current position must be passed as an excluded target"
    )


# ---------------------------------------------------------------------------
# 2. The just-completed active_path_goal_xy must be excluded from the next
#    selection cycle.
# ---------------------------------------------------------------------------


def test_frontier_reached_does_not_reselect_same_completed_path_goal():
    agent = _make_agent(position=REACHED_TARGET)
    agent.assign_path(
        target=REACHED_TARGET,
        waypoints=[REACHED_TARGET],
        planner_reason="initial route",
    )
    # Mirrors step 3 of ExplorationBehavior.update(): exploration_target_xy
    # is cleared before _pick_next_target() is called, but
    # active_path_goal_xy (the just-reached point) is deliberately left
    # untouched here to exercise the exclusion under test.
    agent.exploration_target_xy = None

    behavior = ExplorationBehavior()
    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25, current_time=1.0)

    # The fake planner would (mis)propose the exact point just reached
    # unless the caller excludes it.
    fake_services = _FakePlannerServices(target=REACHED_TARGET, exclusion_radius=0.3)

    result = behavior._pick_next_target(agent, observation, fake_services)

    assert result is None, "the just-completed path goal must be excluded from re-selection"
    excluded = fake_services.calls[0]["excluded_targets"]
    assert any(
        math.hypot(REACHED_TARGET[0] - p[0], REACHED_TARGET[1] - p[1]) <= 1e-6 for p in excluded
    ), "active_path_goal_xy must be included in excluded_targets for this selection cycle"


# ---------------------------------------------------------------------------
# 3. End-to-end: reaching a frontier must not repeatedly assign a
#    near-zero-length route back to the same already-reached goal.
# ---------------------------------------------------------------------------


def test_near_zero_length_exploration_route_is_not_assigned_repeatedly():
    agent = _make_agent(position=REACHED_TARGET)
    agent.assign_path(
        target=REACHED_TARGET,
        waypoints=[REACHED_TARGET],
        planner_reason="initial route",
    )

    behavior = ExplorationBehavior()
    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25, current_time=1.0)

    # Planner keeps proposing the same already-reached point every time it
    # is asked (worst case: it never learns from exclusions).
    fake_services = _FakePlannerServices(target=REACHED_TARGET, exclusion_radius=0.0)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind != "REQUEST_PLAN" or decision.target != REACHED_TARGET, (
        "must not request a new plan for the exact target that was just reached "
        f"(a near-zero-length route); decision={decision!r}"
    )
    # The realistic outcome here is HOLD, since the only candidate on offer
    # is the one that was just rejected.
    assert decision.kind == "HOLD"
    assert "frontier reached" in decision.reason


def test_near_zero_length_exploration_route_recovers_with_farther_target():
    """Same reached-frontier scenario, but a genuinely different farther
    candidate exists and the planner honors the exclusion -- exploration
    must continue onto it instead of holding."""
    agent = _make_agent(position=REACHED_TARGET)
    agent.assign_path(
        target=REACHED_TARGET,
        waypoints=[REACHED_TARGET],
        planner_reason="initial route",
    )

    behavior = ExplorationBehavior()
    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25, current_time=1.0)

    fake_services = _FakePlannerServices(target=FAR_TARGET, exclusion_radius=0.3)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == FAR_TARGET
    assert decision.reason == "frontier reached; requesting next frontier"
