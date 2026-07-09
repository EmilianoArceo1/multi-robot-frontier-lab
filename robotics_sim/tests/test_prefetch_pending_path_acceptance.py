"""
Regression tests for a valid prefetched pending path being abandoned when
the current frontier is reached.

Manual Office.sim telemetry (after the reachability-alignment fix landed):

    [PREFETCH] requested target=(7.25, 3.75)
    [PREFETCH] success waypoints=8
    ... (moments later)
    [NAV] R1 kind=HOLD reason="frontier reached; no valid next frontier available"
    active=(5.25,2.75) path_goal=(1.75,4.25) pending=(7.25,3.75)

A successful prefetch (agent.pending_path/pending_target_xy) already
existed when the robot reached its current path_goal, yet the HOLD
decision was made as if nothing were pending -- explored progress stalled
around 8.3%.

Root cause: ExplorationBehavior.update() step 3 ("frontier reached") went
straight to _pick_next_target() whenever path_goal_reached became True,
never checking agent.pending_path/pending_target_xy first. Step 2 ("pending
path ready -- should we switch?") is the ONLY place that normally accepts a
pending path, but it gates on agent.distance_to_active_target() -- and
agent.active_target() is the agent's own shadow WaypointManager's current
waypoint, which (by long-standing design -- see robot_agent.py) is never
advanced as the robot physically moves. It can stay pinned to an
already-passed intermediate waypoint from whenever the route was last
assigned, so its distance to the robot's real position may never drop
below step 2's threshold even as the robot's actual position (tracked via
active_path_goal_xy / distance_to_active_path_goal(), which step 3 uses)
genuinely reaches the destination. Step 3 fires, finds a pending path
sitting right there, and ignored it anyway.

Fix: step 3 now checks for a valid pending_path/pending_target_xy FIRST,
before attempting _pick_next_target() at all, and returns
ACCEPT_PENDING_PATH immediately if one exists -- mirroring step 2's own
accept-pending-path reasoning (should_brake_for_turn() for the sharp-turn
vs. smooth-turn reason text).

These tests exercise ExplorationBehavior.update() directly (no engine, no
Qt) with a _FakePlannerServices stub, matching the pattern in
test_frontier_reached_target_rejection.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.observation import RobotObservation


REACHED_PATH_GOAL = (1.75, 4.25)
STALE_ACTIVE_WAYPOINT = (5.25, 2.75)  # never advanced by the agent's shadow WaypointManager
PENDING_TARGET = (7.25, 3.75)
PENDING_PATH = [(6.25, 3.25), PENDING_TARGET]
RECOVERY_CANDIDATE = (0.5, 0.5)


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


def _make_agent_with_reached_route(*, pending_path=None, pending_target_xy=None) -> RobotAgent:
    """An agent whose active route's path_goal has just been reached, with
    the agent's own (possibly stale) active_target() left at an
    already-passed intermediate waypoint -- exactly the scenario from the
    manual log."""
    agent = RobotAgent(robot_id=0, position=REACHED_PATH_GOAL, planner_mode="FoV-aware directional frontier")
    agent.assign_path(
        target=REACHED_PATH_GOAL,
        waypoints=[STALE_ACTIVE_WAYPOINT, REACHED_PATH_GOAL],
        planner_reason="initial route",
    )
    agent.pending_path = list(pending_path) if pending_path is not None else None
    agent.pending_target_xy = pending_target_xy
    return agent


def _make_observation(**overrides) -> RobotObservation:
    defaults = dict(
        robot_xy=REACHED_PATH_GOAL,
        robot_heading=0.0,
        robot_radius=0.2,
        belief_map=None,
        planning_grid=None,
        mapped_obstacle_points=[],
        dynamic_obstacles=[],
        active_segment_blocked=False,
        predicted_collision=False,
        current_time=1.0,
        grid_resolution=0.5,
        goal_tolerance=0.25,
        sensor_range=2.5,
        final_goal_xy=None,
    )
    defaults.update(overrides)
    return RobotObservation(**defaults)


# ---------------------------------------------------------------------------
# 1. A valid pending path must be accepted instead of holding, even when
#    _pick_next_target() would return nothing.
# ---------------------------------------------------------------------------


def test_frontier_reached_accepts_existing_pending_path_before_holding():
    agent = _make_agent_with_reached_route(pending_path=PENDING_PATH, pending_target_xy=PENDING_TARGET)
    assert agent.distance_to_active_path_goal() <= 0.25, "sanity check: path_goal must actually be reached"

    behavior = ExplorationBehavior()
    observation = _make_observation()
    fake_services = _FakePlannerServices(target=None)  # _pick_next_target() would find nothing

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "ACCEPT_PENDING_PATH"
    assert decision.reason != "frontier reached; no valid next frontier available"
    assert len(fake_services.calls) == 0, "_pick_next_target() must not even be attempted when a pending path exists"


# ---------------------------------------------------------------------------
# 2. Accepting the pending path wins over recovery too -- recovery is never
#    reached because step 3 returns before step 6 could run.
# ---------------------------------------------------------------------------


def test_frontier_reached_accepts_pending_path_before_recovery():
    agent = _make_agent_with_reached_route(pending_path=PENDING_PATH, pending_target_xy=PENDING_TARGET)
    # A recovery candidate is available if recovery were ever consulted.
    agent.recent_safe_positions.append(RECOVERY_CANDIDATE)

    behavior = ExplorationBehavior()
    observation = _make_observation()
    fake_services = _FakePlannerServices(target=None)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "ACCEPT_PENDING_PATH"
    assert decision.target != RECOVERY_CANDIDATE
    assert "recovery" not in decision.reason.lower()
    assert agent.recovery_targets() == [], "recovery must never have been attempted, let alone recorded"


# ---------------------------------------------------------------------------
# 3. Existing behavior is unchanged with no pending path at all.
# ---------------------------------------------------------------------------


def test_frontier_reached_holds_when_no_pending_path_and_no_next_target():
    agent = _make_agent_with_reached_route(pending_path=None, pending_target_xy=None)

    behavior = ExplorationBehavior()
    observation = _make_observation()
    fake_services = _FakePlannerServices(target=None)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "HOLD"
    assert decision.reason == "frontier reached; no valid next frontier available"


# ---------------------------------------------------------------------------
# 4/5. A pending path is only accepted when BOTH fields are present and
# non-empty -- a partial/inconsistent state must fall through to existing
# behavior rather than being accepted anyway.
# ---------------------------------------------------------------------------


def test_pending_path_is_not_accepted_if_pending_target_missing():
    agent = _make_agent_with_reached_route(pending_path=PENDING_PATH, pending_target_xy=None)

    behavior = ExplorationBehavior()
    observation = _make_observation()
    fake_services = _FakePlannerServices(target=None)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind != "ACCEPT_PENDING_PATH"
    assert decision.kind == "HOLD"
    assert decision.reason == "frontier reached; no valid next frontier available"


def test_pending_target_is_not_accepted_if_pending_path_missing():
    agent = _make_agent_with_reached_route(pending_path=None, pending_target_xy=PENDING_TARGET)

    behavior = ExplorationBehavior()
    observation = _make_observation()
    fake_services = _FakePlannerServices(target=None)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind != "ACCEPT_PENDING_PATH"
    assert decision.kind == "HOLD"
    assert decision.reason == "frontier reached; no valid next frontier available"


def test_pending_target_is_not_accepted_if_pending_path_empty():
    agent = _make_agent_with_reached_route(pending_path=[], pending_target_xy=PENDING_TARGET)

    behavior = ExplorationBehavior()
    observation = _make_observation()
    fake_services = _FakePlannerServices(target=None)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind != "ACCEPT_PENDING_PATH"
    assert decision.kind == "HOLD"
