"""
Regression tests for the "exploration exhausted too early" bug.

Manual Office.sim telemetry, after NavigationSupervisor made the
state-machine coherent, showed exploration ending around 20% explored:

    [NAV] HOLD reason="no reachable frontier candidates remain"
    [NAV] HOLD reason="exploration exhausted: no reachable frontier candidates"

The console state was no longer buggy/looping -- just too conservative:
normal frontier selection failing once was treated as equivalent to
"nothing useful remains anywhere", with no attempt to fall back to a
nearby, previously-known-reachable point first.

Fix: RecoveryPolicy, a small deterministic fallback-target proposer, tried
in ExplorationBehavior.update()'s step 6 ("no path — need first plan")
exactly where next_target is None -- before registering the attempt as a
failure and holding. RobotAgent gained one new bounded field,
recent_safe_positions (a deque(maxlen=20) of positions recorded whenever a
route was successfully assigned/accepted), which RecoveryPolicy searches
most-recent-first for a candidate that is not the robot's current position,
not within goal_tolerance, not recently failed, and not the active/pending
route goal.

Recovery only changes what happens when frontier selection already failed;
it never touches frontier scoring, A*, or the exhaustion budget mechanics
themselves. If a recovery target's route also fails to plan, it flows
through the existing apply_route_result() failure path exactly like any
other exploration target -- register_exploration_failure() still applies,
so recovery does not create a way to dodge exhaustion forever, only a
bounded, deterministic extra attempt before declaring it.

Recovery memory itself went through two designs -- see the "round 1"/"round 2"
comment blocks below for the map_signature-scoped attempt and why it still
ping-ponged, and the episode-scoped design that replaced it.

Tests 1-4 exercise RecoveryPolicy directly (pure, no engine/Qt). The rest
exercise ExplorationBehavior.update() (and RobotAgent's recovery-memory
helpers) with a _FakePlannerServices stub, matching the pattern in
test_frontier_reached_target_rejection.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.navigation.recovery_policy import RecoveryPolicy
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.observation import RobotObservation


RECENT_SAFE_POSE = (1.0, 1.0)
OLDER_SAFE_POSE = (0.5, 0.5)
FAR_TARGET = (9.0, 9.0)


# ---------------------------------------------------------------------------
# 1-3. RecoveryPolicy direct tests.
# ---------------------------------------------------------------------------


def test_recovery_policy_returns_recent_safe_pose_when_frontiers_unavailable():
    robot_xy = (5.0, 5.0)
    recent_safe_positions = deque([OLDER_SAFE_POSE, RECENT_SAFE_POSE], maxlen=20)

    target = RecoveryPolicy.propose_recovery_target(
        robot_xy, goal_tolerance=0.25, recent_safe_positions=recent_safe_positions
    )

    assert target is not None
    assert target != robot_xy
    assert ((target[0] - robot_xy[0]) ** 2 + (target[1] - robot_xy[1]) ** 2) ** 0.5 > 0.25
    # Most-recent-first: RECENT_SAFE_POSE was appended last.
    assert target == RECENT_SAFE_POSE


def test_recovery_policy_ignores_targets_too_close_to_robot():
    robot_xy = (1.02, 1.01)  # within goal_tolerance of RECENT_SAFE_POSE
    recent_safe_positions = deque([OLDER_SAFE_POSE, RECENT_SAFE_POSE], maxlen=20)

    target = RecoveryPolicy.propose_recovery_target(
        robot_xy, goal_tolerance=0.25, recent_safe_positions=recent_safe_positions
    )

    assert target == OLDER_SAFE_POSE, "the too-close candidate must be skipped, falling back to the older one"


def test_recovery_policy_ignores_recently_failed_targets():
    robot_xy = (5.0, 5.0)
    recent_safe_positions = deque([OLDER_SAFE_POSE, RECENT_SAFE_POSE], maxlen=20)

    target = RecoveryPolicy.propose_recovery_target(
        robot_xy,
        goal_tolerance=0.25,
        recent_safe_positions=recent_safe_positions,
        recently_failed_targets=[RECENT_SAFE_POSE],
    )

    assert target == OLDER_SAFE_POSE, "a recently-failed candidate must be skipped"


def test_recovery_policy_returns_none_when_no_candidate_qualifies():
    robot_xy = (5.0, 5.0)
    recent_safe_positions = deque([OLDER_SAFE_POSE, RECENT_SAFE_POSE], maxlen=20)

    target = RecoveryPolicy.propose_recovery_target(
        robot_xy,
        goal_tolerance=0.25,
        recent_safe_positions=recent_safe_positions,
        recently_failed_targets=[RECENT_SAFE_POSE, OLDER_SAFE_POSE],
    )

    assert target is None


# ---------------------------------------------------------------------------
# Recovery oscillation ("ping-pong") regression -- round 1.
#
# Manual Office.sim telemetry after the recovery fix above:
#
#   frontier reached; no valid next frontier available
#   recovery: trying recent safe target before exhaustion
#   [ROUTE ok] goal=(-7.25,0.25)
#   ... (later)
#   frontier reached; no valid next frontier available
#   recovery: trying recent safe target before exhaustion
#   [ROUTE ok] goal=(-3.75,1.75)
#   ... repeating, explored stuck around 21.8%
#
# Root cause: RecoveryPolicy picked recent_safe_positions most-recent-first
# with no memory of what it had already proposed. Each successful recovery
# route records the robot's new position as a fresh recent_safe_positions
# entry (assign_path()/accept_pending_path()), so after reaching B, A
# becomes "the most recent safe position not at the robot's current spot"
# again -- and vice versa after reaching A. Two targets ping-pong forever.
#
# First fix attempt: RobotAgent.recent_recovery_targets, scoped to a single
# map_signature. This passed its own tests but manual Office.sim STILL
# ping-ponged -- see the next section below for why, and the actual fix.
# ---------------------------------------------------------------------------


def test_recovery_policy_skips_recent_recovery_targets():
    robot_xy = (5.0, 5.0)
    target_a = (0.5, 0.5)
    target_b = (1.0, 1.0)
    # target_a is the most recent (last-appended) safe position, so without
    # the recovery-memory exclusion it would be picked first.
    recent_safe_positions = deque([target_b, target_a], maxlen=20)

    target = RecoveryPolicy.propose_recovery_target(
        robot_xy,
        goal_tolerance=0.25,
        recent_safe_positions=recent_safe_positions,
        recent_recovery_targets=[target_a],
    )

    assert target == target_b, "an already-attempted recovery target must be skipped in favor of a valid one"


def test_recovery_policy_returns_none_when_all_recent_safe_positions_already_used_for_recovery():
    robot_xy = (5.0, 5.0)
    target_a = (0.5, 0.5)
    target_b = (1.0, 1.0)
    recent_safe_positions = deque([target_a, target_b], maxlen=20)

    target = RecoveryPolicy.propose_recovery_target(
        robot_xy,
        goal_tolerance=0.25,
        recent_safe_positions=recent_safe_positions,
        recent_recovery_targets=[target_a, target_b],
    )

    assert target is None


# ---------------------------------------------------------------------------
# Recovery oscillation ("ping-pong") regression -- round 2 (the actual fix).
#
# Manual Office.sim telemetry after round 1's map_signature-scoped fix
# landed and passed all its own tests:
#
#   recovery: trying recent safe target before exhaustion
#   [ROUTE ok] goal=(-4.25,1.75)   mapped_obs=1775
#   ...
#   recovery: trying recent safe target before exhaustion
#   [ROUTE ok] goal=(-7.25,0.25)   mapped_obs=1784
#   ...
#   recovery: trying recent safe target before exhaustion
#   [ROUTE ok] goal=(-4.25,1.75)   mapped_obs=1784
#   ... alternating, explored flat around 21.8%-22.4%
#
# Root cause: map_signature (len(mapped_obstacle_points)) is the wrong
# reset trigger. In a live run it changes on nearly every tick from
# routine sensor updates picking up a handful of new, often unrelated
# boundary samples -- 1775 -> 1784 above happened from ordinary continued
# mapping, not from anything that made (-4.25,1.75) newly reachable or
# useful. reset_recovery_memory_if_map_changed() wiped recent_recovery_targets
# on essentially every recovery cycle, so an "already tried" target became
# "fresh" again within a tick or two -- reproducing the exact ping-pong
# the memory was built to prevent.
#
# Fix: recovery memory is now scoped to a recovery EPISODE, not to raw
# map_signature. RobotAgent.mark_recovery_target_attempted()/
# recovery_targets() no longer take or depend on map_signature at all --
# the memory is cleared only by clear_recovery_memory(), called from
# ExplorationBehavior.update() at the one and only point that means
# "exploration is making real progress again": a normal frontier target
# (from _pick_next_target(), not RecoveryPolicy) was found.
# ---------------------------------------------------------------------------


def test_recovery_memory_does_not_reset_on_raw_map_signature_change():
    agent = _make_agent(position=(5.0, 5.0))
    target_a = (0.5, 0.5)
    target_b = (1.0, 1.0)
    agent.recent_safe_positions.extend([target_a, target_b])
    agent.mark_recovery_target_attempted(target_a)

    behavior = ExplorationBehavior()
    fake_services = _FakePlannerServices(target=None)  # no normal frontier target appears

    # A routine sensor update between attempts: map_signature moves from
    # whatever it implicitly was (0, from the empty mapped_obstacle_points
    # default) to 101 -- a raw count change, unrelated to target_a's
    # reachability.
    observation = _make_observation(
        robot_xy=agent.position, mapped_obstacle_points=[(0.0, 0.0)] * 101, current_time=10.0
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind != "REQUEST_PLAN" or decision.target != target_a, (
        "a raw map_signature change must not make an already-attempted recovery target eligible again"
    )
    assert target_a in agent.recovery_targets(), "recovery memory must not be cleared by a raw map_signature change"


def test_recovery_memory_clears_when_normal_frontier_target_is_found():
    agent = _make_agent(position=(5.0, 5.0))
    target_a = (0.5, 0.5)
    target_b = (1.0, 1.0)
    agent.mark_recovery_target_attempted(target_a)
    agent.mark_recovery_target_attempted(target_b)
    assert agent.recovery_targets() == [target_a, target_b]

    behavior = ExplorationBehavior()
    fake_services = _FakePlannerServices(target=FAR_TARGET)  # normal frontier selection succeeds
    observation = _make_observation(robot_xy=agent.position, current_time=10.0)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == FAR_TARGET
    assert agent.recovery_targets() == [], "a normal frontier success must end the recovery episode"


def test_recovery_does_not_ping_pong_when_map_signature_changes_slightly():
    target_a = (0.5, 0.5)
    target_b = (1.0, 1.0)
    agent = _make_agent(position=(5.0, 5.0))
    agent.recent_safe_positions.extend([target_a, target_b])

    behavior = ExplorationBehavior()
    fake_services = _FakePlannerServices(target=None)

    proposed = []
    decision = None
    for tick in range(6):
        # map_signature drifts up by a small amount only while candidates
        # are still being proposed (ticks 0-1), mirroring the routine
        # sensor updates seen in the manual log -- this must NOT reset
        # recovery memory or make target_a/target_b eligible again. Once
        # both are exhausted, hold map_signature steady: this test is
        # about recovery-target memory specifically, not about
        # exploration_exhausted()'s own separate (and unrelated to this
        # fix) exact-map_signature-match bookkeeping for the failure
        # budget, which is a pre-existing mechanism this round does not
        # touch and would otherwise itself keep resetting on every tick's
        # signature change, unrelated to what this test is checking.
        map_signature = 100 + min(tick, 1)
        observation = _make_observation(
            robot_xy=agent.position,
            mapped_obstacle_points=[(0.0, 0.0)] * map_signature,
            current_time=float(tick) * 2.0,
        )
        decision = behavior.update(agent, observation, fake_services)
        if decision.kind == "REQUEST_PLAN":
            proposed.append(decision.target)
        elif decision.kind == "HOLD":
            agent.invalidate_route(reason=decision.reason)

    assert len(proposed) == len(set(proposed)), (
        f"a recovery target was proposed more than once despite slight map_signature drift: {proposed}"
    )
    assert decision.kind == "HOLD"
    assert "exhausted" in decision.reason


def test_recovery_route_success_does_not_create_infinite_new_recovery_candidates():
    target_a = (0.5, 0.5)
    target_b = (1.0, 1.0)
    agent = _make_agent(position=(5.0, 5.0))
    agent.recent_safe_positions.extend([target_a, target_b])

    behavior = ExplorationBehavior()
    fake_services = _FakePlannerServices(target=None)
    map_signature = 3  # held constant: map_signature is no longer what bounds recovery
    current_time = 0.0
    decisions = []

    # Enough ticks to cover: each recovery success also appends the
    # position the robot moved FROM as a fresh recent_safe_positions entry
    # (assign_path()'s own bookkeeping, unrelated to this fix) -- so the
    # candidate pool can grow by a few extra points beyond target_a/target_b
    # before recovery memory (bounded to the same _RECENT_SAFE_POSITION_LIMIT
    # as recent_safe_positions, so it can never be outpaced) catches up and
    # every candidate is exhausted.
    for _ in range(10):
        observation = _make_observation(
            robot_xy=agent.position,
            mapped_obstacle_points=[(0.0, 0.0)] * map_signature,
            current_time=current_time,
        )
        decision = behavior.update(agent, observation, fake_services)
        decisions.append(decision)
        if decision.kind == "REQUEST_PLAN":
            # Simulate the engine successfully committing this recovery
            # route: assign_path() records the route's start position into
            # recent_safe_positions (unrelated existing bookkeeping) and
            # resets consecutive_exploration_failures, exactly like any
            # other successful route assignment would.
            agent.assign_path(target=decision.target, waypoints=[decision.target], planner_reason="recovery route")
            agent.set_position(decision.target)
        elif decision.kind == "HOLD":
            # Simulate apply_navigation_decision()'s HOLD handler, which
            # always calls invalidate_route() -- without this, a stale
            # active_path_goal_xy from the prior recovery route would keep
            # step 3 ("frontier reached") re-firing forever instead of ever
            # reaching step 6 (where recovery memory is checked) again.
            agent.invalidate_route(reason=decision.reason)
        current_time += 2.0  # past _FAILURE_RETRY_COOLDOWN each iteration

    assert decisions[-1].kind == "HOLD"
    assert "exhausted" in decisions[-1].reason, (
        "recovery must not be able to loop forever just because assign_path() keeps generating "
        "fresh candidates from its own route starts -- it must still converge to exhausted"
    )
    proposed_targets = [d.target for d in decisions if d.kind == "REQUEST_PLAN"]
    assert len(proposed_targets) == len(set(proposed_targets)), (
        f"a recovery target was proposed more than once: {proposed_targets}"
    )


# ---------------------------------------------------------------------------
# 4-6. ExplorationBehavior.update() integration.
# ---------------------------------------------------------------------------


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


def _make_agent(position=(5.0, 5.0)) -> RobotAgent:
    return RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")


def _make_observation(**overrides) -> RobotObservation:
    defaults = dict(
        robot_xy=(5.0, 5.0),
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


def test_exploration_behavior_attempts_recovery_before_exhaustion():
    agent = _make_agent(position=(5.0, 5.0))
    agent.recent_safe_positions.append(RECENT_SAFE_POSE)

    behavior = ExplorationBehavior()
    observation = _make_observation(robot_xy=agent.position, current_time=1.0)
    fake_services = _FakePlannerServices(target=None)  # normal frontier selection fails

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == RECENT_SAFE_POSE
    assert "recovery" in decision.reason.lower()
    assert agent.consecutive_exploration_failures == 0, (
        "a successful recovery attempt must not be counted as an exploration failure"
    )


def test_exploration_behavior_exhausts_only_after_recovery_fails():
    agent = _make_agent(position=(5.0, 5.0))
    # No recent_safe_positions at all -> RecoveryPolicy has nothing to offer.
    assert list(agent.recent_safe_positions) == []
    # One failure short of the budget; this call's failure will push it over.
    agent.consecutive_exploration_failures = RobotAgent._EXPLORATION_FAILURE_BUDGET - 1

    behavior = ExplorationBehavior()
    map_signature = 7
    observation = _make_observation(
        robot_xy=agent.position, mapped_obstacle_points=[(0.0, 0.0)] * map_signature, current_time=1.0
    )
    fake_services = _FakePlannerServices(target=None)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "HOLD"
    assert "exploration exhausted" in decision.reason
    assert agent.exploration_exhausted(map_signature=map_signature)


def test_recovery_attempt_resets_when_new_map_information_arrives():
    agent = _make_agent(position=(5.0, 5.0))
    old_map_signature = 7
    for _ in range(RobotAgent._EXPLORATION_FAILURE_BUDGET):
        agent.register_exploration_failure(map_signature=old_map_signature)
    assert agent.exploration_exhausted(map_signature=old_map_signature)

    behavior = ExplorationBehavior()
    new_map_signature = 12
    observation = _make_observation(
        robot_xy=agent.position, mapped_obstacle_points=[(0.0, 0.0)] * new_map_signature, current_time=5.0
    )
    fake_services = _FakePlannerServices(target=FAR_TARGET)

    decision = behavior.update(agent, observation, fake_services)

    assert len(fake_services.calls) == 1, "new map information must let frontier selection be attempted again"
    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == FAR_TARGET
    assert not agent.exploration_exhausted(map_signature=new_map_signature)
