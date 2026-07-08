"""
Regression tests for the single-robot safety-replan loop.

Observed bug (manual Office.sim run, after the exploration-recovery fixes in
test_robot_agent_route_failure.py / test_exploration_target_recovery.py):

    [NAV] kind=REPLAN_FOR_SAFETY brake=True reason='active segment blocked'
    safety replan: active segment blocked Replanning...
    Planner: A* ... FoV-aware directional frontier: kept current target by hysteresis ...
    R1 route assigned ...
    [NAV] kind=REPLAN_FOR_SAFETY brake=True reason='active segment blocked'
    ... (repeats)

Root causes:
    1. engine.replan_after_new_information() (the single-robot safety-replan
       path) had no throttle at all -- unlike
       engine.multi_safety_replan_allowed() for multi-robot -- so every tick
       with active_segment_blocked=True launched another planner request,
       even for the exact same blocked segment/target as last tick.
    2. Even when a fresh route was accepted, nothing validated that its
       first segment was actually clear before committing it as the active
       path. FoV-aware hysteresis could "keep current target" and A* could
       return *a* route whose first segment still crossed the same mapped
       obstacle sample that tripped active_segment_blocked in the first
       place, immediately re-triggering REPLAN_FOR_SAFETY next tick.

Fix:
    - RobotAgent.safety_replan_allowed() throttles identical
      (reason, target) safety replans per agent, mirroring
      engine.multi_safety_replan_allowed().
    - When a repeated identical safety replan is throttled in exploration
      mode, engine.apply_navigation_decision() now holds and marks the
      current exploration target failed via
      RobotAgent.invalidate_failed_exploration_route() (reusing the
      cooldown + blacklist recovery machinery from the exploration-target
      fixes), instead of retrying the same doomed target forever.
    - engine.route_first_segment_blocked() (a module-level, Qt-free
      function) validates a newly-planned route's first segment with the
      same CollisionChecker.check_segment_points() rule
      build_observation() uses for active_segment_blocked, before
      engine.apply_route_result() accepts the route as active.

These tests exercise RobotAgent, ExplorationBehavior, and the standalone
route_first_segment_blocked() helper directly -- no Qt, no canvas, no full
engine/GUI instantiation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.engine import route_first_segment_blocked
from robotics_sim.simulation.observation import RobotObservation


BLOCKED_TARGET = (7.75, -4.75)
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
# 1. Identical safety replans must be throttled.
# ---------------------------------------------------------------------------


def test_single_robot_safety_replan_is_throttled_for_same_blocked_segment():
    """The first REPLAN_FOR_SAFETY for a given (reason, target) signature is
    allowed; an identical one inside the cooldown is rejected; after the
    cooldown elapses, it is allowed again."""
    agent = _make_agent()
    reason = "active segment blocked"
    target = BLOCKED_TARGET
    cooldown = 0.5

    # First request for this signature: allowed.
    assert agent.safety_replan_allowed(
        reason=reason, target=target, current_time=0.0, cooldown=cooldown
    ), "the first safety replan request must be allowed"

    # Same signature, well within the cooldown: must be rejected -- this is
    # the "repeated request inside cooldown rejected/held without launching
    # another planner worker" contract. Before the fix there was no such
    # guard at all, so this call would (incorrectly) also return True.
    assert not agent.safety_replan_allowed(
        reason=reason, target=target, current_time=0.05, cooldown=cooldown
    ), "an identical safety replan within the cooldown must be throttled"

    # Still within cooldown, still rejected.
    assert not agent.safety_replan_allowed(
        reason=reason, target=target, current_time=0.4, cooldown=cooldown
    )

    # A *different* reason or target is a different signature and is not
    # throttled by the previous one.
    assert agent.safety_replan_allowed(
        reason="predicted collision", target=target, current_time=0.41, cooldown=cooldown
    ), "a different safety-replan reason must not be throttled by the previous signature"

    # Once the cooldown has elapsed for the original signature, it is
    # allowed again.
    assert agent.safety_replan_allowed(
        reason=reason, target=target, current_time=1.5, cooldown=cooldown
    ), "the same signature must be allowed again once the cooldown has elapsed"


# ---------------------------------------------------------------------------
# 2. A target that keeps producing a blocked segment must not be retried
#    forever -- it gets blacklisted and exploration recovers onto a fresh
#    target when one exists.
# ---------------------------------------------------------------------------


def test_single_robot_safety_replan_does_not_keep_same_blocked_target_forever():
    """Simulates what engine.apply_navigation_decision() does when the
    safety-replan throttle rejects a repeat: mark the current exploration
    target failed and let ExplorationBehavior recover onto a different
    target, instead of holding on the same blocked target forever."""
    agent = _make_agent()
    agent.set_exploration_target(BLOCKED_TARGET, reason="frontier reached; requesting next frontier")
    agent.assign_path(
        target=BLOCKED_TARGET,
        waypoints=[BLOCKED_TARGET],
        planner_reason="initial route",
    )

    reason = "active segment blocked"
    cooldown = 0.5

    # Tick 1: first safety replan for this target -- allowed, route gets
    # (re)computed elsewhere; exploration_target_xy is untouched here.
    assert agent.safety_replan_allowed(
        reason=reason, target=BLOCKED_TARGET, current_time=0.0, cooldown=cooldown
    )
    assert agent.exploration_target_xy == BLOCKED_TARGET

    # Tick 2: identical safety replan again, within the cooldown -- this is
    # the "replan repeats with same target/segment blocked" case. The
    # engine's REPLAN_FOR_SAFETY handler reacts to the rejected throttle by
    # marking the target failed (mirrors engine.py's
    # apply_navigation_decision REPLAN_FOR_SAFETY branch).
    allowed_again = agent.safety_replan_allowed(
        reason=reason, target=BLOCKED_TARGET, current_time=0.1, cooldown=cooldown
    )
    assert not allowed_again
    agent.invalidate_failed_exploration_route(
        reason=f"repeated safety replan: {reason}",
        current_time=0.1,
    )

    assert agent.exploration_target_xy is None, (
        "the repeatedly-blocked target must be cleared, not kept forever"
    )
    assert BLOCKED_TARGET in agent.recently_failed_exploration_targets(
        current_time=0.1, cooldown=5.0
    ), "the repeatedly-blocked target must be remembered so it is excluded from re-selection"

    # Once the exploration retry cooldown elapses, a fresh selection must
    # prefer a different reachable target (B) over the blacklisted one (A).
    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    behavior = ExplorationBehavior()
    observation = _make_observation(
        robot_xy=agent.position,
        current_time=0.1 + behavior._FAILURE_RETRY_COOLDOWN + 0.1,
    )
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == ALTERNATE_TARGET
    assert decision.target != BLOCKED_TARGET
    excluded = fake_services.calls[0]["excluded_targets"]
    assert BLOCKED_TARGET in excluded


# ---------------------------------------------------------------------------
# 3. A newly-planned route must not be accepted if its first segment is
#    already unsafe by the same rule build_observation() uses.
# ---------------------------------------------------------------------------


def test_single_robot_rejects_route_with_blocked_initial_segment_if_feasible():
    """route_first_segment_blocked() must flag a route whose first segment
    crosses a mapped obstacle point within the robot's safety radius, using
    the exact same CollisionChecker.check_segment_points() rule
    build_observation() uses to compute active_segment_blocked -- and must
    not flag a route whose first segment is clear."""
    collision_checker = CollisionChecker()
    robot_radius = 0.3

    start_xy = (0.0, 0.0)
    blocked_target = (2.0, 0.0)
    # A mapped obstacle point sitting on the straight-line segment, well
    # within the robot's safety radius.
    obstacle_points = [(1.0, 0.0)]

    assert route_first_segment_blocked(
        collision_checker, start_xy, blocked_target, obstacle_points, robot_radius
    ), "a route whose first segment crosses a mapped obstacle point must be flagged as blocked"

    clear_target = (2.0, 5.0)  # far from the obstacle point
    assert not route_first_segment_blocked(
        collision_checker, start_xy, clear_target, obstacle_points, robot_radius
    ), "a route whose first segment is clear of mapped obstacle points must not be flagged"

    # No collision checker or no target: nothing to validate, never blocks.
    assert not route_first_segment_blocked(
        None, start_xy, blocked_target, obstacle_points, robot_radius
    )
    assert not route_first_segment_blocked(
        collision_checker, start_xy, None, obstacle_points, robot_radius
    )
