"""
Regression tests for the single-robot exploration "stuck target" bug.

Confirmed root cause (see conversation / debug report):
    In single-robot exploration mode, when a route planning attempt fails,
    engine.apply_route_result() cleared self.current_exploration_target (an
    engine-level field) but called RobotAgent.invalidate_route(), which does
    NOT clear agent.exploration_target_xy. ExplorationBehavior then had no
    active waypoint/path, fell through to desired_target_from_mode(), which
    still returned the same failed exploration_target_xy, and immediately
    requested another plan for it. FoV-aware hysteresis kept biasing toward
    that stuck target, producing a repeated planner-failure loop.

Fix: RobotAgent gained a dedicated
RobotAgent.invalidate_failed_exploration_route() method that clears the
route AND exploration_target_xy; engine.apply_route_result() now calls it
on the exploration-mode planner-failure path instead of plain
invalidate_route(). invalidate_route() itself was left unchanged, since
set_exploration_target() relies on it NOT clearing exploration_target_xy
(it assigns the new target, then calls invalidate_route() to discard the
now-stale path for the previous target).

These tests exercise RobotAgent and ExplorationBehavior directly (no Qt,
no canvas, no engine.py) and describe the desired contract: they failed
against the pre-fix implementation and pass now that the stuck target is
cleared.
"""
from __future__ import annotations

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.planner_services import PlannerServices


FAILED_TARGET = (7.75, -4.75)


def _make_agent(position=(0.0, 0.0)) -> RobotAgent:
    return RobotAgent(
        robot_id=0,
        position=position,
        planner_mode="FoV-aware directional frontier",
    )


def _make_observation(**overrides) -> RobotObservation:
    """Minimal RobotObservation for exercising ExplorationBehavior.update().

    belief_map/planning_grid are left as None: the "no active path" branch of
    ExplorationBehavior.update() (step 6) never touches them, it only reads
    agent.desired_target_from_mode(), so a real BeliefMap is not needed here.
    """
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


def test_robot_agent_route_failure_can_clear_exploration_target():
    """RobotAgent's route-failure/invalidation path must clear the failed
    exploration target, not just the active/pending path.

    This mirrors exactly what engine.apply_route_result() does on a planner
    failure in exploration mode:
        agent.invalidate_failed_exploration_route(reason=f"planner failed: {reason}")

    Plain RobotAgent.invalidate_route() only clears waypoints,
    active_path_goal_xy, active_path_mode, pending_path, and
    pending_target_xy -- it never touches exploration_target_xy (by design:
    set_exploration_target() relies on that). invalidate_failed_exploration_route()
    is the dedicated method for the planner-failure case: it clears
    everything invalidate_route() clears, plus exploration_target_xy.
    """
    agent = _make_agent()
    agent.set_exploration_target(FAILED_TARGET, reason="frontier reached; requesting next frontier")
    agent.assign_path(
        target=FAILED_TARGET,
        waypoints=[FAILED_TARGET],
        planner_reason="initial route",
    )

    # Simulate the exact call engine.apply_route_result() makes when the
    # planner fails in exploration mode (engine.py, apply_route_result()).
    agent.invalidate_failed_exploration_route(reason="planner failed: no path found")

    # Active/pending path state must be cleared (this part already passes).
    assert not agent.waypoints.has_path(), "active waypoints should be cleared after a planner failure"
    assert agent.pending_path is None, "pending path should be cleared after a planner failure"
    assert agent.active_path_goal_xy is None, "active_path_goal_xy should be cleared after a planner failure"

    # The failed frontier target itself must not survive the failure, or the
    # exploration loop will immediately re-request the same doomed target.
    assert agent.exploration_target_xy is None, (
        "exploration_target_xy must be cleared after a planner failure, otherwise "
        "desired_target_from_mode() keeps returning the same unreachable target"
    )
    assert agent.desired_target_from_mode() is None, (
        "desired_target_from_mode() must not return a target that just failed to plan"
    )


def test_exploration_behavior_does_not_replan_same_failed_target_immediately():
    """After a planner failure, the very next ExplorationBehavior.update()
    call must not immediately re-request a plan for the same failed target.

    Acceptable outcomes for the tick right after a failure:
        - HOLD (no valid target available yet)
        - REQUEST_PLAN for a *different* target (cooldown expired / new
          target selected)
    Not acceptable:
        - REQUEST_PLAN with the exact same target that just failed.

    This reproduces the observed console loop:
        [NAV] kind=REQUEST_PLAN ... reason='frontier reached; requesting next frontier'
        Planner failed in exploration mode: no path found. Holding current position...
        [NAV] kind=REQUEST_PLAN ... reason='no active path; requesting initial frontier plan'
        Planner failed in exploration mode: no path found. Holding current position...
        ... (repeats)

    Before the fix, step 6 of ExplorationBehavior.update() ("no active path
    -- need first plan") read agent.desired_target_from_mode(), which still
    returned FAILED_TARGET, and emitted REQUEST_PLAN(FAILED_TARGET) on the
    very next tick. Now that the planner-failure path clears
    exploration_target_xy, desired_target_from_mode() returns None and the
    agent HOLDs instead.
    """
    agent = _make_agent()
    agent.set_exploration_target(FAILED_TARGET, reason="frontier reached; requesting next frontier")
    agent.assign_path(
        target=FAILED_TARGET,
        waypoints=[FAILED_TARGET],
        planner_reason="initial route",
    )

    # Simulate the planner failure the way engine.apply_route_result() does.
    agent.invalidate_failed_exploration_route(reason="planner failed: no path found")

    observation = _make_observation(robot_xy=agent.position, current_time=1.0)
    decision = agent.behavior.update(agent, observation, PlannerServices())

    same_target_requested = (
        decision.kind == "REQUEST_PLAN" and decision.target == FAILED_TARGET
    )
    assert not same_target_requested, (
        f"ExplorationBehavior.update() re-requested a plan for the exact target "
        f"that just failed ({FAILED_TARGET!r}); decision={decision!r}"
    )
