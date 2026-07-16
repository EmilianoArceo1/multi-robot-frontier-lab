"""
Regression tests for a stale pending path surviving a safety replan / route
repair and later overwriting the repaired route.

Manual Office.sim log sequence:

    REPLAN_FOR_SAFETY reason="active segment blocked" active=(5.25,3.25)
        path_goal=(7.75,3.25) pending=(1.25,-0.75)
    [ROUTE ok] start=(6.57,3.58) goal=(7.75,3.25)
    NAV ACCEPT_PENDING_PATH ... active=(7.75,3.25) path_goal=(7.75,3.25)
        pending=(1.25,-0.75)
    Next state:
    target=(7.75,3.25) path_goal=(1.25,-0.75) wp=1/4

Root cause: engine.apply_navigation_decision()'s REPLAN_FOR_SAFETY branch
(the single-robot "if allowed:" path) and simulation_step()'s route_affected
branch both called replan_after_new_information() to repair the route, but
neither ever cleared agent.pending_path/pending_target_xy first. A prefetch
computed for a completely different, unrelated target under the OLD route
context survived the repair untouched. Once the repaired route was accepted
(agent.assign_path() set active_path_goal_xy to the repair goal), the next
tick's ExplorationBehavior.update() step 2 ("pending path ready -- should we
switch?") found the stale pending path still sitting there, was close
enough to the just-repaired route's first waypoint, and promoted it --
RobotAgent.accept_pending_path() blindly overwrote active_path_goal_xy with
the stale, unrelated pending_target_xy. The robot then tracked a waypoint
list belonging to one target while active_path_goal_xy pointed at another,
producing predicted-collision / first-segment-blocked churn and premature
exploration exhaustion (worse than before the frontier-exhaustion fix, since
the robot was now being sent into inconsistent, effectively invalid state).

Fix (RobotAgent):
    - mark_pending_path_requested(target): stamps pending_target_xy plus
      the route context a prefetch was requested under
      (pending_path_route_generation = current route_generation,
      pending_path_created_for_active_goal = current active_path_goal_xy).
      Used everywhere a prefetch is launched (engine.request_prefetch_route_async()).
    - invalidate_pending_path(reason): discards pending_path/pending_target_xy
      and the stamped context WITHOUT touching the active route. Called by
      engine.apply_navigation_decision()'s REPLAN_FOR_SAFETY branch and
      simulation_step()'s route_affected branch, before requesting the
      repair route.
    - accept_pending_path(): now rejects (via reject_pending_path()) instead
      of promoting a pending path whose stamped route_generation or
      active-goal context no longer matches the CURRENT agent state --
      belt-and-suspenders even if some other path forgets to call
      invalidate_pending_path() explicitly. A pending path with no stamped
      context (None, e.g. raw field assignment in a test) is treated as
      trusted, preserving existing behavior exactly.
    - invalidate_route()/reject_pending_path() also clear the new context
      fields, for consistency.

These tests exercise RobotAgent directly (pure, no engine/Qt) for the
agent-level contract, and the same lightweight duck-typed engine fake
test_recovery_rejects_reached_targets.py / test_frontier_exhaustion_recovery.py
use for the engine-boundary REPLAN_FOR_SAFETY behavior.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.telemetry import TelemetryLogger


PATH_GOAL_A = (7.75, 3.25)
STALE_TARGET_B = (1.25, -0.75)
NEW_PATH_GOAL_C = (2.0, 2.0)


def _make_agent(position=(6.57, 3.58)) -> RobotAgent:
    return RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")


def _make_observation(**overrides) -> RobotObservation:
    defaults = dict(
        robot_xy=(6.57, 3.58),
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


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine(*, robot_xy=(6.57, 3.58)) -> SimpleNamespace:
    robot = _FakeRobot(x=robot_xy[0], y=robot_xy[1])
    agent = _make_agent(position=robot_xy)

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        config=SimpleNamespace(goal_tolerance=0.25),
        mapped_obstacle_points=[],
        simulation_time=5.0,
        console_logs=[],
        request_route_async_calls=[],
        current_exploration_target=None,
        exploration_targets=[],
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.canvas = SimpleNamespace(set_exploration_target=lambda target: fake.exploration_targets.append(target))
    fake.is_exploration_mode = lambda: True
    fake.runtime_agent = lambda robot_index=None: fake.agent
    fake.set_robot_goal_or_waypoints = lambda robot_obj, waypoints: robot_obj.set_waypoints(
        waypoints or [(robot_obj.x, robot_obj.y)]
    )

    def _spy_request_route_async(reason, *, target_override=None):
        fake.request_route_async_calls.append((reason, target_override))
        return False

    def _spy_replan_after_new_information(reason):
        fake.request_route_async_calls.append((reason, "route_repair_goal"))
        return True

    fake.request_route_async = _spy_request_route_async
    fake.replan_after_new_information = _spy_replan_after_new_information
    fake.safety_replan_cooldown_seconds = lambda: 1.5
    fake.apply_navigation_decision = SimulationControllerMixin.apply_navigation_decision.__get__(fake)
    fake._invalidate_prefetch_request = SimulationControllerMixin._invalidate_prefetch_request.__get__(fake)
    return fake


# ---------------------------------------------------------------------------
# A. REPLAN_FOR_SAFETY invalidates an existing pending path.
# ---------------------------------------------------------------------------


def test_safety_replan_invalidates_existing_pending_path():
    fake = _build_fake_engine()
    agent = fake.agent
    agent.assign_path(target=PATH_GOAL_A, waypoints=[(6.9, 3.4), PATH_GOAL_A], planner_reason="initial route")
    agent.mark_pending_path_requested(STALE_TARGET_B)
    agent.pending_path = [(2.0, 0.0), STALE_TARGET_B]
    assert agent.pending_path is not None

    decision = SimpleNamespace(
        kind="REPLAN_FOR_SAFETY",
        reason="active segment blocked",
        target=PATH_GOAL_A,
        brake=True,
        force_new_target=False,
    )

    SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, agent, decision)

    assert agent.pending_path is None, "the stale pending path must be discarded by the safety replan"
    assert agent.pending_target_xy is None
    # The safety-repair route (to the SAME path_goal A) is then accepted --
    # mirrors engine.apply_route_result() -> agent.assign_path().
    agent.assign_path(target=PATH_GOAL_A, waypoints=[(7.0, 3.3), PATH_GOAL_A], planner_reason="safety replan")

    # ACCEPT_PENDING_PATH cannot switch to the stale target B: there is
    # nothing pending to accept anymore.
    assert agent.accept_pending_path() is None
    assert agent.active_path_goal_xy == PATH_GOAL_A, (
        "the repaired route's path_goal must not be overwritten by the stale prefetch"
    )


# ---------------------------------------------------------------------------
# B. A route-repair (route_affected) replan invalidates an existing pending
#    path without touching the active route/target it is repairing.
# ---------------------------------------------------------------------------


def test_route_affected_replan_invalidates_existing_pending_path():
    agent = _make_agent()
    agent.set_exploration_target(PATH_GOAL_A, reason="test target")
    agent.assign_path(target=PATH_GOAL_A, waypoints=[(6.9, 3.4), PATH_GOAL_A], planner_reason="initial route")
    agent.mark_pending_path_requested(STALE_TARGET_B)
    agent.pending_path = [(2.0, 0.0), STALE_TARGET_B]

    # Mirrors simulation_step()'s route_affected branch: invalidate_pending_path()
    # is called before replan_after_new_information() repairs the route.
    agent.invalidate_pending_path(reason="route_affected: new obstacle affects current route")

    assert agent.pending_path is None
    assert agent.pending_target_xy is None
    # The active route/target being repaired must be left alone.
    assert agent.active_path_goal_xy == PATH_GOAL_A
    assert agent.exploration_target_xy == PATH_GOAL_A


# ---------------------------------------------------------------------------
# C. A planner failure that leads to HOLD clears the pending path too
#    (pre-existing invalidate_failed_exploration_route()/invalidate_route()
#    behavior -- covered here as part of the overall contract).
# ---------------------------------------------------------------------------


def test_planner_failure_hold_clears_pending_path():
    agent = _make_agent()
    agent.set_exploration_target(PATH_GOAL_A, reason="test target")
    agent.assign_path(target=PATH_GOAL_A, waypoints=[PATH_GOAL_A], planner_reason="initial route")
    agent.mark_pending_path_requested(STALE_TARGET_B)
    agent.pending_path = [STALE_TARGET_B]

    agent.invalidate_failed_exploration_route(
        reason="planner failed: no path found", current_time=1.0, map_signature=0
    )

    assert agent.pending_path is None
    assert agent.pending_target_xy is None


# ---------------------------------------------------------------------------
# D. Valid prefetch acceptance still works when no replan/invalidation
#    happens -- the normal, common-case flow must be unaffected.
# ---------------------------------------------------------------------------


def test_valid_prefetch_acceptance_still_works_without_replan_invalidation():
    agent = _make_agent(position=PATH_GOAL_A)
    agent.assign_path(target=PATH_GOAL_A, waypoints=[(7.5, 3.25), PATH_GOAL_A], planner_reason="initial route")
    # Prefetch requested under the current (unchanged) route context.
    agent.mark_pending_path_requested(STALE_TARGET_B)
    agent.pending_path = [(4.0, 1.0), STALE_TARGET_B]

    behavior = ExplorationBehavior()
    observation = _make_observation(robot_xy=PATH_GOAL_A, goal_tolerance=0.25, current_time=1.0)

    decision = behavior.update(agent, observation, _FailingPlannerServices())

    assert decision.kind == "ACCEPT_PENDING_PATH"

    waypoints = agent.accept_pending_path()
    assert waypoints == [(4.0, 1.0), STALE_TARGET_B]
    assert agent.active_path_goal_xy == STALE_TARGET_B
    assert agent.pending_path is None


class _FailingPlannerServices:
    """_pick_next_target() must not even be consulted when a pending path
    is accepted directly -- mirrors test_prefetch_pending_path_acceptance.py."""

    def select_exploration_target(self, **kwargs) -> ExplorationPlannerResult:
        raise AssertionError("the frontier planner must not be consulted when accepting a pending path")


# ---------------------------------------------------------------------------
# E. A pending path whose stamped context no longer matches the current
#    agent state is rejected, not accepted.
# ---------------------------------------------------------------------------


def test_pending_path_rejected_when_path_goal_context_changed():
    agent = _make_agent()
    agent.assign_path(target=PATH_GOAL_A, waypoints=[PATH_GOAL_A], planner_reason="initial route")
    agent.mark_pending_path_requested(STALE_TARGET_B)
    agent.pending_path = [STALE_TARGET_B]
    assert agent.pending_path_created_for_active_goal == PATH_GOAL_A

    # active_path_goal_xy changes (e.g. a safety-repair route was accepted
    # for a different goal) without anything explicitly invalidating the
    # pending path this time -- the belt-and-suspenders context check in
    # accept_pending_path() must still catch it.
    agent.active_path_goal_xy = NEW_PATH_GOAL_C

    result = agent.accept_pending_path()

    assert result is None, "a pending path created for a stale path_goal context must not be accepted"
    assert agent.pending_path is None
    assert agent.pending_target_xy is None
    assert agent.active_path_goal_xy == NEW_PATH_GOAL_C, "the current active path_goal must be left untouched"


def test_pending_path_rejected_when_route_generation_changed():
    """Same guard, exercised via route_generation instead of active_path_goal_xy
    (e.g. a route was re-accepted for the SAME target/goal point, which
    would not trip the active_path_goal_xy comparison alone)."""
    agent = _make_agent()
    agent.assign_path(target=PATH_GOAL_A, waypoints=[PATH_GOAL_A], planner_reason="initial route")
    agent.mark_pending_path_requested(STALE_TARGET_B)
    agent.pending_path = [STALE_TARGET_B]
    stamped_generation = agent.pending_path_route_generation

    # A new route is accepted for the exact same path_goal (route_generation
    # advances even though active_path_goal_xy ends up unchanged).
    agent.assign_path(target=PATH_GOAL_A, waypoints=[(7.0, 3.0), PATH_GOAL_A], planner_reason="safety replan")
    assert agent.route_generation != stamped_generation
    assert agent.active_path_goal_xy == PATH_GOAL_A  # unchanged -- the goal-context check alone would not catch this

    result = agent.accept_pending_path()

    assert result is None
    assert agent.pending_path is None
