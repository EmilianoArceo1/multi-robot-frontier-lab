"""
Regression tests for recovery still accepting a target already effectively
reached by the robot.

Manual Office.sim telemetry, after route-repair replanning was fixed to
preserve the current path_goal:

    [ROUTE ok] R1 start=(2.68,-2.18) goal=(2.75,-2.25) wp=1 length=0.10

goal_tolerance=0.25, and distance((2.68,-2.18), (2.75,-2.25)) ~= 0.099, well
inside it -- this route should never have been requested.

Investigation: RecoveryPolicy.propose_recovery_target() already rejects a
candidate within goal_tolerance of the CURRENT robot_xy at selection time
(tests 1, 2, 5 below confirm this still works), and
NavigationSupervisor.normalize_decision() already rejects a REQUEST_PLAN
whose target is within goal_tolerance of robot_xy at the engine boundary,
regardless of the decision's reason string (test 3 confirms this too, and
is reason-agnostic by design -- it doesn't special-case "recovery:").

Both of those checks compare the candidate against the robot's position AT
THE MOMENT OF SELECTION/APPLICATION. Neither is a mechanism the robot's
own exact resting pose can fail to trip on their own -- so the real gap
found here (test 4) is different: recent_safe_positions can contain the
point the robot JUST finished a route at (active_path_goal_xy, recorded
into recent_safe_positions by a LATER, unrelated assign_path() call made
from that same resting position -- see RobotAgent.assign_path()'s own
bookkeeping). Distance-based exclusion alone only protects against
selecting a target close to wherever the robot happens to be standing
RIGHT NOW; it says nothing about "this exact point was just the
destination of the route that ended here", which should be excluded on
its own terms, independent of the robot's current precise distance to it
(e.g. after the robot has since drifted, or after recovery is evaluated a
few ticks later than the moment path_goal was reached).

Fix: RobotAgent gains one small field, last_completed_path_goal_xy,
recorded by ExplorationBehavior.update() step 3 whenever path_goal_reached
fires (the robot's active_path_goal_xy right before it is cleared for the
next selection cycle). RecoveryPolicy.propose_recovery_target() gained a
new optional last_completed_path_goal parameter, checked the same way as
active_path_goal/pending_target.

Tests 1-2, 5 exercise RecoveryPolicy/ExplorationBehavior directly (pure, no
engine/Qt). Test 3 exercises apply_navigation_decision() via a minimal
duck-typed engine fake. Test 4 is the one that isolates the actual gap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.navigation.recovery_policy import RecoveryPolicy
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.telemetry import TelemetryLogger


ROBOT_XY = (2.68, -2.18)
ALREADY_REACHED_TARGET = (2.75, -2.25)  # distance from ROBOT_XY ~= 0.099
FAR_CANDIDATE = (9.0, 9.0)
GOAL_TOLERANCE = 0.25


# ---------------------------------------------------------------------------
# 1. RecoveryPolicy directly: a candidate within goal_tolerance of the
#    robot's CURRENT position must be rejected.
# ---------------------------------------------------------------------------


def test_recovery_policy_rejects_target_within_goal_tolerance_of_current_pose():
    target = RecoveryPolicy.propose_recovery_target(
        ROBOT_XY,
        GOAL_TOLERANCE,
        recent_safe_positions=[ALREADY_REACHED_TARGET],
    )
    assert target is None


# ---------------------------------------------------------------------------
# 2. ExplorationBehavior.update(): with no normal frontier and the only
#    recovery candidate already reached, the decision must be HOLD, not a
#    REQUEST_PLAN to a near-zero-length route.
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


def _make_agent(position=ROBOT_XY) -> RobotAgent:
    return RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")


def _make_observation(**overrides) -> RobotObservation:
    defaults = dict(
        robot_xy=ROBOT_XY,
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
        goal_tolerance=GOAL_TOLERANCE,
        sensor_range=2.5,
        final_goal_xy=None,
    )
    defaults.update(overrides)
    return RobotObservation(**defaults)


def test_exploration_behavior_does_not_request_recovery_route_to_reached_target():
    agent = _make_agent()
    agent.recent_safe_positions.append(ALREADY_REACHED_TARGET)

    behavior = ExplorationBehavior()
    observation = _make_observation()
    fake_services = _FakePlannerServices(target=None)  # no normal frontier

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "HOLD"
    assert decision.kind != "REQUEST_PLAN"


# ---------------------------------------------------------------------------
# 3. Engine boundary: a REQUEST_PLAN decision (whatever its reason) whose
#    target is already within goal_tolerance of the robot must never reach
#    request_route_async().
# ---------------------------------------------------------------------------


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine(*, position=ROBOT_XY, goal_tolerance=GOAL_TOLERANCE) -> SimpleNamespace:
    robot = _FakeRobot(x=position[0], y=position[1])
    agent = _make_agent(position=position)

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        config=SimpleNamespace(goal_tolerance=goal_tolerance),
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

    fake.request_route_async = _spy_request_route_async
    fake.apply_navigation_decision = SimulationControllerMixin.apply_navigation_decision.__get__(fake)
    fake._invalidate_prefetch_request = SimulationControllerMixin._invalidate_prefetch_request.__get__(fake)
    return fake


def test_runtime_blocks_recovery_request_plan_to_already_reached_target():
    fake = _build_fake_engine()
    decision = SimpleNamespace(
        kind="REQUEST_PLAN",
        reason="recovery: trying recent safe target before exhaustion",
        target=ALREADY_REACHED_TARGET,
        brake=False,
        force_new_target=True,
    )

    should_brake = SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    assert fake.request_route_async_calls == [], "no route request may be launched for an already-reached target"
    assert should_brake is False
    assert fake.robot.waypoints == [ROBOT_XY], "the robot must hold at its current position"


# ---------------------------------------------------------------------------
# 4. The actual gap: a point that was just the destination of the route
#    that ended here (recorded via a later, unrelated assign_path() call
#    from that same resting position -- see RobotAgent.assign_path()) must
#    be excluded from recovery on its own terms, independent of the
#    robot's current precise distance to it.
# ---------------------------------------------------------------------------


def test_recovery_does_not_accept_last_completed_path_goal_as_recovery_target():
    agent = _make_agent(position=ALREADY_REACHED_TARGET)
    # The robot reaches its path_goal: ExplorationBehavior step 3 records it
    # as last_completed_path_goal_xy before clearing state for the next cycle.
    agent.assign_path(
        target=ALREADY_REACHED_TARGET, waypoints=[ALREADY_REACHED_TARGET], planner_reason="initial route"
    )
    behavior = ExplorationBehavior()
    reached_observation = _make_observation(robot_xy=ALREADY_REACHED_TARGET, current_time=1.0)
    fake_services = _FakePlannerServices(target=None)  # no frontier at the reached point either

    behavior.update(agent, reached_observation, fake_services)
    assert agent.last_completed_path_goal_xy == ALREADY_REACHED_TARGET

    # Later, recovery is evaluated with the robot elsewhere (e.g. it has
    # since settled a bit further away than goal_tolerance) -- distance
    # alone would no longer exclude ALREADY_REACHED_TARGET, but it must
    # still be rejected because it is the last completed path_goal.
    robot_xy_now = (ALREADY_REACHED_TARGET[0] + 1.0, ALREADY_REACHED_TARGET[1])
    assert (
        (robot_xy_now[0] - ALREADY_REACHED_TARGET[0]) ** 2 + (robot_xy_now[1] - ALREADY_REACHED_TARGET[1]) ** 2
    ) ** 0.5 > GOAL_TOLERANCE

    target = RecoveryPolicy.propose_recovery_target(
        robot_xy_now,
        GOAL_TOLERANCE,
        recent_safe_positions=agent.recent_safe_positions,
        last_completed_path_goal=agent.last_completed_path_goal_xy,
    )

    assert target != ALREADY_REACHED_TARGET, "the last completed path_goal must not be re-proposed for recovery"


# ---------------------------------------------------------------------------
# 5. A farther, otherwise-valid candidate must still be selectable when the
#    nearest one is already reached -- rejection must not turn into "never
#    recover anything".
# ---------------------------------------------------------------------------


def test_recovery_selects_farther_candidate_when_nearest_candidate_is_already_reached():
    target = RecoveryPolicy.propose_recovery_target(
        ROBOT_XY,
        GOAL_TOLERANCE,
        recent_safe_positions=[FAR_CANDIDATE, ALREADY_REACHED_TARGET],
    )
    assert target == FAR_CANDIDATE
