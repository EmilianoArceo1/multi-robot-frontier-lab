"""
Regression tests for accepting a "successful" route that does not actually
reach the goal it was asked to route to.

Manual Office.sim telemetry:

    [PREFETCH] requested target=(2.75,-4.75)
    ...
    [PREFETCH] success waypoints=1
    ... (ACCEPT_PENDING_PATH)
    [STATE ...] target=(3.25,-4.75) path_goal=(2.75,-4.75) wp=1/1
    ... robot reaches ~(3.02,-4.73), enters STOP, v=0.00
    [STATE ...] state=STOP target=(3.25,-4.75) path_goal=(2.75,-4.75) wp=1/1
    [STATE ...] state=STOP target=(3.25,-4.75) path_goal=(2.75,-4.75) wp=1/1
    ... (repeats forever)

Root cause: RobotAgent.accept_pending_path() sets
active_path_goal_xy = pending_target_xy (the ORIGINALLY REQUESTED frontier
target) directly, decoupled from whatever the prefetched route's own final
waypoint actually is. compute_planned_waypoints() can return success=True
with a route that does not end exactly at the requested goal (goal-cell
relocation when the goal cell was occupied, grid quantization, etc.).
engine.on_prefetch_route_ready() previously accepted such a route into
agent.pending_path unconditionally. Once accepted, the robot follows the
route faithfully to its real (different) endpoint, runs out of waypoints,
and stops -- but active_path_goal_xy still points at the original,
never-reached target, so ExplorationBehavior's distance_to_active_path_goal()
never drops below goal_tolerance and "frontier reached" never fires. The
robot is stuck in STOP forever with a stale path_goal.

The same class of mismatch is structurally impossible in
apply_route_result()'s existing REQUEST_PLAN path today (it derives
active_path_goal_xy from the route's own final waypoint,
target=clean_waypoints[-1]) -- but nothing there validates that the route
actually reached the *exploration target it was asked to reach*
(self.current_exploration_target) either, so a route that quietly drifts to
a different endpoint would still be accepted as "success" without ever
being flagged.

Fix: a shared helper, engine.route_reaches_goal(waypoints, goal, tolerance),
is used in two places:
    - apply_route_result(): before accepting a "successful" route in
      exploration mode, its final waypoint must be within goal_tolerance of
      self.current_exploration_target (the goal the route was actually
      requested for). If not, the route is rejected and treated exactly
      like a planner failure (falls through to the existing
      invalidate_failed_exploration_route() / [ROUTE fail] handling).
    - on_prefetch_route_ready(): before accepting a prefetched route into
      agent.pending_path, its final waypoint must be within goal_tolerance
      of agent.pending_target_xy (what accept_pending_path() will use as
      the new active_path_goal_xy). If not, agent.reject_pending_path() is
      called instead -- exactly this scenario is what reproduces the
      observed bug, and test_prefetch_route_rejected_when_final_waypoint_
      does_not_reach_pending_target below exercises it directly.

These tests exercise the standalone route_reaches_goal() helper directly,
and apply_route_result()/on_prefetch_route_ready() via a minimal duck-typed
engine fake, matching the pattern already used in
test_multi_robot_route_validation.py and test_safety_replan_without_active_route.py.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.simulation.engine import SimulationControllerMixin, route_reaches_goal
from robotics_sim.simulation.telemetry import TelemetryLogger


PENDING_TARGET = (2.75, -4.75)
ROUTE_ENDPOINT_TOO_FAR = (3.25, -4.75)  # ~0.5 m from PENDING_TARGET
ROUTE_ENDPOINT_CLOSE_ENOUGH = (2.80, -4.70)  # ~0.07 m from PENDING_TARGET


# ---------------------------------------------------------------------------
# route_reaches_goal() -- the standalone helper.
# ---------------------------------------------------------------------------


def test_route_reaches_goal_true_when_within_tolerance():
    assert route_reaches_goal([ROUTE_ENDPOINT_CLOSE_ENOUGH], PENDING_TARGET, 0.25) is True


def test_route_reaches_goal_false_when_outside_tolerance():
    assert route_reaches_goal([ROUTE_ENDPOINT_TOO_FAR], PENDING_TARGET, 0.25) is False


def test_route_reaches_goal_false_for_empty_waypoints_or_missing_goal():
    assert route_reaches_goal([], PENDING_TARGET, 0.25) is False
    assert route_reaches_goal([ROUTE_ENDPOINT_CLOSE_ENOUGH], None, 0.25) is False


# ---------------------------------------------------------------------------
# Engine-level fixture for apply_route_result()/on_prefetch_route_ready().
# ---------------------------------------------------------------------------


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine(*, position=(3.0, -4.7), goal_tolerance=0.25) -> SimpleNamespace:
    robot = _FakeRobot(x=position[0], y=position[1])
    agent = RobotAgent(
        robot_id=0,
        position=position,
        planner_mode="FoV-aware directional frontier",
    )

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        config=SimpleNamespace(
            planner_type="A*",
            path_simplifier="Direction changes",
            exploration_planner="FoV-aware directional frontier",
            goal_tolerance=goal_tolerance,
            grid_resolution=0.5,
        ),
        mapped_obstacle_points=[],
        current_exploration_target=None,
        route_result_count=0,
        last_goal_selection_reason="frontier selection reason",
        simulation_time=0.0,
        console_logs=[],
        planned_paths=[],
        exploration_targets=[],
        prefetch_workers={},
        prefetch_request_ids={0: 1},
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.collision_checker = CollisionChecker()  # no obstacle points -> never blocks
    fake.canvas = SimpleNamespace(
        set_planned_path=lambda path: fake.planned_paths.append(path),
        set_exploration_target=lambda target: fake.exploration_targets.append(target),
        set_status=lambda message: None,
    )
    fake.is_exploration_mode = lambda: True
    fake.safety_radius = lambda: 0.2
    fake.planner_label = lambda: "A* / Direction changes + FoV-aware directional frontier"
    fake.clean_waypoints_for_current_start = lambda waypoints: [tuple(p) for p in waypoints]
    fake.final_goal_xy = lambda: (0.0, 0.0)
    fake.runtime_agent = lambda robot_index=None: fake.agent

    for name in (
        "apply_route_result",
        "log_route_assignment",
        "on_prefetch_route_ready",
        "_invalidate_prefetch_request",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    return fake


# ---------------------------------------------------------------------------
# 1. apply_route_result(): a "successful" route whose final waypoint misses
#    the requested exploration target must be rejected, not assigned.
# ---------------------------------------------------------------------------


def test_route_success_rejected_when_final_waypoint_does_not_reach_path_goal():
    fake = _build_fake_engine()
    fake.current_exploration_target = PENDING_TARGET
    fake.agent.set_exploration_target(PENDING_TARGET, reason="frontier selected")

    fake.apply_route_result(True, "path found with A*", [ROUTE_ENDPOINT_TOO_FAR])

    assert fake.agent.active_path_goal_xy is None, (
        "the unreachable target must not become the active path goal"
    )
    assert fake.agent.active_target() is None
    # The robot must be holding at its own current position, not following
    # the rejected route.
    assert fake.robot.waypoints == [(3.0, -4.7)]
    # Recovery path used: the attempted target is remembered as failed.
    assert any(
        math.hypot(pt[0] - PENDING_TARGET[0], pt[1] - PENDING_TARGET[1]) < 1e-6
        for pt in fake.agent.recently_failed_exploration_targets(
            current_time=fake.simulation_time, cooldown=999.0
        )
    )
    assert any("[ROUTE fail]" in str(line) for line in fake.console_logs)
    assert not any("[ROUTE ok]" in str(line) for line in fake.console_logs)


# ---------------------------------------------------------------------------
# 2. apply_route_result(): a route whose final waypoint reaches the
#    requested target is assigned normally.
# ---------------------------------------------------------------------------


def test_route_success_allowed_when_final_waypoint_reaches_path_goal():
    fake = _build_fake_engine()
    fake.current_exploration_target = PENDING_TARGET
    fake.agent.set_exploration_target(PENDING_TARGET, reason="frontier selected")

    fake.apply_route_result(True, "path found with A*", [ROUTE_ENDPOINT_CLOSE_ENOUGH])

    assert fake.agent.active_path_goal_xy == ROUTE_ENDPOINT_CLOSE_ENOUGH
    assert fake.agent.active_target() == ROUTE_ENDPOINT_CLOSE_ENOUGH
    assert fake.robot.waypoints == [ROUTE_ENDPOINT_CLOSE_ENOUGH]
    assert any("[ROUTE ok]" in str(line) for line in fake.console_logs)
    assert not any("[ROUTE fail]" in str(line) for line in fake.console_logs)


# ---------------------------------------------------------------------------
# 3. A route ending near-but-not-at the goal: this implementation chooses
#    strict rejection (no automatic "append the goal" behavior), since
#    there is no existing helper that safely validates an arbitrary final
#    segment the way route_first_segment_blocked() validates the first one.
#    Documented and tested explicitly, per the task's instruction to pick
#    one behavior and test it.
# ---------------------------------------------------------------------------


def test_route_success_rejected_when_final_waypoint_is_near_but_not_within_tolerance():
    fake = _build_fake_engine(goal_tolerance=0.25)
    fake.current_exploration_target = PENDING_TARGET
    fake.agent.set_exploration_target(PENDING_TARGET, reason="frontier selected")

    near_miss = (3.05, -4.75)  # 0.30 m away: close, but outside goal_tolerance=0.25
    assert math.hypot(near_miss[0] - PENDING_TARGET[0], near_miss[1] - PENDING_TARGET[1]) > 0.25

    fake.apply_route_result(True, "path found with A*", [near_miss])

    assert fake.agent.active_path_goal_xy is None, (
        "a near-miss final waypoint is still rejected -- strict rejection, no silent goal-append"
    )
    assert any("[ROUTE fail]" in str(line) for line in fake.console_logs)


# ---------------------------------------------------------------------------
# 4. An incomplete route must never leave the robot stuck in STOP with a
#    stale, unreached path_goal.
# ---------------------------------------------------------------------------


def test_incomplete_route_does_not_leave_robot_stuck_in_stop_with_stale_path_goal():
    fake = _build_fake_engine()
    fake.current_exploration_target = PENDING_TARGET
    fake.agent.set_exploration_target(PENDING_TARGET, reason="frontier selected")

    fake.apply_route_result(True, "path found with A*", [ROUTE_ENDPOINT_TOO_FAR])

    # The exact bug: wp exhausted (no waypoints) while path_goal still
    # points at a target the robot never reached. Assert this specific
    # combination cannot occur -- both must be cleared together.
    assert fake.agent.active_path_goal_xy is None
    assert fake.agent.active_target() is None
    assert not fake.agent.waypoints.has_path()


# ---------------------------------------------------------------------------
# The actual observed-bug code path: on_prefetch_route_ready() /
# accept_pending_path(). This is what produced the exact log evidence
# (pending=(2.75,-4.75), then target=(3.25,-4.75) path_goal=(2.75,-4.75)).
# ---------------------------------------------------------------------------


def test_prefetch_route_rejected_when_final_waypoint_does_not_reach_pending_target():
    fake = _build_fake_engine()
    fake.agent.pending_target_xy = PENDING_TARGET
    # The target request_id=1 was launched for -- on_prefetch_route_ready()
    # validates against this captured target, not agent.pending_target_xy.
    fake.prefetch_targets = {0: PENDING_TARGET}

    fake.on_prefetch_route_ready(1, 0, True, "path found with A*", [ROUTE_ENDPOINT_TOO_FAR])

    assert fake.agent.pending_path is None, (
        "a prefetched route that does not reach the pending target must not be accepted"
    )
    assert any("[PREFETCH] rejected" in str(line) for line in fake.console_logs)

    # Even if ACCEPT_PENDING_PATH were to run now, there is nothing pending
    # to accept -- the exact mismatch from the bug report cannot occur.
    accepted = fake.agent.accept_pending_path()
    assert accepted is None
    assert fake.agent.active_path_goal_xy is None


def test_prefetch_route_accepted_when_final_waypoint_reaches_pending_target():
    fake = _build_fake_engine()
    fake.agent.pending_target_xy = PENDING_TARGET
    fake.prefetch_targets = {0: PENDING_TARGET}

    fake.on_prefetch_route_ready(1, 0, True, "path found with A*", [ROUTE_ENDPOINT_CLOSE_ENOUGH])

    assert fake.agent.pending_path == [ROUTE_ENDPOINT_CLOSE_ENOUGH]

    accepted = fake.agent.accept_pending_path()
    assert accepted == [ROUTE_ENDPOINT_CLOSE_ENOUGH]
    assert fake.agent.active_path_goal_xy == PENDING_TARGET
    assert fake.agent.active_target() == ROUTE_ENDPOINT_CLOSE_ENOUGH
    # The accepted route's endpoint and the tracked path goal are now
    # consistent (within tolerance) -- no stale-STOP scenario possible.
    assert math.hypot(
        fake.agent.active_target()[0] - fake.agent.active_path_goal_xy[0],
        fake.agent.active_target()[1] - fake.agent.active_path_goal_xy[1],
    ) <= 0.25
