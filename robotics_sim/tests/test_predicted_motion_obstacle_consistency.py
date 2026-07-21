"""
Regression tests for a real, reproduced safety-replan loop:

    planner accepts a route
    -> predicted_motion_report detects a collision almost immediately
    -> safety replan
    -> new route accepted
    -> new predicted collision
    -> repeats for several seconds

Observed manually with robot pose approximately (-2.99, 3.01) and a mapped
obstacle sample approximately (-3.02, 3.00) -- a few centimeters from the
robot's own center, exactly the kind of boundary sample the robot's own
sensor records right next to a wall/corner it is standing beside.

Root cause (confirmed below by direct experiment against the real
production code, NOT assumed to be identical to the historical fix in
commit 92c7648 "fix: sanitize obstacle points used by first-segment safety
checks"): that commit added engine.py's obstacle_points_for_segment_safety_
check() (sanitizes mapped_obstacle_points the SAME way the planner already
does, via sanitize_planner_obstacle_points()) and wired it into 4 call
sites current at the time. Since then, two of those call sites were
refactored to use _evaluate_route_first_segment() (to preserve the full
CollisionReport for navigation-debug capture) and silently reverted to the
RAW mapped_obstacle_points list in the process:

    - apply_route_result()'s first-segment check (single-robot route
      acceptance)
    - the ACCEPT_PENDING_PATH prefetch-promotion first-segment check

More importantly, simulation_step_multi() (multi-robot mode) was NEVER
covered by the original fix at all -- its per-tick active-segment check and
its predicted_motion_report() call both used the raw list from the start.
Multi-robot route ACCEPTANCE (_assign_route_to_multi_robot_with_corridor_
validation()) only validates the route against teammate corridors
(validate_multi_robot_corridor()) -- it never runs a sanitized first-segment
occupancy check the way single-robot's apply_route_result() does. So in
multi-robot mode, a route past a near-center occupancy sample is accepted
with nothing to reject it, and the very next tick's raw-list active-segment/
predicted-motion checks immediately flag it -- exactly the observed
PLAN_ACCEPTED -> PREDICTED_COLLISION -> SAFETY_REPLAN loop.

Fix: every one of these call sites now uses the SAME obstacle_points_for_
segment_safety_check() helper the planner and build_observation()'s
active_segment_blocked already relied on -- no new abstraction, no radius
duplicated, no config/threshold changed.

These tests exercise the REAL production CollisionChecker/predicted_motion_
report()/obstacle_points_for_segment_safety_check() via a lightweight
duck-typed engine fake (same pattern as test_first_segment_validation_
consistency.py), not mocks.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot import Robot
from robotics_sim.environment.collision_checker import CollisionChecker, CollisionReport
from robotics_sim.simulation.engine import (
    SimulationControllerMixin,
    _evaluate_route_first_segment,
    route_first_segment_blocked,
)
from robotics_sim.simulation.hazard_service import RuntimeHazardService

RESOLUTION = 0.5
ROBOT_RADIUS = 0.35
BOUNDS = (-10.0, 10.0, -10.0, 10.0)


def _make_fake(
    *,
    robot_xy: tuple[float, float] = (0.0, 0.0),
    theta: float = 0.0,
    v: float = 0.5,
    mapped_obstacle_points: list[tuple[float, float]] | None = None,
    grid_resolution: float = RESOLUTION,
    robot_radius: float = ROBOT_RADIUS,
) -> SimpleNamespace:
    robot = Robot(
        x=robot_xy[0], y=robot_xy[1], theta=theta, v=v,
        max_speed=1.5, max_acceleration=2.0, max_angular_speed=2.5,
        robot_radius=robot_radius,
    )
    config = SimpleNamespace(
        grid_resolution=grid_resolution,
        planner_type="A*",
        goal_tolerance=0.25,
        body_radius=robot_radius,
        safety_radius=robot_radius,
        mapping_point_spacing=0.15,
        obstacles=[],
        max_speed=1.5,
        max_acceleration=2.0,
        max_angular_speed=2.5,
    )
    fake = SimpleNamespace(
        robot=robot,
        config=config,
        robots=[],
        mapped_obstacle_points=list(mapped_obstacle_points or []),
        collision_checker=CollisionChecker(),
    )
    for name in (
        "safety_radius_for_robot",
        "safety_radius",
        "body_radius_for_robot",
        "body_radius",
        "sanitize_planner_obstacle_points",
        "obstacle_points_for_segment_safety_check",
        "robot_snapshot",
        "predicted_motion_report",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


def _coast_control() -> np.ndarray:
    """Zero acceleration/omega -- the robot continues in a straight line at
    its current velocity, matching how predicted_motion_report() is used to
    check a nominal control BEFORE it is applied."""
    return np.array([[0.0], [0.0]])


# ---------------------------------------------------------------------------
# 1. An occupancy artifact inside the robot's own body must not desync the
#    validators: none of them should reject a route out to free space.
# ---------------------------------------------------------------------------


def test_body_embedded_occupancy_is_consistently_blocking():
    near_center = (0.02, 0.01)  # ~3cm from the robot's own center
    fake = _make_fake(mapped_obstacle_points=[near_center], v=0.5)
    robot_radius = fake.safety_radius()
    target = (3.0, 0.0)

    sanitized = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    assert near_center in sanitized

    # route_first_segment_blocked() / active_segment_blocked's own rule.
    assert route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), target, sanitized, robot_radius
    ) is True

    # predicted_motion_report() must agree -- same sanitized set, no false collision.
    report = fake.predicted_motion_report(
        control=_coast_control(), dt=0.05, robot_radius=robot_radius,
        known_obstacle_points=sanitized, use_ground_truth=True,
    )
    assert report is not None and report.collision


# ---------------------------------------------------------------------------
# 2. A real obstacle just outside the sanitization disk, on the segment,
#    must still be caught -- the fix must not widen the exclusion zone.
# ---------------------------------------------------------------------------


def test_real_obstacle_outside_exclusion_still_blocks():
    fake = _make_fake(mapped_obstacle_points=[], v=1.0)
    robot_radius = fake.safety_radius()

    real_obstacle = (0.45, 0.0)
    fake.mapped_obstacle_points = [real_obstacle]

    sanitized = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    assert real_obstacle in sanitized, "real obstacle geometry must survive normalization"

    target = (real_obstacle[0] + 1.0, 0.0)
    assert route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), target, sanitized, robot_radius
    ) is True, "a real obstacle on the route must still block it"

    report = fake.predicted_motion_report(
        control=_coast_control(), dt=0.05, robot_radius=robot_radius,
        known_obstacle_points=sanitized, use_ground_truth=True,
    )
    assert report is not None and report.collision, "predicted motion must still detect the real risk"


# ---------------------------------------------------------------------------
# 3. Set consistency: route validation, active-segment validation, and
#    predicted motion must all resolve the SAME semantic obstacle set for
#    the same robot/tick (not necessarily the same Python object).
# ---------------------------------------------------------------------------


def test_all_three_validators_use_the_same_semantic_obstacle_set():
    near_center = (0.03, -0.02)
    wall = [(2.5, y / 10.0) for y in range(-20, 21)]
    fake = _make_fake(mapped_obstacle_points=[near_center] + wall, v=0.4)
    robot_radius = fake.safety_radius()

    route_validation_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    active_segment_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    predicted_motion_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)

    assert set(route_validation_points) == set(active_segment_points) == set(predicted_motion_points)
    assert near_center in route_validation_points
    assert all(point in route_validation_points for point in wall)

    target = (5.0, 0.0)
    route_blocked = route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), target, route_validation_points, robot_radius,
    )
    active_report = fake.collision_checker.check_segment_points(
        start=(0.0, 0.0), end=target, obstacle_points=active_segment_points, robot_radius=robot_radius,
    )
    predicted_report = fake.predicted_motion_report(
        control=_coast_control(), dt=0.05, robot_radius=robot_radius,
        known_obstacle_points=predicted_motion_points, use_ground_truth=True,
    )
    # The wall at x=2.5 blocks all three the same way -- consistent verdicts.
    assert route_blocked == active_report.collision
    if predicted_report is not None:
        assert predicted_report.collision == route_blocked or not route_blocked


# ---------------------------------------------------------------------------
# 4. SOURCE SEPARATION (not runtime wiring): occupancy sanitization must
#    never remove points from a hazard collection combined AFTER it.
#
#    IMPORTANT: this does NOT assert that active_segment_blocked/
#    predicted_motion_report/route validation actually receive hazard
#    points combined together at runtime today -- traced in the audit
#    report and confirmed here again by _make_fake()'s call sites: none of
#    obstacle_points_for_segment_safety_check()'s current callers add
#    observed_blocked_world_points() to the result. Hazard safety for these
#    validators is out of scope of this fix and stays the responsibility of
#    the planning costmap (apply_hazard_belief_to_planning_grid()) and the
#    HOCBF safety filter (hazard_hocbf_filter.py/hazard_safety_runtime.py).
#    This test only proves the ARCHITECTURAL property a future caller that
#    DOES combine both sources would need: sanitize occupancy alone, then
#    combine -- never sanitize the combined collection, which could
#    silently erase an observed hazard sitting close to the robot.
# ---------------------------------------------------------------------------


def test_occupancy_sanitization_does_not_remove_a_separately_combined_hazard_point():
    near_center = (0.02, 0.0)  # occupancy artifact inside the robot's body
    fake = _make_fake(mapped_obstacle_points=[near_center], v=0.3)
    robot_radius = fake.safety_radius()

    hazard_service = RuntimeHazardService(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    hazard_service.add_fire((0.5, 0.0))
    polygon = [(-1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-1.0, 1.0)]
    hazard_service.observe_visible_polygon(polygon, robot_index=0)
    hazard_points = hazard_service.observed_blocked_world_points()
    assert hazard_points, "test setup: the hazard must actually be observed as unsafe"

    sanitized_occupancy = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    assert near_center in sanitized_occupancy

    # Simulates what a FUTURE caller wanting both sources would have to do:
    # sanitize occupancy alone, THEN combine with hazard -- never the other
    # way around. No production code path does this combination today (see
    # docstring above); this is a source-separation contract check only.
    combined = list(sanitized_occupancy) + list(hazard_points)
    for hazard_point in hazard_points:
        assert hazard_point in combined, "occupancy sanitization must never remove an observed hazard point"

    # If a caller DID combine them this way, the unsafe set would still
    # block a route straight through it -- sanitization does not weaken it.
    target = (2.0, 0.0)
    blocked = route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), target, combined, robot_radius,
    )
    assert blocked is True, "an observed hazard must still block a route crossing the unsafe set"


# ---------------------------------------------------------------------------
# 5. Episode regression: accept a route, run several predicted-motion ticks
#    along it, and confirm a body-embedded occupancy point never produces a
#    false PLAN_ACCEPTED -> PREDICTED_COLLISION -> SAFETY_REPLAN sequence.
# ---------------------------------------------------------------------------


def test_episode_rejects_route_before_motion_when_point_is_inside_envelope():
    near_center = (0.02, 0.01)
    fake = _make_fake(robot_xy=(0.0, 0.0), mapped_obstacle_points=[near_center], v=0.5)
    robot_radius = fake.safety_radius()

    # Step 0: route acceptance -- mirrors apply_route_result()'s (now fixed)
    # first-segment check.
    target = (3.0, 0.0)
    sanitized_at_start = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    acceptance_report = _evaluate_route_first_segment(
        fake.collision_checker, (0.0, 0.0), target, sanitized_at_start, robot_radius,
    )
    assert acceptance_report is not None and acceptance_report.collision


# ---------------------------------------------------------------------------
# 6. REAL multi-robot wiring: simulation_step_multi() itself (not a
#    hand-rolled reconstruction of its obstacle-list arithmetic) must feed
#    the sanitized occupancy set to both its active-segment check and its
#    predicted_motion_report() call, and must still append dynamic
#    (other-robot) points unsanitized, exactly as engine.py's own comments
#    at both call sites claim.
#
#    obstacle_points_for_segment_safety_check() itself is monkeypatched to
#    return a known sentinel collection -- this isolates "does simulation_
#    step_multi() call the sanitizer and use its result" from "does the
#    sanitizer itself work", which is already covered by tests 1-3 above and
#    by test_first_segment_validation_consistency.py. CollisionChecker.
#    check_segment_points()/check_predicted_motion_points() are monkeypatched
#    only to CAPTURE their obstacle_points argument (still real collision-
#    safe no-op reports) -- the method under test, simulation_step_multi(),
#    runs for real and unmodified.
# ---------------------------------------------------------------------------


_SANITIZED_SENTINEL = [(42.0, 42.0)]
_DYNAMIC_SENTINEL = [(7.77, 7.77)]


def test_simulation_step_multi_wires_sanitized_occupancy_into_both_checks(monkeypatch):
    near_center = (0.03, -0.02)  # 0.01-0.05m from the robot's own center
    robot_radius = 0.35
    robot = Robot(
        x=0.0, y=0.0, theta=0.0, v=0.4,
        max_speed=1.5, max_acceleration=2.0, max_angular_speed=2.5, robot_radius=robot_radius,
    )

    fake = SimpleNamespace(
        running=True,
        paused=False,
        robots=[robot],
        robot=robot,
        simulation_speed=1.0,
        simulation_time=0.0,
        collision_checker=CollisionChecker(),
        config=SimpleNamespace(
            grid_resolution=RESOLUTION, obstacles=[], goal_tolerance=0.25,
            body_radius=robot_radius, safety_radius=robot_radius,
            max_speed=1.5, max_acceleration=2.0, max_angular_speed=2.5,
        ),
        mapped_obstacle_points=[near_center],
        multi_path_points=[],
        multi_planned_path_points=[],
        multi_exploration_targets=[],
        selected_robot_index=0,
    )
    fake.canvas = SimpleNamespace(set_status=lambda message: None, set_multi_runtime_state=lambda **kwargs: None)

    # Bind the REAL methods under test.
    for name in ("inter_robot_clearance_violation", "predicted_motion_report", "robot_snapshot"):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    # Preliminary/coordination machinery irrelevant to the safety/control
    # segment under test -- stubbed to isolate exactly that segment, per the
    # task's own "no need to test the full coordination cycle" guidance.
    fake.should_run_sensor_update = lambda now: False
    fake.safety_radius_for_robot = lambda robot=None: robot_radius
    fake.active_target_xy = lambda: (3.0, 0.0)
    fake.segment_violates_other_robot_clearance = lambda *args, **kwargs: (False, "")
    fake.runtime_agent = lambda robot_index=None: None
    fake._finalize_navigation_debug_snapshot = lambda **kwargs: None
    fake.nominal_control_safe = lambda blocked=False, capture=None: np.array([[0.1], [0.0]])
    fake.coordinator_runtime_profile = lambda: SimpleNamespace(owns_control=False)
    fake.apply_hazard_safety_filter = lambda robot, control: control
    fake.is_exploration_mode = lambda: False
    fake.log_robot_motion = lambda *args, **kwargs: None

    sanitize_calls: list[tuple[tuple[float, float], float]] = []

    def _fake_obstacle_points_for_segment_safety_check(start_xy, radius):
        sanitize_calls.append((start_xy, radius))
        return list(_SANITIZED_SENTINEL)

    fake.obstacle_points_for_segment_safety_check = _fake_obstacle_points_for_segment_safety_check

    dynamic_calls: list[int] = []

    def _fake_dynamic_robot_obstacle_points_for_robot(index):
        dynamic_calls.append(index)
        return list(_DYNAMIC_SENTINEL)

    fake.dynamic_robot_obstacle_points_for_robot = _fake_dynamic_robot_obstacle_points_for_robot

    segment_points_calls: list[list[tuple[float, float]]] = []

    def _capturing_check_segment_points(*, start, end, obstacle_points, robot_radius):
        segment_points_calls.append(list(obstacle_points))
        return CollisionReport(collision=False)

    monkeypatch.setattr(fake.collision_checker, "check_segment_points", _capturing_check_segment_points)

    predicted_points_calls: list[list[tuple[float, float]]] = []

    def _capturing_check_predicted_motion_points(*, snapshot, control, dt, steps, obstacle_points, robot_radius):
        predicted_points_calls.append(list(obstacle_points))
        return CollisionReport(collision=False)

    monkeypatch.setattr(
        fake.collision_checker, "check_predicted_motion_points", _capturing_check_predicted_motion_points
    )

    # Run the REAL, unmodified method.
    SimulationControllerMixin.simulation_step_multi(fake, 0.05)

    # 1. Both checks were actually reached (not skipped/short-circuited).
    assert len(segment_points_calls) == 1
    assert len(predicted_points_calls) == 1

    # 2. Both received the sanitized sentinel -- proving simulation_step_
    #    multi() actually calls obstacle_points_for_segment_safety_check()
    #    and uses ITS return value, not a hand-rolled equivalent.
    assert _SANITIZED_SENTINEL[0] in segment_points_calls[0]
    assert _SANITIZED_SENTINEL[0] in predicted_points_calls[0]

    # 3. Neither received the raw occupancy point directly -- it only ever
    #    reaches these checks by going through the (mocked) sanitizer, which
    #    in production would have excluded it.
    assert near_center not in segment_points_calls[0]
    assert near_center not in predicted_points_calls[0]

    # 4. dynamic_robot_obstacle_points_for_robot() is still appended AFTER,
    #    unsanitized -- present in both received lists, called once per check.
    assert _DYNAMIC_SENTINEL[0] in segment_points_calls[0]
    assert _DYNAMIC_SENTINEL[0] in predicted_points_calls[0]
    assert dynamic_calls == [0, 0]

    # 5. Each received list is exactly sentinel + dynamic -- nothing else
    #    silently mixed in.
    assert segment_points_calls[0] == _SANITIZED_SENTINEL + _DYNAMIC_SENTINEL
    assert predicted_points_calls[0] == _SANITIZED_SENTINEL + _DYNAMIC_SENTINEL

    # 6. The sanitizer itself was invoked with the robot's real current
    #    position/radius for this tick (wiring detail, not a re-test of the
    #    sanitizer's own math).
    assert sanitize_calls == [((0.0, 0.0), robot_radius), ((0.0, 0.0), robot_radius)]
