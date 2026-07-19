"""
Regression tests for a real, reproduced Office.sim failure sequence:

    - reachability considers a frontier candidate reachable
    - A* finds a path on the sanitized planning grid
    - the simplifier agrees with A*
    - route validation (apply_route_result()/accept_pending_path()) rejects
      the SAME route's first segment as "blocked on arrival"
    - the active-segment/predicted-collision safety checks agree with the
      rejection
    - every frontier candidate eventually fails the same way, and
      exploration is declared exhausted with most of the map unexplored

Root cause (confirmed by direct experiment against the real production
code, before any fix): engine.py's sanitize_planner_obstacle_points()
already removes mapped_obstacle_points samples that fall within a small
disk of the robot's own current position before the PLANNER (A*/
reachability) ever sees them -- this is deliberate and documented (a
boundary sample the robot's own sensor just placed a few centimeters from
its center is expected, not a real obstacle, and would otherwise make A*
reject the route before it starts). But route_first_segment_blocked() (used
by both apply_route_result() and accept_pending_path()), active_segment_
blocked (build_observation()), and predicted_collision
(predicted_motion_report()) all used to pass the RAW, unsanitized
mapped_obstacle_points list straight into CollisionChecker.check_segment_
points()/check_predicted_motion_points() -- whose "distance from any
obstacle point to the segment" rule always finds that same near-start
sample sitting inside robot_radius of start_xy (t=0 of any segment starting
AT the robot), and rejects the segment regardless of which direction it
points. The fix: engine.py's obstacle_points_for_segment_safety_check()
applies the exact SAME sanitize_planner_obstacle_points() sanitization the
planner already relies on, and all four call sites now use it instead of
the raw list.

These tests exercise the REAL production code (BeliefMap, CollisionChecker,
sanitize_planner_obstacle_points(), build_planning_grid_for_robot(),
compute_planned_waypoints() -- the actual A*/simplifier/cell<->world
conversion pipeline in planning/planner_registry.py) via a lightweight
duck-typed engine fake (the same pattern used throughout this test suite),
not mocks -- so a regression in the real geometry would fail these tests.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.planning.planner_registry import compute_planned_waypoints
from robotics_sim.simulation.config import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN
from robotics_sim.simulation.engine import (
    SimulationControllerMixin,
    candidate_reachable_on_planning_grid,
    route_first_segment_blocked,
)

RESOLUTION = 0.5
ROBOT_RADIUS = 0.3
BOUNDS = (WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX)


def _make_engine_fake(
    *,
    robot_xy: tuple[float, float] = (0.0, 0.0),
    mapped_obstacle_points: list[tuple[float, float]] | None = None,
    grid_resolution: float = RESOLUTION,
    robot_radius: float = ROBOT_RADIUS,
) -> SimpleNamespace:
    config = SimpleNamespace(
        grid_resolution=grid_resolution,
        planner_type="A*",
        goal_tolerance=0.25,
        body_radius=robot_radius,
        safety_radius=robot_radius,
        mapping_point_spacing=0.15,
        obstacles=[],
    )
    robot = SimpleNamespace(x=float(robot_xy[0]), y=float(robot_xy[1]), theta=0.0, vision=3.0)
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
        "build_planning_grid_for_robot",
        "obstacle_points_for_segment_safety_check",
        "make_exploration_reachability_check",
        "reset_belief_map",
        "ensure_belief_map",
        "push_discovered_hazard_frame",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


def _plan_route(fake, start_xy, goal_xy):
    """Drive the REAL planner pipeline exactly like engine.py's
    build_planner_kwargs()/build_planner_kwargs_for_goal() do: sanitize ->
    build planning grid -> compute_planned_waypoints() (real A* + real
    simplifier + real cell<->world conversion)."""
    robot_radius = fake.safety_radius()
    resolution = float(fake.config.grid_resolution)
    sanitized_points, _ = fake.sanitize_planner_obstacle_points(
        list(fake.mapped_obstacle_points), start_xy=start_xy, robot_radius=robot_radius, resolution=resolution,
    )
    planning_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=sanitized_points, robot_radius=robot_radius)
    return compute_planned_waypoints(
        planner_type=fake.config.planner_type,
        start_xy=start_xy,
        goal_xy=goal_xy,
        bounds=BOUNDS,
        resolution=resolution,
        robot_radius=robot_radius,
        planning_grid=planning_grid,
        unknown_is_traversable=True,
        obstacle_points=[],
    )


def _wall_points(x: float, y_min: float, y_max: float, spacing: float = 0.2) -> list[tuple[float, float]]:
    """Dense boundary samples forming a straight wall segment at x, from
    y_min to y_max -- stands in for mapped_obstacle_points detected along a
    real wall."""
    count = int((y_max - y_min) / spacing) + 1
    return [(x, round(y_min + spacing * i, 3)) for i in range(count)]


# ---------------------------------------------------------------------------
# 1. Raw A* valid and the simplified path is also valid (control case).
# ---------------------------------------------------------------------------


def test_open_space_route_is_valid_for_planner_and_validator():
    fake = _make_engine_fake(robot_xy=(0.0, 0.0), mapped_obstacle_points=[])

    success, reason, waypoints = _plan_route(fake, (0.0, 0.0), (4.0, 0.0))

    assert success, reason
    assert waypoints
    robot_radius = fake.safety_radius()
    obstacle_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    assert route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), waypoints[0], obstacle_points, robot_radius,
    ) is False


# ---------------------------------------------------------------------------
# 2. Raw A* valid, but a simplifier shortcut would be invalid: the
#    simplifier must keep enough waypoints to stay geometrically safe --
#    exercised through "Line of sight grid-safe", which is what Office.sim
#    evidence names explicitly.
# ---------------------------------------------------------------------------


def test_simplifier_does_not_shortcut_through_an_l_shaped_corner():
    # An L-shaped corridor: a wall spans most of x=2 (blocking a straight
    # line), forcing the route around through a gap at the top.
    mapped_obstacle_points = _wall_points(x=2.0, y_min=-4.0, y_max=2.0)
    fake = _make_engine_fake(robot_xy=(0.0, 0.0), mapped_obstacle_points=mapped_obstacle_points)

    success, reason, waypoints = _plan_route(fake, (0.0, 0.0), (4.0, -3.0))
    assert success, reason
    assert len(waypoints) >= 2, "an L-shaped detour must keep at least a turning waypoint, not a straight shortcut"

    robot_radius = fake.safety_radius()
    obstacle_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    # Every simplified segment (not just the first) must stay clear of the
    # known wall -- a bad shortcut would cut through the corner at x=2.
    full_route = [(0.0, 0.0)] + list(waypoints)
    for start, end in zip(full_route, full_route[1:]):
        report = fake.collision_checker.check_segment_points(
            start=start, end=end, obstacle_points=obstacle_points, robot_radius=robot_radius,
        )
        assert not report.collision, f"simplified segment {start}->{end} cuts through the known wall"


# ---------------------------------------------------------------------------
# 3. THE REPRODUCED BUG: the robot starts with a mapped-obstacle sample a
#    few centimeters from its own center (as its own sensor would record
#    right next to a wall/corner) -- clearing the start cell for A* must
#    not leave the route validator rejecting the resulting first segment.
# ---------------------------------------------------------------------------


def test_robot_near_known_obstacle_does_not_get_an_impossible_first_segment():
    near_robot_sample = (0.10, 0.05)  # inside ROBOT_RADIUS (0.3) of (0, 0)
    wall = _wall_points(x=3.0, y_min=-3.0, y_max=3.0)
    fake = _make_engine_fake(robot_xy=(0.0, 0.0), mapped_obstacle_points=[near_robot_sample] + wall)

    success, reason, waypoints = _plan_route(fake, (0.0, 0.0), (6.0, 0.0))
    assert success, reason
    assert waypoints

    robot_radius = fake.safety_radius()
    obstacle_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    assert near_robot_sample not in obstacle_points, (
        "the near-start sample must be sanitized the same way the planner already sanitizes it"
    )
    assert route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), waypoints[0], obstacle_points, robot_radius,
    ) is False, (
        "a route the planner already found safe on the sanitized grid must not be "
        "rejected by a validator using a stricter, unsanitized obstacle set"
    )


# ---------------------------------------------------------------------------
# 4. A corridor whose width is physically sufficient for the robot must be
#    accepted (both by the planner and by first-segment validation).
# ---------------------------------------------------------------------------


def test_physically_sufficient_corridor_is_accepted():
    # Two walls 1.4m apart (center-to-center); ROBOT_RADIUS=0.3 means the
    # robot's inflated footprint (diameter 0.6) fits with clearance to spare
    # in this corridor.
    left_wall = _wall_points(x=1.0, y_min=-4.0, y_max=4.0)
    right_wall = _wall_points(x=2.4, y_min=-4.0, y_max=4.0)
    fake = _make_engine_fake(robot_xy=(0.0, 0.0), mapped_obstacle_points=left_wall + right_wall)

    success, reason, waypoints = _plan_route(fake, (0.0, 0.0), (3.4, 0.0))

    assert success, reason
    assert waypoints
    robot_radius = fake.safety_radius()
    obstacle_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    assert route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), waypoints[0], obstacle_points, robot_radius,
    ) is False


# ---------------------------------------------------------------------------
# 5. A corridor whose width is physically insufficient must continue to be
#    rejected -- the fix must not weaken genuine safety checks.
# ---------------------------------------------------------------------------


def test_physically_insufficient_corridor_is_rejected():
    # Two walls only 0.4m apart (center-to-center); ROBOT_RADIUS=0.3 means
    # the robot's inflated footprint (diameter 0.6) cannot fit between them.
    # Walls span almost the full world height so there is no way around --
    # the ONLY route to the goal is through the too-narrow gap.
    left_wall = _wall_points(x=1.0, y_min=WORLD_Y_MIN + 0.1, y_max=WORLD_Y_MAX - 0.1)
    right_wall = _wall_points(x=1.4, y_min=WORLD_Y_MIN + 0.1, y_max=WORLD_Y_MAX - 0.1)
    fake = _make_engine_fake(robot_xy=(0.0, 0.0), mapped_obstacle_points=left_wall + right_wall)

    success, reason, waypoints = _plan_route(fake, (0.0, 0.0), (3.0, 0.0))

    if success and waypoints:
        # If the planner (wrongly) found a path through the too-narrow gap,
        # first-segment validation must still catch it -- it must never
        # silently accept a physically impossible corridor.
        robot_radius = fake.safety_radius()
        obstacle_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
        full_route = [(0.0, 0.0)] + list(waypoints)
        collision_found = any(
            fake.collision_checker.check_segment_points(
                start=start, end=end, obstacle_points=obstacle_points, robot_radius=robot_radius,
            ).collision
            for start, end in zip(full_route, full_route[1:])
        )
        assert collision_found, "a route through a too-narrow corridor must be caught by segment validation"
    else:
        assert not success


# ---------------------------------------------------------------------------
# 6. The SAME segment must produce the SAME result under planner-side
#    validation (route_first_segment_blocked) and active-segment safety
#    validation (the same check build_observation() runs), for the exact
#    same map snapshot -- no undocumented difference between the two.
# ---------------------------------------------------------------------------


def test_route_validation_and_active_segment_check_agree():
    near_robot_sample = (0.10, 0.05)
    wall = _wall_points(x=3.0, y_min=-3.0, y_max=3.0)
    fake = _make_engine_fake(robot_xy=(0.0, 0.0), mapped_obstacle_points=[near_robot_sample] + wall)

    success, reason, waypoints = _plan_route(fake, (0.0, 0.0), (6.0, 0.0))
    assert success, reason
    target = waypoints[0]
    robot_radius = fake.safety_radius()

    route_validation_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    route_blocked = route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), target, route_validation_points, robot_radius,
    )

    # Mirrors build_observation()'s active_segment_blocked computation
    # exactly (same helper, same CollisionChecker method, no dynamic_pts in
    # single-robot mode).
    active_segment_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    active_segment_report = fake.collision_checker.check_segment_points(
        start=(0.0, 0.0), end=target, obstacle_points=active_segment_points, robot_radius=robot_radius,
    )

    assert route_blocked == active_segment_report.collision


# ---------------------------------------------------------------------------
# 7. A candidate reachability approves can produce a valid initial route
#    under the SAME map snapshot -- reachability and the real planner must
#    not disagree about whether a frontier is worth requesting a route to.
# ---------------------------------------------------------------------------


def test_reachable_candidate_produces_a_valid_initial_route():
    near_robot_sample = (0.10, 0.05)
    wall = _wall_points(x=3.0, y_min=-3.0, y_max=3.0)
    fake = _make_engine_fake(robot_xy=(0.0, 0.0), mapped_obstacle_points=[near_robot_sample] + wall)
    # Grid-cell-center-aligned so the endpoint-reaches-goal_tolerance check
    # inside candidate_reachable_on_planning_grid() does not fail purely on
    # cell-center quantization -- unrelated to the bug under test here.
    candidate = (6.25, 0.25)

    is_reachable = fake.make_exploration_reachability_check(fake.robot)
    assert is_reachable is not None
    assert is_reachable(candidate) is True

    success, reason, waypoints = _plan_route(fake, (0.0, 0.0), candidate)
    assert success, reason
    assert waypoints

    robot_radius = fake.safety_radius()
    obstacle_points = fake.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    assert route_first_segment_blocked(
        fake.collision_checker, (0.0, 0.0), waypoints[0], obstacle_points, robot_radius,
    ) is False
