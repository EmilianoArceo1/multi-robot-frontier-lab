"""
Audit characterization: what REAL composition does ``obstacle_points`` have
at every production call site of
``SimulationControllerMixin.build_planning_grid_for_robot()``, before
PlanningCostmapBuilder is connected to runtime?

This matters because PlanningCostmapBuilder's own contract (robotics_sim/
planning/planning_costmap_builder.py) takes an explicit
``ObservedObstacleSnapshot`` -- a single, static, sensor-observed point
list -- as its only source of observed-obstacle geometry. If the REAL
runtime's ``obstacle_points`` argument is not equivalent to (static
geometry) + (explicit, separately-modeled dynamic points), migrating a
caller to the new builder would either silently drop something the runtime
currently blocks, or require a dynamic-points parameter the builder does
not have yet.

Call-site inventory (AST-verified, not grep-only -- see
test_build_planning_grid_for_robot_call_sites_are_inventoried_with_
composition_detail and the whole-package tripwire test below):

    engine.py: build_planner_kwargs()
        Single-robot / legacy path. obstacle_points = sanitize_planner_
        obstacle_points(list(self.mapped_obstacle_points), ...). NO other
        robots, NO ground truth, NO hazard (hazard enters later, inside
        build_planning_grid_for_robot() itself, via hazard_service).

    engine.py: build_planner_kwargs_for_goal()
        Known-goal path (frontier target already selected elsewhere). Same
        composition as build_planner_kwargs(): sanitize_planner_obstacle_
        points(list(self.mapped_obstacle_points), ...) only.

    engine.py: build_planner_kwargs_for_multi_robot()
        Multi-robot path. obstacle_points = sanitize_planner_obstacle_
        points(list(self.mapped_obstacle_points) + dynamic_points, ...)
        where dynamic_points = self.dynamic_robot_obstacle_points_for_
        robot(robot_index) -- a dense point-cloud sample (center + ring of
        boundary samples) of every OTHER runtime robot. This is the ONLY
        call site where another robot's position enters obstacle_points.

    engine.py: make_exploration_reachability_check()'s nested _build_
    context()
        Reachability path (FoV-aware target filtering). obstacle_points =
        sanitize_planner_obstacle_points(list(self.mapped_obstacle_points),
        ...) only -- like the single-robot path, it never adds dynamic_
        robot_obstacle_points_for_robot(). It builds its OWN OccupancyGrid
        (a fresh call to build_planning_grid_for_robot()), never reusing
        whatever grid the actual planner call built. See Case 9 below: in
        multi-robot mode this is a real, confirmed composition gap between
        reachability's grid and the multi-robot planner's own grid.

Classification of obstacle_points sources (per this file's Case
tests, letters per the task spec):

    a. static observed geometry  -- self.mapped_obstacle_points, ALWAYS
       present, in every call site above.
    b. per-robot sanitized geometry -- sanitize_planner_obstacle_points()
       removes points within a small disk of the CALLING robot's own
       start_xy; this depends on start_xy, so the same underlying points
       sanitize differently per robot/per call (Case 2, Case 3).
    c. other-robot points -- dynamic_robot_obstacle_points_for_robot(),
       ONLY merged in by build_planner_kwargs_for_multi_robot() (Case 4).
    d. dynamic obstacles (temporal, non-robot) -- NONE found feeding
       obstacle_points anywhere in engine.py; the only other "dynamic_
       obstacles" concept in this codebase (RobotObservation.
       dynamic_obstacles, engine.py's build_observation(), consumed by
       exploration_planners.py's frontier-candidate SCORING as (cx, cy,
       radius) proximity-penalty disks) is a completely separate
       composition that never touches obstacle_points/build_planning_
       grid_for_robot at all.
    e. ground truth -- NEVER present in obstacle_points at any of the 4
       call sites (AST-verified: none of their traced obstacle_points
       expressions contain "self.config.obstacles"). The only place
       config.obstacles reaches ANY safety check is engine.py's
       check_predicted_motion(..., obstacles=self.config.obstacles),
       gated by a `use_ground_truth` flag, inside predicted-motion
       CONTINUOUS collision prediction -- a documented, explicit backstop
       for a different layer entirely, never merged into obstacle_points
       or the discrete planning grid (Case 7).
    f. hazard -- NEVER present in obstacle_points. build_planning_grid_
       for_robot() applies hazard_service.belief (discovered-only
       HazardBelief) to the OccupancyGrid directly, as its own separate
       step, after obstacle_points has already been rasterized (Case 6).
    g. other -- none found beyond (a)-(f) above.

Fakes below bind the REAL SimulationControllerMixin methods under test
(sanitize_planner_obstacle_points, obstacle_points_for_segment_safety_
check, dynamic_robot_obstacle_points_for_robot, build_planning_grid_for_
robot, build_planner_kwargs, make_exploration_reachability_check,
observed_obstacle_snapshot, reset_belief_map, ...) -- the same convention
already used by test_observed_obstacle_coverage_characterization.py /
test_map_snapshot_producers.py / test_planning_costmap_builder.py /
test_reachability_instrumentation.py. Sensor-geometry collaborators are not
involved here at all: this file starts from an already-populated
mapped_obstacle_points list, since it is about what happens to that list
downstream, not how it was produced (see the other characterization file
for that).
"""
from __future__ import annotations

import ast
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.environment.occupancy_grid import OCCUPIED as OG_OCCUPIED
from robotics_sim.simulation.engine import SimulationControllerMixin

RESOLUTION = 1.0
ROBOT_RADIUS = 0.3


def _make_fake_engine(
    *,
    robot_positions: list[tuple[float, float]] | None = None,
    obstacles: list | None = None,
) -> SimpleNamespace:
    positions = robot_positions if robot_positions is not None else [(0.0, 0.0)]
    robots = [SimpleNamespace(x=x, y=y, theta=0.0, vision=3.0) for x, y in positions]

    config = SimpleNamespace(
        grid_resolution=RESOLUTION,
        mapping_point_spacing=0.5,
        body_radius=0.2,
        safety_radius=ROBOT_RADIUS,
        planner_type="A*",
        goal_tolerance=0.25,
        exploration_planner="Goal seeking",  # simplest branch of select_navigation_goal()
        goal_x=9.0,
        goal_y=9.0,
        default_fire_intensity=1.0,
        default_fire_radius=2.0,
        fire_selection_radius=0.6,
        hazard_block_threshold=0.55,
        obstacles=list(obstacles or []),
    )
    fake = SimpleNamespace(
        robot=robots[0],
        robots=robots,
        config=config,
        mapped_obstacle_points=[],
        multi_exploration_targets=[],
        canvas=SimpleNamespace(
            append_mapped_obstacle_points=lambda points: None,
            set_status=lambda message: None,
            set_exploration_target=lambda target: None,
            set_multi_exploration_targets=lambda targets: None,
        ),
    )
    # Not the subject under test here (agent/registry wiring is orthogonal
    # to obstacle_points composition) -- select_navigation_goal()'s and
    # select_navigation_goal_for_multi_robot()'s goal-seeking branches both
    # tolerate None.
    fake.runtime_agent = lambda index=None: None

    for name in (
        "reset_belief_map",
        "ensure_belief_map",
        "sync_legacy_map_views_from_belief",
        "push_discovered_hazard_frame",
        "force_robot_pose_free_in_belief",
        "safety_radius_for_robot",
        "safety_radius",
        "body_radius_for_robot",
        "body_radius",
        "sanitize_planner_obstacle_points",
        "obstacle_points_for_segment_safety_check",
        "dynamic_robot_obstacle_points_for_robot",
        "build_planning_grid_for_robot",
        "_planning_costmap_inputs_for_robot",
        "_planning_grid_from_costmap_snapshot",
        "_dynamic_obstacle_points_for_robot_object",
        "final_goal_xy",
        "select_navigation_goal",
        "select_navigation_goal_for_multi_robot",
        "ensure_multi_exploration_target_slots",
        "publish_multi_exploration_targets",
        "is_exploration_mode",
        "exploration_planner_name",
        "build_planner_kwargs",
        "build_planner_kwargs_for_goal",
        "build_planner_kwargs_for_multi_robot",
        "observed_obstacle_snapshot",
        "make_exploration_reachability_check",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    fake.reset_belief_map()
    return fake


def _run_real_multi_robot_path(
    fake: SimpleNamespace, robot_index: int = 0, *, force_new_exploration_target: bool = False,
) -> tuple[dict, str]:
    """Executes the REAL, already-bound SimulationControllerMixin.
    build_planner_kwargs_for_multi_robot() -- never a hand-composed
    substitute. Exists so the multi-robot test and the reachability
    comparison test both go through the exact same real call, instead of
    each re-deriving the multi-robot composition independently."""
    return fake.build_planner_kwargs_for_multi_robot(robot_index, force_new_exploration_target)


def _capture_calls(fake: SimpleNamespace, method_name: str) -> list[dict]:
    """Wrap the ALREADY-bound real method on fake, recording every call's
    arguments and return value before delegating to the real
    implementation -- never a copy of the method's own logic, just a thin
    recording shim around the real, already-bound callable.

    obstacle_points is captured RAW (None stays None) -- under the runtime
    costmap-builder integration, obstacle_points is None whenever a
    production caller uses the NEW path (the four production call sites
    all do), so `is None` is itself the signal that the new path was
    taken, not an accident to normalize away like the old None-vs-empty-
    list ambiguity this used to guard against.
    """
    real_method = getattr(fake, method_name)
    calls: list[dict] = []

    def _wrapper(robot, *, obstacle_points=None, robot_radius=None, dynamic_obstacle_points=()):
        result = real_method(
            robot,
            obstacle_points=obstacle_points,
            robot_radius=robot_radius,
            dynamic_obstacle_points=dynamic_obstacle_points,
        )
        calls.append(
            {
                "robot": robot,
                "obstacle_points_id": id(obstacle_points),
                "obstacle_points": obstacle_points,
                "robot_radius": robot_radius,
                "dynamic_obstacle_points": tuple(dynamic_obstacle_points),
                "result": result,
            }
        )
        return result

    setattr(fake, method_name, _wrapper)
    return calls


# ---------------------------------------------------------------------------
# Case 1: static observed points.
# ---------------------------------------------------------------------------


def test_static_observed_points_reach_preparation_unmutated_with_characterized_order():
    fake = _make_fake_engine()
    p1 = (8.0, 8.0)
    p2 = (2.0, 2.0)
    p_dup = (8.0, 8.0)  # deliberate duplicate of p1
    fake.mapped_obstacle_points = [p1, p2, p_dup]
    start_xy = (0.0, 0.0)  # far from all three points -- nothing gets sanitized away

    # Mirrors every real call site's own first step: list(self.mapped_
    # obstacle_points) -- a fresh copy, never the live list itself.
    input_copy = list(fake.mapped_obstacle_points)
    prepared, removed = fake.sanitize_planner_obstacle_points(
        input_copy, start_xy=start_xy, robot_radius=ROBOT_RADIUS, resolution=RESOLUTION,
    )

    assert removed == 0, "sanity: the robot is far from every point here"
    # Order is preserved exactly -- characterized, not assumed: the
    # function only filters, it never sorts or reorders.
    assert prepared == [p1, p2, p_dup]
    # Duplicates are NOT deduplicated -- characterized, not assumed.
    assert prepared.count(p1) == 2
    # Neither the live list nor the copy passed in is mutated.
    assert fake.mapped_obstacle_points == [p1, p2, p_dup]
    assert input_copy == [p1, p2, p_dup]
    assert prepared is not input_copy


# ---------------------------------------------------------------------------
# Case 2: per-robot sanitization.
# ---------------------------------------------------------------------------


def test_sanitize_planner_obstacle_points_removes_only_the_near_robot_point():
    fake = _make_fake_engine()
    start_xy = (5.0, 5.0)
    near_point = (5.0, 5.0)  # exactly at the robot's own position -- must be removed
    far_point = (55.0, 5.0)  # 50m away -- must survive regardless of the exact clear_radius formula

    # The real method, called directly -- its clear_radius formula is
    # deliberately NOT reproduced here (see its own docstring for that);
    # only the qualitative near-vs-far outcome is characterized.
    prepared, removed = fake.sanitize_planner_obstacle_points(
        [near_point, far_point], start_xy=start_xy, robot_radius=ROBOT_RADIUS, resolution=RESOLUTION,
    )

    assert removed == 1
    assert prepared == [far_point]
    assert near_point not in prepared
    assert far_point in prepared


# ---------------------------------------------------------------------------
# Case 3: sanitization differs by robot.
# ---------------------------------------------------------------------------


def test_sanitization_result_differs_by_robot_start_position():
    fake = _make_fake_engine()
    shared_points = [(5.0, 5.0), (1.0, 1.0)]  # identical observed geometry for both robots

    robot_a_start = (5.0, 5.0)  # sits exactly on the first point
    robot_b_start = (1.0, 1.0)  # sits exactly on the second point

    prepared_a, removed_a = fake.sanitize_planner_obstacle_points(
        list(shared_points), start_xy=robot_a_start, robot_radius=ROBOT_RADIUS, resolution=RESOLUTION,
    )
    prepared_b, removed_b = fake.sanitize_planner_obstacle_points(
        list(shared_points), start_xy=robot_b_start, robot_radius=ROBOT_RADIUS, resolution=RESOLUTION,
    )

    assert removed_a == 1 and (5.0, 5.0) not in prepared_a and (1.0, 1.0) in prepared_a
    assert removed_b == 1 and (1.0, 1.0) not in prepared_b and (5.0, 5.0) in prepared_b
    assert prepared_a != prepared_b, (
        "the SAME observed geometry produces a DIFFERENT sanitized result depending on which "
        "robot's start_xy is used -- this per-robot filtering is not a property a single "
        "shared ObservedObstacleSnapshot could carry on its own"
    )


# ---------------------------------------------------------------------------
# Case 4: other robots.
# ---------------------------------------------------------------------------


def test_other_robot_position_enters_obstacle_points_via_dynamic_robot_points():
    """Unit test of dynamic_robot_obstacle_points_for_robot() itself, plus a
    hand-composed sanitize+build call using it -- kept as a narrow,
    self-contained characterization of that ONE helper in isolation. This
    is NOT the primary evidence that the real multi-robot path includes
    other robots: that is
    test_build_planner_kwargs_for_multi_robot_intercepted_reveals_exact_
    composition below, which executes the REAL build_planner_kwargs_for_
    multi_robot() end to end and intercepts its own internal call.
    """
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    fake.mapped_obstacle_points = []
    robot_index = 0
    other_robot_xy = (5.0, 5.0)

    dynamic_points = fake.dynamic_robot_obstacle_points_for_robot(robot_index)
    assert dynamic_points, "sanity: a second robot exists, so there is something to sample"
    assert any(
        math.hypot(px - other_robot_xy[0], py - other_robot_xy[1]) < 1e-6 for px, py in dynamic_points
    ), "dynamic_robot_obstacle_points_for_robot() must sample the OTHER robot's own center"

    obstacle_points, _ = fake.sanitize_planner_obstacle_points(
        list(fake.mapped_obstacle_points) + dynamic_points,
        start_xy=(0.0, 0.0), robot_radius=ROBOT_RADIUS, resolution=RESOLUTION,
    )

    calls = _capture_calls(fake, "build_planning_grid_for_robot")
    result_grid = fake.build_planning_grid_for_robot(
        fake.robots[0], obstacle_points=obstacle_points, robot_radius=ROBOT_RADIUS,
    )

    assert calls[0]["obstacle_points"] == obstacle_points
    other_robot_cell = result_grid.world_to_grid(*other_robot_xy)
    assert result_grid.get_value(other_robot_cell) == OG_OCCUPIED, (
        "the other robot's position DOES enter obstacle_points and DOES block the resulting "
        "planning grid, in today's multi-robot obstacle-preparation path -- this is "
        "characterized current behavior, not an asserted architectural requirement"
    )


# ---------------------------------------------------------------------------
# Case 5: dynamic points vs. static snapshot.
# ---------------------------------------------------------------------------


def test_observed_obstacle_snapshot_excludes_dynamic_other_robot_points():
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    fake.mapped_obstacle_points = [(2.0, 2.0)]
    fake.mapped_obstacle_revision = 1

    dynamic_points = fake.dynamic_robot_obstacle_points_for_robot(0)
    assert dynamic_points, "sanity: dynamic other-robot points genuinely exist in this scenario (see Case 4)"

    snapshot = fake.observed_obstacle_snapshot()

    assert snapshot.points == ((2.0, 2.0),)
    assert not (set(dynamic_points) & set(snapshot.points)), (
        "observed_obstacle_snapshot() must contain ONLY static mapped_obstacle_points -- the "
        "dynamic other-robot points the multi-robot planning path adds separately must never "
        "appear in it"
    )


# ---------------------------------------------------------------------------
# Case 6: hazard independence.
# ---------------------------------------------------------------------------


def test_hazard_does_not_change_obstacle_points_but_blocks_the_grid():
    fake = _make_fake_engine()
    fake.mapped_obstacle_points = [(2.0, 2.0)]
    obstacle_points, _ = fake.sanitize_planner_obstacle_points(
        list(fake.mapped_obstacle_points), start_xy=(0.0, 0.0), robot_radius=ROBOT_RADIUS, resolution=RESOLUTION,
    )
    obstacle_points_before = list(obstacle_points)

    hazard_row, hazard_col = 6, 6  # far from (2, 2)
    fake.hazard_service.belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)

    calls = _capture_calls(fake, "build_planning_grid_for_robot")
    result_grid = fake.build_planning_grid_for_robot(
        fake.robot, obstacle_points=obstacle_points, robot_radius=ROBOT_RADIUS,
    )

    assert obstacle_points == obstacle_points_before, "adding a hazard observation must never mutate obstacle_points"
    assert calls[0]["obstacle_points"] == obstacle_points_before
    assert result_grid.get_value(GridCell(hazard_row, hazard_col)) == OG_OCCUPIED, (
        "the hazard must still block the final grid -- via hazard_service/HazardBelief inside "
        "build_planning_grid_for_robot() itself, never via obstacle_points"
    )


# ---------------------------------------------------------------------------
# Case 7: ground-truth exclusion.
# ---------------------------------------------------------------------------


def test_ground_truth_obstacles_are_excluded_from_obstacle_points_preparation():
    ground_truth_rect = (4.0, 4.0, 2.0, 2.0)  # x, y, width, height
    fake = _make_fake_engine(obstacles=[ground_truth_rect])
    # No sensing was ever run -- mapped_obstacle_points starts and stays empty.

    obstacle_points, removed = fake.sanitize_planner_obstacle_points(
        list(fake.mapped_obstacle_points), start_xy=(0.0, 0.0), robot_radius=ROBOT_RADIUS, resolution=RESOLUTION,
    )
    assert obstacle_points == []
    assert removed == 0

    result_grid = fake.build_planning_grid_for_robot(
        fake.robot, obstacle_points=obstacle_points, robot_radius=ROBOT_RADIUS,
    )

    x, y, w, h = ground_truth_rect
    geometry = fake.belief_map.geometry
    steps = 5
    sampled_any = False
    for i in range(steps + 1):
        for j in range(steps + 1):
            cell = geometry.world_to_grid(x + w * i / steps, y + h * j / steps)
            if cell is None:
                continue
            sampled_any = True
            assert result_grid.get_value(cell) != OG_OCCUPIED, (
                f"cell {cell!r} inside ground-truth rectangle {ground_truth_rect} must not be "
                "occupied -- config.obstacles never reaches obstacle_points preparation"
            )
    assert sampled_any, "sanity: the rectangle actually maps to real grid cells"

    # Documented, explicit backstop location -- NOT this preparation path:
    # engine.py's predicted-motion collision check calls
    # self.collision_checker.check_predicted_motion(..., obstacles=self.
    # config.obstacles, ...), gated by a `use_ground_truth` flag, entirely
    # inside CONTINUOUS collision prediction. That call is never reachable
    # from sanitize_planner_obstacle_points()/build_planning_grid_for_
    # robot() -- see this file's module docstring and Case 10 below.


# ---------------------------------------------------------------------------
# Case 8: planner grid input -- executed for real across all three
# build_planner_kwargs* paths (single-robot, known-goal, multi-robot), each
# intercepting its OWN internal call to build_planning_grid_for_robot() via
# the same _capture_calls() recorder. None of these hand-compose a
# substitute call: every one below drives the real, already-bound
# orchestrating method.
#
# Under the runtime costmap-builder integration, obstacle_points and static/
# dynamic sanitization now happen INSIDE build_planning_grid_for_robot()
# itself (obstacle_points is None on every one of these calls -- that IS
# the new-path signal, see _capture_calls()'s own docstring), so "what was
# passed in" is no longer where static-point composition is observable.
# These tests instead verify the RESULT: the actual OccupancyGrid cells,
# plus (for radius/robot/dynamic points) the arguments still passed
# directly. Expected radius/dynamic points are still derived by calling the
# same real helper functions (sanitize_planner_obstacle_points()/
# dynamic_robot_obstacle_points_for_robot()), never a copied formula.
# ---------------------------------------------------------------------------


def test_build_planner_kwargs_intercepted_reveals_exact_planning_grid_input():
    fake = _make_fake_engine()
    near_point = (0.0, 0.0)  # exactly at the robot -- must be sanitized away
    far_point = (8.0, 0.0)  # far, but still within WORLD_X_MIN/MAX -- must survive
    fake.mapped_obstacle_points = [near_point, far_point]
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    result = fake.build_planner_kwargs((0.0, 0.0))

    assert len(calls) == 1, f"expected exactly one build_planning_grid_for_robot() call, got {len(calls)}: {calls!r}"
    call = calls[0]

    assert call["robot"] is fake.robot
    expected_radius = fake.safety_radius()
    assert call["robot_radius"] == expected_radius
    # NEW runtime path signal: no obstacle_points passed in, no dynamic
    # points either (single-robot path never has other robots to include).
    assert call["obstacle_points"] is None
    assert call["dynamic_obstacle_points"] == ()

    grid = call["result"]
    assert grid.get_value(grid.world_to_grid(*near_point)) != OG_OCCUPIED, (
        "the near-start static point must be sanitized away"
    )
    assert grid.get_value(grid.world_to_grid(*far_point)) == OG_OCCUPIED, "the far static point must block"
    assert result["planning_grid"] is call["result"]


def test_build_planner_kwargs_for_goal_intercepted_reveals_exact_planning_grid_input():
    fake = _make_fake_engine()
    near_point = (0.0, 0.0)  # exactly at the robot -- must be sanitized away
    far_point = (8.0, 0.0)  # far, but still within WORLD_X_MIN/MAX -- must survive
    fake.mapped_obstacle_points = [near_point, far_point]
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    start_xy = (0.0, 0.0)
    goal_xy = (9.0, 9.0)
    result = fake.build_planner_kwargs_for_goal(start_xy, goal_xy, robot=fake.robot)

    assert len(calls) == 1, f"expected exactly one build_planning_grid_for_robot() call, got {len(calls)}: {calls!r}"
    call = calls[0]

    assert call["robot"] is fake.robot
    expected_radius = fake.safety_radius_for_robot(fake.robot)
    assert call["robot_radius"] == expected_radius
    assert call["obstacle_points"] is None
    # build_planner_kwargs_for_goal() has no dynamic_robot_obstacle_points_
    # for_robot() call at all (AST-confirmed below: its traced
    # obstacle_points expression never contains "dynamic_points").
    assert call["dynamic_obstacle_points"] == ()

    grid = call["result"]
    assert grid.get_value(grid.world_to_grid(*near_point)) != OG_OCCUPIED, (
        "the near-start static point must be sanitized away"
    )
    assert grid.get_value(grid.world_to_grid(*far_point)) == OG_OCCUPIED, "the far static point must block"
    assert result["planning_grid"] is call["result"]


def test_build_planner_kwargs_for_multi_robot_intercepted_reveals_exact_composition():
    """Executes the REAL build_planner_kwargs_for_multi_robot() end to end
    and intercepts its own internal call to build_planning_grid_for_robot()
    -- this is the primary evidence for the multi-robot obstacle_points
    composition, not test_other_robot_position_enters_obstacle_points_via_
    dynamic_robot_points (which only unit-tests the helper in isolation).
    """
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    target_robot = fake.robots[0]
    target_start_xy = (float(target_robot.x), float(target_robot.y))

    near_point = target_start_xy  # exactly at robot 0's own position -- must be sanitized away
    far_point = (8.0, target_start_xy[1])  # far, but still within WORLD_X_MIN/MAX -- must survive
    fake.mapped_obstacle_points = [near_point, far_point]

    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    kwargs, _reason = _run_real_multi_robot_path(fake, robot_index=0)

    # 1. exactly one call.
    assert len(calls) == 1, f"expected exactly one build_planning_grid_for_robot() call, got {len(calls)}: {calls!r}"
    call = calls[0]

    # 2. correct target robot.
    assert call["robot"] is target_robot

    # 3. correct radius, via the real helper -- not a copied formula.
    expected_radius = fake.safety_radius_for_robot(target_robot)
    assert call["robot_radius"] == expected_radius

    # NEW runtime path signal: obstacle_points is never passed by the real
    # multi-robot orchestrator -- dynamic_obstacle_points carries the other
    # robot's points explicitly instead.
    assert call["obstacle_points"] is None

    # 4. exact dynamic points, built ONLY from the real helper -- no
    #    reimplemented circular sampling formula.
    expected_dynamic = tuple(fake.dynamic_robot_obstacle_points_for_robot(0))
    assert call["dynamic_obstacle_points"] == expected_dynamic

    grid = call["result"]
    # 5. the near-start static point was removed.
    assert grid.get_value(grid.world_to_grid(*near_point)) != OG_OCCUPIED, (
        "the near-start static point must be sanitized away"
    )
    # 6. the far static point remained.
    assert grid.get_value(grid.world_to_grid(*far_point)) == OG_OCCUPIED, "the far static point must block"
    # 7. the other robot's own position blocks the grid.
    other_robot_xy = (float(fake.robots[1].x), float(fake.robots[1].y))
    assert grid.get_value(grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED, (
        "the real multi-robot path must block the other robot's own position"
    )

    # 8. the SAME grid object the wrapper captured is what ends up in the
    #    returned kwargs.
    assert kwargs["planning_grid"] is call["result"]


# ---------------------------------------------------------------------------
# Case 9: reachability input.
# ---------------------------------------------------------------------------


def test_reachability_now_matches_multi_robot_planner_dynamic_composition():
    """Pins the FIX for the previously-confirmed gap: reachability used to
    never include other runtime robots at all, while the multi-robot
    planner did. Reuses the real make_exploration_reachability_check()
    entry point already exercised by test_reachability_instrumentation.py
    (same duck-typed-fake convention). The multi-robot side of the
    comparison is NOT hand-composed here: it goes through
    _run_real_multi_robot_path(), the same helper that drives
    build_planner_kwargs_for_multi_robot() for real in
    test_build_planner_kwargs_for_multi_robot_intercepted_reveals_exact_
    composition above, sharing the SAME fake/wrapped-recorder so both
    captures come from actually running the real production methods.
    """
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    fake.mapped_obstacle_points = []
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    is_reachable = fake.make_exploration_reachability_check(fake.robots[0])
    is_reachable((9.0, 9.0))  # first invocation triggers the lazy _build_context()

    assert len(calls) == 1, "reachability must build exactly one grid on its first use"
    reachability_call = calls[0]

    _run_real_multi_robot_path(fake, robot_index=0)

    assert len(calls) == 2, "the REAL multi-robot path above goes through the same wrapped method"
    multi_robot_call = calls[1]

    # dynamic_robot_obstacle_points_for_robot() is called here only to get
    # a REFERENCE set for the containment checks below -- not to hand-build
    # a competing grid/obstacle_points list (that role is now filled by the
    # real _run_real_multi_robot_path()/make_exploration_reachability_check()
    # calls above).
    dynamic_points_reference = tuple(fake.dynamic_robot_obstacle_points_for_robot(0))
    assert dynamic_points_reference, "sanity: a second robot exists, so there is something to reference"

    # THE FIX: reachability's own dynamic_obstacle_points now equals the
    # SAME reference the multi-robot planner uses for this robot -- both
    # resolve robot index 0 and exclude only itself.
    assert reachability_call["dynamic_obstacle_points"] == dynamic_points_reference
    assert multi_robot_call["dynamic_obstacle_points"] == dynamic_points_reference
    assert reachability_call["dynamic_obstacle_points"] == multi_robot_call["dynamic_obstacle_points"]

    other_robot_xy = (float(fake.robots[1].x), float(fake.robots[1].y))
    reachability_grid = reachability_call["result"]
    multi_robot_grid = multi_robot_call["result"]
    assert reachability_grid.get_value(reachability_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED, (
        "reachability's own grid must now block the other robot's position too"
    )
    assert multi_robot_grid.get_value(multi_robot_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED

    # No object-identity requirement (per this migration's own contract):
    # each caller still builds its OWN OccupancyGrid independently, even
    # though both now use the same layers/composition.
    assert reachability_grid is not multi_robot_grid


# ---------------------------------------------------------------------------
# Case 10: continuous safety is separate.
# ---------------------------------------------------------------------------


def test_continuous_safety_points_match_planning_for_static_geometry_but_never_add_dynamic_points_themselves():
    fake = _make_fake_engine()
    fake.mapped_obstacle_points = [(6.0, 6.0), (0.05, 0.05)]  # one far, one near-start
    start_xy = (0.0, 0.0)
    robot_radius = fake.safety_radius_for_robot(fake.robot)

    planning_points, _ = fake.sanitize_planner_obstacle_points(
        list(fake.mapped_obstacle_points), start_xy=start_xy, robot_radius=robot_radius, resolution=RESOLUTION,
    )
    safety_points = fake.obstacle_points_for_segment_safety_check(start_xy, robot_radius)

    assert safety_points == planning_points, (
        "for STATIC-only geometry, obstacle_points_for_segment_safety_check() uses the exact "
        "same sanitize_planner_obstacle_points(mapped_obstacle_points, ...) composition "
        "planning uses -- documented as intentional in its own docstring. No equality is "
        "asserted beyond this static case: see below for the deliberate divergence."
    )

    # Deliberate divergence: dynamic other-robot points are never inside
    # obstacle_points_for_segment_safety_check() itself. Production call
    # sites append them manually (engine.py: `+ dynamic_pts`/`+ dynamic_
    # points` at its own call sites) -- the SAME caller-side composition
    # pattern sanitize_planner_obstacle_points()'s own callers use for
    # planning (see Case 4/9). Neither function bakes dynamic points in.
    fake_multi = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    dynamic_points = fake_multi.dynamic_robot_obstacle_points_for_robot(0)
    safety_points_multi = fake_multi.obstacle_points_for_segment_safety_check((0.0, 0.0), robot_radius)
    assert not (set(dynamic_points) & set(safety_points_multi)), (
        "obstacle_points_for_segment_safety_check() never includes dynamic other-robot points "
        "on its own -- exactly like sanitize_planner_obstacle_points(), callers must add them "
        "explicitly"
    )
    # Ground truth is absent from both sides too -- see Case 7 for the one
    # documented backstop location (predicted-motion collision), which
    # neither of these two functions is part of.


# ---------------------------------------------------------------------------
# AST inspection obligatoria: detailed inventory of every
# build_planning_grid_for_robot() call site inside engine.py, including
# keyword-vs-positional passing and the exact source expression used for
# obstacle_points/robot_radius -- never a grep-only assertion.
# ---------------------------------------------------------------------------

_TARGET_METHOD = "build_planning_grid_for_robot"
_EXCLUDED_DIR_NAMES = {"tests", "__pycache__"}
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../robotics_sim
_REPO_ROOT = _PACKAGE_ROOT.parent
_ENGINE_PATH = _PACKAGE_ROOT / "simulation" / "engine.py"


def _last_assignment_source(function_node, target_name: str, before_lineno: int, source: str) -> str | None:
    """Within one FunctionDef/AsyncFunctionDef's own body, find the most
    recent (highest lineno strictly before before_lineno) top-level
    assignment to target_name (plain `x = ...` or tuple-unpacking `x, y =
    ...`) and return the AST source text of its RHS value expression. This
    is what actually reveals composition differences between call sites
    that all pass the SAME bare identifier as a keyword argument -- the
    real difference lives in how that local variable was built.
    """
    best = None
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Assign) or node.lineno >= before_lineno:
            continue
        target_names: list[str] = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                target_names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                target_names.extend(elt.id for elt in target.elts if isinstance(elt, ast.Name))
        if target_name in target_names and (best is None or node.lineno > best.lineno):
            best = node
    return ast.get_source_segment(source, best.value) if best is not None else None


_TRACKED_KEYWORDS = ("obstacle_points", "robot_radius", "dynamic_obstacle_points")


class _PlanningGridCallVisitor(ast.NodeVisitor):
    """Records every call to .build_planning_grid_for_robot(...) in one
    module's AST: its nearest enclosing def/async def (innermost lexical
    scope, "<module>" if none), line number, and -- for each of
    _TRACKED_KEYWORDS -- whether it is passed by keyword, the exact source
    expression used, and (when that expression is a bare identifier) the
    traced source of its most recent local assignment.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._function_stack: list[ast.AST] = []
        self.calls: list[dict] = []

    def _visit_function(self, node) -> None:
        self._function_stack.append(node)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == _TARGET_METHOD:
            enclosing_node = self._function_stack[-1] if self._function_stack else None
            enclosing_name = enclosing_node.name if enclosing_node is not None else "<module>"

            keyword_values = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
            positional_arg_exprs = [ast.get_source_segment(self._source, arg) for arg in node.args]

            record = {
                "enclosing_function": enclosing_name,
                "lineno": node.lineno,
                "positional_arg_exprs": positional_arg_exprs,
            }

            for keyword_name in _TRACKED_KEYWORDS:
                is_keyword = keyword_name in keyword_values
                call_expr = (
                    ast.get_source_segment(self._source, keyword_values[keyword_name])
                    if is_keyword
                    else None
                )
                traced_expr = None
                if enclosing_node is not None and call_expr is not None and call_expr.isidentifier():
                    traced_expr = _last_assignment_source(
                        enclosing_node, call_expr, node.lineno, self._source,
                    )
                record[f"{keyword_name}_is_keyword"] = is_keyword
                record[f"{keyword_name}_call_expr"] = call_expr
                record[f"{keyword_name}_traced_expr"] = traced_expr

            self.calls.append(record)
        self.generic_visit(node)


def _find_planning_grid_calls(path: Path) -> list[dict]:
    """AST-parses one production file's own source text (never a text/
    substring search) and returns every call-site record for
    build_planning_grid_for_robot() found in it."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _PlanningGridCallVisitor(source)
    visitor.visit(tree)
    return visitor.calls


def test_build_planning_grid_for_robot_call_sites_are_inventoried_with_composition_detail():
    """Under the runtime costmap-builder integration, obstacle_points is
    never passed at these 4 call sites anymore -- that IS the signal that
    routes each of them through PlanningCostmapBuilder inside
    build_planning_grid_for_robot() itself (obstacle_points is None
    triggers the new path; see that method's own docstring). What
    distinguishes the 4 call sites now is dynamic_obstacle_points instead.
    """
    calls = _find_planning_grid_calls(_ENGINE_PATH)
    enclosing_names = [call["enclosing_function"] for call in calls]

    assert len(calls) == 4, (
        f"expected exactly 4 build_planning_grid_for_robot() call sites in engine.py, "
        f"found {len(calls)}: {calls!r}"
    )
    assert len(enclosing_names) == len(set(enclosing_names)), (
        "a second call inside the SAME enclosing function would otherwise collapse silently "
        f"when indexed by enclosing function name: {enclosing_names!r}"
    )
    assert set(enclosing_names) == {
        "build_planner_kwargs",
        "build_planner_kwargs_for_goal",
        "build_planner_kwargs_for_multi_robot",
        "_build_context",
    }

    by_function = {call["enclosing_function"]: call for call in calls}

    for name, call in by_function.items():
        assert call["obstacle_points_is_keyword"] is False, (name, call)
        assert call["robot_radius_is_keyword"] is True, (name, call)
        assert call["robot_radius_call_expr"] == "robot_radius", (name, call)

    # Only build_planner_kwargs_for_multi_robot() and _build_context()
    # (reachability) pass dynamic_obstacle_points at all -- the single-
    # robot/known-goal paths never have another robot to include (confirmed
    # by the ABSENCE of the keyword here, not assumed from the name).
    for name in ("build_planner_kwargs", "build_planner_kwargs_for_goal"):
        assert by_function[name]["dynamic_obstacle_points_is_keyword"] is False, name

    for name in ("build_planner_kwargs_for_multi_robot", "_build_context"):
        call = by_function[name]
        assert call["dynamic_obstacle_points_is_keyword"] is True, name
        expr = call["dynamic_obstacle_points_call_expr"] or ""
        assert "dynamic_points" in expr, (name, expr)

    # build_planner_kwargs_for_multi_robot() wraps its dynamic points in
    # tuple(...) right at the call site -- visible directly, no need to
    # trace further back.
    assert (
        by_function["build_planner_kwargs_for_multi_robot"]["dynamic_obstacle_points_call_expr"]
        == "tuple(dynamic_points)"
    )

    # _build_context()'s dynamic_obstacle_points is a bare identifier --
    # trace it back to its own assignment to confirm it comes from
    # _dynamic_obstacle_points_for_robot_object(), not a copied formula.
    reachability_traced = by_function["_build_context"]["dynamic_obstacle_points_traced_expr"] or ""
    assert "_dynamic_obstacle_points_for_robot_object" in reachability_traced

    # Ground truth never enters through any tracked keyword at any call site.
    for name, call in by_function.items():
        for keyword_name in ("obstacle_points", "robot_radius", "dynamic_obstacle_points"):
            expr = call.get(f"{keyword_name}_call_expr") or ""
            traced = call.get(f"{keyword_name}_traced_expr") or ""
            assert "self.config.obstacles" not in expr, (name, keyword_name, expr)
            assert "self.config.obstacles" not in traced, (name, keyword_name, traced)


# ---------------------------------------------------------------------------
# Tripwire AST: whole-package inventory of build_planning_grid_for_robot()
# call sites, so a NEW call site anywhere in robotics_sim/ (not just
# engine.py) fails this test until classified.
# ---------------------------------------------------------------------------


def _iter_production_python_files():
    """Every .py file under robotics_sim/, excluding robotics_sim/tests/
    and __pycache__/ (no other generated files exist in this tree today)."""
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative_parts = path.relative_to(_PACKAGE_ROOT).parts
        if _EXCLUDED_DIR_NAMES.intersection(relative_parts):
            continue
        yield path


def _module_name_for_path(path: Path) -> str:
    parts = path.relative_to(_REPO_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def test_all_production_build_planning_grid_for_robot_call_sites_are_the_audited_set():
    """Regression tripwire for this audit's call-site inventory (see module
    docstring). Walks every production .py file under robotics_sim/ and
    tallies a Counter over (module, enclosing_function, called_method) for
    every call to build_planning_grid_for_robot() found anywhere in the
    package -- not just engine.py. A Counter (never a set) so a SECOND call
    inside an already-known function also changes the tally instead of
    being silently deduplicated. A brand-new call site anywhere in
    robotics_sim/, a new call in a different function of an already-known
    module, or a second call inside an already-known function all change
    this Counter and fail the test -- forcing that new/changed call site to
    be classified (which obstacle_points source(s) feed it) before anyone
    assumes PlanningCostmapBuilder integration is still safe.
    """
    found: Counter[tuple[str, str, str]] = Counter()
    call_lines: dict[tuple[str, str, str], list[int]] = {}

    for path in _iter_production_python_files():
        module_name = _module_name_for_path(path)
        for call in _find_planning_grid_calls(path):
            key = (module_name, call["enclosing_function"], _TARGET_METHOD)
            found[key] += 1
            call_lines.setdefault(key, []).append(call["lineno"])

    audited = Counter(
        {
            ("robotics_sim.simulation.engine", "build_planner_kwargs", _TARGET_METHOD): 1,
            ("robotics_sim.simulation.engine", "build_planner_kwargs_for_goal", _TARGET_METHOD): 1,
            ("robotics_sim.simulation.engine", "build_planner_kwargs_for_multi_robot", _TARGET_METHOD): 1,
            ("robotics_sim.simulation.engine", "_build_context", _TARGET_METHOD): 1,
        }
    )

    if found != audited:
        all_keys = sorted(set(found) | set(audited))
        new_sites = {key: found[key] for key in all_keys if found[key] and not audited[key]}
        missing_sites = {key: audited[key] for key in all_keys if audited[key] and not found[key]}
        differing_counts = {
            key: {"expected": audited[key], "found": found[key]}
            for key in all_keys
            if found[key] != audited[key] and key not in new_sites and key not in missing_sites
        }
        lines_by_site = {key: call_lines.get(key, []) for key in all_keys}
        raise AssertionError(
            "unaudited build_planning_grid_for_robot() call site(s) detected under "
            "robotics_sim/ (production code only, robotics_sim/tests/ and __pycache__/ "
            "excluded).\n"
            f"new call sites (module, enclosing_function, called_method) -> count: {new_sites!r}\n"
            f"missing call sites (expected but not found): {missing_sites!r}\n"
            f"differing counts (expected vs. found): {differing_counts!r}\n"
            f"line numbers found per call site (for diagnosis): {lines_by_site!r}\n"
            "Classify what obstacle_points composition this new/changed call site actually "
            "uses (static / per-robot sanitized / other-robot / dynamic / ground-truth / "
            "hazard / other) before assuming PlanningCostmapBuilder integration is still safe."
        )
