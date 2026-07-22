"""
Characterization tests for the current, real routes the simulator uses to
construct or interpret occupancy for planning, exploration, reachability,
multi-robot frontier detection, path simplification, and continuous safety.

Purpose: pin down actual behavior, via real execution of the real functions,
BEFORE any unification work happens (see the architecture plan discussed on
this branch). This file does not fix anything and does not judge whether a
divergence is "correct" -- it documents what the current code does, as
evidence for later phases (e.g. a PlanningCostmapBuilder unifying the
several construction routes characterized here).

Covered routes:
    1. SimulationControllerMixin.build_planning_grid_for_robot() includes
       observed obstacle points (mapped_obstacle_points) that BeliefMap.grid
       alone does not know about.
    2. build_planning_grid_for_robot() includes observed hazards
       (HazardBelief, via apply_hazard_belief_to_planning_grid()).
    3. FoVAwareDirectionalFrontierPlanner.select_goal()'s FALLBACK scoring
       grid (belief.to_planning_grid(), built without mapped obstacle
       points) -- used only when no caller supplies planning_grid= --
       omits what (1) shows the real runtime grid includes. engine.py's
       select_navigation_goal() now supplies a PlanningCostmapBuilder-
       backed planning_grid= on the real runtime path (see
       test_fov_costmap_integration.py); the two tests here characterize
       only the pre-existing, still-unchanged fallback used by direct/test
       callers that never pass that kwarg.
    4. The same fallback scoring grid omits what (2) shows the real runtime
       grid includes for hazards -- same caveat as (3).
    5. candidate_reachable_on_planning_grid() can be driven with the EXACT
       runtime grid build_planning_grid_for_robot() produces.
    6. AStarPlanner.plan() and the path simplifier receive the same grid
       object within one compute_planned_waypoints() call.
    7. coordinated_frontier_planner._occupied_cells_from_points() (the
       multi-robot frontier rasterizer) and OccupancyGrid.add_obstacle_points()
       (what the single-robot runtime grid uses) can disagree about which
       world locations are occupied, for the same input point/resolution/
       robot_radius.
    8. Continuous safety (CollisionChecker.check_segment_points(), fed via
       obstacle_points_for_segment_safety_check()) treats an OCCUPIED
       BeliefMap cell and an observed mapped_obstacle_point as two distinct
       things -- belief occupancy alone is not continuous collision
       geometry.

All fakes below are lightweight duck-typed SimpleNamespace objects binding
the REAL SimulationControllerMixin methods under characterization -- the
same pattern already used by test_first_segment_validation_consistency.py
and test_discovered_hazard_planning.py -- not mocks, and nothing here reads
production source as text to make an assertion.
"""
from __future__ import annotations

from types import SimpleNamespace

import robotics_sim.planning.exploration_planners as exploration_planners
import robotics_sim.planning.planner_registry as planner_registry_module
from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.environment.occupancy_grid import (
    FREE as OG_FREE,
    OCCUPIED as OG_OCCUPIED,
    UNKNOWN as OG_UNKNOWN,
    OccupancyGrid,
)
from robotics_sim.planning.coordinated_frontier_planner import (
    _cell_center,
    _occupied_cells_from_points,
)
from robotics_sim.planning.exploration_planners import select_exploration_goal
from robotics_sim.planning.grid_planners import AStarPlanner
from robotics_sim.planning.planner_registry import compute_planned_waypoints
from robotics_sim.simulation.engine import (
    SimulationControllerMixin,
    candidate_reachable_on_planning_grid,
)
from robotics_sim.simulation.hazard_service import RuntimeHazardService

BOUNDS = (0.0, 10.0, 0.0, 10.0)
RESOLUTION = 1.0  # -> 10x10 grid, cell centers at 0.5, 1.5, ..., 9.5
ROBOT_RADIUS = 0.3


def _make_belief_all_free(bounds=BOUNDS, resolution=RESOLUTION, robot_count: int = 1) -> BeliefMap:
    belief = BeliefMap(bounds=bounds, resolution=resolution, robot_count=robot_count)
    for row in range(belief.height):
        for col in range(belief.width):
            belief.mark_free_cell((row, col))
    return belief


def _make_fake_engine(
    *,
    belief_map: BeliefMap,
    mapped_obstacle_points: list[tuple[float, float]] | None = None,
    hazard_service: RuntimeHazardService | None = None,
    grid_resolution: float = RESOLUTION,
    body_radius: float = ROBOT_RADIUS,
    safety_radius: float = ROBOT_RADIUS,
) -> SimpleNamespace:
    """Lightweight duck-typed engine fake.

    Binds the REAL SimulationControllerMixin methods under characterization
    to a SimpleNamespace; only their own direct collaborator
    (ensure_belief_map) is stubbed, so this test controls exactly which
    BeliefMap instance is used without going through reset_belief_map()'s
    extra config/canvas plumbing.
    """
    config = SimpleNamespace(
        grid_resolution=grid_resolution,
        planner_type="A*",
        goal_tolerance=0.25,
        body_radius=body_radius,
        safety_radius=safety_radius,
        obstacles=[],
    )
    robot = SimpleNamespace(x=0.0, y=0.0, theta=0.0, vision=3.0)
    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        config=config,
        mapped_obstacle_points=list(mapped_obstacle_points or []),
        belief_map=belief_map,
        hazard_service=hazard_service,
        collision_checker=CollisionChecker(),
    )
    fake.ensure_belief_map = lambda: fake.belief_map
    for name in (
        "safety_radius_for_robot",
        "safety_radius",
        "body_radius_for_robot",
        "body_radius",
        "sanitize_planner_obstacle_points",
        "build_planning_grid_for_robot",
        "obstacle_points_for_segment_safety_check",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


# ---------------------------------------------------------------------------
# 1. Runtime planning grid includes observed obstacle points.
# ---------------------------------------------------------------------------


def test_runtime_planning_grid_includes_observed_obstacle_points():
    belief = _make_belief_all_free()
    grid_before = belief.grid.copy()

    obstacle_point = (5.5, 5.5)  # a cell the belief itself currently says is FREE
    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[obstacle_point])

    planning_grid = fake.build_planning_grid_for_robot(
        fake.robot, obstacle_points=[obstacle_point], robot_radius=ROBOT_RADIUS,
    )

    cell = planning_grid.world_to_grid(*obstacle_point)
    assert planning_grid.get_value(cell) == OG_OCCUPIED
    assert (belief.grid == grid_before).all(), "belief_map.grid must never be mutated by building a planning grid"
    assert planning_grid.bounds == belief.bounds
    assert planning_grid.resolution == belief.resolution


# ---------------------------------------------------------------------------
# 2. Runtime planning grid excludes observed hazards for aerial robots.
# ---------------------------------------------------------------------------


def test_runtime_planning_grid_keeps_observed_hazards_traversable():
    belief = _make_belief_all_free()
    grid_before = belief.grid.copy()

    hazard_service = RuntimeHazardService(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1, block_threshold=0.55)
    observed_point = (5.5, 5.5)
    row, col = belief.world_to_cell(observed_point)
    hazard_service.belief.observe_cells([row], [col], [0.8], robot_index=0)  # observed, above threshold
    unobserved_point = (2.5, 2.5)  # never observed at all -- ground truth is irrelevant here

    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[], hazard_service=hazard_service)

    planning_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=[], robot_radius=ROBOT_RADIUS)

    hazard_cell = planning_grid.world_to_grid(*observed_point)
    other_cell = planning_grid.world_to_grid(*unobserved_point)
    assert planning_grid.get_value(hazard_cell) != OG_OCCUPIED
    assert planning_grid.get_value(other_cell) != OG_OCCUPIED
    assert (belief.grid == grid_before).all(), "hazard sensing must never mutate occupancy"


# ---------------------------------------------------------------------------
# 3. FoV-aware internal scoring grid currently omits mapped obstacle
#    projection that the real runtime grid (case 1) includes.
# ---------------------------------------------------------------------------


def test_fov_scoring_grid_currently_omits_mapped_obstacle_projection(monkeypatch):
    belief = _make_belief_all_free()
    robot_xy = (0.5, 5.5)
    obstacle_point = (2.5, 5.5)  # directly ahead of the robot (heading=0), within max_forward_distance

    captured: dict = {}
    original_score_candidate = exploration_planners._score_candidate

    def _spy(**kwargs):
        captured.setdefault("planning_grid", kwargs["planning_grid"])
        return original_score_candidate(**kwargs)

    monkeypatch.setattr(exploration_planners, "_score_candidate", _spy)

    result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief,
        robot_xy=robot_xy,
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_count=1,
        robot_radius=ROBOT_RADIUS,
        sensor_range=3.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
    )

    assert result.success
    assert "planning_grid" in captured, "expected select_goal() to score at least one candidate"
    internal_grid = captured["planning_grid"]
    internal_cell = internal_grid.world_to_grid(*obstacle_point)

    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[obstacle_point])
    runtime_grid = fake.build_planning_grid_for_robot(
        fake.robot, obstacle_points=[obstacle_point], robot_radius=ROBOT_RADIUS,
    )
    runtime_cell = runtime_grid.world_to_grid(*obstacle_point)

    assert runtime_grid.get_value(runtime_cell) == OG_OCCUPIED, (
        "sanity: the real runtime planning grid does represent the mapped obstacle point"
    )
    assert internal_grid.get_value(internal_cell) != OG_OCCUPIED, (
        "when no planning_grid= is supplied (as here -- a direct select_exploration_goal() "
        "call, not through engine.py's select_navigation_goal()), FoVAwareDirectionalFrontier"
        "Planner.select_goal() still falls back to belief.to_planning_grid() alone, with no "
        "mapped obstacle points -- unchanged fallback behavior. The REAL runtime path "
        "(engine.py's select_navigation_goal()) now supplies a PlanningCostmapBuilder-backed "
        "planning_grid= that DOES include mapped obstacle points -- see "
        "test_fov_costmap_integration.py for that path's own coverage."
    )


# ---------------------------------------------------------------------------
# 4. FoV scoring and runtime navigation both keep hazards traversable.
# ---------------------------------------------------------------------------


def test_fov_scoring_and_runtime_both_keep_hazard_traversable(monkeypatch):
    belief = _make_belief_all_free()
    robot_xy = (0.5, 5.5)
    hazard_point = (2.5, 5.5)  # directly ahead of the robot, within max_forward_distance

    hazard_service = RuntimeHazardService(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1, block_threshold=0.55)
    row, col = belief.world_to_cell(hazard_point)
    hazard_service.belief.observe_cells([row], [col], [0.8], robot_index=0)

    captured: dict = {}
    original_score_candidate = exploration_planners._score_candidate

    def _spy(**kwargs):
        captured.setdefault("planning_grid", kwargs["planning_grid"])
        return original_score_candidate(**kwargs)

    monkeypatch.setattr(exploration_planners, "_score_candidate", _spy)

    result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief,
        robot_xy=robot_xy,
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_count=1,
        robot_radius=ROBOT_RADIUS,
        sensor_range=3.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
    )

    assert result.success
    assert "planning_grid" in captured
    internal_grid = captured["planning_grid"]
    internal_cell = internal_grid.world_to_grid(*hazard_point)

    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[], hazard_service=hazard_service)
    runtime_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=[], robot_radius=ROBOT_RADIUS)
    runtime_cell = runtime_grid.world_to_grid(*hazard_point)

    assert runtime_grid.get_value(runtime_cell) != OG_OCCUPIED
    assert internal_grid.get_value(internal_cell) != OG_OCCUPIED, (
        "hazard is an information layer and must not become physical occupancy"
    )


# ---------------------------------------------------------------------------
# 5. Reachability can consume the exact runtime grid.
# ---------------------------------------------------------------------------


def test_reachability_consumes_the_exact_runtime_planning_grid():
    belief = _make_belief_all_free()
    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[])

    # A small radius keeps add_obstacle_points()'s padding (radius +
    # resolution*sqrt(2)/2) under 1 resolution cell, so it marks exactly the
    # wall cells below and nothing else -- letting this test reverse the
    # block by mutating the SAME grid object rather than rebuilding one.
    radius = 0.1
    start_xy = (0.5, 5.5)
    candidate_xy = (9.5, 5.5)
    wall = [(5.5, y + 0.5) for y in range(10)]  # full-height wall: no route around it

    planning_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=wall, robot_radius=radius)

    blocked = candidate_reachable_on_planning_grid(
        planning_grid, "A*", start_xy, candidate_xy,
        bounds=BOUNDS, resolution=RESOLUTION, robot_radius=radius, goal_tolerance=0.25,
    )
    assert blocked is False

    # Remove the wall directly from the SAME runtime grid object -- not a
    # freshly (re)built equivalent grid -- then re-check reachability.
    # (candidate_reachable_on_planning_grid() itself calls planning_grid.copy()
    # once internally before running A*; that per-call copy is not what is
    # being mutated or re-asserted on here -- planning_grid, the object this
    # test owns and passes in both times, is.)
    for point in wall:
        planning_grid.set_value(planning_grid.world_to_grid(*point), OG_FREE)

    reachable = candidate_reachable_on_planning_grid(
        planning_grid, "A*", start_xy, candidate_xy,
        bounds=BOUNDS, resolution=RESOLUTION, robot_radius=radius, goal_tolerance=0.25,
    )
    assert reachable is True


# ---------------------------------------------------------------------------
# 6. A* and the path simplifier share one grid object within one
#    compute_planned_waypoints() call.
# ---------------------------------------------------------------------------


def test_astar_and_simplifier_share_the_same_grid_object_within_one_call(monkeypatch):
    belief = _make_belief_all_free()
    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[])

    radius = 0.1  # keeps inflation from spilling into the gap rows below
    # Wall blocks column 5 for rows 0-7 (y=0.5..7.5); rows 8-9 stay open as
    # a gap, forcing a real detour -- not a direct line-of-sight shortcut.
    wall_with_gap = [(5.5, y + 0.5) for y in range(8)]

    planning_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=wall_with_gap, robot_radius=radius)

    captured: dict = {}
    original_plan = AStarPlanner.plan

    def _spy_plan(self, *args, **kwargs):
        captured["astar_grid"] = kwargs.get("grid", args[0] if args else None)
        return original_plan(self, *args, **kwargs)

    monkeypatch.setattr(AStarPlanner, "plan", _spy_plan)

    original_simplify = planner_registry_module.simplify_grid_path

    def _spy_simplify(path, *, method, grid=None):
        captured["simplifier_grid"] = grid
        return original_simplify(path, method=method, grid=grid)

    monkeypatch.setattr(planner_registry_module, "simplify_grid_path", _spy_simplify)

    success, reason, waypoints = compute_planned_waypoints(
        planner_type="A*",
        start_xy=(0.5, 0.5),
        goal_xy=(9.5, 0.5),
        bounds=BOUNDS,
        resolution=RESOLUTION,
        robot_radius=radius,
        planning_grid=planning_grid,
        unknown_is_traversable=True,
        obstacle_points=[],
    )

    assert success, reason
    assert waypoints
    assert "astar_grid" in captured, "expected real A* to run (no direct line-of-sight shortcut through the wall)"
    assert "simplifier_grid" in captured
    assert captured["astar_grid"] is captured["simplifier_grid"], (
        "A* and the path simplifier must receive the exact same grid object "
        "within one compute_planned_waypoints() call"
    )
    # compute_planned_waypoints() copies a caller-provided planning_grid once
    # per call (see planner_registry._build_planning_grid()) -- so the
    # shared object above is that private per-call copy, not our own input.
    assert captured["astar_grid"] is not planning_grid


# ---------------------------------------------------------------------------
# 7. Multi-robot rasterizer differs from the OccupancyGrid projection the
#    single-robot runtime grid uses, for the same input.
# ---------------------------------------------------------------------------


def test_multi_robot_rasterizer_differs_from_occupancy_grid_projection():
    # Empirically confirmed (not asserted from formula alone): the two
    # functions disagree for this ordinary point/resolution/robot_radius,
    # not only in some razor-thin band between resolution*0.75 and
    # resolution*sqrt(2)/2. coordinated_frontier_planner._cell_key() rounds
    # to the nearest multiple of resolution (round(x/resolution)) while
    # OccupancyGrid floors relative to its origin with a +0.5-cell center
    # offset -- two different grid alignments, not merely two different
    # inflation constants. This test reports both formulas without deciding
    # which (if either) should become the shared source of truth.
    point = (5.5, 5.5)

    coordinated_cells = _occupied_cells_from_points([point], RESOLUTION, ROBOT_RADIUS)
    coordinated_world = {_cell_center(cell, RESOLUTION) for cell in coordinated_cells}

    grid = OccupancyGrid.from_bounds(*BOUNDS, RESOLUTION, initial_value=OG_UNKNOWN)
    grid.add_obstacle_points([point], padding=ROBOT_RADIUS)
    occupancy_world = {
        grid.grid_to_world(GridCell(row, col))
        for row in range(grid.height)
        for col in range(grid.width)
        if grid.get_value(GridCell(row, col)) == OG_OCCUPIED
    }

    difference = coordinated_world.symmetric_difference(occupancy_world)
    assert difference, (
        "expected coordinated_frontier_planner._occupied_cells_from_points() and "
        "OccupancyGrid.add_obstacle_points() to disagree about at least one "
        f"occupied world location for point={point!r}, resolution={RESOLUTION}, "
        f"robot_radius={ROBOT_RADIUS}; "
        f"_occupied_cells_from_points world cells={sorted(coordinated_world)!r}, "
        f"OccupancyGrid world cells={sorted(occupancy_world)!r}, "
        f"symmetric difference={sorted(difference)!r} "
        "(if this ever becomes empty, the two rasterizers would have converged "
        "and this characterization test itself would need re-deriving a fixture "
        "that still shows a difference, or documenting that none was found)"
    )


# ---------------------------------------------------------------------------
# 8. Continuous safety uses observed points, not belief occupancy.
# ---------------------------------------------------------------------------


def test_continuous_safety_uses_observed_points_not_belief_occupancy():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    occupied_point = (5.5, 5.5)
    occupied_cell = belief.world_to_cell(occupied_point)
    belief.mark_occupied_cell(occupied_cell)
    assert belief.cell_state(occupied_point) != 0  # sanity: belief really does say OCCUPIED here

    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[])

    # Far enough from the OCCUPIED cell that sanitize_planner_obstacle_points()'s
    # near-start clearing (see obstacle_points_for_segment_safety_check()'s
    # own docstring) cannot remove the point once it IS observed below.
    start = (3.0, 5.5)
    end = (8.0, 5.5)  # segment passes directly through (5.5, 5.5)
    robot_radius = 0.3

    obstacle_points_before = fake.obstacle_points_for_segment_safety_check(start, robot_radius)
    report_before = fake.collision_checker.check_segment_points(
        start=start, end=end, obstacle_points=obstacle_points_before, robot_radius=robot_radius,
    )
    assert report_before.collision is False, (
        "an OCCUPIED belief_map cell alone must not become continuous collision "
        "geometry for check_segment_points -- obstacle_points_for_segment_safety_check() "
        "only ever reads mapped_obstacle_points, never belief_map.grid"
    )

    # Now the sensor actually observes/maps that same location.
    fake.mapped_obstacle_points.append(occupied_point)

    obstacle_points_after = fake.obstacle_points_for_segment_safety_check(start, robot_radius)
    report_after = fake.collision_checker.check_segment_points(
        start=start, end=end, obstacle_points=obstacle_points_after, robot_radius=robot_radius,
    )
    assert report_after.collision is True, (
        "once the point is an observed mapped_obstacle_point, continuous safety "
        "does react to it"
    )
