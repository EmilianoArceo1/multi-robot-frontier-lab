"""
Integration tests: the runtime costmap migration (robotics_sim/simulation/
engine.py) actually routes SimulationControllerMixin.build_planner_kwargs(),
build_planner_kwargs_for_goal(), build_planner_kwargs_for_multi_robot(), and
make_exploration_reachability_check() through PlanningCostmapBuilder
(robotics_sim/planning/planning_costmap_builder.py), instead of each
building its own OccupancyGrid ad hoc.

This is NOT another characterization file: robotics_sim/tests/
test_planning_obstacle_input_characterization.py and robotics_sim/tests/
test_observed_obstacle_coverage_characterization.py already document the
PRE-migration composition in exhaustive detail; this file pins the POST-
migration runtime contract -- that the builder is genuinely wired in, that
static/dynamic/hazard stay separated end to end, that reachability and the
multi-robot planner now agree on which cells are blocked, that the legacy
direct-obstacle_points call path still works unchanged, and that no
production caller double-projects the same static point through two
different lists at once.

Fakes bind the REAL SimulationControllerMixin methods under test (the same
convention already used throughout this test suite -- see
test_planning_obstacle_input_characterization.py's own module docstring for
the precedent). Where "does this call PlanningCostmapBuilder" needs to be
observed directly, a monkeypatch spy wraps the REAL
PlanningCostmapBuilder.build classmethod-equivalent (an instance method) and
delegates to it -- never a reimplementation of what it does.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.environment.occupancy_grid import OCCUPIED as OG_OCCUPIED
from robotics_sim.environment.occupancy_grid import OccupancyGrid
from robotics_sim.planning.planning_costmap_builder import PlanningCostmapBuilder
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
    # to the costmap-builder integration) -- both goal-seeking branches
    # (single- and multi-robot) tolerate None.
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


def _spy_planning_costmap_builder(monkeypatch) -> list[dict]:
    """Wrap the REAL PlanningCostmapBuilder.build with a monkeypatched spy
    that records each call's kwargs before delegating to the real
    implementation -- proves the runtime actually calls the builder,
    without copying or reimplementing anything it does."""
    calls: list[dict] = []
    real_build = PlanningCostmapBuilder.build

    def _spy(self, **kwargs):
        calls.append(kwargs)
        return real_build(self, **kwargs)

    monkeypatch.setattr(PlanningCostmapBuilder, "build", _spy)
    return calls


def _capture_calls(fake: SimpleNamespace, method_name: str) -> list[dict]:
    """Wrap the ALREADY-bound real method on fake, recording every call's
    arguments and return value before delegating to the real
    implementation."""
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
# 1. Single robot.
# ---------------------------------------------------------------------------


def test_single_robot_build_planner_kwargs_uses_builder_and_static_blocks(monkeypatch):
    fake = _make_fake_engine()
    static_point = (5.0, 5.0)
    fake.mapped_obstacle_points = [static_point]
    points_before = list(fake.mapped_obstacle_points)
    builder_calls = _spy_planning_costmap_builder(monkeypatch)

    result = fake.build_planner_kwargs((0.0, 0.0))

    assert len(builder_calls) == 1, "build_planner_kwargs() must route through PlanningCostmapBuilder exactly once"
    grid = result["planning_grid"]
    assert isinstance(grid, OccupancyGrid)
    assert grid.get_value(grid.world_to_grid(*static_point)) == OG_OCCUPIED
    assert fake.mapped_obstacle_points == points_before, "the original mapped_obstacle_points list must never be mutated"


# ---------------------------------------------------------------------------
# 2. Goal path.
# ---------------------------------------------------------------------------


def test_goal_path_uses_new_flow_returns_occupancy_grid_static_and_hazard_block(monkeypatch):
    fake = _make_fake_engine()
    static_point = (5.0, 5.0)
    fake.mapped_obstacle_points = [static_point]
    builder_calls = _spy_planning_costmap_builder(monkeypatch)

    hazard_row, hazard_col = 2, 2
    fake.hazard_service.belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)

    result = fake.build_planner_kwargs_for_goal((0.0, 0.0), (9.0, 9.0), robot=fake.robot)

    assert len(builder_calls) == 1
    grid = result["planning_grid"]
    assert isinstance(grid, OccupancyGrid)
    assert grid.get_value(grid.world_to_grid(*static_point)) == OG_OCCUPIED
    assert grid.get_value(GridCell(hazard_row, hazard_col)) == OG_OCCUPIED


# ---------------------------------------------------------------------------
# 3. Multi-robot.
# ---------------------------------------------------------------------------


def test_multi_robot_static_blocks_other_robot_blocks_dynamic_stays_out_of_snapshot(monkeypatch):
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    static_point = (8.0, 0.0)
    fake.mapped_obstacle_points = [static_point]
    other_robot_xy = (5.0, 5.0)
    builder_calls = _spy_planning_costmap_builder(monkeypatch)

    kwargs, _reason = fake.build_planner_kwargs_for_multi_robot(0)

    assert len(builder_calls) == 1
    call_kwargs = builder_calls[0]
    assert call_kwargs["dynamic_obstacle_points"], "the other robot must enter as explicit dynamic_obstacle_points"

    # dynamic points must never appear in observed_obstacle_snapshot().
    snapshot = fake.observed_obstacle_snapshot()
    assert not any(
        math.hypot(px - other_robot_xy[0], py - other_robot_xy[1]) < 1e-6 for px, py in snapshot.points
    ), "dynamic other-robot points must never leak into observed_obstacle_snapshot()"

    grid = kwargs["planning_grid"]
    assert grid.get_value(grid.world_to_grid(*static_point)) == OG_OCCUPIED
    assert grid.get_value(grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED


# ---------------------------------------------------------------------------
# 4. Reachability parity.
# ---------------------------------------------------------------------------


def test_reachability_and_multi_robot_planner_both_block_other_robot_cell():
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    fake.mapped_obstacle_points = []
    other_robot_xy = (5.0, 5.0)
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    is_reachable = fake.make_exploration_reachability_check(fake.robots[0])
    is_reachable((9.0, 9.0))  # triggers the lazy _build_context()
    assert len(calls) == 1
    reachability_grid = calls[0]["result"]

    multi_kwargs, _reason = fake.build_planner_kwargs_for_multi_robot(0)
    multi_grid = multi_kwargs["planning_grid"]

    assert reachability_grid.get_value(reachability_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED
    assert multi_grid.get_value(multi_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED
    # No object-identity requirement -- each caller still builds its OWN
    # OccupancyGrid independently, even though both now agree on content.
    assert reachability_grid is not multi_grid


# ---------------------------------------------------------------------------
# 5. Hazard.
# ---------------------------------------------------------------------------


def test_hazard_blocks_without_modifying_static_or_dynamic_points():
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    static_point = (8.0, 0.0)
    fake.mapped_obstacle_points = [static_point]
    static_points_before = list(fake.mapped_obstacle_points)
    dynamic_points_before = list(fake.dynamic_robot_obstacle_points_for_robot(0))

    hazard_row, hazard_col = 2, 2
    fake.hazard_service.belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)

    kwargs, _reason = fake.build_planner_kwargs_for_multi_robot(0)
    grid = kwargs["planning_grid"]

    assert grid.get_value(GridCell(hazard_row, hazard_col)) == OG_OCCUPIED
    assert fake.mapped_obstacle_points == static_points_before
    assert list(fake.dynamic_robot_obstacle_points_for_robot(0)) == dynamic_points_before


# ---------------------------------------------------------------------------
# 6. Ground truth.
# ---------------------------------------------------------------------------


def test_ground_truth_without_sensing_does_not_block_grid():
    ground_truth_rect = (2.0, 2.0, 2.0, 2.0)  # x, y, width, height
    fake = _make_fake_engine(obstacles=[ground_truth_rect])
    # No sensing ever ran -- mapped_obstacle_points stays empty.

    result = fake.build_planner_kwargs((0.0, 0.0))
    grid = result["planning_grid"]

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
            assert grid.get_value(cell) != OG_OCCUPIED, (
                f"cell {cell!r} inside ground-truth rectangle {ground_truth_rect} must not be occupied"
            )
    assert sampled_any, "sanity: the rectangle actually maps to real grid cells"


# ---------------------------------------------------------------------------
# 7. Per-robot sanitization.
# ---------------------------------------------------------------------------


def test_per_robot_sanitization_differs_between_two_robots(monkeypatch):
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    # Each point sits exactly on one robot's own position.
    fake.mapped_obstacle_points = [(0.0, 0.0), (5.0, 5.0)]
    builder_calls = _spy_planning_costmap_builder(monkeypatch)

    fake.build_planner_kwargs_for_multi_robot(0)
    fake.build_planner_kwargs_for_multi_robot(1)

    assert len(builder_calls) == 2
    points_for_robot_0 = builder_calls[0]["observed_obstacles"].points
    points_for_robot_1 = builder_calls[1]["observed_obstacles"].points
    assert points_for_robot_0 != points_for_robot_1, (
        "the SAME static geometry must sanitize differently depending on which robot's "
        "start_xy the builder was given"
    )
    assert (0.0, 0.0) not in points_for_robot_0, "robot 0's own-start sample must be sanitized away for robot 0"
    assert (5.0, 5.0) not in points_for_robot_1, "robot 1's own-start sample must be sanitized away for robot 1"


# ---------------------------------------------------------------------------
# 8. Legacy compatibility.
# ---------------------------------------------------------------------------


def test_legacy_direct_call_with_explicit_obstacle_points_still_works(monkeypatch):
    fake = _make_fake_engine()
    builder_calls = _spy_planning_costmap_builder(monkeypatch)
    explicit_point = (3.0, 3.0)

    grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=[explicit_point], robot_radius=ROBOT_RADIUS)

    assert builder_calls == [], "an explicit obstacle_points= call must take the LEGACY path, never the builder"
    assert isinstance(grid, OccupancyGrid)
    assert grid.get_value(grid.world_to_grid(*explicit_point)) == OG_OCCUPIED

    # An explicit EMPTY list is still "passed explicitly" -- also legacy.
    empty_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=[], robot_radius=ROBOT_RADIUS)
    assert builder_calls == []
    assert isinstance(empty_grid, OccupancyGrid)


# ---------------------------------------------------------------------------
# 9. Builder invocation.
# ---------------------------------------------------------------------------


def test_all_four_production_paths_call_planning_costmap_builder(monkeypatch):
    """Demonstrates, via a monkeypatch spy around the REAL
    PlanningCostmapBuilder.build (never copied or reimplemented), that all
    four production runtime paths route through it."""
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    fake.mapped_obstacle_points = [(2.0, 2.0)]
    builder_calls = _spy_planning_costmap_builder(monkeypatch)

    fake.build_planner_kwargs((0.0, 0.0))
    assert len(builder_calls) == 1, "build_planner_kwargs()"

    fake.build_planner_kwargs_for_goal((0.0, 0.0), (9.0, 9.0), robot=fake.robot)
    assert len(builder_calls) == 2, "build_planner_kwargs_for_goal()"

    fake.build_planner_kwargs_for_multi_robot(0)
    assert len(builder_calls) == 3, "build_planner_kwargs_for_multi_robot()"

    is_reachable = fake.make_exploration_reachability_check(fake.robots[0])
    is_reachable((9.0, 9.0))
    assert len(builder_calls) == 4, "make_exploration_reachability_check()'s _build_context()"


# ---------------------------------------------------------------------------
# 10. No double projection.
# ---------------------------------------------------------------------------


def test_no_double_projection_all_four_production_callers_use_only_new_path():
    """A static point must never be inflated twice by appearing
    simultaneously in the raw snapshot AND a legacy obstacle_points list --
    guaranteed here by confirming none of the four production callers ever
    passes obstacle_points at all (obstacle_points is None is the NEW-path
    signal; see build_planning_grid_for_robot()'s own docstring)."""
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    static_point = (2.0, 2.0)
    fake.mapped_obstacle_points = [static_point]
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    fake.build_planner_kwargs((0.0, 0.0))
    fake.build_planner_kwargs_for_goal((0.0, 0.0), (9.0, 9.0), robot=fake.robot)
    fake.build_planner_kwargs_for_multi_robot(0)
    is_reachable = fake.make_exploration_reachability_check(fake.robots[0])
    is_reachable((9.0, 9.0))

    assert len(calls) == 4
    for call in calls:
        assert call["obstacle_points"] is None, (
            "no production caller may pass obstacle_points explicitly -- doing so would risk "
            "the same static point being projected through two different lists (the raw legacy "
            "list AND the snapshot the new path already projects internally) at once"
        )

    grid = calls[0]["result"]
    assert grid.get_value(grid.world_to_grid(*static_point)) == OG_OCCUPIED
