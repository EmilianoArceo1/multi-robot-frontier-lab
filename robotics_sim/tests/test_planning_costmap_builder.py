"""
Tests for the internal PlanningCostmapBuilder (robotics_sim/planning/
planning_costmap_builder.py) -- not connected to runtime yet.

These pin the builder's own contract: given ExplorationMapSnapshot +
ObservedObstacleSnapshot (+ optional HazardBeliefFrame) + a
PlanningCostmapPolicy, it reproduces SimulationControllerMixin.
build_planning_grid_for_robot()'s composition order and inflation policy,
WITH one intentional, pinned divergence: legacy exploration-belief
OCCUPIED cells are no longer treated as physical obstacle occupancy (see
test_legacy_exploration_occupied_cell_does_not_block_without_observed_geometry
and
test_builder_intentionally_ignores_legacy_belief_occupancy_without_observed_geometry).
Where a case's fixture also gives the same physical occupancy to
ObservedObstacleSnapshot, builder/runtime equivalence still holds exactly
(see test_builder_matches_current_runtime_planning_grid_for_same_inputs).
None of this touches engine.py, Qt, config, or any live BeliefMap/
HazardBelief.
"""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.environment.grid_geometry import GridCell, GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief, HazardBeliefFrame
from robotics_sim.environment.map_snapshots import ExplorationMapSnapshot, ObservedObstacleSnapshot
from robotics_sim.environment.occupancy_grid import FREE as OG_FREE
from robotics_sim.environment.occupancy_grid import OCCUPIED as OG_OCCUPIED
from robotics_sim.environment.occupancy_grid import UNKNOWN as OG_UNKNOWN
from robotics_sim.environment.occupancy_grid import OccupancyGrid
from robotics_sim.planning.costmap_snapshot import FREE as COSTMAP_FREE
from robotics_sim.planning.costmap_snapshot import OCCUPIED as COSTMAP_OCCUPIED
from robotics_sim.planning.costmap_snapshot import UNKNOWN as COSTMAP_UNKNOWN
from robotics_sim.planning.planning_costmap_builder import (
    PlanningCostmapBuilder,
    PlanningCostmapPolicy,
)
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.hazard_service import RuntimeHazardService

BOUNDS = (0.0, 10.0, 0.0, 10.0)
RESOLUTION = 1.0  # -> 10x10 grid, cell centers at 0.5, 1.5, ..., 9.5
ROBOT_RADIUS = 0.3


def _cell_index(point: tuple[float, float], bounds=BOUNDS, resolution=RESOLUTION) -> tuple[int, int]:
    cell = GridGeometry(bounds, resolution).world_to_grid(*point)
    return (cell.row, cell.col)


def _all_free_exploration(revision: int = 1) -> ExplorationMapSnapshot:
    grid = np.zeros((10, 10), dtype=np.int8)  # 0 == FREE
    return ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=revision)


def _empty_observed_obstacles(revision: int = 1) -> ObservedObstacleSnapshot:
    return ObservedObstacleSnapshot(points=(), bounds=BOUNDS, resolution=RESOLUTION, revision=revision)


def _exploration_snapshot_for(
    bounds: tuple[float, float, float, float], resolution: float, revision: int = 1
) -> ExplorationMapSnapshot:
    """An all-UNKNOWN ExplorationMapSnapshot sized via GridGeometry (never a
    manually copied width/height formula) for arbitrary bounds/resolution."""
    geometry = GridGeometry(bounds, resolution)
    grid = np.full((geometry.height, geometry.width), -1, dtype=np.int8)  # -1 == UNKNOWN
    return ExplorationMapSnapshot(grid=grid, bounds=bounds, resolution=resolution, revision=revision)


# ---------------------------------------------------------------------------
# PlanningCostmapPolicy: strict numeric/boolean validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 1.5, np.int64(2), np.float32(0.75)])
def test_policy_accepts_numeric_types_and_normalizes_obstacle_padding_to_float(value):
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=value)
    assert type(policy.obstacle_padding) is float
    assert policy.obstacle_padding == pytest.approx(float(value))


@pytest.mark.parametrize("value", [True, False, np.bool_(True)])
def test_policy_rejects_bool_obstacle_padding(value):
    with pytest.raises(ValueError):
        PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=value)


def test_policy_rejects_numeric_string_obstacle_padding():
    with pytest.raises(ValueError):
        PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding="0.5")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_policy_rejects_non_finite_obstacle_padding(value):
    with pytest.raises(ValueError):
        PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=value)


def test_policy_rejects_negative_obstacle_padding():
    with pytest.raises(ValueError):
        PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=-0.1)


def test_policy_accepts_zero_obstacle_padding():
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0)
    assert policy.obstacle_padding == 0.0


@pytest.mark.parametrize("value", [0.0, -0.5])
def test_policy_rejects_non_positive_hazard_block_threshold(value):
    with pytest.raises(ValueError):
        PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=value)


def test_policy_rejects_hazard_block_threshold_above_one():
    with pytest.raises(ValueError):
        PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=1.5)


def test_policy_accepts_hazard_block_threshold_of_exactly_one():
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=1.0)
    assert policy.hazard_block_threshold == 1.0
    assert type(policy.hazard_block_threshold) is float


def test_policy_accepts_numpy_bool_for_unknown_is_traversable_and_normalizes_to_bool():
    policy = PlanningCostmapPolicy(unknown_is_traversable=np.bool_(True), obstacle_padding=0.3)
    assert policy.unknown_is_traversable is True
    assert type(policy.unknown_is_traversable) is bool


@pytest.mark.parametrize("value", [0, 1, "True", None])
def test_policy_rejects_non_bool_unknown_is_traversable(value):
    with pytest.raises(ValueError):
        PlanningCostmapPolicy(unknown_is_traversable=value, obstacle_padding=0.3)


# ---------------------------------------------------------------------------
# 1. Basic construction.
# ---------------------------------------------------------------------------


def test_basic_construction_produces_matching_geometry_and_source_revisions():
    grid = np.array(
        [[-1 if (r + c) % 3 == 0 else 0 for c in range(10)] for r in range(10)],
        dtype=np.int8,
    )
    grid[2, 2] = 1  # one OCCUPIED cell, mixed in with FREE/UNKNOWN
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=4)
    observed = _empty_observed_obstacles(revision=9)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)
    grid_before = exploration.grid.copy()

    result = PlanningCostmapBuilder().build(exploration=exploration, observed_obstacles=observed, policy=policy)

    assert result.bounds == exploration.bounds
    assert result.resolution == exploration.resolution
    assert result.shape == exploration.shape
    assert result.grid.flags.writeable is False
    assert (exploration.grid == grid_before).all(), "exploration.grid must never be mutated by build()"
    assert result.source_revisions == (("exploration", 4), ("observed_obstacles", 9))


# ---------------------------------------------------------------------------
# 1b. Legacy exploration-belief OCCUPIED cells vs. observed obstacle
# geometry -- the intentional separation this task introduces. See the
# module docstring in planning_costmap_builder.py, "Legacy belief occupancy
# vs. observed obstacle geometry".
# ---------------------------------------------------------------------------


def test_legacy_exploration_occupied_cell_does_not_block_without_observed_geometry():
    grid = np.zeros((10, 10), dtype=np.int8)
    occupied_row, occupied_col = 4, 4
    grid[occupied_row, occupied_col] = 1  # legacy belief OCCUPIED, no observed geometry there
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)

    result = PlanningCostmapBuilder().build(exploration=exploration, observed_obstacles=observed, policy=policy)

    assert result.grid[occupied_row, occupied_col] != COSTMAP_OCCUPIED, (
        "a legacy belief-OCCUPIED cell with no corroborating ObservedObstacleSnapshot "
        "geometry must not be treated as physical occupancy"
    )
    assert result.grid[occupied_row, occupied_col] == COSTMAP_FREE


def test_legacy_exploration_occupied_cell_blocks_when_also_observed():
    grid = np.zeros((10, 10), dtype=np.int8)
    occupied_row, occupied_col = 4, 4
    grid[occupied_row, occupied_col] = 1
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)

    point = GridGeometry(BOUNDS, RESOLUTION).grid_to_world(GridCell(occupied_row, occupied_col))
    observed = ObservedObstacleSnapshot(points=(point,), bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0)

    result = PlanningCostmapBuilder().build(exploration=exploration, observed_obstacles=observed, policy=policy)

    assert result.grid[occupied_row, occupied_col] == COSTMAP_OCCUPIED, (
        "the same cell becomes occupied once ObservedObstacleSnapshot independently "
        "confirms it -- occupancy comes from observed geometry, not the legacy belief state"
    )


@pytest.mark.parametrize("unknown_is_traversable", [True, False])
def test_unknown_cells_still_resolve_purely_by_policy(unknown_is_traversable):
    """UNKNOWN handling is untouched by the OCCUPIED-cell projection change --
    only what happens to OCCUPIED=1 cells changed, not -1 cells."""
    grid = np.full((10, 10), -1, dtype=np.int8)  # all UNKNOWN
    unknown_row, unknown_col = 5, 5
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=unknown_is_traversable, obstacle_padding=0.3)

    result = PlanningCostmapBuilder().build(exploration=exploration, observed_obstacles=observed, policy=policy)

    expected = COSTMAP_FREE if unknown_is_traversable else COSTMAP_UNKNOWN
    assert result.grid[unknown_row, unknown_col] == expected
    assert result.grid[unknown_row, unknown_col] != COSTMAP_OCCUPIED


def test_no_physical_inflation_around_legacy_occupied_cell_without_observed_geometry():
    grid = np.zeros((10, 10), dtype=np.int8)
    occupied_row, occupied_col = 5, 5
    grid[occupied_row, occupied_col] = 1
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    observed = _empty_observed_obstacles()
    # A large padding: under the old (pre-divergence) behavior this would
    # have rasterized the belief-OCCUPIED cell center through
    # add_obstacle_points() and inflated into every one of these neighbors.
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.9)

    result = PlanningCostmapBuilder().build(exploration=exploration, observed_obstacles=observed, policy=policy)

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            row, col = occupied_row + dr, occupied_col + dc
            assert result.grid[row, col] != COSTMAP_OCCUPIED, (
                f"cell ({row},{col}) must not be inflated from a legacy belief-OCCUPIED "
                "cell that has no corroborating observed obstacle geometry"
            )


def test_observed_obstacle_inflation_matches_raw_occupancy_grid_add_obstacle_points_footprint():
    """Observed-obstacle inflation itself is unchanged: the builder's output
    footprint for an ObservedObstacleSnapshot point must match calling
    OccupancyGrid.add_obstacle_points() directly with the same point and
    padding on an all-FREE grid."""
    exploration = _all_free_exploration()
    point = (5.5, 5.5)
    padding = 0.9
    observed = ObservedObstacleSnapshot(points=(point,), bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=padding)

    result = PlanningCostmapBuilder().build(exploration=exploration, observed_obstacles=observed, policy=policy)

    reference_grid = OccupancyGrid.from_bounds(
        x_min=BOUNDS[0], x_max=BOUNDS[1], y_min=BOUNDS[2], y_max=BOUNDS[3],
        resolution=RESOLUTION, initial_value=OG_FREE, unknown_is_traversable=True,
    )
    reference_grid.add_obstacle_points([point], padding=padding)

    assert np.array_equal(result.grid == COSTMAP_OCCUPIED, reference_grid.data == OG_OCCUPIED), (
        "observed-obstacle inflation footprint must match a raw "
        "OccupancyGrid.add_obstacle_points() call with the same point and padding"
    )


@pytest.mark.parametrize("exploration_cell_state", [0, 1], ids=["exploration_free", "exploration_legacy_occupied"])
def test_hazard_blocks_cell_regardless_of_underlying_exploration_state(exploration_cell_state):
    grid = np.zeros((10, 10), dtype=np.int8)
    hazard_row, hazard_col = 5, 5
    grid[hazard_row, hazard_col] = exploration_cell_state
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    observed = _empty_observed_obstacles()

    hazard_belief = HazardBelief(GridGeometry(BOUNDS, RESOLUTION), robot_count=1)
    hazard_belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)
    frame = hazard_belief.snapshot()

    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0, hazard_block_threshold=0.55)

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        hazard_belief=frame, hazard_geometry=GridGeometry(BOUNDS, RESOLUTION),
    )

    assert result.grid[hazard_row, hazard_col] == COSTMAP_OCCUPIED, (
        "hazard blocking must apply independent of whether the underlying exploration "
        f"cell was FREE or legacy-OCCUPIED (state={exploration_cell_state})"
    )


def test_build_does_not_mutate_exploration_grid_occupied_values():
    grid = np.zeros((10, 10), dtype=np.int8)
    occupied_row, occupied_col = 3, 3
    grid[occupied_row, occupied_col] = 1
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)
    grid_before = exploration.grid.copy()

    PlanningCostmapBuilder().build(exploration=exploration, observed_obstacles=observed, policy=policy)

    assert (exploration.grid == grid_before).all()
    assert exploration.grid[occupied_row, occupied_col] == 1, (
        "exploration.grid's own OCCUPIED=1 encoding must remain untouched -- only the "
        "builder's OUTPUT interpretation of that cell changes, never the input snapshot"
    )


# ---------------------------------------------------------------------------
# 2. Observed obstacles.
# ---------------------------------------------------------------------------


def test_observed_obstacle_point_becomes_occupied_with_padding():
    exploration = _all_free_exploration()
    observed = ObservedObstacleSnapshot(points=((5.5, 5.5),), bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    builder = PlanningCostmapBuilder()

    no_padding = builder.build(
        exploration=exploration,
        observed_obstacles=observed,
        policy=PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0),
    )
    with_padding = builder.build(
        exploration=exploration,
        observed_obstacles=observed,
        policy=PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.9),
    )

    base_row, base_col = _cell_index((5.5, 5.5))
    neighbor_row, neighbor_col = _cell_index((6.5, 5.5))

    assert no_padding.grid[base_row, base_col] == COSTMAP_OCCUPIED
    assert no_padding.grid[neighbor_row, neighbor_col] != COSTMAP_OCCUPIED

    assert with_padding.grid[base_row, base_col] == COSTMAP_OCCUPIED
    assert with_padding.grid[neighbor_row, neighbor_col] == COSTMAP_OCCUPIED, (
        "expected policy.obstacle_padding to inflate the observed obstacle point "
        "into the neighboring cell, via OccupancyGrid.add_obstacle_points()"
    )


# ---------------------------------------------------------------------------
# 3. Hazard observed.
# ---------------------------------------------------------------------------


def test_hazard_observed_above_threshold_becomes_occupied_unobserved_cell_does_not():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()

    hazard_belief = HazardBelief(GridGeometry(BOUNDS, RESOLUTION), robot_count=1)
    hazard_row, hazard_col = _cell_index((5.5, 5.5))
    hazard_belief.observe_cells([hazard_row], [hazard_col], [0.8], robot_index=0)
    frame = hazard_belief.snapshot()

    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=0.55)

    result = PlanningCostmapBuilder().build(
        exploration=exploration,
        observed_obstacles=observed,
        policy=policy,
        hazard_belief=frame,
        hazard_geometry=GridGeometry(BOUNDS, RESOLUTION),
    )

    other_row, other_col = _cell_index((2.5, 2.5))  # never observed
    assert result.grid[hazard_row, hazard_col] == COSTMAP_OCCUPIED
    assert result.grid[other_row, other_col] != COSTMAP_OCCUPIED
    assert result.source_revisions == (
        ("exploration", exploration.revision),
        ("hazard", frame.revision),
        ("observed_obstacles", observed.revision),
    )


# ---------------------------------------------------------------------------
# 4. No hazard.
# ---------------------------------------------------------------------------


def test_no_hazard_belief_omits_hazard_source_and_does_not_change_cells():
    exploration = _all_free_exploration()
    observed = ObservedObstacleSnapshot(points=((5.5, 5.5),), bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)
    builder = PlanningCostmapBuilder()

    result_default = builder.build(exploration=exploration, observed_obstacles=observed, policy=policy)
    result_explicit_none = builder.build(
        exploration=exploration, observed_obstacles=observed, policy=policy, hazard_belief=None,
    )

    assert "hazard" not in dict(result_default.source_revisions)
    assert "hazard" not in dict(result_explicit_none.source_revisions)
    assert (result_default.grid == result_explicit_none.grid).all()


# ---------------------------------------------------------------------------
# Hazard geometry contract: hazard_belief and hazard_geometry must be given
# together, and must describe the same grid as exploration.
# ---------------------------------------------------------------------------


def _observed_hazard_frame(bounds=BOUNDS, resolution=RESOLUTION, cell: tuple[int, int] = (1, 1)) -> HazardBeliefFrame:
    belief = HazardBelief(GridGeometry(bounds, resolution), robot_count=1)
    row, col = cell
    belief.observe_cells([row], [col], [0.8], robot_index=0)
    return belief.snapshot()


def test_build_rejects_hazard_belief_without_hazard_geometry():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=0.55)
    frame = _observed_hazard_frame()

    with pytest.raises(ValueError):
        PlanningCostmapBuilder().build(
            exploration=exploration, observed_obstacles=observed, policy=policy,
            hazard_belief=frame, hazard_geometry=None,
        )


def test_build_rejects_hazard_geometry_without_hazard_belief():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)

    with pytest.raises(ValueError):
        PlanningCostmapBuilder().build(
            exploration=exploration, observed_obstacles=observed, policy=policy,
            hazard_belief=None, hazard_geometry=GridGeometry(BOUNDS, RESOLUTION),
        )


def test_build_rejects_hazard_geometry_with_mismatched_bounds():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=0.55)

    # Same resulting shape (10x10) as BOUNDS at RESOLUTION, but genuinely
    # different bounds -- isolates the bounds check from the shape check.
    other_bounds = (5.0, 15.0, 5.0, 15.0)
    frame = _observed_hazard_frame(bounds=other_bounds)

    with pytest.raises(ValueError):
        PlanningCostmapBuilder().build(
            exploration=exploration, observed_obstacles=observed, policy=policy,
            hazard_belief=frame, hazard_geometry=GridGeometry(other_bounds, RESOLUTION),
        )


def test_build_rejects_hazard_geometry_with_mismatched_resolution():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=0.55)

    other_resolution = 2.0
    frame = _observed_hazard_frame(resolution=other_resolution)

    with pytest.raises(ValueError):
        PlanningCostmapBuilder().build(
            exploration=exploration, observed_obstacles=observed, policy=policy,
            hazard_belief=frame, hazard_geometry=GridGeometry(BOUNDS, other_resolution),
        )


def test_build_rejects_hazard_belief_frame_shape_mismatch_with_matching_geometry():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=0.55)

    matching_geometry = GridGeometry(BOUNDS, RESOLUTION)  # 10x10, matches exploration
    wrong_shape_frame = HazardBeliefFrame(
        values=np.zeros((5, 5), dtype=np.float32),
        observed=np.zeros((5, 5), dtype=bool),
        observed_by_robot=np.zeros((1, 5, 5), dtype=bool),
        revision=1,
    )

    with pytest.raises(ValueError):
        PlanningCostmapBuilder().build(
            exploration=exploration, observed_obstacles=observed, policy=policy,
            hazard_belief=wrong_shape_frame, hazard_geometry=matching_geometry,
        )


def test_build_accepts_matching_hazard_geometry():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3, hazard_block_threshold=0.55)
    frame = _observed_hazard_frame()

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        hazard_belief=frame, hazard_geometry=GridGeometry(BOUNDS, RESOLUTION),
    )

    assert ("hazard", frame.revision) in result.source_revisions


# ---------------------------------------------------------------------------
# 5. Immutability.
# ---------------------------------------------------------------------------


def test_output_is_read_only_and_inputs_are_not_mutated():
    grid = np.zeros((10, 10), dtype=np.int8)
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    observed = ObservedObstacleSnapshot(points=((5.5, 5.5),), bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)
    exploration_grid_before = exploration.grid.copy()
    observed_points_before = observed.points

    result = PlanningCostmapBuilder().build(exploration=exploration, observed_obstacles=observed, policy=policy)

    assert (exploration.grid == exploration_grid_before).all()
    assert observed.points == observed_points_before

    assert result.grid is not exploration.grid
    assert not np.shares_memory(result.grid, exploration.grid)
    assert result.grid.flags.writeable is False
    with pytest.raises(ValueError):
        result.grid[0, 0] = 1


# ---------------------------------------------------------------------------
# 6. Repeatability.
# ---------------------------------------------------------------------------


def test_repeated_build_with_same_inputs_produces_equal_but_distinct_grids():
    exploration = _all_free_exploration()
    observed = ObservedObstacleSnapshot(points=((5.5, 5.5),), bounds=BOUNDS, resolution=RESOLUTION, revision=3)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)
    builder = PlanningCostmapBuilder()

    result_a = builder.build(exploration=exploration, observed_obstacles=observed, policy=policy)
    result_b = builder.build(exploration=exploration, observed_obstacles=observed, policy=policy)

    assert (result_a.grid == result_b.grid).all()
    assert result_a.source_revisions == result_b.source_revisions
    assert result_a.grid is not result_b.grid


# ---------------------------------------------------------------------------
# 7. Different revisions.
# ---------------------------------------------------------------------------


def test_different_revisions_same_content_produce_equal_grid_but_different_source_revisions():
    exploration_a = ExplorationMapSnapshot(
        grid=np.zeros((10, 10), dtype=np.int8), bounds=BOUNDS, resolution=RESOLUTION, revision=1,
    )
    exploration_b = ExplorationMapSnapshot(
        grid=np.zeros((10, 10), dtype=np.int8), bounds=BOUNDS, resolution=RESOLUTION, revision=2,
    )
    observed = _empty_observed_obstacles(revision=5)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)
    builder = PlanningCostmapBuilder()

    result_a = builder.build(exploration=exploration_a, observed_obstacles=observed, policy=policy)
    result_b = builder.build(exploration=exploration_b, observed_obstacles=observed, policy=policy)

    assert (result_a.grid == result_b.grid).all()
    assert result_a.source_revisions != result_b.source_revisions
    assert result_a.source_revisions == (("exploration", 1), ("observed_obstacles", 5))
    assert result_b.source_revisions == (("exploration", 2), ("observed_obstacles", 5))


# ---------------------------------------------------------------------------
# 8. Equivalence with the current runtime.
# ---------------------------------------------------------------------------


def _make_fake_engine(
    *,
    belief_map: BeliefMap,
    mapped_obstacle_points: list[tuple[float, float]] | None = None,
    hazard_service: RuntimeHazardService | None = None,
    grid_resolution: float = RESOLUTION,
    body_radius: float = ROBOT_RADIUS,
    safety_radius: float = ROBOT_RADIUS,
) -> SimpleNamespace:
    """Same lightweight duck-typed engine fake as test_planning_map_
    characterization.py: binds the REAL SimulationControllerMixin methods,
    with only ensure_belief_map stubbed to return a caller-controlled
    BeliefMap."""
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


def _pick_cell(geometry: GridGeometry, row_fraction: float, col_fraction: float) -> tuple[int, int]:
    """Derive a cell index from fractional position within the grid, so
    fixtures scale with whatever bounds/resolution a given case uses
    instead of hardcoding row/col numbers sized for one specific grid."""
    row = min(geometry.height - 1, max(0, int(geometry.height * row_fraction)))
    col = min(geometry.width - 1, max(0, int(geometry.width * col_fraction)))
    return row, col


# Case A: unit resolution, no padding, no hazard.
# Case B: the simulator's actual default world bounds, half resolution, with padding and hazard.
# Case C: off-center, non-integer bounds and resolution, with padding and hazard.
# Case D: non-centered bounds, fine resolution, no padding, no hazard.
_EQUIVALENCE_CASES = [
    pytest.param(
        {"bounds": (0.0, 10.0, 0.0, 10.0), "resolution": 1.0, "padding": 0.0, "include_hazard": False},
        id="case_a_unit_resolution_no_padding_no_hazard",
    ),
    pytest.param(
        {"bounds": (-10.0, 10.0, -8.0, 8.0), "resolution": 0.5, "padding": 0.3, "include_hazard": True},
        id="case_b_default_world_bounds_half_resolution_with_hazard",
    ),
    pytest.param(
        # This exact combination triggers OccupancyGrid.from_bounds()'s
        # internal max-bound expansion (x_min + width*resolution) landing on
        # a float64 value that is not idempotent if re-derived through
        # GridGeometry a second time (see test_builder_preserves_canonical_
        # exploration_bounds_when_occupancy_grid_expands_max_bounds below,
        # and the PlanningCostmapBuilder.build() fix that returns
        # exploration.bounds/exploration.resolution -- the canonical,
        # already-validated source geometry -- instead of grid.bounds/
        # grid.resolution). Kept here deliberately, not replaced with more
        # convenient bounds.
        {"bounds": (-2.1, 3.2, -1.7, 2.4), "resolution": 0.3, "padding": 0.2, "include_hazard": True},
        id="case_c_offcenter_fractional_bounds_with_hazard",
    ),
    pytest.param(
        {"bounds": (1.0, 9.0, 2.0, 6.0), "resolution": 0.25, "padding": 0.0, "include_hazard": False},
        id="case_d_noncentered_bounds_fine_resolution_no_padding",
    ),
]


@pytest.mark.parametrize("case", _EQUIVALENCE_CASES)
def test_builder_matches_current_runtime_planning_grid_for_same_inputs(case):
    bounds = case["bounds"]
    resolution = case["resolution"]
    padding = case["padding"]
    include_hazard = case["include_hazard"]

    geometry = GridGeometry(bounds, resolution)
    belief = BeliefMap(bounds=bounds, resolution=resolution, robot_count=1)

    # A small FREE region -- exercises FREE while leaving most cells at
    # their default UNKNOWN belief state.
    free_row, free_col = _pick_cell(geometry, 0.3, 0.3)
    for dr in range(3):
        for dc in range(3):
            row, col = free_row + dr, free_col + dc
            if row < geometry.height and col < geometry.width:
                belief.mark_free_cell((row, col))

    # A belief-native OCCUPIED cell -- also given as an observed obstacle
    # point below, so runtime/builder equivalence holds here regardless of
    # PlanningCostmapBuilder's intentional legacy-OCCUPIED divergence (the
    # cell's physical occupancy is independently confirmed by observed
    # geometry, not just legacy belief state). See
    # test_builder_intentionally_ignores_legacy_belief_occupancy_without_observed_geometry
    # for the case where it is NOT also observed.
    occupied_row, occupied_col = _pick_cell(geometry, 0.1, 0.1)
    belief.mark_occupied_cell((occupied_row, occupied_col))
    occupied_cell_point = tuple(round(v, 6) for v in geometry.grid_to_world(GridCell(occupied_row, occupied_col)))

    # An observed obstacle point placed near a cell boundary, not its
    # center -- exercises the round(..., 3) rasterization path faithfully.
    boundary_row, boundary_col = _pick_cell(geometry, 0.5, 0.5)
    cell_center_x, cell_center_y = geometry.grid_to_world(GridCell(boundary_row, boundary_col))
    near_boundary_point = (round(cell_center_x + resolution * 0.49, 6), cell_center_y)
    mapped_points = [near_boundary_point, occupied_cell_point]  # already-sanitized, as the real caller would pass

    hazard_service = None
    hazard_row = hazard_col = unobserved_row = unobserved_col = None
    if include_hazard:
        hazard_service = RuntimeHazardService(
            bounds=bounds, resolution=resolution, robot_count=1, block_threshold=0.55,
        )
        hazard_row, hazard_col = _pick_cell(geometry, 0.7, 0.2)
        hazard_service.belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)
        # A cell that is never observed -- must not block either output.
        unobserved_row, unobserved_col = _pick_cell(geometry, 0.2, 0.8)

    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=mapped_points, hazard_service=hazard_service)
    runtime_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=mapped_points, robot_radius=padding)

    exploration = ExplorationMapSnapshot(grid=belief.grid, bounds=belief.bounds, resolution=belief.resolution, revision=1)
    observed = ObservedObstacleSnapshot(
        points=tuple(mapped_points), bounds=belief.bounds, resolution=belief.resolution, revision=2,
    )
    policy_kwargs: dict = {"unknown_is_traversable": True, "obstacle_padding": padding}
    policy = PlanningCostmapPolicy(**policy_kwargs)

    builder_result = PlanningCostmapBuilder().build(
        exploration=exploration,
        observed_obstacles=observed,
        policy=policy,
        hazard_belief=None,
        hazard_geometry=None,
    )

    # Equivalence with the runtime is defined by cell data, shape, origin
    # (x_min/y_min), and resolution -- NOT full bounds equality.
    # runtime_grid.bounds carries OccupancyGrid's internally *expanded* max
    # bound (x_min + width*resolution, y_min + height*resolution), which for
    # some decimal bounds/resolution combinations (case C) is not the same
    # value as exploration.bounds/builder_result.bounds, even though both
    # describe the identical grid. See test_builder_preserves_canonical_
    # exploration_bounds_when_occupancy_grid_expands_max_bounds for a
    # fixture that pins exactly this distinction.
    assert builder_result.bounds[0] == runtime_grid.bounds[0]
    assert builder_result.bounds[2] == runtime_grid.bounds[2]
    assert builder_result.resolution == runtime_grid.resolution
    assert builder_result.shape == (runtime_grid.height, runtime_grid.width)
    assert builder_result.bounds == exploration.bounds
    assert builder_result.resolution == exploration.resolution
    assert np.array_equal(builder_result.grid, runtime_grid.data), (
        f"builder/runtime grid mismatch for bounds={bounds}, resolution={resolution}, padding={padding}"
    )

    if include_hazard:
        assert runtime_grid.get_value(GridCell(hazard_row, hazard_col)) != OG_OCCUPIED
        assert runtime_grid.get_value(GridCell(unobserved_row, unobserved_col)) != OG_OCCUPIED
        assert builder_result.grid[hazard_row, hazard_col] != COSTMAP_OCCUPIED
        assert builder_result.grid[unobserved_row, unobserved_col] != COSTMAP_OCCUPIED


def test_builder_intentionally_ignores_legacy_belief_occupancy_without_observed_geometry():
    """Pins the intentional divergence from build_planning_grid_for_robot()
    that this task introduces (see planning_costmap_builder.py's module
    docstring, "Legacy belief occupancy vs. observed obstacle geometry").

    A live BeliefMap OCCUPIED cell with no corroborating observed obstacle
    geometry still blocks the CURRENT runtime's planning grid --
    BeliefMap.to_planning_grid() unconditionally rasterizes/marks
    belief-OCCUPIED cells as physical occupancy. PlanningCostmapBuilder no
    longer reproduces that: it treats the same belief state as observed and
    traversable, since nothing has independently confirmed occupancy there.
    This is not a bug -- separating the two is the entire point of this
    task -- but it is a real, deliberate behavior difference from today's
    runtime, which is why it is pinned here explicitly rather than only
    implied by the equivalence cases above (all of which give the same
    physical occupancy to ObservedObstacleSnapshot too, so they cannot by
    themselves prove this divergence exists).
    """
    bounds = BOUNDS
    resolution = RESOLUTION
    padding = 0.3

    belief = BeliefMap(bounds=bounds, resolution=resolution, robot_count=1)
    occupied_row, occupied_col = 4, 4
    belief.mark_occupied_cell((occupied_row, occupied_col))

    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[])
    runtime_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=[], robot_radius=padding)

    exploration = ExplorationMapSnapshot(
        grid=belief.grid, bounds=belief.bounds, resolution=belief.resolution, revision=1,
    )
    observed = _empty_observed_obstacles(revision=2)  # deliberately no corroborating geometry
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=padding)

    builder_result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
    )

    assert runtime_grid.get_value(GridCell(occupied_row, occupied_col)) == OG_OCCUPIED, (
        "sanity check: today's runtime still blocks this cell via legacy belief occupancy alone"
    )
    assert builder_result.grid[occupied_row, occupied_col] != COSTMAP_OCCUPIED, (
        "PlanningCostmapBuilder must NOT reproduce that legacy behavior -- this divergence "
        "is intentional, not a regression"
    )


def test_builder_preserves_canonical_exploration_bounds_when_occupancy_grid_expands_max_bounds():
    # The exact case that first exposed the issue: OccupancyGrid.from_bounds()
    # stores back an internally *expanded* max bound (x_min + width*resolution,
    # y_min + height*resolution) that is not exactly representable for this
    # bounds/resolution combination, so runtime_grid.bounds genuinely differs
    # from the canonical bounds it was built from -- while still describing
    # the identical grid (same data, origin, resolution, shape).
    bounds = (-2.1, 3.2, -1.7, 2.4)
    resolution = 0.3
    padding = 0.2

    belief = BeliefMap(bounds=bounds, resolution=resolution, robot_count=1)
    belief.mark_free_cell((1, 1))
    exploration = ExplorationMapSnapshot(
        grid=belief.grid, bounds=belief.bounds, resolution=belief.resolution, revision=1,
    )
    observed = ObservedObstacleSnapshot(points=(), bounds=belief.bounds, resolution=belief.resolution, revision=2)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=padding)

    fake = _make_fake_engine(belief_map=belief, mapped_obstacle_points=[])
    runtime_grid = fake.build_planning_grid_for_robot(fake.robot, obstacle_points=[], robot_radius=padding)

    # Sanity: this fixture genuinely exercises the expansion -- if this ever
    # stops being true (e.g. OccupancyGrid changes), the rest of this test
    # would no longer be testing what its name claims.
    assert runtime_grid.bounds != exploration.bounds, (
        "expected this fixture to still exercise OccupancyGrid's internal max-bound "
        "expansion -- if it no longer diverges, this test needs a different fixture"
    )

    builder_result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
    )

    # The snapshot preserves the canonical (pre-expansion) exploration
    # geometry, not OccupancyGrid's internally expanded bookkeeping bound.
    assert builder_result.bounds == exploration.bounds

    # But grid data, origin, resolution, and shape remain equivalent to the
    # runtime grid built from the same inputs.
    assert np.array_equal(builder_result.grid, runtime_grid.data)
    assert builder_result.bounds[0] == runtime_grid.bounds[0]
    assert builder_result.bounds[2] == runtime_grid.bounds[2]
    assert builder_result.resolution == runtime_grid.resolution
    assert builder_result.shape == (runtime_grid.height, runtime_grid.width)


def test_builder_result_bounds_round_trip_through_occupancy_grid_from_bounds():
    bounds = (-2.1, 3.2, -1.7, 2.4)
    resolution = 0.3

    exploration = _exploration_snapshot_for(bounds, resolution)
    observed = ObservedObstacleSnapshot(points=(), bounds=bounds, resolution=resolution, revision=1)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.2)

    builder_result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
    )

    round_trip_grid = OccupancyGrid.from_bounds(
        *builder_result.bounds, builder_result.resolution, initial_value=OG_UNKNOWN,
    )

    assert round_trip_grid.data.shape == builder_result.shape


# ---------------------------------------------------------------------------
# 9. Ground truth absent.
# ---------------------------------------------------------------------------


def test_build_signature_has_no_ground_truth_parameter():
    parameters = set(inspect.signature(PlanningCostmapBuilder.build).parameters)
    forbidden = {"config", "obstacles", "ground_truth", "belief_map", "hazard_service"}
    assert not (parameters & forbidden), (
        f"PlanningCostmapBuilder.build() must not accept a ground-truth/live-simulator "
        f"parameter; found {parameters & forbidden!r} in signature {parameters!r}"
    )
    assert parameters == {
        "self", "exploration", "observed_obstacles", "policy", "dynamic_obstacle_points",
        "hazard_belief", "hazard_geometry",
    }


def test_builder_module_does_not_import_config():
    import robotics_sim.planning.planning_costmap_builder as builder_module

    source = inspect.getsource(builder_module)
    tree = ast.parse(source)

    imported_module_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_module_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_module_names.append(node.module)

    violations = [name for name in imported_module_names if "config" in name.lower()]
    assert violations == [], f"planning_costmap_builder.py must not import config: {violations}"


# ---------------------------------------------------------------------------
# 10. Dependency isolation.
# ---------------------------------------------------------------------------


def test_builder_module_does_not_import_forbidden_modules():
    """AST-based inspection of planning_costmap_builder.py's own import
    statements -- not a sys.modules check (polluted by whatever else pytest
    already imported in this process) and not a text substring search."""
    import robotics_sim.planning.planning_costmap_builder as builder_module

    source = inspect.getsource(builder_module)
    tree = ast.parse(source)

    forbidden_prefixes = ("engine", "qt", "pyside", "pyqt", "mainwindow", "canvas")
    imported_module_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_module_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_module_names.append(node.module)

    violations = [
        name
        for name in imported_module_names
        if any(keyword in name.lower() for keyword in forbidden_prefixes)
    ]
    assert violations == [], f"planning_costmap_builder.py must not import: {violations}"


# ---------------------------------------------------------------------------
# 11. Dynamic obstacle points -- a third, ephemeral physical-occupancy
# source alongside ObservedObstacleSnapshot.points and observed hazard
# cells (see planning_costmap_builder.py's module docstring, "Dynamic
# obstacle points"). Never part of ObservedObstacleSnapshot, never tracked
# in source_revisions.
# ---------------------------------------------------------------------------


def test_empty_dynamic_obstacle_points_does_not_change_result():
    exploration = _all_free_exploration()
    observed = ObservedObstacleSnapshot(points=((5.5, 5.5),), bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)
    builder = PlanningCostmapBuilder()

    result_without_param = builder.build(exploration=exploration, observed_obstacles=observed, policy=policy)
    result_with_empty_tuple = builder.build(
        exploration=exploration, observed_obstacles=observed, policy=policy, dynamic_obstacle_points=(),
    )

    assert np.array_equal(result_without_param.grid, result_with_empty_tuple.grid)
    assert result_without_param.source_revisions == result_with_empty_tuple.source_revisions


def test_single_dynamic_obstacle_point_becomes_occupied():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0)

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        dynamic_obstacle_points=((5.5, 5.5),),
    )

    row, col = _cell_index((5.5, 5.5))
    assert result.grid[row, col] == COSTMAP_OCCUPIED


def test_static_and_dynamic_points_share_identical_inflation_footprint():
    exploration = _all_free_exploration()
    padding = 0.9
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=padding)

    static_result = PlanningCostmapBuilder().build(
        exploration=exploration,
        observed_obstacles=ObservedObstacleSnapshot(
            points=((5.5, 5.5),), bounds=BOUNDS, resolution=RESOLUTION, revision=1,
        ),
        policy=policy,
    )
    dynamic_result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=_empty_observed_obstacles(), policy=policy,
        dynamic_obstacle_points=((5.5, 5.5),),
    )

    assert np.array_equal(static_result.grid == COSTMAP_OCCUPIED, dynamic_result.grid == COSTMAP_OCCUPIED), (
        "the SAME point at the SAME padding must produce an IDENTICAL occupied footprint "
        "whether it arrives via observed_obstacles.points or dynamic_obstacle_points"
    )


def test_static_and_dynamic_point_at_the_same_location_does_not_error():
    exploration = _all_free_exploration()
    observed = ObservedObstacleSnapshot(points=((5.5, 5.5),), bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        dynamic_obstacle_points=((5.5, 5.5),),
    )

    row, col = _cell_index((5.5, 5.5))
    assert result.grid[row, col] == COSTMAP_OCCUPIED


def test_dynamic_obstacle_points_order_and_duplicates_do_not_mutate_input():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0)
    dynamic_points = [(5.5, 5.5), (2.5, 2.5), (5.5, 5.5)]  # a plain list, with a duplicate
    dynamic_points_before = list(dynamic_points)

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        dynamic_obstacle_points=dynamic_points,
    )

    assert dynamic_points == dynamic_points_before, "the caller's own list must never be mutated or reordered"
    row1, col1 = _cell_index((5.5, 5.5))
    row2, col2 = _cell_index((2.5, 2.5))
    assert result.grid[row1, col1] == COSTMAP_OCCUPIED
    assert result.grid[row2, col2] == COSTMAP_OCCUPIED


def test_hazard_remains_independent_of_dynamic_obstacle_points():
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()

    hazard_belief = HazardBelief(GridGeometry(BOUNDS, RESOLUTION), robot_count=1)
    hazard_row, hazard_col = _cell_index((7.5, 7.5))
    hazard_belief.observe_cells([hazard_row], [hazard_col], [0.8], robot_index=0)
    frame = hazard_belief.snapshot()

    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0, hazard_block_threshold=0.55)

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        dynamic_obstacle_points=((2.5, 2.5),),
        hazard_belief=frame, hazard_geometry=GridGeometry(BOUNDS, RESOLUTION),
    )

    dyn_row, dyn_col = _cell_index((2.5, 2.5))
    assert result.grid[dyn_row, dyn_col] == COSTMAP_OCCUPIED
    assert result.grid[hazard_row, hazard_col] == COSTMAP_OCCUPIED
    source_names = {name for name, _ in result.source_revisions}
    assert source_names == {"exploration", "observed_obstacles", "hazard"}, (
        "dynamic_obstacle_points must never appear in source_revisions -- it is ephemeral, "
        "per-call input, not a versioned layer"
    )


def test_legacy_belief_occupied_still_does_not_block_with_dynamic_points_present():
    grid = np.zeros((10, 10), dtype=np.int8)
    legacy_row, legacy_col = 4, 4
    grid[legacy_row, legacy_col] = 1  # legacy OCCUPIED, no corroborating geometry
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        dynamic_obstacle_points=((8.5, 8.5),),  # unrelated dynamic point elsewhere
    )

    assert result.grid[legacy_row, legacy_col] != COSTMAP_OCCUPIED, (
        "legacy BeliefMap.OCCUPIED must still not block on its own, even when unrelated "
        "dynamic_obstacle_points are present in the same build() call"
    )


@pytest.mark.parametrize(
    "bad_points",
    [
        pytest.param(((float("nan"), 0.0),), id="nan_coordinate"),
        pytest.param(((float("inf"), 0.0),), id="inf_coordinate"),
        pytest.param(((1.0,),), id="malformed_not_a_pair"),
        pytest.param(((True, False),), id="bool_coordinates"),
        pytest.param("ab", id="not_an_iterable_of_pairs"),
    ],
)
def test_invalid_dynamic_obstacle_point_raises(bad_points):
    exploration = _all_free_exploration()
    observed = _empty_observed_obstacles()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0)

    with pytest.raises(ValueError):
        PlanningCostmapBuilder().build(
            exploration=exploration, observed_obstacles=observed, policy=policy,
            dynamic_obstacle_points=bad_points,
        )


def test_dynamic_obstacle_points_build_does_not_mutate_original_snapshots():
    grid = np.zeros((10, 10), dtype=np.int8)
    exploration = ExplorationMapSnapshot(grid=grid, bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    observed = ObservedObstacleSnapshot(points=((5.5, 5.5),), bounds=BOUNDS, resolution=RESOLUTION, revision=1)
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.3)
    exploration_grid_before = exploration.grid.copy()
    observed_points_before = observed.points

    PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        dynamic_obstacle_points=((2.5, 2.5),),
    )

    assert (exploration.grid == exploration_grid_before).all()
    assert observed.points == observed_points_before


def test_combined_result_contains_static_dynamic_and_hazard_occupancy():
    exploration = _all_free_exploration()
    static_point = (1.5, 1.5)
    dynamic_point = (5.5, 5.5)
    hazard_row, hazard_col = _cell_index((8.5, 8.5))

    observed = ObservedObstacleSnapshot(points=(static_point,), bounds=BOUNDS, resolution=RESOLUTION, revision=1)

    hazard_belief = HazardBelief(GridGeometry(BOUNDS, RESOLUTION), robot_count=1)
    hazard_belief.observe_cells([hazard_row], [hazard_col], [0.8], robot_index=0)
    frame = hazard_belief.snapshot()

    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0, hazard_block_threshold=0.55)

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        dynamic_obstacle_points=(dynamic_point,),
        hazard_belief=frame, hazard_geometry=GridGeometry(BOUNDS, RESOLUTION),
    )

    static_row, static_col = _cell_index(static_point)
    dynamic_row, dynamic_col = _cell_index(dynamic_point)

    assert result.grid[static_row, static_col] == COSTMAP_OCCUPIED
    assert result.grid[dynamic_row, dynamic_col] == COSTMAP_OCCUPIED
    assert result.grid[hazard_row, hazard_col] == COSTMAP_OCCUPIED
