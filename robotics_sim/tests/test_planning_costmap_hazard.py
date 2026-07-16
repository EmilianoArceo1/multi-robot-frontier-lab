"""
Tests for apply_hazard_to_planning_grid() (robotics_sim.planning.planning_
costmap) -- the one place a continuous HazardField is projected into a
discrete OccupancyGrid for A*/Dijkstra to consume. It must never touch
BeliefMap; it mutates the fresh planning-grid projection it is given (see
its own docstring), which is exactly what makes the "remove the source ->
costmap unblocks" case (8) meaningful: a stale costmap is never reused,
each planning request projects the hazard field fresh.
"""
from __future__ import annotations

from robotics_sim.environment.hazard_field import HazardField
from robotics_sim.environment.occupancy_grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid
from robotics_sim.planning.planning_costmap import apply_hazard_to_planning_grid

_BOUNDS = (-10.0, 10.0, -10.0, 10.0)
_RESOLUTION = 0.5


def _make_hazard_field() -> HazardField:
    return HazardField(bounds=_BOUNDS, resolution=_RESOLUTION)


def _make_planning_grid() -> OccupancyGrid:
    return OccupancyGrid.from_bounds(*_BOUNDS, _RESOLUTION, initial_value=UNKNOWN)


# ---------------------------------------------------------------------------
# 7. Cells above the block threshold become OCCUPIED in the planning grid;
# cells below it do not.
# ---------------------------------------------------------------------------


def test_hazard_above_threshold_blocks_the_planning_grid():
    hazard = _make_hazard_field()
    # Exactly on a cell center (resolution=0.5 -> centers at ..., -0.25,
    # 0.25, ...) so its contribution is the full intensity.
    hazard.add_fire((0.25, 0.25), intensity=1.0, radius=2.0)
    planning_grid = _make_planning_grid()

    apply_hazard_to_planning_grid(planning_grid, hazard, block_threshold=0.9)

    cell = planning_grid.world_to_grid(0.25, 0.25)
    assert planning_grid.get_value(cell) == OCCUPIED


def test_hazard_below_threshold_does_not_block_the_planning_grid():
    hazard = _make_hazard_field()
    hazard.add_fire((0.25, 0.25), intensity=0.3, radius=2.0)
    planning_grid = _make_planning_grid()

    apply_hazard_to_planning_grid(planning_grid, hazard, block_threshold=0.9)

    cell = planning_grid.world_to_grid(0.25, 0.25)
    assert planning_grid.get_value(cell) == UNKNOWN  # untouched -- below threshold


def test_cells_outside_the_fire_radius_are_never_blocked():
    hazard = _make_hazard_field()
    hazard.add_fire((0.0, 0.0), intensity=1.0, radius=1.0)
    planning_grid = _make_planning_grid()

    apply_hazard_to_planning_grid(planning_grid, hazard, block_threshold=0.1)

    far_cell = planning_grid.world_to_grid(9.0, 9.0)
    assert planning_grid.get_value(far_cell) == UNKNOWN


def test_apply_hazard_never_touches_existing_occupied_or_free_cells():
    hazard = _make_hazard_field()
    hazard.add_fire((0.25, 0.25), intensity=1.0, radius=2.0)
    planning_grid = _make_planning_grid()
    free_cell = planning_grid.world_to_grid(5.0, 5.0)
    occupied_cell = planning_grid.world_to_grid(-5.0, -5.0)
    planning_grid.mark_free(free_cell)
    planning_grid.mark_occupied(occupied_cell)

    apply_hazard_to_planning_grid(planning_grid, hazard, block_threshold=0.9)

    assert planning_grid.get_value(free_cell) == FREE
    assert planning_grid.get_value(occupied_cell) == OCCUPIED


# ---------------------------------------------------------------------------
# 8. Removing the source removes the block -- because each planning request
# projects a fresh grid; nothing caches the old block.
# ---------------------------------------------------------------------------


def test_removing_the_source_unblocks_a_freshly_projected_grid():
    hazard = _make_hazard_field()
    source = hazard.add_fire((0.25, 0.25), intensity=1.0, radius=2.0)
    cell = OccupancyGrid.from_bounds(*_BOUNDS, _RESOLUTION, initial_value=UNKNOWN).world_to_grid(0.25, 0.25)

    blocked_grid = _make_planning_grid()
    apply_hazard_to_planning_grid(blocked_grid, hazard, block_threshold=0.9)
    assert blocked_grid.get_value(cell) == OCCUPIED

    hazard.remove_fire(source.fire_id)
    fresh_grid = _make_planning_grid()  # a new planning request's projection
    apply_hazard_to_planning_grid(fresh_grid, hazard, block_threshold=0.9)

    assert fresh_grid.get_value(cell) == UNKNOWN


def test_apply_hazard_with_no_field_is_a_noop():
    planning_grid = _make_planning_grid()

    result = apply_hazard_to_planning_grid(planning_grid, None, block_threshold=0.9)

    assert result is planning_grid
    assert (result.data == UNKNOWN).all()


def test_apply_hazard_with_no_sources_leaves_grid_untouched():
    hazard = _make_hazard_field()  # no fires added
    planning_grid = _make_planning_grid()

    apply_hazard_to_planning_grid(planning_grid, hazard, block_threshold=0.9)

    assert (planning_grid.data == UNKNOWN).all()


def test_apply_hazard_inflate_radius_pads_the_blocked_region():
    hazard = _make_hazard_field()
    hazard.add_fire((0.25, 0.25), intensity=1.0, radius=2.0)
    planning_grid = _make_planning_grid()

    apply_hazard_to_planning_grid(planning_grid, hazard, block_threshold=0.9, inflate_radius=1.0)

    neighbor_cell = planning_grid.world_to_grid(0.25 + _RESOLUTION, 0.25)
    assert planning_grid.get_value(neighbor_cell) == OCCUPIED
