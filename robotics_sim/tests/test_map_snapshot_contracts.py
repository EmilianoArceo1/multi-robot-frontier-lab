"""
Contract tests for the map-layer snapshot dataclasses (Phase 1 of the
map-layer separation plan -- see refactor/map-layer-contracts).

These live inside robotics_sim, not robotics_interfaces: they are
simulator-HOST-internal contracts. External coordination plugins must never
receive a live BeliefMap, an internal OccupancyGrid, a PlanningCostmapSnapshot,
raw mapped_obstacle_points, or engine.py -- plugins consume host-computed
results instead (future FrontierObservation/RouteEvaluation/
RouteReservation/HazardBeliefQuery/CoordinationResult contracts in
robotics_interfaces). The simulator keeps sole responsibility for building
the costmap, applying inflation, and running A*/Dijkstra.

    - ObservedObstacleSnapshot / ExplorationMapSnapshot ->
      robotics_sim/environment/map_snapshots.py
    - PlanningCostmapSnapshot ->
      robotics_sim/planning/costmap_snapshot.py

Nothing here is wired to a producer or consumer yet: this file only pins the
contracts' own shape/validation/immutability, not any runtime behavior.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect

import numpy as np
import pytest

from robotics_sim.environment.map_snapshots import (
    ExplorationMapSnapshot,
    ObservedObstacleSnapshot,
)
from robotics_sim.planning.costmap_snapshot import PlanningCostmapSnapshot

_BOUNDS = (-10.0, 10.0, -8.0, 8.0)
_SMALL_BOUNDS = (-1.0, 1.0, -1.0, 1.0)
# bounds/resolution pairs whose GridGeometry-implied (height, width) matches
# a specific small grid shape, for the geometric-coherence tests below.
_BOUNDS_2X3 = (-1.5, 1.5, -1.0, 1.0)  # resolution=1.0 -> width=3, height=2
_BOUNDS_3X3 = (-1.5, 1.5, -1.5, 1.5)  # resolution=1.0 -> width=3, height=3


# ---------------------------------------------------------------------------
# ObservedObstacleSnapshot
# ---------------------------------------------------------------------------


def test_observed_obstacle_snapshot_valid_construction():
    snapshot = ObservedObstacleSnapshot(
        points=((1.0, 2.0), (3.0, 4.0)),
        bounds=_BOUNDS,
        resolution=0.5,
        revision=3,
    )
    assert snapshot.points == ((1.0, 2.0), (3.0, 4.0))
    assert snapshot.bounds == _BOUNDS
    assert snapshot.resolution == 0.5
    assert snapshot.revision == 3
    assert snapshot.source == "observed_obstacles"


def test_observed_obstacle_snapshot_is_frozen():
    snapshot = ObservedObstacleSnapshot(points=(), bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.revision = 5


def test_observed_obstacle_snapshot_points_stored_as_tuple():
    snapshot = ObservedObstacleSnapshot(
        points=[[1.0, 2.0], [3.0, 4.0]],  # list of lists on purpose
        bounds=_BOUNDS,
        resolution=0.5,
        revision=1,
    )
    assert isinstance(snapshot.points, tuple)
    assert all(isinstance(point, tuple) for point in snapshot.points)


def test_observed_obstacle_snapshot_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=(10.0, -10.0, -8.0, 8.0), resolution=0.5, revision=0)


def test_observed_obstacle_snapshot_rejects_non_positive_resolution():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.0, revision=0)
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=-1.0, revision=0)


def test_observed_obstacle_snapshot_rejects_negative_revision():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision=-1)


def test_observed_obstacle_snapshot_rejects_non_finite_points():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=((float("nan"), 0.0),), bounds=_BOUNDS, resolution=0.5, revision=0)
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=((float("inf"), 0.0),), bounds=_BOUNDS, resolution=0.5, revision=0)


def test_observed_obstacle_snapshot_does_not_derive_revision_from_point_count():
    # Two snapshots with the same point count but different explicit
    # revisions must keep their own revision -- proves revision is not
    # silently recomputed as len(points) anywhere in construction.
    same_points = ((0.0, 0.0), (1.0, 1.0))
    early = ObservedObstacleSnapshot(points=same_points, bounds=_BOUNDS, resolution=0.5, revision=1)
    later = ObservedObstacleSnapshot(points=same_points, bounds=_BOUNDS, resolution=0.5, revision=7)
    assert early.revision == 1
    assert later.revision == 7


def test_observed_obstacle_snapshot_rejects_float_revision():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision=1.8)


def test_observed_obstacle_snapshot_rejects_bool_revision():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision=True)


def test_observed_obstacle_snapshot_rejects_string_revision():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision="3")


def test_observed_obstacle_snapshot_accepts_numpy_integer_revision():
    snapshot = ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision=np.int64(3))
    assert snapshot.revision == 3
    assert type(snapshot.revision) is int


def test_observed_obstacle_snapshot_rejects_empty_source():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision=0, source="")


def test_observed_obstacle_snapshot_rejects_whitespace_only_source():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision=0, source="   ")


def test_observed_obstacle_snapshot_rejects_non_string_source():
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision=0, source=None)
    with pytest.raises(ValueError):
        ObservedObstacleSnapshot(points=(), bounds=_BOUNDS, resolution=0.5, revision=0, source=123)


def test_observed_obstacle_snapshot_accepts_valid_source():
    snapshot = ObservedObstacleSnapshot(
        points=(), bounds=_BOUNDS, resolution=0.5, revision=0, source="lidar_obstacles"
    )
    assert snapshot.source == "lidar_obstacles"


# ---------------------------------------------------------------------------
# ExplorationMapSnapshot
# ---------------------------------------------------------------------------


def test_exploration_map_snapshot_valid_construction():
    grid = np.array([[-1, 0, 1], [0, 0, -1]], dtype=np.int64)
    snapshot = ExplorationMapSnapshot(grid=grid, bounds=_BOUNDS_2X3, resolution=1.0, revision=2)
    assert snapshot.shape == (2, 3)
    assert snapshot.height == 2
    assert snapshot.width == 3
    assert snapshot.revision == 2


def test_exploration_map_snapshot_copies_input_array():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)
    assert snapshot.grid is not grid
    assert not np.shares_memory(snapshot.grid, grid)


def test_exploration_map_snapshot_unaffected_by_mutating_original_array():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)
    grid[0, 0] = 1
    assert snapshot.grid[0, 0] == 0


def test_exploration_map_snapshot_grid_is_not_writeable():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)
    assert snapshot.grid.flags.writeable is False
    with pytest.raises(ValueError):
        snapshot.grid[0, 0] = 1


def test_exploration_map_snapshot_grid_dtype_is_int8():
    grid = np.zeros((2, 2), dtype=np.float64)
    snapshot = ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)
    assert snapshot.grid.dtype == np.int8


def test_exploration_map_snapshot_rejects_1d_and_3d_arrays():
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=np.zeros(4, dtype=np.int8), bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=np.zeros((2, 2, 2), dtype=np.int8), bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)


def test_exploration_map_snapshot_rejects_invalid_cell_state():
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=np.array([[2, 0], [0, -1]]), bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)


def test_exploration_map_snapshot_rejects_invalid_resolution_bounds_revision():
    valid_grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=valid_grid, bounds=_SMALL_BOUNDS, resolution=0.0, revision=0)
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=valid_grid, bounds=(1.0, -1.0, -1.0, 1.0), resolution=1.0, revision=0)
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=valid_grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=-1)


def test_exploration_map_snapshot_rejects_float_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=1.5)


def test_exploration_map_snapshot_rejects_bool_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=True)


def test_exploration_map_snapshot_rejects_string_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision="3")


def test_exploration_map_snapshot_accepts_numpy_integer_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=np.int64(3))
    assert snapshot.revision == 3
    assert type(snapshot.revision) is int


def test_exploration_map_snapshot_accepts_grid_matching_geometry():
    # SMALL_BOUNDS + resolution=1.0 -> GridGeometry(width=2, height=2).
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = ExplorationMapSnapshot(grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)
    assert snapshot.shape == (2, 2)


def test_exploration_map_snapshot_rejects_grid_shape_mismatched_with_geometry():
    # SMALL_BOUNDS + resolution=1.0 implies shape (2, 2); a (3, 3) grid must
    # be rejected instead of silently accepted with the wrong geometry.
    mismatched_grid = np.zeros((3, 3), dtype=np.int8)
    with pytest.raises(ValueError) as excinfo:
        ExplorationMapSnapshot(grid=mismatched_grid, bounds=_SMALL_BOUNDS, resolution=1.0, revision=0)

    message = str(excinfo.value)
    assert "(3, 3)" in message
    assert "(2, 2)" in message
    assert repr(_SMALL_BOUNDS) in message
    assert "1.0" in message


# ---------------------------------------------------------------------------
# PlanningCostmapSnapshot
# ---------------------------------------------------------------------------


def test_planning_costmap_snapshot_valid_construction():
    grid = np.zeros((3, 3), dtype=np.int8)
    snapshot = PlanningCostmapSnapshot(
        grid=grid,
        bounds=_BOUNDS_3X3,
        resolution=1.0,
        unknown_is_traversable=True,
        source_revisions=(("belief", 4), ("observed_obstacles", 2)),
    )
    assert snapshot.shape == (3, 3)
    assert snapshot.unknown_is_traversable is True
    assert snapshot.source_revisions == (("belief", 4), ("observed_obstacles", 2))


def test_planning_costmap_snapshot_grid_copied_and_read_only():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = PlanningCostmapSnapshot(
        grid=grid,
        bounds=_SMALL_BOUNDS,
        resolution=1.0,
        unknown_is_traversable=True,
        source_revisions=(),
    )
    assert snapshot.grid is not grid
    assert snapshot.grid.flags.writeable is False
    grid[0, 0] = 1
    assert snapshot.grid[0, 0] == 0


def test_planning_costmap_snapshot_source_revisions_sorted_deterministically():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = PlanningCostmapSnapshot(
        grid=grid,
        bounds=_SMALL_BOUNDS,
        resolution=1.0,
        unknown_is_traversable=True,
        source_revisions=(("obstacles", 1), ("belief", 3), ("hazard", 2)),
    )
    assert snapshot.source_revisions == (("belief", 3), ("hazard", 2), ("obstacles", 1))


def test_planning_costmap_snapshot_source_revisions_is_tuple_of_tuples_not_mapping():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = PlanningCostmapSnapshot(
        grid=grid,
        bounds=_SMALL_BOUNDS,
        resolution=1.0,
        unknown_is_traversable=True,
        source_revisions=(("belief", 1),),
    )
    assert isinstance(snapshot.source_revisions, tuple)
    assert not isinstance(snapshot.source_revisions, dict)
    assert all(isinstance(entry, tuple) for entry in snapshot.source_revisions)


def test_planning_costmap_snapshot_rejects_duplicate_source_names():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid,
            bounds=_SMALL_BOUNDS,
            resolution=1.0,
            unknown_is_traversable=True,
            source_revisions=(("belief", 1), ("belief", 2)),
        )


def test_planning_costmap_snapshot_rejects_empty_source_name():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid,
            bounds=_SMALL_BOUNDS,
            resolution=1.0,
            unknown_is_traversable=True,
            source_revisions=(("", 1),),
        )


def test_planning_costmap_snapshot_rejects_negative_source_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid,
            bounds=_SMALL_BOUNDS,
            resolution=1.0,
            unknown_is_traversable=True,
            source_revisions=(("belief", -1),),
        )


def test_planning_costmap_snapshot_rejects_invalid_cell_state():
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=np.array([[5, 0], [0, 1]]),
            bounds=_SMALL_BOUNDS,
            resolution=1.0,
            unknown_is_traversable=True,
            source_revisions=(),
        )


def test_planning_costmap_snapshot_unknown_is_traversable_preserved():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot_true = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True, source_revisions=(),
    )
    snapshot_false = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=False, source_revisions=(),
    )
    assert snapshot_true.unknown_is_traversable is True
    assert snapshot_false.unknown_is_traversable is False


def test_planning_costmap_snapshot_has_no_built_at_attribute():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True, source_revisions=(),
    )
    assert not hasattr(snapshot, "built_at")
    field_names = {f.name for f in dataclasses.fields(snapshot)}
    assert "built_at" not in field_names


def test_planning_costmap_snapshot_rejects_float_source_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
            source_revisions=(("belief", 1.5),),
        )


def test_planning_costmap_snapshot_rejects_bool_source_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
            source_revisions=(("belief", True),),
        )


def test_planning_costmap_snapshot_rejects_string_source_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
            source_revisions=(("belief", "3"),),
        )


def test_planning_costmap_snapshot_accepts_numpy_integer_source_revision():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
        source_revisions=(("belief", np.int64(3)),),
    )
    assert snapshot.source_revisions == (("belief", 3),)
    assert type(snapshot.source_revisions[0][1]) is int


# --- unknown_is_traversable: strict bool validation (not bool(value)) ------


def test_planning_costmap_snapshot_accepts_real_bool_unknown_is_traversable():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot_true = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True, source_revisions=(),
    )
    snapshot_false = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=False, source_revisions=(),
    )
    assert snapshot_true.unknown_is_traversable is True
    assert snapshot_false.unknown_is_traversable is False


def test_planning_costmap_snapshot_accepts_numpy_bool_unknown_is_traversable():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot_true = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0,
        unknown_is_traversable=np.bool_(True), source_revisions=(),
    )
    snapshot_false = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0,
        unknown_is_traversable=np.bool_(False), source_revisions=(),
    )
    assert snapshot_true.unknown_is_traversable is True
    assert type(snapshot_true.unknown_is_traversable) is bool
    assert snapshot_false.unknown_is_traversable is False
    assert type(snapshot_false.unknown_is_traversable) is bool


@pytest.mark.parametrize("bad_value", [0, 1, "True", "False", None, [], ()])
def test_planning_costmap_snapshot_rejects_non_bool_unknown_is_traversable(bad_value):
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0,
            unknown_is_traversable=bad_value, source_revisions=(),
        )


# --- source_revisions names: no str(name) coercion --------------------------


def test_planning_costmap_snapshot_rejects_non_string_source_revision_name():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
            source_revisions=((None, 1),),
        )
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
            source_revisions=((123, 1),),
        )


def test_planning_costmap_snapshot_rejects_empty_or_whitespace_only_source_revision_name():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
            source_revisions=(("", 1),),
        )
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
            source_revisions=(("   ", 1),),
        )


def test_planning_costmap_snapshot_strips_source_revision_names():
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
        source_revisions=(("belief", 1), (" observed_obstacles ", 2)),
    )
    assert snapshot.source_revisions == (("belief", 1), ("observed_obstacles", 2))


def test_planning_costmap_snapshot_rejects_duplicate_names_after_stripping():
    grid = np.zeros((2, 2), dtype=np.int8)
    with pytest.raises(ValueError):
        PlanningCostmapSnapshot(
            grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True,
            source_revisions=(("belief", 1), (" belief ", 2)),
        )


# --- geometric coherence: grid.shape must match GridGeometry(bounds, resolution) --


def test_planning_costmap_snapshot_accepts_grid_matching_geometry():
    # SMALL_BOUNDS + resolution=1.0 -> GridGeometry(width=2, height=2).
    grid = np.zeros((2, 2), dtype=np.int8)
    snapshot = PlanningCostmapSnapshot(
        grid=grid, bounds=_SMALL_BOUNDS, resolution=1.0, unknown_is_traversable=True, source_revisions=(),
    )
    assert snapshot.shape == (2, 2)


def test_planning_costmap_snapshot_rejects_grid_shape_mismatched_with_geometry():
    # SMALL_BOUNDS + resolution=1.0 implies shape (2, 2); a (3, 3) grid must
    # be rejected instead of silently accepted with the wrong geometry.
    mismatched_grid = np.zeros((3, 3), dtype=np.int8)
    with pytest.raises(ValueError) as excinfo:
        PlanningCostmapSnapshot(
            grid=mismatched_grid, bounds=_SMALL_BOUNDS, resolution=1.0,
            unknown_is_traversable=True, source_revisions=(),
        )

    message = str(excinfo.value)
    assert "(3, 3)" in message
    assert "(2, 2)" in message
    assert repr(_SMALL_BOUNDS) in message
    assert "1.0" in message


# ---------------------------------------------------------------------------
# Dependency isolation
# ---------------------------------------------------------------------------

# These modules now live inside robotics_sim on purpose (see module
# docstrings): external algorithms/plugins must not receive them, but
# robotics_sim-internal imports are fine. What they must still never import
# is Qt/PySide/PyQt, MainWindow, canvas, or engine.py.
_FORBIDDEN_IMPORT_KEYWORDS = ("qt", "pyside", "pyqt", "mainwindow", "canvas", "engine")


def _collect_imported_names(module) -> list[str]:
    """Static AST inspection of a module's own import statements.

    Deliberately not a sys.modules check (which would be polluted by
    whatever else pytest already imported in this same process) and not a
    naive substring search over the raw source text (which could
    false-positive on a comment or docstring).
    """
    source = inspect.getsource(module)
    tree = ast.parse(source)

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
            names.extend(alias.name for alias in node.names)
    return names


def _find_forbidden_imports(module) -> list[str]:
    violations = []
    for name in _collect_imported_names(module):
        normalized = name.lower().replace("_", "")
        if any(keyword in normalized for keyword in _FORBIDDEN_IMPORT_KEYWORDS):
            violations.append(name)
    return violations


def test_map_snapshots_module_does_not_import_forbidden_modules():
    import robotics_sim.environment.map_snapshots as map_snapshots_module

    violations = _find_forbidden_imports(map_snapshots_module)
    assert violations == [], f"environment/map_snapshots.py must not import: {violations}"


def test_costmap_snapshot_module_does_not_import_forbidden_modules():
    import robotics_sim.planning.costmap_snapshot as costmap_snapshot_module

    violations = _find_forbidden_imports(costmap_snapshot_module)
    assert violations == [], f"planning/costmap_snapshot.py must not import: {violations}"
