"""
Discrete belief map for exploration.

This module is the logical map used by the simulator. The visual blue explored
area is only a rendering cache; planners, coordinators, metrics, and frontier
extraction should use this BeliefMap instead.

Official grid convention:
    - cells are square areas
    - world_to_cell uses floor
    - cell_to_world returns the center of the cell
    - bounds are semi-open [x_min, x_max), [y_min, y_max)

Cell states:
    UNKNOWN  = -1
    FREE     = 0
    OCCUPIED = 1

The map also stores per-robot observation layers, so multi-robot coordination can
measure overlap and avoid sending robots through already saturated areas.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Iterable

import numpy as np

from robotics_sim.environment.grid_geometry import GridCell, GridGeometry

UNKNOWN = -1
FREE = 0
OCCUPIED = 1


def _validate_initial_revision(value) -> int:
    """Accept only a real, non-boolean integer (numpy integers included --
    bool is a subclass of int in Python, so isinstance(x, int) alone would
    otherwise wrongly accept it). Rejects float/str/None/negative values.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(
            f"initial_revision must be a non-boolean integer, got {value!r} ({type(value).__name__})."
        )

    result = int(value)
    if result < 0:
        raise ValueError(f"initial_revision must be >= 0, got {value!r}.")

    return result


def _point_inside_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test in world coordinates."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / denom + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


@dataclass(frozen=True)
class BeliefMapStats:
    unknown_cells: int
    free_cells: int
    occupied_cells: int
    known_cells: int
    total_cells: int
    coverage_percent: float
    overlap_cells: int
    overlap_ratio: float
    revisited_cells: int
    revisit_ratio: float
    total_free_observations: int
    average_visits_per_free_cell: float


class BeliefMap:
    """
    Occupancy/belief grid shared by single and multi-robot exploration.

    The world is discretized once using GridGeometry. This gives the planner a
    real memory of:
        - what is unknown,
        - what is known free,
        - what is known occupied,
        - how often each cell has been seen,
        - which robot observed each free cell.
    """

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        resolution: float,
        robot_count: int = 1,
        initial_revision: int = 0,
    ):
        self.geometry = GridGeometry(bounds, resolution)

        self.x_min = self.geometry.x_min
        self.x_max = self.geometry.x_max
        self.y_min = self.geometry.y_min
        self.y_max = self.geometry.y_max
        self.resolution = self.geometry.resolution
        self.width = self.geometry.width
        self.height = self.geometry.height
        self.robot_count = max(1, int(robot_count))

        self.grid = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)
        self.visit_count = np.zeros((self.height, self.width), dtype=np.uint16)
        self.explored_by_robot = np.zeros((self.robot_count, self.height, self.width), dtype=bool)
        self.last_seen = np.full((self.height, self.width), -1.0, dtype=np.float32)
        # Monotonic visual-state revision used by immutable debug replay and
        # by ExplorationMapSnapshot producers (see snapshot()). It changes
        # only when occupancy or per-robot explored masks change, not for
        # visit-count/last-seen updates that do not alter rendering. Private
        # so external code cannot silently desynchronize it from the grid it
        # is meant to describe -- see revision property and restore_grid_
        # state() (the one place besides this class's own methods that
        # legitimately needs to bump it, for navigation-debug-snapshot
        # restore's wholesale array replacement).
        #
        # initial_revision defaults to 0 for a standalone/first-ever
        # BeliefMap. A host that REPLACES one BeliefMap instance with a new
        # one (see engine.py's reset_belief_map()) must seed the new
        # instance's starting revision at old_revision + 1, so the overall
        # exploration-map revision sequence a consumer observes (e.g. via
        # PlanningCostmapSnapshot.source_revisions) never goes backwards
        # just because the underlying object was replaced.
        self._grid_revision = _validate_initial_revision(initial_revision)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.geometry.bounds

    @property
    def revision(self) -> int:
        """Read-only. Starts at 0, increases monotonically for the life of
        this BeliefMap instance whenever a mutating operation writes or
        attempts to write self.grid (mark_free_cell/force_free_cell/
        mark_occupied_cell/reset/restore_grid_state -- mark_visible_polygon/
        mark_occupied_points call into mark_free_cell/mark_occupied_cell, so
        they are covered too). Does not depend on the number of known
        cells -- two BeliefMaps with identical grid content do not
        necessarily share a revision, and this counter is never derived
        from a cell count.
        """
        return self._grid_revision

    def reset(self, robot_count: int | None = None) -> None:
        if robot_count is not None and int(robot_count) != self.robot_count:
            self.robot_count = max(1, int(robot_count))
            self.explored_by_robot = np.zeros((self.robot_count, self.height, self.width), dtype=bool)

        self.grid.fill(UNKNOWN)
        self.visit_count.fill(0)
        self.explored_by_robot.fill(False)
        self.last_seen.fill(-1.0)
        self._grid_revision += 1

    def restore_grid_state(
        self,
        *,
        grid: np.ndarray,
        explored_by_robot: np.ndarray,
        visit_count: np.ndarray,
        last_seen: np.ndarray,
    ) -> None:
        """Wholesale-replace the belief's internal arrays and bump revision
        by exactly one.

        Used only by navigation-debug-snapshot restore, which replays a
        captured historical frame (grid/explored_by_robot/visit_count/
        last_seen together) rather than incremental per-cell updates -- the
        caller already owns shape validation against the live arrays before
        calling this. Always bumps once per call rather than comparing
        old/new content: the caller is restoring a specific historical
        revision, which must be distinguishable from whatever revision
        preceded the restore even on the (rare) chance the restored content
        happens to be byte-identical.
        """
        self.grid = grid
        self.explored_by_robot = explored_by_robot
        self.visit_count = visit_count
        self.last_seen = last_seen
        self._grid_revision += 1

    def snapshot(self) -> "ExplorationMapSnapshot":
        """Immutable ExplorationMapSnapshot of the current grid/bounds/
        resolution/revision. The grid array is copied and frozen by
        ExplorationMapSnapshot's own contract (see map_snapshots.py) --
        this does not duplicate that freezing logic here.
        """
        from robotics_sim.environment.map_snapshots import ExplorationMapSnapshot

        return ExplorationMapSnapshot(
            grid=self.grid,
            bounds=self.bounds,
            resolution=self.resolution,
            revision=self.revision,
        )

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def in_bounds_world(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.geometry.in_bounds_world(x, y)

    def world_to_cell(self, point: tuple[float, float], *, clamp: bool = False) -> tuple[int, int] | None:
        cell = self.geometry.world_to_grid(point[0], point[1], clamp=clamp)
        if cell is None:
            return None
        return int(cell.row), int(cell.col)

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        x, y = self.geometry.grid_to_world(GridCell(row=int(row), col=int(col)))
        return (round(float(x), 3), round(float(y), 3))

    def cell_bounds(self, cell: tuple[int, int]) -> tuple[float, float, float, float]:
        row, col = cell
        return self.geometry.cell_bounds(GridCell(row=int(row), col=int(col)))

    def _valid_cell(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= int(row) < self.height and 0 <= int(col) < self.width

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def mark_free_cell(
        self,
        cell: tuple[int, int],
        robot_index: int | None = None,
        time_s: float | None = None,
    ) -> None:
        row, col = map(int, cell)
        if not self._valid_cell((row, col)):
            return

        changed = False
        if self.grid[row, col] != OCCUPIED and self.grid[row, col] != FREE:
            self.grid[row, col] = FREE
            changed = True

        self.visit_count[row, col] = min(
            int(self.visit_count[row, col]) + 1,
            np.iinfo(np.uint16).max,
        )

        if robot_index is not None and 0 <= int(robot_index) < self.robot_count:
            robot_index = int(robot_index)
            if not self.explored_by_robot[robot_index, row, col]:
                self.explored_by_robot[robot_index, row, col] = True
                changed = True

        if time_s is not None:
            self.last_seen[row, col] = float(time_s)

        if changed:
            self._grid_revision += 1

    def force_free_cell(
        self,
        cell: tuple[int, int],
        robot_index: int | None = None,
        time_s: float | None = None,
    ) -> bool:
        """Force a cell to FREE, even if it was previously OCCUPIED.

        Reserved for the cell currently occupied by a live robot center. Do not
        use this as a general obstacle eraser.
        """
        row, col = map(int, cell)
        if not self._valid_cell((row, col)):
            return False

        changed = self.grid[row, col] != FREE
        self.grid[row, col] = FREE
        self.visit_count[row, col] = max(1, int(self.visit_count[row, col]))

        if robot_index is not None and 0 <= int(robot_index) < self.robot_count:
            robot_index = int(robot_index)
            if not self.explored_by_robot[robot_index, row, col]:
                self.explored_by_robot[robot_index, row, col] = True
                changed = True

        if time_s is not None:
            self.last_seen[row, col] = float(time_s)

        if changed:
            self._grid_revision += 1
        return bool(changed)

    def force_free_point(
        self,
        point: tuple[float, float],
        robot_index: int | None = None,
        time_s: float | None = None,
    ) -> bool:
        cell = self.world_to_cell(point, clamp=True)
        if cell is None:
            return False
        return self.force_free_cell(cell, robot_index=robot_index, time_s=time_s)

    def mark_occupied_cell(self, cell: tuple[int, int], time_s: float | None = None) -> None:
        row, col = map(int, cell)
        if not self._valid_cell((row, col)):
            return

        changed = self.grid[row, col] != OCCUPIED
        self.grid[row, col] = OCCUPIED
        if time_s is not None:
            self.last_seen[row, col] = float(time_s)
        if changed:
            self._grid_revision += 1

    def mark_visible_polygon(
        self,
        polygon: list[tuple[float, float]],
        *,
        robot_index: int | None = None,
        time_s: float | None = None,
    ) -> int:
        """Rasterize an occlusion-aware sensor polygon into FREE cells."""
        if len(polygon) < 3:
            return 0

        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]

        min_x = max(self.x_min, min(xs))
        max_x = min(self.x_max, max(xs))
        min_y = max(self.y_min, min(ys))
        max_y = min(self.y_max, max(ys))

        start_cell = self.geometry.world_to_grid(min_x, min_y, clamp=True)
        end_cell = self.geometry.world_to_grid(max_x, max_y, clamp=True)

        if start_cell is None or end_cell is None:
            return 0

        r0 = max(0, min(start_cell.row, end_cell.row))
        r1 = min(self.height - 1, max(start_cell.row, end_cell.row))
        c0 = max(0, min(start_cell.col, end_cell.col))
        c1 = min(self.width - 1, max(start_cell.col, end_cell.col))

        changed = 0
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                world = self.cell_to_world((row, col))
                if not _point_inside_polygon(world, polygon):
                    continue

                before = self.grid[row, col]
                self.mark_free_cell((row, col), robot_index=robot_index, time_s=time_s)
                if before == UNKNOWN:
                    changed += 1

        return changed

    def mark_occupied_points(
        self,
        points: Iterable[tuple[float, float]],
        *,
        time_s: float | None = None,
    ) -> int:
        """Mark obstacle-hit points as OCCUPIED cells."""
        changed = 0

        for point in points:
            cell = self.world_to_cell(point)
            if cell is None:
                continue

            row, col = cell
            before = self.grid[row, col]
            self.mark_occupied_cell(cell, time_s=time_s)

            if before != OCCUPIED:
                changed += 1

        return changed

    # ------------------------------------------------------------------
    # Query/export helpers
    # ------------------------------------------------------------------

    def explored_points(self) -> set[tuple[float, float]]:
        rows, cols = np.where(self.grid == FREE)
        return {self.cell_to_world((int(r), int(c))) for r, c in zip(rows, cols)}

    def occupied_points(self) -> list[tuple[float, float]]:
        rows, cols = np.where(self.grid == OCCUPIED)
        return [self.cell_to_world((int(r), int(c))) for r, c in zip(rows, cols)]

    def robot_explored_points(self, robot_index: int) -> set[tuple[float, float]]:
        idx = int(robot_index)
        if idx < 0 or idx >= self.robot_count:
            return set()

        rows, cols = np.where(self.explored_by_robot[idx])
        return {self.cell_to_world((int(r), int(c))) for r, c in zip(rows, cols)}

    def cell_state(self, point: tuple[float, float]) -> int:
        cell = self.world_to_cell(point)
        if cell is None:
            return OCCUPIED
        row, col = cell
        return int(self.grid[row, col])

    def seen_count_at_cell(self, cell: tuple[int, int]) -> int:
        row, col = map(int, cell)
        if not self._valid_cell((row, col)):
            return 0
        return int(self.visit_count[row, col])

    def seen_counts_dict(self) -> dict[tuple[int, int], int]:
        rows, cols = np.where(self.visit_count > 0)
        return {
            (int(r), int(c)): int(self.visit_count[int(r), int(c)])
            for r, c in zip(rows, cols)
        }

    def route_known_reuse_fraction(self, route_points: list[tuple[float, float]]) -> float:
        """Fraction of sampled route cells that are already FREE."""
        if len(route_points) < 2:
            return 0.0

        samples = []
        step = max(self.resolution, 1e-6)

        for a, b in zip(route_points[:-1], route_points[1:]):
            ax, ay = a
            bx, by = b
            length = math.hypot(bx - ax, by - ay)
            count = max(1, int(math.ceil(length / step)))

            for k in range(count + 1):
                t = k / count
                samples.append((ax + (bx - ax) * t, ay + (by - ay) * t))

        if not samples:
            return 0.0

        known = 0
        for point in samples:
            if self.cell_state(point) == FREE:
                known += 1

        return known / len(samples)

    def average_seen_penalty(
        self,
        cells: Iterable[tuple[int, int]],
        *,
        saturation: float = 5.0,
    ) -> float:
        """Average saturated visit penalty for a set/list of cells."""
        unique = set((int(r), int(c)) for r, c in cells)
        if not unique:
            return 0.0

        sat = max(float(saturation), 1e-6)
        total = 0.0

        for cell in unique:
            total += min(1.0, self.seen_count_at_cell(cell) / sat)

        return total / len(unique)

    def to_planning_grid(
        self,
        *,
        unknown_is_traversable: bool = True,
        inflate_radius: float = 0.0,
    ):
        """Create an OccupancyGrid projection for A*/Dijkstra.

        BeliefMap stores the logical observation state. The planning grid is a
        derived projection:
            - OCCUPIED remains OCCUPIED and can be inflated.
            - UNKNOWN is either traversable or blocked depending on policy.
            - FREE remains FREE.

        Inflation is applied to occupied cell centers, not to the belief map.
        """
        from robotics_sim.environment.occupancy_grid import (
            FREE as OG_FREE,
            OCCUPIED as OG_OCCUPIED,
            UNKNOWN as OG_UNKNOWN,
            OccupancyGrid,
        )

        initial = OG_FREE if unknown_is_traversable else OG_UNKNOWN

        grid = OccupancyGrid.from_bounds(
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=self.y_max,
            resolution=self.resolution,
            initial_value=initial,
            unknown_is_traversable=unknown_is_traversable,
        )

        free_rows, free_cols = np.where(self.grid == FREE)
        for row, col in zip(free_rows, free_cols):
            grid.set_value(GridCell(int(row), int(col)), OG_FREE)

        occupied_points = self.occupied_points()
        if inflate_radius > 0.0:
            grid.add_obstacle_points(occupied_points, padding=float(inflate_radius))
        else:
            for row, col in zip(*np.where(self.grid == OCCUPIED)):
                grid.set_value(GridCell(int(row), int(col)), OG_OCCUPIED)

        return grid

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def stats(self) -> BeliefMapStats:
        total = int(self.grid.size)
        free_mask = self.grid == FREE
        occupied_mask = self.grid == OCCUPIED

        unknown = int(np.count_nonzero(self.grid == UNKNOWN))
        free = int(np.count_nonzero(free_mask))
        occupied = int(np.count_nonzero(occupied_mask))
        known = free + occupied

        coverage = 100.0 * free / max(1, total)

        if self.robot_count > 1:
            observed_layers = self.explored_by_robot.sum(axis=0)
            free_observed_layers = observed_layers[free_mask]
            overlap_cells = int(np.count_nonzero(free_observed_layers > 1))
            observed_any = int(np.count_nonzero(free_observed_layers > 0))
            overlap_ratio = overlap_cells / max(1, observed_any)
        else:
            overlap_cells = 0
            overlap_ratio = 0.0

        free_visit_counts = self.visit_count[free_mask]
        revisited_cells = int(np.count_nonzero(free_visit_counts > 1))
        total_free_observations = int(np.sum(free_visit_counts, dtype=np.uint64)) if free else 0
        revisit_ratio = revisited_cells / max(1, free)
        average_visits = total_free_observations / max(1, free)

        return BeliefMapStats(
            unknown_cells=unknown,
            free_cells=free,
            occupied_cells=occupied,
            known_cells=known,
            total_cells=total,
            coverage_percent=coverage,
            overlap_cells=overlap_cells,
            overlap_ratio=overlap_ratio,
            revisited_cells=revisited_cells,
            revisit_ratio=revisit_ratio,
            total_free_observations=total_free_observations,
            average_visits_per_free_cell=average_visits,
        )

    def per_robot_explored_counts(self) -> list[int]:
        free_mask = self.grid == FREE
        counts: list[int] = []

        for idx in range(self.robot_count):
            counts.append(int(np.count_nonzero(self.explored_by_robot[idx] & free_mask)))

        return counts

    def per_robot_overlap_counts(self) -> list[int]:
        if self.robot_count <= 1:
            return [0 for _ in range(self.robot_count)]

        free_mask = self.grid == FREE
        observed_layers = self.explored_by_robot.sum(axis=0)
        overlap_mask = (observed_layers > 1) & free_mask

        counts: list[int] = []
        for idx in range(self.robot_count):
            counts.append(int(np.count_nonzero(self.explored_by_robot[idx] & overlap_mask)))

        return counts
