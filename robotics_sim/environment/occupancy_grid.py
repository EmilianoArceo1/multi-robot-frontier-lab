"""
Occupancy-grid representation used by grid-based planners.

This class is a planning/grid abstraction. It uses the project-wide grid
convention defined in robotics_sim.environment.grid_geometry:

    - cells are square areas
    - world_to_grid uses floor
    - grid_to_world returns cell centers
    - bounds are semi-open [x_min, x_max), [y_min, y_max)

Cell states:
    UNKNOWN  = -1
    FREE     = 0
    OCCUPIED = 1
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from robotics_sim.environment.grid_geometry import GridCell, GridGeometry

FREE = 0
OCCUPIED = 1
UNKNOWN = -1

_ALLOWED_VALUES = {FREE, OCCUPIED, UNKNOWN}


class OccupancyGrid:
    """
    Discrete map used by A* and Dijkstra.

    This can represent either:
        - a belief grid, with UNKNOWN/FREE/OCCUPIED, or
        - a planning projection, where UNKNOWN may be treated as traversable.

    The object does not know the ground truth. It only stores the values provided
    by mapping or by a projection from BeliefMap.
    """

    def __init__(
        self,
        width: int,
        height: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        *,
        initial_value: int = UNKNOWN,
        unknown_is_traversable: bool = False,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("Grid width and height must be positive.")
        if resolution <= 0:
            raise ValueError("Grid resolution must be positive.")
        if int(initial_value) not in _ALLOWED_VALUES:
            raise ValueError(
                f"Invalid initial_value={initial_value}. "
                f"Use FREE={FREE}, OCCUPIED={OCCUPIED}, or UNKNOWN={UNKNOWN}."
            )

        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.unknown_is_traversable = bool(unknown_is_traversable)

        bounds = (
            self.origin_x,
            self.origin_x + self.width * self.resolution,
            self.origin_y,
            self.origin_y + self.height * self.resolution,
        )
        self.geometry = GridGeometry(bounds, self.resolution)

        self.data = np.full(
            (self.height, self.width),
            int(initial_value),
            dtype=np.int8,
        )

    @classmethod
    def from_bounds(
        cls,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float,
        *,
        initial_value: int = UNKNOWN,
        unknown_is_traversable: bool = False,
    ) -> "OccupancyGrid":
        geometry = GridGeometry((x_min, x_max, y_min, y_max), resolution)
        return cls(
            width=geometry.width,
            height=geometry.height,
            resolution=geometry.resolution,
            origin_x=geometry.x_min,
            origin_y=geometry.y_min,
            initial_value=initial_value,
            unknown_is_traversable=unknown_is_traversable,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.geometry.bounds

    def copy(self) -> "OccupancyGrid":
        other = OccupancyGrid(
            width=self.width,
            height=self.height,
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            initial_value=UNKNOWN,
            unknown_is_traversable=self.unknown_is_traversable,
        )
        other.data = self.data.copy()
        return other

    def in_bounds(self, cell: GridCell) -> bool:
        return self.geometry.in_bounds_cell(cell)

    def world_to_grid(self, x: float, y: float, *, clamp: bool = False) -> GridCell:
        cell = self.geometry.world_to_grid(float(x), float(y), clamp=clamp)
        if cell is None:
            # Preserve old behavior: return an out-of-bounds cell instead of None.
            col = math.floor((float(x) - self.origin_x) / self.resolution)
            row = math.floor((float(y) - self.origin_y) / self.resolution)
            return GridCell(row=int(row), col=int(col))
        return cell

    def grid_to_world(self, cell: GridCell) -> tuple[float, float]:
        return self.geometry.grid_to_world(cell)

    def cell_bounds(self, cell: GridCell) -> tuple[float, float, float, float]:
        return self.geometry.cell_bounds(cell)

    def get_value(self, cell: GridCell) -> int:
        if not self.in_bounds(cell):
            raise IndexError("Grid cell out of bounds.")
        return int(self.data[cell.row, cell.col])

    def set_value(self, cell: GridCell, value: int) -> None:
        value = int(value)
        if value not in _ALLOWED_VALUES:
            raise ValueError(
                f"Invalid grid value={value}. "
                f"Use FREE={FREE}, OCCUPIED={OCCUPIED}, or UNKNOWN={UNKNOWN}."
            )

        if self.in_bounds(cell):
            self.data[cell.row, cell.col] = value

    def mark_free(self, cell: GridCell) -> None:
        self.set_value(cell, FREE)

    def mark_occupied(self, cell: GridCell) -> None:
        self.set_value(cell, OCCUPIED)

    def mark_unknown(self, cell: GridCell) -> None:
        self.set_value(cell, UNKNOWN)

    def is_free(self, cell: GridCell) -> bool:
        return self.in_bounds(cell) and self.get_value(cell) == FREE

    def is_occupied(self, cell: GridCell) -> bool:
        return self.in_bounds(cell) and self.get_value(cell) == OCCUPIED

    def is_unknown(self, cell: GridCell) -> bool:
        return self.in_bounds(cell) and self.get_value(cell) == UNKNOWN

    def is_traversable(self, cell: GridCell, allow_unknown: bool | None = None) -> bool:
        if not self.in_bounds(cell):
            return False

        value = self.get_value(cell)
        if value == OCCUPIED:
            return False
        if value == FREE:
            return True
        if value == UNKNOWN:
            return self.unknown_is_traversable if allow_unknown is None else bool(allow_unknown)
        return False

    def set_obstacle_rect_world(
        self,
        rect: tuple[float, float, float, float],
        padding: float = 0.0,
    ) -> None:
        """Mark cells whose square area overlaps an inflated obstacle rectangle."""
        x, y, width, height = map(float, rect)
        if width < 0 or height < 0:
            raise ValueError("Obstacle width and height must be non-negative.")

        padding = max(0.0, float(padding))

        x_min = x - padding
        y_min = y - padding
        x_max = x + width + padding
        y_max = y + height + padding

        row_start = max(0, int(math.floor((y_min - self.origin_y) / self.resolution)))
        row_end = min(self.height - 1, int(math.floor((y_max - self.origin_y) / self.resolution)))
        col_start = max(0, int(math.floor((x_min - self.origin_x) / self.resolution)))
        col_end = min(self.width - 1, int(math.floor((x_max - self.origin_x) / self.resolution)))

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                cell = GridCell(row=row, col=col)
                cx0, cx1, cy0, cy1 = self.cell_bounds(cell)

                overlaps = (
                    cx1 >= x_min
                    and cx0 <= x_max
                    and cy1 >= y_min
                    and cy0 <= y_max
                )
                if overlaps:
                    self.mark_occupied(cell)

    def add_rectangular_obstacles(
        self,
        obstacles: Iterable[tuple[float, float, float, float]],
        padding: float = 0.0,
    ) -> None:
        for obstacle in obstacles:
            self.set_obstacle_rect_world(tuple(obstacle), padding=padding)

    def set_obstacle_point_world(
        self,
        point: tuple[float, float],
        padding: float = 0.0,
    ) -> None:
        """
        Mark grid cells around a mapped obstacle point as occupied.

        Even with padding=0, the cell containing the obstacle point is marked.
        Padding inflates the point by a safety radius.
        """
        px, py = float(point[0]), float(point[1])
        radius = max(0.0, float(padding))

        base_cell = self.world_to_grid(px, py)
        if self.in_bounds(base_cell):
            self.mark_occupied(base_cell)

        if radius == 0.0:
            return

        min_cell = self.world_to_grid(px - radius, py - radius)
        max_cell = self.world_to_grid(px + radius, py + radius)

        row_start = max(0, min(min_cell.row, max_cell.row))
        row_end = min(self.height - 1, max(min_cell.row, max_cell.row))
        col_start = max(0, min(min_cell.col, max_cell.col))
        col_end = min(self.width - 1, max(min_cell.col, max_cell.col))

        threshold = radius + self.resolution * math.sqrt(2.0) * 0.5

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                cell = GridCell(row=row, col=col)
                cx, cy = self.grid_to_world(cell)

                if math.hypot(cx - px, cy - py) <= threshold:
                    self.mark_occupied(cell)

    def add_obstacle_points(
        self,
        points: Iterable[tuple[float, float]],
        padding: float = 0.0,
    ) -> None:
        for point in points:
            self.set_obstacle_point_world(tuple(point), padding=padding)

    def neighbors(
        self,
        cell: GridCell,
        allow_diagonal: bool = True,
        prevent_corner_cutting: bool = True,
        allow_unknown: bool | None = None,
    ) -> list[tuple[GridCell, float]]:
        straight = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
        ]
        diagonal = [
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        ]
        directions = straight + diagonal if allow_diagonal else straight

        result: list[tuple[GridCell, float]] = []

        for dr, dc, cost in directions:
            nxt = GridCell(cell.row + dr, cell.col + dc)
            if not self.is_traversable(nxt, allow_unknown=allow_unknown):
                continue

            if allow_diagonal and dr != 0 and dc != 0 and prevent_corner_cutting:
                side_a = GridCell(cell.row + dr, cell.col)
                side_b = GridCell(cell.row, cell.col + dc)

                if (
                    not self.is_traversable(side_a, allow_unknown=allow_unknown)
                    or not self.is_traversable(side_b, allow_unknown=allow_unknown)
                ):
                    continue

            result.append((nxt, cost * self.resolution))

        return result
