"""
Shared grid geometry conventions for the simulator.

Official convention:
    - A grid cell is a square area, not a point sample.
    - World coordinates are continuous (x, y).
    - Grid coordinates are discrete GridCell(row, col).
    - x maps to col, y maps to row.
    - Bounds are semi-open:
          x_min <= x < x_max
          y_min <= y < y_max
    - world_to_grid uses floor, because it returns the cell containing a point.
    - grid_to_world returns the center of the cell.

Every map/planner/sensor module should use this geometry instead of implementing
its own world <-> grid conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int


class GridGeometry:
    """Canonical conversion between continuous world coordinates and grid cells."""

    def __init__(self, bounds: tuple[float, float, float, float], resolution: float):
        self.x_min, self.x_max, self.y_min, self.y_max = map(float, bounds)
        self.resolution = max(float(resolution), 1e-6)

        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min.")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min.")

        self.width = int(math.ceil((self.x_max - self.x_min) / self.resolution))
        self.height = int(math.ceil((self.y_max - self.y_min) / self.resolution))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x_min, self.x_max, self.y_min, self.y_max

    def in_bounds_world(self, x: float, y: float) -> bool:
        return (
            self.x_min <= float(x) < self.x_max
            and self.y_min <= float(y) < self.y_max
        )

    def clamp_world_point(self, x: float, y: float) -> tuple[float, float]:
        """Clamp a point into the semi-open world domain.

        Useful for GUI clicks exactly on x_max/y_max.
        """
        eps = self.resolution * 1e-9
        cx = min(max(float(x), self.x_min), self.x_max - eps)
        cy = min(max(float(y), self.y_min), self.y_max - eps)
        return cx, cy

    def in_bounds_cell(self, cell: GridCell) -> bool:
        return 0 <= int(cell.row) < self.height and 0 <= int(cell.col) < self.width

    def world_to_grid(self, x: float, y: float, *, clamp: bool = False) -> GridCell | None:
        if clamp:
            x, y = self.clamp_world_point(x, y)

        if not self.in_bounds_world(x, y):
            return None

        col = int(math.floor((float(x) - self.x_min) / self.resolution))
        row = int(math.floor((float(y) - self.y_min) / self.resolution))

        cell = GridCell(row=row, col=col)
        if not self.in_bounds_cell(cell):
            return None

        return cell

    def grid_to_world(self, cell: GridCell) -> tuple[float, float]:
        if not self.in_bounds_cell(cell):
            raise IndexError("Grid cell out of bounds.")

        x = self.x_min + (int(cell.col) + 0.5) * self.resolution
        y = self.y_min + (int(cell.row) + 0.5) * self.resolution
        return float(x), float(y)

    def cell_bounds(self, cell: GridCell) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) for a cell area."""
        if not self.in_bounds_cell(cell):
            raise IndexError("Grid cell out of bounds.")

        x0 = self.x_min + int(cell.col) * self.resolution
        x1 = x0 + self.resolution
        y0 = self.y_min + int(cell.row) * self.resolution
        y1 = y0 + self.resolution
        return float(x0), float(x1), float(y0), float(y1)

    def iter_cells(self):
        for row in range(self.height):
            for col in range(self.width):
                yield GridCell(row=row, col=col)
