"""
Grid-path simplification utilities.

These functions simplify a path after A*/Dijkstra has produced a valid grid path.

Important rule:
    Simplification must not create a segment that crosses blocked cells.

The grid decides traversability through grid.is_traversable(...). Therefore this
module stays consistent with the planner's OccupancyGrid convention.
"""

from __future__ import annotations

import math

from robotics_sim.environment.occupancy_grid import GridCell, OccupancyGrid


DEFAULT_PATH_SIMPLIFIER = "Direction changes"

PATH_SIMPLIFIER_OPTIONS = [
    "Raw grid path",
    "Direction changes",
    "Direction changes + spacing",
    "Line of sight grid-safe",
    "RDP grid-safe",
]


def grid_path_to_world_path(
    grid: OccupancyGrid,
    grid_path: list[GridCell],
) -> list[tuple[float, float]]:
    """Convert grid cells to world-coordinate waypoints at cell centers."""
    return [grid.grid_to_world(cell) for cell in grid_path]


def simplify_path_by_direction_changes(path: list[GridCell]) -> list[GridCell]:
    """
    Keep only the start, direction-change cells, and final cell.

    This is safe because every remaining straight segment follows the original
    grid path direction.
    """
    if len(path) <= 2:
        return path[:]

    simplified = [path[0]]

    prev_dr = path[1].row - path[0].row
    prev_dc = path[1].col - path[0].col

    for i in range(1, len(path) - 1):
        cur = path[i]
        nxt = path[i + 1]

        dr = nxt.row - cur.row
        dc = nxt.col - cur.col

        if (dr, dc) != (prev_dr, prev_dc):
            simplified.append(cur)
            prev_dr, prev_dc = dr, dc

    simplified.append(path[-1])
    return simplified


def _bresenham_cells(a: GridCell, b: GridCell) -> list[GridCell]:
    """
    Return cells on a discrete line between two grid cells.

    This is used for grid-safe line-of-sight checks. It is conservative enough
    for the simulator baseline, but not a continuous geometry proof.
    """
    r0, c0 = int(a.row), int(a.col)
    r1, c1 = int(b.row), int(b.col)

    cells: list[GridCell] = []

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    step_r = 1 if r0 < r1 else -1
    step_c = 1 if c0 < c1 else -1

    r, c = r0, c0

    if dc > dr:
        error = dc / 2.0
        while c != c1:
            cells.append(GridCell(r, c))
            error -= dr
            if error < 0:
                r += step_r
                error += dc
            c += step_c
    else:
        error = dr / 2.0
        while r != r1:
            cells.append(GridCell(r, c))
            error -= dc
            if error < 0:
                c += step_c
                error += dr
            r += step_r

    cells.append(GridCell(r1, c1))
    return cells


def _is_traversable(grid: OccupancyGrid, cell: GridCell) -> bool:
    if hasattr(grid, "is_traversable"):
        return bool(grid.is_traversable(cell))
    if hasattr(grid, "is_free"):
        return bool(grid.is_free(cell))
    return False


def _line_of_sight_grid_safe(grid: OccupancyGrid, a: GridCell, b: GridCell) -> bool:
    """Return whether the segment a-b stays in traversable cells."""
    for cell in _bresenham_cells(a, b):
        if not _is_traversable(grid, cell):
            return False
    return True


def simplify_path_by_line_of_sight(
    path: list[GridCell],
    grid: OccupancyGrid,
) -> list[GridCell]:
    """
    Greedily connect each kept point to the farthest visible future point.

    Every shortcut is validated against the grid.
    """
    if len(path) <= 2:
        return path[:]

    simplified = [path[0]]
    anchor_idx = 0

    while anchor_idx < len(path) - 1:
        next_idx = anchor_idx + 1

        for candidate_idx in range(len(path) - 1, anchor_idx, -1):
            if _line_of_sight_grid_safe(grid, path[anchor_idx], path[candidate_idx]):
                next_idx = candidate_idx
                break

        simplified.append(path[next_idx])
        anchor_idx = next_idx

    return simplified


def simplify_path_by_direction_changes_and_spacing(
    path: list[GridCell],
    *,
    min_spacing_cells: int = 3,
) -> list[GridCell]:
    """
    Direction-change simplification plus minimum spacing between kept waypoints.

    This is useful when too many small zig-zags remain after direction-change
    simplification.
    """
    base = simplify_path_by_direction_changes(path)

    if len(base) <= 2:
        return base

    min_spacing_cells = max(1, int(min_spacing_cells))
    simplified = [base[0]]
    last = base[0]

    for cell in base[1:-1]:
        grid_dist = max(abs(cell.row - last.row), abs(cell.col - last.col))
        if grid_dist >= min_spacing_cells:
            simplified.append(cell)
            last = cell

    simplified.append(base[-1])
    return simplified


def _perpendicular_distance(cell: GridCell, a: GridCell, b: GridCell) -> float:
    """Distance from a grid cell to segment a-b in cell coordinates."""
    x = float(cell.col)
    y = float(cell.row)
    x1 = float(a.col)
    y1 = float(a.row)
    x2 = float(b.col)
    y2 = float(b.row)

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0.0 and dy == 0.0:
        return math.hypot(x - x1, y - y1)

    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.hypot(x - proj_x, y - proj_y)


def _rdp(path: list[GridCell], epsilon_cells: float) -> list[GridCell]:
    if len(path) <= 2:
        return path[:]

    start = path[0]
    end = path[-1]

    max_dist = -1.0
    max_index = -1

    for i in range(1, len(path) - 1):
        dist = _perpendicular_distance(path[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    if max_dist <= epsilon_cells:
        return [start, end]

    left = _rdp(path[: max_index + 1], epsilon_cells)
    right = _rdp(path[max_index:], epsilon_cells)

    return left[:-1] + right


def simplify_path_by_rdp_grid_safe(
    path: list[GridCell],
    grid: OccupancyGrid,
    *,
    epsilon_cells: float = 1.5,
) -> list[GridCell]:
    """
    Ramer-Douglas-Peucker simplification with grid-safe repair.

    RDP proposes a shorter polyline. Each proposed segment is accepted only if
    line-of-sight is traversable. Otherwise the corresponding original subpath
    is simplified with grid-safe line-of-sight.
    """
    if len(path) <= 2:
        return path[:]

    proposed = _rdp(path, float(epsilon_cells))

    if len(proposed) <= 2 and _line_of_sight_grid_safe(grid, proposed[0], proposed[-1]):
        return proposed

    # Build a position map to avoid path.index(...) ambiguity.
    positions: dict[GridCell, list[int]] = {}
    for idx, cell in enumerate(path):
        positions.setdefault(cell, []).append(idx)

    repaired = [proposed[0]]
    search_start = 0

    for point in proposed[1:]:
        candidate_indices = [idx for idx in positions.get(point, []) if idx >= search_start]

        if not candidate_indices:
            continue

        point_idx = candidate_indices[0]
        anchor = repaired[-1]

        if _line_of_sight_grid_safe(grid, anchor, point):
            repaired.append(point)
            search_start = point_idx
            continue

        anchor_indices = [idx for idx in positions.get(anchor, []) if idx <= point_idx]
        anchor_idx = anchor_indices[-1] if anchor_indices else search_start

        subpath = path[anchor_idx : point_idx + 1]
        safe_subpath = simplify_path_by_line_of_sight(subpath, grid)

        repaired.extend(safe_subpath[1:])
        search_start = point_idx

    if repaired[-1] != path[-1]:
        if _line_of_sight_grid_safe(grid, repaired[-1], path[-1]):
            repaired.append(path[-1])
        else:
            tail_start = positions.get(repaired[-1], [0])[-1]
            tail = simplify_path_by_line_of_sight(path[tail_start:], grid)
            repaired.extend(tail[1:])

    return repaired


def simplify_grid_path(
    path: list[GridCell],
    *,
    method: str = DEFAULT_PATH_SIMPLIFIER,
    grid: OccupancyGrid | None = None,
) -> list[GridCell]:
    """Dispatch path simplification by method name."""
    if not path:
        return []

    mode = str(method or DEFAULT_PATH_SIMPLIFIER).strip().lower()

    if "raw" in mode:
        return path[:]

    if "line" in mode and "sight" in mode:
        if grid is None:
            return simplify_path_by_direction_changes(path)
        return simplify_path_by_line_of_sight(path, grid)

    if "rdp" in mode:
        if grid is None:
            return simplify_path_by_direction_changes(path)
        return simplify_path_by_rdp_grid_safe(path, grid)

    if "spacing" in mode:
        return simplify_path_by_direction_changes_and_spacing(path)

    return simplify_path_by_direction_changes(path)
