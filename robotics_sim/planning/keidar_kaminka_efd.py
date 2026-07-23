"""Incremental Wavefront Frontier Detector from Keidar and Kaminka.

Source:
    M. Keidar and G. A. Kaminka, "Efficient frontier detection for robot
    exploration", International Journal of Robotics Research 33(2), 2014,
    pp. 215-236. https://doi.org/10.1177/0278364913494911

This implements WFD-INC (Algorithm 6.1): a persistent frontier database,
wavefront/BFS traversal of known free space, an active area bounded by cells
changed since the previous map event, and maintenance that removes stale
frontiers inside that area. The simulator uses navigable FREE boundary cells
instead of the paper's UNKNOWN-side representation so downstream path planners
receive reachable goals.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import weakref

import numpy as np

from robotics_sim.environment.belief_map import BeliefMap, FREE, UNKNOWN


KEIDAR_KAMINKA_WFD_INC = "Keidar-Kaminka WFD-INC frontier detector"
KEIDAR_KAMINKA_EFD_CITATION = (
    "Keidar, M.; Kaminka, G.A. Efficient frontier detection for robot "
    "exploration. IJRR 33(2), 215-236. "
    "https://doi.org/10.1177/0278364913494911"
)

GridCell = tuple[int, int]


def _neighbors4(cell: GridCell) -> tuple[GridCell, ...]:
    row, col = cell
    return ((row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col))


def _neighbors8(cell: GridCell) -> tuple[GridCell, ...]:
    row, col = cell
    return tuple(
        (row + dr, col + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )


def _valid(belief: BeliefMap, cell: GridCell) -> bool:
    row, col = cell
    return 0 <= row < belief.height and 0 <= col < belief.width


def _frontier(belief: BeliefMap, cell: GridCell) -> bool:
    if not _valid(belief, cell):
        return False
    row, col = cell
    if int(belief.grid[row, col]) != FREE:
        return False
    return any(
        _valid(belief, neighbor)
        and int(belief.grid[neighbor[0], neighbor[1]]) == UNKNOWN
        for neighbor in _neighbors4(cell)
    )


def _nearest_free(belief: BeliefMap, robot_xy: tuple[float, float]) -> GridCell | None:
    cell = belief.world_to_cell(robot_xy)
    if cell is not None and int(belief.grid[cell[0], cell[1]]) == FREE:
        return cell
    rows, cols = np.nonzero(belief.grid == FREE)
    if len(rows) == 0:
        return None
    return min(
        zip(rows.tolist(), cols.tolist()),
        key=lambda item: (
            (belief.cell_to_world(item)[0] - robot_xy[0]) ** 2
            + (belief.cell_to_world(item)[1] - robot_xy[1]) ** 2,
            item,
        ),
    )


@dataclass(frozen=True)
class EFDResult:
    frontier_cells: tuple[GridCell, ...]
    active_area: tuple[int, int, int, int]
    full_scan: bool
    scanned_cells: int
    citation: str = KEIDAR_KAMINKA_EFD_CITATION


class WFDIncrementalDetector:
    def __init__(self) -> None:
        self._previous_grid: np.ndarray | None = None
        self._frontiers: set[GridCell] = set()

    def detect(
        self,
        belief: BeliefMap,
        robot_xy: tuple[float, float],
    ) -> EFDResult:
        grid = np.asarray(belief.grid)
        full_scan = (
            self._previous_grid is None
            or self._previous_grid.shape != grid.shape
        )
        if full_scan:
            active = (0, belief.height - 1, 0, belief.width - 1)
            self._frontiers.clear()
        else:
            changed = np.argwhere(self._previous_grid != grid)
            if changed.size == 0:
                valid = {cell for cell in self._frontiers if _frontier(belief, cell)}
                self._frontiers = valid
                return EFDResult(
                    frontier_cells=tuple(sorted(valid)),
                    active_area=(0, -1, 0, -1),
                    full_scan=False,
                    scanned_cells=0,
                )
            padding = 1
            active = (
                max(0, int(changed[:, 0].min()) - padding),
                min(belief.height - 1, int(changed[:, 0].max()) + padding),
                max(0, int(changed[:, 1].min()) - padding),
                min(belief.width - 1, int(changed[:, 1].max()) + padding),
            )

        self._previous_grid = grid.copy()
        row_min, row_max, col_min, col_max = active

        # Algorithm 6.1 maintenance: invalidate old frontier records in the
        # active area before merging the wavefront's new detections.
        self._frontiers = {
            cell
            for cell in self._frontiers
            if not (
                row_min <= cell[0] <= row_max
                and col_min <= cell[1] <= col_max
            )
            and _frontier(belief, cell)
        }

        start = _nearest_free(belief, robot_xy)
        if start is None:
            return EFDResult(tuple(sorted(self._frontiers)), active, full_scan, 0)

        queue: deque[GridCell] = deque([start])
        opened = {start}
        closed: set[GridCell] = set()
        new_frontiers: set[GridCell] = set()
        while queue:
            cell = queue.popleft()
            if cell in closed:
                continue
            closed.add(cell)
            if _frontier(belief, cell):
                frontier_queue: deque[GridCell] = deque([cell])
                while frontier_queue:
                    frontier_cell = frontier_queue.popleft()
                    if frontier_cell in new_frontiers or not _frontier(belief, frontier_cell):
                        continue
                    new_frontiers.add(frontier_cell)
                    for neighbor in _neighbors8(frontier_cell):
                        if (
                            row_min <= neighbor[0] <= row_max
                            and col_min <= neighbor[1] <= col_max
                        ):
                            frontier_queue.append(neighbor)

            for neighbor in _neighbors4(cell):
                if neighbor in opened or not _valid(belief, neighbor):
                    continue
                if not (
                    row_min <= neighbor[0] <= row_max
                    and col_min <= neighbor[1] <= col_max
                ):
                    continue
                row, col = neighbor
                has_free_neighbor = any(
                    _valid(belief, adjacent)
                    and int(belief.grid[adjacent[0], adjacent[1]]) == FREE
                    for adjacent in _neighbors4(neighbor)
                )
                if int(belief.grid[row, col]) == FREE and has_free_neighbor:
                    opened.add(neighbor)
                    queue.append(neighbor)

        self._frontiers.update(new_frontiers)
        return EFDResult(
            frontier_cells=tuple(sorted(self._frontiers)),
            active_area=active,
            full_scan=full_scan,
            scanned_cells=len(closed),
        )


_DETECTORS: "weakref.WeakKeyDictionary[BeliefMap, WFDIncrementalDetector]" = (
    weakref.WeakKeyDictionary()
)


def detect_frontiers_wfd_inc(
    belief: BeliefMap,
    robot_xy: tuple[float, float],
) -> EFDResult:
    detector = _DETECTORS.get(belief)
    if detector is None:
        detector = WFDIncrementalDetector()
        _DETECTORS[belief] = detector
    return detector.detect(belief, robot_xy)
