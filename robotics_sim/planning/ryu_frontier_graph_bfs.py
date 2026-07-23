"""BFS frontier exploration adapted from Ryu's frontier-graph method.

Source:
    H. Ryu, "Graph Search-Based Exploration Method Using a Frontier-Graph
    Structure for Mobile Robots", Sensors 20(21), 6270, 2020.
    https://doi.org/10.3390/s20216270

The paper detects free cells adjacent to unknown space, segments them with
8-connected CCL, and uses breadth-first graph exploration to choose the next
frontier node (Algorithms 5 and 6).  The simulator has one shared occupancy
grid rather than the paper's local-map database, so reachable FREE cells form
the graph searched here.  DBSCAN clusters can be supplied as the frontier
nodes; when that stage cannot supply nodes, the paper's 8-connected CCL is the
explicit fallback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

from robotics_sim.environment.belief_map import BeliefMap, FREE, UNKNOWN


RYU_FRONTIER_GRAPH_BFS = "Ryu frontier-graph BFS exploration"
RYU_FRONTIER_GRAPH_BFS_CITATION = (
    "Ryu, H. Graph Search-Based Exploration Method Using a Frontier-Graph "
    "Structure for Mobile Robots. Sensors 2020, 20(21), 6270. "
    "https://doi.org/10.3390/s20216270"
)

GridCell = tuple[int, int]


@dataclass(frozen=True)
class BFSFrontierNode:
    cells: tuple[GridCell, ...]
    representative: GridCell
    bfs_depth: int


def _valid(belief: BeliefMap, cell: GridCell) -> bool:
    row, col = cell
    return 0 <= row < belief.height and 0 <= col < belief.width


def _neighbors4(cell: GridCell) -> tuple[GridCell, ...]:
    row, col = cell
    return (
        (row - 1, col),
        (row, col - 1),
        (row, col + 1),
        (row + 1, col),
    )


def _neighbors8(cell: GridCell) -> tuple[GridCell, ...]:
    row, col = cell
    return tuple(
        (row + dr, col + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    )


def _is_frontier(belief: BeliefMap, cell: GridCell) -> bool:
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


def _nearest_free_cell(belief: BeliefMap, robot_xy: tuple[float, float]) -> GridCell | None:
    start = belief.world_to_cell(robot_xy)
    if start is not None and int(belief.grid[start[0], start[1]]) == FREE:
        return start

    free_rows, free_cols = (belief.grid == FREE).nonzero()
    if len(free_rows) == 0:
        return None
    return min(
        zip(free_rows.tolist(), free_cols.tolist()),
        key=lambda cell: math.dist(belief.cell_to_world(cell), robot_xy),
    )


def reachable_free_depths(
    belief: BeliefMap,
    robot_xy: tuple[float, float],
) -> dict[GridCell, int]:
    """Return deterministic four-connected BFS depths from the robot."""
    start = _nearest_free_cell(belief, robot_xy)
    if start is None:
        return {}

    depths = {start: 0}
    queue: deque[GridCell] = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in _neighbors4(cell):
            if neighbor in depths or not _valid(belief, neighbor):
                continue
            if int(belief.grid[neighbor[0], neighbor[1]]) != FREE:
                continue
            depths[neighbor] = depths[cell] + 1
            queue.append(neighbor)
    return depths


def _ccl8(frontiers: Iterable[GridCell]) -> tuple[tuple[GridCell, ...], ...]:
    """Paper Section 3.1.2 frontier segmentation using 8-connectivity."""
    remaining = set(frontiers)
    components: list[tuple[GridCell, ...]] = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        queue: deque[GridCell] = deque([seed])
        component = [seed]
        while queue:
            for neighbor in _neighbors8(queue.popleft()):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
                component.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _representative(cells: tuple[GridCell, ...]) -> GridCell:
    """Cell nearest the component median, as specified by Ryu."""
    rows = sorted(cell[0] for cell in cells)
    cols = sorted(cell[1] for cell in cells)
    middle = len(cells) // 2
    median = (rows[middle], cols[middle])
    return min(
        cells,
        key=lambda cell: (
            (cell[0] - median[0]) ** 2 + (cell[1] - median[1]) ** 2,
            cell,
        ),
    )


def bfs_frontier_nodes(
    belief: BeliefMap,
    robot_xy: tuple[float, float],
    *,
    dbscan_clusters: Iterable[Iterable[GridCell]] | None = None,
) -> tuple[BFSFrontierNode, ...]:
    """Build frontier nodes ordered by BFS depth and deterministic TSP tie order."""
    depths = reachable_free_depths(belief, robot_xy)
    if not depths:
        return ()

    if dbscan_clusters is None:
        components = _ccl8(cell for cell in depths if _is_frontier(belief, cell))
    else:
        components = tuple(
            tuple(sorted(set((int(row), int(col)) for row, col in cluster)))
            for cluster in dbscan_clusters
        )

    nodes: list[BFSFrontierNode] = []
    for component in components:
        reachable = tuple(cell for cell in component if cell in depths and _is_frontier(belief, cell))
        if not reachable:
            continue
        representative = _representative(reachable)
        nodes.append(
            BFSFrontierNode(
                cells=reachable,
                representative=representative,
                bfs_depth=min(depths[cell] for cell in reachable),
            )
        )

    # Algorithm 6 orders equal-priority adjacent nodes deterministically. Grid
    # coordinates provide the stable node IDs used by this runtime adapter.
    return tuple(
        sorted(
            nodes,
            key=lambda node: (
                node.bfs_depth,
                node.representative,
                -len(node.cells),
            ),
        )
    )
