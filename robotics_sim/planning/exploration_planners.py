"""
Exploration target selection.

Responsibility split:
    exploration planner -> chooses where to go next
    path planner        -> computes how to get there
    controller          -> follows waypoints

Default frontier detector:
    Ryu frontier-graph BFS exploration

This file now receives BeliefMap directly when available. That avoids rebuilding
a separate grid convention from explored_points and mapped_obstacle_points.

Compatibility:
    Older engine code can still pass explored_points/mapped_obstacle_points with
    bounds/resolution. In that case a temporary BeliefMap is built using the
    same shared grid convention.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from typing import Any, Iterable, Mapping

import numpy as np

from robotics_sim.environment.belief_map import (
    BeliefMap,
    FREE,
    OCCUPIED,
    UNKNOWN,
)
from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.planning.grid_planners import AStarPlanner
from robotics_sim.planning.ryu_frontier_graph_bfs import (
    RYU_FRONTIER_GRAPH_BFS,
    RYU_FRONTIER_GRAPH_BFS_CITATION,
    bfs_frontier_nodes,
)


DEFAULT_EXPLORATION_PLANNER = RYU_FRONTIER_GRAPH_BFS
NAV2D_NEAREST_FRONTIER_PLANNER = "Nav2D nearest-frontier wavefront"

EXPLORATION_PLANNER_OPTIONS = [
    "Goal seeking",
    RYU_FRONTIER_GRAPH_BFS,
    NAV2D_NEAREST_FRONTIER_PLANNER,
    "Nearest frontier",
    "Largest frontier",
    "Utility frontier",
    "Informative frontier / IPP-lite",
    "FoV-aware directional frontier",
]


@dataclass(frozen=True)
class FrontierCandidate:
    target: tuple[float, float]
    size: int
    distance_from_robot: float
    score: float
    reason: str
    information_gain: float = 0.0
    cluster_points: tuple[tuple[float, float], ...] = ()
    cluster_resolution: float = 0.0
    heading_alignment: float = 0.0


@dataclass(frozen=True)
class ExplorationPlannerResult:
    success: bool
    target: tuple[float, float] | None
    reason: str
    candidates: tuple[FrontierCandidate, ...] = ()


@dataclass(frozen=True)
class _InternalCandidate:
    target_cell: tuple[int, int]
    target: tuple[float, float]
    size: int
    kind: str
    cluster_cells: tuple[tuple[int, int], ...] = ()


def _as_point(point) -> tuple[float, float]:
    return (float(point[0]), float(point[1]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _angle_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0]))


def _normalize_targets(raw_targets) -> list[tuple[float, float]]:
    targets: list[tuple[float, float]] = []

    if not raw_targets:
        return targets

    for item in raw_targets:
        if item is None:
            continue
        try:
            targets.append(_as_point(item))
        except (TypeError, ValueError, IndexError):
            continue

    return targets


def _normalize_dynamic_obstacles(raw_obstacles) -> list[tuple[float, float, float]]:
    obstacles: list[tuple[float, float, float]] = []

    if not raw_obstacles:
        return obstacles

    for item in raw_obstacles:
        if item is None:
            continue

        try:
            if isinstance(item, dict):
                x = float(item.get("x", 0.0))
                y = float(item.get("y", 0.0))
                radius = float(item.get("radius", item.get("r", 0.0)))
            else:
                values = list(item)
                if len(values) < 2:
                    continue
                x = float(values[0])
                y = float(values[1])
                radius = float(values[2]) if len(values) >= 3 else 0.0
        except (TypeError, ValueError):
            continue

        obstacles.append((x, y, max(radius, 0.0)))

    return obstacles


def _belief_from_kwargs(kwargs: dict[str, Any]) -> BeliefMap:
    existing = kwargs.get("belief_map")

    if isinstance(existing, BeliefMap):
        return existing

    bounds = kwargs["bounds"]
    resolution = float(kwargs["resolution"])
    robot_count = int(kwargs.get("robot_count", kwargs.get("num_robots", 1)))

    belief = BeliefMap(
        bounds=bounds,
        resolution=resolution,
        robot_count=robot_count,
    )

    for point in kwargs.get("explored_points", []) or []:
        cell = belief.world_to_cell(_as_point(point))
        if cell is not None:
            belief.mark_free_cell(cell)

    for point in kwargs.get("mapped_obstacle_points", []) or []:
        cell = belief.world_to_cell(_as_point(point))
        if cell is not None:
            belief.mark_occupied_cell(cell)

    robot_xy = kwargs.get("robot_xy")
    if robot_xy is not None:
        belief.force_free_point(_as_point(robot_xy))

    return belief


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))


def _neighbors8(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return (
        (r + 1, c),
        (r - 1, c),
        (r, c + 1),
        (r, c - 1),
        (r + 1, c + 1),
        (r + 1, c - 1),
        (r - 1, c + 1),
        (r - 1, c - 1),
    )


def _valid_cell(belief: BeliefMap, cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < belief.height and 0 <= c < belief.width


def is_frontier_cell(belief: BeliefMap, cell: tuple[int, int]) -> bool:
    """The project's one definition of "frontier" cell: FREE/observed with
    at least one UNKNOWN 4-neighbor.

    A single O(1) local check -- not a map scan -- so it is cheap enough to
    call once per tick to revalidate an already-assigned target (see
    ExplorationBehavior.update()'s active-target staleness check and
    _current_candidate() below), unlike _frontier_cells(), which scans the
    whole grid to build the full candidate set from scratch.
    """
    if not _valid_cell(belief, cell):
        return False
    row, col = cell
    if int(belief.grid[row, col]) != FREE:
        return False

    for neighbor in _neighbors4(cell):
        if not _valid_cell(belief, neighbor):
            continue
        nr, nc = neighbor
        if int(belief.grid[nr, nc]) == UNKNOWN:
            return True

    return False


def _frontier_cells(belief: BeliefMap) -> set[tuple[int, int]]:
    frontiers: set[tuple[int, int]] = set()

    for row in range(belief.height):
        for col in range(belief.width):
            cell = (row, col)
            if is_frontier_cell(belief, cell):
                frontiers.add(cell)

    return frontiers


def detect_frontier_cells(belief: BeliefMap) -> set[tuple[int, int]]:
    """Public detector-stage API used before explicit clustering."""
    return _frontier_cells(belief)


def _cluster_frontiers(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    remaining = set(cells)
    clusters: list[list[tuple[int, int]]] = []

    while remaining:
        start = remaining.pop()
        queue: deque[tuple[int, int]] = deque([start])
        cluster = [start]

        while queue:
            cell = queue.popleft()

            for neighbor in _neighbors8(cell):
                if neighbor not in remaining:
                    continue

                remaining.remove(neighbor)
                queue.append(neighbor)
                cluster.append(neighbor)

        clusters.append(cluster)

    return clusters


def _cluster_frontiers4(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Four-connected frontier components for viewpoint-sensitive planners.

    Diagonal contact often joins two different doorway/wall faces under
    eight-connectivity.  FoV should evaluate those openings independently.
    """
    remaining = set(cells)
    clusters: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        queue: deque[tuple[int, int]] = deque([start])
        cluster = [start]
        while queue:
            cell = queue.popleft()
            for neighbor in _neighbors4(cell):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
                cluster.append(neighbor)
        clusters.append(cluster)
    return clusters


def _candidate_from_cluster(belief: BeliefMap, cluster: list[tuple[int, int]]) -> _InternalCandidate | None:
    if not cluster:
        return None

    points = [belief.cell_to_world(cell) for cell in cluster]
    centroid = (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )

    target_cell = min(cluster, key=lambda cell: _distance(belief.cell_to_world(cell), centroid))
    target = belief.cell_to_world(target_cell)

    return _InternalCandidate(
        target_cell=target_cell,
        target=target,
        size=len(cluster),
        kind="frontier",
        cluster_cells=tuple(cluster),
    )


def _target_near_reserved(
    target: tuple[float, float],
    reserved: list[tuple[float, float]],
    radius: float,
) -> bool:
    return any(_distance(target, item) <= radius for item in reserved)


def _target_near_dynamic_obstacle(
    target: tuple[float, float],
    obstacles: list[tuple[float, float, float]],
    *,
    robot_radius: float,
    margin: float,
) -> bool:
    for ox, oy, radius in obstacles:
        if _distance(target, (ox, oy)) <= robot_radius + radius + margin:
            return True
    return False


def _frontier_candidates(
    *,
    belief: BeliefMap,
    reserved_targets: list[tuple[float, float]],
    dynamic_obstacles: list[tuple[float, float, float]],
    target_exclusion_radius: float,
    robot_radius: float,
    dynamic_obstacle_margin: float,
    viewpoints_per_cluster: int = 1,
    robot_xy: tuple[float, float] | None = None,
    robot_heading: float = 0.0,
    clusters: list[list[tuple[int, int]]] | None = None,
) -> list[_InternalCandidate]:
    candidates: list[_InternalCandidate] = []

    if clusters is None:
        raise ValueError(
            "frontier candidates require explicit clusters; no default clustering "
            "algorithm is available"
        )
    frontier_clusters = clusters

    for cluster in frontier_clusters:
        candidate = _candidate_from_cluster(belief, cluster)
        if candidate is None:
            continue

        viewpoints = [candidate]
        if viewpoints_per_cluster > 1 and robot_xy is not None:
            viewpoints = _sample_frontier_viewpoints(
                candidate,
                belief=belief,
                robot_xy=robot_xy,
                robot_heading=robot_heading,
                limit=viewpoints_per_cluster,
            )

        # Filter individual viewpoints, not the whole connected component.
        # A recently failed cell (or a nearby dynamic obstacle) must not make
        # every other safe point on a long frontier disappear.
        for viewpoint in viewpoints:
            if _target_near_reserved(
                viewpoint.target,
                reserved_targets,
                target_exclusion_radius,
            ):
                continue

            if _target_near_dynamic_obstacle(
                viewpoint.target,
                dynamic_obstacles,
                robot_radius=robot_radius,
                margin=dynamic_obstacle_margin,
            ):
                continue

            candidates.append(viewpoint)

    return candidates


def _bresenham(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b

    cells: list[tuple[int, int]] = []

    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    step_r = 1 if r0 < r1 else -1
    step_c = 1 if c0 < c1 else -1

    r, c = r0, c0

    if dc > dr:
        error = dc / 2.0
        while c != c1:
            cells.append((r, c))
            error -= dr
            if error < 0:
                r += step_r
                error += dc
            c += step_c
    else:
        error = dr / 2.0
        while r != r1:
            cells.append((r, c))
            error -= dc
            if error < 0:
                c += step_c
                error += dr
            r += step_r

    cells.append((r1, c1))
    return cells


def _line_of_sight_clear(belief: BeliefMap, a: tuple[int, int], b: tuple[int, int]) -> bool:
    cells = _bresenham(a, b)

    for cell in cells[1:-1]:
        if not _valid_cell(belief, cell):
            return False
        r, c = cell
        if int(belief.grid[r, c]) == OCCUPIED:
            return False

    return True


def _fov_cells(
    *,
    belief: BeliefMap,
    position: tuple[float, float],
    heading: float,
    sensor_range: float,
    fov_angle: float,
    use_occlusion: bool,
) -> set[tuple[int, int]]:
    origin = belief.world_to_cell(position, clamp=True)
    if origin is None:
        return set()

    radius_cells = max(1, int(math.ceil(float(sensor_range) / belief.resolution)))
    fov_angle = max(1e-6, min(float(fov_angle), 2.0 * math.pi))
    out: set[tuple[int, int]] = set()

    r0, c0 = origin

    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            cell = (r0 + dr, c0 + dc)

            if not _valid_cell(belief, cell):
                continue

            world = belief.cell_to_world(cell)
            dx = world[0] - position[0]
            dy = world[1] - position[1]
            d = math.hypot(dx, dy)

            if d > sensor_range:
                continue

            if d > 1e-9:
                bearing = math.atan2(dy, dx)
                if abs(_wrap_angle(bearing - heading)) > 0.5 * fov_angle:
                    continue

            if use_occlusion and not _line_of_sight_clear(belief, origin, cell):
                continue

            out.add(cell)

    return out


def _path_heading(
    belief: BeliefMap,
    path: list[GridCell],
    index: int,
    fallback: float,
) -> float:
    if len(path) <= 1:
        return fallback

    if index < len(path) - 1:
        a = belief.cell_to_world((path[index].row, path[index].col))
        b = belief.cell_to_world((path[index + 1].row, path[index + 1].col))
        return _angle_between(a, b)

    a = belief.cell_to_world((path[index - 1].row, path[index - 1].col))
    b = belief.cell_to_world((path[index].row, path[index].col))
    return _angle_between(a, b)


def _swept_fov(
    *,
    belief: BeliefMap,
    path: list[GridCell],
    robot_heading: float,
    sensor_range: float,
    fov_angle: float,
    stride_cells: int,
    use_occlusion: bool,
) -> set[tuple[int, int]]:
    if not path:
        return set()

    stride = max(1, int(stride_cells))
    indices = list(range(0, len(path), stride))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)

    swept: set[tuple[int, int]] = set()

    for index in indices:
        cell = path[index]
        position = belief.cell_to_world((cell.row, cell.col))
        heading = _path_heading(belief, path, index, robot_heading)
        swept.update(
            _fov_cells(
                belief=belief,
                position=position,
                heading=heading,
                sensor_range=sensor_range,
                fov_angle=fov_angle,
                use_occlusion=use_occlusion,
            )
        )

    return swept


def _forward_candidate(
    *,
    belief: BeliefMap,
    robot_xy: tuple[float, float],
    robot_heading: float,
    max_forward_distance: float,
) -> _InternalCandidate | None:
    step = max(belief.resolution, 1e-6)
    distance_m = step
    last_cell = None

    while distance_m <= max(float(max_forward_distance), step) + 1e-9:
        point = (
            robot_xy[0] + distance_m * math.cos(robot_heading),
            robot_xy[1] + distance_m * math.sin(robot_heading),
        )
        cell = belief.world_to_cell(point)
        if cell is None:
            break

        r, c = cell
        if int(belief.grid[r, c]) == OCCUPIED:
            break

        last_cell = cell
        distance_m += step

    if last_cell is None:
        return None

    target = belief.cell_to_world(last_cell)
    return _InternalCandidate(
        target_cell=last_cell,
        target=target,
        size=1,
        kind="forward",
        cluster_cells=(last_cell,),
    )


def _current_candidate(
    belief: BeliefMap,
    current_target,
) -> _InternalCandidate | None:
    if current_target is None:
        return None

    try:
        point = _as_point(current_target)
    except (TypeError, ValueError, IndexError):
        return None

    cell = belief.world_to_cell(point, clamp=True)
    if cell is None:
        return None

    r, c = cell
    if int(belief.grid[r, c]) == OCCUPIED:
        return None

    # A target only remains eligible for hysteresis "keep current target"
    # reuse while it is still an actual frontier cell (see is_frontier_
    # cell()'s docstring for the definition). Without this, once every
    # UNKNOWN neighbor around a previously selected target has since been
    # observed (by this robot or a teammate), it keeps being treated as a
    # live "current" candidate purely because it is not OCCUPIED -- and
    # once it is the only candidate left (e.g. the rest of the map is now
    # fully explored too), select_goal() below still picks it as "best",
    # with a reason string ("selected best FoV-aware target") that reads
    # like a fresh, informed choice instead of a zero-information repeat
    # of a dead cell. This is a single local neighbor check, not a fresh
    # full-map frontier scan.
    if not is_frontier_cell(belief, cell):
        return None

    target = belief.cell_to_world(cell)
    return _InternalCandidate(
        target_cell=cell,
        target=target,
        size=1,
        kind="current",
        cluster_cells=(cell,),
    )


def _sample_frontier_viewpoints(
    candidate: _InternalCandidate,
    *,
    belief: BeliefMap,
    robot_xy: tuple[float, float],
    robot_heading: float,
    limit: int,
) -> list[_InternalCandidate]:
    """Expose several useful viewpoints from one connected frontier.

    A frontier surrounding a newly-observed patch is commonly one large
    connected component.  Reducing that component to only the cell nearest
    its centroid makes the directional scorer ineffective: it never gets to
    compare the forward-facing, nearby, and opposite sides of the component.

    Keep the old representative for continuity, then add the nearest and most
    heading-aligned cells.  Remaining slots use farthest-point sampling so a
    long wall/ring is represented across its extent rather than by adjacent
    cells.  The returned viewpoints all retain the component's full size for
    scoring; only their route/FoV geometry differs.
    """
    if candidate.kind != "frontier" or limit <= 1 or not candidate.cluster_cells:
        return [candidate]

    cells = sorted(set(candidate.cluster_cells))
    selected: list[tuple[int, int]] = []

    def add(cell: tuple[int, int]) -> None:
        if cell not in selected:
            selected.append(cell)

    add(candidate.target_cell)

    nearest = min(
        cells,
        key=lambda cell: (
            _distance(robot_xy, belief.cell_to_world(cell)),
            cell[0],
            cell[1],
        ),
    )
    add(nearest)

    aligned = min(
        cells,
        key=lambda cell: (
            abs(_wrap_angle(_angle_between(robot_xy, belief.cell_to_world(cell)) - robot_heading)),
            _distance(robot_xy, belief.cell_to_world(cell)),
            cell[0],
            cell[1],
        ),
    )
    add(aligned)

    while len(selected) < min(max(1, int(limit)), len(cells)):
        remaining = [cell for cell in cells if cell not in selected]
        if not remaining:
            break

        # Grid distance is sufficient here and avoids repeated world
        # conversions.  Deterministic row/column tie-breaks make target
        # choice reproducible across runs.
        farthest = min(
            remaining,
            key=lambda cell: (
                -min(
                    (cell[0] - chosen[0]) ** 2 + (cell[1] - chosen[1]) ** 2
                    for chosen in selected
                ),
                cell[0],
                cell[1],
            ),
        )
        add(farthest)

    return [
        _InternalCandidate(
            target_cell=cell,
            target=belief.cell_to_world(cell),
            size=candidate.size,
            kind="frontier",
            cluster_cells=candidate.cluster_cells,
        )
        for cell in selected[: max(1, int(limit))]
    ]


def _grid_diagonal(belief: BeliefMap) -> float:
    return max(1e-6, math.hypot(belief.x_max - belief.x_min, belief.y_max - belief.y_min))


def _turn_cost(
    belief: BeliefMap,
    path: list[GridCell],
    robot_heading: float,
) -> float:
    if len(path) <= 1:
        return 0.0

    headings: list[float] = []

    for a, b in zip(path[:-1], path[1:]):
        pa = belief.cell_to_world((a.row, a.col))
        pb = belief.cell_to_world((b.row, b.col))
        headings.append(_angle_between(pa, pb))

    total = abs(_wrap_angle(headings[0] - robot_heading))

    for a, b in zip(headings[:-1], headings[1:]):
        total += abs(_wrap_angle(b - a))

    return total


def _alignment(
    belief: BeliefMap,
    path: list[GridCell],
    robot_heading: float,
) -> float:
    if len(path) <= 1:
        return 1.0

    a = belief.cell_to_world((path[0].row, path[0].col))
    b = belief.cell_to_world((path[1].row, path[1].col))
    first = _angle_between(a, b)

    return math.cos(_wrap_angle(first - robot_heading))


def _seen_penalty(belief: BeliefMap, cells: Iterable[tuple[int, int]], saturation: float) -> float:
    return belief.average_seen_penalty(cells, saturation=saturation)


def _score_candidate(
    *,
    candidate: _InternalCandidate,
    belief: BeliefMap,
    planning_grid,
    robot_xy: tuple[float, float],
    robot_heading: float,
    current_target: tuple[float, float] | None,
    reserved_targets: list[tuple[float, float]],
    dynamic_obstacles: list[tuple[float, float, float]],
    sensor_range: float,
    fov_angle: float,
    fov_stride_cells: int,
    use_occlusion: bool,
    seen_saturation: float,
    max_frontier_size: int,
    target_exclusion_radius: float,
    robot_radius: float,
    dynamic_obstacle_margin: float,
    weights: dict[str, float],
) -> FrontierCandidate | None:
    start_cell_tuple = belief.world_to_cell(robot_xy, clamp=True)
    if start_cell_tuple is None:
        return None

    start = GridCell(*start_cell_tuple)
    goal = GridCell(*candidate.target_cell)

    if planning_grid.in_bounds(start):
        planning_grid.set_value(start, 0)

    if planning_grid.in_bounds(goal) and planning_grid.get_value(goal) == OCCUPIED:
        return None

    planner = AStarPlanner(
        allow_diagonal=True,
        prevent_corner_cutting=True,
    )

    result = planner.plan(
        grid=planning_grid,
        start_xy=robot_xy,
        goal_xy=candidate.target,
    )

    if not result.success or not result.grid_path:
        return None

    path = result.grid_path

    swept = _swept_fov(
        belief=belief,
        path=path,
        robot_heading=robot_heading,
        sensor_range=sensor_range,
        fov_angle=fov_angle,
        stride_cells=fov_stride_cells,
        use_occlusion=use_occlusion,
    )

    if swept:
        info = float(sum(1 for r, c in swept if int(belief.grid[r, c]) == UNKNOWN))
        novelty = info / len(swept)
    else:
        info = 0.0
        novelty = 0.0

    terminal_heading = _path_heading(belief, path, len(path) - 1, robot_heading)
    terminal_fov = _fov_cells(
        belief=belief,
        position=candidate.target,
        heading=terminal_heading,
        sensor_range=sensor_range,
        fov_angle=fov_angle,
        use_occlusion=use_occlusion,
    )
    terminal_info = float(
        sum(1 for r, c in terminal_fov if int(belief.grid[r, c]) == UNKNOWN)
    )
    terminal_novelty = terminal_info / len(terminal_fov) if terminal_fov else 0.0

    # Preserve the useful density signal, but do not dilute every distant
    # frontier merely because its route crosses a long known corridor.  A
    # full terminal FoV's worth of newly observed cells is already enough to
    # saturate the absolute-gain term.
    gain_norm = min(1.0, info / max(1, len(terminal_fov)))
    information_utility = 0.40 * novelty + 0.40 * terminal_novelty + 0.20 * gain_norm

    path_cells = [(cell.row, cell.col) for cell in path]
    fov_repeat = _seen_penalty(belief, swept, seen_saturation)
    path_repeat = _seen_penalty(belief, path_cells, seen_saturation)

    length_norm = float(result.total_cost) / _grid_diagonal(belief)
    turn_norm = min(1.0, _turn_cost(belief, path, robot_heading) / math.pi)
    align = _alignment(belief, path, robot_heading)
    direct_distance = _distance(robot_xy, candidate.target)
    route_efficiency = min(1.0, direct_distance / max(float(result.total_cost), 1e-9))
    detour_penalty = 1.0 - route_efficiency
    target_alignment = math.cos(
        _wrap_angle(_angle_between(robot_xy, candidate.target) - robot_heading)
    ) if direct_distance > 1e-9 else 1.0
    backtrack_penalty = max(0.0, -target_alignment)

    frontier_norm = (
        math.log1p(candidate.size) / math.log1p(max(1, max_frontier_size))
    )

    switch_penalty = 0.0
    if current_target is None or _distance(current_target, candidate.target) > belief.resolution:
        switch_penalty = 1.0

    multi_penalty = 0.0
    sigma = max(float(target_exclusion_radius), 1e-6)
    for target in reserved_targets:
        d = _distance(candidate.target, target)
        multi_penalty += math.exp(-(d * d) / (sigma * sigma))

    for ox, oy, radius in dynamic_obstacles:
        safe = max(1e-6, robot_radius + radius + dynamic_obstacle_margin)
        d = _distance(candidate.target, (ox, oy))
        multi_penalty += math.exp(-(d * d) / (safe * safe))

    score = (
        weights["information"] * information_utility
        + weights["frontier"] * frontier_norm
        + weights["alignment"] * align
        - weights["length"] * length_norm
        - weights["fov_repetition"] * fov_repeat
        - weights["path_repetition"] * path_repeat
        - weights["turn"] * turn_norm
        - weights["detour"] * detour_penalty
        - weights["backtrack"] * backtrack_penalty
        - weights["switch"] * switch_penalty
        - weights["multi_robot"] * multi_penalty
    )

    reason = (
        f"kind={candidate.kind}, size={candidate.size}, "
        f"info={info:.0f}, novelty={novelty:.6f}, terminal_info={terminal_info:.0f}, "
        f"info_utility={information_utility:.6f}, "
        f"frontier_norm={frontier_norm:.6f}, "
        f"fov_repeat={fov_repeat:.6f}, path_repeat={path_repeat:.6f}, "
        f"length={result.total_cost:.6f}, length_norm={length_norm:.6f}, turn={turn_norm:.6f}, "
        f"align={align:.6f}, detour={detour_penalty:.6f}, "
        f"backtrack={backtrack_penalty:.6f}, switch={switch_penalty:.0f}, "
        f"multi={multi_penalty:.6f}, score={score:.6f}"
    )

    return FrontierCandidate(
        target=candidate.target,
        size=candidate.size,
        distance_from_robot=_distance(robot_xy, candidate.target),
        score=float(score),
        reason=reason,
        information_gain=info,
        cluster_points=tuple(belief.cell_to_world(cell) for cell in candidate.cluster_cells),
        cluster_resolution=float(belief.resolution),
        heading_alignment=float(align),
    )


def _prefer_forward_continuation(
    candidates: list[FrontierCandidate],
    *,
    score_margin: float,
    min_alignment: float,
) -> tuple[FrontierCandidate, bool]:
    """Avoid an unnecessary U-turn when a useful forward option is close."""
    best = max(candidates, key=lambda item: item.score)
    forward = [
        item for item in candidates
        if item.heading_alignment >= min_alignment and item.information_gain > 0.0
    ]
    if best.heading_alignment < 0.0 and forward:
        best_forward = max(forward, key=lambda item: item.score)
        if best_forward.score >= best.score - score_margin:
            return best_forward, True
    return best, False


class BaseExplorationPlanner:
    name = "Base"
    uses_frontier_clustering = False

    def select_goal(self, **kwargs) -> ExplorationPlannerResult:
        raise NotImplementedError


class GoalSeekingPlanner(BaseExplorationPlanner):
    name = "Goal seeking"

    def select_goal(self, **kwargs) -> ExplorationPlannerResult:
        final_goal_xy = kwargs.get("final_goal_xy")
        if final_goal_xy is None:
            return ExplorationPlannerResult(False, None, "Goal seeking requires final_goal_xy")
        return ExplorationPlannerResult(True, _as_point(final_goal_xy), "using final mission goal")


class Nav2DNearestFrontierPlanner(BaseExplorationPlanner):
    """Faithful native port of Nav2D's ``NearestFrontierPlanner``.

    Nav2D does not rank frontier clusters by straight-line distance.  It
    expands a four-connected, unit-cost wavefront through *known free* cells
    and selects the first frontier reached.  Consequently the selected target
    is nearest by traversable grid distance, including detours around walls.

    The small exclusion/reachability checks are host integration guards: they
    prevent a just-reached/failed cell from being returned forever and ensure
    the target is accepted by the simulator's real navigation costmap.  They
    do not alter the wavefront ordering among valid candidates.
    """

    name = NAV2D_NEAREST_FRONTIER_PLANNER

    @staticmethod
    def _nearest_free_start(
        belief: BeliefMap,
        robot_xy: tuple[float, float],
    ) -> tuple[int, int] | None:
        start = belief.world_to_cell(robot_xy, clamp=True)
        if start is not None and int(belief.grid[start[0], start[1]]) == FREE:
            return start

        free_cells = np.argwhere(belief.grid == FREE)
        if free_cells.size == 0:
            return None

        if start is None:
            start = (belief.height // 2, belief.width // 2)
        best = min(
            ((int(row), int(col)) for row, col in free_cells),
            key=lambda cell: (
                abs(cell[0] - start[0]) + abs(cell[1] - start[1]),
                cell[0],
                cell[1],
            ),
        )
        return best

    def select_goal(self, **kwargs) -> ExplorationPlannerResult:
        belief = _belief_from_kwargs(kwargs)
        robot_xy = _as_point(kwargs["robot_xy"])
        start = self._nearest_free_start(belief, robot_xy)
        if start is None:
            return ExplorationPlannerResult(
                False,
                None,
                f"{self.name}: map contains no known free start cell",
                (),
            )

        excluded = _normalize_targets(kwargs.get("excluded_targets"))
        exclusion_radius = max(0.0, float(kwargs.get("target_exclusion_radius", 0.0)))
        dynamic = _normalize_dynamic_obstacles(kwargs.get("dynamic_obstacles"))
        robot_radius = max(0.0, float(kwargs.get("robot_radius", 0.0)))
        dynamic_margin = max(0.0, float(kwargs.get("dynamic_obstacle_margin", 0.25)))
        is_candidate_reachable = kwargs.get("is_candidate_reachable")

        queue: deque[tuple[int, int]] = deque([start])
        distance_steps = {start: 0}
        checked = 0
        rejected = 0
        candidates: list[FrontierCandidate] = []
        probe_candidate: FrontierCandidate | None = None

        while queue:
            row, col = queue.popleft()
            cell = (row, col)
            checked += 1

            if is_frontier_cell(belief, cell):
                target = belief.cell_to_world(cell)
                blocked = any(_distance(target, item) <= exclusion_radius for item in excluded)
                blocked = blocked or _target_near_dynamic_obstacle(
                    target,
                    dynamic,
                    robot_radius=robot_radius,
                    margin=dynamic_margin,
                )

                reachable = True
                if not blocked and callable(is_candidate_reachable):
                    try:
                        reachable = bool(is_candidate_reachable(target))
                    except Exception:
                        # Match the other planners: a diagnostic callback must
                        # never crash exploration; unknown means assume valid.
                        reachable = True

                path_distance = distance_steps[cell] * belief.resolution
                candidate = FrontierCandidate(
                    target=target,
                    size=1,
                    distance_from_robot=path_distance,
                    score=-path_distance,
                    information_gain=float(
                        sum(
                            1
                            for neighbor in _neighbors8(cell)
                            if _valid_cell(belief, neighbor)
                            and int(belief.grid[neighbor[0], neighbor[1]]) == UNKNOWN
                        )
                    ),
                    reason=f"wavefront_distance={path_distance:.2f} m",
                )
                candidates.append(candidate)
                if not blocked and probe_candidate is None:
                    probe_candidate = candidate
                if not blocked and reachable:
                    return ExplorationPlannerResult(
                        True,
                        target,
                        (
                            f"{self.name}: first reachable frontier in four-connected "
                            f"wavefront; distance={path_distance:.2f} m, "
                            f"checked={checked}, rejected={rejected}"
                        ),
                        tuple(candidates),
                    )
                rejected += 1

            # Nav2D's single-robot implementation expands only the four
            # cardinal neighbors and only through known-free cells.
            for neighbor in _neighbors4(cell):
                if neighbor in distance_steps or not _valid_cell(belief, neighbor):
                    continue
                nr, nc = neighbor
                if int(belief.grid[nr, nc]) != FREE:
                    continue
                distance_steps[neighbor] = distance_steps[cell] + 1
                queue.append(neighbor)

        if probe_candidate is not None:
            return ExplorationPlannerResult(
                True,
                probe_candidate.target,
                (
                    f"{self.name}: all frontier cells failed the navigation reachability "
                    f"precheck; probing first wavefront candidate; checked={checked}, "
                    f"rejected={rejected}"
                ),
                tuple(candidates),
            )

        return ExplorationPlannerResult(
            False,
            None,
            (
                f"{self.name}: no reachable frontier after checking {checked} "
                f"known-free cell(s); rejected={rejected}"
            ),
            tuple(candidates),
        )


class FrontierExplorationPlanner(BaseExplorationPlanner):
    name = "Frontier"
    uses_frontier_clustering = True

    def cluster_frontiers(self, belief: BeliefMap) -> list[list[tuple[int, int]]]:
        """Legacy direct-call compatibility for archived planner tests.

        The interactive runtime never calls this method: it supplies
        ``frontier_clusters`` from the explicitly selected clustering stage.
        """
        return _cluster_frontiers(_frontier_cells(belief))

    def _clusters_for_call(
        self,
        belief: BeliefMap,
        kwargs: Mapping[str, Any],
    ) -> list[list[tuple[int, int]]]:
        if "clustering_algorithm" in kwargs:
            supplied = kwargs.get("frontier_clusters")
            if supplied is None:
                raise ValueError(
                    "frontier detector requires clusters from the selected "
                    "Clustering Algorithm"
                )
            return [list(cluster) for cluster in supplied]
        return self.cluster_frontiers(belief)

    def frontier_candidates(self, **kwargs) -> list[FrontierCandidate]:
        belief = _belief_from_kwargs(kwargs)
        robot_xy = _as_point(kwargs["robot_xy"])
        # kwargs.get(..., robot_xy) only falls back when the key is MISSING --
        # PlannerServices always passes final_goal_xy explicitly (even as
        # None, for pure-exploration runs with no configured mission goal),
        # so that default never actually applied; _as_point(None) crashed
        # instead. final_goal_xy is legitimately Optional (see
        # RobotObservation.final_goal_xy), so an explicit None must fall
        # back to robot_xy too.
        final_goal_xy = _as_point(kwargs.get("final_goal_xy") or robot_xy)
        robot_radius = float(kwargs.get("robot_radius", 0.0))
        target_exclusion_radius = float(kwargs.get("target_exclusion_radius", 1.0))
        dynamic_obstacle_margin = float(kwargs.get("dynamic_obstacle_margin", 0.25))

        reserved = _normalize_targets(kwargs.get("excluded_targets"))
        dynamic = _normalize_dynamic_obstacles(kwargs.get("dynamic_obstacles"))

        internal = _frontier_candidates(
            belief=belief,
            reserved_targets=reserved,
            dynamic_obstacles=dynamic,
            target_exclusion_radius=target_exclusion_radius,
            robot_radius=robot_radius,
            dynamic_obstacle_margin=dynamic_obstacle_margin,
            clusters=self._clusters_for_call(belief, kwargs),
        )

        candidates: list[FrontierCandidate] = []

        for item in internal:
            distance_from_robot = _distance(robot_xy, item.target)
            distance_to_goal = _distance(item.target, final_goal_xy)
            info = self.estimate_information_gain(
                belief=belief,
                target=item.target,
                sensor_range=float(kwargs.get("sensor_range", 2.5)),
            )
            score = self.score_candidate(
                size=item.size,
                distance_from_robot=distance_from_robot,
                distance_to_final_goal=distance_to_goal,
                information_gain=info,
                ipp_distance_penalty=float(kwargs.get("ipp_distance_penalty", 0.20)),
            )
            candidates.append(
                FrontierCandidate(
                    target=item.target,
                    size=item.size,
                    distance_from_robot=distance_from_robot,
                    score=score,
                    reason=(
                        f"frontier size={item.size}, info_gain={info:.1f}, "
                        f"distance={distance_from_robot:.2f}, "
                        f"goal_distance={distance_to_goal:.2f}, score={score:.2f}"
                    ),
                    information_gain=info,
                    cluster_points=tuple(belief.cell_to_world(cell) for cell in item.cluster_cells),
                    cluster_resolution=float(belief.resolution),
                )
            )

        return candidates

    def estimate_information_gain(
        self,
        *,
        belief: BeliefMap,
        target: tuple[float, float],
        sensor_range: float,
    ) -> float:
        cells = _fov_cells(
            belief=belief,
            position=target,
            heading=0.0,
            sensor_range=sensor_range,
            fov_angle=2.0 * math.pi,
            use_occlusion=False,
        )
        return float(sum(1 for r, c in cells if int(belief.grid[r, c]) == UNKNOWN))

    def score_candidate(
        self,
        *,
        size: int,
        distance_from_robot: float,
        distance_to_final_goal: float,
        information_gain: float,
        ipp_distance_penalty: float,
    ) -> float:
        return float(size)

    def choose_candidate(self, candidates: list[FrontierCandidate]) -> FrontierCandidate:
        raise NotImplementedError

    def select_goal(self, **kwargs) -> ExplorationPlannerResult:
        try:
            candidates = self.frontier_candidates(**kwargs)
        except ValueError as exc:
            return ExplorationPlannerResult(False, None, str(exc), ())

        if not candidates:
            return ExplorationPlannerResult(False, None, "no valid frontier candidates found", ())

        candidates_public = tuple(candidates)

        # Optional reachability gate: `frontier_candidates()` above only
        # consults the belief map, not the dense-obstacle-point-padded grid
        # the real single-robot navigation A* plans on, so a candidate this
        # scorer accepts can still come back "no path found" from the actual
        # planner. When the caller supplies is_candidate_reachable(xy) ->
        # bool (typically backed by that same real planning grid), reject
        # candidates it rejects here, before final selection, instead of
        # sending them to A* and failing downstream.
        is_candidate_reachable = kwargs.get("is_candidate_reachable")
        filtered_unreachable = 0

        if callable(is_candidate_reachable):
            reachable_candidates: list[FrontierCandidate] = []
            annotated_candidates: list[FrontierCandidate] = []
            for item in candidates:
                try:
                    reachability = is_candidate_reachable(item.target)
                    reachable = bool(reachability)
                    reachability_reason = str(getattr(reachability, "reason", ""))
                except Exception:
                    # A broken reachability callback must not take down
                    # exploration -- treat it as "unknown, assume reachable".
                    reachable = True
                    reachability_reason = "reachability callback failed; assumed reachable"
                annotated = replace(
                    item,
                    reason=(
                        f"{item.reason}, reachability={'reachable' if reachable else 'rejected'}, "
                        f"reachability_reason={reachability_reason or 'not reported'}"
                    ),
                )
                annotated_candidates.append(annotated)
                if reachable:
                    reachable_candidates.append(annotated)
                else:
                    filtered_unreachable += 1
            candidates_public = tuple(annotated_candidates)
        else:
            reachable_candidates = candidates

        if not reachable_candidates:
            # A precheck is advisory, not authority over exploration.  Return
            # the planner's own best candidate so the normal route pipeline
            # performs one explicit attempt; a real failure then blacklists
            # this target and the next ranked candidate is tried.
            chosen = self.choose_candidate(candidates)
            return ExplorationPlannerResult(
                True,
                chosen.target,
                (
                    f"{self.name}: all {len(candidates)} candidates failed the navigation "
                    f"reachability precheck; probing planner-ranked candidate {chosen.target}; "
                    f"generated={len(candidates)}, filtered_unreachable={filtered_unreachable}"
                ),
                candidates_public,
            )

        chosen = self.choose_candidate(reachable_candidates)
        return ExplorationPlannerResult(
            True,
            chosen.target,
            (
                f"{self.name}: {chosen.reason}; generated={len(candidates)}, "
                f"filtered_unreachable={filtered_unreachable}"
            ),
            candidates_public,
        )


class NearestFrontierPlanner(FrontierExplorationPlanner):
    name = "Nearest frontier"

    def choose_candidate(self, candidates: list[FrontierCandidate]) -> FrontierCandidate:
        return min(candidates, key=lambda item: (item.distance_from_robot, -item.size))


class RyuFrontierGraphBFSPlanner(BaseExplorationPlanner):
    """Paper-cited BFS frontier detector/selector with DBSCAN input or CCL fallback."""

    name = RYU_FRONTIER_GRAPH_BFS
    uses_frontier_clustering = True
    citation = RYU_FRONTIER_GRAPH_BFS_CITATION

    def select_goal(self, **kwargs) -> ExplorationPlannerResult:
        belief = _belief_from_kwargs(kwargs)
        robot_xy = _as_point(kwargs["robot_xy"])
        supplied_clusters = kwargs.get("frontier_clusters")
        nodes = bfs_frontier_nodes(
            belief,
            robot_xy,
            dbscan_clusters=supplied_clusters,
        )
        if not nodes:
            return ExplorationPlannerResult(
                False,
                None,
                f"{self.name}: no reachable frontier nodes; citation={self.citation}",
                (),
            )

        excluded = _normalize_targets(kwargs.get("excluded_targets"))
        exclusion_radius = float(kwargs.get("target_exclusion_radius", 0.0))
        dynamic = _normalize_dynamic_obstacles(kwargs.get("dynamic_obstacles"))
        robot_radius = float(kwargs.get("robot_radius", 0.0))
        dynamic_margin = float(kwargs.get("dynamic_obstacle_margin", 0.25))
        candidates: list[FrontierCandidate] = []
        selected: FrontierCandidate | None = None
        for node in nodes:
            target = belief.cell_to_world(node.representative)
            rejected = _target_near_reserved(target, excluded, exclusion_radius) or (
                _target_near_dynamic_obstacle(
                    target,
                    dynamic,
                    robot_radius=robot_radius,
                    margin=dynamic_margin,
                )
            )
            candidate = FrontierCandidate(
                target=target,
                size=len(node.cells),
                distance_from_robot=node.bfs_depth * float(belief.resolution),
                score=-float(node.bfs_depth),
                reason=(
                    f"BFS depth={node.bfs_depth}, frontier_size={len(node.cells)}, "
                    f"rejected={rejected}"
                ),
                cluster_points=tuple(belief.cell_to_world(cell) for cell in node.cells),
                cluster_resolution=float(belief.resolution),
            )
            candidates.append(candidate)
            if selected is None and not rejected:
                selected = candidate

        if selected is None:
            return ExplorationPlannerResult(
                False,
                None,
                f"{self.name}: all reachable BFS frontier nodes were excluded",
                tuple(candidates),
            )

        fallback_reason = str(kwargs.get("clustering_fallback_reason", "")).strip()
        stage = (
            f"8-connected CCL fallback ({fallback_reason})"
            if fallback_reason
            else "selected DBSCAN frontier nodes"
        )
        return ExplorationPlannerResult(
            True,
            selected.target,
            (
                f"{self.name}: first eligible node at BFS depth "
                f"{selected.distance_from_robot / float(belief.resolution):.0f}; "
                f"source={stage}; citation={self.citation}"
            ),
            tuple(candidates),
        )


class LargestFrontierPlanner(FrontierExplorationPlanner):
    name = "Largest frontier"

    def choose_candidate(self, candidates: list[FrontierCandidate]) -> FrontierCandidate:
        return max(candidates, key=lambda item: (item.size, -item.distance_from_robot))


class UtilityFrontierPlanner(FrontierExplorationPlanner):
    name = "Utility frontier"

    def score_candidate(
        self,
        *,
        size: int,
        distance_from_robot: float,
        distance_to_final_goal: float,
        information_gain: float,
        ipp_distance_penalty: float,
    ) -> float:
        return float(size) - 0.75 * float(distance_from_robot) - 0.15 * float(distance_to_final_goal)

    def choose_candidate(self, candidates: list[FrontierCandidate]) -> FrontierCandidate:
        return max(candidates, key=lambda item: (item.score, item.size, -item.distance_from_robot))


class InformativeFrontierPlanner(FrontierExplorationPlanner):
    name = "Informative frontier / IPP-lite"

    def score_candidate(
        self,
        *,
        size: int,
        distance_from_robot: float,
        distance_to_final_goal: float,
        information_gain: float,
        ipp_distance_penalty: float,
    ) -> float:
        return float(information_gain) - float(ipp_distance_penalty) * float(distance_from_robot)

    def choose_candidate(self, candidates: list[FrontierCandidate]) -> FrontierCandidate:
        return max(candidates, key=lambda item: (item.score, item.information_gain, item.size, -item.distance_from_robot))


class FoVAwareDirectionalFrontierPlanner(BaseExplorationPlanner):
    name = "FoV-aware directional frontier"
    uses_frontier_clustering = True

    def cluster_frontiers(self, belief: BeliefMap) -> list[list[tuple[int, int]]]:
        """Legacy direct-call compatibility for archived FoV tests only."""
        return _cluster_frontiers4(_frontier_cells(belief))

    def _clusters_for_call(
        self,
        belief: BeliefMap,
        kwargs: Mapping[str, Any],
    ) -> list[list[tuple[int, int]]]:
        if "clustering_algorithm" in kwargs:
            supplied = kwargs.get("frontier_clusters")
            if supplied is None:
                raise ValueError(
                    "FoV-aware directional frontier requires clusters from the "
                    "selected Clustering Algorithm"
                )
            return [list(cluster) for cluster in supplied]
        return self.cluster_frontiers(belief)

    def select_goal(self, **kwargs) -> ExplorationPlannerResult:
        belief = _belief_from_kwargs(kwargs)
        robot_xy = _as_point(kwargs["robot_xy"])
        robot_heading = float(kwargs.get("robot_heading", kwargs.get("heading", kwargs.get("theta", 0.0))))
        robot_radius = float(kwargs.get("robot_radius", 0.0))
        safety_margin = float(kwargs.get("safety_margin", 0.0))
        sensor_range = float(kwargs.get("sensor_range", 2.5))
        fov_angle = float(kwargs.get("fov_angle", math.radians(120.0)))
        fov_stride_cells = int(kwargs.get("fov_stride_cells", 2))
        use_occlusion = bool(kwargs.get("fov_use_occlusion", True))
        seen_saturation = float(kwargs.get("seen_saturation", 5.0))

        target_exclusion_radius = float(kwargs.get("target_exclusion_radius", 1.0))
        dynamic_obstacle_margin = float(kwargs.get("dynamic_obstacle_margin", 0.25))

        current_target = kwargs.get("current_target", kwargs.get("current_goal", kwargs.get("assigned_target", None)))
        if current_target is not None:
            try:
                current_target = _as_point(current_target)
            except (TypeError, ValueError, IndexError):
                current_target = None

        reserved = _normalize_targets(kwargs.get("excluded_targets"))
        dynamic = _normalize_dynamic_obstacles(kwargs.get("dynamic_obstacles"))

        weights = {
            "information": float(kwargs.get("w_information", 3.0)),
            "frontier": float(kwargs.get("w_frontier", 0.7)),
            "alignment": float(kwargs.get("w_alignment", 1.2)),
            "length": float(kwargs.get("w_length", 1.0)),
            "fov_repetition": float(kwargs.get("w_fov_repetition", 2.2)),
            "path_repetition": float(kwargs.get("w_path_repetition", 0.8)),
            "turn": float(kwargs.get("w_turn", 1.0)),
            "detour": float(kwargs.get("w_detour", 0.45)),
            "backtrack": float(kwargs.get("w_backtrack", 0.75)),
            "switch": float(kwargs.get("w_switch", 0.6)),
            "multi_robot": float(kwargs.get("w_multi_robot", 1.2)),
        }

        hysteresis_margin = float(kwargs.get("hysteresis_margin", 0.15))
        forward_continuation_margin = float(kwargs.get("forward_continuation_margin", 1.5))
        min_forward_alignment = float(kwargs.get("min_forward_alignment", 0.5))
        min_current_information_gain = float(kwargs.get("min_current_information_gain", 1.0))
        max_candidates = max(1, int(kwargs.get("max_fov_candidates", 32)))
        viewpoints_per_cluster = max(1, int(kwargs.get("max_frontier_viewpoints_per_cluster", 5)))
        max_forward_distance = float(kwargs.get("max_forward_distance", max(sensor_range, 4.0 * belief.resolution)))

        try:
            frontier_clusters = self._clusters_for_call(belief, kwargs)
        except ValueError as exc:
            return ExplorationPlannerResult(False, None, str(exc), ())

        candidates = _frontier_candidates(
            belief=belief,
            reserved_targets=reserved,
            dynamic_obstacles=dynamic,
            target_exclusion_radius=target_exclusion_radius,
            robot_radius=robot_radius,
            dynamic_obstacle_margin=dynamic_obstacle_margin,
            viewpoints_per_cluster=viewpoints_per_cluster,
            robot_xy=robot_xy,
            robot_heading=robot_heading,
            clusters=frontier_clusters,
        )

        forward = _forward_candidate(
            belief=belief,
            robot_xy=robot_xy,
            robot_heading=robot_heading,
            max_forward_distance=max_forward_distance,
        )
        if forward is not None:
            candidates.append(forward)

        current = _current_candidate(belief, current_target)
        if current is not None:
            candidates.append(current)

        if not candidates:
            return ExplorationPlannerResult(False, None, "no frontier, forward, or current-target candidates found", ())

        candidates = self._preselect(
            candidates,
            robot_xy,
            max_candidates,
            robot_heading=robot_heading,
        )
        max_frontier_size = max((c.size for c in candidates), default=1)

        # Three-tier grid source, in priority order -- both #1 and #2 carry
        # sanitized static observed geometry, other-robot dynamic points,
        # and observed hazard (built by engine.py via the SAME
        # PlanningCostmapBuilder-backed adapter build_planner_kwargs()/
        # make_exploration_reachability_check() already use -- see
        # SimulationControllerMixin.build_planning_grid_for_robot()), never
        # just the belief map alone:
        #   1. kwargs["planning_grid"] -- an already-built grid, when a
        #      caller happens to have one on hand.
        #   2. kwargs["planning_grid_provider"] -- a Callable[[], grid],
        #      invoked HERE, at most once, only by this planner. Other
        #      planners in this module receive the same kwarg but never
        #      read it, so they never trigger this build. This keeps grid
        #      construction lazy: engine.py registers the provider once per
        #      tick (see ensure_planner_services()) but nothing actually
        #      builds a grid unless FoV scoring is genuinely reached.
        #   3. Neither supplied (e.g. direct/test callers, or callers with
        #      no live robot object to build one from): falls back to the
        #      belief-only grid this planner has always built on its own --
        #      this fallback's behavior is unchanged from before.
        # #1/#2 are copied here so this function never mutates whatever the
        # caller's grid/provider produced.
        supplied_planning_grid = kwargs.get("planning_grid")
        if supplied_planning_grid is not None:
            planning_grid = supplied_planning_grid.copy()
        else:
            planning_grid_provider = kwargs.get("planning_grid_provider")
            if callable(planning_grid_provider):
                planning_grid = planning_grid_provider().copy()
            else:
                planning_grid = belief.to_planning_grid(
                    unknown_is_traversable=True,
                    inflate_radius=max(0.0, robot_radius + safety_margin),
                )

        scored: list[FrontierCandidate] = []

        for candidate in candidates:
            item = _score_candidate(
                candidate=candidate,
                belief=belief,
                planning_grid=planning_grid.copy(),
                robot_xy=robot_xy,
                robot_heading=robot_heading,
                current_target=current_target,
                reserved_targets=reserved,
                dynamic_obstacles=dynamic,
                sensor_range=sensor_range,
                fov_angle=fov_angle,
                fov_stride_cells=fov_stride_cells,
                use_occlusion=use_occlusion,
                seen_saturation=seen_saturation,
                max_frontier_size=max_frontier_size,
                target_exclusion_radius=target_exclusion_radius,
                robot_radius=robot_radius,
                dynamic_obstacle_margin=dynamic_obstacle_margin,
                weights=weights,
            )
            if item is not None:
                scored.append(item)

        if not scored:
            return ExplorationPlannerResult(False, None, "candidate targets existed, but no candidate path was valid", ())

        candidates_public = tuple(sorted(scored, key=lambda item: item.score, reverse=True))

        # Optional reachability gate: the internal `planning_grid` above is
        # built from the belief map alone (unknown traversable, only
        # robot-radius padding). The real single-robot navigation A* also
        # inflates around dense mapped-obstacle-point samples, so a
        # candidate this scorer considers reachable can still come back
        # "no path found" from the actual planner. When the caller supplies
        # is_candidate_reachable(xy) -> bool (typically backed by the same
        # planning grid engine.py's real A* uses), candidates it rejects
        # are dropped here, before final selection, instead of being
        # requested and failing downstream.
        is_candidate_reachable = kwargs.get("is_candidate_reachable")
        filtered_unreachable = 0

        if callable(is_candidate_reachable):
            reachable_scored: list[FrontierCandidate] = []
            annotated_scored: list[FrontierCandidate] = []
            for item in scored:
                try:
                    reachability = is_candidate_reachable(item.target)
                    reachable = bool(reachability)
                    reachability_reason = str(getattr(reachability, "reason", ""))
                except Exception:
                    # A broken reachability callback must not take down
                    # exploration -- treat it as "unknown, assume reachable".
                    reachable = True
                    reachability_reason = "reachability callback failed; assumed reachable"
                annotated = replace(
                    item,
                    reason=(
                        f"{item.reason}, reachability={'reachable' if reachable else 'rejected'}, "
                        f"reachability_reason={reachability_reason or 'not reported'}"
                    ),
                )
                annotated_scored.append(annotated)
                if reachable:
                    reachable_scored.append(annotated)
                else:
                    filtered_unreachable += 1
            candidates_public = tuple(sorted(annotated_scored, key=lambda item: item.score, reverse=True))
        else:
            reachable_scored = scored

        debug_counts = (
            f"generated={len(candidates)}, excluded_recently_failed={len(reserved)}, "
            f"filtered_unreachable={filtered_unreachable}"
        )

        if not reachable_scored:
            chosen, directional_probe = _prefer_forward_continuation(
                scored,
                score_margin=forward_continuation_margin,
                min_alignment=min_forward_alignment,
            )
            return ExplorationPlannerResult(
                True,
                chosen.target,
                (
                    f"{self.name}: all {len(scored)} candidates failed the navigation "
                    f"reachability precheck; probing "
                    f"{'forward-continuation' if directional_probe else 'best FoV-ranked'} candidate; "
                    f"{chosen.reason}; {debug_counts}, selected={chosen.target}"
                ),
                candidates_public,
            )

        best, directional_override = _prefer_forward_continuation(
            reachable_scored,
            score_margin=forward_continuation_margin,
            min_alignment=min_forward_alignment,
        )
        current_scored = None

        for item in reachable_scored:
            if current_target is not None and _distance(item.target, current_target) <= belief.resolution:
                current_scored = item
                break

        if (
            current_scored is not None
            and current_scored.information_gain >= min_current_information_gain
            and best.score < current_scored.score + hysteresis_margin
        ):
            chosen = current_scored
            prefix = "kept current target by hysteresis"
        else:
            chosen = best
            prefix = (
                "selected forward-continuation target instead of avoidable backtrack"
                if directional_override else "selected best FoV-aware target"
            )

        return ExplorationPlannerResult(
            True,
            chosen.target,
            f"{self.name}: {prefix}; {chosen.reason}; {debug_counts}, selected={chosen.target}",
            candidates_public,
        )

    def _preselect(
        self,
        candidates: list[_InternalCandidate],
        robot_xy: tuple[float, float],
        max_candidates: int,
        *,
        robot_heading: float = 0.0,
    ) -> list[_InternalCandidate]:
        if len(candidates) <= max_candidates:
            return candidates

        special = [c for c in candidates if c.kind in {"current", "forward"}]
        frontiers = [c for c in candidates if c.kind == "frontier"]

        budget = max(0, max_candidates - len(special))
        if budget <= 0:
            return special[:max_candidates]

        by_size = sorted(
            frontiers,
            key=lambda c: (-c.size, _distance(robot_xy, c.target), c.target_cell),
        )
        by_distance = sorted(
            frontiers,
            key=lambda c: (_distance(robot_xy, c.target), -c.size, c.target_cell),
        )
        by_alignment = sorted(
            frontiers,
            key=lambda c: (
                abs(_wrap_angle(_angle_between(robot_xy, c.target) - robot_heading)),
                _distance(robot_xy, c.target),
                -c.size,
                c.target_cell,
            ),
        )

        chosen: list[_InternalCandidate] = []
        seen: set[tuple[int, int]] = set()

        # Interleave rankings.  Previously `by_size` always contained every
        # frontier and therefore consumed the whole budget before
        # `by_distance` was ever consulted, despite the apparent two-pool
        # implementation.  Directional candidates now receive an equal
        # opportunity as well.
        pools = (by_size, by_distance, by_alignment)
        indices = [0] * len(pools)
        while len(chosen) < budget:
            added_this_round = False
            for pool_index, pool in enumerate(pools):
                while indices[pool_index] < len(pool):
                    candidate = pool[indices[pool_index]]
                    indices[pool_index] += 1
                    if candidate.target_cell in seen:
                        continue
                    seen.add(candidate.target_cell)
                    chosen.append(candidate)
                    added_this_round = True
                    break
                if len(chosen) >= budget:
                    break
            if not added_this_round:
                break

        return special + chosen[:budget]


class ExplorationPlannerRegistry:
    def __init__(self):
        self._planners: dict[str, BaseExplorationPlanner] = {
            GoalSeekingPlanner.name: GoalSeekingPlanner(),
            RyuFrontierGraphBFSPlanner.name: RyuFrontierGraphBFSPlanner(),
            Nav2DNearestFrontierPlanner.name: Nav2DNearestFrontierPlanner(),
            NearestFrontierPlanner.name: NearestFrontierPlanner(),
            LargestFrontierPlanner.name: LargestFrontierPlanner(),
            UtilityFrontierPlanner.name: UtilityFrontierPlanner(),
            InformativeFrontierPlanner.name: InformativeFrontierPlanner(),
            FoVAwareDirectionalFrontierPlanner.name: FoVAwareDirectionalFrontierPlanner(),
        }

    def names(self) -> list[str]:
        return list(self._planners.keys())

    def get(self, name: str) -> BaseExplorationPlanner:
        return self._planners.get(name, self._planners[DEFAULT_EXPLORATION_PLANNER])

    def select_goal(self, planner_name: str, **kwargs) -> ExplorationPlannerResult:
        return self.get(planner_name).select_goal(**kwargs)


_REGISTRY = ExplorationPlannerRegistry()


def select_exploration_goal(planner_name: str, **kwargs) -> ExplorationPlannerResult:
    return _REGISTRY.select_goal(planner_name, **kwargs)


def exploration_planner_requires_clustering(planner_name: str) -> bool:
    """Whether the detector consumes clusters before ranking candidates."""
    return bool(_REGISTRY.get(str(planner_name)).uses_frontier_clustering)
