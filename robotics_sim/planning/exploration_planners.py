"""
Exploration target selection.

Responsibility split:
    exploration planner -> chooses where to go next
    path planner        -> computes how to get there
    controller          -> follows waypoints

Main planner:
    FoV-aware directional frontier

This file now receives BeliefMap directly when available. That avoids rebuilding
a separate grid convention from explored_points and mapped_obstacle_points.

Compatibility:
    Older engine code can still pass explored_points/mapped_obstacle_points with
    bounds/resolution. In that case a temporary BeliefMap is built using the
    same shared grid convention.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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


DEFAULT_EXPLORATION_PLANNER = "FoV-aware directional frontier"

EXPLORATION_PLANNER_OPTIONS = [
    "Goal seeking",
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


def _frontier_cells(belief: BeliefMap) -> set[tuple[int, int]]:
    frontiers: set[tuple[int, int]] = set()

    for row in range(belief.height):
        for col in range(belief.width):
            if int(belief.grid[row, col]) != FREE:
                continue

            cell = (row, col)

            for neighbor in _neighbors4(cell):
                if not _valid_cell(belief, neighbor):
                    continue

                nr, nc = neighbor
                if int(belief.grid[nr, nc]) == UNKNOWN:
                    frontiers.add(cell)
                    break

    return frontiers


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
) -> list[_InternalCandidate]:
    candidates: list[_InternalCandidate] = []

    for cluster in _cluster_frontiers(_frontier_cells(belief)):
        candidate = _candidate_from_cluster(belief, cluster)
        if candidate is None:
            continue

        if _target_near_reserved(candidate.target, reserved_targets, target_exclusion_radius):
            continue

        if _target_near_dynamic_obstacle(
            candidate.target,
            dynamic_obstacles,
            robot_radius=robot_radius,
            margin=dynamic_obstacle_margin,
        ):
            continue

        candidates.append(candidate)

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

    target = belief.cell_to_world(cell)
    return _InternalCandidate(
        target_cell=cell,
        target=target,
        size=1,
        kind="current",
        cluster_cells=(cell,),
    )


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

    path_cells = [(cell.row, cell.col) for cell in path]
    fov_repeat = _seen_penalty(belief, swept, seen_saturation)
    path_repeat = _seen_penalty(belief, path_cells, seen_saturation)

    length_norm = float(result.total_cost) / _grid_diagonal(belief)
    turn_norm = min(1.0, _turn_cost(belief, path, robot_heading) / math.pi)
    align = _alignment(belief, path, robot_heading)

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
        weights["information"] * novelty
        + weights["frontier"] * frontier_norm
        + weights["alignment"] * align
        - weights["length"] * length_norm
        - weights["fov_repetition"] * fov_repeat
        - weights["path_repetition"] * path_repeat
        - weights["turn"] * turn_norm
        - weights["switch"] * switch_penalty
        - weights["multi_robot"] * multi_penalty
    )

    reason = (
        f"kind={candidate.kind}, size={candidate.size}, "
        f"info={info:.0f}, novelty={novelty:.2f}, "
        f"fov_repeat={fov_repeat:.2f}, path_repeat={path_repeat:.2f}, "
        f"length={result.total_cost:.2f}, turn={turn_norm:.2f}, "
        f"align={align:.2f}, switch={switch_penalty:.0f}, "
        f"multi={multi_penalty:.2f}, score={score:.3f}"
    )

    return FrontierCandidate(
        target=candidate.target,
        size=candidate.size,
        distance_from_robot=_distance(robot_xy, candidate.target),
        score=float(score),
        reason=reason,
        information_gain=info,
    )


class BaseExplorationPlanner:
    name = "Base"

    def select_goal(self, **kwargs) -> ExplorationPlannerResult:
        raise NotImplementedError


class GoalSeekingPlanner(BaseExplorationPlanner):
    name = "Goal seeking"

    def select_goal(self, **kwargs) -> ExplorationPlannerResult:
        final_goal_xy = kwargs.get("final_goal_xy")
        if final_goal_xy is None:
            return ExplorationPlannerResult(False, None, "Goal seeking requires final_goal_xy")
        return ExplorationPlannerResult(True, _as_point(final_goal_xy), "using final mission goal")


class FrontierExplorationPlanner(BaseExplorationPlanner):
    name = "Frontier"

    def frontier_candidates(self, **kwargs) -> list[FrontierCandidate]:
        belief = _belief_from_kwargs(kwargs)
        robot_xy = _as_point(kwargs["robot_xy"])
        final_goal_xy = _as_point(kwargs.get("final_goal_xy", robot_xy))
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
                        f"distance={distance_from_robot:.2f}, score={score:.2f}"
                    ),
                    information_gain=info,
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
        candidates = self.frontier_candidates(**kwargs)

        if not candidates:
            return ExplorationPlannerResult(False, None, "no valid frontier candidates found", ())

        chosen = self.choose_candidate(candidates)
        return ExplorationPlannerResult(True, chosen.target, f"{self.name}: {chosen.reason}", tuple(candidates))


class NearestFrontierPlanner(FrontierExplorationPlanner):
    name = "Nearest frontier"

    def choose_candidate(self, candidates: list[FrontierCandidate]) -> FrontierCandidate:
        return min(candidates, key=lambda item: (item.distance_from_robot, -item.size))


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
            "switch": float(kwargs.get("w_switch", 0.6)),
            "multi_robot": float(kwargs.get("w_multi_robot", 1.2)),
        }

        hysteresis_margin = float(kwargs.get("hysteresis_margin", 0.15))
        min_current_information_gain = float(kwargs.get("min_current_information_gain", 1.0))
        max_candidates = max(1, int(kwargs.get("max_fov_candidates", 32)))
        max_forward_distance = float(kwargs.get("max_forward_distance", max(sensor_range, 4.0 * belief.resolution)))

        candidates = _frontier_candidates(
            belief=belief,
            reserved_targets=reserved,
            dynamic_obstacles=dynamic,
            target_exclusion_radius=target_exclusion_radius,
            robot_radius=robot_radius,
            dynamic_obstacle_margin=dynamic_obstacle_margin,
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

        candidates = self._preselect(candidates, robot_xy, max_candidates)
        max_frontier_size = max((c.size for c in candidates), default=1)

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

        best = max(scored, key=lambda item: item.score)
        current_scored = None

        for item in scored:
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
            prefix = "selected best FoV-aware target"

        candidates_public = tuple(sorted(scored, key=lambda item: item.score, reverse=True))

        return ExplorationPlannerResult(
            True,
            chosen.target,
            f"{self.name}: {prefix}; {chosen.reason}",
            candidates_public,
        )

    def _preselect(
        self,
        candidates: list[_InternalCandidate],
        robot_xy: tuple[float, float],
        max_candidates: int,
    ) -> list[_InternalCandidate]:
        if len(candidates) <= max_candidates:
            return candidates

        special = [c for c in candidates if c.kind in {"current", "forward"}]
        frontiers = [c for c in candidates if c.kind == "frontier"]

        budget = max(0, max_candidates - len(special))
        if budget <= 0:
            return special[:max_candidates]

        by_size = sorted(frontiers, key=lambda c: c.size, reverse=True)
        by_distance = sorted(frontiers, key=lambda c: _distance(robot_xy, c.target))

        chosen: list[_InternalCandidate] = []
        seen: set[tuple[int, int]] = set()

        for pool in (by_size, by_distance):
            for candidate in pool:
                if len(chosen) >= budget:
                    break
                if candidate.target_cell in seen:
                    continue
                seen.add(candidate.target_cell)
                chosen.append(candidate)

        return special + chosen[:budget]


class ExplorationPlannerRegistry:
    def __init__(self):
        self._planners: dict[str, BaseExplorationPlanner] = {
            GoalSeekingPlanner.name: GoalSeekingPlanner(),
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
