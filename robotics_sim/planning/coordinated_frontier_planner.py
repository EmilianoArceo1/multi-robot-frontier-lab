"""
Coordinated multi-robot frontier planner.

This module replaces the previous two-step behavior:

    per-robot exploration planner -> coordinator tries to repair conflicts

with one coordinated policy:

    shared map -> global frontiers/viewpoints -> robot-viewpoint scoring ->
    team assignment.

The key design rule is that frontier selection is a team decision.  Robots may
cross already known free space, but they should not waste sensor footprints on
areas that a teammate has already mapped or is already moving to inspect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from robotics_sim.planning.exploration_planners import FrontierCandidate


@dataclass(frozen=True)
class CoordinatedFrontierAssignment:
    target: tuple[float, float]
    score: float
    information_gain: float
    distance: float
    other_map_ratio: float
    route_overlap_ratio: float
    reason: str


@dataclass(frozen=True)
class CoordinatedFrontierPlannerResult:
    targets: tuple[tuple[float, float] | None, ...]
    reasons: tuple[str, ...]
    assignments: tuple[CoordinatedFrontierAssignment | None, ...]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))




def _distance_point_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    closest = (ax + t * dx, ay + t * dy)
    return _distance(point, closest)


def _segment_crosses_any_disk(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    disks: Sequence[tuple[float, float, float]],
    margin: float,
    ignore_near_start: float = 0.0,
) -> bool:
    """Return True if a proposed corridor crosses a robot safety disk.

    This catches the failure mode where a frontier target is valid, but the
    first committed segment passes through a teammate that is holding position.
    The local safety controller should remain the final guard, but the planner
    should avoid assigning these corridors in the first place.
    """
    for cx, cy, radius in disks:
        center = (float(cx), float(cy))
        if ignore_near_start > 0.0 and _distance(start, center) <= ignore_near_start:
            continue
        if _distance_point_to_segment(center, start, end) <= float(radius) + max(float(margin), 0.0):
            return True
    return False


def _polyline_segments(route: Sequence[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    points = [(float(p[0]), float(p[1])) for p in route if p is not None]
    if len(points) < 2:
        return []
    return list(zip(points[:-1], points[1:]))


def _route_segment_overlap_ratio(
    *,
    start: tuple[float, float],
    target: tuple[float, float],
    routes: Sequence[Sequence[tuple[float, float]]],
    radius: float,
    sample_spacing: float,
) -> float:
    samples = _sample_route_to_target(start, target, sample_spacing)
    if not samples:
        return 0.0

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for route in routes:
        segments.extend(_polyline_segments(route))
    if not segments:
        return 0.0

    hits = 0
    radius = max(float(radius), 0.0)
    for sample in samples:
        if any(_distance_point_to_segment(sample, a, b) <= radius for a, b in segments):
            hits += 1
    return hits / len(samples)


def _cell_key(point: tuple[float, float], resolution: float) -> tuple[int, int]:
    return (int(round(float(point[0]) / resolution)), int(round(float(point[1]) / resolution)))


def _cell_center(cell: tuple[int, int], resolution: float) -> tuple[float, float]:
    return (float(cell[0]) * resolution, float(cell[1]) * resolution)


def _inside_bounds(point: tuple[float, float], bounds: tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def _normalize_target(target) -> tuple[float, float] | None:
    if target is None:
        return None
    try:
        x, y = target
        return (float(x), float(y))
    except (TypeError, ValueError):
        return None


def _target_list(items: Iterable | None) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not items:
        return out
    for item in items:
        target = _normalize_target(item)
        if target is not None:
            out.append(target)
    return out


def _occupied_cells_from_points(
    mapped_obstacle_points: Sequence[tuple[float, float]],
    resolution: float,
    robot_radius: float,
) -> set[tuple[int, int]]:
    occupied: set[tuple[int, int]] = set()
    if not mapped_obstacle_points:
        return occupied

    resolution = max(float(resolution), 1e-6)
    radius_cells = max(1, int(math.ceil(float(robot_radius) / resolution)))

    for point in mapped_obstacle_points:
        center = _cell_key(point, resolution)
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                cell = (center[0] + dx, center[1] + dy)
                world = _cell_center(cell, resolution)
                if _distance(world, point) <= robot_radius + resolution * 0.75:
                    occupied.add(cell)
    return occupied


def _cluster_cells(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    offsets = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    remaining = set(cells)
    clusters: list[list[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        cluster = [start]
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            cell = queue.popleft()
            for dx, dy in offsets:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
                cluster.append(neighbor)
        clusters.append(cluster)
    return clusters


def _detect_global_frontier_viewpoints(
    *,
    explored_points: Sequence[tuple[float, float]],
    mapped_obstacle_points: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    resolution: float,
    robot_radius: float,
    sensor_range: float,
) -> list[FrontierCandidate]:
    """Detect frontiers once from the shared map and return viewpoint candidates."""
    resolution = max(float(resolution), 1e-6)
    explored = {
        _cell_key(point, resolution)
        for point in explored_points
        if _inside_bounds(point, bounds)
    }
    if not explored:
        return []

    occupied = _occupied_cells_from_points(mapped_obstacle_points, resolution, robot_radius)
    explored_free = explored - occupied
    if not explored_free:
        return []

    frontier_cells: set[tuple[int, int]] = set()
    for cell in explored_free:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (cell[0] + dx, cell[1] + dy)
            neighbor_world = _cell_center(neighbor, resolution)
            if not _inside_bounds(neighbor_world, bounds):
                continue
            if neighbor not in explored:
                frontier_cells.add(cell)
                break

    if not frontier_cells:
        return []

    candidates: list[FrontierCandidate] = []
    for cluster in _cluster_cells(frontier_cells):
        if not cluster:
            continue
        cluster_world = [_cell_center(cell, resolution) for cell in cluster]
        centroid = (
            sum(point[0] for point in cluster_world) / len(cluster_world),
            sum(point[1] for point in cluster_world) / len(cluster_world),
        )
        if len(cluster_world) >= 8:
            if max(max(point[0] for point in cluster_world) - min(point[0] for point in cluster_world),
                   max(point[1] for point in cluster_world) - min(point[1] for point in cluster_world)) >= 1.5 * resolution:
                if max(point[0] for point in cluster_world) - min(point[0] for point in cluster_world) >= \
                   max(point[1] for point in cluster_world) - min(point[1] for point in cluster_world):
                    sorted_points = sorted(cluster_world, key=lambda point: point[0])
                    slices = [sorted_points[i * len(sorted_points) // 3:(i + 1) * len(sorted_points) // 3] for i in range(3)]
                else:
                    sorted_points = sorted(cluster_world, key=lambda point: point[1])
                    slices = [sorted_points[i * len(sorted_points) // 3:(i + 1) * len(sorted_points) // 3] for i in range(3)]
                cluster_slices = [slice_points for slice_points in slices if slice_points]
            else:
                cluster_slices = [cluster_world]
        else:
            cluster_slices = [cluster_world]

        for slice_world in cluster_slices:
            if not slice_world:
                continue
            target = min(slice_world, key=lambda point: _distance(point, centroid))
            if not _inside_bounds(target, bounds):
                continue
            information_gain = _estimate_information_gain(
                target=target,
                explored=explored,
                occupied=occupied,
                bounds=bounds,
                resolution=resolution,
                sensor_range=sensor_range,
            )
            candidates.append(
                FrontierCandidate(
                    target=(float(target[0]), float(target[1])),
                    size=len(slice_world),
                    distance_from_robot=0.0,
                    information_gain=information_gain,
                    score=information_gain,
                    reason=f"frontier size={len(slice_world)}, info_gain={information_gain:.1f}",
                )
            )
    return candidates


def _estimate_information_gain(
    *,
    target: tuple[float, float],
    explored: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    bounds: tuple[float, float, float, float],
    resolution: float,
    sensor_range: float,
) -> float:
    resolution = max(float(resolution), 1e-6)
    radius_cells = max(1, int(math.ceil(max(float(sensor_range), resolution) / resolution)))
    target_cell = _cell_key(target, resolution)
    gain = 0
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            cell = (target_cell[0] + dx, target_cell[1] + dy)
            if cell in explored or cell in occupied:
                continue
            world = _cell_center(cell, resolution)
            if not _inside_bounds(world, bounds):
                continue
            if _distance(target, world) <= sensor_range:
                gain += 1
    return float(gain)


def _footprint_cells(
    target: tuple[float, float],
    *,
    resolution: float,
    sensor_range: float,
    bounds: tuple[float, float, float, float],
) -> set[tuple[int, int]]:
    resolution = max(float(resolution), 1e-6)
    radius_cells = max(1, int(math.ceil(max(float(sensor_range), resolution) / resolution)))
    center = _cell_key(target, resolution)
    cells: set[tuple[int, int]] = set()
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            cell = (center[0] + dx, center[1] + dy)
            world = _cell_center(cell, resolution)
            if _inside_bounds(world, bounds) and _distance(target, world) <= sensor_range:
                cells.add(cell)
    return cells


def _sample_route_to_target(
    start: tuple[float, float],
    target: tuple[float, float],
    spacing: float,
) -> list[tuple[float, float]]:
    spacing = max(float(spacing), 1e-6)
    length = _distance(start, target)
    if length <= 1e-9:
        return [start]
    steps = max(1, int(math.ceil(length / spacing)))
    return [
        (
            start[0] + (target[0] - start[0]) * (i / steps),
            start[1] + (target[1] - start[1]) * (i / steps),
        )
        for i in range(steps + 1)
    ]


def _ratio_near_points(
    samples: Sequence[tuple[float, float]],
    points: Sequence[tuple[float, float]],
    radius: float,
) -> float:
    if not samples or not points:
        return 0.0
    radius = max(float(radius), 0.0)
    radius_sq = radius * radius
    hits = 0
    for sample in samples:
        sx, sy = sample
        for px, py in points:
            dx = sx - px
            dy = sy - py
            if dx * dx + dy * dy <= radius_sq:
                hits += 1
                break
    return hits / max(len(samples), 1)


def _target_too_close(
    target: tuple[float, float],
    others: Sequence[tuple[float, float]],
    radius: float,
) -> bool:
    return any(_distance(target, other) <= radius for other in others)


def _build_debug_reason(
    base_reason: str,
    *,
    raw_candidates: int,
    rejected_by_invalidated: int,
    rejected_by_target_too_close: int,
    rejected_by_teammate_distance: int,
    rejected_by_corridor_overlap: int,
    rejected_by_route_overlap: int,
    rejected_by_zero_information_gain: int,
    selected_target: tuple[float, float] | None,
) -> str:
    selected_text = "HOLD" if selected_target is None else f"{selected_target[0]:.2f},{selected_target[1]:.2f}"
    return (
        f"{base_reason}; raw_candidates={raw_candidates}; "
        f"rejected_by_invalidated={rejected_by_invalidated}; "
        f"rejected_by_target_too_close={rejected_by_target_too_close}; "
        f"rejected_by_teammate_distance={rejected_by_teammate_distance}; "
        f"rejected_by_corridor_overlap={rejected_by_corridor_overlap}; "
        f"rejected_by_route_overlap={rejected_by_route_overlap}; "
        f"rejected_by_zero_information_gain={rejected_by_zero_information_gain}; "
        f"selected_target={selected_text}"
    )


def assign_frontier_viewpoints(
    *,
    robot_states,
    existing_targets: Sequence[tuple[float, float] | None],
    robots_to_assign: Sequence[int],
    invalidated_targets_by_robot: Sequence[Sequence[tuple[float, float]]] | None,
    explored_points: Sequence[tuple[float, float]],
    mapped_obstacle_points: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    resolution: float,
    final_goal_xy: tuple[float, float],
    ipp_distance_penalty: float,
    target_exclusion_radius: float,
    dynamic_obstacle_margin: float,
    route_points_by_robot: Sequence[Sequence[tuple[float, float]]] | None = None,
    explored_points_by_robot: Sequence[Sequence[tuple[float, float]]] | None = None,
) -> CoordinatedFrontierPlannerResult:
    """Assign global frontier viewpoints to robots in one synchronized pass."""
    count = len(robot_states)
    assign_set = {int(index) for index in robots_to_assign if 0 <= int(index) < count}
    targets: list[tuple[float, float] | None] = [
        _normalize_target(existing_targets[index]) if index < len(existing_targets) else None
        for index in range(count)
    ]
    reasons: list[str] = [
        "kept existing target" if targets[index] is not None else "no target assigned yet"
        for index in range(count)
    ]
    assignments: list[CoordinatedFrontierAssignment | None] = [None for _ in range(count)]

    if not assign_set:
        return CoordinatedFrontierPlannerResult(tuple(targets), tuple(reasons), tuple(assignments))

    resolution = max(float(resolution), 1e-6)
    max_robot_radius = max((float(state.safety_radius) for state in robot_states), default=0.0)
    avg_sensor_range = sum(float(state.sensor_range) for state in robot_states) / max(count, 1)
    target_exclusion_radius = max(float(target_exclusion_radius), resolution)
    dynamic_obstacle_margin = max(float(dynamic_obstacle_margin), 0.0)

    global_candidates = _detect_global_frontier_viewpoints(
        explored_points=explored_points,
        mapped_obstacle_points=mapped_obstacle_points,
        bounds=bounds,
        resolution=resolution,
        robot_radius=max_robot_radius,
        sensor_range=avg_sensor_range,
    )
    if not global_candidates:
        for index in assign_set:
            targets[index] = None
            reasons[index] = "coordinated frontier planner: no global frontier candidates"
        return CoordinatedFrontierPlannerResult(tuple(targets), tuple(reasons), tuple(assignments))

    kept_targets = [target for idx, target in enumerate(targets) if idx not in assign_set and target is not None]
    route_points_by_robot = route_points_by_robot or [[] for _ in range(count)]
    explored_points_by_robot = explored_points_by_robot or [[] for _ in range(count)]

    # Per-robot explored sets are used only for footprint overlap penalties.  A
    # robot may drive over known free cells; it just should not select a sensor
    # footprint mostly already seen by another robot.
    explored_cells_by_robot = []
    for points in explored_points_by_robot:
        explored_cells_by_robot.append({_cell_key(point, resolution) for point in points})

    pair_options: dict[int, list[CoordinatedFrontierAssignment]] = {index: [] for index in assign_set}
    debug_tracker: dict[int, dict[str, int]] = {index: {
        "raw_candidates": 0,
        "rejected_by_invalidated": 0,
        "rejected_by_target_too_close": 0,
        "rejected_by_teammate_distance": 0,
        "rejected_by_corridor_overlap": 0,
        "rejected_by_route_overlap": 0,
        "rejected_by_zero_information_gain": 0,
    } for index in assign_set}
    for index in sorted(assign_set):
        state = robot_states[index]
        invalidated = _target_list(
            invalidated_targets_by_robot[index]
            if invalidated_targets_by_robot and index < len(invalidated_targets_by_robot)
            else []
        )
        teammates = [
            (float(other.xy[0]), float(other.xy[1]))
            for j, other in enumerate(robot_states)
            if j != index
        ]
        teammate_block_radius = float(state.safety_radius) + max_robot_radius + dynamic_obstacle_margin

        other_explored_cells: set[tuple[int, int]] = set()
        for j, cells in enumerate(explored_cells_by_robot):
            if j != index:
                other_explored_cells.update(cells)

        other_routes = [list(route) for j, route in enumerate(route_points_by_robot) if j != index]
        teammate_disks = [
            (float(other.xy[0]), float(other.xy[1]), float(other.safety_radius))
            for j, other in enumerate(robot_states)
            if j != index
        ]

        for candidate in global_candidates:
            debug_tracker[index]["raw_candidates"] += 1
            target = candidate.target
            if _target_too_close(target, kept_targets, target_exclusion_radius):
                debug_tracker[index]["rejected_by_invalidated"] += 1
                continue

            distance = _distance(state.xy, target)
            invalidated_penalty = 2.0 if _target_too_close(target, invalidated, target_exclusion_radius) else 0.0
            if invalidated_penalty > 0.0:
                debug_tracker[index]["rejected_by_invalidated"] += 1
            footprint = _footprint_cells(
                target,
                resolution=resolution,
                sensor_range=float(state.sensor_range),
                bounds=bounds,
            )
            other_map_ratio = (
                len(footprint & other_explored_cells) / max(len(footprint), 1)
                if footprint else 0.0
            )
            corridor_margin = max(dynamic_obstacle_margin, resolution * 0.75)
            corridor_conflict = False
            for cx, cy, radius in teammate_disks:
                center = (float(cx), float(cy))
                if _distance(target, center) > float(radius) + float(state.safety_radius) + corridor_margin:
                    continue
                if _distance_point_to_segment(center, state.xy, target) <= float(radius) + float(state.safety_radius) + corridor_margin:
                    corridor_conflict = True
                    break
            if corridor_conflict:
                debug_tracker[index]["rejected_by_corridor_overlap"] += 1
                continue

            route_overlap_ratio = _route_segment_overlap_ratio(
                start=state.xy,
                target=target,
                routes=other_routes,
                radius=max(float(state.safety_radius) * 2.0, resolution),
                sample_spacing=resolution,
            )
            if route_overlap_ratio > 0.6:
                debug_tracker[index]["rejected_by_route_overlap"] += 1

            information_gain = float(candidate.information_gain)
            if information_gain <= 0.0:
                debug_tracker[index]["rejected_by_zero_information_gain"] += 1
                continue

            # Overlap and route redundancy are treated as soft penalties.  A useful
            # frontier should still be available when the team starts close or when
            # the initial FoV overlaps; the planner should only reject candidates
            # that are physically unsafe or otherwise impossible.
            score = (
                information_gain
                - float(ipp_distance_penalty) * distance
                - 6.0 * other_map_ratio
                - 3.0 * route_overlap_ratio
                - invalidated_penalty
            )
            reason = (
                "coordinated frontier: assigned; "
                f"J={score:.2f}; IG={information_gain:.1f}; "
                f"d={distance:.2f}; other_map={other_map_ratio:.2f}; "
                f"other_route={route_overlap_ratio:.2f}; {candidate.reason}"
            )
            pair_options[index].append(
                CoordinatedFrontierAssignment(
                    target=target,
                    score=score,
                    information_gain=information_gain,
                    distance=distance,
                    other_map_ratio=other_map_ratio,
                    route_overlap_ratio=route_overlap_ratio,
                    reason=reason,
                )
            )
        pair_options[index].sort(
            key=lambda item: (item.score, item.information_gain, -item.distance),
            reverse=True,
        )

        if not pair_options[index]:
            relaxed_candidates: list[CoordinatedFrontierAssignment] = []
            for candidate in global_candidates:
                target = candidate.target
                if _target_too_close(target, kept_targets, target_exclusion_radius):
                    continue
                if _target_too_close(target, invalidated, target_exclusion_radius):
                    debug_tracker[index]["rejected_by_invalidated"] += 1
                distance = _distance(state.xy, target)
                info_gain = float(candidate.information_gain)
                if info_gain <= 0.0:
                    debug_tracker[index]["rejected_by_zero_information_gain"] += 1
                    continue
                score = info_gain - float(ipp_distance_penalty) * distance
                relaxed_candidates.append(
                    CoordinatedFrontierAssignment(
                        target=target,
                        score=score,
                        information_gain=info_gain,
                        distance=distance,
                        other_map_ratio=0.0,
                        route_overlap_ratio=0.0,
                        reason=(
                            "coordinated frontier: relaxed fallback; "
                            f"J={score:.2f}; IG={info_gain:.1f}; d={distance:.2f}; {candidate.reason}"
                        ),
                    )
                )
            relaxed_candidates.sort(
                key=lambda item: (item.score, item.information_gain, -item.distance),
                reverse=True,
            )
            pair_options[index].extend(relaxed_candidates[:1])

    # Exhaustive search is fine for GUI-scale teams; cap each robot to its best
    # candidates so 3-10 robots stay responsive without scipy/Hungarian.
    active = [index for index in sorted(assign_set) if pair_options.get(index)]
    for index in sorted(assign_set - set(active)):
        targets[index] = None
        reasons[index] = _build_debug_reason(
            "coordinated frontier planner: no candidate survived filtering",
            raw_candidates=debug_tracker.get(index, {}).get("raw_candidates", 0),
            rejected_by_invalidated=debug_tracker.get(index, {}).get("rejected_by_invalidated", 0),
            rejected_by_target_too_close=debug_tracker.get(index, {}).get("rejected_by_target_too_close", 0),
            rejected_by_teammate_distance=debug_tracker.get(index, {}).get("rejected_by_teammate_distance", 0),
            rejected_by_corridor_overlap=debug_tracker.get(index, {}).get("rejected_by_corridor_overlap", 0),
            rejected_by_route_overlap=debug_tracker.get(index, {}).get("rejected_by_route_overlap", 0),
            rejected_by_zero_information_gain=debug_tracker.get(index, {}).get("rejected_by_zero_information_gain", 0),
            selected_target=None,
        )

    best_score = -float("inf")
    best: dict[int, CoordinatedFrontierAssignment] = {}
    chosen_targets: list[tuple[float, float]] = list(kept_targets)

    def dfs(pos: int, current_score: float, partial: dict[int, CoordinatedFrontierAssignment]) -> None:
        nonlocal best_score, best, chosen_targets
        if pos >= len(active):
            # Holding is only a last resort. Useful candidates should be selected
            # before the planner falls back to a permanent hold.
            total = current_score - 1000.0 * (len(active) - len(partial))
            if total > best_score:
                best_score = total
                best = dict(partial)
            return

        robot_index = active[pos]
        # HOLD branch: better than assigning the exact same frontier to multiple robots.
        dfs(pos + 1, current_score - 1000.0, partial)

        for option in pair_options[robot_index][:12]:
            if _target_too_close(option.target, chosen_targets, target_exclusion_radius):
                continue
            extra_penalty = 0.0
            for other_robot_index, other in partial.items():
                separation = _distance(option.target, other.target)
                if separation < target_exclusion_radius:
                    extra_penalty += 1000.0
                elif separation < avg_sensor_range:
                    extra_penalty += 4.0 * (avg_sensor_range - separation) / max(avg_sensor_range, 1e-6)

                # Two newly assigned robots should also not receive corridors
                # that immediately pass through each other's current safety
                # zones. This is not robot-robot avoidance control; it is target
                # assignment refusing an obviously bad crossing.
                other_state = robot_states[other_robot_index]
                current_state = robot_states[robot_index]
                if _segment_crosses_any_disk(
                    start=current_state.xy,
                    end=option.target,
                    disks=[(other_state.xy[0], other_state.xy[1], float(other_state.safety_radius))],
                    margin=float(current_state.safety_radius) + dynamic_obstacle_margin,
                ):
                    extra_penalty += 1000.0
                if _segment_crosses_any_disk(
                    start=other_state.xy,
                    end=other.target,
                    disks=[(current_state.xy[0], current_state.xy[1], float(current_state.safety_radius))],
                    margin=float(other_state.safety_radius) + dynamic_obstacle_margin,
                ):
                    extra_penalty += 1000.0
            if extra_penalty >= 1000.0:
                continue
            partial[robot_index] = option
            chosen_targets.append(option.target)
            dfs(pos + 1, current_score + option.score - extra_penalty, partial)
            chosen_targets.pop()
            partial.pop(robot_index, None)

    dfs(0, 0.0, {})

    for index, assignment in best.items():
        targets[index] = assignment.target
        assignments[index] = assignment
        reasons[index] = _build_debug_reason(
            assignment.reason,
            raw_candidates=debug_tracker.get(index, {}).get("raw_candidates", 0),
            rejected_by_invalidated=debug_tracker.get(index, {}).get("rejected_by_invalidated", 0),
            rejected_by_target_too_close=debug_tracker.get(index, {}).get("rejected_by_target_too_close", 0),
            rejected_by_teammate_distance=debug_tracker.get(index, {}).get("rejected_by_teammate_distance", 0),
            rejected_by_corridor_overlap=debug_tracker.get(index, {}).get("rejected_by_corridor_overlap", 0),
            rejected_by_route_overlap=debug_tracker.get(index, {}).get("rejected_by_route_overlap", 0),
            rejected_by_zero_information_gain=debug_tracker.get(index, {}).get("rejected_by_zero_information_gain", 0),
            selected_target=assignment.target,
        )

    for index in sorted(assign_set - set(best.keys())):
        targets[index] = None
        if not reasons[index] or reasons[index] == "no target assigned yet":
            reasons[index] = _build_debug_reason(
                "coordinated frontier planner: held to avoid duplicate/low-value frontier",
                raw_candidates=debug_tracker.get(index, {}).get("raw_candidates", 0),
                rejected_by_invalidated=debug_tracker.get(index, {}).get("rejected_by_invalidated", 0),
                rejected_by_target_too_close=debug_tracker.get(index, {}).get("rejected_by_target_too_close", 0),
                rejected_by_teammate_distance=debug_tracker.get(index, {}).get("rejected_by_teammate_distance", 0),
                rejected_by_corridor_overlap=debug_tracker.get(index, {}).get("rejected_by_corridor_overlap", 0),
                rejected_by_route_overlap=debug_tracker.get(index, {}).get("rejected_by_route_overlap", 0),
                rejected_by_zero_information_gain=debug_tracker.get(index, {}).get("rejected_by_zero_information_gain", 0),
                selected_target=None,
            )

    return CoordinatedFrontierPlannerResult(tuple(targets), tuple(reasons), tuple(assignments))
