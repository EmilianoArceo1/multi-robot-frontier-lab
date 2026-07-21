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

from robotics_interfaces.frontiers import FrontierCluster, ViewpointCandidate
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


@dataclass(frozen=True)
class CorridorValidationResult:
    """Outcome of validating a full waypoint corridor before it goes ACTIVE.

    reason_code is "" when is_valid is True, otherwise one of:
        "route_conflict_with_robot_safety_zone"
        "route_conflict_with_active_route"
        "corridor_reservation_conflict"
    """

    is_valid: bool
    reason_code: str
    detail: str


def _sample_polyline(points: Sequence[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    normalized = [(float(p[0]), float(p[1])) for p in points if p is not None]
    if len(normalized) < 2:
        return list(normalized)
    samples: list[tuple[float, float]] = []
    for start, end in zip(normalized[:-1], normalized[1:]):
        samples.extend(_sample_route_to_target(start, end, spacing))
    return samples


def validate_multi_robot_corridor(
    *,
    start: tuple[float, float],
    waypoints: Sequence[tuple[float, float]],
    ego_safety_radius: float,
    other_robot_disks: Sequence[tuple[float, float, float]] = (),
    other_routes: Sequence[Sequence[tuple[float, float]]] = (),
    reserved_corridors: Sequence[Sequence[tuple[float, float]]] = (),
    margin: float = 0.0,
    sample_spacing: float = 0.25,
) -> CorridorValidationResult:
    """Validate a FULL candidate corridor before it is accepted as ACTIVE.

    This runs earlier and is stricter than the per-frame movement safety veto,
    which only checks the immediate next segment once a route is already
    moving. Here every planned segment (Direct's single segment included) is
    checked against:
        - other robots' current position + safety radius
        - other robots' active routes
        - reserved corridors (if any are supplied; empty by default today)

    A robot that already starts within another robot's combined safety
    clearance is not treated as a violation -- that is the team's starting
    formation, not a corridor crossing (mirrors the same fix applied to
    assign_frontier_viewpoints's teammate-corridor check).
    """
    corridor = [(float(start[0]), float(start[1]))] + [
        (float(point[0]), float(point[1])) for point in waypoints
    ]
    segments = _polyline_segments(corridor)
    if not segments:
        return CorridorValidationResult(True, "", "no movement segment to validate")

    ego_safety_radius = float(ego_safety_radius)
    margin = max(float(margin), 0.0)

    for cx, cy, radius in other_robot_disks:
        center = (float(cx), float(cy))
        required = ego_safety_radius + float(radius) + margin
        if _distance(corridor[0], center) <= required:
            # Already this close at the start is the spawn/starting formation,
            # not a corridor the robot is choosing to drive through.
            continue
        for seg_start, seg_end in segments:
            if _distance_point_to_segment(center, seg_start, seg_end) <= required:
                return CorridorValidationResult(
                    False,
                    "route_conflict_with_robot_safety_zone",
                    f"corridor passes within {required:.2f} m of a teammate at "
                    f"({center[0]:.2f}, {center[1]:.2f})",
                )

    samples = _sample_polyline(corridor, max(float(sample_spacing), 1e-3))
    required_clearance = ego_safety_radius + margin

    def _crosses_any(routes: Sequence[Sequence[tuple[float, float]]]) -> bool:
        for route in routes:
            route_segments = _polyline_segments(route)
            if not route_segments:
                continue
            for sample in samples:
                if any(
                    _distance_point_to_segment(sample, a, b) <= required_clearance
                    for a, b in route_segments
                ):
                    return True
        return False

    if _crosses_any(other_routes):
        return CorridorValidationResult(
            False,
            "route_conflict_with_active_route",
            "corridor crosses a teammate's active route",
        )

    if _crosses_any(reserved_corridors):
        return CorridorValidationResult(
            False,
            "corridor_reservation_conflict",
            "corridor crosses a reserved corridor",
        )

    return CorridorValidationResult(True, "", "corridor clear")


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
    """8-connected components of `cells`, fully deterministic.

    `cells` is a set, so its own iteration order can depend on the set's
    internal hash-table history (insertion/deletion order), not just its
    final membership. Seeding each component from `sorted(cells)` instead
    of `set.pop()`, and sorting each component's cells before returning,
    means the same cell membership always produces the same components in
    the same order with the same cell order -- regardless of what order
    the cells were originally added in. No hash() or hash-order dependence
    anywhere in this function.
    """
    offsets = (
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    )
    remaining = set(cells)
    clusters: list[list[tuple[int, int]]] = []
    for start in sorted(cells):
        if start not in remaining:
            continue
        remaining.discard(start)
        cluster = [start]
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            cell = queue.popleft()
            for dx, dy in offsets:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in remaining:
                    continue
                remaining.discard(neighbor)
                queue.append(neighbor)
                cluster.append(neighbor)
        clusters.append(sorted(cluster))
    return clusters


def _frontier_cells_from_map(
    *,
    explored_points: Sequence[tuple[float, float]],
    mapped_obstacle_points: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    resolution: float,
    robot_radius: float,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Shared first stage of frontier detection: discretize the map and
    return (frontier_cells, explored, occupied). A cell is frontier iff it
    is explored, free (not occupied), and has at least one in-bounds
    4-connected (cardinal) neighbor that is not in `explored` -- i.e. its
    status is genuinely unknown, not merely occupied. This is pure set
    membership, so the result never depends on the order of
    explored_points/mapped_obstacle_points, only on which points they
    contain."""
    explored = {
        _cell_key(point, resolution)
        for point in explored_points
        if _inside_bounds(point, bounds)
    }
    if not explored:
        return set(), explored, set()

    occupied = _occupied_cells_from_points(mapped_obstacle_points, resolution, robot_radius)
    explored_free = explored - occupied
    if not explored_free:
        return set(), explored, occupied

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

    return frontier_cells, explored, occupied


def _component_slices(
    cluster_world: Sequence[tuple[float, float]], *, resolution: float
) -> list[list[tuple[float, float]]]:
    """Split one connected component into up to three viewpoint slices.

    Preserves the existing heuristic exactly: components under 8 cells, or
    whose bounding-box extent is under 1.5*resolution in both axes, get a
    single slice (one viewpoint). Larger/elongated components are cut into
    up to three slices along their longer axis, so one big frontier is
    covered by more than one viewpoint instead of a single one far from
    most of it. Ties in the sort key are broken by `cluster_world`'s own
    (already-deterministic) order, since sorted() is stable.
    """
    if len(cluster_world) >= 8:
        min_x = min(point[0] for point in cluster_world)
        max_x = max(point[0] for point in cluster_world)
        min_y = min(point[1] for point in cluster_world)
        max_y = max(point[1] for point in cluster_world)
        width = max_x - min_x
        height = max_y - min_y
        if max(width, height) >= 1.5 * resolution:
            sort_key = (lambda point: point[0]) if width >= height else (lambda point: point[1])
            sorted_points = sorted(cluster_world, key=sort_key)
            slices = [
                sorted_points[i * len(sorted_points) // 3 : (i + 1) * len(sorted_points) // 3]
                for i in range(3)
            ]
            return [slice_points for slice_points in slices if slice_points]
    return [list(cluster_world)]


def detect_connected_frontier_components(
    *,
    explored_points: Sequence[tuple[float, float]],
    mapped_obstacle_points: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    resolution: float,
    robot_radius: float,
    sensor_range: float,
) -> tuple[FrontierCluster, ...]:
    """Detect connected frontier components from the shared map, once, and
    return them as real FrontierCluster objects.

    Pure function: no engine/Qt/MainWindow/canvas import, no global mutable
    state -- everything it needs is passed in. Reuses the exact same
    discretization/occupied-cell/frontier-cell/8-connectivity/information-
    gain/viewpoint-slicing semantics `_detect_global_frontier_viewpoints`
    used to implement directly (see _frontier_cells_from_map(),
    _cluster_cells(), _component_slices(), _estimate_information_gain());
    this commit only changes what those cells/slices are packaged into.

    Determinism: `_cluster_cells()` returns components (and cells within
    each component) sorted lexicographically by discrete (int, int)
    coordinate, seeded from `sorted(cells)` rather than `set.pop()` -- so
    the same map always yields the same components in the same order,
    regardless of explored_points/mapped_obstacle_points input order. IDs
    are assigned after that ordering as frontier-component-0000,
    frontier-component-0001, ... and are only stable for one observation;
    this commit does not track cluster identity across calls.
    """
    resolution = max(float(resolution), 1e-6)
    frontier_cells, explored, occupied = _frontier_cells_from_map(
        explored_points=explored_points,
        mapped_obstacle_points=mapped_obstacle_points,
        bounds=bounds,
        resolution=resolution,
        robot_radius=robot_radius,
    )
    if not frontier_cells:
        return ()

    clusters: list[FrontierCluster] = []
    for component_index, component_cells in enumerate(_cluster_cells(frontier_cells)):
        if not component_cells:
            continue
        cluster_world = [_cell_center(cell, resolution) for cell in component_cells]
        centroid = (
            sum(point[0] for point in cluster_world) / len(cluster_world),
            sum(point[1] for point in cluster_world) / len(cluster_world),
        )
        cluster_id = f"frontier-component-{component_index:04d}"

        viewpoints: list[ViewpointCandidate] = []
        for slice_index, slice_world in enumerate(_component_slices(cluster_world, resolution=resolution)):
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
            viewpoints.append(
                ViewpointCandidate(
                    xy=(float(target[0]), float(target[1])),
                    information_gain=information_gain,
                    visible_cell_count=len(slice_world),
                    metadata={
                        "component_id": cluster_id,
                        "slice_index": slice_index,
                        "slice_cell_count": len(slice_world),
                    },
                )
            )

        cluster_information_gain = max((vp.information_gain for vp in viewpoints), default=0.0)

        clusters.append(
            FrontierCluster(
                cluster_id=cluster_id,
                cells=tuple((float(point[0]), float(point[1])) for point in cluster_world),
                centroid=(float(centroid[0]), float(centroid[1])),
                viewpoints=tuple(viewpoints),
                information_gain=cluster_information_gain,
                metadata={
                    "source": "connected_frontier_component",
                    "frontier_cell_count": len(component_cells),
                    "resolution": resolution,
                    "viewpoint_count": len(viewpoints),
                },
                valid=True,
            )
        )

    return tuple(clusters)


def _detect_global_frontier_viewpoints(
    *,
    explored_points: Sequence[tuple[float, float]],
    mapped_obstacle_points: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    resolution: float,
    robot_radius: float,
    sensor_range: float,
) -> list[FrontierCandidate]:
    """Legacy flat-candidate view over detect_connected_frontier_components().

    Kept as a module-level function (rather than folded into
    detect_global_frontier_candidates()) because it is directly monkeypatched
    and called by name from legacy callers/tests that predate FrontierCluster
    -- assign_frontier_viewpoints() below, and
    robotics_sim/tests/test_noic_frontier_regressions.py. One FrontierCandidate
    is emitted per cluster viewpoint (slice), matching what this function
    returned before clusters existed.
    """
    clusters = detect_connected_frontier_components(
        explored_points=explored_points,
        mapped_obstacle_points=mapped_obstacle_points,
        bounds=bounds,
        resolution=resolution,
        robot_radius=robot_radius,
        sensor_range=sensor_range,
    )

    candidates: list[FrontierCandidate] = []
    for cluster in clusters:
        for viewpoint in cluster.viewpoints:
            slice_cell_count = int(viewpoint.metadata.get("slice_cell_count", viewpoint.visible_cell_count))
            information_gain = float(viewpoint.information_gain)
            candidates.append(
                FrontierCandidate(
                    target=(float(viewpoint.xy[0]), float(viewpoint.xy[1])),
                    size=slice_cell_count,
                    distance_from_robot=0.0,
                    information_gain=information_gain,
                    score=information_gain,
                    reason=f"frontier size={slice_cell_count}, info_gain={information_gain:.1f}",
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


_DEBUG_COUNTER_KEYS = (
    "raw_candidates",
    "rejected_by_invalidated",
    "rejected_by_target_too_close",
    "rejected_by_teammate_distance",
    "rejected_by_corridor_overlap",
    "rejected_by_route_overlap",
    "rejected_by_zero_information_gain",
    # Named to match what an operator actually asks when reading a HOLD/penalty
    # reason: "why was this candidate rejected or penalized?"
    "too_close_to_robot",
    "target_reservation_conflict",
    "active_route_overlap",
    "fov_overlap_penalty",
    "safety_replan_blacklist",
)


def _new_debug_counters() -> dict[str, int]:
    return {key: 0 for key in _DEBUG_COUNTER_KEYS}


def _build_debug_reason(
    base_reason: str,
    *,
    counters: dict[str, int],
    selected_target: tuple[float, float] | None,
) -> str:
    selected_text = "HOLD" if selected_target is None else f"{selected_target[0]:.2f},{selected_target[1]:.2f}"
    counters_text = "; ".join(f"{key}={counters.get(key, 0)}" for key in _DEBUG_COUNTER_KEYS)
    return f"{base_reason}; {counters_text}; selected_target={selected_text}"

def detect_global_frontier_candidates(
    *,
    explored_points: Sequence[tuple[float, float]],
    mapped_obstacle_points: Sequence[tuple[float, float]],
    bounds: tuple[float, float, float, float],
    resolution: float,
    robot_radius: float,
    sensor_range: float,
) -> tuple[FrontierCandidate, ...]:
    """Return raw shared-map frontier candidates without assigning robots.

    This is the candidate-generation half of the coordinated frontier planner.
    TeamFrontierProvider adapters should call this function, not
    assign_frontier_viewpoints(), because providers expose candidate pools while
    coordinator plugins perform task allocation.
    """
    return tuple(
        _detect_global_frontier_viewpoints(
            explored_points=explored_points,
            mapped_obstacle_points=mapped_obstacle_points,
            bounds=bounds,
            resolution=resolution,
            robot_radius=robot_radius,
            sensor_range=sensor_range,
        )
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
    debug_tracker: dict[int, dict[str, int]] = {index: _new_debug_counters() for index in assign_set}
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

            # A target already reserved by a robot that is not being
            # reassigned this call is a strong soft penalty, not a veto: if it
            # is genuinely the only usable candidate, assigning it anyway
            # (accepting some duplication) still beats leaving this robot on
            # HOLD indefinitely.
            reservation_penalty = 0.0
            if _target_too_close(target, kept_targets, target_exclusion_radius):
                debug_tracker[index]["rejected_by_invalidated"] += 1
                debug_tracker[index]["target_reservation_conflict"] += 1
                reservation_penalty = 8.0

            distance = _distance(state.xy, target)
            invalidated_penalty = 2.0 if _target_too_close(target, invalidated, target_exclusion_radius) else 0.0
            if invalidated_penalty > 0.0:
                debug_tracker[index]["rejected_by_invalidated"] += 1
                debug_tracker[index]["safety_replan_blacklist"] += 1
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
            if other_map_ratio > 0.3:
                debug_tracker[index]["fov_overlap_penalty"] += 1

            # Penalize the complete direct corridor, not just candidates whose
            # endpoint happens to be close to a teammate.  The old endpoint
            # pre-filter missed R1 -> R2 -> frontier crossings whenever the
            # frontier itself was far beyond R2.
            corridor_margin = max(dynamic_obstacle_margin, resolution * 0.75)
            corridor_penalty = 0.0
            for cx, cy, radius in teammate_disks:
                if _segment_crosses_any_disk(
                    start=state.xy,
                    end=target,
                    disks=[(cx, cy, radius)],
                    margin=float(state.safety_radius) + corridor_margin,
                    ignore_near_start=teammate_block_radius,
                ):
                    corridor_penalty = 8.0
                    debug_tracker[index]["rejected_by_corridor_overlap"] += 1
                    debug_tracker[index]["too_close_to_robot"] += 1
                    break

            route_overlap_ratio = _route_segment_overlap_ratio(
                start=state.xy,
                target=target,
                routes=other_routes,
                radius=max(float(state.safety_radius) * 2.0, resolution),
                sample_spacing=resolution,
            )
            if route_overlap_ratio > 0.6:
                debug_tracker[index]["rejected_by_route_overlap"] += 1
                debug_tracker[index]["active_route_overlap"] += 1

            information_gain = float(candidate.information_gain)
            if information_gain <= 0.0:
                debug_tracker[index]["rejected_by_zero_information_gain"] += 1
                continue

            # Overlap, reservation conflicts, corridor crossings, and route
            # redundancy are all treated as soft penalties. A useful frontier
            # should still be available when the team starts close together or
            # the initial FoV overlaps; the planner only hard-rejects
            # candidates with zero information gain (see above).
            score = (
                information_gain
                - float(ipp_distance_penalty) * distance
                - 6.0 * other_map_ratio
                - 3.0 * route_overlap_ratio
                - invalidated_penalty
                - reservation_penalty
                - corridor_penalty
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
            counters=debug_tracker.get(index, {}),
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
            duplicate_penalty = 0.0
            if _target_too_close(option.target, chosen_targets, target_exclusion_radius):
                # A candidate that duplicates an already-reserved/chosen target
                # is nearly useless (both robots would drive to the same
                # spot), but it is still better than leaving a robot on HOLD
                # when nothing else is available -- so this is a large soft
                # penalty, not a pruning `continue`.
                duplicate_penalty = 1000.0

            extra_penalty = duplicate_penalty
            for other_robot_index, other in partial.items():
                separation = _distance(option.target, other.target)
                if separation < target_exclusion_radius:
                    extra_penalty += 1000.0
                elif separation < avg_sensor_range:
                    extra_penalty += 4.0 * (avg_sensor_range - separation) / max(avg_sensor_range, 1e-6)

                # Two newly assigned robots should also avoid corridors that
                # cross each other's current safety zones mid-route. This is
                # not robot-robot avoidance control; it is target assignment
                # discouraging an obviously bad crossing. It is a soft penalty
                # (not a veto) so HOLD remains a last resort. ignore_near_start
                # keeps it from firing just because the team already starts
                # close together (e.g. spawning side by side) -- that is not a
                # crossing, it is the starting formation.
                other_state = robot_states[other_robot_index]
                current_state = robot_states[robot_index]
                same_start_radius = (
                    float(current_state.safety_radius) + float(other_state.safety_radius) + dynamic_obstacle_margin
                )
                if _segment_crosses_any_disk(
                    start=current_state.xy,
                    end=option.target,
                    disks=[(other_state.xy[0], other_state.xy[1], float(other_state.safety_radius))],
                    margin=float(current_state.safety_radius) + dynamic_obstacle_margin,
                    ignore_near_start=same_start_radius,
                ):
                    extra_penalty += 400.0
                if _segment_crosses_any_disk(
                    start=other_state.xy,
                    end=other.target,
                    disks=[(current_state.xy[0], current_state.xy[1], float(current_state.safety_radius))],
                    margin=float(other_state.safety_radius) + dynamic_obstacle_margin,
                    ignore_near_start=same_start_radius,
                ):
                    extra_penalty += 400.0
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
            counters=debug_tracker.get(index, {}),
            selected_target=assignment.target,
        )

    for index in sorted(assign_set - set(best.keys())):
        targets[index] = None
        if not reasons[index] or reasons[index] == "no target assigned yet":
            reasons[index] = _build_debug_reason(
                "coordinated frontier planner: held to avoid duplicate/low-value frontier",
                counters=debug_tracker.get(index, {}),
                selected_target=None,
            )

    return CoordinatedFrontierPlannerResult(tuple(targets), tuple(reasons), tuple(assignments))
