"""
Multi-robot frontier coordination strategies.

This module is intentionally independent from Qt and from the visual canvas.
It receives plain robot/map data and returns target assignments. The simulation
engine owns the robots and calls this coordinator when it needs frontier targets.

Separation of responsibilities:
    exploration_planners.py -> generates and scores frontier candidates
    coordination.py         -> decides which robot gets which frontier
    engine.py               -> applies assignments, plans paths, moves robots
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol

from robotics_sim.planning.exploration_planners import (
    DEFAULT_EXPLORATION_PLANNER,
    select_exploration_goal,
    FrontierCandidate,
)
from robotics_sim.planning.coordinated_frontier_planner import assign_frontier_viewpoints

NOIC_COORDINATOR = "NOIC information coordinator"
DEFAULT_COORDINATOR = NOIC_COORDINATOR
COORDINATOR_OPTIONS = [
    NOIC_COORDINATOR,
]


@dataclass(frozen=True)
class RobotCoordinationState:
    """Plain robot state needed to assign a frontier target."""

    xy: tuple[float, float]
    safety_radius: float
    sensor_range: float
    vision_model: str
    theta: float = 0.0


@dataclass(frozen=True)
class CoordinationResult:
    """Result returned by a multi-robot coordinator."""

    targets: tuple[tuple[float, float] | None, ...]
    reasons: tuple[str, ...]
    strategy: str

@dataclass(frozen=True)
class CoordinationRequest:
    """Stable request object consumed by coordination algorithms.

    This keeps the public engine-facing call signature intact while giving this
    module an internal plugin boundary.  Future coordinators should consume this
    request instead of depending on engine, Qt, canvas, or RobotAgent objects.
    """

    planner_name: str
    robot_states: list[RobotCoordinationState]
    existing_targets: list[tuple[float, float] | None]
    robots_to_assign: list[int]
    invalidated_targets_by_robot: list[list[tuple[float, float]]] | None
    explored_points: list[tuple[float, float]]
    mapped_obstacle_points: list[tuple[float, float]]
    bounds: tuple[float, float, float, float]
    resolution: float
    final_goal_xy: tuple[float, float] | None = None
    ipp_distance_penalty: float = 0.5
    target_exclusion_radius: float = 1.5
    dynamic_obstacle_margin: float = 0.5
    route_points_by_robot: list[list[tuple[float, float]]] | None = None
    explored_points_by_robot: list[list[tuple[float, float]]] | None = None


class CoordinationAlgorithm(Protocol):
    """Internal interface implemented by coordination algorithms.

    The simulator-facing host remains MultiRobotCoordinator.  Algorithm
    implementations only receive a CoordinationRequest and return the legacy
    CoordinationResult that engine.py already understands.
    """

    name: str

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        ...



def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Fast Euclidean distance for sanitized 2D tuples."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Squared distance used in tight rejection loops."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _normalize_target(target) -> tuple[float, float] | None:
    if target is None:
        return None
    try:
        x, y = target
        return (float(x), float(y))
    except (TypeError, ValueError):
        return None


def _target_list(items: Iterable | None) -> list[tuple[float, float]]:
    targets: list[tuple[float, float]] = []
    if not items:
        return targets
    for item in items:
        target = _normalize_target(item)
        if target is not None:
            targets.append(target)
    return targets


class SpatialHash2D:
    """Tiny spatial hash for fast radius queries in a bounded 2D world.

    This avoids repeatedly scanning all reserved targets, robots, and route
    samples inside the coordinator. It is intentionally dependency-free; scipy's
    cKDTree would be faster for large point clouds, but this project should run
    in a simple Python/PySide environment without extra installs.
    """

    def __init__(self, cell_size: float):
        self.cell_size = max(float(cell_size), 1e-6)
        self.buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def _key(self, point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(math.floor(point[0] / self.cell_size)),
            int(math.floor(point[1] / self.cell_size)),
        )

    def add(self, point: tuple[float, float]) -> None:
        self.buckets.setdefault(self._key(point), []).append(point)

    def extend(self, points: Iterable[tuple[float, float]]) -> None:
        for point in points:
            self.add(point)

    def any_within(self, point: tuple[float, float], radius: float) -> bool:
        radius = max(float(radius), 0.0)
        radius_sq = radius * radius
        key_x, key_y = self._key(point)
        span = int(math.ceil(radius / self.cell_size))
        for ix in range(key_x - span, key_x + span + 1):
            for iy in range(key_y - span, key_y + span + 1):
                for other in self.buckets.get((ix, iy), ()):
                    if _distance_sq(point, other) <= radius_sq:
                        return True
        return False


def _sample_segment_points(
    start: tuple[float, float],
    end: tuple[float, float],
    spacing: float,
) -> list[tuple[float, float]]:
    """Sample a segment as points for fast corridor/routing conflicts."""
    spacing = max(float(spacing), 1e-6)
    length = _distance(start, end)
    if length <= 1e-9:
        return [start]
    steps = max(1, int(math.ceil(length / spacing)))
    sx, sy = start
    ex, ey = end
    return [
        (sx + (ex - sx) * k / steps, sy + (ey - sy) * k / steps)
        for k in range(steps + 1)
    ]


def _target_near_reserved(
    target: tuple[float, float],
    reserved: list[tuple[float, float]],
    radius: float,
) -> bool:
    radius = max(float(radius), 0.0)
    radius_sq = radius * radius
    return any(_distance_sq(target, other) <= radius_sq for other in reserved)


def _target_near_dynamic_robot(
    target: tuple[float, float],
    robot_states: list[RobotCoordinationState],
    robot_index: int,
    margin: float,
) -> bool:
    ego_radius = float(robot_states[robot_index].safety_radius)
    margin = max(float(margin), 0.0)
    for other_index, other in enumerate(robot_states):
        if other_index == robot_index:
            continue
        required = ego_radius + float(other.safety_radius) + margin
        if _distance_sq(target, other.xy) <= required * required:
            return True
    return False


def _distance_point_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    closest = (ax + t * dx, ay + t * dy)
    return _distance(point, closest)


def _target_near_other_routes(
    target: tuple[float, float],
    robot_states: list[RobotCoordinationState],
    robot_index: int,
    route_points_by_robot: list[list[tuple[float, float]]] | None,
    margin: float,
) -> bool:
    if not route_points_by_robot:
        return False
    ego_radius = float(robot_states[robot_index].safety_radius)
    for other_index, route in enumerate(route_points_by_robot):
        if other_index == robot_index or len(route) < 2:
            continue
        required = ego_radius + float(robot_states[other_index].safety_radius) + max(float(margin), 0.0)
        for start, end in zip(route[:-1], route[1:]):
            if _distance_point_to_segment(target, start, end) <= required:
                return True
    return False


def _segment_near_other_routes(
    start: tuple[float, float],
    end: tuple[float, float],
    robot_states: list[RobotCoordinationState],
    robot_index: int,
    route_points_by_robot: list[list[tuple[float, float]]] | None,
    margin: float,
    sample_spacing: float,
) -> bool:
    """Approximate whether a candidate travel corridor conflicts with teammates."""
    if not route_points_by_robot:
        return False

    ego_radius = float(robot_states[robot_index].safety_radius)
    samples = _sample_segment_points(start, end, sample_spacing)
    for other_index, route in enumerate(route_points_by_robot):
        if other_index == robot_index or len(route) < 2:
            continue
        required = ego_radius + float(robot_states[other_index].safety_radius) + max(float(margin), 0.0)
        for sample in samples:
            for seg_start, seg_end in zip(route[:-1], route[1:]):
                if _distance_point_to_segment(sample, seg_start, seg_end) <= required:
                    return True
    return False


def _dynamic_obstacles_for_robot(
    robot_states: list[RobotCoordinationState],
    robot_index: int,
) -> list[tuple[float, float, float]]:
    disks: list[tuple[float, float, float]] = []
    for other_index, other in enumerate(robot_states):
        if other_index == robot_index:
            continue
        disks.append((float(other.xy[0]), float(other.xy[1]), float(other.safety_radius)))
    return disks


def _candidate_score(planner_name: str, candidate: FrontierCandidate) -> float:
    """Planner-aware score used by the synchronized greedy assignment."""
    if planner_name == "Nearest frontier":
        return -float(candidate.distance_from_robot)
    if planner_name == "Largest frontier":
        return float(candidate.size) - 0.001 * float(candidate.distance_from_robot)
    return float(candidate.score)


def _coverage_aware_score(
    *,
    planner_name: str,
    candidate: FrontierCandidate,
    distance_penalty: float,
    route_reuse_penalty: float,
    route_conflict_penalty: float,
    spread_bonus: float,
) -> float:
    """Score that favors new information and discourages redundant motion.

    It does not replace A*/Dijkstra. It only selects a better frontier target:
    high information gain, reasonable travel cost, spatial separation from other
    assignments, and low expected overlap with teammate routes.
    """
    base = _candidate_score(planner_name, candidate)
    information_density = candidate.information_gain / max(candidate.distance_from_robot, 0.50)
    return (
        base
        + 0.45 * information_density
        + 0.10 * float(candidate.size)
        + spread_bonus
        - distance_penalty * float(candidate.distance_from_robot)
        - route_reuse_penalty
        - route_conflict_penalty
    )


def _build_point_index(
    points: Iterable[tuple[float, float]],
    cell_size: float,
) -> SpatialHash2D:
    index = SpatialHash2D(cell_size)
    index.extend(points)
    return index


def _route_samples_by_robot(
    route_points_by_robot: list[list[tuple[float, float]]] | None,
    sample_spacing: float,
    *,
    exclude_robot_index: int | None = None,
) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    if not route_points_by_robot:
        return samples
    for robot_index, route in enumerate(route_points_by_robot):
        if exclude_robot_index is not None and robot_index == exclude_robot_index:
            continue
        if len(route) < 2:
            continue
        for start, end in zip(route[:-1], route[1:]):
            samples.extend(_sample_segment_points(start, end, sample_spacing))
    return samples


def _segment_near_index(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    index: SpatialHash2D,
    radius: float,
    sample_spacing: float,
) -> bool:
    """Fast corridor check against pre-sampled route points."""
    for sample in _sample_segment_points(start, end, sample_spacing):
        if index.any_within(sample, radius):
            return True
    return False


def _corridor_index_hit_ratio(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    index: SpatialHash2D,
    radius: float,
    sample_spacing: float,
    ignore_start_distance: float = 0.0,
) -> float:
    """Return how much of a candidate corridor lies near indexed points.

    We use this to estimate route reuse. A robot necessarily starts inside an
    explored region, so samples near the robot are ignored; otherwise every
    candidate would be unfairly penalized at the beginning of the route.
    """
    samples = _sample_segment_points(start, end, sample_spacing)
    usable = [point for point in samples if _distance(start, point) > ignore_start_distance]
    if not usable:
        return 0.0
    hits = 0
    for point in usable:
        if index.any_within(point, radius):
            hits += 1
    return hits / len(usable)


def _route_crossing_penalty(
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    robot_states: list[RobotCoordinationState],
    robot_index: int,
    route_points_by_robot: list[list[tuple[float, float]]] | None,
    margin: float,
) -> float:
    """Soft penalty for candidate corridors that pass close to teammates.

    This is stronger than only rejecting targets near another robot. It looks at
    the whole straight corridor from the robot to the candidate frontier and
    penalizes proximity to active routes owned by other robots.
    """
    if not route_points_by_robot:
        return 0.0

    ego_radius = float(robot_states[robot_index].safety_radius)
    penalty = 0.0
    for other_index, route in enumerate(route_points_by_robot):
        if other_index == robot_index or len(route) < 2:
            continue
        required = ego_radius + float(robot_states[other_index].safety_radius) + max(float(margin), 0.0)
        for route_start, route_end in zip(route[:-1], route[1:]):
            # Sample the candidate corridor and measure closest approach to the
            # teammate's active route segment. This is cheap enough for a small
            # team and avoids adding a full segment-distance implementation.
            min_distance = min(
                _distance_point_to_segment(sample, route_start, route_end)
                for sample in _sample_segment_points(start, end, max(required * 0.50, 0.20))
            )
            if min_distance <= required:
                penalty += 18.0
            elif min_distance <= required * 2.0:
                penalty += 6.0 * (1.0 - (min_distance - required) / required)
    return penalty


def _anti_overlap_score(
    *,
    planner_name: str,
    candidate: FrontierCandidate,
    distance_penalty: float,
    explored_reuse_ratio: float,
    crossing_penalty: float,
    route_reuse_penalty: float,
    spread_bonus: float,
) -> float:
    """Score for our custom exploration-efficient coordinator.

    Main design goal:
        maximize new information while avoiding redundant travel through already
        explored corridors and reducing robot-robot route crossings.

    The score is intentionally interpretable rather than learned.
    """
    base = _candidate_score(planner_name, candidate)
    distance = max(float(candidate.distance_from_robot), 0.50)
    information_density = float(candidate.information_gain) / distance
    new_area_bias = 0.30 * float(candidate.information_gain) + 0.25 * information_density
    explored_reuse_penalty = explored_reuse_ratio * distance * 2.25
    return (
        base
        + new_area_bias
        + 0.08 * float(candidate.size)
        + spread_bonus
        - max(float(distance_penalty), 0.05) * distance
        - explored_reuse_penalty
        - crossing_penalty
        - route_reuse_penalty
    )


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Signed area / orientation test for three 2D points."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """Return True when two closed 2D line segments intersect."""
    eps = 1e-9

    def on_segment(p, q, r) -> bool:
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True
    return False


def _segment_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> float:
    """Minimum distance between two 2D segments."""
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _distance_point_to_segment(a, c, d),
        _distance_point_to_segment(b, c, d),
        _distance_point_to_segment(c, a, b),
        _distance_point_to_segment(d, a, b),
    )


def _team_center(robot_states: list[RobotCoordinationState]) -> tuple[float, float]:
    if not robot_states:
        return (0.0, 0.0)
    return (
        sum(state.xy[0] for state in robot_states) / len(robot_states),
        sum(state.xy[1] for state in robot_states) / len(robot_states),
    )


def _angle_from(center: tuple[float, float], point: tuple[float, float]) -> float:
    return math.atan2(point[1] - center[1], point[0] - center[0])


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _distributed_candidate_score(
    *,
    planner_name: str,
    candidate: FrontierCandidate,
    ipp_distance_penalty: float,
    team_center: tuple[float, float],
    robot_count: int,
) -> float:
    """Individual utility before pairwise coordination penalties.

    Design choice:
        Do not punish every meter through already explored space. A frontier is
        by definition reached from known free space, so over-penalizing explored
        corridors makes robots choose poor local targets. Instead, reward
        expected new information per meter and leave redundancy/crossing to the
        pairwise team assignment terms.
    """
    distance = max(float(candidate.distance_from_robot), 0.50)
    info_gain = float(candidate.information_gain)
    size = float(candidate.size)
    base = _candidate_score(planner_name, candidate)
    info_density = info_gain / distance

    # Targets farther from the team center usually expand a different frontier
    # lobe instead of keeping the whole team clustered around the same boundary.
    outward = _distance(tuple(candidate.target), team_center)
    outward_bonus = min(outward, 6.0) * (0.15 + 0.03 * max(robot_count - 1, 0))

    return (
        base
        + 0.70 * info_density
        + 0.22 * info_gain
        + 0.08 * size
        + outward_bonus
        - max(float(ipp_distance_penalty), 0.05) * distance
    )


def _pairwise_assignment_penalty(
    *,
    robot_i: int,
    target_i: tuple[float, float],
    robot_j: int,
    target_j: tuple[float, float],
    robot_states: list[RobotCoordinationState],
    effective_target_radius: float,
    dynamic_obstacle_margin: float,
    team_center: tuple[float, float],
) -> float | None:
    """Soft/hard compatibility between two robot-target assignments.

    Returns:
        None  -> pair is invalid and should not be assigned together.
        float -> non-negative penalty to subtract from the joint score.
    """
    start_i = robot_states[robot_i].xy
    start_j = robot_states[robot_j].xy
    ri = float(robot_states[robot_i].safety_radius)
    rj = float(robot_states[robot_j].safety_radius)
    clearance = ri + rj + max(float(dynamic_obstacle_margin), 0.0)

    target_distance = _distance(target_i, target_j)
    if target_distance < effective_target_radius:
        return None

    corridor_distance = _segment_distance(start_i, target_i, start_j, target_j)

    penalty = 0.0
    if corridor_distance < clearance * 1.20:
        # Soft, not hard. A*/dynamic obstacles can often route around this,
        # while a hard reject can leave robots with no assignment at startup.
        penalty += 26.0 * (1.0 - corridor_distance / max(clearance * 1.20, 1e-6))

    # Sensor-footprint overlap approximation. If the two assigned frontiers are
    # close compared with sensor range, the robots are likely to observe the same
    # unknown band and waste coverage.
    view_radius = 0.65 * min(
        float(robot_states[robot_i].sensor_range),
        float(robot_states[robot_j].sensor_range),
    )
    view_radius = max(view_radius, effective_target_radius)
    if target_distance < view_radius * 1.60:
        penalty += 22.0 * (1.0 - target_distance / (view_radius * 1.60))

    # Route crossing. We are not computing full A* paths here, but straight
    # intent corridors are a good early warning. Hard reject true intersections
    # unless the starts are almost co-located; otherwise apply a strong penalty
    # for near misses.
    starts_close = _distance(start_i, start_j) < clearance * 1.4
    if _segments_intersect(start_i, target_i, start_j, target_j) and not starts_close:
        penalty += 34.0
    if corridor_distance < clearance * 3.5:
        penalty += 12.0 * (1.0 - corridor_distance / (clearance * 3.5))

    # Angular diversity around the team center. This makes the assignment prefer
    # different frontier lobes instead of multiple targets on the same local arc.
    angle_i = _angle_from(team_center, target_i)
    angle_j = _angle_from(team_center, target_j)
    min_angle = math.pi / max(len(robot_states), 2)
    diff = _angle_diff(angle_i, angle_j)
    if diff < min_angle:
        penalty += 10.0 * (1.0 - diff / max(min_angle, 1e-6))

    return penalty


# ---------------------------------------------------------------------------
# NOIC helper functions
# ---------------------------------------------------------------------------

def _disk_samples(
    center: tuple[float, float],
    radius: float,
    spacing: float,
) -> list[tuple[float, float]]:
    """Sample a circular approximate future sensor footprint."""
    radius = max(float(radius), 0.0)
    spacing = max(float(spacing), 1e-6)
    if radius <= 1e-9:
        return [center]
    cx, cy = center
    samples: list[tuple[float, float]] = []
    steps = int(math.ceil(radius / spacing))
    for ix in range(-steps, steps + 1):
        for iy in range(-steps, steps + 1):
            x = cx + ix * spacing
            y = cy + iy * spacing
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius:
                samples.append((x, y))
    return samples or [center]


def _make_robot_explored_indices(
    explored_points_by_robot: list[list[tuple[float, float]]] | None,
    robot_count: int,
    resolution: float,
) -> list[SpatialHash2D]:
    """Build one spatial hash per robot observation layer."""
    cell_size = max(float(resolution), 0.25)
    indices = [SpatialHash2D(cell_size) for _ in range(max(int(robot_count), 0))]
    if not explored_points_by_robot:
        return indices
    for idx in range(min(len(indices), len(explored_points_by_robot))):
        for point in explored_points_by_robot[idx] or []:
            target = _normalize_target(point)
            if target is not None:
                indices[idx].add(target)
    return indices


def _index_hit_any(
    point: tuple[float, float],
    indices: list[SpatialHash2D],
    radius: float,
    *,
    exclude_index: int | None = None,
) -> bool:
    for idx, index in enumerate(indices):
        if exclude_index is not None and idx == exclude_index:
            continue
        if index.any_within(point, radius):
            return True
    return False


def _footprint_overlap_ratio(
    *,
    target: tuple[float, float],
    robot_index: int,
    robot_states: list[RobotCoordinationState],
    explored_indices: list[SpatialHash2D],
    resolution: float,
    teammate_only: bool,
) -> float:
    """Estimate how much future sensing would repeat already observed cells.

    This is only an approximation because the final heading at the frontier is
    not fixed yet. The coordinator uses a disk around the candidate target as a
    conservative footprint proxy. The local reactive layer may later choose the
    exact heading.
    """
    if not explored_indices:
        return 0.0
    sensor = float(robot_states[robot_index].sensor_range)
    radius = max(min(sensor * 0.75, 2.25), float(resolution) * 2.0)
    spacing = max(float(resolution), 0.50)
    hit_radius = max(float(resolution) * 0.75, 0.35)
    samples = _disk_samples(target, radius, spacing)
    if not samples:
        return 0.0
    hits = 0
    for sample in samples:
        if teammate_only:
            if _index_hit_any(sample, explored_indices, hit_radius, exclude_index=robot_index):
                hits += 1
        else:
            own_index = explored_indices[robot_index] if 0 <= robot_index < len(explored_indices) else None
            if own_index is not None and own_index.any_within(sample, hit_radius):
                hits += 1
    return hits / max(len(samples), 1)


def _corridor_teammate_mapped_ratio(
    *,
    start: tuple[float, float],
    target: tuple[float, float],
    robot_index: int,
    robot_states: list[RobotCoordinationState],
    explored_indices: list[SpatialHash2D],
    resolution: float,
) -> float:
    """Estimate how much of a travel corridor enters teammates' mapped zones."""
    if not explored_indices:
        return 0.0
    spacing = max(float(resolution), 0.35)
    hit_radius = max(float(resolution) * 0.75, 0.35)
    ignore_start = max(min(float(robot_states[robot_index].sensor_range) * 0.35, 1.25), 0.50)
    samples = [p for p in _sample_segment_points(start, target, spacing) if _distance(start, p) > ignore_start]
    if not samples:
        return 0.0
    hits = 0
    for sample in samples:
        if _index_hit_any(sample, explored_indices, hit_radius, exclude_index=robot_index):
            hits += 1
    return hits / max(len(samples), 1)


def _point_in_current_view(
    point: tuple[float, float],
    viewer: RobotCoordinationState,
) -> bool:
    """Approximate whether a point is currently inside another robot's sensor."""
    dist = _distance(point, viewer.xy)
    if dist > float(viewer.sensor_range):
        return False
    if "Camera" not in str(viewer.vision_model):
        return True
    angle = math.atan2(point[1] - viewer.xy[1], point[0] - viewer.xy[0])
    diff = _angle_diff(angle, float(viewer.theta))
    return diff <= math.radians(35.0)


def _teammate_current_view_penalty(
    *,
    target: tuple[float, float],
    robot_index: int,
    robot_states: list[RobotCoordinationState],
) -> float:
    """Penalty when a candidate sends a robot into a teammate's current viewer."""
    penalty = 0.0
    for other_index, other in enumerate(robot_states):
        if other_index == robot_index:
            continue
        if _point_in_current_view(target, other):
            # Strong but not hard: if the map has only one viable frontier, the
            # solver may still accept it instead of freezing the whole team.
            dist = max(_distance(target, other.xy), 0.10)
            penalty += 18.0 * (1.0 - min(dist / max(float(other.sensor_range), 1e-6), 1.0)) + 6.0
    return penalty


def _noic_score_components(
    *,
    planner_name: str,
    candidate: FrontierCandidate,
    robot_index: int,
    robot_states: list[RobotCoordinationState],
    explored_indices: list[SpatialHash2D],
    route_points_by_robot: list[list[tuple[float, float]]] | None,
    team_center: tuple[float, float],
    ipp_distance_penalty: float,
    resolution: float,
    dynamic_obstacle_margin: float,
) -> tuple[float, dict[str, float]]:
    """Return NOIC score and interpretable components.

    Mathematical objective implemented here:

        J_i(F) = a IG_i(F) + b IG_i(F)/d_i(F) + c S_i(F)
                 - l d_i(F)
                 - g M^-i(F)
                 - r C^-i(F)
                 - o V^-i(F)
                 - x X_i(F)

    where M^-i is teammate-mapped footprint ratio, C^-i is teammate-mapped
    corridor ratio, V^-i is current teammate viewer overlap, and X_i is route
    crossing penalty.
    """
    target = tuple(candidate.target)
    start = robot_states[robot_index].xy
    distance = max(float(candidate.distance_from_robot), 0.50)
    info_gain = float(candidate.information_gain)
    size = float(candidate.size)
    base = _candidate_score(planner_name, candidate)
    info_density = info_gain / distance
    outward = _distance(target, team_center)
    outward_bonus = min(outward, 6.0) * 0.20

    teammate_map_ratio = _footprint_overlap_ratio(
        target=target,
        robot_index=robot_index,
        robot_states=robot_states,
        explored_indices=explored_indices,
        resolution=resolution,
        teammate_only=True,
    )
    own_revisit_ratio = _footprint_overlap_ratio(
        target=target,
        robot_index=robot_index,
        robot_states=robot_states,
        explored_indices=explored_indices,
        resolution=resolution,
        teammate_only=False,
    )
    teammate_corridor_ratio = _corridor_teammate_mapped_ratio(
        start=start,
        target=target,
        robot_index=robot_index,
        robot_states=robot_states,
        explored_indices=explored_indices,
        resolution=resolution,
    )
    current_view_penalty = _teammate_current_view_penalty(
        target=target,
        robot_index=robot_index,
        robot_states=robot_states,
    )
    crossing_penalty = _route_crossing_penalty(
        start=start,
        end=target,
        robot_states=robot_states,
        robot_index=robot_index,
        route_points_by_robot=route_points_by_robot,
        margin=dynamic_obstacle_margin,
    )

    score = (
        base
        + 0.34 * info_gain
        + 0.85 * info_density
        + 0.06 * size
        + outward_bonus
        - max(float(ipp_distance_penalty), 0.05) * distance
        # These are deliberately soft penalties. The previous NOIC draft made
        # them too strong and the global optimizer preferred holding robots
        # over assigning imperfect but still useful frontiers.
        - 18.0 * teammate_map_ratio
        - 5.0 * own_revisit_ratio
        - 14.0 * teammate_corridor_ratio
        - 0.85 * current_view_penalty
        - 0.50 * crossing_penalty
    )
    components = {
        "IG": info_gain,
        "d": distance,
        "other_map": teammate_map_ratio,
        "own_revisit": own_revisit_ratio,
        "other_corridor": teammate_corridor_ratio,
        "other_view_penalty": current_view_penalty,
        "route_cross": crossing_penalty,
        "J": score,
    }
    return score, components


class LegacyNoicCoordinatorAlgorithm:
    """Current NOIC/coordinated-frontier behavior behind an algorithm boundary.

    This class intentionally preserves the existing behavior and output format.
    It is the first internal algorithm implementation consumed by
    MultiRobotCoordinator, which now acts as a host instead of owning the whole
    coordination strategy directly.
    """

    name = NOIC_COORDINATOR

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        count = len(request.robot_states)
        targets: list[tuple[float, float] | None] = [
            _normalize_target(request.existing_targets[index])
            if index < len(request.existing_targets)
            else None
            for index in range(count)
        ]
        reasons: list[str] = [
            "kept existing target" if targets[index] is not None else "no target assigned yet"
            for index in range(count)
        ]

        assign_set = {int(index) for index in request.robots_to_assign if 0 <= int(index) < count}
        if not assign_set:
            return CoordinationResult(tuple(targets), tuple(reasons), self.name)

        planner_res = assign_frontier_viewpoints(
            robot_states=request.robot_states,
            existing_targets=targets,
            robots_to_assign=sorted(assign_set),
            invalidated_targets_by_robot=request.invalidated_targets_by_robot,
            explored_points=request.explored_points,
            mapped_obstacle_points=request.mapped_obstacle_points,
            bounds=request.bounds,
            resolution=request.resolution,
            final_goal_xy=request.final_goal_xy or (0.0, 0.0),
            ipp_distance_penalty=request.ipp_distance_penalty,
            target_exclusion_radius=request.target_exclusion_radius,
            dynamic_obstacle_margin=request.dynamic_obstacle_margin,
            route_points_by_robot=request.route_points_by_robot,
            explored_points_by_robot=request.explored_points_by_robot,
        )

        assigned_indices: set[int] = set()
        for idx, assignment in enumerate(planner_res.assignments):
            if idx in assign_set and assignment is not None:
                targets[idx] = assignment.target
                reasons[idx] = (
                    f"{self.name}: asignación coordinada; {assignment.reason}; "
                    f"info_gain={assignment.information_gain:.1f}; dist={assignment.distance:.1f}; "
                    f"map_overlap={assignment.other_map_ratio:.2f}; "
                    f"route_overlap={assignment.route_overlap_ratio:.2f}"
                )
                assigned_indices.add(idx)

        # Tactical hold: if the modern planner rejects every candidate for a robot,
        # keep it idle instead of forcing duplicated sensing or route conflicts.
        for idx in sorted(assign_set - assigned_indices):
            targets[idx] = None
            if idx < len(planner_res.reasons) and planner_res.reasons[idx] != "no target assigned yet":
                reasons[idx] = (
                    f"{self.name}: En espera (HOLD) para evitar solapamiento de visión "
                    f"o rutas cruzadas; {planner_res.reasons[idx]}"
                )
            else:
                reasons[idx] = (
                    f"{self.name}: En espera (HOLD) para evitar solapamiento de visión "
                    f"o rutas cruzadas"
                )

        return CoordinationResult(tuple(targets), tuple(reasons), self.name)


COORDINATION_ALGORITHM_REGISTRY: dict[str, type[CoordinationAlgorithm]] = {
    NOIC_COORDINATOR: LegacyNoicCoordinatorAlgorithm,
}


class MultiRobotCoordinator:
    """Assign frontier targets to multiple robots.

    Available strategies:
        Independent frontiers:
            each robot chooses its own best frontier with no target reservation.
            Useful as a baseline, but can duplicate targets.

        Reserved frontiers:
            robots are processed sequentially. Earlier assignments reserve a
            radius around their targets so later robots do not duplicate them.

        Synchronized greedy:
            builds robot-frontier candidate pairs and greedily assigns the best
            non-conflicting pairs. This is the recommended baseline.

        Coverage-aware greedy:
            prefers high information-gain frontiers and rejects obvious route
            conflicts with teammate routes.

        Anti-overlap greedy:
            our strongest interpretable coordinator. It penalizes long travel
            through already explored area, avoids crossing teammate corridors,
            and spreads robots toward frontiers with high expected new coverage.
    """

    def __init__(self, strategy: str = DEFAULT_COORDINATOR):
        self.strategy = strategy if strategy in COORDINATOR_OPTIONS else DEFAULT_COORDINATOR
        algorithm_cls = COORDINATION_ALGORITHM_REGISTRY[self.strategy]
        self.algorithm: CoordinationAlgorithm = algorithm_cls()

    def assign_frontiers(
        self,
        *,
        planner_name: str,
        robot_states: list[RobotCoordinationState],
        existing_targets: list[tuple[float, float] | None],
        robots_to_assign: list[int],
        invalidated_targets_by_robot: list[list[tuple[float, float]]] | None,
        explored_points: list[tuple[float, float]],
        mapped_obstacle_points: list[tuple[float, float]],
        bounds: tuple[float, float, float, float],
        resolution: float,
        final_goal_xy: tuple[float, float] | None = None,
        ipp_distance_penalty: float = 0.5,
        target_exclusion_radius: float = 1.5,
        dynamic_obstacle_margin: float = 0.5,
        route_points_by_robot: list[list[tuple[float, float]]] | None = None,
        explored_points_by_robot: list[list[tuple[float, float]]] | None = None,
    ) -> CoordinationResult:
        """Assign coordinated frontier targets through the active algorithm.

        The public signature stays compatible with engine.py.  Internally the
        arguments are normalized into CoordinationRequest and passed to the
        selected CoordinationAlgorithm.  For now, the only registered algorithm
        is the current NOIC/coordinated-frontier behavior.
        """
        request = CoordinationRequest(
            planner_name=planner_name,
            robot_states=robot_states,
            existing_targets=existing_targets,
            robots_to_assign=robots_to_assign,
            invalidated_targets_by_robot=invalidated_targets_by_robot,
            explored_points=explored_points,
            mapped_obstacle_points=mapped_obstacle_points,
            bounds=bounds,
            resolution=resolution,
            final_goal_xy=final_goal_xy,
            ipp_distance_penalty=ipp_distance_penalty,
            target_exclusion_radius=target_exclusion_radius,
            dynamic_obstacle_margin=dynamic_obstacle_margin,
            route_points_by_robot=route_points_by_robot,
            explored_points_by_robot=explored_points_by_robot,
        )
        return self.algorithm.assign(request)

    def _invalidated_for(self, invalidated_targets_by_robot, index: int) -> list[tuple[float, float]]:
        if not invalidated_targets_by_robot or index >= len(invalidated_targets_by_robot):
            return []
        return _target_list(invalidated_targets_by_robot[index])

    def _select_single_goal(
        self,
        *,
        planner_name: str,
        robot_state: RobotCoordinationState,
        robot_index: int,
        excluded_targets: list[tuple[float, float]],
        robot_states: list[RobotCoordinationState],
        explored_points: list[tuple[float, float]],
        mapped_obstacle_points: list[tuple[float, float]],
        bounds: tuple[float, float, float, float],
        resolution: float,
        final_goal_xy: tuple[float, float],
        ipp_distance_penalty: float,
        target_exclusion_radius: float,
        dynamic_obstacle_margin: float,
        use_dynamic_obstacles: bool,
    ):
        return select_exploration_goal(
            planner_name,
            robot_xy=robot_state.xy,
            final_goal_xy=final_goal_xy,
            explored_points=explored_points,
            mapped_obstacle_points=mapped_obstacle_points,
            bounds=bounds,
            resolution=resolution,
            robot_radius=robot_state.safety_radius,
            sensor_range=robot_state.sensor_range,
            vision_model=robot_state.vision_model,
            ipp_distance_penalty=ipp_distance_penalty,
            excluded_targets=excluded_targets,
            target_exclusion_radius=target_exclusion_radius,
            dynamic_obstacles=(
                _dynamic_obstacles_for_robot(robot_states, robot_index)
                if use_dynamic_obstacles
                else []
            ),
            dynamic_obstacle_margin=dynamic_obstacle_margin,
        )

    def _assign_noic(self, **kwargs) -> CoordinationResult:
        """NOIC compatibility entry point backed by the coordinated frontier planner.

        The old NOIC implementation tried to repair frontiers after the
        exploration planner had already selected them.  That created HOLD loops
        and repeated reassignment of the same weak candidates.  The new behavior
        delegates frontier selection to one synchronized planner that detects
        global frontiers once and assigns robot-viewpoint pairs as a team.
        """
        result = assign_frontier_viewpoints(
            robot_states=kwargs["robot_states"],
            existing_targets=kwargs["targets"],
            robots_to_assign=sorted(set(kwargs["assign_set"])),
            invalidated_targets_by_robot=kwargs.get("invalidated_targets_by_robot"),
            explored_points=kwargs["explored_points"],
            mapped_obstacle_points=kwargs["mapped_obstacle_points"],
            bounds=kwargs["bounds"],
            resolution=kwargs["resolution"],
            final_goal_xy=kwargs["final_goal_xy"],
            ipp_distance_penalty=kwargs["ipp_distance_penalty"],
            target_exclusion_radius=kwargs["target_exclusion_radius"],
            dynamic_obstacle_margin=kwargs["dynamic_obstacle_margin"],
            route_points_by_robot=kwargs.get("route_points_by_robot"),
            explored_points_by_robot=kwargs.get("explored_points_by_robot"),
        )
        return CoordinationResult(result.targets, result.reasons, self.strategy)

    def _assign_independent(self, **kwargs) -> CoordinationResult:
        targets = kwargs["targets"]
        reasons = kwargs["reasons"]
        for index in sorted(kwargs["assign_set"]):
            excluded = self._invalidated_for(kwargs["invalidated_targets_by_robot"], index)
            result = self._select_single_goal(
                planner_name=kwargs["planner_name"],
                robot_state=kwargs["robot_states"][index],
                robot_index=index,
                excluded_targets=excluded,
                robot_states=kwargs["robot_states"],
                explored_points=kwargs["explored_points"],
                mapped_obstacle_points=kwargs["mapped_obstacle_points"],
                bounds=kwargs["bounds"],
                resolution=kwargs["resolution"],
                final_goal_xy=kwargs["final_goal_xy"],
                ipp_distance_penalty=kwargs["ipp_distance_penalty"],
                target_exclusion_radius=kwargs.get("target_exclusion_radius", 0.0),
                dynamic_obstacle_margin=kwargs.get("dynamic_obstacle_margin", 0.0),
                use_dynamic_obstacles=False,
            )
            if result.success:
                targets[index] = tuple(result.target)
                reasons[index] = f"Independent frontiers: {result.reason}"
            else:
                targets[index] = None
                reasons[index] = f"Independent frontiers: {result.reason}"
        return CoordinationResult(tuple(targets), tuple(reasons), self.strategy)

    def _assign_reserved(self, **kwargs) -> CoordinationResult:
        targets = kwargs["targets"]
        reasons = kwargs["reasons"]
        reserved = [target for i, target in enumerate(targets) if i not in kwargs["assign_set"] and target is not None]
        for index in sorted(kwargs["assign_set"]):
            excluded = reserved + self._invalidated_for(kwargs["invalidated_targets_by_robot"], index)
            result = self._select_single_goal(
                planner_name=kwargs["planner_name"],
                robot_state=kwargs["robot_states"][index],
                robot_index=index,
                excluded_targets=excluded,
                robot_states=kwargs["robot_states"],
                explored_points=kwargs["explored_points"],
                mapped_obstacle_points=kwargs["mapped_obstacle_points"],
                bounds=kwargs["bounds"],
                resolution=kwargs["resolution"],
                final_goal_xy=kwargs["final_goal_xy"],
                ipp_distance_penalty=kwargs["ipp_distance_penalty"],
                target_exclusion_radius=kwargs["target_exclusion_radius"],
                dynamic_obstacle_margin=kwargs["dynamic_obstacle_margin"],
                use_dynamic_obstacles=True,
            )
            if result.success:
                target = tuple(result.target)
                targets[index] = target
                reserved.append(target)
                reasons[index] = f"Reserved frontiers: {result.reason}"
            else:
                targets[index] = None
                reasons[index] = f"Reserved frontiers: {result.reason}"
        return CoordinationResult(tuple(targets), tuple(reasons), self.strategy)

    def _assign_synchronized_greedy(self, **kwargs) -> CoordinationResult:
        targets = kwargs["targets"]
        reasons = kwargs["reasons"]
        robot_states = kwargs["robot_states"]
        assign_set = set(kwargs["assign_set"])
        radius = float(kwargs["target_exclusion_radius"])
        margin = float(kwargs["dynamic_obstacle_margin"])
        route_points_by_robot = kwargs.get("route_points_by_robot")

        reserved: list[tuple[float, float]] = [
            target for index, target in enumerate(targets)
            if index not in assign_set and target is not None
        ]

        pairs: list[tuple[float, int, FrontierCandidate]] = []
        for index in sorted(assign_set):
            excluded = reserved + self._invalidated_for(kwargs["invalidated_targets_by_robot"], index)
            result = self._select_single_goal(
                planner_name=kwargs["planner_name"],
                robot_state=robot_states[index],
                robot_index=index,
                excluded_targets=excluded,
                robot_states=robot_states,
                explored_points=kwargs["explored_points"],
                mapped_obstacle_points=kwargs["mapped_obstacle_points"],
                bounds=kwargs["bounds"],
                resolution=kwargs["resolution"],
                final_goal_xy=kwargs["final_goal_xy"],
                ipp_distance_penalty=kwargs["ipp_distance_penalty"],
                target_exclusion_radius=radius,
                dynamic_obstacle_margin=margin,
                use_dynamic_obstacles=True,
            )
            if not result.candidates:
                targets[index] = None
                reasons[index] = f"Synchronized greedy: {result.reason}"
                continue
            for candidate in result.candidates:
                pairs.append((_candidate_score(kwargs["planner_name"], candidate), index, candidate))

        assigned_robots: set[int] = set()
        pairs.sort(key=lambda item: (item[0], item[2].information_gain, item[2].size, -item[2].distance_from_robot), reverse=True)

        for _, index, candidate in pairs:
            if index in assigned_robots:
                continue
            target = tuple(candidate.target)
            if _target_near_reserved(target, reserved, radius):
                continue
            if _target_near_dynamic_robot(target, robot_states, index, margin):
                continue
            if _target_near_other_routes(target, robot_states, index, route_points_by_robot, margin):
                continue

            targets[index] = target
            reserved.append(target)
            assigned_robots.add(index)
            reasons[index] = f"Synchronized greedy: {candidate.reason}"

        for index in sorted(assign_set - assigned_robots):
            if targets[index] is None:
                reasons[index] = "Synchronized greedy: no non-conflicting frontier available"

        return CoordinationResult(tuple(targets), tuple(reasons), self.strategy)


    def _assign_coverage_aware_greedy(self, **kwargs) -> CoordinationResult:
        """Assign targets with information gain, dispersion, and route-awareness.

        Compared with Synchronized greedy, this strategy is stricter:
            - frontiers are reserved by radius;
            - candidate corridors are rejected if they cross teammate routes;
            - candidates near active routes are penalized;
            - candidates with high information gain per meter are preferred;
            - a heap avoids globally sorting all robot-frontier pairs.

        It is still a baseline, not an optimal assignment solver. The goal is to
        produce cleaner exploration behavior without introducing MARL or a heavy
        combinatorial optimizer.
        """
        import heapq

        targets = kwargs["targets"]
        reasons = kwargs["reasons"]
        robot_states = kwargs["robot_states"]
        assign_set = set(kwargs["assign_set"])
        radius = float(kwargs["target_exclusion_radius"])
        margin = float(kwargs["dynamic_obstacle_margin"])
        resolution = max(float(kwargs["resolution"]), 1e-6)
        route_points_by_robot = kwargs.get("route_points_by_robot")

        sample_spacing = max(resolution * 0.75, 0.20)
        reserved: list[tuple[float, float]] = [
            target for index, target in enumerate(targets)
            if index not in assign_set and target is not None
        ]

        # Spatial indexes replace repeated linear scans in the hot assignment loop.
        reserved_index = _build_point_index(reserved, max(radius, resolution))
        route_indices_by_robot: list[SpatialHash2D] = []
        for robot_index in range(len(robot_states)):
            route_samples = _route_samples_by_robot(
                route_points_by_robot,
                sample_spacing,
                exclude_robot_index=robot_index,
            )
            route_indices_by_robot.append(
                _build_point_index(route_samples, max(sample_spacing, resolution))
            )

        # Heap items are (-score, tie_breaker, robot_index, candidate). We use a
        # heap instead of sorting the full pair list so only the best available
        # candidates are popped as needed.
        heap: list[tuple[float, int, int, FrontierCandidate]] = []
        tie = 0

        for index in sorted(assign_set):
            excluded = reserved + self._invalidated_for(kwargs["invalidated_targets_by_robot"], index)
            result = self._select_single_goal(
                planner_name=kwargs["planner_name"],
                robot_state=robot_states[index],
                robot_index=index,
                excluded_targets=excluded,
                robot_states=robot_states,
                explored_points=kwargs["explored_points"],
                mapped_obstacle_points=kwargs["mapped_obstacle_points"],
                bounds=kwargs["bounds"],
                resolution=resolution,
                final_goal_xy=kwargs["final_goal_xy"],
                ipp_distance_penalty=kwargs["ipp_distance_penalty"],
                target_exclusion_radius=radius,
                dynamic_obstacle_margin=margin,
                use_dynamic_obstacles=True,
            )
            if not result.candidates:
                targets[index] = None
                reasons[index] = f"Coverage-aware greedy: {result.reason}"
                continue

            for candidate in result.candidates:
                target = tuple(candidate.target)
                start = robot_states[index].xy

                # Hard dynamic rejections before adding to the heap.
                if reserved_index.any_within(target, radius):
                    continue
                if _target_near_dynamic_robot(target, robot_states, index, margin):
                    continue
                route_clearance = (
                    robot_states[index].safety_radius
                    + max(
                        (state.safety_radius for other_i, state in enumerate(robot_states) if other_i != index),
                        default=0.0,
                    )
                    + margin
                )
                if _segment_near_index(
                    start=start,
                    end=target,
                    index=route_indices_by_robot[index],
                    radius=route_clearance,
                    sample_spacing=sample_spacing,
                ):
                    continue

                # Soft penalties. These do not reject the candidate; they only
                # make redundant/overlapping frontiers less attractive.
                route_reuse_penalty = 0.0
                if route_indices_by_robot[index].any_within(
                    target,
                    robot_states[index].safety_radius + margin + resolution,
                ):
                    route_reuse_penalty += 4.0

                # Encourage spatial spread between robots/frontiers. If no
                # target is reserved yet, this term is neutral.
                spread_bonus = 0.0
                if reserved:
                    nearest_reserved = min(_distance(target, other) for other in reserved)
                    spread_bonus = min(nearest_reserved, radius * 2.0) * 0.20

                # Penalize targets too close to current robot positions, except
                # the robot that owns the candidate. This helps reduce crossing
                # into occupied zones even before exact collision checking.
                route_conflict_penalty = 0.0
                for other_index, other in enumerate(robot_states):
                    if other_index == index:
                        continue
                    required = robot_states[index].safety_radius + other.safety_radius + margin
                    dist = _distance(target, other.xy)
                    if dist < required + resolution:
                        route_conflict_penalty += 10.0

                score = _coverage_aware_score(
                    planner_name=kwargs["planner_name"],
                    candidate=candidate,
                    distance_penalty=max(float(kwargs["ipp_distance_penalty"]), 0.05),
                    route_reuse_penalty=route_reuse_penalty,
                    route_conflict_penalty=route_conflict_penalty,
                    spread_bonus=spread_bonus,
                )
                heapq.heappush(heap, (-score, tie, index, candidate))
                tie += 1

        assigned_robots: set[int] = set()

        while heap and len(assigned_robots) < len(assign_set):
            negative_score, _, index, candidate = heapq.heappop(heap)
            if index in assigned_robots:
                continue

            target = tuple(candidate.target)
            start = robot_states[index].xy

            # Re-check after each assignment because reserved targets change.
            if reserved_index.any_within(target, radius):
                continue
            if _target_near_dynamic_robot(target, robot_states, index, margin):
                continue
            route_clearance = (
                robot_states[index].safety_radius
                + max(
                    (state.safety_radius for other_i, state in enumerate(robot_states) if other_i != index),
                    default=0.0,
                )
                + margin
            )
            if _segment_near_index(
                start=start,
                end=target,
                index=route_indices_by_robot[index],
                radius=route_clearance,
                sample_spacing=sample_spacing,
            ):
                continue

            targets[index] = target
            reserved.append(target)
            reserved_index.add(target)
            assigned_robots.add(index)
            reasons[index] = (
                f"Coverage-aware greedy: {candidate.reason}; "
                f"assignment_score={-negative_score:.2f}"
            )

        for index in sorted(assign_set - assigned_robots):
            if targets[index] is None:
                reasons[index] = "Coverage-aware greedy: no non-conflicting informative frontier available"

        return CoordinationResult(tuple(targets), tuple(reasons), self.strategy)

    def _assign_anti_overlap_greedy(self, **kwargs) -> CoordinationResult:
        """Assign frontiers while penalizing revisits and route crossings.

        This is our custom coordinator for the current simulator. It is not a
        neural MARL policy; it is an interpretable multi-objective heuristic:

            score = frontier utility
                    + expected new information
                    + team spread bonus
                    - travel cost
                    - explored corridor reuse
                    - teammate route crossing risk

        Compared with Coverage-aware greedy, this version explicitly estimates
        how much of the robot->frontier corridor lies in already explored space.
        Long corridors through already explored cells are discouraged, so robots
        are pushed toward nearby expansion boundaries instead of repeatedly
        crossing mapped regions.
        """
        import heapq

        targets = kwargs["targets"]
        reasons = kwargs["reasons"]
        robot_states = kwargs["robot_states"]
        assign_set = set(kwargs["assign_set"])
        radius = float(kwargs["target_exclusion_radius"])
        margin = float(kwargs["dynamic_obstacle_margin"])
        resolution = max(float(kwargs["resolution"]), 1e-6)
        route_points_by_robot = kwargs.get("route_points_by_robot")

        sample_spacing = max(resolution * 0.50, 0.18)
        reserved: list[tuple[float, float]] = [
            target for index, target in enumerate(targets)
            if index not in assign_set and target is not None
        ]

        reserved_index = _build_point_index(reserved, max(radius, resolution))

        # Shared explored-area index. This lets us estimate whether a candidate
        # requires traveling through a long corridor that is already known.
        explored_index = _build_point_index(
            ((float(x), float(y)) for x, y in kwargs["explored_points"]),
            max(resolution, 0.20),
        )

        route_indices_by_robot: list[SpatialHash2D] = []
        for robot_index in range(len(robot_states)):
            route_samples = _route_samples_by_robot(
                route_points_by_robot,
                sample_spacing,
                exclude_robot_index=robot_index,
            )
            route_indices_by_robot.append(
                _build_point_index(route_samples, max(sample_spacing, resolution))
            )

        heap: list[tuple[float, int, int, FrontierCandidate]] = []
        tie = 0

        for index in sorted(assign_set):
            excluded = reserved + self._invalidated_for(kwargs["invalidated_targets_by_robot"], index)
            result = self._select_single_goal(
                planner_name=kwargs["planner_name"],
                robot_state=robot_states[index],
                robot_index=index,
                excluded_targets=excluded,
                robot_states=robot_states,
                explored_points=kwargs["explored_points"],
                mapped_obstacle_points=kwargs["mapped_obstacle_points"],
                bounds=kwargs["bounds"],
                resolution=resolution,
                final_goal_xy=kwargs["final_goal_xy"],
                ipp_distance_penalty=kwargs["ipp_distance_penalty"],
                target_exclusion_radius=radius,
                dynamic_obstacle_margin=margin,
                use_dynamic_obstacles=True,
            )
            if not result.candidates:
                targets[index] = None
                reasons[index] = f"Anti-overlap greedy: {result.reason}"
                continue

            start = robot_states[index].xy
            ego_radius = float(robot_states[index].safety_radius)
            max_other_radius = max(
                (state.safety_radius for other_i, state in enumerate(robot_states) if other_i != index),
                default=0.0,
            )
            route_clearance = ego_radius + float(max_other_radius) + margin

            for candidate in result.candidates:
                target = tuple(candidate.target)

                # Hard safety filters. These keep the target itself and the
                # straight intent corridor away from teammates and their routes.
                if reserved_index.any_within(target, radius):
                    continue
                if _target_near_dynamic_robot(target, robot_states, index, margin):
                    continue
                if _segment_near_index(
                    start=start,
                    end=target,
                    index=route_indices_by_robot[index],
                    radius=route_clearance,
                    sample_spacing=sample_spacing,
                ):
                    continue

                explored_reuse_ratio = _corridor_index_hit_ratio(
                    start=start,
                    end=target,
                    index=explored_index,
                    radius=max(resolution * 0.90, 0.25),
                    sample_spacing=sample_spacing,
                    ignore_start_distance=max(robot_states[index].sensor_range * 0.35, ego_radius * 2.0),
                )

                crossing_penalty = _route_crossing_penalty(
                    start=start,
                    end=target,
                    robot_states=robot_states,
                    robot_index=index,
                    route_points_by_robot=route_points_by_robot,
                    margin=margin,
                )

                route_reuse_penalty = 0.0
                if route_indices_by_robot[index].any_within(target, route_clearance + resolution):
                    route_reuse_penalty += 8.0

                # Spread robots spatially. This term grows when the candidate is
                # far from already reserved frontiers, up to a bounded bonus.
                spread_bonus = 0.0
                if reserved:
                    nearest_reserved = min(_distance(target, other) for other in reserved)
                    spread_bonus = min(nearest_reserved, radius * 2.5) * 0.35

                score = _anti_overlap_score(
                    planner_name=kwargs["planner_name"],
                    candidate=candidate,
                    distance_penalty=max(float(kwargs["ipp_distance_penalty"]), 0.05),
                    explored_reuse_ratio=explored_reuse_ratio,
                    crossing_penalty=crossing_penalty,
                    route_reuse_penalty=route_reuse_penalty,
                    spread_bonus=spread_bonus,
                )
                heapq.heappush(heap, (-score, tie, index, candidate))
                tie += 1

        assigned_robots: set[int] = set()

        while heap and len(assigned_robots) < len(assign_set):
            negative_score, _, index, candidate = heapq.heappop(heap)
            if index in assigned_robots:
                continue

            target = tuple(candidate.target)
            start = robot_states[index].xy
            ego_radius = float(robot_states[index].safety_radius)
            max_other_radius = max(
                (state.safety_radius for other_i, state in enumerate(robot_states) if other_i != index),
                default=0.0,
            )
            route_clearance = ego_radius + float(max_other_radius) + margin

            # Re-check after each assignment because reserved targets changed.
            if reserved_index.any_within(target, radius):
                continue
            if _target_near_dynamic_robot(target, robot_states, index, margin):
                continue
            if _segment_near_index(
                start=start,
                end=target,
                index=route_indices_by_robot[index],
                radius=route_clearance,
                sample_spacing=sample_spacing,
            ):
                continue

            targets[index] = target
            reserved.append(target)
            reserved_index.add(target)
            assigned_robots.add(index)
            reasons[index] = (
                f"Anti-overlap greedy: {candidate.reason}; "
                f"assignment_score={-negative_score:.2f}"
            )

        for index in sorted(assign_set - assigned_robots):
            if targets[index] is None:
                reasons[index] = "Anti-overlap greedy: no non-overlapping informative frontier available"

        return CoordinationResult(tuple(targets), tuple(reasons), self.strategy)



    def _assign_distributed_auction(self, **kwargs) -> CoordinationResult:
        """Joint frontier assignment with dispersion and anti-crossing.

        This replaces the earlier anti-overlap behavior. The previous version
        punished travel through explored space too aggressively. In frontier
        exploration, a robot must usually travel through known free space to
        reach the boundary; over-penalizing that made the team inefficient.

        This strategy instead solves a small joint assignment problem:
            1. collect top frontier candidates for each robot;
            2. score individual information gain / distance;
            3. add pairwise penalties for duplicated sensor footprints, target
               clustering, and route crossings;
            4. choose the best compatible combination.

        It is still interpretable and dependency-free, but it behaves more like
        a coordinated team assignment than independent greedy target picking.
        """
        targets = kwargs["targets"]
        reasons = kwargs["reasons"]
        robot_states = kwargs["robot_states"]
        assign_set = set(kwargs["assign_set"])
        resolution = max(float(kwargs["resolution"]), 1e-6)
        base_radius = float(kwargs["target_exclusion_radius"])
        margin = float(kwargs["dynamic_obstacle_margin"])
        route_points_by_robot = kwargs.get("route_points_by_robot")
        team_center = _team_center(robot_states)

        if not assign_set:
            return CoordinationResult(tuple(targets), tuple(reasons), self.strategy)

        avg_sensor = (
            sum(float(state.sensor_range) for state in robot_states) / max(len(robot_states), 1)
        )
        # This is the critical change: reservation should approximate sensor
        # overlap, not only exact frontier equality. Otherwise F1/F2/F3 can be
        # technically different but still explore the same local patch.
        effective_target_radius = max(
            base_radius,
            resolution * 3.0,
            min(avg_sensor * 0.65, 3.25),
        )

        existing_reserved: list[tuple[float, float]] = [
            target for index, target in enumerate(targets)
            if index not in assign_set and target is not None
        ]

        # If existing routes belong to robots that are not being reassigned,
        # avoid sending new targets through them.
        static_route_samples = _route_samples_by_robot(
            route_points_by_robot,
            sample_spacing=max(resolution * 0.75, 0.20),
            exclude_robot_index=None,
        )
        route_index = _build_point_index(static_route_samples, max(resolution, 0.25))

        candidates_by_robot: dict[int, list[tuple[float, FrontierCandidate]]] = {}
        top_k = 12

        for index in sorted(assign_set):
            excluded = (
                existing_reserved
                + self._invalidated_for(kwargs["invalidated_targets_by_robot"], index)
            )
            result = self._select_single_goal(
                planner_name=kwargs["planner_name"],
                robot_state=robot_states[index],
                robot_index=index,
                excluded_targets=excluded,
                robot_states=robot_states,
                explored_points=kwargs["explored_points"],
                mapped_obstacle_points=kwargs["mapped_obstacle_points"],
                bounds=kwargs["bounds"],
                resolution=resolution,
                final_goal_xy=kwargs["final_goal_xy"],
                ipp_distance_penalty=kwargs["ipp_distance_penalty"],
                target_exclusion_radius=effective_target_radius,
                dynamic_obstacle_margin=margin,
                use_dynamic_obstacles=True,
            )

            scored: list[tuple[float, FrontierCandidate]] = []
            for candidate in result.candidates:
                target = tuple(candidate.target)
                start = robot_states[index].xy

                # Hard filters against already committed targets/routes.
                if _target_near_reserved(target, existing_reserved, effective_target_radius):
                    continue
                if _target_near_dynamic_robot(target, robot_states, index, margin):
                    continue

                ego_radius = float(robot_states[index].safety_radius)
                max_other_radius = max(
                    (state.safety_radius for other_i, state in enumerate(robot_states) if other_i != index),
                    default=0.0,
                )
                clearance = ego_radius + float(max_other_radius) + margin
                if route_index.any_within(target, clearance):
                    continue
                if _segment_near_index(
                    start=start,
                    end=target,
                    index=route_index,
                    radius=clearance,
                    sample_spacing=max(resolution * 0.75, 0.20),
                ):
                    continue

                score = _distributed_candidate_score(
                    planner_name=kwargs["planner_name"],
                    candidate=candidate,
                    ipp_distance_penalty=kwargs["ipp_distance_penalty"],
                    team_center=team_center,
                    robot_count=len(robot_states),
                )
                scored.append((score, candidate))

            scored.sort(
                key=lambda item: (
                    item[0],
                    item[1].information_gain,
                    item[1].size,
                    -item[1].distance_from_robot,
                ),
                reverse=True,
            )
            candidates_by_robot[index] = scored[:top_k]
            if not scored:
                targets[index] = None
                reasons[index] = f"{self.strategy}: no safe distributed frontier candidates"

        active_robots = [index for index in sorted(assign_set) if candidates_by_robot.get(index)]
        if not active_robots:
            return CoordinationResult(tuple(targets), tuple(reasons), self.strategy)

        # Assign robots with fewest options first. This reduces the chance that
        # an easy robot consumes the only viable region for a constrained robot.
        active_robots.sort(key=lambda idx: len(candidates_by_robot[idx]))

        # Pre-compute an optimistic upper bound for branch-and-bound.
        best_individual_by_robot = {
            idx: max(score for score, _ in candidates_by_robot[idx])
            for idx in active_robots
        }
        suffix_bound: list[float] = [0.0] * (len(active_robots) + 1)
        for pos in range(len(active_robots) - 1, -1, -1):
            suffix_bound[pos] = suffix_bound[pos + 1] + best_individual_by_robot[active_robots[pos]]

        best_score = -float("inf")
        best_assignment: dict[int, tuple[FrontierCandidate, float]] = {}

        def dfs(
            pos: int,
            current_score: float,
            assigned: dict[int, tuple[FrontierCandidate, float]],
        ) -> None:
            nonlocal best_score, best_assignment

            if current_score + suffix_bound[pos] <= best_score:
                return

            if pos >= len(active_robots):
                if current_score > best_score:
                    best_score = current_score
                    best_assignment = dict(assigned)
                return

            robot_index = active_robots[pos]

            # Option 1: hold position with a small penalty. This is better than
            # forcing two robots into the same frontier lobe when the map only
            # contains one useful opening.
            dfs(pos + 1, current_score - 3.0, assigned)

            for individual_score, candidate in candidates_by_robot[robot_index]:
                target = tuple(candidate.target)

                if _target_near_reserved(target, existing_reserved, effective_target_radius):
                    continue

                pair_penalty = 0.0
                valid = True
                for other_robot, (other_candidate, _) in assigned.items():
                    penalty = _pairwise_assignment_penalty(
                        robot_i=robot_index,
                        target_i=target,
                        robot_j=other_robot,
                        target_j=tuple(other_candidate.target),
                        robot_states=robot_states,
                        effective_target_radius=effective_target_radius,
                        dynamic_obstacle_margin=margin,
                        team_center=team_center,
                    )
                    if penalty is None:
                        valid = False
                        break
                    pair_penalty += penalty

                if not valid:
                    continue

                assigned[robot_index] = (candidate, individual_score - pair_penalty)
                dfs(pos + 1, current_score + individual_score - pair_penalty, assigned)
                assigned.pop(robot_index, None)

        dfs(0, 0.0, {})

        assigned_robots = set(best_assignment.keys())
        for index, (candidate, assignment_score) in best_assignment.items():
            targets[index] = tuple(candidate.target)
            reasons[index] = (
                f"{self.strategy}: distributed assignment; "
                f"{candidate.reason}; joint_score={assignment_score:.2f}; "
                f"reserve_radius={effective_target_radius:.2f}"
            )

        for index in sorted(assign_set - assigned_robots):
            if targets[index] is None:
                reasons[index] = f"{self.strategy}: holding position to avoid redundant/crossing exploration"

        return CoordinationResult(tuple(targets), tuple(reasons), self.strategy)

