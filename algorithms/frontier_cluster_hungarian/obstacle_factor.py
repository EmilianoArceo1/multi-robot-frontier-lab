"""Pure geometric obstacle-avoidance heuristic: a five-parallel-line
clearance check between a robot and a candidate target.

Uses only the observed obstacle points the host already reports
(request.world.mapped_obstacle_points in the caller) -- never ground-truth
obstacles, never the collision-checking service, never A*, never an
occupancy grid, never a line-of-sight grid, never a hazard field. When
there are no observed obstacle points at all, every line is clear by
definition: blocked_line_fraction == 0.0 and clearance_score == 1.0.
"""
from __future__ import annotations

import math
from typing import Sequence

Point2D = tuple[float, float]

# Five parallel lines, offset from the direct robot->target segment along
# its perpendicular by these factors of the configured half-width
# (typically the robot's safety_radius): -1x, -0.5x, 0 (the direct
# segment), +0.5x, +1x.
LINE_OFFSET_FACTORS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
LINE_COUNT = len(LINE_OFFSET_FACTORS)


def point_to_segment_distance(point: Point2D, start: Point2D, end: Point2D) -> float:
    """Euclidean distance from `point` to the closest point on segment
    [start, end]. Degenerates to point-to-point distance when start == end,
    without raising."""
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
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _unit_perpendicular(start: Point2D, end: Point2D) -> tuple[float, float]:
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        # Zero-length segment: no well-defined direction, so every offset
        # collapses onto the same point. This never raises -- feasibility
        # of a near-zero-distance target is resolved upstream via
        # min_frontier_travel_distance, not here.
        return (0.0, 0.0)
    return (-dy / length, dx / length)


def five_line_blocked_fraction(
    *,
    robot_xy: Point2D,
    target_xy: Point2D,
    observed_obstacle_points: Sequence[Point2D],
    safety_radius: float,
    point_tolerance: float,
) -> float:
    """Fraction (in [0.0, 1.0]) of the five parallel robot->target lines
    that are blocked by at least one observed obstacle point within
    point_tolerance of that line.

    Both endpoints of the direct robot->target segment are shifted by the
    same perpendicular offset for each of the five lines, so all five stay
    parallel to the direct segment. With no observed obstacle points, this
    is always 0.0 (see module docstring).
    """
    obstacles = [(float(x), float(y)) for x, y in observed_obstacle_points]
    if not obstacles:
        return 0.0

    safety_radius = float(safety_radius)
    point_tolerance = max(float(point_tolerance), 0.0)
    perp_x, perp_y = _unit_perpendicular(robot_xy, target_xy)

    blocked_lines = 0
    for factor in LINE_OFFSET_FACTORS:
        offset = factor * safety_radius
        shifted_start = (robot_xy[0] + perp_x * offset, robot_xy[1] + perp_y * offset)
        shifted_end = (target_xy[0] + perp_x * offset, target_xy[1] + perp_y * offset)
        if any(
            point_to_segment_distance(obstacle, shifted_start, shifted_end) <= point_tolerance
            for obstacle in obstacles
        ):
            blocked_lines += 1

    return blocked_lines / float(LINE_COUNT)


def five_line_clearance_score(blocked_line_fraction: float) -> float:
    """1.0 - blocked_line_fraction, clamped to [0.0, 1.0]."""
    score = 1.0 - float(blocked_line_fraction)
    return max(0.0, min(1.0, score))
