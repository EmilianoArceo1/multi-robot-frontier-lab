"""
Collision-checking utilities for the 2D robotics simulator.

This module intentionally stays independent from the GUI and from the robot
controller. It answers geometric safety questions:

    - Is the robot currently inside an obstacle?
    - Is the local segment toward the active waypoint blocked?
    - Would the predicted next motion enter an obstacle?

The robot is modeled as a disk. Rectangular obstacles are expanded by the
robot radius, so collision checking can reason about the robot center as a
point.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


Point2D = tuple[float, float]
RectObstacle = tuple[float, float, float, float]


@dataclass(frozen=True)
class CollisionReport:
    """
    Result of a collision query.

    The boolean `collision` is the main decision signal. The other fields exist
    for status messages, debugging, and future visualization.

    `distance` is only populated on a collision hit for the point-cloud checks
    (check_segment_points/check_position_points/check_predicted_motion_points),
    since that is the only case where those checks already compute a scalar
    distance before comparing it to robot_radius. On a clear result there is
    no single distance value to report -- the loop simply finds nothing within
    radius of any sampled point, so this stays None rather than being invented
    (e.g. as a nearest-point search the real check never performs).
    """

    collision: bool
    reason: str = "clear"
    obstacle: RectObstacle | None = None
    point: Point2D | None = None
    distance: float | None = None


@dataclass(frozen=True)
class RobotSnapshot:
    """
    Minimal robot state needed to predict short-horizon motion.

    This is not a replacement for RobotState. It is just a lightweight adapter
    so collision checking does not depend on the full robot class.
    """

    x: float
    y: float
    theta: float
    v: float
    max_speed: float
    max_acceleration: float
    max_angular_speed: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def expanded_rect(
    rect: RectObstacle,
    padding: float,
) -> tuple[float, float, float, float]:
    """
    Expand a rectangle by a safety margin.

    Input convention:
        rect = (x, y, width, height), where x,y is the lower-left corner.

    Output convention:
        (x_min, y_min, x_max, y_max)
    """
    x, y, width, height = rect
    return (
        float(x) - padding,
        float(y) - padding,
        float(x) + float(width) + padding,
        float(y) + float(height) + padding,
    )


def point_inside_expanded_rect(
    point: Point2D,
    rect: RectObstacle,
    padding: float = 0.0,
) -> bool:
    """
    Return whether a point lies inside a rectangle expanded by padding.

    Abstraction:
        Instead of checking disk-vs-rectangle directly, the obstacle is expanded
        by the robot radius and the robot center is checked as a point.
    """
    px, py = point
    x_min, y_min, x_max, y_max = expanded_rect(rect, padding)

    return x_min <= px <= x_max and y_min <= py <= y_max


def orientation(a: Point2D, b: Point2D, c: Point2D) -> int:
    """
    Orientation test for ordered triplet (a, b, c).

    Returns:
        0 for collinear
        1 for clockwise
        2 for counterclockwise
    """
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])

    if abs(value) < 1e-12:
        return 0

    return 1 if value > 0 else 2


def on_segment(a: Point2D, b: Point2D, c: Point2D) -> bool:
    """
    Return whether b lies on segment ac, assuming collinearity.
    """
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(
    p1: Point2D,
    q1: Point2D,
    p2: Point2D,
    q2: Point2D,
) -> bool:
    """
    Return whether two 2D line segments intersect.

    This is a geometric predicate. It does not know about robots or maps.
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def rect_edges(
    rect: RectObstacle,
    padding: float = 0.0,
) -> list[tuple[Point2D, Point2D]]:
    """
    Return the edges of an expanded rectangular obstacle.
    """
    x_min, y_min, x_max, y_max = expanded_rect(rect, padding)

    bottom_left = (x_min, y_min)
    bottom_right = (x_max, y_min)
    top_right = (x_max, y_max)
    top_left = (x_min, y_max)

    return [
        (bottom_left, bottom_right),
        (bottom_right, top_right),
        (top_right, top_left),
        (top_left, bottom_left),
    ]


def segment_intersects_expanded_rect(
    start: Point2D,
    end: Point2D,
    rect: RectObstacle,
    padding: float = 0.0,
) -> bool:
    """
    Return whether a segment intersects an expanded rectangle.

    This is the core test for local path blocking:
        robot center -> active waypoint
    """
    if point_inside_expanded_rect(start, rect, padding):
        return True

    if point_inside_expanded_rect(end, rect, padding):
        return True

    for edge_start, edge_end in rect_edges(rect, padding):
        if segments_intersect(start, end, edge_start, edge_end):
            return True

    return False


def distance_point_to_segment(point: Point2D, start: Point2D, end: Point2D) -> float:
    """
    Minimum Euclidean distance from a point to a line segment.

    Mapped obstacles are represented as sparse points. To keep clearance from
    those points, the local segment is unsafe when this distance is smaller than
    the robot safety radius.
    """
    px, py = point
    sx, sy = start
    ex, ey = end

    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy

    if length_sq <= 1e-12:
        return math.hypot(px - sx, py - sy)

    t = ((px - sx) * dx + (py - sy) * dy) / length_sq
    t = clamp(t, 0.0, 1.0)

    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def point_inside_disk(point: Point2D, center: Point2D, radius: float) -> bool:
    """
    Return whether point lies inside a disk centered at center.
    """
    return math.hypot(point[0] - center[0], point[1] - center[1]) <= radius


class CollisionChecker:
    """
    Geometric safety checker for rectangular 2D obstacles.

    Responsibility:
        Provide collision decisions to the simulator loop.

    It does not:
        - change the robot state
        - compute controls
        - compute A* or Dijkstra paths
        - draw anything directly
    """

    def check_position(
        self,
        position: Point2D,
        obstacles: Iterable[RectObstacle],
        robot_radius: float,
    ) -> CollisionReport:
        """
        Check whether the robot center is inside any expanded obstacle.
        """
        for obstacle in obstacles:
            if point_inside_expanded_rect(position, obstacle, robot_radius):
                return CollisionReport(
                    collision=True,
                    reason="robot center is inside an expanded obstacle",
                    obstacle=obstacle,
                    point=position,
                )

        return CollisionReport(collision=False)

    def check_segment(
        self,
        start: Point2D,
        end: Point2D | None,
        obstacles: Iterable[RectObstacle],
        robot_radius: float,
    ) -> CollisionReport:
        """
        Check whether the local segment from start to end is blocked.

        If end is None, there is no active target and therefore no segment to
        validate.
        """
        if end is None:
            return CollisionReport(collision=False)

        for obstacle in obstacles:
            if segment_intersects_expanded_rect(start, end, obstacle, robot_radius):
                return CollisionReport(
                    collision=True,
                    reason="local path segment intersects an expanded obstacle",
                    obstacle=obstacle,
                    point=end,
                )

        return CollisionReport(collision=False)

    def check_position_points(
        self,
        position: Point2D,
        obstacle_points: Iterable[Point2D],
        robot_radius: float,
    ) -> CollisionReport:
        """
        Check whether the robot center is too close to any mapped obstacle point.
        """
        for point in obstacle_points:
            mapped_point = (float(point[0]), float(point[1]))
            distance = math.hypot(position[0] - mapped_point[0], position[1] - mapped_point[1])
            if distance <= robot_radius:
                return CollisionReport(
                    collision=True,
                    reason="robot center is inside a mapped obstacle point radius",
                    obstacle=None,
                    point=mapped_point,
                    distance=distance,
                )

        return CollisionReport(collision=False)

    def check_segment_points(
        self,
        start: Point2D,
        end: Point2D | None,
        obstacle_points: Iterable[Point2D],
        robot_radius: float,
    ) -> CollisionReport:
        """
        Check whether a local segment comes too close to mapped obstacle points.
        """
        if end is None:
            return CollisionReport(collision=False)

        for point in obstacle_points:
            mapped_point = (float(point[0]), float(point[1]))
            distance = distance_point_to_segment(mapped_point, start, end)
            if distance <= robot_radius:
                return CollisionReport(
                    collision=True,
                    reason="local path segment intersects mapped obstacle point radius",
                    obstacle=None,
                    point=mapped_point,
                    distance=distance,
                )

        return CollisionReport(collision=False)

    def check_predicted_motion_points(
        self,
        snapshot: RobotSnapshot,
        control,
        dt: float,
        steps: int,
        obstacle_points: Iterable[Point2D],
        robot_radius: float,
    ) -> CollisionReport:
        """
        Check whether a short predicted trajectory enters mapped point radius.
        """
        predicted_points = self.predict_unicycle_points(snapshot, control, dt, steps)

        for point in predicted_points:
            report = self.check_position_points(point, obstacle_points, robot_radius)
            if report.collision:
                return CollisionReport(
                    collision=True,
                    reason="predicted motion enters a mapped obstacle point radius",
                    obstacle=None,
                    point=report.point,
                    distance=report.distance,
                )

        return CollisionReport(collision=False)

    def predict_unicycle_points(
        self,
        snapshot: RobotSnapshot,
        control,
        dt: float,
        steps: int,
    ) -> list[Point2D]:
        """
        Predict short-horizon robot center positions.

        This uses the same DynamicUnicycle2D abstraction:
            x_dot     = v cos(theta)
            y_dot     = v sin(theta)
            theta_dot = omega
            v_dot     = a

        The prediction is not a replacement for the real dynamics. It is a
        safety lookahead used before applying a control.
        """
        if dt <= 0 or steps <= 0:
            return []

        control_array = np.asarray(control, dtype=float).reshape(-1)
        if control_array.size != 2:
            raise ValueError("control must have two components: [a, omega].")

        acceleration = clamp(
            float(control_array[0]),
            -snapshot.max_acceleration,
            snapshot.max_acceleration,
        )
        angular_velocity = clamp(
            float(control_array[1]),
            -snapshot.max_angular_speed,
            snapshot.max_angular_speed,
        )

        x = float(snapshot.x)
        y = float(snapshot.y)
        theta = float(snapshot.theta)
        v = float(snapshot.v)

        points: list[Point2D] = []

        for _ in range(steps):
            x = x + v * math.cos(theta) * dt
            y = y + v * math.sin(theta) * dt
            theta = wrap_angle(theta + angular_velocity * dt)
            v = clamp(v + acceleration * dt, 0.0, snapshot.max_speed)
            points.append((x, y))

        return points

    def check_predicted_motion(
        self,
        snapshot: RobotSnapshot,
        control,
        dt: float,
        steps: int,
        obstacles: Iterable[RectObstacle],
        robot_radius: float,
    ) -> CollisionReport:
        """
        Check whether a short predicted trajectory would collide.
        """
        predicted_points = self.predict_unicycle_points(snapshot, control, dt, steps)

        for point in predicted_points:
            report = self.check_position(point, obstacles, robot_radius)
            if report.collision:
                return CollisionReport(
                    collision=True,
                    reason="predicted motion enters an expanded obstacle",
                    obstacle=report.obstacle,
                    point=point,
                )

        return CollisionReport(collision=False)
