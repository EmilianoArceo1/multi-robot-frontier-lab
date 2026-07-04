"""
Waypoint sequence manager.

Responsibility:
    Store and advance through an executable route.

It does not:
    - compute paths
    - choose exploration targets
    - control the robot
    - detect obstacles
"""

from __future__ import annotations

import numpy as np

from robotics_sim.core.geometry import distance, parse_point


class WaypointManager:
    """
    Ordered sequence of local targets.

    Important distinction:
        active_waypoint:
            the waypoint currently being tracked by the controller

        final_goal:
            the final waypoint of the current route

        display_goal:
            the point the GUI should show; this is usually the active waypoint,
            but after finishing the route it remains the final waypoint
    """

    def __init__(self, waypoints=None):
        self.waypoints: list[np.ndarray] = []
        self.current_index: int = 0

        if waypoints is not None:
            self.set_waypoints(waypoints)

    def set_goal(self, goal) -> None:
        """Convert a single goal into a route with one waypoint."""
        self.set_waypoints([goal])

    def set_waypoints(self, waypoints) -> None:
        """Replace the active route."""
        parsed = [parse_point(waypoint) for waypoint in waypoints]
        self.waypoints = parsed
        self.current_index = 0

    def clear(self) -> None:
        """Remove any active route."""
        self.waypoints = []
        self.current_index = 0

    def has_path(self) -> bool:
        """Return whether a route has been assigned."""
        return len(self.waypoints) > 0

    def is_finished(self) -> bool:
        """Return whether all waypoints have been reached."""
        return self.has_path() and self.current_index >= len(self.waypoints)

    def active_waypoint(self):
        """Return the waypoint the controller should track now."""
        if not self.has_path():
            return None

        if self.current_index >= len(self.waypoints):
            return None

        return self.waypoints[self.current_index]

    def final_goal(self):
        """Return the final goal of the current route."""
        if not self.has_path():
            return None
        return self.waypoints[-1]

    def display_goal(self):
        """
        Return a point for visualization.

        If the route is finished, the final waypoint is preserved so the GUI can
        still display the final target.
        """
        active = self.active_waypoint()

        if active is not None:
            return active

        return self.final_goal()

    def advance_if_reached(
        self,
        position: tuple[float, float],
        tolerance: float,
    ) -> bool:
        """
        Advance to the next waypoint if the current one has been reached.

        Criterion:
            distance(position, active_waypoint) <= tolerance
        """
        active = self.active_waypoint()

        if active is None:
            return False

        waypoint_position = (float(active[0]), float(active[1]))

        if distance(position, waypoint_position) <= tolerance:
            self.current_index += 1
            return True

        return False

    def same_active_waypoint_as(self, goal, tolerance: float = 1e-9) -> bool:
        """Return whether the active waypoint is essentially the given point."""
        active = self.active_waypoint()

        if active is None:
            return False

        new_goal = parse_point(goal)

        return float(np.linalg.norm(active - new_goal)) <= tolerance

    def same_final_goal_as(self, goal, tolerance: float = 1e-9) -> bool:
        """
        Return whether the final route goal is essentially the given point.

        This is the method the engine/RobotAgent should use to avoid resetting a
        multi-waypoint route when the same final target is repeatedly sent.
        """
        current_goal = self.final_goal()

        if current_goal is None:
            return False

        new_goal = parse_point(goal)

        return float(np.linalg.norm(current_goal - new_goal)) <= tolerance

    def same_goal_as(self, goal, tolerance: float = 1e-9) -> bool:
        """
        Backward-compatible alias.

        Older code used same_goal_as(...). The safer meaning is now final-goal
        comparison, not display-goal comparison.
        """
        return self.same_final_goal_as(goal, tolerance=tolerance)
