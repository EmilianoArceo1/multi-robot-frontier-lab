"""Concrete simulator-side implementations of robotics_interfaces.services.

This module is on the simulator side of the boundary: it may import
robotics_sim/robotics_sim.planning internals. External algorithms must never
import this module directly -- they only depend on the protocols in
robotics_interfaces.services, and the simulator host injects instances of the
classes below through CoordinationServices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from robotics_interfaces.observations import Point2D
from robotics_interfaces.results import (
    CollisionCheckResult,
    PathPlanningRequest,
    PathPlanningResponse,
)
from robotics_sim.planning.coordinated_frontier_planner import validate_multi_robot_corridor
from robotics_sim.planning.planner_registry import compute_planned_waypoints


@dataclass(frozen=True)
class RuntimePathPlanningService:
    """PathPlanningService backed by the simulator's existing planner registry
    (Direct/A*/Dijkstra), so an algorithm can request a path without knowing
    which concrete planner is configured."""

    def plan_path(self, request: PathPlanningRequest) -> PathPlanningResponse:
        success, reason, waypoints = compute_planned_waypoints(
            planner_type=request.planner_type,
            start_xy=request.start,
            goal_xy=request.goal,
            bounds=request.bounds,
            resolution=request.resolution,
            robot_radius=request.robot_radius,
            obstacle_points=list(request.obstacle_points),
        )
        return PathPlanningResponse(
            success=bool(success),
            waypoints=tuple(waypoints),
            reason=str(reason),
        )


@dataclass(frozen=True)
class RuntimeCollisionCheckingService:
    """CollisionCheckingService backed by validate_multi_robot_corridor(), the
    same corridor validator the engine uses before accepting a route as
    ACTIVE. Reusing it here keeps "is this path safe" consistent between what
    the engine enforces and what an algorithm can ask about in advance.
    """

    other_robot_disks_by_id: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    other_routes_by_id: dict[int, tuple[Point2D, ...]] = field(default_factory=dict)
    margin: float = 0.25

    def is_path_safe(
        self,
        path: Sequence[Point2D],
        robot_id: int,
        safety_radius: float,
    ) -> CollisionCheckResult:
        if len(path) < 1:
            return CollisionCheckResult(is_safe=True, detail="empty path has nothing to validate")

        start = path[0]
        waypoints = path[1:]
        other_disks = [
            disk for rid, disk in self.other_robot_disks_by_id.items() if rid != robot_id
        ]
        other_routes = [
            route for rid, route in self.other_routes_by_id.items() if rid != robot_id
        ]

        result = validate_multi_robot_corridor(
            start=start,
            waypoints=waypoints,
            ego_safety_radius=safety_radius,
            other_robot_disks=other_disks,
            other_routes=other_routes,
            margin=self.margin,
        )
        return CollisionCheckResult(
            is_safe=result.is_valid,
            reason_code=result.reason_code,
            detail=result.detail,
        )
