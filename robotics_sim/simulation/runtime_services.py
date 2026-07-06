"""Concrete simulator-side implementations of robotics_interfaces.services.

This module is on the simulator side of the boundary: it may import
robotics_sim/robotics_sim.planning internals. External algorithms must never
import this module directly -- they only depend on the protocols in
robotics_interfaces.services, and the simulator host injects instances of the
classes below through CoordinationServices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from robotics_interfaces.observations import Point2D, WorldBounds
from robotics_interfaces.results import (
    CollisionCheckResult,
    MapQuerySnapshot,
    MetricsEvent,
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

        conflict_robot_id = None
        if not result.is_valid and result.reason_code == "route_conflict_with_robot_safety_zone":
            conflict_robot_id = self._find_conflicting_robot_id(start, waypoints, safety_radius, robot_id)

        return CollisionCheckResult(
            is_safe=result.is_valid,
            reason_code=result.reason_code,
            detail=result.detail,
            conflict_robot_id=conflict_robot_id,
        )

    def _find_conflicting_robot_id(
        self,
        start: Point2D,
        waypoints: Sequence[Point2D],
        safety_radius: float,
        robot_id: int,
    ) -> int | None:
        """Best-effort: identify which single teammate disk the corridor
        crosses, by re-checking one disk at a time. validate_multi_robot_corridor
        only reports a reason code, not which teammate triggered it."""
        for other_id, disk in self.other_robot_disks_by_id.items():
            if other_id == robot_id:
                continue
            single = validate_multi_robot_corridor(
                start=start,
                waypoints=waypoints,
                ego_safety_radius=safety_radius,
                other_robot_disks=[disk],
                margin=self.margin,
            )
            if not single.is_valid and single.reason_code == "route_conflict_with_robot_safety_zone":
                return other_id
        return None


@dataclass(frozen=True)
class RuntimeMapQueryService:
    """MapQueryService backed by a snapshot captured when the coordination
    request was built. This is read-only and does not re-query the live
    engine, so it reflects the map as of that request, not "right now"."""

    explored_points: tuple[Point2D, ...] = ()
    mapped_obstacle_points: tuple[Point2D, ...] = ()
    bounds: WorldBounds | None = None
    resolution: float = 0.5

    def map_snapshot(self) -> MapQuerySnapshot:
        return MapQuerySnapshot(
            explored_points=self.explored_points,
            mapped_obstacle_points=self.mapped_obstacle_points,
            bounds=self.bounds,
            resolution=self.resolution,
        )


@dataclass(frozen=True)
class RuntimeMetricsService:
    """Minimal MetricsService: records events through the stdlib logger.

    This is deliberately the simplest safe implementation -- it never raises
    and never blocks -- rather than wiring a real dashboard/metrics store
    before any algorithm actually needs one.
    """

    logger_name: str = "robotics_sim.runtime_metrics"

    def record_event(self, event: MetricsEvent) -> None:
        logging.getLogger(self.logger_name).debug(
            "metrics event: name=%s data=%s", event.name, dict(event.data)
        )
