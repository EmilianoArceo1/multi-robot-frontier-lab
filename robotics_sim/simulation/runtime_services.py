"""Concrete simulator-side implementations of robotics_interfaces.services.

This module is on the simulator side of the boundary: it may import
robotics_sim/robotics_sim.planning internals. External algorithms must never
import this module directly -- they only depend on the protocols in
robotics_interfaces.services, and the simulator host injects instances of the
classes below through CoordinationServices.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Sequence

from robotics_interfaces.frontiers import FrontierCluster, ViewpointCandidate
from robotics_interfaces.observations import Point2D, WorldBounds
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_interfaces.regions import CoveragePath, RegionTask
from robotics_interfaces.results import (
    CollisionCheckResult,
    MapQuerySnapshot,
    MetricsEvent,
    PathPlanningRequest,
    PathPlanningResponse,
)
from robotics_sim.planning.coordinated_frontier_planner import (
    detect_connected_frontier_components,
    validate_multi_robot_corridor,
)
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


def frontier_clusters_from_candidates(
    candidates: Sequence[ExplorationCandidate],
    id_prefix: str = "legacy",
) -> tuple[FrontierCluster, ...]:
    """Convert already-computed ExplorationCandidate objects (e.g. from
    RuntimeTeamFrontierProvider/RuntimeFrontierProvider) into simple
    single-viewpoint FrontierCluster objects.

    This is not real re-clustering -- each candidate becomes its own cluster
    -- so a FrontierInformationService consumer has something usable even
    when fresh map-based frontier detection has not found anything yet (e.g.
    a plugin that only knows about FrontierInformationService and does not
    want to reach for team/single-robot providers itself). These are
    adapters, not real connected components: cells is always empty and
    metadata always marks legacy_adapter=True so a consumer can tell the
    difference from a detect_connected_frontier_components() result.
    cluster_id ends up in each viewpoint's metadata via
    FrontierCluster.as_exploration_candidate(), so callers do not need it
    stamped here too.
    """
    clusters = []
    for index, candidate in enumerate(candidates):
        viewpoint = ViewpointCandidate(
            xy=candidate.target,
            heading_rad=candidate.heading_rad,
            information_gain=float(candidate.information_gain),
            travel_cost=float(candidate.travel_cost),
            safety_cost=float(candidate.safety_cost),
            metadata=dict(candidate.metadata),
        )
        clusters.append(
            FrontierCluster(
                cluster_id=f"{id_prefix}-{index}",
                cells=(),
                centroid=candidate.target,
                viewpoints=(viewpoint,),
                information_gain=float(candidate.information_gain),
                metadata={"source": candidate.source, "legacy_adapter": True},
                valid=True,
            )
        )
    return tuple(clusters)


@dataclass(frozen=True)
class RuntimeFrontierInformationService:
    """FrontierInformationService backed by the same connected-component
    frontier detector the coordinated frontier planner uses
    (detect_connected_frontier_components()). Each call runs the detector at
    most once and returns its real FrontierCluster objects unmodified --
    full cells, real centroid, one or more viewpoints per component -- so a
    FUEL/RACER-style plugin can observe the same geometry the legacy
    candidate pipeline is flattened from.

    If map-based detection finds nothing (e.g. not enough explored_points
    yet), it falls back to converting whatever legacy_candidates_by_robot was
    supplied at construction time (typically pre-computed from
    RuntimeTeamFrontierProvider/RuntimeFrontierProvider) instead of returning
    empty. Real detected components and legacy adapter clusters are never
    mixed in one response: if the detector found anything, that is the whole
    response. If there is truly no data anywhere, it returns an empty tuple
    instead of raising.
    """

    explored_points: tuple[Point2D, ...] = ()
    mapped_obstacle_points: tuple[Point2D, ...] = ()
    bounds: WorldBounds | None = None
    resolution: float = 0.5
    robot_radius: float = 0.35
    sensor_range: float = 2.5
    legacy_candidates_by_robot: dict[int, tuple[ExplorationCandidate, ...]] = field(default_factory=dict)

    def get_frontier_clusters(self, robot_id: int | None = None) -> tuple[FrontierCluster, ...]:
        if self.bounds is not None and self.explored_points:
            clusters = detect_connected_frontier_components(
                explored_points=self.explored_points,
                mapped_obstacle_points=self.mapped_obstacle_points,
                bounds=self.bounds,
                resolution=self.resolution,
                robot_radius=self.robot_radius,
                sensor_range=self.sensor_range,
            )
            if clusters:
                if robot_id is None:
                    return clusters
                # Only metadata is annotated per-robot -- cluster_id, cells,
                # centroid, viewpoints, information_gain, and valid are the
                # detector's geometry and must stay identical regardless of
                # which robot asked.
                return tuple(
                    replace(
                        cluster,
                        metadata={**dict(cluster.metadata), "requested_for_robot_id": robot_id},
                    )
                    for cluster in clusters
                )

        if robot_id is not None:
            legacy = self.legacy_candidates_by_robot.get(robot_id, ())
        else:
            legacy = tuple(
                candidate
                for candidates in self.legacy_candidates_by_robot.values()
                for candidate in candidates
            )
        if not legacy:
            return ()
        return frontier_clusters_from_candidates(legacy, id_prefix="legacy")


@dataclass(frozen=True)
class RuntimeRegionDecompositionService:
    """RegionDecompositionService derived from the same frontier detection
    RuntimeFrontierInformationService uses: each frontier cluster becomes one
    unassigned region task. This is a placeholder decomposition (not a real
    space-filling/Voronoi partition), so the contract has something real to
    return before a proper region planner exists.
    """

    frontier_information_service: RuntimeFrontierInformationService

    def get_region_tasks(self) -> tuple[RegionTask, ...]:
        clusters = self.frontier_information_service.get_frontier_clusters()
        if not clusters:
            return ()

        return tuple(
            RegionTask(
                region_id=f"region-{cluster.cluster_id}",
                centroid=cluster.centroid if cluster.centroid is not None else (0.0, 0.0),
                unknown_cell_count=sum(vp.visible_cell_count for vp in cluster.viewpoints),
                cells=cluster.cells,
                metadata={"source_cluster_id": cluster.cluster_id},
            )
            for cluster in clusters
        )


@dataclass(frozen=True)
class RuntimeCoveragePathService:
    """CoveragePathService that orders region centroids by greedy nearest-
    neighbor distance from the origin. This is a placeholder ("trivial
    coverage path"), not a CVRP/TSP solver -- a real solver can replace this
    later without changing the contract algorithms depend on.
    """

    def plan_coverage_path(self, robot_id: int, regions: tuple[RegionTask, ...]) -> CoveragePath:
        if not regions:
            return CoveragePath(
                robot_id=robot_id,
                waypoints=(),
                metadata={"reason": "no region tasks available"},
            )

        remaining = list(regions)
        origin = (0.0, 0.0)
        ordered: list[RegionTask] = []
        cursor = origin
        while remaining:
            remaining.sort(
                key=lambda region: math.hypot(
                    region.centroid[0] - cursor[0], region.centroid[1] - cursor[1]
                )
            )
            next_region = remaining.pop(0)
            ordered.append(next_region)
            cursor = next_region.centroid

        estimated_cost = 0.0
        cursor = origin
        for region in ordered:
            estimated_cost += math.hypot(region.centroid[0] - cursor[0], region.centroid[1] - cursor[1])
            cursor = region.centroid

        return CoveragePath(
            robot_id=robot_id,
            waypoints=tuple(region.centroid for region in ordered),
            region_ids=tuple(region.region_id for region in ordered),
            estimated_cost=estimated_cost,
            metadata={"ordering": "nearest_neighbor_from_origin"},
        )
