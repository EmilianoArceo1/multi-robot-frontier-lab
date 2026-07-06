from __future__ import annotations

import math
from typing import Any, Mapping

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.observations import Point2D, RobotCoordinationState
from robotics_interfaces.plugins import (
    CoordinationPlugin,
    PluginCapability,
    PluginMetadata,
)
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate

FUEL_FRONTIER_BASELINE_COORDINATOR = "FUEL frontier baseline coordinator"


class FuelFrontierBaselinePlugin:
    """FUEL-inspired frontier/viewpoint baseline for the 2D simulator.

    This is intentionally not a port of HKUST FUEL/RACER.  It keeps the useful
    algorithm shape from FUEL -- frontier clusters, candidate viewpoints with
    heading/yaw, local viewpoint scoring, and a simple global ordering bias --
    while staying simulator-independent and compatible with the current 2D
    runtime.  The implementation only depends on robotics_interfaces so the
    same contract can later be extended to 3D snapshots/viewpoints without
    importing simulator internals.
    """

    metadata = PluginMetadata(
        name=FUEL_FRONTIER_BASELINE_COORDINATOR,
        version="0.1.0",
        description=(
            "FUEL-inspired baseline: clusters frontier/viewpoint candidates, "
            "scores information gain, travel cost and heading change, and "
            "returns RobotCommand target + optional heading."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TARGET_GENERATION,
            PluginCapability.TASK_ALLOCATION,
        ),
        source="FUEL/RACER-inspired 2D adapter; no middleware/C++ dependency",
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        robots_by_id = {robot.robot_id: robot for robot in request.robot_states}
        robots_to_assign = self._robots_to_assign(request, robots_by_id)
        team_candidates = self._team_candidate_pool(request)

        reserved_target_keys: set[tuple[int, int]] = {
            self._target_key(robot.current_target, request)
            for robot in request.robot_states
            if robot.robot_id not in robots_to_assign and robot.current_target is not None
        }
        reserved_cluster_ids: set[str] = set()

        assignments: list[CoordinationAssignment] = []
        commands: list[RobotCommand] = []
        targets_by_robot: dict[int, Point2D | None] = {}
        reasons_by_robot: dict[int, str] = {}
        debug_by_robot: dict[int, dict[str, Any]] = {}

        for robot_id in robots_to_assign:
            robot = robots_by_id.get(robot_id)
            if robot is None:
                reason = "robot id not present in request.robot_states"
                assignments.append(
                    CoordinationAssignment(
                        robot_id=robot_id,
                        status="FAILED",
                        target=None,
                        reason=reason,
                    )
                )
                commands.append(RobotCommand(robot_id=robot_id, status="FAILED", reason=reason))
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                debug_by_robot[robot_id] = {"rejected": {"missing_robot": 1}}
                continue

            raw_candidates = self._candidate_pool(request, robot, team_candidates)
            clustered = self._best_candidate_per_frontier_cluster(raw_candidates, robot, request)
            chosen, rejection_counts = self._choose_candidate(
                candidates=clustered,
                robot=robot,
                request=request,
                reserved_target_keys=reserved_target_keys,
                reserved_cluster_ids=reserved_cluster_ids,
            )

            debug_by_robot[robot_id] = {
                "raw_candidates": len(raw_candidates),
                "clustered_candidates": len(clustered),
                "rejected": rejection_counts,
            }

            if chosen is None:
                reason = "no valid FUEL-style frontier viewpoint candidate"
                assignments.append(
                    CoordinationAssignment(
                        robot_id=robot_id,
                        status="HOLD",
                        target=None,
                        reason=reason,
                    )
                )
                commands.append(RobotCommand(robot_id=robot_id, status="HOLD", reason=reason))
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                continue

            target = chosen.target
            cluster_id = self._cluster_id(chosen, request)
            reserved_target_keys.add(self._target_key(target, request))
            if cluster_id is not None:
                reserved_cluster_ids.add(cluster_id)

            reason = "selected by FUEL frontier baseline"
            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=target,
                    proposal=chosen,
                    reason=reason,
                )
            )
            commands.append(
                RobotCommand(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=target,
                    heading_rad=chosen.heading_rad,
                    reason=reason,
                    metadata={
                        "source": chosen.source,
                        "cluster_id": cluster_id,
                        "fuel_score": self._fuel_score(chosen, robot, request),
                    },
                )
            )
            targets_by_robot[robot_id] = target
            reasons_by_robot[robot_id] = reason

        targets = tuple(
            targets_by_robot.get(robot.robot_id, robot.current_target) for robot in request.robot_states
        )
        reasons = tuple(
            reasons_by_robot.get(
                robot.robot_id,
                "kept existing target" if robot.current_target is not None else "not requested",
            )
            for robot in request.robot_states
        )

        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            commands=tuple(commands),
            debug={
                "plugin": self.metadata.name,
                "capabilities": tuple(capability.value for capability in self.metadata.capabilities),
                "robots_to_assign": tuple(robots_to_assign),
                "candidate_source_order": (
                    "explicit_proposals",
                    "frontier_information_service",
                    "team_frontier_provider",
                    "frontier_provider",
                ),
                "per_robot": debug_by_robot,
            },
        )

    def _robots_to_assign(
        self,
        request: CoordinationRequest,
        robots_by_id: dict[int, RobotCoordinationState],
    ) -> tuple[int, ...]:
        if request.robots_to_assign:
            return tuple(int(robot_id) for robot_id in request.robots_to_assign)
        return tuple(robot_id for robot_id, robot in robots_by_id.items() if robot.is_active)

    def _team_candidate_pool(
        self,
        request: CoordinationRequest,
    ) -> Mapping[int, tuple[ExplorationCandidate, ...]]:
        if request.services is None:
            return {}
        team_provider = getattr(request.services, "team_frontier_provider", None)
        if team_provider is None:
            return {}
        return {
            int(robot_id): tuple(self._as_candidate(candidate) for candidate in candidates)
            for robot_id, candidates in team_provider.candidates_for_team(request).items()
        }

    def _candidate_pool(
        self,
        request: CoordinationRequest,
        robot: RobotCoordinationState,
        team_candidates: Mapping[int, tuple[ExplorationCandidate, ...]],
    ) -> tuple[ExplorationCandidate, ...]:
        explicit = request.proposals_by_robot.get(robot.robot_id, ())
        if explicit:
            return tuple(self._as_candidate(candidate) for candidate in explicit)

        frontier_information = getattr(request.services, "frontier_information_service", None)
        if frontier_information is not None:
            clusters = frontier_information.get_frontier_clusters(robot_id=robot.robot_id)
            candidates = tuple(self._candidate_from_cluster(cluster) for cluster in clusters)
            candidates = tuple(candidate for candidate in candidates if candidate is not None)
            if candidates:
                return candidates

        if robot.robot_id in team_candidates:
            return team_candidates[robot.robot_id]

        frontier_provider = getattr(request.services, "frontier_provider", None)
        if frontier_provider is None or request.world is None:
            return ()

        blocked = request.blocked_targets_by_robot.get(robot.robot_id, ())
        return tuple(
            self._as_candidate(candidate)
            for candidate in frontier_provider.candidates_for_robot(
                robot=robot,
                world=request.world,
                blocked_targets=blocked,
            )
        )

    def _best_candidate_per_frontier_cluster(
        self,
        candidates: tuple[ExplorationCandidate, ...],
        robot: RobotCoordinationState,
        request: CoordinationRequest,
    ) -> tuple[ExplorationCandidate, ...]:
        best_by_cluster: dict[str, ExplorationCandidate] = {}
        for candidate in candidates:
            cluster_id = self._cluster_id(candidate, request)
            if cluster_id is None:
                cluster_id = f"target:{self._target_key(candidate.target, request)}"
            previous = best_by_cluster.get(cluster_id)
            if previous is None or self._fuel_score(candidate, robot, request) > self._fuel_score(
                previous, robot, request
            ):
                best_by_cluster[cluster_id] = candidate
        return tuple(best_by_cluster.values())

    def _choose_candidate(
        self,
        candidates: tuple[ExplorationCandidate, ...],
        robot: RobotCoordinationState,
        request: CoordinationRequest,
        reserved_target_keys: set[tuple[int, int]],
        reserved_cluster_ids: set[str],
    ) -> tuple[ExplorationCandidate | None, dict[str, int]]:
        rejection_counts: dict[str, int] = {}
        blocked_targets = set(
            self._target_key(target, request)
            for target in request.blocked_targets_by_robot.get(robot.robot_id, ())
        )

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                self._fuel_score(candidate, robot, request),
                -self._distance(robot.xy, candidate.target),
                candidate.target[0],
                candidate.target[1],
            ),
            reverse=True,
        )

        for candidate in ranked:
            target_key = self._target_key(candidate.target, request)
            cluster_id = self._cluster_id(candidate, request)

            if self._distance(robot.xy, candidate.target) < self._min_frontier_travel_distance(request):
                self._count(rejection_counts, "too_close_to_robot")
                continue
            if target_key in blocked_targets:
                self._count(rejection_counts, "blocked_target")
                continue
            if target_key in reserved_target_keys:
                self._count(rejection_counts, "target_reservation_conflict")
                continue
            if cluster_id is not None and cluster_id in reserved_cluster_ids:
                self._count(rejection_counts, "cluster_reservation_conflict")
                continue
            return candidate, rejection_counts

        return None, rejection_counts

    def _fuel_score(
        self,
        candidate: ExplorationCandidate,
        robot: RobotCoordinationState,
        request: CoordinationRequest,
    ) -> float:
        distance_weight = float(
            request.parameters.get(
                "fuel_distance_weight",
                request.parameters.get("ipp_distance_penalty", 0.2),
            )
        )
        heading_weight = float(request.parameters.get("fuel_heading_weight", 0.25))
        safety_weight = float(request.parameters.get("fuel_safety_weight", 1.0))
        overlap_weight = float(request.parameters.get("fuel_overlap_weight", 1.0))

        travel_cost = candidate.travel_cost
        if travel_cost <= 0.0:
            travel_cost = self._distance(robot.xy, candidate.target)

        heading_cost = candidate.heading_cost
        if candidate.heading_rad is not None:
            heading_cost += self._angle_distance(robot.theta, candidate.heading_rad)

        return (
            candidate.information_gain
            - distance_weight * travel_cost
            - heading_weight * heading_cost
            - safety_weight * candidate.safety_cost
            - overlap_weight * candidate.overlap_cost
        )

    def _as_candidate(self, value: ExplorationCandidate | CandidateProposal | Any) -> ExplorationCandidate:
        if isinstance(value, ExplorationCandidate):
            return value
        if isinstance(value, CandidateProposal):
            return value.as_candidate(source="explicit_proposal")

        target = getattr(value, "target", None)
        if target is None:
            target = getattr(value, "xy", None)
        if target is None:
            raise TypeError("candidate-like value must expose target or xy")

        metadata = dict(getattr(value, "metadata", {}) or {})
        if hasattr(value, "cluster_id"):
            metadata.setdefault("cluster_id", str(getattr(value, "cluster_id")))

        return ExplorationCandidate(
            target=(float(target[0]), float(target[1])),
            source=str(getattr(value, "source", "duck_typed_viewpoint")),
            information_gain=float(getattr(value, "information_gain", 0.0)),
            travel_cost=float(getattr(value, "travel_cost", 0.0)),
            safety_cost=float(getattr(value, "safety_cost", 0.0)),
            overlap_cost=float(getattr(value, "overlap_cost", 0.0)),
            heading_cost=float(getattr(value, "heading_cost", 0.0)),
            heading_rad=getattr(value, "heading_rad", None),
            metadata=metadata,
        )

    def _candidate_from_cluster(self, cluster: Any) -> ExplorationCandidate | None:
        viewpoints = tuple(getattr(cluster, "viewpoints", ()) or ())
        cluster_id = str(getattr(cluster, "cluster_id", "")) or None
        if viewpoints:
            ranked = sorted(
                viewpoints,
                key=lambda viewpoint: (
                    float(getattr(viewpoint, "information_gain", 0.0)),
                    float(getattr(viewpoint, "coverage_fraction", 0.0)),
                    int(getattr(viewpoint, "visible_cell_count", 0)),
                    -float(getattr(viewpoint, "travel_cost", 0.0)),
                ),
                reverse=True,
            )
            candidate = self._as_candidate(ranked[0])
        else:
            centroid = getattr(cluster, "centroid", None)
            if centroid is None:
                return None
            candidate = ExplorationCandidate(
                target=(float(centroid[0]), float(centroid[1])),
                source="frontier_cluster_centroid",
                information_gain=float(getattr(cluster, "information_gain", 0.0)),
            )

        metadata = dict(candidate.metadata)
        if cluster_id is not None:
            metadata.setdefault("cluster_id", cluster_id)
        return ExplorationCandidate(
            target=candidate.target,
            source="frontier_information_service",
            information_gain=candidate.information_gain,
            travel_cost=candidate.travel_cost,
            safety_cost=candidate.safety_cost,
            overlap_cost=candidate.overlap_cost,
            heading_cost=candidate.heading_cost,
            heading_rad=candidate.heading_rad,
            metadata=metadata,
        )

    def _cluster_id(self, candidate: ExplorationCandidate, request: CoordinationRequest) -> str | None:
        value = candidate.metadata.get("cluster_id")
        if value is not None:
            return str(value)
        radius = float(request.parameters.get("fuel_cluster_radius", self._default_cluster_radius(request)))
        radius = max(radius, 1e-6)
        return f"implicit:{round(candidate.target[0] / radius)}:{round(candidate.target[1] / radius)}"

    def _target_key(self, target: Point2D, request: CoordinationRequest) -> tuple[int, int]:
        resolution = float(request.parameters.get("reservation_resolution", 0.01))
        resolution = max(resolution, 1e-6)
        return (round(target[0] / resolution), round(target[1] / resolution))

    def _default_cluster_radius(self, request: CoordinationRequest) -> float:
        grid_resolution = float(request.parameters.get("grid_resolution", 0.5))
        return max(2.0 * grid_resolution, 1.0)

    def _min_frontier_travel_distance(self, request: CoordinationRequest) -> float:
        if "min_frontier_travel_distance" in request.parameters:
            return float(request.parameters["min_frontier_travel_distance"])
        grid_resolution = float(request.parameters.get("grid_resolution", 0.5))
        goal_tolerance = float(request.parameters.get("goal_tolerance", 0.25))
        return max(2.0 * goal_tolerance, 2.0 * grid_resolution, 0.75)

    def _distance(self, a: Point2D, b: Point2D) -> float:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _angle_distance(self, a: float, b: float) -> float:
        delta = (float(b) - float(a) + math.pi) % (2.0 * math.pi) - math.pi
        return abs(delta)

    def _count(self, counts: dict[str, int], reason: str) -> None:
        counts[reason] = counts.get(reason, 0) + 1


def create_plugin() -> CoordinationPlugin:
    return FuelFrontierBaselinePlugin()
