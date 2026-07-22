"""Two-stage frontier Hungarian coordinator.

Inspired by Gao et al., "A Novel Frontier-Based Multi-Robot Cooperative
Exploration Method": this plugin treats the FrontierCluster objects the
host already exposes through FrontierInformationService as the first
clustering stage, applies one more deterministic reduction pass (see
algorithms.frontier_cluster_hungarian.clustering -- a grid-density
APPROXIMATION, not GriT-DBSCAN), builds a global robot-task utility matrix
(information + distance + a five-parallel-line obstacle clearance factor),
and solves a global assignment with a pure-Python Hungarian solver (no
SciPy). This is a controlled, incomplete approximation of the paper's
method -- it does not claim to reproduce GriT-DBSCAN or every detail of the
utility formulation.

This plugin never detects frontiers itself (it only ever calls
request.services.frontier_information_service.get_frontier_clusters(),
exactly once per assign()) and never plans paths (no A*, no
path_planning_service, no collision_checking_service). It only depends on
robotics_interfaces, the Python standard library, and its own sibling
modules in this package.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.frontiers import FrontierCluster
from robotics_interfaces.observations import Point2D, RobotCoordinationState
from robotics_interfaces.plugins import (
    CandidateInputMode,
    CoordinationPlugin,
    PluginCapability,
    PluginMetadata,
)

from algorithms.frontier_cluster_hungarian.assignment import solve_max_utility_assignment
from algorithms.frontier_cluster_hungarian.clustering import (
    CLUSTERING_METHOD,
    reduce_frontier_clusters_with_diagnostics,
)
from algorithms.frontier_cluster_hungarian.utility import (
    UtilityWeights,
    build_utility_matrix,
    normalize_weights,
)

FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR = "Frontier cluster Hungarian coordinator"

DEFAULT_INFORMATION_WEIGHT = 0.45
DEFAULT_DISTANCE_WEIGHT = 0.35
DEFAULT_OBSTACLE_WEIGHT = 0.20

DEFAULT_GRID_RESOLUTION = 0.5
DEFAULT_ASSIGNMENT_DUPLICATE_TOLERANCE = 1e-6
DEFAULT_MIN_FRONTIER_TRAVEL_DISTANCE = 0.0

_REASON_NO_SERVICE = "no frontier information service"
_REASON_NO_VALID_CLUSTERS = "no valid frontier clusters"
_REASON_NO_TASKS = "no reduced frontier tasks"
_REASON_NO_FEASIBLE_TASK = "no feasible frontier task"
_REASON_HUNGARIAN_HOLD = "assigned to Hungarian HOLD dummy"
_REASON_UNKNOWN_ROBOT = "robot id not present in request.robot_states"


def _finite_non_negative(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")
    return result


def _finite_positive(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite, positive number, got {value!r}")
    return result


def _resolve_resolution(request: CoordinationRequest) -> float:
    if request.world is not None:
        candidate = request.world.resolution
    else:
        candidate = request.parameters.get("grid_resolution", DEFAULT_GRID_RESOLUTION)
    return _finite_positive(candidate, name="grid_resolution")


def _resolve_grid_size(request: CoordinationRequest, resolution: float) -> float:
    default = max(1.0, 4.0 * resolution)
    candidate = request.parameters.get("secondary_cluster_grid_size", default)
    return _finite_positive(candidate, name="secondary_cluster_grid_size")


def _resolve_merge_radius(request: CoordinationRequest, grid_size: float) -> float:
    default = 1.5 * grid_size
    candidate = request.parameters.get("secondary_cluster_merge_radius", default)
    return _finite_non_negative(candidate, name="secondary_cluster_merge_radius")


def _resolve_weights(request: CoordinationRequest) -> UtilityWeights:
    information = request.parameters.get("hungarian_information_weight", DEFAULT_INFORMATION_WEIGHT)
    distance = request.parameters.get("hungarian_distance_weight", DEFAULT_DISTANCE_WEIGHT)
    obstacle = request.parameters.get("hungarian_obstacle_weight", DEFAULT_OBSTACLE_WEIGHT)
    return normalize_weights(information_weight=information, distance_weight=distance, obstacle_weight=obstacle)


def _resolve_duplicate_tolerance(request: CoordinationRequest) -> float:
    candidate = request.parameters.get(
        "assignment_duplicate_tolerance", DEFAULT_ASSIGNMENT_DUPLICATE_TOLERANCE
    )
    return _finite_non_negative(candidate, name="assignment_duplicate_tolerance")


def _resolve_blocked_target_tolerance(request: CoordinationRequest, duplicate_tolerance: float) -> float:
    candidate = request.parameters.get("target_exclusion_radius", duplicate_tolerance)
    return _finite_non_negative(candidate, name="target_exclusion_radius")


def _resolve_min_frontier_travel_distance(request: CoordinationRequest) -> float:
    candidate = request.parameters.get(
        "min_frontier_travel_distance", DEFAULT_MIN_FRONTIER_TRAVEL_DISTANCE
    )
    return _finite_non_negative(candidate, name="min_frontier_travel_distance")


def _resolve_obstacle_point_tolerance(request: CoordinationRequest, resolution: float) -> float:
    default = max(0.05, 0.75 * resolution)
    candidate = request.parameters.get("obstacle_point_tolerance", default)
    return _finite_non_negative(candidate, name="obstacle_point_tolerance")


def _resolve_obstacle_line_half_width_override(request: CoordinationRequest) -> float | None:
    if "obstacle_line_half_width" not in request.parameters:
        return None
    return _finite_non_negative(
        request.parameters["obstacle_line_half_width"], name="obstacle_line_half_width"
    )


def _rejection_reason(cluster: FrontierCluster) -> str | None:
    """Hard-filter rule, checked in this exact order:
    cluster.valid is the sole authority for validity -- checked first, and
    a cluster failing it never reaches any other check (or clustering, or
    the matrix)."""
    if not cluster.valid:
        return "cluster marked invalid"
    if cluster.centroid is None:
        return "missing centroid"
    if not cluster.viewpoints:
        return "no viewpoints"
    candidate = cluster.as_exploration_candidate()
    if candidate is None:
        return "no assignable candidate"
    coords = (
        cluster.centroid[0],
        cluster.centroid[1],
        candidate.target[0],
        candidate.target[1],
    )
    if not all(math.isfinite(value) for value in coords):
        return "non-finite geometry"
    return None


def _filter_clusters(
    raw_clusters: tuple[FrontierCluster, ...],
) -> tuple[tuple[FrontierCluster, ...], tuple[dict[str, str], ...]]:
    valid: list[FrontierCluster] = []
    rejected: list[dict[str, str]] = []
    for cluster in raw_clusters:
        reason = _rejection_reason(cluster)
        if reason is None:
            valid.append(cluster)
        else:
            rejected.append({"cluster_id": cluster.cluster_id, "reason": reason})
    # Sorted by cluster_id so this diagnostic never depends on the order
    # the service happened to return raw_clusters in (see rule: reordering
    # the service's response must never change diagnostics).
    rejected.sort(key=lambda item: item["cluster_id"])
    return tuple(valid), tuple(rejected)


def _resolve_robots_to_assign(
    request: CoordinationRequest, robots_by_id: Mapping[int, RobotCoordinationState]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Returns (known_robots_to_assign_ids, unknown_ids). unknown_ids are
    requested robot_ids absent from request.robot_states -- they always
    become FAILED, never HOLD/ASSIGNED."""
    if request.robots_to_assign:
        requested = sorted({int(robot_id) for robot_id in request.robots_to_assign})
    else:
        requested = sorted(robot_id for robot_id, robot in robots_by_id.items() if robot.is_active)

    known = tuple(robot_id for robot_id in requested if robot_id in robots_by_id)
    unknown = tuple(robot_id for robot_id in requested if robot_id not in robots_by_id)
    return known, unknown


def _reserved_targets(
    request: CoordinationRequest,
    robots_by_id: Mapping[int, RobotCoordinationState],
    known_robots_to_assign_ids: tuple[int, ...],
) -> tuple[Point2D, ...]:
    """Targets belonging to robots that exist but are NOT being reassigned
    this round -- globally reserved so no reassigned robot duplicates
    them. A robot that IS being reassigned never reserves its own previous
    target."""
    reassigning = set(known_robots_to_assign_ids)
    reserved: list[Point2D] = []
    for robot_id, robot in robots_by_id.items():
        if robot_id in reassigning:
            continue
        target = request.existing_targets_by_robot.get(robot_id)
        if target is None:
            target = robot.current_target
        if target is not None:
            reserved.append((float(target[0]), float(target[1])))
    return tuple(reserved)


def _targets_and_reasons(
    request: CoordinationRequest,
    targets_by_robot: Mapping[int, Point2D | None],
    reasons_by_robot: Mapping[int, str],
) -> tuple[tuple[Point2D | None, ...], tuple[str, ...]]:
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
    return targets, reasons


def _round_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[round(float(value), 12) for value in row] for row in matrix]


def _default_cluster_task_debug(known_ids: tuple[int, ...]) -> dict[str, Any]:
    """Baseline diagnostic fields for every early-exit HOLD path, so
    result.debug always exposes the same key set regardless of which stage
    the plugin stopped at (no service / no valid clusters / no reduced
    tasks) -- callers never need to guard with .get()."""
    return {
        "raw_cluster_count": 0,
        "valid_cluster_count": 0,
        "rejected_cluster_count": 0,
        "rejected_clusters": [],
        "reduced_task_count": 0,
        "duplicate_task_targets_removed": 0,
        "task_ids": [],
        "task_source_cluster_ids": {},
        "weight_configuration": None,
        "utility_matrix": [],
        "feasible_matrix": [],
        "selected_task_by_robot": {str(robot_id): None for robot_id in known_ids},
    }


class FrontierClusterHungarianPlugin:
    """Two-stage frontier coordinator: reduce host-provided FrontierCluster
    components, score robot-task pairs, solve a global Hungarian
    assignment. See this module's docstring for the full pipeline and the
    Gao et al. inspiration/caveats."""

    metadata = PluginMetadata(
        name=FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR,
        version="0.1.0",
        description=(
            "Two-stage frontier coordinator inspired by Gao et al.'s "
            "frontier-based multi-robot exploration method: consumes host-"
            "detected FrontierCluster components, applies a deterministic "
            "grid-density approximation second-stage reduction (not "
            "GriT-DBSCAN), builds a global robot-task utility matrix "
            "(information + distance + five-line obstacle clearance), and "
            "solves task allocation with a pure-Python Hungarian solver."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TASK_ALLOCATION,
            # The grid-density reduction stage turns many host-detected
            # FrontierClusters into a smaller set of tasks before allocation
            # -- genuine task generation, not frontier detection (it never
            # touches raw occupancy/explored points itself).
            PluginCapability.TASK_GENERATION,
        ),
        # The host detects connected frontier components
        # (FrontierInformationService.get_frontier_clusters(), called exactly
        # once per assign() -- see debug["frontier_information_service_calls"]);
        # this plugin only reduces and allocates. It never falls back to
        # frontier_provider/team_frontier_provider/its own detection.
        candidate_input_mode=CandidateInputMode.HOST_FRONTIER_CLUSTERS,
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        robots_by_id = {robot.robot_id: robot for robot in request.robot_states}
        known_ids, unknown_ids = _resolve_robots_to_assign(request, robots_by_id)
        reserved_targets = _reserved_targets(request, robots_by_id, known_ids)

        frontier_information_service = None
        if request.services is not None:
            frontier_information_service = request.services.frontier_information_service

        if frontier_information_service is None:
            return self._hold_result(
                request=request,
                known_ids=known_ids,
                unknown_ids=unknown_ids,
                hold_reason=_REASON_NO_SERVICE,
                service_calls=0,
                extra_debug={},
            )

        raw_clusters = tuple(frontier_information_service.get_frontier_clusters())
        service_calls = 1

        valid_clusters, rejected_clusters = _filter_clusters(raw_clusters)

        if not valid_clusters:
            return self._hold_result(
                request=request,
                known_ids=known_ids,
                unknown_ids=unknown_ids,
                hold_reason=_REASON_NO_VALID_CLUSTERS,
                service_calls=service_calls,
                extra_debug={
                    "raw_cluster_count": len(raw_clusters),
                    "valid_cluster_count": 0,
                    "rejected_cluster_count": len(rejected_clusters),
                    "rejected_clusters": [dict(item) for item in rejected_clusters],
                },
            )

        resolution = _resolve_resolution(request)
        grid_size = _resolve_grid_size(request, resolution)
        merge_radius = _resolve_merge_radius(request, grid_size)
        duplicate_tolerance = _resolve_duplicate_tolerance(request)

        tasks, removed_duplicate_task_ids = reduce_frontier_clusters_with_diagnostics(
            valid_clusters,
            grid_size=grid_size,
            merge_radius=merge_radius,
            duplicate_tolerance=duplicate_tolerance,
        )

        if not tasks:
            return self._hold_result(
                request=request,
                known_ids=known_ids,
                unknown_ids=unknown_ids,
                hold_reason=_REASON_NO_TASKS,
                service_calls=service_calls,
                extra_debug={
                    "raw_cluster_count": len(raw_clusters),
                    "valid_cluster_count": len(valid_clusters),
                    "rejected_cluster_count": len(rejected_clusters),
                    "rejected_clusters": [dict(item) for item in rejected_clusters],
                    "reduced_task_count": 0,
                    "duplicate_task_targets_removed": len(removed_duplicate_task_ids),
                },
            )

        weights = _resolve_weights(request)
        min_travel = _resolve_min_frontier_travel_distance(request)
        blocked_tolerance = _resolve_blocked_target_tolerance(request, duplicate_tolerance)
        obstacle_tolerance = _resolve_obstacle_point_tolerance(request, resolution)
        obstacle_override = _resolve_obstacle_line_half_width_override(request)

        robot_xy_by_id = {robot_id: robots_by_id[robot_id].xy for robot_id in known_ids}
        safety_radius_by_robot = {robot_id: robots_by_id[robot_id].safety_radius for robot_id in known_ids}
        blocked_targets_by_robot = {
            robot_id: tuple(request.blocked_targets_by_robot.get(robot_id, ())) for robot_id in known_ids
        }
        observed_obstacle_points = tuple(
            (float(x), float(y))
            for x, y in (request.world.mapped_obstacle_points if request.world is not None else ())
        )

        ordered_tasks = tuple(sorted(tasks, key=lambda task: task.task_id))
        task_ids_ordered = [task.task_id for task in ordered_tasks]
        task_by_id = {task.task_id: task for task in ordered_tasks}

        utility_cells = build_utility_matrix(
            robot_ids=known_ids,
            robot_xy_by_id=robot_xy_by_id,
            tasks=ordered_tasks,
            blocked_targets_by_robot=blocked_targets_by_robot,
            reserved_targets=reserved_targets,
            observed_obstacle_points=observed_obstacle_points,
            safety_radius_by_robot=safety_radius_by_robot,
            obstacle_line_half_width_override=obstacle_override,
            point_tolerance=obstacle_tolerance,
            min_frontier_travel_distance=min_travel,
            blocked_target_tolerance=blocked_tolerance,
            weights=weights,
        )
        cell_by_pair = {(cell.robot_id, cell.task_id): cell for cell in utility_cells}

        utility_matrix = [
            [cell_by_pair[(robot_id, task_id)].utility for task_id in task_ids_ordered]
            for robot_id in known_ids
        ]
        feasible_matrix = [
            [cell_by_pair[(robot_id, task_id)].feasible for task_id in task_ids_ordered]
            for robot_id in known_ids
        ]

        selected_indices = solve_max_utility_assignment(utility_matrix, feasible_matrix)

        assignments: list[CoordinationAssignment] = []
        commands: list[RobotCommand] = []
        targets_by_robot: dict[int, Point2D | None] = {}
        reasons_by_robot: dict[int, str] = {}
        selected_task_by_robot: dict[str, str | None] = {}

        for unknown_id in unknown_ids:
            assignments.append(
                CoordinationAssignment(
                    robot_id=unknown_id, target=None, status="FAILED", reason=_REASON_UNKNOWN_ROBOT
                )
            )
            commands.append(RobotCommand(robot_id=unknown_id, status="FAILED", reason=_REASON_UNKNOWN_ROBOT))

        for row_index, robot_id in enumerate(known_ids):
            selected_index = selected_indices[row_index]

            if selected_index is None:
                row_has_feasible_task = any(feasible_matrix[row_index])
                reason = _REASON_NO_FEASIBLE_TASK if not row_has_feasible_task else _REASON_HUNGARIAN_HOLD
                assignments.append(
                    CoordinationAssignment(robot_id=robot_id, target=None, status="HOLD", reason=reason)
                )
                commands.append(RobotCommand(robot_id=robot_id, status="HOLD", reason=reason))
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                selected_task_by_robot[str(robot_id)] = None
                continue

            task_id = task_ids_ordered[selected_index]
            task = task_by_id[task_id]
            cell = cell_by_pair[(robot_id, task_id)]

            reason = (
                f"assigned by {self.metadata.name}: task={task.task_id}, "
                f"representative_cluster={task.representative_cluster_id}"
            )
            proposal_metadata = {
                **dict(task.candidate.metadata),
                "task_id": task.task_id,
                "source_cluster_ids": task.source_cluster_ids,
                "representative_cluster_id": task.representative_cluster_id,
                "reduced_cluster_count": len(task.source_cluster_ids),
                "reduced_frontier_cell_count": len(task.cells),
                "information_score": cell.information_score,
                "distance_score": cell.distance_score,
                "blocked_line_fraction": cell.blocked_line_fraction,
                "clearance_score": cell.clearance_score,
                "assignment_utility": cell.utility,
                "clustering_method": CLUSTERING_METHOD,
            }
            proposal = replace(
                task.candidate,
                information_gain=task.information_gain,
                travel_cost=cell.distance,
                safety_cost=cell.blocked_line_fraction,
                metadata=proposal_metadata,
            )

            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id, target=task.target, status="ASSIGNED", proposal=proposal, reason=reason
                )
            )
            commands.append(
                RobotCommand(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=task.target,
                    heading_rad=task.heading_rad,
                    reason=reason,
                )
            )
            targets_by_robot[robot_id] = task.target
            reasons_by_robot[robot_id] = reason
            selected_task_by_robot[str(robot_id)] = task.task_id

        targets, reasons = _targets_and_reasons(request, targets_by_robot, reasons_by_robot)

        debug: dict[str, Any] = {
            "plugin": self.metadata.name,
            "capabilities": [capability.value for capability in self.metadata.capabilities],
            "robots_to_assign": list(known_ids),
            "frontier_information_service_calls": service_calls,
            "raw_cluster_count": len(raw_clusters),
            "valid_cluster_count": len(valid_clusters),
            "rejected_cluster_count": len(rejected_clusters),
            "rejected_clusters": [dict(item) for item in rejected_clusters],
            "reduced_task_count": len(tasks),
            "duplicate_task_targets_removed": len(removed_duplicate_task_ids),
            "task_ids": list(task_ids_ordered),
            "task_source_cluster_ids": {
                task.task_id: list(task.source_cluster_ids) for task in ordered_tasks
            },
            "weight_configuration": {
                "information": weights.information,
                "distance": weights.distance,
                "obstacle": weights.obstacle,
            },
            "utility_matrix": _round_matrix(utility_matrix),
            "feasible_matrix": [[bool(value) for value in row] for row in feasible_matrix],
            "selected_task_by_robot": selected_task_by_robot,
        }

        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            debug=debug,
            commands=tuple(commands),
        )

    def _hold_result(
        self,
        *,
        request: CoordinationRequest,
        known_ids: tuple[int, ...],
        unknown_ids: tuple[int, ...],
        hold_reason: str,
        service_calls: int,
        extra_debug: Mapping[str, Any],
    ) -> CoordinationResult:
        assignments: list[CoordinationAssignment] = []
        commands: list[RobotCommand] = []
        targets_by_robot: dict[int, Point2D | None] = {}
        reasons_by_robot: dict[int, str] = {}

        for unknown_id in unknown_ids:
            assignments.append(
                CoordinationAssignment(
                    robot_id=unknown_id, target=None, status="FAILED", reason=_REASON_UNKNOWN_ROBOT
                )
            )
            commands.append(RobotCommand(robot_id=unknown_id, status="FAILED", reason=_REASON_UNKNOWN_ROBOT))

        for robot_id in known_ids:
            assignments.append(
                CoordinationAssignment(robot_id=robot_id, target=None, status="HOLD", reason=hold_reason)
            )
            commands.append(RobotCommand(robot_id=robot_id, status="HOLD", reason=hold_reason))
            targets_by_robot[robot_id] = None
            reasons_by_robot[robot_id] = hold_reason

        targets, reasons = _targets_and_reasons(request, targets_by_robot, reasons_by_robot)

        debug: dict[str, Any] = {
            "plugin": self.metadata.name,
            "capabilities": [capability.value for capability in self.metadata.capabilities],
            "robots_to_assign": list(known_ids),
            "frontier_information_service_calls": service_calls,
            **_default_cluster_task_debug(known_ids),
            **extra_debug,
        }

        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            debug=debug,
            commands=tuple(commands),
        )


def create_plugin() -> CoordinationPlugin:
    return FrontierClusterHungarianPlugin()
