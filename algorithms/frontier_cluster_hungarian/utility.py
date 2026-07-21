"""Global robot-task utility matrix: information + distance + five-line
obstacle clearance, combined into one deterministic Hungarian-ready score
per (robot, task) pair.

    utility_ij = alpha * information_score_j
               + beta  * distance_score_ij
               + gamma * clearance_score_ij

alpha/beta/gamma are the normalized hungarian_information_weight/
hungarian_distance_weight/hungarian_obstacle_weight parameters (see
normalize_weights()). Obstacle presence is only ever rewarded through
clearance (fewer blocked lines -> higher score); it is never added as a
positive "obstacle occurrence" term on its own, which would perversely
reward obstacles instead of penalizing them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence, TypeVar

from robotics_interfaces.observations import Point2D

from algorithms.frontier_cluster_hungarian.clustering import ReducedFrontierTask
from algorithms.frontier_cluster_hungarian.obstacle_factor import (
    five_line_blocked_fraction,
    five_line_clearance_score,
)

K = TypeVar("K")


@dataclass(frozen=True)
class UtilityWeights:
    information: float
    distance: float
    obstacle: float


@dataclass(frozen=True)
class UtilityCell:
    robot_id: int
    task_id: str
    feasible: bool
    distance: float
    information_score: float
    distance_score: float
    blocked_line_fraction: float
    clearance_score: float
    utility: float
    rejection_reason: str | None = None


def normalize_weights(
    *, information_weight: float, distance_weight: float, obstacle_weight: float
) -> UtilityWeights:
    """Validate (finite, non-negative, sum > 0) and normalize the three
    Hungarian weights to sum to 1.0. Never mutates its inputs."""
    raw = (float(information_weight), float(distance_weight), float(obstacle_weight))
    for weight in raw:
        if not math.isfinite(weight):
            raise ValueError(f"hungarian weights must be finite numbers, got {raw!r}")
        if weight < 0.0:
            raise ValueError(f"hungarian weights must be non-negative, got {raw!r}")
    total = sum(raw)
    if total <= 0.0:
        raise ValueError(f"hungarian weights must sum to more than zero, got {raw!r}")
    information, distance, obstacle = (weight / total for weight in raw)
    return UtilityWeights(information=information, distance=distance, obstacle=obstacle)


def _dense_rank_scores(values_by_key: Mapping[K, float], *, higher_is_better: bool) -> dict[K, float]:
    """Dense-rank-normalize values to [0.0, 1.0]: the best value maps to
    1.0, the worst to 0.0, ties share the same score, and a single distinct
    value (including a single key) maps everyone to 1.0."""
    if not values_by_key:
        return {}
    distinct = sorted(set(values_by_key.values()), reverse=higher_is_better)
    if len(distinct) <= 1:
        return {key: 1.0 for key in values_by_key}
    rank_by_value = {value: rank for rank, value in enumerate(distinct)}
    max_rank = len(distinct) - 1
    return {key: 1.0 - (rank_by_value[value] / max_rank) for key, value in values_by_key.items()}


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def evaluate_feasibility(
    *,
    robot_xy: Point2D,
    target: Point2D,
    blocked_targets: Sequence[Point2D],
    reserved_targets: Sequence[Point2D],
    min_frontier_travel_distance: float,
    blocked_target_tolerance: float,
) -> tuple[bool, str | None, float]:
    """Pure per-(robot, task) feasibility check. Returns
    (feasible, rejection_reason, distance). A pair is infeasible when:
      - the target has non-finite coordinates;
      - the robot-target distance is below min_frontier_travel_distance;
      - the target is blocked for this robot (blocked_targets);
      - the target is reserved by a robot that is not being reassigned
        this round (reserved_targets).
    """
    if not (math.isfinite(target[0]) and math.isfinite(target[1])):
        return False, "target coordinates are not finite", float("inf")

    distance = _distance(robot_xy, target)

    if distance < float(min_frontier_travel_distance):
        return False, "target closer than min_frontier_travel_distance", distance

    tolerance = max(float(blocked_target_tolerance), 0.0)
    if any(_distance(target, blocked) <= tolerance for blocked in blocked_targets):
        return False, "target blocked for this robot", distance

    if any(_distance(target, reserved) <= tolerance for reserved in reserved_targets):
        return False, "target reserved by a non-reassigned robot", distance

    return True, None, distance


def build_utility_matrix(
    *,
    robot_ids: Sequence[int],
    robot_xy_by_id: Mapping[int, Point2D],
    tasks: Sequence[ReducedFrontierTask],
    blocked_targets_by_robot: Mapping[int, Sequence[Point2D]],
    reserved_targets: Sequence[Point2D],
    observed_obstacle_points: Sequence[Point2D],
    safety_radius_by_robot: Mapping[int, float],
    obstacle_line_half_width_override: float | None,
    point_tolerance: float,
    min_frontier_travel_distance: float,
    blocked_target_tolerance: float,
    weights: UtilityWeights,
) -> tuple[UtilityCell, ...]:
    """Build the full, deterministic robot x task utility matrix.

    Rows are robot_ids in ascending order; columns are task_ids in
    ascending order. information_score is ranked globally across all
    tasks (rule: "ranking denso normalizado global entre tareas");
    distance_score is ranked per robot row, over only the tasks feasible
    for that robot (infeasible tasks never enter that row's ranking).
    """
    ordered_robot_ids = sorted(robot_ids)
    ordered_tasks = sorted(tasks, key=lambda task: task.task_id)

    information_gain_by_task = {task.task_id: task.information_gain for task in ordered_tasks}
    information_score_by_task = _dense_rank_scores(information_gain_by_task, higher_is_better=True)

    cells: list[UtilityCell] = []
    for robot_id in ordered_robot_ids:
        robot_xy = robot_xy_by_id[robot_id]
        blocked_targets = tuple(blocked_targets_by_robot.get(robot_id, ()))
        safety_radius = float(safety_radius_by_robot.get(robot_id, 0.0))
        line_half_width = (
            float(obstacle_line_half_width_override)
            if obstacle_line_half_width_override is not None
            else safety_radius
        )

        row_feasibility: dict[str, tuple[bool, str | None, float]] = {}
        row_blocked_fraction: dict[str, float] = {}
        for task in ordered_tasks:
            feasible, reason, distance = evaluate_feasibility(
                robot_xy=robot_xy,
                target=task.target,
                blocked_targets=blocked_targets,
                reserved_targets=reserved_targets,
                min_frontier_travel_distance=min_frontier_travel_distance,
                blocked_target_tolerance=blocked_target_tolerance,
            )
            row_feasibility[task.task_id] = (feasible, reason, distance)
            row_blocked_fraction[task.task_id] = (
                five_line_blocked_fraction(
                    robot_xy=robot_xy,
                    target_xy=task.target,
                    observed_obstacle_points=observed_obstacle_points,
                    safety_radius=line_half_width,
                    point_tolerance=point_tolerance,
                )
                if feasible
                else 0.0
            )

        feasible_distance_by_task = {
            task.task_id: row_feasibility[task.task_id][2]
            for task in ordered_tasks
            if row_feasibility[task.task_id][0]
        }
        distance_score_by_task = _dense_rank_scores(feasible_distance_by_task, higher_is_better=False)

        for task in ordered_tasks:
            feasible, reason, distance = row_feasibility[task.task_id]
            blocked_fraction = row_blocked_fraction[task.task_id]
            information_score = information_score_by_task.get(task.task_id, 0.0)

            if feasible:
                clearance_score = five_line_clearance_score(blocked_fraction)
                distance_score = distance_score_by_task.get(task.task_id, 0.0)
                utility = (
                    weights.information * information_score
                    + weights.distance * distance_score
                    + weights.obstacle * clearance_score
                )
            else:
                clearance_score = 0.0
                distance_score = 0.0
                utility = 0.0

            cells.append(
                UtilityCell(
                    robot_id=robot_id,
                    task_id=task.task_id,
                    feasible=feasible,
                    distance=distance,
                    information_score=information_score,
                    distance_score=distance_score,
                    blocked_line_fraction=blocked_fraction,
                    clearance_score=clearance_score,
                    utility=utility,
                    rejection_reason=reason,
                )
            )

    return tuple(cells)
