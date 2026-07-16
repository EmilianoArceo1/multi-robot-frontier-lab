"""Runtime service adapters for simulator-independent coordination plugins.

This module is on the simulator side of the boundary.  It may import
robotics_sim internals, but external algorithms must not import this module.

The goal is to expose simulator capabilities through robotics_interfaces
protocols, so plugins can request frontier candidates without depending on
engine.py, Qt, canvas objects, or concrete planner modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.planning.coordinated_frontier_planner import (
    assign_frontier_viewpoints,
    detect_global_frontier_candidates,
)

Point2D = tuple[float, float]


@dataclass(frozen=True)
class RuntimeTeamFrontierProvider:
    """TeamFrontierProvider backed by the current coordinated frontier planner.

    Important: this calls the legacy planner once with the whole team.  Calling
    the planner independently per robot loses target reservations and is the
    failure mode that caused duplicated/too-close frontiers in MMPF.
    """

    ipp_distance_penalty: float = 0.5
    target_exclusion_radius: float = 1.5
    dynamic_obstacle_margin: float = 0.5

    def candidates_for_team(
        self,
        request: CoordinationRequest,
    ) -> Mapping[int, tuple[ExplorationCandidate, ...]]:
        world = request.world
        if world is None or world.bounds is None or not request.robot_states:
            return {}

        robot_ids_to_assign = (
            tuple(int(robot_id) for robot_id in request.robots_to_assign)
            if request.robots_to_assign
            else tuple(int(robot.robot_id) for robot in request.robot_states if robot.is_active)
        )
        if not robot_ids_to_assign:
            return {}

        active_assignable_robot_ids = {
            int(robot.robot_id)
            for robot in request.robot_states
            if int(robot.robot_id) in robot_ids_to_assign and robot.is_active
        }
        if not active_assignable_robot_ids:
            return {}

        max_robot_radius = max(
            (float(robot.safety_radius) for robot in request.robot_states),
            default=0.0,
        )
        avg_sensor_range = (
            sum(float(robot.sensor_range) for robot in request.robot_states)
            / max(len(request.robot_states), 1)
        )

        global_candidates = detect_global_frontier_candidates(
            explored_points=tuple(_valid_points(world.explored_points)),
            mapped_obstacle_points=tuple(_valid_points(world.mapped_obstacle_points)),
            bounds=tuple(float(value) for value in world.bounds),
            resolution=float(world.resolution),
            robot_radius=max_robot_radius,
            sensor_range=avg_sensor_range,
        )

        candidates_by_robot: dict[int, tuple[ExplorationCandidate, ...]] = {}

        for robot in request.robot_states:
            robot_id = int(robot.robot_id)
            if robot_id not in active_assignable_robot_ids:
                continue

            blocked_targets = tuple(
                _valid_points(request.blocked_targets_by_robot.get(robot_id, ()))
            )

            existing_targets = tuple(
                point
                for other_id, target in request.existing_targets_by_robot.items()
                if int(other_id) != robot_id
                if (point := _normalize_point(target)) is not None
            )

            robot_candidates: list[ExplorationCandidate] = []

            for candidate in global_candidates:
                target = _normalize_point(candidate.target)
                if target is None:
                    continue

                # Hard filter: a target this robot has blacklisted must never
                # be handed back to it as a candidate. Raising safety_cost or
                # attaching metadata is not enough -- that only works if
                # every plugin consuming this pool happens to read and
                # respect it, which is exactly the gap Codex review found.
                # Uses the same spatial-tolerance helper/radius as the
                # `reserved` check below rather than exact float equality.
                if _point_near_any(target, blocked_targets, self.target_exclusion_radius):
                    continue

                distance = _distance(robot.xy, target)
                reserved = _point_near_any(
                    target,
                    existing_targets,
                    self.target_exclusion_radius,
                )

                safety_cost = 8.0 if reserved else 0.0

                metadata = {
                    "robot_id": robot_id,
                    "provider": type(self).__name__,
                    "reason": str(
                        getattr(candidate, "reason", "runtime team frontier candidate")
                    ),
                    "team_synchronized": True,
                    "distance": distance,
                    "frontier_size": int(getattr(candidate, "size", 0)),
                    "raw_score": float(getattr(candidate, "score", 0.0)),
                    "reserved_by_existing_target": reserved,
                }

                robot_candidates.append(
                    ExplorationCandidate(
                        target=target,
                        source="runtime_team_frontier_provider",
                        information_gain=float(getattr(candidate, "information_gain", 0.0)),
                        travel_cost=float(self.ipp_distance_penalty) * distance,
                        safety_cost=safety_cost,
                        metadata=metadata,
                    )
                )

            robot_candidates.sort(
                key=lambda item: (
                    item.utility,
                    item.information_gain,
                    -float(item.metadata.get("distance", 0.0)),
                ),
                reverse=True,
            )

            candidates_by_robot[robot_id] = tuple(robot_candidates)

        return candidates_by_robot


@dataclass(frozen=True)
class RuntimeFrontierProvider:
    """Single-robot fallback FrontierProvider backed by the simulator planner.

    Prefer RuntimeTeamFrontierProvider for multi-robot coordination. This class
    remains for simple plugins and compatibility tests.
    """

    ipp_distance_penalty: float = 0.5
    target_exclusion_radius: float = 1.5
    dynamic_obstacle_margin: float = 0.5
    route_points_by_robot: tuple[tuple[Point2D, ...], ...] = ()
    explored_points_by_robot: tuple[tuple[Point2D, ...], ...] = ()

    def candidates_for_robot(
        self,
        robot: RobotCoordinationState,
        world: WorldSnapshot,
        blocked_targets: tuple[Point2D, ...] = (),
    ) -> tuple[ExplorationCandidate, ...]:
        bounds = world.bounds
        if bounds is None:
            return ()

        final_goal_xy = world.final_goal_xy or (0.0, 0.0)

        result = assign_frontier_viewpoints(
            robot_states=(robot,),
            existing_targets=(None,),
            robots_to_assign=(0,),
            invalidated_targets_by_robot=(tuple(blocked_targets),),
            explored_points=tuple(_valid_points(world.explored_points)),
            mapped_obstacle_points=tuple(_valid_points(world.mapped_obstacle_points)),
            bounds=tuple(float(value) for value in bounds),
            resolution=float(world.resolution),
            final_goal_xy=_normalize_point(final_goal_xy) or (0.0, 0.0),
            ipp_distance_penalty=float(self.ipp_distance_penalty),
            target_exclusion_radius=float(self.target_exclusion_radius),
            dynamic_obstacle_margin=float(self.dynamic_obstacle_margin),
            route_points_by_robot=(
                self._route_for_robot(robot.robot_id),
            ),
            explored_points_by_robot=(
                self._explored_points_for_robot(robot.robot_id),
            ),
        )

        if not result.targets or result.targets[0] is None:
            return ()

        assignment = None
        if getattr(result, "assignments", None):
            assignment = result.assignments[0]

        target = _normalize_point(result.targets[0])
        if target is None:
            return ()

        reason = "runtime frontier provider"
        information_gain = 1.0
        travel_cost = 0.0
        score = None

        if assignment is not None:
            reason = str(getattr(assignment, "reason", reason))
            information_gain = float(getattr(assignment, "information_gain", information_gain))
            distance = float(getattr(assignment, "distance", 0.0))
            travel_cost = float(self.ipp_distance_penalty) * distance
            score = getattr(assignment, "score", None)

        metadata = {
            "robot_id": robot.robot_id,
            "provider": type(self).__name__,
            "reason": reason,
            "team_synchronized": False,
        }
        if isinstance(score, (int, float)):
            metadata["score"] = float(score)

        return (
            ExplorationCandidate(
                target=target,
                source="runtime_frontier_provider",
                information_gain=information_gain,
                travel_cost=travel_cost,
                metadata=metadata,
            ),
        )

    def _route_for_robot(self, robot_id: int) -> tuple[Point2D, ...]:
        if 0 <= int(robot_id) < len(self.route_points_by_robot):
            return tuple(_valid_points(self.route_points_by_robot[int(robot_id)]))
        return ()

    def _explored_points_for_robot(self, robot_id: int) -> tuple[Point2D, ...]:
        if 0 <= int(robot_id) < len(self.explored_points_by_robot):
            return tuple(_valid_points(self.explored_points_by_robot[int(robot_id)]))
        return ()

def _distance(a: Point2D, b: Point2D) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _point_near_any(
    point: Point2D,
    others: Sequence[Point2D],
    radius: float,
) -> bool:
    radius = max(float(radius), 0.0)
    return any(_distance(point, other) <= radius for other in others)

def _explored_points_by_robot(
    request: CoordinationRequest,
) -> tuple[tuple[Point2D, ...], ...]:
    raw = request.shared.get("explored_points_by_robot", ())
    if not raw:
        return tuple(() for _ in request.robot_states)
    return tuple(
        tuple(_valid_points(points))
        for points in tuple(raw)[: len(request.robot_states)]
    ) + tuple(
        () for _ in range(max(0, len(request.robot_states) - len(tuple(raw))))
    )


def _align_routes(
    routes: Sequence[Sequence[Point2D]] | None,
    *,
    count: int,
) -> tuple[tuple[Point2D, ...], ...]:
    normalized = tuple(tuple(_valid_points(route)) for route in (routes or ()))
    if len(normalized) >= count:
        return normalized[:count]
    return normalized + tuple(() for _ in range(count - len(normalized)))


def _valid_points(points: Sequence[object] | None) -> tuple[Point2D, ...]:
    return tuple(
        point
        for value in (points or ())
        if (point := _normalize_point(value)) is not None
    )


def _normalize_point(value: object) -> Point2D | None:
    if value is None:
        return None
    try:
        x, y = value  # type: ignore[misc]
        return (float(x), float(y))
    except (TypeError, ValueError):
        return None
