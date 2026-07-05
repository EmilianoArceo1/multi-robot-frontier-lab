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
from robotics_sim.planning.coordinated_frontier_planner import assign_frontier_viewpoints

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

        index_by_robot_id = {
            int(robot.robot_id): index
            for index, robot in enumerate(request.robot_states)
        }
        robot_ids_to_assign = (
            tuple(int(robot_id) for robot_id in request.robots_to_assign)
            if request.robots_to_assign
            else tuple(int(robot.robot_id) for robot in request.robot_states if robot.is_active)
        )
        planner_indices = tuple(
            index_by_robot_id[robot_id]
            for robot_id in robot_ids_to_assign
            if robot_id in index_by_robot_id
        )
        if not planner_indices:
            return {}

        existing_targets = tuple(
            _normalize_point(request.existing_targets_by_robot.get(robot.robot_id))
            for robot in request.robot_states
        )
        invalidated_targets = tuple(
            tuple(_valid_points(request.blocked_targets_by_robot.get(robot.robot_id, ())))
            for robot in request.robot_states
        )
        route_points_by_robot = _align_routes(
            request.route_points_by_robot,
            count=len(request.robot_states),
        )
        explored_by_robot = _explored_points_by_robot(request)

        result = assign_frontier_viewpoints(
            robot_states=request.robot_states,
            existing_targets=existing_targets,
            robots_to_assign=planner_indices,
            invalidated_targets_by_robot=invalidated_targets,
            explored_points=tuple(_valid_points(world.explored_points)),
            mapped_obstacle_points=tuple(_valid_points(world.mapped_obstacle_points)),
            bounds=tuple(float(value) for value in world.bounds),
            resolution=float(world.resolution),
            final_goal_xy=_normalize_point(world.final_goal_xy) or (0.0, 0.0),
            ipp_distance_penalty=float(self.ipp_distance_penalty),
            target_exclusion_radius=float(self.target_exclusion_radius),
            dynamic_obstacle_margin=float(self.dynamic_obstacle_margin),
            route_points_by_robot=route_points_by_robot,
            explored_points_by_robot=explored_by_robot,
        )

        candidates_by_robot: dict[int, tuple[ExplorationCandidate, ...]] = {}
        for robot in request.robot_states:
            robot_id = int(robot.robot_id)
            index = index_by_robot_id[robot_id]
            target = (
                _normalize_point(result.targets[index])
                if index < len(getattr(result, "targets", ()))
                else None
            )
            if target is None:
                candidates_by_robot[robot_id] = ()
                continue

            assignment = None
            assignments = getattr(result, "assignments", ())
            if index < len(assignments):
                assignment = assignments[index]

            reason = "runtime team frontier provider"
            information_gain = 1.0
            travel_cost = 0.0
            overlap_cost = 0.0
            score = None

            if assignment is not None:
                reason = str(getattr(assignment, "reason", reason))
                information_gain = float(getattr(assignment, "information_gain", information_gain))
                distance = float(getattr(assignment, "distance", 0.0))
                travel_cost = float(self.ipp_distance_penalty) * distance
                overlap_cost = float(getattr(assignment, "route_overlap_ratio", 0.0))
                score = getattr(assignment, "score", None)

            metadata = {
                "robot_id": robot_id,
                "provider": type(self).__name__,
                "reason": reason,
                "team_synchronized": True,
            }
            if isinstance(score, (int, float)):
                metadata["score"] = float(score)

            candidates_by_robot[robot_id] = (
                ExplorationCandidate(
                    target=target,
                    source="runtime_team_frontier_provider",
                    information_gain=information_gain,
                    travel_cost=travel_cost,
                    overlap_cost=overlap_cost,
                    metadata=metadata,
                ),
            )

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
