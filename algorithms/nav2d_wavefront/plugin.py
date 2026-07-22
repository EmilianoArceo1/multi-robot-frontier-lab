"""Native port of Nav2D's multi-robot wavefront frontier allocator.

Reference implementation:
https://github.com/skasperski/navigation_2d/blob/master/nav2d_exploration/src/MultiWavefrontPlanner.cpp

The plugin intentionally depends only on ``robotics_interfaces``.  It rebuilds
the known-free/occupied/unknown grid from the immutable request snapshot,
grows one synchronized wavefront from every robot, and assigns each robot the
first frontier claimed by its Voronoi region.  If a region has no usable
frontier, a second wave is allowed to cross another robot's region, matching
Nav2D's ``mWaitForOthers == false`` fallback.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Iterable

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.observations import Point2D, RobotCoordinationState, WorldBounds
from robotics_interfaces.plugins import (
    CandidateInputMode,
    CoordinationPlugin,
    PluginCapability,
    PluginMetadata,
)
from robotics_interfaces.proposals import ExplorationCandidate

NAV2D_WAVEFRONT_COORDINATOR = "Nav2D multi-wavefront coordinator"

_UNKNOWN = 0
_FREE = 1
_OCCUPIED = 2


class _Grid:
    def __init__(self, bounds: WorldBounds, resolution: float):
        self.x_min, self.x_max, self.y_min, self.y_max = (float(value) for value in bounds)
        self.resolution = max(float(resolution), 1e-6)
        self.width = max(1, int(math.ceil((self.x_max - self.x_min) / self.resolution)))
        self.height = max(1, int(math.ceil((self.y_max - self.y_min) / self.resolution)))
        self.data = bytearray(self.width * self.height)

    def valid(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def index(self, cell: tuple[int, int]) -> int:
        row, col = cell
        return row * self.width + col

    def state(self, cell: tuple[int, int]) -> int:
        return self.data[self.index(cell)] if self.valid(cell) else _UNKNOWN

    def set_state(self, cell: tuple[int, int], state: int) -> None:
        if self.valid(cell):
            self.data[self.index(cell)] = int(state)

    def world_to_cell(self, point: Point2D) -> tuple[int, int] | None:
        col = int(math.floor((float(point[0]) - self.x_min) / self.resolution))
        row = int(math.floor((float(point[1]) - self.y_min) / self.resolution))
        cell = (row, col)
        return cell if self.valid(cell) else None

    def cell_to_world(self, cell: tuple[int, int]) -> Point2D:
        row, col = cell
        return (
            self.x_min + (col + 0.5) * self.resolution,
            self.y_min + (row + 0.5) * self.resolution,
        )

    def free_cells(self) -> Iterable[tuple[int, int]]:
        for index, state in enumerate(self.data):
            if state == _FREE:
                yield (index // self.width, index % self.width)


def _neighbors8(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    row, col = cell
    return tuple(
        (row + dr, col + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    )


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _is_frontier(grid: _Grid, cell: tuple[int, int]) -> bool:
    if grid.state(cell) != _FREE:
        return False
    # GridMap::isFrontier() in Nav2D uses all eight neighbors; an index
    # outside the map reads as unknown, so a known-free boundary cell is a
    # frontier too.
    return any(grid.state(neighbor) == _UNKNOWN for neighbor in _neighbors8(cell))


def _nearest_free_cell(grid: _Grid, point: Point2D) -> tuple[int, int] | None:
    requested = grid.world_to_cell(point)
    if requested is not None and grid.state(requested) == _FREE:
        return requested
    if requested is None:
        requested = (grid.height // 2, grid.width // 2)
    return min(
        grid.free_cells(),
        key=lambda cell: (
            abs(cell[0] - requested[0]) + abs(cell[1] - requested[1]),
            cell[0],
            cell[1],
        ),
        default=None,
    )


def _build_grid(request: CoordinationRequest) -> _Grid | None:
    world = request.world
    if world is None or world.bounds is None:
        return None

    resolution = max(float(world.resolution), 1e-6)
    grid = _Grid(world.bounds, resolution)

    # Exploration observations first establish known-free space.  Confirmed
    # obstacle samples then override those cells and receive the same lethal
    # padding Nav2D applies before running its exploration wavefront.
    for point in world.explored_points:
        cell = grid.world_to_cell(point)
        if cell is not None:
            grid.set_state(cell, _FREE)

    obstacle_cells = {
        cell
        for point in world.mapped_obstacle_points
        if (cell := grid.world_to_cell(point)) is not None
    }
    max_safety_radius = max(
        (float(robot.safety_radius) for robot in request.robot_states),
        default=0.0,
    )
    cell_radius = int(math.ceil(max_safety_radius / resolution))
    center_threshold = max_safety_radius + resolution * math.sqrt(2.0) * 0.5
    offsets = tuple(
        (dr, dc)
        for dr in range(-cell_radius - 1, cell_radius + 2)
        for dc in range(-cell_radius - 1, cell_radius + 2)
        if math.hypot(dr * resolution, dc * resolution) <= center_threshold
    )
    for row, col in obstacle_cells:
        for dr, dc in offsets:
            grid.set_state((row + dr, col + dc), _OCCUPIED)

    # Nav2D clears the footprint around the current pose after map inflation.
    # Clearing every team seed is the centralized equivalent and prevents a
    # mapped wall sample/noisy overlap from deleting a robot's wave source.
    for robot in request.robot_states:
        center = grid.world_to_cell(robot.xy)
        if center is None:
            continue
        radius = max(0.0, float(robot.safety_radius))
        cells = int(math.ceil(radius / resolution))
        for dr in range(-cells, cells + 1):
            for dc in range(-cells, cells + 1):
                if math.hypot(dr * resolution, dc * resolution) <= radius:
                    grid.set_state((center[0] + dr, center[1] + dc), _FREE)

    return grid


class Nav2DMultiWavefrontPlugin:
    metadata = PluginMetadata(
        name=NAV2D_WAVEFRONT_COORDINATOR,
        version="1.0.0",
        description=(
            "Port of navigation_2d MultiWavefrontPlanner: synchronized "
            "multi-source wavefront frontier allocation with cross-region fallback."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TARGET_GENERATION,
            PluginCapability.TASK_ALLOCATION,
            # This plugin rebuilds its own occupancy grid from
            # request.world.explored_points/mapped_obstacle_points and finds
            # frontier cells itself (_build_grid/_is_frontier) -- real
            # frontier detection, not host-provided candidates.
            PluginCapability.FRONTIER_DETECTION,
            # The synchronized wavefront claims cells and selects one target
            # per robot from its own detected frontier cells -- task
            # generation over its own detection output, not host reduction.
            PluginCapability.TASK_GENERATION,
        ),
        source="https://github.com/skasperski/navigation_2d",
        # This plugin never reads request.services at all (no
        # frontier_provider/team_frontier_provider/frontier_information_
        # service) -- it must keep working with services=None. See
        # test_nav2d_wavefront_assigns_targets_without_any_host_frontier_service.
        candidate_input_mode=CandidateInputMode.PLUGIN_INTERNAL,
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        robots_by_id = {int(robot.robot_id): robot for robot in request.robot_states}
        requested = tuple(
            int(robot_id)
            for robot_id in (
                request.robots_to_assign
                or tuple(robot.robot_id for robot in request.robot_states if robot.is_active)
            )
        )
        grid = _build_grid(request)
        if grid is None:
            return self._all_hold(request, requested, "world snapshot or bounds unavailable")

        seed_by_robot = {
            robot_id: _nearest_free_cell(grid, robot.xy)
            for robot_id, robot in robots_by_id.items()
            if robot.is_active
        }
        seed_by_robot = {robot_id: cell for robot_id, cell in seed_by_robot.items() if cell is not None}
        if not seed_by_robot:
            return self._all_hold(request, requested, "known map contains no free wavefront seed")

        reservation_radius = max(
            float(request.parameters.get("target_exclusion_radius", grid.resolution)),
            grid.resolution,
        )
        min_travel = max(
            0.0,
            float(request.parameters.get("min_frontier_travel_distance", grid.resolution)),
        )
        reserved = [
            target
            for robot_id, target in request.existing_targets_by_robot.items()
            if int(robot_id) not in requested and target is not None
        ]

        labels = [-1] * len(grid.data)
        distances = [-1] * len(grid.data)
        queue: deque[tuple[int, int, int, int]] = deque()
        for robot_id, cell in sorted(seed_by_robot.items()):
            index = grid.index(cell)
            # Deterministic tie handling when two robots quantize to one cell.
            if labels[index] >= 0 and labels[index] < robot_id:
                continue
            labels[index] = robot_id
            distances[index] = 0
            queue.append((0, robot_id, cell[0], cell[1]))

        selected: dict[int, tuple[Point2D, int, str]] = {}
        wavefront_owners_needed = {
            robot_id
            for robot_id in requested
            if robot_id in robots_by_id and robot_id in seed_by_robot
        }
        claimed_cells_by_robot: dict[int, int] = {robot_id: 0 for robot_id in seed_by_robot}
        frontier_count = 0

        # Every transition has unit cost, so a FIFO queue is the exact
        # Dijkstra specialization for this graph and avoids O(log n) heap
        # overhead on the 128k-cell experiment grid.
        while queue:
            steps, owner, row, col = queue.popleft()
            cell = (row, col)
            index = grid.index(cell)
            if labels[index] != owner or distances[index] != steps:
                continue
            claimed_cells_by_robot[owner] = claimed_cells_by_robot.get(owner, 0) + 1

            if _is_frontier(grid, cell):
                frontier_count += 1
                target = grid.cell_to_world(cell)
                robot = robots_by_id.get(owner)
                if (
                    owner in requested
                    and owner not in selected
                    and robot is not None
                    and self._target_allowed(
                        target,
                        robot,
                        request,
                        reserved,
                        min_travel=min_travel,
                        reservation_radius=reservation_radius,
                    )
                ):
                    selected[owner] = (
                        target,
                        steps,
                        "first frontier claimed by synchronized Nav2D wavefront",
                    )
                    reserved.append(target)

                    # The reference implementation returns as soon as the
                    # current robot's wave reaches a frontier.  This central
                    # port can stop once every requested owner's first
                    # frontier is known; finishing the unused remainder of
                    # the Voronoi partition would not change an assignment.
                    if wavefront_owners_needed.issubset(selected):
                        queue.clear()
                        break

            for neighbor in _neighbors8(cell):
                if not grid.valid(neighbor) or grid.state(neighbor) != _FREE:
                    continue
                neighbor_index = grid.index(neighbor)
                if labels[neighbor_index] >= 0:
                    continue
                candidate_steps = steps + 1
                labels[neighbor_index] = owner
                distances[neighbor_index] = candidate_steps
                queue.append((candidate_steps, owner, neighbor[0], neighbor[1]))

        # Nav2D's second queue crosses another robot's wave region when the
        # robot's own region has no frontier.  A per-robot BFS over the same
        # known-free grid is the equivalent once the complete Voronoi labels
        # above are available.
        for robot_id in requested:
            if robot_id in selected or robot_id not in robots_by_id or robot_id not in seed_by_robot:
                continue
            fallback = self._cross_region_frontier(
                grid,
                seed_by_robot[robot_id],
                robots_by_id[robot_id],
                request,
                reserved,
                min_travel=min_travel,
                reservation_radius=reservation_radius,
            )
            if fallback is not None:
                target, steps = fallback
                selected[robot_id] = (
                    target,
                    steps,
                    "own wave region had no usable frontier; crossed another Nav2D region",
                )
                reserved.append(target)

        assignments: list[CoordinationAssignment] = []
        commands: list[RobotCommand] = []
        target_by_id = {
            int(robot.robot_id): request.existing_targets_by_robot.get(
                int(robot.robot_id), robot.current_target
            )
            for robot in request.robot_states
        }
        reason_by_id = {
            int(robot.robot_id): "kept existing target"
            for robot in request.robot_states
        }

        for robot_id in requested:
            robot = robots_by_id.get(robot_id)
            choice = selected.get(robot_id)
            if robot is None:
                reason = "robot id not present in request"
                status = "FAILED"
                target = None
                proposal = None
            elif choice is None:
                reason = "no reachable unreserved frontier in known-free map"
                status = "HOLD"
                target = None
                proposal = None
            else:
                target, steps, detail = choice
                distance = steps * grid.resolution
                reason = f"{detail}; wavefront_distance={distance:.2f} m"
                status = "ASSIGNED"
                proposal = ExplorationCandidate(
                    target=target,
                    source="nav2d_multi_wavefront",
                    travel_cost=distance,
                    information_gain=1.0,
                    metadata={"wavefront_steps": steps, "owner_robot_id": robot_id},
                )

            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id,
                    target=target,
                    status=status,
                    reason=reason,
                    proposal=proposal,
                )
            )
            commands.append(
                RobotCommand(robot_id=robot_id, status=status, target=target, reason=reason)
            )
            target_by_id[robot_id] = target
            reason_by_id[robot_id] = reason

        ordered_ids = [int(robot.robot_id) for robot in request.robot_states]
        return CoordinationResult(
            targets=tuple(target_by_id.get(robot_id) for robot_id in ordered_ids),
            reasons=tuple(reason_by_id.get(robot_id, "not requested") for robot_id in ordered_ids),
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            commands=tuple(commands),
            debug={
                "plugin": self.metadata.name,
                "frontier_cells_seen_during_wave": frontier_count,
                "known_free_cells": sum(1 for state in grid.data if state == _FREE),
                "claimed_cells_by_robot": claimed_cells_by_robot,
                "selected_by_robot": {
                    robot_id: {"target": choice[0], "wavefront_steps": choice[1], "reason": choice[2]}
                    for robot_id, choice in selected.items()
                },
            },
        )

    @staticmethod
    def _target_allowed(
        target: Point2D,
        robot: RobotCoordinationState,
        request: CoordinationRequest,
        reserved: list[Point2D],
        *,
        min_travel: float,
        reservation_radius: float,
    ) -> bool:
        if _distance(target, robot.xy) < min_travel:
            return False
        if any(_distance(target, item) <= reservation_radius for item in reserved):
            return False
        blocked = request.blocked_targets_by_robot.get(int(robot.robot_id), ())
        return not any(_distance(target, item) <= reservation_radius for item in blocked)

    def _cross_region_frontier(
        self,
        grid: _Grid,
        start: tuple[int, int],
        robot: RobotCoordinationState,
        request: CoordinationRequest,
        reserved: list[Point2D],
        *,
        min_travel: float,
        reservation_radius: float,
    ) -> tuple[Point2D, int] | None:
        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            cell, steps = queue.popleft()
            if _is_frontier(grid, cell):
                target = grid.cell_to_world(cell)
                if self._target_allowed(
                    target,
                    robot,
                    request,
                    reserved,
                    min_travel=min_travel,
                    reservation_radius=reservation_radius,
                ):
                    return target, steps
            for neighbor in _neighbors8(cell):
                if neighbor in visited or not grid.valid(neighbor) or grid.state(neighbor) != _FREE:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, steps + 1))
        return None

    def _all_hold(
        self,
        request: CoordinationRequest,
        requested: tuple[int, ...],
        reason: str,
    ) -> CoordinationResult:
        assignments = tuple(
            CoordinationAssignment(robot_id=robot_id, target=None, status="HOLD", reason=reason)
            for robot_id in requested
        )
        commands = tuple(
            RobotCommand(robot_id=robot_id, target=None, status="HOLD", reason=reason)
            for robot_id in requested
        )
        requested_set = set(requested)
        return CoordinationResult(
            targets=tuple(
                None if int(robot.robot_id) in requested_set else robot.current_target
                for robot in request.robot_states
            ),
            reasons=tuple(
                reason if int(robot.robot_id) in requested_set else "kept existing target"
                for robot in request.robot_states
            ),
            strategy=self.metadata.name,
            assignments=assignments,
            commands=commands,
            debug={"plugin": self.metadata.name, "hold_reason": reason},
        )


def create_plugin() -> CoordinationPlugin:
    return Nav2DMultiWavefrontPlugin()
