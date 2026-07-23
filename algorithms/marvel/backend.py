"""Simulator-to-tensor bridge for the official MARVEL PolicyNet.

The observation construction follows the authors' implementation in
``utils/agent.py``, ``utils/node_manager.py`` and ``utils/utils.py``:

* paper-scale 4 m viewpoint lattice or a dimensionless scaled equivalent;
* free cells with between 2 and 7 unknown neighbours as frontiers;
* six node inputs (relative x/y, utility, guidepost, occupancy and heading);
* 36-bin frontier/visited-heading vectors and three heading candidates.

Only the environment adapter is implemented here.  Waypoint/heading selection
is always made from the logits produced by the cited PolicyNet checkpoint; the
bridge deliberately has no heuristic task-assignment fallback.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.observations import (
    Point2D,
    RobotCoordinationState,
    WorldSnapshot,
)


MARVEL_SOURCE = "https://arxiv.org/abs/2502.20217"
MARVEL_COORDINATOR = "MARVEL CTDE graph-attention policy"
MARVEL_SCALED_COORDINATOR = (
    "MARVEL CTDE graph-attention policy (scaled environment)"
)
PAPER_SPATIAL_MODE = "paper"
SCALED_SPATIAL_MODE = "scaled"
NODE_RESOLUTION = 4.0
UPDATING_MAP_SIZE = 60.0
FRONTIER_CELL_SIZE = 0.8
NUM_NODE_NEIGHBORS = 5
NUM_ANGLES_BIN = 36
NUM_HEADING_CANDIDATES = 3
DEFAULT_FOV_DEGREES = 120.0
PAPER_SENSOR_RANGE = 10.0

GridCell = tuple[int, int]


@dataclass(frozen=True)
class MarvelSpatialConfiguration:
    """Physical constants used to construct a dimensionless MARVEL graph.

    The authors trained with a 10 m sensor, 4 m nodes, 0.8 m frontier
    downsampling and a 60 m local map.  The scaled adapter multiplies those
    four lengths by the same factor, preserving every ratio seen by PolicyNet.
    """

    mode: str
    scale_factor: float
    reference_sensor_range_m: float
    node_resolution_m: float
    updating_map_size_m: float
    frontier_cell_size_m: float

    @classmethod
    def paper(cls) -> "MarvelSpatialConfiguration":
        return cls(
            mode=PAPER_SPATIAL_MODE,
            scale_factor=1.0,
            reference_sensor_range_m=PAPER_SENSOR_RANGE,
            node_resolution_m=NODE_RESOLUTION,
            updating_map_size_m=UPDATING_MAP_SIZE,
            frontier_cell_size_m=FRONTIER_CELL_SIZE,
        )

    @classmethod
    def scaled(
        cls,
        *,
        sensor_range_m: float,
        grid_resolution_m: float,
    ) -> "MarvelSpatialConfiguration":
        sensor_range = float(sensor_range_m)
        grid_resolution = max(float(grid_resolution_m), 1e-6)
        if sensor_range <= 0.0:
            raise ValueError("MARVEL scaled sensor range must be positive")
        scale = sensor_range / PAPER_SENSOR_RANGE
        node_resolution = NODE_RESOLUTION * scale
        if node_resolution < 2.0 * grid_resolution:
            minimum_range = (
                2.0 * grid_resolution * PAPER_SENSOR_RANGE / NODE_RESOLUTION
            )
            raise ValueError(
                "MARVEL scaled range is too small for the belief-grid "
                f"resolution; use at least {minimum_range:.2f} m for a "
                f"{grid_resolution:.2f} m grid"
            )
        return cls(
            mode=SCALED_SPATIAL_MODE,
            scale_factor=scale,
            reference_sensor_range_m=sensor_range,
            node_resolution_m=node_resolution,
            updating_map_size_m=UPDATING_MAP_SIZE * scale,
            # A voxel cannot be represented below one host belief-map cell.
            frontier_cell_size_m=max(FRONTIER_CELL_SIZE * scale, grid_resolution),
        )


@dataclass(frozen=True)
class _BeliefGrid:
    bounds: tuple[float, float, float, float]
    resolution: float
    width: int
    height: int
    free: frozenset[GridCell]
    occupied: frozenset[GridCell]

    @classmethod
    def from_world(
        cls,
        world: WorldSnapshot,
        robot_states: Sequence[RobotCoordinationState],
    ) -> "_BeliefGrid":
        if world.bounds is None:
            raise ValueError("MARVEL requires finite world bounds")
        x_min, x_max, y_min, y_max = (float(value) for value in world.bounds)
        resolution = max(float(world.resolution), 1e-6)
        width = max(1, int(math.ceil((x_max - x_min) / resolution)))
        height = max(1, int(math.ceil((y_max - y_min) / resolution)))

        def cell_for(point: Point2D) -> GridCell | None:
            x, y = float(point[0]), float(point[1])
            if not (x_min <= x < x_max and y_min <= y < y_max):
                return None
            col = int(math.floor((x - x_min) / resolution))
            row = int(math.floor((y - y_min) / resolution))
            if 0 <= row < height and 0 <= col < width:
                return row, col
            return None

        occupied = {
            cell
            for point in world.mapped_obstacle_points
            if (cell := cell_for(point)) is not None
        }
        free = {
            cell
            for point in world.explored_points
            if (cell := cell_for(point)) is not None and cell not in occupied
        }
        # A robot's current pose is necessarily observed free space.  This
        # closes the one-frame race between motion and the immutable map
        # snapshot without exposing any unobserved cell to the policy.
        for robot in robot_states:
            cell = cell_for(robot.xy)
            if cell is not None and cell not in occupied:
                free.add(cell)

        return cls(
            bounds=(x_min, x_max, y_min, y_max),
            resolution=resolution,
            width=width,
            height=height,
            free=frozenset(free),
            occupied=frozenset(occupied),
        )

    def cell_for(self, point: Point2D) -> GridCell | None:
        x_min, x_max, y_min, y_max = self.bounds
        x, y = float(point[0]), float(point[1])
        if not (x_min <= x < x_max and y_min <= y < y_max):
            return None
        col = int(math.floor((x - x_min) / self.resolution))
        row = int(math.floor((y - y_min) / self.resolution))
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def point_for(self, cell: GridCell) -> Point2D:
        row, col = cell
        x_min, _x_max, y_min, _y_max = self.bounds
        return (
            x_min + (col + 0.5) * self.resolution,
            y_min + (row + 0.5) * self.resolution,
        )

    def connected_free(self, start: Point2D) -> frozenset[GridCell]:
        start_cell = self.cell_for(start)
        if start_cell is None or start_cell not in self.free:
            return frozenset()
        reached = {start_cell}
        queue = deque([start_cell])
        while queue:
            row, col = queue.popleft()
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    candidate = row + d_row, col + d_col
                    if candidate in self.free and candidate not in reached:
                        reached.add(candidate)
                        queue.append(candidate)
        return frozenset(reached)

    def frontiers(
        self,
        connected_free: frozenset[GridCell],
        *,
        frontier_cell_size: float,
    ) -> tuple[Point2D, ...]:
        frontier_points: list[Point2D] = []
        for row, col in connected_free:
            unknown_count = 0
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    neighbor = row + d_row, col + d_col
                    in_bounds = (
                        0 <= neighbor[0] < self.height
                        and 0 <= neighbor[1] < self.width
                    )
                    if in_bounds and neighbor not in self.free and neighbor not in self.occupied:
                        unknown_count += 1
            # Exact frontier condition from the authors' get_frontier_in_map.
            if 1 < unknown_count < 8:
                frontier_points.append(self.point_for((row, col)))
        return _downsample_frontiers(
            frontier_points,
            frontier_cell_size=frontier_cell_size,
        )

    def line_is_known_free(self, start: Point2D, end: Point2D) -> bool:
        start_cell = self.cell_for(start)
        end_cell = self.cell_for(end)
        if start_cell is None or end_cell is None:
            return False
        for cell in _bresenham_cells(start_cell, end_cell):
            if cell not in self.free:
                return False
        return True


@dataclass(frozen=True)
class _MarvelGraph:
    nodes: tuple[Point2D, ...]
    adjacency: tuple[tuple[int, ...], ...]
    current_index: int
    neighbor_indices: tuple[int, ...]
    utilities: tuple[float, ...]
    guidepost: tuple[float, ...]
    occupancy: tuple[float, ...]
    highest_utility_angles: tuple[float, ...]
    frontier_distribution: tuple[tuple[float, ...], ...]
    heading_visited: tuple[tuple[float, ...], ...]
    heading_candidate_features: tuple[tuple[tuple[float, ...], ...], ...]
    heading_candidate_indices: tuple[tuple[int, ...], ...]
    frontier_count: int
    spatial: MarvelSpatialConfiguration


class MarvelInferenceBackend:
    """Build MARVEL observations and decode PolicyNet waypoint-heading actions."""

    def __init__(
        self,
        *,
        strategy_name: str = MARVEL_COORDINATOR,
        spatial_mode: str = PAPER_SPATIAL_MODE,
    ) -> None:
        if spatial_mode not in {PAPER_SPATIAL_MODE, SCALED_SPATIAL_MODE}:
            raise ValueError(f"unsupported MARVEL spatial mode: {spatial_mode!r}")
        self.strategy_name = str(strategy_name)
        self.spatial_mode = str(spatial_mode)
        self._visited_headings: dict[
            tuple[int, str, tuple[int, int]], set[int]
        ] = {}

    def assign(self, request: CoordinationRequest, policy) -> CoordinationResult:
        world = request.world
        if world is None:
            return self._hold_all(request, "MARVEL requires a belief-map snapshot")

        architecture = str(
            request.shared.get(
                "mapping_architecture",
                world.metadata.get("mapping_architecture", "centralized"),
            )
        )
        if architecture != "centralized":
            return self._hold_all(
                request,
                "MARVEL requires the paper's shared centralized belief map "
                "(CTDE with decentralized policy execution)",
            )

        incompatible = [
            robot
            for robot in request.robot_states
            if robot.is_active
            and "Camera" not in str(robot.vision_model)
        ]
        if incompatible:
            details = ", ".join(
                f"R{robot.robot_id + 1}={robot.vision_model}/{robot.sensor_range:.2f}m"
                for robot in incompatible
            )
            return self._hold_all(
                request,
                "MARVEL requires a directional Camera / FoV observation; "
                f"incompatible sensing model: {details}. Sensor range and "
                "FoV angle remain user-adjustable experimental parameters.",
            )

        try:
            belief = _BeliefGrid.from_world(world, request.robot_states)
        except (TypeError, ValueError) as exc:
            return self._hold_all(request, f"MARVEL belief-map conversion failed: {exc}")
        try:
            spatial = self._spatial_configuration(request, belief)
        except ValueError as exc:
            return self._hold_all(request, str(exc))

        requested_ids = set(int(value) for value in request.robots_to_assign)
        if not requested_ids:
            requested_ids = {
                robot.robot_id for robot in request.robot_states if robot.is_active
            }
        robots_by_id = {robot.robot_id: robot for robot in request.robot_states}

        targets_by_id = {
            robot.robot_id: robot.current_target for robot in request.robot_states
        }
        reasons_by_id: dict[int, str] = {}
        assignments: list[CoordinationAssignment] = []
        commands: list[RobotCommand] = []
        debug_by_robot: dict[str, object] = {}

        reservation_radius = max(
            min(
                float(request.parameters.get("target_exclusion_radius", 1.5)),
                0.5 * spatial.node_resolution_m,
            ),
            belief.resolution,
        )
        requested_min_travel = max(
            float(request.parameters.get("min_frontier_travel_distance", 0.0)),
            0.0,
        )
        min_travel = max(
            2.0 * float(request.parameters.get("goal_tolerance", 0.25)),
            min(
                requested_min_travel,
                0.5 * spatial.node_resolution_m,
            ),
        )
        reserved = [
            target
            for robot_id, target in request.existing_targets_by_robot.items()
            if robot_id not in requested_ids and target is not None
        ]

        for robot_id in sorted(requested_ids):
            robot = robots_by_id.get(robot_id)
            if robot is None or not robot.is_active:
                continue
            try:
                graph = self._build_graph(request, belief, robot, spatial)
            except ValueError as exc:
                reason = str(exc)
                self._append_hold(
                    robot, reason, targets_by_id, reasons_by_id, assignments, commands
                )
                debug_by_robot[str(robot_id)] = {"status": "HOLD", "reason": reason}
                continue

            tensors = self._observation_tensors(graph, robot)
            try:
                import torch

                with torch.no_grad():
                    logp = policy(*tensors)
                ranked_actions = torch.argsort(logp[0], descending=True).tolist()
            except Exception as exc:
                reason = f"MARVEL PolicyNet inference failed: {exc}"
                self._append_hold(
                    robot, reason, targets_by_id, reasons_by_id, assignments, commands
                )
                debug_by_robot[str(robot_id)] = {"status": "HOLD", "reason": reason}
                continue

            selected: tuple[Point2D, int, int, float] | None = None
            blocked = request.blocked_targets_by_robot.get(robot_id, ())
            for action in ranked_actions:
                edge_position, heading_position = divmod(
                    int(action), NUM_HEADING_CANDIDATES
                )
                if edge_position >= len(graph.neighbor_indices):
                    continue
                node_index = graph.neighbor_indices[edge_position]
                if node_index == graph.current_index:
                    continue
                target = graph.nodes[node_index]
                if math.dist(robot.xy, target) < min_travel:
                    continue
                if any(math.dist(target, item) < reservation_radius for item in reserved):
                    continue
                if any(math.dist(target, item) < reservation_radius for item in blocked):
                    continue
                heading_bin = graph.heading_candidate_indices[edge_position][
                    heading_position
                ]
                selected = (
                    target,
                    node_index,
                    heading_bin,
                    float(logp[0, int(action)].item()),
                )
                break

            if selected is None:
                reason = (
                    "MARVEL PolicyNet returned no unreserved reachable "
                    "waypoint-heading action"
                )
                self._append_hold(
                    robot, reason, targets_by_id, reasons_by_id, assignments, commands
                )
                debug_by_robot[str(robot_id)] = {
                    "status": "HOLD",
                    "reason": reason,
                    "nodes": len(graph.nodes),
                    "frontiers": graph.frontier_count,
                    "current_node": graph.nodes[graph.current_index],
                    "neighbor_nodes": tuple(
                        graph.nodes[index]
                        for index in graph.neighbor_indices
                    ),
                    "reserved_targets": tuple(reserved),
                    "blocked_targets": tuple(blocked),
                    "reservation_radius_m": reservation_radius,
                    "min_travel_m": min_travel,
                }
                continue

            target, node_index, heading_bin, action_logp = selected
            heading_rad = math.radians(heading_bin * (360.0 / NUM_ANGLES_BIN))
            reason = "selected by the MARVEL PolicyNet waypoint-heading policy"
            targets_by_id[robot_id] = target
            reasons_by_id[robot_id] = reason
            reserved.append(target)
            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=target,
                    reason=reason,
                )
            )
            commands.append(
                RobotCommand(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=target,
                    heading_rad=heading_rad,
                    reason=reason,
                    metadata={
                        "policy": "MARVEL PolicyNet",
                        "node_index": node_index,
                        "heading_bin": heading_bin,
                        "action_log_probability": action_logp,
                    },
                )
            )
            debug_by_robot[str(robot_id)] = {
                "status": "ASSIGNED",
                "nodes": len(graph.nodes),
                "frontiers": graph.frontier_count,
                "current_node": graph.nodes[graph.current_index],
                "selected_node": target,
                "heading_bin": heading_bin,
                "action_log_probability": action_logp,
            }

        targets = tuple(
            targets_by_id.get(robot.robot_id, robot.current_target)
            for robot in request.robot_states
        )
        reasons = tuple(
            reasons_by_id.get(
                robot.robot_id,
                "kept existing target" if robot.current_target is not None else "not requested",
            )
            for robot in request.robot_states
        )
        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.strategy_name,
            assignments=tuple(assignments),
            commands=tuple(commands),
            debug={
                "ready": True,
                "paper_source": MARVEL_SOURCE,
                "mapping_architecture": architecture,
                "observation": {
                    "spatial_mode": spatial.mode,
                    "scale_factor": spatial.scale_factor,
                    "reference_sensor_range_m": (
                        spatial.reference_sensor_range_m
                    ),
                    "node_resolution_m": spatial.node_resolution_m,
                    "updating_map_size_m": spatial.updating_map_size_m,
                    "frontier_cell_size_m": spatial.frontier_cell_size_m,
                    "target_reservation_radius_m": reservation_radius,
                    "node_features": 6,
                    "heading_bins": NUM_ANGLES_BIN,
                    "heading_candidates": NUM_HEADING_CANDIDATES,
                    "neighbor_stencil": NUM_NODE_NEIGHBORS,
                    "camera_fov_degrees": float(
                        request.parameters.get(
                            "marvel_fov_degrees",
                            DEFAULT_FOV_DEGREES,
                        )
                    ),
                    "sensor_ranges_m": tuple(
                        float(robot.sensor_range)
                        for robot in request.robot_states
                    ),
                    "paper_defaults": {
                        "sensor_range_m": PAPER_SENSOR_RANGE,
                        "camera_fov_degrees": DEFAULT_FOV_DEGREES,
                    },
                },
                "per_robot": debug_by_robot,
            },
        )

    def _spatial_configuration(
        self,
        request: CoordinationRequest,
        belief: _BeliefGrid,
    ) -> MarvelSpatialConfiguration:
        if self.spatial_mode == PAPER_SPATIAL_MODE:
            return MarvelSpatialConfiguration.paper()

        ranges = tuple(
            float(robot.sensor_range)
            for robot in request.robot_states
            if robot.is_active
        )
        if not ranges:
            raise ValueError("MARVEL scaled mode requires at least one active robot")
        if max(ranges) - min(ranges) > max(belief.resolution, 1e-6):
            raise ValueError(
                "MARVEL scaled mode requires one shared sensor range because "
                "the team uses one shared viewpoint graph"
            )
        reference_range = sum(ranges) / len(ranges)
        return MarvelSpatialConfiguration.scaled(
            sensor_range_m=reference_range,
            grid_resolution_m=belief.resolution,
        )

    def _build_graph(
        self,
        request: CoordinationRequest,
        belief: _BeliefGrid,
        robot: RobotCoordinationState,
        spatial: MarvelSpatialConfiguration,
    ) -> _MarvelGraph:
        connected = belief.connected_free(robot.xy)
        if not connected:
            raise ValueError("MARVEL has no known-free component at the robot pose")
        frontiers = belief.frontiers(
            connected,
            frontier_cell_size=spatial.frontier_cell_size_m,
        )
        if not frontiers:
            raise ValueError("MARVEL found no frontiers in the shared belief map")

        nodes = _lattice_nodes(belief, connected, robot.xy, spatial)
        if not nodes:
            raise ValueError(
                "MARVEL has no "
                f"{spatial.node_resolution_m:.2f} m viewpoint node in the "
                "robot's known-free component"
            )
        current_index = min(
            range(len(nodes)), key=lambda index: math.dist(robot.xy, nodes[index])
        )
        adjacency = _graph_adjacency(
            nodes,
            belief,
            node_resolution=spatial.node_resolution_m,
        )
        neighbor_indices = tuple(
            index for index, value in enumerate(adjacency[current_index]) if value == 0
        )
        if len(neighbor_indices) <= 1:
            raise ValueError(
                "MARVEL current viewpoint has no reachable "
                f"{spatial.node_resolution_m:.2f} m graph neighbour; "
                "increase the visible starting area or use the scaled adapter"
            )

        fov = float(request.parameters.get("marvel_fov_degrees", DEFAULT_FOV_DEGREES))
        utility_range = 0.9 * float(robot.sensor_range)
        distributions: list[tuple[float, ...]] = []
        utilities: list[float] = []
        best_angles: list[float] = []
        raw_distributions: list[tuple[float, ...]] = []
        for node in nodes:
            distribution = [0.0] * NUM_ANGLES_BIN
            for frontier in frontiers:
                if math.dist(node, frontier) >= utility_range:
                    continue
                if not belief.line_is_known_free(node, frontier):
                    continue
                angle = math.degrees(
                    math.atan2(frontier[1] - node[1], frontier[0] - node[0])
                ) % 360.0
                distribution[int(angle / 360.0 * NUM_ANGLES_BIN) % NUM_ANGLES_BIN] += 1.0
            utility = float(sum(distribution))
            if utility <= 1.0:
                distribution = [0.0] * NUM_ANGLES_BIN
                utility = 0.0
                best_angle = -360.0
            else:
                best_angle = _best_heading_bins(distribution, fov, 1)[0] * (
                    360.0 / NUM_ANGLES_BIN
                )
            raw_distributions.append(tuple(distribution))
            utilities.append(utility)
            best_angles.append(best_angle)

        guidepost, path = _guidepost_to_nearest_utility(
            nodes, adjacency, current_index, utilities
        )
        occupancy = [0.0] * len(nodes)
        for other in request.robot_states:
            if not other.is_active:
                continue
            index = min(
                range(len(nodes)), key=lambda value: math.dist(other.xy, nodes[value])
            )
            occupancy[index] = -1.0 if other.robot_id == robot.robot_id else 1.0

        heading_visited = [[0.0] * NUM_ANGLES_BIN for _ in nodes]
        current_key = _node_key(
            nodes[current_index],
            node_resolution=spatial.node_resolution_m,
        )
        current_heading_bin = int(
            (math.degrees(float(robot.theta)) % 360.0)
            / 360.0
            * NUM_ANGLES_BIN
        ) % NUM_ANGLES_BIN
        self._visited_headings.setdefault(
            (robot.robot_id, spatial.mode, current_key), set()
        ).add(current_heading_bin)
        for index, node in enumerate(nodes):
            for heading_bin in self._visited_headings.get(
                (
                    robot.robot_id,
                    spatial.mode,
                    _node_key(
                        node,
                        node_resolution=spatial.node_resolution_m,
                    ),
                ),
                (),
            ):
                heading_visited[index][heading_bin] = 1.0

        heading_features: list[tuple[tuple[float, ...], ...]] = []
        heading_indices: list[tuple[int, ...]] = []
        for node_index in neighbor_indices:
            distribution = raw_distributions[node_index]
            if utilities[node_index] > 0.0:
                candidates = _best_heading_bins(
                    distribution, fov, NUM_HEADING_CANDIDATES
                )
            else:
                guide_angle = _guide_heading(
                    nodes, adjacency, path, node_index, current_index
                )
                center = int(guide_angle / 360.0 * NUM_ANGLES_BIN) % NUM_ANGLES_BIN
                candidates = tuple(
                    (center + offset) % NUM_ANGLES_BIN for offset in (-1, 0, 1)
                )
            heading_indices.append(tuple(int(value) for value in candidates))
            heading_features.append(
                tuple(_heading_window(value, fov) for value in candidates)
            )

        frontier_normalizer = max(
            (
                2.0
                * float(robot.sensor_range)
                * math.pi
                // spatial.frontier_cell_size_m
            )
            / NUM_ANGLES_BIN,
            1.0,
        )
        for distribution in raw_distributions:
            distributions.append(
                tuple(value / frontier_normalizer for value in distribution)
            )

        return _MarvelGraph(
            nodes=nodes,
            adjacency=adjacency,
            current_index=current_index,
            neighbor_indices=neighbor_indices,
            utilities=tuple(utilities),
            guidepost=guidepost,
            occupancy=tuple(occupancy),
            highest_utility_angles=tuple(best_angles),
            frontier_distribution=tuple(distributions),
            heading_visited=tuple(tuple(row) for row in heading_visited),
            heading_candidate_features=tuple(heading_features),
            heading_candidate_indices=tuple(heading_indices),
            frontier_count=len(frontiers),
            spatial=spatial,
        )

    def _observation_tensors(
        self,
        graph: _MarvelGraph,
        robot: RobotCoordinationState,
    ):
        import torch

        current = graph.nodes[graph.current_index]
        utility_normalizer = max(
            (
                2.0
                * float(robot.sensor_range)
                * math.pi
                // graph.spatial.frontier_cell_size_m
            ),
            1.0,
        )
        node_inputs = []
        for index, node in enumerate(graph.nodes):
            node_inputs.append(
                (
                    (
                        (node[0] - current[0])
                        / graph.spatial.updating_map_size_m
                        / 2.0
                    ),
                    (
                        (node[1] - current[1])
                        / graph.spatial.updating_map_size_m
                        / 2.0
                    ),
                    graph.utilities[index] / utility_normalizer,
                    graph.guidepost[index],
                    graph.occupancy[index],
                    graph.highest_utility_angles[index] / 360.0,
                )
            )

        node_count = len(graph.nodes)
        current_in_edge = graph.neighbor_indices.index(graph.current_index)
        edge_padding_mask = torch.zeros(
            (1, 1, len(graph.neighbor_indices)), dtype=torch.int16
        )
        edge_padding_mask[0, 0, current_in_edge] = 1
        return (
            torch.tensor(node_inputs, dtype=torch.float32).unsqueeze(0),
            torch.zeros((1, 1, node_count), dtype=torch.int16),
            torch.tensor(graph.adjacency, dtype=torch.int16).unsqueeze(0),
            torch.tensor([graph.current_index], dtype=torch.long).reshape(1, 1, 1),
            torch.tensor(graph.neighbor_indices, dtype=torch.long)
            .reshape(1, -1, 1),
            edge_padding_mask,
            torch.tensor(graph.frontier_distribution, dtype=torch.float32)
            .unsqueeze(0),
            torch.tensor(graph.heading_visited, dtype=torch.float32).unsqueeze(0),
            torch.tensor(graph.heading_candidate_features, dtype=torch.float32)
            .unsqueeze(0),
        )

    @staticmethod
    def _append_hold(
        robot: RobotCoordinationState,
        reason: str,
        targets_by_id: dict[int, Point2D | None],
        reasons_by_id: dict[int, str],
        assignments: list[CoordinationAssignment],
        commands: list[RobotCommand],
    ) -> None:
        targets_by_id[robot.robot_id] = None
        reasons_by_id[robot.robot_id] = reason
        assignments.append(
            CoordinationAssignment(robot.robot_id, "HOLD", None, reason)
        )
        commands.append(
            RobotCommand(robot_id=robot.robot_id, status="HOLD", reason=reason)
        )

    def _hold_all(
        self, request: CoordinationRequest, reason: str
    ) -> CoordinationResult:
        requested = set(int(value) for value in request.robots_to_assign)
        if not requested:
            requested = {
                robot.robot_id for robot in request.robot_states if robot.is_active
            }
        assignments = tuple(
            CoordinationAssignment(robot.robot_id, "HOLD", None, reason)
            for robot in request.robot_states
            if robot.robot_id in requested
        )
        commands = tuple(
            RobotCommand(robot_id=item.robot_id, status="HOLD", reason=reason)
            for item in assignments
        )
        return CoordinationResult(
            targets=tuple(
                None if robot.robot_id in requested else robot.current_target
                for robot in request.robot_states
            ),
            reasons=tuple(
                reason if robot.robot_id in requested else "not requested"
                for robot in request.robot_states
            ),
            strategy=self.strategy_name,
            assignments=assignments,
            commands=commands,
            debug={
                "ready": False,
                "paper_source": MARVEL_SOURCE,
                "reason": reason,
            },
        )


def _lattice_nodes(
    belief: _BeliefGrid,
    connected: frozenset[GridCell],
    robot_xy: Point2D,
    spatial: MarvelSpatialConfiguration,
) -> tuple[Point2D, ...]:
    x_min, x_max, y_min, y_max = belief.bounds
    half = spatial.updating_map_size_m / 2.0
    local_x_min = max(x_min, robot_xy[0] - half)
    local_x_max = min(x_max, robot_xy[0] + half)
    local_y_min = max(y_min, robot_xy[1] - half)
    local_y_max = min(y_max, robot_xy[1] + half)
    node_resolution = spatial.node_resolution_m
    first_x = math.ceil(local_x_min / node_resolution) * node_resolution
    first_y = math.ceil(local_y_min / node_resolution) * node_resolution
    nodes: list[Point2D] = []
    x = first_x
    while x < local_x_max:
        y = first_y
        while y < local_y_max:
            point = (round(float(x), 6), round(float(y), 6))
            cell = belief.cell_for(point)
            if cell is not None and cell in connected:
                nodes.append(point)
            y += node_resolution
        x += node_resolution
    return tuple(sorted(nodes))


def _graph_adjacency(
    nodes: Sequence[Point2D],
    belief: _BeliefGrid,
    *,
    node_resolution: float,
) -> tuple[tuple[int, ...], ...]:
    node_count = len(nodes)
    adjacency = [[1] * node_count for _ in range(node_count)]
    max_offset = (NUM_NODE_NEIGHBORS // 2) * node_resolution + 1e-6
    for left in range(node_count):
        adjacency[left][left] = 0
        for right in range(left + 1, node_count):
            dx = abs(nodes[left][0] - nodes[right][0])
            dy = abs(nodes[left][1] - nodes[right][1])
            if dx > max_offset or dy > max_offset:
                continue
            if belief.line_is_known_free(nodes[left], nodes[right]):
                adjacency[left][right] = 0
                adjacency[right][left] = 0
    return tuple(tuple(row) for row in adjacency)


def _guidepost_to_nearest_utility(
    nodes: Sequence[Point2D],
    adjacency: Sequence[Sequence[int]],
    current_index: int,
    utilities: Sequence[float],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    distances = [math.inf] * len(nodes)
    previous: list[int | None] = [None] * len(nodes)
    distances[current_index] = 0.0
    queue: list[tuple[float, int]] = [(0.0, current_index)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, masked in enumerate(adjacency[node]):
            if masked or neighbor == node:
                continue
            candidate = distance + math.dist(nodes[node], nodes[neighbor])
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))

    utility_nodes = [
        index
        for index, utility in enumerate(utilities)
        if (
            index != current_index
            and utility > 0.0
            and math.isfinite(distances[index])
        )
    ]
    if not utility_nodes:
        guidepost = [0.0] * len(nodes)
        guidepost[current_index] = 1.0
        return tuple(guidepost), (current_index,)
    destination = min(utility_nodes, key=lambda index: distances[index])
    path = [destination]
    while path[-1] != current_index:
        parent = previous[path[-1]]
        if parent is None:
            break
        path.append(parent)
    path.reverse()
    guidepost = [0.0] * len(nodes)
    for index in path[1:]:
        guidepost[index] = 1.0
    return tuple(guidepost), tuple(path)


def _guide_heading(
    nodes: Sequence[Point2D],
    adjacency: Sequence[Sequence[int]],
    guide_path: Sequence[int],
    node_index: int,
    current_index: int,
) -> float:
    if node_index in guide_path:
        position = guide_path.index(node_index)
        if position + 1 < len(guide_path):
            target = nodes[guide_path[position + 1]]
            return math.degrees(
                math.atan2(
                    target[1] - nodes[node_index][1],
                    target[0] - nodes[node_index][0],
                )
            ) % 360.0
    neighbors = [
        index
        for index, masked in enumerate(adjacency[node_index])
        if not masked and index != node_index
    ]
    if neighbors:
        target = nodes[neighbors[0]]
        return math.degrees(
            math.atan2(
                target[1] - nodes[node_index][1],
                target[0] - nodes[node_index][0],
            )
        ) % 360.0
    target = nodes[current_index]
    return math.degrees(
        math.atan2(
            target[1] - nodes[node_index][1],
            target[0] - nodes[node_index][0],
        )
    ) % 360.0


def _best_heading_bins(
    distribution: Sequence[float],
    fov_degrees: float,
    count: int,
) -> tuple[int, ...]:
    half_bins = max(
        0,
        int((float(fov_degrees) / 360.0) * NUM_ANGLES_BIN / 2.0),
    )
    scores = []
    for center in range(NUM_ANGLES_BIN):
        score = sum(
            distribution[(center + offset) % NUM_ANGLES_BIN]
            for offset in range(-half_bins, half_bins + 1)
        )
        scores.append((float(score), center))
    scores.sort(key=lambda item: (-item[0], item[1]))
    selected = [center for _score, center in scores[:count]]
    while len(selected) < count:
        selected.append(selected[-1] if selected else 0)
    return tuple(selected)


def _heading_window(center: int, fov_degrees: float) -> tuple[float, ...]:
    half_bins = max(
        0,
        int((float(fov_degrees) / 360.0) * NUM_ANGLES_BIN / 2.0),
    )
    result = [0.0] * NUM_ANGLES_BIN
    for offset in range(-half_bins, half_bins + 1):
        result[(int(center) + offset) % NUM_ANGLES_BIN] = 1.0
    return tuple(result)


def _downsample_frontiers(
    points: Iterable[Point2D],
    *,
    frontier_cell_size: float,
) -> tuple[Point2D, ...]:
    by_voxel: dict[tuple[int, int], Point2D] = {}
    for point in points:
        key = (
            int(float(point[0]) / frontier_cell_size),
            int(float(point[1]) / frontier_cell_size),
        )
        anchor = (
            key[0] * frontier_cell_size,
            key[1] * frontier_cell_size,
        )
        current = by_voxel.get(key)
        if current is None or math.dist(point, anchor) < math.dist(current, anchor):
            by_voxel[key] = point
    return tuple(sorted(by_voxel.values()))


def _bresenham_cells(start: GridCell, end: GridCell) -> tuple[GridCell, ...]:
    y0, x0 = start
    y1, x1 = end
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    cells: list[GridCell] = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        twice_error = 2 * error
        if twice_error >= dy:
            error += dy
            x0 += step_x
        if twice_error <= dx:
            error += dx
            y0 += step_y
    return tuple(cells)


def _node_key(
    point: Point2D,
    *,
    node_resolution: float,
) -> tuple[int, int]:
    return (
        int(round(float(point[0]) / node_resolution)),
        int(round(float(point[1]) / node_resolution)),
    )
