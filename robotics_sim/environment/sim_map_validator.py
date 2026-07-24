"""
Pure geometric validator for the .sim smoke map corpus
(experiments/maps/smoke_v0/).

This module answers one question: is a .sim map file (plus the scenario
overlays -- robot starts and fire placements -- that go with it) safe and
well formed enough to serve as a smoke-test map for exploration and
coordination? It never touches the GUI, the physics engine, or a learning
pipeline; it only reads geometry already produced by the real .sim parser
(robotics_sim.simulation.config.load_sim_file/config_from_sim_payload) and
rasterizes it with the same OccupancyGrid helper the runtime planner uses
(robotics_sim.environment.occupancy_grid.OccupancyGrid.add_rectangular_
obstacles -- see robotics_sim/planning/planner_registry.py for the runtime
call site this mirrors).

Map vs. scenario
-----------------
A .sim file only carries *map* geometry (world bounds, grid_resolution,
obstacles) plus, because the current serializer format requires it, one
scenario's worth of robot start poses (robot/multi_robot blocks). It has no
field for fire placement or a geometry seed at all. This validator therefore
takes robot starts from the loaded SimulationConfig (the only source of
truth the parser gives us -- see normalized_robot_start_configs, which is
exactly how robotics_sim.simulation.engine spawns robots at runtime) but
takes fire scenarios as a separate argument supplied by the caller (in
practice, experiments/maps/smoke_v0/manifest.json), since fires are not
part of the .sim format today.

Nothing here does file I/O other than validate_sim_map_file's read, and
nothing here mutates its inputs.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from robotics_sim.environment.collision_checker import (
    distance_segment_to_rect,
    point_inside_expanded_rect,
)
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.occupancy_grid import FREE, OCCUPIED, OccupancyGrid
from robotics_sim.simulation.config import (
    SimulationConfig,
    WORLD_X_MAX,
    WORLD_X_MIN,
    WORLD_Y_MAX,
    WORLD_Y_MIN,
    load_sim_file,
    normalized_robot_start_configs,
)

# The runtime always builds its ground-truth/belief grids against these fixed
# module constants, never against a .sim file's "world" block (that block is
# written on save but never read back by config_from_sim_payload -- see
# engine.py's BeliefMap(bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN,
# WORLD_Y_MAX), ...) call sites). Validating against anything else would not
# match what the simulator actually does with the same file.
WORLD_BOUNDS: tuple[float, float, float, float] = (
    WORLD_X_MIN,
    WORLD_X_MAX,
    WORLD_Y_MIN,
    WORLD_Y_MAX,
)

# At least 98% of free cells must be reachable from the robot start cells.
MIN_CONNECTED_FREE_FRACTION = 0.98

Point2D = tuple[float, float]
RectObstacle = tuple[float, float, float, float]


@dataclass(frozen=True)
class MapValidationReport:
    """Immutable result of validating one .sim map (plus its scenario overlays)."""

    map_id: str
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    width: int
    height: int
    resolution: float
    free_cell_count: int
    occupied_cell_count: int
    obstacle_fraction: float
    connected_free_fraction: float
    connected_component_count: int
    start_positions_valid: bool
    fire_positions_valid: bool
    minimum_estimated_clearance_cells: float


def _required_corridor_width_m(safety_radius: float, resolution: float) -> float:
    """2 * safety_radius + 2 * grid_resolution.

    Matches the empirical finding in test_grid_resolution_corridor_
    diagnostic.py: obstacle rasterization (OccupancyGrid.set_obstacle_
    rect_world) occupies a whole cell as soon as the inflated obstacle rect
    touches it, so a real corridor has to clear the nominal 2*safety_radius
    continuous-space requirement by roughly one extra cell of width on each
    side before A* can actually route through it at the configured
    resolution. This is a smoke-corpus acceptance threshold, not a claim
    that anything narrower is geometrically impossible in continuous space.
    """
    return 2.0 * float(safety_radius) + 2.0 * float(resolution)


def _rasterize_ground_truth(
    obstacles: Sequence[RectObstacle],
    resolution: float,
) -> OccupancyGrid:
    """Same rasterization the runtime uses for planning grids: OccupancyGrid.
    from_bounds() + add_rectangular_obstacles(), no clearance padding (padding
    is applied separately per-query at runtime, e.g. by CollisionChecker /
    effective_planning_clearance -- see planner_registry.py's identical
    from_bounds()+add_rectangular_obstacles() pairing)."""
    grid = OccupancyGrid.from_bounds(
        x_min=WORLD_BOUNDS[0],
        x_max=WORLD_BOUNDS[1],
        y_min=WORLD_BOUNDS[2],
        y_max=WORLD_BOUNDS[3],
        resolution=resolution,
        initial_value=FREE,
        unknown_is_traversable=True,
    )
    grid.add_rectangular_obstacles(obstacles, padding=0.0)
    return grid


def _free_mask(grid: OccupancyGrid):
    return grid.data == FREE


def _occupied_mask(grid: OccupancyGrid):
    return grid.data == OCCUPIED


def _bfs_reachable(free_mask, seeds: Sequence[tuple[int, int]]):
    """4-connected BFS from `seeds` over `free_mask` (numpy bool[h, w]).

    4-connectivity matches the project's one existing whole-grid reachability
    walk, reachable_free_depths() in robotics_sim/planning/ryu_frontier_
    graph_bfs.py, so "reachable" means the same thing here as it does for the
    real exploration/frontier code that will eventually run on this map.
    """
    height, width = free_mask.shape
    visited = [[False] * width for _ in range(height)]
    queue: deque[tuple[int, int]] = deque()

    for row, col in seeds:
        if 0 <= row < height and 0 <= col < width and free_mask[row, col] and not visited[row][col]:
            visited[row][col] = True
            queue.append((row, col))

    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and free_mask[nr, nc] and not visited[nr][nc]:
                visited[nr][nc] = True
                queue.append((nr, nc))

    return visited


def _count_connected_components(free_mask) -> int:
    """4-connected component count over the whole free mask (not just the
    reachable-from-start subset) -- used only for the informational
    connected_component_count field."""
    height, width = free_mask.shape
    visited = [[False] * width for _ in range(height)]
    components = 0

    for row in range(height):
        for col in range(width):
            if not free_mask[row, col] or visited[row][col]:
                continue
            components += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row][col] = True
            while queue:
                r, c = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and free_mask[nr, nc] and not visited[nr][nc]:
                        visited[nr][nc] = True
                        queue.append((nr, nc))

    return components


def _minimum_clearance_cells(
    occupied_mask,
    reachable_mask,
) -> float:
    """Approximate, grid-based clearance (in cells) at the narrowest point of
    the reachable free space's medial axis (skeleton).

    This is a documented approximation, not an exact continuous-space
    channel width. It has two stages:

    1. An 8-connected (Chebyshev-style) multi-source BFS distance, in grid
       steps, from every free cell to the nearest OCCUPIED cell -- a
       standard discrete distance transform. The world boundary itself is
       deliberately NOT treated as a wall here -- corridor-width validation
       is about clearance between obstacles a robot must pass between, not
       about the edge of the simulated arena, and the six smoke maps
       intentionally let free space run up to the world border in open
       areas.

    2. The raw per-cell distance from stage 1 is NOT by itself a corridor
       width: in any open room, the free cell hugging a wall always has a
       tiny distance-to-nearest-obstacle even though the room is huge, so
       taking a bare minimum over all free cells would flag every map with
       obstacles as "too narrow." What actually characterizes a corridor's
       width is its medial-axis/skeleton cells -- local maxima of the
       distance transform, i.e. free cells whose distance is >= every free
       neighbor's distance. Along a straight corridor these ridge cells sit
       on the centerline and their distance equals half the corridor width;
       wall-hugging cells in a wide room are not ridge cells (a neighbor
       further from the wall always has a strictly larger distance), so
       they do not pull the estimate down. The minimum distance among ridge
       cells in the reachable component is what's returned here;
       validate_sim_map compares 2 * that value * resolution against the
       continuous-space corridor threshold.

    There is no existing public "distance transform over an occupancy grid"
    utility in the codebase to reuse (the only prior art, ryu_frontier_
    graph_bfs._ccl8/coordinated_frontier_planner._cluster_cells, labels
    connected components of a pre-selected cell set, not a distance field),
    so this is a small, local, pure implementation. If a map has no
    obstacles at all, clearance is unbounded (float('inf')), which trivially
    satisfies the corridor-width check.
    """
    height, width = occupied_mask.shape
    if height == 0 or width == 0:
        return 0.0

    inf = float("inf")
    dist = [[inf] * width for _ in range(height)]
    queue: deque[tuple[int, int]] = deque()

    for row in range(height):
        for col in range(width):
            if occupied_mask[row, col]:
                dist[row][col] = 0.0
                queue.append((row, col))

    neighbors8 = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    while queue:
        row, col = queue.popleft()
        base = dist[row][col]
        for dr, dc in neighbors8:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and base + 1.0 < dist[nr][nc]:
                dist[nr][nc] = base + 1.0
                queue.append((nr, nc))

    minimum = inf
    for row in range(height):
        for col in range(width):
            if not reachable_mask[row][col] or occupied_mask[row, col]:
                continue

            cell_dist = dist[row][col]
            is_ridge = True
            for dr, dc in neighbors8:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if occupied_mask[nr, nc]:
                    continue
                if dist[nr][nc] > cell_dist:
                    is_ridge = False
                    break

            if is_ridge and cell_dist < minimum:
                minimum = cell_dist

    return float(minimum)


def _validate_geometry_in_bounds(
    obstacles: Sequence[RectObstacle],
    geometry: GridGeometry,
) -> list[str]:
    errors = []
    x_min, x_max, y_min, y_max = geometry.bounds
    for index, (ox, oy, ow, oh) in enumerate(obstacles):
        if ox < x_min or oy < y_min or (ox + ow) > x_max or (oy + oh) > y_max:
            errors.append(
                f"obstacle[{index}]=({ox}, {oy}, {ow}, {oh}) extends outside world bounds {geometry.bounds}"
            )
    return errors


def _validate_start_positions(
    starts: Sequence,
    obstacles: Sequence[RectObstacle],
    geometry: GridGeometry,
    expected_robot_count: int | None,
) -> tuple[bool, list[str], list[tuple[int, int]]]:
    """Returns (start_positions_valid, errors, seed_cells).

    Collision uses the same padding the runtime collision checker uses
    (CollisionChecker.check_position: point_inside_expanded_rect(position,
    obstacle, robot_radius)) -- i.e. a start is rejected if the robot would
    already be touching an expanded obstacle at t=0, not only if its center
    is literally inside the raw rectangle.
    """
    errors: list[str] = []
    valid = True
    seeds: list[tuple[int, int]] = []

    if expected_robot_count is not None and len(starts) != expected_robot_count:
        errors.append(
            f"expected {expected_robot_count} robot start positions, found {len(starts)}"
        )
        valid = False

    for index, robot in enumerate(starts):
        point = (float(robot.x), float(robot.y))

        if not geometry.in_bounds_world(point[0], point[1]):
            errors.append(f"robot[{index}] start {point} is out of world bounds {geometry.bounds}")
            valid = False
            continue

        safety_radius = float(robot.safety_radius)
        for obstacle in obstacles:
            if point_inside_expanded_rect(point, obstacle, safety_radius):
                errors.append(
                    f"robot[{index}] start {point} collides with obstacle {obstacle} "
                    f"(safety_radius={safety_radius})"
                )
                valid = False
                break

        cell = geometry.world_to_grid(point[0], point[1], clamp=False)
        if cell is not None:
            seeds.append((cell.row, cell.col))

    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            a, b = starts[i], starts[j]
            dx = float(a.x) - float(b.x)
            dy = float(a.y) - float(b.y)
            distance = (dx * dx + dy * dy) ** 0.5
            required = float(a.safety_radius) + float(b.safety_radius)
            if distance < required:
                errors.append(
                    f"robot[{i}] and robot[{j}] starts overlap: distance={distance:.3f} "
                    f"< required={required:.3f}"
                )
                valid = False

    return valid, errors, seeds


def _validate_fire_scenarios(
    fire_scenarios: Sequence[Sequence[Point2D]],
    obstacles: Sequence[RectObstacle],
    geometry: GridGeometry,
    starts: Sequence,
) -> tuple[bool, list[str]]:
    """Fire v0 semantics: fires are traversable (checked with padding=0.0,
    unlike robot starts), must not sit inside solid geometry, must stay in
    bounds, and must not be visible from the initial region -- approximated
    as "within some robot's sensor range AND with an unobstructed straight
    line to that robot", using the same opaque-rectangle test the collision
    checker already provides (distance_segment_to_rect == 0 means the
    segment touches/crosses the obstacle)."""
    errors: list[str] = []
    valid = True

    for scenario_index, fires in enumerate(fire_scenarios):
        for fire_index, fire in enumerate(fires):
            point = (float(fire[0]), float(fire[1]))
            label = f"fire_scenarios[{scenario_index}][{fire_index}]={point}"

            if not geometry.in_bounds_world(point[0], point[1]):
                errors.append(f"{label} is out of world bounds {geometry.bounds}")
                valid = False
                continue

            for obstacle in obstacles:
                if point_inside_expanded_rect(point, obstacle, 0.0):
                    errors.append(f"{label} is inside obstacle {obstacle}")
                    valid = False
                    break

            for robot in starts:
                start = (float(robot.x), float(robot.y))
                dx = point[0] - start[0]
                dy = point[1] - start[1]
                distance = (dx * dx + dy * dy) ** 0.5
                if distance > float(robot.vision):
                    continue

                blocked = any(
                    distance_segment_to_rect(start, point, obstacle) <= 1e-9
                    for obstacle in obstacles
                )
                if not blocked:
                    errors.append(
                        f"{label} is visible from initial region start {start} "
                        f"(distance={distance:.3f} <= vision={float(robot.vision):.3f})"
                    )
                    valid = False
                    break

    return valid, errors


def validate_sim_map(
    *,
    map_id: str,
    config: SimulationConfig,
    fire_scenarios: Sequence[Sequence[Point2D]] = (),
    expected_robot_count: int | None = None,
) -> MapValidationReport:
    """Validate one already-loaded .sim map (SimulationConfig) plus the fire
    scenarios that will run on it (supplied separately -- see module
    docstring). Pure computation: no file I/O, no mutation of `config`."""
    errors: list[str] = []
    warnings: list[str] = []

    resolution = float(config.grid_resolution)
    obstacles = [tuple(float(v) for v in obstacle) for obstacle in config.obstacles]
    starts = normalized_robot_start_configs(config)

    geometry = GridGeometry(WORLD_BOUNDS, resolution)
    grid = _rasterize_ground_truth(obstacles, resolution)
    free_mask = _free_mask(grid)
    occupied_mask = _occupied_mask(grid)

    free_cell_count = int(free_mask.sum())
    occupied_cell_count = int(occupied_mask.sum())
    total_cells = grid.width * grid.height
    obstacle_fraction = occupied_cell_count / total_cells if total_cells else 0.0

    errors.extend(_validate_geometry_in_bounds(obstacles, geometry))

    start_positions_valid, start_errors, seed_cells = _validate_start_positions(
        starts, obstacles, geometry, expected_robot_count
    )
    errors.extend(start_errors)

    fire_positions_valid, fire_errors = _validate_fire_scenarios(
        fire_scenarios, obstacles, geometry, starts
    )
    errors.extend(fire_errors)

    reachable = _bfs_reachable(free_mask, seed_cells)
    reachable_count = sum(1 for row in reachable for cell in row if cell)
    connected_free_fraction = (
        reachable_count / free_cell_count if free_cell_count else 0.0
    )
    if connected_free_fraction < MIN_CONNECTED_FREE_FRACTION:
        errors.append(
            f"connected_free_fraction={connected_free_fraction:.4f} is below the "
            f"required {MIN_CONNECTED_FREE_FRACTION} (isolated free space detected)"
        )

    connected_component_count = _count_connected_components(free_mask)
    if connected_component_count > 1 and connected_free_fraction >= MIN_CONNECTED_FREE_FRACTION:
        warnings.append(
            f"{connected_component_count} disconnected free-space components detected "
            "(within tolerance, but map is fragmented)"
        )

    minimum_estimated_clearance_cells = _minimum_clearance_cells(occupied_mask, reachable)

    max_safety_radius = max((float(robot.safety_radius) for robot in starts), default=0.0)
    required_width_m = _required_corridor_width_m(max_safety_radius, resolution)
    estimated_width_m = 2.0 * minimum_estimated_clearance_cells * resolution
    if free_cell_count > 0 and estimated_width_m < required_width_m:
        errors.append(
            f"estimated narrowest corridor width={estimated_width_m:.3f}m is below the "
            f"required 2*safety_radius+2*grid_resolution={required_width_m:.3f}m"
        )

    valid = not errors

    return MapValidationReport(
        map_id=map_id,
        valid=valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
        width=grid.width,
        height=grid.height,
        resolution=resolution,
        free_cell_count=free_cell_count,
        occupied_cell_count=occupied_cell_count,
        obstacle_fraction=obstacle_fraction,
        connected_free_fraction=connected_free_fraction,
        connected_component_count=connected_component_count,
        start_positions_valid=start_positions_valid,
        fire_positions_valid=fire_positions_valid,
        minimum_estimated_clearance_cells=minimum_estimated_clearance_cells,
    )


def validate_sim_map_file(
    path: str,
    *,
    map_id: str,
    fire_scenarios: Sequence[Sequence[Point2D]] = (),
    expected_robot_count: int | None = None,
) -> MapValidationReport:
    """Convenience wrapper: load_sim_file() (the real parser) + validate_sim_map()."""
    config = load_sim_file(path)
    return validate_sim_map(
        map_id=map_id,
        config=config,
        fire_scenarios=fire_scenarios,
        expected_robot_count=expected_robot_count,
    )


def validate_manifest(
    manifest: dict,
    *,
    maps_dir: str,
) -> dict[str, MapValidationReport]:
    """Validate every map entry of a smoke-corpus manifest.json.

    `manifest["maps"]` entries are expected to carry at least `map_id`,
    `filename`, and `fire_scenarios` (a list of scenarios, each a list of
    {"x": ..., "y": ...} points) -- see experiments/maps/smoke_v0/
    manifest.json. `manifest["base_robot_count"]`, if present, is used as
    the expected robot-start count for every map unless a per-map entry
    overrides it with its own `expected_robot_count`. Returns one report
    per map_id, in manifest order.
    """
    default_robot_count = manifest.get("base_robot_count")

    reports: dict[str, MapValidationReport] = {}
    for entry in manifest.get("maps", []):
        map_id = entry["map_id"]
        path = os.path.join(maps_dir, entry["filename"])
        fire_scenarios = [
            [(float(fire["x"]), float(fire["y"])) for fire in scenario.get("fires", [])]
            for scenario in entry.get("fire_scenarios", [])
        ]
        expected_robot_count = entry.get("expected_robot_count", default_robot_count)
        reports[map_id] = validate_sim_map_file(
            path,
            map_id=map_id,
            fire_scenarios=fire_scenarios,
            expected_robot_count=expected_robot_count,
        )

    return reports
