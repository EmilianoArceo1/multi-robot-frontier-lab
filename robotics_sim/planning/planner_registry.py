"""
Planner registry for the 2D robotics simulator.

Public contract:
    compute_planned_waypoints(...) -> tuple[bool, str, list[tuple[float, float]]]

This module is a thin orchestration layer:
    - obtains or builds a planning OccupancyGrid
    - selects A*, Dijkstra, or Direct
    - dispatches path simplification to path_simplifier.py
    - converts grid paths to executable world waypoints

It does not implement its own grid conversion, obstacle inflation, A*, Dijkstra,
or path simplification.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Iterable, Any

from robotics_sim.diagnostics.capture import PlanDebugCapture
from robotics_sim.environment.occupancy_grid import (
    FREE,
    OCCUPIED,
    GridCell,
    OccupancyGrid,
)
from robotics_sim.planning.grid_planners import AStarPlanner, DijkstraPlanner
from robotics_sim.planning.path_simplifier import (
    DEFAULT_PATH_SIMPLIFIER,
    grid_path_to_world_path,
    line_of_sight_grid_safe,
    simplify_grid_path,
)


Point = tuple[float, float]
Rect = tuple[float, float, float, float]


def _as_point(point: Any) -> Point:
    return (float(point[0]), float(point[1]))


def _cell_from_world_or_none(grid: OccupancyGrid, point: Point, *, clamp: bool = False) -> GridCell | None:
    cell = grid.world_to_grid(float(point[0]), float(point[1]), clamp=clamp)
    if not grid.in_bounds(cell):
        return None
    return cell


def _is_occupied(grid: OccupancyGrid, cell: GridCell) -> bool:
    return grid.in_bounds(cell) and grid.get_value(cell) == OCCUPIED


def _nearest_traversable_cell(
    grid: OccupancyGrid,
    start: GridCell,
    *,
    max_depth: int = 12,
) -> GridCell | None:
    if grid.is_traversable(start):
        return start

    visited: set[GridCell] = {start}
    queue: deque[tuple[GridCell, int]] = deque([(start, 0)])

    while queue:
        current, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for neighbor, _ in grid.neighbors(
            current,
            allow_diagonal=True,
            prevent_corner_cutting=True,
        ):
            if neighbor in visited:
                continue

            if grid.is_traversable(neighbor):
                return neighbor

            visited.add(neighbor)
            queue.append((neighbor, depth + 1))

    return None


def _clear_start_cell_for_live_robot(grid: OccupancyGrid, start_cell: GridCell) -> bool:
    """Let a live robot leave its current cell if inflation marked it occupied."""
    if not grid.in_bounds(start_cell):
        return False

    was_occupied = _is_occupied(grid, start_cell)
    grid.set_value(start_cell, FREE)
    return was_occupied


def _select_planner(planner_type: str):
    planner = str(planner_type or "A*").strip().lower()

    if planner == "direct":
        return None, "Direct"

    if planner in {"dijkstra", "dijsktra"}:
        return DijkstraPlanner(
            allow_diagonal=True,
            prevent_corner_cutting=True,
        ), "Dijkstra"

    return AStarPlanner(
        allow_diagonal=True,
        prevent_corner_cutting=True,
    ), "A*"


def _build_planning_grid_from_legacy_inputs(
    *,
    bounds: tuple[float, float, float, float],
    resolution: float,
    robot_radius: float,
    obstacle_points: Iterable[Point] | None,
    obstacles: Iterable[Rect] | None,
    unknown_is_traversable: bool,
) -> OccupancyGrid:
    """
    Compatibility path for old engine calls that do not pass BeliefMap.

    Unknown space is represented by the initial value:
        optimistic: FREE
        conservative: UNKNOWN
    """
    initial_value = FREE if unknown_is_traversable else -1

    grid = OccupancyGrid.from_bounds(
        x_min=float(bounds[0]),
        x_max=float(bounds[1]),
        y_min=float(bounds[2]),
        y_max=float(bounds[3]),
        resolution=float(resolution),
        initial_value=initial_value,
        unknown_is_traversable=unknown_is_traversable,
    )

    padding = max(0.0, float(robot_radius))

    if obstacle_points is not None:
        grid.add_obstacle_points(obstacle_points, padding=padding)

    if obstacles is not None:
        grid.add_rectangular_obstacles(obstacles, padding=padding)

    return grid


def _build_planning_grid(
    *,
    belief_map=None,
    planning_grid: OccupancyGrid | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    robot_radius: float = 0.0,
    obstacle_points: Iterable[Point] | None = None,
    obstacles: Iterable[Rect] | None = None,
    unknown_is_traversable: bool = True,
    safety_margin: float = 0.0,
) -> OccupancyGrid:
    if planning_grid is not None:
        return planning_grid.copy()

    inflate_radius = max(0.0, float(robot_radius) + float(safety_margin))

    if belief_map is not None:
        return belief_map.to_planning_grid(
            unknown_is_traversable=unknown_is_traversable,
            inflate_radius=inflate_radius,
        )

    if bounds is None or resolution is None:
        raise ValueError("bounds and resolution are required when belief_map/planning_grid are not provided.")

    return _build_planning_grid_from_legacy_inputs(
        bounds=bounds,
        resolution=resolution,
        robot_radius=inflate_radius,
        obstacle_points=obstacle_points,
        obstacles=obstacles,
        unknown_is_traversable=unknown_is_traversable,
    )


def _drop_start_waypoint(
    *,
    grid: OccupancyGrid,
    start_cell: GridCell,
    world_path: list[Point],
) -> list[Point]:
    """Remove the start-cell waypoint from the executable route."""
    if not world_path:
        return []

    cleaned: list[Point] = []
    skipping_start = True

    for point in world_path:
        cell = _cell_from_world_or_none(grid, point, clamp=True)

        if skipping_start and cell == start_cell:
            continue

        skipping_start = False
        cleaned.append((round(float(point[0]), 3), round(float(point[1]), 3)))

    return cleaned


def compute_planned_waypoints(
    *,
    planner_type: str,
    start_xy: Point,
    goal_xy: Point,
    obstacles: Iterable[Rect] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    resolution: float | None = None,
    robot_radius: float = 0.0,
    obstacle_points: Iterable[Point] | None = None,
    mapped_obstacle_points: Iterable[Point] | None = None,
    path_simplifier: str = DEFAULT_PATH_SIMPLIFIER,
    belief_map=None,
    planning_grid: OccupancyGrid | None = None,
    unknown_is_traversable: bool = True,
    safety_margin: float = 0.0,
    debug_capture: PlanDebugCapture | None = None,
) -> tuple[bool, str, list[Point]]:
    """
    Compute executable world-coordinate waypoints.

    Returns:
        (success, reason, waypoints)

    Failure returns an empty waypoint list. The controller must not execute a
    route when success is False.

    debug_capture: optional outparam. When provided and planning succeeds,
    filled in place with the raw/simplified grid path, start/first-waypoint
    cells, and planner/simplifier names -- values this function already
    computes locally and would otherwise discard. Every existing caller
    omits it (default None); nothing below runs when it is None.
    """
    start_xy = _as_point(start_xy)
    goal_xy = _as_point(goal_xy)

    selected_planner, planner_name = _select_planner(planner_type)

    if planner_name == "Direct":
        return True, "direct route; no grid collision checking", [goal_xy]

    known_obstacle_points = obstacle_points if obstacle_points is not None else mapped_obstacle_points

    grid = _build_planning_grid(
        belief_map=belief_map,
        planning_grid=planning_grid,
        bounds=bounds,
        resolution=resolution,
        robot_radius=robot_radius,
        obstacle_points=known_obstacle_points,
        obstacles=obstacles,
        unknown_is_traversable=unknown_is_traversable,
        safety_margin=safety_margin,
    )

    start_cell = _cell_from_world_or_none(grid, start_xy, clamp=True)
    goal_cell = _cell_from_world_or_none(grid, goal_xy, clamp=True)

    if start_cell is None:
        return False, "start is outside planning bounds", []

    if goal_cell is None:
        return False, "goal is outside planning bounds", []

    start_was_occupied = _clear_start_cell_for_live_robot(grid, start_cell)

    adjusted_goal = False
    if _is_occupied(grid, goal_cell):
        nearest = _nearest_traversable_cell(grid, goal_cell)

        if nearest is None:
            return False, "goal cell is occupied and no nearby traversable cell was found", []

        goal_cell = nearest
        goal_xy = grid.grid_to_world(goal_cell)
        grid.set_value(goal_cell, FREE)
        adjusted_goal = True

    assert selected_planner is not None

    # A*/Dijkstra search is unnecessary when the selected target is already
    # visible through the exact derived planning grid. This check uses the
    # same UNKNOWN policy, obstacle inflation, dynamic obstacles and hazard
    # projection as the planner, so it cannot create a shortcut through a
    # cell that A*/Dijkstra considers blocked. It also prevents grid-search
    # tie-breaking from returning a long loop for a frontier immediately
    # behind the robot: the executable action is simply rotate, then follow
    # the clear segment to the target.
    if line_of_sight_grid_safe(grid, start_cell, goal_cell):
        direct_goal = _as_point(goal_xy)
        if debug_capture is not None:
            start_world = grid.grid_to_world(start_cell)
            debug_capture.planner_name = planner_name
            debug_capture.simplifier_name = str(path_simplifier)
            debug_capture.raw_world_path = (start_world, direct_goal)
            debug_capture.simplified_world_path = (start_world, direct_goal)
            debug_capture.start_cell = start_cell
            debug_capture.start_cell_world = start_world
            debug_capture.first_waypoint_cell = goal_cell
            debug_capture.first_waypoint_world = direct_goal
            debug_capture.unknown_is_traversable = bool(unknown_is_traversable)
            debug_capture.start_cell_cleared = bool(start_was_occupied)
            debug_capture.total_cost = float(math.dist(start_world, direct_goal))
            debug_capture.expanded_nodes = 0
            debug_capture.goal_cell = goal_cell
            debug_capture.grid_resolution = float(grid.resolution)

        reason_parts = [
            f"direct line-of-sight route with {planner_name}",
            f"simplifier={path_simplifier}",
            f"unknown_is_traversable={unknown_is_traversable}",
        ]
        if adjusted_goal:
            reason_parts.append("goal adjusted to nearest traversable cell")
        if start_was_occupied:
            reason_parts.append("start cell cleared because robot is already there")
        return True, "; ".join(reason_parts), [direct_goal]

    result = selected_planner.plan(
        grid=grid,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )

    if not result.success:
        return False, result.reason, []

    simplified_grid_path = simplify_grid_path(
        result.grid_path,
        method=path_simplifier,
        grid=grid,
    )

    world_path = grid_path_to_world_path(grid, simplified_grid_path)

    waypoints = _drop_start_waypoint(
        grid=grid,
        start_cell=start_cell,
        world_path=world_path,
    )

    if not waypoints:
        waypoints = [tuple(round(float(v), 3) for v in grid.grid_to_world(goal_cell))]

    reason_parts = [
        f"path found with {planner_name}",
        f"simplifier={path_simplifier}",
        f"unknown_is_traversable={unknown_is_traversable}",
    ]

    if adjusted_goal:
        reason_parts.append("goal adjusted to nearest traversable cell")

    if start_was_occupied:
        reason_parts.append("start cell cleared because robot is already there")

    if debug_capture is not None:
        first_waypoint_cell = _cell_from_world_or_none(grid, waypoints[0], clamp=True) if waypoints else None
        debug_capture.planner_name = planner_name
        debug_capture.simplifier_name = str(path_simplifier)
        debug_capture.raw_world_path = tuple(grid.grid_to_world(cell) for cell in result.grid_path)
        debug_capture.simplified_world_path = tuple(world_path)
        debug_capture.start_cell = start_cell
        debug_capture.start_cell_world = grid.grid_to_world(start_cell)
        debug_capture.first_waypoint_cell = first_waypoint_cell
        debug_capture.first_waypoint_world = (
            grid.grid_to_world(first_waypoint_cell) if first_waypoint_cell is not None else None
        )
        debug_capture.unknown_is_traversable = bool(unknown_is_traversable)
        debug_capture.start_cell_cleared = bool(start_was_occupied)
        debug_capture.total_cost = float(result.total_cost)
        debug_capture.expanded_nodes = int(result.expanded_nodes)
        debug_capture.goal_cell = goal_cell
        debug_capture.grid_resolution = float(grid.resolution)

    return True, "; ".join(reason_parts), waypoints
