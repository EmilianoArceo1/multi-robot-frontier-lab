"""
Grid-based planners for the simulator.

A* and Dijkstra share the same graph-search implementation. Dijkstra is A* with
zero heuristic.

This module does not choose exploration targets. It only computes a path on an
OccupancyGrid using that grid's traversability policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from robotics_sim.environment.occupancy_grid import GridCell, OccupancyGrid
from robotics_sim.planning.path_simplifier import (
    grid_path_to_world_path,
    simplify_path_by_direction_changes,
)


@dataclass
class PlanningResult:
    success: bool
    reason: str
    grid_path: list[GridCell]
    simplified_grid_path: list[GridCell]
    world_path: list[tuple[float, float]]
    total_cost: float
    expanded_nodes: int = 0


class AStarPlanner:
    """
    A* over an OccupancyGrid.

    The planner does not decide whether UNKNOWN is traversable. That policy
    belongs to the grid/projection passed to plan().

    Examples:
        conservative planning grid:
            UNKNOWN blocked
        optimistic planning grid:
            UNKNOWN traversable
    """

    def __init__(
        self,
        allow_diagonal: bool = True,
        prevent_corner_cutting: bool = True,
        allow_unknown: bool | None = None,
    ):
        self.allow_diagonal = bool(allow_diagonal)
        self.prevent_corner_cutting = bool(prevent_corner_cutting)
        self.allow_unknown = allow_unknown

    def heuristic(self, a: GridCell, b: GridCell, resolution: float) -> float:
        dr = abs(a.row - b.row)
        dc = abs(a.col - b.col)

        if self.allow_diagonal:
            diag = min(dr, dc)
            straight = max(dr, dc) - diag
            return (math.sqrt(2.0) * diag + straight) * resolution

        return (dr + dc) * resolution

    @staticmethod
    def reconstruct_path(came_from: dict[GridCell, GridCell], current: GridCell) -> list[GridCell]:
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def plan(
        self,
        grid: OccupancyGrid,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> PlanningResult:
        start = grid.world_to_grid(*start_xy)
        goal = grid.world_to_grid(*goal_xy)

        if not grid.in_bounds(start):
            return PlanningResult(False, "start is outside the grid", [], [], [], math.inf)

        if not grid.in_bounds(goal):
            return PlanningResult(False, "goal is outside the grid", [], [], [], math.inf)

        if not grid.is_traversable(start, allow_unknown=self.allow_unknown):
            return PlanningResult(False, "start cell is not traversable", [], [], [], math.inf)

        if not grid.is_traversable(goal, allow_unknown=self.allow_unknown):
            return PlanningResult(False, "goal cell is not traversable", [], [], [], math.inf)

        open_heap: list[tuple[float, int, GridCell]] = []
        counter = 0

        heapq.heappush(
            open_heap,
            (self.heuristic(start, goal, grid.resolution), counter, start),
        )

        came_from: dict[GridCell, GridCell] = {}
        g_score: dict[GridCell, float] = {start: 0.0}
        visited: set[GridCell] = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                grid_path = self.reconstruct_path(came_from, current)
                simplified = simplify_path_by_direction_changes(grid_path)
                world_path = grid_path_to_world_path(grid, simplified)

                return PlanningResult(
                    True,
                    "path found",
                    grid_path,
                    simplified,
                    world_path,
                    g_score[current],
                    len(visited),
                )

            for neighbor, move_cost in grid.neighbors(
                current,
                allow_diagonal=self.allow_diagonal,
                prevent_corner_cutting=self.prevent_corner_cutting,
                allow_unknown=self.allow_unknown,
            ):
                tentative_g = g_score[current] + move_cost

                if tentative_g >= g_score.get(neighbor, math.inf):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + self.heuristic(neighbor, goal, grid.resolution)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))

        return PlanningResult(False, "no path found", [], [], [], math.inf)


class DijkstraPlanner(AStarPlanner):
    """Dijkstra implemented as A* with zero heuristic."""

    def heuristic(self, a: GridCell, b: GridCell, resolution: float) -> float:
        return 0.0
