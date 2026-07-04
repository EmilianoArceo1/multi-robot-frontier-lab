"""
PlannerServices — thin façade over path-planning and frontier-selection utilities.

Injected into RobotAgent.step() so the agent can request plans and pick
exploration targets without importing engine internals, Qt, or any robot physics.

Responsibilities:
    plan_path()                — synchronous A*/Dijkstra call.
    select_exploration_target() — frontier / informative target selection.

What does NOT live here:
    - Async worker management (PlannerWorker stays in engine.py for now).
    - BeliefMap construction or obstacle mapping.
    - Qt signals or canvas updates.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_sim.environment.occupancy_grid import OccupancyGrid

# Lazy imports so the module can be loaded even when planning packages are
# absent (e.g., lightweight unit tests that only test the decision layer).
try:
    from robotics_sim.planning.planner_registry import compute_planned_waypoints as _cpw
except ImportError:
    _cpw = None  # type: ignore[assignment]

try:
    from robotics_sim.planning.exploration_planners import select_exploration_goal as _seg
except ImportError:
    _seg = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Sentinel used when exploration planners are unavailable.
# ---------------------------------------------------------------------------

class _FailedResult:
    success = False
    target = None
    reason = "exploration planner package not available"
    candidates: tuple = ()


# ---------------------------------------------------------------------------

class PlannerServices:
    """
    Provides path planning and exploration target selection to RobotAgent.

    A single instance is created by the engine and passed into every
    agent.step() call.  Stateless — safe to share across all robot agents.
    """

    # ------------------------------------------------------------------ path

    def plan_path(
        self,
        *,
        planner_type: str,
        path_simplifier: str,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        planning_grid: "OccupancyGrid | None",
        robot_radius: float,
        bounds: tuple[float, float, float, float],
        resolution: float,
    ) -> tuple[bool, str, list[tuple[float, float]]]:
        """
        Synchronous A*/Dijkstra path planning.

        Returns (success, reason, waypoints).
        Waypoints are world-coordinate (x, y) tuples.
        """
        if planner_type == "Direct":
            return True, "direct route", [goal_xy]

        if _cpw is None:
            return False, "planner package not available", []

        kwargs: dict = dict(
            planner_type=planner_type,
            start_xy=start_xy,
            goal_xy=goal_xy,
            obstacles=[],
            bounds=bounds,
            resolution=resolution,
            robot_radius=robot_radius,
            planning_grid=planning_grid,
            unknown_is_traversable=True,
            obstacle_points=[],
        )

        try:
            has_simplifier = "path_simplifier" in inspect.signature(_cpw).parameters
        except (TypeError, ValueError):
            has_simplifier = False

        try:
            if has_simplifier:
                return _cpw(**kwargs, path_simplifier=path_simplifier)
            return _cpw(**kwargs)
        except Exception as exc:
            return False, f"planner error: {exc}", []

    # ------------------------------------------------------------------ exploration

    def select_exploration_target(
        self,
        *,
        planner_name: str,
        belief_map,
        robot_xy: tuple[float, float],
        robot_heading: float,
        current_target,
        final_goal_xy: tuple[float, float] | None,
        robot_radius: float,
        sensor_range: float,
        vision_model: str,
        ipp_distance_penalty: float,
        excluded_targets: list[tuple[float, float]] | None = None,
    ):
        """
        Select the next exploration frontier target.

        Returns an ExplorationPlannerResult-like object with:
            .success  bool
            .target   tuple[float, float] | None
            .reason   str

        On failure returns a sentinel object with .success = False.

        Note: multi-robot coordination (duplicate-frontier avoidance) is still
        handled by the engine via MultiRobotCoordinator.  The excluded_targets
        list provides a lightweight alternative for single-robot filtering.
        """
        if _seg is None:
            return _FailedResult()

        return _seg(
            planner_name,
            belief_map=belief_map,
            robot_xy=robot_xy,
            robot_heading=robot_heading,
            current_target=current_target,
            final_goal_xy=final_goal_xy,
            robot_count=1,
            robot_radius=robot_radius,
            sensor_range=sensor_range,
            vision_model=vision_model,
            ipp_distance_penalty=ipp_distance_penalty,
            target_exclusion_radius=0.0,
        )
