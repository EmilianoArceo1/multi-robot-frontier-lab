"""
Simulation controller logic.

This is the main file to read for behavior. It contains the methods that
start/reset the simulation, assign goals/frontiers, request A*/Dijkstra
routes, update sensor mapping, run each simulation step, check robot-obstacle
and robot-robot safety, and compute metrics.

It is implemented as a mixin so the Qt MainWindow can keep UI construction
separate from simulation behavior without a risky rewrite of all state
references in one step.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import time

import numpy as np
from PySide6.QtCore import Qt, Signal, QObject, QRunnable
from PySide6.QtWidgets import QFileDialog, QMessageBox

from robot import Robot

from robotics_sim.simulation.config import *
from robotics_sim.planning.exploration_planners import (
    DEFAULT_EXPLORATION_PLANNER,
    select_exploration_goal,
)
from robotics_sim.simulation.navigation_modes import (
    GOAL_SEEKING_PLANNER,
    is_goal_seeking_planner,
    is_exploration_planner,
)
from robotics_sim.simulation.runtime_robot_registry import RuntimeRobotRegistry
from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.app.widgets import make_icon, SimulationMetricsWindow, SimulationConsoleWindow
from robotics_interfaces.plugins import PluginMetadata, build_runtime_profile
from robotics_sim.planning.coordinated_frontier_planner import validate_multi_robot_corridor
from robotics_sim.simulation.coordination import (
    MultiRobotCoordinator,
    RobotCoordinationState,
    map_robot_commands_by_id,
    runtime_profile_for_strategy,
    select_runtime_control_source,
    select_runtime_path_source,
)
from robotics_sim.simulation.plugin_loader import PluginLoadError

_LOGGER = logging.getLogger(__name__)

try:
    from robotics_sim.planning.planner_registry import compute_planned_waypoints
except ImportError:
    compute_planned_waypoints = None

try:
    from robotics_sim.environment.collision_checker import (
        CollisionChecker,
        RobotSnapshot,
    )
except ImportError:
    CollisionChecker = None
    RobotSnapshot = None

# New POO architecture — imported lazily inside methods to avoid circular deps.
# These imports are only used by the new build_observation / apply_navigation_decision
# / planner_services plumbing.  The existing simulation loop is unchanged.
try:
    from robotics_sim.simulation.observation import RobotObservation
    from robotics_sim.simulation.planner_services import PlannerServices
except ImportError:  # pragma: no cover
    RobotObservation = None  # type: ignore[assignment,misc]
    PlannerServices = None  # type: ignore[assignment]

class PlannerWorkerSignals(QObject):
    route_ready = Signal(int, bool, str, list)


class PlannerWorker(QRunnable):
    """
    Compute A*/Dijkstra routes outside the GUI thread.

    Only immutable/simple data is passed into the worker. It must never touch
    Qt widgets or the live Robot object.
    """

    def __init__(self, request_id: int, planner_kwargs: dict, path_simplifier: str):
        super().__init__()
        self.setAutoDelete(False)
        self.request_id = int(request_id)
        self.planner_kwargs = dict(planner_kwargs)
        self.path_simplifier = str(path_simplifier)
        self.signals = PlannerWorkerSignals()

    def run(self):
        if bool(self.planner_kwargs.get("__hold__", False)):
            self.signals.route_ready.emit(
                self.request_id,
                False,
                str(self.planner_kwargs.get("__hold_reason__", "holding position")),
                [],
            )
            return

        if compute_planned_waypoints is None:
            self.signals.route_ready.emit(
                self.request_id,
                False,
                "planner package is not available",
                [],
            )
            return

        try:
            supports_simplifier = False
            try:
                supports_simplifier = "path_simplifier" in inspect.signature(compute_planned_waypoints).parameters
            except (TypeError, ValueError):
                supports_simplifier = False

            if supports_simplifier:
                success, reason, waypoints = compute_planned_waypoints(
                    **self.planner_kwargs,
                    path_simplifier=self.path_simplifier,
                )
            else:
                success, reason, waypoints = compute_planned_waypoints(**self.planner_kwargs)
        except Exception as exc:  # noqa: BLE001 - report planner failures to GUI safely.
            success = False
            reason = f"planner worker failed: {exc}"
            waypoints = []

        self.signals.route_ready.emit(
            self.request_id,
            bool(success),
            str(reason),
            [tuple(point) for point in waypoints],
        )



# ============================================================
# METRICS WINDOW


class SimulationControllerMixin:
    ROUTE_STATE_ACTIVE = "ACTIVE"
    ROUTE_STATE_HOLD_NO_FRONTIER = "HOLD_NO_FRONTIER"
    ROUTE_STATE_STUCK_SAFETY = "STUCK_SAFETY"
    ROUTE_STATE_ESCAPE_LOCAL = "ESCAPE_LOCAL"
    # A route candidate exists but corridor validation rejected it -- this is
    # not "no frontier", it is "no safe route to the frontier we do have".
    ROUTE_STATE_HOLD_ROUTE_BLOCKED = "HOLD_ROUTE_BLOCKED"
    # Specifically a route_conflict_with_active_route rejection: the target
    # itself is fine, a teammate's active route is just in the way right now.
    ROUTE_STATE_WAITING_FOR_CORRIDOR = "WAITING_FOR_CORRIDOR"

    # How many candidate targets to try (1 initial attempt + retries) before
    # a corridor-blocked robot is allowed to fall back to HOLD/WAITING.
    MAX_ROUTE_RECOVERY_ATTEMPTS = 3

    # NAVIGATION MODE / ROBOT AGENT HELPERS
    # ========================================================

    def exploration_planner_name(self) -> str:
        return str(getattr(self.config, "exploration_planner", GOAL_SEEKING_PLANNER))

    def is_goal_seeking_mode(self) -> bool:
        return is_goal_seeking_planner(self.exploration_planner_name())

    def is_exploration_mode(self) -> bool:
        return is_exploration_planner(self.exploration_planner_name())

    def ensure_runtime_robot_registry(self) -> RuntimeRobotRegistry:
        if not hasattr(self, "runtime_robot_registry") or self.runtime_robot_registry is None:
            self.runtime_robot_registry = RuntimeRobotRegistry()
            self.robot_agents = self.runtime_robot_registry.agents
        return self.runtime_robot_registry

    def sync_runtime_robot_agents(self) -> None:
        registry = self.ensure_runtime_robot_registry()
        robots = list(getattr(self, "robots", []) or [])
        if not robots and getattr(self, "robot", None) is not None:
            robots = [self.robot]

        radii = [self.safety_radius_for_robot(robot) for robot in robots]
        registry.sync_from_robots(
            robots=robots,
            planner_mode=self.exploration_planner_name(),
            final_goal_xy=self.final_goal_xy() if hasattr(self, "config") else None,
            radii=radii,
        )
        self.robot_agents = registry.agents

    def runtime_agent(self, robot_index: int | None = None):
        self.sync_runtime_robot_agents()
        if not getattr(self, "robot_agents", None):
            return None
        if robot_index is None:
            return self.robot_agents[0]
        index = int(robot_index)
        if 0 <= index < len(self.robot_agents):
            return self.robot_agents[index]
        return None

    # BELIEF MAP
    # ========================================================

    def reset_belief_map(self, robot_count: int = 1) -> None:
        """Create a fresh logical occupancy/belief map.

        This is the source of truth for exploration logic. The canvas pixmaps are
        rendering caches only.
        """
        self.belief_map = BeliefMap(
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=max(float(self.config.grid_resolution), 0.10),
            robot_count=max(1, int(robot_count)),
        )
        self.explored_free_points = set()
        # Dense, visible obstacle-boundary samples. This is intentionally
        # separate from belief_map.grid OCCUPIED cells; the grid is logical,
        # while these samples preserve the obstacle contour for rendering and
        # local safety checks.
        self.mapped_obstacle_points = []
        self.mapped_obstacle_point_keys: set[tuple[float, float]] = set()

    def ensure_belief_map(self) -> BeliefMap:
        """Return the active belief map, creating it if needed."""
        if not hasattr(self, "belief_map") or self.belief_map is None:
            count = len(getattr(self, "robots", [])) if getattr(self, "robots", None) else 1
            self.reset_belief_map(robot_count=max(1, count))
        return self.belief_map

    def sync_legacy_map_views_from_belief(self) -> None:
        """Update legacy views without destroying boundary obstacle samples.

        The logical FREE/UNKNOWN/OCCUPIED state lives in ``belief_map.grid``.
        However, the visual obstacle trace and local route safety checks need
        dense boundary samples, not just one center point per occupied cell.

        Therefore this method exports explored FREE cells from the belief map,
        but deliberately does *not* replace ``self.mapped_obstacle_points``.
        Those mapped obstacle points are maintained by ``update_sensed_obstacles``
        as visible boundary samples.
        """
        belief = self.ensure_belief_map()
        self.explored_free_points = belief.explored_points()
        if not hasattr(self, "mapped_obstacle_points"):
            self.mapped_obstacle_points = []
        if not hasattr(self, "mapped_obstacle_point_keys"):
            self.mapped_obstacle_point_keys = {
                (round(float(p[0]), 3), round(float(p[1]), 3))
                for p in self.mapped_obstacle_points
            }

    # CONFIG
    # ========================================================

    def read_config(self) -> SimulationConfig:
        return SimulationConfig(
            x=float(self.x_input.value()),
            y=float(self.y_input.value()),
            theta=float(self.theta_input.value()),
            v=float(self.v_slider.value()),
            vision=float(self.vision_slider.value()),
            body_radius=float(self.body_radius_slider.value()),
            safety_radius=max(float(self.safety_radius_slider.value()), float(self.body_radius_slider.value())),
            goal_x=float(self.goal_x_input.value()),
            goal_y=float(self.goal_y_input.value()),
            max_speed=float(self.max_speed_input.value()),
            max_acceleration=float(self.max_accel_input.value()),
            max_angular_speed=float(self.max_omega_input.value()),
            goal_tolerance=float(self.goal_tol_input.value()),
            acceleration_gain=float(self.accel_gain_input.value()),
            planner_type=self.planner_combo.currentText(),
            path_simplifier=self.path_simplifier_combo.currentText(),
            exploration_planner=self.exploration_planner_combo.currentText(),
            coordinator_type=self.coordinator_combo.currentText() if hasattr(self, "coordinator_combo") else self.config.coordinator_type,
            exploration_replan_cooldown=max(0.0, float(self.exploration_cooldown_input.value())),
            ipp_distance_penalty=max(0.0, float(self.ipp_lambda_input.value())),
            vision_model=self.vision_combo.currentText(),
            agent_mode=self.top_bar.mode_selector.currentText(),
            grid_resolution=self.config.grid_resolution,
            obstacles=list(self.config.obstacles),
            show_goal_preview=self.preview_switch.isChecked(),
            show_path=True,
            show_vision=True,
            show_explored_area=self.explored_area_switch.isChecked(),
            show_obstacles=self.obstacles_switch.isChecked(),
            show_robot_orders=self.orders_switch.isChecked(),
            mapping_point_spacing=self.config.mapping_point_spacing,
            robot_count=max(1, min(8, int(round(float(self.robot_count_input.value()))))) if hasattr(self, "robot_count_input") else self.config.robot_count,
            selected_robot_index=int(getattr(self, "selected_robot_index", 0)),
            same_robot_configuration=self.same_config_switch.isChecked() if hasattr(self, "same_config_switch") else self.config.same_robot_configuration,
            robots=list(getattr(self, "multi_robot_configs", self.config.robots)),
        )

    def enforce_radius_consistency(self, *_):
        """
        Keep safety radius r physically valid.

        r is a clearance radius, so it cannot be smaller than the robot body.
        """
        body_radius = float(self.body_radius_slider.value())
        safety_radius = float(self.safety_radius_slider.value())
        if safety_radius < body_radius:
            self.safety_radius_slider.setValue(body_radius)

    def enforce_selected_multi_radius_consistency(self, *_):
        """Keep per-robot safety radius physically valid in multi config."""
        if not hasattr(self, "multi_body_radius_slider"):
            return
        body_radius = float(self.multi_body_radius_slider.value())
        safety_radius = float(self.multi_safety_radius_slider.value())
        if safety_radius < body_radius:
            self.multi_safety_radius_slider.setValue(body_radius)

    def update_preview(self):
        self.enforce_radius_consistency()
        self.enforce_selected_multi_radius_consistency()
        self.update_relevant_parameter_visibility()
        self.config = self.read_config()
        self.canvas.set_preview_config(self.config)

    def apply_config_to_widgets(self, config: SimulationConfig) -> None:
        """
        Push a loaded .sim configuration back into the GUI controls.
        """
        self.x_input.setValue(config.x)
        self.y_input.setValue(config.y)
        self.theta_input.setValue(config.theta)
        self.v_slider.setValue(config.v)
        self.vision_slider.setValue(config.vision)
        self.body_radius_slider.setValue(config.body_radius)
        self.safety_radius_slider.setValue(max(config.safety_radius, config.body_radius))
        self.goal_x_input.setValue(config.goal_x)
        self.goal_y_input.setValue(config.goal_y)
        self.max_speed_input.setValue(config.max_speed)
        self.max_omega_input.setValue(config.max_angular_speed)
        self.max_accel_input.setValue(config.max_acceleration)
        self.goal_tol_input.setValue(config.goal_tolerance)
        self.accel_gain_input.setValue(config.acceleration_gain)
        self.preview_switch.setChecked(config.show_goal_preview)
        self.orders_switch.setChecked(config.show_robot_orders)
        self.obstacles_switch.setChecked(config.show_obstacles)
        self.explored_area_switch.setChecked(config.show_explored_area)
        self.planner_combo.setCurrentText(config.planner_type)
        self.path_simplifier_combo.setCurrentText(config.path_simplifier)
        self.exploration_planner_combo.setCurrentText(config.exploration_planner)
        if hasattr(self, "coordinator_combo"):
            self.coordinator_combo.setCurrentText(config.coordinator_type)
        self.exploration_cooldown_input.setValue(config.exploration_replan_cooldown)
        self.ipp_lambda_input.setValue(config.ipp_distance_penalty)
        self.vision_combo.setCurrentText(config.vision_model)
        self.top_bar.mode_selector.setCurrentText(config.agent_mode)

        self.multi_robot_configs = normalized_robot_start_configs(config)
        self.selected_robot_index = max(0, min(int(config.selected_robot_index), len(self.multi_robot_configs) - 1))
        if hasattr(self, "robot_count_input"):
            self.robot_count_input.setValue(max(1, min(8, int(config.robot_count))))
            self.same_config_switch.setChecked(bool(config.same_robot_configuration))
            self.load_selected_robot_into_panel()

        self.config = config
        self.spatial_index.rebuild(self.config.obstacles)
        self.update_relevant_parameter_visibility()
        self.set_configuration_locked(self.running or self.robot is not None)
        self.canvas.set_preview_config(self.config)
        self.canvas.set_planned_path([])
        self.canvas.set_exploration_target(None)

    def save_simulation_config(self) -> None:
        self.config = self.read_config()

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save simulation scenario",
            "scenario.sim",
            "Simulation files (*.sim);;JSON files (*.json);;All files (*)",
        )

        if not path:
            return

        if not path.lower().endswith((".sim", ".json")):
            path += ".sim"

        try:
            save_sim_file(path, self.config)
            self.canvas.set_status(f"Saved scenario: {os.path.basename(path)}")
        except OSError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def load_simulation_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load simulation scenario",
            "",
            "Simulation files (*.sim);;JSON files (*.json);;All files (*)",
        )

        if not path:
            return

        try:
            config = load_sim_file(path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.reset_simulation()
        self.apply_config_to_widgets(config)
        self.canvas.set_status(f"Loaded scenario: {os.path.basename(path)}")

    def final_goal_xy(self) -> tuple[float, float]:
        return (float(self.config.goal_x), float(self.config.goal_y))

    def select_navigation_goal(self, start_xy: tuple[float, float]) -> tuple[tuple[float, float] | None, str]:
        """
        Select the current navigation target.

        Goal seeking:
            the GUI final goal G is executable.

        Exploration:
            G is only a visual/reference goal. The executable target must come
            from the exploration planner. If no exploration target exists, return
            None so the caller can enter HOLD instead of planning to G.
        """
        final_goal = self.final_goal_xy()
        planner_name = str(self.config.exploration_planner)

        if is_goal_seeking_planner(planner_name):
            self.current_exploration_target = None
            self.last_goal_selection_reason = "using final mission goal"
            self.canvas.set_exploration_target(None)
            agent = self.runtime_agent(None)
            if agent is not None:
                agent.exploration_target_xy = None
            return final_goal, self.last_goal_selection_reason

        belief = self.ensure_belief_map()
        agent = self.runtime_agent(None)
        current_target = agent.exploration_target_xy if agent is not None else self.current_exploration_target

        result = select_exploration_goal(
            planner_name,
            belief_map=belief,
            robot_xy=(float(start_xy[0]), float(start_xy[1])),
            robot_heading=float(getattr(self.robot, "theta", 0.0)) if self.robot is not None else 0.0,
            current_target=current_target,
            final_goal_xy=final_goal,
            robot_count=1,
            robot_radius=float(self.safety_radius()),
            sensor_range=float(self.config.vision),
            vision_model=str(self.config.vision_model),
            ipp_distance_penalty=float(self.config.ipp_distance_penalty),
            target_exclusion_radius=0.0,
        )

        if not result.success or result.target is None:
            self.current_exploration_target = None
            self.last_goal_selection_reason = str(result.reason)
            self.canvas.set_exploration_target(None)
            if agent is not None:
                agent.exploration_target_xy = None
            return None, self.last_goal_selection_reason

        target = (float(result.target[0]), float(result.target[1]))
        self.current_exploration_target = target
        self.last_goal_selection_reason = str(result.reason)
        self.canvas.set_exploration_target(target)
        if agent is not None:
            agent.set_exploration_target(target, reason=self.last_goal_selection_reason)
        return target, self.last_goal_selection_reason

    def force_robot_pose_free_in_belief(self, robot_index: int | None = None) -> bool:
        """Ensure the active robot center is FREE in the logical map.

        A live robot pose must never be rejected by A*/Dijkstra as an occupied
        start cell. This fixes false deadlocks caused by obstacle-point
        quantization or by an obstacle boundary sample landing on the robot's
        current grid cell. Ground-truth collision checks still remain active.
        """
        belief = self.ensure_belief_map()
        if robot_index is None:
            robot = getattr(self, "robot", None)
            idx = None
        else:
            idx = int(robot_index)
            if not (0 <= idx < len(getattr(self, "robots", []) or [])):
                return False
            robot = self.robots[idx]
        if robot is None:
            return False
        changed = belief.force_free_point(
            (float(robot.x), float(robot.y)),
            robot_index=idx,
            time_s=float(getattr(self, "simulation_time", 0.0)),
        )
        if changed:
            self.sync_legacy_map_views_from_belief()
        return bool(changed)

    def force_all_robot_poses_free_in_belief(self) -> int:
        """Force every live robot center to FREE and return changed cells."""
        changed = 0
        robots = list(getattr(self, "robots", []) or [])
        if robots:
            for index in range(len(robots)):
                changed += int(self.force_robot_pose_free_in_belief(index))
        elif getattr(self, "robot", None) is not None:
            changed += int(self.force_robot_pose_free_in_belief(None))
        return changed

    def sanitize_planner_obstacle_points(
        self,
        obstacle_points: list[tuple[float, float]],
        *,
        start_xy: tuple[float, float],
        robot_radius: float,
        resolution: float,
    ) -> tuple[list[tuple[float, float]], int]:
        """Remove obstacle samples that falsely occupy the robot's own start cell.

        The planner inflates obstacle points by the robot safety radius. If a
        quantized mapped-obstacle point falls on the current robot cell, A* can
        reject the route before it even starts. We clear only a small disk around
        the robot's current center for the planner input. This does not remove
        ground-truth obstacle checks or teammate dynamic obstacles outside the
        start cell.
        """
        sx, sy = float(start_xy[0]), float(start_xy[1])
        # Enough to clear the start cell and immediate quantization error, but
        # not enough to open corridors through real obstacles.
        clear_radius = max(float(resolution) * 1.25, min(float(robot_radius) * 0.75, float(resolution) * 2.5))
        clear_radius = max(clear_radius, 1e-6)
        kept: list[tuple[float, float]] = []
        removed = 0
        for point in obstacle_points:
            px, py = float(point[0]), float(point[1])
            if math.hypot(px - sx, py - sy) <= clear_radius:
                removed += 1
                continue
            kept.append((px, py))
        return kept, removed

    def build_planning_grid_for_robot(
        self,
        robot,
        *,
        obstacle_points: list[tuple[float, float]] | None = None,
        robot_radius: float | None = None,
    ):
        """Build a planning grid from BeliefMap plus dense mapped/dynamic samples.

        BeliefMap is the source of logical UNKNOWN/FREE/OCCUPIED state.
        Dense boundary samples remain useful for route safety and are added only
        to this derived planning grid, not to the belief map itself.
        """
        belief = self.ensure_belief_map()
        radius = self.safety_radius_for_robot(robot) if robot_radius is None else float(robot_radius)
        planning_grid = belief.to_planning_grid(
            unknown_is_traversable=True,
            inflate_radius=max(0.0, radius),
        )
        if obstacle_points:
            planning_grid.add_obstacle_points(obstacle_points, padding=max(0.0, radius))
        return planning_grid

    def planner_accepts_path_simplifier(self) -> bool:
        """Return True when the installed planner registry supports path_simplifier."""
        if compute_planned_waypoints is None:
            return False
        try:
            return "path_simplifier" in inspect.signature(compute_planned_waypoints).parameters
        except (TypeError, ValueError):
            return False

    def call_compute_planned_waypoints(
        self,
        planner_kwargs: dict,
        *,
        path_simplifier: str | None = None,
    ) -> tuple[bool, str, list[tuple[float, float]]]:
        """Call the planner without spamming TypeError fallback messages."""
        if bool(planner_kwargs.get("__hold__", False)):
            return False, str(planner_kwargs.get("__hold_reason__", "holding position")), []

        if compute_planned_waypoints is None:
            return False, "planner package is not available", []
        if path_simplifier is not None and self.planner_accepts_path_simplifier():
            return compute_planned_waypoints(**planner_kwargs, path_simplifier=path_simplifier)
        return compute_planned_waypoints(**planner_kwargs)


    def build_planner_kwargs(self, start_xy: tuple[float, float]) -> dict:
        """
        Build an immutable input packet for synchronous or asynchronous planning.
        """
        self.force_robot_pose_free_in_belief(None)
        goal_xy, goal_reason = self.select_navigation_goal(start_xy)
        self.last_goal_selection_reason = goal_reason

        resolution = float(self.config.grid_resolution)
        robot_radius = float(self.safety_radius())

        if goal_xy is None:
            return dict(
                __hold__=True,
                __hold_reason__=goal_reason,
                planner_type=self.config.planner_type,
                start_xy=(float(start_xy[0]), float(start_xy[1])),
                goal_xy=(float(start_xy[0]), float(start_xy[1])),
                planning_grid=None,
                bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
                resolution=resolution,
                robot_radius=robot_radius,
                obstacle_points=[],
            )

        obstacle_points, removed = self.sanitize_planner_obstacle_points(
            list(self.mapped_obstacle_points),
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            robot_radius=robot_radius,
            resolution=resolution,
        )

        if removed:
            self.last_goal_selection_reason = f"{goal_reason}; ignored {removed} own-start obstacle sample(s) for planning"

        planning_grid = self.build_planning_grid_for_robot(
            self.robot,
            obstacle_points=obstacle_points,
            robot_radius=robot_radius,
        )

        return dict(
            planner_type=self.config.planner_type,
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            obstacles=[],
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=resolution,
            robot_radius=robot_radius,
            planning_grid=planning_grid,
            unknown_is_traversable=True,
            obstacle_points=[],
        )

    def build_planner_kwargs_for_goal(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        *,
        robot=None,
    ) -> dict:
        """
        Build an immutable planning input packet for a *known* goal.

        Unlike build_planner_kwargs() this method does not call
        select_navigation_goal(); the caller already knows where to go (e.g.
        a prefetched frontier target chosen by ExplorationBehavior).
        """
        robot = robot if robot is not None else self.robot
        resolution = float(self.config.grid_resolution)
        robot_radius = self.safety_radius_for_robot(robot)

        obstacle_points, _ = self.sanitize_planner_obstacle_points(
            list(self.mapped_obstacle_points),
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            robot_radius=robot_radius,
            resolution=resolution,
        )

        planning_grid = self.build_planning_grid_for_robot(
            robot,
            obstacle_points=obstacle_points,
            robot_radius=robot_radius,
        )

        return dict(
            planner_type=self.config.planner_type,
            start_xy=(float(start_xy[0]), float(start_xy[1])),
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            obstacles=[],
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=resolution,
            robot_radius=robot_radius,
            planning_grid=planning_grid,
            unknown_is_traversable=True,
            obstacle_points=[],
        )

    def ensure_multi_exploration_target_slots(self) -> None:
        """Keep one exploration target and blacklist slot per runtime robot."""
        count = len(self.robots)
        if len(self.multi_exploration_targets) < count:
            self.multi_exploration_targets.extend([None] * (count - len(self.multi_exploration_targets)))
        elif len(self.multi_exploration_targets) > count:
            self.multi_exploration_targets = self.multi_exploration_targets[:count]

        if not hasattr(self, "multi_invalidated_exploration_targets"):
            self.multi_invalidated_exploration_targets = []
        if len(self.multi_invalidated_exploration_targets) < count:
            self.multi_invalidated_exploration_targets.extend([[] for _ in range(count - len(self.multi_invalidated_exploration_targets))])
        elif len(self.multi_invalidated_exploration_targets) > count:
            self.multi_invalidated_exploration_targets = self.multi_invalidated_exploration_targets[:count]

    def multi_reserved_exploration_targets(self, exclude_robot_index: int) -> list[tuple[float, float]]:
        """Return frontier targets already assigned to other robots.

        This is the minimal coordination layer: a robot may share the map with
        the team, but it must not share the same local frontier target F.
        """
        self.ensure_multi_exploration_target_slots()
        reserved: list[tuple[float, float]] = []
        for index, target in enumerate(self.multi_exploration_targets):
            if index == int(exclude_robot_index) or target is None:
                continue
            reserved.append((float(target[0]), float(target[1])))
        return reserved

    def multi_frontier_exclusion_radius(self) -> float:
        """Minimum distance between two reserved frontier targets.

        A radius of roughly two grid cells avoids duplicate or nearly duplicate
        F markers without being so large that robots starve in small maps.
        """
        return max(0.75, 2.0 * float(self.config.grid_resolution))

    def multi_dynamic_target_margin(self) -> float:
        """Extra clearance used when assigning frontiers around teammates."""
        return max(0.25, 0.5 * float(self.config.grid_resolution))

    def dynamic_robot_obstacles_for_target_selection(
        self,
        robot_index: int,
    ) -> list[tuple[float, float, float]]:
        """Return other runtime robots as dynamic disks for frontier selection."""
        robot_index = int(robot_index)
        disks: list[tuple[float, float, float]] = []
        for other_index, other in enumerate(self.robots):
            if other_index == robot_index:
                continue
            disks.append(
                (
                    float(other.x),
                    float(other.y),
                    float(self.safety_radius_for_robot(other)),
                )
            )
        return disks

    def target_is_clear_of_dynamic_robots(
        self,
        robot_index: int,
        target: tuple[float, float],
    ) -> tuple[bool, str]:
        """Validate that a proposed F_i is not inside a teammate safety zone."""
        if not (0 <= int(robot_index) < len(self.robots)):
            return False, "invalid robot index"

        robot = self.robots[int(robot_index)]
        ego_radius = float(self.safety_radius_for_robot(robot))
        margin = self.multi_dynamic_target_margin()

        for other_index, other in enumerate(self.robots):
            if other_index == int(robot_index):
                continue
            required = ego_radius + float(self.safety_radius_for_robot(other)) + margin
            distance = math.hypot(float(target[0]) - float(other.x), float(target[1]) - float(other.y))
            if distance <= required:
                return (
                    False,
                    f"target too close to R{other_index + 1} "
                    f"({distance:.2f} m < {required:.2f} m)",
                )

        return True, "target clear of dynamic robots"

    def target_is_clear_of_reserved_frontiers(
        self,
        robot_index: int,
        target: tuple[float, float],
    ) -> tuple[bool, str]:
        """Validate that F_i is not a near-duplicate of another reserved F_j."""
        radius = self.multi_frontier_exclusion_radius()
        for other_index, other_target in enumerate(self.multi_exploration_targets):
            if other_index == int(robot_index) or other_target is None:
                continue
            distance = math.hypot(float(target[0]) - float(other_target[0]), float(target[1]) - float(other_target[1]))
            if distance <= radius:
                return (
                    False,
                    f"target too close to F{other_index + 1} "
                    f"({distance:.2f} m < {radius:.2f} m)",
                )
        return True, "target clear of reserved frontiers"

    def target_is_clear_of_other_active_routes(
        self,
        robot_index: int,
        target: tuple[float, float],
    ) -> tuple[bool, str]:
        """Avoid assigning a frontier directly on a teammate's active path."""
        if not (0 <= int(robot_index) < len(self.robots)):
            return False, "invalid robot index"

        ego_radius = self.safety_radius_for_robot(self.robots[int(robot_index)])
        margin = self.multi_dynamic_target_margin()
        for other_index, other in enumerate(self.robots):
            if other_index == int(robot_index):
                continue
            route = self.current_route_points_for_robot(other)
            if len(route) < 2:
                continue
            required = ego_radius + self.safety_radius_for_robot(other) + margin
            for start, end in zip(route[:-1], route[1:]):
                distance = self.distance_point_to_segment(target, start, end)
                if distance <= required:
                    return (
                        False,
                        f"target too close to R{other_index + 1} active route "
                        f"({distance:.2f} m < {required:.2f} m)",
                    )
        return True, "target clear of active teammate routes"

    def multi_exploration_target_is_valid(
        self,
        robot_index: int,
        target: tuple[float, float],
    ) -> tuple[bool, str]:
        """Full validation for an already assigned or newly proposed F_i."""
        checks = (
            self.target_is_clear_of_reserved_frontiers,
            self.target_is_clear_of_dynamic_robots,
            self.target_is_clear_of_other_active_routes,
        )
        for check in checks:
            ok, reason = check(robot_index, target)
            if not ok:
                return False, reason
        return True, "target valid"

    def temporary_separation_target_for_robot(self, robot_index: int) -> tuple[float, float] | None:
        """Create a short-range separation target when robots start too close together."""
        if not (0 <= int(robot_index) < len(self.robots)):
            return None

        robot = self.robots[int(robot_index)]
        others = [other for idx, other in enumerate(self.robots) if idx != int(robot_index)]
        if not others:
            return None

        own_radius = float(self.safety_radius_for_robot(robot))
        max_other_radius = max((float(self.safety_radius_for_robot(other)) for other in others), default=own_radius)
        required_clearance = own_radius + max_other_radius + self.multi_dynamic_target_margin()
        nearest_distance = min(
            math.hypot(float(robot.x) - float(other.x), float(robot.y) - float(other.y))
            for other in others
        )
        if nearest_distance > max(required_clearance * 1.75, 1.0):
            return None

        centroid_x = sum(float(other.x) for other in others) / len(others)
        centroid_y = sum(float(other.y) for other in others) / len(others)
        dx = float(robot.x) - centroid_x
        dy = float(robot.y) - centroid_y
        if abs(dx) < 1e-8 and abs(dy) < 1e-8:
            dx, dy = 1.0, 0.0
        norm = math.hypot(dx, dy)
        if norm < 1e-8:
            dx, dy = 1.0, 0.0
            norm = 1.0

        step = max(required_clearance * 1.25, 0.75)
        target = (
            float(robot.x) + dx / norm * step,
            float(robot.y) + dy / norm * step,
        )
        target = (
            min(max(target[0], WORLD_X_MIN), WORLD_X_MAX),
            min(max(target[1], WORLD_Y_MIN), WORLD_Y_MAX),
        )
        return target

    def invalidated_frontiers_for_robot(self, robot_index: int) -> list[tuple[float, float]]:
        self.ensure_multi_exploration_target_slots()
        if not (0 <= int(robot_index) < len(self.multi_invalidated_exploration_targets)):
            return []
        return list(self.multi_invalidated_exploration_targets[int(robot_index)])

    def invalidate_current_multi_frontier(self, robot_index: int, reason: str = "") -> None:
        """Blacklist the current F_i for this robot and clear its assignment."""
        self.ensure_multi_exploration_target_slots()
        robot_index = int(robot_index)
        if not (0 <= robot_index < len(self.multi_exploration_targets)):
            return
        target = self.multi_exploration_targets[robot_index]
        if target is not None:
            invalid = self.multi_invalidated_exploration_targets[robot_index]
            target_tuple = (float(target[0]), float(target[1]))
            if all(math.hypot(target_tuple[0] - old[0], target_tuple[1] - old[1]) > 1e-6 for old in invalid):
                invalid.append(target_tuple)
            # Keep the blacklist bounded so a robot is not starved forever.
            if len(invalid) > 12:
                self.multi_invalidated_exploration_targets[robot_index] = invalid[-12:]
        self.multi_exploration_targets[robot_index] = None
        self.publish_multi_exploration_targets()

    def publish_multi_exploration_targets(self) -> None:
        if hasattr(self, "canvas"):
            self.canvas.set_multi_exploration_targets(self.multi_exploration_targets)

    def ensure_multi_route_state_slots(self) -> None:
        """Create per-robot route-state storage.

        This separates real navigation states from route-assignment messages.
        A robot that has no frontier must not keep asking A* for a fake
        one-cell route to its current position; it should be in an explicit
        HOLD/ STUCK state until a useful frontier or escape maneuver exists.
        """
        count = len(getattr(self, "robots", []))

        if not hasattr(self, "multi_route_states"):
            self.multi_route_states = []
        if len(self.multi_route_states) < count:
            self.multi_route_states.extend([self.ROUTE_STATE_ACTIVE] * (count - len(self.multi_route_states)))
        elif len(self.multi_route_states) > count:
            self.multi_route_states = self.multi_route_states[:count]

        if not hasattr(self, "multi_route_state_reasons"):
            self.multi_route_state_reasons = []
        if len(self.multi_route_state_reasons) < count:
            self.multi_route_state_reasons.extend([""] * (count - len(self.multi_route_state_reasons)))
        elif len(self.multi_route_state_reasons) > count:
            self.multi_route_state_reasons = self.multi_route_state_reasons[:count]

        if not hasattr(self, "multi_last_route_state_log_times"):
            self.multi_last_route_state_log_times = []
        if len(self.multi_last_route_state_log_times) < count:
            self.multi_last_route_state_log_times.extend([-1.0e9] * (count - len(self.multi_last_route_state_log_times)))
        elif len(self.multi_last_route_state_log_times) > count:
            self.multi_last_route_state_log_times = self.multi_last_route_state_log_times[:count]

    def set_multi_route_state(self, robot_index: int, state: str, reason: str = "", *, force_log: bool = False) -> None:
        """Set and log route-state transitions without spamming every frame."""
        self.ensure_multi_route_state_slots()
        robot_index = int(robot_index)
        if not (0 <= robot_index < len(self.multi_route_states)):
            return

        previous_state = self.multi_route_states[robot_index]
        previous_reason = self.multi_route_state_reasons[robot_index]
        reason = str(reason or "").strip()
        self.multi_route_states[robot_index] = str(state)
        self.multi_route_state_reasons[robot_index] = reason

        now = float(getattr(self, "simulation_time", 0.0))
        elapsed = now - float(self.multi_last_route_state_log_times[robot_index])
        changed = previous_state != state or previous_reason != reason
        if force_log or changed or elapsed >= 5.0:
            self.multi_last_route_state_log_times[robot_index] = now
            message = f"R{robot_index + 1} state={state}"
            if reason:
                message += f"; reason={reason}"
            self.log_console_message(message)

    def multi_goal_selection_is_hold(self, start_xy, goal_xy, reason: str) -> bool:
        """Detect a planner request that is really a hold/no-frontier state."""
        text = str(reason or "").lower()
        if (
            "no valid frontier" in text
            or "holding position" in text
            or "assigned frontier invalid" in text
        ):
            return True
        try:
            return math.hypot(float(goal_xy[0]) - float(start_xy[0]), float(goal_xy[1]) - float(start_xy[1])) <= max(1e-6, 0.10 * float(self.config.grid_resolution))
        except Exception:
            return False

    def dynamic_robot_obstacle_points_for_robot(
        self,
        robot_index: int,
        samples_per_robot: int = 16,
    ) -> list[tuple[float, float]]:
        """
        Approximate every *other* robot as a dynamic obstacle point cloud.

        The path planner already knows how to avoid mapped obstacle points by
        inflating them with the current robot radius. To make another robot
        behave like a disk obstacle with its own radius, we sample its safety
        boundary plus its center. When the current robot's radius is applied by
        the planner, this approximates the required pairwise clearance
        r_i + r_j.
        """
        if not self.robots:
            return []

        points: list[tuple[float, float]] = []
        robot_index = int(robot_index)
        samples = max(8, int(samples_per_robot))

        for other_index, other in enumerate(self.robots):
            if other_index == robot_index:
                continue

            cx = float(other.x)
            cy = float(other.y)
            radius = max(0.02, float(self.safety_radius_for_robot(other)))
            points.append((cx, cy))

            for k in range(samples):
                angle = 2.0 * math.pi * k / samples
                points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

        return points

    @staticmethod
    def distance_point_to_segment(
        point: tuple[float, float],
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> float:
        """Distance from a point to a finite 2D segment."""
        px, py = float(point[0]), float(point[1])
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / denom
        t = clamp(t, 0.0, 1.0)
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        return math.hypot(px - closest_x, py - closest_y)

    def segment_violates_other_robot_clearance(
        self,
        robot_index: int,
        start: tuple[float, float],
        end: tuple[float, float] | None,
    ) -> tuple[bool, str]:
        """
        Check whether a proposed local segment would pass through another robot.

        This is separate from the hard pairwise position check. It treats other
        robots as dynamic obstacles before the robot commits to the next control,
        so routes do not simply cross through another robot's body/safety zone.
        """
        if end is None or not (0 <= int(robot_index) < len(self.robots)):
            return False, ""

        robot_index = int(robot_index)
        ego = self.robots[robot_index]
        ego_radius = self.safety_radius_for_robot(ego)

        for other_index, other in enumerate(self.robots):
            if other_index == robot_index:
                continue
            other_xy = (float(other.x), float(other.y))
            required = ego_radius + self.safety_radius_for_robot(other)
            distance = self.distance_point_to_segment(other_xy, start, end)
            if distance <= required:
                return (
                    True,
                    f"ROBOT OBSTACLE: R{robot_index + 1} local segment crosses R{other_index + 1} "
                    f"safety zone ({distance:.2f} m < {required:.2f} m).",
                )
        return False, ""

    def coordinator_runtime_profile(self):
        """Return the selected coordinator plugin's runtime profile, memoized.

        This is read from the per-frame multi-robot loop (path/control source
        selection), so it is cached per coordinator_type instead of re-running
        plugin discovery every frame.
        """
        strategy = str(self.config.coordinator_type)
        if getattr(self, "_cached_runtime_profile_strategy", None) != strategy:
            try:
                self._cached_runtime_profile = runtime_profile_for_strategy(strategy)
            except PluginLoadError:
                self._cached_runtime_profile = build_runtime_profile(
                    PluginMetadata(name=strategy, version="", description="", capabilities=())
                )
            self._cached_runtime_profile_strategy = strategy
        return self._cached_runtime_profile

    def multi_robot_coordination_states(self) -> list[RobotCoordinationState]:
        """Return plain robot state packets for the coordinator."""
        states: list[RobotCoordinationState] = []
        for robot in self.robots:
            states.append(
                RobotCoordinationState(
                    xy=(float(robot.x), float(robot.y)),
                    safety_radius=float(self.safety_radius_for_robot(robot)),
                    sensor_range=float(getattr(robot, "vision", self.config.vision)),
                    vision_model=str(self.config.vision_model),
                    theta=float(robot.theta),
                )
            )
        return states

    def multi_active_route_points_by_robot(self) -> list[list[tuple[float, float]]]:
        """Return the current active route of every robot for coordination."""
        routes: list[list[tuple[float, float]]] = []
        for robot in self.robots:
            try:
                routes.append(self.current_route_points_for_robot(robot))
            except Exception:
                routes.append([(float(robot.x), float(robot.y))])
        return routes

    def synchronize_multi_frontier_targets(
        self,
        requesting_robot_index: int,
        force_new_target: bool = False,
    ) -> None:
        """Assign missing frontier targets using the selected coordinator.

        The map is shared by the team, but each robot must own an independent
        local target F_i. This method is the bridge between the engine and
        robotics_sim.simulation.coordination.MultiRobotCoordinator.
        """
        if self.is_goal_seeking_mode():
            return

        self.ensure_multi_exploration_target_slots()
        requesting_robot_index = int(requesting_robot_index)

        if force_new_target and 0 <= requesting_robot_index < len(self.multi_exploration_targets):
            self.multi_exploration_targets[requesting_robot_index] = None

        robots_to_assign: list[int] = []
        for index, target in enumerate(self.multi_exploration_targets):
            if target is None:
                robots_to_assign.append(index)

        if force_new_target and requesting_robot_index not in robots_to_assign:
            robots_to_assign.append(requesting_robot_index)

        if not robots_to_assign:
            return

        coordinator = MultiRobotCoordinator(self.config.coordinator_type)

        # Per-robot explored footprints are required by the coordinated frontier
        # planner to penalize duplicated sensing.  Passing only the shared map is
        # not enough: it makes teammate-overlap ratios collapse to zero because
        # the planner cannot distinguish who already observed each cell.
        explored_points_by_robot = [
            list(self.belief_map.robot_explored_points(index))
            for index in range(len(self.robots))
        ]

        result = coordinator.assign_frontiers(
            planner_name=str(self.config.exploration_planner),
            robot_states=self.multi_robot_coordination_states(),
            existing_targets=list(self.multi_exploration_targets),
            robots_to_assign=robots_to_assign,
            invalidated_targets_by_robot=list(self.multi_invalidated_exploration_targets),
            explored_points=list(self.explored_free_points),
            mapped_obstacle_points=list(self.mapped_obstacle_points),
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=float(self.config.grid_resolution),
            final_goal_xy=self.final_goal_xy(),
            ipp_distance_penalty=float(self.config.ipp_distance_penalty),
            target_exclusion_radius=self.multi_frontier_exclusion_radius(),
            dynamic_obstacle_margin=self.multi_dynamic_target_margin(),
            route_points_by_robot=self.multi_active_route_points_by_robot(),
            explored_points_by_robot=explored_points_by_robot,
            goal_tolerance=float(self.config.goal_tolerance),
        )

        if not hasattr(self, "multi_robot_commands_by_id"):
            self.multi_robot_commands_by_id = {}
        commands_by_id = map_robot_commands_by_id(result.commands)
        self.multi_robot_commands_by_id.update(commands_by_id)

        # Preference order: command.target (richer, plugin-authoritative) ->
        # result.targets[index] (plain legacy field) -> the target the robot
        # already had. The third tier matters because a plugin only returns an
        # entry for the robots it was asked to (re)assign this call;
        # result.targets is None for every other robot, and blindly assigning
        # list(result.targets) would wipe out targets that were not part of
        # this batch.
        previous_targets = list(self.multi_exploration_targets)
        updated_targets: list[tuple[float, float] | None] = []
        for index in range(len(result.targets)):
            command = commands_by_id.get(index)
            if command is not None and command.target is not None:
                updated_targets.append(command.target)
            elif result.targets[index] is not None:
                updated_targets.append(result.targets[index])
            elif index < len(previous_targets):
                updated_targets.append(previous_targets[index])
            else:
                updated_targets.append(None)
        self.multi_exploration_targets = updated_targets

        registry = self.ensure_runtime_robot_registry()
        registry.sync_exploration_targets_from_legacy_list(self.multi_exploration_targets)
        self.robot_agents = registry.agents
        if 0 <= requesting_robot_index < len(result.reasons):
            self.last_goal_selection_reason = (
                f"R{requesting_robot_index + 1}: {result.reasons[requesting_robot_index]} "
                f"[{result.strategy}]"
            )
        self.publish_multi_exploration_targets()

    def select_navigation_goal_for_multi_robot(
        self,
        robot_index: int,
        start_xy: tuple[float, float],
        force_new_target: bool = False,
    ) -> tuple[tuple[float, float], str]:
        """Select a navigation target for exactly one robot.

        Important multi-robot rule:
            each robot owns its own frontier target F.

        Replanning due to safety should usually keep the same F and only
        recompute the path to it. Selecting a new F should happen when the
        current F is reached or when the robot has no assigned frontier yet.
        """
        final_goal = self.final_goal_xy()
        planner_name = str(self.config.exploration_planner)
        robot_index = int(robot_index)
        self.ensure_multi_exploration_target_slots()

        if is_goal_seeking_planner(planner_name):
            if 0 <= robot_index < len(self.multi_exploration_targets):
                self.multi_exploration_targets[robot_index] = None
            self.publish_multi_exploration_targets()
            return final_goal, f"R{robot_index + 1}: using shared final mission goal"

        if force_new_target and 0 <= robot_index < len(self.multi_exploration_targets):
            self.multi_exploration_targets[robot_index] = None

        existing_target = None
        if 0 <= robot_index < len(self.multi_exploration_targets):
            existing_target = self.multi_exploration_targets[robot_index]

        if existing_target is not None and not force_new_target:
            target = (float(existing_target[0]), float(existing_target[1]))
            still_valid, validity_reason = self.multi_exploration_target_is_valid(robot_index, target)
            if still_valid:
                self.publish_multi_exploration_targets()
                return target, f"R{robot_index + 1}: keeping assigned frontier F{robot_index + 1}"

            # A teammate may have moved into this frontier or reserved a nearby
            # one. Clear only this robot's F_i; do not disturb the other robots.
            self.invalidate_current_multi_frontier(robot_index, validity_reason)

        self.synchronize_multi_frontier_targets(
            requesting_robot_index=robot_index,
            force_new_target=force_new_target,
        )

        target = None
        if 0 <= robot_index < len(self.multi_exploration_targets):
            target = self.multi_exploration_targets[robot_index]

        if target is None:
            recovery_target = self.temporary_separation_target_for_robot(robot_index)
            if recovery_target is not None:
                ok, reason = self.multi_exploration_target_is_valid(robot_index, recovery_target)
                if ok:
                    self.multi_exploration_targets[robot_index] = recovery_target
                    self.publish_multi_exploration_targets()
                    return recovery_target, (
                        f"R{robot_index + 1}: temporary separation target while waiting for frontier"
                    )

            # Do not fall back to G while an exploration planner is selected.
            # The robot should hold its current position until a unique frontier exists.
            return (float(start_xy[0]), float(start_xy[1])), (
                f"R{robot_index + 1}: no valid frontier assigned by "
                f"{self.config.coordinator_type}; holding position"
            )

        target = (float(target[0]), float(target[1]))
        target_valid, target_valid_reason = self.multi_exploration_target_is_valid(robot_index, target)
        if not target_valid:
            self.multi_exploration_targets[robot_index] = None
            self.publish_multi_exploration_targets()
            return (float(start_xy[0]), float(start_xy[1])), (
                f"R{robot_index + 1}: assigned frontier invalid after validation; "
                f"{target_valid_reason}"
            )

        return target, (
            f"R{robot_index + 1}: frontier assigned by "
            f"{self.config.coordinator_type}"
        )

    def build_planner_kwargs_for_multi_robot(
        self,
        robot_index: int,
        force_new_exploration_target: bool = False,
    ) -> tuple[dict, str]:
        """Build planner inputs for one robot, including other robots as obstacles."""
        robot_index = int(robot_index)
        robot = self.robots[robot_index]
        start_xy = (float(robot.x), float(robot.y))

        agent = self.runtime_agent(robot_index)
        if agent is not None:
            agent.set_position(start_xy)
            agent.set_heading(float(robot.theta))

        goal_xy, goal_reason = self.select_navigation_goal_for_multi_robot(
            robot_index,
            start_xy,
            force_new_target=force_new_exploration_target,
        )

        if self.is_exploration_mode() and self.multi_goal_selection_is_hold(start_xy, goal_xy, goal_reason):
            return dict(
                __hold__=True,
                __hold_reason__=goal_reason,
                planner_type=self.config.planner_type,
                start_xy=start_xy,
                goal_xy=start_xy,
                planning_grid=None,
                obstacles=[],
                bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
                resolution=float(self.config.grid_resolution),
                robot_radius=float(self.safety_radius_for_robot(robot)),
                obstacle_points=[],
            ), goal_reason

        self.force_robot_pose_free_in_belief(robot_index)
        dynamic_points = self.dynamic_robot_obstacle_points_for_robot(robot_index)
        resolution = float(self.config.grid_resolution)
        robot_radius = float(self.safety_radius_for_robot(robot))
        obstacle_points, removed = self.sanitize_planner_obstacle_points(
            list(self.mapped_obstacle_points) + dynamic_points,
            start_xy=start_xy,
            robot_radius=robot_radius,
            resolution=resolution,
        )

        if removed:
            goal_reason = f"{goal_reason}; ignored {removed} own-start obstacle sample(s) for planning"

        planning_grid = self.build_planning_grid_for_robot(
            robot,
            obstacle_points=obstacle_points,
            robot_radius=robot_radius,
        )

        kwargs = dict(
            planner_type=self.config.planner_type,
            start_xy=start_xy,
            goal_xy=(float(goal_xy[0]), float(goal_xy[1])),
            obstacles=[],
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=resolution,
            robot_radius=robot_radius,
            planning_grid=planning_grid,
            unknown_is_traversable=True,
            obstacle_points=[],
        )
        return kwargs, goal_reason

    def compute_route_for_multi_robot(
        self,
        robot_index: int,
        force_new_exploration_target: bool = False,
    ) -> tuple[bool, str, list[tuple[float, float]]]:
        """Compute one robot's route.

        If the selected coordinator plugin owns PATH_PLANNING and supplied a
        usable command.path for this robot, that path is authoritative and the
        external A*/Direct planner below is never invoked. Otherwise (this is
        the case for MMPF and NOIC legacy today, since neither declares
        PATH_PLANNING) the external planner runs exactly as before.
        """

        def _legacy_route() -> tuple[bool, str, list[tuple[float, float]]]:
            planner_kwargs, goal_reason = self.build_planner_kwargs_for_multi_robot(
                robot_index,
                force_new_exploration_target=force_new_exploration_target,
            )
            if bool(planner_kwargs.get("__hold__", False)):
                return False, goal_reason, []

            goal_xy = tuple(planner_kwargs["goal_xy"])

            if self.config.planner_type == "Direct":
                return True, f"direct route; {goal_reason}", [goal_xy]

            if compute_planned_waypoints is None:
                return False, "planner package is not available", []

            success, reason, waypoints = self.call_compute_planned_waypoints(
                planner_kwargs,
                path_simplifier=self.config.path_simplifier,
            )

            return success, f"{goal_reason}; {reason}", waypoints

        profile = self.coordinator_runtime_profile()
        command = getattr(self, "multi_robot_commands_by_id", {}).get(int(robot_index))
        success, reason, waypoints = select_runtime_path_source(profile, command, _legacy_route)
        if profile.owns_path_planning and "fallback" in reason:
            self.log_console_message(f"R{int(robot_index) + 1}: {reason}")
        return success, reason, waypoints

    def segment_clear_for_robot_against_points(
        self,
        robot,
        start: tuple[float, float],
        end: tuple[float, float],
        obstacle_points: list[tuple[float, float]],
    ) -> bool:
        if self.collision_checker is None:
            return True
        report = self.collision_checker.check_segment_points(
            start=(float(start[0]), float(start[1])),
            end=(float(end[0]), float(end[1])),
            obstacle_points=list(obstacle_points),
            robot_radius=float(self.safety_radius_for_robot(robot)),
        )
        return not bool(report.collision)

    def clean_waypoints_for_robot(
        self,
        robot,
        waypoints: list[tuple[float, float]],
        obstacle_points: list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, float]]:
        """Clean waypoints using a specific robot pose and obstacle context."""
        if robot is None or not waypoints:
            return [tuple(point) for point in waypoints]

        start = (float(robot.x), float(robot.y))
        raw_points = [tuple((float(point[0]), float(point[1]))) for point in waypoints]
        cleaned: list[tuple[float, float]] = []

        for point in raw_points:
            if math.hypot(point[0] - start[0], point[1] - start[1]) <= 1e-6:
                continue
            if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) <= 1e-6:
                continue
            cleaned.append(point)

        if not cleaned:
            return []

        if self.config.path_simplifier != "Line of sight grid-safe":
            return cleaned

        points_for_clearance = list(self.mapped_obstacle_points if obstacle_points is None else obstacle_points)
        simplified: list[tuple[float, float]] = []
        current = start
        index = 0

        while index < len(cleaned):
            farthest_visible = index
            for candidate_index in range(len(cleaned) - 1, index - 1, -1):
                candidate = cleaned[candidate_index]
                if self.segment_clear_for_robot_against_points(robot, current, candidate, points_for_clearance):
                    farthest_visible = candidate_index
                    break
            next_point = cleaned[farthest_visible]
            simplified.append(next_point)
            current = next_point
            index = farthest_visible + 1

        return simplified

    def compute_route(self, start_xy: tuple[float, float]) -> tuple[bool, str, list[tuple[float, float]]]:
        """
        Ask the selected planner for world-coordinate waypoints.

        This synchronous version is still used for initial startup and explicit
        goal changes. Replanning during motion uses PlannerWorker so expensive
        A*/Dijkstra calls do not freeze the GUI thread.
        """
        planner_kwargs = self.build_planner_kwargs(start_xy)
        if bool(planner_kwargs.get("__hold__", False)):
            return False, str(planner_kwargs.get("__hold_reason__", "holding position")), []
        goal_xy = tuple(planner_kwargs["goal_xy"])

        if self.config.planner_type == "Direct":
            return True, f"direct route; {self.last_goal_selection_reason}", [goal_xy]

        if compute_planned_waypoints is None:
            return False, "planner package is not available", []


        return self.call_compute_planned_waypoints(
            planner_kwargs,
            path_simplifier=self.config.path_simplifier,
        )

    def planner_label(self) -> str:
        exploration = self.config.exploration_planner
        if self.config.planner_type == "Direct":
            return f"Direct + {exploration}"
        return f"{self.config.planner_type} / {self.config.path_simplifier} + {exploration}"

    def segment_clear_against_current_map(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        """
        Return True when a continuous segment is safe with respect to the
        robot's current partial map.

        This is intentionally checked against mapped_obstacle_points, not the
        ground-truth rectangles. The robot should be allowed to plan through
        unknown space; if a hidden obstacle is discovered later, the safety
        layer will trigger replanning.
        """
        if self.collision_checker is None:
            return True

        report = self.collision_checker.check_segment_points(
            start=(float(start[0]), float(start[1])),
            end=(float(end[0]), float(end[1])),
            obstacle_points=list(self.mapped_obstacle_points),
            robot_radius=float(self.safety_radius()),
        )
        return not bool(report.collision)

    def clean_waypoints_for_current_start(
        self,
        waypoints: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """
        Clean planner output using the robot's actual continuous pose.

        Why this exists:
            A*/Dijkstra work on cell centers. Even after an aggressive
            simplifier, the first returned waypoint can be an artificial cell
            center beside the robot. Visually this looks like a useless target
            between S and F/G, and dynamically it can make the unicycle turn in
            a way that was not intended.

        Policy:
            - Always remove near-duplicate consecutive points.
            - In Line-of-sight mode, greedily collapse waypoints using the
              continuous partial map. If the real segment robot -> F/G is safe,
              the route becomes exactly [F/G].
            - Do not apply this aggressive cleanup to conservative modes; those
              modes should preserve the grid route topology for comparison.
        """
        if self.robot is None or not waypoints:
            return [tuple(point) for point in waypoints]

        start = (float(self.robot.x), float(self.robot.y))
        raw_points = [tuple((float(point[0]), float(point[1]))) for point in waypoints]

        # Remove consecutive duplicates and points already essentially reached
        # from the real robot pose.
        cleaned: list[tuple[float, float]] = []
        for point in raw_points:
            if math.hypot(point[0] - start[0], point[1] - start[1]) <= 1e-6:
                continue
            if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) <= 1e-6:
                continue
            cleaned.append(point)

        if not cleaned:
            return []

        # The most aggressive simplifier should be allowed to remove the
        # artificial cell-center waypoint produced by grid planning when the
        # continuous segment is actually safe.
        if self.config.path_simplifier != "Line of sight grid-safe":
            return cleaned

        simplified: list[tuple[float, float]] = []
        current = start
        index = 0

        while index < len(cleaned):
            farthest_visible = index

            for candidate_index in range(len(cleaned) - 1, index - 1, -1):
                candidate = cleaned[candidate_index]
                if self.segment_clear_against_current_map(current, candidate):
                    farthest_visible = candidate_index
                    break

            next_point = cleaned[farthest_visible]
            simplified.append(next_point)
            current = next_point
            index = farthest_visible + 1

        return simplified

    def apply_route_result(
        self,
        success: bool,
        reason: str,
        waypoints: list[tuple[float, float]],
    ) -> None:
        if self.robot is None:
            return

        self.route_result_count += 1

        if success and waypoints:
            clean_waypoints = self.clean_waypoints_for_current_start(waypoints)

            if not clean_waypoints:
                if self.is_exploration_mode():
                    clean_waypoints = [(float(self.robot.x), float(self.robot.y))]
                else:
                    clean_waypoints = [self.final_goal_xy()]

            # Belt-and-suspenders: if the planner returned the target the robot
            # just reached (hysteresis slipped through), refuse to reassign the
            # same route.  This prevents the infinite REQUEST_PLAN loop when
            # exploration hysteresis returns "kind=current" with length=0.
            if self.is_exploration_mode() and self.robot is not None and clean_waypoints:
                agent_check = self.runtime_agent(None)
                new_goal = clean_waypoints[-1]
                old_goal = getattr(agent_check, "active_path_goal_xy", None) if agent_check is not None else None
                robot_xy = (float(self.robot.x), float(self.robot.y))
                same_target_radius = max(
                    float(self.config.grid_resolution),
                    2.0 * float(self.config.goal_tolerance),
                )
                if (
                    old_goal is not None
                    and math.hypot(new_goal[0] - old_goal[0], new_goal[1] - old_goal[1]) <= same_target_radius
                    and math.hypot(robot_xy[0] - old_goal[0], robot_xy[1] - old_goal[1]) <= float(self.config.goal_tolerance) * 2.0
                ):
                    self.log_console_message(
                        f"[NAV] apply_route_result: planner returned already-reached target "
                        f"{new_goal}; forcing re-search."
                    )
                    if agent_check is not None:
                        agent_check.exploration_target_xy = None
                        agent_check.invalidate_route(reason="planner returned completed target; forcing re-search")
                    self.current_exploration_target = None
                    self.canvas.set_exploration_target(None)
                    return

            if hasattr(self.robot, "set_waypoints"):
                self.robot.set_waypoints(clean_waypoints)
            elif hasattr(self.robot, "set_goal"):
                self.robot.set_goal(clean_waypoints[-1])
            else:
                self.robot.goal = np.array(clean_waypoints[-1], dtype=float)

            # Sync RobotAgent so agent.active_target() is non-None next frame.
            # Without this, agent.step() keeps emitting REQUEST_PLAN because
            # agent.waypoints is never populated.
            agent = self.runtime_agent(None)
            if agent is not None and clean_waypoints:
                agent.assign_path(
                    target=clean_waypoints[-1],
                    waypoints=clean_waypoints,
                    planner_reason=reason,
                )

            self.canvas.set_planned_path([(self.robot.x, self.robot.y)] + clean_waypoints)
            if self.is_exploration_mode() and clean_waypoints:
                self.canvas.set_exploration_target(clean_waypoints[-1])
            self.canvas.set_status(
                f"Planner: {self.planner_label()}. {self.last_goal_selection_reason}. {reason}. "
                f"Mapped points: {len(self.mapped_obstacle_points)}."
            )
            self.log_route_assignment(
                None,
                (float(self.robot.x), float(self.robot.y)),
                clean_waypoints,
                f"{self.last_goal_selection_reason}; {reason}",
            )
            return

        if self.is_exploration_mode():
            hold_xy = (float(self.robot.x), float(self.robot.y))
            if hasattr(self.robot, "set_waypoints"):
                self.robot.set_waypoints([hold_xy])
            elif hasattr(self.robot, "set_goal"):
                self.robot.set_goal(hold_xy)
            else:
                self.robot.goal = np.array(hold_xy, dtype=float)

            # Keep agent in sync: no path, no stale goal, and no stale
            # exploration target -- otherwise desired_target_from_mode()
            # keeps returning the target that just failed to plan, and the
            # agent immediately re-requests a plan for it next tick.
            agent = self.runtime_agent(None)
            if agent is not None:
                agent.invalidate_failed_exploration_route(reason=f"planner failed: {reason}")

            self.current_exploration_target = None
            self.canvas.set_exploration_target(None)
            self.canvas.set_planned_path([hold_xy])
            self.canvas.set_status(
                f"Planner failed in exploration mode: {reason}. Holding current position; not falling back to G."
            )
            self.log_console_message(
                f"R1 holding position at {self._xy_text(hold_xy)}; planner failed in exploration mode: {reason}"
            )
            return

        # Goal-seeking failure: fall back to direct goal.
        goal_xy = self.final_goal_xy()
        if hasattr(self.robot, "set_goal"):
            self.robot.set_goal(goal_xy)
        else:
            self.robot.goal = np.array(goal_xy, dtype=float)

        # Sync agent to the fallback waypoint so it emits FOLLOW_PATH next
        # frame instead of re-requesting a plan and looping forever.
        agent = self.runtime_agent(None)
        if agent is not None:
            agent.assign_path(
                target=goal_xy,
                waypoints=[goal_xy],
                planner_reason=f"fallback direct: {reason}",
            )

        self.canvas.set_planned_path([(self.robot.x, self.robot.y), goal_xy])
        self.canvas.set_status(
            f"Planner failed: {reason}. Falling back to direct goal."
        )

    def assign_route_to_robot(self) -> None:
        if self.robot is None:
            return

        self.route_request_count += 1
        success, reason, waypoints = self.compute_route((self.robot.x, self.robot.y))
        self.apply_route_result(success, reason, waypoints)

    def request_route_async(self, reason: str) -> bool:
        """
        Start a background replan and keep the GUI responsive.
        """
        if self.robot is None:
            return False

        self.route_request_count += 1

        # Direct mode has no expensive global path planner, but exploration still
        # has to update F. Compute and apply the new exploration target
        # synchronously so "Planner = Direct" means "drive straight to the
        # selected frontier", not "freeze the first frontier forever".
        if self.config.planner_type == "Direct":
            success, route_reason, waypoints = self.compute_route((self.robot.x, self.robot.y))
            self.apply_route_result(success, f"{reason} {route_reason}", waypoints)
            return bool(success and waypoints)

        if compute_planned_waypoints is None:
            return False

        if self.planning_in_progress:
            return True

        self.route_request_id += 1
        request_id = self.route_request_id
        planner_kwargs = self.build_planner_kwargs((self.robot.x, self.robot.y))
        if bool(planner_kwargs.get("__hold__", False)):
            self.apply_route_result(False, str(planner_kwargs.get("__hold_reason__", "holding position")), [])
            return False

        worker = PlannerWorker(
            request_id=request_id,
            planner_kwargs=planner_kwargs,
            path_simplifier=self.config.path_simplifier,
        )
        worker.signals.route_ready.connect(self.on_async_route_ready)
        self.active_planner_workers[request_id] = worker

        self.planning_in_progress = True
        self.last_control = self.brake_control_for_collision()
        self.canvas.set_last_control(self.last_control)
        self.canvas.set_status(f"{reason} Planning in background...")
        self.thread_pool.start(worker)
        return True

    def on_async_route_ready(
        self,
        request_id: int,
        success: bool,
        reason: str,
        waypoints: list,
    ) -> None:
        self.active_planner_workers.pop(int(request_id), None)

        if request_id != self.route_request_id:
            return

        self.planning_in_progress = False
        clean_waypoints = [tuple(point) for point in waypoints]
        self.apply_route_result(success, reason, clean_waypoints)


    # ========================================================
    # LIVE METRICS
    # ========================================================

    def open_metrics_window(self) -> None:
        if self.metrics_window is None:
            self.metrics_window = SimulationMetricsWindow(self)
            # Open near the main window, but as an independent movable window.
            self.metrics_window.move(self.geometry().right() - 560, self.geometry().top() + 90)

        self.metrics_window.show()
        self.metrics_window.raise_()
        self.metrics_window.activateWindow()

    def open_console_window(self) -> None:
        if getattr(self, "console_window", None) is None:
            self.console_window = SimulationConsoleWindow(self)
            # Open below the metrics area, but keep it independent and movable.
            self.console_window.move(self.geometry().right() - 860, self.geometry().top() + 130)

        self.console_window.show()
        self.console_window.raise_()
        self.console_window.activateWindow()

    def get_console_lines(self) -> list[str]:
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "status_history_lines"):
            return canvas.status_history_lines()
        message = getattr(canvas, "status_message", "") if canvas is not None else ""
        return [str(message)] if message else []

    def clear_console_messages(self) -> None:
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "clear_status_history"):
            canvas.clear_status_history()

    def log_console_message(self, message: str, *, visible_status: bool = False) -> None:
        """Write a readable debugging message to the simulation console.

        visible_status=True also replaces the short status shown at the top of
        the canvas. Most detailed traces should keep visible_status=False so the
        canvas does not become noisy or truncated.
        """
        message = str(message).strip()
        if not message:
            return

        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return

        if visible_status and hasattr(canvas, "set_status"):
            canvas.set_status(message)
        elif hasattr(canvas, "append_console_message"):
            canvas.append_console_message(message)
        elif hasattr(canvas, "_append_status_history"):
            canvas._append_status_history(message)

    def _xy_text(self, point) -> str:
        if point is None:
            return "--"
        try:
            return f"({float(point[0]):.2f}, {float(point[1]):.2f})"
        except Exception:
            return str(point)

    def _control_text(self, control) -> str:
        try:
            arr = np.asarray(control, dtype=float).reshape(-1)
            if arr.size >= 2:
                return f"u=({arr[0]:.3f}, {arr[1]:.3f})"
            if arr.size == 1:
                return f"u=({arr[0]:.3f})"
        except Exception:
            pass
        return "u=--"

    def simulation_start_summary(self, *, multi: bool) -> str:
        """Return a multi-line, copyable summary of the exact run configuration."""
        cfg = self.config
        mode = "Multiple Robot Mode" if multi else "Single Robot Mode"
        try:
            profile = runtime_profile_for_strategy(cfg.coordinator_type)
        except PluginLoadError:
            profile = None

        if profile is not None and profile.owns_target_generation:
            exploration_lines = [
                f"Exploration source: {cfg.coordinator_type}",
                f"Legacy frontier service (fallback only): {cfg.exploration_planner}",
            ]
        else:
            exploration_lines = [f"Exploration planner: {cfg.exploration_planner}"]

        lines = [
            "=== Simulation started ===",
            f"Mode: {mode}",
            f"Planner: {cfg.planner_type}",
            f"Path simplifier: {cfg.path_simplifier}",
            *exploration_lines,
            f"Multi-robot coordinator: {cfg.coordinator_type}",
        ]
        if profile is not None:
            lines.append(
                "Algorithm runtime profile: "
                f"owns_target_generation={profile.owns_target_generation}, "
                f"owns_task_allocation={profile.owns_task_allocation}, "
                f"owns_path_planning={profile.owns_path_planning}, "
                f"owns_control={profile.owns_control}, "
                f"uses_legacy_frontier_service={profile.uses_legacy_frontier_service}, "
                f"uses_external_path_planner={profile.uses_external_path_planner}, "
                f"uses_external_motion_controller={profile.uses_external_motion_controller}"
            )
        lines += [
            f"Vision model: {cfg.vision_model}",
            f"Sensor range: {float(cfg.vision):.2f} m",
            f"Grid resolution: {float(cfg.grid_resolution):.2f} m/cell",
            f"Goal G: ({float(cfg.goal_x):.2f}, {float(cfg.goal_y):.2f})",
            f"Robot body radius: {float(cfg.body_radius):.2f} m",
            f"Safety radius r: {float(cfg.safety_radius):.2f} m",
            f"Max speed: {float(cfg.max_speed):.2f} m/s",
            f"Max acceleration: {float(cfg.max_acceleration):.2f} m/s²",
            f"Max angular speed: {float(cfg.max_angular_speed):.2f} rad/s",
            f"Goal tolerance: {float(cfg.goal_tolerance):.2f} m",
            f"IPP λ distance penalty: {float(cfg.ipp_distance_penalty):.2f}",
            f"Exploration replan cooldown: {float(cfg.exploration_replan_cooldown):.2f} s",
            f"Obstacles in scenario: {len(cfg.obstacles)}",
        ]

        if multi:
            lines.append(f"Robot count: {len(getattr(self, 'robots', []) or [])}")
            lines.append(f"Same robot configuration: {bool(getattr(cfg, 'same_robot_configuration', True))}")
            for index, robot in enumerate(getattr(self, "robots", []) or []):
                lines.append(
                    f"R{index + 1} start: pos=({float(robot.x):.2f}, {float(robot.y):.2f}), "
                    f"theta={float(robot.theta):.3f} rad, v={float(robot.v):.3f} m/s, "
                    f"vision={float(getattr(robot, '_sim_vision', cfg.vision)):.2f} m, "
                    f"r={float(self.safety_radius_for_robot(robot)):.2f} m"
                )
        else:
            lines.append(
                f"R1 start: pos=({float(cfg.x):.2f}, {float(cfg.y):.2f}), "
                f"theta={float(cfg.theta):.3f} rad, v={float(cfg.v):.3f} m/s"
            )

        return "\n".join(lines)

    def log_route_assignment(
        self,
        robot_index: int | None,
        start_xy: tuple[float, float],
        waypoints: list[tuple[float, float]],
        reason: str,
    ) -> None:
        label = f"R{int(robot_index) + 1}" if robot_index is not None else "R1"
        target = waypoints[-1] if waypoints else None
        self.log_console_message(
            f"{label} route assigned: start={self._xy_text(start_xy)}, "
            f"target={self._xy_text(target)}, waypoints={len(waypoints)}, "
            f"planner={self.config.planner_type}, exploration={self.config.exploration_planner}, "
            f"reason={reason}"
        )

    def log_robot_motion(
        self,
        robot,
        *,
        robot_index: int | None = None,
        control=None,
        target=None,
        force: bool = False,
    ) -> None:
        """Log throttled robot motion traces with coordinates and target."""
        if robot is None:
            return

        interval = 0.50
        now = float(getattr(self, "simulation_time", 0.0))
        if robot_index is None:
            last = float(getattr(self, "last_motion_log_time", -1.0e9))
            if (not force) and now - last < interval:
                return
            self.last_motion_log_time = now
            label = "R1"
        else:
            log_times = getattr(self, "multi_last_motion_log_times", None)
            if log_times is None:
                log_times = {}
                self.multi_last_motion_log_times = log_times
            last = float(log_times.get(int(robot_index), -1.0e9))
            if (not force) and now - last < interval:
                return
            log_times[int(robot_index)] = now
            label = f"R{int(robot_index) + 1}"

        if target is None:
            target = self.active_target_xy()

        self.log_console_message(
            f"{label} move @ t={now:.2f}s: pos=({float(robot.x):.2f}, {float(robot.y):.2f}), "
            f"theta={float(robot.theta):.3f} rad, v={float(robot.v):.3f} m/s, "
            f"target={self._xy_text(target)}, {self._control_text(control)}"
        )

    def latest_decision_message(self) -> str:
        status = ""
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            status = str(getattr(canvas, "status_message", "") or "").strip()
        reason = str(getattr(self, "last_goal_selection_reason", "") or "").strip()

        if reason and status and reason not in status:
            return f"{reason}\nStatus: {status}"
        return reason or status or "--"

    def estimated_explored_percent(self) -> float:
        return self.ensure_belief_map().stats().coverage_percent

    def point_inside_ground_truth_obstacle(self, point: tuple[float, float]) -> bool:
        """Return True if a world point is inside a scenario obstacle.

        This is used only for evaluation metrics, not for planning decisions.
        The planner still receives only the partial belief map.
        """
        x, y = point
        for obstacle in self.config.obstacles:
            ox, oy, width, height = map(float, obstacle)
            x0, x1 = sorted((ox, ox + width))
            y0, y1 = sorted((oy, oy + height))
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    def ground_truth_free_cell_count(self) -> int:
        """Count traversable cells in the full scenario for metrics only.

        The denominator for exploration quality should not be the whole
        rectangle, because obstacle interiors are not traversable. This metric
        deliberately uses ground truth only in the dashboard/evaluation layer.
        """
        belief = self.ensure_belief_map()
        count = 0
        for row in range(belief.height):
            for col in range(belief.width):
                if not self.point_inside_ground_truth_obstacle(belief.cell_to_world((row, col))):
                    count += 1
        return count

    def estimated_free_space_coverage_percent(self) -> float:
        belief = self.ensure_belief_map()
        free_cells = belief.stats().free_cells
        traversable_cells = self.ground_truth_free_cell_count()
        return 100.0 * free_cells / max(1, traversable_cells)

    def remaining_waypoint_count(self) -> int:
        if self.robot is None:
            return 0
        waypoint_manager = getattr(self.robot, "waypoints", None)
        raw_waypoints = getattr(waypoint_manager, "waypoints", None)
        current_index = getattr(waypoint_manager, "current_index", None)
        if raw_waypoints is not None and isinstance(current_index, int):
            return max(0, len(raw_waypoints) - int(current_index))
        return 1 if self.active_target_xy() is not None else 0

    def get_metrics_snapshot(self) -> list[tuple[str, str]]:
        robot_state = "None" if self.robot is None else mode_name(self.robot)
        robot_xy = "--"
        robot_theta = "--"
        robot_v = "--"
        target_xy = "--"
        distance_to_target = "--"
        distance_to_goal = "--"

        if self.robot is not None:
            robot_xy = f"({self.robot.x:.2f}, {self.robot.y:.2f})"
            robot_theta = f"{self.robot.theta:.3f} rad"
            robot_v = f"{self.robot.v:.3f} m/s"
            target = self.active_target_xy()
            if target is not None:
                target_xy = f"({target[0]:.2f}, {target[1]:.2f})"
                distance_to_target = f"{math.hypot(float(self.robot.x) - target[0], float(self.robot.y) - target[1]):.3f} m"
            gx, gy = self.final_goal_xy()
            distance_to_goal = f"{math.hypot(float(self.robot.x) - gx, float(self.robot.y) - gy):.3f} m"

        exploration_target = "--"
        if self.current_exploration_target is not None:
            exploration_target = f"({self.current_exploration_target[0]:.2f}, {self.current_exploration_target[1]:.2f})"

        belief = self.ensure_belief_map()
        stats = belief.stats()
        metrics = [
            ("Running", "Yes" if self.running and not self.paused else "No"),
            ("Robot state", robot_state),
            ("FPS", f"{self.canvas.fps:.1f}"),
            ("Simulation time", f"{self.simulation_time:.2f} s"),
            ("Simulation speed", f"{self.simulation_speed:.2f}x"),
            ("Robot position", robot_xy),
            ("Robot theta", robot_theta),
            ("Robot velocity", robot_v),
            ("Active target", target_xy),
            ("Exploration target F", exploration_target),
            ("Distance to active target", distance_to_target),
            ("Distance to final goal", distance_to_goal),
            ("Total distance traveled", f"{self.total_distance_traveled:.2f} m"),
            ("Path planner", self.config.planner_type),
            ("Path simplifier", self.config.path_simplifier),
            ("Exploration planner", self.config.exploration_planner),
            ("Multi-robot coordinator", self.config.coordinator_type),
            ("UI coordinator selection", self.coordinator_combo.currentText() if hasattr(self, "coordinator_combo") else "--"),
            ("Coordinator synced", "Yes" if (not hasattr(self, "coordinator_combo") or self.coordinator_combo.currentText() == self.config.coordinator_type) else "No"),
            ("IPP distance penalty λ", f"{self.config.ipp_distance_penalty:.2f}"),
            ("Planner requests", str(self.route_request_count)),
            ("Planner results applied", str(self.route_result_count)),
            ("Exploration replans", str(self.exploration_replan_count)),
            ("Safety replans", str(self.safety_replan_count)),
            ("Planning in background", "Yes" if self.planning_in_progress else "No"),
            ("Remaining waypoints", str(self.remaining_waypoint_count())),
            ("Belief FREE cells", str(stats.free_cells)),
            ("Belief OCCUPIED cells", str(stats.occupied_cells)),
            ("Belief UNKNOWN cells", str(stats.unknown_cells)),
            ("Belief known cells", str(stats.known_cells)),
            ("Belief coverage of rectangle", f"{stats.coverage_percent:.1f}%"),
            ("Free-space coverage", f"{self.estimated_free_space_coverage_percent():.1f}%"),
            ("Revisited free cells", str(stats.revisited_cells)),
            ("Revisit ratio", f"{100.0 * stats.revisit_ratio:.1f}%"),
            ("Avg visits per free cell", f"{stats.average_visits_per_free_cell:.2f}"),
            ("Multi-robot overlap cells", str(stats.overlap_cells)),
            ("Multi-robot overlap ratio", f"{100.0 * stats.overlap_ratio:.1f}%"),
        ]

        if getattr(self, "robots", None):
            per_robot_counts = belief.per_robot_explored_counts()
            per_robot_overlap = belief.per_robot_overlap_counts()
            for index, count in enumerate(per_robot_counts):
                metrics.append((f"R{index + 1} free cells", str(count)))
            if len(per_robot_counts) > 1:
                for index, count in enumerate(per_robot_overlap):
                    metrics.append((f"R{index + 1} overlap cells", str(count)))

        metrics.extend([
            ("Sensor updates", str(self.sensor_update_count)),
            ("Mapping updates", str(self.mapping_update_count)),
        ])
        return metrics

    # ========================================================
    # SIMULATION CONTROLS
    # ========================================================

    def update_start_pause_button(self) -> None:
        """
        Keep the main action button stateful.

        Start Simulation creates a new run only when there is no active run.
        During a run, the same button pauses/resumes. Restart is handled by the
        separate Restart button so Start no longer behaves like an accidental
        reset.
        """
        if not self.running:
            self.start_button.setText("Start Simulation")
            self.start_button.setIcon(make_icon("play", "white"))
        elif self.paused:
            self.start_button.setText("Resume Simulation")
            self.start_button.setIcon(make_icon("play", "white"))
        else:
            self.start_button.setText("Pause Simulation")
            self.start_button.setIcon(make_icon("pause", "white"))

    def handle_start_pause_button(self) -> None:
        has_runtime_robot = self.robot is not None or bool(getattr(self, "robots", []))
        if not self.running or not has_runtime_robot:
            self.start_simulation()
            return

        self.toggle_pause()

    def cycle_simulation_speed(self) -> None:
        self.simulation_speed_index = (
            self.simulation_speed_index + 1
        ) % len(self.simulation_speed_options)
        self.simulation_speed = self.simulation_speed_options[self.simulation_speed_index]
        self.speed_button.setText(f"Speed {self.simulation_speed:.2f}x")
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)
        self.canvas.set_status(f"Simulation speed set to {self.simulation_speed:.2f}x.")

    def restart_simulation(self) -> None:
        """
        Reset the run and leave the simulator stopped.

        This button returns the simulator to the configured initial state. It
        must not auto-start; the user explicitly presses Start Simulation when
        ready to run again.
        """
        self.reset_simulation()
        self.canvas.set_status("Restart complete. Press Start Simulation to run.")

    # ========================================================
    # SIMULATION
    # ========================================================

    def create_robot_instance(self, start_cfg: RobotStartConfig):
        """Create one Robot from a per-robot start configuration."""
        body_radius = max(0.01, float(start_cfg.body_radius))
        safety_radius = max(float(start_cfg.safety_radius), body_radius)
        initial_goal = (
            (float(start_cfg.x), float(start_cfg.y))
            if self.is_exploration_mode()
            else (float(self.config.goal_x), float(self.config.goal_y))
        )

        robot_kwargs = dict(
            x=float(start_cfg.x),
            y=float(start_cfg.y),
            theta=float(start_cfg.theta),
            v=float(start_cfg.v),
            vision=float(start_cfg.vision),
            goal=initial_goal,
            max_speed=float(start_cfg.max_speed),
            max_acceleration=float(start_cfg.max_acceleration),
            max_angular_speed=float(start_cfg.max_angular_speed),
            goal_tolerance=float(start_cfg.goal_tolerance),
            robot_radius=body_radius,
        )

        try:
            robot = Robot(**robot_kwargs)
        except TypeError:
            robot_kwargs.pop("robot_radius", None)
            robot = Robot(**robot_kwargs)

            limits = getattr(robot, "limits", None)
            if limits is not None and hasattr(limits, "robot_radius"):
                limits.robot_radius = body_radius

        # Store simulator-side radii/dynamics because the Robot class may not
        # expose a dedicated safety-radius field. Collision checking and
        # drawing read these attributes when present.
        robot._sim_body_radius = body_radius
        robot._sim_safety_radius = safety_radius
        robot._sim_acceleration_gain = float(start_cfg.acceleration_gain)
        robot._sim_goal_tolerance = float(start_cfg.goal_tolerance)

        self.apply_controller_parameters(robot, acceleration_gain=float(start_cfg.acceleration_gain))
        return robot

    def set_robot_goal_or_waypoints(self, robot, waypoints: list[tuple[float, float]]) -> None:
        if not waypoints:
            waypoints = [self.final_goal_xy()]

        # Always give each robot its own waypoint list. Reusing the same list
        # object across robots can make debugging look like a robot is following
        # another robot's route.
        robot_waypoints = [(float(point[0]), float(point[1])) for point in waypoints]

        if hasattr(robot, "set_waypoints"):
            robot.set_waypoints(robot_waypoints)
        elif hasattr(robot, "set_goal"):
            robot.set_goal(robot_waypoints[-1])
        else:
            robot.goal = np.array(robot_waypoints[-1], dtype=float)

    def current_route_points_for_robot(self, robot) -> list[tuple[float, float]]:
        """
        Return the remaining route assigned to a specific runtime robot.

        This is the multi-robot equivalent of current_route_points(). It lets
        the safety/replanning logic check whether newly mapped obstacle points
        actually invalidate that robot's active route.
        """
        if robot is None:
            return []

        points: list[tuple[float, float]] = [(float(robot.x), float(robot.y))]
        waypoint_manager = getattr(robot, "waypoints", None)
        raw_waypoints = getattr(waypoint_manager, "waypoints", None)
        current_index = getattr(waypoint_manager, "current_index", None)

        if raw_waypoints is not None and current_index is not None:
            for waypoint in raw_waypoints[int(current_index):]:
                waypoint_array = np.asarray(waypoint, dtype=float).reshape(-1)
                if waypoint_array.size >= 2:
                    points.append((float(waypoint_array[0]), float(waypoint_array[1])))
        else:
            goal = getattr(robot, "goal", None)
            if goal is not None:
                goal_array = np.asarray(goal, dtype=float).reshape(-1)
                if goal_array.size >= 2:
                    points.append((float(goal_array[0]), float(goal_array[1])))

        cleaned: list[tuple[float, float]] = []
        for point in points:
            if not cleaned or math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-6:
                cleaned.append(point)
        return cleaned

    def ensure_multi_replan_guard_slots(self) -> None:
        """Create per-robot cooldown state for repeated replanning triggers.

        Safety checks can fire every frame while a robot is stopped in front of
        a known obstacle. Without a guard, the simulator accepts the same route,
        rejects the same first segment, and logs hundreds of identical route
        assignments. This guard throttles identical replans while keeping the
        robot braked.
        """
        count = len(getattr(self, "robots", []))

        if not hasattr(self, "multi_last_safety_replan_sim_times"):
            self.multi_last_safety_replan_sim_times = []
        if len(self.multi_last_safety_replan_sim_times) < count:
            self.multi_last_safety_replan_sim_times.extend([-1.0e9] * (count - len(self.multi_last_safety_replan_sim_times)))
        elif len(self.multi_last_safety_replan_sim_times) > count:
            self.multi_last_safety_replan_sim_times = self.multi_last_safety_replan_sim_times[:count]

        if not hasattr(self, "multi_last_safety_replan_signatures"):
            self.multi_last_safety_replan_signatures = []
        if len(self.multi_last_safety_replan_signatures) < count:
            self.multi_last_safety_replan_signatures.extend([None] * (count - len(self.multi_last_safety_replan_signatures)))
        elif len(self.multi_last_safety_replan_signatures) > count:
            self.multi_last_safety_replan_signatures = self.multi_last_safety_replan_signatures[:count]

        if not hasattr(self, "multi_last_exploration_replan_sim_times"):
            self.multi_last_exploration_replan_sim_times = []
        if len(self.multi_last_exploration_replan_sim_times) < count:
            self.multi_last_exploration_replan_sim_times.extend([-1.0e9] * (count - len(self.multi_last_exploration_replan_sim_times)))
        elif len(self.multi_last_exploration_replan_sim_times) > count:
            self.multi_last_exploration_replan_sim_times = self.multi_last_exploration_replan_sim_times[:count]

    def multi_safety_replan_allowed(
        self,
        robot_index: int,
        reason: str,
        target: tuple[float, float] | None,
    ) -> bool:
        """Throttle identical safety replans for a robot.

        Returning False means: keep the robot stopped this frame and retry later,
        instead of logging the same rejected route again.
        """
        self.ensure_multi_replan_guard_slots()
        robot_index = int(robot_index)
        if not (0 <= robot_index < len(self.multi_last_safety_replan_sim_times)):
            return True

        cooldown = max(0.35, 0.75 * max(0.1, float(self.config.exploration_replan_cooldown)))
        target_key = None
        if target is not None:
            target_key = (round(float(target[0]), 2), round(float(target[1]), 2))
        signature = (str(reason), target_key)
        elapsed = float(self.simulation_time) - float(self.multi_last_safety_replan_sim_times[robot_index])
        same_signature = signature == self.multi_last_safety_replan_signatures[robot_index]
        if same_signature and elapsed < cooldown:
            return False

        self.multi_last_safety_replan_sim_times[robot_index] = float(self.simulation_time)
        self.multi_last_safety_replan_signatures[robot_index] = signature
        return True

    def multi_exploration_target_replan_allowed(self, robot_index: int) -> bool:
        """Per-robot cooldown for target-reached frontier replans."""
        self.ensure_multi_replan_guard_slots()
        robot_index = int(robot_index)
        if not (0 <= robot_index < len(self.multi_last_exploration_replan_sim_times)):
            return True
        cooldown = max(0.25, float(self.config.exploration_replan_cooldown))
        elapsed = float(self.simulation_time) - float(self.multi_last_exploration_replan_sim_times[robot_index])
        if elapsed < cooldown:
            return False
        self.multi_last_exploration_replan_sim_times[robot_index] = float(self.simulation_time)
        return True

    def route_points_intersect_new_map_information(
        self,
        route_points: list[tuple[float, float]],
        mapped_points: list[tuple[float, float]],
        robot_radius: float | None = None,
    ) -> bool:
        if self.collision_checker is None or len(route_points) < 2 or not mapped_points:
            return False

        robot_radius = self.safety_radius() if robot_radius is None else float(robot_radius)
        for start, end in zip(route_points[:-1], route_points[1:]):
            report = self.collision_checker.check_segment_points(
                start=start,
                end=end,
                obstacle_points=mapped_points,
                robot_radius=robot_radius,
            )
            if report.collision:
                return True
        return False

    def hold_multi_robot_position(
        self,
        robot_index: int,
        reason: str = "",
        *,
        state: str | None = None,
    ) -> bool:
        """Assign a zero-length hold target to one robot.

        This is critical in exploration mode: if no valid frontier exists or the
        path planner fails, the robot must *not* fall back to the shared final
        goal G. G is only a visual mission reference while a frontier planner is
        active.

        state lets a caller that already knows *why* it is holding (e.g. a
        corridor rejection, not a missing frontier) say so explicitly instead
        of relying on substring-sniffing the reason text below, which only
        covers the case where nothing more specific is known.
        """
        if not (0 <= int(robot_index) < len(self.robots)):
            return False

        robot_index = int(robot_index)
        robot = self.robots[robot_index]
        hold_xy = (float(robot.x), float(robot.y))

        self.set_robot_goal_or_waypoints(robot, [hold_xy])

        while len(self.multi_planned_path_points) <= robot_index:
            self.multi_planned_path_points.append([])
        self.multi_planned_path_points[robot_index] = [hold_xy]

        self.ensure_multi_exploration_target_slots()
        if 0 <= robot_index < len(self.multi_exploration_targets):
            self.multi_exploration_targets[robot_index] = None
        self.publish_multi_exploration_targets()

        if reason:
            self.last_goal_selection_reason = f"R{robot_index + 1}: holding position; {reason}"
        else:
            self.last_goal_selection_reason = f"R{robot_index + 1}: holding position"

        reason_text = str(reason or "")
        if state is not None:
            resolved_state = state
        else:
            resolved_state = self.ROUTE_STATE_HOLD_NO_FRONTIER
            if "collision" in reason_text.lower() or "blocked" in reason_text.lower() or "safety" in reason_text.lower():
                resolved_state = self.ROUTE_STATE_STUCK_SAFETY
        self.set_multi_route_state(robot_index, resolved_state, reason_text or "hold position")
        return True

    def assign_route_to_multi_robot(
        self,
        robot_index: int,
        reason: str = "",
        force_new_exploration_target: bool = False,
    ) -> bool:
        """
        Assign a route to one runtime robot using the shared planner selectors.

        Other robots are treated as dynamic obstacles during planning. In
        exploration modes, the shared final goal is ignored and each robot gets
        a frontier target instead.
        """
        return self._assign_route_to_multi_robot_with_corridor_validation(
            robot_index,
            reason=reason,
            force_new_exploration_target=force_new_exploration_target,
            remaining_corridor_retries=self.MAX_ROUTE_RECOVERY_ATTEMPTS - 1,
        )

    def compute_grid_safe_fallback_route_for_multi_robot(
        self,
        robot_index: int,
        force_new_exploration_target: bool = False,
    ) -> tuple[bool, str, list[tuple[float, float]]]:
        """One-off A* fallback used only when Direct's corridor is rejected.

        This does not change self.config.planner_type -- Direct stays the
        globally selected planner. It just asks the grid-safe planner for one
        alternate route to the same goal before the target itself is given
        up on, since a straight line can be blocked while a route around the
        obstruction is not.
        """
        if compute_planned_waypoints is None:
            return False, "planner package is not available", []

        planner_kwargs, goal_reason = self.build_planner_kwargs_for_multi_robot(
            robot_index,
            force_new_exploration_target=force_new_exploration_target,
        )
        if bool(planner_kwargs.get("__hold__", False)):
            return False, goal_reason, []

        fallback_kwargs = dict(planner_kwargs)
        fallback_kwargs["planner_type"] = "A*"

        success, reason, waypoints = self.call_compute_planned_waypoints(
            fallback_kwargs,
            path_simplifier=self.config.path_simplifier,
        )
        return bool(success), f"{goal_reason}; grid-safe fallback (A*): {reason}", waypoints

    def _activate_multi_robot_route(
        self,
        robot_index: int,
        robot,
        old_robot,
        waypoints: list[tuple[float, float]],
        route_reason: str,
        reason: str,
    ) -> bool:
        """Shared tail: commit a validated route as ACTIVE and restore self.robot."""
        self.set_robot_goal_or_waypoints(robot, waypoints)
        self.set_multi_route_state(robot_index, self.ROUTE_STATE_ACTIVE, route_reason)

        while len(self.multi_planned_path_points) <= robot_index:
            self.multi_planned_path_points.append([])
        self.multi_planned_path_points[robot_index] = [(float(robot.x), float(robot.y))] + list(waypoints)

        self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
        self.route_request_count += 1
        self.route_result_count += 1
        if reason:
            self.last_goal_selection_reason = f"R{robot_index + 1}: {reason}; {route_reason}"
        else:
            self.last_goal_selection_reason = route_reason
        self.log_route_assignment(
            robot_index,
            (float(robot.x), float(robot.y)),
            list(waypoints),
            self.last_goal_selection_reason,
        )
        self.publish_multi_exploration_targets()
        return True

    def _assign_route_to_multi_robot_with_corridor_validation(
        self,
        robot_index: int,
        *,
        reason: str,
        force_new_exploration_target: bool,
        remaining_corridor_retries: int,
    ) -> bool:
        if not (0 <= int(robot_index) < len(self.robots)):
            return False

        robot_index = int(robot_index)
        robot = self.robots[robot_index]
        old_robot = self.robot
        self.robot = robot

        success, route_reason, waypoints = self.compute_route_for_multi_robot(
            robot_index,
            force_new_exploration_target=force_new_exploration_target,
        )

        if (not success or not waypoints) and self.is_exploration_mode():
            held = self.hold_multi_robot_position(
                robot_index,
                f"no valid exploration route; {route_reason}",
            )
            self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
            return held

        if not success or not waypoints:
            waypoints = [self.final_goal_xy()]

        obstacle_points = list(self.mapped_obstacle_points) + self.dynamic_robot_obstacle_points_for_robot(robot_index)
        waypoints = self.clean_waypoints_for_robot(robot, waypoints, obstacle_points=obstacle_points)

        if not waypoints:
            if self.is_exploration_mode():
                held = self.hold_multi_robot_position(
                    robot_index,
                    f"target already reached or no safe frontier waypoint; {route_reason}",
                )
                self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
                return held
            waypoints = [self.final_goal_xy()]

        # Validate the FULL corridor against teammates before this route is
        # allowed to become ACTIVE. This runs earlier and stricter than the
        # per-frame movement safety veto (segment_violates_other_robot_clearance
        # in the movement loop), which only checks the immediate next segment
        # once a route is already active -- that veto remains as a final
        # backstop, it is just no longer the first place a conflict is caught.
        # Direct is included: a single-segment route is still a full corridor.
        if self.is_exploration_mode():
            corridor_check = validate_multi_robot_corridor(
                start=(float(robot.x), float(robot.y)),
                waypoints=waypoints,
                ego_safety_radius=float(self.safety_radius_for_robot(robot)),
                other_robot_disks=self.dynamic_robot_obstacles_for_target_selection(robot_index),
                other_routes=[
                    route
                    for j, route in enumerate(self.multi_active_route_points_by_robot())
                    if j != robot_index
                ],
                margin=self.multi_dynamic_target_margin(),
            )
            if not corridor_check.is_valid:
                self.log_console_message(
                    f"R{robot_index + 1}: route candidate rejected: reason={corridor_check.reason_code}; "
                    f"{corridor_check.detail}"
                )

                # A straight line can be blocked while a route around the
                # obstruction is not. Try the grid-safe planner once, for the
                # SAME target, before giving up on it -- this only applies
                # when Direct is the globally selected planner AND no plugin
                # owns PATH_PLANNING. A PATH_PLANNING-owning plugin's
                # command.path is authoritative (see compute_route_for_multi_
                # robot/select_runtime_path_source); this local A* fallback
                # must not silently override it just because the (now
                # disabled) planner combo still shows "Direct".
                if (
                    self.config.planner_type == "Direct"
                    and not self.coordinator_runtime_profile().owns_path_planning
                ):
                    self.log_console_message(
                        f"R{robot_index + 1}: Direct route rejected, trying A* fallback"
                    )
                    fb_success, fb_reason, fb_waypoints = self.compute_grid_safe_fallback_route_for_multi_robot(
                        robot_index,
                        force_new_exploration_target=False,
                    )
                    if fb_success and fb_waypoints:
                        fb_waypoints = self.clean_waypoints_for_robot(
                            robot, fb_waypoints, obstacle_points=obstacle_points
                        )
                    if fb_success and fb_waypoints:
                        fb_corridor_check = validate_multi_robot_corridor(
                            start=(float(robot.x), float(robot.y)),
                            waypoints=fb_waypoints,
                            ego_safety_radius=float(self.safety_radius_for_robot(robot)),
                            other_robot_disks=self.dynamic_robot_obstacles_for_target_selection(robot_index),
                            other_routes=[
                                route
                                for j, route in enumerate(self.multi_active_route_points_by_robot())
                                if j != robot_index
                            ],
                            margin=self.multi_dynamic_target_margin(),
                        )
                        if fb_corridor_check.is_valid:
                            _LOGGER.debug(
                                "R%d: route_accepted_after_corridor_validation (A* fallback)",
                                robot_index + 1,
                            )
                            return self._activate_multi_robot_route(
                                robot_index, robot, old_robot, fb_waypoints, fb_reason, reason
                            )

                self.invalidate_current_multi_frontier(robot_index, corridor_check.detail)
                self.log_console_message(
                    f"R{robot_index + 1}: target_blacklisted_after_route_rejection"
                )

                if remaining_corridor_retries > 0:
                    attempt_number = self.MAX_ROUTE_RECOVERY_ATTEMPTS - remaining_corridor_retries
                    self.log_console_message(
                        f"R{robot_index + 1}: route rejected, trying alternative target "
                        f"{attempt_number + 1}/{self.MAX_ROUTE_RECOVERY_ATTEMPTS}"
                    )
                    self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
                    retry_reason = f"retry after {corridor_check.reason_code}"
                    if reason:
                        retry_reason = f"{reason}; {retry_reason}"
                    return self._assign_route_to_multi_robot_with_corridor_validation(
                        robot_index,
                        reason=retry_reason,
                        force_new_exploration_target=True,
                        remaining_corridor_retries=remaining_corridor_retries - 1,
                    )

                # Candidates exhausted. A conflict with a teammate's active
                # route is transient (they are moving; the corridor may clear
                # on its own), so it waits rather than reporting a permanent
                # hold. Any other corridor conflict is reported as a blocked
                # route, not as "no frontier" -- a target/frontier did exist.
                if corridor_check.reason_code == "route_conflict_with_active_route":
                    self.log_console_message(
                        f"R{robot_index + 1}: waiting for corridor instead of HOLD_NO_FRONTIER"
                    )
                    held = self.hold_multi_robot_position(
                        robot_index,
                        f"waiting for corridor; {corridor_check.detail}",
                        state=self.ROUTE_STATE_WAITING_FOR_CORRIDOR,
                    )
                else:
                    self.log_console_message(
                        f"R{robot_index + 1}: HOLD_ROUTE_BLOCKED; candidates exhausted"
                    )
                    held = self.hold_multi_robot_position(
                        robot_index,
                        f"no safe corridor available after retry; {corridor_check.detail}",
                        state=self.ROUTE_STATE_HOLD_ROUTE_BLOCKED,
                    )
                self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)
                return held

            _LOGGER.debug("R%d: route_accepted_after_corridor_validation", robot_index + 1)

        return self._activate_multi_robot_route(robot_index, robot, old_robot, waypoints, route_reason, reason)

    def replan_multi_robots_affected_by_points(
        self,
        newly_mapped: list[tuple[float, float]],
        reason: str,
    ) -> int:
        """
        Replan only the robots whose current routes are invalidated by new map data.

        This runs regardless of planner_type: a Direct route is a straight
        line to a target, which can still be crossed by a newly discovered
        obstacle. assign_route_to_multi_robot() resolves Direct/A*/Dijkstra/
        plugin-owned paths uniformly, so there is nothing Direct-specific to
        special-case here.
        """
        if not newly_mapped:
            return 0

        replanned = 0
        for index, robot in enumerate(self.robots):
            route_points = self.current_route_points_for_robot(robot)
            if self.route_points_intersect_new_map_information(
                route_points,
                newly_mapped,
                robot_radius=self.safety_radius_for_robot(robot),
            ):
                if self.assign_route_to_multi_robot(index, reason=reason):
                    replanned += 1
        if replanned:
            self.safety_replan_count += replanned
        return replanned

    def start_multi_robot_simulation(self):
        """
        Start the executable multi-robot baseline.

        Current policy:
            - all robots share the global Planner / Path Simplifier selectors;
            - each robot receives its own route from its own current position;
            - the map is shared, but sensing/explored-area layers stay colored
              per robot;
            - if Same Configuration is OFF, only pose/initial-v overrides are
              per robot for now. Per-robot planner selection is intentionally a
              later experiment, because it would complicate comparisons.
        """
        self.spatial_index.rebuild(self.config.obstacles)
        self.planning_in_progress = False
        self.route_request_id += 1
        self.active_planner_workers.clear()
        getattr(self, "prefetch_workers", {}).clear()
        getattr(self, "prefetch_request_ids", {}).clear()

        self.ensure_multi_robot_configs()
        robot_starts = normalized_robot_start_configs(self.config)
        self.robots = [self.create_robot_instance(start_cfg) for start_cfg in robot_starts]
        self.robot = self.robots[0] if self.robots else None
        self.sync_runtime_robot_agents()

        # Reset shared mapping/metrics before the first routes are computed.
        # The previous version computed routes first and reset the map after,
        # which made multi-robot planning look like it ignored the selected
        # planner or used stale information from a previous run.
        self.known_obstacles = []
        self.explored_area_polygons = []
        self.reset_belief_map(robot_count=len(self.robots) if getattr(self, "robots", None) else 1)
        self.current_exploration_target = None
        self.multi_exploration_targets = []
        self.multi_invalidated_exploration_targets = []
        self.last_exploration_replan_sim_time = -1.0e9
        self.last_exploration_gate_message_time = -1.0e9
        self.last_goal_selection_reason = "multi-robot baseline using shared final goal"
        self.route_request_count = 0
        self.route_result_count = 0
        self.sensor_update_count = 0
        self.mapping_update_count = 0
        self.safety_replan_count = 0
        self.exploration_replan_count = 0
        self.total_distance_traveled = 0.0
        self.last_explored_pose = None
        self.multi_last_explored_poses = {}
        self.last_sensor_update_time = 0.0
        self.last_sensor_update_pose = None

        self.multi_path_points = [[(float(robot.x), float(robot.y))] for robot in self.robots]
        self.multi_robot_commands_by_id = {}
        self.multi_exploration_targets = [None for _ in self.robots]
        self.multi_invalidated_exploration_targets = [[] for _ in self.robots]
        self.multi_planned_path_points = [[] for _ in self.robots]
        self.multi_last_controls = [np.array([[0.0], [0.0]], dtype=float) for _ in self.robots]
        self.multi_route_states = [self.ROUTE_STATE_ACTIVE for _ in self.robots]
        self.multi_route_state_reasons = [""] * len(self.robots)
        self.multi_last_route_state_log_times = [-1.0e9] * len(self.robots)
        self.path_points = self.multi_path_points[0] if self.multi_path_points else []
        self.last_control = self.multi_last_controls[0] if self.multi_last_controls else np.array([[0.0], [0.0]], dtype=float)
        self.last_motion_log_time = -1.0e9
        self.multi_last_motion_log_times = {}
        self.simulation_time = 0.0
        self.last_time = time.perf_counter()

        self.log_console_message(self.simulation_start_summary(multi=True))

        self.canvas.set_mapped_obstacle_points(self.mapped_obstacle_points)
        self.canvas.set_explored_area_polygons(self.explored_area_polygons)
        self.canvas.set_known_obstacles(self.known_obstacles)
        self.canvas.set_planned_path([])
        self.canvas.set_exploration_target(None)
        self.canvas.set_multi_exploration_targets(self.multi_exploration_targets)
        self.canvas.invalidate_explored_area_cache()
        self.canvas.invalidate_sensor_cache()

        # Initialize the shared map from all robot sensors before assigning
        # routes. This lets frontier exploration and obstacle-aware A*/Dijkstra
        # start from the team observation instead of an empty or stale map.
        for robot_index, robot in enumerate(self.robots):
            old_robot = self.robot
            self.robot = robot
            self.record_explored_area(force=True, robot_index=robot_index)
            self.update_sensed_obstacles(force_status=False)
            self.force_robot_pose_free_in_belief(robot_index)
            self.robot = old_robot

        # Global planner applies to every robot. Each robot still gets its own
        # route because the start pose is different.
        for robot_index in range(len(self.robots)):
            self.assign_route_to_multi_robot(robot_index, reason="Initial multi-robot route")

        self.running = True
        self.paused = False
        self.set_configuration_locked(True)
        self.update_start_pause_button()
        self.speed_button.setText(f"Speed {self.simulation_speed:.2f}x")
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)
        self.canvas.set_multi_robots(
            self.robots,
            self.multi_path_points,
            self.multi_last_controls,
            planned_path_points=self.multi_planned_path_points,
            exploration_targets=self.multi_exploration_targets,
        )
        self.canvas.set_status(
            f"Multi-robot simulation running with {len(self.robots)} robots. "
            f"Planner shared: {self.config.planner_type}."
        )
        for robot_index, robot in enumerate(self.robots):
            target = (
                self.multi_exploration_targets[robot_index]
                if robot_index < len(self.multi_exploration_targets)
                else self.final_goal_xy()
            )
            self.log_robot_motion(
                robot,
                robot_index=robot_index,
                control=self.multi_last_controls[robot_index] if robot_index < len(self.multi_last_controls) else None,
                target=target,
                force=True,
            )
        self.top_bar.set_status("running")

    def start_simulation(self):
        self.config = self.read_config()
        if "Multiple" in self.config.agent_mode:
            self.start_multi_robot_simulation()
            return

        self.robots = []
        self.multi_path_points = []
        self.multi_planned_path_points = []
        self.multi_last_controls = []
        self.spatial_index.rebuild(self.config.obstacles)
        self.planning_in_progress = False
        self.route_request_id += 1
        self.active_planner_workers.clear()
        getattr(self, "prefetch_workers", {}).clear()
        getattr(self, "prefetch_request_ids", {}).clear()

        initial_goal = (
            (float(self.config.x), float(self.config.y))
            if self.is_exploration_mode()
            else (float(self.config.goal_x), float(self.config.goal_y))
        )

        robot_kwargs = dict(
            x=self.config.x,
            y=self.config.y,
            theta=self.config.theta,
            v=self.config.v,
            vision=self.config.vision,
            goal=initial_goal,
            max_speed=self.config.max_speed,
            max_acceleration=self.config.max_acceleration,
            max_angular_speed=self.config.max_angular_speed,
            goal_tolerance=self.config.goal_tolerance,
            robot_radius=self.config.body_radius,
        )

        try:
            self.robot = Robot(**robot_kwargs)
        except TypeError:
            robot_kwargs.pop("robot_radius", None)
            self.robot = Robot(**robot_kwargs)

            limits = getattr(self.robot, "limits", None)
            if limits is not None and hasattr(limits, "robot_radius"):
                limits.robot_radius = self.config.body_radius

        self.apply_controller_parameters()
        self.sync_runtime_robot_agents()

        self.known_obstacles = []
        self.explored_area_polygons = []
        self.reset_belief_map(robot_count=len(self.robots) if getattr(self, "robots", None) else 1)
        self.current_exploration_target = None
        self.multi_exploration_targets = []
        self.multi_invalidated_exploration_targets = []
        self.last_exploration_replan_sim_time = -1.0e9
        self.last_exploration_gate_message_time = -1.0e9
        self.last_goal_selection_reason = "using final mission goal"
        self.route_request_count = 0
        self.route_result_count = 0
        self.sensor_update_count = 0
        self.mapping_update_count = 0
        self.safety_replan_count = 0
        self.exploration_replan_count = 0
        self.total_distance_traveled = 0.0
        self.last_explored_pose = None
        self.last_motion_log_time = -1.0e9
        self.multi_last_motion_log_times = {}
        self.log_console_message(self.simulation_start_summary(multi=False))
        self.canvas.set_mapped_obstacle_points(self.mapped_obstacle_points)
        self.canvas.set_explored_area_polygons(self.explored_area_polygons)
        self.record_explored_area(force=True)
        self.update_sensed_obstacles(force_status=False)
        self.force_robot_pose_free_in_belief(None)
        self.assign_route_to_robot()

        self.running = True
        self.paused = False
        self.set_configuration_locked(True)

        self.path_points = [(self.robot.x, self.robot.y)]
        self.last_control = np.array([[0.0], [0.0]], dtype=float)
        self.simulation_time = 0.0
        self.last_time = time.perf_counter()
        self.last_sensor_update_time = 0.0
        self.last_sensor_update_pose = None

        self.update_start_pause_button()
        self.speed_button.setText(f"Speed {self.simulation_speed:.2f}x")
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)

        self.canvas.set_robot(self.robot)
        self.canvas.set_path(self.path_points)
        self.canvas.set_known_obstacles(self.known_obstacles)
        self.canvas.set_last_control(self.last_control)
        if self.config.planner_type == "Direct":
            self.canvas.set_status("Simulation running with direct route.")

        self.log_robot_motion(
            self.robot,
            robot_index=None,
            control=self.last_control,
            target=self.active_target_xy(),
            force=True,
        )
        self.top_bar.set_status("running")

    def reset_simulation(self):
        self.robot = None
        self.ensure_runtime_robot_registry().reset()
        self.robot_agents = self.ensure_runtime_robot_registry().agents
        self.running = False
        self.paused = False
        self.set_configuration_locked(False)

        self.collision_checker = CollisionChecker() if CollisionChecker is not None else None
        self.last_collision_report = None
        self.spatial_index.rebuild(self.config.obstacles)
        self.known_obstacles: list[tuple[float, float, float, float]] = []
        self.explored_area_polygons: list[list[tuple[float, float]]] = []
        self.reset_belief_map(robot_count=1)
        self.current_exploration_target: tuple[float, float] | None = None
        self.multi_exploration_targets = []
        self.multi_invalidated_exploration_targets = []
        self.last_exploration_replan_sim_time = -1.0e9
        self.last_exploration_gate_message_time = -1.0e9
        self.last_goal_selection_reason = "using final mission goal"
        self.route_request_count = 0
        self.route_result_count = 0
        self.sensor_update_count = 0
        self.mapping_update_count = 0
        self.safety_replan_count = 0
        self.exploration_replan_count = 0
        self.total_distance_traveled = 0.0
        self.last_explored_pose: tuple[float, float, float] | None = None
        self.last_sensor_update_time = 0.0
        self.last_sensor_update_pose = None
        self.last_motion_log_time = -1.0e9
        self.multi_last_motion_log_times = {}
        self.planning_in_progress = False
        self.route_request_id += 1
        self.active_planner_workers.clear()
        getattr(self, "prefetch_workers", {}).clear()
        getattr(self, "prefetch_request_ids", {}).clear()

        self.path_points = []
        self.robots = []
        self.multi_path_points = []
        self.multi_planned_path_points = []
        self.multi_last_controls = []
        self.last_control = np.array([[0.0], [0.0]], dtype=float)
        self.simulation_time = 0.0
        self.last_time = time.perf_counter()

        self.update_start_pause_button()
        self.canvas.set_simulation_metrics(self.simulation_time, self.simulation_speed)

        self.canvas.set_robot(None)
        self.canvas.set_multi_robots([], [], [], exploration_targets=[])
        self.canvas.set_path([])
        self.canvas.set_planned_path([])
        self.canvas.set_exploration_target(None)
        self.canvas.set_known_obstacles(self.known_obstacles)
        self.canvas.set_mapped_obstacle_points(self.mapped_obstacle_points)
        self.canvas.set_explored_area_polygons(self.explored_area_polygons)
        self.canvas.set_last_control(self.last_control)
        self.canvas.set_status("Reset complete. Press Start Simulation to run.")

        self.top_bar.set_status("ready")

    def toggle_pause(self):
        if not self.running:
            return

        self.paused = not self.paused

        if self.paused:
            self.canvas.set_status("Simulation paused.")
            self.top_bar.set_status("paused")
        else:
            self.last_time = time.perf_counter()
            self.canvas.set_status("Simulation running.")
            self.top_bar.set_status("running")

        self.update_start_pause_button()

    def set_goal_from_canvas(self, gx: float, gy: float):
        self.goal_x_input.setValue(gx)
        self.goal_y_input.setValue(gy)
        self.config = self.read_config()

        self.sync_runtime_robot_agents()
        registry = self.ensure_runtime_robot_registry()
        registry.set_final_goal_for_all(self.final_goal_xy())

        # In exploration modes the mission goal remains visible as a reference,
        # but it must not overwrite frontier targets.
        if self.is_exploration_mode():
            self.canvas.set_status(
                "Final goal updated visually. Exploration mode is active, so robots keep following frontiers."
            )
            return

        # Goal seeking is the only mode where G is executable. Changing G must
        # immediately invalidate old routes and assign fresh routes.
        if self.robots:
            for agent in getattr(self, "robot_agents", []) or []:
                agent.invalidate_route(reason="manual goal changed in Goal seeking")
            for robot_index in range(len(self.robots)):
                self.assign_route_to_multi_robot(robot_index, reason="Shared final goal updated")
            self.canvas.set_status("Goal updated and routes reassigned for all robots.")
            return

        if self.robot is not None:
            agent = self.runtime_agent(None)
            if agent is not None:
                agent.invalidate_route(reason="manual goal changed in Goal seeking")
            self.assign_route_to_robot()
            self.canvas.set_status("Goal updated by canvas click and route reassigned.")

    def body_radius_for_robot(self, robot=None) -> float:
        """Return physical body radius for a runtime robot or the global config."""
        target_robot = self.robot if robot is None else robot
        if target_robot is not None:
            if hasattr(target_robot, "_sim_body_radius"):
                return float(target_robot._sim_body_radius)
            limits = getattr(target_robot, "limits", None)
            if limits is not None and hasattr(limits, "robot_radius"):
                return float(limits.robot_radius)
            if hasattr(target_robot, "robot_radius"):
                return float(target_robot.robot_radius)
        return float(self.config.body_radius)

    def safety_radius_for_robot(self, robot=None) -> float:
        """Return clearance radius r for a runtime robot or the global config."""
        target_robot = self.robot if robot is None else robot
        body = self.body_radius_for_robot(target_robot)
        if target_robot is not None and hasattr(target_robot, "_sim_safety_radius"):
            return max(float(target_robot._sim_safety_radius), body)
        return max(float(self.config.safety_radius), body)

    def body_radius(self) -> float:
        """Backward-compatible alias for the current robot body radius."""
        return self.body_radius_for_robot(self.robot)

    def safety_radius(self) -> float:
        """Backward-compatible alias for the current robot clearance radius."""
        return self.safety_radius_for_robot(self.robot)

    def robot_radius(self) -> float:
        """
        Backward-compatible alias for the safety radius used by old calls.
        """
        return self.safety_radius()

    def apply_controller_parameters(self, robot=None, acceleration_gain: float | None = None) -> None:
        """
        Push GUI/per-robot controller parameters into the robot when the
        implementation exposes a modular TrackingController.
        """
        target_robot = self.robot if robot is None else robot
        if target_robot is None:
            return
        controller = getattr(target_robot, "controller", None)
        gain = float(self.config.acceleration_gain if acceleration_gain is None else acceleration_gain)
        if controller is not None and hasattr(controller, "acceleration_gain"):
            controller.acceleration_gain = gain

    def active_target_xy(self) -> tuple[float, float] | None:
        """
        Return the local target the robot is currently trying to reach.

        For the modular robot, this is active_waypoint(). For older robot
        versions, this falls back to robot.goal.
        """
        if self.robot is None:
            return None

        if hasattr(self.robot, "active_waypoint"):
            target = self.robot.active_waypoint()
            if target is not None:
                target_array = np.asarray(target, dtype=float).reshape(-1)
                return float(target_array[0]), float(target_array[1])

        goal = getattr(self.robot, "goal", None)
        if goal is not None:
            goal_array = np.asarray(goal, dtype=float).reshape(-1)
            if goal_array.size >= 2:
                return float(goal_array[0]), float(goal_array[1])

        return self.config.goal_x, self.config.goal_y

    def robot_snapshot(self):
        """
        Create a minimal dynamic snapshot for short-horizon collision prediction.
        """
        if self.robot is None or RobotSnapshot is None:
            return None

        return RobotSnapshot(
            x=float(self.robot.x),
            y=float(self.robot.y),
            theta=float(self.robot.theta),
            v=float(self.robot.v),
            max_speed=float(getattr(self.robot, "max_speed", self.config.max_speed)),
            max_acceleration=float(
                getattr(self.robot, "max_acceleration", self.config.max_acceleration)
            ),
            max_angular_speed=float(
                getattr(self.robot, "max_angular_speed", self.config.max_angular_speed)
            ),
        )

    def brake_control_for_collision(self) -> np.ndarray:
        """
        Return a braking control compatible with the robot interface.
        """
        if self.robot is not None and hasattr(self.robot, "brake_control"):
            return self.robot.brake_control()

        max_acceleration = float(
            getattr(self.robot, "max_acceleration", self.config.max_acceleration)
        )
        return np.array([[-max_acceleration], [0.0]], dtype=float)

    def stop_for_collision(self, message: str) -> None:
        """
        Stop the simulation after detecting an unsafe condition.

        The robot state is preserved so the canvas shows where the safety logic
        intervened.
        """
        self.running = False
        self.paused = False
        self.last_control = self.brake_control_for_collision()
        self.canvas.set_last_control(self.last_control)
        self.canvas.set_status(message)
        self.top_bar.set_status("paused")
        self.update_start_pause_button()

    def nominal_control_safe(self, blocked: bool = False) -> np.ndarray:
        """
        Call the robot nominal controller while supporting old and new APIs.
        """
        try:
            return self.robot.nominal_control(blocked=blocked)
        except TypeError:
            return self.robot.nominal_control()

    @staticmethod
    def distance_point_to_rect(point, obstacle) -> float:
        px, py = point
        ox, oy, ow, oh = obstacle
        closest_x = clamp(px, ox, ox + ow)
        closest_y = clamp(py, oy, oy + oh)
        return math.hypot(px - closest_x, py - closest_y)

    @staticmethod
    def sample_obstacle_boundary_points(
        obstacle: tuple[float, float, float, float],
        spacing: float,
    ) -> list[tuple[float, float]]:
        """
        Approximate a rectangular obstacle boundary with sparse points.

        The robot does not reveal the full rectangle when it senses it. It only
        adds visible boundary samples to its internal map, which creates a more
        realistic incremental mapping effect.
        """
        ox, oy, ow, oh = obstacle
        spacing = max(float(spacing), 0.015)
        points: list[tuple[float, float]] = []

        nx = max(1, int(math.ceil(ow / spacing)))
        ny = max(1, int(math.ceil(oh / spacing)))

        for i in range(nx + 1):
            x = ox + ow * i / nx
            points.append((x, oy))
            points.append((x, oy + oh))

        for j in range(1, ny):
            y = oy + oh * j / ny
            points.append((ox, y))
            points.append((ox + ow, y))

        return points

    @staticmethod
    def quantize_map_point(point: tuple[float, float], resolution: float) -> tuple[float, float]:
        """
        Quantize mapped points to avoid storing hundreds of near-duplicates.
        """
        # Keep points on the actual sampled boundary. Coarse grid quantization
        # made some mapped points look shifted relative to the rectangles.
        return (round(float(point[0]), 3), round(float(point[1]), 3))

    def visible_candidate_obstacles(self) -> list[tuple[float, float, float, float]]:
        """
        Return only obstacles that can affect the current sensor footprint.
        """
        if self.robot is None:
            return list(self.config.obstacles)

        return self.spatial_index.query_circle(
            origin=(float(self.robot.x), float(self.robot.y)),
            radius=float(getattr(self.robot, "vision", self.config.vision)),
            padding=max(self.safety_radius(), self.config.mapping_point_spacing),
        )

    def should_run_sensor_update(self, now: float) -> bool:
        """
        Throttle expensive sensor/mapping work.

        Robot dynamics still runs at the GUI timer rate. Sensor mapping runs
        around 10 Hz or sooner if the robot moves enough to reveal new geometry.
        """
        if self.robot is None:
            return False

        pose = (float(self.robot.x), float(self.robot.y), float(self.robot.theta))
        if self.last_sensor_update_pose is None:
            self.last_sensor_update_pose = pose
            self.last_sensor_update_time = float(now)
            return True

        last_x, last_y, last_theta = self.last_sensor_update_pose
        moved = math.hypot(pose[0] - last_x, pose[1] - last_y)
        rotated = abs(wrapped_angle_error(pose[2], last_theta))
        elapsed = float(now) - float(self.last_sensor_update_time)

        if (
            elapsed >= SENSOR_UPDATE_PERIOD_SEC
            or moved >= MIN_SENSOR_UPDATE_DISTANCE
            or rotated >= MIN_SENSOR_UPDATE_ROTATION
        ):
            self.last_sensor_update_pose = pose
            self.last_sensor_update_time = float(now)
            return True

        return False

    def point_visible_from_robot(
        self,
        point: tuple[float, float],
        candidate_obstacles: list[tuple[float, float, float, float]] | None = None,
    ) -> bool:
        """
        Return whether a boundary point is visible from the robot sensor.

        Visibility has three conditions:
            1. the point is inside sensor range;
            2. the point is inside the sensor angular model;
            3. no closer obstacle boundary occludes it.

        This prevents the map from being painted behind an obstacle. To map a
        full object, the robot must observe it from multiple sides.
        """
        if self.robot is None:
            return False

        rx = float(self.robot.x)
        ry = float(self.robot.y)
        px, py = point
        sensor_range = float(getattr(self.robot, "vision", self.config.vision))

        dx = float(px) - rx
        dy = float(py) - ry
        point_distance = math.hypot(dx, dy)

        if point_distance > sensor_range:
            return False

        if point_distance <= 1e-9:
            return False

        point_angle = math.atan2(dy, dx)

        if not angle_is_inside_sensor_model(
            angle=point_angle,
            robot_theta=float(self.robot.theta),
            vision_model=self.config.vision_model,
        ):
            return False

        first_hit = first_ray_hit_distance(
            origin=(rx, ry),
            angle=point_angle,
            obstacles=candidate_obstacles if candidate_obstacles is not None else self.visible_candidate_obstacles(),
            max_range=sensor_range,
        )

        # A boundary point is visible if it lies on the first surface hit by the
        # ray. If another obstacle is closer, the point is occluded.
        return point_distance <= first_hit + max(0.018, self.config.mapping_point_spacing * 0.70)

    @staticmethod
    def point_inside_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        """Return True when a world point is inside a polygon."""
        x, y = point
        inside = False
        n = len(polygon)
        if n < 3:
            return False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-12 else 1e-12) + xi
            )
            if intersects:
                inside = not inside
            j = i

        return inside

    def update_explored_free_points_from_polygon(
        self,
        polygon: list[tuple[float, float]],
        robot_index: int | None = None,
    ) -> None:
        """Backward-compatible wrapper around BeliefMap rasterization."""
        belief = self.ensure_belief_map()
        belief.mark_visible_polygon(
            polygon,
            robot_index=robot_index,
            time_s=float(getattr(self, "simulation_time", 0.0)),
        )
        self.explored_free_points = belief.explored_points()

    def record_explored_area(self, force: bool = False, robot_index: int | None = None) -> None:
        """
        Store the current occlusion-aware sensor footprint as explored area.

        The trace is a visualization of coverage, independent from Robot Orders.
        To keep the GUI responsive, a new polygon is recorded only after the
        robot moves or rotates enough to add visible information.
        """
        if self.robot is None:
            return

        x = float(self.robot.x)
        y = float(self.robot.y)
        theta = float(self.robot.theta)
        vision = float(self.robot.vision)

        last_pose = self.last_explored_pose if robot_index is None else self.multi_last_explored_poses.get(int(robot_index))
        if last_pose is not None and not force:
            last_x, last_y, last_theta = last_pose
            moved = math.hypot(x - last_x, y - last_y)
            rotated = abs(wrapped_angle_error(theta, last_theta))

            # For omnidirectional/LiDAR sensors, orientation does not change the
            # visible footprint much. For camera/FoV, rotation matters.
            min_move = max(0.12, min(0.35, vision * 0.06))
            min_turn = 0.18 if "Camera" in self.config.vision_model else math.inf

            if moved < min_move and rotated < min_turn:
                return

        polygon = sensor_visible_polygon_world(
            origin=(x, y),
            theta=theta,
            vision=vision,
            vision_model=self.config.vision_model,
            obstacles=self.visible_candidate_obstacles(),
            ray_count=EXPLORED_RAYS_CAMERA if "Camera" in self.config.vision_model else EXPLORED_RAYS_OMNI,
        )

        if len(polygon) < 3:
            return

        # Update the geometric explored map used by frontier planners. This is
        # independent from the optimized visual pixmap cache.
        self.update_explored_free_points_from_polygon(polygon, robot_index=robot_index)

        # The canvas stores the accumulated explored area in a pixmap. Keep
        # only a short Python-side history to avoid growing copy costs over
        # long runs. The visual explored layer remains complete.
        self.explored_area_polygons.append(polygon)
        if len(self.explored_area_polygons) > EXPLORED_POLYGON_HISTORY_LIMIT:
            self.explored_area_polygons = self.explored_area_polygons[-EXPLORED_POLYGON_HISTORY_LIMIT:]

        if robot_index is None:
            self.last_explored_pose = (x, y, theta)
            self.canvas.append_explored_area_polygon(polygon, robot_index=None)
        else:
            self.multi_last_explored_poses[int(robot_index)] = (x, y, theta)
            self.canvas.append_explored_area_polygon(polygon, robot_index=int(robot_index))

    def update_sensed_obstacles(self, force_status: bool = True) -> list[tuple[float, float]]:
        """Update the partial obstacle map with visible boundary samples.

        Two map representations are updated together but kept separate:

        1. ``mapped_obstacle_points`` stores dense boundary samples. These points
           are used for the pink/red visual trace and for local known-obstacle
           safety checks. They should remain dense enough to look like an edge.

        2. ``belief_map.grid`` stores OCCUPIED cells. This is used for logical
           coverage/frontier metrics. It may be coarse, so it must not replace
           the boundary samples.
        """
        newly_mapped: list[tuple[float, float]] = []
        belief = self.ensure_belief_map()

        if not hasattr(self, "mapped_obstacle_points"):
            self.mapped_obstacle_points = []
        if not hasattr(self, "mapped_obstacle_point_keys"):
            self.mapped_obstacle_point_keys = {
                (round(float(p[0]), 3), round(float(p[1]), 3))
                for p in self.mapped_obstacle_points
            }

        spacing = max(float(self.config.mapping_point_spacing), 0.015)
        quantization = spacing
        candidate_obstacles = self.visible_candidate_obstacles()

        for obstacle in candidate_obstacles:
            for point in self.sample_obstacle_boundary_points(tuple(obstacle), spacing):
                if not self.point_visible_from_robot(point, candidate_obstacles):
                    continue

                # Keep the sampled boundary location, only rounded to a stable
                # key. Do not collapse it to the belief cell center; doing so
                # destroys the visible line and weakens route safety checks.
                mapped_point = self.quantize_map_point(point, quantization)
                key = (round(float(mapped_point[0]), 3), round(float(mapped_point[1]), 3))
                if key in self.mapped_obstacle_point_keys:
                    continue

                self.mapped_obstacle_point_keys.add(key)
                self.mapped_obstacle_points.append(mapped_point)
                newly_mapped.append(mapped_point)

        if newly_mapped:
            changed_cells = belief.mark_occupied_points(
                newly_mapped,
                time_s=float(getattr(self, "simulation_time", 0.0)),
            )
            # A live robot center is always traversable for its own next plan.
            # This does not erase dense obstacle-boundary samples; it only fixes
            # the exact start cell in the logical grid.
            self.force_all_robot_poses_free_in_belief()
            self.sync_legacy_map_views_from_belief()
            self.canvas.append_mapped_obstacle_points(newly_mapped)
            if force_status:
                self.canvas.set_status(
                    f"Mapped {len(newly_mapped)} obstacle boundary sample(s); "
                    f"{changed_cells} occupied belief cell(s)."
                )

        return newly_mapped

    def current_route_points(self) -> list[tuple[float, float]]:
        """
        Return the remaining route currently assigned to the robot.

        The first point is always the robot's current position. The rest are the
        active waypoint and the future waypoints, when the modular robot exposes
        a WaypointManager. This route is used only to decide whether newly mapped
        obstacle points actually affect the current plan.
        """
        if self.robot is None:
            return []

        points: list[tuple[float, float]] = [(float(self.robot.x), float(self.robot.y))]

        waypoint_manager = getattr(self.robot, "waypoints", None)
        raw_waypoints = getattr(waypoint_manager, "waypoints", None)
        current_index = getattr(waypoint_manager, "current_index", None)

        if raw_waypoints is not None and current_index is not None:
            for waypoint in raw_waypoints[int(current_index):]:
                waypoint_array = np.asarray(waypoint, dtype=float).reshape(-1)
                if waypoint_array.size >= 2:
                    points.append((float(waypoint_array[0]), float(waypoint_array[1])))
        else:
            target = self.active_target_xy()
            if target is not None:
                points.append(target)

        # Remove near-duplicate consecutive points. They create zero-length
        # route segments that can look like false safety interventions.
        cleaned: list[tuple[float, float]] = []
        for point in points:
            if not cleaned or math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) > 1e-6:
                cleaned.append(point)

        return cleaned

    def route_intersects_mapped_points(
        self,
        route_points: list[tuple[float, float]],
        mapped_points: list[tuple[float, float]],
    ) -> bool:
        """
        Return whether mapped obstacle points invalidate the current route.

        A newly sensed point should not trigger replanning just because it exists.
        It should trigger replanning only if it violates the safety radius around
        the current route segments.
        """
        if self.collision_checker is None:
            return False

        if len(route_points) < 2 or not mapped_points:
            return False

        robot_radius = self.safety_radius()

        for start, end in zip(route_points[:-1], route_points[1:]):
            report = self.collision_checker.check_segment_points(
                start=start,
                end=end,
                obstacle_points=mapped_points,
                robot_radius=robot_radius,
            )
            if report.collision:
                return True

        return False

    def new_information_affects_current_route(
        self,
        newly_mapped: list[tuple[float, float]],
    ) -> bool:
        """
        Decide whether new sensor information requires replanning.

        Mapping and replanning are intentionally separated:
            - mapping updates the partial map whenever the sensor sees something;
            - replanning happens only when the new information threatens the
              route that the robot is currently executing.

        This prevents irrelevant discoveries, such as a wall behind or beside the
        robot, from changing a perfectly valid route.
        """
        route_points = self.current_route_points()
        return self.route_intersects_mapped_points(route_points, newly_mapped)

    def exploration_replan_allowed(self) -> tuple[bool, float]:
        """
        Gate frontier-target replans so exploration does not constantly destroy
        an aggressive path simplification result.

        This cooldown applies only to exploration target changes. Safety replans
        caused by a newly mapped obstacle or a predicted collision bypass this
        gate.
        """
        cooldown = max(0.0, float(self.config.exploration_replan_cooldown))
        elapsed = float(self.simulation_time) - float(self.last_exploration_replan_sim_time)
        remaining = max(0.0, cooldown - elapsed)
        return remaining <= 1e-9, remaining

    def request_exploration_route_async(self, reason: str) -> bool:
        """
        Request a new frontier target only when the exploration cooldown allows it.
        """
        allowed, remaining = self.exploration_replan_allowed()
        if not allowed:
            # Avoid spamming the status text every frame while the robot waits
            # at a reached local frontier target.
            if float(self.simulation_time) - float(self.last_exploration_gate_message_time) >= 0.50:
                self.canvas.set_status(
                    f"{reason} Waiting {remaining:.2f}s before next exploration replan."
                )
                self.last_exploration_gate_message_time = float(self.simulation_time)
            return False

        requested = self.request_route_async(reason)
        if requested:
            self.exploration_replan_count += 1
            self.last_exploration_replan_sim_time = float(self.simulation_time)
        return requested

    def replan_after_new_information(self, reason: str) -> bool:
        """
        Recompute the route using the robot's current partial map.

        The robot should not stop permanently when a local segment is blocked. It
        should update its map and ask the selected planner for a new route from
        its current state.
        """
        if self.robot is None:
            return False

        if self.config.planner_type == "Direct":
            return False

        self.safety_replan_count += 1
        return self.request_route_async(
            f"{reason} Replanning with {len(self.mapped_obstacle_points)} mapped boundary sample(s)."
        )

    def inter_robot_clearance_violation(self) -> tuple[bool, str]:
        """
        Check pairwise robot-robot safety clearance.

        Each robot is modeled as a disk with its own safety radius r. A violation
        occurs when the distance between centers is smaller than r_i + r_j. This
        is the first multi-robot safety layer; later we can replace the hard stop
        with CBF-based avoidance.
        """
        if len(self.robots) < 2:
            return False, ""

        for i in range(len(self.robots)):
            ri = self.robots[i]
            xi, yi = float(ri.x), float(ri.y)
            radius_i = self.safety_radius_for_robot(ri)
            for j in range(i + 1, len(self.robots)):
                rj = self.robots[j]
                xj, yj = float(rj.x), float(rj.y)
                radius_j = self.safety_radius_for_robot(rj)
                distance = math.hypot(xi - xj, yi - yj)
                minimum_distance = radius_i + radius_j
                if distance <= minimum_distance:
                    return (
                        True,
                        f"ROBOT-ROBOT COLLISION: R{i + 1} and R{j + 1} are too close "
                        f"({distance:.2f} m < {minimum_distance:.2f} m).",
                    )

        return False, ""

    def predicted_motion_report(
        self,
        *,
        control: np.ndarray,
        dt: float,
        robot_radius: float,
        known_obstacle_points: list[tuple[float, float]] | None = None,
        use_ground_truth: bool = True,
    ):
        """Check short-horizon motion before applying a control.

        Known mapped-obstacle points are checked first when the installed
        CollisionChecker supports point-cloud prediction. Ground-truth rectangles
        are also checked as a simulator integrity guard: the simulator should
        stop before a collision, not after the robot has already entered an
        obstacle safety region.
        """
        if self.collision_checker is None:
            return None
        snapshot = self.robot_snapshot()
        if snapshot is None:
            return None

        safe_dt = max(float(dt), 1e-3)
        steps = 10

        if known_obstacle_points and hasattr(self.collision_checker, "check_predicted_motion_points"):
            report = self.collision_checker.check_predicted_motion_points(
                snapshot=snapshot,
                control=control,
                dt=safe_dt,
                steps=steps,
                obstacle_points=known_obstacle_points,
                robot_radius=robot_radius,
            )
            if getattr(report, "collision", False):
                return report

        if use_ground_truth and hasattr(self.collision_checker, "check_predicted_motion"):
            report = self.collision_checker.check_predicted_motion(
                snapshot=snapshot,
                control=control,
                dt=safe_dt,
                steps=steps,
                obstacles=self.config.obstacles,
                robot_radius=robot_radius,
            )
            if getattr(report, "collision", False):
                return report

        return None

    def simulation_step_multi(self, real_dt: float) -> None:
        if not self.running or self.paused or not self.robots:
            return

        dt = min(real_dt, 0.05) * float(self.simulation_speed)
        self.simulation_time += dt

        if self.collision_checker is None:
            self.canvas.set_status("Collision checker unavailable.")
            return

        violation, message = self.inter_robot_clearance_violation()
        if violation:
            self.stop_for_collision(message)
            return

        run_sensor_update = self.should_run_sensor_update(time.perf_counter())
        if run_sensor_update:
            self.sensor_update_count += 1
            old_robot = self.robot
            newly_discovered_all: list[tuple[float, float]] = []

            for robot_index, robot in enumerate(self.robots):
                self.robot = robot
                self.record_explored_area(force=True, robot_index=robot_index)
                newly = self.update_sensed_obstacles(force_status=False)
                newly_discovered_all.extend(newly)

            self.robot = old_robot if old_robot in self.robots else (self.robots[0] if self.robots else None)

            if newly_discovered_all:
                self.mapping_update_count += 1
                replanned = self.replan_multi_robots_affected_by_points(
                    newly_discovered_all,
                    reason="New mapped obstacle affects robot route",
                )
                if replanned:
                    self.canvas.set_status(
                        f"Multi-robot mapping: {len(newly_discovered_all)} new boundary sample(s). "
                        f"Replanned {replanned} robot route(s)."
                    )
                else:
                    self.canvas.set_status(
                        f"Multi-robot mapping: {len(newly_discovered_all)} new obstacle boundary sample(s)."
                    )

        new_controls: list[np.ndarray] = []

        for index, robot in enumerate(self.robots):
            robot_position = (float(robot.x), float(robot.y))
            robot_radius = self.safety_radius_for_robot(robot)

            current_collision = self.collision_checker.check_position(
                position=robot_position,
                obstacles=self.config.obstacles,
                robot_radius=robot_radius,
            )
            if current_collision.collision:
                self.last_collision_report = current_collision
                self.robot = robot
                self.stop_for_collision(f"COLLISION: robot {index + 1} is inside an obstacle safety region.")
                return

            self.robot = robot
            target = self.active_target_xy()
            if target is not None:
                dynamic_points = self.dynamic_robot_obstacle_points_for_robot(index)
                active_segment_report = self.collision_checker.check_segment_points(
                    start=robot_position,
                    end=target,
                    obstacle_points=list(self.mapped_obstacle_points) + dynamic_points,
                    robot_radius=robot_radius,
                )
                robot_obstacle_violation, robot_obstacle_message = self.segment_violates_other_robot_clearance(
                    index,
                    robot_position,
                    target,
                )
                if active_segment_report.collision or robot_obstacle_violation:
                    block_reason = robot_obstacle_message if robot_obstacle_violation else "Active segment blocked by known obstacle"
                    self.set_multi_route_state(index, self.ROUTE_STATE_STUCK_SAFETY, block_reason)

                    # If the currently assigned frontier produces an unsafe first
                    # segment, do not keep re-planning to that same frontier.
                    # Blacklist it for this robot and request a different target.
                    if self.is_exploration_mode():
                        self.invalidate_current_multi_frontier(index, block_reason)

                    # Re-target regardless of planner_type: assign_route_to_multi_robot
                    # already resolves Direct/A*/Dijkstra/plugin-owned paths
                    # uniformly (see compute_route_for_multi_robot) and falls back
                    # to HOLD_ROUTE_BLOCKED/WAITING_FOR_CORRIDOR on its own when no
                    # safe route exists -- gating this on planner_type == "Direct"
                    # used to skip straight to stop_for_collision() (halting the
                    # WHOLE simulation) on the very first blocked segment whenever
                    # Direct was selected, since Direct is the default planner.
                    if self.multi_safety_replan_allowed(index, block_reason, target):
                        if self.assign_route_to_multi_robot(
                            index,
                            reason=block_reason,
                            force_new_exploration_target=True,
                        ):
                            control = self.brake_control_for_collision()
                            new_controls.append(control)
                            continue

                    # During the cooldown, stay stopped instead of logging the
                    # same rejected route every frame.
                    control = self.brake_control_for_collision()
                    new_controls.append(control)
                    continue

            # nominal_control_safe() also advances the robot's state machine
            # (active waypoint, ARRIVED/BLOCKED mode), so it always runs even
            # when a plugin owns CONTROL -- only the resulting control vector
            # may be replaced below. The safety veto further down (predicted
            # collision check) still runs on whatever control is used here, so
            # a CONTROL-owning plugin cannot bypass it.
            legacy_control = self.nominal_control_safe(blocked=False)
            control_profile = self.coordinator_runtime_profile()
            robot_command = getattr(self, "multi_robot_commands_by_id", {}).get(index)
            control, control_reason = select_runtime_control_source(
                control_profile, robot_command, legacy_control
            )
            if control_profile.owns_control:
                _LOGGER.debug("R%d control source: %s", index + 1, control_reason)
            control = np.asarray(control, dtype=float).reshape(np.asarray(legacy_control).shape)

            prediction_report = self.predicted_motion_report(
                control=control,
                dt=dt,
                robot_radius=robot_radius,
                known_obstacle_points=list(self.mapped_obstacle_points) + self.dynamic_robot_obstacle_points_for_robot(index),
                use_ground_truth=True,
            )
            if prediction_report is not None and getattr(prediction_report, "collision", False):
                self.last_collision_report = prediction_report
                block_reason = "Predicted collision before motion update"
                self.set_multi_route_state(index, self.ROUTE_STATE_STUCK_SAFETY, block_reason)
                if self.is_exploration_mode():
                    self.invalidate_current_multi_frontier(index, block_reason)

                # Re-target regardless of planner_type -- see the matching
                # comment on the active-segment block above for why gating
                # this on planner_type == "Direct" was wrong.
                if self.multi_safety_replan_allowed(index, block_reason, target):
                    if self.assign_route_to_multi_robot(
                        index,
                        reason=block_reason,
                        force_new_exploration_target=True,
                    ):
                        control = self.brake_control_for_collision()
                        new_controls.append(control)
                        continue

                control = self.brake_control_for_collision()
                new_controls.append(control)
                continue

            robot.update(control, dt)
            new_controls.append(control)
            self.log_robot_motion(
                robot,
                robot_index=index,
                control=control,
                target=target,
            )

            post_position = (float(robot.x), float(robot.y))
            post_collision = self.collision_checker.check_position(
                position=post_position,
                obstacles=self.config.obstacles,
                robot_radius=robot_radius,
            )
            if post_collision.collision:
                self.last_collision_report = post_collision
                self.stop_for_collision(f"COLLISION: robot {index + 1} entered an obstacle safety region after update.")
                return

            violation, message = self.inter_robot_clearance_violation()
            if violation:
                self.stop_for_collision(message)
                return

            # If frontier exploration is active, each robot can request a new
            # target after reaching its current one. This is intentionally simple
            # assignment for now; duplicate-frontier avoidance is the next layer.
            if self.is_exploration_mode():
                target = self.active_target_xy()
                tolerance = max(float(getattr(robot, "_sim_goal_tolerance", self.config.goal_tolerance)), 0.25)
                if target is not None and math.hypot(float(robot.x) - target[0], float(robot.y) - target[1]) <= tolerance:
                    if self.multi_exploration_target_replan_allowed(index):
                        if self.assign_route_to_multi_robot(
                            index,
                            reason="Exploration target reached",
                            force_new_exploration_target=True,
                        ):
                            self.exploration_replan_count += 1

            while len(self.multi_path_points) <= index:
                self.multi_path_points.append([])
            path = self.multi_path_points[index]
            new_path_point = (float(robot.x), float(robot.y))
            if path:
                self.total_distance_traveled += math.hypot(
                    new_path_point[0] - float(path[-1][0]),
                    new_path_point[1] - float(path[-1][1]),
                )
            path.append(new_path_point)
            if len(path) > 900:
                self.multi_path_points[index] = path[-900:]

        selected = max(0, min(int(self.selected_robot_index), len(self.robots) - 1))
        self.robot = self.robots[selected]
        self.path_points = self.multi_path_points[selected] if selected < len(self.multi_path_points) else []
        self.multi_last_controls = new_controls
        self.last_control = new_controls[selected] if selected < len(new_controls) else np.array([[0.0], [0.0]], dtype=float)

        self.canvas.set_multi_runtime_state(
            robots=self.robots,
            path_points=self.multi_path_points,
            planned_path_points=self.multi_planned_path_points,
            exploration_targets=self.multi_exploration_targets,
            last_controls=self.multi_last_controls,
            simulation_time=self.simulation_time,
            simulation_speed=self.simulation_speed,
        )

    def simulation_step(self):
        now = time.perf_counter()
        real_dt = now - self.last_time
        self.last_time = now
        real_dt = min(real_dt, 0.05)

        if self.running and self.robots:
            self.simulation_step_multi(real_dt)
            return

        if not self.running or self.paused or self.robot is None:
            return

        dt = real_dt * float(self.simulation_speed)
        self.simulation_time += dt

        if self.collision_checker is None:
            self.canvas.set_status("Collision checker unavailable.")
            return

        if self.planning_in_progress:
            # Keep the robot still while a new global route is being computed,
            # but do not block the GUI thread.
            self.last_control = self.brake_control_for_collision()
            self.canvas.set_runtime_state(
                robot=self.robot,
                path_points=self.path_points,
                last_control=self.last_control,
                simulation_time=self.simulation_time,
                simulation_speed=self.simulation_speed,
            )
            return

        run_sensor_update = self.should_run_sensor_update(now)
        if run_sensor_update:
            self.sensor_update_count += 1
            self.record_explored_area(force=False)
            newly_discovered = self.update_sensed_obstacles(force_status=False)
            if newly_discovered:
                self.mapping_update_count += 1
            if newly_discovered and self.config.planner_type != "Direct":
                if self.new_information_affects_current_route(newly_discovered):
                    self.replan_after_new_information("New obstacle affects current route.")
                    return

                self.canvas.set_status(
                    f"Mapped {len(newly_discovered)} new obstacle boundary sample(s). Current route unchanged."
                )

        robot_position = (float(self.robot.x), float(self.robot.y))
        robot_radius = self.safety_radius()
        target = self.active_target_xy()

        current_collision = self.collision_checker.check_position(
            position=robot_position,
            obstacles=self.config.obstacles,
            robot_radius=robot_radius,
        )

        if current_collision.collision:
            self.last_collision_report = current_collision
            self.stop_for_collision(
                "COLLISION: robot is inside an obstacle safety region."
            )
            return

        # ── Phase 2A: agent-based navigation decision ──────────────────────
        # The agent owns navigation state and policy; the engine is the executor.
        # If the agent layer is not yet available (first frame, registry not
        # initialised) we fall back to the legacy code path so the sim never
        # stalls.
        agent = self.runtime_agent(None)
        old_mode = mode_name(self.robot)

        if agent is not None and RobotObservation is not None:
            # ── New OOP flow ──────────────────────────────────────────────
            # build_observation pre-computes active_segment_blocked.
            obs = self.build_observation(self.robot, agent, None)

            # Compute nominal control first so predicted_motion_report() can
            # use it; pass the blocked flag so the controller can slow down.
            self.last_control = self.nominal_control_safe(
                blocked=obs.active_segment_blocked
            )

            predicted_report = self.predicted_motion_report(
                control=self.last_control,
                dt=dt,
                robot_radius=robot_radius,
                known_obstacle_points=list(self.mapped_obstacle_points),
                use_ground_truth=True,
            )
            if predicted_report is not None and getattr(predicted_report, "collision", False):
                self.last_collision_report = predicted_report
                obs.predicted_collision = True

            planner_services = self.ensure_planner_services()
            decision = agent.step(obs, planner_services, dt)
            should_brake = self.apply_navigation_decision(self.robot, agent, decision)

            if should_brake:
                self.last_control = self.brake_control_for_collision()
                self.canvas.set_runtime_state(
                    robot=self.robot,
                    path_points=self.path_points,
                    last_control=self.last_control,
                    simulation_time=self.simulation_time,
                    simulation_speed=self.simulation_speed,
                )
                return

            target = self.active_target_xy()
            self.robot.update(self.last_control, dt)
            self.log_robot_motion(
                self.robot,
                robot_index=None,
                control=self.last_control,
                target=target,
            )

        else:
            # ── Legacy fallback (agent layer unavailable) ─────────────────
            # Segment blocking is checked against the robot's current map, not
            # against omniscient ground truth. If blocked, it requests
            # replanning instead of treating a sharp turn as a terminal failure.
            local_path_report = self.collision_checker.check_segment_points(
                start=robot_position,
                end=target,
                obstacle_points=self.mapped_obstacle_points,
                robot_radius=robot_radius,
            )

            if local_path_report.collision:
                self.last_collision_report = local_path_report
                if self.replan_after_new_information("Active segment blocked by known obstacle."):
                    return

                self.last_control = self.nominal_control_safe(blocked=True)
                self.canvas.set_last_control(self.last_control)
                self.stop_for_collision(
                    "BLOCKED: direct segment intersects a known obstacle and replanning is unavailable."
                )
                return

            self.last_control = self.nominal_control_safe(blocked=False)

            predicted_report = self.predicted_motion_report(
                control=self.last_control,
                dt=dt,
                robot_radius=robot_radius,
                known_obstacle_points=list(self.mapped_obstacle_points),
                use_ground_truth=True,
            )
            if predicted_report is not None and getattr(predicted_report, "collision", False):
                self.last_collision_report = predicted_report
                if self.replan_after_new_information("Predicted collision before motion update."):
                    return

                self.stop_for_collision(
                    "PREDICTED COLLISION: control would enter an obstacle safety region before next update."
                )
                return

            self.robot.update(self.last_control, dt)
            self.log_robot_motion(
                self.robot,
                robot_index=None,
                control=self.last_control,
                target=target,
            )

            # In exploration mode, reaching a frontier target should select the
            # next frontier instead of leaving the robot permanently DONE.
            if self.is_exploration_mode():
                target = self.active_target_xy()
                if target is not None and math.hypot(
                    float(self.robot.x) - target[0], float(self.robot.y) - target[1]
                ) <= max(self.config.goal_tolerance, 0.25):
                    if self.request_exploration_route_async("Exploration target reached."):
                        return

        # ── Shared post-step checks ───────────────────────────────────────
        new_mode = mode_name(self.robot)
        if old_mode != new_mode:
            self.canvas.set_status(f"State transition: {old_mode} → {new_mode}")

        post_position = (float(self.robot.x), float(self.robot.y))
        post_collision = self.collision_checker.check_position(
            position=post_position,
            obstacles=self.config.obstacles,
            robot_radius=robot_radius,
        )

        if post_collision.collision:
            self.last_collision_report = post_collision
            self.stop_for_collision(
                "COLLISION: robot entered an obstacle safety region after update."
            )
            return

        new_path_point = (float(self.robot.x), float(self.robot.y))
        if self.path_points:
            self.total_distance_traveled += math.hypot(
                new_path_point[0] - float(self.path_points[-1][0]),
                new_path_point[1] - float(self.path_points[-1][1]),
            )
        self.path_points.append(new_path_point)

        if len(self.path_points) > 1200:
            self.path_points = self.path_points[-1200:]

        self.canvas.set_runtime_state(
            robot=self.robot,
            path_points=self.path_points,
            last_control=self.last_control,
            simulation_time=self.simulation_time,
            simulation_speed=self.simulation_speed,
        )

    # ========================================================
    # NEW POO INTERFACE — gradual migration helpers
    #
    # These methods wrap the new RobotObservation / NavigationDecision /
    # PlannerServices layer.  The existing simulation_step / simulation_step_multi
    # loops are untouched; call these from the new architecture incrementally.
    # ========================================================

    def ensure_planner_services(self):
        """Return the shared PlannerServices instance, creating it if needed."""
        if PlannerServices is None:
            return None
        if not hasattr(self, "_planner_services") or self._planner_services is None:
            self._planner_services = PlannerServices()
        return self._planner_services

    def build_observation(self, robot, agent, robot_index=None):
        """
        Build a RobotObservation snapshot for one robot.

        The engine pre-computes the two safety flags (active_segment_blocked,
        predicted_collision) so the agent's step() never touches engine internals.

        Parameters
        ----------
        robot:
            The live Robot physics object.
        agent:
            The RobotAgent for this robot.
        robot_index:
            Index in self.robots (None for single-robot mode).
        """
        if RobotObservation is None:
            return None

        robot_xy = (float(robot.x), float(robot.y))
        robot_radius = self.safety_radius_for_robot(robot)
        sensor_range = float(getattr(robot, "vision", self.config.vision))

        # Pre-compute active segment blocked flag.
        active_segment_blocked = False
        if self.collision_checker is not None:
            target = self.active_target_xy()
            if target is not None:
                dynamic_pts = (
                    self.dynamic_robot_obstacle_points_for_robot(int(robot_index))
                    if robot_index is not None and self.robots
                    else []
                )
                report = self.collision_checker.check_segment_points(
                    start=robot_xy,
                    end=target,
                    obstacle_points=list(self.mapped_obstacle_points) + dynamic_pts,
                    robot_radius=robot_radius,
                )
                active_segment_blocked = bool(report.collision)

        # Dynamic obstacles: other robots as (cx, cy, radius) disks.
        dynamic_obstacles: list[tuple[float, float, float]] = []
        if self.robots and robot_index is not None:
            for other_idx, other in enumerate(self.robots):
                if other_idx == int(robot_index):
                    continue
                dynamic_obstacles.append(
                    (float(other.x), float(other.y), self.safety_radius_for_robot(other))
                )

        # Excluded frontier targets: other robots' current frontiers.
        excluded: list[tuple[float, float]] = []
        if hasattr(self, "multi_exploration_targets") and self.multi_exploration_targets:
            for idx, t in enumerate(self.multi_exploration_targets):
                if idx != (robot_index if robot_index is not None else 0) and t is not None:
                    excluded.append((float(t[0]), float(t[1])))

        return RobotObservation(
            robot_xy=robot_xy,
            robot_heading=float(robot.theta),
            robot_radius=robot_radius,
            belief_map=self.ensure_belief_map(),
            planning_grid=None,  # built lazily by PlannerServices when needed
            mapped_obstacle_points=list(self.mapped_obstacle_points),
            dynamic_obstacles=dynamic_obstacles,
            active_segment_blocked=active_segment_blocked,
            predicted_collision=False,  # caller can set after nominal control
            current_time=float(self.simulation_time),
            grid_resolution=float(self.config.grid_resolution),
            goal_tolerance=float(
                getattr(robot, "_sim_goal_tolerance", self.config.goal_tolerance)
            ),
            sensor_range=sensor_range,
            final_goal_xy=self.final_goal_xy(),
            vision_model=str(self.config.vision_model),
            ipp_distance_penalty=float(self.config.ipp_distance_penalty),
            excluded_targets=excluded,
            route_points_by_robot=self.multi_active_route_points_by_robot()
            if self.robots
            else [],
        )

    def apply_navigation_decision(self, robot, agent, decision) -> bool:
        """
        Apply a NavigationDecision returned by agent.step() to robot and planner.

        Returns True when the engine should use brake control this frame.

        This method is the counterpart of build_observation().  Together they
        form the new "engine as executor" contract:

            observation = self.build_observation(robot, agent, idx)
            decision    = agent.step(observation, self.ensure_planner_services(), dt)
            should_brake = self.apply_navigation_decision(robot, agent, decision)

        Integration notes
        -----------------
        The existing simulation_step / simulation_step_multi loops are not yet
        replaced.  Wire these calls in incrementally; both paths can coexist.

        FOLLOW_PATH:
            Engine does nothing extra; robot follows its existing waypoints.

        BRAKE:
            Engine uses brake_control_for_collision() for this robot.

        HOLD:
            Engine sets robot target to current position.
            NEVER falls back to G while an exploration planner is active.

        REQUEST_PLAN:
            Engine asks the planner for a new route to decision.target.
            Uses async worker for non-Direct planners; brakes if decision.brake.

        PREFETCH_NEXT_TARGET:
            Engine stores decision.target in agent.pending_target_xy.
            TODO (next phase): kick off an async PlannerWorker for the prefetch
            and write the result into agent.pending_path when ready.

        ACCEPT_PENDING_PATH:
            Engine calls agent.accept_pending_path() and pushes the waypoints
            into the Robot object.

        REPLAN_FOR_SAFETY:
            Engine triggers a safety replan and brakes while computing.
        """
        kind = decision.kind

        if kind != "FOLLOW_PATH":
            self.log_console_message(
                f"[NAV] kind={kind} brake={decision.brake} reason={decision.reason!r} "
                f"active_target={getattr(agent, 'active_target', lambda: None)()} "
                f"path_goal={getattr(agent, 'active_path_goal_xy', None)} "
                f"pending_target={getattr(agent, 'pending_target_xy', None)}"
            )

        if kind == "FOLLOW_PATH":
            return False

        if kind == "BRAKE":
            return True

        if kind == "HOLD":
            hold_xy = (float(robot.x), float(robot.y))
            self.set_robot_goal_or_waypoints(robot, [hold_xy])
            agent.invalidate_route(reason=decision.reason or "hold")
            return False

        if kind == "ACCEPT_PENDING_PATH":
            waypoints = agent.accept_pending_path()
            if waypoints:
                self.set_robot_goal_or_waypoints(robot, waypoints)
                self.canvas.set_planned_path(
                    [(float(robot.x), float(robot.y))] + list(waypoints)
                )
                if self.is_exploration_mode():
                    self.canvas.set_exploration_target(waypoints[-1])
            return False

        if kind == "PREFETCH_NEXT_TARGET":
            agent.last_prefetch_time = float(self.simulation_time)
            self.request_prefetch_route_async(robot, agent, decision)
            return False

        if kind == "REQUEST_PLAN":
            # Route through the existing planner infrastructure.
            if self.robots:
                robot_index = next(
                    (i for i, r in enumerate(self.robots) if r is robot), 0
                )
                self.assign_route_to_multi_robot(
                    robot_index,
                    reason=decision.reason or "agent requested plan",
                    force_new_exploration_target=True,
                )
            else:
                if decision.force_new_target and agent is not None:
                    # The frontier was just reached.  Clear exploration_target_xy
                    # so select_navigation_goal() inside request_route_async()
                    # also sees current_target=None and cannot return it by hysteresis.
                    agent.exploration_target_xy = None
                self.request_route_async(decision.reason or "agent requested plan")
            return bool(decision.brake)

        if kind == "REPLAN_FOR_SAFETY":
            if self.robots:
                robot_index = next(
                    (i for i, r in enumerate(self.robots) if r is robot), 0
                )
                if self.multi_safety_replan_allowed(robot_index, decision.reason, decision.target):
                    self.assign_route_to_multi_robot(
                        robot_index,
                        reason=f"safety replan: {decision.reason}",
                        force_new_exploration_target=bool(self.is_exploration_mode()),
                    )
            else:
                self.replan_after_new_information(
                    f"safety replan: {decision.reason}"
                )
            return True  # always brake for safety replans

        return False

    def request_prefetch_route_async(
        self,
        robot,
        agent,
        decision,
        robot_index: int = 0,
    ) -> bool:
        """
        Launch a background planner for the *next* frontier without stopping
        the robot or touching planning_in_progress.

        The result lands in agent.pending_path via on_prefetch_route_ready().
        ExplorationBehavior decides when to promote it to the active path
        (ACCEPT_PENDING_PATH).
        """
        if robot is None or agent is None:
            return False

        target = (
            decision.target
            if decision.target is not None
            else agent.pending_target_xy
        )
        if target is None:
            return False

        idx = int(robot_index)

        # Avoid double-launching: if a worker is already running for this
        # robot, leave it alone.
        if idx in getattr(self, "prefetch_workers", {}):
            return False

        # "Direct" planner needs no A* — store the path immediately.
        if self.config.planner_type == "Direct":
            agent.pending_path = [target]
            agent.pending_target_xy = target
            agent.prefetch_success_count += 1
            self.log_console_message(f"[PREFETCH] direct route to target={target}")
            return True

        if compute_planned_waypoints is None:
            return False

        start_xy = (float(robot.x), float(robot.y))
        planner_kwargs = self.build_planner_kwargs_for_goal(
            start_xy, target, robot=robot
        )

        if not hasattr(self, "prefetch_request_counter"):
            self.prefetch_request_counter = 0
        self.prefetch_request_counter += 1
        request_id = self.prefetch_request_counter

        if not hasattr(self, "prefetch_request_ids"):
            self.prefetch_request_ids = {}
        self.prefetch_request_ids[idx] = request_id

        worker = PlannerWorker(
            request_id=request_id,
            planner_kwargs=planner_kwargs,
            path_simplifier=self.config.path_simplifier,
        )
        # Capture idx in the closure so stale callbacks go to the right robot.
        captured_idx = idx
        worker.signals.route_ready.connect(
            lambda rid, ok, rsn, wps: self.on_prefetch_route_ready(
                rid, captured_idx, ok, rsn, wps
            )
        )

        if not hasattr(self, "prefetch_workers"):
            self.prefetch_workers = {}
        self.prefetch_workers[idx] = worker

        # Store target now so agent.step() can track pending_target_xy.
        agent.pending_target_xy = target

        self.thread_pool.start(worker)
        self.log_console_message(f"[PREFETCH] requested target={target}")
        return True

    def on_prefetch_route_ready(
        self,
        request_id: int,
        robot_index: int,
        success: bool,
        reason: str,
        waypoints: list,
    ) -> None:
        """
        Callback fired when a prefetch PlannerWorker finishes.

        Never touches planning_in_progress, never brakes the robot, and never
        clears the current active path.  The agent decides when to switch via
        ACCEPT_PENDING_PATH.
        """
        idx = int(robot_index)

        if not hasattr(self, "prefetch_workers"):
            return
        self.prefetch_workers.pop(idx, None)

        # Stale result: a newer prefetch was started for this robot slot.
        stored_id = getattr(self, "prefetch_request_ids", {}).get(idx)
        if stored_id != int(request_id):
            return

        agent = self.runtime_agent(None if robot_index == 0 else robot_index)
        if agent is None:
            return

        if success and waypoints:
            clean_waypoints = [(float(p[0]), float(p[1])) for p in waypoints]
            agent.pending_path = clean_waypoints
            # pending_target_xy was set when the worker launched; keep it.
            agent.prefetch_success_count += 1
            self.log_console_message(
                f"[PREFETCH] success waypoints={len(clean_waypoints)}"
            )
        else:
            agent.reject_pending_path(reason)
            self.log_console_message(
                f"[PREFETCH] failed; keeping current route — {reason}"
            )

    # ========================================================

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.handle_start_pause_button()
        elif event.key() == Qt.Key_R:
            self.restart_simulation()

    # ========================================================
