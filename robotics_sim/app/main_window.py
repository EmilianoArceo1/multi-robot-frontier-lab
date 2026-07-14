"""
Main application window.

This file builds the interface and connects UI events to SimulationControllerMixin.
The heavy behavior lives in robotics_sim.simulation.engine; the canvas drawing
lives in robotics_sim.app.simulation_canvas.
"""

from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize, QThreadPool
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from robotics_sim.simulation.config import *
from robotics_sim.diagnostics.event_log import NavigationDebugEventLog
from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.simulation.engine import PlannerWorker, SimulationControllerMixin
from robotics_sim.app.widgets import (
    HeroHeader,
    NumericStepper,
    SectionCard,
    SliderValueRow,
    ToggleSwitch,
    TopBar,
    make_icon,
)
from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.app.config_panel import build_config_panel as build_right_config_panel
from robotics_sim.app.map_editor import (
    MIN_EDITOR_OBSTACLE_SIZE,
    create_free_draw_obstacles_from_path,
    create_rect_obstacle_from_drag,
    merge_obstacles,
    move_obstacle_to,
    move_obstacles_by,
    normalize_obstacles,
    remove_obstacle_at,
)
from robotics_sim.simulation.coordination import runtime_profile_for_strategy
from robotics_sim.simulation.gui_policy import compute_gui_control_policy
from robotics_sim.simulation.navigation_modes import is_exploration_planner
from robotics_sim.simulation.plugin_loader import PluginLoadError
from robotics_sim.simulation.runtime_robot_registry import RuntimeRobotRegistry

try:
    from robotics_sim.environment.collision_checker import CollisionChecker
except ImportError:
    CollisionChecker = None

class MainWindow(SimulationControllerMixin, QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Robotics Simulation Lab")

        # Quita la barra negra nativa del sistema.
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

        self.config = SimulationConfig()
        self.multi_robot_configs: list[RobotStartConfig] = normalized_robot_start_configs(self.config)
        self.selected_robot_index = 0

        self.robot = None
        self.runtime_robot_registry = RuntimeRobotRegistry()
        self.robot_agents = self.runtime_robot_registry.agents
        self.running = False
        self.paused = False

        # Navigation debug overlay: off by default, persists across
        # simulation resets (a user preference, not run state) -- mirrors
        # canvas.grid_overlay_enabled's lifecycle. The bounded event log
        # itself IS reset per simulation run (see reset_simulation_state()),
        # since events from a previous run are no longer meaningful.
        self.navigation_debug_enabled = False
        self.navigation_debug_log = NavigationDebugEventLog()
        self._nav_debug_seq = 0
        # None = showing the live (always-current) snapshot. An int is an
        # index into navigation_debug_log's events while the user is
        # stepping through history with the </> buttons (paused only).
        self._nav_debug_history_index = None
        self.editor_mode = False
        self.editor_tool = "rectangles"
        self.editor_interaction_mode = "paint"
        self.editor_brush_size = 0.2
        self.editor_pan_zoom_label = None
        self.editor_pending_draw_points: list[tuple[float, float]] = []
        self.editor_undo_stack: list[tuple[tuple[tuple[float, float, float, float], ...], tuple[float, float, float, float]]] = []
        self.editor_redo_stack: list[tuple[tuple[tuple[float, float, float, float], ...], tuple[float, float, float, float]]] = []
        self.editor_history_limit = 100
        self.editor_panel = None
        self.simulation_panel = None
        self.side_panel_container = None

        # Simulation clock and speed multiplier. The GUI still targets 60 FPS;
        # this multiplier changes simulated dt, not the QTimer interval.
        self.simulation_speed_options = [0.25, 0.50, 1.00, 1.50, 2.00]
        self.simulation_speed_index = 2
        self.simulation_speed = self.simulation_speed_options[self.simulation_speed_index]
        self.simulation_time = 0.0

        self.collision_checker = CollisionChecker() if CollisionChecker is not None else None
        self.last_collision_report = None
        self.spatial_index = SpatialObstacleIndex(SPATIAL_BUCKET_SIZE)
        self.spatial_index.rebuild(self.config.obstacles)

        self.known_obstacles: list[tuple[float, float, float, float]] = []
        self.mapped_obstacle_points: list[tuple[float, float]] = []
        self.explored_area_polygons: list[list[tuple[float, float]]] = []
        self.explored_free_points: set[tuple[float, float]] = set()
        self.belief_map = BeliefMap(
            bounds=(WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX),
            resolution=max(float(self.config.grid_resolution), 0.10),
            robot_count=max(1, int(self.config.robot_count)),
        )
        self.current_exploration_target: tuple[float, float] | None = None
        self.multi_exploration_targets: list[tuple[float, float] | None] = []
        self.multi_invalidated_exploration_targets: list[list[tuple[float, float]]] = []
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
        self.multi_last_explored_poses: dict[int, tuple[float, float, float]] = {}
        self.last_sensor_update_time = 0.0
        self.last_sensor_update_pose: tuple[float, float, float] | None = None

        self.thread_pool = QThreadPool.globalInstance()
        self.route_request_id = 0
        self.planning_in_progress = False
        self.active_planner_workers: dict[int, PlannerWorker] = {}

        # Prefetch async state — keyed by robot_index (0 = single robot).
        # Independent of planning_in_progress so the robot never brakes for a prefetch.
        self.prefetch_workers: dict[int, PlannerWorker] = {}
        self.prefetch_request_ids: dict[int, int] = {}
        self.prefetch_request_counter: int = 0

        # Metrics windows. Runtime counters are initialized above.
        self.metrics_window: SimulationMetricsWindow | None = None
        self.console_window = None

        self.path_points = []
        self.robots: list = []
        self.multi_path_points: list[list[tuple[float, float]]] = []
        self.multi_planned_path_points: list[list[tuple[float, float]]] = []
        self.multi_last_controls: list[np.ndarray] = []
        self.last_control = np.array([[0.0], [0.0]], dtype=float)
        self.last_time = time.perf_counter()

        self.setStyleSheet(self.stylesheet())
        self.build_ui()

        self.resize_to_screen()
        QTimer.singleShot(0, self.center_on_screen)

        self.timer = QTimer(self)
        # on_simulation_tick() times simulation_step() (see
        # PerfMonitor/SIM_PERF_LOG) and then calls it -- simulation_step()
        # itself is completely unchanged by this indirection.
        self.timer.timeout.connect(self.on_simulation_tick)
        self.timer.start(TARGET_FRAME_MS)

    # ========================================================
    # WINDOW SIZE / POSITION
    # ========================================================

    def resize_to_screen(self):
        screen = QApplication.primaryScreen()

        if screen is None:
            self.resize(WINDOW_TARGET_WIDTH, WINDOW_TARGET_HEIGHT)
            return

        available = screen.availableGeometry()

        width = min(WINDOW_TARGET_WIDTH, available.width() - 70)
        height = min(WINDOW_TARGET_HEIGHT, available.height() - 70)

        width = max(width, 1060)
        height = max(height, 660)

        self.resize(width, height)
        self.setMinimumSize(1040, 640)

    def center_on_screen(self):
        screen = QApplication.primaryScreen()

        if screen is None:
            return

        available = screen.availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(available.center())
        self.move(frame.topLeft())

    # ========================================================
    # UI
    # ========================================================

    def build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.top_bar = TopBar(self)
        outer.addWidget(self.top_bar)

        body = QWidget()
        body.setObjectName("body")

        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(14, 14, 14, 14)
        body_layout.setSpacing(14)

        self.canvas = SimulationCanvas()
        self.canvas.set_editor_brush_size(self.editor_brush_size)
        self.canvas.goalClicked.connect(self.set_goal_from_canvas)
        self.canvas.robotDragged.connect(self.move_robot_from_canvas)
        self.canvas.robotSelected.connect(self.select_robot_panel)
        self.canvas.editor_interaction_started.connect(self.on_editor_interaction_started)
        self.canvas.editor_interaction_progress.connect(self.on_editor_interaction_progress)
        self.canvas.editor_interaction_finished.connect(self.on_editor_interaction_finished)
        self.canvas.editor_camera_changed.connect(self.on_editor_camera_changed)
        self.canvas.editor_camera_interaction_started.connect(self.push_editor_undo_state)
        self.canvas.editor_obstacle_move_started.connect(self.push_editor_undo_state)
        self.canvas.editor_obstacle_moved.connect(self.on_editor_obstacle_moved)
        self.canvas.editor_view_changed.connect(self.refresh_editor_status_label)
        self.canvas.navigation_debug_step_back_button.clicked.connect(self.on_navigation_debug_step_back)
        self.canvas.navigation_debug_step_forward_button.clicked.connect(self.on_navigation_debug_step_forward)
        self.canvas.navigationDebugToggleRequested.connect(self.on_navigation_debug_eye_clicked)

        self.simulation_panel = self.build_config_panel()
        self.editor_panel = self.build_editor_panel()
        self.side_panel_container = QWidget()
        self.side_panel_container.setObjectName("sidePanelContainer")
        self.side_panel_container_layout = QVBoxLayout(self.side_panel_container)
        self.side_panel_container_layout.setContentsMargins(0, 0, 0, 0)
        self.side_panel_container_layout.setSpacing(0)
        self.side_panel_container_layout.addWidget(self.simulation_panel)
        self.side_panel_container_layout.addWidget(self.editor_panel)
        self.switch_panel_to_simulation()

        body_layout.addWidget(self.canvas, 1)
        body_layout.addWidget(self.side_panel_container, 0)

        outer.addWidget(body, 1)

    def build_config_panel(self):
        """Build the right-side simulation configuration panel."""
        return build_right_config_panel(self)

    def build_editor_panel(self):
        from robotics_sim.app.config_panel import build_editor_panel
        return build_editor_panel(self)

    def read_config(self) -> SimulationConfig:
        """Read GUI configuration and preserve editor camera settings."""
        config = super().read_config()
        if hasattr(self, "editor_camera_x_input"):
            config.camera_center_x = float(self.editor_camera_x_input.value())
            config.camera_center_y = float(self.editor_camera_y_input.value())
            config.camera_width = max(1.0, float(self.editor_camera_width_input.value()))
            config.camera_height = max(1.0, float(self.editor_camera_height_input.value()))
        else:
            config.camera_center_x = float(getattr(self.config, "camera_center_x", config.camera_center_x))
            config.camera_center_y = float(getattr(self.config, "camera_center_y", config.camera_center_y))
            config.camera_width = max(1.0, float(getattr(self.config, "camera_width", config.camera_width)))
            config.camera_height = max(1.0, float(getattr(self.config, "camera_height", config.camera_height)))
        return config

    def apply_config_to_widgets(self, config: SimulationConfig) -> None:
        """Apply loaded config, including the editor/simulation camera."""
        super().apply_config_to_widgets(config)
        if hasattr(self, "editor_camera_x_input"):
            self.set_editor_camera_controls(
                config.camera_center_x,
                config.camera_center_y,
                config.camera_width,
                config.camera_height,
                emit_preview=False,
            )
            self.config = self.read_config()
            self.canvas.set_preview_config(self.config)
            self.refresh_editor_status_label()

    def switch_panel_to_editor(self) -> None:
        if self.simulation_panel is not None:
            self.simulation_panel.setVisible(False)
        if self.editor_panel is not None:
            self.editor_panel.setVisible(True)
            self.editor_panel.setEnabled(True)
        self.refresh_editor_status_label()

    def switch_panel_to_simulation(self) -> None:
        if self.editor_panel is not None:
            self.editor_panel.setVisible(False)
            self.editor_panel.setEnabled(False)
        if self.simulation_panel is not None:
            self.simulation_panel.setVisible(True)
            self.simulation_panel.setEnabled(True)

    def toggle_editor_mode_from_button(self) -> None:
        self.set_editor_mode(not self.editor_mode)

    def ensure_multi_robot_configs(self) -> None:
        """Keep the editable multi-robot list consistent with the robot count."""
        count = max(1, min(8, int(round(float(self.robot_count_input.value())))))
        self.config.robot_count = count
        self.config.robots = list(self.multi_robot_configs)
        self.config.same_robot_configuration = self.same_config_switch.isChecked()
        self.multi_robot_configs = normalized_robot_start_configs(self.config)
        self.selected_robot_index = max(0, min(self.selected_robot_index, count - 1))

    def refresh_same_position_rows(self) -> None:
        if not hasattr(self, "same_position_inputs"):
            return

        self.ensure_multi_robot_configs()
        count = len(self.multi_robot_configs)

        for index, (row_widget, x_stepper, y_stepper) in enumerate(self.same_position_inputs):
            row_widget.setVisible(index < count)
            if index >= count:
                continue

            robot_cfg = self.multi_robot_configs[index]
            widgets = [x_stepper, y_stepper]
            blocked = [widget.blockSignals(True) for widget in widgets]
            x_stepper.setValue(robot_cfg.x)
            y_stepper.setValue(robot_cfg.y)
            for widget, was_blocked in zip(widgets, blocked):
                widget.blockSignals(was_blocked)

    def load_selected_robot_into_panel(self) -> None:
        self.ensure_multi_robot_configs()
        robot_cfg = self.multi_robot_configs[self.selected_robot_index]

        widgets = [
            self.multi_x_input,
            self.multi_y_input,
            self.multi_theta_input,
            self.multi_v_slider,
            self.multi_vision_slider,
            self.multi_body_radius_slider,
            self.multi_safety_radius_slider,
            self.multi_max_speed_input,
            self.multi_max_omega_input,
            self.multi_max_accel_input,
            self.multi_accel_gain_input,
            self.multi_goal_tol_input,
        ]
        blocked = [widget.blockSignals(True) for widget in widgets]
        self.multi_x_input.setValue(robot_cfg.x)
        self.multi_y_input.setValue(robot_cfg.y)
        self.multi_theta_input.setValue(robot_cfg.theta)
        self.multi_v_slider.setValue(robot_cfg.v)
        self.multi_vision_slider.setValue(robot_cfg.vision)
        self.multi_body_radius_slider.setValue(robot_cfg.body_radius)
        self.multi_safety_radius_slider.setValue(max(robot_cfg.safety_radius, robot_cfg.body_radius))
        self.multi_max_speed_input.setValue(robot_cfg.max_speed)
        self.multi_max_omega_input.setValue(robot_cfg.max_angular_speed)
        self.multi_max_accel_input.setValue(robot_cfg.max_acceleration)
        self.multi_accel_gain_input.setValue(robot_cfg.acceleration_gain)
        self.multi_goal_tol_input.setValue(robot_cfg.goal_tolerance)
        for widget, was_blocked in zip(widgets, blocked):
            widget.blockSignals(was_blocked)

        self.refresh_same_position_rows()

        count = len(self.multi_robot_configs)
        selected_color = ROBOT_COLOR_HEXES[self.selected_robot_index % len(ROBOT_COLOR_HEXES)]
        self.selected_robot_label.setText(f"Robot {self.selected_robot_index + 1} / {count}")
        self.selected_robot_label.setStyleSheet(f"color: {selected_color}; font-weight: 900;")
        self.update_multi_robot_panel_style()

    def update_multi_robot_panel_style(self) -> None:
        """
        Make the multi-robot setup visually meaningful without adding clutter.

        Colored borders are shown only when Multiple Robot Mode is active and
        Same Configuration is OFF. In that case the selected robot has its own
        editable configuration, so the color helps identify which robot is being
        adjusted. When robots share one configuration, no colored border is used.
        """
        if not hasattr(self, "multi_robot_card"):
            return

        is_multi = "Multiple" in self.top_bar.mode_selector.currentText()
        same_config = self.same_config_switch.isChecked() if hasattr(self, "same_config_switch") else True
        selected_color = ROBOT_COLOR_HEXES[self.selected_robot_index % len(ROBOT_COLOR_HEXES)]

        if is_multi and not same_config:
            card_border = selected_color
            robot_panel_style = (
                "QWidget#selectedRobotConfigPanel {"
                "background: #FFFFFF;"
                f"border: 2px solid {selected_color};"
                "border-radius: 8px;"
                "}"
            )
        else:
            card_border = BORDER_SOFT
            robot_panel_style = (
                "QWidget#selectedRobotConfigPanel {"
                "background: transparent;"
                "border: none;"
                "}"
            )

        self.multi_robot_card.setStyleSheet(
            "QFrame#sectionCard {"
            f"background: {PANEL_CARD};"
            f"border: 1px solid {card_border};"
            "border-radius: 9px;"
            "}"
        )
        self.multi_position_row.setStyleSheet(robot_panel_style)
        self.multi_sensing_row.setStyleSheet(robot_panel_style)
        self.multi_dynamics_row.setStyleSheet(robot_panel_style)

    def sync_same_positions_from_panel(self, *_):
        if not hasattr(self, "same_position_inputs") or not self.same_config_switch.isChecked():
            return

        self.ensure_multi_robot_configs()
        for index, (_, x_stepper, y_stepper) in enumerate(self.same_position_inputs):
            if index >= len(self.multi_robot_configs):
                continue
            self.multi_robot_configs[index] = RobotStartConfig(
                x=float(x_stepper.value()),
                y=float(y_stepper.value()),
                theta=float(self.theta_input.value()),
                v=float(self.v_slider.value()),
                vision=float(self.vision_slider.value()),
                body_radius=float(self.body_radius_slider.value()),
                safety_radius=max(float(self.safety_radius_slider.value()), float(self.body_radius_slider.value())),
                max_speed=float(self.max_speed_input.value()),
                max_acceleration=float(self.max_accel_input.value()),
                max_angular_speed=float(self.max_omega_input.value()),
                goal_tolerance=float(self.goal_tol_input.value()),
                acceleration_gain=float(self.accel_gain_input.value()),
            )
        self.update_preview()

    def sync_selected_robot_from_panel(self, *_):
        if not hasattr(self, "multi_x_input"):
            return

        self.ensure_multi_robot_configs()
        index = self.selected_robot_index
        current = self.multi_robot_configs[index]
        if self.same_config_switch.isChecked():
            updated = RobotStartConfig(
                x=float(self.multi_x_input.value()),
                y=float(self.multi_y_input.value()),
                theta=float(self.theta_input.value()),
                v=float(self.v_slider.value()),
                vision=float(self.vision_slider.value()),
                body_radius=float(self.body_radius_slider.value()),
                safety_radius=max(float(self.safety_radius_slider.value()), float(self.body_radius_slider.value())),
                max_speed=float(self.max_speed_input.value()),
                max_acceleration=float(self.max_accel_input.value()),
                max_angular_speed=float(self.max_omega_input.value()),
                goal_tolerance=float(self.goal_tol_input.value()),
                acceleration_gain=float(self.accel_gain_input.value()),
            )
        else:
            body_radius = float(self.multi_body_radius_slider.value())
            updated = RobotStartConfig(
                x=float(self.multi_x_input.value()),
                y=float(self.multi_y_input.value()),
                theta=float(self.multi_theta_input.value()),
                v=float(self.multi_v_slider.value()),
                vision=float(self.multi_vision_slider.value()),
                body_radius=body_radius,
                safety_radius=max(float(self.multi_safety_radius_slider.value()), body_radius),
                max_speed=float(self.multi_max_speed_input.value()),
                max_acceleration=float(self.multi_max_accel_input.value()),
                max_angular_speed=float(self.multi_max_omega_input.value()),
                goal_tolerance=float(self.multi_goal_tol_input.value()),
                acceleration_gain=float(self.multi_accel_gain_input.value()),
            )

        self.multi_robot_configs[index] = updated
        self.update_preview()

    def select_robot_panel(self, index: int) -> None:
        if not hasattr(self, "robot_count_input"):
            return
        self.ensure_multi_robot_configs()
        self.selected_robot_index = max(0, min(int(index), len(self.multi_robot_configs) - 1))
        self.load_selected_robot_into_panel()
        self.update_preview()

    def select_previous_robot(self) -> None:
        self.ensure_multi_robot_configs()
        self.select_robot_panel((self.selected_robot_index - 1) % len(self.multi_robot_configs))

    def select_next_robot(self) -> None:
        self.ensure_multi_robot_configs()
        self.select_robot_panel((self.selected_robot_index + 1) % len(self.multi_robot_configs))

    def on_robot_count_changed(self, *_):
        self.ensure_multi_robot_configs()
        self.load_selected_robot_into_panel()
        self.update_preview()

    def on_same_config_toggled(self, *_):
        self.ensure_multi_robot_configs()
        self.load_selected_robot_into_panel()
        self.update_relevant_parameter_visibility()
        self.update_preview()

    def on_agent_mode_changed(self, *_):
        self.update_relevant_parameter_visibility()
        self.update_preview()

    def on_grid_resolution_control_changed(self, value: float) -> None:
        """Show the temporary red grid preview while the user adjusts
        grid_resolution. Purely visual -- update_preview() (already
        connected to this same control via numeric_widgets) is what
        actually applies the value into self.config; this only drives the
        canvas overlay, and never rebuilds any occupancy/planning grid
        mid-run."""
        self.canvas.show_grid_resolution_preview(float(value))
        self.canvas.set_grid_overlay_resolution(float(value))

    def on_grid_overlay_toggled(self, enabled: bool) -> None:
        """Toggle the persistent "Show Grid" overlay.

        Rendering-only: it never touches self.config, never rebuilds the
        belief/occupancy or planning grid, and is intentionally allowed to
        change while the simulation is running (see grid_overlay_toggle's
        exclusion from locked_during_run_widgets in config_panel.py)."""
        self.canvas.set_grid_overlay_enabled(bool(enabled))

    def on_navigation_debug_eye_clicked(self) -> None:
        """The canvas eye icon only emits navigationDebugToggleRequested
        (never decides state itself, same as goalClicked) -- this reads the
        current state and flips it, so there is exactly one control (the
        eye icon) and exactly one place that decides the new value."""
        self.on_navigation_debug_toggled(not self.navigation_debug_enabled)

    def on_navigation_debug_toggled(self, enabled: bool) -> None:
        """Apply the navigation debug on/off state.

        Gates both snapshot capture (engine.py checks
        self.navigation_debug_enabled before building a
        NavigationDebugCapture at all -- zero cost while off) and its
        rendering (canvas checks its own mirrored flag before drawing).
        Never touches self.config, robot/agent state, or the simulation
        itself -- only which values get computed for display -- so it is
        intentionally allowed to change while paused or running. The
        canvas eye icon is always visible (unlike a side-panel control),
        so this can fire at any time, running or paused."""
        self.navigation_debug_enabled = bool(enabled)
        self.canvas.set_navigation_debug_enabled(bool(enabled))
        if not enabled:
            self.resume_navigation_debug_live_view()
        self.update_navigation_debug_step_buttons()

    def on_navigation_debug_step_back(self) -> None:
        self.step_navigation_debug_history(-1)

    def on_navigation_debug_step_forward(self) -> None:
        self.step_navigation_debug_history(1)

    def move_robot_from_canvas(self, index: int, x: float, y: float) -> None:
        if self.running or self.robot is not None or bool(getattr(self, "robots", [])):
            return

        # Single-robot preview dragging.
        if int(index) < 0:
            self.x_input.setValue(float(x))
            self.y_input.setValue(float(y))
            self.update_preview()
            return

        self.ensure_multi_robot_configs()
        if not (0 <= int(index) < len(self.multi_robot_configs)):
            return

        self.selected_robot_index = int(index)
        robot_cfg = self.multi_robot_configs[self.selected_robot_index]
        self.multi_robot_configs[self.selected_robot_index] = RobotStartConfig(
            x=float(x),
            y=float(y),
            theta=float(robot_cfg.theta),
            v=float(robot_cfg.v),
            vision=float(robot_cfg.vision),
            body_radius=float(robot_cfg.body_radius),
            safety_radius=max(float(robot_cfg.safety_radius), float(robot_cfg.body_radius)),
            max_speed=float(robot_cfg.max_speed),
            max_acceleration=float(robot_cfg.max_acceleration),
            max_angular_speed=float(robot_cfg.max_angular_speed),
            goal_tolerance=float(robot_cfg.goal_tolerance),
            acceleration_gain=float(robot_cfg.acceleration_gain),
        )
        self.load_selected_robot_into_panel()
        self.update_preview()

    # ========================================================
    # PARAMETER VISIBILITY / LOCKING
    # ========================================================

    def update_relevant_parameter_visibility(self) -> None:
        """
        Hide parameter fields that are irrelevant for the current selections.

        This keeps the panel from becoming noisy:
            - Direct planning does not use a path simplifier.
            - Goal seeking does not use frontier replanning cooldown.
            - IPP λ only matters for the informative frontier planner.
        """
        planner = self.planner_combo.currentText()
        exploration = self.exploration_planner_combo.currentText()

        uses_grid_path_planner = planner in ("A*", "Dijkstra")
        uses_frontier_exploration = is_exploration_planner(exploration)
        uses_ipp_lite = exploration == "Informative frontier / IPP-lite"
        is_multi_robot_mode = "Multiple" in self.top_bar.mode_selector.currentText()
        same_config = getattr(self, "same_config_switch", None) is not None and self.same_config_switch.isChecked()

        self.path_simplifier_field.setVisible(uses_grid_path_planner)
        self.coordinator_field.setVisible(is_multi_robot_mode and uses_frontier_exploration)
        self.exploration_cooldown_field.setVisible(uses_frontier_exploration)
        self.ipp_lambda_field.setVisible(uses_ipp_lite)
        self.apply_algorithm_ownership_gui_policy()

        if hasattr(self, "multi_robot_card"):
            self.multi_robot_card.setVisible(is_multi_robot_mode)
            self.same_positions_widget.setVisible(is_multi_robot_mode and same_config)
            self.robot_nav_widget.setVisible(is_multi_robot_mode and not same_config)
            self.selected_robot_section_label.setVisible(is_multi_robot_mode and not same_config)
            self.multi_position_row.setVisible(is_multi_robot_mode and not same_config)
            self.multi_sensing_row.setVisible(is_multi_robot_mode and not same_config)
            self.multi_dynamics_row.setVisible(is_multi_robot_mode and not same_config)
            self.robot_count_input.setVisible(is_multi_robot_mode)
            self.same_config_switch.setVisible(is_multi_robot_mode)
            self.prev_robot_button.setEnabled(is_multi_robot_mode and not same_config and not self.running)
            self.next_robot_button.setEnabled(is_multi_robot_mode and not same_config and not self.running)
            self.refresh_same_position_rows()
            self.update_multi_robot_panel_style()

        # In multi-robot mode with different configurations, the shared Robot
        # Parameters and Dynamics cards would be misleading because those values
        # are now edited per selected robot inside Multi-Robot Setup. Goal Setup
        # remains shared by the team in both modes.
        if hasattr(self, "robot_card"):
            self.robot_card.setVisible(not is_multi_robot_mode or same_config)
        if hasattr(self, "dynamics_card"):
            self.dynamics_card.setVisible(not is_multi_robot_mode or same_config)

    def apply_algorithm_ownership_gui_policy(self) -> None:
        """
        Grey out (not hide) controls the selected plugin already owns.

        This does not decide plugin behavior -- robotics_sim.simulation.
        gui_policy.compute_gui_control_policy() is the pure, Qt-free source of
        truth. This method only applies that decision to widgets, so a plugin
        declaring TARGET_GENERATION does not leave "FoV-aware directional
        frontier" looking like the active exploration algorithm.
        """
        if not hasattr(self, "coordinator_combo"):
            return

        try:
            profile = runtime_profile_for_strategy(self.coordinator_combo.currentText())
        except PluginLoadError:
            return

        policy = compute_gui_control_policy(profile)
        # A locked (running) configuration always wins over the ownership
        # policy: set_configuration_locked() disables these same widgets and
        # then calls update_relevant_parameter_visibility(), so re-enabling a
        # plugin-owned-but-otherwise-allowed control here would unlock it
        # mid-run.
        not_locked = not getattr(self, "running", False)

        self.exploration_planner_field.setEnabled(policy.exploration_planner_enabled and not_locked)
        self.exploration_planner_field.setToolTip(policy.exploration_planner_reason)
        if hasattr(self.exploration_planner_field, "field_label"):
            base_text = getattr(self.exploration_planner_field, "field_label_base_text", "Exploration Planner")
            label_text = (
                base_text
                if policy.exploration_planner_enabled
                else f"{base_text} (provided by algorithm / fallback service)"
            )
            self.exploration_planner_field.field_label.setText(label_text)
        self.planner_combo.setEnabled(policy.path_planner_enabled and not_locked)
        self.path_simplifier_field.setEnabled(policy.path_simplifier_enabled and not_locked)
        self.control_combo.setEnabled(policy.control_enabled and not_locked)

    def set_configuration_locked(self, locked: bool) -> None:
        """
        Disable simulation-defining controls while a run is active.

        Pausing does not unlock them. A paused run still has a robot, map,
        planner state, async route requests, and metrics tied to the current
        configuration. Restart/reset returns the UI to an editable state.
        """
        widgets = getattr(self, "locked_during_run_widgets", [])
        for widget in widgets:
            widget.setEnabled(not locked and not self.editor_mode)

        if hasattr(self, "editor_tool_combo"):
            self.editor_tool_combo.setEnabled(not locked and self.editor_mode)
        if hasattr(self.top_bar, "editor_button"):
            self.top_bar.editor_button.setEnabled(True)

        # Keep visibility rules active even while the controls are disabled.
        self.update_relevant_parameter_visibility()

    def set_editor_mode(self, enabled: bool) -> None:
        self.editor_mode = bool(enabled)
        self.canvas.set_editor_mode(self.editor_mode)
        if self.editor_mode:
            self.switch_panel_to_editor()
            if hasattr(self, "editor_tool_combo"):
                self.editor_tool_combo.setEnabled(True)
            self.canvas.set_status("Editor mode enabled. Create or edit the map.")
        else:
            self.switch_panel_to_simulation()
            if hasattr(self, "editor_tool_combo"):
                self.editor_tool_combo.setEnabled(False)
            self.canvas.set_status("Editor mode disabled. Simulation interactions restored.")
        self.update_preview()
        if hasattr(self.top_bar, "editor_button"):
            self.top_bar.editor_button.setChecked(self.editor_mode)
        self.set_configuration_locked(self.running or self.robot is not None)

    def set_editor_tool(self, tool: str) -> None:
        tool_name = str(tool).lower()
        if "camera" in tool_name or "viewport" in tool_name:
            self.editor_tool = "camera"
        elif "move obstacle" in tool_name or "select" in tool_name:
            # Legacy UI text from older builds. Object movement is no longer a
            # separate tool; click-drag any existing object while editing.
            self.editor_tool = "rectangles"
        elif "free" in tool_name:
            self.editor_tool = "free"
        elif "erase" in tool_name:
            self.editor_tool = "erase"
        elif "square" in tool_name:
            self.editor_tool = "squares"
        else:
            self.editor_tool = "rectangles"

        self.canvas.set_editor_tool(self.editor_tool)

        if hasattr(self, "editor_brush_size_input") and self.editor_brush_size_input is not None:
            self.editor_brush_size_input.setEnabled(self.editor_tool == "free")

        if hasattr(self, "editor_tool_combo") and self.editor_tool_combo is not None:
            desired_text = {
                "camera": "Camera view",
                "free": "Free draw",
                "erase": "Erase",
                "squares": "Squares",
                "rectangles": "Rectangles",
            }.get(self.editor_tool, "Rectangles")
            if self.editor_tool_combo.currentText() != desired_text:
                blocked = self.editor_tool_combo.blockSignals(True)
                self.editor_tool_combo.setCurrentText(desired_text)
                self.editor_tool_combo.blockSignals(blocked)

        self.refresh_editor_tool_buttons()
        self.refresh_editor_status_label()
        status_by_tool = {
            "rectangles": "Map editor: click empty space to draw rectangles; click an object to drag it.",
            "squares": "Map editor: click empty space to draw squares; click an object to drag it.",
            "free": "Map editor: free-draw with circular brush; click an object to drag it.",
            "erase": "Map editor: click an obstacle to remove it.",
            "camera": "Map editor: drag/resize the red simulation camera viewport.",
        }
        self.canvas.set_status(status_by_tool.get(self.editor_tool, "Map editor tool updated."))

    def set_editor_brush_size(self, brush_size: float) -> None:
        self.editor_brush_size = max(0.05, float(brush_size))
        self.canvas.set_editor_brush_size(self.editor_brush_size)

        slider = getattr(self, "editor_brush_size_slider", None)
        if slider is not None:
            blocked = slider.blockSignals(True)
            slider.setValue(int(round(self.editor_brush_size * 100.0)))
            slider.blockSignals(blocked)

        preview = getattr(self, "editor_brush_size_preview", None)
        if preview is not None:
            preview.set_brush_size(self.editor_brush_size)

        value_label = getattr(self, "editor_brush_size_value_label", None)
        if value_label is not None:
            value_label.setText(f"{self.editor_brush_size:.2f} m")

    def set_editor_brush_size_from_slider(self, raw_value: int) -> None:
        self.set_editor_brush_size(float(raw_value) / 100.0)

    def set_editor_interaction_mode(self, mode: str) -> None:
        self.editor_interaction_mode = "move" if str(mode).lower() == "move" else "paint"
        self.canvas.set_editor_interaction_mode(self.editor_interaction_mode)
        self.refresh_editor_tool_buttons()
        if self.editor_interaction_mode == "move":
            self.canvas.set_status("Pan/Zoom mode: drag the map or use the wheel. Object tools are locked.")
        else:
            self.canvas.set_status("Edit objects mode: draw on empty space, or click-drag an existing object to move it.")

    def refresh_editor_tool_buttons(self) -> None:
        tool_names = ("rectangles", "squares", "free", "erase", "camera")
        tool_controls_enabled = self.editor_interaction_mode != "move"

        for name in tool_names:
            button = getattr(self, f"editor_{name}_button", None)
            if button is not None:
                button.setChecked(self.editor_tool == name and tool_controls_enabled)
                button.setEnabled(tool_controls_enabled)

        tool_combo = getattr(self, "editor_tool_combo", None)
        if tool_combo is not None:
            tool_combo.setEnabled(tool_controls_enabled)

        brush_enabled = tool_controls_enabled and self.editor_tool == "free"
        for attr in ("editor_brush_size_input", "editor_brush_size_slider", "editor_brush_size_preview"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setEnabled(brush_enabled)

        # Viewport numeric controls are only editable when the editor is in
        # object-edit mode. Pan/Zoom mode is exclusively for moving the editor
        # camera, not the simulation viewport.
        camera_enabled = tool_controls_enabled
        for attr in (
            "editor_camera_x_input",
            "editor_camera_y_input",
            "editor_camera_width_input",
            "editor_camera_height_input",
            "editor_camera_reset_button",
            "editor_camera_fit_button",
        ):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.setEnabled(camera_enabled)

        paint_button = getattr(self, "editor_paint_button", None)
        move_button = getattr(self, "editor_move_button", None)
        if paint_button is not None:
            paint_button.setChecked(self.editor_interaction_mode == "paint")
        if move_button is not None:
            move_button.setChecked(self.editor_interaction_mode == "move")

    def set_editor_camera_controls(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        *,
        emit_preview: bool = True,
    ) -> None:
        if not hasattr(self, "editor_camera_x_input"):
            return

        self._syncing_editor_camera = True
        try:
            values = (float(center_x), float(center_y), max(1.0, float(width)), max(1.0, float(height)))
            widgets = (
                self.editor_camera_x_input,
                self.editor_camera_y_input,
                self.editor_camera_width_input,
                self.editor_camera_height_input,
            )
            for widget, value in zip(widgets, values):
                blocked = widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(blocked)
        finally:
            self._syncing_editor_camera = False

        self.config.camera_center_x = float(center_x)
        self.config.camera_center_y = float(center_y)
        self.config.camera_width = max(1.0, float(width))
        self.config.camera_height = max(1.0, float(height))
        self.canvas.set_camera_view(
            self.config.camera_center_x,
            self.config.camera_center_y,
            self.config.camera_width,
            self.config.camera_height,
            emit_signal=False,
        )
        self.refresh_editor_status_label()
        if emit_preview:
            self.update_preview()

    def sync_editor_camera_from_panel(self, *_):
        if getattr(self, "_syncing_editor_camera", False):
            return
        if not hasattr(self, "editor_camera_x_input"):
            return
        self.push_editor_undo_state()
        self.set_editor_camera_controls(
            self.editor_camera_x_input.value(),
            self.editor_camera_y_input.value(),
            self.editor_camera_width_input.value(),
            self.editor_camera_height_input.value(),
        )

    def on_editor_camera_changed(self, camera_tuple: tuple) -> None:
        if len(camera_tuple) != 4:
            return
        self.set_editor_camera_controls(
            float(camera_tuple[0]),
            float(camera_tuple[1]),
            float(camera_tuple[2]),
            float(camera_tuple[3]),
        )

    def reset_editor_camera(self) -> None:
        self.push_editor_undo_state()
        self.set_editor_camera_controls(
            (WORLD_X_MIN + WORLD_X_MAX) / 2.0,
            (WORLD_Y_MIN + WORLD_Y_MAX) / 2.0,
            WORLD_X_MAX - WORLD_X_MIN,
            WORLD_Y_MAX - WORLD_Y_MIN,
        )
        self.canvas.set_status("Simulation camera reset to the full world.")

    def fit_editor_camera_to_obstacles(self) -> None:
        self.push_editor_undo_state()
        if not self.config.obstacles:
            self.reset_editor_camera()
            return

        xs = [obstacle[0] for obstacle in self.config.obstacles] + [obstacle[0] + obstacle[2] for obstacle in self.config.obstacles]
        ys = [obstacle[1] for obstacle in self.config.obstacles] + [obstacle[1] + obstacle[3] for obstacle in self.config.obstacles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max(1.0, (max_x - min_x) * 1.25)
        height = max(1.0, (max_y - min_y) * 1.25)
        self.set_editor_camera_controls(
            (min_x + max_x) / 2.0,
            (min_y + max_y) / 2.0,
            width,
            height,
        )
        self.canvas.set_status("Simulation camera fitted to current obstacles.")

    def editor_snapshot(self) -> tuple[tuple[tuple[float, float, float, float], ...], tuple[float, float, float, float]]:
        """Return an undoable snapshot of obstacles and camera viewport."""
        obstacles = tuple(tuple(map(float, obstacle)) for obstacle in self.config.obstacles)
        camera = (
            float(getattr(self.config, "camera_center_x", (WORLD_X_MIN + WORLD_X_MAX) / 2.0)),
            float(getattr(self.config, "camera_center_y", (WORLD_Y_MIN + WORLD_Y_MAX) / 2.0)),
            float(getattr(self.config, "camera_width", WORLD_X_MAX - WORLD_X_MIN)),
            float(getattr(self.config, "camera_height", WORLD_Y_MAX - WORLD_Y_MIN)),
        )
        return obstacles, camera

    def restore_editor_snapshot(self, snapshot: tuple[tuple[tuple[float, float, float, float], ...], tuple[float, float, float, float]]) -> None:
        obstacles, camera = snapshot
        self.config.obstacles = [tuple(map(float, obstacle)) for obstacle in obstacles]
        self.set_editor_camera_controls(camera[0], camera[1], camera[2], camera[3], emit_preview=False)
        self.rebuild_editor_state()

    def push_editor_undo_state(self, *_args) -> None:
        if not self.editor_mode:
            return
        snapshot = self.editor_snapshot()
        if self.editor_undo_stack and self.editor_undo_stack[-1] == snapshot:
            return
        self.editor_undo_stack.append(snapshot)
        if len(self.editor_undo_stack) > self.editor_history_limit:
            self.editor_undo_stack = self.editor_undo_stack[-self.editor_history_limit:]
        self.editor_redo_stack.clear()

    def undo_editor_change(self) -> None:
        if not self.editor_mode or not self.editor_undo_stack:
            self.canvas.set_status("Nothing to undo in the map editor.")
            return
        current = self.editor_snapshot()
        previous = self.editor_undo_stack.pop()
        self.editor_redo_stack.append(current)
        self.restore_editor_snapshot(previous)
        self.canvas.set_status("Editor undo applied.")

    def redo_editor_change(self) -> None:
        if not self.editor_mode or not self.editor_redo_stack:
            self.canvas.set_status("Nothing to redo in the map editor.")
            return
        current = self.editor_snapshot()
        next_snapshot = self.editor_redo_stack.pop()
        self.editor_undo_stack.append(current)
        self.restore_editor_snapshot(next_snapshot)
        self.canvas.set_status("Editor redo applied.")

    def on_editor_obstacle_moved(self, move_tuple: tuple) -> None:
        if len(move_tuple) != 3:
            return

        first = move_tuple[0]
        if isinstance(first, (list, tuple)):
            indices = [int(index) for index in first]
            dx = float(move_tuple[1])
            dy = float(move_tuple[2])
            moved = move_obstacles_by(self.config.obstacles, indices, (dx, dy))
            label = "Object moved." if len(indices) > 1 else "Obstacle moved."
        else:
            # Backward-compatible path for older canvas payloads.
            index = int(first)
            left = float(move_tuple[1])
            bottom = float(move_tuple[2])
            moved = move_obstacle_to(self.config.obstacles, index, (left, bottom))
            label = "Obstacle moved."

        if moved:
            self.rebuild_editor_state()
            self.canvas.set_status(label)

    def commit_editor_map(self) -> None:
        self.rebuild_editor_state()
        self.canvas.fit_to_obstacles(self.config.obstacles)
        self.canvas.set_status("Map updated and ready to use.")

    def new_editor_map(self) -> None:
        self.push_editor_undo_state()
        self.config.obstacles = []
        self.rebuild_editor_state()
        self.canvas.fit_to_obstacles(self.config.obstacles)
        self.canvas.set_status("Blank map created.")

    def clear_editor_map(self) -> None:
        self.push_editor_undo_state()
        self.config.obstacles = []
        self.rebuild_editor_state()
        self.canvas.set_status("Map cleared.")

    def on_editor_interaction_started(self, start_xy: tuple[float, float]) -> None:
        if self.running or self.robot is not None or bool(getattr(self, "robots", [])):
            return
        self.editor_pending_draw_points = [tuple(start_xy)]
        self.canvas.set_editor_drag_start(start_xy)

    def on_editor_interaction_progress(self, point_xy: tuple[float, float]) -> None:
        self.editor_pending_draw_points.append(tuple(point_xy))

    def on_editor_interaction_finished(self, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> None:
        if self.running or self.robot is not None or bool(getattr(self, "robots", [])):
            return
        if not self.editor_mode:
            return

        if self.editor_tool == "camera":
            self.canvas.set_status("Camera tool active. Drag the red frame or its handles.")
            return

        if self.editor_tool == "erase":
            self.push_editor_undo_state()
            removed = remove_obstacle_at(self.config.obstacles, end_xy)
            if removed:
                self.rebuild_editor_state()
                self.canvas.set_status("Obstacle removed.")
            else:
                self.canvas.set_status("No obstacle selected for editing.")
            return

        if self.editor_tool == "free":
            free_draw_obstacles = create_free_draw_obstacles_from_path(
                self.editor_pending_draw_points,
                brush_size=self.editor_brush_size,
            )
            if free_draw_obstacles:
                self.push_editor_undo_state()
                self.config.obstacles.extend(free_draw_obstacles)
                self.rebuild_editor_state()
                self.canvas.set_status("Smooth free-draw object added.")
            else:
                self.canvas.set_status("No free-draw stroke created.")
            return

        if self.editor_tool == "squares":
            from robotics_sim.app.map_editor import create_square_obstacle_from_drag
            obstacle = create_square_obstacle_from_drag(start_xy, end_xy, min_size=MIN_EDITOR_OBSTACLE_SIZE)
            if obstacle is None:
                self.canvas.set_status("Obstacle too small. Drag a larger square.")
                return
            self.push_editor_undo_state()
            self.config.obstacles.append(obstacle)
            self.rebuild_editor_state()
            self.canvas.set_status("Square obstacle added.")
            return

        obstacle = create_rect_obstacle_from_drag(start_xy, end_xy, min_size=MIN_EDITOR_OBSTACLE_SIZE)
        if obstacle is None:
            self.canvas.set_status("Obstacle too small. Drag a larger rectangle.")
            return

        self.push_editor_undo_state()
        self.config.obstacles.append(obstacle)
        self.rebuild_editor_state()
        self.canvas.set_status("Obstacle added.")


    def keyPressEvent(self, event):
        if self.editor_mode and event.key() == Qt.Key_Z:
            modifiers = event.modifiers()
            if modifiers & Qt.ControlModifier and modifiers & Qt.AltModifier:
                self.redo_editor_change()
                event.accept()
                return
            if modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier:
                self.redo_editor_change()
                event.accept()
                return
            if modifiers & Qt.ControlModifier:
                self.undo_editor_change()
                event.accept()
                return
        super().keyPressEvent(event)

    def rebuild_editor_state(self) -> None:
        # Do not auto-union user-created obstacles. Destructive bounding-box
        # merges change free-draw strokes and L-shaped compositions into shapes
        # the user never drew. Explicit tools may still call merge_obstacles
        # when they know the merge preserves geometry.
        self.config.obstacles = normalize_obstacles([tuple(obstacle) for obstacle in self.config.obstacles])
        self.spatial_index.rebuild(self.config.obstacles)
        self.update_preview()
        self.refresh_editor_status_label()

    def refresh_editor_status_label(self) -> None:
        if hasattr(self, "editor_status_label") and self.editor_status_label is not None:
            self.editor_status_label.setText(self.canvas.editor_status_text() if self.editor_mode else "Editor disabled")

    # ========================================================

    # STYLE
    # ========================================================

    def stylesheet(self):
        return f"""
        QWidget#root {{
            background: {BG};
            font-family: "Segoe UI";
            color: {TEXT};
        }}

        QWidget#body {{
            background: {BG};
        }}

        QFrame#topBar {{
            background: qlineargradient(
                x1: 0, y1: 0,
                x2: 1, y2: 0,
                stop: 0 {MAROON_DARK},
                stop: 1 {MAROON}
            );
        }}

        QLabel#topTitle {{
            color: white;
            font-size: 14px;
            font-weight: 900;
            background: transparent;
        }}

        QLabel#statusReady {{
            color: {GREEN};
            font-size: 12px;
            font-weight: 800;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 5px 13px;
        }}

        QLabel#statusRunning {{
            color: {GREEN};
            font-size: 12px;
            font-weight: 800;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 5px 13px;
        }}

        QLabel#statusPaused {{
            color: {ORANGE};
            font-size: 12px;
            font-weight: 800;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 5px 13px;
        }}

        QPushButton#topIconButton,
        QPushButton#windowButton,
        QPushButton#closeButton {{
            background: transparent;
            border: none;
            border-radius: 6px;
        }}

        QPushButton#topIconButton:hover,
        QPushButton#windowButton:hover {{
            background: rgba(255,255,255,0.12);
        }}

        QPushButton#closeButton:hover {{
            background: #B42318;
        }}

        QComboBox#topModeSelector {{
            background: rgba(255,255,255,0.08);
            color: #FFFFFF;
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 6px;
            padding-left: 9px;
            font-size: 11px;
            font-weight: 800;
            min-height: 28px;
        }}

        QComboBox#topModeSelector::drop-down {{
            width: 22px;
            border: none;
        }}

        QPushButton#modeSegmentButton {{
            background: rgba(255,255,255,0.08);
            color: #FFFFFF;
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 7px;
            font-size: 11px;
            font-weight: 900;
        }}

        QPushButton#modeSegmentButton:hover {{
            background: rgba(255,255,255,0.14);
        }}

        QPushButton#modeSegmentButton:checked {{
            background: #FFFFFF;
            color: {MAROON};
            border: 1px solid #FFFFFF;
        }}

        QPushButton#modeSegmentButton:disabled {{
            color: rgba(255,255,255,0.45);
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
        }}

        QFrame#sidePanel {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 14px;
        }}

        QScrollArea#configScroll {{
            background: transparent;
            border: none;
        }}

        QWidget#scrollContent {{
            background: transparent;
        }}

        QFrame#sectionCard {{
            background: {PANEL_CARD};
            border: 1px solid {BORDER_SOFT};
            border-radius: 9px;
        }}

        QFrame#actionPanelBottom {{
            background: {CARD};
            border-top: 1px solid {BORDER_SOFT};
            border-bottom-left-radius: 14px;
            border-bottom-right-radius: 14px;
        }}

        QLabel#sectionTitle {{
            color: {MAROON};
            font-size: 13px;
            font-weight: 900;
        }}

        QLabel#fieldLabel {{
            color: {TEXT_MUTED};
            font-size: 10px;
            font-weight: 700;
        }}

        QLabel#subsectionLabel {{
            color: #5E6673;
            font-size: 10px;
            font-weight: 900;
            padding-top: 3px;
        }}

        QPushButton#stepperButton {{
            background: #F8F9FB;
            color: {MAROON};
            border: 1px solid {BORDER};
            border-radius: 5px;
            min-height: 28px;
            font-size: 13px;
            font-weight: 900;
        }}

        QPushButton#stepperButton:hover {{
            background: #F0F2F5;
            border: 1px solid {MAROON};
        }}

        QLineEdit#numericInput {{
            background: white;
            color: #000000;
            border: 1px solid {BORDER};
            border-radius: 5px;
            min-height: 28px;
            font-size: 11px;
            font-weight: 900;
            padding-left: 4px;
            padding-right: 4px;
        }}

        QLineEdit#numericInput:focus {{
            border: 2px solid {MAROON};
        }}

        QLineEdit#smallNumericInput {{
            background: white;
            color: #000000;
            border: 1px solid {BORDER};
            border-radius: 5px;
            min-height: 28px;
            font-size: 11px;
            font-weight: 900;
        }}

        QLineEdit#smallNumericInput:focus {{
            border: 2px solid {MAROON};
        }}

                QComboBox {{
            background-color: #FFFFFF;
            color: #111827;
            border: 1px solid {BORDER};
            border-radius: 7px;
            min-height: 34px;
            padding-left: 10px;
            padding-right: 28px;
            font-size: 12px;
            font-weight: 800;
            selection-background-color: #F4EAEA;
            selection-color: {MAROON};
        }}

        QComboBox:hover {{
            background-color: #FBFCFE;
            border: 1px solid #B8C0CC;
        }}

        QComboBox:focus {{
            background-color: #FFFFFF;
            border: 2px solid {MAROON};
        }}

        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 28px;
            border-left: 1px solid {BORDER_SOFT};
            border-top-right-radius: 7px;
            border-bottom-right-radius: 7px;
            background-color: #FFFFFF;
        }}

        QComboBox::down-arrow {{
            image: none;
            width: 0px;
            height: 0px;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #111827;
            margin-right: 8px;
        }}

        QComboBox QAbstractItemView {{
            background-color: #FFFFFF;
            color: #111827;
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 4px;
            outline: 0px;
            selection-background-color: #F4EAEA;
            selection-color: {MAROON};
        }}

        QComboBox QAbstractItemView::item {{
            min-height: 28px;
            padding: 6px 8px;
            color: #111827;
            background-color: #FFFFFF;
        }}

        QComboBox QAbstractItemView::item:selected {{
            color: {MAROON};
            background-color: #F4EAEA;
        }}

        QComboBox QAbstractItemView::item:hover {{
            color: {MAROON};
            background-color: #FAF2F2;
        }}

        QSlider::groove:horizontal {{
            height: 4px;
            background: #E0E2E7;
            border-radius: 2px;
        }}

        QSlider::sub-page:horizontal {{
            background: {MAROON};
            border-radius: 2px;
        }}

        QSlider::handle:horizontal {{
            background: {MAROON};
            border: 2px solid white;
            width: 13px;
            height: 13px;
            margin: -5px 0;
            border-radius: 7px;
        }}

        QCheckBox {{
            color: {TEXT};
            font-size: 11px;
            font-weight: 700;
        }}

        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
            border-radius: 3px;
            border: 1.5px solid {MAROON};
            background: white;
        }}

        QCheckBox::indicator:checked {{
            background: {MAROON};
        }}

        QPushButton#startButton {{
            background: {MAROON};
            color: white;
            border: none;
            border-radius: 7px;
            min-height: 40px;
            font-size: 14px;
            font-weight: 900;
        }}

        QPushButton#startButton:hover {{
            background: #6A0000;
        }}

        QPushButton#secondaryButton {{
            background: white;
            color: {TEXT};
            border: 1px solid {BORDER};
            border-radius: 6px;
            min-height: 32px;
            font-size: 11px;
            font-weight: 800;
        }}

        QPushButton#secondaryButton:hover {{
            background: #F5F6F8;
        }}

        QPushButton#secondaryButton:checked {{
            background: #F4EAEA;
            color: {MAROON};
            border: 1px solid {MAROON};
        }}

        QTableWidget {{
            background: #1F1F1F;
            color: #FFFFFF;
            gridline-color: #3D3D3D;
            border: 1px solid #4B4B4B;
            border-radius: 8px;
            font-size: 12px;
        }}

        QTableWidget::item {{
            padding: 7px 8px;
        }}

        QTableWidget::item:selected {{
            background: #0B79D0;
            color: #FFFFFF;
        }}

        QHeaderView::section {{
            background: #343434;
            color: #FFFFFF;
            border: none;
            border-bottom: 1px solid #525252;
            padding: 8px;
            font-size: 12px;
            font-weight: 900;
        }}

        QLabel#metricsMessageBox {{
            background: #252525;
            color: #FFFFFF;
            border: 1px solid #4B4B4B;
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
        }}

        QPlainTextEdit#consoleText {{
            background: #171717;
            color: #F3F4F6;
            border: 1px solid #4B4B4B;
            border-radius: 8px;
            padding: 10px;
            font-family: Consolas, "Cascadia Mono", monospace;
            font-size: 11px;
        }}

        QScrollBar:vertical {{
            border: none;
            background: transparent;
            width: 5px;
        }}

        QScrollBar::handle:vertical {{
            background: {BORDER};
            border-radius: 2px;
        }}

        QScrollBar:horizontal {{
            height: 0px;
        }}
        """

