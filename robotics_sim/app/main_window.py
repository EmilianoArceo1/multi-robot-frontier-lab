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
from robotics_sim.simulation.navigation_modes import is_exploration_planner
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
        self.timer.timeout.connect(self.simulation_step)
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
        self.canvas.goalClicked.connect(self.set_goal_from_canvas)
        self.canvas.robotDragged.connect(self.move_robot_from_canvas)
        self.canvas.robotSelected.connect(self.select_robot_panel)

        panel = self.build_config_panel()

        body_layout.addWidget(self.canvas, 1)
        body_layout.addWidget(panel, 0)

        outer.addWidget(body, 1)

    def build_config_panel(self):
        """Build the right-side configuration panel.

        The actual panel construction lives in robotics_sim.app.config_panel so
        MainWindow stays focused on the top-level app window and Qt signal
        wiring.
        """
        return build_right_config_panel(self)

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

    def set_configuration_locked(self, locked: bool) -> None:
        """
        Disable simulation-defining controls while a run is active.

        Pausing does not unlock them. A paused run still has a robot, map,
        planner state, async route requests, and metrics tied to the current
        configuration. Restart/reset returns the UI to an editable state.
        """
        widgets = getattr(self, "locked_during_run_widgets", [])
        for widget in widgets:
            widget.setEnabled(not locked)

        # Keep visibility rules active even while the controls are disabled.
        self.update_relevant_parameter_visibility()

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

