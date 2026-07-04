"""
Right-side configuration panel for the robotics simulator.

This module builds the UI panel and attaches all created widgets to the
MainWindow instance passed as ``window``.

It intentionally does not implement simulation behavior. It only creates
widgets and connects their signals to methods already owned by MainWindow /
SimulationControllerMixin.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from robotics_sim.simulation.config import *
from robotics_sim.app.widgets import (
    HeroHeader,
    NumericStepper,
    SectionCard,
    SliderValueRow,
    ToggleSwitch,
    make_icon,
)


def labeled_toggle(label: str, switch: ToggleSwitch):
    """
    Build a compact label + toggle switch row for boolean options.
    """
    box = QWidget()

    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)

    lbl = QLabel(label)
    lbl.setObjectName("fieldLabel")

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(switch)
    row.addStretch()

    layout.addWidget(lbl)
    layout.addLayout(row)

    return box



def labeled_combo(label: str, combo: QComboBox):
    """
    Build a labeled combo box used in configuration cards.

    The popup style is applied directly to the combo view because, on some
    Windows themes, the dropdown list ignores part of the global QSS and can
    show unreadable white text.
    """
    box = QWidget()

    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)

    lbl = QLabel(label)
    lbl.setObjectName("fieldLabel")

    combo.setMinimumHeight(34)
    combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    combo.view().setStyleSheet(f"""
        QListView {{
            background-color: #FFFFFF;
            color: #111827;
            border: 1px solid {BORDER};
            border-radius: 6px;
            padding: 4px;
            outline: 0px;
            selection-background-color: #F4EAEA;
            selection-color: {MAROON};
        }}

        QListView::item {{
            min-height: 28px;
            padding: 6px 8px;
            color: #111827;
            background-color: #FFFFFF;
        }}

        QListView::item:selected {{
            color: {MAROON};
            background-color: #F4EAEA;
        }}

        QListView::item:hover {{
            color: {MAROON};
            background-color: #FAF2F2;
        }}
    """)

    layout.addWidget(lbl)
    layout.addWidget(combo)

    return box

# ========================================================
# MULTI-ROBOT CONFIGURATION
# ========================================================



def build_config_panel(window):
    panel = QFrame()
    panel.setObjectName("sidePanel")
    panel.setFixedWidth(SIDE_PANEL_WIDTH)

    main_layout = QVBoxLayout(panel)
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    image_path = find_tamu_image()
    main_layout.addWidget(HeroHeader(image_path))

    scroll = QScrollArea()
    scroll.setObjectName("configScroll")
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setFrameShape(QFrame.NoFrame)

    content = QWidget()
    content.setObjectName("scrollContent")

    layout = QVBoxLayout(content)
    layout.setContentsMargins(9, 9, 9, 9)
    layout.setSpacing(9)

    # Simulation Options
    options_card = SectionCard("options", "Simulation Options")

    options_grid = QGridLayout()
    options_grid.setHorizontalSpacing(10)
    options_grid.setVerticalSpacing(9)

    window.state_switch = ToggleSwitch(True)

    window.planner_combo = QComboBox()
    window.planner_combo.addItems([
        "Direct",
        "A*",
        "Dijkstra",
        "RRT (future)",
    ])

    window.path_simplifier_combo = QComboBox()
    window.path_simplifier_combo.addItems(PATH_SIMPLIFIER_OPTIONS)
    window.path_simplifier_combo.setCurrentText(DEFAULT_PATH_SIMPLIFIER)

    window.exploration_planner_combo = QComboBox()
    window.exploration_planner_combo.addItems(EXPLORATION_PLANNER_OPTIONS)
    window.exploration_planner_combo.setCurrentText(DEFAULT_EXPLORATION_PLANNER)

    window.coordinator_combo = QComboBox()
    window.coordinator_combo.addItems(COORDINATOR_OPTIONS)
    window.coordinator_combo.setCurrentText(DEFAULT_COORDINATOR)

    window.control_combo = QComboBox()
    window.control_combo.addItems(["Nominal", "Adaptive"])

    window.vision_combo = QComboBox()
    window.vision_combo.addItems([
        "LiDAR",
        "Camera / FoV",
        "Omnidirectional",
    ])


    window.orders_switch = ToggleSwitch(False)
    window.obstacles_switch = ToggleSwitch(True)
    window.explored_area_switch = ToggleSwitch(True)

    # Visual toggles go first because they are runtime-safe and apply to
    # all robots, independent of whether robots share configuration.
    options_grid.addWidget(
        labeled_toggle("Robot Orders", window.orders_switch),
        0,
        0,
    )
    options_grid.addWidget(
        labeled_toggle("Show Obstacles", window.obstacles_switch),
        0,
        1,
    )
    options_grid.addWidget(
        labeled_toggle("Explored Area", window.explored_area_switch),
        1,
        0,
    )
    options_grid.addWidget(
        labeled_toggle("State Machine", window.state_switch),
        1,
        1,
    )
    options_grid.addWidget(
        labeled_combo("Planner", window.planner_combo),
        2,
        0,
    )
    options_grid.addWidget(
        labeled_combo("Control", window.control_combo),
        2,
        1,
    )
    options_grid.addWidget(
        labeled_combo("Vision Model", window.vision_combo),
        3,
        0,
        1,
        2,
    )
    window.path_simplifier_field = labeled_combo(
        "Path Simplifier",
        window.path_simplifier_combo,
    )
    options_grid.addWidget(
        window.path_simplifier_field,
        4,
        0,
        1,
        2,
    )

    window.exploration_planner_field = labeled_combo(
        "Exploration Planner",
        window.exploration_planner_combo,
    )
    options_grid.addWidget(
        window.exploration_planner_field,
        5,
        0,
        1,
        2,
    )

    window.coordinator_field = labeled_combo(
        "Multi-Robot Coordinator",
        window.coordinator_combo,
    )
    options_grid.addWidget(
        window.coordinator_field,
        6,
        0,
        1,
        2,
    )

    window.exploration_cooldown_input = NumericStepper(
        "Exploration Replan Cooldown (s)",
        window.config.exploration_replan_cooldown,
        0.00,
        10.00,
        0.25,
    )
    window.exploration_cooldown_field = window.exploration_cooldown_input
    options_grid.addWidget(
        window.exploration_cooldown_field,
        7,
        0,
        1,
        2,
    )

    window.ipp_lambda_input = NumericStepper(
        "IPP Distance Penalty λ",
        window.config.ipp_distance_penalty,
        0.00,
        5.00,
        0.05,
    )
    window.ipp_lambda_field = window.ipp_lambda_input
    options_grid.addWidget(
        window.ipp_lambda_field,
        8,
        0,
        1,
        2,
    )

    options_grid.setColumnStretch(0, 1)
    options_grid.setColumnStretch(1, 1)

    options_card.root.addLayout(options_grid)
    layout.addWidget(options_card)

    # Multi-robot setup. Hidden in Single Robot Mode. The global robot
    # parameters still define shared dynamics/sensing. This card only
    # manages initial poses and per-robot overrides for the selected robot.
    window.multi_robot_card = SectionCard("multi_robot", "Multi-Robot Setup")

    multi_grid = QGridLayout()
    multi_grid.setHorizontalSpacing(8)
    multi_grid.setVerticalSpacing(7)

    window.robot_count_input = NumericStepper("Robot Count", 3, 1, 8, 1, decimals=0)
    window.same_config_switch = ToggleSwitch(True)

    multi_grid.addWidget(window.robot_count_input, 0, 0)
    multi_grid.addWidget(labeled_toggle("Same Configuration", window.same_config_switch), 0, 1)
    multi_grid.setColumnStretch(0, 1)
    multi_grid.setColumnStretch(1, 1)
    window.multi_robot_card.root.addLayout(multi_grid)

    nav_row = QHBoxLayout()
    nav_row.setContentsMargins(0, 0, 0, 0)
    nav_row.setSpacing(8)

    window.prev_robot_button = QPushButton("‹")
    window.prev_robot_button.setObjectName("secondaryButton")
    window.prev_robot_button.setFixedWidth(42)
    window.next_robot_button = QPushButton("›")
    window.next_robot_button.setObjectName("secondaryButton")
    window.next_robot_button.setFixedWidth(42)
    window.selected_robot_label = QLabel("Robot 1 / 3")
    window.selected_robot_label.setObjectName("robotSelectionLabel")
    window.selected_robot_label.setAlignment(Qt.AlignCenter)

    nav_row.addWidget(window.prev_robot_button)
    nav_row.addWidget(window.selected_robot_label, 1)
    nav_row.addWidget(window.next_robot_button)
    window.robot_nav_widget = QWidget()
    window.robot_nav_widget.setLayout(nav_row)
    window.multi_robot_card.root.addWidget(window.robot_nav_widget)

    window.same_positions_widget = QWidget()
    window.same_positions_layout = QVBoxLayout(window.same_positions_widget)
    window.same_positions_layout.setContentsMargins(0, 0, 0, 0)
    window.same_positions_layout.setSpacing(7)
    window.same_position_inputs: list[tuple[QWidget, NumericStepper, NumericStepper]] = []

    for robot_index in range(8):
        row_widget = QWidget()
        row_layout = QGridLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setHorizontalSpacing(8)
        row_layout.setVerticalSpacing(5)

        x_stepper = NumericStepper(f"R{robot_index + 1} X", 0.0, WORLD_X_MIN, WORLD_X_MAX, 0.25)
        y_stepper = NumericStepper(f"R{robot_index + 1} Y", 0.0, WORLD_Y_MIN, WORLD_Y_MAX, 0.25)
        x_stepper.valueChanged.connect(window.sync_same_positions_from_panel)
        y_stepper.valueChanged.connect(window.sync_same_positions_from_panel)

        row_layout.addWidget(x_stepper, 0, 0)
        row_layout.addWidget(y_stepper, 0, 1)
        row_layout.setColumnStretch(0, 1)
        row_layout.setColumnStretch(1, 1)
        window.same_positions_layout.addWidget(row_widget)
        window.same_position_inputs.append((row_widget, x_stepper, y_stepper))

    window.multi_robot_card.root.addWidget(window.same_positions_widget)

    window.selected_robot_section_label = QLabel("Selected robot configuration")
    window.selected_robot_section_label.setObjectName("subsectionLabel")
    window.multi_robot_card.root.addWidget(window.selected_robot_section_label)

    window.multi_x_input = NumericStepper("Robot X (m)", -1.0, WORLD_X_MIN, WORLD_X_MAX, 0.25)
    window.multi_y_input = NumericStepper("Robot Y (m)", -0.6, WORLD_Y_MIN, WORLD_Y_MAX, 0.25)
    window.multi_theta_input = NumericStepper("Robot Theta", 0.0, -math.pi, math.pi, 0.1)
    window.multi_v_slider = SliderValueRow("Initial v", 0.0, 0.0, 5.0)
    window.multi_vision_slider = SliderValueRow("Vision Radius", window.config.vision, 0.25, 8.0)
    window.multi_body_radius_slider = SliderValueRow("Body Radius", window.config.body_radius, 0.05, 1.00)
    window.multi_safety_radius_slider = SliderValueRow("Safety Radius r", window.config.safety_radius, 0.05, 1.50)
    window.multi_max_speed_input = NumericStepper("Max Speed", window.config.max_speed, 0.1, 20.0, 0.1)
    window.multi_max_omega_input = NumericStepper("Max Angular Speed", window.config.max_angular_speed, 0.1, 20.0, 0.1)
    window.multi_max_accel_input = NumericStepper("Max Acceleration", window.config.max_acceleration, 0.1, 50.0, 0.1)
    window.multi_accel_gain_input = NumericStepper("Accel Gain k_a", window.config.acceleration_gain, 0.05, 5.0, 0.05)
    window.multi_goal_tol_input = NumericStepper("Goal Tolerance", window.config.goal_tolerance, 0.01, 5.0, 0.05)

    window.multi_position_row = QWidget()
    window.multi_position_row.setObjectName("selectedRobotConfigPanel")
    position_layout = QGridLayout(window.multi_position_row)
    position_layout.setContentsMargins(8, 8, 8, 8)
    position_layout.setHorizontalSpacing(8)
    position_layout.setVerticalSpacing(7)
    pose_title = QLabel("Pose")
    pose_title.setObjectName("subsectionLabel")
    position_layout.addWidget(pose_title, 0, 0, 1, 2)
    position_layout.addWidget(window.multi_x_input, 1, 0)
    position_layout.addWidget(window.multi_y_input, 1, 1)
    position_layout.addWidget(window.multi_theta_input, 2, 0)
    position_layout.addWidget(window.multi_v_slider, 2, 1)
    position_layout.setColumnStretch(0, 1)
    position_layout.setColumnStretch(1, 1)

    window.multi_sensing_row = QWidget()
    window.multi_sensing_row.setObjectName("selectedRobotConfigPanel")
    sensing_layout = QGridLayout(window.multi_sensing_row)
    sensing_layout.setContentsMargins(8, 8, 8, 8)
    sensing_layout.setHorizontalSpacing(8)
    sensing_layout.setVerticalSpacing(7)
    sensing_title = QLabel("Sensing and clearance")
    sensing_title.setObjectName("subsectionLabel")
    sensing_layout.addWidget(sensing_title, 0, 0, 1, 2)
    sensing_layout.addWidget(window.multi_vision_slider, 1, 0, 1, 2)
    sensing_layout.addWidget(window.multi_body_radius_slider, 2, 0)
    sensing_layout.addWidget(window.multi_safety_radius_slider, 2, 1)
    sensing_layout.setColumnStretch(0, 1)
    sensing_layout.setColumnStretch(1, 1)

    window.multi_dynamics_row = QWidget()
    window.multi_dynamics_row.setObjectName("selectedRobotConfigPanel")
    dynamics_layout = QGridLayout(window.multi_dynamics_row)
    dynamics_layout.setContentsMargins(8, 8, 8, 8)
    dynamics_layout.setHorizontalSpacing(8)
    dynamics_layout.setVerticalSpacing(7)
    dynamics_title = QLabel("Dynamics")
    dynamics_title.setObjectName("subsectionLabel")
    dynamics_layout.addWidget(dynamics_title, 0, 0, 1, 2)
    dynamics_layout.addWidget(window.multi_max_speed_input, 1, 0)
    dynamics_layout.addWidget(window.multi_max_omega_input, 1, 1)
    dynamics_layout.addWidget(window.multi_max_accel_input, 2, 0)
    dynamics_layout.addWidget(window.multi_accel_gain_input, 2, 1)
    dynamics_layout.addWidget(window.multi_goal_tol_input, 3, 0, 1, 2)
    dynamics_layout.setColumnStretch(0, 1)
    dynamics_layout.setColumnStretch(1, 1)

    # Backward-compatible alias used by older style/visibility code.
    window.multi_override_row = window.multi_sensing_row

    window.multi_robot_card.root.addWidget(window.multi_position_row)
    window.multi_robot_card.root.addWidget(window.multi_sensing_row)
    window.multi_robot_card.root.addWidget(window.multi_dynamics_row)

    hint = QLabel("Tip: before starting, drag robots directly on the map.")
    hint.setObjectName("fieldLabel")
    window.multi_robot_card.root.addWidget(hint)

    layout.insertWidget(1, window.multi_robot_card)

    # Robot Parameters
    window.robot_card = SectionCard("robot", "Robot Parameters")
    robot_card = window.robot_card

    pose_label = QLabel("Initial pose")
    pose_label.setObjectName("subsectionLabel")
    robot_card.root.addWidget(pose_label)

    robot_grid = QGridLayout()
    robot_grid.setHorizontalSpacing(8)
    robot_grid.setVerticalSpacing(7)

    window.x_input = NumericStepper("X Position (m)", window.config.x, WORLD_X_MIN, WORLD_X_MAX, 0.25)
    window.y_input = NumericStepper("Y Position (m)", window.config.y, WORLD_Y_MIN, WORLD_Y_MAX, 0.25)
    window.theta_input = NumericStepper("Theta (rad)", window.config.theta, -math.pi, math.pi, 0.1)

    robot_grid.addWidget(window.x_input, 0, 0)
    robot_grid.addWidget(window.y_input, 0, 1)
    robot_grid.addWidget(window.theta_input, 1, 0, 1, 2)
    robot_grid.setColumnStretch(0, 1)
    robot_grid.setColumnStretch(1, 1)

    robot_card.root.addLayout(robot_grid)

    motion_label = QLabel("Motion and sensing")
    motion_label.setObjectName("subsectionLabel")
    robot_card.root.addWidget(motion_label)

    motion_grid = QGridLayout()
    motion_grid.setHorizontalSpacing(8)
    motion_grid.setVerticalSpacing(7)

    window.v_slider = SliderValueRow("Initial Velocity (m/s)", window.config.v, 0.0, 5.0)
    window.vision_slider = SliderValueRow("Vision Radius (m)", window.config.vision, 0.25, 8.0)

    motion_grid.addWidget(window.v_slider, 0, 0)
    motion_grid.addWidget(window.vision_slider, 0, 1)
    motion_grid.setColumnStretch(0, 1)
    motion_grid.setColumnStretch(1, 1)

    robot_card.root.addLayout(motion_grid)

    radius_label = QLabel("Physical size and clearance")
    radius_label.setObjectName("subsectionLabel")
    robot_card.root.addWidget(radius_label)

    radius_grid = QGridLayout()
    radius_grid.setHorizontalSpacing(8)
    radius_grid.setVerticalSpacing(7)

    window.body_radius_slider = SliderValueRow("Body Radius (m)", window.config.body_radius, 0.05, 1.00)
    window.safety_radius_slider = SliderValueRow("Safety Radius r (m)", window.config.safety_radius, 0.05, 1.50)

    radius_grid.addWidget(window.body_radius_slider, 0, 0)
    radius_grid.addWidget(window.safety_radius_slider, 0, 1)
    radius_grid.setColumnStretch(0, 1)
    radius_grid.setColumnStretch(1, 1)

    robot_card.root.addLayout(radius_grid)
    layout.addWidget(robot_card)

    # Dynamics
    window.dynamics_card = SectionCard("dynamics", "Dynamics & Limits")
    dynamics_card = window.dynamics_card

    dynamics_grid = QGridLayout()
    dynamics_grid.setHorizontalSpacing(8)
    dynamics_grid.setVerticalSpacing(7)

    window.max_speed_input = NumericStepper("Max Speed (m/s)", window.config.max_speed, 0.1, 20.0, 0.1)
    window.max_omega_input = NumericStepper("Max Angular Speed", window.config.max_angular_speed, 0.1, 20.0, 0.1)
    window.max_accel_input = NumericStepper("Max Acceleration", window.config.max_acceleration, 0.1, 50.0, 0.1)
    window.accel_gain_input = NumericStepper("Accel Gain k_a", window.config.acceleration_gain, 0.05, 5.0, 0.05)
    window.goal_tol_input = NumericStepper("Goal Tolerance", window.config.goal_tolerance, 0.01, 5.0, 0.05)

    dynamics_grid.addWidget(window.max_speed_input, 0, 0)
    dynamics_grid.addWidget(window.max_omega_input, 0, 1)
    dynamics_grid.addWidget(window.max_accel_input, 1, 0)
    dynamics_grid.addWidget(window.accel_gain_input, 1, 1)
    dynamics_grid.addWidget(window.goal_tol_input, 2, 0, 1, 2)
    dynamics_grid.setColumnStretch(0, 1)
    dynamics_grid.setColumnStretch(1, 1)

    dynamics_card.root.addLayout(dynamics_grid)
    layout.addWidget(dynamics_card)

    # Goal
    window.goal_card = SectionCard("goal", "Goal Setup")
    goal_card = window.goal_card

    goal_grid = QGridLayout()
    goal_grid.setHorizontalSpacing(8)
    goal_grid.setVerticalSpacing(7)

    window.goal_x_input = NumericStepper("Goal X (m)", window.config.goal_x, WORLD_X_MIN, WORLD_X_MAX, 0.25)
    window.goal_y_input = NumericStepper("Goal Y (m)", window.config.goal_y, WORLD_Y_MIN, WORLD_Y_MAX, 0.25)

    window.preview_switch = ToggleSwitch(True)

    goal_grid.addWidget(window.goal_x_input, 0, 0)
    goal_grid.addWidget(window.goal_y_input, 0, 1)
    goal_grid.addWidget(labeled_toggle("Goal Preview", window.preview_switch), 1, 0, 1, 2)
    goal_grid.setColumnStretch(0, 1)
    goal_grid.setColumnStretch(1, 1)

    goal_card.root.addLayout(goal_grid)
    layout.addWidget(goal_card)

    layout.addStretch()
    scroll.setWidget(content)

    main_layout.addWidget(scroll, 1)

    # Bottom actions
    actions = QFrame()
    actions.setObjectName("actionPanelBottom")

    actions_layout = QVBoxLayout(actions)
    actions_layout.setContentsMargins(12, 12, 12, 12)
    actions_layout.setSpacing(8)

    window.start_button = QPushButton("Start Simulation")
    window.start_button.setObjectName("startButton")
    window.start_button.setIcon(make_icon("play", "white"))
    window.start_button.setIconSize(QSize(22, 22))
    window.start_button.clicked.connect(window.handle_start_pause_button)

    bottom_buttons = QHBoxLayout()
    bottom_buttons.setSpacing(8)

    file_buttons = QHBoxLayout()
    file_buttons.setSpacing(8)

    window.reset_button = QPushButton("Restart")
    window.reset_button.setObjectName("secondaryButton")
    window.reset_button.setIcon(make_icon("reset", TEXT))
    window.reset_button.setIconSize(QSize(18, 18))
    window.reset_button.clicked.connect(window.restart_simulation)

    window.speed_button = QPushButton(f"Speed {window.simulation_speed:.2f}x")
    window.speed_button.setObjectName("secondaryButton")
    window.speed_button.setIcon(make_icon("gear", TEXT))
    window.speed_button.setIconSize(QSize(18, 18))
    window.speed_button.clicked.connect(window.cycle_simulation_speed)

    window.metrics_button = QPushButton("Metrics")
    window.metrics_button.setObjectName("secondaryButton")
    window.metrics_button.setIcon(make_icon("maximize", TEXT))
    window.metrics_button.setIconSize(QSize(18, 18))
    window.metrics_button.clicked.connect(window.open_metrics_window)

    window.console_button = QPushButton("Console")
    window.console_button.setObjectName("secondaryButton")
    window.console_button.setIcon(make_icon("console", TEXT))
    window.console_button.setIconSize(QSize(18, 18))
    window.console_button.clicked.connect(window.open_console_window)

    window.load_button = QPushButton("Load .sim")
    window.load_button.setObjectName("secondaryButton")
    window.load_button.setIcon(make_icon("reset", TEXT))
    window.load_button.setIconSize(QSize(18, 18))
    window.load_button.clicked.connect(window.load_simulation_config)

    window.save_button = QPushButton("Save .sim")
    window.save_button.setObjectName("secondaryButton")
    window.save_button.setIcon(make_icon("save", TEXT))
    window.save_button.setIconSize(QSize(18, 18))
    window.save_button.clicked.connect(window.save_simulation_config)

    monitor_buttons = QHBoxLayout()
    monitor_buttons.setSpacing(8)

    bottom_buttons.addWidget(window.reset_button)
    bottom_buttons.addWidget(window.speed_button)

    monitor_buttons.addWidget(window.metrics_button)
    monitor_buttons.addWidget(window.console_button)

    file_buttons.addWidget(window.load_button)
    file_buttons.addWidget(window.save_button)

    actions_layout.addWidget(window.start_button)
    actions_layout.addLayout(bottom_buttons)
    actions_layout.addLayout(monitor_buttons)
    actions_layout.addLayout(file_buttons)

    main_layout.addWidget(actions)

    # Signals
    numeric_widgets = [
        window.x_input,
        window.y_input,
        window.theta_input,
        window.max_speed_input,
        window.max_omega_input,
        window.max_accel_input,
        window.goal_tol_input,
        window.accel_gain_input,
        window.exploration_cooldown_input,
        window.ipp_lambda_input,
        window.goal_x_input,
        window.goal_y_input,
        window.v_slider,
        window.vision_slider,
        window.body_radius_slider,
        window.safety_radius_slider,
        window.robot_count_input,
        window.multi_x_input,
        window.multi_y_input,
        window.multi_theta_input,
        window.multi_v_slider,
        window.multi_vision_slider,
        window.multi_body_radius_slider,
        window.multi_safety_radius_slider,
        window.multi_max_speed_input,
        window.multi_max_omega_input,
        window.multi_max_accel_input,
        window.multi_accel_gain_input,
        window.multi_goal_tol_input,
    ]

    for widget in numeric_widgets:
        widget.valueChanged.connect(window.update_preview)

    window.preview_switch.toggled.connect(window.update_preview)
    window.state_switch.toggled.connect(window.update_preview)
    window.planner_combo.currentTextChanged.connect(window.update_preview)
    window.path_simplifier_combo.currentTextChanged.connect(window.update_preview)
    window.exploration_planner_combo.currentTextChanged.connect(window.update_preview)
    window.coordinator_combo.currentTextChanged.connect(window.update_preview)
    window.vision_combo.currentTextChanged.connect(window.update_preview)
    window.orders_switch.toggled.connect(window.update_preview)
    window.obstacles_switch.toggled.connect(window.update_preview)
    window.explored_area_switch.toggled.connect(window.update_preview)
    window.body_radius_slider.valueChanged.connect(window.enforce_radius_consistency)
    window.safety_radius_slider.valueChanged.connect(window.enforce_radius_consistency)
    window.top_bar.mode_selector.currentTextChanged.connect(window.on_agent_mode_changed)
    window.robot_count_input.valueChanged.connect(window.on_robot_count_changed)
    window.same_config_switch.toggled.connect(window.on_same_config_toggled)
    window.prev_robot_button.clicked.connect(window.select_previous_robot)
    window.next_robot_button.clicked.connect(window.select_next_robot)
    window.multi_x_input.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_y_input.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_theta_input.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_v_slider.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_vision_slider.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_body_radius_slider.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_safety_radius_slider.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_max_speed_input.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_max_omega_input.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_max_accel_input.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_accel_gain_input.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_goal_tol_input.valueChanged.connect(window.sync_selected_robot_from_panel)
    window.multi_body_radius_slider.valueChanged.connect(window.enforce_selected_multi_radius_consistency)
    window.multi_safety_radius_slider.valueChanged.connect(window.enforce_selected_multi_radius_consistency)

    # Controls that define the initial conditions, algorithms, and physical
    # model are locked while a simulation is active. Display-only controls
    # such as Robot Orders, Show Obstacles, Explored Area, Metrics, and
    # Speed stay available because they do not invalidate the running state.
    window.locked_during_run_widgets = [
        window.top_bar.mode_selector,
        window.state_switch,
        window.planner_combo,
        window.path_simplifier_combo,
        window.exploration_planner_combo,
        window.coordinator_combo,
        window.control_combo,
        window.vision_combo,
        window.robot_count_input,
        window.same_config_switch,
        window.prev_robot_button,
        window.next_robot_button,
        window.multi_x_input,
        window.multi_y_input,
        window.multi_theta_input,
        window.multi_v_slider,
        window.multi_vision_slider,
        window.multi_body_radius_slider,
        window.multi_safety_radius_slider,
        window.multi_max_speed_input,
        window.multi_max_omega_input,
        window.multi_max_accel_input,
        window.multi_accel_gain_input,
        window.multi_goal_tol_input,
        window.exploration_cooldown_input,
        window.ipp_lambda_input,
        window.x_input,
        window.y_input,
        window.theta_input,
        window.v_slider,
        window.vision_slider,
        window.body_radius_slider,
        window.safety_radius_slider,
        window.max_speed_input,
        window.max_omega_input,
        window.max_accel_input,
        window.accel_gain_input,
        window.goal_tol_input,
        window.goal_x_input,
        window.goal_y_input,
        window.load_button,
        window.top_bar.single_mode_button,
        window.top_bar.multi_mode_button,
    ]
    for _, x_stepper, y_stepper in window.same_position_inputs:
        window.locked_during_run_widgets.extend([x_stepper, y_stepper])

    window.update_relevant_parameter_visibility()
    window.set_configuration_locked(False)
    window.update_preview()

    return panel


