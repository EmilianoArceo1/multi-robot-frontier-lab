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

from PySide6.QtCore import QSize, Qt, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QBrush, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QColorDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from robotics_sim.simulation.config import *
from robotics_sim.app.theme import ThemeMode, dropdown_popup_stylesheet, theme_colors
from robotics_sim.app.widgets import (
    HeroHeader,
    NumericStepper,
    SectionCard,
    SliderValueRow,
    SteppedSliderRow,
    ToggleSwitch,
    make_icon,
)


class ColorPickerButton(QPushButton):
    """Compact RGB selector that stores a normalized #RRGGBB value."""

    colorChanged = Signal(str)

    def __init__(self, color: str, dialog_title: str):
        super().__init__()
        self._color = QColor(color)
        if not self._color.isValid():
            self._color = QColor("#000000")
        self._dialog_title = str(dialog_title)
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.clicked.connect(self._choose_color)
        self._refresh_swatch()

    def color_hex(self) -> str:
        return self._color.name().upper()

    def set_color(self, color: str) -> None:
        candidate = QColor(str(color))
        if not candidate.isValid():
            return
        normalized = candidate.name().upper()
        changed = normalized != self.color_hex()
        self._color = candidate
        self._refresh_swatch()
        if changed:
            self.colorChanged.emit(normalized)

    def _choose_color(self) -> None:
        selected = QColorDialog.getColor(self._color, self, self._dialog_title)
        if selected.isValid():
            self.set_color(selected.name())

    def _refresh_swatch(self) -> None:
        swatch = QPixmap(20, 20)
        swatch.fill(Qt.transparent)
        painter = QPainter(swatch)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(96, 101, 112), 1.0))
        painter.setBrush(QBrush(self._color))
        painter.drawRoundedRect(QRectF(1.0, 1.0, 18.0, 18.0), 3.0, 3.0)
        painter.end()
        self.setIcon(QIcon(swatch))
        self.setIconSize(QSize(20, 20))
        self.setText(self.color_hex())


class BrushSizePreview(QWidget):
    """Small visual brush-size selector preview used by the map editor panel."""

    def __init__(self, brush_size: float = 0.2):
        super().__init__()
        self.brush_size = float(brush_size)
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Starts light, corrected by MainWindow._apply_theme() (which finds
        # every BrushSizePreview via findChildren()) once the saved theme is
        # known -- same pattern as ToggleSwitch.set_theme_mode().
        self._theme_mode = ThemeMode.LIGHT

    def set_brush_size(self, brush_size: float) -> None:
        self.brush_size = max(0.05, float(brush_size))
        self.update()

    def set_theme_mode(self, mode) -> None:
        self._theme_mode = ThemeMode(mode)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        colors = theme_colors(self._theme_mode)

        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        painter.setPen(QPen(QColor(colors.border), 1.0))
        painter.setBrush(QBrush(QColor(colors.card_background)))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        center_x = rect.left() + 32.0
        center_y = rect.center().y()
        radius = max(4.0, min(19.0, 4.0 + self.brush_size * 8.0))

        # The brush swatch itself stays the brand maroon in both themes --
        # it previews a paint/erase brush color choice, not app chrome.
        painter.setPen(QPen(QColor(MAROON), 1.6))
        painter.setBrush(QBrush(QColor(122, 0, 25, 48)))
        painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2.0, radius * 2.0))

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(colors.text_primary))
        painter.drawText(
            QRectF(center_x + 30.0, rect.top(), rect.width() - 68.0, rect.height()),
            Qt.AlignVCenter | Qt.AlignLeft,
            f"Brush {self.brush_size:.2f} m",
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



def labeled_color_picker(label: str, picker: ColorPickerButton):
    box = QWidget()

    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)

    lbl = QLabel(label)
    lbl.setObjectName("fieldLabel")

    layout.addWidget(lbl)
    layout.addWidget(picker)
    return box


def labeled_combo(label: str, combo: QComboBox):
    """
    Build a labeled combo box used in configuration cards.

    The popup style is applied directly to the combo view because, on some
    Windows themes, the dropdown list ignores part of the global QSS and can
    show unreadable white text. Built light here (matches the app's own
    light-first default -- see MainWindow._load_saved_theme()); `combo` is
    tagged themedDropdownPopup=True so MainWindow._apply_theme() can find
    and re-style every such popup when the theme changes (its own
    setStyleSheet() is not reachable by the ordinary app-level QSS cascade).
    """
    box = QWidget()

    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)

    lbl = QLabel(label)
    lbl.setObjectName("fieldLabel")

    combo.setMinimumHeight(34)
    combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    combo.setProperty("themedDropdownPopup", True)
    combo.view().setStyleSheet(dropdown_popup_stylesheet(ThemeMode.LIGHT))

    layout.addWidget(lbl)
    layout.addWidget(combo)

    # Stashed so callers can update the visible label later (e.g. marking
    # "Frontier Algorithm Detector" as algorithm-provided/fallback) without needing to
    # change this helper's signature or rebuild the widget tree.
    box.field_label = lbl
    box.field_label_base_text = label

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
    window.hero_header = HeroHeader(image_path)
    main_layout.addWidget(window.hero_header)

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
    window.simulation_options_grid = options_grid
    options_grid.setHorizontalSpacing(10)
    options_grid.setVerticalSpacing(9)

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
    window.exploration_planner_combo.addItems(FRONTIER_ALGORITHM_DETECTOR_OPTIONS)
    if DEFAULT_EXPLORATION_PLANNER in FRONTIER_ALGORITHM_DETECTOR_OPTIONS:
        window.exploration_planner_combo.setCurrentText(DEFAULT_EXPLORATION_PLANNER)

    window.clustering_algorithm_combo = QComboBox()
    window.clustering_algorithm_combo.addItems(
        [NO_CLUSTERING_ALGORITHM, *CLUSTERING_ALGORITHM_OPTIONS]
    )
    window.clustering_algorithm_combo.setPlaceholderText(NO_CLUSTERING_ALGORITHM)
    window.clustering_algorithm_combo.setCurrentText(NO_CLUSTERING_ALGORITHM)
    window.clustering_algorithm_combo.setToolTip(
        "Groups the cells produced by the frontier detector. Only algorithms "
        "implemented from a citable paper are listed; there is no implicit default."
    )

    window.coordinator_combo = QComboBox()
    window.coordinator_combo.addItems(TASK_ASSIGN_ALGORITHM_OPTIONS)
    window.coordinator_combo.setPlaceholderText(NO_TASK_ASSIGN_ALGORITHM)
    if TASK_ASSIGN_ALGORITHM_OPTIONS:
        window.coordinator_combo.setCurrentIndex(0)
    else:
        window.coordinator_combo.setCurrentIndex(-1)
        window.coordinator_combo.setEnabled(False)

    window.safety_algorithm_combo = QComboBox()
    window.safety_algorithm_combo.addItems(SAFETY_ALGORITHM_OPTIONS)
    window.safety_algorithm_combo.setCurrentText(WANG_AMES_BARRIER_CERTIFICATE)
    window.safety_algorithm_combo.setToolTip(
        "Wang, Ames & Egerstedt, IEEE Transactions on Robotics 33(3), 2017."
    )

    window.vision_combo = QComboBox()
    window.vision_combo.addItems([
        "LiDAR",
        "Camera / FoV",
        "Omnidirectional",
    ])

    window.map_visualization_combo = QComboBox()
    window.map_visualization_combo.addItems(MAP_VISUALIZATION_OPTIONS)
    window.map_visualization_combo.setCurrentText(DEFAULT_MAP_VISUALIZATION)

    window.custom_unexplored_color_button = ColorPickerButton(
        DEFAULT_CUSTOM_UNEXPLORED_COLOR,
        "Choose unexplored-area color",
    )
    window.custom_explored_color_button = ColorPickerButton(
        DEFAULT_CUSTOM_EXPLORED_COLOR,
        "Choose explored-area color",
    )
    window.custom_obstacle_color_button = ColorPickerButton(
        DEFAULT_CUSTOM_OBSTACLE_COLOR,
        "Choose obstacle color",
    )
    window.custom_obstacle_color_button.setToolTip(
        "Visual color for physical obstacles in Custom Discovery; mapped samples keep their own color."
    )
    window.custom_explored_opacity_input = SteppedSliderRow(
        "Explored opacity",
        DEFAULT_CUSTOM_EXPLORED_OPACITY * 100.0,
        0.0,
        100.0,
        5.0,
        decimals=0,
        unit_suffix="%",
    )
    window.custom_explored_opacity_input.setToolTip(
        "Opacity of the discovered free-space layer in Custom Discovery."
    )
    window.mapped_obstacle_line_width_input = SteppedSliderRow(
        "Mapped obstacle width",
        DEFAULT_MAPPED_OBSTACLE_LINE_WIDTH,
        0.25,
        6.0,
        0.25,
        decimals=2,
        unit_suffix="px",
    )
    window.mapped_obstacle_line_width_input.setToolTip(
        "Screen-space thickness of discovered obstacle boundary samples; planning is unchanged."
    )

    window.robot_icon_combo = QComboBox()
    window.robot_icon_combo.addItems(ROBOT_ICON_OPTIONS)
    window.robot_icon_combo.setCurrentText(DEFAULT_ROBOT_ICON)

    window.obstacles_switch = ToggleSwitch(True)
    window.explored_area_switch = ToggleSwitch(True)
    window.traveled_path_switch = ToggleSwitch(False)
    window.planned_route_switch = ToggleSwitch(True)
    window.traveled_path_switch.setToolTip(
        "Draw the accumulated trajectory already traveled by each robot."
    )
    window.planned_route_switch.setToolTip(
        "Draw the remaining accepted route, its waypoints, and the selected frontier."
    )

    # Visual toggles go first because they are runtime-safe and apply to
    # all robots, independent of whether robots share configuration.
    # "Robot Orders"/"State Machine"/"Motion Control Service" were removed:
    # the first two are superseded by the Navigation Debug eye icon on the
    # canvas (see simulation_canvas.py), and Motion Control Service's
    # Nominal/Adaptive choice was never read anywhere -- confirmed dead,
    # decorative UI with zero effect on simulation behavior.
    options_grid.addWidget(
        labeled_toggle("Show Obstacles", window.obstacles_switch),
        0,
        0,
    )
    options_grid.addWidget(
        labeled_toggle("Explored Area", window.explored_area_switch),
        0,
        1,
    )
    options_grid.addWidget(
        labeled_toggle("Traveled Route", window.traveled_path_switch),
        1,
        0,
    )
    options_grid.addWidget(
        labeled_toggle("Planned Route + Frontier", window.planned_route_switch),
        1,
        1,
    )
    options_grid.addWidget(
        labeled_combo("Map Visualization", window.map_visualization_combo),
        2,
        0,
        1,
        2,
    )
    window.custom_unexplored_color_field = labeled_color_picker(
        "Unexplored Color",
        window.custom_unexplored_color_button,
    )
    window.custom_explored_color_field = labeled_color_picker(
        "Explored Color",
        window.custom_explored_color_button,
    )
    window.custom_obstacle_color_field = labeled_color_picker(
        "Obstacle Color",
        window.custom_obstacle_color_button,
    )
    window.custom_explored_opacity_field = window.custom_explored_opacity_input
    options_grid.addWidget(window.custom_unexplored_color_field, 3, 0)
    options_grid.addWidget(window.custom_explored_color_field, 3, 1)
    options_grid.addWidget(window.custom_obstacle_color_field, 4, 0)
    options_grid.addWidget(window.custom_explored_opacity_field, 4, 1)
    options_grid.addWidget(window.mapped_obstacle_line_width_input, 5, 0, 1, 2)
    options_grid.addWidget(
        labeled_combo("Robot Icon", window.robot_icon_combo),
        6,
        0,
        1,
        2,
    )
    window.path_planner_field = labeled_combo(
        "Path Planner Service",
        window.planner_combo,
    )
    options_grid.addWidget(
        window.path_planner_field,
        7,
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
        13,
        0,
        1,
        2,
    )

    window.clustering_algorithm_field = labeled_combo(
        "Clustering Algorithm",
        window.clustering_algorithm_combo,
    )
    options_grid.addWidget(
        window.clustering_algorithm_field,
        8,
        0,
        1,
        2,
    )

    window.exploration_planner_field = labeled_combo(
        "Frontier Algorithm Detector",
        window.exploration_planner_combo,
    )
    options_grid.addWidget(
        window.exploration_planner_field,
        9,
        0,
        1,
        2,
    )

    window.coordinator_field = labeled_combo(
        "Task Assign Algorithm",
        window.coordinator_combo,
    )
    options_grid.addWidget(
        window.coordinator_field,
        10,
        0,
        1,
        2,
    )

    window.vision_model_field = labeled_combo(
        "Vision Model",
        window.vision_combo,
    )
    options_grid.addWidget(window.vision_model_field, 11, 0, 1, 2)

    window.camera_fov_input = NumericStepper(
        "Camera FoV (degrees)",
        window.config.camera_fov_degrees,
        10.0,
        360.0,
        5.0,
    )
    window.camera_fov_field = window.camera_fov_input
    options_grid.addWidget(window.camera_fov_field, 12, 0, 1, 2)

    window.safety_algorithm_field = labeled_combo(
        "Safety Algorithm",
        window.safety_algorithm_combo,
    )
    options_grid.addWidget(window.safety_algorithm_field, 14, 0, 1, 2)

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
        15,
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
        15,
        0,
        1,
        2,
    )

    # Planning grid cell size. Not a planning-algorithm change -- purely
    # exposes the existing SimulationConfig.grid_resolution field (already
    # used throughout the planning grid / reachability pipeline and already
    # serialized in .sim files) so 0.50 vs 0.25 m/cell can be compared via
    # manual A/B testing without hand-editing a scenario file.
    window.grid_resolution_input = SteppedSliderRow(
        "Grid resolution",
        window.config.grid_resolution,
        0.10,
        1.00,
        0.05,
        unit_suffix="m/cell",
    )
    window.grid_resolution_field = window.grid_resolution_input
    options_grid.addWidget(
        window.grid_resolution_field,
        16,
        0,
        1,
        2,
    )

    # "Show Grid" is a rendering-only toggle: it keeps the (otherwise
    # auto-hiding) grid preview visible persistently and, while the
    # simulation is running, layers translucent occupied/free/unknown cell
    # colors on top. It never touches SimulationConfig, so it is
    # deliberately left out of both numeric_widgets and
    # locked_during_run_widgets below -- it must stay interactive while a
    # simulation is running.
    window.grid_overlay_toggle = ToggleSwitch(False)
    options_grid.addWidget(
        labeled_toggle("Show Grid", window.grid_overlay_toggle),
        17,
        0,
    )
    window.grid_cell_values_toggle = ToggleSwitch(False)
    window.frontier_decisions_toggle = ToggleSwitch(False)
    window.grid_cell_values_toggle.setToolTip(
        "Show occupancy values: -1 unknown, 0 discovered/free, 1 occupied."
    )
    window.frontier_decisions_toggle.setToolTip(
        "Show BFS levels from the selected robot and highlight frontier cells as F:level."
    )
    options_grid.addWidget(
        labeled_toggle("Cell Values", window.grid_cell_values_toggle),
        18,
        0,
    )
    options_grid.addWidget(
        labeled_toggle("Frontier Decisions", window.frontier_decisions_toggle),
        18,
        1,
    )

    # "Hazard Map" / "Fire Markers" are two independent rendering-only
    # ground-truth DEBUG overlays, both default OFF: what the team has
    # actually discovered (the warm hazard heatmap and its discovered fire
    # sources) is ALWAYS visible regardless of these -- turning a toggle ON
    # only ADDS the full ground-truth layer on top, it never hides
    # discovered information (see SimulationCanvas.draw_ground_truth_
    # hazard_map()/draw_fire_markers()'s own docstrings). Neither touches
    # SimulationConfig/HazardBelief/planning, so both are deliberately left
    # out of numeric_widgets and locked_during_run_widgets below and stay
    # interactive while a simulation is running -- see MainWindow.
    # on_hazard_map_toggled()/on_fire_markers_toggled().
    window.hazard_map_toggle = ToggleSwitch(False)
    window.hazard_map_toggle.setToolTip(
        "Show the complete ground-truth hazard field. Visualization only."
    )
    options_grid.addWidget(
        labeled_toggle("Hazard Map", window.hazard_map_toggle),
        17,
        1,
    )
    window.fire_markers_toggle = ToggleSwitch(False)
    window.fire_markers_toggle.setToolTip(
        "Show all fire sources. Discovered sources remain visible when off."
    )
    options_grid.addWidget(
        labeled_toggle("Fire Markers", window.fire_markers_toggle),
        19,
        0,
    )
    window.cursor_coordinates_toggle = ToggleSwitch(True)
    window.cursor_coordinates_toggle.setToolTip(
        "Show the world (x, y) coordinate beside the mouse over the map."
    )
    options_grid.addWidget(
        labeled_toggle("Mouse Coordinates", window.cursor_coordinates_toggle),
        19,
        1,
    )

    # "Navigation Debug" is controlled solely by the Navigation switch in
    # main_window._build_navigation_snapshot_bar() (docked above the canvas
    # header), not by a side-panel control -- a single, always-visible
    # toggle instead of two controls for the same state.

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
    window.multi_vision_slider = SliderValueRow("Vision Radius", window.config.vision, 0.25, 20.0)
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
    window.vision_slider = SliderValueRow("Vision Radius (m)", window.config.vision, 0.25, 20.0)

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

    # Runtime actions live in the canvas action bar. The configuration panel
    # now contains only configuration fields; Load/Save are exposed from the
    # top-bar gear menu.

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
        window.camera_fov_input,
        window.grid_resolution_input,
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
    # Editor controls are created in build_editor_panel(). Do not create the
    # editor tool combo here; doing so would leave the editor with wrong options.
    window.planner_combo.currentTextChanged.connect(window.update_preview)
    window.path_simplifier_combo.currentTextChanged.connect(window.update_preview)
    window.exploration_planner_combo.currentTextChanged.connect(window.update_preview)
    window.clustering_algorithm_combo.currentTextChanged.connect(window.update_preview)
    window.coordinator_combo.currentTextChanged.connect(window.update_preview)
    window.coordinator_combo.currentTextChanged.connect(
        window.update_mapping_architecture_badge
    )
    window.coordinator_combo.currentTextChanged.connect(
        window.apply_task_assignment_dependencies
    )
    window.safety_algorithm_combo.currentTextChanged.connect(window.update_preview)
    window.vision_combo.currentTextChanged.connect(window.update_preview)
    def update_custom_color_visibility(mode: str) -> None:
        visible = str(mode) == "Custom Discovery"
        window.custom_unexplored_color_field.setVisible(visible)
        window.custom_explored_color_field.setVisible(visible)
        window.custom_obstacle_color_field.setVisible(visible)
        window.custom_explored_opacity_field.setVisible(visible)

    window.map_visualization_combo.currentTextChanged.connect(update_custom_color_visibility)
    window.map_visualization_combo.currentTextChanged.connect(window.update_preview)
    window.custom_unexplored_color_button.colorChanged.connect(window.update_preview)
    window.custom_explored_color_button.colorChanged.connect(window.update_preview)
    window.custom_obstacle_color_button.colorChanged.connect(window.update_preview)
    window.custom_explored_opacity_input.valueChanged.connect(window.update_preview)
    window.mapped_obstacle_line_width_input.valueChanged.connect(window.update_preview)
    update_custom_color_visibility(window.map_visualization_combo.currentText())
    window.robot_icon_combo.currentTextChanged.connect(window.update_preview)
    window.obstacles_switch.toggled.connect(window.update_preview)
    window.explored_area_switch.toggled.connect(window.update_preview)
    window.traveled_path_switch.toggled.connect(window.update_preview)
    window.planned_route_switch.toggled.connect(window.update_preview)
    window.body_radius_slider.valueChanged.connect(window.enforce_radius_consistency)
    window.safety_radius_slider.valueChanged.connect(window.enforce_radius_consistency)
    window.grid_resolution_input.valueChanged.connect(window.on_grid_resolution_control_changed)
    window.grid_overlay_toggle.toggled.connect(window.on_grid_overlay_toggled)
    window.grid_cell_values_toggle.toggled.connect(window.on_grid_cell_values_toggled)
    window.frontier_decisions_toggle.toggled.connect(window.on_frontier_decisions_toggled)
    window.hazard_map_toggle.toggled.connect(window.on_hazard_map_toggled)
    window.fire_markers_toggle.toggled.connect(window.on_fire_markers_toggled)
    window.cursor_coordinates_toggle.toggled.connect(window.on_cursor_coordinates_toggled)
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
    # such as Show Obstacles, Explored Area, Metrics, and Speed stay
    # available because they do not invalidate the running state.
    window.locked_during_run_widgets = [
        window.top_bar.mode_selector,
        window.planner_combo,
        window.path_simplifier_combo,
        window.exploration_planner_combo,
        window.clustering_algorithm_combo,
        window.coordinator_combo,
        window.safety_algorithm_combo,
        window.vision_combo,
        window.camera_fov_input,
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
        window.grid_resolution_input,
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
        window.top_bar.single_mode_button,
        window.top_bar.multi_mode_button,
    ]
    for _, x_stepper, y_stepper in window.same_position_inputs:
        window.locked_during_run_widgets.extend([x_stepper, y_stepper])

    window.update_relevant_parameter_visibility()
    window.set_configuration_locked(False)
    window.update_preview()

    return panel


def build_editor_panel(window):
    panel = QFrame()
    panel.setObjectName("sidePanel")
    panel.setFixedWidth(SIDE_PANEL_WIDTH)

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    layout.addWidget(HeroHeader(find_tamu_image()))

    scroll = QScrollArea()
    scroll.setObjectName("configScroll")
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setFrameShape(QFrame.NoFrame)

    content = QWidget()
    content.setObjectName("scrollContent")
    content_layout = QVBoxLayout(content)
    content_layout.setContentsMargins(9, 9, 9, 9)
    content_layout.setSpacing(9)

    # --------------------------------------------------------
    # Editor tools
    # --------------------------------------------------------
    editor_card = SectionCard("editor", "Map Editor")
    editor_grid = QGridLayout()
    editor_grid.setHorizontalSpacing(8)
    editor_grid.setVerticalSpacing(7)

    window.editor_status_label = QLabel("Editor disabled")
    # Its own object name (not the generic "fieldLabel") because it wants a
    # larger font-size/line-height for its longer wrapped sentence -- styled
    # via the global stylesheet's QLabel#editorStatusLabel rule (see
    # theme.py) so it stays correctly themed after a theme toggle without
    # its own propagation call.
    window.editor_status_label.setObjectName("editorStatusLabel")
    window.editor_status_label.setWordWrap(True)

    window.editor_tool_combo = QComboBox()
    window.editor_tool_combo.addItems(["Rectangles", "Squares", "Free draw", "Erase", "Camera view"])
    window.editor_tool_combo.setCurrentText("Rectangles")
    window.editor_tool_combo.currentTextChanged.connect(window.set_editor_tool)

    mode_button_row = QHBoxLayout()
    mode_button_row.setSpacing(6)
    window.editor_mode_button_group = QButtonGroup(window)
    window.editor_mode_button_group.setExclusive(True)
    for mode_name, label, icon_name in (
        ("paint", "Edit objects", "gear"),
        ("move", "Pan / Zoom map", "maximize"),
    ):
        button = QPushButton(label)
        button.setObjectName("secondaryButton")
        button.setCheckable(True)
        button.setIcon(make_icon(icon_name, TEXT))
        button.setIconSize(QSize(16, 16))
        button.clicked.connect(lambda checked=False, name=mode_name: window.set_editor_interaction_mode(name))
        window.editor_mode_button_group.addButton(button)
        mode_button_row.addWidget(button)
        setattr(window, f"editor_{mode_name}_button", button)

    tool_button_row = QHBoxLayout()
    tool_button_row.setSpacing(6)
    window.editor_tool_button_group = QButtonGroup(window)
    window.editor_tool_button_group.setExclusive(True)
    for tool_name, label, icon_name in (
        ("rectangles", "Rect", "maximize"),
        ("squares", "Square", "maximize"),
        ("free", "Free", "gear"),
        ("erase", "Erase", "reset"),
        ("camera", "Viewport", "console"),
    ):
        button = QPushButton(label)
        button.setObjectName("secondaryButton")
        button.setCheckable(True)
        button.setIcon(make_icon(icon_name, TEXT))
        button.setIconSize(QSize(16, 16))
        button.clicked.connect(lambda checked=False, name=tool_name: window.set_editor_tool(name))
        window.editor_tool_button_group.addButton(button)
        tool_button_row.addWidget(button)
        setattr(window, f"editor_{tool_name}_button", button)

    window.editor_brush_size_preview = BrushSizePreview(getattr(window, "editor_brush_size", 0.2))
    window.editor_brush_size_value_label = QLabel(f"{getattr(window, 'editor_brush_size', 0.2):.2f} m")
    window.editor_brush_size_value_label.setObjectName("fieldLabel")
    window.editor_brush_size_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    window.editor_brush_size_slider = QSlider(Qt.Horizontal)
    window.editor_brush_size_slider.setMinimum(5)
    window.editor_brush_size_slider.setMaximum(200)
    window.editor_brush_size_slider.setSingleStep(5)
    window.editor_brush_size_slider.setPageStep(20)
    window.editor_brush_size_slider.setValue(int(round(getattr(window, "editor_brush_size", 0.2) * 100.0)))
    window.editor_brush_size_slider.valueChanged.connect(window.set_editor_brush_size_from_slider)

    # Compatibility name used by MainWindow.refresh_editor_tool_buttons().
    window.editor_brush_size_input = window.editor_brush_size_slider

    brush_size_row = QVBoxLayout()
    brush_size_row.setSpacing(5)
    brush_header = QHBoxLayout()
    brush_header.setSpacing(6)
    brush_size_label = QLabel("Free-draw brush")
    brush_size_label.setObjectName("fieldLabel")
    brush_header.addWidget(brush_size_label)
    brush_header.addWidget(window.editor_brush_size_value_label)
    brush_size_row.addLayout(brush_header)
    brush_size_row.addWidget(window.editor_brush_size_preview)
    brush_size_row.addWidget(window.editor_brush_size_slider)

    editor_hint = QLabel(
        "Edit objects and Pan/Zoom are exclusive modes. Click any obstacle/object and drag it directly. "
        "Free draw uses circular stamps and renders connected stamps as one smooth object. "
        "Shortcuts: Ctrl+Z undo, Ctrl+Alt+Z redo."
    )
    editor_hint.setObjectName("fieldLabel")
    editor_hint.setWordWrap(True)

    interaction_label = QLabel("Interaction mode")
    interaction_label.setObjectName("subsectionLabel")
    tools_label = QLabel("Object tool")
    tools_label.setObjectName("subsectionLabel")

    editor_grid.addWidget(window.editor_status_label, 0, 0, 1, 2)
    editor_grid.addWidget(interaction_label, 1, 0, 1, 2)
    editor_grid.addLayout(mode_button_row, 2, 0, 1, 2)
    editor_grid.addWidget(tools_label, 3, 0, 1, 2)
    editor_grid.addWidget(labeled_combo("Tool", window.editor_tool_combo), 4, 0, 1, 2)
    editor_grid.addLayout(tool_button_row, 5, 0, 1, 2)
    editor_grid.addLayout(brush_size_row, 6, 0, 1, 2)
    editor_grid.addWidget(editor_hint, 7, 0, 1, 2)
    editor_grid.setColumnStretch(0, 1)
    editor_grid.setColumnStretch(1, 1)
    editor_card.root.addLayout(editor_grid)
    content_layout.addWidget(editor_card)

    # --------------------------------------------------------
    # Simulation camera / viewport
    # --------------------------------------------------------
    camera_card = SectionCard("camera", "Simulation Camera")
    camera_grid = QGridLayout()
    camera_grid.setHorizontalSpacing(8)
    camera_grid.setVerticalSpacing(7)

    window.editor_camera_x_input = NumericStepper(
        "Center X",
        getattr(window.config, "camera_center_x", 0.0),
        -100.0,
        100.0,
        0.25,
    )
    window.editor_camera_y_input = NumericStepper(
        "Center Y",
        getattr(window.config, "camera_center_y", 0.0),
        -100.0,
        100.0,
        0.25,
    )
    window.editor_camera_width_input = NumericStepper(
        "Width (m)",
        getattr(window.config, "camera_width", WORLD_X_MAX - WORLD_X_MIN),
        1.0,
        80.0,
        0.5,
    )
    window.editor_camera_height_input = NumericStepper(
        "Height (m)",
        getattr(window.config, "camera_height", WORLD_Y_MAX - WORLD_Y_MIN),
        1.0,
        80.0,
        0.5,
    )

    for widget in (
        window.editor_camera_x_input,
        window.editor_camera_y_input,
        window.editor_camera_width_input,
        window.editor_camera_height_input,
    ):
        widget.valueChanged.connect(window.sync_editor_camera_from_panel)

    window.editor_camera_reset_button = QPushButton("Reset to full world")
    window.editor_camera_reset_button.setIcon(make_icon("reset", TEXT))
    window.editor_camera_reset_button.setIconSize(QSize(16, 16))
    window.editor_camera_reset_button.setObjectName("secondaryButton")
    window.editor_camera_reset_button.clicked.connect(window.reset_editor_camera)

    window.editor_camera_fit_button = QPushButton("Fit camera to obstacles")
    window.editor_camera_fit_button.setIcon(make_icon("maximize", TEXT))
    window.editor_camera_fit_button.setIconSize(QSize(16, 16))
    window.editor_camera_fit_button.setObjectName("secondaryButton")
    window.editor_camera_fit_button.clicked.connect(window.fit_editor_camera_to_obstacles)

    camera_hint = QLabel(
        "Drag the red border on the map or type exact numbers here. This viewport is saved with the .sim file."
    )
    camera_hint.setObjectName("fieldLabel")
    camera_hint.setWordWrap(True)

    camera_grid.addWidget(window.editor_camera_x_input, 0, 0)
    camera_grid.addWidget(window.editor_camera_y_input, 0, 1)
    camera_grid.addWidget(window.editor_camera_width_input, 1, 0)
    camera_grid.addWidget(window.editor_camera_height_input, 1, 1)
    camera_grid.addWidget(window.editor_camera_reset_button, 2, 0)
    camera_grid.addWidget(window.editor_camera_fit_button, 2, 1)
    camera_grid.addWidget(camera_hint, 3, 0, 1, 2)
    camera_grid.setColumnStretch(0, 1)
    camera_grid.setColumnStretch(1, 1)
    camera_card.root.addLayout(camera_grid)
    content_layout.addWidget(camera_card)

    # --------------------------------------------------------
    # Map actions
    # --------------------------------------------------------
    actions_card = SectionCard("map_actions", "Map Actions")
    actions_grid = QGridLayout()
    actions_grid.setHorizontalSpacing(8)
    actions_grid.setVerticalSpacing(7)

    window.editor_new_map_button = QPushButton("New blank map")
    window.editor_new_map_button.setIcon(make_icon("reset", TEXT))
    window.editor_new_map_button.setIconSize(QSize(16, 16))
    window.editor_new_map_button.setObjectName("secondaryButton")
    window.editor_new_map_button.clicked.connect(window.new_editor_map)

    window.editor_clear_button = QPushButton("Clear obstacles")
    window.editor_clear_button.setIcon(make_icon("reset", TEXT))
    window.editor_clear_button.setIconSize(QSize(16, 16))
    window.editor_clear_button.setObjectName("secondaryButton")
    window.editor_clear_button.clicked.connect(window.clear_editor_map)

    window.editor_undo_button = QPushButton("Undo  Ctrl+Z")
    window.editor_undo_button.setIcon(make_icon("reset", TEXT))
    window.editor_undo_button.setIconSize(QSize(16, 16))
    window.editor_undo_button.setObjectName("secondaryButton")
    window.editor_undo_button.clicked.connect(window.undo_editor_change)

    window.editor_redo_button = QPushButton("Redo  Ctrl+Alt+Z")
    window.editor_redo_button.setIcon(make_icon("reset", TEXT))
    window.editor_redo_button.setIconSize(QSize(16, 16))
    window.editor_redo_button.setObjectName("secondaryButton")
    window.editor_redo_button.clicked.connect(window.redo_editor_change)

    window.editor_commit_button = QPushButton("Accept map")
    window.editor_commit_button.setIcon(make_icon("save", "white"))
    window.editor_commit_button.setIconSize(QSize(18, 18))
    window.editor_commit_button.setObjectName("startButton")
    window.editor_commit_button.clicked.connect(window.commit_editor_map)

    actions_grid.addWidget(window.editor_undo_button, 0, 0)
    actions_grid.addWidget(window.editor_redo_button, 0, 1)
    actions_grid.addWidget(window.editor_new_map_button, 1, 0)
    actions_grid.addWidget(window.editor_clear_button, 1, 1)
    actions_grid.addWidget(window.editor_commit_button, 2, 0, 1, 2)
    actions_grid.setColumnStretch(0, 1)
    actions_grid.setColumnStretch(1, 1)
    actions_card.root.addLayout(actions_grid)
    content_layout.addWidget(actions_card)

    content_layout.addStretch()
    scroll.setWidget(content)
    layout.addWidget(scroll, 1)

    bottom_actions = QFrame()
    bottom_actions.setObjectName("actionPanelBottom")
    actions_layout = QVBoxLayout(bottom_actions)
    actions_layout.setContentsMargins(12, 12, 12, 12)
    actions_layout.setSpacing(8)

    window.editor_back_button = QPushButton("Back to Simulation")
    window.editor_back_button.setIcon(make_icon("play", TEXT))
    window.editor_back_button.setIconSize(QSize(16, 16))
    window.editor_back_button.setObjectName("secondaryButton")
    window.editor_back_button.clicked.connect(lambda: window.set_editor_mode(False))
    actions_layout.addWidget(window.editor_back_button)
    layout.addWidget(bottom_actions)

    window.refresh_editor_tool_buttons()
    return panel
