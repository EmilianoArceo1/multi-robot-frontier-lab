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
from PySide6.QtCore import Qt, QTimer, QSize, QThreadPool, QEasingCurve, QPropertyAnimation
from PySide6.QtGui import QAction, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from robotics_sim.simulation.config import *
from robotics_sim.diagnostics.event_log import NavigationDebugEventLog
from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.simulation.engine import PlannerWorker, SimulationControllerMixin
from robotics_sim.app.theme import (
    THEME_SETTINGS_KEY,
    ThemeColors,
    ThemeMode,
    apply_application_theme,
    build_application_stylesheet,
    dropdown_popup_stylesheet,
    open_theme_settings,
    parse_theme_mode,
    theme_colors,
    with_alpha,
)
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
from robotics_sim.app.navigation_reasoning_window import NavigationReasoningWindow
from robotics_sim.app.config_panel import BrushSizePreview
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
        # Last ACCEPTED plan's planner/simplifier/raw/simplified-path,
        # persisted so routine ticks (which do not recompute a plan) keep
        # describing the route currently being executed -- see apply_route_
        # result()'s diagnostics block and _finalize_navigation_debug_
        # snapshot()'s fallback.
        self._nav_debug_last_accepted_plan = None
        # Multi-robot diagnostics keep the equivalent route provenance and
        # live frame independently for every robot.  The scalar aliases above
        # remain the single-robot/backward-compatible view used by the canvas.
        self._nav_debug_last_accepted_plan_by_robot: dict[int, object] = {}
        # Last live snapshot is kept separately from the historical view so
        # stepping backward while paused can always return to the actual live
        # state instead of leaving a stale history frame on the canvas.
        self._nav_debug_live_snapshot = None
        self._nav_debug_live_snapshots_by_robot: dict[int, object] = {}
        self._nav_debug_last_event_by_robot: dict[int, object] = {}
        self._nav_debug_pending_plan_capture_by_robot: dict[int, object] = {}
        self._nav_debug_current_plan_capture_by_robot: dict[int, object] = {}

        # Press-and-hold history scrubbing. The first press moves one frame,
        # then a single-shot timer repeats with a smooth 1x -> 20x ramp.
        # _nav_history_scrub_current_multiplier is the ramp value the *last*
        # step was taken at (1.0 while idle) -- read by update_navigation_
        # debug_step_buttons() so the snapshot bar's counter can show "x4"
        # etc. without a second timer/state machine of its own.
        self._nav_history_scrub_direction = 0
        self._nav_history_scrub_started_at = 0.0
        self._nav_history_scrub_current_multiplier = 1.0
        self._nav_history_scrub_initial_delay_ms = 320
        self._nav_history_scrub_base_interval_ms = 180
        self._nav_history_scrub_ramp_seconds = 1.8
        self._nav_history_scrub_timer = QTimer(self)
        self._nav_history_scrub_timer.setSingleShot(True)
        self._nav_history_scrub_timer.timeout.connect(self._continue_navigation_history_scrub)

        # RuntimeHazardService is created lazily by reset_belief_map(), using
        # the same bounds/resolution as the logical occupancy belief. Temporary
        # fire sources never live inside BeliefMap.grid.
        self.hazard_service = None
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
        # Panel visibility is explicit UI state. Do not infer it with
        # QWidget.isVisible(): during build_ui() the parent window is not yet
        # shown, so isVisible() returns False even for a child that is intended
        # to be visible. That previously hid the whole side column and made the
        # gear-menu action unable to recover it.
        self._configuration_panel_visible = True
        # Deliberately independent of navigation_debug_enabled (see
        # on_navigation_reasoning_panel_visibility_toggled() /
        # on_navigation_debug_toggled()): the Navigation switch controls
        # capture, the gear-menu "Navigation Reasoning" action controls only
        # whether the docked panel/tab is shown. Neither one drives the
        # other -- closing the panel must not stop capture, and toggling
        # capture off must not change which panel tab is selected.
        self._navigation_reasoning_panel_visible = False

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

        # Loaded before the first stylesheet()/build_ui() call -- both read
        # self._theme_mode. Starts light the very first run (no saved
        # value) and on any invalid saved value; see _load_saved_theme()'s
        # docstring for the exact persistence rules.
        self._theme_mode = self._load_saved_theme()
        self._theme_transition_animation = None
        self._theme_transition_overlay = None

        self.setStyleSheet(self.stylesheet())
        self.build_ui()

        # Custom-painted children (canvas, every ToggleSwitch, the docked
        # reasoning panel) default to light at construction time regardless
        # of the loaded theme -- this first _apply_theme() call corrects
        # them and wires the theme button now that top_bar exists. Also
        # applies the app-level stylesheet so QMenu/QToolTip popups (which
        # do not inherit MainWindow's own setStyleSheet()) render correctly
        # from the very first paint.
        self.top_bar.theme_button.clicked.connect(self._toggle_theme)
        self._apply_theme(self._theme_mode)

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
        self.canvas.fireToggleRequested.connect(self.on_fire_toggle_requested)
        self._build_canvas_action_bar()
        self._build_navigation_snapshot_bar()

        self.navigation_reasoning_window = NavigationReasoningWindow(self)
        # Closing the docked panel only hides it -- it must never disable
        # capture (see on_navigation_reasoning_panel_visibility_toggled()).
        # History stepping itself lives solely in navigation_snapshot_bar
        # now; the panel no longer owns its own `‹`/`›` buttons.
        self.navigation_reasoning_window.closeRequested.connect(
            lambda: self.on_navigation_reasoning_panel_visibility_toggled(False)
        )
        self.navigation_reasoning_window.nextRobotRequested.connect(self.select_next_robot)
        self.canvas.set_navigation_reasoning_window(self.navigation_reasoning_window)

        self.simulation_panel = self.build_config_panel()
        self.editor_panel = self.build_editor_panel()

        self.config_panel_stack = QWidget()
        self.config_panel_stack.setObjectName("configPanelStack")
        config_stack_layout = QVBoxLayout(self.config_panel_stack)
        config_stack_layout.setContentsMargins(0, 0, 0, 0)
        config_stack_layout.setSpacing(0)
        config_stack_layout.addWidget(self.simulation_panel)
        config_stack_layout.addWidget(self.editor_panel)

        self._install_config_panel_close_button(self.simulation_panel)
        self._install_config_panel_close_button(self.editor_panel)

        # A narrow side column cannot support two vertically stacked, content-
        # heavy panels without making both unpleasant to use. Keep each panel
        # full-height and switch between them with tabs when both are enabled.
        self.side_panel_tabs = QTabWidget()
        self.side_panel_tabs.setObjectName("sidePanelTabs")
        self.side_panel_tabs.setDocumentMode(True)
        self.side_panel_tabs.setMovable(False)
        self.side_panel_tabs.setTabsClosable(False)
        self.side_panel_tabs.tabBar().setExpanding(True)

        self.side_panel_container = QWidget()
        self.side_panel_container.setObjectName("sidePanelContainer")
        self.side_panel_container.setFixedWidth(SIDE_PANEL_WIDTH)
        self.side_panel_container_layout = QVBoxLayout(self.side_panel_container)
        self.side_panel_container_layout.setContentsMargins(0, 0, 0, 0)
        self.side_panel_container_layout.setSpacing(0)
        self.side_panel_container_layout.addWidget(self.side_panel_tabs)

        self._build_panel_visibility_menu()
        self.switch_panel_to_simulation()
        # Configuration is the default panel. Keep this as explicit state
        # rather than relying on effective visibility before the window has
        # been shown.
        self._configuration_panel_visible = True
        self.config_panel_stack.setVisible(True)
        self.navigation_reasoning_window.hide()

        canvas_column = QWidget()
        canvas_column.setObjectName("canvasColumn")
        canvas_column_layout = QVBoxLayout(canvas_column)
        canvas_column_layout.setContentsMargins(0, 0, 0, 0)
        canvas_column_layout.setSpacing(8)
        canvas_column_layout.addWidget(self.navigation_snapshot_bar)
        canvas_column_layout.addWidget(self.canvas, 1)

        body_layout.addWidget(canvas_column, 1)
        body_layout.addWidget(self.side_panel_container, 0)

        outer.addWidget(body, 1)
        self._sync_side_panel_layout()
        QTimer.singleShot(0, self._sync_side_panel_layout)

    def _build_canvas_action_bar(self) -> None:
        """Create the primary runtime controls in the canvas footer.

        The old telemetry strip was passive and duplicated state that now
        belongs in Navigation Reasoning. These are real Qt buttons placed in
        the same reserved footer region, so they remain usable even when the
        configuration panel is hidden.
        """
        bar = QFrame(self.canvas)
        bar.setObjectName("canvasActionBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(9, 5, 9, 5)
        layout.setSpacing(8)

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("canvasStartButton")
        self.start_button.setIcon(make_icon("play", "white"))
        self.start_button.setIconSize(QSize(18, 18))
        self.start_button.clicked.connect(self.handle_start_pause_button)

        self.reset_button = QPushButton("Restart")
        self.reset_button.setObjectName("canvasActionButton")
        self.reset_button.setIcon(make_icon("reset", TEXT))
        self.reset_button.setIconSize(QSize(16, 16))
        self.reset_button.clicked.connect(self.restart_simulation)

        self.speed_button = QPushButton(f"Speed {self.simulation_speed:.2f}x")
        self.speed_button.setObjectName("canvasActionButton")
        self.speed_button.setIcon(make_icon("gear", TEXT))
        self.speed_button.setIconSize(QSize(16, 16))
        self.speed_button.clicked.connect(self.cycle_simulation_speed)

        self.metrics_button = QPushButton("Metrics")
        self.metrics_button.setObjectName("canvasActionButton")
        self.metrics_button.setIcon(make_icon("maximize", TEXT))
        self.metrics_button.setIconSize(QSize(16, 16))
        self.metrics_button.clicked.connect(self.open_metrics_window)

        self.console_button = QPushButton("Console")
        self.console_button.setObjectName("canvasActionButton")
        self.console_button.setIcon(make_icon("console", TEXT))
        self.console_button.setIconSize(QSize(16, 16))
        self.console_button.clicked.connect(self.open_console_window)

        layout.addWidget(self.start_button, 2)
        layout.addWidget(self.reset_button, 1)
        layout.addWidget(self.speed_button, 1)
        layout.addWidget(self.metrics_button, 1)
        layout.addWidget(self.console_button, 1)

        self.canvas_action_bar = bar
        self.canvas.set_action_bar(bar)

    def _update_canvas_action_bar_icons(self) -> None:
        """Re-render the canvas action bar's hand-drawn icons in the current
        theme's text color. The QPushButton labels/background already
        update for free via the QSS#canvasActionButton rule -- only the
        baked QIcon bitmaps need a manual refresh. canvasStartButton keeps
        its hardcoded white icon; its maroon background never changes with
        theme, same as the Resume button in the navigation snapshot bar."""
        color = theme_colors(self._theme_mode).text_primary
        icon_specs = (
            ("reset_button", "reset", 16),
            ("speed_button", "gear", 16),
            ("metrics_button", "maximize", 16),
            ("console_button", "console", 16),
        )
        for attr_name, icon_type, size in icon_specs:
            button = getattr(self, attr_name, None)
            if button is not None:
                button.setIcon(make_icon(icon_type, color))
                button.setIconSize(QSize(size, size))

    def _navigation_snapshot_bar_stylesheet(self, c: ThemeColors) -> str:
        """Stylesheet for the navigation snapshot bar (Navigation switch,
        `<`/`>` history step buttons, Resume from snapshot). Rebuilt and
        reapplied on every theme change by _apply_theme() since it is a
        self-contained inline stylesheet, not reachable by the app-level
        QSS cascade."""
        return f"""
            QFrame#navigationSnapshotBar {{
                background: {c.card_background};
                border: 1px solid {c.border};
                border-radius: 10px;
            }}
            QLabel#navigationSnapshotLabel {{
                color: {c.text_primary};
                font-size: 11px;
                font-weight: 800;
            }}
            QLabel#navigationSnapshotCounter {{
                color: {c.text_primary};
                font-family: Consolas, "Courier New", monospace;
                font-size: 11px;
                font-weight: 700;
            }}
            QPushButton#navigationSnapshotStepButton {{
                background: {c.card_background};
                border: 1px solid {c.border};
                border-radius: 6px;
                color: {c.text_primary};
                font-size: 13px;
                font-weight: 900;
            }}
            QPushButton#navigationSnapshotStepButton:hover:enabled {{
                border-color: {c.accent};
                background: {with_alpha(c.accent, 32)};
                color: {c.accent};
            }}
            QPushButton#navigationSnapshotStepButton:disabled {{
                color: {c.text_disabled};
                background: {c.elevated_background};
            }}
            QPushButton#navigationSnapshotResumeButton {{
                background: {MAROON};
                border: none;
                border-radius: 7px;
                color: white;
                font-size: 11px;
                font-weight: 800;
                padding: 0 12px;
            }}
            QPushButton#navigationSnapshotResumeButton:hover:enabled {{
                background: {MAROON_DARK};
            }}
            QPushButton#navigationSnapshotResumeButton:disabled {{
                background: {c.elevated_background};
                color: {c.text_disabled};
            }}
            """

    def _build_navigation_snapshot_bar(self) -> None:
        """Primary Navigation Debug control, docked above the canvas header.

        Replaces the old canvas-painted eye icon as the activator: a real
        ToggleSwitch (navigation_snapshot_switch) is the single place that
        turns capture on/off. `<`/`>` (navigation_snapshot_back/forward_
        button) reuse the same press/hold acceleration timer as the docked
        reasoning panel's legacy footer buttons -- start/stop_navigation_
        history_scrub() -- so both controls always agree. update_navigation_
        debug_step_buttons() (engine.py) owns all of this bar's enabled
        state and counter text via update_state(); this method only builds
        the widgets and wires the raw Qt signals.
        """
        bar = QFrame()
        bar.setObjectName("navigationSnapshotBar")
        bar.setFixedHeight(46)
        bar.setStyleSheet(self._navigation_snapshot_bar_stylesheet(theme_colors(self._theme_mode)))
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        label = QLabel("Navigation")
        label.setObjectName("navigationSnapshotLabel")
        layout.addWidget(label)

        self.navigation_snapshot_switch = ToggleSwitch(checked=False)
        self.navigation_snapshot_switch.setToolTip("Enable Navigation Debug capture")
        self.navigation_snapshot_switch.toggled.connect(self.on_navigation_debug_toggled)
        layout.addWidget(self.navigation_snapshot_switch)

        layout.addSpacing(10)

        self.navigation_snapshot_back_button = QPushButton("<")
        self.navigation_snapshot_forward_button = QPushButton(">")
        for button in (self.navigation_snapshot_back_button, self.navigation_snapshot_forward_button):
            button.setObjectName("navigationSnapshotStepButton")
            button.setFixedSize(30, 30)
            button.setEnabled(False)
            # MainWindow owns press-and-hold acceleration (1x -> 20x) with a
            # single timer -- see start/_continue/stop_navigation_history_
            # scrub(). Qt's own fixed-rate autoRepeat is left off so one
            # press cannot trigger two independent repeat loops.
            button.setAutoRepeat(False)
        self.navigation_snapshot_back_button.setToolTip("Previous snapshot (hold to accelerate)")
        self.navigation_snapshot_forward_button.setToolTip("Next snapshot (hold to accelerate)")
        self.navigation_snapshot_back_button.pressed.connect(lambda: self.start_navigation_history_scrub(-1))
        self.navigation_snapshot_back_button.released.connect(self.stop_navigation_history_scrub)
        self.navigation_snapshot_forward_button.pressed.connect(lambda: self.start_navigation_history_scrub(1))
        self.navigation_snapshot_forward_button.released.connect(self.stop_navigation_history_scrub)
        layout.addWidget(self.navigation_snapshot_back_button)

        self.navigation_snapshot_counter_label = QLabel("OFF")
        self.navigation_snapshot_counter_label.setObjectName("navigationSnapshotCounter")
        self.navigation_snapshot_counter_label.setAlignment(Qt.AlignCenter)
        self.navigation_snapshot_counter_label.setFixedWidth(88)
        layout.addWidget(self.navigation_snapshot_counter_label)

        layout.addWidget(self.navigation_snapshot_forward_button)

        layout.addSpacing(10)

        self.navigation_snapshot_resume_button = QPushButton("Resume from snapshot")
        self.navigation_snapshot_resume_button.setObjectName("navigationSnapshotResumeButton")
        self.navigation_snapshot_resume_button.setEnabled(False)
        self.navigation_snapshot_resume_button.clicked.connect(self.on_resume_from_snapshot_clicked)
        layout.addWidget(self.navigation_snapshot_resume_button)

        layout.addStretch(1)

        self.navigation_snapshot_bar = bar
        self.navigation_snapshot_bar.update_state = self._update_navigation_snapshot_bar_state

    def _update_navigation_snapshot_bar_state(
        self,
        *,
        navigation_enabled: bool,
        position: int | None,
        total: int,
        back_enabled: bool,
        forward_enabled: bool,
        multiplier: float,
        resume_enabled: bool,
        resume_reason: str,
    ) -> None:
        """The single place that turns engine-owned history state into the
        bar's text/enabled state -- called from engine.update_navigation_
        debug_step_buttons() so this widget never has to poll."""
        self._set_action_checked_ish(self.navigation_snapshot_switch, navigation_enabled)

        if not navigation_enabled:
            counter_text = "OFF"
        elif position is None:
            counter_text = "LIVE"
        else:
            counter_text = f"{position + 1}/{total}"
            if multiplier >= 2.0:
                counter_text += f" · x{multiplier:.0f}"
        self.navigation_snapshot_counter_label.setText(counter_text)

        self.navigation_snapshot_back_button.setEnabled(bool(back_enabled))
        self.navigation_snapshot_forward_button.setEnabled(bool(forward_enabled))

        self.navigation_snapshot_resume_button.setEnabled(bool(resume_enabled))
        self.navigation_snapshot_resume_button.setToolTip(
            resume_reason or "Restore the simulation to this snapshot."
        )

    @staticmethod
    def _set_action_checked_ish(switch, checked: bool) -> None:
        if switch.isChecked() == bool(checked):
            return
        switch.blockSignals(True)
        switch.setChecked(bool(checked))
        switch.blockSignals(False)
        switch.update()

    def on_resume_from_snapshot_clicked(self) -> None:
        self.restore_navigation_debug_snapshot()

    def build_config_panel(self):
        """Build the right-side simulation configuration panel."""
        return build_right_config_panel(self)

    def build_editor_panel(self):
        from robotics_sim.app.config_panel import build_editor_panel
        return build_editor_panel(self)

    def _install_config_panel_close_button(self, panel: QWidget) -> None:
        """Attach a small close control without modifying the panel builders.

        Styled via the global stylesheet's QPushButton#configPanelCloseButton
        rule (see theme.py) rather than an inline setStyleSheet() here, so it
        stays correctly themed after a later theme toggle without needing
        its own propagation call.
        """
        button = QPushButton("×", panel)
        button.setObjectName("configPanelCloseButton")
        button.setFixedSize(26, 24)
        button.move(max(0, SIDE_PANEL_WIDTH - 34), 8)
        button.setToolTip("Close configuration panel")
        button.clicked.connect(lambda: self.set_configuration_panel_visible(False))
        button.raise_()

    def _load_saved_theme(self) -> ThemeMode:
        """Read the persisted theme preference.

        Read-only -- never writes. Missing/invalid values resolve to light
        (see theme.parse_theme_mode()), which is also the correct behavior
        the very first time the app ever runs, before any value has been
        saved.
        """
        settings = open_theme_settings()
        return parse_theme_mode(settings.value(THEME_SETTINGS_KEY))

    def _apply_theme(self, mode: ThemeMode | str) -> None:
        """Apply `mode` everywhere the app's own chrome needs it -- and
        nothing else. Deliberately does not touch SimulationEngine state:
        no reset, no snapshot/history change, no pause/resume, no robot
        rebuild, no panel visibility change, no Single/Multiple/Editor mode
        change. Safe to call at any point in a run, including mid-simulation.

        Three kinds of propagation, each cheap:
        1. QSS-selector-styled widgets (top bar, panels, tabs, menus,
           inputs, buttons, tooltips, scrollbars) update for free the
           moment the stylesheet is reapplied -- no per-widget call needed.
        2. Widgets with their own inline stylesheet or custom QPainter
           output that the global QSS cannot reach (every ToggleSwitch,
           the docked NavigationReasoningWindow, SimulationCanvas's themed
           chrome layers) get one explicit set_theme_mode() call each.
        3. update_navigation_debug_step_buttons() is NOT called here --
           theme has no effect on history/capture state, only on how it's
           painted, which set_theme_mode() above already covers.

        Every color swap above is still an instant, synchronous hard cut --
        callers (tests included) can rely on self._theme_mode and every
        widget's styling being fully updated the moment this method
        returns. The soft crossfade started at the bottom is a purely
        cosmetic overlay painted on top of the already-final state; it
        never delays or defers any of the above.
        """
        mode = ThemeMode(mode)
        theme_changed = mode != self._theme_mode
        before_snapshot = self.grab() if theme_changed and self.isVisible() else None
        self._theme_mode = mode

        app = QApplication.instance()
        if isinstance(app, QApplication):
            # Reaches QMenu/QToolTip popups, which do not inherit
            # MainWindow's own setStyleSheet() (see build_application_
            # stylesheet()'s docstring).
            apply_application_theme(app, mode)
        self.setStyleSheet(self.stylesheet())

        for switch in self.findChildren(ToggleSwitch):
            switch.set_theme_mode(mode)

        for brush_preview in self.findChildren(BrushSizePreview):
            brush_preview.set_theme_mode(mode)

        # Combo popups tagged by config_panel.labeled_combo() -- their
        # QListView stylesheet is not reachable by the ordinary app-level
        # QSS cascade (see dropdown_popup_stylesheet()'s docstring).
        popup_qss = dropdown_popup_stylesheet(mode)
        for combo in self.findChildren(QComboBox):
            if combo.property("themedDropdownPopup"):
                combo.view().setStyleSheet(popup_qss)

        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "set_theme_mode"):
            canvas.set_theme_mode(mode)

        reasoning_window = getattr(self, "navigation_reasoning_window", None)
        if reasoning_window is not None and hasattr(reasoning_window, "set_theme_mode"):
            reasoning_window.set_theme_mode(mode)

        # Self-contained inline stylesheets not reachable by the app-level
        # QSS cascade (same reason as the combo popups above).
        snapshot_bar = getattr(self, "navigation_snapshot_bar", None)
        if snapshot_bar is not None:
            snapshot_bar.setStyleSheet(self._navigation_snapshot_bar_stylesheet(theme_colors(mode)))

        self._update_canvas_action_bar_icons()

        self._update_theme_button()
        self.update()

        if before_snapshot is not None:
            self._start_theme_crossfade(before_snapshot)

    def _start_theme_crossfade(self, before: "QPixmap") -> None:
        """Fade the pre-toggle appearance out over the newly-applied theme
        so the switch reads as a smooth transition instead of a hard cut.

        Purely cosmetic and additive: by the time this runs, _apply_theme()
        has already applied every color synchronously, so this only ever
        layers a fading snapshot on top -- it cannot affect simulation
        state, widget visibility, or any theme-toggle test assertion.
        """
        if self._theme_transition_animation is not None:
            self._theme_transition_animation.stop()
        if self._theme_transition_overlay is not None:
            self._theme_transition_overlay.deleteLater()

        overlay = QLabel(self)
        overlay.setPixmap(before)
        overlay.setGeometry(self.rect())
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        effect = QGraphicsOpacityEffect(overlay)
        overlay.setGraphicsEffect(effect)
        overlay.show()
        overlay.raise_()

        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(220)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)

        def _finish():
            overlay.deleteLater()
            self._theme_transition_overlay = None
            self._theme_transition_animation = None

        animation.finished.connect(_finish)
        self._theme_transition_overlay = overlay
        self._theme_transition_animation = animation
        animation.start()

    def _toggle_theme(self) -> None:
        """The theme_button's click handler. Flips light<->dark, applies
        it, and persists immediately (see theme.py's persistence rules --
        only the theme name is ever written)."""
        new_mode = ThemeMode.DARK if self._theme_mode == ThemeMode.LIGHT else ThemeMode.LIGHT
        self._apply_theme(new_mode)
        settings = open_theme_settings()
        settings.setValue(THEME_SETTINGS_KEY, new_mode.value)
        settings.sync()

    def _update_theme_button(self) -> None:
        """The button's icon always shows the CURRENTLY ACTIVE mode (sun
        while light is active, moon while dark is active) -- never the
        mode a click would switch to."""
        button = getattr(self.top_bar, "theme_button", None)
        if button is None:
            return
        if self._theme_mode == ThemeMode.DARK:
            button.setIcon(make_icon("theme_dark", "white"))
            button.setToolTip("Dark mode active — switch to light mode")
        else:
            button.setIcon(make_icon("theme_light", "white"))
            button.setToolTip("Light mode active — switch to dark mode")

    def _build_panel_visibility_menu(self) -> None:
        """Use the top-bar menu button ("⋮") as the panel and file-action
        menu. Independent of theme_button ("☀"/"☾"), which never opens this
        menu and toggles only the app theme -- see _toggle_theme().

        Snapshot export is deliberately placed before the scenario actions and
        revalidated immediately before every popup.  This avoids a stale native
        menu geometry hiding the last action after actions are appended during
        window construction.
        """
        self.panel_visibility_menu = QMenu(self)
        self.panel_visibility_menu.setSeparatorsCollapsible(False)

        self.configuration_panel_action = QAction("Configuration", self)
        self.configuration_panel_action.setObjectName("configurationPanelAction")
        self.configuration_panel_action.setCheckable(True)
        self.configuration_panel_action.setChecked(True)
        self.configuration_panel_action.toggled.connect(self.set_configuration_panel_visible)
        self.panel_visibility_menu.addAction(self.configuration_panel_action)

        self.navigation_reasoning_panel_action = QAction("Navigation Reasoning", self)
        self.navigation_reasoning_panel_action.setObjectName("navigationReasoningPanelAction")
        self.navigation_reasoning_panel_action.setCheckable(True)
        self.navigation_reasoning_panel_action.setChecked(False)
        # Visibility only -- see on_navigation_reasoning_panel_visibility_
        # toggled()'s docstring. Capture is the Navigation switch's job.
        self.navigation_reasoning_panel_action.toggled.connect(
            self.on_navigation_reasoning_panel_visibility_toggled
        )
        self.panel_visibility_menu.addAction(self.navigation_reasoning_panel_action)

        self.panel_visibility_menu.addSeparator()

        # Keep export visible even with an empty log.  The handler already shows
        # a useful "No snapshots" message, which is better UX than silently
        # hiding the feature the user is trying to discover.
        self.export_snapshots_action = QAction(
            make_icon("save", TEXT), "Export snapshots to Excel…", self
        )
        self.export_snapshots_action.setObjectName("exportSnapshotsExcelAction")
        self.export_snapshots_action.triggered.connect(self.export_navigation_snapshots)
        self.panel_visibility_menu.addAction(self.export_snapshots_action)

        self.panel_visibility_menu.addSeparator()
        self.load_sim_action = QAction(make_icon("reset", TEXT), "Load .sim…", self)
        self.load_sim_action.setObjectName("loadSimulationAction")
        self.load_sim_action.triggered.connect(self.load_simulation_config)
        self.panel_visibility_menu.addAction(self.load_sim_action)

        self.save_sim_action = QAction(make_icon("save", TEXT), "Save .sim…", self)
        self.save_sim_action.setObjectName("saveSimulationAction")
        self.save_sim_action.triggered.connect(self.save_simulation_config)
        self.panel_visibility_menu.addAction(self.save_sim_action)

        self.panel_visibility_menu.aboutToShow.connect(
            self._prepare_panel_visibility_menu
        )
        # Tooltip already set by TopBar itself ("Open application menu");
        # not overridden here so there is exactly one place that owns it.
        self.top_bar.menu_button.clicked.connect(self._show_panel_visibility_menu)

    def _prepare_panel_visibility_menu(self) -> None:
        """Guarantee that the export action is present and the popup is resized.

        QMenu can cache native popup geometry.  Explicitly refreshing the action
        and size before showing prevents a late-added final action from being
        clipped or omitted by the platform menu implementation.
        """
        menu = self.panel_visibility_menu
        actions = menu.actions()
        if self.export_snapshots_action not in actions:
            before = self.load_sim_action if self.load_sim_action in actions else None
            if before is None:
                menu.addAction(self.export_snapshots_action)
            else:
                menu.insertAction(before, self.export_snapshots_action)

        self.export_snapshots_action.setVisible(True)
        self.export_snapshots_action.setEnabled(True)
        menu.ensurePolished()
        menu.adjustSize()

    def _show_panel_visibility_menu(self) -> None:
        self._prepare_panel_visibility_menu()
        button = self.top_bar.menu_button
        position = button.mapToGlobal(button.rect().bottomLeft())
        self.panel_visibility_menu.exec(position)

    @staticmethod
    def _set_action_checked(action: QAction | None, checked: bool) -> None:
        if action is None or action.isChecked() == bool(checked):
            return
        action.blockSignals(True)
        action.setChecked(bool(checked))
        action.blockSignals(False)

    def set_configuration_panel_visible(self, visible: bool) -> None:
        visible = bool(visible)
        self._configuration_panel_visible = visible
        stack = getattr(self, "config_panel_stack", None)
        if stack is not None:
            stack.setVisible(visible)
        self._set_action_checked(getattr(self, "configuration_panel_action", None), visible)
        self._sync_side_panel_layout()
        if visible:
            tabs = getattr(self, "side_panel_tabs", None)
            if tabs is not None and stack is not None and tabs.indexOf(stack) >= 0:
                tabs.setCurrentWidget(stack)
        QTimer.singleShot(0, self._sync_side_panel_layout)

    def _ensure_side_panel_tab(
        self,
        widget: QWidget | None,
        *,
        title: str,
        visible: bool,
        preferred_index: int,
    ) -> None:
        """Add/remove one panel tab without deleting the panel widget."""
        tabs = getattr(self, "side_panel_tabs", None)
        if tabs is None or widget is None:
            return

        current_index = tabs.indexOf(widget)
        if visible:
            if current_index < 0:
                # Clear any explicit hidden state before QTabWidget takes
                # ownership of page visibility.
                widget.setVisible(True)
                tabs.insertTab(min(preferred_index, tabs.count()), widget, title)
        else:
            if current_index >= 0:
                tabs.removeTab(current_index)
            widget.setVisible(False)

    def _sync_side_panel_layout(self) -> None:
        """Synchronize the adaptive right-side panel deck.

        Panel visibility is explicit state. When both panels are enabled they
        are presented as full-height tabs, not a 50/50 vertical split. This
        preserves the complete configuration workflow and gives Navigation
        Reasoning enough room for readable diagnostics.
        """
        container = getattr(self, "side_panel_container", None)
        tabs = getattr(self, "side_panel_tabs", None)
        config_stack = getattr(self, "config_panel_stack", None)
        reasoning = getattr(self, "navigation_reasoning_window", None)
        if container is None or tabs is None:
            return

        config_visible = bool(
            getattr(self, "_configuration_panel_visible", True)
            and config_stack is not None
        )
        reasoning_visible = bool(
            getattr(self, "_navigation_reasoning_panel_visible", False)
            and reasoning is not None
        )

        self._ensure_side_panel_tab(
            config_stack,
            title="Configuration",
            visible=config_visible,
            preferred_index=0,
        )
        self._ensure_side_panel_tab(
            reasoning,
            title="Navigation",
            visible=reasoning_visible,
            preferred_index=1,
        )

        # A single visible panel does not need a tab bar; hiding it returns the
        # vertical space to the panel content. With two panels the tab bar is
        # the compact, predictable switcher.
        tabs.tabBar().setVisible(tabs.count() > 1)
        container.setVisible(tabs.count() > 0)

        if tabs.count() == 1:
            tabs.setCurrentIndex(0)

    @staticmethod
    def navigation_history_scrub_multiplier(elapsed_seconds: float, ramp_seconds: float = 1.8) -> float:
        """Smoothly ramp hold-to-scrub speed from 1x to a hard 20x cap."""
        elapsed = max(0.0, float(elapsed_seconds))
        ramp = max(1e-6, float(ramp_seconds))
        return min(20.0, 1.0 + 19.0 * min(1.0, elapsed / ramp))

    def start_navigation_history_scrub(self, direction: int) -> None:
        direction = -1 if int(direction) < 0 else 1
        if not self.navigation_debug_enabled or not self.paused:
            return
        self.stop_navigation_history_scrub()
        self._nav_history_scrub_direction = direction
        self._nav_history_scrub_started_at = time.perf_counter()
        # The first press always moves exactly one frame at 1x -- the ramp
        # only applies to the repeats a sustained hold triggers below.
        self._nav_history_scrub_current_multiplier = 1.0
        self.step_navigation_debug_history(direction)
        self._nav_history_scrub_timer.start(self._nav_history_scrub_initial_delay_ms)

    def _continue_navigation_history_scrub(self) -> None:
        direction = int(getattr(self, "_nav_history_scrub_direction", 0))
        if direction == 0 or not self.navigation_debug_enabled or not self.paused:
            self.stop_navigation_history_scrub()
            return

        elapsed = time.perf_counter() - float(self._nav_history_scrub_started_at)
        multiplier = self.navigation_history_scrub_multiplier(
            elapsed, self._nav_history_scrub_ramp_seconds
        )
        # Stored before stepping (not a fixed interval) so update_navigation_
        # debug_step_buttons() -- called from inside step_navigation_debug_
        # history() below -- picks up the multiplier this exact step is
        # taken at, and the snapshot bar's "current/total · xN" counter
        # reflects the real ramped speed rather than a guessed one.
        self._nav_history_scrub_current_multiplier = multiplier
        self.step_navigation_debug_history(direction)
        interval_ms = max(20, int(round(self._nav_history_scrub_base_interval_ms / multiplier)))
        self._nav_history_scrub_timer.start(interval_ms)

    def stop_navigation_history_scrub(self) -> None:
        timer = getattr(self, "_nav_history_scrub_timer", None)
        if timer is not None:
            timer.stop()
        self._nav_history_scrub_direction = 0
        self._nav_history_scrub_current_multiplier = 1.0
        updater = getattr(self, "update_navigation_debug_step_buttons", None)
        if callable(updater):
            updater()

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
        self._sync_side_panel_layout()

    def switch_panel_to_simulation(self) -> None:
        if self.editor_panel is not None:
            self.editor_panel.setVisible(False)
            self.editor_panel.setEnabled(False)
        if self.simulation_panel is not None:
            self.simulation_panel.setVisible(True)
            self.simulation_panel.setEnabled(True)
        self._sync_side_panel_layout()

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
        reasoning_panel = getattr(self, "navigation_reasoning_window", None)
        if reasoning_panel is not None and hasattr(reasoning_panel, "set_robot_selector"):
            multiple = "Multiple" in str(getattr(self.config, "agent_mode", ""))
            reasoning_panel.set_robot_selector(
                self.selected_robot_index if multiple else 0,
                len(self.multi_robot_configs) if multiple else 1,
            )
        # During a live multi-robot run the reasoning panel/overlay follows
        # the robot selected on the canvas/setup controls.  Configuration
        # selection existed before navigation debugging, so keep this call
        # optional for lightweight UI test doubles.
        select_debug_robot = getattr(self, "select_navigation_debug_robot", None)
        if callable(select_debug_robot):
            select_debug_robot(self.selected_robot_index)

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
        reasoning_panel = getattr(self, "navigation_reasoning_window", None)
        if reasoning_panel is not None and hasattr(reasoning_panel, "set_robot_selector"):
            multiple = "Multiple" in str(getattr(self.config, "agent_mode", ""))
            reasoning_panel.set_robot_selector(
                self.selected_robot_index if multiple else 0,
                len(self.multi_robot_configs) if multiple else 1,
            )

    def on_same_config_toggled(self, *_):
        self.ensure_multi_robot_configs()
        self.load_selected_robot_into_panel()
        self.update_relevant_parameter_visibility()
        self.update_preview()

    def on_agent_mode_changed(self, *_):
        self.update_relevant_parameter_visibility()
        self.update_preview()
        reasoning_panel = getattr(self, "navigation_reasoning_window", None)
        if reasoning_panel is not None and hasattr(reasoning_panel, "set_robot_selector"):
            multiple = "Multiple" in str(getattr(self.config, "agent_mode", ""))
            count = len(getattr(self, "multi_robot_configs", [])) if multiple else 1
            reasoning_panel.set_robot_selector(
                self.selected_robot_index if multiple else 0,
                max(1, count),
            )

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

    def on_hazard_map_toggled(self, enabled: bool) -> None:
        """Toggle the full ground-truth hazard DEBUG overlay -- ADDS the
        complete HazardField as a blue heatmap under the (always-visible)
        discovered hazard layer; never hides anything. Independent of Fire
        Markers below. Rendering-only, same contract as Show Grid: never
        touches self.config/HazardBelief/planning, stays interactive while
        a simulation is running."""
        self.canvas.set_hazard_map_enabled(bool(enabled))

    def on_fire_markers_toggled(self, enabled: bool) -> None:
        """Toggle whether UNDISCOVERED fire sources are also shown -- ADDS
        the full ground-truth FireSource set; discovered sources are
        always drawn regardless. Independent of Hazard Map above. Same
        rendering-only contract."""
        self.canvas.set_fire_markers_enabled(bool(enabled))

    def on_navigation_debug_toggled(self, enabled: bool) -> None:
        """Enable/disable navigation-debug capture -- and *only* capture.

        Driven exclusively by navigation_snapshot_switch now. Deliberately
        does not touch panel/tab visibility in either direction: turning
        capture off must not change which side-panel tab is selected (see
        on_navigation_reasoning_panel_visibility_toggled(), the menu
        action's handler, which owns that independently).
        """
        enabled = bool(enabled)
        self.navigation_debug_enabled = enabled
        self.canvas.set_navigation_debug_enabled(enabled)

        if not enabled:
            self.stop_navigation_history_scrub()
            self.resume_navigation_debug_live_view()

        self.update_navigation_debug_step_buttons()

    def on_navigation_reasoning_panel_visibility_toggled(self, visible: bool) -> None:
        """Show/hide the docked Navigation Reasoning tab -- and *only*
        visibility. Driven by the gear-menu "Navigation Reasoning" action
        and the panel's own close button. Never touches navigation_debug_
        enabled: closing this panel must not stop capture (see
        on_navigation_debug_toggled(), the switch's handler, which owns
        capture independently)."""
        visible = bool(visible)
        self._navigation_reasoning_panel_visible = visible
        self._set_action_checked(
            getattr(self, "navigation_reasoning_panel_action", None), visible
        )
        self._sync_side_panel_layout()
        if visible:
            tabs = getattr(self, "side_panel_tabs", None)
            panel = getattr(self, "navigation_reasoning_window", None)
            if tabs is not None and panel is not None and tabs.indexOf(panel) >= 0:
                tabs.setCurrentWidget(panel)
        QTimer.singleShot(0, self._sync_side_panel_layout)

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

        load_action = getattr(self, "load_sim_action", None)
        if load_action is not None:
            load_action.setEnabled(not locked and not self.editor_mode)

        if hasattr(self, "editor_tool_combo"):
            self.editor_tool_combo.setEnabled(not locked and self.editor_mode)
        if hasattr(self.top_bar, "editor_button"):
            self.top_bar.editor_button.setEnabled(True)

        # Keep visibility rules active even while the controls are disabled.
        self.update_relevant_parameter_visibility()

    def set_editor_mode(self, enabled: bool) -> None:
        self.editor_mode = bool(enabled)
        self.canvas.set_editor_mode(self.editor_mode)
        self.canvas.set_action_bar_visible(not self.editor_mode)
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
        """Return the full application stylesheet for the current theme.

        Delegates entirely to theme.build_application_stylesheet() -- see
        that module for the actual QSS. Kept as a method (not inlined at
        call sites) because SimulationMetricsWindow/SimulationConsoleWindow
        already call owner.stylesheet() to pick up whatever theme is
        active at the moment they are constructed.
        """
        return build_application_stylesheet(self._theme_mode)
