"""
Simulation canvas and rendering logic.

This module draws the current simulator snapshot: grid, obstacles, mapped
points, explored area, robots, FoV/LiDAR, routes, frontiers, and telemetry.
It emits interaction events, but it does not choose frontiers or compute routes.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QBrush,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from robotics_sim.simulation.config import *
from robotics_sim.app.map_editor import (
    MIN_EDITOR_OBSTACLE_SIZE,
    connected_obstacle_indices,
    find_obstacle_group_at,
    remove_obstacle_at,
)
from robotics_sim.app.render_perf import (
    PerfGuiWarningGate,
    RenderPerfMonitor,
    format_gui_perf_warning,
)

# Soft cap on how many occupancy cells the grid overlay will color-fill in a
# single cache rebuild. Above this, per-cell coloring is skipped for that
# rebuild (grid lines are still drawn) so a small grid_resolution over a
# large visible area can never freeze the UI trying to draw every cell.
MAX_GRID_OVERLAY_CELLS = 20000

# Radii (px) of planned-route markers: the small numbered waypoint dots,
# the active/current waypoint (larger so it still stands out, but not so
# large it dominates the route), and the S (start) / F,G (frontier or
# final-goal endpoint) markers. Purely visual -- none of these affect
# waypoint coordinates, route geometry, or planning in any way.
WAYPOINT_MARKER_RADIUS = 4
MULTI_ROBOT_WAYPOINT_MARKER_RADIUS = 3
ACTIVE_WAYPOINT_MARKER_RADIUS = 6
ACTIVE_WAYPOINT_HALO_PADDING = 6
START_MARKER_RADIUS = 6
FRONTIER_OR_ENDPOINT_MARKER_RADIUS = 7

DEFAULT_RENDER_THROTTLE_FPS = 30.0


class RenderThrottler:
    """Decides whether a high-frequency, simulation-driven repaint request
    should actually trigger self.update() right now, or be coalesced
    (skipped) because a repaint already happened recently enough to hit
    target_fps.

    Pure/Qt-free on purpose (no QWidget dependency) so it is unit-testable
    without a running Qt application. Coalescing loses nothing visually:
    Qt's paintEvent always paints the CURRENT widget/simulation state, not
    a queue of past ones, so skipping an update() call between two accepted
    calls only skips a redundant repaint of state that either looked
    identical or is about to be superseded by the next accepted call.

    Only wired into the two per-tick setters (set_runtime_state()/
    set_multi_runtime_state()) that the engine calls every simulation
    tick while running and unpaused -- every other self.update() call in
    this class (mouse/editor interactions, status/config changes, which
    already only ever fire on user action or while not actively
    simulating) is untouched and stays immediate, matching "render
    immediately after user interactions".

    target_fps defaults to the SIM_RENDER_FPS environment variable
    (read at construction time, mirroring RobotTrace/PerfMonitor's own
    env-reading convention) when not given explicitly, falling back to
    DEFAULT_RENDER_THROTTLE_FPS if that env var is unset. Pass `env=`
    explicitly in tests for a deterministic instance.
    """

    def __init__(
        self,
        target_fps: float | None = None,
        *,
        env: "dict[str, str] | None" = None,
    ):
        if target_fps is None:
            source = env if env is not None else os.environ
            target_fps = float(source.get("SIM_RENDER_FPS", DEFAULT_RENDER_THROTTLE_FPS))
        self.target_fps = float(target_fps)
        self._min_interval = (1.0 / self.target_fps) if self.target_fps > 0 else 0.0
        self._last_render_time: float | None = None

    def should_render(self, now: float | None = None, *, force: bool = False) -> bool:
        now = time.perf_counter() if now is None else float(now)
        if force or self._last_render_time is None or (now - self._last_render_time) >= self._min_interval:
            self._last_render_time = now
            return True
        return False


class SimulationCanvas(QWidget):
    goalClicked = Signal(float, float)
    robotDragged = Signal(int, float, float)
    robotSelected = Signal(int)
    editor_interaction_started = Signal(tuple)
    editor_interaction_progress = Signal(tuple)
    editor_interaction_finished = Signal(tuple, tuple)
    editor_camera_changed = Signal(tuple)
    editor_camera_interaction_started = Signal()
    editor_obstacle_move_started = Signal()
    editor_obstacle_moved = Signal(tuple)
    editor_view_changed = Signal()

    def __init__(self):
        super().__init__()

        self.setObjectName("canvasCard")
        self.setMinimumSize(610, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.robot = None
        self.robots: list = []
        self.config = SimulationConfig()
        self.path_points: list[tuple[float, float]] = []
        self.multi_path_points: list[list[tuple[float, float]]] = []
        self.multi_last_controls: list[np.ndarray] = []
        self.planned_path_points: list[tuple[float, float]] = []
        self.exploration_target_xy: tuple[float, float] | None = None
        self.multi_exploration_targets: list[tuple[float, float] | None] = []
        self.multi_invalidated_exploration_targets: list[list[tuple[float, float]]] = []
        self.explored_area_polygons: list[list[tuple[float, float]]] = []
        self.mapped_obstacle_points: list[tuple[float, float]] = []
        self.known_obstacles: list[tuple[float, float, float, float]] = []
        self.status_message = "Configure parameters and press Start."
        self.status_history: list[str] = []
        self.status_history_limit = 2000
        self._append_status_history(self.status_message)
        self.last_control = np.array([[0.0], [0.0]], dtype=float)

        self.plot_margin_left = 30
        self.plot_margin_top = 60
        self.plot_margin_right = 30
        self.plot_margin_bottom = 70

        # Cached visual layers. Static map background is expensive because it
        # draws topographic curves and grid lines. Explored area is expensive
        # when rebuilt from hundreds of polygons. Both are cached as pixmaps.
        self._static_plot_cache: QPixmap | None = None
        self._static_plot_cache_size: QSize | None = None
        self._explored_area_cache: QPixmap | None = None
        self._explored_area_cache_size: QSize | None = None
        self._explored_area_cached_count = 0
        self._explored_area_caches_by_robot: dict[int, QPixmap] = {}
        self._explored_area_cache_sizes_by_robot: dict[int, QSize] = {}

        # Mapped obstacle points can become thousands of tiny ellipses. Drawing
        # each point every paintEvent is expensive, so they are rasterized into
        # a transparent cache and updated only when new points arrive.
        self._mapped_points_cache: QPixmap | None = None
        self._mapped_points_cache_size: QSize | None = None
        self._mapped_points_cached_count = 0

        # Obstacle completion opacity depends on mapped points. Recomputing
        # coverage in every paintEvent is O(boundary_samples * mapped_points),
        # so the values are cached and invalidated only when mapping changes.
        self._obstacle_coverage_cache: dict[int, float] = {}
        self._obstacle_coverage_cache_count = -1

        # Ground-truth obstacle rendering is cached separately. Showing obstacles
        # used to drop FPS because coverage was recomputed and rectangles were
        # redrawn during paintEvent. Now the obstacle layer is rasterized only
        # when the canvas changes size, the obstacle list changes, or new mapped
        # points may change completion opacity.
        self._obstacles_cache: QPixmap | None = None
        self._obstacles_cache_size: QSize | None = None
        self._obstacles_cache_mapped_count = -1
        self._obstacles_cache_signature: tuple | None = None

        # Runtime metrics. FPS is measured in paintEvent because that is the
        # rate the user actually sees, not just the QTimer tick rate.
        self.fps = 0.0
        self._fps_frame_count = 0
        self._fps_last_time = time.perf_counter()
        self.simulation_time = 0.0
        self.simulation_speed = 1.0
        self.metrics_visible = True

        # Temporary red grid-resolution preview shown while the user adjusts
        # SimulationConfig.grid_resolution in the config panel. Purely visual
        # -- it never touches self.config or any simulation-facing state, and
        # auto-hides itself shortly after the last change so it never becomes
        # a permanent, easy-to-forget overlay.
        self._grid_resolution_preview_active = False
        self._grid_resolution_preview_resolution: float | None = None
        self._grid_resolution_preview_timer = QTimer(self)
        self._grid_resolution_preview_timer.setSingleShot(True)
        self._grid_resolution_preview_timer.timeout.connect(self.hide_grid_resolution_preview)

        # Persistent "Show Grid" overlay ("Grid Overlay" toggle). Unlike the
        # temporary preview above, this does not auto-hide -- it stays on
        # until the user turns it off, including while the simulation is
        # running. Purely visual/debug: it never touches self.config and
        # never rebuilds any occupancy/planning grid. _grid_overlay_snapshot
        # is an optional read-only copy of the current belief/occupancy
        # grid (resolution/bounds/grid array) pushed in from outside, used
        # to color occupied/free/unknown cells while running; when absent
        # (not running, or no belief map yet) only resolution grid lines
        # are drawn.
        self.grid_overlay_enabled = False
        self._grid_overlay_resolution = 0.50
        self._grid_overlay_snapshot: dict | None = None
        self._grid_overlay_snapshot_version = 0
        self._grid_overlay_snapshot_pushed_at: float | None = None

        # Rendered-overlay cache. Rebuilding requires looping over every
        # visible occupancy cell and issuing a QPainter.drawRect() call per
        # cell -- fine once, ruinous if repeated every frame at a fine
        # grid_resolution. The cache is reused as long as resolution, canvas
        # size, view bounds, and the occupancy snapshot are all unchanged;
        # otherwise it is rebuilt once and reused again.
        self._grid_overlay_cache: QPixmap | None = None
        self._grid_overlay_cache_key: tuple | None = None
        self._grid_overlay_last_cache_status = "off"
        self._grid_overlay_last_visible_cells = 0
        self._grid_overlay_degraded = False

        # Render-only FPS/frame-time telemetry. Independent of the engine --
        # this only ever measures how fast paintEvent itself is running.
        # Routine samples are NEVER printed to stdout/terminal and NEVER
        # appended to the GUI console (that would just trade one spam
        # problem for another) -- they are only kept in-memory as
        # latest_perf_status, inspectable by an optional in-app "Show FPS"
        # display without any terminal or GUI console output. Only a
        # genuinely severe, much less frequent FPS drop reaches the GUI
        # console, via _perf_gui_warning_gate.
        self._render_perf_monitor = RenderPerfMonitor()
        self._perf_gui_warning_gate = PerfGuiWarningGate()
        # Throttles only the high-frequency, simulation-driven repaint
        # requests (set_runtime_state()/set_multi_runtime_state()) to at
        # most DEFAULT_RENDER_THROTTLE_FPS repaints/second -- see
        # RenderThrottler's docstring. Does not affect any other
        # self.update() call in this class.
        self._render_throttler = RenderThrottler()
        self.latest_perf_status: dict | None = None
        # Gates GUI-console perf warnings only (see
        # _maybe_emit_perf_gui_warning/draw_grid_overlay's degraded notice)
        # -- a low paint_fps during setup/load/reset, or with the overlay
        # off, is not meaningful and must not be reported as if Show Grid
        # were the cause. Set via set_simulation_running_for_perf().
        self._simulation_running_for_perf = False
        # Tracks whether the one-time "grid overlay degraded" console line
        # has already been shown for the CURRENT run + degraded streak --
        # separate from _grid_overlay_degraded (which also gates cache-key/
        # snapshot-throttle logic and must stay accurate even while idle).
        self._grid_overlay_degraded_notice_shown = False

        # Dragging support for pre-simulation multi-robot placement.
        self.dragging_robot_index: int | None = None
        self.dragging_robot_offset: tuple[float, float] = (0.0, 0.0)
        self.editor_mode = False
        self.editor_tool = "rectangles"
        self.editor_drag_start: tuple[float, float] | None = None
        self.editor_drag_current: tuple[float, float] | None = None
        self.editor_preview_points: list[tuple[float, float]] = []
        self.editor_pan_offset: tuple[float, float] = (0.0, 0.0)
        self.editor_zoom = 1.0
        self.editor_brush_size = 0.2
        self.editor_interaction_mode = "paint"
        self.editor_pan_active = False
        self.editor_last_pan_pos: tuple[float, float] | None = None
        self.editor_camera_active_handle: str | None = None
        self.editor_camera_drag_start_world: tuple[float, float] | None = None
        self.editor_camera_start_bounds: tuple[float, float, float, float] | None = None
        self.editor_obstacle_drag_index: int | None = None
        self.editor_obstacle_drag_indices: list[int] = []
        self.editor_obstacle_drag_offset: tuple[float, float] = (0.0, 0.0)
        self.editor_obstacle_drag_last_world: tuple[float, float] | None = None

        # Cached current blue sensor footprint. This avoids recomputing
        # ray-casting in every paintEvent when the robot moved only a tiny
        # amount since the previous frame.
        self._sensor_polygon_cache: list[tuple[float, float]] = []
        self._sensor_polygon_pose: tuple[float, float, float] | None = None
        self._sensor_polygon_signature: tuple | None = None
        self._sensor_polygon_caches_by_robot: dict[int, tuple[tuple[float, float, float], tuple, list[tuple[float, float]]]] = {}

    def resizeEvent(self, event):
        self.invalidate_static_plot_cache()
        self.invalidate_explored_area_cache()
        self.invalidate_mapped_points_cache()
        self.invalidate_obstacles_cache()
        super().resizeEvent(event)

    def invalidate_static_plot_cache(self):
        self._static_plot_cache = None
        self._static_plot_cache_size = None

    def invalidate_explored_area_cache(self):
        self._explored_area_cache = None
        self._explored_area_cache_size = None
        self._explored_area_cached_count = 0
        self._explored_area_caches_by_robot = {}
        self._explored_area_cache_sizes_by_robot = {}

    def invalidate_mapped_points_cache(self):
        self._mapped_points_cache = None
        self._mapped_points_cache_size = None
        self._mapped_points_cached_count = 0

    def invalidate_obstacle_coverage_cache(self):
        self._obstacle_coverage_cache = {}
        self._obstacle_coverage_cache_count = -1

    def invalidate_obstacles_cache(self):
        self._obstacles_cache = None
        self._obstacles_cache_size = None
        self._obstacles_cache_mapped_count = -1
        self._obstacles_cache_signature = None

    def invalidate_view_transform_caches(self):
        """
        Invalidate all pixmap caches whose pixels depend on world_to_screen().

        Pan/zoom changes do not change widget size, but they do change the
        world-to-screen transform. Any cached layer drawn in screen coordinates
        must be rebuilt after camera movement.
        """
        self.invalidate_static_plot_cache()
        self.invalidate_explored_area_cache()
        self.invalidate_mapped_points_cache()
        self.invalidate_obstacles_cache()

    def invalidate_sensor_cache(self):
        self._sensor_polygon_cache = []
        self._sensor_polygon_pose = None
        self._sensor_polygon_signature = None
        self._sensor_polygon_caches_by_robot = {}

    def set_preview_config(self, config: SimulationConfig):
        previous_spacing = getattr(self.config, "mapping_point_spacing", None)
        previous_obstacles = getattr(self.config, "obstacles", None)
        previous_vision = getattr(self.config, "vision", None)
        previous_vision_model = getattr(self.config, "vision_model", None)
        previous_camera = (
            getattr(self.config, "camera_center_x", None),
            getattr(self.config, "camera_center_y", None),
            getattr(self.config, "camera_width", None),
            getattr(self.config, "camera_height", None),
        )
        self.config = config
        if previous_spacing != config.mapping_point_spacing or previous_obstacles != config.obstacles:
            self.invalidate_obstacle_coverage_cache()
            self.invalidate_obstacles_cache()
        if (
            previous_obstacles != config.obstacles
            or previous_vision != config.vision
            or previous_vision_model != config.vision_model
        ):
            self.invalidate_sensor_cache()

        current_camera = (
            getattr(config, "camera_center_x", None),
            getattr(config, "camera_center_y", None),
            getattr(config, "camera_width", None),
            getattr(config, "camera_height", None),
        )
        if previous_camera != current_camera:
            self.invalidate_view_transform_caches()

        self.update()

    def set_robot(self, robot):
        self.robot = robot
        if robot is not None:
            self.robots = []
            self.multi_path_points = []
            self.multi_planned_path_points = []
            self.multi_last_controls = []
            self.multi_exploration_targets = []
        self.update()

    def set_multi_robots(
        self,
        robots,
        path_points=None,
        last_controls=None,
        planned_path_points=None,
        exploration_targets=None,
    ):
        self.robots = list(robots or [])
        self.robot = self.robots[0] if self.robots else None
        if path_points is not None:
            self.multi_path_points = [list(path) for path in path_points]
        if planned_path_points is not None:
            self.multi_planned_path_points = [list(path) for path in planned_path_points]
        if last_controls is not None:
            self.multi_last_controls = list(last_controls)
        if exploration_targets is not None:
            self.multi_exploration_targets = [None if target is None else tuple(target) for target in exploration_targets]
        self.update()

    def set_path(self, path_points):
        self.path_points = path_points
        self.update()

    def set_planned_path(self, planned_path_points):
        self.planned_path_points = planned_path_points
        self.update()

    def set_exploration_target(self, target_xy):
        self.exploration_target_xy = None if target_xy is None else tuple(target_xy)
        self.update()

    def set_multi_exploration_targets(self, targets):
        """Store one exploration target per robot for drawing independent F markers."""
        self.multi_exploration_targets = [None if target is None else tuple(target) for target in (targets or [])]
        self.update()

    def set_explored_area_polygons(self, polygons):
        new_polygons = [list(polygon) for polygon in polygons]

        # Incremental update: if polygons were appended, paint only the new
        # polygons onto the explored-area pixmap. If the history was reset or
        # truncated, rebuild the cache once.
        previous_count = len(self.explored_area_polygons)
        self.explored_area_polygons = new_polygons

        if len(new_polygons) == 0:
            self.invalidate_explored_area_cache()
        elif (
            self._explored_area_cache is not None
            and self._explored_area_cache_size == self.size()
            and len(new_polygons) > previous_count
            and previous_count == self._explored_area_cached_count
        ):
            for polygon in new_polygons[previous_count:]:
                self.paint_explored_polygon_to_cache(polygon)
            self._explored_area_cached_count = len(new_polygons)
        else:
            self.rebuild_explored_area_cache()

        self.update()

    def append_explored_area_polygon(self, polygon: list[tuple[float, float]], robot_index: int | None = None):
        """
        Append one explored sensor footprint without copying the whole history.

        For single-robot mode the footprint is painted into the standard blue
        homogeneous cache. For multi-robot mode each robot gets its own colored
        cache, so coverage remains attributable without cluttering the main UI.
        """
        if len(polygon) < 3:
            return

        polygon_copy = list(polygon)
        self.explored_area_polygons.append(polygon_copy)
        if len(self.explored_area_polygons) > EXPLORED_POLYGON_HISTORY_LIMIT:
            self.explored_area_polygons = self.explored_area_polygons[-EXPLORED_POLYGON_HISTORY_LIMIT:]

        if robot_index is None:
            if (
                self._explored_area_cache is None
                or self._explored_area_cache_size != self.size()
            ):
                self.rebuild_explored_area_cache()
            else:
                self.paint_explored_polygon_to_cache(polygon_copy, robot_index=None)
                self._explored_area_cached_count = len(self.explored_area_polygons)
        else:
            self.paint_explored_polygon_to_cache(polygon_copy, robot_index=int(robot_index))

        self.update()

    def set_runtime_state(
        self,
        robot=None,
        path_points=None,
        last_control=None,
        simulation_time: float | None = None,
        simulation_speed: float | None = None,
    ):
        """
        Update high-frequency runtime data with a single repaint request.

        The old code called update() three times per physics tick via separate
        setters. At 60 FPS, redundant repaint requests can become visible as
        frame jitter.
        """
        if robot is not None:
            self.robot = robot
        if path_points is not None:
            self.path_points = path_points
        if last_control is not None:
            self.last_control = last_control
        if simulation_time is not None:
            self.simulation_time = float(simulation_time)
        if simulation_speed is not None:
            self.simulation_speed = float(simulation_speed)
        if self._render_throttler.should_render():
            self.update()

    def set_multi_runtime_state(
        self,
        robots=None,
        path_points=None,
        last_controls=None,
        planned_path_points=None,
        exploration_targets=None,
        simulation_time: float | None = None,
        simulation_speed: float | None = None,
    ):
        if robots is not None:
            self.robots = list(robots)
            self.robot = self.robots[0] if self.robots else None
        if path_points is not None:
            self.multi_path_points = [list(path) for path in path_points]
        if planned_path_points is not None:
            self.multi_planned_path_points = [list(path) for path in planned_path_points]
        if last_controls is not None:
            self.multi_last_controls = list(last_controls)
        if exploration_targets is not None:
            self.multi_exploration_targets = [None if target is None else tuple(target) for target in exploration_targets]
        if simulation_time is not None:
            self.simulation_time = float(simulation_time)
        if simulation_speed is not None:
            self.simulation_speed = float(simulation_speed)
        if self._render_throttler.should_render():
            self.update()

    def set_simulation_metrics(self, simulation_time: float, simulation_speed: float):
        self.simulation_time = float(simulation_time)
        self.simulation_speed = float(simulation_speed)
        self.update()

    def record_render_frame(self):
        """
        Estimate user-visible FPS from paintEvent calls.

        This deliberately measures rendering cadence, not physics updates. The
        value is refreshed about four times per second so the telemetry does not
        create extra repaint pressure by itself.
        """
        self._fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_last_time

        if elapsed >= 0.25:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_last_time = now

    def set_known_obstacles(self, obstacles):
        # Backward-compatible setter. Full obstacle rectangles are no longer
        # revealed during mapping, but this keeps older calls harmless.
        self.known_obstacles = [tuple(obstacle) for obstacle in obstacles]
        self.update()

    def set_mapped_obstacle_points(self, points):
        new_points = [tuple(point) for point in points]
        previous_count = len(self.mapped_obstacle_points)
        self.mapped_obstacle_points = new_points
        self.invalidate_obstacle_coverage_cache()
        self.invalidate_obstacles_cache()

        if len(new_points) == 0:
            self.invalidate_mapped_points_cache()
        elif (
            self._mapped_points_cache is not None
            and self._mapped_points_cache_size == self.size()
            and len(new_points) > previous_count
            and previous_count == self._mapped_points_cached_count
        ):
            self.paint_mapped_points_to_cache(new_points[previous_count:])
            self._mapped_points_cached_count = len(new_points)
        else:
            self.rebuild_mapped_points_cache()

        self.update()

    def append_mapped_obstacle_points(self, points: list[tuple[float, float]]):
        """
        Append newly sensed obstacle points without copying/rebuilding the full
        point cloud cache.

        This removes a growing cost that appeared late in long simulations. The
        gray obstacle opacity cache is refreshed only every
        OBSTACLE_VISUAL_REFRESH_POINT_STEP points because it is visual feedback,
        not collision logic.
        """
        if not points:
            return

        new_points = [tuple(point) for point in points]
        self.mapped_obstacle_points.extend(new_points)

        if (
            self._mapped_points_cache is None
            or self._mapped_points_cache_size != self.size()
        ):
            self.rebuild_mapped_points_cache()
        else:
            self.paint_mapped_points_to_cache(new_points)
            self._mapped_points_cached_count = len(self.mapped_obstacle_points)

        # Do not rebuild the obstacle opacity layer after every single sensor
        # point. That was the main cause of FPS falling as mapping progressed.
        if (
            self._obstacles_cache_mapped_count < 0
            or len(self.mapped_obstacle_points) - self._obstacles_cache_mapped_count
            >= OBSTACLE_VISUAL_REFRESH_POINT_STEP
        ):
            self.invalidate_obstacle_coverage_cache()
            self.invalidate_obstacles_cache()

        self.update()

    def _append_status_history(self, message: str) -> None:
        raw_message = str(message).strip()
        if not raw_message:
            return

        timestamp = time.strftime("%H:%M:%S")
        for line in raw_message.splitlines():
            line = line.strip()
            if not line:
                continue
            entry = f"[{timestamp}] {line}"

            # Avoid flooding the console with repeated status messages emitted by
            # periodic replanning gates. The latest visible status is still updated
            # every time; only identical consecutive console lines are collapsed.
            if self.status_history and self.status_history[-1].endswith(line):
                continue

            self.status_history.append(entry)

        if len(self.status_history) > self.status_history_limit:
            self.status_history = self.status_history[-self.status_history_limit:]

    def append_console_message(self, message: str) -> None:
        """Append a message to the console history without changing the top status."""
        self._append_status_history(message)

    def set_status(self, message: str):
        self.status_message = str(message)
        self._append_status_history(self.status_message)
        self.update()

    def status_history_lines(self) -> list[str]:
        return list(self.status_history)

    def clear_status_history(self) -> None:
        self.status_history.clear()
        self._append_status_history("Console cleared.")
        self.update()

    def set_last_control(self, control):
        self.last_control = control
        self.update()

    def plot_rect(self):
        return self.rect().adjusted(
            self.plot_margin_left,
            self.plot_margin_top,
            -self.plot_margin_right,
            -self.plot_margin_bottom,
        )

    def editor_view_span_world(self) -> tuple[float, float]:
        """Return the world span currently visible in editor mode."""
        zoom = max(0.10, float(self.editor_zoom))
        return (
            max(0.25, (WORLD_X_MAX - WORLD_X_MIN) / zoom),
            max(0.25, (WORLD_Y_MAX - WORLD_Y_MIN) / zoom),
        )

    def simulation_camera_span_world(self) -> tuple[float, float]:
        """Return the simulation camera span stored in the config."""
        return (
            max(0.50, float(getattr(self.config, "camera_width", WORLD_X_MAX - WORLD_X_MIN))),
            max(0.50, float(getattr(self.config, "camera_height", WORLD_Y_MAX - WORLD_Y_MIN))),
        )

    def active_view_center_world(self) -> tuple[float, float]:
        if self.editor_mode:
            return (float(self.editor_pan_offset[0]), float(self.editor_pan_offset[1]))
        return (
            float(getattr(self.config, "camera_center_x", (WORLD_X_MIN + WORLD_X_MAX) / 2.0)),
            float(getattr(self.config, "camera_center_y", (WORLD_Y_MIN + WORLD_Y_MAX) / 2.0)),
        )

    def active_view_span_world(self) -> tuple[float, float]:
        if self.editor_mode:
            return self.editor_view_span_world()
        return self.simulation_camera_span_world()

    def active_view_bounds_world(self) -> tuple[float, float, float, float]:
        """Return left, right, bottom, top of the visible world rectangle."""
        center_x, center_y = self.active_view_center_world()
        span_x, span_y = self.active_view_span_world()
        return (
            center_x - span_x / 2.0,
            center_x + span_x / 2.0,
            center_y - span_y / 2.0,
            center_y + span_y / 2.0,
        )

    def world_to_screen(self, x: float, y: float):
        rect = self.plot_rect()
        center_x, center_y = self.active_view_center_world()
        span_x, span_y = self.active_view_span_world()
        sx = rect.left() + (rect.width() / 2.0) + (float(x) - center_x) * (rect.width() / span_x)
        sy = rect.bottom() - (rect.height() / 2.0) - (float(y) - center_y) * (rect.height() / span_y)
        return sx, sy

    def screen_to_world(self, sx: float, sy: float):
        rect = self.plot_rect()
        center_x, center_y = self.active_view_center_world()
        span_x, span_y = self.active_view_span_world()
        x = center_x + ((float(sx) - (rect.left() + rect.width() / 2.0)) / rect.width()) * span_x
        y = center_y - ((float(sy) - (rect.bottom() - rect.height() / 2.0)) / rect.height()) * span_y
        return x, y

    def telemetry_rect(self):
        r = self.rect()
        return r.adjusted(30, r.height() - 50, -30, -16)

    def metrics_rect(self) -> QRectF:
        """
        Center-top badge for FPS, simulation time and simulation speed.
        """
        width = min(272.0, max(224.0, self.width() * 0.30))
        height = 25.0
        eye_width = 28.0
        gap = 6.0
        group_width = width + gap + eye_width
        x = (self.width() - group_width) / 2.0
        y = 16.0
        return QRectF(x, y, width, height)

    def metrics_eye_rect(self) -> QRectF:
        """Return the clickable eye button rectangle."""
        height = 25.0
        eye_width = 28.0
        y = 16.0

        if self.metrics_visible:
            metrics = self.metrics_rect()
            return QRectF(metrics.right() + 6.0, y, eye_width, height)

        # When metrics are hidden, keep only the eye button centered so the user
        # can bring the counters back without searching elsewhere.
        return QRectF((self.width() - eye_width) / 2.0, y, eye_width, height)

    def metrics_reserved_rect(self) -> QRectF:
        """Area reserved by the metric controls in the header row."""
        eye = self.metrics_eye_rect()
        if not self.metrics_visible:
            return eye
        metrics = self.metrics_rect()
        return QRectF(metrics.left(), metrics.top(), eye.right() - metrics.left(), metrics.height())


    def multi_robot_screen_positions(self) -> list[tuple[int, float, float, RobotStartConfig]]:
        if "Multiple" not in self.config.agent_mode:
            return []

        robots = normalized_robot_start_configs(self.config)
        positions: list[tuple[int, float, float, RobotStartConfig]] = []
        for index, robot_cfg in enumerate(robots):
            sx, sy = self.world_to_screen(robot_cfg.x, robot_cfg.y)
            positions.append((index, sx, sy, robot_cfg))
        return positions

    def pixels_per_meter(self) -> float:
        span_x, _ = self.active_view_span_world()
        return max(1.0, self.plot_rect().width() / max(0.1, span_x))

    def robot_index_at_screen_position(self, sx: float, sy: float) -> tuple[int, RobotStartConfig] | None:
        """
        Return the preview robot under the cursor before the simulation starts.

        Index convention:
            -1  -> single-robot preview
             0+ -> multi-robot preview robot index

        Runtime robots are intentionally not draggable here. Dragging during
        simulation would teleport the state and invalidate dynamics/collision
        metrics.
        """
        if self.robot is not None or self.robots:
            return None

        px_per_meter = self.pixels_per_meter()
        body_px = max(7.0, float(self.config.body_radius) * px_per_meter)
        hit_radius = max(13.0, body_px + 5.0)

        if "Multiple" not in self.config.agent_mode:
            rx, ry = self.world_to_screen(float(self.config.x), float(self.config.y))
            if math.hypot(float(sx) - rx, float(sy) - ry) <= hit_radius:
                return -1, RobotStartConfig(
                    x=float(self.config.x),
                    y=float(self.config.y),
                    theta=float(self.config.theta),
                    v=float(self.config.v),
                )
            return None

        # Reverse order so the visually topmost/highest-index robot is easier to pick.
        for index, rx, ry, robot_cfg in reversed(self.multi_robot_screen_positions()):
            if math.hypot(float(sx) - rx, float(sy) - ry) <= hit_radius:
                return index, robot_cfg

        return None

    def set_editor_mode(self, enabled: bool) -> None:
        self.editor_mode = bool(enabled)
        self.editor_drag_start = None
        self.editor_drag_current = None
        self.editor_preview_points = []
        self.editor_pan_active = False
        self.editor_last_pan_pos = None
        self.editor_camera_active_handle = None
        self.editor_camera_drag_start_world = None
        self.editor_camera_start_bounds = None
        self.editor_obstacle_drag_index = None
        self.editor_obstacle_drag_indices = []
        self.editor_obstacle_drag_offset = (0.0, 0.0)
        self.editor_obstacle_drag_last_world = None
        if not self.editor_mode:
            self.editor_pan_offset = (0.0, 0.0)
            self.editor_zoom = 1.0
            self.invalidate_view_transform_caches()
            self.editor_view_changed.emit()
            self.update()
            return

        self.fit_to_obstacles(self.config.obstacles)

    def set_editor_tool(self, tool: str) -> None:
        self.editor_tool = str(tool)
        self.editor_drag_start = None
        self.editor_drag_current = None
        self.editor_preview_points = []
        self.editor_camera_active_handle = None
        self.editor_camera_drag_start_world = None
        self.editor_camera_start_bounds = None
        self.editor_obstacle_drag_index = None
        self.editor_obstacle_drag_indices = []
        self.editor_obstacle_drag_offset = (0.0, 0.0)
        self.editor_obstacle_drag_last_world = None
        self.editor_view_changed.emit()
        self.update()

    def set_editor_drag_start(self, start_xy: tuple[float, float]) -> None:
        self.editor_drag_start = tuple(start_xy)
        self.editor_drag_current = tuple(start_xy)
        self.editor_preview_points = [tuple(start_xy)]
        self.update()

    def set_editor_brush_size(self, brush_size: float) -> None:
        self.editor_brush_size = max(0.05, float(brush_size))
        self.invalidate_obstacles_cache()
        self.update()

    def set_editor_interaction_mode(self, mode: str) -> None:
        mode_name = str(mode).lower()
        self.editor_interaction_mode = "move" if mode_name == "move" else "paint"
        self.editor_drag_start = None
        self.editor_drag_current = None
        self.editor_preview_points = []
        self.editor_pan_active = False
        self.editor_last_pan_pos = None
        self.editor_camera_active_handle = None
        self.editor_camera_drag_start_world = None
        self.editor_camera_start_bounds = None
        self.editor_obstacle_drag_index = None
        self.editor_obstacle_drag_indices = []
        self.editor_obstacle_drag_offset = (0.0, 0.0)
        self.editor_obstacle_drag_last_world = None
        self.update()

    def fit_to_obstacles(self, obstacles: list[tuple[float, float, float, float]]) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return

        if not obstacles:
            self.editor_pan_offset = ((WORLD_X_MIN + WORLD_X_MAX) / 2.0, (WORLD_Y_MIN + WORLD_Y_MAX) / 2.0)
            self.editor_zoom = 1.0
            self.invalidate_view_transform_caches()
            self.editor_view_changed.emit()
            self.update()
            return

        xs = [obstacle[0] for obstacle in obstacles] + [obstacle[0] + obstacle[2] for obstacle in obstacles]
        ys = [obstacle[1] for obstacle in obstacles] + [obstacle[1] + obstacle[3] for obstacle in obstacles]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        padding_x = max(0.5, span_x * 0.08)
        padding_y = max(0.5, span_y * 0.08)

        world_span_x = max(span_x + padding_x * 2.0, 1.0)
        world_span_y = max(span_y + padding_y * 2.0, 1.0)
        zoom_x = (WORLD_X_MAX - WORLD_X_MIN) / world_span_x
        zoom_y = (WORLD_Y_MAX - WORLD_Y_MIN) / world_span_y
        self.editor_zoom = max(0.35, min(3.0, min(zoom_x, zoom_y)))
        self.editor_pan_offset = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        self.invalidate_view_transform_caches()
        self.editor_view_changed.emit()
        self.update()

    def editor_status_text(self) -> str:
        zoom_percent = max(0.0, self.editor_zoom * 100.0)
        cam_x = float(getattr(self.config, "camera_center_x", 0.0))
        cam_y = float(getattr(self.config, "camera_center_y", 0.0))
        cam_w = float(getattr(self.config, "camera_width", WORLD_X_MAX - WORLD_X_MIN))
        cam_h = float(getattr(self.config, "camera_height", WORLD_Y_MAX - WORLD_Y_MIN))
        return (
            f"Editor zoom {zoom_percent:.0f}%  ·  View center ({self.editor_pan_offset[0]:.1f}, {self.editor_pan_offset[1]:.1f})  ·  "
            f"Simulation camera center ({cam_x:.1f}, {cam_y:.1f}) size {cam_w:.1f} × {cam_h:.1f} m"
        )

    def camera_bounds_world(self) -> tuple[float, float, float, float]:
        """Return simulation camera left, right, bottom, top in world coordinates."""
        center_x = float(getattr(self.config, "camera_center_x", 0.0))
        center_y = float(getattr(self.config, "camera_center_y", 0.0))
        width = max(0.50, float(getattr(self.config, "camera_width", WORLD_X_MAX - WORLD_X_MIN)))
        height = max(0.50, float(getattr(self.config, "camera_height", WORLD_Y_MAX - WORLD_Y_MIN)))
        return (
            center_x - width / 2.0,
            center_x + width / 2.0,
            center_y - height / 2.0,
            center_y + height / 2.0,
        )

    def camera_rect_screen(self) -> QRectF:
        left, right, bottom, top = self.camera_bounds_world()
        x1, y1 = self.world_to_screen(left, bottom)
        x2, y2 = self.world_to_screen(right, top)
        return QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def set_camera_view(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        *,
        emit_signal: bool = False,
    ) -> None:
        """Update the red editor camera rectangle and simulation viewport."""
        width = max(0.50, float(width))
        height = max(0.50, float(height))
        self.config.camera_center_x = float(center_x)
        self.config.camera_center_y = float(center_y)
        self.config.camera_width = width
        self.config.camera_height = height
        self.invalidate_view_transform_caches()
        self.editor_view_changed.emit()
        if emit_signal:
            self.editor_camera_changed.emit((
                self.config.camera_center_x,
                self.config.camera_center_y,
                self.config.camera_width,
                self.config.camera_height,
            ))
        self.update()

    def camera_handle_at_screen_position(self, sx: float, sy: float) -> str | None:
        """Return resize/move handle under the cursor for the camera frame."""
        rect = self.camera_rect_screen()
        if rect.isNull() or rect.width() <= 0.0 or rect.height() <= 0.0:
            return None

        point = QPointF(float(sx), float(sy))
        handle_radius = 10.0
        corners = {
            "nw": rect.topLeft(),
            "ne": rect.topRight(),
            "sw": rect.bottomLeft(),
            "se": rect.bottomRight(),
        }
        for name, corner in corners.items():
            if math.hypot(point.x() - corner.x(), point.y() - corner.y()) <= handle_radius:
                return name

        edge_tol = 7.0
        if rect.left() - edge_tol <= point.x() <= rect.right() + edge_tol:
            if abs(point.y() - rect.top()) <= edge_tol:
                return "n"
            if abs(point.y() - rect.bottom()) <= edge_tol:
                return "s"
        if rect.top() - edge_tol <= point.y() <= rect.bottom() + edge_tol:
            if abs(point.x() - rect.left()) <= edge_tol:
                return "w"
            if abs(point.x() - rect.right()) <= edge_tol:
                return "e"

        if rect.adjusted(0, 0, 0, 0).contains(point):
            return "move"
        return None

    def update_camera_from_drag(self, current_world: tuple[float, float]) -> None:
        if (
            self.editor_camera_active_handle is None
            or self.editor_camera_drag_start_world is None
            or self.editor_camera_start_bounds is None
        ):
            return

        start_x, start_y = self.editor_camera_drag_start_world
        dx = float(current_world[0]) - start_x
        dy = float(current_world[1]) - start_y
        left, right, bottom, top = self.editor_camera_start_bounds
        handle = self.editor_camera_active_handle
        min_size = 0.75

        if handle == "move":
            left += dx
            right += dx
            bottom += dy
            top += dy
        else:
            if "w" in handle:
                left += dx
            if "e" in handle:
                right += dx
            if "s" in handle:
                bottom += dy
            if "n" in handle:
                top += dy

            if right - left < min_size:
                if "w" in handle:
                    left = right - min_size
                else:
                    right = left + min_size
            if top - bottom < min_size:
                if "s" in handle:
                    bottom = top - min_size
                else:
                    top = bottom + min_size

        center_x = (left + right) / 2.0
        center_y = (bottom + top) / 2.0
        width = right - left
        height = top - bottom
        self.set_camera_view(center_x, center_y, width, height, emit_signal=True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position()

            if self.metrics_eye_rect().contains(QPointF(pos.x(), pos.y())):
                self.metrics_visible = not self.metrics_visible
                self.update()
                return

            if self.plot_rect().contains(pos.toPoint()):
                if self.editor_mode and not self.robot and not self.robots:
                    # Pan/Zoom mode is exclusive. It must never select, move,
                    # erase, resize the viewport, or create obstacles.
                    if self.editor_interaction_mode == "move":
                        self.editor_pan_active = True
                        self.editor_last_pan_pos = (pos.x(), pos.y())
                        self.setCursor(Qt.ClosedHandCursor)
                        return

                    world_x, world_y = self.screen_to_world(pos.x(), pos.y())

                    if self.editor_tool == "camera":
                        handle = self.camera_handle_at_screen_position(pos.x(), pos.y())
                        if handle is not None:
                            self.editor_camera_interaction_started.emit()
                            self.editor_camera_active_handle = handle
                            self.editor_camera_drag_start_world = (float(world_x), float(world_y))
                            self.editor_camera_start_bounds = self.camera_bounds_world()
                            self.setCursor(Qt.ClosedHandCursor if handle == "move" else Qt.SizeAllCursor)
                            return
                        # Camera mode should never create obstacles by accident.
                        return

                    if self.editor_tool == "erase":
                        self.editor_drag_start = (world_x, world_y)
                        self.editor_drag_current = self.editor_drag_start
                        self.editor_interaction_started.emit(self.editor_drag_start)
                        self.update()
                        return

                    # Object movement is no longer a separate tool. In edit mode,
                    # clicking an existing connected object starts dragging it;
                    # clicking empty space keeps the currently selected draw tool.
                    group_indices = find_obstacle_group_at(self.config.obstacles, (world_x, world_y))
                    if group_indices:
                        self.editor_obstacle_move_started.emit()
                        self.editor_obstacle_drag_index = int(group_indices[-1])
                        self.editor_obstacle_drag_indices = list(group_indices)
                        self.editor_obstacle_drag_last_world = (float(world_x), float(world_y))
                        self.setCursor(Qt.ClosedHandCursor)
                        self.update()
                        return

                    self.editor_drag_start = (world_x, world_y)
                    self.editor_drag_current = self.editor_drag_start
                    self.editor_interaction_started.emit(self.editor_drag_start)
                    self.update()
                    return

                hit = self.robot_index_at_screen_position(pos.x(), pos.y())
                if hit is not None:
                    index, robot_cfg = hit
                    self.dragging_robot_index = index
                    world_x, world_y = self.screen_to_world(pos.x(), pos.y())
                    self.dragging_robot_offset = (robot_cfg.x - world_x, robot_cfg.y - world_y)
                    if index >= 0:
                        self.robotSelected.emit(index)
                    self.setCursor(Qt.ClosedHandCursor)
                    return

                x, y = self.screen_to_world(pos.x(), pos.y())
                self.goalClicked.emit(x, y)

    def mouseMoveEvent(self, event):
        if self.editor_mode and self.editor_pan_active and self.editor_last_pan_pos is not None:
            pos = event.position()
            dx = pos.x() - self.editor_last_pan_pos[0]
            dy = pos.y() - self.editor_last_pan_pos[1]
            span_x, span_y = self.editor_view_span_world()
            self.editor_pan_offset = (
                self.editor_pan_offset[0] - dx * span_x / max(1.0, self.plot_rect().width()),
                self.editor_pan_offset[1] + dy * span_y / max(1.0, self.plot_rect().height()),
            )
            self.editor_last_pan_pos = (pos.x(), pos.y())
            self.invalidate_view_transform_caches()
            self.editor_view_changed.emit()
            self.update()
            return

        if self.editor_mode and self.editor_camera_active_handle is not None:
            pos = event.position()
            self.update_camera_from_drag(self.screen_to_world(pos.x(), pos.y()))
            return

        if self.editor_mode and self.editor_obstacle_drag_indices and self.editor_obstacle_drag_last_world is not None:
            pos = event.position()
            world_x, world_y = self.screen_to_world(pos.x(), pos.y())
            last_x, last_y = self.editor_obstacle_drag_last_world
            dx = float(world_x) - float(last_x)
            dy = float(world_y) - float(last_y)
            if abs(dx) > 1.0e-9 or abs(dy) > 1.0e-9:
                self.editor_obstacle_moved.emit((tuple(self.editor_obstacle_drag_indices), dx, dy))
                self.editor_obstacle_drag_last_world = (float(world_x), float(world_y))
            self.update()
            return

        if self.editor_mode and self.editor_drag_start is not None:
            pos = event.position()
            world_x, world_y = self.screen_to_world(pos.x(), pos.y())
            self.editor_drag_current = (world_x, world_y)
            if self.editor_tool == "free":
                if not self.editor_preview_points or math.hypot(world_x - self.editor_preview_points[-1][0], world_y - self.editor_preview_points[-1][1]) >= 0.05:
                    self.editor_preview_points.append((world_x, world_y))
                    self.editor_interaction_progress.emit((world_x, world_y))
            self.update()
            return

        if self.dragging_robot_index is None:
            return

        pos = event.position()
        x, y = self.screen_to_world(pos.x(), pos.y())
        dx, dy = self.dragging_robot_offset
        x = clamp(x + dx, WORLD_X_MIN, WORLD_X_MAX)
        y = clamp(y + dy, WORLD_Y_MIN, WORLD_Y_MAX)
        self.robotDragged.emit(int(self.dragging_robot_index), float(x), float(y))

    def mouseReleaseEvent(self, event):
        if self.editor_mode and self.editor_pan_active:
            self.editor_pan_active = False
            self.editor_last_pan_pos = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self.editor_mode and self.editor_camera_active_handle is not None:
            self.editor_camera_active_handle = None
            self.editor_camera_drag_start_world = None
            self.editor_camera_start_bounds = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self.editor_mode and self.editor_obstacle_drag_indices:
            self.editor_obstacle_drag_index = None
            self.editor_obstacle_drag_indices = []
            self.editor_obstacle_drag_offset = (0.0, 0.0)
            self.editor_obstacle_drag_last_world = None
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        if self.editor_mode and self.editor_drag_start is not None:
            pos = event.position()
            world_x, world_y = self.screen_to_world(pos.x(), pos.y())
            self.editor_drag_current = (world_x, world_y)
            self.editor_interaction_finished.emit(self.editor_drag_start, (world_x, world_y))
            self.editor_drag_start = None
            self.editor_drag_current = None
            self.update()
            return

        if self.dragging_robot_index is not None:
            self.dragging_robot_index = None
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        if not self.editor_mode:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        pos = event.position()
        world_before = self.screen_to_world(pos.x(), pos.y())

        zoom_factor = 1.10 if delta > 0 else 0.90
        self.editor_zoom = max(0.35, min(8.0, self.editor_zoom * zoom_factor))

        world_after = self.screen_to_world(pos.x(), pos.y())
        self.editor_pan_offset = (
            self.editor_pan_offset[0] + (world_before[0] - world_after[0]),
            self.editor_pan_offset[1] + (world_before[1] - world_after[1]),
        )
        self.invalidate_view_transform_caches()
        self.editor_view_changed.emit()
        self.update()

    def paintEvent(self, event):
        self.record_render_frame()
        frame_start = time.perf_counter()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self.draw_card(painter)
        self.draw_title(painter)
        self.draw_plot(painter)
        self.draw_telemetry(painter)

        self._report_render_perf(frame_start)

    def _report_render_perf(self, frame_start: float) -> None:
        """Update in-app perf diagnostics from this frame's measured paint
        time. Purely observational -- never touches self.config or any
        simulation state.

        Routine samples are NEVER printed to stdout/terminal and NEVER
        appended to the GUI console: they are only stored in
        latest_perf_status, so an optional "Show FPS" display can read the
        current numbers without any terminal or GUI console output. Only a
        genuinely severe, heavily throttled FPS drop reaches the GUI
        console, via _maybe_emit_perf_gui_warning().
        """
        paint_ms = (time.perf_counter() - frame_start) * 1000.0

        snapshot_age_ms = None
        if self.grid_overlay_enabled and self._grid_overlay_snapshot_pushed_at is not None:
            snapshot_age_ms = (time.perf_counter() - self._grid_overlay_snapshot_pushed_at) * 1000.0

        # record_frame() still throttles its returned formatted line (kept
        # for callers/tests that want the exact [PERF] text), but nothing
        # here prints or GUI-console-appends it -- only the always-current
        # rolling paint_fps/paint_ms values are kept, in latest_perf_status.
        self._render_perf_monitor.record_frame(
            paint_ms=paint_ms,
            overlay_enabled=self.grid_overlay_enabled,
            grid_resolution=self._grid_overlay_resolution,
            visible_cells=self._grid_overlay_last_visible_cells if self.grid_overlay_enabled else None,
            cache_status=self._grid_overlay_last_cache_status,
            snapshot_age_ms=snapshot_age_ms,
        )

        self.latest_perf_status = {
            "paint_fps": self._render_perf_monitor.paint_fps,
            "paint_ms": self._render_perf_monitor.paint_ms,
            "overlay_enabled": self.grid_overlay_enabled,
            "grid_resolution": self._grid_overlay_resolution,
            "visible_cells": self._grid_overlay_last_visible_cells if self.grid_overlay_enabled else None,
            "cache_status": self._grid_overlay_last_cache_status,
            "snapshot_age_ms": snapshot_age_ms,
        }

        self._maybe_emit_perf_gui_warning()

    def _maybe_emit_perf_gui_warning(self) -> None:
        """Append a rare, heavily throttled GUI-console line when paint_fps
        is severely low -- the only case where perf diagnostics reach the
        GUI console at all, since routine samples never do (see
        _report_render_perf's latest_perf_status).

        Gated on simulation_running AND grid_overlay_enabled: a low
        paint_fps during setup/load/reset is not meaningful (nothing is
        actually rendering the overlay yet), and with the overlay off,
        Show Grid cannot be the cause -- reporting it as an "overlay is
        low fps" warning in either case would be a false lead.
        """
        if not self._simulation_running_for_perf or not self.grid_overlay_enabled:
            return

        if self._perf_gui_warning_gate.should_warn(self._render_perf_monitor.paint_fps):
            self.append_console_message(
                format_gui_perf_warning(
                    paint_fps=self._render_perf_monitor.paint_fps,
                    overlay_enabled=self.grid_overlay_enabled,
                    grid_resolution=self._grid_overlay_resolution,
                )
            )

    def draw_card(self, painter: QPainter):
        rect = QRectF(self.rect().adjusted(0, 0, -1, -1))
        path = QPainterPath()
        path.addRoundedRect(rect, 12, 12)
        painter.fillPath(path, QColor(CARD))
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawPath(path)

    def draw_title(self, painter: QPainter):
        """
        Draw the canvas header.

        Layout rule:
            left   -> title
            center -> FPS / simulation time / speed + eye button
            right  -> short status message
        """
        reserved_rect = self.metrics_reserved_rect()

        # Left title. Keep it in its own small area so it never collides with
        # the centered metrics controls.
        painter.setFont(QFont("Segoe UI", 14, QFont.Bold))
        painter.setPen(QColor(TEXT))
        title_rect = QRectF(24, 13, max(120.0, reserved_rect.left() - 36.0), 28)
        title = painter.fontMetrics().elidedText(
            "Simulation Preview",
            Qt.ElideRight,
            int(max(90.0, title_rect.width())),
        )
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, title)

        # Center metrics badge. The eye button remains visible even when the
        # counters are hidden.
        if self.metrics_visible:
            self.draw_metrics_badge(painter, self.metrics_rect())
        self.draw_metrics_eye_button(painter, self.metrics_eye_rect())

        # Right status. Long status messages are elided because the center
        # metrics controls have priority in this header row.
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(TEXT_FAINT))
        status_left = reserved_rect.right() + 16.0
        status_width = max(0.0, self.width() - status_left - 24.0)
        if status_width >= 70.0:
            status_rect = QRectF(status_left, 16, status_width, 22)
            status_text = self.editor_status_text() if self.editor_mode else self.status_message
            status = painter.fontMetrics().elidedText(
                status_text,
                Qt.ElideRight,
                int(status_rect.width()),
            )
            painter.drawText(status_rect, Qt.AlignRight | Qt.AlignVCenter, status)

    def draw_metrics_badge(self, painter: QPainter, rect: QRectF):
        """
        Draw runtime counters in a compact top-center pill.
        """
        painter.save()

        path = QPainterPath()
        path.addRoundedRect(rect, 12.5, 12.5)

        painter.setPen(QPen(QColor(218, 223, 231, 190), 1.0))
        painter.setBrush(QBrush(QColor(255, 255, 255, 218)))
        painter.drawPath(path)

        dot_color = QColor(GREEN) if self.fps >= 50.0 else QColor(ORANGE)
        if self.fps < 35.0 and self.fps > 0.0:
            dot_color = QColor(RED)

        dot_x = rect.left() + 11.0
        dot_y = rect.center().y() - 3.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(QRectF(dot_x, dot_y, 6.0, 6.0))

        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.setPen(QColor(TEXT))

        text = (
            f"FPS {self.fps:04.1f}"
            f"  ·  {self.simulation_time:05.2f}s"
            f"  ·  {self.simulation_speed:.2f}x"
        )
        painter.drawText(
            rect.adjusted(23, 0, -8, 0),
            Qt.AlignVCenter | Qt.AlignLeft,
            text,
        )

        painter.restore()

    def draw_metrics_eye_button(self, painter: QPainter, rect: QRectF):
        """Draw the open/closed eye button used to hide/show counters."""
        painter.save()

        path = QPainterPath()
        path.addRoundedRect(rect, 12.5, 12.5)
        painter.setPen(QPen(QColor(218, 223, 231, 190), 1.0))
        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))
        painter.drawPath(path)

        cx = rect.center().x()
        cy = rect.center().y()
        eye_color = QColor(TEXT if self.metrics_visible else TEXT_MUTED)
        painter.setPen(QPen(eye_color, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)

        eye_path = QPainterPath()
        eye_path.moveTo(cx - 8.5, cy)
        eye_path.cubicTo(cx - 5.5, cy - 5.0, cx + 5.5, cy - 5.0, cx + 8.5, cy)
        eye_path.cubicTo(cx + 5.5, cy + 5.0, cx - 5.5, cy + 5.0, cx - 8.5, cy)
        painter.drawPath(eye_path)

        if self.metrics_visible:
            painter.setBrush(QBrush(eye_color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QRectF(cx - 2.3, cy - 2.3, 4.6, 4.6))
        else:
            painter.setPen(QPen(eye_color, 1.7, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(cx - 8.0, cy + 7.0), QPointF(cx + 8.0, cy - 7.0))

        painter.restore()

    def ensure_static_plot_cache(self):
        if (
            self._static_plot_cache is not None
            and self._static_plot_cache_size == self.size()
        ):
            return

        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)

        cache_painter = QPainter(cache)
        cache_painter.setRenderHint(QPainter.Antialiasing)

        rect = self.plot_rect()
        cache_painter.save()
        cache_painter.setClipRect(rect)
        cache_painter.fillRect(rect, QColor("#F9FBFD"))
        self.draw_grid(cache_painter, rect)
        cache_painter.restore()
        cache_painter.end()

        self._static_plot_cache = cache
        self._static_plot_cache_size = QSize(self.size())

    def polygon_to_screen_path(self, polygon: list[tuple[float, float]]) -> QPainterPath:
        path = QPainterPath()
        if len(polygon) < 3:
            return path

        sx, sy = self.world_to_screen(*polygon[0])
        path.moveTo(sx, sy)

        for point in polygon[1:]:
            px, py = self.world_to_screen(*point)
            path.lineTo(px, py)

        path.closeSubpath()
        return path

    def ensure_explored_area_cache(self):
        if (
            self._explored_area_cache is not None
            and self._explored_area_cache_size == self.size()
        ):
            return

        self.rebuild_explored_area_cache()

    def rebuild_explored_area_cache(self):
        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)
        self._explored_area_cache = cache
        self._explored_area_cache_size = QSize(self.size())
        self._explored_area_cached_count = 0

        for polygon in self.explored_area_polygons:
            self.paint_explored_polygon_to_cache(polygon)

        self._explored_area_cached_count = len(self.explored_area_polygons)

    def ensure_robot_explored_area_cache(self, robot_index: int) -> QPixmap:
        cache = self._explored_area_caches_by_robot.get(int(robot_index))
        if cache is not None and self._explored_area_cache_sizes_by_robot.get(int(robot_index)) == self.size():
            return cache

        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)
        self._explored_area_caches_by_robot[int(robot_index)] = cache
        self._explored_area_cache_sizes_by_robot[int(robot_index)] = QSize(self.size())
        return cache

    def paint_explored_polygon_to_cache(self, polygon: list[tuple[float, float]], robot_index: int | None = None):
        if len(polygon) < 3:
            return

        if robot_index is None:
            if (
                self._explored_area_cache is None
                or self._explored_area_cache_size != self.size()
            ):
                self.rebuild_explored_area_cache()
                return
            target_cache = self._explored_area_cache
            fill_color = QColor(35, 111, 207, 24)
            composition_mode = QPainter.CompositionMode_Source
        else:
            target_cache = self.ensure_robot_explored_area_cache(int(robot_index))
            fill_color = robot_color(int(robot_index))
            fill_color.setAlpha(24)

            # Same principle as single-robot explored area: each robot owns a
            # homogeneous cache. Repainting the same zone by the same robot
            # should not get darker over time. Different robot caches are drawn
            # on top of each other later, so overlap between robots remains
            # visually distinguishable without accumulating within one robot.
            composition_mode = QPainter.CompositionMode_Source

        path = self.polygon_to_screen_path(polygon)
        if path.isEmpty():
            return

        cache_painter = QPainter(target_cache)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        cache_painter.setClipRect(self.plot_rect())
        cache_painter.setCompositionMode(composition_mode)
        cache_painter.setPen(Qt.NoPen)
        cache_painter.setBrush(QBrush(fill_color))
        cache_painter.drawPath(path)
        cache_painter.end()

    def ensure_mapped_points_cache(self):
        if (
            self._mapped_points_cache is not None
            and self._mapped_points_cache_size == self.size()
            and self._mapped_points_cached_count == len(self.mapped_obstacle_points)
        ):
            return
        self.rebuild_mapped_points_cache()

    def rebuild_mapped_points_cache(self):
        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)
        self._mapped_points_cache = cache
        self._mapped_points_cache_size = QSize(self.size())
        self._mapped_points_cached_count = 0

        if self.mapped_obstacle_points:
            self.paint_mapped_points_to_cache(self.mapped_obstacle_points)
            self._mapped_points_cached_count = len(self.mapped_obstacle_points)

    def paint_mapped_points_to_cache(self, points: list[tuple[float, float]]):
        if self._mapped_points_cache is None:
            self.rebuild_mapped_points_cache()
            return

        if not points:
            return

        cache_painter = QPainter(self._mapped_points_cache)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        cache_painter.setClipRect(self.plot_rect())
        cache_painter.setPen(QPen(QColor(130, 0, 42, 150), 0.35))
        cache_painter.setBrush(QBrush(QColor(224, 45, 96, 225)))

        # Keep this tiny. The density comes from mapping_point_spacing, not from
        # drawing large circles.
        point_radius = 0.18

        for px, py in points:
            sx, sy = self.world_to_screen(px, py)
            cache_painter.drawEllipse(
                QRectF(
                    sx - point_radius,
                    sy - point_radius,
                    2 * point_radius,
                    2 * point_radius,
                )
            )

        cache_painter.end()

    def draw_plot(self, painter: QPainter):
        rect = self.plot_rect()

        painter.save()
        painter.setClipRect(rect)

        self.ensure_static_plot_cache()
        if self._static_plot_cache is not None:
            painter.drawPixmap(0, 0, self._static_plot_cache)
        else:
            painter.fillRect(rect, QColor("#F9FBFD"))
            self.draw_grid(painter, rect)

        # Persistent "Show Grid" overlay, drawn just above the background so
        # every other layer below (obstacles, mapped points, routes, robot,
        # safety radius, FoV, labels) stays clearly visible on top of it.
        self.draw_grid_overlay(painter, rect)

        # Always-visible physical world layers.
        # These are not "robot orders"; they are what the simulation world
        # actually contains or what the robot has already sensed.
        self.draw_explored_area_trace(painter)
        self.draw_sensor_range(painter)

        if self.config.show_robot_orders:
            self.draw_safety_radius(painter)

        # Ground-truth obstacles are a human-facing visual layer. They can be
        # hidden without changing the robot's partial map or planner inputs.
        if self.config.show_obstacles:
            self.draw_ground_truth_obstacles(painter)

        self.draw_editor_preview(painter)
        self.draw_editor_move_selection(painter)
        self.draw_editor_camera_frame(painter)

        # Mapped points remain visible because they represent the discovered map.
        # They are drawn above the vision/r layer and below routes/waypoints/robot.
        self.draw_mapped_obstacle_points(painter)

        # Robot Orders layers. These reveal internal commands/decisions.
        if self.config.show_robot_orders:
            if self.robots and "Multiple" in self.config.agent_mode:
                self.draw_multi_planned_routes(painter)
            else:
                self.draw_planned_route(painter)
                self.draw_executed_path(painter)

        self.draw_goal_and_robot(painter)

        # Drawn last so the temporary red preview is clearly visible over
        # every other layer while the user is comparing grid resolutions.
        self.draw_grid_resolution_preview(painter, rect)

        painter.restore()

        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawRect(rect)

    def draw_topography(self, painter: QPainter, rect):
        painter.save()
        painter.setPen(QPen(QColor(96, 110, 130, 24), 1))

        centers = [
            (rect.left() + rect.width() * 0.20, rect.top() + rect.height() * 0.28, 90, 55, 0.3),
            (rect.left() + rect.width() * 0.66, rect.top() + rect.height() * 0.26, 125, 70, 1.9),
            (rect.left() + rect.width() * 0.42, rect.top() + rect.height() * 0.72, 145, 82, 2.7),
            (rect.left() + rect.width() * 0.82, rect.top() + rect.height() * 0.72, 110, 64, 0.9),
        ]

        for cx, cy, rx0, ry0, phase in centers:
            for level in range(1, 6):
                path = QPainterPath()
                rx = rx0 + level * 16
                ry = ry0 + level * 11

                for k in range(90):
                    t = 2 * math.pi * k / 89
                    wobble = 1.0 + 0.04 * math.sin(3 * t + phase)
                    px = cx + rx * wobble * math.cos(t)
                    py = cy + ry * wobble * math.sin(t)
                    if k == 0:
                        path.moveTo(px, py)
                    else:
                        path.lineTo(px, py)

                path.closeSubpath()
                painter.drawPath(path)

        painter.restore()

    def nice_grid_step(self, visible_span: float, pixel_span: float, target_pixels: float = 58.0) -> float:
        """Choose a readable coordinate-grid spacing for the current zoom."""
        approx_lines = max(2.0, float(pixel_span) / max(20.0, float(target_pixels)))
        raw_step = max(1.0e-9, float(visible_span) / approx_lines)
        exponent = math.floor(math.log10(raw_step))
        scale = 10.0 ** exponent
        for multiplier in (1.0, 2.0, 5.0, 10.0):
            step = multiplier * scale
            if step >= raw_step:
                return step
        return 10.0 * scale

    def format_grid_label(self, value: float, step: float) -> str:
        if abs(value) < step * 1.0e-4:
            value = 0.0
        if step >= 1.0:
            return f"{value:.0f}"
        if step >= 0.1:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def draw_grid(self, painter: QPainter, rect):
        """Draw an infinite-style coordinate grid for the current view."""
        left, right, bottom, top = self.active_view_bounds_world()
        span_x = max(0.1, right - left)
        span_y = max(0.1, top - bottom)
        step = self.nice_grid_step(min(span_x, span_y), min(rect.width(), rect.height()))
        minor_step = step / 2.0

        painter.save()

        # Minor grid.
        painter.setPen(QPen(QColor(235, 238, 243), 1))
        start_x = math.floor(left / minor_step) * minor_step
        x = start_x
        while x <= right + minor_step * 0.5:
            sx, _ = self.world_to_screen(x, 0.0)
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))
            x += minor_step

        start_y = math.floor(bottom / minor_step) * minor_step
        y = start_y
        while y <= top + minor_step * 0.5:
            _, sy = self.world_to_screen(0.0, y)
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))
            y += minor_step

        # Major grid and coordinate labels.
        painter.setFont(QFont("Consolas", 7))
        label_color = QColor(116, 126, 142, 185)
        major_pen = QPen(QColor(210, 216, 226), 1.15)
        axis_pen = QPen(GRID_AXIS, 1.8)

        x = math.floor(left / step) * step
        while x <= right + step * 0.5:
            sx, _ = self.world_to_screen(x, 0.0)
            is_axis = left <= 0.0 <= right and abs(x) <= step * 1.0e-4
            painter.setPen(axis_pen if is_axis else major_pen)
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))

            if rect.left() + 4 <= sx <= rect.right() - 4:
                painter.setPen(label_color)
                painter.drawText(QRectF(sx - 22, rect.bottom() - 18, 44, 14), Qt.AlignCenter, self.format_grid_label(x, step))
            x += step

        y = math.floor(bottom / step) * step
        while y <= top + step * 0.5:
            _, sy = self.world_to_screen(0.0, y)
            is_axis = bottom <= 0.0 <= top and abs(y) <= step * 1.0e-4
            painter.setPen(axis_pen if is_axis else major_pen)
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))

            if rect.top() + 6 <= sy <= rect.bottom() - 6:
                painter.setPen(label_color)
                painter.drawText(QRectF(rect.left() + 5, sy - 7, 42, 14), Qt.AlignLeft | Qt.AlignVCenter, self.format_grid_label(y, step))
            y += step

        painter.restore()

    # ------------------------------------------------------------------
    # Grid resolution preview (temporary red overlay).
    #
    # Shown while the user adjusts SimulationConfig.grid_resolution in the
    # config panel, so 0.50 vs 0.25 m/cell can be compared visually before
    # running the simulation. Purely a rendering overlay: it never mutates
    # self.config or any simulation/runtime state, and it does not rebuild
    # any occupancy/planning grid. It auto-hides itself shortly after the
    # last change via a single-shot QTimer, so it is never left on
    # permanently.
    # ------------------------------------------------------------------

    def show_grid_resolution_preview(self, resolution: float, duration_ms: int = 800) -> None:
        """Show the red grid preview at *resolution*, auto-hiding after
        *duration_ms* (default within the requested 700-1000ms range).

        Safe to call repeatedly while the user is still adjusting the
        control -- each call restarts the auto-hide timer, so the preview
        only disappears after the user stops changing the value. If the
        persistent "Show Grid" overlay is enabled, the overlay already
        keeps a grid visible permanently, so the auto-hide timer is not
        armed -- there is nothing for it to hide.
        """
        self._grid_resolution_preview_active = True
        self._grid_resolution_preview_resolution = max(0.01, float(resolution))
        if self.grid_overlay_enabled:
            self._grid_resolution_preview_timer.stop()
        else:
            self._grid_resolution_preview_timer.start(int(duration_ms))
        self.update()

    def hide_grid_resolution_preview(self) -> None:
        """Hide the red grid preview immediately."""
        self._grid_resolution_preview_active = False
        self._grid_resolution_preview_timer.stop()
        self.update()

    def is_grid_resolution_preview_active(self) -> bool:
        return bool(self._grid_resolution_preview_active)

    def grid_resolution_preview_value(self) -> float | None:
        return self._grid_resolution_preview_resolution

    def draw_grid_resolution_preview(self, painter: QPainter, rect) -> None:
        """Draw a lightweight red grid at the previewed resolution.

        Only within the visible world bounds, only while the preview is
        active. Deliberately simpler than draw_grid(): no minor/major
        distinction, no coordinate labels -- this is a quick visual
        comparison aid, not a permanent map layer.
        """
        if not self._grid_resolution_preview_active or not self._grid_resolution_preview_resolution:
            return

        resolution = self._grid_resolution_preview_resolution
        left, right, bottom, top = self.active_view_bounds_world()

        painter.save()
        painter.setClipRect(rect)
        painter.setPen(QPen(QColor(220, 40, 40, 190), 1))

        x = math.floor(left / resolution) * resolution
        while x <= right + resolution * 0.5:
            sx, _ = self.world_to_screen(x, 0.0)
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))
            x += resolution

        y = math.floor(bottom / resolution) * resolution
        while y <= top + resolution * 0.5:
            _, sy = self.world_to_screen(0.0, y)
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))
            y += resolution

        painter.restore()

    # ------------------------------------------------------------------
    # Persistent grid overlay ("Show Grid" toggle).
    #
    # Unlike the temporary preview above, this stays visible until the user
    # turns it off -- including while the simulation is running -- so it
    # uses its own state instead of the preview's auto-hide timer. Purely a
    # rendering overlay: it never mutates self.config, never rebuilds any
    # occupancy/planning grid, and the occupancy snapshot it colors is a
    # read-only copy pushed in from outside (see engine.py's
    # occupancy_grid_snapshot()), never a live reference.
    # ------------------------------------------------------------------

    def set_grid_overlay_enabled(self, enabled: bool) -> None:
        self.grid_overlay_enabled = bool(enabled)
        self.update()

    def is_grid_overlay_enabled(self) -> bool:
        return bool(self.grid_overlay_enabled)

    def set_simulation_running_for_perf(self, running: bool) -> None:
        """Tell the canvas whether the simulation is actively running, for
        perf-diagnostic gating only (see _maybe_emit_perf_gui_warning and
        the grid-overlay-degraded console notice) -- does not affect
        rendering or any simulation state. A low paint_fps while idle
        (before Start, or after Reset) is not a meaningful signal and must
        not produce a console warning."""
        running = bool(running)
        if running and not self._simulation_running_for_perf:
            # Fresh run starting: if the overlay was already degraded before
            # Start was pressed, give it a fresh chance to notify now that
            # the simulation is actually running, instead of staying
            # permanently suppressed because the degrade happened while idle.
            self._grid_overlay_degraded_notice_shown = False
        self._simulation_running_for_perf = running

    def set_grid_overlay_resolution(self, resolution: float) -> None:
        self._grid_overlay_resolution = max(0.01, float(resolution))
        self.update()

    def set_grid_overlay_snapshot(self, snapshot: dict | None) -> None:
        """Store a read-only occupancy snapshot (resolution/bounds/grid) for
        cell coloring. Pass None to fall back to resolution-only grid lines
        (e.g. before the simulation has started, or no belief map yet).

        Each call bumps a version counter (rather than diffing the grid
        array's contents, which would be as expensive as the render work
        it's meant to avoid) -- draw_grid_overlay()'s cache key includes
        this version, so a genuinely new snapshot always invalidates the
        cache, and repeated pushes of "no new data" never do.
        """
        self._grid_overlay_snapshot = snapshot
        self._grid_overlay_snapshot_version += 1
        self._grid_overlay_snapshot_pushed_at = time.perf_counter()
        self.update()

    def is_grid_overlay_degraded(self) -> bool:
        return bool(self._grid_overlay_degraded)

    def grid_overlay_cache_status(self) -> str:
        return self._grid_overlay_last_cache_status

    def grid_overlay_visible_cell_count(self) -> int:
        return int(self._grid_overlay_last_visible_cells)

    def _grid_overlay_cell_bounds(
        self, resolution: float, snapshot: dict | None
    ) -> tuple[int, int, int, int] | None:
        """(col_start, col_end, row_start, row_end) of snapshot cells inside
        the current view, or None if there is no snapshot/nothing visible."""
        if snapshot is None:
            return None

        grid = snapshot.get("grid")
        bounds = snapshot.get("bounds")
        snapshot_resolution = float(snapshot.get("resolution") or resolution)
        if grid is None or bounds is None or snapshot_resolution <= 0.0:
            return None

        x_min, x_max, y_min, y_max = bounds
        left, right, bottom, top = self.active_view_bounds_world()

        col_start = max(0, int(math.floor((left - x_min) / snapshot_resolution)))
        col_end = min(grid.shape[1] - 1, int(math.ceil((right - x_min) / snapshot_resolution)))
        row_start = max(0, int(math.floor((bottom - y_min) / snapshot_resolution)))
        row_end = min(grid.shape[0] - 1, int(math.ceil((top - y_min) / snapshot_resolution)))

        if col_start > col_end or row_start > row_end:
            return None

        return col_start, col_end, row_start, row_end

    def draw_grid_overlay(self, painter: QPainter, rect) -> None:
        """Draw the persistent grid overlay: resolution grid lines, plus
        translucent occupied/free/unknown cell colors when a snapshot is
        available. Deliberately drawn just above the background/base map
        and below obstacles, mapped points, routes, and the robot, so it
        never hides them.

        Rebuilding the overlay means looping over every visible occupancy
        cell (one QPainter.drawRect() call each), which is only affordable
        once, not every frame -- so the result is cached into a QPixmap and
        reused as long as resolution/canvas size/view bounds/snapshot are
        unchanged (see _grid_overlay_cache_key below). If the number of
        visible cells exceeds MAX_GRID_OVERLAY_CELLS (e.g. a fine
        grid_resolution zoomed far out), cell coloring is skipped for that
        rebuild -- grid lines are still drawn -- so this can never freeze
        the UI trying to draw every cell.
        """
        if not self.grid_overlay_enabled:
            self._grid_overlay_last_cache_status = "off"
            self._grid_overlay_last_visible_cells = 0
            self._grid_overlay_degraded = False
            return

        resolution = self._grid_overlay_resolution
        snapshot = self._grid_overlay_snapshot

        cell_bounds = self._grid_overlay_cell_bounds(resolution, snapshot)
        if cell_bounds is not None:
            col_start, col_end, row_start, row_end = cell_bounds
            visible_cells = (col_end - col_start + 1) * (row_end - row_start + 1)
        else:
            visible_cells = 0

        degraded = visible_cells > MAX_GRID_OVERLAY_CELLS
        if degraded and not self._grid_overlay_degraded_notice_shown and self._simulation_running_for_perf:
            # Only surfaced to the console while the simulation is actually
            # running -- during setup/load/reset this would just be console
            # noise about a state the user isn't looking at yet. Still
            # tracked in latest_perf_status's cache_status field either way.
            self.append_console_message(
                f"[PERF] grid overlay degraded due visible_cells={visible_cells}"
            )
            self._grid_overlay_degraded_notice_shown = True
        if not degraded:
            self._grid_overlay_degraded_notice_shown = False
        if degraded:
            cell_bounds = None  # skip per-cell coloring; grid lines only.

        self._grid_overlay_degraded = degraded
        self._grid_overlay_last_visible_cells = visible_cells

        cache_key = (
            round(float(resolution), 3),
            self.width(),
            self.height(),
            tuple(round(float(bound), 2) for bound in self.active_view_bounds_world()),
            self._grid_overlay_snapshot_version if cell_bounds is not None else -1,
        )

        if self._grid_overlay_cache is not None and self._grid_overlay_cache_key == cache_key:
            painter.drawPixmap(0, 0, self._grid_overlay_cache)
            self._grid_overlay_last_cache_status = "hit"
            return

        self._grid_overlay_cache_key = cache_key
        self._grid_overlay_cache = self._rebuild_grid_overlay_cache(
            rect, resolution, snapshot, cell_bounds
        )
        self._grid_overlay_last_cache_status = "degraded" if degraded else "rebuild"
        painter.drawPixmap(0, 0, self._grid_overlay_cache)

    def _rebuild_grid_overlay_cache(
        self,
        rect,
        resolution: float,
        snapshot: dict | None,
        cell_bounds: tuple[int, int, int, int] | None,
    ) -> QPixmap:
        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)

        cache_painter = QPainter(cache)
        cache_painter.save()
        cache_painter.setClipRect(rect)

        if snapshot is not None and cell_bounds is not None:
            self._draw_grid_overlay_cells(cache_painter, snapshot, cell_bounds)

        self._draw_grid_overlay_lines(cache_painter, rect, resolution)

        cache_painter.restore()
        cache_painter.end()
        return cache

    def _draw_grid_overlay_lines(self, painter: QPainter, rect, resolution: float) -> None:
        left, right, bottom, top = self.active_view_bounds_world()

        painter.setPen(QPen(QColor(90, 90, 90, 70), 1))

        x = math.floor(left / resolution) * resolution
        while x <= right + resolution * 0.5:
            sx, _ = self.world_to_screen(x, 0.0)
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))
            x += resolution

        y = math.floor(bottom / resolution) * resolution
        while y <= top + resolution * 0.5:
            _, sy = self.world_to_screen(0.0, y)
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))
            y += resolution

    def _draw_grid_overlay_cells(
        self,
        painter: QPainter,
        snapshot: dict,
        cell_bounds: tuple[int, int, int, int],
    ) -> None:
        """Fill each visible cell with a translucent color based on its
        occupancy state (unknown/free/occupied). All colors are low-alpha
        so obstacles, routes, and the robot underneath/above remain
        readable -- this is a debug aid, not an opaque map layer. Only
        called during a cache rebuild, never every frame.
        """
        grid = snapshot.get("grid")
        resolution = float(snapshot.get("resolution") or 0.0)
        bounds = snapshot.get("bounds")
        if grid is None or resolution <= 0.0 or bounds is None:
            return

        x_min, _x_max, y_min, _y_max = bounds
        col_start, col_end, row_start, row_end = cell_bounds

        unknown_brush = QBrush(QColor(120, 120, 120, 35))
        free_brush = QBrush(QColor(60, 140, 220, 45))
        occupied_brush = QBrush(QColor(220, 40, 40, 80))

        painter.setPen(Qt.NoPen)

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                state = int(grid[row, col])
                if state == 1:
                    painter.setBrush(occupied_brush)
                elif state == 0:
                    painter.setBrush(free_brush)
                else:
                    painter.setBrush(unknown_brush)

                cx0 = x_min + col * resolution
                cy0 = y_min + row * resolution
                sxA, syA = self.world_to_screen(cx0, cy0)
                sxB, syB = self.world_to_screen(cx0 + resolution, cy0 + resolution)
                painter.drawRect(
                    QRectF(
                        min(sxA, sxB),
                        min(syA, syB),
                        abs(sxB - sxA),
                        abs(syB - syA),
                    )
                )

    def current_robot_pose(self) -> tuple[float, float, float, float]:
        if self.robot is not None:
            return (
                float(self.robot.x),
                float(self.robot.y),
                float(self.robot.theta),
                float(self.robot.vision),
            )

        if "Multiple" in self.config.agent_mode:
            robots = normalized_robot_start_configs(self.config)
            if robots:
                index = max(0, min(int(self.config.selected_robot_index), len(robots) - 1))
                robot_cfg = robots[index]
                return (
                    float(robot_cfg.x),
                    float(robot_cfg.y),
                    float(robot_cfg.theta),
                    float(self.config.vision),
                )

        return (
            float(self.config.x),
            float(self.config.y),
            float(self.config.theta),
            float(self.config.vision),
        )

    def current_goal_xy(self) -> tuple[float, float]:
        # The final mission goal should always remain visible, even when the
        # robot is internally tracking an intermediate waypoint.
        return float(self.config.goal_x), float(self.config.goal_y)

    def draw_explored_area_trace(self, painter: QPainter):
        """
        Draw explored coverage from a cached pixmap.

        The cache is updated incrementally when a new sensor footprint is
        recorded, so paintEvent no longer rebuilds a large QPainterPath every
        frame.
        """
        if not self.config.show_explored_area:
            return

        painter.save()

        if self._explored_area_caches_by_robot:
            for robot_index in sorted(self._explored_area_caches_by_robot):
                cache = self._explored_area_caches_by_robot.get(robot_index)
                if cache is not None:
                    painter.drawPixmap(0, 0, cache)
            painter.restore()
            return

        if not self.explored_area_polygons:
            painter.restore()
            return

        self.ensure_explored_area_cache()
        if self._explored_area_cache is not None:
            painter.drawPixmap(0, 0, self._explored_area_cache)
        painter.restore()

    def sensor_polygon_for_pose(
        self,
        cache_key: int,
        x: float,
        y: float,
        theta: float,
        vision: float,
    ) -> list[tuple[float, float]]:
        """Return a cached occlusion-aware sensor polygon for one robot."""
        signature = (
            round(float(vision), 3),
            str(self.config.vision_model),
            self.obstacles_cache_signature(),
        )
        pose = (float(x), float(y), float(theta))
        cached = self._sensor_polygon_caches_by_robot.get(int(cache_key))
        if cached is not None:
            cached_pose, cached_signature, cached_polygon = cached
            moved = math.hypot(pose[0] - cached_pose[0], pose[1] - cached_pose[1])
            rotated = abs(wrapped_angle_error(pose[2], cached_pose[2]))
            if (
                cached_signature == signature
                and moved < SENSOR_DRAW_RECOMPUTE_DISTANCE
                and rotated < SENSOR_DRAW_RECOMPUTE_ROTATION
            ):
                return cached_polygon

        polygon = sensor_visible_polygon_world(
            origin=(pose[0], pose[1]),
            theta=pose[2],
            vision=float(vision),
            vision_model=self.config.vision_model,
            obstacles=self.config.obstacles,
            ray_count=SENSOR_DRAW_RAYS_CAMERA if "Camera" in self.config.vision_model else SENSOR_DRAW_RAYS_OMNI,
        )
        self._sensor_polygon_caches_by_robot[int(cache_key)] = (pose, signature, polygon)
        return polygon

    def draw_sensor_polygon(
        self,
        painter: QPainter,
        polygon: list[tuple[float, float]],
        color: QColor,
        alpha_fill: int = 16,
        alpha_stroke: int = 58,
    ) -> None:
        if len(polygon) < 3:
            return

        fill = QColor(color)
        fill.setAlpha(alpha_fill)
        stroke = QColor(color)
        stroke.setAlpha(alpha_stroke)

        visible_path = QPainterPath()
        sx, sy = self.world_to_screen(*polygon[0])
        visible_path.moveTo(sx, sy)
        for point in polygon[1:]:
            px, py = self.world_to_screen(*point)
            visible_path.lineTo(px, py)
        visible_path.closeSubpath()

        painter.setPen(QPen(stroke, 1.4))
        painter.setBrush(QBrush(fill))
        painter.drawPath(visible_path)

    def sensor_display_poses(self) -> list[tuple[int, float, float, float, float]]:
        """Return all sensor poses that should be visible on the canvas."""
        if "Multiple" in self.config.agent_mode:
            if self.robots:
                return [
                    (index, float(robot.x), float(robot.y), float(robot.theta), float(robot.vision))
                    for index, robot in enumerate(self.robots)
                ]
            if self.robot is None:
                return [
                    (index, float(cfg.x), float(cfg.y), float(cfg.theta), float(self.config.vision))
                    for index, cfg in enumerate(normalized_robot_start_configs(self.config))
                ]

        x, y, theta, vision = self.current_robot_pose()
        return [(-1, x, y, theta, vision)]

    def body_radius_for_display_key(self, cache_key: int) -> float:
        if int(cache_key) >= 0:
            if self.robots and int(cache_key) < len(self.robots):
                return float(getattr(self.robots[int(cache_key)], "_sim_body_radius", self.config.body_radius))
            configs = normalized_robot_start_configs(self.config)
            if int(cache_key) < len(configs):
                return float(configs[int(cache_key)].body_radius)
        return float(self.config.body_radius)

    def safety_radius_for_display_key(self, cache_key: int) -> float:
        if int(cache_key) >= 0:
            if self.robots and int(cache_key) < len(self.robots):
                body = float(getattr(self.robots[int(cache_key)], "_sim_body_radius", self.config.body_radius))
                return max(float(getattr(self.robots[int(cache_key)], "_sim_safety_radius", self.config.safety_radius)), body)
            configs = normalized_robot_start_configs(self.config)
            if int(cache_key) < len(configs):
                return max(float(configs[int(cache_key)].safety_radius), float(configs[int(cache_key)].body_radius))
        return max(float(self.config.safety_radius), float(self.config.body_radius))

    def draw_sensor_range(self, painter: QPainter):
        """
        Draw the actually visible sensor regions.

        In multi-robot mode the LiDAR/FoV of every robot is always drawn with
        the robot's own color. This layer is world/sensing information, not a
        robot-order/debug layer, so it does not depend on Robot Orders.
        """
        if not self.config.show_vision:
            return

        painter.save()
        for cache_key, x, y, theta, vision in self.sensor_display_poses():
            color = QColor(BLUE) if cache_key < 0 else robot_color(cache_key)
            polygon = self.sensor_polygon_for_pose(cache_key, x, y, theta, vision)
            self.draw_sensor_polygon(painter, polygon, color)
        painter.restore()

    def obstacle_boundary_samples_for_display(
        self,
        obstacle: tuple[float, float, float, float],
    ) -> list[tuple[float, float]]:
        """
        Sample an obstacle boundary for display-only discovery coverage.

        This mirrors the mapping abstraction without changing planner behavior:
        a rectangle is treated as fully discovered only when the robot has
        observed most of its boundary samples from visible viewpoints.
        """
        ox, oy, ow, oh = obstacle
        spacing = max(float(self.config.mapping_point_spacing), 0.015)
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

    def obstacle_boundary_sample_count(
        self,
        obstacle: tuple[float, float, float, float],
    ) -> int:
        """
        Return how many boundary samples would represent this obstacle.

        This is used as the denominator for completion opacity. It avoids
        building and comparing every sample against every mapped point during
        paintEvent.
        """
        ox, oy, ow, oh = obstacle
        spacing = max(float(self.config.mapping_point_spacing), 0.015)
        nx = max(1, int(math.ceil(ow / spacing)))
        ny = max(1, int(math.ceil(oh / spacing)))
        return max(1, 2 * (nx + 1) + 2 * max(0, ny - 1))

    def mapped_point_lies_on_obstacle_boundary(
        self,
        point: tuple[float, float],
        obstacle: tuple[float, float, float, float],
    ) -> bool:
        """
        Fast boundary-membership test for visual completion opacity.

        The mapped points are generated from obstacle boundaries, so we do not
        need the previous O(boundary_samples * mapped_points) nearest-neighbor
        coverage check. Testing each mapped point against each rectangle edge is
        much cheaper and removes the FPS drop caused by Show Obstacles.
        """
        px, py = point
        ox, oy, ow, oh = obstacle
        tol = max(0.025, float(self.config.mapping_point_spacing) * 0.75)

        inside_x_span = (ox - tol) <= px <= (ox + ow + tol)
        inside_y_span = (oy - tol) <= py <= (oy + oh + tol)

        on_bottom_or_top = inside_x_span and (
            abs(py - oy) <= tol or abs(py - (oy + oh)) <= tol
        )
        on_left_or_right = inside_y_span and (
            abs(px - ox) <= tol or abs(px - (ox + ow)) <= tol
        )

        return bool(on_bottom_or_top or on_left_or_right)

    def obstacle_mapping_coverage(
        self,
        obstacle: tuple[float, float, float, float],
    ) -> float:
        """
        Estimate obstacle-boundary coverage in O(mapped_points), not O(samples
        * mapped_points).

        This is intentionally visual only. Planning still uses the mapped point
        cloud; this value only controls opacity of the gray ground-truth layer.
        """
        if not self.mapped_obstacle_points:
            return 0.0

        sample_count = self.obstacle_boundary_sample_count(obstacle)
        covered = 0

        for point in self.mapped_obstacle_points:
            if self.mapped_point_lies_on_obstacle_boundary(point, obstacle):
                covered += 1

        return min(1.0, covered / sample_count)

    def ensure_obstacle_coverage_cache(self):
        if self._obstacle_coverage_cache_count == len(self.mapped_obstacle_points):
            return

        self._obstacle_coverage_cache = {}
        for index, obstacle in enumerate(self.config.obstacles):
            self._obstacle_coverage_cache[index] = self.obstacle_mapping_coverage(tuple(obstacle))
        self._obstacle_coverage_cache_count = len(self.mapped_obstacle_points)

    def obstacles_cache_signature(self) -> tuple:
        return tuple(
            (
                round(float(ox), 4),
                round(float(oy), 4),
                round(float(ow), 4),
                round(float(oh), 4),
            )
            for ox, oy, ow, oh in self.config.obstacles
        )

    def ensure_obstacles_cache(self):
        signature = self.obstacles_cache_signature()
        mapped_count = len(self.mapped_obstacle_points)

        base_cache_is_valid = (
            self._obstacles_cache is not None
            and self._obstacles_cache_size == self.size()
            and self._obstacles_cache_signature == signature
        )

        if base_cache_is_valid:
            mapped_delta = mapped_count - self._obstacles_cache_mapped_count
            if 0 <= mapped_delta < OBSTACLE_VISUAL_REFRESH_POINT_STEP:
                return

        self.rebuild_obstacles_cache(signature)

    def obstacle_is_squareish_stamp(self, obstacle: tuple[float, float, float, float]) -> bool:
        """Return whether an obstacle looks like one free-draw brush stamp.

        Free-draw strokes are stored as small square bounding boxes because the
        runtime planner/collision code still consumes rectangles. Rendering is
        allowed to interpret connected dense stamps as circles so the user sees
        one smooth object instead of a chain of tiny squares.
        """
        _, _, width, height = obstacle
        width = abs(float(width))
        height = abs(float(height))
        if width <= 0.0 or height <= 0.0:
            return False

        squareish = abs(width - height) <= max(0.025, 0.12 * max(width, height))
        # Do not depend on the current brush slider value. The user may draw a
        # stroke, change brush size, then run the simulation. A visual stamp
        # should still render as a stamp. Keep the cap high enough for normal
        # editor brush sizes, but low enough that large square obstacles remain
        # rectangles.
        plausible_stamp_size = max(width, height) <= 2.25
        return bool(squareish and plausible_stamp_size)

    def obstacle_group_looks_like_free_draw(self, indices: list[int]) -> bool:
        """Heuristic for deciding when a connected object is a free-draw stroke."""
        if len(indices) < 3:
            return False

        obstacles = [tuple(self.config.obstacles[index]) for index in indices if 0 <= index < len(self.config.obstacles)]
        if len(obstacles) < 3:
            return False

        squareish_count = sum(1 for obstacle in obstacles if self.obstacle_is_squareish_stamp(obstacle))
        if squareish_count / len(obstacles) < 0.70:
            return False

        sizes = [max(abs(float(obstacle[2])), abs(float(obstacle[3]))) for obstacle in obstacles]
        min_size = max(min(sizes), 1.0e-9)
        max_size = max(sizes)
        similar_sizes = (max_size / min_size) <= 2.25
        return bool(similar_sizes)

    def obstacle_screen_path(
        self,
        obstacle: tuple[float, float, float, float],
        *,
        as_brush_stamp: bool = False,
    ) -> QPainterPath:
        """Return the visual path for one obstacle in screen coordinates."""
        ox, oy, ow, oh = obstacle
        x1, y1 = self.world_to_screen(ox, oy)
        x2, y2 = self.world_to_screen(ox + ow, oy + oh)
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        path = QPainterPath()
        if as_brush_stamp:
            path.addEllipse(rect)
        else:
            path.addRect(rect)
        return path

    def obstacle_visual_groups(self) -> list[list[int]]:
        """Return connected obstacle groups for seam-free rendering.

        Data remains as individual rectangles. This method affects only the
        display layer, so joined objects and free-draw strokes look like one
        object in both editor mode and simulation mode.
        """
        groups: list[list[int]] = []
        visited: set[int] = set()

        for index in range(len(self.config.obstacles)):
            if index in visited:
                continue
            group = connected_obstacle_indices(list(self.config.obstacles), index)
            if not group:
                group = [index]
            for group_index in group:
                visited.add(group_index)
            groups.append(group)

        return groups

    def obstacle_group_screen_path(self, indices: list[int]) -> QPainterPath:
        """Build a unified visual path for one connected obstacle object."""
        union_path = QPainterPath()
        draw_as_free_stroke = self.obstacle_group_looks_like_free_draw(indices)

        for index in indices:
            if index < 0 or index >= len(self.config.obstacles):
                continue

            obstacle_path = self.obstacle_screen_path(
                tuple(self.config.obstacles[index]),
                as_brush_stamp=draw_as_free_stroke,
            )
            if union_path.isEmpty():
                union_path = obstacle_path
            else:
                union_path = union_path.united(obstacle_path)

        return union_path.simplified()

    def obstacle_group_mapping_coverage(self, indices: list[int]) -> float:
        """Return display-only completion coverage for a connected object."""
        valid_indices = [index for index in indices if 0 <= index < len(self.config.obstacles)]
        if not valid_indices:
            return 0.0
        return float(
            sum(self._obstacle_coverage_cache.get(index, 0.0) for index in valid_indices)
            / len(valid_indices)
        )

    def draw_obstacle_group(
        self,
        painter: QPainter,
        indices: list[int],
    ) -> None:
        """Draw one connected obstacle object without internal seams."""
        path = self.obstacle_group_screen_path(indices)
        if path.isEmpty():
            return

        if self.editor_mode:
            fill = QColor(178, 181, 188, 105)
            stroke = QColor(82, 84, 92, 165)
            pen_width = 1.35
        else:
            coverage = self.obstacle_group_mapping_coverage(indices)
            fully_discovered = coverage >= OBSTACLE_COMPLETE_COVERAGE

            if fully_discovered:
                fill = QColor(190, 194, 202, 170)
                stroke = QColor(82, 84, 92, 210)
                pen_width = 1.7
            else:
                fill = QColor(178, 181, 188, 85)
                stroke = QColor(82, 84, 92, 105)
                pen_width = 1.2

        painter.setPen(QPen(stroke, pen_width))
        painter.setBrush(QBrush(fill))
        painter.drawPath(path)

    def rebuild_obstacles_cache(self, signature: tuple | None = None):
        cache = QPixmap(self.size())
        cache.fill(Qt.transparent)
        self._obstacles_cache = cache
        self._obstacles_cache_size = QSize(self.size())
        self._obstacles_cache_mapped_count = len(self.mapped_obstacle_points)
        self._obstacles_cache_signature = signature if signature is not None else self.obstacles_cache_signature()

        if not self.config.obstacles:
            return

        self.ensure_obstacle_coverage_cache()

        cache_painter = QPainter(self._obstacles_cache)
        cache_painter.setRenderHint(QPainter.Antialiasing)
        cache_painter.setClipRect(self.plot_rect())

        for group in self.obstacle_visual_groups():
            self.draw_obstacle_group(cache_painter, group)

        cache_painter.end()

    def draw_ground_truth_obstacles(self, painter: QPainter):
        """
        Draw scenario obstacles from a cached pixmap.

        This keeps the human-facing gray obstacles visible without recomputing
        completion opacity or redrawing rectangles every frame.
        """
        if not self.config.obstacles:
            return

        self.ensure_obstacles_cache()
        if self._obstacles_cache is None:
            return

        painter.save()
        painter.drawPixmap(0, 0, self._obstacles_cache)
        painter.restore()

    def draw_editor_preview(self, painter: QPainter):
        if not self.editor_mode or self.editor_drag_start is None or self.editor_drag_current is None:
            return

        if self.editor_tool == "free":
            if len(self.editor_preview_points) >= 2:
                painter.save()
                stroke_width = max(1.6, self.editor_brush_size * self.pixels_per_meter() * 0.8)
                painter.setPen(QPen(QColor(BLUE), stroke_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setOpacity(0.75)
                path = QPainterPath()
                first = self.editor_preview_points[0]
                x0, y0 = self.world_to_screen(first[0], first[1])
                path.moveTo(x0, y0)
                for point in self.editor_preview_points[1:]:
                    x, y = self.world_to_screen(point[0], point[1])
                    path.lineTo(x, y)
                painter.drawPath(path)

                # Live circular brush cursor at the last stamp position.
                last = self.editor_preview_points[-1]
                cx, cy = self.world_to_screen(last[0], last[1])
                radius = max(2.0, self.editor_brush_size * self.pixels_per_meter() / 2.0)
                painter.setPen(QPen(QColor(BLUE_DARK), 1.4))
                painter.setBrush(QBrush(QColor(255, 255, 255, 95)))
                painter.drawEllipse(QRectF(cx - radius, cy - radius, 2.0 * radius, 2.0 * radius))
                painter.restore()
            return

        if self.editor_tool not in {"rectangles", "squares"}:
            return

        start_x, start_y = self.editor_drag_start
        current_x, current_y = self.editor_drag_current
        left = min(start_x, current_x)
        bottom = min(start_y, current_y)
        width = abs(current_x - start_x)
        height = abs(current_y - start_y)

        if width < MIN_EDITOR_OBSTACLE_SIZE and height < MIN_EDITOR_OBSTACLE_SIZE:
            return

        if self.editor_tool == "squares":
            size = max(width, height)
            width = size
            height = size

        x1, y1 = self.world_to_screen(left, bottom)
        x2, y2 = self.world_to_screen(left + width, bottom + height)
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        painter.save()
        painter.setPen(QPen(QColor(BLUE), 2, Qt.DashLine))
        painter.setBrush(QBrush(QColor(BLUE_LIGHT)))
        painter.setOpacity(0.35)
        painter.drawRect(rect)
        painter.restore()


    def draw_editor_move_selection(self, painter: QPainter):
        """Highlight the connected object currently being moved in editor mode."""
        if not self.editor_mode or not self.editor_obstacle_drag_indices:
            return

        selection_path = QPainterPath()
        for index in self.editor_obstacle_drag_indices:
            if index < 0 or index >= len(self.config.obstacles):
                continue
            path = self.obstacle_screen_path(tuple(self.config.obstacles[index]))
            selection_path = path if selection_path.isEmpty() else selection_path.united(path)

        if selection_path.isEmpty():
            return

        painter.save()
        painter.setPen(QPen(QColor(220, 52, 52), 2.0, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(selection_path.simplified())
        painter.restore()

    def draw_editor_camera_frame(self, painter: QPainter):
        """Draw the adjustable red simulation camera frame in editor mode."""
        if not self.editor_mode:
            return

        rect = self.camera_rect_screen()
        plot = QRectF(self.plot_rect())
        if rect.isNull() or rect.width() <= 0.0 or rect.height() <= 0.0:
            return

        painter.save()
        painter.setClipRect(plot)

        # Soft outside overlay so the user understands this frame is the future
        # simulation viewport, not an obstacle.
        outside = QPainterPath()
        outside.addRect(plot)
        inside = QPainterPath()
        inside.addRect(rect)
        outside = outside.subtracted(inside)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(220, 52, 52, 20)))
        painter.drawPath(outside)

        painter.setPen(QPen(QColor(220, 52, 52), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        painter.setPen(QPen(QColor(255, 255, 255), 1.4))
        painter.setBrush(QBrush(QColor(220, 52, 52)))
        handle_size = 7.0
        for point in (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()):
            painter.drawRect(QRectF(point.x() - handle_size / 2.0, point.y() - handle_size / 2.0, handle_size, handle_size))

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        label = "Simulation camera viewport"
        label_rect = QRectF(rect.left() + 8, rect.top() + 8, 172, 20)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 225)))
        painter.drawRoundedRect(label_rect, 6, 6)
        painter.setPen(QColor(160, 20, 20))
        painter.drawText(label_rect.adjusted(8, 0, -8, 0), Qt.AlignVCenter | Qt.AlignLeft, label)

        painter.restore()

    def draw_safety_radius(self, painter: QPainter):
        """
        Draw safety radius r below mapped obstacles and waypoints.

        In multi-robot mode every robot gets its own colored safety radius when
        Robot Orders is enabled.
        """
        px_per_meter = self.pixels_per_meter()

        painter.save()
        for cache_key, x, y, _, _ in self.sensor_display_poses():
            rx, ry = self.world_to_screen(x, y)
            radius = self.safety_radius_for_display_key(cache_key) * px_per_meter
            color = QColor(122, 30, 36) if cache_key < 0 else robot_color(cache_key)
            stroke = QColor(color)
            stroke.setAlpha(105)
            fill = QColor(color)
            fill.setAlpha(18)
            painter.setPen(QPen(stroke, 1.8, Qt.DashLine))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(QRectF(rx - radius, ry - radius, radius * 2, radius * 2))

        painter.restore()

    def draw_mapped_obstacle_points(self, painter: QPainter):
        """
        Draw discovered obstacle samples from a cached pixmap.

        This avoids redrawing thousands of tiny ellipses every frame. The cache
        is updated only when new mapped points are added or when the canvas is
        resized.
        """
        if not self.mapped_obstacle_points:
            return

        self.ensure_mapped_points_cache()
        if self._mapped_points_cache is None:
            return

        painter.save()
        painter.drawPixmap(0, 0, self._mapped_points_cache)
        painter.restore()

    def active_planned_waypoint_index(self) -> int:
        if self.robot is None or len(self.planned_path_points) < 2:
            return -1

        waypoint_manager = getattr(self.robot, "waypoints", None)
        current_index = getattr(waypoint_manager, "current_index", None)
        if isinstance(current_index, int):
            planned_index = current_index + 1
            if 0 <= planned_index < len(self.planned_path_points):
                return planned_index

        return -1

    def draw_planned_route(self, painter: QPainter):
        if len(self.planned_path_points) < 2:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setPen(QPen(QColor(ORANGE), 2.4, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
        for i in range(len(self.planned_path_points) - 1):
            x1, y1 = self.world_to_screen(*self.planned_path_points[i])
            x2, y2 = self.world_to_screen(*self.planned_path_points[i + 1])
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        active_index = self.active_planned_waypoint_index()
        last_index = len(self.planned_path_points) - 1
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))

        for i, point in enumerate(self.planned_path_points):
            sx, sy = self.world_to_screen(*point)

            if i == 0:
                label = "S"
                radius = START_MARKER_RADIUS
                fill = QColor(BLUE_DARK)
                stroke = QColor("white")
                text_color = QColor("white")
            elif i == last_index:
                goal_xy = self.current_goal_xy()
                is_final_goal = math.hypot(point[0] - goal_xy[0], point[1] - goal_xy[1]) <= max(0.20, self.config.goal_tolerance)
                label = "G" if is_final_goal else "F"
                radius = FRONTIER_OR_ENDPOINT_MARKER_RADIUS
                fill = QColor(GREEN) if is_final_goal else QColor(146, 62, 160)
                stroke = QColor("white")
                text_color = QColor("white")
            else:
                label = str(i)
                radius = WAYPOINT_MARKER_RADIUS
                fill = QColor("white")
                stroke = QColor(ORANGE)
                text_color = QColor(MAROON)

            if i == active_index:
                halo_radius = ACTIVE_WAYPOINT_MARKER_RADIUS + ACTIVE_WAYPOINT_HALO_PADDING
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(225, 126, 38, 55)))
                painter.drawEllipse(QRectF(sx - halo_radius, sy - halo_radius, 2 * halo_radius, 2 * halo_radius))
                radius = ACTIVE_WAYPOINT_MARKER_RADIUS
                fill = QColor(ORANGE)
                stroke = QColor("white")
                text_color = QColor("white")

            painter.setPen(QPen(stroke, 2.0))
            painter.setBrush(QBrush(fill))
            painter.drawEllipse(QRectF(sx - radius, sy - radius, 2 * radius, 2 * radius))
            painter.setPen(QPen(text_color))
            painter.drawText(QRectF(sx - radius, sy - radius, 2 * radius, 2 * radius), Qt.AlignCenter, label)

        painter.restore()

    def draw_multi_planned_routes(self, painter: QPainter):
        """Draw planned routes/waypoints for every runtime robot."""
        if not self.multi_planned_path_points:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))

        for robot_index, route in enumerate(self.multi_planned_path_points):
            if len(route) < 2:
                continue

            color = robot_color(robot_index)
            route_color = QColor(color)
            route_color.setAlpha(210)
            painter.setPen(QPen(route_color, 2.0, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))

            for i in range(len(route) - 1):
                x1, y1 = self.world_to_screen(*route[i])
                x2, y2 = self.world_to_screen(*route[i + 1])
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            last_index = len(route) - 1
            for i, point in enumerate(route):
                sx, sy = self.world_to_screen(*point)
                if i == 0:
                    label = f"S{robot_index + 1}"
                    radius = START_MARKER_RADIUS
                    fill = QColor(color)
                    stroke = QColor("white")
                    text_color = QColor("white")
                elif i == last_index:
                    goal_xy = self.current_goal_xy()
                    is_final_goal = math.hypot(point[0] - goal_xy[0], point[1] - goal_xy[1]) <= max(0.20, self.config.goal_tolerance)
                    label = "G" if is_final_goal else "F"
                    radius = FRONTIER_OR_ENDPOINT_MARKER_RADIUS
                    fill = QColor(GREEN) if is_final_goal else QColor(146, 62, 160)
                    stroke = QColor("white")
                    text_color = QColor("white")
                else:
                    label = str(i)
                    radius = MULTI_ROBOT_WAYPOINT_MARKER_RADIUS
                    fill = QColor("white")
                    stroke = QColor(color)
                    text_color = QColor(MAROON)

                painter.setPen(QPen(stroke, 1.8))
                painter.setBrush(QBrush(fill))
                painter.drawEllipse(QRectF(sx - radius, sy - radius, 2 * radius, 2 * radius))
                painter.setPen(QPen(text_color))
                painter.drawText(QRectF(sx - radius, sy - radius, 2 * radius, 2 * radius), Qt.AlignCenter, label)

        painter.restore()

    def draw_executed_path(self, painter: QPainter):
        if len(self.path_points) < 2:
            return

        painter.save()
        painter.setPen(QPen(QColor(BLUE), 1.7))
        for i in range(len(self.path_points) - 1):
            x1, y1 = self.world_to_screen(*self.path_points[i])
            x2, y2 = self.world_to_screen(*self.path_points[i + 1])
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        painter.restore()

    def draw_goal_and_robot(self, painter: QPainter):
        x, y, theta, _ = self.current_robot_pose()
        gx, gy = self.current_goal_xy()

        rx, ry = self.world_to_screen(x, y)
        gx_s, gy_s = self.world_to_screen(gx, gy)

        # Goal marker: always visible.
        painter.save()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(GREEN_LIGHT)))
        painter.drawEllipse(QRectF(gx_s - 15, gy_s - 15, 30, 30))
        painter.setBrush(QBrush(QColor(GREEN)))
        painter.drawEllipse(QRectF(gx_s - 8, gy_s - 8, 16, 16))
        painter.setBrush(QBrush(QColor("white")))
        painter.drawEllipse(QRectF(gx_s - 3, gy_s - 3, 6, 6))
        painter.restore()

        # Exploration target marker(s): visible only with Robot Orders because
        # frontiers are internal targets selected by the exploration planner.
        # In multi-robot mode each robot owns its own F marker; do not draw a
        # single shared F because that makes the robots look coupled.
        if self.config.show_robot_orders:
            if self.robots and "Multiple" in self.config.agent_mode:
                painter.save()
                painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
                for target_index, target in enumerate(self.multi_exploration_targets):
                    if target is None:
                        continue
                    tx, ty = float(target[0]), float(target[1])
                    if math.hypot(tx - gx, ty - gy) <= max(0.20, self.config.goal_tolerance):
                        continue
                    tx_s, ty_s = self.world_to_screen(tx, ty)
                    color = robot_color(target_index)
                    painter.setPen(QPen(QColor("white"), 2.0))
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(QRectF(tx_s - 11, ty_s - 11, 22, 22))
                    painter.setPen(QPen(QColor("white")))
                    painter.drawText(QRectF(tx_s - 11, ty_s - 11, 22, 22), Qt.AlignCenter, f"F{target_index + 1}")
                painter.restore()
            elif self.exploration_target_xy is not None:
                tx, ty = self.exploration_target_xy
                if math.hypot(tx - gx, ty - gy) > max(0.20, self.config.goal_tolerance):
                    tx_s, ty_s = self.world_to_screen(tx, ty)
                    painter.save()
                    painter.setPen(QPen(QColor("white"), 2.0))
                    painter.setBrush(QBrush(QColor(146, 62, 160)))
                    painter.drawEllipse(QRectF(tx_s - 10, ty_s - 10, 20, 20))
                    painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                    painter.setPen(QPen(QColor("white")))
                    painter.drawText(QRectF(tx_s - 10, ty_s - 10, 20, 20), Qt.AlignCenter, "F")
                    painter.restore()

        # Multi-robot preview: before the simulation starts, show every robot
        # start pose and allow click-drag placement. The runtime multi-robot
        # controller is a separate implementation step; this keeps configuration
        # stable first.
        if self.robot is None and "Multiple" in self.config.agent_mode:
            painter.save()
            px_per_meter = self.pixels_per_meter()
            selected_index = max(0, min(int(self.config.selected_robot_index), int(self.config.robot_count) - 1))

            for index, robot_cfg in enumerate(normalized_robot_start_configs(self.config)):
                sx, sy = self.world_to_screen(robot_cfg.x, robot_cfg.y)
                body_px = max(7.0, float(robot_cfg.body_radius) * px_per_meter)
                is_selected = index == selected_index

                if is_selected:
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(QColor(225, 126, 38, 45)))
                    painter.drawEllipse(QRectF(sx - body_px - 8, sy - body_px - 8, 2 * (body_px + 8), 2 * (body_px + 8)))

                fill = robot_color(index)
                painter.setPen(QPen(QColor("white"), 2.4 if is_selected else 1.8))
                painter.setBrush(QBrush(fill))
                painter.drawEllipse(QRectF(sx - body_px, sy - body_px, 2 * body_px, 2 * body_px))

                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.setPen(QPen(QColor("white")))
                painter.drawText(QRectF(sx - body_px, sy - body_px, 2 * body_px, 2 * body_px), Qt.AlignCenter, str(index + 1))

                if self.config.show_robot_orders:
                    arrow_len = 25
                    hx = sx + arrow_len * math.cos(robot_cfg.theta)
                    hy = sy - arrow_len * math.sin(robot_cfg.theta)
                    painter.setPen(QPen(QColor(RED), 2.4, Qt.SolidLine, Qt.RoundCap))
                    painter.drawLine(QPointF(sx, sy), QPointF(hx, hy))

            painter.restore()
            return

        # Runtime multi-robot drawing. This is the first executable multi-robot
        # baseline: every robot is visible and moves as an independent agent.
        if self.robots and "Multiple" in self.config.agent_mode:
            painter.save()
            px_per_meter = self.pixels_per_meter()

            if self.config.show_robot_orders:
                for index, path_points in enumerate(self.multi_path_points):
                    if len(path_points) < 2:
                        continue
                    color = robot_color(index)
                    color.setAlpha(175)
                    painter.setPen(QPen(color, 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    for i in range(len(path_points) - 1):
                        x1, y1 = self.world_to_screen(*path_points[i])
                        x2, y2 = self.world_to_screen(*path_points[i + 1])
                        painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            for index, robot in enumerate(self.robots):
                sx, sy = self.world_to_screen(float(robot.x), float(robot.y))
                color = robot_color(index)
                body_px = max(5.0, float(getattr(robot, "_sim_body_radius", self.config.body_radius)) * px_per_meter)

                painter.setPen(QPen(QColor("white"), 2.2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QRectF(sx - body_px, sy - body_px, 2 * body_px, 2 * body_px))

                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.setPen(QPen(QColor("white")))
                painter.drawText(QRectF(sx - body_px, sy - body_px, 2 * body_px, 2 * body_px), Qt.AlignCenter, str(index + 1))

                if self.config.show_robot_orders:
                    arrow_len = 28
                    hx = sx + arrow_len * math.cos(float(robot.theta))
                    hy = sy - arrow_len * math.sin(float(robot.theta))
                    painter.setPen(QPen(color, 3, Qt.SolidLine, Qt.RoundCap))
                    painter.drawLine(QPointF(sx, sy), QPointF(hx, hy))

            painter.restore()
            return

        # Robot marker: always visible. Its size follows body_radius. The safety
        # radius r is a separate layer shown only when Robot Orders is ON.
        painter.save()
        px_per_meter = self.pixels_per_meter()
        body_px = max(5.0, float(self.config.body_radius) * px_per_meter)
        painter.setPen(QPen(QColor("white"), 2.0))
        painter.setBrush(QBrush(QColor(BLUE)))
        painter.drawEllipse(QRectF(rx - body_px, ry - body_px, 2 * body_px, 2 * body_px))

        if self.config.show_robot_orders:
            arrow_len = 34
            hx = rx + arrow_len * math.cos(theta)
            hy = ry - arrow_len * math.sin(theta)
            painter.setPen(QPen(QColor(RED), 3, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(rx, ry), QPointF(hx, hy))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(RED)))
            painter.drawEllipse(QRectF(hx - 4, hy - 4, 8, 8))

        painter.restore()

    def draw_telemetry(self, painter: QPainter):
        rect = QRectF(self.telemetry_rect())
        path = QPainterPath()
        path.addRoundedRect(rect, 7, 7)
        painter.fillPath(path, QColor("#FAFBFD"))
        painter.setPen(QPen(QColor(BORDER), 1))
        painter.drawPath(path)

        painter.setFont(QFont("Consolas", 9))
        painter.setPen(QColor(TEXT))

        if self.robot is None:
            text = (
                "state: CONFIG     x: --     y: --     theta: --     "
                "v: --     a: --     omega: --     distance: --"
            )
        else:
            gx, gy = self.current_goal_xy()
            distance = np.linalg.norm(self.robot.displacement(gx, gy))
            text = (
                f"state: {mode_name(self.robot)}     "
                f"x: {self.robot.x: .2f}     "
                f"y: {self.robot.y: .2f}     "
                f"theta: {self.robot.theta: .3f}     "
                f"v: {self.robot.v: .3f}     "
                f"a: {self.last_control[0, 0]: .3f}     "
                f"omega: {self.last_control[1, 0]: .3f}     "
                f"distance: {distance: .3f}     "
                f"mapped pts: {len(self.mapped_obstacle_points)}"
            )

        painter.drawText(rect.adjusted(12, 0, -12, 0), Qt.AlignVCenter | Qt.AlignLeft, text)


