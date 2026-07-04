"""
Simulation canvas and rendering logic.

This module draws the current simulator snapshot: grid, obstacles, mapped
points, explored area, robots, FoV/LiDAR, routes, frontiers, and telemetry.
It emits interaction events, but it does not choose frontiers or compute routes.
"""

from __future__ import annotations

import math
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

class SimulationCanvas(QWidget):
    goalClicked = Signal(float, float)
    robotDragged = Signal(int, float, float)
    robotSelected = Signal(int)

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

        # Dragging support for pre-simulation multi-robot placement.
        self.dragging_robot_index: int | None = None
        self.dragging_robot_offset: tuple[float, float] = (0.0, 0.0)

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

    def world_to_screen(self, x: float, y: float):
        rect = self.plot_rect()
        sx = rect.left() + (x - WORLD_X_MIN) / (WORLD_X_MAX - WORLD_X_MIN) * rect.width()
        sy = rect.bottom() - (y - WORLD_Y_MIN) / (WORLD_Y_MAX - WORLD_Y_MIN) * rect.height()
        return sx, sy

    def screen_to_world(self, sx: float, sy: float):
        rect = self.plot_rect()
        x = WORLD_X_MIN + (sx - rect.left()) / rect.width() * (WORLD_X_MAX - WORLD_X_MIN)
        y = WORLD_Y_MIN + (rect.bottom() - sy) / rect.height() * (WORLD_Y_MAX - WORLD_Y_MIN)
        return x, y

    def multi_robot_screen_positions(self) -> list[tuple[int, float, float, RobotStartConfig]]:
        if "Multiple" not in self.config.agent_mode:
            return []

        robots = normalized_robot_start_configs(self.config)
        positions: list[tuple[int, float, float, RobotStartConfig]] = []
        for index, robot_cfg in enumerate(robots):
            sx, sy = self.world_to_screen(robot_cfg.x, robot_cfg.y)
            positions.append((index, sx, sy, robot_cfg))
        return positions

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

        px_per_meter = self.plot_rect().width() / (WORLD_X_MAX - WORLD_X_MIN)
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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.position()

            if self.metrics_eye_rect().contains(QPointF(pos.x(), pos.y())):
                self.metrics_visible = not self.metrics_visible
                self.update()
                return

            if self.plot_rect().contains(pos.toPoint()):
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
        if self.dragging_robot_index is None:
            return

        pos = event.position()
        x, y = self.screen_to_world(pos.x(), pos.y())
        dx, dy = self.dragging_robot_offset
        x = clamp(x + dx, WORLD_X_MIN, WORLD_X_MAX)
        y = clamp(y + dy, WORLD_Y_MIN, WORLD_Y_MAX)
        self.robotDragged.emit(int(self.dragging_robot_index), float(x), float(y))

    def mouseReleaseEvent(self, event):
        if self.dragging_robot_index is not None:
            self.dragging_robot_index = None
            self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, event):
        self.record_render_frame()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        self.draw_card(painter)
        self.draw_title(painter)
        self.draw_plot(painter)
        self.draw_telemetry(painter)

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
            status = painter.fontMetrics().elidedText(
                self.status_message,
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
        self.draw_topography(cache_painter, rect)
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
            self.draw_topography(painter, rect)
            self.draw_grid(painter, rect)

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

    def draw_grid(self, painter: QPainter, rect):
        for x in range(math.ceil(WORLD_X_MIN), math.floor(WORLD_X_MAX) + 1):
            sx, _ = self.world_to_screen(x, 0)
            painter.setPen(QPen(GRID_AXIS, 1.5) if x == 0 else QPen(GRID, 1))
            painter.drawLine(QPointF(sx, rect.top()), QPointF(sx, rect.bottom()))

        for y in range(math.ceil(WORLD_Y_MIN), math.floor(WORLD_Y_MAX) + 1):
            _, sy = self.world_to_screen(0, y)
            painter.setPen(QPen(GRID_AXIS, 1.5) if y == 0 else QPen(GRID, 1))
            painter.drawLine(QPointF(rect.left(), sy), QPointF(rect.right(), sy))

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

        for index, obstacle in enumerate(self.config.obstacles):
            ox, oy, ow, oh = obstacle
            coverage = self._obstacle_coverage_cache.get(index, 0.0)
            fully_discovered = coverage >= OBSTACLE_COMPLETE_COVERAGE

            if fully_discovered:
                fill = QColor(190, 194, 202, 170)
                stroke = QColor(82, 84, 92, 210)
                pen_width = 1.7
            else:
                fill = QColor(178, 181, 188, 85)
                stroke = QColor(82, 84, 92, 105)
                pen_width = 1.2

            cache_painter.setPen(QPen(stroke, pen_width))
            cache_painter.setBrush(QBrush(fill))

            x1, y1 = self.world_to_screen(ox, oy)
            x2, y2 = self.world_to_screen(ox + ow, oy + oh)

            left = min(x1, x2)
            top = min(y1, y2)
            width = abs(x2 - x1)
            height = abs(y2 - y1)

            cache_painter.drawRect(QRectF(left, top, width, height))

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

    def draw_safety_radius(self, painter: QPainter):
        """
        Draw safety radius r below mapped obstacles and waypoints.

        In multi-robot mode every robot gets its own colored safety radius when
        Robot Orders is enabled.
        """
        px_per_meter = self.plot_rect().width() / (WORLD_X_MAX - WORLD_X_MIN)

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
                radius = 9
                fill = QColor(BLUE_DARK)
                stroke = QColor("white")
                text_color = QColor("white")
            elif i == last_index:
                goal_xy = self.current_goal_xy()
                is_final_goal = math.hypot(point[0] - goal_xy[0], point[1] - goal_xy[1]) <= max(0.20, self.config.goal_tolerance)
                label = "G" if is_final_goal else "F"
                radius = 10
                fill = QColor(GREEN) if is_final_goal else QColor(146, 62, 160)
                stroke = QColor("white")
                text_color = QColor("white")
            else:
                label = str(i)
                radius = 7
                fill = QColor("white")
                stroke = QColor(ORANGE)
                text_color = QColor(MAROON)

            if i == active_index:
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(225, 126, 38, 55)))
                painter.drawEllipse(QRectF(sx - 17, sy - 17, 34, 34))
                radius = 11
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
                    radius = 8
                    fill = QColor(color)
                    stroke = QColor("white")
                    text_color = QColor("white")
                elif i == last_index:
                    goal_xy = self.current_goal_xy()
                    is_final_goal = math.hypot(point[0] - goal_xy[0], point[1] - goal_xy[1]) <= max(0.20, self.config.goal_tolerance)
                    label = "G" if is_final_goal else "F"
                    radius = 9
                    fill = QColor(GREEN) if is_final_goal else QColor(146, 62, 160)
                    stroke = QColor("white")
                    text_color = QColor("white")
                else:
                    label = str(i)
                    radius = 6
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
            px_per_meter = self.plot_rect().width() / (WORLD_X_MAX - WORLD_X_MIN)
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
            px_per_meter = self.plot_rect().width() / (WORLD_X_MAX - WORLD_X_MIN)

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
        px_per_meter = self.plot_rect().width() / (WORLD_X_MAX - WORLD_X_MIN)
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


