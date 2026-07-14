"""
Lightweight, throttled render-performance reporting for SimulationCanvas.

Decoupled from Qt and from the simulation engine on purpose: this only
measures/reports how fast paintEvent is actually running (paint_fps,
paint_ms), never simulation/physics cadence -- the canvas decides when a
frame was drawn, the engine has no role in render FPS.

Naming note: earlier versions of this module called these fields
render_fps/frame_ms. That was misleading -- what is actually measured is
paint EVENT frequency and the time spent inside paintEvent, not a
theoretical FPS derived independently from frame_ms (the two can and do
disagree once other work, e.g. the simulation tick or route planning,
delays paintEvent calls without directly costing time inside paintEvent
itself). Renamed to paint_fps/paint_ms to say exactly what is measured.
"""
from __future__ import annotations

import os
import time
from typing import Callable

DEFAULT_PERF_LOG_INTERVAL_S = 1.5
DEFAULT_GUI_WARNING_FPS_THRESHOLD = 20.0
DEFAULT_GUI_WARNING_INTERVAL_S = 5.0
DEFAULT_RENDER_DETAIL_LOG_INTERVAL_S = 2.0


def format_perf_line(
    *,
    paint_fps: float,
    paint_ms: float,
    overlay_enabled: bool,
    grid_resolution: float,
    visible_cells: int | None = None,
    cache_status: str = "n/a",
    snapshot_age_ms: float | None = None,
) -> str:
    """Pure formatter for a single periodic [PERF] console line.

    Kept separate from RenderPerfMonitor's counters/throttling so the exact
    line format can be tested without driving a real frame-timing loop.
    """
    overlay_text = "on" if overlay_enabled else "off"
    cells_text = "n/a" if visible_cells is None else str(int(visible_cells))
    line = (
        f"[PERF] paint_fps={float(paint_fps):.1f} paint_ms={float(paint_ms):.1f} "
        f"overlay={overlay_text} grid_res={float(grid_resolution):.2f} "
        f"visible_cells={cells_text} cache={cache_status}"
    )
    if snapshot_age_ms is not None:
        line += f" snapshot_age_ms={float(snapshot_age_ms):.0f}"
    return line


def format_gui_perf_warning(
    *, paint_fps: float, overlay_enabled: bool, grid_resolution: float
) -> str:
    """Pure formatter for the rare, throttled GUI-console warning line --
    distinct from the routine [PERF] line, which by default goes to stdout
    only (see SimulationCanvas.perf_to_gui_console)."""
    overlay_text = "on" if overlay_enabled else "off"
    return (
        f"[PERF WARN] paint_fps={float(paint_fps):.1f} is low "
        f"(overlay={overlay_text}, grid_res={float(grid_resolution):.2f})"
    )


def format_route_plan_perf_line(
    *,
    route_plan_ms: float,
    reason: str,
    grid_resolution: float,
    mapped_obs: int,
    result: str,
) -> str:
    """Pure formatter for a route-planning timing line, measured at the
    existing PlannerWorker.run() call boundary around
    compute_planned_waypoints() -- never inside A*/planner internals."""
    return (
        f"[PERF] route_plan_ms={float(route_plan_ms):.1f} reason={reason} "
        f"grid_res={float(grid_resolution):.2f} mapped_obs={int(mapped_obs)} result={result}"
    )


class RenderPerfMonitor:
    """Rolling paint_fps/paint_ms counter with a throttled [PERF] emitter.

    record_frame() is meant to be called once per paintEvent. It always
    updates the rolling paint_fps/paint_ms counters, but only returns a
    formatted log line about once every log_interval_s seconds of wall
    time -- most calls return None. Callers push the returned line (when
    not None) to whatever console/log sink they use; nothing is printed
    here directly, so this stays fully unit-testable without a GUI or real
    wall-clock waits (pass `now=` explicitly to drive it deterministically).
    """

    def __init__(self, log_interval_s: float = DEFAULT_PERF_LOG_INTERVAL_S):
        self.log_interval_s = float(log_interval_s)
        self.paint_fps = 0.0
        self.paint_ms = 0.0
        self._frame_count = 0
        self._frame_paint_ms_sum = 0.0
        self._window_start: float | None = None
        self._last_log_time: float | None = None

    def record_frame(
        self,
        *,
        paint_ms: float,
        overlay_enabled: bool,
        grid_resolution: float,
        visible_cells: int | None = None,
        cache_status: str = "n/a",
        snapshot_age_ms: float | None = None,
        now: float | None = None,
    ) -> str | None:
        now = time.perf_counter() if now is None else float(now)

        if self._window_start is None:
            self._window_start = now

        self._frame_count += 1
        self._frame_paint_ms_sum += float(paint_ms)
        elapsed = now - self._window_start

        if elapsed >= 0.25:
            self.paint_fps = self._frame_count / elapsed
            self.paint_ms = self._frame_paint_ms_sum / self._frame_count
            self._frame_count = 0
            self._frame_paint_ms_sum = 0.0
            self._window_start = now

        if self._last_log_time is not None and (now - self._last_log_time) < self.log_interval_s:
            return None

        self._last_log_time = now
        return format_perf_line(
            paint_fps=self.paint_fps,
            paint_ms=self.paint_ms,
            overlay_enabled=overlay_enabled,
            grid_resolution=grid_resolution,
            visible_cells=visible_cells,
            cache_status=cache_status,
            snapshot_age_ms=snapshot_age_ms,
        )


class PerfGuiWarningGate:
    """Decides whether a low paint_fps reading is worth a GUI-console line.

    Independent of RenderPerfMonitor's own ~1-2s [PERF] cadence: appending
    to the GUI console has its own cost (per the caller's Part B change,
    routine [PERF] lines go to stdout only), so only a genuinely severe,
    much less frequent drop is allowed to reach the GUI console at all.
    """

    def __init__(
        self,
        fps_threshold: float = DEFAULT_GUI_WARNING_FPS_THRESHOLD,
        interval_s: float = DEFAULT_GUI_WARNING_INTERVAL_S,
    ):
        self.fps_threshold = float(fps_threshold)
        self.interval_s = float(interval_s)
        self._last_warning_time: float | None = None

    def should_warn(self, paint_fps: float, now: float | None = None) -> bool:
        if paint_fps >= self.fps_threshold:
            return False

        now = time.perf_counter() if now is None else float(now)
        if self._last_warning_time is not None and (now - self._last_warning_time) < self.interval_s:
            return False

        self._last_warning_time = now
        return True


def _env_enabled(source, name: str, default: str = "0") -> bool:
    return str(source.get(name, default)).strip().lower() not in {"0", "false", "no", "off"}


def format_render_detail_line(
    *,
    total_ms: float,
    background_ms: float,
    map_layer_ms: float,
    robot_body_ms: float = 0.0,
    robot_fov_ms: float = 0.0,
    route_path_ms: float = 0.0,
    planned_route_build_ms: float = 0.0,
    planned_route_paint_ms: float = 0.0,
    executed_trail_build_ms: float = 0.0,
    executed_trail_paint_ms: float = 0.0,
    executed_trail_points: int = 0,
    executed_trail_segments_painted: int = 0,
    executed_trail_cache_hit: bool = False,
    sensor_debug_overlay_ms: float = 0.0,
    overlays_ms: float = 0.0,
    cache_hit: bool = False,
) -> str:
    """Pure formatter for the optional [RENDER] per-layer breakdown line,
    kept separate from RenderDetailLogger's throttle state so the exact
    line format can be tested without driving a real paintEvent.

    Layer grouping (see SimulationCanvas.draw_plot()):
        background_ms            ensure_static_plot_cache() + the
                                  background/grid-lines pixmap draw (or
                                  direct draw_grid() on a cache miss)
        map_layer_ms              grid overlay, explored-area trace,
                                  ground-truth obstacles, mapped obstacle
                                  points -- static/semi-static "what the
                                  map knows" layers
        robot_body_ms             goal marker + exploration-target labels
                                  + the robot glyph/heading arrow
                                  (draw_goal_and_robot())
        robot_fov_ms              the sensor/FoV cone (draw_sensor_range());
                                  cheap by design -- the expensive
                                  raycasting itself is cached in world
                                  space per robot pose (see
                                  sensor_polygon_for_pose()), so this is
                                  normally just a screen-space polygon
                                  transform + one drawPath() call
        route_path_ms             total time in planned + executed
                                  route/path drawing (draw_planned_route()/
                                  draw_executed_path(), or
                                  draw_multi_planned_routes() in
                                  multi-robot mode) -- see the finer-
                                  grained fields below for where it goes
        planned_route_build_ms   time spent checking/rebuilding the
                                  cached planned-route QPainterPath
                                  (cheap -- planned_path_points is short)
        planned_route_paint_ms   time spent drawing the cached planned-
                                  route path plus its waypoint/label
                                  markers
        executed_trail_build_ms  time spent checking the executed-trail
                                  pixmap cache and, on a miss, painting
                                  new/all segments into it
        executed_trail_paint_ms  time spent blitting the cached executed-
                                  trail pixmap (drawPixmap()) -- should
                                  stay ~constant regardless of trail
                                  length, unlike a QPainterPath redraw
        executed_trail_points    total points in the executed trail this
                                  frame (diagnostic only)
        executed_trail_segments_painted  segments actually painted into
                                  the trail pixmap this frame (0 when the
                                  pixmap was simply reused unchanged)
        executed_trail_cache_hit whether the trail pixmap was reused/
                                  appended to (True) rather than fully
                                  rebuilt (False) this frame
        sensor_debug_overlay_ms   the safety-radius "r" debug ring
                                  (draw_safety_radius(), only when Robot
                                  Orders is shown)
        overlays_ms               editor preview/selection/camera-frame,
                                  the grid-resolution preview, plus the
                                  card/title/telemetry chrome drawn
                                  outside draw_plot()
    robot_labels_ms is deliberately not broken out separately: waypoint/
    goal/frontier labels are drawn inline with their markers inside
    robot_body_ms's/route_path_ms's own drawing loops, and separating
    them would need restructuring that drawing code -- not worth the
    risk for a purely diagnostic distinction.
    cache_hit reflects whether the background layer (the most expensive
    cache) was reused this frame rather than rebuilt.
    """
    return (
        f"[RENDER] total_ms={float(total_ms):.1f} background_ms={float(background_ms):.1f} "
        f"map_layer_ms={float(map_layer_ms):.1f} robot_body_ms={float(robot_body_ms):.1f} "
        f"robot_fov_ms={float(robot_fov_ms):.1f} route_path_ms={float(route_path_ms):.1f} "
        f"planned_route_build_ms={float(planned_route_build_ms):.1f} "
        f"planned_route_paint_ms={float(planned_route_paint_ms):.1f} "
        f"executed_trail_build_ms={float(executed_trail_build_ms):.1f} "
        f"executed_trail_paint_ms={float(executed_trail_paint_ms):.1f} "
        f"executed_trail_points={int(executed_trail_points)} "
        f"executed_trail_segments_painted={int(executed_trail_segments_painted)} "
        f"executed_trail_cache_hit={bool(executed_trail_cache_hit)} "
        f"sensor_debug_overlay_ms={float(sensor_debug_overlay_ms):.1f} "
        f"overlays_ms={float(overlays_ms):.1f} cache_hit={bool(cache_hit)}"
    )


class RenderDetailLogger:
    """Optional, throttled per-layer paint-time breakdown -- disabled by
    default, enabled via SIM_RENDER_DETAIL_LOG=1 (read at construction
    time, mirroring RobotTrace/PerfMonitor/RenderThrottler's own env-
    reading convention). At most one [RENDER] line every
    DEFAULT_RENDER_DETAIL_LOG_INTERVAL_S seconds of wall-clock time, even
    when enabled -- never spams."""

    def __init__(self, env: "dict[str, str] | None" = None):
        source = env if env is not None else os.environ
        self.enabled = _env_enabled(source, "SIM_RENDER_DETAIL_LOG")
        self._last_log_time: float | None = None

    def maybe_log(
        self,
        *,
        total_ms: float,
        background_ms: float,
        map_layer_ms: float,
        robot_body_ms: float = 0.0,
        robot_fov_ms: float = 0.0,
        route_path_ms: float = 0.0,
        planned_route_build_ms: float = 0.0,
        planned_route_paint_ms: float = 0.0,
        executed_trail_build_ms: float = 0.0,
        executed_trail_paint_ms: float = 0.0,
        executed_trail_points: int = 0,
        executed_trail_segments_painted: int = 0,
        executed_trail_cache_hit: bool = False,
        sensor_debug_overlay_ms: float = 0.0,
        overlays_ms: float = 0.0,
        cache_hit: bool = False,
        log: Callable[[str], None] | None = None,
        now: float | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        now = time.perf_counter() if now is None else float(now)
        if self._last_log_time is not None and (now - self._last_log_time) < DEFAULT_RENDER_DETAIL_LOG_INTERVAL_S:
            return False
        self._last_log_time = now
        line = format_render_detail_line(
            total_ms=total_ms,
            background_ms=background_ms,
            map_layer_ms=map_layer_ms,
            robot_body_ms=robot_body_ms,
            robot_fov_ms=robot_fov_ms,
            route_path_ms=route_path_ms,
            planned_route_build_ms=planned_route_build_ms,
            planned_route_paint_ms=planned_route_paint_ms,
            executed_trail_build_ms=executed_trail_build_ms,
            executed_trail_paint_ms=executed_trail_paint_ms,
            executed_trail_points=executed_trail_points,
            executed_trail_segments_painted=executed_trail_segments_painted,
            executed_trail_cache_hit=executed_trail_cache_hit,
            sensor_debug_overlay_ms=sensor_debug_overlay_ms,
            overlays_ms=overlays_ms,
            cache_hit=cache_hit,
        )
        (log or print)(line)
        return True
