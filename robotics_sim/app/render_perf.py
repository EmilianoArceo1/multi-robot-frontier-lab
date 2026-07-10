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

import time

DEFAULT_PERF_LOG_INTERVAL_S = 1.5
DEFAULT_GUI_WARNING_FPS_THRESHOLD = 20.0
DEFAULT_GUI_WARNING_INTERVAL_S = 5.0


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
