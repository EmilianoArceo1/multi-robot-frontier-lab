"""
perf_monitor.py

Lightweight, low-overhead timing instrumentation for the simulation loop --
where time goes each tick (sim_step, sensor update, route_affected check,
belief-trace queue pressure), independent of both ROBOT_TRACE (what the
robot did) and render_perf.py's RenderPerfMonitor (paint_fps/paint_ms,
measured from inside paintEvent). This module never touches Qt and knows
nothing about painting itself; engine.py feeds it a render_ms number from
render_perf.py's existing measurement when producing the combined summary.

Recording a timing (record()/time_phase()) is always cheap: a handful of
time.perf_counter() calls and a bounded rolling-window append. The
periodic [PERF] summary line is the only thing gated behind SIM_PERF_LOG
(env var, default off) plus its own throttle -- so this stays silent by
default and never spams even when enabled.
"""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Callable

DEFAULT_LOG_INTERVAL_S = 2.0
_ROLLING_WINDOW = 120  # a few seconds of ticks at typical simulation rates


def _env_enabled(source, name: str, default: str = "0") -> bool:
    return str(source.get(name, default)).strip().lower() not in {"0", "false", "no", "off"}


def format_perf_summary_line(
    *,
    sim_step_ms: float,
    render_ms: float,
    trace_queue: int,
    dropped_trace_events: int,
    fps: float,
) -> str:
    """Pure formatter, kept separate from PerfMonitor's timing/throttle
    state so the exact line format can be tested without driving a real
    timing loop."""
    return (
        f"[PERF] sim_step_ms={float(sim_step_ms):.1f} render_ms={float(render_ms):.1f} "
        f"trace_queue={int(trace_queue)} dropped_trace_events={int(dropped_trace_events)} "
        f"fps={float(fps):.1f}"
    )


class _PhaseTimer:
    """Context manager returned by PerfMonitor.time_phase(); records the
    elapsed wall-clock time on __exit__ regardless of whether the wrapped
    block raised."""

    def __init__(self, monitor: "PerfMonitor", phase: str):
        self._monitor = monitor
        self._phase = phase
        self._start = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._monitor.record(self._phase, time.perf_counter() - self._start)
        return False


class PerfMonitor:
    """Rolling per-phase timing counters plus a throttled [PERF] summary.

    Reads SIM_PERF_LOG from the environment at CONSTRUCTION time (not
    import time), mirroring RobotTrace's own env-reading convention, so
    tests can pass an explicit `env` mapping for a deterministic instance.
    """

    def __init__(self, env: "dict[str, str] | None" = None):
        source = env if env is not None else os.environ
        self.logging_enabled = _env_enabled(source, "SIM_PERF_LOG")
        self._durations: dict[str, deque] = {}
        self._last_log_time: float | None = None
        self._tick_count = 0
        self._last_fps_time: float | None = None
        self._last_fps_tick_count = 0

    def record(self, phase: str, duration_s: float) -> None:
        """Record one timing sample for *phase*. Ignores negative/NaN
        durations instead of raising -- a bad timing sample must never
        break the simulation loop."""
        try:
            duration_s = float(duration_s)
        except (TypeError, ValueError):
            return
        if duration_s < 0 or duration_s != duration_s:  # NaN check
            return
        bucket = self._durations.setdefault(phase, deque(maxlen=_ROLLING_WINDOW))
        bucket.append(duration_s)

    def time_phase(self, phase: str) -> _PhaseTimer:
        """with monitor.time_phase("sim_step"): ... -- records elapsed
        wall-clock time under `phase` on exit."""
        return _PhaseTimer(self, phase)

    def average_ms(self, phase: str) -> float:
        bucket = self._durations.get(phase)
        if not bucket:
            return 0.0
        return 1000.0 * sum(bucket) / len(bucket)

    def note_tick(self) -> None:
        """Call once per simulation tick -- feeds the fps figure in the
        [PERF] summary line."""
        self._tick_count += 1

    def maybe_log_summary(
        self,
        *,
        render_ms: float = 0.0,
        trace_queue_size: int = 0,
        dropped_trace_events: int = 0,
        log: Callable[[str], None] | None = None,
        now: float | None = None,
    ) -> bool:
        """Throttled to at most once every DEFAULT_LOG_INTERVAL_S seconds
        of real wall-clock time -- a no-op unless SIM_PERF_LOG is enabled.
        `log` defaults to print(); pass a console-message sink instead for
        GUI visibility. Returns True iff a line was actually emitted."""
        if not self.logging_enabled:
            return False
        now = time.monotonic() if now is None else float(now)
        if self._last_log_time is not None and (now - self._last_log_time) < DEFAULT_LOG_INTERVAL_S:
            return False
        self._last_log_time = now

        fps = self._compute_fps(now)
        line = format_perf_summary_line(
            sim_step_ms=self.average_ms("sim_step"),
            render_ms=render_ms,
            trace_queue=trace_queue_size,
            dropped_trace_events=dropped_trace_events,
            fps=fps,
        )
        (log or print)(line)
        return True

    def _compute_fps(self, now: float) -> float:
        if self._last_fps_time is None:
            self._last_fps_time = now
            self._last_fps_tick_count = self._tick_count
            return 0.0
        elapsed = now - self._last_fps_time
        ticks = self._tick_count - self._last_fps_tick_count
        fps = (ticks / elapsed) if elapsed > 0 else 0.0
        self._last_fps_time = now
        self._last_fps_tick_count = self._tick_count
        return fps
