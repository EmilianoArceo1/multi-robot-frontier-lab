"""
perf_monitor.py

Lightweight, low-overhead timing instrumentation for the simulation loop --
where time goes each tick and every major section inside it, belief-trace
queue pressure, and job/replan/failure counts -- independent of both
ROBOT_TRACE (what the robot did) and render_perf.py's RenderPerfMonitor
(paint_fps/paint_ms, measured from inside paintEvent). This module never
touches Qt and knows nothing about painting itself; engine.py feeds it a
render_ms number from render_perf.py's existing measurement when producing
the combined summary.

Recording a timing (record()/time_phase()) is always cheap: a couple of
dict updates. The periodic [PERF] summary line is the only thing gated
behind SIM_PERF_LOG (env var, default off) plus its own throttle -- so
this stays silent by default and never spams even when enabled.

Per-window semantics (important): every *_ms/*_jobs_*/*_replans/*_failures
figure in the emitted line is accumulated ONLY since the PREVIOUS emitted
[PERF] line, never carried forward. Section timing accumulators
(_section_sum/_section_count/_section_max) are cleared immediately after
each successful emit. The five job/replan/failure counters are supplied by
the caller as running CUMULATIVE totals (since engine.py just keeps
incrementing plain integers) and PerfMonitor itself diffs each one against
a stored baseline from the previous window -- so e.g. planner_dispatch_ms
or planner_jobs_started correctly read 0.0/0 in a window where nothing new
happened, instead of showing a stale figure from an old, no-longer-
representative sample still sitting in a rolling window.

sim_step reports BOTH avg_sim_step_ms and max_sim_step_ms (a single
"sim_step_ms" was ambiguous about which is meant, and repeated occasional
long ticks can hide inside an otherwise-low average).

unaccounted_ms = avg_sim_step_ms minus the sum of the TOP-LEVEL, mutually
non-overlapping sections measured INSIDE simulation_step() (see
_UNACCOUNTED_SECTIONS below). Deliberately excludes:
    - render_ms: paintEvent runs on a separate Qt callback, not nested
      inside simulation_step() at all, and is not reset per-window the
      same way (it is RenderPerfMonitor's own short rolling average) --
      never subtracted, and frame_total_ms is not computed at all, since
      render throttling means a given tick's simulation_step() call does
      not reliably correspond to the same paintEvent.
    - planner_dispatch/route_result_handling/pending_path_acceptance/
      telemetry/console_log: each of these can be (and usually is) a
      NESTED sub-timing of apply_decision (e.g. a REQUEST_PLAN decision
      calls request_route_async() -> planner_dispatch, from inside
      apply_navigation_decision() -> apply_decision), or can fire from a
      queued Qt signal callback delivered on a different event-loop turn
      than the sim_step window it would otherwise be compared against
      (route_result_handling, for an async planner result). Including
      them in the unaccounted-ms subtraction would double-count time
      already inside apply_decision_ms, or subtract time that was never
      part of this tick's sim_step_ms in the first place. They are still
      reported on the [PERF] line as their own fields for visibility into
      which one dominates -- just not folded into unaccounted_ms.
"""
from __future__ import annotations

import os
import time
from typing import Callable

DEFAULT_LOG_INTERVAL_S = 2.0

# Top-level, non-overlapping sections inside simulation_step() used to
# compute unaccounted_ms -- see module docstring for why the others
# (planner_dispatch, route_result_handling, pending_path_acceptance,
# telemetry, console_log) are excluded.
_UNACCOUNTED_SECTIONS: tuple[str, ...] = (
    "explored_update",
    "obstacle_extract",
    "belief_update",
    "runtime_state_build",
    "controller",
    "route_affected_check",
    "nav_decision",
    "apply_decision",
    "motion_update",
    "canvas_state_update",
    "belief_snapshot",
)

# Cumulative counters the caller reports each window; PerfMonitor diffs
# each against a stored baseline so the emitted figure is the delta since
# the previous window, not the lifetime total.
_WINDOW_COUNTER_NAMES: tuple[str, ...] = (
    "planner_jobs_started",
    "planner_jobs_completed",
    "safety_replans",
    "route_failures",
    "repeated_safety_replans",
)


def _env_enabled(source, name: str, default: str = "0") -> bool:
    return str(source.get(name, default)).strip().lower() not in {"0", "false", "no", "off"}


def format_perf_summary_line(
    *,
    avg_sim_step_ms: float,
    max_sim_step_ms: float = 0.0,
    explored_update_ms: float = 0.0,
    obstacle_extract_ms: float = 0.0,
    belief_update_ms: float = 0.0,
    runtime_state_build_ms: float = 0.0,
    controller_ms: float = 0.0,
    nav_ms: float = 0.0,
    apply_decision_ms: float = 0.0,
    motion_ms: float = 0.0,
    route_check_ms: float = 0.0,
    planner_dispatch_ms: float = 0.0,
    route_result_ms: float = 0.0,
    pending_path_ms: float = 0.0,
    snapshot_ms: float = 0.0,
    telemetry_ms: float = 0.0,
    canvas_ms: float = 0.0,
    render_ms: float = 0.0,
    console_ms: float = 0.0,
    unaccounted_ms: float = 0.0,
    mapped_obs: int = 0,
    explored_percent: float | None = None,
    nav_state: str | None = None,
    fps: float = 0.0,
    trace_queue: int = 0,
    dropped_trace_events: int = 0,
    planner_jobs_started: int = 0,
    planner_jobs_completed: int = 0,
    safety_replans: int = 0,
    route_failures: int = 0,
    repeated_safety_replans: int = 0,
) -> str:
    """Pure formatter, kept separate from PerfMonitor's timing/throttle
    state so the exact line format can be tested without driving a real
    timing loop.

    explored_percent/nav_state are the only genuinely optional fields
    (omitted entirely, not shown as a placeholder, when not given/cheap);
    everything else always appears.
    """
    parts = [
        "[PERF]",
        f"avg_sim_step_ms={float(avg_sim_step_ms):.1f}",
        f"max_sim_step_ms={float(max_sim_step_ms):.1f}",
        f"explored_update_ms={float(explored_update_ms):.1f}",
        f"obstacle_extract_ms={float(obstacle_extract_ms):.1f}",
        f"belief_update_ms={float(belief_update_ms):.1f}",
        f"runtime_state_build_ms={float(runtime_state_build_ms):.1f}",
        f"controller_ms={float(controller_ms):.1f}",
        f"nav_ms={float(nav_ms):.1f}",
        f"apply_ms={float(apply_decision_ms):.1f}",
        f"motion_ms={float(motion_ms):.1f}",
        f"route_check_ms={float(route_check_ms):.1f}",
        f"planner_dispatch_ms={float(planner_dispatch_ms):.1f}",
        f"route_result_ms={float(route_result_ms):.1f}",
        f"pending_path_ms={float(pending_path_ms):.1f}",
        f"snapshot_ms={float(snapshot_ms):.1f}",
        f"telemetry_ms={float(telemetry_ms):.1f}",
        f"canvas_ms={float(canvas_ms):.1f}",
        f"render_ms={float(render_ms):.1f}",
        f"console_ms={float(console_ms):.1f}",
        f"unaccounted_ms={float(unaccounted_ms):.1f}",
        f"mapped_obs={int(mapped_obs)}",
    ]
    if explored_percent is not None:
        parts.append(f"explored={float(explored_percent):.1f}%")
    if nav_state:
        parts.append(f"nav={nav_state}")
    parts.extend([
        f"fps={float(fps):.1f}",
        f"trace_queue={int(trace_queue)}",
        f"dropped_trace_events={int(dropped_trace_events)}",
        f"planner_jobs_started={int(planner_jobs_started)}",
        f"planner_jobs_completed={int(planner_jobs_completed)}",
        f"safety_replans={int(safety_replans)}",
        f"route_failures={int(route_failures)}",
        f"repeated_safety_replans={int(repeated_safety_replans)}",
    ])
    return " ".join(parts)


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
    """Per-window timing/counter accumulator plus a throttled [PERF] summary.

    Reads SIM_PERF_LOG from the environment at CONSTRUCTION time (not
    import time), mirroring RobotTrace's own env-reading convention, so
    tests can pass an explicit `env` mapping for a deterministic instance.
    """

    def __init__(self, env: "dict[str, str] | None" = None):
        source = env if env is not None else os.environ
        self.logging_enabled = _env_enabled(source, "SIM_PERF_LOG")
        self._section_sum: dict[str, float] = {}
        self._section_count: dict[str, int] = {}
        self._section_max: dict[str, float] = {}
        self._counter_baseline: dict[str, int] = {}
        self._last_log_time: float | None = None
        self._tick_count = 0
        self._last_fps_time: float | None = None
        self._last_fps_tick_count = 0

    def record(self, phase: str, duration_s: float) -> None:
        """Record one timing sample for *phase*, accumulated into the
        CURRENT window only (cleared after the next successful
        maybe_log_summary() emit). Ignores negative/NaN durations instead
        of raising -- a bad timing sample must never break the simulation
        loop."""
        try:
            duration_s = float(duration_s)
        except (TypeError, ValueError):
            return
        if duration_s < 0 or duration_s != duration_s:  # NaN check
            return
        self._section_sum[phase] = self._section_sum.get(phase, 0.0) + duration_s
        self._section_count[phase] = self._section_count.get(phase, 0) + 1
        self._section_max[phase] = max(self._section_max.get(phase, 0.0), duration_s)

    def time_phase(self, phase: str) -> _PhaseTimer:
        """with monitor.time_phase("sim_step"): ... -- records elapsed
        wall-clock time under `phase` on exit."""
        return _PhaseTimer(self, phase)

    def average_ms(self, phase: str) -> float:
        count = self._section_count.get(phase, 0)
        if count == 0:
            return 0.0
        return 1000.0 * self._section_sum.get(phase, 0.0) / count

    def max_ms(self, phase: str) -> float:
        return 1000.0 * self._section_max.get(phase, 0.0)

    def note_tick(self) -> None:
        """Call once per simulation tick -- feeds the fps figure in the
        [PERF] summary line."""
        self._tick_count += 1

    def _window_delta(self, name: str, cumulative_value: int) -> int:
        """Diff a caller-supplied CUMULATIVE counter against its value at
        the start of the current window, returning only the delta. A
        counter never seen before establishes its own baseline (delta 0
        for that first window) rather than reporting everything
        accumulated since process start."""
        cumulative_value = int(cumulative_value)
        baseline = self._counter_baseline.get(name, cumulative_value)
        self._counter_baseline[name] = cumulative_value
        return cumulative_value - baseline

    def maybe_log_summary(
        self,
        *,
        render_ms: float = 0.0,
        trace_queue_size: int = 0,
        dropped_trace_events: int = 0,
        mapped_obstacle_count: int = 0,
        explored_percent: float | None = None,
        nav_state: str | None = None,
        planner_jobs_started: int = 0,
        planner_jobs_completed: int = 0,
        safety_replans: int = 0,
        route_failures: int = 0,
        repeated_safety_replans: int = 0,
        log: Callable[[str], None] | None = None,
        now: float | None = None,
    ) -> bool:
        """Throttled to at most once every DEFAULT_LOG_INTERVAL_S seconds
        of real wall-clock time -- a no-op unless SIM_PERF_LOG is enabled.
        `log` defaults to print(); pass a console-message sink instead for
        GUI visibility. Returns True iff a line was actually emitted.

        Every *_ms section in the emitted line (other than render_ms) is
        pulled from this instance's own PER-WINDOW accumulators
        (record()/time_phase()), which are reset immediately after a
        successful emit -- see the module docstring. The five job/replan/
        failure counters are diffed against their own per-window baseline
        (_window_delta()) so they too read 0 in a window where nothing
        new happened, never a stale carried-forward figure.
        """
        if not self.logging_enabled:
            return False
        now = time.monotonic() if now is None else float(now)
        if self._last_log_time is not None and (now - self._last_log_time) < DEFAULT_LOG_INTERVAL_S:
            return False
        self._last_log_time = now

        fps = self._compute_fps(now)

        avg_sim_step_ms = self.average_ms("sim_step")
        max_sim_step_ms = self.max_ms("sim_step")
        measured_inside_sim_step = sum(self.average_ms(phase) for phase in _UNACCOUNTED_SECTIONS)
        unaccounted_ms = avg_sim_step_ms - measured_inside_sim_step

        line = format_perf_summary_line(
            avg_sim_step_ms=avg_sim_step_ms,
            max_sim_step_ms=max_sim_step_ms,
            explored_update_ms=self.average_ms("explored_update"),
            obstacle_extract_ms=self.average_ms("obstacle_extract"),
            belief_update_ms=self.average_ms("belief_update"),
            runtime_state_build_ms=self.average_ms("runtime_state_build"),
            controller_ms=self.average_ms("controller"),
            nav_ms=self.average_ms("nav_decision"),
            apply_decision_ms=self.average_ms("apply_decision"),
            motion_ms=self.average_ms("motion_update"),
            route_check_ms=self.average_ms("route_affected_check"),
            planner_dispatch_ms=self.average_ms("planner_dispatch"),
            route_result_ms=self.average_ms("route_result_handling"),
            pending_path_ms=self.average_ms("pending_path_acceptance"),
            snapshot_ms=self.average_ms("belief_snapshot"),
            telemetry_ms=self.average_ms("telemetry"),
            canvas_ms=self.average_ms("canvas_state_update"),
            render_ms=render_ms,
            console_ms=self.average_ms("console_log"),
            unaccounted_ms=unaccounted_ms,
            mapped_obs=mapped_obstacle_count,
            explored_percent=explored_percent,
            nav_state=nav_state,
            fps=fps,
            trace_queue=trace_queue_size,
            dropped_trace_events=dropped_trace_events,
            planner_jobs_started=self._window_delta("planner_jobs_started", planner_jobs_started),
            planner_jobs_completed=self._window_delta("planner_jobs_completed", planner_jobs_completed),
            safety_replans=self._window_delta("safety_replans", safety_replans),
            route_failures=self._window_delta("route_failures", route_failures),
            repeated_safety_replans=self._window_delta("repeated_safety_replans", repeated_safety_replans),
        )
        (log or print)(line)
        self._reset_window()
        return True

    def _reset_window(self) -> None:
        """Clear per-window timing accumulators after a successful emit --
        NOT the counter baselines (_counter_baseline) or the fps/log
        throttle timestamps, which must persist across windows to keep
        diffing/throttling correctly."""
        self._section_sum.clear()
        self._section_count.clear()
        self._section_max.clear()

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
