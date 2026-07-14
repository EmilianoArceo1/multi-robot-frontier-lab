"""Tests for perf_monitor.py: low-overhead sim-loop timing, silent unless
SIM_PERF_LOG is explicitly enabled."""
from __future__ import annotations

from robotics_sim.simulation.perf_monitor import PerfMonitor, format_perf_summary_line


# ---------------------------------------------------------------------------
# H. Disabled by default: recording timings never produces output.
# ---------------------------------------------------------------------------


def test_perf_monitor_disabled_by_default_no_output(capsys):
    monitor = PerfMonitor(env={})

    with monitor.time_phase("sim_step"):
        pass
    monitor.note_tick()

    emitted = monitor.maybe_log_summary(render_ms=5.0, trace_queue_size=3, dropped_trace_events=0)

    assert emitted is False
    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# I. SIM_PERF_LOG=1 reports a summary line with the expected fields.
# ---------------------------------------------------------------------------


def test_perf_monitor_enabled_reports_summary(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})

    monitor.record("sim_step", 0.010)
    monitor.record("sim_step", 0.020)
    monitor.note_tick()
    monitor.note_tick()

    emitted = monitor.maybe_log_summary(
        render_ms=8.5, trace_queue_size=12, dropped_trace_events=3, now=0.0,
    )

    assert emitted is True
    captured = capsys.readouterr()
    assert "[PERF]" in captured.out
    assert "sim_step_ms=15.0" in captured.out
    assert "render_ms=8.5" in captured.out
    assert "trace_queue=12" in captured.out
    assert "dropped_trace_events=3" in captured.out
    assert "fps=" in captured.out

    # Throttled: a second call within DEFAULT_LOG_INTERVAL_S must not emit again.
    again = monitor.maybe_log_summary(render_ms=8.5, trace_queue_size=12, dropped_trace_events=3, now=0.5)
    assert again is False
    captured = capsys.readouterr()
    assert captured.out == ""


def test_format_perf_summary_line_is_pure_and_stable():
    line = format_perf_summary_line(
        sim_step_ms=12.3, render_ms=4.5, trace_queue=7, dropped_trace_events=1, fps=29.9,
    )
    assert line == "[PERF] sim_step_ms=12.3 render_ms=4.5 trace_queue=7 dropped_trace_events=1 fps=29.9"
