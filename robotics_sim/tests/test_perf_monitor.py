"""
Tests for perf_monitor.py: low-overhead sim-loop timing, silent unless
SIM_PERF_LOG is explicitly enabled.

Manual Office.sim PERF evidence found the previous (rolling-window) design
internally inconsistent: planner_dispatch_ms stayed at an old, stale value
(e.g. 24.0) in windows where no planner job was dispatched at all, because
samples sat in a 120-sample rolling deque long after they stopped being
representative. This round switches to strict PER-WINDOW accounting: every
section accumulator (and the five job/replan/failure counters) is reset
immediately after each emitted [PERF] line, so a quiet window correctly
reads 0.0/0 instead of an old sample. sim_step_ms was also ambiguous
(single figure hiding whether it was typical or a rare spike) -- replaced
with avg_sim_step_ms + max_sim_step_ms.
"""
from __future__ import annotations

from robotics_sim.simulation.perf_monitor import PerfMonitor, format_perf_summary_line


# ---------------------------------------------------------------------------
# Disabled by default: recording timings never produces output.
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
    assert "avg_sim_step_ms=15.0" in captured.out
    assert "max_sim_step_ms=20.0" in captured.out
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
        avg_sim_step_ms=36.8, max_sim_step_ms=52.0, explored_update_ms=1.1,
        obstacle_extract_ms=2.2, belief_update_ms=20.4, runtime_state_build_ms=0.5,
        controller_ms=0.3, nav_ms=0.4, apply_decision_ms=0.3, motion_ms=0.1,
        route_check_ms=0.1, planner_dispatch_ms=1.7, route_result_ms=0.3,
        pending_path_ms=0.0, snapshot_ms=0.2, telemetry_ms=0.1, canvas_ms=0.0,
        render_ms=21.0, console_ms=0.0, unaccounted_ms=9.8, mapped_obs=3350,
        fps=9.3, trace_queue=0, dropped_trace_events=0,
    )
    assert line == (
        "[PERF] avg_sim_step_ms=36.8 max_sim_step_ms=52.0 explored_update_ms=1.1 "
        "obstacle_extract_ms=2.2 belief_update_ms=20.4 runtime_state_build_ms=0.5 "
        "controller_ms=0.3 nav_ms=0.4 apply_ms=0.3 motion_ms=0.1 route_check_ms=0.1 "
        "planner_dispatch_ms=1.7 route_result_ms=0.3 pending_path_ms=0.0 snapshot_ms=0.2 "
        "telemetry_ms=0.1 canvas_ms=0.0 render_ms=21.0 console_ms=0.0 unaccounted_ms=9.8 "
        "mapped_obs=3350 fps=9.3 trace_queue=0 dropped_trace_events=0 "
        "planner_jobs_started=0 planner_jobs_completed=0 safety_replans=0 "
        "route_failures=0 repeated_safety_replans=0"
    )


def test_format_perf_summary_line_omits_explored_and_nav_when_not_given():
    line = format_perf_summary_line(avg_sim_step_ms=1.0)
    assert "explored=" not in line
    assert "nav=" not in line


def test_format_perf_summary_line_includes_explored_and_nav_when_given():
    line = format_perf_summary_line(avg_sim_step_ms=1.0, explored_percent=42.5, nav_state="exhausted")
    assert "explored=42.5%" in line
    assert "nav=exhausted" in line


# ---------------------------------------------------------------------------
# A. Section timings are reset to 0 after a [PERF] line is emitted -- a
#    quiet window must never show a stale figure from an earlier one.
# ---------------------------------------------------------------------------


def test_perf_monitor_resets_window_timings_after_emit(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.050)
    monitor.record("nav_decision", 0.020)
    monitor.record("planner_dispatch", 0.500)

    monitor.maybe_log_summary(now=0.0)
    captured = capsys.readouterr()
    assert "nav_ms=20.0" in captured.out
    assert "planner_dispatch_ms=500.0" in captured.out

    # Second window: nothing recorded at all -- everything must read 0.0,
    # not the previous window's values.
    emitted = monitor.maybe_log_summary(now=3.0)
    assert emitted is True
    captured = capsys.readouterr()
    assert "avg_sim_step_ms=0.0" in captured.out
    assert "max_sim_step_ms=0.0" in captured.out
    assert "nav_ms=0.0" in captured.out
    assert "planner_dispatch_ms=0.0" in captured.out


# ---------------------------------------------------------------------------
# B. planner_dispatch_ms specifically must not carry forward once no new
#    planner jobs are dispatched -- the exact bug from the manual evidence
#    (planner_dispatch_ms staying 24.0 after exhaustion).
# ---------------------------------------------------------------------------


def test_planner_dispatch_ms_does_not_carry_forward_without_new_jobs(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.030)
    monitor.record("planner_dispatch", 0.024)
    monitor.maybe_log_summary(now=0.0)
    capsys.readouterr()

    # Robot is now exhausted -- no more planner dispatches happen in this
    # (or any later) window.
    monitor.record("sim_step", 0.005)
    monitor.maybe_log_summary(now=3.0)
    captured = capsys.readouterr()
    assert "planner_dispatch_ms=0.0" in captured.out

    monitor.record("sim_step", 0.005)
    monitor.maybe_log_summary(now=6.0)
    captured = capsys.readouterr()
    assert "planner_dispatch_ms=0.0" in captured.out


# ---------------------------------------------------------------------------
# C/D. unaccounted_ms excludes render_ms (measured outside simulation_step)
#      and nested/cross-turn sections (planner_dispatch, route_result,
#      pending_path, telemetry, console_log).
# ---------------------------------------------------------------------------


def test_unaccounted_excludes_render_time(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.050)
    monitor.record("nav_decision", 0.010)

    monitor.maybe_log_summary(render_ms=999.0, now=0.0)

    captured = capsys.readouterr()
    # measured = 10ms; sim_step=50ms -> unaccounted=40.0, NOT reduced by
    # the huge render_ms figure.
    assert "unaccounted_ms=40.0" in captured.out


def test_unaccounted_uses_only_sim_step_sections(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.050)
    monitor.record("apply_decision", 0.010)
    monitor.record("planner_dispatch", 0.500)  # nested inside apply_decision -- must be ignored
    monitor.record("route_result_handling", 0.500)  # separate event-loop turn -- must be ignored
    monitor.record("telemetry", 0.300)  # partially nested -- must be ignored
    monitor.record("console_log", 0.200)  # cross-cutting -- must be ignored

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    # measured = 10ms (apply_decision only); sim_step=50ms -> unaccounted=40.0
    assert "unaccounted_ms=40.0" in captured.out


# ---------------------------------------------------------------------------
# E. avg_sim_step_ms and max_sim_step_ms are reported separately.
# ---------------------------------------------------------------------------


def test_perf_monitor_reports_avg_and_max_sim_step(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.010)
    monitor.record("sim_step", 0.020)
    monitor.record("sim_step", 0.052)

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    # avg = (10+20+52)/3 = 27.33 -> 27.3
    assert "avg_sim_step_ms=27.3" in captured.out
    assert "max_sim_step_ms=52.0" in captured.out


# ---------------------------------------------------------------------------
# F. planner_jobs_started/completed, safety_replans, route_failures, and
#    repeated_safety_replans are per-window deltas of a caller-supplied
#    cumulative total, not the raw cumulative figure itself.
# ---------------------------------------------------------------------------


def test_perf_monitor_counts_are_per_window(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.010)

    # First window establishes the baseline -- delta is 0 even though the
    # cumulative total is already 10.
    monitor.maybe_log_summary(planner_jobs_started=10, now=0.0)
    captured = capsys.readouterr()
    assert "planner_jobs_started=0" in captured.out

    # Second window: 5 more jobs started since the baseline.
    monitor.record("sim_step", 0.010)
    monitor.maybe_log_summary(planner_jobs_started=15, now=3.0)
    captured = capsys.readouterr()
    assert "planner_jobs_started=5" in captured.out

    # Third window: no new jobs -- delta must be 0, not the stale 15.
    monitor.record("sim_step", 0.010)
    monitor.maybe_log_summary(planner_jobs_started=15, now=6.0)
    captured = capsys.readouterr()
    assert "planner_jobs_started=0" in captured.out


# ---------------------------------------------------------------------------
# G. belief_update_ms is reported as its own section (occupancy/belief map
#    update, distinct from obstacle_extract_ms/explored_update_ms).
# ---------------------------------------------------------------------------


def test_perf_monitor_can_report_belief_update_section(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.100)
    monitor.record("belief_update", 0.0204)

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    assert "belief_update_ms=20.4" in captured.out
