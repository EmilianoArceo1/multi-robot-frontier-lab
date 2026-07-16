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

import pytest

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
        "[PERF] avg_sim_step_ms=36.8 max_sim_step_ms=52.0 fast_path_avg_ms=0.00 "
        "full_pipeline_avg_ms=0.00 explored_update_ms=1.1 "
        "obstacle_extract_ms=2.2 belief_update_ms=20.4 runtime_state_build_ms=0.5 "
        "controller_ms=0.3 planner_services_refresh_ms=0.0 "
        "reachability_context_build_ms=0.0 reachability_obstacle_prepare_ms=0.0 "
        "reachability_grid_build_ms=0.0 reachability_context_builds=0 "
        "visible_candidate_obstacles_ms=0.0 "
        "nav_ms=0.4 apply_ms=0.3 motion_ms=0.1 route_check_ms=0.1 "
        "planner_dispatch_ms=1.7 route_result_ms=0.3 pending_path_ms=0.0 snapshot_ms=0.2 "
        "telemetry_ms=0.1 canvas_ms=0.0 render_ms=21.0 console_ms=0.0 misc_ms=0.0 "
        "top_level_sum_ms=0.0 "
        "unaccounted_ms=9.8 phase_coverage_pct=0.0 "
        "mapped_obs=3350 fps=9.3 trace_queue=0 dropped_trace_events=0 "
        "planner_jobs_started=0 planner_jobs_completed=0 safety_replans=0 "
        "route_failures=0 repeated_safety_replans=0 "
        "exhausted_idle_fast_path_hits=0 exhausted_idle_full_updates=0 "
        "exhausted_idle_skipped_canvas_updates=0 exhausted_idle_skipped_sensor_updates=0"
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


# ---------------------------------------------------------------------------
# H. The four exhausted-idle fast-path counters are per-window deltas of a
#    caller-supplied cumulative total, exactly like planner_jobs_started/
#    safety_replans/etc -- a quiet window with no new fast-path activity
#    must read 0, not the stale cumulative figure.
# ---------------------------------------------------------------------------


def test_perf_reports_exhausted_idle_counters(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.002)

    monitor.maybe_log_summary(
        exhausted_idle_fast_path_hits=40,
        exhausted_idle_full_updates=2,
        exhausted_idle_skipped_canvas_updates=40,
        exhausted_idle_skipped_sensor_updates=40,
        now=0.0,
    )
    captured = capsys.readouterr()
    # First window establishes the baseline -- delta is 0 even though the
    # cumulative total is already 40.
    assert "exhausted_idle_fast_path_hits=0" in captured.out
    assert "exhausted_idle_full_updates=0" in captured.out
    assert "exhausted_idle_skipped_canvas_updates=0" in captured.out
    assert "exhausted_idle_skipped_sensor_updates=0" in captured.out

    monitor.record("sim_step", 0.002)
    monitor.maybe_log_summary(
        exhausted_idle_fast_path_hits=100,
        exhausted_idle_full_updates=3,
        exhausted_idle_skipped_canvas_updates=100,
        exhausted_idle_skipped_sensor_updates=100,
        now=3.0,
    )
    captured = capsys.readouterr()
    assert "exhausted_idle_fast_path_hits=60" in captured.out
    assert "exhausted_idle_full_updates=1" in captured.out
    assert "exhausted_idle_skipped_canvas_updates=60" in captured.out
    assert "exhausted_idle_skipped_sensor_updates=60" in captured.out


# ---------------------------------------------------------------------------
# I. misc_ms is folded into unaccounted_ms's subtraction (a top-level,
#    non-overlapping section) -- confirms it behaves like the other
#    _UNACCOUNTED_SECTIONS entries, not like the excluded nested ones.
# ---------------------------------------------------------------------------


def test_misc_ms_is_subtracted_from_unaccounted(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.050)
    monitor.record("misc", 0.015)

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    assert "misc_ms=15.0" in captured.out
    # measured = 15ms; sim_step=50ms -> unaccounted=35.0
    assert "unaccounted_ms=35.0" in captured.out


# ---------------------------------------------------------------------------
# Exhausted-idle fast path accounting fix.
#
# Real Office.sim evidence: avg_sim_step_ms=0.6, obstacle_extract_ms=4.3,
# controller_ms=2.3, unaccounted_ms=-6.9, exhausted_idle_fast_path_hits=123,
# exhausted_idle_full_updates=2. The bug: average_ms(phase) divides by
# phase's OWN occurrence count (2, since obstacle_extract/controller only
# run on full-pipeline ticks), while avg_sim_step_ms divides by the total
# sim_step tick count (125, fast-path skips included) -- summing several
# such per-occurrence averages and subtracting from a per-tick average
# mixed two different denominators and went negative.
# ---------------------------------------------------------------------------


def test_fast_path_phase_metrics_use_total_tick_denominator():
    """125 total sim_step ticks (123 fast-path skips + 2 full-pipeline),
    but obstacle_extract only ever runs on the 2 full-pipeline ticks --
    per_tick_ms() must divide by 125, not by 2."""
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    for _ in range(123):
        monitor.record("sim_step", 0.0006)  # fast-path-skip tick: ~0.6ms
    for _ in range(2):
        monitor.record("sim_step", 0.0043)  # full-pipeline tick
        monitor.record("obstacle_extract", 0.0043)

    # Per-occurrence (unchanged, existing semantics): "when it runs, it
    # costs ~4.3ms" -- matches the real bad-line evidence exactly.
    assert monitor.average_ms("obstacle_extract") == pytest.approx(4.3, abs=0.01)

    # Per-tick (the fix): the SAME total obstacle_extract time (8.6ms)
    # spread over all 125 ticks, not just the 2 it happened to run on.
    expected_per_tick = 1000.0 * (0.0043 * 2) / 125
    assert monitor.per_tick_ms("obstacle_extract") == pytest.approx(expected_per_tick, abs=0.001)
    assert monitor.per_tick_ms("obstacle_extract") < monitor.average_ms("obstacle_extract"), (
        "per-tick contribution must be much smaller than the per-occurrence average "
        "once most ticks skip this phase entirely"
    )


def test_unaccounted_not_negative_when_fast_path_dominates(capsys):
    """Direct repro of the real bad Office.sim line: avg_sim_step_ms=0.6,
    obstacle_extract_ms=4.3, controller_ms=2.3, previously
    unaccounted_ms=-6.9. Must not go negative once per_tick_ms() is used."""
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    for _ in range(123):
        monitor.record("sim_step", 0.0006)
    for _ in range(2):
        monitor.record("sim_step", 0.0043)
        monitor.record("obstacle_extract", 0.0043)
        monitor.record("controller", 0.0023)

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    assert "unaccounted_ms=-" not in captured.out, "unaccounted_ms must never be negative"
    assert "obstacle_extract_ms=4.3" in captured.out, (
        "the per-occurrence field itself is still useful and must be unchanged"
    )


def test_full_update_metrics_reported_separately_from_per_tick_metrics(capsys):
    """fast_path_avg_ms/full_pipeline_avg_ms report each tick CATEGORY's
    own average cost separately, instead of blending a ~0.6ms skip tick
    and a ~5ms full-pipeline tick into one misleading avg_sim_step_ms."""
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    for _ in range(123):
        monitor.record("sim_step", 0.0006)
        monitor.record("sim_step_fast_path", 0.0006)
    for _ in range(2):
        monitor.record("sim_step", 0.0050)
        monitor.record("sim_step_full_pipeline", 0.0050)

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    assert "fast_path_avg_ms=0.60" in captured.out
    assert "full_pipeline_avg_ms=5.00" in captured.out


def test_render_ms_not_subtracted_from_sim_step_unaccounted(capsys):
    """render_ms is a render-side (paintEvent) measurement, not part of
    simulation_step() at all -- it must never be folded into
    top_level_sum_ms/unaccounted_ms, regardless of how large it is
    relative to a now-tiny (fast-path-dominated) avg_sim_step_ms."""
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.0006)
    monitor.record("misc", 0.0002)

    monitor.maybe_log_summary(render_ms=14.0, now=0.0)

    captured = capsys.readouterr()
    assert "render_ms=14.0" in captured.out
    # measured = 0.2ms; sim_step=0.6ms -> unaccounted=0.4, NOT reduced by
    # the much larger render_ms figure.
    assert "unaccounted_ms=0.4" in captured.out
    assert "unaccounted_ms=-" not in captured.out


def test_exhausted_idle_counters_still_reported(capsys):
    """Regression guard for this round's refactor: the four exhausted-idle
    counters added last round must still be correctly diffed/reported
    after switching unaccounted_ms/top_level_sum_ms to per_tick_ms()."""
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.0006)

    monitor.maybe_log_summary(
        exhausted_idle_fast_path_hits=123,
        exhausted_idle_full_updates=2,
        exhausted_idle_skipped_canvas_updates=123,
        exhausted_idle_skipped_sensor_updates=123,
        now=0.0,
    )
    captured = capsys.readouterr()
    # First window establishes the baseline.
    assert "exhausted_idle_fast_path_hits=0" in captured.out

    monitor.record("sim_step", 0.0006)
    monitor.maybe_log_summary(
        exhausted_idle_fast_path_hits=200,
        exhausted_idle_full_updates=3,
        exhausted_idle_skipped_canvas_updates=200,
        exhausted_idle_skipped_sensor_updates=200,
        now=3.0,
    )
    captured = capsys.readouterr()
    assert "exhausted_idle_fast_path_hits=77" in captured.out
    assert "exhausted_idle_full_updates=1" in captured.out
    assert "exhausted_idle_skipped_canvas_updates=77" in captured.out
    assert "exhausted_idle_skipped_sensor_updates=77" in captured.out


# ---------------------------------------------------------------------------
# J. planner_services_refresh_ms accounting (diagnosis-only round for
#    unaccounted_ms growth traced to ensure_planner_services()).
#
# Real Office.sim evidence: unaccounted_ms grew progressively with
# mapped_obs while nav_decision itself had separate 15-27ms spikes.
# ensure_planner_services() sits between the "controller" and
# "nav_decision" timers with no timer of its own, and its own docstring
# says it refreshes is_candidate_reachable on every call -- these tests
# confirm the new top-level timer correctly reduces unaccounted_ms, and
# that its nested reachability_* sub-timings are never subtracted a
# second time (the same double-counting bug pattern already fixed once
# for the exhausted-idle fast path).
# ---------------------------------------------------------------------------


def test_planner_services_refresh_ms_is_top_level_and_reduces_unaccounted(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.020)
    monitor.record("planner_services_refresh", 0.008)
    # Nested INSIDE planner_services_refresh -- must NOT be subtracted again.
    monitor.record("reachability_context_build", 0.007)
    monitor.record("reachability_obstacle_prepare", 0.003)
    monitor.record("reachability_grid_build", 0.004)

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    assert "planner_services_refresh_ms=8.0" in captured.out
    # measured = 8ms (planner_services_refresh only); sim_step=20ms ->
    # unaccounted=12.0, NOT reduced again by the nested reachability_*
    # figures (which would drive it negative: 20 - 8 - 7 - 3 - 4 < 0).
    assert "unaccounted_ms=12.0" in captured.out


def test_reachability_subfields_reported_without_double_counting(capsys):
    """reachability_context_build/obstacle_prepare/grid_build are NESTED
    entirely inside planner_services_refresh (the same call) and must not
    be subtracted from unaccounted_ms a second time.
    visible_candidate_obstacles, by contrast, is a SEPARATE top-level
    section (a genuine, non-overlapping gap inside update_sensed_obstacles(),
    which has no wrapping timer of its own) and DOES reduce unaccounted_ms
    on top of planner_services_refresh -- see perf_monitor.py's module
    docstring for why each is classified the way it is."""
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.050)
    monitor.record("planner_services_refresh", 0.030)
    monitor.record("reachability_context_build", 0.029)
    monitor.record("reachability_obstacle_prepare", 0.012)
    monitor.record("reachability_grid_build", 0.015)
    monitor.record("visible_candidate_obstacles", 0.001)

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    assert "reachability_context_build_ms=29.0" in captured.out
    assert "reachability_obstacle_prepare_ms=12.0" in captured.out
    assert "reachability_grid_build_ms=15.0" in captured.out
    assert "visible_candidate_obstacles_ms=1.0" in captured.out
    # measured = 30ms (planner_services_refresh) + 1ms
    # (visible_candidate_obstacles, a SEPARATE top-level section) = 31ms;
    # sim_step=50ms -> unaccounted=19.0. NOT 20.0 (visible_candidate_obstacles
    # must still count) and NOT negative (the nested reachability_* figures
    # must not be subtracted a second time on top of planner_services_refresh).
    assert "unaccounted_ms=19.0" in captured.out


def test_reachability_fields_have_safe_defaults(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.010)

    monitor.maybe_log_summary(now=0.0)

    captured = capsys.readouterr()
    assert "planner_services_refresh_ms=0.0" in captured.out
    assert "reachability_context_build_ms=0.0" in captured.out
    assert "reachability_obstacle_prepare_ms=0.0" in captured.out
    assert "reachability_grid_build_ms=0.0" in captured.out
    assert "reachability_context_builds=0" in captured.out
    assert "visible_candidate_obstacles_ms=0.0" in captured.out


def test_reachability_fields_silent_when_logging_disabled(capsys):
    monitor = PerfMonitor(env={})
    monitor.record("sim_step", 0.010)
    monitor.record("planner_services_refresh", 0.005)

    emitted = monitor.maybe_log_summary(reachability_context_builds=42, now=0.0)

    assert emitted is False
    captured = capsys.readouterr()
    assert captured.out == ""


def test_reachability_context_builds_is_per_window_delta(capsys):
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    monitor.record("sim_step", 0.010)

    monitor.maybe_log_summary(reachability_context_builds=10, now=0.0)
    captured = capsys.readouterr()
    # First window establishes the baseline -- delta is 0 even though the
    # cumulative total is already 10.
    assert "reachability_context_builds=0" in captured.out

    monitor.record("sim_step", 0.010)
    monitor.maybe_log_summary(reachability_context_builds=15, now=3.0)
    captured = capsys.readouterr()
    assert "reachability_context_builds=5" in captured.out
