"""
Tests for the opt-in, terminal-only ROBOT_TRACE diagnostic layer.

Manual motivation: Office.sim still gets stuck around ~19.7% explored with
repeated no_path/exploration-exhausted holds, and the existing app console
does not explain enough about what the robot believes is occupied/free/
unknown, which obstacle sections are blocking a narrow passage, which
frontier candidates were rejected, or why a route failed.

robot_trace.py adds a separate, richer, terminal-only trace layer for this
kind of debugging -- disabled by default, enabled via the ROBOT_TRACE
environment variable (e.g. ROBOT_TRACE=map,obstacles,decision,frontier,
route,safety or ROBOT_TRACE=all). It is deliberately independent of:
    - render_perf.py's PERF diagnostics (different concern, different env
      var, but the same "never print unless explicitly asked" principle),
    - the in-app GUI console (trace lines only ever go to stdout via
      print(), never SimulationCanvas.append_console_message()),
    - telemetry.py's [STATE]/[MAP]/[ROUTE]/[NAV]/[FRONTIER] lines (an
      always-available, GUI-console-aimed summary; this is a separate,
      terminal-only, off-by-default trace).

These tests exercise robot_trace.py directly (pure, no engine/Qt) --
RobotTrace reads its environment at CONSTRUCTION time (not import time),
via an explicit `env` mapping here, so each test gets a fresh, isolated,
deterministic instance without needing to mutate/restore os.environ.

Windows-safety note: a manual run on plain Windows PowerShell (stdout
encoding cp1252, not UTF-8) crashed with UnicodeEncodeError on the Unicode
approx sign "u2248" ("~=") previously used in format_safety_trace_line().
Fixed two ways: (1) every format_*_line() in robot_trace.py is ASCII-only
by construction now (min_clearance~0.38, not min_clearance u2248 0.38);
(2) RobotTrace._emit() also defensively encode/decode-sanitizes with
errors="replace" before printing, as a second line of defense against any
other unexpected non-ASCII content, so a narrow terminal encoding can
never crash the simulation.
"""
from __future__ import annotations

import io
import sys

from robotics_sim.simulation.robot_trace import (
    CATEGORIES,
    RobotTrace,
    format_decision_trace_line,
    format_frontier_trace_line,
    format_map_trace_line,
    format_obstacle_trace_line,
    format_route_trace_line,
    format_safety_trace_line,
    group_obstacle_points_into_sections,
    parse_categories,
)


class _Cp1252Stream(io.TextIOBase):
    """Minimal stand-in for a real Windows console with cp1252 stdout:
    raises UnicodeEncodeError on write() for anything cp1252 can't
    encode, exactly like the real crash this module fixes."""

    encoding = "cp1252"

    def __init__(self):
        self.written: list[str] = []

    def write(self, s: str) -> int:
        s.encode("cp1252")  # raises UnicodeEncodeError if not cp1252-safe
        self.written.append(s)
        return len(s)

    def flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# A. Disabled by default.
# ---------------------------------------------------------------------------


def test_robot_trace_disabled_by_default(capsys):
    trace = RobotTrace(env={})

    assert trace.enabled is False
    for category in CATEGORIES:
        assert not trace.is_enabled(category)

    emitted = trace.trace_map(
        sim_time=1.0,
        robot_label="R1",
        pose=(0.0, 0.0),
        explored_percent=10.0,
        mapped_obstacle_samples=5,
    )
    emitted_decision = trace.trace_decision(
        sim_time=1.0,
        robot_label="R1",
        kind="HOLD",
        reason="test",
        active_target=None,
        path_goal=None,
        pending_target=None,
    )

    assert emitted is False
    assert emitted_decision is False
    captured = capsys.readouterr()
    assert captured.out == "", "no ROBOT_TRACE output when unset/empty"


# ---------------------------------------------------------------------------
# B. Selected categories only.
# ---------------------------------------------------------------------------


def test_robot_trace_enables_selected_categories():
    trace = RobotTrace(env={"ROBOT_TRACE": "map,decision"})

    assert trace.is_enabled("map")
    assert trace.is_enabled("decision")
    assert not trace.is_enabled("route")
    assert not trace.is_enabled("frontier")
    assert not trace.is_enabled("obstacles")
    assert not trace.is_enabled("safety")


# ---------------------------------------------------------------------------
# C. "all" enables every category.
# ---------------------------------------------------------------------------


def test_robot_trace_all_enables_all_categories():
    trace = RobotTrace(env={"ROBOT_TRACE": "all"})

    for category in CATEGORIES:
        assert trace.is_enabled(category)


def test_parse_categories_ignores_unknown_tokens_and_whitespace():
    assert parse_categories(" Map , bogus ,ROUTE") == frozenset({"map", "route"})
    assert parse_categories(None) == frozenset()
    assert parse_categories("") == frozenset()


# ---------------------------------------------------------------------------
# D. Obstacle points grouped into compact line-like sections.
# ---------------------------------------------------------------------------


def test_obstacle_section_grouping_axis_aligned_points():
    points = [(1.5, 2.0), (1.5, 2.5), (1.5, 3.0), (2.0, 4.8), (2.5, 4.8)]

    sections = group_obstacle_points_into_sections(points)

    assert len(sections) == 2
    x_section = next(s for s in sections if s.axis == "x")
    y_section = next(s for s in sections if s.axis == "y")

    assert x_section.coordinate == 1.5
    assert x_section.span_min == 2.0
    assert x_section.span_max == 3.0
    assert x_section.count == 3

    assert y_section.coordinate == 4.8
    assert y_section.span_min == 2.0
    assert y_section.span_max == 2.5
    assert y_section.count == 2


def test_obstacle_section_grouping_never_drops_a_point():
    """A point with no matching x or y partner still becomes its own
    section -- the total point count is never silently reduced."""
    points = [(1.5, 2.0), (1.5, 2.5), (9.0, 9.0)]  # last point is isolated

    sections = group_obstacle_points_into_sections(points)

    assert sum(s.count for s in sections) == 3


# ---------------------------------------------------------------------------
# E. The obstacle-trace formatter limits displayed sections but still
#    reports the true total.
# ---------------------------------------------------------------------------


def test_obstacle_section_formatter_limits_output():
    points = []
    for i in range(10):
        # 10 distinct x-values, each with 2 points -> 10 sections.
        points.append((float(i), 0.0))
        points.append((float(i), 1.0))

    sections = group_obstacle_points_into_sections(points)
    assert len(sections) == 10

    line = format_obstacle_trace_line(
        sim_time=10.0,
        robot_label="R1",
        sample_points=len(points),
        sections=sections,
        max_sections=6,
    )

    assert "sections=10" in line
    assert f"sample_points={len(points)}" in line
    assert "(+4 more)" in line
    # Only 6 "n=" section entries should actually be listed.
    assert line.count("n=") == 6


# ---------------------------------------------------------------------------
# F. Map trace line contains the expected belief-map fields.
# ---------------------------------------------------------------------------


def test_trace_map_line_contains_belief_fields():
    line = format_map_trace_line(
        sim_time=50.9,
        robot_label="R1",
        pose=(-0.33, 0.94),
        explored_percent=14.0,
        mapped_obstacle_samples=830,
        free_unlocked=84,
        occupied_new=12,
        unknown_remaining=8421,
    )

    assert "[TRACE MAP" in line
    assert "t=50.9" in line
    assert "pose=(-0.33,0.94)" in line
    assert "free_unlocked=84" in line
    assert "occupied_new=12" in line
    assert "unknown_remaining=8421" in line
    assert "explored=14.0%" in line
    assert "mapped_obs=830" in line


def test_trace_map_line_handles_missing_fields_gracefully():
    line = format_map_trace_line(
        sim_time=1.0,
        robot_label="R1",
        pose=(0.0, 0.0),
        explored_percent=0.0,
        mapped_obstacle_samples=0,
    )

    assert "free_unlocked=n/a" in line
    assert "occupied_new=n/a" in line
    assert "unknown_remaining=n/a" in line


# ---------------------------------------------------------------------------
# G. Decision trace line contains active/path_goal/pending.
# ---------------------------------------------------------------------------


def test_trace_decision_line_contains_active_path_goal_pending():
    line = format_decision_trace_line(
        sim_time=51.0,
        robot_label="R1",
        kind="REPLAN_FOR_SAFETY",
        reason="active segment blocked",
        active_target=(1.75, 4.25),
        path_goal=(2.25, 3.75),
        pending_target=None,
    )

    assert "[TRACE DECISION" in line
    assert "kind=REPLAN_FOR_SAFETY" in line
    assert 'reason="active segment blocked"' in line
    assert "active=(1.75,4.25)" in line
    assert "path_goal=(2.25,3.75)" in line
    assert "pending=None" in line


# ---------------------------------------------------------------------------
# H. Route failure trace line contains reason and goal.
# ---------------------------------------------------------------------------


def test_trace_route_failure_line_contains_reason_and_goal():
    line = format_route_trace_line(
        sim_time=138.2,
        robot_label="R1",
        result="fail",
        start=(-0.21, 0.96),
        goal=(7.25, 3.75),
        reason="no_path",
        mapped_obstacle_count=1533,
    )

    assert "[TRACE ROUTE" in line
    assert "result=fail" in line
    assert "reason=no_path" in line
    assert "start=(-0.21,0.96)" in line
    assert "goal=(7.25,3.75)" in line
    assert "mapped_obs=1533" in line


# ---------------------------------------------------------------------------
# I. Repeated map events inside the throttle interval do not print
#    multiple lines.
# ---------------------------------------------------------------------------


def test_trace_throttles_periodic_map_events(capsys):
    trace = RobotTrace(env={"ROBOT_TRACE": "map"})
    kwargs = dict(
        robot_label="R1",
        pose=(0.0, 0.0),
        explored_percent=10.0,
        mapped_obstacle_samples=5,
    )

    assert trace.trace_map(sim_time=0.0, interval=1.0, **kwargs) is True
    assert trace.trace_map(sim_time=0.2, interval=1.0, **kwargs) is False
    assert trace.trace_map(sim_time=0.9, interval=1.0, **kwargs) is False

    captured = capsys.readouterr()
    assert captured.out.count("[TRACE MAP") == 1, (
        "repeated map trace calls inside the throttle interval must print only once"
    )

    assert trace.trace_map(sim_time=1.1, interval=1.0, **kwargs) is True
    captured = capsys.readouterr()
    assert captured.out.count("[TRACE MAP") == 1, "once the interval elapses, tracing resumes"


def test_trace_obstacles_only_emits_when_there_are_new_points(capsys):
    trace = RobotTrace(env={"ROBOT_TRACE": "obstacles"})

    assert trace.trace_obstacles(sim_time=1.0, robot_label="R1", points=[]) is False
    captured = capsys.readouterr()
    assert captured.out == "", "an empty sensor update must not print an obstacle trace line"

    assert trace.trace_obstacles(
        sim_time=1.0, robot_label="R1", points=[(1.0, 1.0), (1.0, 2.0)]
    ) is True
    captured = capsys.readouterr()
    assert "[TRACE OBS" in captured.out


def test_trace_methods_are_no_ops_for_disabled_categories(capsys):
    trace = RobotTrace(env={"ROBOT_TRACE": "map"})

    assert trace.trace_route(
        sim_time=1.0, robot_label="R1", result="ok", start=(0.0, 0.0), goal=(1.0, 1.0)
    ) is False
    assert trace.trace_frontier(sim_time=1.0, source="x", selected=None) is False
    assert trace.trace_safety(sim_time=1.0, robot_label="R1", goal=None, repair_status="throttled") is False

    captured = capsys.readouterr()
    assert captured.out == ""


# ---------------------------------------------------------------------------
# Windows-terminal safety: no non-ASCII characters, and _emit() never lets
# an encoding error escape even if some content somehow were non-ASCII.
# ---------------------------------------------------------------------------


def test_trace_safety_line_is_ascii_or_cp1252_safe():
    line = format_safety_trace_line(
        sim_time=100.9,
        robot_label="R1",
        goal=(7.25, 3.75),
        repair_status="throttled",
        min_clearance=0.38,
    )

    line.encode("ascii")  # must not raise
    line.encode("cp1252")  # must not raise
    assert "≈" not in line, "the Unicode approx sign must not appear in trace output"
    assert "min_clearance=~0.38" in line


def test_robot_trace_emit_does_not_raise_on_cp1252_stdout(monkeypatch):
    stream = _Cp1252Stream()
    monkeypatch.setattr(sys, "stdout", stream)

    trace = RobotTrace(env={"ROBOT_TRACE": "safety"})
    trace.trace_safety(
        sim_time=100.9,
        robot_label="R1",
        goal=(7.25, 3.75),
        repair_status="throttled",
        min_clearance=0.38,
    )

    assert any("[TRACE SAFETY" in s for s in stream.written)


def test_robot_trace_emit_sanitizes_unexpected_non_ascii_on_narrow_encoding(monkeypatch):
    """Belt-and-suspenders: even if some future/unexpected content did
    contain a non-cp1252 character, _emit() must sanitize (not crash)."""
    stream = _Cp1252Stream()
    monkeypatch.setattr(sys, "stdout", stream)

    trace = RobotTrace(env={"ROBOT_TRACE": "safety"})
    trace._emit("[TRACE TEST] contains ≈ unicode and 中文 too")

    assert stream.written, "must still write something -- never silently drop the line"
    "".join(stream.written).encode("cp1252")  # must not raise


def test_all_trace_formatters_avoid_problematic_unicode():
    sections = group_obstacle_points_into_sections([(1.0, 1.0), (1.0, 2.0)])
    lines = [
        format_map_trace_line(
            sim_time=1.0, robot_label="R1", pose=(0.0, 0.0), explored_percent=10.0, mapped_obstacle_samples=5
        ),
        format_obstacle_trace_line(sim_time=1.0, robot_label="R1", sample_points=2, sections=sections),
        format_decision_trace_line(
            sim_time=1.0,
            robot_label="R1",
            kind="HOLD",
            reason="test",
            active_target=None,
            path_goal=None,
            pending_target=None,
        ),
        format_frontier_trace_line(sim_time=1.0, source="x", selected=None),
        format_route_trace_line(
            sim_time=1.0, robot_label="R1", result="fail", start=(0.0, 0.0), goal=(1.0, 1.0), reason="no_path"
        ),
        format_safety_trace_line(
            sim_time=1.0, robot_label="R1", goal=(1.0, 1.0), repair_status="throttled", min_clearance=0.38
        ),
    ]

    for line in lines:
        line.encode("ascii")  # must not raise for any formatter


# ---------------------------------------------------------------------------
# Decision trace dedup/throttle: repeated HOLD is suppressed; REQUEST_PLAN/
# REPLAN_FOR_SAFETY/ACCEPT_PENDING_PATH/PREFETCH_NEXT_TARGET always show.
# ---------------------------------------------------------------------------


def test_trace_decision_dedups_repeated_hold(capsys):
    trace = RobotTrace(env={"ROBOT_TRACE": "decision"})
    kwargs = dict(
        robot_label="R1",
        kind="HOLD",
        reason="recovering after planner failure; retry cooldown active",
        active_target=None,
        path_goal=None,
        pending_target=None,
    )

    assert trace.trace_decision(sim_time=0.0, repeat_interval=2.0, **kwargs) is True
    assert trace.trace_decision(sim_time=0.1, repeat_interval=2.0, **kwargs) is False
    assert trace.trace_decision(sim_time=1.9, repeat_interval=2.0, **kwargs) is False

    captured = capsys.readouterr()
    assert captured.out.count("[TRACE DECISION") == 1, (
        "a repeated identical HOLD must print at most once per throttle interval"
    )

    assert trace.trace_decision(sim_time=2.1, repeat_interval=2.0, **kwargs) is True
    captured = capsys.readouterr()
    assert "repeated=2" in captured.out


def test_trace_decision_shows_different_hold_reason_immediately():
    trace = RobotTrace(env={"ROBOT_TRACE": "decision"})

    assert trace.trace_decision(
        sim_time=0.0,
        robot_label="R1",
        kind="HOLD",
        reason="recovering after planner failure; retry cooldown active",
        active_target=None,
        path_goal=None,
        pending_target=None,
    ) is True
    # A different HOLD reason is a different signature -- must not be
    # suppressed just because the previous HOLD was recent.
    assert trace.trace_decision(
        sim_time=0.05,
        robot_label="R1",
        kind="HOLD",
        reason="exploration exhausted: no reachable frontier candidates",
        active_target=None,
        path_goal=None,
        pending_target=None,
    ) is True


def test_trace_decision_always_shows_always_visible_kinds_even_when_repeated():
    trace = RobotTrace(env={"ROBOT_TRACE": "decision"})
    kwargs = dict(
        robot_label="R1",
        kind="REQUEST_PLAN",
        reason="frontier reached; requesting next frontier",
        active_target=(1.0, 1.0),
        path_goal=(2.0, 2.0),
        pending_target=None,
    )

    assert trace.trace_decision(sim_time=0.0, **kwargs) is True
    assert trace.trace_decision(sim_time=0.05, **kwargs) is True
    assert trace.trace_decision(sim_time=0.1, **kwargs) is True


# ---------------------------------------------------------------------------
# Frontier trace dedup/throttle: repeated "nothing selected" (exhaustion)
# is suppressed; a genuine selection always shows.
# ---------------------------------------------------------------------------


def test_trace_frontier_dedups_repeated_exhaustion(capsys):
    trace = RobotTrace(env={"ROBOT_TRACE": "frontier"})
    kwargs = dict(source="map-wide-fallback", selected=None, generated=3)

    assert trace.trace_frontier(sim_time=0.0, repeat_interval=2.0, **kwargs) is True
    assert trace.trace_frontier(sim_time=0.5, repeat_interval=2.0, **kwargs) is False
    assert trace.trace_frontier(sim_time=1.9, repeat_interval=2.0, **kwargs) is False

    captured = capsys.readouterr()
    assert captured.out.count("[TRACE FRONTIER") == 1

    assert trace.trace_frontier(sim_time=2.1, repeat_interval=2.0, **kwargs) is True
    captured = capsys.readouterr()
    assert "repeated=2" in captured.out


def test_trace_frontier_always_shows_a_genuine_selection():
    trace = RobotTrace(env={"ROBOT_TRACE": "frontier"})

    assert trace.trace_frontier(sim_time=0.0, source="x", selected=(1.0, 1.0)) is True
    assert trace.trace_frontier(sim_time=0.05, source="x", selected=(1.0, 1.0)) is True
    assert trace.trace_frontier(sim_time=0.1, source="x", selected=(1.0, 1.0)) is True


# ---------------------------------------------------------------------------
# A-H. Belief-trace artifact files are generated by default, independent of
# ROBOT_TRACE (which only ever controls terminal [TRACE ...] printing).
# RobotTrace.start_run() is the explicit lifecycle hook engine.py calls at
# Start/Restart Simulation -- never __init__, never a trace_*() call -- so
# every test above this section (none of which calls start_run()) continues
# to touch zero real files regardless of what ROBOT_TRACE_DIR/env it uses.
# ---------------------------------------------------------------------------


def test_belief_trace_artifacts_enabled_by_default(tmp_path):
    """No env vars at all (matching `python .\\main.py` with nothing set) ->
    starting a run must still create the artifact directory + files."""
    trace = RobotTrace(env={"ROBOT_TRACE_DIR": str(tmp_path)})

    run_dir = trace.start_run()

    assert run_dir is not None
    assert trace.writer is not None
    assert run_dir.exists()
    assert (run_dir / "belief_events.jsonl").exists()
    assert (run_dir / "route_events.csv").exists()


def test_belief_trace_artifacts_can_be_disabled(tmp_path):
    trace = RobotTrace(env={"ROBOT_TRACE_DIR": str(tmp_path), "BELIEF_TRACE_ARTIFACTS": "0"})

    run_dir = trace.start_run()

    assert run_dir is None
    assert trace.writer is None
    assert list(tmp_path.iterdir()) == [], "BELIEF_TRACE_ARTIFACTS=0 must create no files at all"


def test_robot_trace_env_only_controls_stdout_categories(tmp_path, capsys):
    """ROBOT_TRACE unset: the file sink still exists (artifacts are always
    on by default), but routine [TRACE ...] stdout stays silent."""
    trace = RobotTrace(env={"ROBOT_TRACE_DIR": str(tmp_path)})
    trace.start_run()
    capsys.readouterr()

    assert trace.writer is not None
    trace.trace_route(
        sim_time=1.0, robot_label="R1", result="ok", start=(0.0, 0.0), goal=(1.0, 1.0),
    )

    captured = capsys.readouterr()
    assert "[TRACE" not in captured.out, (
        "ROBOT_TRACE unset must never print routine [TRACE ...] terminal lines"
    )

    route_csv = (trace.file_output_dir / "route_events.csv").read_text(encoding="utf-8")
    assert "R1" in route_csv, (
        "artifact files must still be populated even when ROBOT_TRACE is unset"
    )


def test_robot_trace_env_enabled_prints_terminal_trace_and_keeps_files(tmp_path, capsys):
    """ROBOT_TRACE=map,route: terminal trace prints those categories AND
    artifact files keep being written -- the two concerns are independent."""
    trace = RobotTrace(env={"ROBOT_TRACE": "route", "ROBOT_TRACE_DIR": str(tmp_path)})
    trace.start_run()
    capsys.readouterr()

    assert trace.trace_route(
        sim_time=1.0, robot_label="R1", result="ok", start=(0.0, 0.0), goal=(1.0, 1.0),
    ) is True

    captured = capsys.readouterr()
    assert "[TRACE ROUTE" in captured.out

    route_csv = (trace.file_output_dir / "route_events.csv").read_text(encoding="utf-8")
    assert "R1" in route_csv and "ok" in route_csv


def test_start_simulation_creates_new_trace_directory(tmp_path):
    """Represents what engine.py's start_simulation() does: call
    start_run() once when a run begins."""
    trace = RobotTrace(env={"ROBOT_TRACE_DIR": str(tmp_path)})

    run_dir = trace.start_run()

    assert run_dir is not None
    assert run_dir.parent == tmp_path
    assert trace.file_output_dir == run_dir


def test_restart_simulation_creates_new_trace_directory(tmp_path):
    """Represents what engine.py's restart_simulation() does: call
    start_run() again -- each call must produce a brand-new directory,
    distinct from any earlier one, never reusing the same writer/files."""
    trace = RobotTrace(env={"ROBOT_TRACE_DIR": str(tmp_path)})

    first_run_dir = trace.start_run()
    first_writer = trace.writer
    # Even within the same wall-clock second, make_run_directory()'s own
    # collision-suffix logic (belief_trace_..._2, _3, ...) guarantees a
    # distinct directory -- no special-casing needed here.
    second_run_dir = trace.start_run()

    assert second_run_dir is not None
    assert second_run_dir != first_run_dir
    assert trace.writer is not first_writer
    assert first_run_dir.exists() and second_run_dir.exists()


def test_trace_directory_is_repo_root_relative_absolute():
    """With no ROBOT_TRACE_DIR override, the default output directory must
    be an absolute, repo-root-based path -- never dependent on cwd."""
    from robotics_sim.simulation.belief_trace_writer import REPO_ROOT

    trace = RobotTrace(env={})
    run_dir = trace.start_run()
    try:
        assert run_dir is not None
        assert run_dir.is_absolute()
        assert run_dir.parent == REPO_ROOT / "runs" / "debug"
    finally:
        if run_dir is not None and run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)


def test_files_exist_immediately_after_simulation_start(tmp_path):
    trace = RobotTrace(env={"ROBOT_TRACE_DIR": str(tmp_path)})

    run_dir = trace.start_run()

    assert run_dir is not None
    for name in (
        "belief_events.jsonl", "run_summary.json", "obstacle_sections.csv",
        "route_events.csv", "frontier_events.csv", "decision_events.csv",
    ):
        assert (run_dir / name).exists(), f"{name} must exist immediately, before any trace_*() call"
