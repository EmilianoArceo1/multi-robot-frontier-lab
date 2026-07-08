"""
Tests for robotics_sim.simulation.telemetry -- structured, throttled console
telemetry that replaces the noisiest raw engine.py console spam:

    - per-tick "R1 move @ t=..." traces
    - repeated "Mapped N new obstacle boundary sample(s)" lines
    - verbose route-assignment/failure messages

Design boundary under test: engine.py decides WHEN something happened (it
would call report_*() every tick/every new sample/every route event,
unconditionally); TelemetryLogger decides HOW OFTEN that actually produces
a console line and how it is formatted. These tests exercise
TelemetryLogger directly with a plain list-based sink -- no Qt, no canvas,
no engine/GUI instantiation.
"""
from __future__ import annotations

from robotics_sim.simulation.telemetry import TelemetryLogger


def _make_logger(level: str = "normal", **kwargs) -> tuple[TelemetryLogger, list[str]]:
    lines: list[str] = []
    logger = TelemetryLogger(level=level, sink=lines.append, **kwargs)
    return logger, lines


# ---------------------------------------------------------------------------
# 1. [STATE] snapshots are throttled by simulation time.
# ---------------------------------------------------------------------------


def test_state_snapshot_is_throttled_by_sim_time():
    logger, lines = _make_logger(state_interval=1.0)

    def _report(sim_time: float) -> bool:
        return logger.report_state(
            sim_time=sim_time,
            speed_multiplier=1.0,
            robot_label="R1",
            pos=(1.0, 2.0),
            theta=0.0,
            v=0.3,
            state="TRACK",
            target=(3.0, 4.0),
            path_goal=(5.0, 6.0),
            wp_index=1,
            wp_total=2,
            mapped_obstacle_count=10,
            explored_percent=12.3,
        )

    assert _report(0.0) is True, "the first snapshot must always be emitted"
    assert _report(0.1) is False, "a snapshot well within the interval must be suppressed"
    assert _report(0.9) is False, "still within the 1.0s interval"
    assert len(lines) == 1

    assert _report(1.05) is True, "a snapshot past the interval must be emitted"
    assert len(lines) == 2

    assert "[STATE" in lines[0]
    assert "pos=(1.00,2.00)" in lines[0]
    assert "target=(3.00,4.00)" in lines[0]
    assert "path_goal=(5.00,6.00)" in lines[0]
    assert "wp=1/2" in lines[0]
    assert "mapped_obs=10" in lines[0]
    assert "explored=12.3%" in lines[0]


# ---------------------------------------------------------------------------
# 2. [MAP] updates aggregate small sample batches and report bbox/centroid.
# ---------------------------------------------------------------------------


def test_map_update_aggregates_samples_and_reports_bbox():
    logger, lines = _make_logger(map_flush_interval=1.0)

    # Several small batches inside the same flush window must not each
    # produce their own line.
    assert logger.report_map_update(
        sim_time=0.0, new_points=[(3.1, -4.9)], total_count=101,
        route_affected=False, explored_percent=10.0,
    ) is False
    assert logger.report_map_update(
        sim_time=0.3, new_points=[(5.8, -3.2), (4.0, -4.0)], total_count=103,
        route_affected=False, explored_percent=10.5,
    ) is False
    assert lines == [], "small batches within the flush window must not emit yet"

    # Past the flush interval, everything buffered so far is aggregated
    # into exactly one line.
    assert logger.report_map_update(
        sim_time=1.2, new_points=[], total_count=103,
        route_affected=False, explored_percent=11.0,
    ) is True
    assert len(lines) == 1

    line = lines[0]
    assert line.startswith("[MAP t=1.2s]")
    assert "+3 obstacle_samples" in line, "must aggregate the count across all buffered batches"
    assert "total=103" in line
    assert "bbox=x[3.1,5.8] y[-4.9,-3.2]" in line
    assert "centroid=" in line
    assert "route_affected=no" in line
    assert "explored=11.0%" in line
    # Only summary statistics -- never a raw coordinate list.
    assert "-4.9)" not in line.replace("y[-4.9,-3.2]", "")


def test_map_update_with_route_affected_flushes_immediately():
    logger, lines = _make_logger(map_flush_interval=100.0)  # would not flush otherwise

    emitted = logger.report_map_update(
        sim_time=29.3, new_points=[(3.6, -4.5), (4.8, -4.1)], total_count=711,
        route_affected=True, explored_percent=20.0,
    )
    assert emitted is True, "route_affected=True must flush immediately regardless of the interval"
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[MAP t=29.3s]")
    assert "route_affected=yes" in line
    assert "+2 obstacle_samples" in line
    assert "total=711" in line
    assert "bbox=x[3.6,4.8] y[-4.5,-4.1]" in line


# ---------------------------------------------------------------------------
# 3. [ROUTE fail] summaries include the attempted target and mapped count.
# ---------------------------------------------------------------------------


def test_route_failure_summary_includes_attempted_target_and_mapped_obstacle_count():
    logger, lines = _make_logger()

    logger.report_route_failure(
        robot_label="R1",
        start_xy=(4.64, -4.27),
        attempted_target=(3.25, 0.75),
        reason="no path found",
        planner_type="A*",
        mapped_obstacle_count=692,
    )

    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[ROUTE fail]")
    assert "start=(4.64,-4.27)" in line
    assert "attempted=(3.25,0.75)" in line
    assert "reason=no_path" in line
    assert "planner=A*" in line
    assert "mapped_obs=692" in line


def test_route_failure_summary_visible_even_at_quiet_level():
    logger, lines = _make_logger(level="quiet")
    logger.report_route_failure(
        robot_label="R1", start_xy=(0.0, 0.0), attempted_target=(1.0, 1.0),
        reason="no path found", planner_type="A*", mapped_obstacle_count=5,
    )
    assert len(lines) == 1, "a planner failure must stay visible even at quiet level"


# ---------------------------------------------------------------------------
# 4. Normal level suppresses detailed per-frame move logs.
# ---------------------------------------------------------------------------


def test_normal_level_suppresses_detailed_move_logs():
    logger, lines = _make_logger(level="normal")

    emitted = logger.report_move(
        sim_time=12.34, robot_label="R1", pos=(1.0, 2.0), theta=0.5, v=0.4,
        target=(3.0, 4.0), control_text="u=(0.100, 0.000)",
    )

    assert emitted is False
    assert lines == [], "normal level must not print per-frame R1 move traces"


# ---------------------------------------------------------------------------
# 5. Debug level allows detailed per-frame move logs.
# ---------------------------------------------------------------------------


def test_debug_level_allows_detailed_move_logs():
    logger, lines = _make_logger(level="debug")

    emitted = logger.report_move(
        sim_time=12.34, robot_label="R1", pos=(1.0, 2.0), theta=0.5, v=0.4,
        target=(3.0, 4.0), control_text="u=(0.100, 0.000)",
    )

    assert emitted is True
    assert len(lines) == 1
    line = lines[0]
    assert "R1 move @ t=12.34s" in line
    assert "pos=(1.00, 2.00)" in line
    assert "target=(3.00,4.00)" in line
    assert "u=(0.100, 0.000)" in line


# ---------------------------------------------------------------------------
# Additional coverage: NAV / FRONTIER / WARN formatting and quiet-level gating.
# ---------------------------------------------------------------------------


def test_nav_decision_is_compact_and_quoted():
    logger, lines = _make_logger()
    logger.report_nav_decision(
        sim_time=0.0,
        robot_label="R1",
        kind="HOLD",
        reason="exploration exhausted: no reachable frontier candidates",
        active_target=None,
        path_goal=None,
        pending_target=None,
    )
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[NAV] R1 kind=HOLD")
    assert 'reason="exploration exhausted: no reachable frontier candidates"' in line
    assert "active=None" in line
    assert "path_goal=None" in line
    assert "pending=None" in line


def test_frontier_selection_parses_existing_debug_counts_from_reason():
    logger, lines = _make_logger()
    logger.report_frontier_selection(
        robot_label="R1",
        success=True,
        selected=(7.25, 3.75),
        reason=(
            "FoV-aware directional frontier: selected best FoV-aware target; "
            "kind=frontier, size=4, score=-0.56; generated=7, "
            "excluded_recently_failed=0, filtered_unreachable=2, selected=(7.25, 3.75)"
        ),
    )
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("[FRONTIER] R1")
    assert "generated=7" in line
    assert "filtered_unreachable=2" in line
    assert "failed_recent=0" in line
    assert "selected=(7.25,3.75)" in line


def test_quiet_level_suppresses_state_map_and_nav_but_not_route_failure():
    logger, lines = _make_logger(level="quiet")

    logger.report_state(
        sim_time=0.0, speed_multiplier=1.0, robot_label="R1", pos=(0, 0), theta=0.0,
        v=0.0, state="TRACK", target=None, path_goal=None, wp_index=0, wp_total=0,
        mapped_obstacle_count=0, explored_percent=0.0, force=True,
    )
    logger.report_map_update(
        sim_time=0.0, new_points=[(1.0, 1.0)], total_count=1,
        route_affected=False, explored_percent=0.0, force=True,
    )
    logger.report_nav_decision(
        sim_time=0.0, robot_label="R1", kind="HOLD", reason="x",
        active_target=None, path_goal=None, pending_target=None,
    )
    assert lines == [], "quiet level must suppress STATE, MAP, and NAV lines"

    logger.report_route_failure(
        robot_label="R1", start_xy=(0.0, 0.0), attempted_target=(1.0, 1.0),
        reason="no path found", planner_type="A*", mapped_obstacle_count=1,
    )
    assert len(lines) == 1, "route failures must still surface even at quiet level"


# ---------------------------------------------------------------------------
# Polish pass: verbose legacy planner detail is debug-only.
# ---------------------------------------------------------------------------


def test_verbose_planner_detail_is_debug_only_or_suppressed_in_normal_mode():
    verbose_line = (
        "Planner: A* / Line of sight grid-safe + FoV-aware directional frontier. "
        "selected best FoV-aware target. path found with A*. Mapped points: 413."
    )

    normal_logger, normal_lines = _make_logger(level="normal")
    emitted_normal = normal_logger.debug(verbose_line)
    assert emitted_normal is False
    assert normal_lines == [], "normal mode must not print the old verbose planner detail line"

    debug_logger, debug_lines = _make_logger(level="debug")
    emitted_debug = debug_logger.debug(verbose_line)
    assert emitted_debug is True
    assert debug_lines == [verbose_line], "debug mode may still print the verbose detail line"

    quiet_logger, quiet_lines = _make_logger(level="quiet")
    assert quiet_logger.debug(verbose_line) is False
    assert quiet_lines == []


# ---------------------------------------------------------------------------
# Polish pass: repeated identical NAV decisions are throttled/deduplicated.
# ---------------------------------------------------------------------------


def test_repeated_identical_nav_decisions_are_throttled():
    logger, lines = _make_logger(nav_repeat_interval=5.0)

    def _report(sim_time: float) -> bool:
        return logger.report_nav_decision(
            sim_time=sim_time,
            robot_label="R1",
            kind="HOLD",
            reason="exploration exhausted: no reachable frontier candidates",
            active_target=None,
            path_goal=None,
            pending_target=None,
        )

    assert _report(0.0) is True, "the first occurrence must be printed immediately"
    assert len(lines) == 1
    assert "repeated=" not in lines[0]

    # Many identical calls within the interval must all be suppressed.
    for t in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        assert _report(t) is False
    assert len(lines) == 1, "identical NAV lines within the interval must not spam the console"

    # Once the interval elapses, a compact repeated-count summary is printed.
    assert _report(5.5) is True
    assert len(lines) == 2
    assert "repeated=6" in lines[1], "must report how many identical calls were suppressed"
    assert lines[1].startswith('[NAV] R1 kind=HOLD reason="exploration exhausted')

    # A distinct decision (different kind/reason/targets) must be printed
    # immediately, not suppressed by the previous streak.
    assert logger.report_nav_decision(
        sim_time=5.6,
        robot_label="R1",
        kind="REQUEST_PLAN",
        reason="recovered after planner failure; requesting fresh frontier",
        active_target=None,
        path_goal=(2.0, 3.0),
        pending_target=None,
    ) is True
    assert len(lines) == 3
    assert "repeated=" not in lines[2]


def test_repeated_nav_signature_includes_robot_and_all_target_fields():
    """A change in ANY signature field (robot, kind, reason, active/path/pending
    target) must be treated as a new, immediately-printed decision, not a
    repeat of the previous one."""
    logger, lines = _make_logger(nav_repeat_interval=5.0)
    base = dict(
        sim_time=0.0, robot_label="R1", kind="HOLD", reason="x",
        active_target=None, path_goal=None, pending_target=None,
    )
    logger.report_nav_decision(**base)
    assert len(lines) == 1

    variant = dict(base)
    variant["path_goal"] = (1.0, 1.0)
    variant["sim_time"] = 0.1
    assert logger.report_nav_decision(**variant) is True
    assert len(lines) == 2, "a different path_goal must not be treated as a repeat"


# ---------------------------------------------------------------------------
# Polish pass: STATE must not report a hold position as `target`.
# ---------------------------------------------------------------------------


def test_state_snapshot_does_not_report_hold_position_as_target_when_path_goal_is_none():
    logger, lines = _make_logger()

    logger.report_state(
        sim_time=0.0,
        speed_multiplier=0.83,
        robot_label="R1",
        pos=(3.13, -4.26),
        theta=-0.9,
        v=0.0,
        state="HOLD",
        target=None,
        path_goal=None,
        hold_pos=(3.13, -4.26),
        wp_index=0,
        wp_total=0,
        mapped_obstacle_count=692,
        explored_percent=25.0,
    )

    assert len(lines) == 1
    line = lines[0]
    assert "target=None" in line
    assert "path_goal=None" in line
    assert "wp=0/0" in line
    assert "hold_pos=(3.13,-4.26)" in line
    # The hold position must never be printed under the `target=` key.
    assert "target=(3.13,-4.26)" not in line


def test_state_snapshot_reports_real_target_when_path_goal_is_set():
    """Confirming the normal (non-holding) case is unaffected: a real active
    route still reports its current waypoint as `target`."""
    logger, lines = _make_logger()

    logger.report_state(
        sim_time=0.0,
        speed_multiplier=1.0,
        robot_label="R1",
        pos=(3.13, -4.26),
        theta=-0.9,
        v=0.3,
        state="TRACK",
        target=(4.75, -4.25),
        path_goal=(7.25, 3.75),
        wp_index=1,
        wp_total=2,
        mapped_obstacle_count=692,
        explored_percent=25.0,
    )

    assert len(lines) == 1
    line = lines[0]
    assert "target=(4.75,-4.25)" in line
    assert "path_goal=(7.25,3.75)" in line
    assert "wp=1/2" in line
    assert "hold_pos=" not in line
