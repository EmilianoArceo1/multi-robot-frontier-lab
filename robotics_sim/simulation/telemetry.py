"""
Structured, throttled console telemetry for the simulator.

Design boundary (see robotics_sim/simulation/engine.py):
    engine.py decides WHEN something happened -- it calls a report_*()
    method every time a relevant event occurs (a tick passes, new obstacle
    samples are mapped, a route is assigned or fails, a NavigationDecision
    is produced), unconditionally and without its own throttling logic.

    telemetry.py decides HOW that is summarized, throttled, aggregated, and
    formatted into a compact, readable line. All buffering/throttling state
    lives here, not in engine.py.

This module has no Qt, canvas, or algorithm dependencies. The console sink
is injected as a plain callable (str) -> None, so engine.py wires it to
whatever the GUI actually uses (MainWindow.log_console_message), and this
module stays fully unit-testable without a GUI.

Categories: [STATE] [MAP] [ROUTE] [NAV] [FRONTIER] [WARN]
Verbosity levels: "quiet" < "normal" < "debug" (default: "normal")
"""
from __future__ import annotations

import re
import time
from typing import Callable, Iterable

Point2D = tuple[float, float]

QUIET = "quiet"
NORMAL = "normal"
DEBUG = "debug"

_LEVEL_ORDER = {QUIET: 0, NORMAL: 1, DEBUG: 2}

# Default cadences. Callers may invoke report_state()/report_map_update()
# every tick -- these intervals are what actually gate emission.
DEFAULT_STATE_INTERVAL_S = 1.0
DEFAULT_MAP_FLUSH_INTERVAL_S = 1.0
DEFAULT_NAV_REPEAT_INTERVAL_S = 5.0

# Known planner-failure reason substrings -> short, stable slugs for
# compact [ROUTE fail] lines. Falls back to a sanitized version of the raw
# reason when nothing matches, so a new/unlisted planner_registry reason
# string never produces a blank or crashing log line.
_ROUTE_FAILURE_REASON_SLUGS: tuple[tuple[str, str], ...] = (
    ("no path found", "no_path"),
    ("goal cell is occupied", "goal_occupied"),
    ("goal cell is not traversable", "goal_blocked"),
    ("start cell is not traversable", "start_blocked"),
    ("start is outside", "start_out_of_bounds"),
    ("goal is outside", "goal_out_of_bounds"),
    ("no valid frontier candidates", "no_frontier_candidates"),
    ("no reachable frontier candidates", "no_reachable_candidates"),
    ("no candidate path was valid", "no_candidate_path"),
    ("repeated safety replan", "repeated_safety_replan"),
    ("first segment blocked", "first_segment_blocked"),
)


def _fmt_point(point: Point2D | None) -> str:
    if point is None:
        return "None"
    try:
        return f"({float(point[0]):.2f},{float(point[1]):.2f})"
    except (TypeError, ValueError, IndexError):
        return "None"


def _slug_reason(reason: str) -> str:
    """Compact, stable identifier for a planner-failure reason string."""
    text = str(reason or "").strip().lower()
    for needle, slug in _ROUTE_FAILURE_REASON_SLUGS:
        if needle in text:
            return slug
    if not text:
        return "unknown"
    # Fallback: first few words, sanitized, so unlisted reasons stay short
    # and log-line-safe instead of dumping a full sentence inline.
    words = re.findall(r"[a-z0-9]+", text)[:4]
    return "_".join(words) if words else "unknown"


def _slug_camel(text: str) -> str:
    """'Line of sight grid-safe' -> 'LineOfSightGridSafe'."""
    words = re.findall(r"[a-zA-Z0-9]+", str(text or ""))
    return "".join(word.capitalize() for word in words) or "--"


def _round_point(point: Point2D | None) -> tuple[float, float] | None:
    if point is None:
        return None
    try:
        return (round(float(point[0]), 2), round(float(point[1]), 2))
    except (TypeError, ValueError, IndexError):
        return None


def _bbox_and_centroid(points: Iterable[Point2D]) -> tuple[str, tuple[float, float]]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    bbox = f"x[{min(xs):.1f},{max(xs):.1f}] y[{min(ys):.1f},{max(ys):.1f}]"
    centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
    return bbox, centroid


class TelemetryLogger:
    """Structured, throttled console telemetry.

    One instance per simulation run. Safe to call every tick -- each
    report_*() method internally decides whether/how to actually emit a
    line, based on `level` and its own aggregation state.
    """

    def __init__(
        self,
        *,
        level: str = NORMAL,
        sink: "Callable[[str], None] | None" = None,
        state_interval: float = DEFAULT_STATE_INTERVAL_S,
        map_flush_interval: float = DEFAULT_MAP_FLUSH_INTERVAL_S,
        nav_repeat_interval: float = DEFAULT_NAV_REPEAT_INTERVAL_S,
    ) -> None:
        self.level = level if level in _LEVEL_ORDER else NORMAL
        self._sink = sink if sink is not None else (lambda line: None)
        self.state_interval = float(state_interval)
        self.map_flush_interval = float(map_flush_interval)
        self.nav_repeat_interval = float(nav_repeat_interval)

        self._last_state_time = -1.0e9
        self._pending_map_points: list[Point2D] = []
        # None means "no aggregation window open yet" -- the window starts
        # at the first buffered sample rather than flushing immediately,
        # which a -1e9 sentinel would otherwise trigger on the very first call.
        self._last_map_flush_time: float | None = None

        # NAV dedup/throttle state -- see report_nav_decision().
        self._last_nav_signature: tuple | None = None
        self._last_nav_emit_time = -1.0e9
        self._nav_repeat_count = 0

    # ------------------------------------------------------------------ setup

    def set_level(self, level: str) -> None:
        self.level = level if level in _LEVEL_ORDER else self.level

    def set_sink(self, sink: "Callable[[str], None]") -> None:
        self._sink = sink

    def _at_least(self, level: str) -> bool:
        return _LEVEL_ORDER.get(self.level, 1) >= _LEVEL_ORDER[level]

    def _emit(self, line: str) -> None:
        self._sink(line)

    # ------------------------------------------------------------------ [STATE]

    def report_state(
        self,
        *,
        sim_time: float,
        wall_time: float | None = None,
        speed_multiplier: float,
        robot_label: str,
        pos: Point2D,
        theta: float,
        v: float,
        state: str,
        target: Point2D | None,
        path_goal: Point2D | None,
        wp_index: int,
        wp_total: int,
        mapped_obstacle_count: int,
        explored_percent: float,
        hold_pos: Point2D | None = None,
        force: bool = False,
    ) -> bool:
        """Periodic robot/exploration snapshot. Throttled to state_interval
        seconds of *simulation* time regardless of how often this is called.

        hold_pos: pass this (with target=None) when the robot has no active
        exploration route (path_goal is None) and is simply holding at its
        current position -- callers must not report that hold position as
        `target`, which would misleadingly look like a real destination.

        Returns True when a line was actually emitted (useful for tests).
        """
        if self.level == QUIET:
            return False
        if not force and (float(sim_time) - self._last_state_time) < self.state_interval:
            return False
        self._last_state_time = float(sim_time)

        hold_pos_text = f"hold_pos={_fmt_point(hold_pos)} " if hold_pos is not None else ""
        self._emit(
            f"[STATE t={float(sim_time):.1f}s speed=x{float(speed_multiplier):.2f}] {robot_label} "
            f"pos=({float(pos[0]):.2f},{float(pos[1]):.2f}) theta={float(theta):.2f} v={float(v):.2f} "
            f"state={state} {hold_pos_text}target={_fmt_point(target)} path_goal={_fmt_point(path_goal)} "
            f"wp={int(wp_index)}/{int(wp_total)} mapped_obs={int(mapped_obstacle_count)} "
            f"explored={float(explored_percent):.1f}%"
        )
        return True

    # ------------------------------------------------------------------ move (debug only)

    def report_move(
        self,
        *,
        sim_time: float,
        robot_label: str,
        pos: Point2D,
        theta: float,
        v: float,
        target: Point2D | None,
        control_text: str = "",
    ) -> bool:
        """Detailed per-tick movement trace. Only ever emitted at DEBUG
        level -- this is exactly the "R1 move @ t=..." spam that must
        disappear from quiet/normal consoles."""
        if not self._at_least(DEBUG):
            return False
        self._emit(
            f"{robot_label} move @ t={float(sim_time):.2f}s: pos=({float(pos[0]):.2f}, {float(pos[1]):.2f}), "
            f"theta={float(theta):.3f} rad, v={float(v):.3f} m/s, target={_fmt_point(target)}, {control_text}"
        )
        return True

    # ------------------------------------------------------------------ [MAP]

    def report_map_update(
        self,
        *,
        sim_time: float,
        new_points: list[Point2D],
        total_count: int,
        route_affected: bool,
        explored_percent: float,
        force: bool = False,
    ) -> bool:
        """Report newly-mapped obstacle samples.

        Buffers points internally and only emits an aggregated [MAP] line
        (count/bbox/centroid, never individual coordinates) once per
        map_flush_interval seconds -- except when route_affected is True,
        which flushes immediately regardless of the interval, since that is
        safety-relevant and must stay visible right away.
        """
        if self.level == QUIET:
            return False
        if new_points:
            self._pending_map_points.extend(new_points)
        if not self._pending_map_points:
            return False

        if self._last_map_flush_time is None:
            # Open the aggregation window here instead of flushing
            # immediately (unless this very first sample is itself urgent).
            self._last_map_flush_time = float(sim_time)
            if not (force or route_affected):
                return False

        should_flush = force or route_affected or (
            (float(sim_time) - self._last_map_flush_time) >= self.map_flush_interval
        )
        if not should_flush:
            return False

        points = self._pending_map_points
        self._pending_map_points = []
        self._last_map_flush_time = float(sim_time)
        bbox, centroid = _bbox_and_centroid(points)

        if route_affected:
            self._emit(
                f"[MAP t={float(sim_time):.1f}s] route_affected=yes +{len(points)} obstacle_samples "
                f"total={int(total_count)} bbox={bbox}"
            )
        else:
            self._emit(
                f"[MAP t={float(sim_time):.1f}s] +{len(points)} obstacle_samples total={int(total_count)} "
                f"bbox={bbox} centroid=({centroid[0]:.1f},{centroid[1]:.1f}) route_affected=no "
                f"explored={float(explored_percent):.1f}%"
            )
        return True

    # ------------------------------------------------------------------ [ROUTE]

    def report_route_success(
        self,
        *,
        robot_label: str,
        start_xy: Point2D,
        goal_xy: Point2D | None,
        wp_count: int,
        planner_type: str,
        simplifier: str,
        length: float,
        mapped_obstacle_count: int,
    ) -> None:
        if self.level == QUIET:
            return
        self._emit(
            f"[ROUTE ok] {robot_label} start={_fmt_point(start_xy)} goal={_fmt_point(goal_xy)} "
            f"wp={int(wp_count)} planner={planner_type} simplifier={_slug_camel(simplifier)} "
            f"length={float(length):.2f} mapped_obs={int(mapped_obstacle_count)}"
        )

    def report_route_failure(
        self,
        *,
        robot_label: str,
        start_xy: Point2D,
        attempted_target: Point2D | None,
        reason: str,
        planner_type: str,
        mapped_obstacle_count: int,
    ) -> None:
        """Route/plan failure summary.

        Always visible (even at QUIET, short of disabling telemetry
        entirely) since a planner failure is safety/behavior relevant, not
        routine noise -- mirrors requirement to keep "planner failed"
        visible regardless of verbosity.
        """
        self._emit(
            f"[ROUTE fail] {robot_label} start={_fmt_point(start_xy)} "
            f"attempted={_fmt_point(attempted_target)} reason={_slug_reason(reason)} "
            f"planner={planner_type} mapped_obs={int(mapped_obstacle_count)}"
        )

    # ------------------------------------------------------------------ [NAV]

    def report_nav_decision(
        self,
        *,
        sim_time: float,
        robot_label: str,
        kind: str,
        reason: str,
        active_target: Point2D | None,
        path_goal: Point2D | None,
        pending_target: Point2D | None,
    ) -> bool:
        """NavigationDecision summary.

        Deduplicated: an identical (robot_label, kind, reason, active_target,
        path_goal, pending_target) signature to the last one emitted is
        suppressed for nav_repeat_interval seconds of simulation time -- this
        is what stops a stable HOLD (e.g. "exploration exhausted") from
        reprinting the same line every tick. A distinct signature is always
        printed immediately. Once the interval elapses while the signature
        is still unchanged, a compact "repeated=N" summary is printed for
        however many identical calls were suppressed in between.

        Returns True when a line was actually emitted (useful for tests).
        """
        if self.level == QUIET:
            return False

        signature = (
            robot_label,
            kind,
            str(reason),
            _round_point(active_target),
            _round_point(path_goal),
            _round_point(pending_target),
        )

        if signature != self._last_nav_signature:
            self._last_nav_signature = signature
            self._last_nav_emit_time = float(sim_time)
            self._nav_repeat_count = 0
            self._emit_nav_line(robot_label, kind, reason, active_target, path_goal, pending_target)
            return True

        if (float(sim_time) - self._last_nav_emit_time) < self.nav_repeat_interval:
            self._nav_repeat_count += 1
            return False

        repeated = self._nav_repeat_count
        self._last_nav_emit_time = float(sim_time)
        self._nav_repeat_count = 0
        self._emit_nav_line(
            robot_label, kind, reason, active_target, path_goal, pending_target, repeated=repeated
        )
        return True

    def _emit_nav_line(
        self,
        robot_label: str,
        kind: str,
        reason: str,
        active_target: Point2D | None,
        path_goal: Point2D | None,
        pending_target: Point2D | None,
        *,
        repeated: int | None = None,
    ) -> None:
        safe_reason = str(reason).replace('"', "'")
        suffix = f" repeated={repeated}" if repeated else ""
        self._emit(
            f'[NAV] {robot_label} kind={kind} reason="{safe_reason}" '
            f"active={_fmt_point(active_target)} path_goal={_fmt_point(path_goal)} "
            f"pending={_fmt_point(pending_target)}{suffix}"
        )

    # ------------------------------------------------------------------ [FRONTIER]

    _FRONTIER_COUNT_PATTERN = re.compile(
        r"generated=(?P<generated>\d+),\s*excluded_recently_failed=(?P<excluded>\d+),\s*"
        r"filtered_unreachable=(?P<filtered>\d+)"
    )

    def report_frontier_selection(
        self,
        *,
        robot_label: str,
        success: bool,
        selected: Point2D | None,
        reason: str = "",
        score: float | None = None,
        candidate_count: int | None = None,
    ) -> None:
        """Compact frontier-selection summary.

        generated/excluded_recently_failed/filtered_unreachable are parsed
        out of the exploration planner's own debug-count reason string when
        present (see exploration_planners.py FoVAwareDirectionalFrontierPlanner);
        this module never recomputes or changes that scoring logic, it only
        reports whatever numbers were already produced.
        """
        if self.level == QUIET:
            return

        match = self._FRONTIER_COUNT_PATTERN.search(str(reason or ""))
        if match:
            generated = match.group("generated")
            excluded = match.group("excluded")
            filtered = match.group("filtered")
        else:
            generated = str(candidate_count) if candidate_count is not None else "--"
            excluded = "--"
            filtered = "--"

        score_text = f"{float(score):.2f}" if score is not None else "--"
        selected_text = _fmt_point(selected) if success else "None"

        self._emit(
            f"[FRONTIER] {robot_label} generated={generated} "
            f"filtered_unreachable={filtered} failed_recent={excluded} "
            f"selected={selected_text} score={score_text}"
        )

    # ------------------------------------------------------------------ [WARN]

    def warn(self, message: str) -> None:
        """Always visible regardless of level, short of being disabled."""
        self._emit(f"[WARN] {message}")

    # ------------------------------------------------------------------ debug-only passthrough

    def debug(self, message: str) -> bool:
        """Free-form line, emitted only at DEBUG level.

        For legacy/verbose detail (e.g. the old long "Planner: A* / ..."
        line) that a normal/quiet console should not see anymore, but that
        is still useful when actively debugging with DEBUG level enabled.

        Returns True when the line was actually emitted (useful for tests).
        """
        if not self._at_least(DEBUG):
            return False
        self._emit(str(message))
        return True
