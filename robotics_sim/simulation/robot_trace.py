"""
robot_trace.py

Opt-in, terminal-only diagnostic trace of the robot's belief-map/
navigation reasoning -- how it maps obstacles, groups new samples into
line-like sections, selects/rejects frontier candidates, and why routes
succeed or fail. Disabled by default; enabled via the ROBOT_TRACE
environment variable.

Deliberately separate from:
    - PERF diagnostics (render_perf.py): different concern (paint timing
      vs navigation reasoning), different env var, but the same design
      principle -- never printed unless explicitly requested, never GUI-
      console output.
    - the in-app GUI console (SimulationCanvas.append_console_message):
      trace lines only ever go to stdout via print(), and only when
      enabled. They never touch the canvas/console history.
    - telemetry.py's [STATE]/[MAP]/[ROUTE]/[NAV]/[FRONTIER] lines, an
      always-available (level-gated), GUI-console-aimed summary. This
      module is a separate, richer, terminal-only trace aimed at
      development/debugging, off by default.

Activation (PowerShell):
    $env:ROBOT_TRACE = "map,obstacles,decision,frontier,route,safety"
    python .\\main.py

Activation (POSIX shells):
    ROBOT_TRACE=map,obstacles,decision,frontier,route,safety python ./main.py

Unset/empty ROBOT_TRACE -> completely silent (matches the existing rule
that nothing prints to stdout by default).

Optional: ROBOT_TRACE_POINTS=1 also prints up to MAX_SAMPLE_POINTS raw
obstacle sample points per trace call; omitted by default to keep lines
compact.

File output: whenever ROBOT_TRACE is non-empty, a companion
BeliefTraceWriter (belief_trace_writer.py) is created automatically so a
run leaves persistent artifacts under runs/debug/belief_trace_<timestamp>/
without needing shell redirection (Tee-Object). Two more env vars control
this:

    ROBOT_TRACE_STDOUT=0   suppress the human-readable [TRACE ...] lines
                           printed to the terminal; file output is
                           unaffected (default "1": print, as before).
    ROBOT_TRACE_DIR=path   base directory for the timestamped run
                           directory (default "runs/debug").

Construction note: the RobotTrace() constructor itself only creates the
file sink automatically when `create_file_sink=True` is passed -- this is
what engine.py's ensure_robot_trace() always does for real simulation
runs. It defaults to False so that plain unit tests constructing
RobotTrace(env={...}) to exercise formatting/throttling logic keep
touching zero real files, with no behavior change from before this file-
output feature existed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

from robotics_sim.simulation.belief_trace_writer import (
    DEFAULT_BASE_DIR,
    BeliefTraceWriter,
    make_run_directory,
)

Point2D = tuple[float, float]

CATEGORIES: tuple[str, ...] = ("map", "obstacles", "frontier", "route", "decision", "safety")

DEFAULT_MAP_TRACE_INTERVAL_S = 1.0
DEFAULT_DECISION_REPEAT_INTERVAL_S = 2.0
DEFAULT_FRONTIER_REPEAT_INTERVAL_S = 2.0
DEFAULT_BELIEF_SNAPSHOT_INTERVAL_S = 5.0
MAX_OBSTACLE_SECTIONS = 6
MAX_SAMPLE_POINTS = 5

# Decision kinds that are always shown immediately, never throttled: each
# is a discrete, one-off event (a plan was requested, a safety replan was
# triggered, a prefetched path was promoted, a prefetch was launched) --
# not a repeating steady state like HOLD/BRAKE can be while the robot sits
# idle or brakes for several consecutive ticks.
ALWAYS_VISIBLE_DECISION_KINDS = frozenset(
    {"REQUEST_PLAN", "REPLAN_FOR_SAFETY", "ACCEPT_PENDING_PATH", "PREFETCH_NEXT_TARGET"}
)
_SECTION_TOLERANCE = 0.05  # meters; grouping tolerance for axis-aligned samples

# Compact, stable identifiers for common route-failure reason substrings --
# an independent, smaller copy of telemetry.py's own reason-slugging (kept
# separate rather than importing telemetry.py's private helper, so this
# module has no dependency on telemetry.py's internals).
_ROUTE_FAILURE_REASON_SLUGS: tuple[tuple[str, str], ...] = (
    ("no path found", "no_path"),
    ("first segment blocked", "first_segment_blocked"),
    ("final waypoint does not reach", "endpoint_mismatch"),
    ("goal cell is occupied", "goal_occupied"),
    ("goal cell is not traversable", "goal_blocked"),
    ("start cell is not traversable", "start_blocked"),
    ("repeated safety replan", "repeated_safety_replan"),
)


def slug_route_failure_reason(reason: str) -> str:
    """Compact, stable identifier for a route-failure reason string, for
    a short [TRACE ROUTE] line instead of a full sentence."""
    text = str(reason or "").strip().lower()
    for needle, slug in _ROUTE_FAILURE_REASON_SLUGS:
        if needle in text:
            return slug
    if not text:
        return "unknown"
    return text.split(";")[0].split(".")[0].strip().replace(" ", "_")[:40]


def parse_categories(raw: str | None) -> frozenset[str]:
    """Parse a comma-separated ROBOT_TRACE value into a set of enabled
    categories.

    Unknown tokens are ignored (forward-compatible, never raises).
    "all" expands to every known category. Empty/None -> empty set, i.e.
    disabled -- this is what running with ROBOT_TRACE unset produces.
    """
    if not raw:
        return frozenset()
    tokens = {token.strip().lower() for token in raw.split(",") if token.strip()}
    if "all" in tokens:
        return frozenset(CATEGORIES)
    return frozenset(token for token in tokens if token in CATEGORIES)


@dataclass(frozen=True)
class ObstacleSection:
    """One compact line-like group of obstacle sample points."""

    axis: str  # "x" (constant-x / vertical wall) or "y" (constant-y / horizontal wall)
    coordinate: float
    span_min: float
    span_max: float
    count: int

    def format(self) -> str:
        other_axis = "y" if self.axis == "x" else "x"
        return (
            f"{self.axis}={self.coordinate:.2f} {other_axis}="
            f"{self.span_min:.2f}..{self.span_max:.2f} n={self.count}"
        )


def group_obstacle_points_into_sections(
    points: Iterable[Point2D],
    *,
    tolerance: float = _SECTION_TOLERANCE,
) -> list[ObstacleSection]:
    """Group axis-aligned obstacle sample points into compact line-like
    sections instead of reporting every raw point.

    Office.sim's obstacles are grid-like (mostly vertical or horizontal
    boundary walls), so this groups points sharing an (approximately)
    equal x first (constant-x / vertical sections), then groups whatever
    is left by equal y (constant-y / horizontal sections). Any point left
    over afterwards (no other point shares its x or y within tolerance)
    becomes its own single-point section, so the total count is never
    silently dropped. Sections are returned largest-first.

    Purely a diagnostic/formatting helper: it never mutates its input,
    never touches the belief map, and has no effect on mapping, planning,
    or navigation.
    """
    remaining = [(float(x), float(y)) for x, y in points]

    def _extract_axis_groups(
        pts: list[Point2D], axis_index: int, axis_name: str
    ) -> tuple[list[ObstacleSection], list[Point2D]]:
        used = [False] * len(pts)
        found: list[ObstacleSection] = []
        for i, p in enumerate(pts):
            if used[i]:
                continue
            group_idx = [i]
            for j in range(i + 1, len(pts)):
                if used[j]:
                    continue
                if abs(pts[j][axis_index] - p[axis_index]) <= tolerance:
                    group_idx.append(j)
            if len(group_idx) >= 2:
                for k in group_idx:
                    used[k] = True
                other_index = 1 - axis_index
                others = [pts[k][other_index] for k in group_idx]
                coordinate = sum(pts[k][axis_index] for k in group_idx) / len(group_idx)
                found.append(
                    ObstacleSection(
                        axis=axis_name,
                        coordinate=coordinate,
                        span_min=min(others),
                        span_max=max(others),
                        count=len(group_idx),
                    )
                )
        leftover = [p for i, p in enumerate(pts) if not used[i]]
        return found, leftover

    x_sections, leftover = _extract_axis_groups(remaining, 0, "x")
    y_sections, leftover = _extract_axis_groups(leftover, 1, "y")

    sections = x_sections + y_sections
    for x, y in leftover:
        sections.append(ObstacleSection(axis="x", coordinate=x, span_min=y, span_max=y, count=1))

    sections.sort(key=lambda s: s.count, reverse=True)
    return sections


def _fmt_point(point: Point2D | None) -> str:
    if point is None:
        return "None"
    return f"({float(point[0]):.2f},{float(point[1]):.2f})"


def _fmt_count(value: int | None) -> str:
    return "n/a" if value is None else str(int(value))


def format_map_trace_line(
    *,
    sim_time: float,
    robot_label: str,
    pose: Point2D,
    explored_percent: float,
    mapped_obstacle_samples: int,
    free_unlocked: int | None = None,
    occupied_new: int | None = None,
    unknown_remaining: int | None = None,
) -> str:
    """[TRACE MAP] line: how the belief map changed from the robot's
    perspective. free_unlocked/occupied_new/unknown_remaining are "n/a"
    when not cheaply available -- callers should not compute anything
    expensive just to fill them in."""
    return (
        f"[TRACE MAP t={float(sim_time):.1f}] {robot_label} pose={_fmt_point(pose)} "
        f"free_unlocked={_fmt_count(free_unlocked)} occupied_new={_fmt_count(occupied_new)} "
        f"unknown_remaining={_fmt_count(unknown_remaining)} explored={float(explored_percent):.1f}% "
        f"mapped_obs={int(mapped_obstacle_samples)}"
    )


def format_obstacle_trace_line(
    *,
    sim_time: float,
    robot_label: str,
    sample_points: int,
    sections: list[ObstacleSection],
    max_sections: int = MAX_OBSTACLE_SECTIONS,
) -> str:
    """[TRACE OBS] line: newly mapped obstacle samples grouped into
    compact sections, not a raw point dump."""
    shown = sections[:max_sections]
    lines_text = "; ".join(section.format() for section in shown)
    if len(sections) > max_sections:
        lines_text += f" (+{len(sections) - max_sections} more)"
    return (
        f"[TRACE OBS t={float(sim_time):.1f}] {robot_label} sections={len(sections)} "
        f"sample_points={int(sample_points)} lines=[{lines_text}]"
    )


def format_decision_trace_line(
    *,
    sim_time: float,
    robot_label: str,
    kind: str,
    reason: str,
    active_target: Point2D | None,
    path_goal: Point2D | None,
    pending_target: Point2D | None,
    repeated: int | None = None,
) -> str:
    """[TRACE DECISION] line: the NavigationDecision just applied.

    repeated, when given and > 0, notes how many identical decisions were
    suppressed since the last line (see RobotTrace.trace_decision()'s
    dedup/throttle) -- mirrors telemetry.py's [NAV] repeated=N convention.
    """
    safe_reason = str(reason).replace('"', "'")
    line = (
        f'[TRACE DECISION t={float(sim_time):.1f}] {robot_label} kind={kind} reason="{safe_reason}" '
        f"active={_fmt_point(active_target)} path_goal={_fmt_point(path_goal)} "
        f"pending={_fmt_point(pending_target)}"
    )
    if repeated:
        line += f" repeated={int(repeated)}"
    return line


def format_frontier_trace_line(
    *,
    sim_time: float,
    source: str,
    selected: Point2D | None,
    generated: int | None = None,
    rejected_failed: int | None = None,
    rejected_unreachable: int | None = None,
    repeated: int | None = None,
) -> str:
    """[TRACE FRONTIER] line: an exploration target selection attempt.

    generated/rejected_* are omitted from the line entirely when not
    available, rather than shown as "n/a" -- keeps the common case (only
    source/selected known) compact, per the "do not overbuild" guidance.
    repeated, when given and > 0, notes how many identical
    (source/selected=None) events were suppressed since the last line --
    see RobotTrace.trace_frontier()'s dedup/throttle.
    """
    parts = [f"[TRACE FRONTIER t={float(sim_time):.1f}]", f"source={source}"]
    if generated is not None:
        parts.append(f"generated={int(generated)}")
    if rejected_failed is not None:
        parts.append(f"rejected_failed={int(rejected_failed)}")
    if rejected_unreachable is not None:
        parts.append(f"rejected_unreachable={int(rejected_unreachable)}")
    parts.append(f"selected={_fmt_point(selected)}")
    if repeated:
        parts.append(f"repeated={int(repeated)}")
    return " ".join(parts)


def format_route_trace_line(
    *,
    sim_time: float,
    robot_label: str,
    result: str,
    start: Point2D | None,
    goal: Point2D | None,
    reason: str = "",
    waypoint_count: int | None = None,
    length: float | None = None,
    mapped_obstacle_count: int = 0,
) -> str:
    """[TRACE ROUTE] line: a route request's outcome (ok/fail)."""
    parts = [f"[TRACE ROUTE t={float(sim_time):.1f}]", robot_label, f"result={result}"]
    if reason:
        parts.append(f"reason={reason}")
    parts.append(f"start={_fmt_point(start)}")
    parts.append(f"goal={_fmt_point(goal)}")
    if waypoint_count is not None:
        parts.append(f"wp={int(waypoint_count)}")
    if length is not None:
        parts.append(f"length={float(length):.2f}")
    parts.append(f"mapped_obs={int(mapped_obstacle_count)}")
    return " ".join(parts)


def format_safety_trace_line(
    *,
    sim_time: float,
    robot_label: str,
    goal: Point2D | None,
    repair_status: str,
    min_clearance: float | None = None,
) -> str:
    """[TRACE SAFETY] line: narrow-passage / route_affected repair status.

    min_clearance is an approximation (nearest mapped-obstacle-point
    distance), never an exact segment-clearance computation -- the "~"
    prefix and "approx" framing are deliberate, matching engine.py's
    existing [NARROW_DIAG] min_clearance semantics. ASCII-only ("~", not
    the Unicode "approx" sign "≈") so this is always encodable on a
    plain cp1252 Windows terminal (see RobotTrace._emit()'s defensive
    encode as a second line of defense for any other unexpected
    non-ASCII content).
    """
    clearance_text = "n/a" if min_clearance is None else f"~{float(min_clearance):.2f}"
    return (
        f"[TRACE SAFETY t={float(sim_time):.1f}] {robot_label} route_affected goal={_fmt_point(goal)} "
        f"min_clearance={clearance_text} repair={repair_status}"
    )


class RobotTrace:
    """Opt-in terminal trace controller.

    Reads ROBOT_TRACE/ROBOT_TRACE_POINTS from the environment at
    CONSTRUCTION time, not import time, so tests can set/clear os.environ
    (or pass an explicit `env` mapping) and get a deterministic, fresh
    instance without any module-level state to reset between tests.

    Every trace_*() method is a no-op (returns False, prints nothing)
    unless its category is enabled -- callers do not need to check
    is_enabled() themselves first, though they may want to for expensive
    argument computation (e.g. skip building an obstacle-section summary
    entirely when "obstacles" is disabled).
    """

    def __init__(self, env: "dict[str, str] | None" = None):
        source = env if env is not None else os.environ
        self.categories = parse_categories(source.get("ROBOT_TRACE"))
        self.include_points = str(source.get("ROBOT_TRACE_POINTS", "")).strip().lower() in {"1", "true", "yes"}
        self.stdout_enabled = str(source.get("ROBOT_TRACE_STDOUT", "1")).strip().lower() not in {
            "0", "false", "no", "off",
        }
        # Belief-trace artifact files are ON by default and independent of
        # ROBOT_TRACE/self.categories entirely -- BELIEF_TRACE_ARTIFACTS is
        # the only thing that can turn them off. This is the opposite
        # polarity from stdout_enabled/categories on purpose: "python
        # .\main.py" with zero env vars set must still produce artifact
        # files (per this module's redesign), while printing nothing.
        self.file_artifacts_enabled = str(source.get("BELIEF_TRACE_ARTIFACTS", "1")).strip().lower() not in {
            "0", "false", "no", "off",
        }
        self._base_dir = source.get("ROBOT_TRACE_DIR") or DEFAULT_BASE_DIR
        self.writer: BeliefTraceWriter | None = None
        self.file_output_dir = None
        self._last_map_trace_time: float | None = None
        self._last_snapshot_time: float | None = None
        # Decision/frontier dedup-and-throttle state -- see
        # trace_decision()/trace_frontier(). Mirrors telemetry.py's [NAV]
        # signature-dedup pattern (same idea, independent implementation:
        # this module has no dependency on telemetry.py).
        self._last_decision_signature: tuple | None = None
        self._last_decision_time: float | None = None
        self._decision_repeat_count = 0
        self._last_frontier_signature: tuple | None = None
        self._last_frontier_time: float | None = None
        self._frontier_repeat_count = 0

    def is_enabled(self, category: str) -> bool:
        return category in self.categories

    @property
    def enabled(self) -> bool:
        return bool(self.categories)

    def start_run(self):
        """Create a brand-new timestamped belief-trace run directory +
        writer for one simulation run, discarding any previous one.

        Deliberately NOT called from __init__ or from any trace_*()
        method: engine.py calls this explicitly exactly when a run starts
        (Start Simulation / Restart Simulation), never merely when
        RobotTrace is constructed or lazily on first use -- so loading a
        scenario alone does not create a directory, but each run does, and
        repeated Restarts each get their own fresh one.

        Independent of ROBOT_TRACE/self.categories: artifact files are
        generated whether or not terminal tracing is enabled.
        BELIEF_TRACE_ARTIFACTS is the only switch that disables this.

        Returns the new run directory (Path), or None if file artifacts
        are disabled or the directory could not be created (best-effort:
        a single warning is printed, never raised).
        """
        if not self.file_artifacts_enabled:
            self.writer = None
            self.file_output_dir = None
            return None
        try:
            run_dir = make_run_directory(self._base_dir)
        except OSError as exc:
            self.writer = None
            self.file_output_dir = None
            self._print(f"[BELIEF TRACE] could not create trace directory: {exc}")
            return None
        self.writer = BeliefTraceWriter(run_dir, categories=tuple(sorted(self.categories)), warn=self._print)
        self.file_output_dir = run_dir
        return run_dir

    def print_line(self, message: str) -> None:
        """Public, Windows-safe, unconditional print for one-off
        operational messages (e.g. engine.py's belief-trace startup line)
        that must always be visible regardless of ROBOT_TRACE_STDOUT."""
        self._print(message)

    def _print(self, line: str) -> None:
        """Print *line* unconditionally (ignores ROBOT_TRACE_STDOUT --
        used for the one-time startup line and file-sink error warnings,
        never for routine per-tick trace lines) without ever letting an
        encoding error reach the caller. Every format_*_line() in this
        module is built to be ASCII-only by construction (see e.g.
        format_safety_trace_line()'s "~" instead of the Unicode approx
        sign), so this is a second, defensive line of protection -- not
        the primary fix -- for any unexpected non-ASCII content (e.g. a
        user-entered reason string) on a narrow-encoding terminal (cp1252
        on plain Windows consoles). Never silently drops the line:
        undecodable characters are replaced (with "?" or a backslash
        escape), not deleted.
        """
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            safe_line = line.encode(encoding, errors="replace").decode(encoding, errors="replace")
        except (LookupError, UnicodeError):
            safe_line = line.encode("ascii", errors="backslashreplace").decode("ascii")
        print(safe_line)

    def _emit(self, line: str) -> None:
        """Routine [TRACE ...] line: respects ROBOT_TRACE_STDOUT, so file-
        only debug runs (ROBOT_TRACE_STDOUT=0) do not spam the terminal
        while file output (see self.writer) continues independently."""
        if self.stdout_enabled:
            self._print(line)

    # ------------------------------------------------------------------ map

    def trace_map(
        self,
        *,
        sim_time: float,
        interval: float = DEFAULT_MAP_TRACE_INTERVAL_S,
        force: bool = False,
        robot_label: str = "R1",
        pose: Point2D | None = None,
        explored_percent: float = 0.0,
        mapped_obstacle_samples: int = 0,
        free_unlocked: int | None = None,
        occupied_new: int | None = None,
        unknown_remaining: int | None = None,
    ) -> bool:
        """Throttled: at most once every `interval` seconds of simulation
        time, regardless of how often this is called. The throttle governs
        both sinks together (file artifacts are meant to capture a
        representative record, not every single tick's identical state);
        is_enabled("map") only decides whether this also prints to
        stdout -- belief-trace artifact files are written whenever
        self.writer exists, independent of ROBOT_TRACE/categories."""
        if not force and self._last_map_trace_time is not None and (
            float(sim_time) - self._last_map_trace_time
        ) < interval:
            return False
        self._last_map_trace_time = float(sim_time)
        emitted = False
        if self.is_enabled("map"):
            self._emit(
                format_map_trace_line(
                    sim_time=sim_time,
                    robot_label=robot_label,
                    pose=pose,
                    explored_percent=explored_percent,
                    mapped_obstacle_samples=mapped_obstacle_samples,
                    free_unlocked=free_unlocked,
                    occupied_new=occupied_new,
                    unknown_remaining=unknown_remaining,
                )
            )
            emitted = True
        if self.writer is not None:
            self.writer.record_event(
                "map",
                simulation_time=sim_time,
                robot_id=robot_label,
                pose=pose,
                explored_percent=explored_percent,
                mapped_obstacle_samples=mapped_obstacle_samples,
                payload=dict(
                    free_unlocked=free_unlocked,
                    occupied_new=occupied_new,
                    unknown_remaining=unknown_remaining,
                ),
            )
            emitted = True
        return emitted

    def maybe_snapshot_belief(
        self,
        *,
        sim_time: float,
        provider: Callable[[], dict | None],
        interval: float = DEFAULT_BELIEF_SNAPSHOT_INTERVAL_S,
        force: bool = False,
    ) -> bool:
        """Best-effort periodic belief_final.json (+ belief_grid_final.npz)
        snapshot, so a crash or hang still leaves useful data (no clean-
        shutdown hook is required). `provider` is only called -- lazily --
        once this decides a snapshot is actually due, so building the
        (possibly grid-scanning) snapshot dict costs nothing on ticks
        where no file sink exists or the interval has not elapsed yet."""
        if self.writer is None:
            return False
        if not force and self._last_snapshot_time is not None and (
            float(sim_time) - self._last_snapshot_time
        ) < interval:
            return False
        self._last_snapshot_time = float(sim_time)
        snapshot = provider()
        if not snapshot:
            return False
        self.writer.write_belief_snapshot(snapshot)
        self.writer.flush_summary()
        return True

    # ------------------------------------------------------------------ obstacles

    def trace_obstacles(
        self,
        *,
        sim_time: float,
        robot_label: str,
        points: list[Point2D],
        max_sections: int = MAX_OBSTACLE_SECTIONS,
        explored_percent: float | None = None,
    ) -> bool:
        """Only emits when there are new points -- never on an empty
        sensor update, so this never fires every frame by itself.
        is_enabled("obstacles") only gates stdout printing; the file
        writer (if present) always receives new-point events regardless
        of ROBOT_TRACE/categories."""
        if not points:
            return False
        sections = group_obstacle_points_into_sections(points)
        emitted = False
        if self.is_enabled("obstacles"):
            self._emit(
                format_obstacle_trace_line(
                    sim_time=sim_time,
                    robot_label=robot_label,
                    sample_points=len(points),
                    sections=sections,
                    max_sections=max_sections,
                )
            )
            if self.include_points:
                sample = [(float(x), float(y)) for x, y in list(points)[:MAX_SAMPLE_POINTS]]
                self._emit(f"[TRACE OBS t={float(sim_time):.1f}] points_sample={sample}")
            emitted = True
        if self.writer is not None:
            self.writer.record_event(
                "obstacles",
                simulation_time=sim_time,
                robot_id=robot_label,
                explored_percent=explored_percent,
                mapped_obstacle_samples=len(points),
                payload={
                    "sections": [
                        dict(
                            orientation=section.axis,
                            coord=section.coordinate,
                            span_min=section.span_min,
                            span_max=section.span_max,
                            n_points=section.count,
                        )
                        for section in sections
                    ],
                },
            )
            for section in sections:
                self.writer.record_obstacle_section(
                    simulation_time=sim_time,
                    robot_id=robot_label,
                    orientation=section.axis,
                    coord=section.coordinate,
                    span_min=section.span_min,
                    span_max=section.span_max,
                    n_points=section.count,
                    raw_sample_count=len(points),
                    explored_percent=explored_percent,
                )
            emitted = True
        return emitted

    # ------------------------------------------------------------------ decision

    def trace_decision(
        self,
        *,
        sim_time: float,
        robot_label: str,
        kind: str,
        reason: str,
        active_target: Point2D | None,
        path_goal: Point2D | None,
        pending_target: Point2D | None,
        repeat_interval: float = DEFAULT_DECISION_REPEAT_INTERVAL_S,
    ) -> bool:
        """Emits a [TRACE DECISION] line, deduplicated/throttled for
        repeating steady states.

        kind in ALWAYS_VISIBLE_DECISION_KINDS (REQUEST_PLAN,
        REPLAN_FOR_SAFETY, ACCEPT_PENDING_PATH, PREFETCH_NEXT_TARGET) is
        always shown immediately -- each is a discrete, one-off event.
        Anything else (chiefly HOLD, which can repeat every tick while the
        robot sits idle/exhausted) is only shown when its full signature
        (kind, reason, active/path_goal/pending) changes, or at most once
        every `repeat_interval` simulated seconds while it stays the same
        -- with a repeated=N count for however many identical calls were
        suppressed in between, mirroring telemetry.py's [NAV] dedup.

        is_enabled("decision") only gates stdout printing; the file
        writer (if present) always receives whatever survives the
        dedup/throttle below, independent of ROBOT_TRACE/categories.
        """
        if kind in ALWAYS_VISIBLE_DECISION_KINDS:
            self._last_decision_signature = None
            self._last_decision_time = None
            self._decision_repeat_count = 0
            emitted = False
            if self.is_enabled("decision"):
                self._emit(
                    format_decision_trace_line(
                        sim_time=sim_time,
                        robot_label=robot_label,
                        kind=kind,
                        reason=reason,
                        active_target=active_target,
                        path_goal=path_goal,
                        pending_target=pending_target,
                    )
                )
                emitted = True
            if self.writer is not None:
                self.writer.record_event(
                    "decision",
                    simulation_time=sim_time,
                    robot_id=robot_label,
                    payload=dict(
                        kind=kind,
                        reason=str(reason),
                        active=list(active_target) if active_target is not None else None,
                        path_goal=list(path_goal) if path_goal is not None else None,
                        pending=list(pending_target) if pending_target is not None else None,
                    ),
                )
                self.writer.record_decision_event(
                    simulation_time=sim_time,
                    robot_id=robot_label,
                    kind=kind,
                    reason=str(reason),
                    active=active_target,
                    path_goal=path_goal,
                    pending=pending_target,
                )
            return emitted

        signature = (kind, str(reason), active_target, path_goal, pending_target)
        same_signature = signature == self._last_decision_signature

        if (
            same_signature
            and self._last_decision_time is not None
            and (float(sim_time) - self._last_decision_time) < repeat_interval
        ):
            self._decision_repeat_count += 1
            return False

        repeated = self._decision_repeat_count if same_signature else None
        self._decision_repeat_count = 0
        self._last_decision_signature = signature
        self._last_decision_time = float(sim_time)
        emitted = False
        if self.is_enabled("decision"):
            self._emit(
                format_decision_trace_line(
                    sim_time=sim_time,
                    robot_label=robot_label,
                    kind=kind,
                    reason=reason,
                    active_target=active_target,
                    path_goal=path_goal,
                    pending_target=pending_target,
                    repeated=repeated,
                )
            )
            emitted = True
        if self.writer is not None:
            self.writer.record_event(
                "decision",
                simulation_time=sim_time,
                robot_id=robot_label,
                payload=dict(
                    kind=kind,
                    reason=str(reason),
                    active=list(active_target) if active_target is not None else None,
                    path_goal=list(path_goal) if path_goal is not None else None,
                    pending=list(pending_target) if pending_target is not None else None,
                    repeated=repeated,
                ),
            )
            self.writer.record_decision_event(
                simulation_time=sim_time,
                robot_id=robot_label,
                kind=kind,
                reason=str(reason),
                active=active_target,
                path_goal=path_goal,
                pending=pending_target,
            )
            emitted = True
        return emitted

    # ------------------------------------------------------------------ frontier

    def trace_frontier(
        self,
        *,
        sim_time: float,
        source: str,
        selected: Point2D | None,
        generated: int | None = None,
        rejected_failed: int | None = None,
        rejected_unreachable: int | None = None,
        repeat_interval: float = DEFAULT_FRONTIER_REPEAT_INTERVAL_S,
    ) -> bool:
        """Emits a [TRACE FRONTIER] line, deduplicated/throttled for
        repeated "nothing selected" events (chiefly exploration-exhausted
        holds, which can repeat every tick). A genuine selection
        (selected is not None) is always shown immediately -- it is a
        real, discrete choice, never a repeating steady state.

        is_enabled("frontier") only gates stdout printing; the file
        writer (if present) always receives whatever survives the
        dedup/throttle below, independent of ROBOT_TRACE/categories.
        """
        if selected is not None:
            self._last_frontier_signature = None
            self._last_frontier_time = None
            self._frontier_repeat_count = 0
            emitted = False
            if self.is_enabled("frontier"):
                self._emit(
                    format_frontier_trace_line(
                        sim_time=sim_time,
                        source=source,
                        selected=selected,
                        generated=generated,
                        rejected_failed=rejected_failed,
                        rejected_unreachable=rejected_unreachable,
                    )
                )
                emitted = True
            if self.writer is not None:
                self.writer.record_event(
                    "frontier",
                    simulation_time=sim_time,
                    robot_id="R1",
                    payload=dict(
                        source=source,
                        generated=generated,
                        selected=list(selected),
                        map_wide_fallback_used=(source == "map-wide-fallback"),
                    ),
                )
                self.writer.record_frontier_event(
                    simulation_time=sim_time,
                    robot_id="R1",
                    source=source,
                    generated_count=generated,
                    selected=selected,
                    map_wide_fallback_used=(source == "map-wide-fallback"),
                    reason="",
                )
                emitted = True
            return emitted

        signature = (source, generated, rejected_failed, rejected_unreachable)
        same_signature = signature == self._last_frontier_signature

        if (
            same_signature
            and self._last_frontier_time is not None
            and (float(sim_time) - self._last_frontier_time) < repeat_interval
        ):
            self._frontier_repeat_count += 1
            return False

        repeated = self._frontier_repeat_count if same_signature else None
        self._frontier_repeat_count = 0
        self._last_frontier_signature = signature
        self._last_frontier_time = float(sim_time)
        emitted = False
        if self.is_enabled("frontier"):
            self._emit(
                format_frontier_trace_line(
                    sim_time=sim_time,
                    source=source,
                    selected=None,
                    generated=generated,
                    rejected_failed=rejected_failed,
                    rejected_unreachable=rejected_unreachable,
                    repeated=repeated,
                )
            )
            emitted = True
        if self.writer is not None:
            reason_bits = []
            if rejected_failed:
                reason_bits.append(f"rejected_failed={int(rejected_failed)}")
            if rejected_unreachable:
                reason_bits.append(f"rejected_unreachable={int(rejected_unreachable)}")
            self.writer.record_event(
                "frontier",
                simulation_time=sim_time,
                robot_id="R1",
                payload=dict(
                    source=source,
                    generated=generated,
                    selected=None,
                    rejected_failed=rejected_failed,
                    rejected_unreachable=rejected_unreachable,
                    repeated=repeated,
                ),
            )
            self.writer.record_frontier_event(
                simulation_time=sim_time,
                robot_id="R1",
                source=source,
                generated_count=generated,
                selected=None,
                map_wide_fallback_used=(source == "map-wide-fallback"),
                reason=" ".join(reason_bits),
            )
            emitted = True
        return emitted

    # ------------------------------------------------------------------ route

    def trace_route(
        self,
        *,
        sim_time: float,
        robot_label: str,
        result: str,
        start: Point2D | None,
        goal: Point2D | None,
        reason: str = "",
        waypoint_count: int | None = None,
        length: float | None = None,
        mapped_obstacle_count: int = 0,
        planner: str = "",
        simplifier: str = "",
    ) -> bool:
        """is_enabled("route") only gates stdout printing; the file writer
        (if present) always receives route events, independent of
        ROBOT_TRACE/categories."""
        emitted = False
        if self.is_enabled("route"):
            self._emit(
                format_route_trace_line(
                    sim_time=sim_time,
                    robot_label=robot_label,
                    result=result,
                    start=start,
                    goal=goal,
                    reason=reason,
                    waypoint_count=waypoint_count,
                    length=length,
                    mapped_obstacle_count=mapped_obstacle_count,
                )
            )
            emitted = True
        if self.writer is not None:
            self.writer.record_event(
                "route",
                simulation_time=sim_time,
                robot_id=robot_label,
                payload=dict(
                    result=result,
                    reason=reason,
                    start=list(start) if start is not None else None,
                    goal=list(goal) if goal is not None else None,
                    waypoint_count=waypoint_count,
                    length=length,
                    mapped_obs=mapped_obstacle_count,
                ),
            )
            self.writer.record_route_event(
                simulation_time=sim_time,
                robot_id=robot_label,
                result=result,
                reason=reason,
                start=start,
                goal=goal,
                waypoint_count=waypoint_count,
                length=length,
                mapped_obs=mapped_obstacle_count,
                planner=planner,
                simplifier=simplifier,
            )
            emitted = True
        return emitted

    # ------------------------------------------------------------------ safety

    def trace_safety(
        self,
        *,
        sim_time: float,
        robot_label: str,
        goal: Point2D | None,
        repair_status: str,
        min_clearance: float | None = None,
    ) -> bool:
        """is_enabled("safety") only gates stdout printing; the file
        writer (if present) always receives safety events, independent of
        ROBOT_TRACE/categories."""
        emitted = False
        if self.is_enabled("safety"):
            self._emit(
                format_safety_trace_line(
                    sim_time=sim_time,
                    robot_label=robot_label,
                    goal=goal,
                    repair_status=repair_status,
                    min_clearance=min_clearance,
                )
            )
            emitted = True
        if self.writer is not None:
            self.writer.record_event(
                "safety",
                simulation_time=sim_time,
                robot_id=robot_label,
                payload=dict(
                    goal=list(goal) if goal is not None else None,
                    repair_status=repair_status,
                    min_clearance=min_clearance,
                ),
            )
            emitted = True
        return emitted

    # ------------------------------------------------------------------ route_affected

    def trace_route_affected(
        self,
        *,
        sim_time: float,
        robot_id: str = "R1",
        path_goal: Point2D | None = None,
        active: Point2D | None = None,
        mapped_obs: int = 0,
        new_obstacle_count: int = 0,
        bbox: tuple[float, float, float, float] | None = None,
        action: str = "",
    ) -> bool:
        """Records one route_affected=yes occurrence (engine.py's
        new_information_affects_current_route() branch), whether the
        resulting repair was throttled or actually requested -- see
        engine.py's two call sites, one per outcome.

        File-only by design: there is no [TRACE ROUTE_AFFECTED] stdout
        line (the existing [NARROW_DIAG] telemetry.debug() line and the
        "New obstacle affects current route" console message already
        cover terminal visibility for this). Always recorded whenever the
        file writer exists, independent of ROBOT_TRACE/categories, so
        total_route_affected in run_summary.json can never silently miss
        an occurrence -- this is the ONE place that counter increments
        (see belief_trace_writer.py's record_route_affected_event()).
        """
        if self.writer is None:
            return False
        self.writer.record_event(
            "route_affected",
            simulation_time=sim_time,
            robot_id=robot_id,
            payload=dict(
                path_goal=list(path_goal) if path_goal is not None else None,
                active=list(active) if active is not None else None,
                mapped_obs=mapped_obs,
                new_obstacle_count=new_obstacle_count,
                bbox=list(bbox) if bbox is not None else None,
                action=action,
            ),
        )
        self.writer.record_route_affected_event(
            simulation_time=sim_time,
            robot_id=robot_id,
            path_goal=path_goal,
            active=active,
            mapped_obs=mapped_obs,
            new_obstacle_count=new_obstacle_count,
            bbox=bbox,
            action=action,
        )
        return True
