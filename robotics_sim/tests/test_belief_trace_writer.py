"""Tests for belief_trace_writer.py: the best-effort file sink that gives
ROBOT_TRACE debug runs persistent artifacts without shell redirection."""
import csv
import json
import re

from robotics_sim.simulation.belief_trace_writer import (
    BeliefTraceWriter,
    make_run_directory,
)


# ---------------------------------------------------------------------------
# A. make_run_directory() creates a fresh, timestamped run directory.
# ---------------------------------------------------------------------------


def test_belief_trace_writer_creates_run_directory(tmp_path):
    run_dir = make_run_directory(str(tmp_path))

    assert run_dir.exists() and run_dir.is_dir()
    assert run_dir.parent == tmp_path
    assert re.fullmatch(r"belief_trace_\d{8}_\d{6}(_\d+)?", run_dir.name)


# ---------------------------------------------------------------------------
# B. record_event() appends one valid-JSON line per event to belief_events.jsonl.
# ---------------------------------------------------------------------------


def test_belief_trace_writer_writes_jsonl_event(tmp_path):
    run_dir = make_run_directory(str(tmp_path))
    writer = BeliefTraceWriter(run_dir, categories=("route",))

    writer.record_event(
        "route",
        simulation_time=90.5,
        robot_id="R1",
        payload={"result": "ok", "start": [7.25, 2.93], "goal": [0.25, 3.75], "waypoint_count": 5},
    )

    lines = (run_dir / "belief_events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_type"] == "route"
    assert record["simulation_time"] == 90.5
    assert record["robot_id"] == "R1"
    assert record["payload"]["result"] == "ok"
    assert record["payload"]["waypoint_count"] == 5


# ---------------------------------------------------------------------------
# C. record_route_event() appends a row with the exact required columns.
# ---------------------------------------------------------------------------


def test_belief_trace_writer_writes_route_csv(tmp_path):
    run_dir = make_run_directory(str(tmp_path))
    writer = BeliefTraceWriter(run_dir, categories=("route",))

    writer.record_route_event(
        simulation_time=90.5,
        robot_id="R1",
        result="fail",
        reason="no_path",
        start=(7.25, 2.93),
        goal=(0.25, 3.75),
        waypoint_count=None,
        length=None,
        mapped_obs=1771,
        planner="AStar",
        simplifier="RDP",
    )

    with open(run_dir / "route_events.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "simulation_time", "robot_id", "result", "reason", "start_x", "start_y",
            "goal_x", "goal_y", "waypoint_count", "length", "mapped_obs", "planner", "simplifier",
        ]
        rows = list(reader)

    assert len(rows) == 1
    row = rows[0]
    assert row["result"] == "fail"
    assert row["reason"] == "no_path"
    assert row["start_x"] == "7.25"
    assert row["goal_y"] == "3.75"
    assert row["mapped_obs"] == "1771"
    assert row["planner"] == "AStar"


# ---------------------------------------------------------------------------
# D. record_frontier_event() appends a row with the exact required columns.
# ---------------------------------------------------------------------------


def test_belief_trace_writer_writes_frontier_csv(tmp_path):
    run_dir = make_run_directory(str(tmp_path))
    writer = BeliefTraceWriter(run_dir, categories=("frontier",))

    writer.record_frontier_event(
        simulation_time=12.0,
        robot_id="R1",
        source="map-wide-fallback",
        generated_count=3,
        selected=(4.5, 1.25),
        map_wide_fallback_used=True,
        reason="",
    )
    writer.record_frontier_event(
        simulation_time=13.0,
        robot_id="R1",
        source="nearest",
        generated_count=0,
        selected=None,
        map_wide_fallback_used=False,
        reason="exploration exhausted: no reachable frontier candidates",
    )

    with open(run_dir / "frontier_events.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "simulation_time", "robot_id", "source", "generated_count",
            "selected_x", "selected_y", "map_wide_fallback_used", "reason",
        ]
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["selected_x"] == "4.5"
    assert rows[0]["map_wide_fallback_used"] == "True"
    assert rows[1]["selected_x"] == ""
    assert "exhausted" in rows[1]["reason"]


# ---------------------------------------------------------------------------
# E. record_obstacle_section() appends a row with the exact required columns.
# ---------------------------------------------------------------------------


def test_belief_trace_writer_writes_obstacle_sections_csv(tmp_path):
    run_dir = make_run_directory(str(tmp_path))
    writer = BeliefTraceWriter(run_dir, categories=("obstacles",))

    writer.record_obstacle_section(
        simulation_time=90.7,
        robot_id="R1",
        orientation="x",
        coord=8.95,
        span_min=3.86,
        span_max=4.81,
        n_points=20,
        raw_sample_count=20,
        explored_percent=18.8,
    )

    with open(run_dir / "obstacle_sections.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "simulation_time", "robot_id", "orientation", "coord", "span_min",
            "span_max", "n_points", "raw_sample_count", "explored_percent",
        ]
        rows = list(reader)

    assert rows == [
        {
            "simulation_time": "90.7", "robot_id": "R1", "orientation": "x", "coord": "8.95",
            "span_min": "3.86", "span_max": "4.81", "n_points": "20", "raw_sample_count": "20",
            "explored_percent": "18.8",
        }
    ]


# ---------------------------------------------------------------------------
# F. run_summary.json aggregates counters and reflects the latest state
#    once flushed.
# ---------------------------------------------------------------------------


def test_belief_trace_writer_updates_run_summary(tmp_path):
    run_dir = make_run_directory(str(tmp_path))
    writer = BeliefTraceWriter(run_dir, categories=("route", "frontier"))

    writer.record_route_event(
        simulation_time=1.0, robot_id="R1", result="ok", start=(0, 0), goal=(1, 1),
        waypoint_count=3, length=2.0, mapped_obs=10,
    )
    writer.record_route_event(
        simulation_time=2.0, robot_id="R1", result="fail", reason="no_path",
        start=(0, 0), goal=(1, 1), mapped_obs=12,
    )
    writer.record_frontier_event(
        simulation_time=3.0, robot_id="R1", source="nearest", selected=None,
        reason="exploration exhausted: no reachable frontier candidates",
    )
    writer.flush_summary()

    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["total_route_ok"] == 1
    assert summary["total_route_fail"] == 1
    assert summary["route_fail_reasons"] == {"no_path": 1}
    assert summary["total_frontier_none"] == 1
    assert summary["last_exhaustion_reason"] == "exploration exhausted: no reachable frontier candidates"
    assert summary["last_simulation_time"] == 3.0
    assert summary["trace_categories"] == ["route", "frontier"]


# ---------------------------------------------------------------------------
# G. Any file-write error disables the writer and warns exactly once --
#    never raises into the caller.
# ---------------------------------------------------------------------------


def test_belief_trace_writer_handles_write_error_without_crashing(tmp_path):
    # A file (not a directory) blocking the run directory's own path makes
    # every open() beneath it raise OSError -- including the constructor's
    # own initial run_summary.json write.
    blocker = tmp_path / "blocker_file"
    blocker.write_text("not a directory", encoding="utf-8")
    unwritable_run_dir = blocker / "belief_trace_20260101_000000"

    warnings = []
    writer = BeliefTraceWriter(unwritable_run_dir, categories=("route",), warn=warnings.append)

    assert writer.enabled is False
    assert len(warnings) == 1

    # Further calls must be silent no-ops, never raise, and must not add a
    # second warning.
    writer.record_route_event(
        simulation_time=1.0, robot_id="R1", result="ok", start=(0, 0), goal=(1, 1),
    )
    writer.record_event("route", simulation_time=1.0, robot_id="R1", payload={})
    writer.flush_summary()

    assert len(warnings) == 1
