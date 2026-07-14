"""
Regression tests for incomplete belief-trace artifact coverage.

Manual Office.sim evidence: the app log showed many [ROUTE ok] lines,
route_affected=yes ("New obstacle affects current route. Replanning..."),
and safety/repair replans, but run_summary.json reported
total_route_affected=0 and total_safety_replans stuck far below what the
log implied, while total_route_ok/total_route_fail were only partially
populated.

Root causes found by auditing every engine.py call site that assigns a new
path or reports route_affected=yes:

1. engine.py's route_affected block (inside simulation_step()) only called
   _emit_robot_trace(..., "trace_safety", ...) in the THROTTLED branch (the
   route_affected_replan_allowed() == False case). The ALLOWED branch --
   where the repair actually proceeds via replan_after_new_information(),
   i.e. the common "New obstacle affects current route. Replanning..."
   case -- had NO trace call at all. Since
   BeliefTraceWriter.record_route_affected() (the old counter-only method)
   was only ever invoked from trace_safety(), total_route_affected only
   ever counted throttled occurrences, undercounting every run where most
   repairs are allowed to proceed (the common case).

2. ACCEPT_PENDING_PATH (a promoted prefetched path) bypasses
   log_route_assignment()/report_route_success() entirely -- no [ROUTE ok]
   telemetry line and no trace_route event were ever produced for it, even
   though it is a real route assignment the robot starts following.

Fixes:
    - New RobotTrace.trace_route_affected() / BeliefTraceWriter's
      record_route_affected_event() + route_affected_events.csv: the ONE
      place total_route_affected increments, called from BOTH the
      throttled branch (action="repair_throttled") and the previously-
      uninstrumented allowed branch (action="repair_requested") in
      simulation_step()'s route_affected block.
    - ACCEPT_PENDING_PATH now also emits a trace_route(result="ok", ...)
      event (reason="accepted pending path (prefetch)") in
      apply_navigation_decision().

These tests exercise RobotTrace/BeliefTraceWriter directly (pure, file-sink
via tmp_path) for the writer-level completeness contract, and the same
lightweight duck-typed SimulationControllerMixin fakes used throughout this
test suite (see test_pending_path_invalidated_by_replan.py) for the
engine-boundary wiring that is practical to exercise without a full Qt/
sensor/planner stack (apply_navigation_decision(), apply_route_result()).
simulation_step()'s route_affected block itself is not directly invoked
here (it requires the full sensor/mapping/collision-checker stack) -- its
two call sites are exercised through RobotTrace.trace_route_affected()
with the exact same arguments/actions engine.py now passes.
"""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.robot_trace import RobotTrace
from robotics_sim.simulation.telemetry import TelemetryLogger


PATH_GOAL_A = (7.75, 3.25)


def _make_agent(position=(6.57, 3.58)) -> RobotAgent:
    return RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]

    def set_goal(self, goal):
        self.goal = tuple(goal)

    def force_stop(self, reason=""):
        self.force_stopped_reason = reason


def _make_trace(tmp_path, *, categories="") -> RobotTrace:
    trace = RobotTrace(env={"ROBOT_TRACE": categories, "ROBOT_TRACE_DIR": str(tmp_path)})
    trace.start_run()
    return trace


def _build_fake_engine(tmp_path, *, robot_xy=(6.57, 3.58)) -> SimpleNamespace:
    """Mirrors test_pending_path_invalidated_by_replan.py's _build_fake_engine(),
    extended with a real RobotTrace file sink."""
    robot = _FakeRobot(x=robot_xy[0], y=robot_xy[1])
    agent = _make_agent(position=robot_xy)

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        config=SimpleNamespace(goal_tolerance=0.25, planner_type="AStar", path_simplifier="RDP"),
        mapped_obstacle_points=[],
        simulation_time=5.0,
        console_logs=[],
        request_route_async_calls=[],
        current_exploration_target=None,
        exploration_targets=[],
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.robot_trace = _make_trace(tmp_path, categories="route,decision")
    fake.canvas = SimpleNamespace(
        set_exploration_target=lambda target: fake.exploration_targets.append(target),
        set_planned_path=lambda points: None,
    )
    fake.is_exploration_mode = lambda: True
    fake.runtime_agent = lambda robot_index=None: fake.agent
    fake.set_robot_goal_or_waypoints = lambda robot_obj, waypoints: robot_obj.set_waypoints(
        waypoints or [(robot_obj.x, robot_obj.y)]
    )

    def _spy_request_route_async(reason, *, target_override=None):
        fake.request_route_async_calls.append((reason, target_override))
        return False

    def _spy_replan_after_new_information(reason):
        fake.request_route_async_calls.append((reason, "route_repair_goal"))
        return True

    fake.request_route_async = _spy_request_route_async
    fake.replan_after_new_information = _spy_replan_after_new_information
    fake.safety_replan_cooldown_seconds = lambda: 1.5
    fake.apply_navigation_decision = SimulationControllerMixin.apply_navigation_decision.__get__(fake)
    return fake


def _build_fake_engine_for_route_result(tmp_path, *, robot_xy=(0.0, 0.0)) -> SimpleNamespace:
    """A fake sized for apply_route_result()/log_route_assignment(), the
    central success/failure handler for single-robot exploration mode."""
    robot = _FakeRobot(x=robot_xy[0], y=robot_xy[1])
    agent = _make_agent(position=robot_xy)

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        collision_checker=None,  # route_first_segment_blocked() short-circuits to False
        config=SimpleNamespace(
            goal_tolerance=0.25, planner_type="AStar", path_simplifier="RDP", grid_resolution=0.5,
        ),
        mapped_obstacle_points=[],
        simulation_time=5.0,
        route_result_count=0,
        console_logs=[],
        planned_paths=[],
        exploration_targets=[],
        statuses=[],
        current_exploration_target=None,
        last_goal_selection_reason="using final mission goal",
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.robot_trace = _make_trace(tmp_path, categories="route")
    fake.canvas = SimpleNamespace(
        set_planned_path=lambda points: fake.planned_paths.append(points),
        set_exploration_target=lambda target: fake.exploration_targets.append(target),
        set_status=lambda message: fake.statuses.append(message),
    )
    fake.is_exploration_mode = lambda: True
    fake.runtime_agent = lambda robot_index=None: fake.agent
    fake.final_goal_xy = lambda: (0.0, 0.0)
    fake.safety_radius = lambda: 0.2
    fake.planner_label = lambda: "AStar"
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.clean_waypoints_for_current_start = SimulationControllerMixin.clean_waypoints_for_current_start.__get__(fake)
    fake.log_route_assignment = SimulationControllerMixin.log_route_assignment.__get__(fake)
    fake.apply_route_result = SimulationControllerMixin.apply_route_result.__get__(fake)
    fake.sanitize_planner_obstacle_points = SimulationControllerMixin.sanitize_planner_obstacle_points.__get__(fake)
    fake.obstacle_points_for_segment_safety_check = (
        SimulationControllerMixin.obstacle_points_for_segment_safety_check.__get__(fake)
    )
    return fake


def _route_events_rows(trace: RobotTrace) -> list[dict]:
    trace.writer.flush()  # AsyncTraceWriter: wait for the background queue to drain
    with open(trace.file_output_dir / "route_events.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _route_affected_events_rows(trace: RobotTrace) -> list[dict]:
    trace.writer.flush()
    with open(trace.file_output_dir / "route_affected_events.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _jsonl_events(trace: RobotTrace) -> list[dict]:
    trace.writer.flush()
    with open(trace.file_output_dir / "belief_events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_summary(trace: RobotTrace) -> dict:
    trace.writer.flush_summary()  # enqueues a forced run_summary.json rewrite
    trace.writer.flush()  # wait for the background queue (including the above) to drain
    with open(trace.file_output_dir / "run_summary.json", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# A. Every route-ok app-log path writes a route_events.csv row + JSONL event
#    -- both the "normal" log_route_assignment() path and the previously-
#    uninstrumented ACCEPT_PENDING_PATH (prefetch) path.
# ---------------------------------------------------------------------------


def test_every_route_ok_app_log_path_writes_route_event(tmp_path):
    fake = _build_fake_engine_for_route_result(tmp_path)
    fake.apply_route_result(True, "exploration target", [(1.0, 0.0), (2.0, 0.0)])

    rows = _route_events_rows(fake.robot_trace)
    assert len(rows) == 1
    assert rows[0]["result"] == "ok"

    # ACCEPT_PENDING_PATH: a promoted prefetched path bypasses
    # log_route_assignment() entirely, but must still be recorded.
    fake2 = _build_fake_engine(tmp_path / "accept")
    fake2.agent.pending_path = [(3.0, 0.0), (4.0, 0.0)]
    fake2.agent.pending_target_xy = (4.0, 0.0)
    decision = SimpleNamespace(
        kind="ACCEPT_PENDING_PATH", reason="pending path ready", target=(4.0, 0.0),
        brake=False, force_new_target=False,
    )
    SimulationControllerMixin.apply_navigation_decision(fake2, fake2.robot, fake2.agent, decision)

    rows2 = _route_events_rows(fake2.robot_trace)
    assert len(rows2) == 1
    assert rows2[0]["result"] == "ok"
    assert rows2[0]["reason"] == "accepted pending path (prefetch)"


# ---------------------------------------------------------------------------
# B. A route_affected=yes occurrence (either outcome) writes a
#    route_affected_events.csv row + JSONL event with the matching action.
# ---------------------------------------------------------------------------


def test_route_affected_map_update_writes_route_affected_event(tmp_path):
    trace = _make_trace(tmp_path, categories="safety")

    assert trace.trace_route_affected(
        sim_time=90.5, path_goal=PATH_GOAL_A, active=(6.5, 3.5),
        mapped_obs=1771, new_obstacle_count=20, bbox=(8.7, 3.8, 9.1, 4.0),
        action="repair_requested",
    ) is True

    rows = _route_affected_events_rows(trace)
    assert len(rows) == 1
    assert rows[0]["action"] == "repair_requested"
    assert rows[0]["new_obstacle_count"] == "20"
    assert rows[0]["bbox_min_x"] == "8.7"

    events = [e for e in _jsonl_events(trace) if e["event_type"] == "route_affected"]
    assert len(events) == 1
    assert events[0]["payload"]["action"] == "repair_requested"


# ---------------------------------------------------------------------------
# C. A REPLAN_FOR_SAFETY decision increments total_safety_replans.
# ---------------------------------------------------------------------------


def test_safety_replan_decision_updates_summary_counter(tmp_path):
    fake = _build_fake_engine(tmp_path)
    fake.agent.assign_path(target=PATH_GOAL_A, waypoints=[(6.9, 3.4), PATH_GOAL_A], planner_reason="initial route")

    decision = SimpleNamespace(
        kind="REPLAN_FOR_SAFETY", reason="active segment blocked", target=PATH_GOAL_A,
        brake=True, force_new_target=False,
    )
    SimulationControllerMixin.apply_navigation_decision(fake, fake.robot, fake.agent, decision)

    summary = _run_summary(fake.robot_trace)  # also flushes the background queue
    assert summary["total_safety_replans"] == 1

    decision_rows_path = fake.robot_trace.file_output_dir / "decision_events.csv"
    with open(decision_rows_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert any(row["kind"] == "REPLAN_FOR_SAFETY" for row in rows)


# ---------------------------------------------------------------------------
# D. run_summary.json's route counters always match route_events.csv.
# ---------------------------------------------------------------------------


def test_run_summary_matches_route_csv_counts(tmp_path):
    trace = _make_trace(tmp_path, categories="route")
    trace.trace_route(sim_time=1.0, robot_label="R1", result="ok", start=(0, 0), goal=(1, 1))
    trace.trace_route(sim_time=2.0, robot_label="R1", result="ok", start=(0, 0), goal=(1, 1))
    trace.trace_route(sim_time=3.0, robot_label="R1", result="fail", start=(0, 0), goal=(1, 1), reason="no_path")

    summary = _run_summary(trace)
    rows = _route_events_rows(trace)

    assert summary["total_route_ok"] == sum(1 for r in rows if r["result"] == "ok")
    assert summary["total_route_fail"] == sum(1 for r in rows if r["result"] == "fail")


# ---------------------------------------------------------------------------
# E. run_summary.json's total_route_affected always matches
#    route_affected_events.csv's row count.
# ---------------------------------------------------------------------------


def test_run_summary_matches_route_affected_counts(tmp_path):
    trace = _make_trace(tmp_path, categories="safety")
    trace.trace_route_affected(sim_time=1.0, path_goal=PATH_GOAL_A, action="repair_throttled")
    trace.trace_route_affected(sim_time=2.0, path_goal=PATH_GOAL_A, action="repair_requested")
    trace.trace_route_affected(sim_time=3.0, path_goal=PATH_GOAL_A, action="repair_throttled")

    summary = _run_summary(trace)
    rows = _route_affected_events_rows(trace)

    assert summary["total_route_affected"] == len(rows) == 3


# ---------------------------------------------------------------------------
# F. A route_affected repair that is allowed to proceed (not throttled) is
#    recorded, and the eventual successful route is recorded too.
# ---------------------------------------------------------------------------


def test_route_repair_success_is_recorded(tmp_path):
    trace = _make_trace(tmp_path, categories="route,safety")

    # Mirrors simulation_step()'s route_affected "allowed" branch (the
    # previously-missing call).
    trace.trace_route_affected(
        sim_time=10.0, path_goal=PATH_GOAL_A, active=(6.5, 3.5),
        mapped_obs=500, new_obstacle_count=5, bbox=(1.0, 1.0, 2.0, 2.0),
        action="repair_requested",
    )
    # Mirrors the eventual apply_route_result()/log_route_assignment() call
    # once the async repair route arrives.
    trace.trace_route(
        sim_time=10.4, robot_label="R1", result="ok", start=(6.5, 3.5), goal=PATH_GOAL_A,
        waypoint_count=2, length=1.5,
    )

    affected_rows = _route_affected_events_rows(trace)
    route_rows = _route_events_rows(trace)
    assert affected_rows[0]["action"] == "repair_requested"
    assert route_rows[0]["result"] == "ok"

    summary = _run_summary(trace)
    assert summary["total_route_affected"] == 1
    assert summary["total_route_ok"] == 1


# ---------------------------------------------------------------------------
# G. A planner failure followed by a later successful route (recovery) --
#    both get recorded, with the correct ok/fail counts.
# ---------------------------------------------------------------------------


def test_recovered_after_planner_failure_route_ok_is_recorded(tmp_path):
    fake = _build_fake_engine_for_route_result(tmp_path)
    fake.agent.set_exploration_target((5.0, 5.0), reason="test target")

    fake.apply_route_result(False, "no path found", [])
    fake.apply_route_result(True, "exploration target", [(1.0, 0.0), (2.0, 0.0)])

    rows = _route_events_rows(fake.robot_trace)
    assert [row["result"] for row in rows] == ["fail", "ok"]

    summary = _run_summary(fake.robot_trace)
    assert summary["total_route_fail"] == 1
    assert summary["total_route_ok"] == 1
