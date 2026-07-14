"""
Regression tests for PerfMonitor wiring inside engine.py -- specifically
the bug the manual Office.sim evidence found: route_check_ms stayed 0.0
despite route_affected=yes events appearing in the app log.

Root cause: the exhausted-hold throttle added in an earlier perf round
(_should_skip_for_exhausted_hold()) gated route_affected_check/
belief_snapshot/canvas updates behind
RobotAgent.exploration_exhausted_map_signature alone. That flag is only
ever CLEARED by exploration_exhausted() itself, which only runs from
ExplorationBehavior's "no active path" branch -- so once the agent
regained an active route (e.g. after a route_affected repair or a safety
replan succeeded) and moved into an active safety-replan-loop/recovering
episode, the STALE signature kept reading "not None" and kept suppressing
the very diagnostics needed to see that episode. Fixed by also requiring
active_path_goal_xy is None before treating the agent as "exhausted" for
throttling purposes (see engine.py's _should_skip_for_exhausted_hold()).

These tests exercise the real engine methods directly via lightweight
duck-typed fakes (the same pattern used throughout this test suite --
see test_trace_artifact_wiring.py/test_pending_path_invalidated_by_replan.py)
rather than the full simulation_step()/Qt/sensor stack.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.perf_monitor import PerfMonitor
from robotics_sim.simulation.telemetry import TelemetryLogger


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


# ---------------------------------------------------------------------------
# B. route_check_ms is recorded whenever the real route_affected=yes code
#    path (new_information_affects_current_route()) runs -- this is the
#    exact fix for the reported instrumentation gap.
# ---------------------------------------------------------------------------


def test_route_affected_yes_records_route_check_time():
    fake = SimpleNamespace()
    fake.new_information_affects_current_route = lambda points: True  # the real "route_affected=yes" branch
    fake._perf_monitor = PerfMonitor(env={})
    fake.ensure_perf_monitor = SimulationControllerMixin.ensure_perf_monitor.__get__(fake)
    fake._timed_route_affected_check = SimulationControllerMixin._timed_route_affected_check.__get__(fake)

    result = fake._timed_route_affected_check([(1.0, 1.0), (2.0, 2.0)])

    assert result is True
    assert fake.ensure_perf_monitor().average_ms("route_affected_check") >= 0.0
    # A real timing sample was recorded (not silently skipped) -- exactly
    # what was missing before: this can never read as "no samples yet".
    assert fake.ensure_perf_monitor()._section_count.get("route_affected_check", 0) == 1


# ---------------------------------------------------------------------------
# apply_route_result()/request_route_async()/log_console_message() each
# record their own timing section via a thin wrapper -- confirms the
# wrapper wiring (not just PerfMonitor in isolation) actually works.
# ---------------------------------------------------------------------------


def test_apply_route_result_records_route_result_handling_timing():
    robot = _FakeRobot(x=0.0, y=0.0)
    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        collision_checker=None,
        config=SimpleNamespace(goal_tolerance=0.25, planner_type="AStar", path_simplifier="RDP", grid_resolution=0.5),
        mapped_obstacle_points=[],
        simulation_time=1.0,
        route_result_count=0,
        current_exploration_target=None,
        last_goal_selection_reason="using final mission goal",
    )
    fake.telemetry = TelemetryLogger(sink=lambda message, **k: None)
    fake.canvas = SimpleNamespace(
        set_planned_path=lambda points: None,
        set_exploration_target=lambda target: None,
        set_status=lambda message: None,
    )
    fake.is_exploration_mode = lambda: True
    fake.runtime_agent = lambda robot_index=None: None
    fake.final_goal_xy = lambda: (0.0, 0.0)
    fake.safety_radius = lambda: 0.2
    fake.planner_label = lambda: "AStar"
    fake.log_console_message = lambda message, **k: None
    fake.clean_waypoints_for_current_start = SimulationControllerMixin.clean_waypoints_for_current_start.__get__(fake)
    fake._perf_monitor = PerfMonitor(env={})
    fake.ensure_perf_monitor = SimulationControllerMixin.ensure_perf_monitor.__get__(fake)
    fake.apply_route_result = SimulationControllerMixin.apply_route_result.__get__(fake)

    fake.apply_route_result(False, "no path found", [])

    assert fake.ensure_perf_monitor().average_ms("route_result_handling") >= 0.0
    assert fake.ensure_perf_monitor()._section_count.get("route_result_handling", 0) == 1


def test_request_route_async_direct_mode_records_planner_dispatch_timing():
    robot = _FakeRobot(x=0.0, y=0.0)
    fake = SimpleNamespace(
        robot=robot,
        config=SimpleNamespace(planner_type="Direct"),
        route_request_count=0,
    )
    fake.compute_route = lambda start_xy: (True, "ok", [(1.0, 0.0)])
    fake.apply_route_result = lambda success, reason, waypoints: None
    fake._perf_monitor = PerfMonitor(env={})
    fake.ensure_perf_monitor = SimulationControllerMixin.ensure_perf_monitor.__get__(fake)
    fake.request_route_async = SimulationControllerMixin.request_route_async.__get__(fake)

    result = fake.request_route_async("test reason")

    assert result is True
    assert fake.ensure_perf_monitor()._section_count.get("planner_dispatch", 0) == 1


def test_log_console_message_records_console_log_timing():
    fake = SimpleNamespace(canvas=SimpleNamespace(append_console_message=lambda message: None))
    fake._perf_monitor = PerfMonitor(env={})
    fake.ensure_perf_monitor = SimulationControllerMixin.ensure_perf_monitor.__get__(fake)
    fake.log_console_message = SimulationControllerMixin.log_console_message.__get__(fake)

    fake.log_console_message("hello")

    assert fake.ensure_perf_monitor()._section_count.get("console_log", 0) == 1


def test_on_async_route_ready_increments_planner_jobs_completed():
    fake = SimpleNamespace(
        active_planner_workers={1: object()},
        route_request_id=1,
        planning_in_progress=True,
    )
    fake.apply_route_result = lambda success, reason, waypoints: None
    fake.on_async_route_ready = SimulationControllerMixin.on_async_route_ready.__get__(fake)

    fake.on_async_route_ready(1, True, "ok", [(1.0, 0.0)])

    assert getattr(fake, "planner_jobs_completed", 0) == 1
    assert fake.planning_in_progress is False


# ---------------------------------------------------------------------------
# E. _compute_nav_state() distinguishes recovering/safety_replan_loop from
#    exhausted, using only existing, already-tracked RobotAgent fields.
# ---------------------------------------------------------------------------


def test_compute_nav_state_labels():
    fake = SimpleNamespace()
    fake._compute_nav_state = SimulationControllerMixin._compute_nav_state.__get__(fake)

    assert fake._compute_nav_state(None) == "idle"

    exhausted_agent = SimpleNamespace(
        active_path_goal_xy=None, exploration_exhausted_map_signature=5,
        consecutive_exploration_failures=3, route_repair_in_progress_for_goal=None,
    )
    assert fake._compute_nav_state(exhausted_agent) == "exhausted"

    safety_replan_agent = SimpleNamespace(
        active_path_goal_xy=(1.0, 1.0), exploration_exhausted_map_signature=5,
        consecutive_exploration_failures=1, route_repair_in_progress_for_goal=(2.0, 2.0),
    )
    assert fake._compute_nav_state(safety_replan_agent) == "safety_replan_loop"

    recovering_agent = SimpleNamespace(
        active_path_goal_xy=None, exploration_exhausted_map_signature=None,
        consecutive_exploration_failures=1, route_repair_in_progress_for_goal=None,
    )
    assert fake._compute_nav_state(recovering_agent) == "recovering"

    running_agent = SimpleNamespace(
        active_path_goal_xy=(3.0, 3.0), exploration_exhausted_map_signature=None,
        consecutive_exploration_failures=0, route_repair_in_progress_for_goal=None,
    )
    assert fake._compute_nav_state(running_agent) == "running"

    idle_agent = SimpleNamespace(
        active_path_goal_xy=None, exploration_exhausted_map_signature=None,
        consecutive_exploration_failures=0, route_repair_in_progress_for_goal=None,
    )
    assert fake._compute_nav_state(idle_agent) == "idle"
