"""
Tests for NavigationDebugEventLog -- the bounded ring buffer navigation
debug events are pushed into. These are pure contract-level tests against a
fabricated NavigationDebugSnapshot; producer wiring is covered separately in
test_navigation_debug_route_validation.py.
"""
from __future__ import annotations

from robotics_sim.diagnostics.event_log import NavigationDebugEvent, NavigationDebugEventLog
from robotics_sim.diagnostics.navigation_snapshot import (
    ControllerDebug,
    FrontierDebug,
    Maybe,
    NavigationDebugEventKind,
    NavigationDebugSnapshot,
    PathDebug,
    PlanningGridDebug,
    Pose,
    PredictedMotionDebug,
    RouteValidationDebug,
    SafetyDebug,
)


def _make_snapshot(snapshot_id: int) -> NavigationDebugSnapshot:
    return NavigationDebugSnapshot(
        snapshot_id=snapshot_id,
        simulation_time=float(snapshot_id),
        robot_id="R1",
        navigation_state="moving",
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        robot_pose=Pose(x=0.0, y=0.0, theta=0.0, v=0.0),
        path=PathDebug(
            raw_path=Maybe.missing(),
            simplified_path=Maybe.missing(),
            active_path=(),
            pending_path=(),
            active_segment=None,
            active_waypoint_index=None,
            planner_name=Maybe.missing(),
            simplifier_name=Maybe.missing(),
        ),
        route=RouteValidationDebug(first_segment=Maybe.missing(), endpoint_reaches_goal=None),
        predicted_motion=PredictedMotionDebug(trajectory=Maybe.missing(), collision=Maybe.missing()),
        safety=SafetyDebug(robot_radius=0.2, safety_radius=0.3, active_segment=Maybe.missing()),
        planning_grid=PlanningGridDebug(
            start_cell=Maybe.missing(),
            start_cell_world=Maybe.missing(),
            first_waypoint_cell=Maybe.missing(),
            first_waypoint_world=Maybe.missing(),
            unknown_is_traversable=Maybe.missing(),
            start_cell_cleared=Maybe.missing(),
        ),
        controller=ControllerDebug(
            v=0.0, omega=0.0, acceleration=0.0, heading_error=Maybe.missing(), distance_to_goal=Maybe.missing()
        ),
        frontier=FrontierDebug(
            candidate_count=Maybe.missing(),
            selected_target=Maybe.missing(),
            selected_score=Maybe.missing(),
            reason=Maybe.missing(),
        ),
    )


def test_bound_is_respected_oldest_evicted_first():
    log = NavigationDebugEventLog(max_size=3)
    for i in range(5):
        log.record(NavigationDebugEventKind.TICK, _make_snapshot(i))

    assert len(log) == 3
    ids = [event.snapshot.snapshot_id for event in log.events()]
    assert ids == [2, 3, 4], "only the 3 most recent events survive"


def test_latest_survives_repeated_reads_with_no_new_record_calls():
    log = NavigationDebugEventLog(max_size=5)
    log.record(NavigationDebugEventKind.PREDICTED_COLLISION, _make_snapshot(1))

    first_read = log.latest()
    for _ in range(10):
        # Simulates a paused simulation: no record() calls happen, but the
        # canvas/HUD may re-read latest() every repaint.
        assert log.latest() is first_read

    assert log.latest().snapshot.snapshot_id == 1
    assert log.latest().event_kind is NavigationDebugEventKind.PREDICTED_COLLISION


def test_idle_tick_never_replaces_last_relevant_event_because_nothing_calls_record():
    log = NavigationDebugEventLog(max_size=5)
    log.record(NavigationDebugEventKind.ROUTE_REJECTED, _make_snapshot(7))

    # An "idle" frame is modeled here as simply not calling record() again --
    # the producer (engine.py) only calls record() from actual tick/route
    # assembly, never from rendering, so this is the real invariant under
    # test, not a simulation of the GUI.
    assert len(log) == 1
    assert log.latest().snapshot.snapshot_id == 7


def test_event_at_supports_indexed_history_access():
    log = NavigationDebugEventLog(max_size=5)
    for i in range(3):
        log.record(NavigationDebugEventKind.TICK, _make_snapshot(i))

    assert log.event_at(0).snapshot.snapshot_id == 0
    assert log.event_at(2).snapshot.snapshot_id == 2
    assert log.event_at(3) is None
    assert log.event_at(-1) is None


def test_empty_log_reports_correctly():
    log = NavigationDebugEventLog(max_size=5)
    assert len(log) == 0
    assert log.latest() is None
    assert log.events() == ()


def test_record_returns_frozen_event_wrapper():
    log = NavigationDebugEventLog(max_size=5)
    log.record(NavigationDebugEventKind.HOLD, _make_snapshot(1))
    event = log.latest()
    assert isinstance(event, NavigationDebugEvent)
    assert event.snapshot.decision_kind == "FOLLOW_PATH"
