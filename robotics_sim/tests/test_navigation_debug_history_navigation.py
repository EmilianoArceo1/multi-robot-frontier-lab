"""
Tests for stepping backward/forward through the bounded navigation-debug
event log (the </> history buttons) -- engine-side wiring only. Uses the
same lightweight duck-typed engine fake pattern as the other
test_navigation_debug_*.py files.
"""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from robotics_sim.diagnostics.event_log import NavigationDebugEventLog
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
from robotics_sim.simulation.engine import SimulationControllerMixin

# toggle_pause() -> update_start_pause_button() -> make_icon() constructs a
# QPixmap/QPainter, which aborts the process if no QApplication exists yet --
# same requirement as test_navigation_debug_canvas_wiring.py.
_app = QApplication.instance() or QApplication([])


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


class _FakeCanvas:
    def __init__(self):
        self.pushed_snapshots: list[int] = []
        self.pushed_events: list[NavigationDebugEventKind] = []
        self.history_positions: list[tuple[int | None, int]] = []

    def set_status(self, message):
        pass

    def set_navigation_debug_snapshot(self, snapshot):
        self.pushed_snapshots.append(snapshot.snapshot_id)

    def set_navigation_debug_last_event(self, event):
        self.pushed_events.append(event.event_kind)

    def set_navigation_debug_history_position(self, position, total):
        self.history_positions.append((position, total))


def _build_fake_engine(*, paused: bool, navigation_debug_enabled: bool = True, event_count: int = 5) -> SimpleNamespace:
    log = NavigationDebugEventLog(max_size=10)
    for i in range(event_count):
        log.record(NavigationDebugEventKind.HOLD, _make_snapshot(i))

    fake = SimpleNamespace(
        navigation_debug_enabled=navigation_debug_enabled,
        navigation_debug_log=log,
        paused=paused,
        _nav_debug_history_index=None,
        canvas=_FakeCanvas(),
    )
    for name in (
        "navigation_debug_history_length",
        "_push_navigation_debug_history_view",
        "step_navigation_debug_history",
        "resume_navigation_debug_live_view",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


def test_step_back_from_live_starts_at_latest_event():
    fake = _build_fake_engine(paused=True)

    fake.step_navigation_debug_history(-1)

    assert fake._nav_debug_history_index == 3  # latest is index 4; one step back -> 3
    assert fake.canvas.pushed_snapshots[-1] == 3


def test_step_forward_and_back_move_the_cursor():
    fake = _build_fake_engine(paused=True)

    fake.step_navigation_debug_history(-1)  # -> 3
    fake.step_navigation_debug_history(-1)  # -> 2
    fake.step_navigation_debug_history(1)  # -> 3

    assert fake._nav_debug_history_index == 3
    assert fake.canvas.pushed_snapshots == [3, 2, 3]


def test_step_clamps_at_bounds():
    fake = _build_fake_engine(paused=True, event_count=3)

    for _ in range(10):
        fake.step_navigation_debug_history(-1)
    assert fake._nav_debug_history_index == 0

    for _ in range(10):
        fake.step_navigation_debug_history(1)
    assert fake._nav_debug_history_index == 2


def test_step_is_noop_while_running():
    fake = _build_fake_engine(paused=False)

    fake.step_navigation_debug_history(-1)

    assert fake._nav_debug_history_index is None
    assert fake.canvas.pushed_snapshots == []


def test_step_is_noop_when_navigation_debug_disabled():
    fake = _build_fake_engine(paused=True, navigation_debug_enabled=False)

    fake.step_navigation_debug_history(-1)

    assert fake._nav_debug_history_index is None
    assert fake.canvas.pushed_snapshots == []


def test_step_is_noop_with_empty_log():
    fake = _build_fake_engine(paused=True, event_count=0)

    fake.step_navigation_debug_history(-1)

    assert fake._nav_debug_history_index is None
    assert fake.canvas.pushed_snapshots == []


def test_history_position_pushed_to_canvas_is_one_based():
    fake = _build_fake_engine(paused=True, event_count=5)

    fake.step_navigation_debug_history(-1)  # latest (index 4) -> one back -> index 3

    assert fake.canvas.history_positions[-1] == (4, 5)  # 1-based: index 3 -> position 4


def test_resume_live_view_clears_history_index_and_notifies_canvas():
    fake = _build_fake_engine(paused=True, event_count=5)
    fake.step_navigation_debug_history(-1)
    assert fake._nav_debug_history_index is not None

    fake.resume_navigation_debug_live_view()

    assert fake._nav_debug_history_index is None
    assert fake.canvas.history_positions[-1] == (None, 5)


def test_toggle_pause_resumes_live_view_on_resume():
    fake = SimpleNamespace(
        running=True,
        paused=True,
        navigation_debug_enabled=True,
        navigation_debug_log=NavigationDebugEventLog(max_size=10),
        _nav_debug_history_index=2,
        canvas=_FakeCanvas(),
        top_bar=SimpleNamespace(set_status=lambda *_a, **_k: None),
        start_button=SimpleNamespace(setText=lambda *_a: None, setIcon=lambda *_a: None),
    )
    fake.navigation_debug_log.record(NavigationDebugEventKind.HOLD, _make_snapshot(0))
    for name in (
        "toggle_pause",
        "update_start_pause_button",
        "update_navigation_debug_step_buttons",
        "navigation_debug_history_length",
        "resume_navigation_debug_live_view",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    fake.toggle_pause()  # paused=True -> False (resume)

    assert fake.paused is False
    assert fake._nav_debug_history_index is None
