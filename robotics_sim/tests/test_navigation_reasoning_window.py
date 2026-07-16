"""
Tests for:
- the standalone NavigationReasoningWindow (separate OS window, not drawn
  on the canvas -- so it cannot obstruct the map/title/FPS);
- on_async_route_ready() now persisting the worker's debug_capture before
  calling apply_route_result(), so planner/simplifier stay available after
  the first (synchronous) route once replanning-during-motion (always
  async for a non-Direct planner) takes over;
- the panel no longer owning its own `<`/`>` history-step buttons -- that
  control now lives solely in main_window's navigation_snapshot_bar (see
  test_navigation_panel_controls.py for its hold-to-repeat behavior).
"""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from robotics_sim.app.navigation_reasoning_window import NavigationReasoningWindow
from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.diagnostics.capture import PlanDebugCapture
from robotics_sim.diagnostics.event_log import NavigationDebugEventKind
from robotics_sim.simulation.engine import SimulationControllerMixin

_app = QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# The canvas no longer draws a HUD card -- draw_navigation_debug_hud() must
# not exist as a callable draw method that paintEvent reaches.
# ---------------------------------------------------------------------------


def test_canvas_has_no_in_canvas_hud_draw_method():
    assert not hasattr(SimulationCanvas, "draw_navigation_debug_hud")


def test_canvas_forwards_snapshot_pushes_to_registered_reasoning_window():
    canvas = SimulationCanvas()
    received = []
    fake_window = SimpleNamespace(update_snapshot=lambda *args: received.append(args))
    canvas.set_navigation_reasoning_window(fake_window)

    canvas.set_navigation_debug_snapshot("a-snapshot")

    assert received and received[0][0] == "a-snapshot"


def test_reasoning_window_shows_placeholder_with_no_snapshot():
    window = NavigationReasoningWindow()
    assert "No navigation decisions" in window._label.text()


# ---------------------------------------------------------------------------
# No duplicate step controls: the panel keeps only the passive history
# label (still driven by update_snapshot()'s history_position argument);
# stepping itself belongs exclusively to navigation_snapshot_bar.
# ---------------------------------------------------------------------------


def test_reasoning_window_has_no_step_buttons_of_its_own():
    window = NavigationReasoningWindow()
    assert not hasattr(window, "step_back_button")
    assert not hasattr(window, "step_forward_button")
    assert not hasattr(window, "set_history_controls")


def test_canvas_has_no_history_step_buttons_of_its_own():
    canvas = SimulationCanvas()
    assert not hasattr(canvas, "navigation_debug_step_back_button")
    assert not hasattr(canvas, "navigation_debug_step_forward_button")


def test_reasoning_window_history_label_still_reflects_history_position():
    window = NavigationReasoningWindow()
    snapshot = SimpleNamespace(
        controller=SimpleNamespace(
            desired_heading=SimpleNamespace(unavailable=True, value=None),
            heading_error=SimpleNamespace(unavailable=True, value=None),
            nominal_control=SimpleNamespace(unavailable=True, value=None),
            applied_control=SimpleNamespace(unavailable=True, value=None),
            distance_to_goal=SimpleNamespace(unavailable=True, value=None),
            v=0.0,
            acceleration=0.0,
            omega=0.0,
        ),
        tracking_mode="TRACK",
        decision_kind="FOLLOW_PATH",
        robot_id="R1",
        simulation_time=1.0,
        snapshot_id=3,
        explanation="",
        decision_reason="",
        robot_pose=SimpleNamespace(x=0.0, y=0.0, theta=0.0),
        rotate_threshold=SimpleNamespace(unavailable=True, value=None),
        path=SimpleNamespace(
            active_waypoint_index=None,
            planner_name=SimpleNamespace(unavailable=True, value=None),
            simplifier_name=SimpleNamespace(unavailable=True, value=None),
        ),
        route=SimpleNamespace(first_segment=SimpleNamespace(unavailable=True, value=None)),
        safety=SimpleNamespace(
            active_segment=SimpleNamespace(unavailable=True, value=None),
            robot_radius=0.2,
            safety_radius=0.3,
        ),
        predicted_motion=SimpleNamespace(collision=SimpleNamespace(unavailable=True, value=None)),
        mapped_obstacle_points_count=0,
        navigation_state="moving",
    )

    # history_position's first element is already 1-based (see
    # engine._push_navigation_debug_history_view()'s index + 1).
    window.update_snapshot(snapshot, None, (3, 10))

    assert "3 of 10" in window._history_label.text()


# ---------------------------------------------------------------------------
# on_async_route_ready() persists the worker's plan capture.
# ---------------------------------------------------------------------------


def test_on_async_route_ready_stashes_worker_debug_capture_before_apply():
    captured_kwargs = {}

    def fake_apply_route_result(success, reason, waypoints):
        captured_kwargs["pending"] = fake._nav_debug_last_plan_capture

    plan_capture = PlanDebugCapture(planner_name="A*", simplifier_name="Direction changes")
    worker = SimpleNamespace(debug_capture=plan_capture)
    fake = SimpleNamespace(
        active_planner_workers={7: worker},
        route_request_id=7,
        planning_in_progress=True,
        apply_route_result=fake_apply_route_result,
    )
    fake.on_async_route_ready = SimulationControllerMixin.on_async_route_ready.__get__(fake)

    fake.on_async_route_ready(7, True, "ok", [(1.0, 0.0)])

    assert captured_kwargs["pending"] is plan_capture
