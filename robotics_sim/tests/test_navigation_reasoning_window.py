"""
Tests for:
- the standalone NavigationReasoningWindow (separate OS window, not drawn
  on the canvas -- so it cannot obstruct the map/title/FPS);
- on_async_route_ready() now persisting the worker's debug_capture before
  calling apply_route_result(), so planner/simplifier stay available after
  the first (synchronous) route once replanning-during-motion (always
  async for a non-Direct planner) takes over;
- the </> buttons' hold-to-repeat (Qt autoRepeat) configuration.
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
# Hold-to-repeat.
# ---------------------------------------------------------------------------


def test_step_buttons_have_auto_repeat_enabled():
    canvas = SimulationCanvas()
    assert canvas.navigation_debug_step_back_button.autoRepeat() is True
    assert canvas.navigation_debug_step_forward_button.autoRepeat() is True


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
