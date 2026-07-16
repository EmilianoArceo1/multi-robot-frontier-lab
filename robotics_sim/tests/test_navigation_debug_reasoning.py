"""
Tests for the "navigation reasoning" additions: the eye-icon toggle on the
canvas, the desired_heading/nominal/applied control capture fields, and the
human-readable one-line explanation built purely from real snapshot fields
(engine._navigation_debug_explanation()).
"""
from __future__ import annotations

import math

from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.diagnostics.navigation_snapshot import (
    ClearanceTerms,
    ControllerDebug,
    Maybe,
    PredictedMotionDebug,
    RouteValidationDebug,
    SafetyDebug,
)
from robotics_sim.simulation.engine import _navigation_debug_explanation

_app = QApplication.instance() or QApplication([])


def _controller(heading_error=None):
    return ControllerDebug(
        v=0.0,
        omega=0.0,
        acceleration=0.0,
        heading_error=Maybe.of(heading_error) if heading_error is not None else Maybe.missing(),
        distance_to_goal=Maybe.missing(),
    )


def _blank_safety():
    return SafetyDebug(robot_radius=0.2, safety_radius=0.35, active_segment=Maybe.missing())


def _blank_predicted():
    return PredictedMotionDebug(trajectory=Maybe.missing(), collision=Maybe.missing())


def _blank_route():
    return RouteValidationDebug(first_segment=Maybe.missing(), endpoint_reaches_goal=None)


# ---------------------------------------------------------------------------
# Explanation text matches the real condition -- not an inference.
# ---------------------------------------------------------------------------


def test_rotate_explanation_reports_real_heading_error_and_threshold():
    text = _navigation_debug_explanation(
        tracking_mode="ROTATE",
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        controller=_controller(heading_error=math.radians(20.9)),
        rotate_threshold=Maybe.of(math.radians(10.0)),
        safety=_blank_safety(),
        predicted_motion=_blank_predicted(),
        route=_blank_route(),
    )
    assert "20.9" in text
    assert "10.0" in text
    assert text.startswith("ROTATE:")


def test_track_explanation_when_aligned():
    text = _navigation_debug_explanation(
        tracking_mode="TRACK",
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        controller=_controller(),
        rotate_threshold=Maybe.missing(),
        safety=_blank_safety(),
        predicted_motion=_blank_predicted(),
        route=_blank_route(),
    )
    assert text.startswith("TRACK:")


def test_stop_explanation_reports_real_predicted_clearance():
    terms = ClearanceTerms(
        checker="check_predicted_motion",
        distance=Maybe.of(0.18),
        required_clearance=0.35,
        blocked=True,
        blocking_point=(1.0, 2.0),
        reason="predicted motion enters an expanded obstacle",
    )
    text = _navigation_debug_explanation(
        tracking_mode="TRACK",
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        controller=_controller(),
        rotate_threshold=Maybe.missing(),
        safety=_blank_safety(),
        predicted_motion=PredictedMotionDebug(trajectory=Maybe.missing(), collision=Maybe.of(terms)),
        route=_blank_route(),
    )
    assert "0.18" in text
    assert "0.35" in text
    assert text.startswith("STOP:")


def test_replan_explanation_uses_real_decision_reason():
    text = _navigation_debug_explanation(
        tracking_mode="TRACK",
        decision_kind="REPLAN_FOR_SAFETY",
        decision_reason="newly mapped obstacle affects current route",
        controller=_controller(),
        rotate_threshold=Maybe.missing(),
        safety=_blank_safety(),
        predicted_motion=_blank_predicted(),
        route=_blank_route(),
    )
    assert text.startswith("REPLAN:")
    assert "newly mapped obstacle affects current route" in text


# ---------------------------------------------------------------------------
# Eye icon: visible always, toggles overlay visibility, never touches
# simulation state.
# ---------------------------------------------------------------------------


def test_toggling_navigation_debug_changes_canvas_overlay_flag_only():
    """The activator is main_window.py's Navigation switch now (see
    _build_navigation_snapshot_bar()), not a canvas-painted eye icon --
    set_navigation_debug_enabled() is still the single method that flips
    the flag, and still mutates nothing else on the canvas."""
    canvas = SimulationCanvas()
    canvas.robot = None
    config_before = canvas.config

    assert canvas.navigation_debug_enabled is False
    canvas.set_navigation_debug_enabled(True)
    assert canvas.navigation_debug_enabled is True
    canvas.set_navigation_debug_enabled(False)
    assert canvas.navigation_debug_enabled is False

    assert canvas.config is config_before
    assert canvas.robot is None


def test_canvas_no_longer_exposes_the_eye_icon_activator():
    """Removed as the primary activator in favor of the Navigation switch
    docked above the canvas -- neither the geometry nor the click signal
    should exist anymore."""
    canvas = SimulationCanvas()
    assert not hasattr(canvas, "navigation_debug_eye_rect")
    assert not hasattr(canvas, "draw_navigation_debug_eye_button")
    assert not hasattr(canvas, "navigationDebugToggleRequested")
