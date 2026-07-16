"""
Tests for the primary Navigation Debug / snapshot control bar docked above
the canvas header (main_window._build_navigation_snapshot_bar()):

- it replaces the old canvas-painted eye icon as the activation control;
- it is a real widget positioned above (not overlaid on) the canvas;
- its counter/button state is driven entirely by engine.update_navigation_
  debug_step_buttons() through _update_navigation_snapshot_bar_state();
- hold-to-repeat on `<`/`>` ramps 1x -> 20x and the counter shows the
  multiplier once it reaches 2x.

A single MainWindow() is constructed once for the module (like the shared
_app QApplication other test_navigation_debug_*.py files already use) --
construction takes ~1s, and everything here is a read-only inspection of
wiring/state, so sharing one instance keeps this file fast without giving
up real integration coverage. Never .show()'n.
"""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QFrame, QVBoxLayout

from robotics_sim.app.main_window import MainWindow
from robotics_sim.app.widgets import ToggleSwitch

_app = QApplication.instance() or QApplication([])
_window = MainWindow()


# ---------------------------------------------------------------------------
# The bar exists, is a real widget, and sits above the canvas -- not
# overlaid on it like the runtime action bar / legacy eye icon were.
# ---------------------------------------------------------------------------


def test_snapshot_bar_is_a_real_widget_docked_above_the_canvas():
    bar = _window.navigation_snapshot_bar
    assert isinstance(bar, QFrame)

    canvas_column = _window.canvas.parentWidget()
    assert bar.parentWidget() is canvas_column, "bar and canvas must share the same column container"

    layout = canvas_column.layout()
    assert isinstance(layout, QVBoxLayout)
    bar_index = layout.indexOf(bar)
    canvas_index = layout.indexOf(_window.canvas)
    assert bar_index != -1 and canvas_index != -1
    assert bar_index < canvas_index, "the snapshot bar must come before the canvas in the column layout"


def test_snapshot_bar_contains_switch_step_buttons_counter_and_resume():
    assert isinstance(_window.navigation_snapshot_switch, ToggleSwitch)
    assert _window.navigation_snapshot_back_button.text() == "<"
    assert _window.navigation_snapshot_forward_button.text() == ">"
    assert _window.navigation_snapshot_resume_button.text() == "Resume from snapshot"
    assert _window.navigation_snapshot_counter_label.parentWidget() is _window.navigation_snapshot_bar


# ---------------------------------------------------------------------------
# The switch is the activator now -- the canvas eye icon is gone entirely.
# ---------------------------------------------------------------------------


def test_switch_is_wired_to_on_navigation_debug_toggled():
    assert _window.navigation_debug_enabled is False
    try:
        _window.navigation_snapshot_switch.setChecked(True)
        assert _window.navigation_debug_enabled is True
    finally:
        _window.navigation_snapshot_switch.setChecked(False)
        assert _window.navigation_debug_enabled is False


def test_canvas_no_longer_has_an_eye_icon_activator():
    assert not hasattr(_window.canvas, "navigationDebugToggleRequested")
    assert not hasattr(_window.canvas, "navigation_debug_eye_rect")
    assert not hasattr(_window.canvas, "draw_navigation_debug_eye_button")
    assert not hasattr(_window, "on_navigation_debug_eye_clicked")


# ---------------------------------------------------------------------------
# Counter text / button enabled state -- driven by _update_navigation_
# snapshot_bar_state(), the exact function engine.update_navigation_debug_
# step_buttons() calls. Exercised directly here (isolated from engine
# plumbing, which test_navigation_snapshot_restore.py already covers).
# ---------------------------------------------------------------------------


def _set_bar_state(**overrides):
    defaults = dict(
        navigation_enabled=True,
        position=None,
        total=0,
        back_enabled=False,
        forward_enabled=False,
        multiplier=1.0,
        resume_enabled=False,
        resume_reason="",
    )
    defaults.update(overrides)
    _window._update_navigation_snapshot_bar_state(**defaults)


def test_counter_shows_off_when_navigation_disabled():
    _set_bar_state(navigation_enabled=False)
    assert _window.navigation_snapshot_counter_label.text() == "OFF"


def test_counter_shows_live_when_enabled_with_no_history():
    _set_bar_state(navigation_enabled=True, position=None, total=0)
    assert _window.navigation_snapshot_counter_label.text() == "LIVE"


def test_counter_shows_position_over_total_in_history():
    _set_bar_state(navigation_enabled=True, position=2, total=10, multiplier=1.0)
    assert _window.navigation_snapshot_counter_label.text() == "3/10"  # 0-based index 2 -> 1-based 3


def test_counter_omits_multiplier_below_two_x():
    _set_bar_state(navigation_enabled=True, position=2, total=10, multiplier=1.9)
    assert "x" not in _window.navigation_snapshot_counter_label.text()


def test_counter_shows_multiplier_from_two_x():
    _set_bar_state(navigation_enabled=True, position=2, total=10, multiplier=4.0)
    assert _window.navigation_snapshot_counter_label.text() == "3/10 · x4"


def test_resume_button_disabled_in_live():
    _set_bar_state(resume_enabled=False, resume_reason="Select a historical snapshot to resume from.")
    assert _window.navigation_snapshot_resume_button.isEnabled() is False
    assert "historical snapshot" in _window.navigation_snapshot_resume_button.toolTip().lower()


def test_resume_button_enabled_in_history():
    _set_bar_state(position=2, total=10, resume_enabled=True, resume_reason="")
    assert _window.navigation_snapshot_resume_button.isEnabled() is True


def test_resume_button_disabled_for_multi_robot_with_explanatory_tooltip():
    """Restore stays single-robot-only in this version (see engine.
    restore_navigation_debug_snapshot()'s docstring) -- can_restore_
    navigation_debug_snapshot() is the source of this exact reason string;
    this only checks it reaches the real button widget's tooltip."""
    _set_bar_state(
        position=2,
        total=10,
        resume_enabled=False,
        resume_reason="Resume from snapshot supports single-robot mode only.",
    )
    assert _window.navigation_snapshot_resume_button.isEnabled() is False
    assert "single-robot" in _window.navigation_snapshot_resume_button.toolTip().lower()


def test_can_restore_reports_multi_robot_reason_from_a_real_window():
    """End-to-end through the real engine method (not a hand-built
    fixture): can_restore_navigation_debug_snapshot() on a MainWindow whose
    config is genuinely in Multiple Robot Mode must refuse with the
    single-robot-only reason, regardless of history state."""
    _window.on_navigation_debug_toggled(True)
    original_agent_mode = _window.config.agent_mode
    try:
        _window.config.agent_mode = "Multiple Robot Mode"
        can_restore, reason = _window.can_restore_navigation_debug_snapshot()
        assert can_restore is False
        assert "single-robot" in reason.lower()
    finally:
        _window.config.agent_mode = original_agent_mode
        _window.on_navigation_debug_toggled(False)


def test_back_forward_buttons_reflect_border_clamping():
    _set_bar_state(position=0, total=5, back_enabled=False, forward_enabled=True)
    assert _window.navigation_snapshot_back_button.isEnabled() is False
    assert _window.navigation_snapshot_forward_button.isEnabled() is True

    _set_bar_state(position=4, total=5, back_enabled=True, forward_enabled=False)
    assert _window.navigation_snapshot_back_button.isEnabled() is True
    assert _window.navigation_snapshot_forward_button.isEnabled() is False


def test_resume_button_click_delegates_to_restore_navigation_debug_snapshot():
    calls = []
    original = _window.restore_navigation_debug_snapshot
    try:
        _window.restore_navigation_debug_snapshot = lambda: calls.append(True)
        _window.on_resume_from_snapshot_clicked()
    finally:
        _window.restore_navigation_debug_snapshot = original
    assert calls == [True]


# ---------------------------------------------------------------------------
# Hold-to-repeat ramp: 1x -> 20x, shown once it reaches 2x (see the counter
# tests above for the display half of this).
# ---------------------------------------------------------------------------


def test_hold_acceleration_starts_at_one_x():
    assert MainWindow.navigation_history_scrub_multiplier(0.0) == 1.0


def test_hold_acceleration_ramps_up_before_the_cap():
    half_ramp = MainWindow.navigation_history_scrub_multiplier(0.9, ramp_seconds=1.8)
    assert 1.0 < half_ramp < 20.0


def test_hold_acceleration_caps_at_twenty_x():
    at_ramp_end = MainWindow.navigation_history_scrub_multiplier(1.8, ramp_seconds=1.8)
    well_past_ramp_end = MainWindow.navigation_history_scrub_multiplier(100.0, ramp_seconds=1.8)
    assert at_ramp_end == 20.0
    assert well_past_ramp_end == 20.0


def test_hold_acceleration_never_exceeds_twenty_x_across_the_whole_ramp():
    for elapsed_ms in range(0, 5000, 25):
        multiplier = MainWindow.navigation_history_scrub_multiplier(elapsed_ms / 1000.0, ramp_seconds=1.8)
        assert 1.0 <= multiplier <= 20.0


# ---------------------------------------------------------------------------
# Capture (navigation_debug_enabled, driven by the switch) and panel
# visibility (_navigation_reasoning_panel_visible, driven by the gear-menu
# "Navigation Reasoning" action / the panel's own close button) are two
# independent pieces of state. Neither handler may touch the other's state.
# Each test restores whatever it changed so order doesn't matter.
# ---------------------------------------------------------------------------


def test_toggling_capture_does_not_change_panel_visibility_state():
    _window.on_navigation_reasoning_panel_visibility_toggled(False)
    try:
        _window.on_navigation_debug_toggled(True)
        assert _window._navigation_reasoning_panel_visible is False
        _window.on_navigation_debug_toggled(False)
        assert _window._navigation_reasoning_panel_visible is False
    finally:
        _window.on_navigation_debug_toggled(False)


def test_toggling_capture_does_not_move_the_panel_action_checkbox():
    _window.on_navigation_reasoning_panel_visibility_toggled(True)
    try:
        _window.on_navigation_debug_toggled(True)
        assert _window.navigation_reasoning_panel_action.isChecked() is True
        _window.on_navigation_debug_toggled(False)
        assert _window.navigation_reasoning_panel_action.isChecked() is True
    finally:
        _window.on_navigation_debug_toggled(False)
        _window.on_navigation_reasoning_panel_visibility_toggled(False)


def test_closing_the_panel_does_not_disable_capture():
    _window.on_navigation_debug_toggled(True)
    _window.on_navigation_reasoning_panel_visibility_toggled(True)
    try:
        _window.on_navigation_reasoning_panel_visibility_toggled(False)  # closeRequested path
        assert _window.navigation_debug_enabled is True
    finally:
        _window.on_navigation_debug_toggled(False)


def test_panel_menu_action_never_touches_capture():
    _window.on_navigation_debug_toggled(False)
    try:
        _window.on_navigation_reasoning_panel_visibility_toggled(True)
        assert _window.navigation_debug_enabled is False
        _window.on_navigation_reasoning_panel_visibility_toggled(False)
        assert _window.navigation_debug_enabled is False
    finally:
        _window.on_navigation_reasoning_panel_visibility_toggled(False)


def test_disabling_capture_does_not_alter_the_selected_side_panel_tab():
    _window.on_navigation_reasoning_panel_visibility_toggled(True)
    _window.on_navigation_debug_toggled(True)
    try:
        tabs = _window.side_panel_tabs
        # Deliberately select a different tab before disabling capture.
        tabs.setCurrentWidget(_window.config_panel_stack)
        selected_before = tabs.currentWidget()

        _window.on_navigation_debug_toggled(False)

        assert tabs.currentWidget() is selected_before
    finally:
        _window.on_navigation_debug_toggled(False)
        _window.on_navigation_reasoning_panel_visibility_toggled(False)
