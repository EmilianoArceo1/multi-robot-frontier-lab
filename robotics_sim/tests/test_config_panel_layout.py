"""Regression coverage for dynamically visible configuration fields."""

from PySide6.QtWidgets import QApplication, QGridLayout

from robotics_sim.app.main_window import MainWindow


_app = QApplication.instance() or QApplication([])
_window = MainWindow()


def _grid_position(widget) -> tuple[int, int, int, int]:
    layout = _window.simulation_options_grid
    assert isinstance(layout, QGridLayout)
    index = layout.indexOf(widget)
    assert index >= 0
    return layout.getItemPosition(index)


def test_vision_model_and_path_simplifier_have_distinct_rows():
    vision_row, *_ = _grid_position(_window.vision_model_field)
    simplifier_row, *_ = _grid_position(_window.path_simplifier_field)

    assert vision_row != simplifier_row


def test_enabling_astar_keeps_vision_model_visible_and_separate():
    _window.planner_combo.setCurrentText("Direct")
    _window.update_relevant_parameter_visibility()
    assert _window.path_simplifier_field.isHidden()

    _window.planner_combo.setCurrentText("A*")
    _window.update_relevant_parameter_visibility()

    assert not _window.vision_model_field.isHidden()
    assert not _window.path_simplifier_field.isHidden()
    assert _grid_position(_window.vision_model_field)[0] != _grid_position(
        _window.path_simplifier_field
    )[0]


def test_frontier_reasoning_tab_is_menu_controlled_not_overlay_controlled():
    _window.on_frontier_reasoning_panel_visibility_toggled(False)
    assert _window.canvas.frontier_reasoning_overlay_enabled is False
    _window.frontier_decisions_toggle.setChecked(False)
    _window.frontier_decisions_toggle.setChecked(True)
    assert _window.canvas.frontier_decisions_enabled is True
    assert _window.side_panel_tabs.indexOf(_window.frontier_reasoning_panel) == -1

    _window.frontier_reasoning_panel_action.setChecked(True)
    assert _window.side_panel_tabs.indexOf(_window.frontier_reasoning_panel) >= 0
    assert _window.canvas.frontier_reasoning_overlay_enabled is True
    assert _window.frontier_decisions_toggle.isChecked() is True

    _window.frontier_reasoning_panel_action.setChecked(False)
    assert _window.side_panel_tabs.indexOf(_window.frontier_reasoning_panel) == -1
    assert _window.canvas.frontier_reasoning_overlay_enabled is False
    assert _window.frontier_decisions_toggle.isChecked() is True

    _window.frontier_decisions_toggle.setChecked(False)


def test_path_reasoning_is_an_independent_optional_tab():
    _window.on_path_reasoning_panel_visibility_toggled(False)
    assert _window.side_panel_tabs.indexOf(_window.path_reasoning_panel) == -1
    _window.path_reasoning_panel_action.setChecked(True)
    assert _window.side_panel_tabs.indexOf(_window.path_reasoning_panel) >= 0
    _window.path_reasoning_panel_action.setChecked(False)
    assert _window.side_panel_tabs.indexOf(_window.path_reasoning_panel) == -1


def test_coordinator_reasoning_is_optional_and_only_available_in_multiple_mode():
    previous_mode = _window.top_bar.mode_selector.currentText()
    try:
        _window.top_bar.mode_selector.setCurrentText("Multiple Robot Mode")
        _window.update_relevant_parameter_visibility()
        assert _window.coordinator_reasoning_panel_action.isVisible()
        _window.coordinator_reasoning_panel_action.setChecked(True)
        assert _window.side_panel_tabs.indexOf(_window.coordinator_reasoning_panel) >= 0

        _window.top_bar.mode_selector.setCurrentText("Single Robot Mode")
        _window.update_relevant_parameter_visibility()
        assert not _window.coordinator_reasoning_panel_action.isVisible()
        assert not _window.coordinator_reasoning_panel_action.isChecked()
        assert _window.side_panel_tabs.indexOf(_window.coordinator_reasoning_panel) == -1
    finally:
        _window.top_bar.mode_selector.setCurrentText(previous_mode)
        _window.update_relevant_parameter_visibility()


def test_mouse_coordinates_can_be_disabled_from_configuration():
    assert _window.cursor_coordinates_toggle.isChecked() is True
    _window.cursor_coordinates_toggle.setChecked(False)
    assert _window.canvas.cursor_coordinates_enabled is False
    _window.cursor_coordinates_toggle.setChecked(True)
    assert _window.canvas.cursor_coordinates_enabled is True
