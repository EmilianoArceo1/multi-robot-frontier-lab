"""Regression coverage for dynamically visible configuration fields."""

from types import SimpleNamespace

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QGridLayout

from robotics_sim.app.main_window import MainWindow
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.planning.ryu_frontier_graph_bfs import RYU_FRONTIER_GRAPH_BFS
from robotics_sim.simulation.config import (
    CLUSTERING_ALGORITHM_OPTIONS,
    FRONTIER_ALGORITHM_DETECTOR_OPTIONS,
    NO_CLUSTERING_ALGORITHM,
    NO_TASK_ASSIGN_ALGORITHM,
    REMOVED_FRONTIER_ALGORITHM_DETECTOR_OPTIONS,
    REMOVED_TASK_ASSIGN_ALGORITHM_OPTIONS,
    TASK_ASSIGN_ALGORITHM_OPTIONS,
    SAFETY_ALGORITHM_OPTIONS,
    WANG_AMES_BARRIER_CERTIFICATE,
    SIDE_PANEL_WIDTH,
    WINDOW_TARGET_HEIGHT,
    WINDOW_TARGET_WIDTH,
)


_app = QApplication.instance() or QApplication([])
_window = MainWindow()


def _grid_position(widget) -> tuple[int, int, int, int]:
    layout = _window.simulation_options_grid
    assert isinstance(layout, QGridLayout)
    index = layout.indexOf(widget)
    assert index >= 0
    return layout.getItemPosition(index)


def _combo_items(combo) -> list[str]:
    return [combo.itemText(index) for index in range(combo.count())]


def test_window_and_configuration_panel_fit_the_extended_selectors():
    assert WINDOW_TARGET_WIDTH >= 1440
    assert WINDOW_TARGET_HEIGHT >= 900
    assert SIDE_PANEL_WIDTH >= 520
    assert _window.side_panel_container.width() == SIDE_PANEL_WIDTH
    assert _window.minimumWidth() >= 1180
    assert _window.minimumHeight() >= 740


def test_combo_wheel_requires_click_and_resets_when_focus_is_lost():
    combo = _window.safety_algorithm_combo
    wheel = QEvent(QEvent.Wheel)

    assert _window.eventFilter(combo, wheel) is True

    _window.eventFilter(combo, QEvent(QEvent.MouseButtonPress))
    assert _window.eventFilter(combo, wheel) is False

    _window.eventFilter(combo, QEvent(QEvent.FocusOut))
    assert _window.eventFilter(combo, wheel) is True


def test_pipeline_algorithm_fields_use_the_new_responsibility_names():
    assert _window.exploration_planner_field.field_label_base_text == (
        "Frontier Algorithm Detector"
    )
    assert _window.coordinator_field.field_label_base_text == "Task Assign Algorithm"
    assert _window.exploration_planner_field.field_label.text().startswith(
        "Frontier Algorithm Detector"
    )
    assert _window.coordinator_field.field_label.text() == "Task Assign Algorithm"


def test_clustering_stage_lists_the_registered_cited_algorithm():
    items = _combo_items(_window.clustering_algorithm_combo)

    assert _window.clustering_algorithm_field.field_label_base_text == (
        "Clustering Algorithm"
    )
    assert len(CLUSTERING_ALGORITHM_OPTIONS) == 1
    assert items == list(CLUSTERING_ALGORITHM_OPTIONS)
    assert _window.clustering_algorithm_combo.currentIndex() == 0
    assert _window.read_config().clustering_algorithm == CLUSTERING_ALGORITHM_OPTIONS[0]


def test_removed_frontier_algorithms_are_not_selectable():
    items = _combo_items(_window.exploration_planner_combo)

    assert items == list(FRONTIER_ALGORITHM_DETECTOR_OPTIONS)
    assert not REMOVED_FRONTIER_ALGORITHM_DETECTOR_OPTIONS.intersection(items)
    assert RYU_FRONTIER_GRAPH_BFS in items
    assert _window.exploration_planner_combo.currentText() == RYU_FRONTIER_GRAPH_BFS


def test_implemented_hungarian_task_assign_algorithm_is_selectable():
    items = _combo_items(_window.coordinator_combo)

    assert TASK_ASSIGN_ALGORITHM_OPTIONS == ("Frontier cluster Hungarian coordinator",)
    assert items == list(TASK_ASSIGN_ALGORITHM_OPTIONS)
    assert not REMOVED_TASK_ASSIGN_ALGORITHM_OPTIONS.intersection(items)
    assert _window.coordinator_combo.currentIndex() == 0
    assert _window.read_config().coordinator_type == TASK_ASSIGN_ALGORITHM_OPTIONS[0]


def test_only_cited_safety_algorithm_is_selectable():
    items = _combo_items(_window.safety_algorithm_combo)
    assert items == list(SAFETY_ALGORITHM_OPTIONS)
    assert items == [WANG_AMES_BARRIER_CERTIFICATE]
    assert _window.read_config().safety_algorithm == WANG_AMES_BARRIER_CERTIFICATE


def test_removed_frontier_value_from_legacy_config_falls_back_to_visible_default():
    legacy = _window.read_config()
    legacy.exploration_planner = "Nearest frontier"

    _window.apply_config_to_widgets(legacy)

    assert _window.exploration_planner_combo.currentText() in (
        FRONTIER_ALGORITHM_DETECTOR_OPTIONS
    )
    assert _window.exploration_planner_combo.currentText() != "Nearest frontier"


def test_multiple_mode_cannot_silently_fall_back_to_a_removed_task_assign_algorithm():
    messages: list[str] = []
    statuses: list[str] = []
    fake = SimpleNamespace(
        config=SimpleNamespace(coordinator_type=NO_TASK_ASSIGN_ALGORITHM),
        log_console_message=messages.append,
        canvas=SimpleNamespace(set_status=statuses.append),
    )

    SimulationControllerMixin.start_multi_robot_simulation(fake)

    assert messages == statuses
    assert len(messages) == 1
    assert "requires a Task Assign Algorithm" in messages[0]


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


def test_route_visualization_toggles_are_independent_and_feed_config():
    _window.traveled_path_switch.setChecked(True)
    _window.planned_route_switch.setChecked(False)

    config = _window.read_config()

    assert config.show_traveled_path is True
    assert config.show_path is False

    _window.traveled_path_switch.setChecked(False)
    _window.planned_route_switch.setChecked(True)


def test_custom_discovery_reveals_obstacle_color_and_opacity_controls():
    previous_mode = _window.map_visualization_combo.currentText()
    try:
        _window.map_visualization_combo.setCurrentText("Current")
        assert _window.custom_obstacle_color_field.isHidden()
        assert _window.custom_explored_opacity_field.isHidden()

        _window.map_visualization_combo.setCurrentText("Custom Discovery")
        assert not _window.custom_obstacle_color_field.isHidden()
        assert not _window.custom_explored_opacity_field.isHidden()
    finally:
        _window.map_visualization_combo.setCurrentText(previous_mode)


def test_custom_visual_values_are_collected_from_configuration_controls():
    previous_color = _window.custom_obstacle_color_button.color_hex()
    previous_opacity = _window.custom_explored_opacity_input.value()
    previous_width = _window.mapped_obstacle_line_width_input.value()
    try:
        _window.custom_obstacle_color_button.set_color("#123456")
        _window.custom_explored_opacity_input.setValue(45.0)
        _window.mapped_obstacle_line_width_input.setValue(2.75)

        config = _window.read_config()

        assert config.custom_obstacle_color == "#123456"
        assert config.custom_explored_opacity == 0.45
        assert config.mapped_obstacle_line_width == 2.75
    finally:
        _window.custom_obstacle_color_button.set_color(previous_color)
        _window.custom_explored_opacity_input.setValue(previous_opacity)
        _window.mapped_obstacle_line_width_input.setValue(previous_width)
