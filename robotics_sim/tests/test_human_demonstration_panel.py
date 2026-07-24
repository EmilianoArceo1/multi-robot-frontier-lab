"""Tests for HumanDemonstrationPanel -- presentation-only widget.

Follows the existing app-test pattern (see test_config_panel_layout.py):
one shared QApplication instance for the whole module, no per-test
QApplication construction.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from PySide6.QtWidgets import QApplication

from robotics_sim.app import human_demonstration_panel as panel_module
from robotics_sim.app.human_demonstration_panel import HumanDemonstrationPanel

_app = QApplication.instance() or QApplication([])
_panel = HumanDemonstrationPanel()
_panel.show()  # isVisible()/setVisible() checks require a shown top-level ancestor


def test_no_additional_qapplication_created() -> None:
    assert QApplication.instance() is _app


def test_no_filesystem_logic_in_widget_module() -> None:
    source = inspect.getsource(panel_module)
    tree = ast.parse(source)
    forbidden_calls = {"open"}
    forbidden_modules = {"json", "pathlib", "os", "shutil"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_modules
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_modules
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_calls


def test_no_planner_or_candidate_computation_in_widget_module() -> None:
    source = inspect.getsource(panel_module)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ("assign", "compute_planned_waypoints")


def test_no_plugin_created_under_algorithms() -> None:
    algorithms_dir = Path(__file__).resolve().parents[2] / "algorithms"
    names = {p.name.lower() for p in algorithms_dir.iterdir() if p.is_dir()}
    assert not any("human" in name or "demonstration" in name for name in names)


def test_collector_selection_emits_signal() -> None:
    seen = []
    _panel.collectorSelected.connect(seen.append)
    try:
        _panel.set_collector_options(["collector_a", "collector_b"])
        _panel.collector_combo.setCurrentText("collector_b")
        assert seen == ["collector_b"]
    finally:
        _panel.collectorSelected.disconnect(seen.append)


def test_map_selection_emits_signal() -> None:
    seen = []
    _panel.mapSelected.connect(seen.append)
    try:
        _panel.set_map_options(["map_a", "map_b"])
        _panel.map_combo.setCurrentText("map_b")
        assert seen == ["map_b"]
    finally:
        _panel.mapSelected.disconnect(seen.append)


def test_episode_selection_emits_int_signal() -> None:
    seen = []
    _panel.episodeSelected.connect(seen.append)
    try:
        _panel.set_episode_options([1, 2, 3])
        _panel.episode_combo.setCurrentText("2")
        assert seen == [2]
    finally:
        _panel.episodeSelected.disconnect(seen.append)


def test_previous_next_buttons_emit_signals() -> None:
    previous_calls = []
    next_calls = []
    _panel.previousEpisodeRequested.connect(lambda: previous_calls.append(1))
    _panel.nextEpisodeRequested.connect(lambda: next_calls.append(1))
    _panel.previous_button.click()
    _panel.next_button.click()
    assert previous_calls == [1]
    assert next_calls == [1]


def test_load_finish_abort_buttons_emit_signals() -> None:
    load_calls = []
    finish_calls = []
    abort_calls = []
    _panel.loadEpisodeRequested.connect(lambda: load_calls.append(1))
    _panel.finishEpisodeRequested.connect(lambda: finish_calls.append(1))
    _panel.abortEpisodeRequested.connect(lambda: abort_calls.append(1))
    # finish/abort start disabled (see __init__) -- a disabled QPushButton
    # does not emit clicked() on click(), so enable them for this signal-
    # wiring check; enablement policy itself is covered elsewhere.
    _panel.set_finish_enabled(True)
    _panel.set_abort_enabled(True)
    _panel.load_button.click()
    _panel.finish_button.click()
    _panel.abort_button.click()
    assert load_calls == [1]
    assert finish_calls == [1]
    assert abort_calls == [1]


def test_episode_position_text_rendered_verbatim() -> None:
    _panel.set_episode_position_text("Episode 3 of 7")
    assert _panel.episode_position_label.text() == "Episode 3 of 7"


def test_recorded_and_accepted_progress_rendered_verbatim() -> None:
    _panel.set_recorded_progress_text("Recorded 2 of 7")
    _panel.set_accepted_progress_text("Accepted 1 of 7")
    assert _panel.recorded_progress_label.text() == "Recorded 2 of 7"
    assert _panel.accepted_progress_label.text() == "Accepted 1 of 7"


def test_fires_loaded_count_rendered() -> None:
    _panel.set_fires_loaded_count(3)
    assert _panel.fires_loaded_label.text() == "Fires loaded: 3"
    _panel.set_fires_loaded_count(None)
    assert _panel.fires_loaded_label.text() == "Fires loaded: --"


def test_selected_robot_and_pending_robots_rendered() -> None:
    _panel.set_selected_robot_text("R2")
    _panel.set_pending_robots_text("R1, R2")
    assert _panel.selected_robot_label.text() == "Selected robot: R2"
    assert _panel.pending_robots_label.text() == "Pending robots: R1, R2"


def test_episode_active_locks_selectors() -> None:
    _panel.set_episode_active(True)
    assert not _panel.collector_combo.isEnabled()
    assert not _panel.map_combo.isEnabled()
    assert not _panel.episode_combo.isEnabled()
    assert not _panel.previous_button.isEnabled()
    assert not _panel.next_button.isEnabled()
    assert not _panel.load_button.isEnabled()

    _panel.set_episode_active(False)
    assert _panel.collector_combo.isEnabled()
    assert _panel.map_combo.isEnabled()
    assert _panel.episode_combo.isEnabled()


def test_map_complete_text_hidden_when_none() -> None:
    _panel.set_map_complete_text("Map complete: Recorded 6 of 6")
    assert _panel.map_complete_label.isVisible()
    assert _panel.map_complete_label.text() == "Map complete: Recorded 6 of 6"

    _panel.set_map_complete_text(None)
    assert not _panel.map_complete_label.isVisible()


def test_last_saved_path_rendered() -> None:
    _panel.set_last_saved_path("/tmp/episode_dir")
    assert "/tmp/episode_dir" in _panel.last_saved_path_label.text()
    _panel.set_last_saved_path(None)
    assert _panel.last_saved_path_label.text() == ""


def test_close_button_emits_close_requested() -> None:
    calls = []
    _panel.closeRequested.connect(lambda: calls.append(1))
    _panel.close_button.click()
    assert calls == [1]
