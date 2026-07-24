"""End-to-end GUI-flow tests for Human Demonstration mode, through a real
MainWindow instance.

Unlike test_config_panel_layout.py's single shared-window pattern, this
module builds one fresh MainWindow per test (all sharing the one module-
level QApplication -- no additional QApplication is ever created): the
episode lifecycle is heavily stateful (collector/map/episode selection,
active episode, active manual round), so sharing a window across tests
would make failures order-dependent. Every test also injects its own
HumanDemonstrationRuntime pointed at tmp_path (via the real manifest/plan,
but never the real experiments/datasets/human_demonstrations_v0/ output
root) so no test ever writes into the actual repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from robotics_interfaces.coordination import CoordinationRequest, CoordinationResult
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.app.main_window import MainWindow
from robotics_sim.learning.demonstration_collection_plan import load_demonstration_collection_plan
from robotics_sim.learning.map_catalog import load_map_catalog
from robotics_sim.simulation.human_demonstration_runtime import (
    HUMAN_DEMONSTRATION_COORDINATOR_LABEL,
    HumanDemonstrationHostBindings,
    HumanDemonstrationRuntime,
)

_app = QApplication.instance() or QApplication([])

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "experiments" / "maps" / "smoke_v0" / "manifest.json"
_PLAN_PATH = _REPO_ROOT / "experiments" / "collection_plans" / "human_demo_smoke_v0.json"


def make_window_with_test_runtime(tmp_path: Path) -> MainWindow:
    window = MainWindow()
    map_catalog = load_map_catalog(_MANIFEST_PATH)
    collection_plan = load_demonstration_collection_plan(_PLAN_PATH, map_catalog=map_catalog)
    host = HumanDemonstrationHostBindings(
        load_sim_file=window._human_demo_load_sim_file,
        clear_fires=window._human_demo_clear_fires,
        add_fire=window._human_demo_add_fire,
        reset_hazard_belief=window._human_demo_reset_hazard_belief,
        request_pause=window._human_demo_request_pause,
        wait_for_human_resume=window._human_demo_wait_for_resume,
        get_simulation_time_s=lambda: float(getattr(window, "simulation_time", 0.0)),
        get_final_metrics=window._human_demo_get_final_metrics,
    )
    window.human_demo_runtime = HumanDemonstrationRuntime(
        map_catalog=map_catalog,
        collection_plan=collection_plan,
        sim_directory=_MANIFEST_PATH.parent,
        output_root=tmp_path / "human_demo_output",
        host=host,
    )
    return window


def activate_human_demo_mode(window: MainWindow) -> None:
    window.coordinator_combo.setCurrentText(HUMAN_DEMONSTRATION_COORDINATOR_LABEL)
    window.human_demo_request_executor = window.human_demo_runtime.request_executor


# --- basic presence / activation -----------------------------------------


def test_human_demonstration_option_appears_in_coordinator_combo() -> None:
    window = MainWindow()
    items = [window.coordinator_combo.itemText(i) for i in range(window.coordinator_combo.count())]
    assert HUMAN_DEMONSTRATION_COORDINATOR_LABEL in items


def test_panel_enabled_only_in_manual_mode() -> None:
    window = MainWindow()
    assert window._human_demo_panel_visible is False
    window.coordinator_combo.setCurrentText(HUMAN_DEMONSTRATION_COORDINATOR_LABEL)
    assert window._human_demo_mode_active is True
    assert window._human_demo_panel_visible is True
    assert window.canvas._human_demo_mode_active is True

    # Switching to a real coordinator turns it back off.
    real_option = next(
        window.coordinator_combo.itemText(i)
        for i in range(window.coordinator_combo.count())
        if window.coordinator_combo.itemText(i) != HUMAN_DEMONSTRATION_COORDINATOR_LABEL
    )
    window.coordinator_combo.setCurrentText(real_option)
    assert window._human_demo_mode_active is False
    assert window._human_demo_panel_visible is False
    assert window.canvas._human_demo_mode_active is False
    assert window.human_demo_request_executor is None


def test_selecting_human_demo_never_reaches_a_fake_plugin(tmp_path: Path) -> None:
    """coordinator_type must always stay a real, loadable plugin name."""

    window = make_window_with_test_runtime(tmp_path)
    activate_human_demo_mode(window)
    config = window.read_config()
    assert config.coordinator_type != HUMAN_DEMONSTRATION_COORDINATOR_LABEL
    assert config.coordinator_type


# --- collector / map / episode selection ----------------------------------


def test_collector_selector_filters_maps(tmp_path: Path) -> None:
    window = make_window_with_test_runtime(tmp_path)
    activate_human_demo_mode(window)

    window.on_human_demo_collector_selected("collector_a")
    maps_a = set(window.human_demo_runtime.setup.available_map_ids)
    window.on_human_demo_collector_selected("collector_b")
    maps_b = set(window.human_demo_runtime.setup.available_map_ids)

    assert maps_a and maps_b
    assert maps_a.isdisjoint(maps_b)
    assert set(window.human_demo_panel.map_combo.itemText(i) for i in range(window.human_demo_panel.map_combo.count())) == maps_b


def test_previous_next_episode(tmp_path: Path) -> None:
    window = make_window_with_test_runtime(tmp_path)
    activate_human_demo_mode(window)
    window.on_human_demo_collector_selected("collector_a")
    map_id = next(iter(window.human_demo_runtime.setup.available_map_ids))
    window.on_human_demo_map_selected(map_id)
    window.on_human_demo_episode_selected(1)
    assert window.human_demo_runtime.setup.selected_episode_number == 1

    window.on_human_demo_next_episode()
    assert window.human_demo_runtime.setup.selected_episode_number == 2

    window.on_human_demo_previous_episode()
    assert window.human_demo_runtime.setup.selected_episode_number == 1


def test_episode_x_of_n_displayed(tmp_path: Path) -> None:
    window = make_window_with_test_runtime(tmp_path)
    activate_human_demo_mode(window)
    window.on_human_demo_collector_selected("collector_a")
    map_id = next(iter(window.human_demo_runtime.setup.available_map_ids))
    window.on_human_demo_map_selected(map_id)
    window.on_human_demo_episode_selected(1)
    assert window.human_demo_panel.episode_position_label.text() == "Episode 1 of 2"


def test_load_episode_locks_selectors(tmp_path: Path) -> None:
    window = make_window_with_test_runtime(tmp_path)
    activate_human_demo_mode(window)
    window.on_human_demo_collector_selected("collector_a")
    map_id = next(iter(window.human_demo_runtime.setup.available_map_ids))
    window.on_human_demo_map_selected(map_id)
    window.on_human_demo_episode_selected(1)
    assert window.human_demo_panel.collector_combo.isEnabled()

    window.on_human_demo_load_episode()
    assert not window.human_demo_panel.collector_combo.isEnabled()
    assert not window.human_demo_panel.map_combo.isEnabled()
    assert not window.human_demo_panel.episode_combo.isEnabled()
    assert not window.human_demo_panel.load_button.isEnabled()


# --- click flow -------------------------------------------------------------


def _make_request(robot_ids, targets) -> CoordinationRequest:
    robot_states = tuple(
        RobotCoordinationState(
            robot_id=rid, xy=(0.0, 0.0), safety_radius=0.3, sensor_range=3.0, vision_model="cone"
        )
        for rid in robot_ids
    )
    return CoordinationRequest(
        robot_states=robot_states,
        robots_to_assign=tuple(robot_ids),
        proposals_by_robot={
            rid: (ExplorationCandidate(target=targets[rid], source="test"),) for rid in robot_ids
        },
    )


def _load_first_episode(window: MainWindow) -> None:
    activate_human_demo_mode(window)
    window.on_human_demo_collector_selected("collector_a")
    map_id = next(iter(window.human_demo_runtime.setup.available_map_ids))
    window.on_human_demo_map_selected(map_id)
    window.on_human_demo_episode_selected(1)
    window.on_human_demo_load_episode()


def test_robot_click_updates_selection(tmp_path: Path) -> None:
    window = make_window_with_test_runtime(tmp_path)
    _load_first_episode(window)

    captured = {}

    def during_wait():
        window.on_human_demo_robot_clicked(0)
        captured["selected_text"] = window.human_demo_panel.selected_robot_label.text()
        captured["session_robot"] = window.human_demo_runtime.active_session.focused_robot_id
        slot = window.human_demo_runtime.candidates_for_robot(0)[0]
        window.on_human_demo_candidate_clicked(slot.candidate_id)
        window.handle_start_pause_button()

    QTimer.singleShot(0, during_wait)
    result = window.human_demo_request_executor(_make_request((0,), {0: (1.0, 1.0)}))

    assert captured["selected_text"] == "Selected robot: R1"
    assert captured["session_robot"] == 0
    assert isinstance(result, CoordinationResult)


def test_candidate_click_uses_exact_marker(tmp_path: Path) -> None:
    window = make_window_with_test_runtime(tmp_path)
    _load_first_episode(window)

    outcomes = {}

    def during_wait():
        window.on_human_demo_robot_clicked(0)
        slots = window.human_demo_runtime.candidates_for_robot(0)
        # Clicking an id that doesn't match any shown candidate must be a
        # silent no-op (never guessed/approximated).
        window.on_human_demo_candidate_clicked("not-a-real-candidate-id")
        outcomes["decisions_after_bogus_click"] = len(window.human_demo_runtime.active_session.decisions())
        window.on_human_demo_candidate_clicked(slots[0].candidate_id)
        outcomes["decisions_after_real_click"] = len(window.human_demo_runtime.active_session.decisions())
        outcomes["target"] = window.human_demo_runtime.active_session.decisions()[0].target_xy
        window.handle_start_pause_button()

    QTimer.singleShot(0, during_wait)
    window.human_demo_request_executor(_make_request((0,), {0: (2.5, 3.5)}))

    assert outcomes["decisions_after_bogus_click"] == 0
    assert outcomes["decisions_after_real_click"] == 1
    assert outcomes["target"] == (2.5, 3.5)


def test_finish_shows_saved_path(tmp_path: Path) -> None:
    window = make_window_with_test_runtime(tmp_path)
    _load_first_episode(window)

    def during_wait():
        window.on_human_demo_robot_clicked(0)
        slot = window.human_demo_runtime.candidates_for_robot(0)[0]
        window.on_human_demo_candidate_clicked(slot.candidate_id)
        window.handle_start_pause_button()

    QTimer.singleShot(0, during_wait)
    window.human_demo_request_executor(_make_request((0,), {0: (1.0, 1.0)}))

    window.on_human_demo_finish_episode()
    saved_text = window.human_demo_panel.last_saved_path_label.text()
    assert saved_text.startswith("Saved: ")
    assert str(tmp_path) in saved_text
    saved_dir = Path(saved_text[len("Saved: "):])
    assert saved_dir.is_dir()
    files = sorted(p.name for p in saved_dir.iterdir())
    assert files == ["decisions.jsonl", "integrity_report.json", "metadata.json", "metrics.json"]


# --- static hygiene checks --------------------------------------------------


def test_no_additional_qapplication_created() -> None:
    assert QApplication.instance() is _app


def test_no_plugin_created_under_algorithms() -> None:
    algorithms_dir = _REPO_ROOT / "algorithms"
    names = {p.name.lower() for p in algorithms_dir.iterdir() if p.is_dir()}
    assert not any("human" in name or "demonstration" in name for name in names)
