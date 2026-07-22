from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas, ipp_uncertainty_rgba
from robotics_sim.app.main_window import MainWindow
from robotics_sim.experiments import load_ipp_bundle
from robotics_sim.simulation.config import (
    SimulationConfig,
    config_from_sim_payload,
    config_to_sim_payload,
    load_sim_file,
)
from robotics_sim.simulation.engine import SimulationControllerMixin


ROOT = Path(__file__).resolve().parents[2]
PRESET = ROOT / "examples" / "rss26_ipp_rbf_smoke.sim"
BUNDLE = ROOT / "examples" / "rss26_ipp_rbf_smoke" / "manifest.json"
PINNED_COMMIT = "f387c57bcaa61bf218d26e63212bf789ef42a534"


def test_experiment_metadata_round_trips_without_affecting_old_scenarios(tmp_path) -> None:
    metadata = {
        "kind": "uncertainty_guaranteed_ipp_rss26",
        "bundle": "assets/manifest.json",
        "nested": {"ratios": [0.9, 0.7, 0.5]},
    }
    payload = config_to_sim_payload(SimulationConfig(experiment=metadata))
    assert payload["experiment"] == metadata
    restored = config_from_sim_payload(payload)
    assert restored.experiment == metadata

    ordinary = config_to_sim_payload(SimulationConfig())
    assert "experiment" not in ordinary

    scenario = tmp_path / "paper.sim"
    scenario.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_sim_file(str(scenario))
    assert loaded.source_path == str(scenario.resolve())


def test_experiment_manifest_is_relative_and_confined_to_scenario(tmp_path) -> None:
    scenario = tmp_path / "paper.sim"
    scenario.write_text("{}", encoding="utf-8")
    config = SimulationConfig(
        experiment={
            "kind": "uncertainty_guaranteed_ipp_rss26",
            "bundle": "assets/manifest.json",
        },
        source_path=str(scenario),
    )
    assert SimulationControllerMixin.ipp_manifest_path_for_config(config) == (
        tmp_path / "assets" / "manifest.json"
    ).resolve()

    config.experiment["bundle"] = "../outside.json"
    with pytest.raises(ValueError, match="escapes"):
        SimulationControllerMixin.ipp_manifest_path_for_config(config)


def test_prescribed_sensing_waypoints_are_not_line_of_sight_collapsed() -> None:
    fake = SimpleNamespace(
        robot=SimpleNamespace(x=0.0, y=0.0),
        config=SimpleNamespace(path_simplifier="Line of sight grid-safe"),
        collision_checker=object(),
        _preserve_next_route_waypoints=True,
    )
    route = [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 1.0)]
    cleaned = SimulationControllerMixin.clean_waypoints_for_current_start(fake, route)
    assert cleaned == route[1:]


def test_rbf_smoke_bundle_is_certified_but_explicitly_not_the_paper_benchmark() -> None:
    bundle = load_ipp_bundle(BUNDLE)
    metrics = bundle.metrics

    assert metrics["fidelity"] == "integration_smoke_not_paper_benchmark"
    assert metrics["kernel"] == "stationary RBF"
    assert metrics["certificate_passed"] is True
    assert metrics["theorem_coverage_passed"] is True
    assert metrics["max_posterior_variance"] <= metrics["target_variance"]
    assert len(bundle.solution_path) >= 2
    assert len(bundle.sensing_points) == metrics["sensing_point_count"]


def test_rss26_smoke_preset_pins_sources_and_stays_single_robot() -> None:
    payload = json.loads(PRESET.read_text(encoding="utf-8"))
    experiment = payload["experiment"]

    assert experiment["kind"] == "uncertainty_guaranteed_ipp_rss26"
    assert experiment["arxiv"] == "2602.05198v3"
    assert experiment["source_commit"] == PINNED_COMMIT
    assert experiment["fidelity"] == "integration_smoke_not_paper_benchmark"
    assert payload["simulation"]["agent_mode"] == "Single Robot Mode"
    assert payload["exploration"]["planner"] == "Goal seeking"
    assert payload["multi_robot"]["robot_count"] == 1


def test_canvas_renders_uncertainty_and_reference_tour_offscreen() -> None:
    _app = QApplication.instance() or QApplication([])
    bundle = load_ipp_bundle(BUNDLE)
    rgba = ipp_uncertainty_rgba(bundle.posterior_variance, bundle.mask)
    assert rgba.shape == (*bundle.posterior_variance.shape, 4)
    assert np.all(rgba[..., 3][~bundle.mask] == 0)

    canvas = SimulationCanvas()
    canvas.resize(800, 640)
    canvas.set_ipp_experiment_bundle(bundle)
    image = QImage(800, 640, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    canvas.draw_ipp_uncertainty_heatmap(painter)
    canvas.draw_ipp_reference_overlay(painter)
    painter.end()
    assert canvas._ipp_variance_pixmap_cache is not None


def test_main_window_loads_preset_and_installs_complete_sensing_tour() -> None:
    _app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        window.apply_config_to_widgets(load_sim_file(str(PRESET)))
        bundle = window.ipp_experiment_bundle
        assert bundle is not None

        window.start_simulation()
        assigned = np.asarray(window.robot.waypoints.waypoints, dtype=float)
        # The first bundle point is the robot's initial pose and is correctly
        # removed; every subsequent sensing-tour vertex must remain.
        np.testing.assert_allclose(assigned, bundle.solution_path[1:])
        assert len(assigned) == len(bundle.solution_path) - 1
    finally:
        window.reset_simulation()
        window.close()
