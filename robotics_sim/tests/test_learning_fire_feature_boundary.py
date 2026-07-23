"""Boundary tests: fire features depend only on the observed belief, never
on ground truth, and the extractor modules import nothing privileged."""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path
from typing import Iterator, Mapping

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.learning import CandidateKind, CandidateObservation
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import (
    CandidateFeatureExtractionInput,
    CandidateFeatureExtractor,
    FeatureNormalizationConfig,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent
FEATURE_MODULES = ("feature_schema_v0.py", "feature_inputs.py", "feature_extractors.py")

CONFIG = FeatureNormalizationConfig(
    distance_scale=10.0,
    information_gain_scale=5.0,
    travel_cost_scale=20.0,
    safety_cost_scale=2.0,
    overlap_cost_scale=4.0,
    heading_cost_scale=1.0,
    sensor_range_scale=8.0,
    safety_radius_scale=1.0,
    fire_window_radius_cells=1,
)

ROBOT = RobotCoordinationState(
    robot_id=0,
    xy=(1.0, 2.0),
    safety_radius=0.5,
    sensor_range=4.0,
    vision_model="cone",
    theta=0.0,
)

OBSERVATION = CandidateObservation(
    candidate_id="c0",
    kind=CandidateKind.FRONTIER_VIEWPOINT,
    xy=(4.0, 6.0),
    heading_candidates=(0.0,),
    source="frontier",
    reachable=True,
)


def make_geometry() -> GridGeometry:
    return GridGeometry(bounds=(0.0, 10.0, 0.0, 10.0), resolution=1.0)


def extract(frame, geometry, candidate=None) -> Mapping[str, float]:
    return CandidateFeatureExtractor().extract(
        CandidateFeatureExtractionInput(
            robot=ROBOT,
            candidate=candidate if candidate is not None else ExplorationCandidate(target=(4.0, 6.0)),
            candidate_observation=OBSERVATION,
            hazard_belief=frame,
            grid_geometry=geometry,
            normalization=CONFIG,
        )
    )


class TestGroundTruthDoesNotLeak:
    def test_different_hidden_fire_same_belief_gives_identical_features(self):
        geometry = make_geometry()

        # Two conceptually different ground-truth fire worlds.  They exist
        # only in this test and are never passed to the extractor -- there is
        # no parameter that could accept them.
        hidden_fire_world_a = {"fires": (((8.0, 8.0), 0.9),), "occupancy_seed": 1}
        hidden_fire_world_b = {"fires": (((1.0, 1.0), 0.2), ((5.0, 9.0), 1.0)), "occupancy_seed": 2}
        assert hidden_fire_world_a != hidden_fire_world_b

        # Identical *observed* belief in both worlds: the team has only seen
        # the same three cells with the same values.
        def observed_belief() -> HazardBelief:
            belief = HazardBelief(geometry)
            belief.observe_cells([6, 5, 0], [4, 4, 0], [0.3, 0.0, 0.0], robot_index=0)
            return belief

        features_a = extract(observed_belief().snapshot(), geometry)
        features_b = extract(observed_belief().snapshot(), geometry)
        assert features_a == features_b

    def test_observable_belief_change_does_change_fire_features(self):
        geometry = make_geometry()
        belief = HazardBelief(geometry)
        belief.observe_cells([6], [4], [0.0], robot_index=0)
        before = extract(belief.snapshot(), geometry)

        belief.observe_cells([6], [4], [0.9], robot_index=0)  # fire now observed
        after = extract(belief.snapshot(), geometry)

        assert before["fire_value_at_target"] == 0.0
        assert after["fire_value_at_target"] == pytest.approx(0.9, rel=1e-6)
        assert before != after
        # Non-fire features are untouched by the belief change.
        for name in ("relative_x_norm", "euclidean_distance_norm", "travel_cost_norm"):
            assert before[name] == after[name]


class TestPrivilegedImports:
    FORBIDDEN_ROOTS = ("PyQt5", "PyQt6", "PySide2", "PySide6", "torch", "pandas")
    FORBIDDEN_MODULES = (
        "robotics_sim.simulation.engine",
        "robotics_sim.app",
        "robotics_sim.environment.hazard_field",
        "robotics_sim.diagnostics",
    )
    FORBIDDEN_NAMES = (
        "HazardField",
        "FireSource",
        "HazardDebug",
        "HazardSourceDebug",
        "GroundTruthSnapshot",
        "CriticState",
    )

    def _imports(self, filename: str) -> Iterator[tuple[str, str]]:
        tree = ast.parse((LEARNING_DIR / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name, ""
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    yield node.module, alias.name

    @pytest.mark.parametrize("filename", FEATURE_MODULES)
    def test_no_privileged_imports(self, filename):
        for module, name in self._imports(filename):
            root = module.split(".")[0]
            assert root not in self.FORBIDDEN_ROOTS, (filename, module)
            assert not root.lower().startswith(("pyqt", "pyside")), (filename, module)
            for forbidden in self.FORBIDDEN_MODULES:
                assert not module.startswith(forbidden), (filename, module)
            assert name not in self.FORBIDDEN_NAMES, (filename, module, name)


class TestInputRejectsGroundTruth:
    def test_input_field_names_have_no_privileged_fields(self):
        names = {f.name for f in dataclasses.fields(CandidateFeatureExtractionInput)}
        assert names == {
            "robot", "candidate", "candidate_observation", "hazard_belief",
            "grid_geometry", "normalization",
        }

    def test_ground_truth_carrier_rejected_as_hazard_belief(self):
        from robotics_interfaces.learning import GroundTruthSnapshot

        snapshot = GroundTruthSnapshot(
            schema_version="0.1.0",
            decision_step=0,
            time_s=0.0,
            true_robot_poses={},
            true_occupancy=(),
            true_fire_locations=((4.0, 6.0),),
            global_coverage_fraction=0.0,
        )
        with pytest.raises(TypeError):
            CandidateFeatureExtractionInput(
                robot=ROBOT,
                candidate=ExplorationCandidate(target=(4.0, 6.0)),
                candidate_observation=OBSERVATION,
                hazard_belief=snapshot,
                grid_geometry=make_geometry(),
                normalization=CONFIG,
            )


class _ExplodingMapping(Mapping):
    """A metadata mapping that fails on any read access."""

    def __getitem__(self, key):  # pragma: no cover - failure path
        raise AssertionError("extractor must not read candidate.metadata")

    def __iter__(self):  # pragma: no cover - failure path
        raise AssertionError("extractor must not iterate candidate.metadata")

    def __len__(self):  # pragma: no cover - failure path
        raise AssertionError("extractor must not measure candidate.metadata")


class TestMetadataIsNeverRead:
    def test_no_metadata_attribute_access_in_extractor_source(self):
        tree = ast.parse((LEARNING_DIR / "feature_extractors.py").read_text(encoding="utf-8"))
        accessed = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "metadata"
        ]
        assert accessed == []

    def test_extraction_succeeds_with_exploding_metadata(self):
        geometry = make_geometry()
        candidate = ExplorationCandidate(
            target=(4.0, 6.0),
            information_gain=1.0,
            heading_rad=0.5,
            metadata=_ExplodingMapping(),
        )
        features = extract(HazardBelief(geometry).snapshot(), geometry, candidate=candidate)
        assert math.isfinite(features["information_gain_norm"])
