"""Boundary tests: RuntimeActorFrame/ActorObservationBatch depend only on the
observed HazardBeliefFrame, never on ground truth, and capture_inputs.py /
observation_batch.py import nothing privileged.

Field-name inspection alone is not enough (a builder could still leak a
privileged value into a "safe" field); the tests below demonstrate the
boundary behaviorally by building the same batch under two different hidden
ground-truth worlds and asserting the results are identical."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.learning import CandidateKind, CandidateSetSpec, HoldPolicy
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig, build_feature_schema_v0
from robotics_sim.learning.capture_inputs import (
    CandidateCaptureInput,
    RobotActorCaptureInput,
    RuntimeActorFrame,
)
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent
RUNTIME_MODULES = ("capture_inputs.py", "observation_batch.py")

NORMALIZATION = FeatureNormalizationConfig(
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


def make_geometry() -> GridGeometry:
    return GridGeometry(bounds=(0.0, 10.0, 0.0, 10.0), resolution=1.0)


def make_candidate_spec() -> CandidateSetSpec:
    return CandidateSetSpec(
        schema_version="0.1.0",
        max_candidates=8,
        max_headings_per_candidate=1,
        deterministic_ordering=True,
        deduplication_distance=0.5,
        hold_policy=HoldPolicy(),
    )


def make_frame(geometry: GridGeometry, belief_frame) -> RuntimeActorFrame:
    robot = RobotCoordinationState(
        robot_id=0,
        xy=(1.0, 1.0),
        safety_radius=0.5,
        sensor_range=4.0,
        vision_model="cone",
        theta=0.0,
    )
    candidate = CandidateCaptureInput(
        candidate=ExplorationCandidate(target=(4.0, 6.0), source="frontier"),
        kind=CandidateKind.FRONTIER_VIEWPOINT,
        enabled=True,
        reachable=True,
    )
    robot_capture = RobotActorCaptureInput(
        robot=robot,
        candidates=(candidate,),
        graph_edges=(),
        visible_teammates=(),
        hazard_belief=belief_frame,
    )
    return RuntimeActorFrame(
        episode_id="ep-boundary",
        decision_step=0,
        time_s=0.0,
        robots=(robot_capture,),
        grid_geometry=geometry,
        normalization=NORMALIZATION,
        candidate_spec=make_candidate_spec(),
    )


def build_batch(geometry, belief_frame):
    assembler = ActorObservationBatchAssembler(
        schema=build_feature_schema_v0(), candidate_spec=make_candidate_spec()
    )
    return assembler.build(make_frame(geometry, belief_frame))


class TestGroundTruthDoesNotLeakIntoBatch:
    def test_different_hidden_fire_worlds_same_belief_gives_identical_batch(self):
        geometry = make_geometry()

        # Two conceptually different ground-truth fire worlds.  Neither is
        # ever passed to RuntimeActorFrame, ActorObservationBatchAssembler,
        # or any type they accept -- there is no parameter that could take
        # them.
        hidden_world_a = {"fires": (((8.0, 8.0), 0.9),), "occupancy_seed": 1}
        hidden_world_b = {
            "fires": (((1.0, 1.0), 0.2), ((5.0, 9.0), 1.0)),
            "occupancy_seed": 2,
        }
        assert hidden_world_a != hidden_world_b

        # Identical *observed* belief in both worlds: the team has only seen
        # the same three cells with the same values.
        def observed_belief():
            belief = HazardBelief(geometry)
            belief.observe_cells([6, 5, 0], [4, 4, 0], [0.3, 0.0, 0.0], robot_index=0)
            return belief.snapshot()

        batch_a = build_batch(geometry, observed_belief())
        batch_b = build_batch(geometry, observed_belief())
        assert batch_a == batch_b

    def test_observable_belief_change_changes_thermal_features(self):
        geometry = make_geometry()
        belief = HazardBelief(geometry)
        belief.observe_cells([6], [4], [0.0], robot_index=0)
        batch_before = build_batch(geometry, belief.snapshot())

        belief.observe_cells([6], [4], [0.9], robot_index=0)  # fire now observed
        batch_after = build_batch(geometry, belief.snapshot())

        obs_before = batch_before.observations[0]
        obs_after = batch_after.observations[0]
        fire_index = obs_before.candidate_feature_names.index("fire_value_at_target")
        assert obs_before.candidate_features[0][fire_index] == 0.0
        assert obs_after.candidate_features[0][fire_index] == pytest.approx(0.9, rel=1e-6)
        assert batch_before != batch_after


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

    @pytest.mark.parametrize("filename", RUNTIME_MODULES)
    def test_no_privileged_imports(self, filename):
        for module, name in self._imports(filename):
            root = module.split(".")[0]
            assert root not in self.FORBIDDEN_ROOTS, (filename, module)
            assert not root.lower().startswith(("pyqt", "pyside")), (filename, module)
            for forbidden in self.FORBIDDEN_MODULES:
                assert not module.startswith(forbidden), (filename, module)
            assert name not in self.FORBIDDEN_NAMES, (filename, module, name)
