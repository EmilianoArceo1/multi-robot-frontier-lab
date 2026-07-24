"""Boundary tests: ground truth reaches only recorder.append(), never a
LearningTransition; transition_inputs.py and transition_assembler.py import
nothing privileged; no arbitrary metadata anywhere in the new types.

Field-name inspection alone is not enough -- the tests below build two real
transitions under two different (conceptually) hidden ground-truth worlds
and demonstrate the resulting LearningTransition and EpisodeRecord.transitions
are identical, while ground truth only shows up in
EpisodeRecord.ground_truth_by_step."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Iterator

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.learning import (
    CONTRACT_VERSIONS,
    CandidateKind,
    CandidateSetSpec,
    CriticState,
    EpisodeMetadata,
    GroundTruthSnapshot,
    HoldPolicy,
    TerminationReason,
    build_contract_manifest,
    compute_contract_bundle_hash,
)
from robotics_interfaces.learning.transitions import LearningTransition, RewardComponent
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig, build_feature_schema_v0
from robotics_sim.learning.action_catalog import ActionCatalogAssembler
from robotics_sim.learning.capture_inputs import CandidateCaptureInput, RobotActorCaptureInput, RuntimeActorFrame
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler
from robotics_sim.learning.recorder import InMemoryTrajectoryRecorder
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import (
    DecisionSelectionBatch,
    RobotActionSelection,
    RobotRewardOutcome,
    TransitionAssemblyInput,
    TransitionOutcomeBatch,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent
TRANSITION_MODULES = ("transition_inputs.py", "transition_assembler.py")

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


def make_candidate_spec(max_candidates: int = 8) -> CandidateSetSpec:
    return CandidateSetSpec(
        schema_version="0.1.0",
        max_candidates=max_candidates,
        max_headings_per_candidate=1,
        deterministic_ordering=True,
        deduplication_distance=0.5,
        hold_policy=HoldPolicy(),
    )


def make_robot(robot_id: int = 0) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id, xy=(1.0, 1.0), safety_radius=0.5, sensor_range=4.0,
        vision_model="cone", theta=0.0,
    )


def make_candidate(target=(4.0, 6.0)) -> CandidateCaptureInput:
    return CandidateCaptureInput(
        candidate=ExplorationCandidate(target=target, source="frontier", information_gain=1.0),
        kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=True, reachable=True,
    )


def build_decision(geometry, episode_id="ep-boundary", decision_step=0, time_s=0.0):
    candidate_spec = make_candidate_spec()
    robot_capture = RobotActorCaptureInput(
        robot=make_robot(0), candidates=(make_candidate(),), graph_edges=(),
        visible_teammates=(), hazard_belief=HazardBelief(geometry).snapshot(),
    )
    frame = RuntimeActorFrame(
        episode_id=episode_id, decision_step=decision_step, time_s=time_s,
        robots=(robot_capture,), grid_geometry=geometry, normalization=NORMALIZATION,
        candidate_spec=candidate_spec,
    )
    assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(
            schema=build_feature_schema_v0(), candidate_spec=candidate_spec
        ),
        catalog_assembler=ActionCatalogAssembler(),
    )
    return assembler.build(frame)


def make_critic_state(decision_step=0, time_s=0.0) -> CriticState:
    return CriticState(
        schema_version="0.1.0", decision_step=decision_step, time_s=time_s,
        global_feature_names=("coverage",), global_features=(0.5,),
        per_robot_feature_names=(), per_robot_features={},
    )


def make_ground_truth(fire_xy, coverage) -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        schema_version="0.1.0", decision_step=0, time_s=0.0, true_robot_poses={},
        true_occupancy=(), true_fire_locations=(fire_xy,), global_coverage_fraction=coverage,
    )


def make_metadata(episode_id="ep-boundary") -> EpisodeMetadata:
    bundle_hash = compute_contract_bundle_hash(build_contract_manifest())
    return EpisodeMetadata(
        episode_id=episode_id, seed=1, map_id="map-1", robot_count=1, fire_count=1,
        sensor_range=4.0, field_of_view_deg=120.0, communication_range=15.0, max_steps=100,
        simulator_commit="deadbeef", contract_versions=dict(CONTRACT_VERSIONS),
        contract_bundle_hash=bundle_hash,
    )


def build_terminal_transition(geometry, episode_id):
    current = build_decision(geometry, episode_id=episode_id, decision_step=0)
    selections = DecisionSelectionBatch(
        episode_id=episode_id, decision_step=0,
        selections=(RobotActionSelection(robot_id=0, action_index=0, issued_at_step=0),),
    )
    outcome = TransitionOutcomeBatch(
        episode_id=episode_id, decision_step=0,
        rewards=(
            RobotRewardOutcome(
                robot_id=0,
                components=(
                    RewardComponent(
                        name="new_coverage", raw_value=1.0, applied_weight=0.5, weighted_value=0.5
                    ),
                ),
            ),
        ),
        terminated=True, truncated=False, termination_reason=TerminationReason.MAX_STEPS,
    )
    build_input = TransitionAssemblyInput(
        current_decision=current, selections=selections, outcome=outcome,
        next_decision=None, critic_state=make_critic_state(),
    )
    return LearningTransitionAssembler().build(build_input)


class TestGroundTruthOnlyInRecorderAppend:
    def test_two_hidden_ground_truth_worlds_give_identical_transitions_and_episode_records(self):
        geometry = make_geometry()

        # Two conceptually different hidden ground-truth worlds.  Neither is
        # ever passed to TransitionAssemblyInput or LearningTransitionAssembler
        # -- there is no parameter that could take them.
        gt_a = make_ground_truth((8.0, 8.0), coverage=0.2)
        gt_b = make_ground_truth((1.0, 1.0), coverage=0.9)
        assert gt_a != gt_b

        transition_a = build_terminal_transition(geometry, episode_id="ep-boundary")
        transition_b = build_terminal_transition(geometry, episode_id="ep-boundary")
        assert transition_a == transition_b  # identical, despite the two hidden worlds

        recorder_a = InMemoryTrajectoryRecorder()
        recorder_a.start_episode(make_metadata(episode_id="ep-boundary"))
        recorder_a.append(transition_a, ground_truth=gt_a)
        record_a = recorder_a.finish_episode()

        recorder_b = InMemoryTrajectoryRecorder()
        recorder_b.start_episode(make_metadata(episode_id="ep-boundary"))
        recorder_b.append(transition_b, ground_truth=gt_b)
        record_b = recorder_b.finish_episode()

        assert record_a.transitions == record_b.transitions
        assert record_a.ground_truth_by_step == ((0, gt_a),)
        assert record_b.ground_truth_by_step == ((0, gt_b),)
        assert record_a.ground_truth_by_step != record_b.ground_truth_by_step

    def test_learning_transition_contains_no_ground_truth_snapshot(self):
        transition = build_terminal_transition(make_geometry(), episode_id="ep-boundary-2")
        for field in dataclasses.fields(LearningTransition):
            value = getattr(transition, field.name)
            assert not isinstance(value, GroundTruthSnapshot), field.name
            if isinstance(value, dict):
                for item in value.values():
                    assert not isinstance(item, GroundTruthSnapshot), field.name

    def test_episode_record_keeps_ground_truth_only_in_dedicated_block(self):
        geometry = make_geometry()
        transition = build_terminal_transition(geometry, episode_id="ep-boundary-3")
        gt = make_ground_truth((3.0, 3.0), coverage=0.4)

        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata(episode_id="ep-boundary-3"))
        recorder.append(transition, ground_truth=gt)
        record = recorder.finish_episode()

        assert record.ground_truth_by_step == ((0, gt),)
        for t in record.transitions:
            for field in dataclasses.fields(LearningTransition):
                assert not isinstance(getattr(t, field.name), GroundTruthSnapshot)


class TestNoArbitraryMetadata:
    @pytest.mark.parametrize(
        "cls",
        [
            TransitionAssemblyInput,
            TransitionOutcomeBatch,
            RobotRewardOutcome,
            DecisionSelectionBatch,
            RobotActionSelection,
        ],
    )
    def test_no_metadata_field(self, cls):
        names = {f.name for f in dataclasses.fields(cls)}
        assert "metadata" not in names


class TestPrivilegedImports:
    FORBIDDEN_ROOTS = ("PyQt5", "PyQt6", "PySide2", "PySide6", "numpy", "torch", "pandas")
    FORBIDDEN_MODULES = (
        "robotics_sim.simulation.engine",
        "robotics_sim.app",
        "robotics_sim.environment.hazard_field",
        "robotics_sim.diagnostics",
    )
    FORBIDDEN_NAMES = ("HazardField", "FireSource", "HazardDebug", "HazardSourceDebug")

    def _imports(self, filename: str) -> Iterator[tuple[str, str]]:
        tree = ast.parse((LEARNING_DIR / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name, ""
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    yield node.module, alias.name

    @pytest.mark.parametrize("filename", TRANSITION_MODULES)
    def test_no_privileged_imports(self, filename):
        for module, name in self._imports(filename):
            root = module.split(".")[0]
            assert root not in self.FORBIDDEN_ROOTS, (filename, module)
            assert not root.lower().startswith(("pyqt", "pyside")), (filename, module)
            for forbidden in self.FORBIDDEN_MODULES:
                assert not module.startswith(forbidden), (filename, module)
            assert name not in self.FORBIDDEN_NAMES, (filename, module, name)

    def test_transition_assembler_does_not_import_ground_truth_snapshot(self):
        # GroundTruthSnapshot is only ever accepted by
        # episode_session.complete_current_decision() and forwarded straight
        # to recorder.append(); the assembler module has no reason to know
        # about it at all.
        for module, name in self._imports("transition_assembler.py"):
            assert name != "GroundTruthSnapshot", (module, name)
