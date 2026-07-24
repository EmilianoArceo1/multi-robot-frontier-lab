"""Tests for build_candidate_id: deterministic, position-based candidate
identity, local to one decision step -- never a persistent spatial identity."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

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
from robotics_sim.learning.observation_batch import (
    ActorObservationBatchAssembler,
    build_candidate_id,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent

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


def make_robot(robot_id: int = 0) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(1.0, 1.0),
        safety_radius=0.5,
        sensor_range=4.0,
        vision_model="cone",
        theta=0.0,
    )


def make_candidate_capture(target=(4.0, 6.0), metadata=None) -> CandidateCaptureInput:
    return CandidateCaptureInput(
        candidate=ExplorationCandidate(
            target=target, source="frontier", metadata=metadata or {}
        ),
        kind=CandidateKind.FRONTIER_VIEWPOINT,
        enabled=True,
        reachable=True,
    )


def make_frame(
    candidates: tuple[CandidateCaptureInput, ...], robot_id: int = 0, decision_step: int = 0
) -> RuntimeActorFrame:
    geometry = make_geometry()
    robot_capture = RobotActorCaptureInput(
        robot=make_robot(robot_id),
        candidates=candidates,
        graph_edges=(),
        visible_teammates=(),
        hazard_belief=HazardBelief(geometry).snapshot(),
    )
    return RuntimeActorFrame(
        episode_id="ep-identity",
        decision_step=decision_step,
        time_s=0.0,
        robots=(robot_capture,),
        grid_geometry=geometry,
        normalization=NORMALIZATION,
        candidate_spec=make_candidate_spec(),
    )


class TestFormat:
    def test_exact_format(self):
        assert build_candidate_id(2, 17, 4) == "robot-2/step-17/candidate-4"

    def test_zero_values(self):
        assert build_candidate_id(0, 0, 0) == "robot-0/step-0/candidate-0"


class TestDeterminism:
    def test_same_arguments_give_same_id(self):
        assert build_candidate_id(3, 5, 1) == build_candidate_id(3, 5, 1)


class TestDistinctness:
    def test_different_index_gives_different_id(self):
        assert build_candidate_id(0, 0, 0) != build_candidate_id(0, 0, 1)

    def test_different_step_gives_different_id(self):
        assert build_candidate_id(0, 0, 0) != build_candidate_id(0, 1, 0)

    def test_different_robot_gives_different_id(self):
        assert build_candidate_id(0, 0, 0) != build_candidate_id(1, 0, 0)


class TestRejectsNegativeIndices:
    @pytest.mark.parametrize(
        "robot_id,decision_step,candidate_index", [(-1, 0, 0), (0, -1, 0), (0, 0, -1)]
    )
    def test_negative_values_rejected(self, robot_id, decision_step, candidate_index):
        with pytest.raises(ValueError):
            build_candidate_id(robot_id, decision_step, candidate_index)


class TestNoHashNoCoordinates:
    def test_source_never_calls_hash(self):
        tree = ast.parse((LEARNING_DIR / "observation_batch.py").read_text(encoding="utf-8"))
        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assert "hash" not in calls

    def test_signature_has_no_candidate_or_coordinate_parameters(self):
        params = set(inspect.signature(build_candidate_id).parameters)
        assert params == {"robot_id", "decision_step", "candidate_index"}


class TestIdentityIgnoresMetadataAndTarget:
    def test_changing_metadata_does_not_change_id(self):
        assembler = ActorObservationBatchAssembler(
            schema=build_feature_schema_v0(), candidate_spec=make_candidate_spec()
        )
        frame_a = make_frame((make_candidate_capture(metadata={"note": "a"}),))
        frame_b = make_frame((make_candidate_capture(metadata={"note": "different"}),))

        ids_a = assembler.build(frame_a).observations[0].candidate_ids
        ids_b = assembler.build(frame_b).observations[0].candidate_ids
        assert ids_a == ids_b == ("robot-0/step-0/candidate-0",)

    def test_changing_target_without_changing_index_does_not_change_id(self):
        assembler = ActorObservationBatchAssembler(
            schema=build_feature_schema_v0(), candidate_spec=make_candidate_spec()
        )
        frame_a = make_frame((make_candidate_capture(target=(4.0, 6.0)),))
        frame_b = make_frame((make_candidate_capture(target=(9.0, 1.0)),))

        ids_a = assembler.build(frame_a).observations[0].candidate_ids
        ids_b = assembler.build(frame_b).observations[0].candidate_ids
        assert ids_a == ids_b == ("robot-0/step-0/candidate-0",)
