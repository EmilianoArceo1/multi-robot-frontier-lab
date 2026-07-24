"""Tests for InMemoryTrajectoryRecorder accepting transitions with unique,
but not necessarily increasing, decision_step values -- and producing a
deterministically decision_step-ordered EpisodeRecord regardless of
arrival order."""

from __future__ import annotations

import dataclasses
import os

import pytest

from robotics_interfaces.learning import (
    ActorObservation,
    CriticState,
    EpisodeFireMetrics,
    EpisodeMetadata,
    GroundTruthSnapshot,
    LearningAction,
    LearningTransition,
    RewardComponent,
    TerminationReason,
    build_contract_manifest,
    compute_contract_bundle_hash,
)
from robotics_sim.learning import DuplicateDecisionStepError, InMemoryTrajectoryRecorder

VALID_HASH = compute_contract_bundle_hash(build_contract_manifest())


def make_metadata(episode_id: str = "ep-1", **overrides) -> EpisodeMetadata:
    kwargs = dict(
        episode_id=episode_id, seed=7, map_id="synthetic", robot_count=1, fire_count=1,
        sensor_range=5.0, field_of_view_deg=90.0, communication_range=10.0, max_steps=100,
        simulator_commit="deadbeef", contract_versions={"ObservationSpec": "0.1.0"},
        contract_bundle_hash=VALID_HASH,
    )
    kwargs.update(overrides)
    return EpisodeMetadata(**kwargs)


def make_actor_observation(step: int) -> ActorObservation:
    return ActorObservation(
        schema_version="0.1.0", robot_id=0, decision_step=step, time_s=float(step),
        robot_feature_names=("x",), robot_features=(1.0,),
        candidate_feature_names=("dist",), candidate_features=((1.0,),),
        candidate_ids=("c0",), action_mask=(True,), graph_edges=(),
        visible_teammate_feature_names=(), visible_teammate_features=(),
    )


def make_critic_state(step: int) -> CriticState:
    return CriticState(
        schema_version="0.1.0", decision_step=step, time_s=float(step),
        global_feature_names=("coverage",), global_features=(0.5,),
        per_robot_feature_names=(), per_robot_features={},
    )


def make_transition(step: int, episode_id: str = "ep-1") -> LearningTransition:
    return LearningTransition(
        schema_version="0.1.0", episode_id=episode_id, decision_step=step,
        actor_observations={0: make_actor_observation(step)},
        critic_state=make_critic_state(step),
        selected_actions={
            0: LearningAction(
                robot_id=0, candidate_id="c0", candidate_index=0,
                heading_index=0, action_index=0, issued_at_step=step,
            )
        },
        reward_components_by_robot={
            0: (RewardComponent(name="new_coverage", raw_value=1.0,
                                applied_weight=1.0, weighted_value=1.0),)
        },
        reward_total_by_robot={0: 1.0},
        next_actor_observations={0: make_actor_observation(step + 1)},
        terminated=False, truncated=False, termination_reason=TerminationReason.RUNNING,
    )


def make_ground_truth(step: int) -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        schema_version="0.1.0", decision_step=step, time_s=float(step),
        true_robot_poses={0: (1.0, 2.0, 0.0)}, true_occupancy=((0, 1),),
        true_fire_locations=((3.0, 4.0),), global_coverage_fraction=0.5,
    )


class TestOutOfOrderAppendAccepted:
    def test_append_8_then_7_is_valid(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(8))
        recorder.append(make_transition(7))  # must not raise
        record = recorder.finish_episode()
        assert len(record.transitions) == 2

    def test_finish_produces_7_then_8(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(8))
        recorder.append(make_transition(7))
        record = recorder.finish_episode()
        assert [t.decision_step for t in record.transitions] == [7, 8]

    def test_append_10_2_6_produces_2_6_10(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(10))
        recorder.append(make_transition(2))
        recorder.append(make_transition(6))
        record = recorder.finish_episode()
        assert [t.decision_step for t in record.transitions] == [2, 6, 10]

    def test_non_consecutive_steps_are_valid(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(0))
        recorder.append(make_transition(50))
        recorder.append(make_transition(51))
        record = recorder.finish_episode()
        assert [t.decision_step for t in record.transitions] == [0, 50, 51]


class TestDuplicateStepRejected:
    def test_duplicate_step_fails(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(5))
        with pytest.raises(DuplicateDecisionStepError):
            recorder.append(make_transition(5))

    def test_duplicate_step_fails_regardless_of_arrival_order(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(9))
        recorder.append(make_transition(3))
        with pytest.raises(DuplicateDecisionStepError):
            recorder.append(make_transition(3))


class TestGroundTruthOrdering:
    def test_ground_truth_arrives_out_of_order_and_comes_out_sorted(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(8), make_ground_truth(8))
        recorder.append(make_transition(3), make_ground_truth(3))
        recorder.append(make_transition(5), make_ground_truth(5))
        record = recorder.finish_episode()
        assert [step for step, _ in record.ground_truth_by_step] == [3, 5, 8]

    def test_ground_truth_is_optional(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(4))  # no ground_truth
        recorder.append(make_transition(1), make_ground_truth(1))
        record = recorder.finish_episode()
        assert record.ground_truth_by_step == ((1, make_ground_truth(1)),)
        assert len(record.transitions) == 2

    def test_transition_and_ground_truth_remain_separate(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(2), make_ground_truth(2))
        record = recorder.finish_episode()
        transition_fields = {f.name for f in dataclasses.fields(record.transitions[0])}
        assert not any("ground_truth" in name for name in transition_fields)
        assert record.ground_truth_by_step == ((2, make_ground_truth(2)),)


class TestAbortAndReuse:
    def test_abort_clears_used_steps(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata("ep-a"))
        recorder.append(make_transition(4, episode_id="ep-a"))
        recorder.abort_episode()

        recorder.start_episode(make_metadata("ep-b"))
        recorder.append(make_transition(4, episode_id="ep-b"))  # must not raise
        record = recorder.finish_episode()
        assert record.transitions[0].decision_step == 4

    def test_second_episode_can_reuse_the_same_steps(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata("ep-1"))
        recorder.append(make_transition(0, episode_id="ep-1"))
        recorder.append(make_transition(1, episode_id="ep-1"))
        recorder.finish_episode()

        recorder.start_episode(make_metadata("ep-2"))
        recorder.append(make_transition(0, episode_id="ep-2"))
        recorder.append(make_transition(1, episode_id="ep-2"))
        record = recorder.finish_episode()
        assert [t.decision_step for t in record.transitions] == [0, 1]


class TestFireMetricsPreserved:
    def test_fire_metrics_preserved_with_out_of_order_appends(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(9))
        recorder.append(make_transition(1))
        recorder.set_fire_metrics(
            EpisodeFireMetrics(fire_crossing_time_s=3.0, fire_overflight_distance=1.5)
        )
        record = recorder.finish_episode()
        assert record.fire_metrics.fire_crossing_time_s == 3.0
        assert record.fire_metrics.fire_overflight_distance == 1.5


class TestNoFilesystemWrites:
    def test_recording_out_of_order_creates_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(5), make_ground_truth(5))
        recorder.append(make_transition(1), make_ground_truth(1))
        recorder.finish_episode()
        assert os.listdir(tmp_path) == []
