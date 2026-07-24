"""Tests for the in-memory trajectory recorder plus a synthetic
end-to-end smoke episode over the full contract pipeline."""

from __future__ import annotations

import dataclasses
import inspect
import os

import pytest

from robotics_interfaces.learning import (
    ActorObservation,
    CandidateKind,
    CandidateObservation,
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
from robotics_sim.learning import (
    ActorObservationBuildInput,
    ActorObservationBuilder,
    CandidateFeatureSource,
    ContractBundleHashMismatchError,
    CriticStateBuildInput,
    CriticStateBuilder,
    DuplicateDecisionStepError,
    EpisodeIdMismatchError,
    FeatureSchema,
    GroundTruthBuildInput,
    GroundTruthSnapshotBuilder,
    InMemoryTrajectoryRecorder,
    RecorderStateError,
    TeammateFeatureSource,
)

VALID_HASH = compute_contract_bundle_hash(build_contract_manifest())


def make_metadata(episode_id: str = "ep-1", **overrides) -> EpisodeMetadata:
    kwargs = dict(
        episode_id=episode_id,
        seed=7,
        map_id="synthetic",
        robot_count=1,
        fire_count=1,
        sensor_range=5.0,
        field_of_view_deg=90.0,
        communication_range=10.0,
        max_steps=100,
        simulator_commit="deadbeef",
        contract_versions={"ObservationSpec": "0.1.0"},
        contract_bundle_hash=VALID_HASH,
    )
    kwargs.update(overrides)
    return EpisodeMetadata(**kwargs)


def make_actor_observation(step: int) -> ActorObservation:
    return ActorObservation(
        schema_version="0.1.0",
        robot_id=0,
        decision_step=step,
        time_s=float(step),
        robot_feature_names=("x",),
        robot_features=(1.0,),
        candidate_feature_names=("dist",),
        candidate_features=((1.0,),),
        candidate_ids=("c0",),
        action_mask=(True,),
        graph_edges=(),
        visible_teammate_feature_names=(),
        visible_teammate_features=(),
    )


def make_critic_state(step: int) -> CriticState:
    return CriticState(
        schema_version="0.1.0",
        decision_step=step,
        time_s=float(step),
        global_feature_names=("coverage",),
        global_features=(0.5,),
        per_robot_feature_names=(),
        per_robot_features={},
    )


def make_transition(step: int, episode_id: str = "ep-1") -> LearningTransition:
    return LearningTransition(
        schema_version="0.1.0",
        episode_id=episode_id,
        decision_step=step,
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
        terminated=False,
        truncated=False,
        termination_reason=TerminationReason.RUNNING,
    )


def make_ground_truth(step: int) -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        schema_version="0.1.0",
        decision_step=step,
        time_s=float(step),
        true_robot_poses={0: (1.0, 2.0, 0.0)},
        true_occupancy=((0, 1),),
        true_fire_locations=((3.0, 4.0),),
        global_coverage_fraction=0.5,
    )


class TestRecorderHappyPath:
    def test_start_append_finish(self):
        recorder = InMemoryTrajectoryRecorder()
        assert recorder.is_recording is False
        recorder.start_episode(make_metadata())
        assert recorder.is_recording is True
        recorder.append(make_transition(0), make_ground_truth(0))
        recorder.append(make_transition(1), make_ground_truth(1))
        record = recorder.finish_episode()
        assert recorder.is_recording is False
        assert len(record.transitions) == 2
        assert record.metadata.episode_id == "ep-1"

    def test_record_is_immutable(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(0))
        record = recorder.finish_episode()
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.transitions = ()
        assert isinstance(record.transitions, tuple)
        assert isinstance(record.ground_truth_by_step, tuple)

    def test_ground_truth_stored_separately_from_transitions(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(0), make_ground_truth(0))
        record = recorder.finish_episode()
        assert record.ground_truth_by_step == ((0, make_ground_truth(0)),)
        transition_fields = {f.name for f in dataclasses.fields(record.transitions[0])}
        assert not any("ground_truth" in name for name in transition_fields)

    def test_non_consecutive_but_increasing_steps_are_allowed(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(0))
        recorder.append(make_transition(5))
        recorder.append(make_transition(6))
        assert len(recorder.finish_episode().transitions) == 3

    def test_set_fire_metrics(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(0))
        recorder.set_fire_metrics(
            EpisodeFireMetrics(fire_crossing_time_s=2.5, fire_overflight_distance=7.0)
        )
        record = recorder.finish_episode()
        assert record.fire_metrics.fire_crossing_time_s == 2.5

    def test_finish_allows_new_episode(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata("ep-1"))
        recorder.append(make_transition(0))
        first = recorder.finish_episode()
        recorder.start_episode(make_metadata("ep-2"))
        recorder.append(make_transition(0, episode_id="ep-2"))
        second = recorder.finish_episode()
        assert first.metadata.episode_id == "ep-1"
        assert second.metadata.episode_id == "ep-2"
        assert first.transitions[0].episode_id == "ep-1"
        assert len(second.transitions) == 1


class TestRecorderRejections:
    def test_episode_id_mismatch(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata("ep-1"))
        with pytest.raises(EpisodeIdMismatchError):
            recorder.append(make_transition(0, episode_id="ep-other"))

    def test_duplicate_step_fails(self):
        # step equal to one already recorded: rejected as a duplicate.
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(3))
        with pytest.raises(DuplicateDecisionStepError):
            recorder.append(make_transition(3))

    def test_smaller_step_after_larger_one_is_now_valid(self):
        # decision_step is episode-global, assigned when a decision opens --
        # arrival order at the recorder is unconstrained, only uniqueness is
        # enforced. A smaller step arriving after a larger one (the real
        # multi-robot runtime is asynchronous) must not raise.
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(3))
        recorder.append(make_transition(2))  # must not raise
        record = recorder.finish_episode()
        assert [t.decision_step for t in record.transitions] == [2, 3]

    def test_append_without_active_episode(self):
        with pytest.raises(RecorderStateError):
            InMemoryTrajectoryRecorder().append(make_transition(0))

    def test_double_start(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        with pytest.raises(RecorderStateError):
            recorder.start_episode(make_metadata("ep-2"))

    def test_double_finish(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.finish_episode()
        with pytest.raises(RecorderStateError):
            recorder.finish_episode()

    def test_wrong_contract_hash_rejected(self):
        recorder = InMemoryTrajectoryRecorder()
        with pytest.raises(ContractBundleHashMismatchError):
            recorder.start_episode(make_metadata(contract_bundle_hash="0" * 64))
        assert recorder.is_recording is False

    def test_correct_contract_hash_accepted(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata(contract_bundle_hash=VALID_HASH))
        assert recorder.is_recording is True

    def test_duplicate_step_rejected_even_with_ground_truth_attached(self):
        # decision_step uniqueness is enforced on the transition itself, so
        # a second ground-truth snapshot for an already-recorded step is
        # rejected as a duplicate step -- not as a special ground-truth rule.
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(3), make_ground_truth(3))
        with pytest.raises(DuplicateDecisionStepError):
            recorder.append(make_transition(3), make_ground_truth(3))

    def test_ground_truth_inside_transition_rejected_defensively(self):
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        bad = dataclasses.replace(make_transition(0))
        object.__setattr__(bad, "reward_total_by_robot", {0: 1.0, 1: make_ground_truth(0)})
        with pytest.raises(RecorderStateError):
            recorder.append(bad)


class TestRecorderIsMemoryOnly:
    def test_no_filesystem_paths_in_public_api(self):
        suspicious = ("path", "file", "dir", "folder", "save", "write", "dump")
        for name, member in inspect.getmembers(InMemoryTrajectoryRecorder):
            if name.startswith("_"):
                continue
            assert not any(s in name.lower() for s in suspicious), name
            if callable(member):
                for param in inspect.signature(member).parameters:
                    assert not any(s in param.lower() for s in suspicious), (name, param)

    def test_recording_creates_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata())
        recorder.append(make_transition(0), make_ground_truth(0))
        recorder.set_fire_metrics(EpisodeFireMetrics())
        recorder.finish_episode()
        assert os.listdir(tmp_path) == []


class TestSyntheticSmokeEpisode:
    """Full contract pipeline on synthetic data: builders -> contracts ->
    recorder.  No simulator, no engine, no GUI, no external maps."""

    SCHEMA = FeatureSchema(
        robot_feature_names=("x", "y"),
        candidate_feature_names=("dist", "gain"),
        teammate_feature_names=("rel_x",),
    )

    def _candidate(self, candidate_id: str, reachable: bool) -> CandidateObservation:
        return CandidateObservation(
            candidate_id=candidate_id,
            kind=CandidateKind.FRONTIER_VIEWPOINT,
            xy=(1.0, 2.0),
            heading_candidates=(0.0,),
            source="synthetic",
            reachable=reachable,
            rejection_reasons=() if reachable else ("unreachable",),
        )

    def _actor_observation(self, step: int) -> ActorObservation:
        return ActorObservationBuilder().build(
            ActorObservationBuildInput(
                schema=self.SCHEMA,
                robot_id=0,
                decision_step=step,
                time_s=float(step),
                robot_features={"x": 1.0 + step, "y": 2.0},
                candidates=(
                    CandidateFeatureSource(
                        candidate=self._candidate(f"valid-{step}", reachable=True),
                        features={"dist": 1.0, "gain": 0.5},
                        enabled=True,
                    ),
                    CandidateFeatureSource(
                        candidate=self._candidate(f"invalid-{step}", reachable=False),
                        features={"dist": 9.0, "gain": 0.0},
                        enabled=False,
                    ),
                ),
                graph_edges=((0, 1),),
                visible_teammates=(),
            )
        )

    def test_three_decision_episode(self):
        critic_builder = CriticStateBuilder()
        gt_builder = GroundTruthSnapshotBuilder()
        recorder = InMemoryTrajectoryRecorder()
        recorder.start_episode(make_metadata("smoke-ep"))

        for step in range(3):
            last = step == 2
            transition = LearningTransition(
                schema_version="0.1.0",
                episode_id="smoke-ep",
                decision_step=step,
                actor_observations={0: self._actor_observation(step)},
                critic_state=critic_builder.build(
                    CriticStateBuildInput(
                        decision_step=step,
                        time_s=float(step),
                        global_feature_names=("coverage",),
                        global_features={"coverage": 0.2 * (step + 1)},
                        per_robot_feature_names=("x",),
                        per_robot_features={0: {"x": 1.0 + step}},
                    )
                ),
                selected_actions={
                    0: LearningAction(
                        robot_id=0, candidate_id=f"valid-{step}", candidate_index=0,
                        heading_index=0, action_index=0, issued_at_step=step,
                    )
                },
                reward_components_by_robot={
                    0: (
                        RewardComponent(name="new_coverage", raw_value=0.2,
                                        applied_weight=1.0, weighted_value=0.2),
                        RewardComponent(name="path_cost", raw_value=-0.1,
                                        applied_weight=0.5, weighted_value=-0.05),
                    )
                },
                reward_total_by_robot={0: 0.15},
                next_actor_observations={0: self._actor_observation(step + 1)},
                terminated=last,
                truncated=False,
                termination_reason=(
                    TerminationReason.COVERAGE_COMPLETE if last else TerminationReason.RUNNING
                ),
            )
            ground_truth = gt_builder.build(
                GroundTruthBuildInput(
                    decision_step=step,
                    time_s=float(step),
                    true_robot_poses={0: (1.0 + step, 2.0, 0.0)},
                    true_occupancy=((0, 1), (1, 0)),
                    true_fire_locations=((6.0, 6.0),),
                    global_coverage_fraction=0.2 * (step + 1),
                )
            )
            recorder.append(transition, ground_truth)

        recorder.set_fire_metrics(
            EpisodeFireMetrics(fire_crossing_time_s=0.0, fire_overflight_distance=0.0)
        )
        record = recorder.finish_episode()

        assert len(record.transitions) == 3
        assert [step for step, _ in record.ground_truth_by_step] == [0, 1, 2]
        assert record.transitions[-1].terminated is True
        assert record.transitions[-1].termination_reason is TerminationReason.COVERAGE_COMPLETE
        first_obs = record.transitions[0].actor_observations[0]
        assert first_obs.action_mask == (True, False)
        assert first_obs.candidate_ids == ("valid-0", "invalid-0")
        assert record.metadata.contract_bundle_hash == VALID_HASH
        assert record.fire_metrics is not None
