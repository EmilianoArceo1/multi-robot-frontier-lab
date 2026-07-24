"""Tests for InMemoryLearningEpisodeSession: the IDLE -> ACTIVE ->
TERMINATED -> IDLE state machine, and a full synthetic episode smoke test."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.learning import (
    CONTRACT_VERSIONS,
    CandidateKind,
    CandidateSetSpec,
    CriticState,
    EpisodeFireMetrics,
    EpisodeMetadata,
    GroundTruthSnapshot,
    HoldPolicy,
    TerminationReason,
    build_contract_manifest,
    compute_contract_bundle_hash,
)
from robotics_interfaces.learning.transitions import RewardComponent
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig, build_feature_schema_v0
from robotics_sim.learning.action_catalog import ActionCatalogAssembler
from robotics_sim.learning.capture_inputs import (
    CandidateCaptureInput,
    RobotActorCaptureInput,
    RuntimeActorFrame,
)
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler
from robotics_sim.learning.episode_session import (
    InMemoryLearningEpisodeSession,
    SessionState,
    SessionStateError,
)
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler
from robotics_sim.learning.recorder import InMemoryTrajectoryRecorder
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import (
    DecisionSelectionBatch,
    RobotActionSelection,
    RobotRewardOutcome,
    TransitionOutcomeBatch,
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


def make_candidate_spec(max_candidates: int = 8) -> CandidateSetSpec:
    return CandidateSetSpec(
        schema_version="0.1.0",
        max_candidates=max_candidates,
        max_headings_per_candidate=1,
        deterministic_ordering=True,
        deduplication_distance=0.5,
        hold_policy=HoldPolicy(),
    )


def make_robot(robot_id: int = 0, xy=(1.0, 1.0)) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id, xy=xy, safety_radius=0.5, sensor_range=4.0,
        vision_model="cone", theta=0.0,
    )


def make_candidate(target=(4.0, 6.0), heading_rad=None) -> CandidateCaptureInput:
    return CandidateCaptureInput(
        candidate=ExplorationCandidate(
            target=target, source="frontier", information_gain=1.0, heading_rad=heading_rad
        ),
        kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=True, reachable=True,
    )


def build_decision(
    geometry, robots_spec, episode_id="ep-session", decision_step=0, time_s=0.0, candidate_spec=None
):
    candidate_spec = candidate_spec or make_candidate_spec()
    robot_captures = tuple(
        RobotActorCaptureInput(
            robot=make_robot(rid), candidates=candidates, graph_edges=(),
            visible_teammates=(), hazard_belief=HazardBelief(geometry).snapshot(),
        )
        for rid, candidates in robots_spec
    )
    frame = RuntimeActorFrame(
        episode_id=episode_id, decision_step=decision_step, time_s=time_s,
        robots=robot_captures, grid_geometry=geometry, normalization=NORMALIZATION,
        candidate_spec=candidate_spec,
    )
    assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(
            schema=build_feature_schema_v0(), candidate_spec=candidate_spec
        ),
        catalog_assembler=ActionCatalogAssembler(),
    )
    return assembler.build(frame)


def make_metadata(episode_id="ep-session") -> EpisodeMetadata:
    bundle_hash = compute_contract_bundle_hash(build_contract_manifest())
    return EpisodeMetadata(
        episode_id=episode_id, seed=1, map_id="map-1", robot_count=1, fire_count=1,
        sensor_range=4.0, field_of_view_deg=120.0, communication_range=15.0, max_steps=100,
        simulator_commit="deadbeef", contract_versions=dict(CONTRACT_VERSIONS),
        contract_bundle_hash=bundle_hash,
    )


def make_critic_state(decision_step=0) -> CriticState:
    return CriticState(
        schema_version="0.1.0", decision_step=decision_step, time_s=float(decision_step),
        global_feature_names=("coverage",), global_features=(0.5,),
        per_robot_feature_names=(), per_robot_features={},
    )


def make_ground_truth(step) -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        schema_version="0.1.0", decision_step=step, time_s=float(step), true_robot_poses={},
        true_occupancy=(), true_fire_locations=((float(step), float(step)),),
        global_coverage_fraction=min(0.1 * step, 1.0),
    )


def make_reward_component(name="new_coverage", raw=1.0, weight=0.5) -> RewardComponent:
    return RewardComponent(name=name, raw_value=raw, applied_weight=weight, weighted_value=raw * weight)


def make_selections(episode_id, decision_step, robot_ids, action_index=0):
    return DecisionSelectionBatch(
        episode_id=episode_id, decision_step=decision_step,
        selections=tuple(
            RobotActionSelection(robot_id=rid, action_index=action_index, issued_at_step=decision_step)
            for rid in robot_ids
        ),
    )


def make_outcome(episode_id, decision_step, robot_ids, terminated=False, truncated=False, reason=None):
    if reason is None:
        reason = TerminationReason.RUNNING if not (terminated or truncated) else TerminationReason.MAX_STEPS
    return TransitionOutcomeBatch(
        episode_id=episode_id, decision_step=decision_step,
        rewards=tuple(
            RobotRewardOutcome(robot_id=rid, components=(make_reward_component(),))
            for rid in robot_ids
        ),
        terminated=terminated, truncated=truncated, termination_reason=reason,
    )


def make_session() -> InMemoryLearningEpisodeSession:
    return InMemoryLearningEpisodeSession(LearningTransitionAssembler(), InMemoryTrajectoryRecorder())


class TestStateMachine:
    def test_initial_state_is_idle(self):
        session = make_session()
        assert session.state is SessionState.IDLE
        assert session.is_active is False
        assert session.current_decision is None

    def test_start_sets_active(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),))
        session.start_episode(make_metadata(), decision)
        assert session.state is SessionState.ACTIVE
        assert session.is_active is True
        assert session.current_decision is decision

    def test_full_flow_one_nonterminal_then_terminal(self):
        geometry = make_geometry()
        session = make_session()
        spec = ((0, (make_candidate(),)),)
        d0 = build_decision(geometry, spec, decision_step=0)
        d1 = build_decision(geometry, spec, decision_step=1)

        session.start_episode(make_metadata(), d0)
        t0 = session.complete_current_decision(
            make_selections("ep-session", 0, (0,)),
            make_outcome("ep-session", 0, (0,)),
            next_decision=d1,
            critic_state=make_critic_state(0),
        )
        assert session.state is SessionState.ACTIVE
        assert session.current_decision is d1  # next_decision becomes current_decision
        assert t0.terminated is False

        t1 = session.complete_current_decision(
            make_selections("ep-session", 1, (0,)),
            make_outcome("ep-session", 1, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(1),
        )
        assert session.state is SessionState.TERMINATED
        assert session.current_decision is None
        assert t1.terminated is True

        record = session.finish_episode()
        assert session.state is SessionState.IDLE
        assert record.transitions == (t0, t1)  # recorder receives transitions in order

    def test_single_transition_terminal_episode(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),))
        session.start_episode(make_metadata(), decision)
        session.complete_current_decision(
            make_selections("ep-session", 0, (0,)),
            make_outcome("ep-session", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        record = session.finish_episode()
        assert len(record.transitions) == 1
        assert record.transitions[0].terminated is True

    def test_two_robots(self):
        geometry = make_geometry()
        session = make_session()
        spec = ((0, (make_candidate(),)), (1, (make_candidate(),)))
        decision = build_decision(geometry, spec)
        session.start_episode(make_metadata(), decision)
        transition = session.complete_current_decision(
            make_selections("ep-session", 0, (0, 1)),
            make_outcome("ep-session", 0, (0, 1), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        assert set(transition.actor_observations) == {0, 1}
        record = session.finish_episode()
        assert len(record.transitions) == 1

    def test_start_twice_fails(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),))
        session.start_episode(make_metadata(), decision)
        with pytest.raises(SessionStateError):
            session.start_episode(make_metadata(episode_id="ep-other"), decision)

    def test_complete_without_start_fails(self):
        session = make_session()
        with pytest.raises(SessionStateError):
            session.complete_current_decision(
                make_selections("ep-session", 0, (0,)),
                make_outcome("ep-session", 0, (0,), terminated=True),
                next_decision=None,
                critic_state=make_critic_state(),
            )

    def test_finish_before_terminal_fails(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),))
        session.start_episode(make_metadata(), decision)
        with pytest.raises(SessionStateError):
            session.finish_episode()

    def test_complete_after_terminal_fails(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),))
        session.start_episode(make_metadata(), decision)
        session.complete_current_decision(
            make_selections("ep-session", 0, (0,)),
            make_outcome("ep-session", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        with pytest.raises(SessionStateError):
            session.complete_current_decision(
                make_selections("ep-session", 1, (0,)),
                make_outcome("ep-session", 1, (0,), terminated=True),
                next_decision=None,
                critic_state=make_critic_state(),
            )

    def test_metadata_episode_id_mismatch_rejected(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),), episode_id="ep-real")
        with pytest.raises(ValueError):
            session.start_episode(make_metadata(episode_id="ep-different"), decision)

    def test_finish_clears_session_and_allows_second_episode(self):
        geometry = make_geometry()
        session = make_session()
        d0 = build_decision(geometry, ((0, (make_candidate(),)),), episode_id="ep-first")
        session.start_episode(make_metadata(episode_id="ep-first"), d0)
        session.complete_current_decision(
            make_selections("ep-first", 0, (0,)),
            make_outcome("ep-first", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        session.finish_episode()
        assert session.state is SessionState.IDLE
        assert session.current_decision is None

        d0_second = build_decision(geometry, ((0, (make_candidate(),)),), episode_id="ep-second")
        session.start_episode(make_metadata(episode_id="ep-second"), d0_second)
        assert session.state is SessionState.ACTIVE


class TestGroundTruthAndFireMetrics:
    def test_ground_truth_passed_separately_in_order(self):
        geometry = make_geometry()
        session = make_session()
        spec = ((0, (make_candidate(),)),)
        d0 = build_decision(geometry, spec, decision_step=0)
        d1 = build_decision(geometry, spec, decision_step=1)

        session.start_episode(make_metadata(), d0)
        gt0 = make_ground_truth(0)
        session.complete_current_decision(
            make_selections("ep-session", 0, (0,)),
            make_outcome("ep-session", 0, (0,)),
            next_decision=d1,
            critic_state=make_critic_state(0),
            ground_truth=gt0,
        )
        gt1 = make_ground_truth(1)
        session.complete_current_decision(
            make_selections("ep-session", 1, (0,)),
            make_outcome("ep-session", 1, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(1),
            ground_truth=gt1,
        )
        record = session.finish_episode()
        assert record.ground_truth_by_step == ((0, gt0), (1, gt1))

    def test_fire_metrics_recorded(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),))
        session.start_episode(make_metadata(), decision)
        metrics = EpisodeFireMetrics(fire_crossing_time_s=1.5, fire_overflight_distance=3.0)
        session.set_fire_metrics(metrics)
        session.complete_current_decision(
            make_selections("ep-session", 0, (0,)),
            make_outcome("ep-session", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        record = session.finish_episode()
        assert record.fire_metrics == metrics


class TestAbort:
    def test_abort_from_active(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),))
        session.start_episode(make_metadata(), decision)
        session.abort_episode()
        assert session.state is SessionState.IDLE
        assert session.current_decision is None

    def test_abort_from_terminated(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),))
        session.start_episode(make_metadata(), decision)
        session.complete_current_decision(
            make_selections("ep-session", 0, (0,)),
            make_outcome("ep-session", 0, (0,), terminated=True),
            next_decision=None,
            critic_state=make_critic_state(),
        )
        assert session.state is SessionState.TERMINATED
        session.abort_episode()
        assert session.state is SessionState.IDLE

    def test_abort_from_idle_fails(self):
        session = make_session()
        with pytest.raises(SessionStateError):
            session.abort_episode()

    def test_abort_leaves_recorder_clean_for_a_new_episode(self):
        geometry = make_geometry()
        session = make_session()
        decision = build_decision(geometry, ((0, (make_candidate(),)),), episode_id="ep-aborted")
        session.start_episode(make_metadata(episode_id="ep-aborted"), decision)
        session.abort_episode()

        # If abort() left the recorder's internal state dirty, this would
        # raise RecorderStateError("episode ... is already active").
        new_decision = build_decision(geometry, ((0, (make_candidate(),)),), episode_id="ep-fresh")
        session.start_episode(make_metadata(episode_id="ep-fresh"), new_decision)
        assert session.state is SessionState.ACTIVE


class TestNoFilesystemWrites:
    def test_episode_session_module_touches_no_filesystem_api(self):
        tree = ast.parse((LEARNING_DIR / "episode_session.py").read_text(encoding="utf-8"))
        forbidden_modules = {"os", "pathlib", "shutil", "tempfile", "io"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_modules
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_modules
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "open"


class TestSmokeEpisode:
    def test_full_synthetic_episode_no_runtime(self):
        geometry = make_geometry()
        candidate_spec = make_candidate_spec(max_candidates=2)
        schema = build_feature_schema_v0()

        def two_candidates():
            return (
                make_candidate(target=(2.0, 2.0), heading_rad=0.2),
                make_candidate(target=(7.0, 7.0)),
            )

        def build(step):
            robot_capture = RobotActorCaptureInput(
                robot=make_robot(0), candidates=two_candidates(), graph_edges=(),
                visible_teammates=(), hazard_belief=HazardBelief(geometry).snapshot(),
            )
            frame = RuntimeActorFrame(
                episode_id="ep-smoke-session", decision_step=step, time_s=float(step),
                robots=(robot_capture,), grid_geometry=geometry, normalization=NORMALIZATION,
                candidate_spec=candidate_spec,
            )
            assembler = DecisionCaptureAssembler(
                actor_assembler=ActorObservationBatchAssembler(
                    schema=schema, candidate_spec=candidate_spec
                ),
                catalog_assembler=ActionCatalogAssembler(),
            )
            return assembler.build(frame)

        d0, d1, d2 = build(0), build(1), build(2)

        session = make_session()
        session.start_episode(make_metadata(episode_id="ep-smoke-session"), d0)

        gt0 = make_ground_truth(0)
        session.complete_current_decision(
            make_selections("ep-smoke-session", 0, (0,), action_index=0),
            TransitionOutcomeBatch(
                episode_id="ep-smoke-session", decision_step=0,
                rewards=(
                    RobotRewardOutcome(
                        robot_id=0,
                        components=(make_reward_component("new_coverage", 1.0, 0.5),),
                    ),
                ),
                terminated=False, truncated=False, termination_reason=TerminationReason.RUNNING,
            ),
            next_decision=d1,
            critic_state=make_critic_state(0),
            ground_truth=gt0,
        )

        gt1 = make_ground_truth(1)
        session.complete_current_decision(
            make_selections("ep-smoke-session", 1, (0,), action_index=1),
            TransitionOutcomeBatch(
                episode_id="ep-smoke-session", decision_step=1,
                rewards=(
                    RobotRewardOutcome(
                        robot_id=0,
                        components=(make_reward_component("fire_information_gain", 2.0, 1.0),),
                    ),
                ),
                terminated=False, truncated=False, termination_reason=TerminationReason.RUNNING,
            ),
            next_decision=d2,
            critic_state=make_critic_state(1),
            ground_truth=gt1,
        )

        gt2 = make_ground_truth(2)
        session.complete_current_decision(
            make_selections("ep-smoke-session", 2, (0,), action_index=0),
            TransitionOutcomeBatch(
                episode_id="ep-smoke-session", decision_step=2,
                rewards=(
                    RobotRewardOutcome(
                        robot_id=0,
                        components=(make_reward_component("completion", 5.0, 1.0),),
                    ),
                ),
                terminated=True, truncated=False,
                termination_reason=TerminationReason.ALL_FIRE_FOUND,
            ),
            next_decision=None,
            critic_state=make_critic_state(2),
            ground_truth=gt2,
        )

        record = session.finish_episode()

        assert [t.decision_step for t in record.transitions] == [0, 1, 2]
        assert record.transitions[0].selected_actions[0].action_index == 0
        assert record.transitions[1].selected_actions[0].action_index == 1
        assert record.transitions[2].selected_actions[0].action_index == 0
        assert record.transitions[0].reward_total_by_robot[0] == pytest.approx(0.5)
        assert record.transitions[1].reward_total_by_robot[0] == pytest.approx(2.0)
        assert record.transitions[2].reward_total_by_robot[0] == pytest.approx(5.0)
        assert record.transitions[0].terminated is False
        assert record.transitions[1].terminated is False
        assert record.transitions[2].terminated is True  # last transition is terminal
        assert record.ground_truth_by_step == ((0, gt0), (1, gt1), (2, gt2))
        assert session.state is SessionState.IDLE
