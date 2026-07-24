"""Tests for InMemoryAsynchronousLearningEpisodeSession: several robots'
learning decisions pending at once, each completed and reopened
independently, with the recorder (not the session) imposing the final
decision_step order; CriticState/GroundTruthSnapshot captured at open time;
and episode-global decision_step uniqueness enforced by the session."""

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
from robotics_sim.learning.asynchronous_episode import (
    AsynchronousEpisodeSessionStateError,
    InMemoryAsynchronousLearningEpisodeSession,
    PendingRobotDecisionError,
)
from robotics_sim.learning.capture_inputs import (
    CandidateCaptureInput,
    RobotActorCaptureInput,
    RuntimeActorFrame,
)
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler
from robotics_sim.learning.recorder import InMemoryTrajectoryRecorder
from robotics_sim.learning.runtime_decision_opening import (
    OpenedLearningDecision,
    OpenedRobotLearningDecision,
    UnresolvedCoordinationDecision,
)
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


def make_robot(robot_id: int, xy=(1.0, 1.0)) -> RobotCoordinationState:
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


def build_decision_capture(
    geometry, robot_id, decision_step, time_s, episode_id="ep-async",
    candidate_spec=None, n_candidates=1,
):
    candidate_spec = candidate_spec or make_candidate_spec()
    candidates = tuple(
        make_candidate(target=(2.0 + i, 3.0 + i)) for i in range(n_candidates)
    )
    robot_capture = RobotActorCaptureInput(
        robot=make_robot(robot_id), candidates=candidates, graph_edges=(),
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


def make_opened_robot(
    geometry, robot_id, decision_step, time_s, episode_id="ep-async",
    action_index=0, candidate_spec=None, n_candidates=1,
) -> OpenedRobotLearningDecision:
    decision_capture = build_decision_capture(
        geometry, robot_id, decision_step, time_s, episode_id, candidate_spec, n_candidates
    )
    selections = DecisionSelectionBatch(
        episode_id=episode_id, decision_step=decision_step,
        selections=(
            RobotActionSelection(
                robot_id=robot_id, action_index=action_index, issued_at_step=decision_step
            ),
        ),
    )
    return OpenedRobotLearningDecision(
        robot_id=robot_id, decision_step=decision_step, time_s=time_s,
        decision_capture=decision_capture, selections=selections,
    )


def make_opened_batch(items, unresolved=(), episode_id="ep-async", time_s=0.0) -> OpenedLearningDecision:
    return OpenedLearningDecision(
        episode_id=episode_id, time_s=time_s, assigned=tuple(items), unresolved=tuple(unresolved)
    )


def make_metadata(episode_id="ep-async") -> EpisodeMetadata:
    bundle_hash = compute_contract_bundle_hash(build_contract_manifest())
    return EpisodeMetadata(
        episode_id=episode_id, seed=1, map_id="map-1", robot_count=2, fire_count=1,
        sensor_range=4.0, field_of_view_deg=120.0, communication_range=15.0, max_steps=100,
        simulator_commit="deadbeef", contract_versions=dict(CONTRACT_VERSIONS),
        contract_bundle_hash=bundle_hash,
    )


def make_critic_state(decision_step=0, coverage=0.5) -> CriticState:
    return CriticState(
        schema_version="0.1.0", decision_step=decision_step, time_s=float(decision_step),
        global_feature_names=("coverage",), global_features=(coverage,),
        per_robot_feature_names=(), per_robot_features={},
    )


def make_ground_truth(step, fire_x=None) -> GroundTruthSnapshot:
    fire_x = float(step) if fire_x is None else fire_x
    return GroundTruthSnapshot(
        schema_version="0.1.0", decision_step=step, time_s=float(step), true_robot_poses={},
        true_occupancy=(), true_fire_locations=((fire_x, fire_x),),
        global_coverage_fraction=min(0.1 * step, 1.0),
    )


def make_reward_component(name="new_coverage", raw=1.0, weight=0.5) -> RewardComponent:
    return RewardComponent(name=name, raw_value=raw, applied_weight=weight, weighted_value=raw * weight)


def make_outcome(
    episode_id, decision_step, robot_id, terminated=False, truncated=False, reason=None,
) -> TransitionOutcomeBatch:
    if reason is None:
        reason = TerminationReason.RUNNING if not (terminated or truncated) else TerminationReason.MAX_STEPS
    return TransitionOutcomeBatch(
        episode_id=episode_id, decision_step=decision_step,
        rewards=(RobotRewardOutcome(robot_id=robot_id, components=(make_reward_component(),)),),
        terminated=terminated, truncated=truncated, termination_reason=reason,
    )


def make_session() -> InMemoryAsynchronousLearningEpisodeSession:
    return InMemoryAsynchronousLearningEpisodeSession(
        LearningTransitionAssembler(), InMemoryTrajectoryRecorder()
    )


def default_critic_states(items):
    return {item.robot_id: make_critic_state(item.decision_step) for item in items}


def register_items(
    session, items, critic_states_by_robot=None, ground_truth_by_robot=None,
    unresolved=(), episode_id="ep-async", time_s=0.0,
):
    if critic_states_by_robot is None:
        critic_states_by_robot = default_critic_states(items)
    batch = make_opened_batch(items, unresolved=unresolved, episode_id=episode_id, time_s=time_s)
    return session.register_opened_decisions(batch, critic_states_by_robot, ground_truth_by_robot)


class TestStartEpisode:
    def test_start_normal(self):
        session = make_session()
        session.start_episode(make_metadata())
        assert session.is_active is True
        assert session.episode_id == "ep-async"
        assert session.pending_count == 0
        assert session.pending_robot_ids == ()

    def test_start_twice_fails(self):
        session = make_session()
        session.start_episode(make_metadata())
        with pytest.raises(AsynchronousEpisodeSessionStateError):
            session.start_episode(make_metadata(episode_id="ep-other"))


class TestRegisterOpenedDecisions:
    def test_register_with_one_robot(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        unresolved = register_items(session, [r0])
        assert unresolved == ()
        assert session.pending_count == 1
        assert session.has_pending(0) is True

    def test_register_with_two_robots(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])
        assert session.pending_count == 2
        assert session.has_pending(0) is True
        assert session.has_pending(1) is True

    def test_pending_order_preserved(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r5 = make_opened_robot(geometry, robot_id=5, decision_step=0, time_s=0.0)
        r2 = make_opened_robot(geometry, robot_id=2, decision_step=1, time_s=0.0)
        register_items(session, [r5, r2])
        assert session.pending_robot_ids == (5, 2)

    def test_register_returns_unresolved(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        unresolved_entry = UnresolvedCoordinationDecision(
            robot_id=1, status="HOLD", reason="no candidates", candidate_count=0
        )
        unresolved = register_items(session, [r0], unresolved=(unresolved_entry,))
        assert unresolved == (unresolved_entry,)

    def test_unresolved_not_converted_to_pending(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        unresolved_entry = UnresolvedCoordinationDecision(
            robot_id=1, status="HOLD", reason="no candidates", candidate_count=0
        )
        register_items(session, [r0], unresolved=(unresolved_entry,))
        assert session.has_pending(1) is False
        assert session.pending_robot_ids == (0,)

    def test_duplicate_robot_in_same_call_rejected_by_contract(self):
        geometry = make_geometry()
        r0a = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r0b = make_opened_robot(geometry, robot_id=0, decision_step=1, time_s=0.0)
        with pytest.raises(ValueError):
            OpenedLearningDecision(
                episode_id="ep-async", time_s=0.0, assigned=(r0a, r0b), unresolved=()
            )

    def test_robot_already_pending_in_another_call_rejected(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0_first = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0_first])
        r0_second = make_opened_robot(geometry, robot_id=0, decision_step=5, time_s=0.0)
        with pytest.raises(PendingRobotDecisionError):
            register_items(session, [r0_second])

    def test_atomic_registration_when_one_robot_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0_first = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0_first])

        r1_new = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        r0_conflicting = make_opened_robot(geometry, robot_id=0, decision_step=9, time_s=0.0)
        with pytest.raises(PendingRobotDecisionError):
            register_items(session, [r1_new, r0_conflicting])

        # Nothing from the rejected batch got registered, including robot 1.
        assert session.has_pending(1) is False
        assert session.pending_robot_ids == (0,)

    def test_register_without_start_fails(self):
        geometry = make_geometry()
        session = make_session()
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        with pytest.raises(AsynchronousEpisodeSessionStateError):
            register_items(session, [r0])


class TestCriticStateRegistration:
    def test_critic_required_for_each_assigned(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(
            session, [r0, r1],
            critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
        )
        assert session.pending_count == 2

    def test_missing_critic_for_assigned_robot_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        with pytest.raises(ValueError):
            register_items(session, [r0, r1], critic_states_by_robot={0: make_critic_state(0)})
        assert session.pending_count == 0

    def test_extra_critic_for_unassigned_robot_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        with pytest.raises(ValueError):
            register_items(
                session, [r0],
                critic_states_by_robot={0: make_critic_state(0), 9: make_critic_state(9)},
            )
        assert session.pending_count == 0

    def test_critic_with_wrong_type_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        with pytest.raises(TypeError):
            register_items(
                session, [r0], critic_states_by_robot={0: make_ground_truth(0)}
            )
        assert session.pending_count == 0


class TestGroundTruthRegistration:
    def test_ground_truth_is_optional(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])  # no ground_truth_by_robot at all
        assert session.pending_count == 1

    def test_ground_truth_for_hold_or_failed_robot_rejected(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        unresolved_entry = UnresolvedCoordinationDecision(
            robot_id=1, status="HOLD", reason="no candidates", candidate_count=0
        )
        with pytest.raises(ValueError):
            register_items(
                session, [r0], unresolved=(unresolved_entry,),
                ground_truth_by_robot={0: make_ground_truth(0), 1: make_ground_truth(1)},
            )
        assert session.pending_count == 0


class TestCriticAndGroundTruthCapturedAtOpen:
    def test_critic_captured_when_registered(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        critic_a = make_critic_state(0, coverage=0.1)
        register_items(session, [r0], critic_states_by_robot={0: critic_a})
        transition = session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        assert transition.critic_state == critic_a

    def test_world_change_after_open_does_not_change_transition_critic_state(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        critic_a = make_critic_state(0, coverage=0.1)
        register_items(session, [r0], critic_states_by_robot={0: critic_a})

        # Conceptually, the world advances after the decision was opened --
        # a later critic state exists, but nothing here ever hands it to
        # the session (complete_robot_decision() has no such parameter).
        critic_b = make_critic_state(0, coverage=0.99)
        assert critic_b != critic_a

        transition = session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        assert transition.critic_state == critic_a
        assert transition.critic_state != critic_b

    def test_ground_truth_captured_when_registered(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        gt_a = make_ground_truth(0, fire_x=1.0)
        register_items(session, [r0], ground_truth_by_robot={0: gt_a})
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        record = session.finish_episode()
        assert record.ground_truth_by_step == ((0, gt_a),)


class TestCompleteRobotDecision:
    def test_complete_without_start_fails(self):
        session = make_session()
        with pytest.raises(AsynchronousEpisodeSessionStateError):
            session.complete_robot_decision(
                robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
            )

    def test_complete_unknown_robot_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(PendingRobotDecisionError):
            session.complete_robot_decision(
                robot_id=1, outcome=make_outcome("ep-async", 0, 1, terminated=True)
            )

    def test_complete_no_longer_accepts_critic_state_kwarg(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(TypeError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0, terminated=True),
                critic_state=make_critic_state(0),
            )

    def test_complete_no_longer_accepts_ground_truth_kwarg(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(TypeError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0, terminated=True),
                ground_truth=make_ground_truth(0),
            )

    def test_outcome_episode_id_mismatch_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0, outcome=make_outcome("ep-wrong", 0, 0, terminated=True)
            )

    def test_outcome_decision_step_mismatch_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0, outcome=make_outcome("ep-async", 7, 0, terminated=True)
            )

    def test_reward_for_missing_robot_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        empty_outcome = TransitionOutcomeBatch(
            episode_id="ep-async", decision_step=0, rewards=(),
            terminated=True, truncated=False, termination_reason=TerminationReason.MAX_STEPS,
        )
        with pytest.raises(ValueError):
            session.complete_robot_decision(robot_id=0, outcome=empty_outcome)

    def test_reward_for_extra_robot_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        extra_outcome = TransitionOutcomeBatch(
            episode_id="ep-async", decision_step=0,
            rewards=(
                RobotRewardOutcome(robot_id=0, components=(make_reward_component(),)),
                RobotRewardOutcome(robot_id=9, components=(make_reward_component(),)),
            ),
            terminated=True, truncated=False, termination_reason=TerminationReason.MAX_STEPS,
        )
        with pytest.raises(ValueError):
            session.complete_robot_decision(robot_id=0, outcome=extra_outcome)

    def test_non_terminal_with_valid_next(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        next0 = make_opened_robot(geometry, robot_id=0, decision_step=3, time_s=1.0)
        transition = session.complete_robot_decision(
            robot_id=0,
            outcome=make_outcome("ep-async", 0, 0),
            next_decision=next0,
            next_critic_state=make_critic_state(3),
        )
        assert transition.terminated is False
        assert session.has_pending(0) is True
        assert session.pending_robot_ids == (0,)

    def test_non_terminal_without_next_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0, outcome=make_outcome("ep-async", 0, 0), next_decision=None,
            )

    def test_non_terminal_without_next_critic_state_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        next0 = make_opened_robot(geometry, robot_id=0, decision_step=3, time_s=1.0)
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0, outcome=make_outcome("ep-async", 0, 0), next_decision=next0,
            )
        assert session.has_pending(0) is True

    def test_next_from_another_robot_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])
        wrong_next = make_opened_robot(geometry, robot_id=1, decision_step=5, time_s=1.0)
        with pytest.raises(PendingRobotDecisionError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0),
                next_decision=wrong_next,
                next_critic_state=make_critic_state(5),
            )

    def test_next_with_step_equal_to_current_fails(self):
        # The current step is always already in _seen_decision_steps (added
        # at register time), so reusing it is caught by the session's own
        # step-uniqueness guard, not delegated to TransitionAssemblyInput.
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=3, time_s=0.0)
        register_items(session, [r0])
        same_step_next = make_opened_robot(geometry, robot_id=0, decision_step=3, time_s=1.0)
        with pytest.raises(PendingRobotDecisionError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 3, 0),
                next_decision=same_step_next,
                next_critic_state=make_critic_state(3),
            )

    def test_next_with_lower_unused_step_fails(self):
        # A lower step that was never seen this episode is rejected by
        # TransitionAssemblyInput's own ordering rule, not the session's
        # step-uniqueness guard.
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=5, time_s=0.0)
        register_items(session, [r0])
        lower_next = make_opened_robot(geometry, robot_id=0, decision_step=2, time_s=1.0)
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 5, 0),
                next_decision=lower_next,
                next_critic_state=make_critic_state(2),
            )

    def test_next_with_lower_time_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=5.0)
        register_items(session, [r0], time_s=5.0)
        earlier_next = make_opened_robot(geometry, robot_id=0, decision_step=1, time_s=1.0)
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0),
                next_decision=earlier_next,
                next_critic_state=make_critic_state(1),
            )

    def test_terminal_with_no_next_valid(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        transition = session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        assert transition.terminated is True
        assert session.has_pending(0) is False

    def test_terminal_with_next_decision_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        next0 = make_opened_robot(geometry, robot_id=0, decision_step=1, time_s=1.0)
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0, terminated=True),
                next_decision=next0,
            )

    def test_terminal_with_next_critic_state_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0, terminated=True),
                next_critic_state=make_critic_state(1),
            )

    def test_terminal_with_next_ground_truth_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(ValueError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0, terminated=True),
                next_ground_truth=make_ground_truth(1),
            )

    def test_truncated_with_no_next_valid(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        transition = session.complete_robot_decision(
            robot_id=0,
            outcome=make_outcome(
                "ep-async", 0, 0, truncated=True, reason=TerminationReason.MAX_STEPS
            ),
        )
        assert transition.truncated is True
        assert session.has_pending(0) is False


class TestGroundTruthNextVsCurrent:
    def test_next_ground_truth_does_not_appear_in_current_transition_step(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        gt0 = make_ground_truth(0, fire_x=1.0)
        register_items(session, [r0], ground_truth_by_robot={0: gt0})

        next0 = make_opened_robot(geometry, robot_id=0, decision_step=1, time_s=1.0)
        gt_next = make_ground_truth(1, fire_x=42.0)
        session.complete_robot_decision(
            robot_id=0,
            outcome=make_outcome("ep-async", 0, 0),
            next_decision=next0,
            next_critic_state=make_critic_state(1),
            next_ground_truth=gt_next,
        )
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 1, 0, terminated=True)
        )
        record = session.finish_episode()
        assert record.ground_truth_by_step == ((0, gt0), (1, gt_next))

    def test_next_ground_truth_appears_when_that_step_completes(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0], ground_truth_by_robot={0: make_ground_truth(0, fire_x=1.0)})

        next0 = make_opened_robot(geometry, robot_id=0, decision_step=1, time_s=1.0)
        gt_next = make_ground_truth(1, fire_x=42.0)
        session.complete_robot_decision(
            robot_id=0,
            outcome=make_outcome("ep-async", 0, 0),
            next_decision=next0,
            next_critic_state=make_critic_state(1),
            next_ground_truth=gt_next,
        )
        transition = session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 1, 0, terminated=True)
        )
        record = session.finish_episode()
        step_1_entry = dict(record.ground_truth_by_step)[1]
        assert step_1_entry == gt_next
        assert transition.decision_step == 1


class TestDecisionStepUniqueness:
    def test_repeated_step_across_two_register_calls_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=5, time_s=0.0)
        register_items(session, [r0])
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=5, time_s=0.0)
        with pytest.raises(PendingRobotDecisionError):
            register_items(session, [r1])
        assert session.has_pending(1) is False

    def test_next_step_matches_another_pending_robot_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])

        next0 = make_opened_robot(geometry, robot_id=0, decision_step=1, time_s=1.0)
        with pytest.raises(PendingRobotDecisionError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0),
                next_decision=next0,
                next_critic_state=make_critic_state(1),
            )

    def test_next_step_matches_a_completed_step_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        # step 0 is completed and no longer pending, but still "seen".
        next1 = make_opened_robot(geometry, robot_id=1, decision_step=0, time_s=5.0)
        with pytest.raises(PendingRobotDecisionError):
            session.complete_robot_decision(
                robot_id=1,
                outcome=make_outcome("ep-async", 1, 1),
                next_decision=next1,
                next_critic_state=make_critic_state(0),
            )

    def test_duplicate_step_failure_does_not_modify_pending(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])

        next0 = make_opened_robot(geometry, robot_id=0, decision_step=1, time_s=1.0)
        with pytest.raises(PendingRobotDecisionError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0),
                next_decision=next0,
                next_critic_state=make_critic_state(1),
            )
        assert session.pending_robot_ids == (0, 1)
        assert session.pending_count == 2

    def test_duplicate_step_failure_does_not_append_transition(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])

        next0 = make_opened_robot(geometry, robot_id=0, decision_step=1, time_s=1.0)
        with pytest.raises(PendingRobotDecisionError):
            session.complete_robot_decision(
                robot_id=0,
                outcome=make_outcome("ep-async", 0, 0),
                next_decision=next0,
                next_critic_state=make_critic_state(1),
            )
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        session.complete_robot_decision(
            robot_id=1, outcome=make_outcome("ep-async", 1, 1, terminated=True)
        )
        record = session.finish_episode()
        assert len(record.transitions) == 2  # not 3 -- the failed attempt never appended


class TestAsynchronousOrdering:
    def test_complete_robots_in_reverse_order(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])

        t1 = session.complete_robot_decision(
            robot_id=1, outcome=make_outcome("ep-async", 1, 1, terminated=True)
        )
        assert session.pending_robot_ids == (0,)
        t0 = session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        assert session.pending_count == 0
        assert t0.decision_step == 0
        assert t1.decision_step == 1

        record = session.finish_episode()
        assert [t.decision_step for t in record.transitions] == [0, 1]

    def test_complete_higher_step_before_lower_step(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=5, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=2, time_s=0.0)
        register_items(session, [r0, r1])

        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 5, 0, terminated=True)
        )
        session.complete_robot_decision(
            robot_id=1, outcome=make_outcome("ep-async", 2, 1, terminated=True)
        )
        record = session.finish_episode()
        assert [t.decision_step for t in record.transitions] == [2, 5]

    def test_recorder_retains_both_transitions(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])
        session.complete_robot_decision(
            robot_id=1, outcome=make_outcome("ep-async", 1, 1, terminated=True)
        )
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        record = session.finish_episode()
        assert len(record.transitions) == 2


class TestFinishEpisode:
    def test_finish_with_pending_fails(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        with pytest.raises(AsynchronousEpisodeSessionStateError):
            session.finish_episode()

    def test_finish_without_episode_fails(self):
        session = make_session()
        with pytest.raises(AsynchronousEpisodeSessionStateError):
            session.finish_episode()

    def test_finish_clears_session_for_next_episode(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        session.finish_episode()
        assert session.is_active is False
        assert session.episode_id is None
        session.start_episode(make_metadata(episode_id="ep-second"))
        assert session.is_active is True

    def test_second_episode_can_reuse_the_same_steps(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata(episode_id="ep-first"))
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0, episode_id="ep-first")
        register_items(session, [r0], episode_id="ep-first")
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-first", 0, 0, terminated=True)
        )
        session.finish_episode()

        session.start_episode(make_metadata(episode_id="ep-second"))
        r0_again = make_opened_robot(
            geometry, robot_id=0, decision_step=0, time_s=0.0, episode_id="ep-second"
        )
        register_items(session, [r0_again], episode_id="ep-second")  # must not raise
        assert session.pending_count == 1


class TestAbort:
    def test_abort_clears_pending(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0)
        register_items(session, [r0, r1])
        session.abort_episode()
        assert session.is_active is False
        assert session.pending_count == 0

    def test_abort_allows_second_episode(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata(episode_id="ep-aborted"))
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0, episode_id="ep-aborted")
        register_items(session, [r0], episode_id="ep-aborted")
        session.abort_episode()

        session.start_episode(make_metadata(episode_id="ep-fresh"))
        assert session.is_active is True
        assert session.pending_count == 0

    def test_abort_without_episode_fails(self):
        session = make_session()
        with pytest.raises(AsynchronousEpisodeSessionStateError):
            session.abort_episode()

    def test_abort_clears_seen_steps(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata(episode_id="ep-aborted"))
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0, episode_id="ep-aborted")
        register_items(session, [r0], episode_id="ep-aborted")
        session.abort_episode()

        session.start_episode(make_metadata(episode_id="ep-fresh"))
        r0_again = make_opened_robot(
            geometry, robot_id=0, decision_step=0, time_s=0.0, episode_id="ep-fresh"
        )
        register_items(session, [r0_again], episode_id="ep-fresh")  # must not raise
        assert session.pending_count == 1


class TestFireMetrics:
    def test_fire_metrics_recorded(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0])
        metrics = EpisodeFireMetrics(fire_crossing_time_s=1.5, fire_overflight_distance=3.0)
        session.set_fire_metrics(metrics)
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        record = session.finish_episode()
        assert record.fire_metrics == metrics

    def test_fire_metrics_without_episode_fails(self):
        session = make_session()
        with pytest.raises(AsynchronousEpisodeSessionStateError):
            session.set_fire_metrics(EpisodeFireMetrics())


class TestGroundTruthSeparateFromTransition:
    def test_ground_truth_never_a_transition_field(self):
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        gt = make_ground_truth(0)
        register_items(session, [r0], ground_truth_by_robot={0: gt})
        transition = session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        import dataclasses

        for field in dataclasses.fields(transition):
            value = getattr(transition, field.name)
            assert not isinstance(value, GroundTruthSnapshot)

        record = session.finish_episode()
        assert record.ground_truth_by_step == ((0, gt),)


class TestNoFilesystemWrites:
    def test_asynchronous_episode_module_touches_no_filesystem_api(self):
        tree = ast.parse((LEARNING_DIR / "asynchronous_episode.py").read_text(encoding="utf-8"))
        forbidden_modules = {"os", "pathlib", "shutil", "tempfile", "io"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_modules
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_modules
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "open"

    def test_full_flow_creates_no_files(self, tmp_path, monkeypatch):
        import os

        monkeypatch.chdir(tmp_path)
        geometry = make_geometry()
        session = make_session()
        session.start_episode(make_metadata())
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0)
        register_items(session, [r0], ground_truth_by_robot={0: make_ground_truth(0)})
        session.complete_robot_decision(
            robot_id=0, outcome=make_outcome("ep-async", 0, 0, terminated=True)
        )
        session.finish_episode()
        assert os.listdir(tmp_path) == []


class TestSmokeAsynchronousEpisode:
    def test_full_asynchronous_episode_out_of_order_completion(self):
        geometry = make_geometry()
        episode_id = "ep-smoke-async"
        session = make_session()
        session.start_episode(make_metadata(episode_id=episode_id))

        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0, episode_id=episode_id)
        r1 = make_opened_robot(geometry, robot_id=1, decision_step=1, time_s=0.0, episode_id=episode_id)
        gt0 = make_ground_truth(0)
        gt1 = make_ground_truth(1)
        register_items(
            session, [r0, r1], episode_id=episode_id, ground_truth_by_robot={0: gt0, 1: gt1}
        )

        metrics = EpisodeFireMetrics(fire_crossing_time_s=2.0, fire_overflight_distance=1.0)
        session.set_fire_metrics(metrics)

        # Robot 1 finishes first, non-terminal, opens a new decision at step 2.
        gt2 = make_ground_truth(2)
        next1 = make_opened_robot(
            geometry, robot_id=1, decision_step=2, time_s=1.0, episode_id=episode_id
        )
        t1 = session.complete_robot_decision(
            robot_id=1,
            outcome=make_outcome(episode_id, 1, 1),
            next_decision=next1,
            next_critic_state=make_critic_state(2),
            next_ground_truth=gt2,
        )
        assert t1.terminated is False
        assert session.pending_robot_ids == (0, 1)

        # Robot 0 finishes after, terminal.
        t0 = session.complete_robot_decision(
            robot_id=0, outcome=make_outcome(episode_id, 0, 0, terminated=True)
        )
        assert t0.terminated is True
        assert session.pending_robot_ids == (1,)

        # Robot 1 finishes its second decision, terminal.
        t2 = session.complete_robot_decision(
            robot_id=1, outcome=make_outcome(episode_id, 2, 1, terminated=True)
        )
        assert t2.terminated is True
        assert session.pending_count == 0

        record = session.finish_episode()

        assert [t.decision_step for t in record.transitions] == [0, 1, 2]
        assert set(record.transitions[0].actor_observations) == {0}
        assert set(record.transitions[1].actor_observations) == {1}
        assert set(record.transitions[2].actor_observations) == {1}
        assert record.transitions[0].terminated is True
        assert record.transitions[1].terminated is False
        assert record.transitions[2].terminated is True
        assert record.ground_truth_by_step == ((0, gt0), (1, gt1), (2, gt2))
        assert record.fire_metrics == metrics
        assert session.is_active is False

        for transition in record.transitions:
            assert not isinstance(transition.critic_state, GroundTruthSnapshot)
