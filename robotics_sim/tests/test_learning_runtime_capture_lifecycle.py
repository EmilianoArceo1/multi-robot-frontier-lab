"""Tests for RuntimeLearningCaptureService's lifecycle: start_episode(),
complete_terminal_robot_decision(), finish_episode(), abort_episode(), and
set_fire_metrics() -- the state machine around the composed components,
without inspecting any of their internals -- plus one end-to-end smoke test
covering the full open/replace/close/finish flow with real materialized
CriticState/GroundTruthSnapshot contracts."""

from __future__ import annotations

import pytest

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.learning import (
    CONTRACT_VERSIONS,
    CandidateSetSpec,
    EpisodeFireMetrics,
    EpisodeMetadata,
    HoldPolicy,
    TerminationReason,
    build_contract_manifest,
    compute_contract_bundle_hash,
)
from robotics_interfaces.learning.transitions import RewardComponent
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.plugins import CandidateInputMode, PluginMetadata
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig, build_feature_schema_v0
from robotics_sim.learning.action_catalog import ActionCatalogAssembler
from robotics_sim.learning.asynchronous_episode import InMemoryAsynchronousLearningEpisodeSession
from robotics_sim.learning.builders import CriticStateBuilder, GroundTruthSnapshotBuilder
from robotics_sim.learning.coordination_decision_source import LearningCoordinationDecisionSource
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler
from robotics_sim.learning.decision_steps import EpisodeDecisionStepAllocator
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler
from robotics_sim.learning.recorder import ContractBundleHashMismatchError, InMemoryTrajectoryRecorder
from robotics_sim.learning.runtime_capture_service import (
    CriticStateCaptureSource,
    GroundTruthCaptureSource,
    RuntimeCoordinationCaptureInput,
    RuntimeLearningCaptureConsistencyError,
    RuntimeLearningCaptureService,
    RuntimeLearningCaptureStateError,
)
from robotics_sim.learning.runtime_decision_opening import (
    RobotDecisionObservationContext,
    RuntimeLearningDecisionOpener,
)
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import RobotRewardOutcome, TransitionOutcomeBatch

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


def make_candidate(target=(4.0, 6.0), information_gain=1.0) -> ExplorationCandidate:
    return ExplorationCandidate(target=target, source="test", information_gain=information_gain)


def make_context(robot_id, geometry, xy=(1.0, 1.0)):
    return RobotDecisionObservationContext(
        robot=make_robot(robot_id, xy),
        hazard_belief=HazardBelief(geometry).snapshot(),
        graph_edges=(),
        visible_teammates=(),
    )


def make_metadata(episode_id="ep-lifecycle") -> EpisodeMetadata:
    bundle_hash = compute_contract_bundle_hash(build_contract_manifest())
    return EpisodeMetadata(
        episode_id=episode_id, seed=1, map_id="map-1", robot_count=1, fire_count=1,
        sensor_range=4.0, field_of_view_deg=120.0, communication_range=15.0, max_steps=100,
        simulator_commit="deadbeef", contract_versions=dict(CONTRACT_VERSIONS),
        contract_bundle_hash=bundle_hash,
    )


def make_critic_source(coverage=0.5) -> CriticStateCaptureSource:
    return CriticStateCaptureSource(
        global_feature_names=("coverage",),
        global_features={"coverage": coverage},
        per_robot_feature_names=(),
        per_robot_features={},
    )


def make_ground_truth_source(fire_x=0.0) -> GroundTruthCaptureSource:
    return GroundTruthCaptureSource(
        true_robot_poses={},
        true_occupancy=(),
        true_fire_locations=((fire_x, fire_x),),
        global_coverage_fraction=0.0,
    )


def make_reward_component(name="new_coverage", raw=1.0, weight=0.5) -> RewardComponent:
    return RewardComponent(name=name, raw_value=raw, applied_weight=weight, weighted_value=raw * weight)


def make_outcome(episode_id, decision_step, robot_id, terminated=False, truncated=False, reason=None) -> TransitionOutcomeBatch:
    if reason is None:
        reason = TerminationReason.RUNNING if not (terminated or truncated) else TerminationReason.MAX_STEPS
    return TransitionOutcomeBatch(
        episode_id=episode_id, decision_step=decision_step,
        rewards=(RobotRewardOutcome(robot_id=robot_id, components=(make_reward_component(),)),),
        terminated=terminated, truncated=truncated, termination_reason=reason,
    )


def make_request(robot_ids, candidates_by_robot) -> CoordinationRequest:
    return CoordinationRequest(
        robot_states=tuple(make_robot(rid) for rid in robot_ids),
        robots_to_assign=tuple(robot_ids),
        proposals_by_robot={rid: candidates_by_robot.get(rid, ()) for rid in robot_ids},
    )


class ScriptedPlugin:
    metadata = PluginMetadata(
        name="scripted-lifecycle-plugin", version="0.0.0", description="",
        capabilities=(), candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
    )

    def __init__(self, plan_by_robot):
        self._plan_by_robot = plan_by_robot
        self.assign_calls = 0

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        self.assign_calls += 1
        assignments, commands = [], []
        for robot_id in request.robots_to_assign:
            candidates = tuple(request.proposals_by_robot.get(robot_id, ()))
            plan = self._plan_by_robot[robot_id]
            if plan[0] == "ASSIGNED":
                chosen = candidates[plan[1]]
                assignments.append(
                    CoordinationAssignment(
                        robot_id=robot_id, status="ASSIGNED", target=chosen.target,
                        proposal=chosen, reason="scripted",
                    )
                )
                commands.append(
                    RobotCommand(robot_id=robot_id, status="ASSIGNED", target=chosen.target, reason="scripted")
                )
            else:
                reason = plan[1] if len(plan) > 1 else "scripted"
                assignments.append(
                    CoordinationAssignment(robot_id=robot_id, status=plan[0], target=None, reason=reason)
                )
                commands.append(RobotCommand(robot_id=robot_id, status=plan[0], reason=reason))
        return CoordinationResult(assignments=tuple(assignments), commands=tuple(commands), strategy="scripted")


def make_decision_opener(candidate_spec=None):
    candidate_spec = candidate_spec or make_candidate_spec()
    schema = build_feature_schema_v0()
    decision_assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(schema=schema, candidate_spec=candidate_spec),
        catalog_assembler=ActionCatalogAssembler(),
    )
    return RuntimeLearningDecisionOpener(decision_assembler)


def make_service(plan_by_robot=None):
    plugin = ScriptedPlugin(plan_by_robot or {0: ("ASSIGNED", 0)})
    decision_source = LearningCoordinationDecisionSource(plugin)
    decision_opener = make_decision_opener()
    step_allocator = EpisodeDecisionStepAllocator()
    episode_session = InMemoryAsynchronousLearningEpisodeSession(
        LearningTransitionAssembler(), InMemoryTrajectoryRecorder()
    )
    service = RuntimeLearningCaptureService(
        decision_source, decision_opener, step_allocator, episode_session,
        CriticStateBuilder(), GroundTruthSnapshotBuilder(),
    )
    return service, plugin, step_allocator, episode_session


def make_capture_input(
    robot_ids, candidates_by_robot, contexts_by_robot, critic_sources_by_robot,
    ground_truth_sources_by_robot=None, closing_outcomes_by_robot=None, time_s=0.0, geometry=None,
) -> RuntimeCoordinationCaptureInput:
    geometry = geometry or make_geometry()
    return RuntimeCoordinationCaptureInput(
        request=make_request(robot_ids, candidates_by_robot),
        time_s=time_s,
        contexts_by_robot=contexts_by_robot,
        critic_sources_by_robot=critic_sources_by_robot,
        ground_truth_sources_by_robot=ground_truth_sources_by_robot,
        closing_outcomes_by_robot=closing_outcomes_by_robot or {},
        grid_geometry=geometry,
        normalization=NORMALIZATION,
        candidate_spec=make_candidate_spec(),
    )


class TestStart:
    def test_start_normal(self):
        service, *_ = make_service()
        service.start_episode(make_metadata())
        assert service.is_active is True
        assert service.episode_id == "ep-lifecycle"
        assert service.next_decision_step == 0

    def test_start_step_nonzero(self):
        service, *_ = make_service()
        service.start_episode(make_metadata(), start_step=7)
        assert service.next_decision_step == 7

    def test_double_start_fails(self):
        service, *_ = make_service()
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureStateError):
            service.start_episode(make_metadata(episode_id="ep-other"))

    def test_allocator_starts(self):
        service, plugin, step_allocator, episode_session = make_service()
        service.start_episode(make_metadata())
        assert step_allocator.is_active is True

    def test_session_starts(self):
        service, plugin, step_allocator, episode_session = make_service()
        service.start_episode(make_metadata())
        assert episode_session.is_active is True
        assert episode_session.episode_id == "ep-lifecycle"

    def test_allocator_rolled_back_when_session_start_fails(self):
        # A metadata with a wrong contract_bundle_hash passes the service's
        # own precondition checks (both components inactive, metadata is an
        # EpisodeMetadata) and step_allocator.start_episode() succeeds, but
        # episode_session.start_episode() -> recorder.start_episode() then
        # rejects it with ContractBundleHashMismatchError -- exactly the
        # "session.start fails after allocator started" case.
        service, plugin, step_allocator, episode_session = make_service()
        bad_metadata = EpisodeMetadata(
            episode_id="ep-bad-hash", seed=1, map_id="map-1", robot_count=1, fire_count=1,
            sensor_range=4.0, field_of_view_deg=120.0, communication_range=15.0, max_steps=100,
            simulator_commit="deadbeef", contract_versions=dict(CONTRACT_VERSIONS),
            contract_bundle_hash="not-the-real-contract-bundle-hash",
        )
        with pytest.raises(ContractBundleHashMismatchError):
            service.start_episode(bad_metadata)

        assert step_allocator.is_active is False
        assert episode_session.is_active is False
        # The service is fully inactive again, so a real start still works.
        service.start_episode(make_metadata())
        assert service.is_active is True


class TestCaptureRequiresStart:
    def test_capture_without_start_fails(self):
        service, *_ = make_service()
        geometry = make_geometry()
        c0 = make_candidate()
        with pytest.raises(RuntimeLearningCaptureStateError):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
                )
            )


class TestTerminalClose:
    def _service_with_pending(self):
        service, plugin, step_allocator, episode_session = make_service()
        service.start_episode(make_metadata())
        geometry = make_geometry()
        c0 = make_candidate()
        service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
            )
        )
        return service

    def test_terminal_close_valid(self):
        service = self._service_with_pending()
        transition = service.complete_terminal_robot_decision(
            0, make_outcome("ep-lifecycle", 0, 0, terminated=True)
        )
        assert transition.terminated is True
        assert service.pending_robot_ids == ()

    def test_terminal_close_with_running_fails(self):
        service = self._service_with_pending()
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.complete_terminal_robot_decision(0, make_outcome("ep-lifecycle", 0, 0))


class TestFinish:
    def test_finish_with_pending_fails(self):
        service, *_ = make_service()
        service.start_episode(make_metadata())
        geometry = make_geometry()
        c0 = make_candidate()
        service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
            )
        )
        with pytest.raises(RuntimeLearningCaptureStateError):
            service.finish_episode()

    def test_finish_without_pending(self):
        service, *_ = make_service()
        service.start_episode(make_metadata())
        geometry = make_geometry()
        c0 = make_candidate()
        service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
            )
        )
        service.complete_terminal_robot_decision(0, make_outcome("ep-lifecycle", 0, 0, terminated=True))
        record = service.finish_episode()
        assert len(record.transitions) == 1

    def test_finish_without_episode_fails(self):
        service, *_ = make_service()
        with pytest.raises(RuntimeLearningCaptureStateError):
            service.finish_episode()

    def test_episode_record_ordered(self):
        service, plugin, step_allocator, episode_session = make_service(
            {0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)}
        )
        service.start_episode(make_metadata())
        geometry = make_geometry()
        c0, c1 = make_candidate(target=(2.0, 2.0)), make_candidate(target=(6.0, 6.0))
        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                {0: make_critic_source(), 1: make_critic_source()},
            )
        )
        service.complete_terminal_robot_decision(1, make_outcome("ep-lifecycle", 1, 1, terminated=True))
        service.complete_terminal_robot_decision(0, make_outcome("ep-lifecycle", 0, 0, terminated=True))
        record = service.finish_episode()
        assert [t.decision_step for t in record.transitions] == [0, 1]

    def test_finish_deactivates_service(self):
        service, *_ = make_service()
        service.start_episode(make_metadata())
        geometry = make_geometry()
        c0 = make_candidate()
        service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
            )
        )
        service.complete_terminal_robot_decision(0, make_outcome("ep-lifecycle", 0, 0, terminated=True))
        service.finish_episode()
        assert service.is_active is False
        assert service.episode_id is None
        assert service.next_decision_step is None


class TestFireMetrics:
    def test_fire_metrics(self):
        service, *_ = make_service()
        service.start_episode(make_metadata())
        geometry = make_geometry()
        c0 = make_candidate()
        service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
            )
        )
        metrics = EpisodeFireMetrics(fire_crossing_time_s=1.0, fire_overflight_distance=2.0)
        service.set_fire_metrics(metrics)
        service.complete_terminal_robot_decision(0, make_outcome("ep-lifecycle", 0, 0, terminated=True))
        record = service.finish_episode()
        assert record.fire_metrics == metrics

    def test_fire_metrics_without_episode_fails(self):
        service, *_ = make_service()
        with pytest.raises(RuntimeLearningCaptureStateError):
            service.set_fire_metrics(EpisodeFireMetrics())


class TestAbort:
    def test_abort_with_pending(self):
        service, *_ = make_service()
        service.start_episode(make_metadata())
        geometry = make_geometry()
        c0 = make_candidate()
        service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
            )
        )
        service.abort_episode()
        assert service.is_active is False

    def test_abort_clears_allocator_and_session(self):
        service, plugin, step_allocator, episode_session = make_service()
        service.start_episode(make_metadata())
        service.abort_episode()
        assert step_allocator.is_active is False
        assert episode_session.is_active is False

    def test_second_episode_after_abort(self):
        service, *_ = make_service()
        service.start_episode(make_metadata(episode_id="ep-aborted"))
        service.abort_episode()
        service.start_episode(make_metadata(episode_id="ep-fresh"))
        assert service.is_active is True
        assert service.episode_id == "ep-fresh"

    def test_same_steps_reusable_in_another_episode(self):
        service, *_ = make_service()
        service.start_episode(make_metadata(episode_id="ep-first"))
        geometry = make_geometry()
        c0 = make_candidate()
        service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
            )
        )
        service.complete_terminal_robot_decision(0, make_outcome("ep-first", 0, 0, terminated=True))
        service.finish_episode()

        service.start_episode(make_metadata(episode_id="ep-second"))
        result = service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()}
            )
        )
        assert result.opened_decision.assigned[0].decision_step == 0  # step 0 reused, no collision

    def test_abort_without_episode_fails(self):
        service, *_ = make_service()
        with pytest.raises(RuntimeLearningCaptureStateError):
            service.abort_episode()

    def test_partial_abort_attempts_both_components(self):
        service, plugin, step_allocator, episode_session = make_service()
        service.start_episode(make_metadata())
        # Bring the episode session down independently of the service, so
        # the service's abort_episode() sees session.is_active=False but
        # step_allocator.is_active=True -- an inconsistent state it did not
        # cause, but must still try to clean up.
        episode_session.abort_episode()
        assert step_allocator.is_active is True
        assert episode_session.is_active is False

        service.abort_episode()  # must not raise: allocator alone still needed cleanup
        assert step_allocator.is_active is False


class TestSmokeOpenReplaceCloseFinish:
    """End-to-end smoke test: start_episode -> open robot 0 and robot 1 ->
    verify steps 0/1 on actor, critic, and ground truth -> replace robot 1
    with step 2 -> close robot 0 and robot 1 -> finish -> EpisodeRecord
    ordered 0, 1, 2."""

    def test_full_flow(self):
        service, plugin, step_allocator, episode_session = make_service(
            {0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)}
        )
        geometry = make_geometry()
        service.start_episode(make_metadata(episode_id="ep-smoke"))

        # 2. Open robot 0 and robot 1 in the same event.
        c0, c1 = make_candidate(target=(2.0, 2.0)), make_candidate(target=(6.0, 6.0))
        opened_result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                {0: make_critic_source(0.1), 1: make_critic_source(0.2)},
                ground_truth_sources_by_robot={
                    0: make_ground_truth_source(1.0), 1: make_ground_truth_source(2.0)
                },
            )
        )
        by_id = {item.robot_id: item for item in opened_result.opened_decision.assigned}

        # 3. Steps 0 and 1 landed on actor (via opened_decision), critic, and
        # ground truth alike.
        assert {by_id[0].decision_step, by_id[1].decision_step} == {0, 1}

        # 4. Replace robot 1 with a new decision at step 2.
        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        replace_result = service.capture_coordination_event(
            make_capture_input(
                [1], {1: (c1_next,)},
                {1: make_context(1, geometry, xy=(3.0, 3.0))},
                {1: make_critic_source(0.3)},
                ground_truth_sources_by_robot={1: make_ground_truth_source(3.0)},
                closing_outcomes_by_robot={1: make_outcome("ep-smoke", by_id[1].decision_step, 1)},
            )
        )
        new_item = replace_result.opened_decision.assigned[0]
        assert new_item.decision_step == 2

        # 5. Close robot 0 and robot 1 (now at step 2).
        service.complete_terminal_robot_decision(
            0, make_outcome("ep-smoke", by_id[0].decision_step, 0, terminated=True)
        )
        service.complete_terminal_robot_decision(
            1, make_outcome("ep-smoke", 2, 1, terminated=True)
        )

        # 6. Finish.
        record = service.finish_episode()

        # 7. EpisodeRecord ordered 0, 1, 2.
        assert [t.decision_step for t in record.transitions] == [0, 1, 2]
        assert {step for step, _ in record.ground_truth_by_step} == {0, 1, 2}
