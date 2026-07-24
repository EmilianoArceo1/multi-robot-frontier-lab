"""Tests for RuntimeLearningCaptureService.capture_coordination_event(): the
plugin-call, candidate-pool, step-allocation, source-materialization, opening,
and register/replace classification flow for one coordination event."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

import robotics_sim.learning as learning_pkg
from algorithms.independent_baseline.plugin import create_plugin as create_independent_baseline
from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.learning import (
    CONTRACT_VERSIONS,
    CandidateSetSpec,
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
from robotics_interfaces.services import CoordinationServices
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
from robotics_sim.learning.recorder import InMemoryTrajectoryRecorder
from robotics_sim.learning.runtime_capture_service import (
    CriticStateCaptureSource,
    GroundTruthCaptureSource,
    RuntimeCoordinationCaptureInput,
    RuntimeLearningCaptureConsistencyError,
    RuntimeLearningCaptureService,
    RuntimeLearningCaptureStateError,
)
from robotics_sim.learning.runtime_decision_opening import RobotDecisionObservationContext, RuntimeLearningDecisionOpener
from robotics_sim.learning.source_models import CriticStateBuildInput, GroundTruthBuildInput
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import RobotRewardOutcome, TransitionOutcomeBatch

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


def make_candidate(target=(4.0, 6.0), information_gain=1.0, heading_rad=None) -> ExplorationCandidate:
    return ExplorationCandidate(
        target=target, source="test", information_gain=information_gain, heading_rad=heading_rad
    )


def make_context(robot_id, geometry, xy=(1.0, 1.0), graph_edges=(), visible_teammates=(), belief=None):
    return RobotDecisionObservationContext(
        robot=make_robot(robot_id, xy),
        hazard_belief=belief if belief is not None else HazardBelief(geometry).snapshot(),
        graph_edges=graph_edges,
        visible_teammates=visible_teammates,
    )


def make_metadata(episode_id="ep-capture") -> EpisodeMetadata:
    bundle_hash = compute_contract_bundle_hash(build_contract_manifest())
    return EpisodeMetadata(
        episode_id=episode_id, seed=1, map_id="map-1", robot_count=2, fire_count=1,
        sensor_range=4.0, field_of_view_deg=120.0, communication_range=15.0, max_steps=100,
        simulator_commit="deadbeef", contract_versions=dict(CONTRACT_VERSIONS),
        contract_bundle_hash=bundle_hash,
    )


def make_critic_source(coverage=0.5, per_robot_features=None) -> CriticStateCaptureSource:
    """Step-agnostic critic source -- no decision_step, no time_s.

    ``coverage`` only distinguishes content between sources in a test; the
    real decision_step is assigned later by the service, from
    EpisodeDecisionStepAllocator.
    """

    return CriticStateCaptureSource(
        global_feature_names=("coverage",),
        global_features={"coverage": coverage},
        per_robot_feature_names=(),
        per_robot_features=per_robot_features or {},
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


def make_request(robot_ids, candidates_by_robot=None, services=None) -> CoordinationRequest:
    candidates_by_robot = candidates_by_robot or {}
    return CoordinationRequest(
        robot_states=tuple(make_robot(rid) for rid in robot_ids),
        robots_to_assign=tuple(robot_ids),
        proposals_by_robot={rid: candidates_by_robot.get(rid, ()) for rid in robot_ids},
        services=services,
    )


class ScriptedPlugin:
    """Deterministic plugin driven entirely by a caller-supplied plan, so
    tests can pick exactly which robots are ASSIGNED/HOLD/FAILED (and which
    candidate index wins) without depending on a real ranking rule."""

    metadata = PluginMetadata(
        name="scripted-test-plugin", version="0.0.0", description="",
        capabilities=(), candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
    )

    def __init__(self, plan_by_robot):
        # plan_by_robot: {robot_id: ("ASSIGNED", index) | ("HOLD", reason) | ("FAILED", reason)}
        self._plan_by_robot = plan_by_robot
        self.assign_calls = 0

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        self.assign_calls += 1
        assignments = []
        commands = []
        for robot_id in request.robots_to_assign:
            candidates = tuple(request.proposals_by_robot.get(robot_id, ()))
            plan = self._plan_by_robot[robot_id]
            kind = plan[0]
            if kind == "ASSIGNED":
                chosen = candidates[plan[1]]
                assignments.append(
                    CoordinationAssignment(
                        robot_id=robot_id, status="ASSIGNED", target=chosen.target,
                        proposal=chosen, reason="scripted",
                    )
                )
                commands.append(
                    RobotCommand(
                        robot_id=robot_id, status="ASSIGNED", target=chosen.target,
                        heading_rad=chosen.heading_rad, reason="scripted",
                    )
                )
            else:
                reason = plan[1] if len(plan) > 1 else "scripted"
                assignments.append(
                    CoordinationAssignment(robot_id=robot_id, status=kind, target=None, reason=reason)
                )
                commands.append(RobotCommand(robot_id=robot_id, status=kind, reason=reason))
        return CoordinationResult(assignments=tuple(assignments), commands=tuple(commands), strategy="scripted")


class CountingDecisionOpener(RuntimeLearningDecisionOpener):
    def __init__(self, decision_assembler):
        super().__init__(decision_assembler)
        self.open_calls = 0

    def open(self, opening_input):
        self.open_calls += 1
        return super().open(opening_input)


class CountingCriticStateBuilder(CriticStateBuilder):
    def __init__(self):
        super().__init__()
        self.build_calls = 0

    def build(self, build_input):
        self.build_calls += 1
        return super().build(build_input)


class CountingGroundTruthBuilder(GroundTruthSnapshotBuilder):
    def __init__(self):
        super().__init__()
        self.build_calls = 0

    def build(self, build_input):
        self.build_calls += 1
        return super().build(build_input)


class FailingCriticStateBuilder(CriticStateBuilder):
    def build(self, build_input):
        raise RuntimeError("simulated critic builder failure")


class FailingGroundTruthBuilder(GroundTruthSnapshotBuilder):
    def build(self, build_input):
        raise RuntimeError("simulated ground truth builder failure")


class FailingDecisionOpener(RuntimeLearningDecisionOpener):
    def open(self, opening_input):
        raise RuntimeError("simulated decision opener failure")


class FailingRegisterEpisodeSession(InMemoryAsynchronousLearningEpisodeSession):
    def register_opened_decisions(self, opened, critic_states_by_robot, ground_truth_by_robot=None):
        raise RuntimeError("simulated register_opened_decisions failure")


class RaisingPlugin:
    """A plugin whose assign() fails outright -- used to distinguish a
    pre-allocation failure (before EpisodeDecisionStepAllocator.
    allocate_many() ever runs) from a post-allocation one."""

    metadata = PluginMetadata(
        name="raising-plugin", version="0.0.0", description="",
        capabilities=(), candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        raise RuntimeError("simulated pre-allocation plugin failure")


def make_decision_opener(candidate_spec=None, counting=False, opener_cls=None):
    candidate_spec = candidate_spec or make_candidate_spec()
    schema = build_feature_schema_v0()
    decision_assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(schema=schema, candidate_spec=candidate_spec),
        catalog_assembler=ActionCatalogAssembler(),
    )
    if opener_cls is not None:
        cls = opener_cls
    else:
        cls = CountingDecisionOpener if counting else RuntimeLearningDecisionOpener
    return cls(decision_assembler)


def make_service(
    plan_by_robot, candidate_spec=None, counting_opener=False,
    critic_state_builder=None, ground_truth_builder=None,
    decision_opener=None, episode_session=None,
):
    plugin = ScriptedPlugin(plan_by_robot)
    decision_source = LearningCoordinationDecisionSource(plugin)
    opener = decision_opener or make_decision_opener(candidate_spec, counting=counting_opener)
    step_allocator = EpisodeDecisionStepAllocator()
    session = episode_session or InMemoryAsynchronousLearningEpisodeSession(
        LearningTransitionAssembler(), InMemoryTrajectoryRecorder()
    )
    service = RuntimeLearningCaptureService(
        decision_source, opener, step_allocator, session,
        critic_state_builder or CriticStateBuilder(),
        ground_truth_builder or GroundTruthSnapshotBuilder(),
    )
    return service, plugin, opener


def make_capture_input(
    robot_ids, candidates_by_robot=None, contexts_by_robot=None, critic_sources_by_robot=None,
    ground_truth_sources_by_robot=None, closing_outcomes_by_robot=None, time_s=0.0,
    geometry=None, candidate_spec=None, services=None,
) -> RuntimeCoordinationCaptureInput:
    geometry = geometry or make_geometry()
    return RuntimeCoordinationCaptureInput(
        request=make_request(robot_ids, candidates_by_robot, services=services),
        time_s=time_s,
        contexts_by_robot=contexts_by_robot or {},
        critic_sources_by_robot=critic_sources_by_robot or {},
        ground_truth_sources_by_robot=ground_truth_sources_by_robot,
        closing_outcomes_by_robot=closing_outcomes_by_robot or {},
        grid_geometry=geometry,
        normalization=NORMALIZATION,
        candidate_spec=candidate_spec or make_candidate_spec(),
    )


class TestInitialEventOneRobot:
    def test_single_robot_assigned(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0], {0: (c0,)},
            contexts_by_robot={0: make_context(0, geometry)},
            critic_sources_by_robot={0: make_critic_source()},
        )
        result = service.capture_coordination_event(capture_input)

        assert result.newly_registered_robot_ids == (0,)
        assert result.replaced_robot_ids == ()
        assert result.unresolved == ()
        assert service.pending_robot_ids == (0,)

    def test_robot_receives_critic_step_0(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_sources_by_robot={0: make_critic_source()},
            )
        )
        transition = service.complete_terminal_robot_decision(
            0, make_outcome("ep-capture", 0, 0, terminated=True)
        )
        assert transition.critic_state.decision_step == 0
        assert transition.critic_state.time_s == 0.0


class TestInitialEventTwoRobots:
    def test_two_robots_assigned(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0, 1], {0: (c0,), 1: (c1,)},
            contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
        )
        result = service.capture_coordination_event(capture_input)

        assert set(result.newly_registered_robot_ids) == {0, 1}
        assert set(service.pending_robot_ids) == {0, 1}

    def test_steps_assigned_globally(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0, 1], {0: (c0,), 1: (c1,)},
            contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
        )
        result = service.capture_coordination_event(capture_input)

        by_id = {item.robot_id: item for item in result.opened_decision.assigned}
        assert {by_id[0].decision_step, by_id[1].decision_step} == {0, 1}
        assert service.next_decision_step == 2

    def test_critic_and_ground_truth_steps_match_actor_steps(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
                ground_truth_sources_by_robot={0: make_ground_truth_source(1.0), 1: make_ground_truth_source(2.0)},
            )
        )
        by_id = {item.robot_id: item for item in result.opened_decision.assigned}

        t0 = service.complete_terminal_robot_decision(
            0, make_outcome("ep-capture", by_id[0].decision_step, 0, terminated=True)
        )
        record_partial = None
        t1 = service.complete_terminal_robot_decision(
            1, make_outcome("ep-capture", by_id[1].decision_step, 1, terminated=True)
        )
        record = service.finish_episode()

        assert t0.critic_state.decision_step == by_id[0].decision_step
        assert t1.critic_state.decision_step == by_id[1].decision_step
        assert {step for step, _ in record.ground_truth_by_step} == {
            by_id[0].decision_step, by_id[1].decision_step
        }

    def test_robot_order_preserved(self):
        geometry = make_geometry()
        c5 = make_candidate(target=(2.0, 2.0))
        c2 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({5: ("ASSIGNED", 0), 2: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [5, 2], {5: (c5,), 2: (c2,)},
            contexts_by_robot={5: make_context(5, geometry), 2: make_context(2, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={5: make_critic_source(), 2: make_critic_source()},
        )
        result = service.capture_coordination_event(capture_input)

        assert result.newly_registered_robot_ids == (5, 2)
        assert result.opened_decision.assigned_robot_ids == (5, 2)

    def test_contexts_and_critic_content_exact(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0], {0: (c0,)},
            contexts_by_robot={0: make_context(0, geometry)},
            critic_sources_by_robot={0: make_critic_source(coverage=0.42)},
        )
        service.capture_coordination_event(capture_input)

        transition = service.complete_terminal_robot_decision(
            0, make_outcome("ep-capture", 0, 0, terminated=True)
        )
        assert transition.critic_state.global_feature_names == ("coverage",)
        assert transition.critic_state.global_features == (0.42,)
        assert transition.critic_state.decision_step == 0
        assert transition.critic_state.time_s == 0.0

    def test_ground_truth_optional(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0], {0: (c0,)},
            contexts_by_robot={0: make_context(0, geometry)},
            critic_sources_by_robot={0: make_critic_source()},
        )  # ground_truth_sources_by_robot omitted entirely
        result = service.capture_coordination_event(capture_input)
        assert result.newly_registered_robot_ids == (0,)


class TestPluginAndProviderCallCounts:
    def test_plugin_called_exactly_once(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0], {0: (c0,)},
            contexts_by_robot={0: make_context(0, geometry)},
            critic_sources_by_robot={0: make_critic_source()},
        )
        service.capture_coordination_event(capture_input)
        assert plugin.assign_calls == 1

    def test_team_frontier_provider_called_exactly_once(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))

        class CountingTeamProvider:
            def __init__(self, candidates):
                self.candidates = candidates
                self.calls = 0

            def candidates_for_team(self, request):
                self.calls += 1
                return self.candidates

        provider = CountingTeamProvider({0: (c0,)})
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0], candidates_by_robot=None,
            contexts_by_robot={0: make_context(0, geometry)},
            critic_sources_by_robot={0: make_critic_source()},
            services=CoordinationServices(team_frontier_provider=provider),
        )
        service.capture_coordination_event(capture_input)
        assert provider.calls == 1

    def test_decision_source_and_opener_not_repeated(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, opener = make_service({0: ("ASSIGNED", 0)}, counting_opener=True)
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0], {0: (c0,)},
            contexts_by_robot={0: make_context(0, geometry)},
            critic_sources_by_robot={0: make_critic_source()},
        )
        service.capture_coordination_event(capture_input)
        assert plugin.assign_calls == 1
        assert opener.open_calls == 1

    def test_critic_builder_called_once_per_assigned(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        critic_builder = CountingCriticStateBuilder()
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)}, critic_state_builder=critic_builder
        )
        service.start_episode(make_metadata())

        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
            )
        )
        assert critic_builder.build_calls == 2

    def test_ground_truth_builder_only_for_present_sources(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        ground_truth_builder = CountingGroundTruthBuilder()
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)}, ground_truth_builder=ground_truth_builder
        )
        service.start_episode(make_metadata())

        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
                # Ground truth source present only for robot 0.
                ground_truth_sources_by_robot={0: make_ground_truth_source(1.0)},
            )
        )
        assert ground_truth_builder.build_calls == 1

    def test_hold_failed_do_not_materialize_snapshots(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        critic_builder = CountingCriticStateBuilder()
        ground_truth_builder = CountingGroundTruthBuilder()
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0), 1: ("HOLD", "no candidates")},
            critic_state_builder=critic_builder, ground_truth_builder=ground_truth_builder,
        )
        service.start_episode(make_metadata())

        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (make_candidate(target=(9.0, 9.0)),)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_sources_by_robot={0: make_critic_source()},
            )
        )
        # Exactly one ASSIGNED robot (robot 0); the HOLD robot never triggers
        # a builder call.
        assert critic_builder.build_calls == 1
        assert ground_truth_builder.build_calls == 0


class TestReplacement:
    def _start_with_two_pending(self, service, geometry):
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service.start_episode(make_metadata())
        capture_input = make_capture_input(
            [0, 1], {0: (c0,), 1: (c1,)},
            contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
        )
        result = service.capture_coordination_event(capture_input)
        by_id = {item.robot_id: item for item in result.opened_decision.assigned}
        return by_id

    def test_robots_pending_after_first_event(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        self._start_with_two_pending(service, geometry)
        assert set(service.pending_robot_ids) == {0, 1}

    def test_second_event_only_robot_1(self):
        geometry = make_geometry()
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)}
        )
        by_id = self._start_with_two_pending(service, geometry)

        # Reprogram the plugin to only be asked about robot 1 this time.
        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        capture_input = make_capture_input(
            [1], {1: (c1_next,)},
            contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={1: make_critic_source()},
            closing_outcomes_by_robot={1: make_outcome("ep-capture", by_id[1].decision_step, 1)},
        )
        result = service.capture_coordination_event(capture_input)

        assert result.replaced_robot_ids == (1,)
        assert result.newly_registered_robot_ids == ()
        assert len(result.completed_transitions) == 1

    def test_robot_1_receives_step_2(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        by_id = self._start_with_two_pending(service, geometry)
        assert {by_id[0].decision_step, by_id[1].decision_step} == {0, 1}

        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        capture_input = make_capture_input(
            [1], {1: (c1_next,)},
            contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={1: make_critic_source()},
            closing_outcomes_by_robot={1: make_outcome("ep-capture", by_id[1].decision_step, 1)},
        )
        result = service.capture_coordination_event(capture_input)
        new_item = result.opened_decision.assigned[0]
        assert new_item.decision_step == 2

    def test_replacement_critic_and_ground_truth_carry_the_new_step(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        by_id = self._start_with_two_pending(service, geometry)

        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        service.capture_coordination_event(
            make_capture_input(
                [1], {1: (c1_next,)},
                contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={1: make_critic_source()},
                ground_truth_sources_by_robot={1: make_ground_truth_source(9.0)},
                closing_outcomes_by_robot={1: make_outcome("ep-capture", by_id[1].decision_step, 1)},
            )
        )
        # The new pending decision (step 2) closes with a critic/ground truth
        # tagged step 2, not step 1 (the one it replaced).
        transition = service.complete_terminal_robot_decision(
            1, make_outcome("ep-capture", 2, 1, terminated=True)
        )
        assert transition.critic_state.decision_step == 2

    def test_robot_1_replaces_its_previous_pending(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        by_id = self._start_with_two_pending(service, geometry)

        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        capture_input = make_capture_input(
            [1], {1: (c1_next,)},
            contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={1: make_critic_source()},
            closing_outcomes_by_robot={1: make_outcome("ep-capture", by_id[1].decision_step, 1)},
        )
        service.capture_coordination_event(capture_input)
        assert service.pending_robot_ids == (0, 1)  # robot 1 keeps its slot, now at step 2

    def test_robot_0_remains_pending_at_step_0(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        by_id = self._start_with_two_pending(service, geometry)
        assert by_id[0].decision_step == 0

        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        capture_input = make_capture_input(
            [1], {1: (c1_next,)},
            contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={1: make_critic_source()},
            closing_outcomes_by_robot={1: make_outcome("ep-capture", by_id[1].decision_step, 1)},
        )
        service.capture_coordination_event(capture_input)
        assert 0 in service.pending_robot_ids

    def test_step_1_transition_completes_before_step_0(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        by_id = self._start_with_two_pending(service, geometry)

        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        capture_input = make_capture_input(
            [1], {1: (c1_next,)},
            contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_sources_by_robot={1: make_critic_source()},
            closing_outcomes_by_robot={1: make_outcome("ep-capture", by_id[1].decision_step, 1)},
        )
        service.capture_coordination_event(capture_input)

        t1 = service.complete_terminal_robot_decision(
            1, make_outcome("ep-capture", 2, 1, terminated=True)
        )
        t0 = service.complete_terminal_robot_decision(
            0, make_outcome("ep-capture", by_id[0].decision_step, 0, terminated=True)
        )
        record = service.finish_episode()
        # step 1's transition closed during the replacement event itself,
        # before either terminal close below -- finish_episode() still
        # returns all three in decision_step order.
        assert [t.decision_step for t in record.transitions] == [0, 1, 2]


class TestClosingOutcomeValidation:
    def test_missing_closing_outcome_for_pending_fails(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
            )
        )
        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [1], {1: (c1_next,)},
                    contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_sources_by_robot={1: make_critic_source()},
                    closing_outcomes_by_robot={},  # missing!
                )
            )

    def test_extra_closing_outcome_for_new_robot_fails(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        c0 = make_candidate(target=(2.0, 2.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                    closing_outcomes_by_robot={0: make_outcome("ep-capture", 0, 0)},  # robot 0 has no pending yet
                )
            )

    def test_terminal_closing_outcome_with_replacement_fails(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
            )
        )
        by_id = {item.robot_id: item for item in result.opened_decision.assigned}
        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [1], {1: (c1_next,)},
                    contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_sources_by_robot={1: make_critic_source()},
                    closing_outcomes_by_robot={
                        1: make_outcome("ep-capture", by_id[1].decision_step, 1, terminated=True)
                    },
                )
            )

    def test_closing_outcome_for_wrong_robot_fails(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
            )
        )
        by_id = {item.robot_id: item for item in result.opened_decision.assigned}
        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [1], {1: (c1_next,)},
                    contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_sources_by_robot={1: make_critic_source()},
                    # Outcome keyed under robot 0 (not pending in this event at all).
                    closing_outcomes_by_robot={0: make_outcome("ep-capture", by_id[0].decision_step, 0)},
                )
            )

    def test_closing_outcome_with_wrong_step_fails_deeper(self):
        # This mismatch can only be caught inside
        # complete_robot_decision() (episode_session does not expose a
        # pending decision's stored decision_step through its public API),
        # which runs *after* allocate_many() -- so this is a post-allocation
        # failure: it gets wrapped into RuntimeLearningCaptureConsistencyError
        # and aborts the whole episode, rather than propagating the raw
        # ValueError from TransitionAssemblyInput.
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        c0 = make_candidate(target=(2.0, 2.0))
        result = service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_sources_by_robot={0: make_critic_source()},
            )
        )
        pending_step = result.opened_decision.assigned[0].decision_step

        plugin._plan_by_robot = {0: ("ASSIGNED", 0)}
        c0_next = make_candidate(target=(9.0, 9.0))
        wrong_step_outcome = make_outcome("ep-capture", pending_step + 99, 0)
        with pytest.raises(RuntimeLearningCaptureConsistencyError) as exc_info:
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0_next,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                    closing_outcomes_by_robot={0: wrong_step_outcome},
                )
            )
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert service.is_active is False


class TestSnapshotValidation:
    def test_missing_critic_source_for_assigned_robot_fails(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0, 1], {0: (c0,), 1: (c1,)},
                    contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_sources_by_robot={0: make_critic_source()},  # robot 1 missing
                )
            )

    def test_extra_critic_source_for_hold_robot_fails(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("HOLD", "no candidates")})
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0, 1], {0: (c0,), 1: (make_candidate(target=(9.0, 9.0)),)},
                    contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
                )
            )

    def test_extra_ground_truth_source_for_non_assigned_robot_fails(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("HOLD", "no candidates")})
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0, 1], {0: (c0,), 1: (make_candidate(target=(9.0, 9.0)),)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                    ground_truth_sources_by_robot={1: make_ground_truth_source(1.0)},
                )
            )


class TestUnresolvedRobots:
    def test_unresolved_without_pending_is_returned(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("HOLD", "no candidates")})
        service.start_episode(make_metadata())
        result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (make_candidate(target=(9.0, 9.0)),)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_sources_by_robot={0: make_critic_source()},
            )
        )
        assert result.unresolved[0].robot_id == 1
        assert result.unresolved[0].status == "HOLD"
        assert service.pending_robot_ids == (0,)

    def test_unresolved_not_registered_as_pending(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("HOLD", "no candidates")})
        service.start_episode(make_metadata())
        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (make_candidate(target=(9.0, 9.0)),)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_sources_by_robot={0: make_critic_source()},
            )
        )
        assert service.pending_robot_ids == (0,)

    def test_unresolved_with_existing_pending_fails(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
            )
        )
        # Now robot 1 goes HOLD while it still has a pending decision.
        plugin._plan_by_robot = {1: ("HOLD", "lost frontier")}
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [1], {1: ()},
                    contexts_by_robot={},
                    critic_sources_by_robot={},
                )
            )

    def test_no_synthetic_hold_action(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("HOLD", "no candidates")})
        service.start_episode(make_metadata())
        result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (make_candidate(target=(9.0, 9.0)),)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_sources_by_robot={0: make_critic_source()},
            )
        )
        assert result.opened_decision.assigned_robot_ids == (0,)  # robot 1 never opened


class TestCandidatePoolVerifiable:
    def test_candidate_pool_and_selected_index_verifiable(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 1)})
        service.start_episode(make_metadata())
        result = service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0, c1)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_sources_by_robot={0: make_critic_source()},
            )
        )
        assert result.prepared_decision.candidate_pool.candidates_by_robot[0] == (c0, c1)
        assert result.prepared_decision.selected_candidate_index_by_robot[0] == 1


class TestNoCandidateMetadataRead:
    def test_module_never_reads_candidate_metadata(self):
        # runtime_capture_service.py never inspects plugin metadata itself
        # (LearningCoordinationDecisionSource already did that at
        # construction) and never reads ExplorationCandidate.metadata --
        # ExplorationCandidate has no such field, and no ".metadata"
        # attribute access appears anywhere in this module at all.
        tree = ast.parse((LEARNING_DIR / "runtime_capture_service.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "metadata"


class TestBuilderFailureAbortsEpisode:
    def test_failing_critic_builder_aborts_episode_and_raises(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, critic_state_builder=FailingCriticStateBuilder()
        )
        service.start_episode(make_metadata())

        with pytest.raises(RuntimeLearningCaptureConsistencyError) as exc_info:
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        assert exc_info.value.__cause__ is not None
        assert service.is_active is False

    def test_failed_event_registers_no_pending_decision(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, critic_state_builder=FailingCriticStateBuilder()
        )
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        # Episode is fully gone -- pending_robot_ids can't even be queried
        # meaningfully, but a fresh start proves nothing was left dangling.
        service.start_episode(make_metadata(episode_id="ep-after-failure"))
        assert service.pending_robot_ids == ()

    def test_failing_builder_does_not_complete_a_replacement_transition(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
            )
        )
        by_id = {item.robot_id: item for item in result.opened_decision.assigned}

        # Swap in a failing critic builder for the replacement event only.
        service._critic_state_builder = FailingCriticStateBuilder()
        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [1], {1: (c1_next,)},
                    contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_sources_by_robot={1: make_critic_source()},
                    closing_outcomes_by_robot={1: make_outcome("ep-capture", by_id[1].decision_step, 1)},
                )
            )
        assert service.is_active is False

    def test_can_start_new_episode_after_abort(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, critic_state_builder=FailingCriticStateBuilder()
        )
        service.start_episode(make_metadata(episode_id="ep-doomed"))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        service.start_episode(make_metadata(episode_id="ep-fresh"))
        assert service.is_active is True
        assert service.episode_id == "ep-fresh"


class TestSourceObjectsNotMutated:
    def test_critic_source_reused_across_events_stays_unchanged(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        shared_source = make_critic_source(coverage=0.7)
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: shared_source, 1: shared_source},
            )
        )
        assert shared_source.global_features == {"coverage": 0.7}
        assert shared_source.global_feature_names == ("coverage",)

    def test_same_ground_truth_source_produces_distinct_snapshots_per_step(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        shared_ground_truth = make_ground_truth_source(fire_x=3.0)
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
                ground_truth_sources_by_robot={0: shared_ground_truth, 1: shared_ground_truth},
            )
        )
        by_id = {item.robot_id: item for item in result.opened_decision.assigned}
        t0 = service.complete_terminal_robot_decision(
            0, make_outcome("ep-capture", by_id[0].decision_step, 0, terminated=True)
        )
        t1 = service.complete_terminal_robot_decision(
            1, make_outcome("ep-capture", by_id[1].decision_step, 1, terminated=True)
        )
        record = service.finish_episode()
        steps_and_gt = dict(record.ground_truth_by_step)
        assert set(steps_and_gt) == {by_id[0].decision_step, by_id[1].decision_step}
        gt0 = steps_and_gt[by_id[0].decision_step]
        gt1 = steps_and_gt[by_id[1].decision_step]
        assert gt0 != gt1  # different decision_step -- never literally the same object
        assert gt0.decision_step != gt1.decision_step
        assert gt0.true_fire_locations == gt1.true_fire_locations  # same world content


class TestSmokeIndependentBaselinePlugin:
    def test_real_plugin_single_robot_event(self):
        geometry = make_geometry()
        low = make_candidate(target=(1.0, 1.0), information_gain=1.0)
        high = make_candidate(target=(5.0, 5.0), information_gain=4.0)

        decision_source = LearningCoordinationDecisionSource(create_independent_baseline())
        decision_opener = make_decision_opener()
        step_allocator = EpisodeDecisionStepAllocator()
        episode_session = InMemoryAsynchronousLearningEpisodeSession(
            LearningTransitionAssembler(), InMemoryTrajectoryRecorder()
        )
        service = RuntimeLearningCaptureService(
            decision_source, decision_opener, step_allocator, episode_session,
            CriticStateBuilder(), GroundTruthSnapshotBuilder(),
        )
        service.start_episode(make_metadata())

        result = service.capture_coordination_event(
            make_capture_input(
                [0], {0: (low, high)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_sources_by_robot={0: make_critic_source()},
            )
        )
        assert result.newly_registered_robot_ids == (0,)
        assert result.opened_decision.assigned[0].selections.selections[0].action_index == 1


class TestCaptureInputRejectsOldFieldNames:
    def test_old_critic_states_by_robot_kwarg_rejected(self):
        geometry = make_geometry()
        with pytest.raises(TypeError):
            RuntimeCoordinationCaptureInput(
                request=make_request([0], {0: (make_candidate(),)}),
                time_s=0.0,
                contexts_by_robot={0: make_context(0, geometry)},
                critic_states_by_robot={0: make_critic_source()},  # old field name
                ground_truth_by_robot=None,
                closing_outcomes_by_robot={},
                grid_geometry=geometry,
                normalization=NORMALIZATION,
                candidate_spec=make_candidate_spec(),
            )


class TestSourcesAreStepAgnostic:
    def test_critic_source_has_no_decision_step_or_time_s_field(self):
        field_names = {f.name for f in dataclasses.fields(CriticStateCaptureSource)}
        assert "decision_step" not in field_names
        assert "time_s" not in field_names

    def test_ground_truth_source_has_no_decision_step_or_time_s_field(self):
        field_names = {f.name for f in dataclasses.fields(GroundTruthCaptureSource)}
        assert "decision_step" not in field_names
        assert "time_s" not in field_names


# --- Deep immutability of the capture sources -------------------------------


class TestCriticStateCaptureSourceDeepImmutability:
    def test_mutating_original_global_features_after_construction_has_no_effect(self):
        original_global = {"coverage": 0.1}
        source = CriticStateCaptureSource(
            global_feature_names=("coverage",), global_features=original_global,
            per_robot_feature_names=(), per_robot_features={},
        )
        original_global["coverage"] = 999.0
        original_global["new_key"] = 1.0
        assert source.global_features == {"coverage": 0.1}

    def test_mutating_original_nested_per_robot_features_after_construction_has_no_effect(self):
        original_per_robot = {0: {"x": 1.0}}
        source = CriticStateCaptureSource(
            global_feature_names=(), global_features={},
            per_robot_feature_names=("x",), per_robot_features=original_per_robot,
        )
        original_per_robot[0]["x"] = 999.0
        assert source.per_robot_features[0]["x"] == 1.0

    def test_mutating_original_outer_per_robot_mapping_after_construction_has_no_effect(self):
        original_per_robot = {0: {"x": 1.0}}
        source = CriticStateCaptureSource(
            global_feature_names=(), global_features={},
            per_robot_feature_names=("x",), per_robot_features=original_per_robot,
        )
        original_per_robot[1] = {"x": 2.0}
        assert set(source.per_robot_features) == {0}

    def test_assigning_into_global_features_through_source_raises(self):
        source = make_critic_source(coverage=0.2)
        with pytest.raises(TypeError):
            source.global_features["coverage"] = 5.0

    def test_assigning_into_outer_per_robot_features_through_source_raises(self):
        source = CriticStateCaptureSource(
            global_feature_names=(), global_features={},
            per_robot_feature_names=("x",), per_robot_features={0: {"x": 1.0}},
        )
        with pytest.raises(TypeError):
            source.per_robot_features[0] = {"x": 2.0}

    def test_assigning_into_nested_per_robot_features_through_source_raises(self):
        source = CriticStateCaptureSource(
            global_feature_names=(), global_features={},
            per_robot_feature_names=("x",), per_robot_features={0: {"x": 1.0}},
        )
        with pytest.raises(TypeError):
            source.per_robot_features[0]["x"] = 5.0

    def test_builder_receives_the_frozen_values_not_the_mutated_originals(self):
        original_global = {"coverage": 0.1}
        original_per_robot = {0: {"x": 1.0}}
        source = CriticStateCaptureSource(
            global_feature_names=("coverage",), global_features=original_global,
            per_robot_feature_names=("x",), per_robot_features=original_per_robot,
        )
        # Mutate the originals *before* the builder ever runs.
        original_global["coverage"] = 999.0
        original_per_robot[0]["x"] = 999.0

        build_input = CriticStateBuildInput(
            decision_step=0, time_s=0.0,
            global_feature_names=source.global_feature_names,
            global_features=source.global_features,
            per_robot_feature_names=source.per_robot_feature_names,
            per_robot_features=source.per_robot_features,
        )
        critic_state = CriticStateBuilder().build(build_input)
        assert critic_state.global_features == (0.1,)
        assert critic_state.per_robot_features[0] == (1.0,)


class TestGroundTruthCaptureSourceDeepImmutability:
    def test_mutating_original_true_robot_poses_after_construction_has_no_effect(self):
        original_poses = {0: (1.0, 2.0, 0.0)}
        source = GroundTruthCaptureSource(
            true_robot_poses=original_poses, true_occupancy=(), true_fire_locations=(),
            global_coverage_fraction=0.0,
        )
        original_poses[0] = (99.0, 99.0, 99.0)
        original_poses[1] = (5.0, 5.0, 5.0)
        assert source.true_robot_poses == {0: (1.0, 2.0, 0.0)}

    def test_mutating_lists_used_for_occupancy_after_construction_has_no_effect(self):
        row_a = [0, 1, 0]
        row_b = [1, 1, 0]
        occupancy = [row_a, row_b]
        source = GroundTruthCaptureSource(
            true_robot_poses={}, true_occupancy=occupancy, true_fire_locations=(),
            global_coverage_fraction=0.0,
        )
        row_a[0] = 9
        occupancy.append([1, 1, 1])
        assert source.true_occupancy == ((0, 1, 0), (1, 1, 0))

    def test_mutating_lists_used_for_fire_locations_after_construction_has_no_effect(self):
        fire_a = [1.0, 2.0]
        fire_locations = [fire_a]
        source = GroundTruthCaptureSource(
            true_robot_poses={}, true_occupancy=(), true_fire_locations=fire_locations,
            global_coverage_fraction=0.0,
        )
        fire_a[0] = 999.0
        fire_locations.append([9.0, 9.0])
        assert source.true_fire_locations == ((1.0, 2.0),)

    def test_assigning_into_true_robot_poses_through_source_raises(self):
        source = GroundTruthCaptureSource(
            true_robot_poses={0: (1.0, 2.0, 0.0)}, true_occupancy=(), true_fire_locations=(),
            global_coverage_fraction=0.0,
        )
        with pytest.raises(TypeError):
            source.true_robot_poses[0] = (0.0, 0.0, 0.0)

    def test_builder_receives_the_frozen_values_not_the_mutated_originals(self):
        original_poses = {0: (1.0, 2.0, 0.0)}
        source = GroundTruthCaptureSource(
            true_robot_poses=original_poses, true_occupancy=((0, 1),), true_fire_locations=((3.0, 4.0),),
            global_coverage_fraction=0.25,
        )
        original_poses[0] = (999.0, 999.0, 999.0)

        build_input = GroundTruthBuildInput(
            decision_step=0, time_s=0.0,
            true_robot_poses=source.true_robot_poses,
            true_occupancy=source.true_occupancy,
            true_fire_locations=source.true_fire_locations,
            global_coverage_fraction=source.global_coverage_fraction,
        )
        ground_truth = GroundTruthSnapshotBuilder().build(build_input)
        assert ground_truth.true_robot_poses == {0: (1.0, 2.0, 0.0)}


# --- Post-allocation failures abort the whole episode; pre-allocation ------
# --- failures propagate unwrapped and change nothing. ----------------------


class TestPreAllocationFailureDoesNotAbort:
    def test_plugin_failure_before_allocation_propagates_unwrapped(self):
        geometry = make_geometry()
        decision_source = LearningCoordinationDecisionSource(RaisingPlugin())
        decision_opener = make_decision_opener()
        step_allocator = EpisodeDecisionStepAllocator()
        episode_session = InMemoryAsynchronousLearningEpisodeSession(
            LearningTransitionAssembler(), InMemoryTrajectoryRecorder()
        )
        service = RuntimeLearningCaptureService(
            decision_source, decision_opener, step_allocator, episode_session,
            CriticStateBuilder(), GroundTruthSnapshotBuilder(),
        )
        service.start_episode(make_metadata())
        next_step_before = service.next_decision_step

        c0 = make_candidate(target=(2.0, 2.0))
        with pytest.raises(RuntimeError, match="simulated pre-allocation plugin failure"):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        # Not wrapped into RuntimeLearningCaptureConsistencyError, and no
        # abort happened: the episode is exactly as it was before the call.
        assert service.is_active is True
        assert service.next_decision_step == next_step_before


class TestPostAllocationFailureAbortsEverything:
    def test_critic_builder_failure_leaves_allocator_and_session_inactive(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, critic_state_builder=FailingCriticStateBuilder()
        )
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError) as exc_info:
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        assert service.is_active is False
        assert service.next_decision_step is None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_ground_truth_builder_failure_leaves_allocator_and_session_inactive(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, ground_truth_builder=FailingGroundTruthBuilder()
        )
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError) as exc_info:
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                    ground_truth_sources_by_robot={0: make_ground_truth_source(1.0)},
                )
            )
        assert service.is_active is False
        assert service.next_decision_step is None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_decision_opener_failure_leaves_allocator_and_session_inactive(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        failing_opener = make_decision_opener(opener_cls=FailingDecisionOpener)
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, decision_opener=failing_opener
        )
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError) as exc_info:
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        assert service.is_active is False
        assert service.next_decision_step is None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_register_opened_decisions_failure_leaves_allocator_and_session_inactive(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        failing_session = FailingRegisterEpisodeSession(
            LearningTransitionAssembler(), InMemoryTrajectoryRecorder()
        )
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, episode_session=failing_session
        )
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError) as exc_info:
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        assert service.is_active is False
        assert service.next_decision_step is None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_first_replacement_succeeds_second_fails_aborts_whole_episode(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata(episode_id="ep-partial"))
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        result = service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
            )
        )
        by_id = {item.robot_id: item for item in result.opened_decision.assigned}

        plugin._plan_by_robot = {0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)}
        c0_next = make_candidate(target=(8.0, 8.0))
        c1_next = make_candidate(target=(7.0, 7.0))
        # Robot 0's closing outcome is correct (its replacement would
        # succeed in isolation); robot 1's references the wrong
        # decision_step, so complete_robot_decision() fails for robot 1
        # *after* robot 0's has already gone through.
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0, 1], {0: (c0_next,), 1: (c1_next,)},
                    contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_sources_by_robot={0: make_critic_source(), 1: make_critic_source()},
                    closing_outcomes_by_robot={
                        0: make_outcome("ep-partial", by_id[0].decision_step, 0),
                        1: make_outcome("ep-partial", by_id[1].decision_step + 99, 1),
                    },
                )
            )
        # Robot 0's just-applied replacement is discarded along with
        # everything else -- the whole episode is gone, not half-applied.
        assert service.is_active is False
        assert service.next_decision_step is None

    def test_no_partial_episode_record_exportable_after_abort(self):
        geometry = make_geometry()
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, critic_state_builder=FailingCriticStateBuilder()
        )
        service.start_episode(make_metadata())
        c0 = make_candidate(target=(2.0, 2.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        with pytest.raises(RuntimeLearningCaptureStateError):
            service.finish_episode()

    def test_can_start_new_episode_after_post_allocation_failure(self):
        geometry = make_geometry()
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, critic_state_builder=FailingCriticStateBuilder()
        )
        service.start_episode(make_metadata(episode_id="ep-doomed"))
        c0 = make_candidate(target=(2.0, 2.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        service.start_episode(make_metadata(episode_id="ep-fresh"))
        assert service.is_active is True
        assert service.episode_id == "ep-fresh"

    def test_new_episode_after_failure_can_reuse_start_step_zero(self):
        geometry = make_geometry()
        service, plugin, _ = make_service(
            {0: ("ASSIGNED", 0)}, critic_state_builder=FailingCriticStateBuilder()
        )
        service.start_episode(make_metadata(episode_id="ep-doomed"))
        c0 = make_candidate(target=(2.0, 2.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_sources_by_robot={0: make_critic_source()},
                )
            )
        service.start_episode(make_metadata(episode_id="ep-fresh"))
        assert service.next_decision_step == 0
