"""Tests for RuntimeLearningCaptureService.capture_coordination_event(): the
plugin-call, candidate-pool, step-allocation, opening, and
register/replace classification flow for one coordination event."""

from __future__ import annotations

import ast
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
from robotics_sim.learning.coordination_decision_source import LearningCoordinationDecisionSource
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler
from robotics_sim.learning.decision_steps import EpisodeDecisionStepAllocator
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler
from robotics_sim.learning.recorder import InMemoryTrajectoryRecorder
from robotics_sim.learning.runtime_capture_service import (
    RuntimeCoordinationCaptureInput,
    RuntimeLearningCaptureConsistencyError,
    RuntimeLearningCaptureService,
)
from robotics_sim.learning.runtime_decision_opening import RobotDecisionObservationContext, RuntimeLearningDecisionOpener
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


def make_decision_opener(candidate_spec=None, counting=False):
    candidate_spec = candidate_spec or make_candidate_spec()
    schema = build_feature_schema_v0()
    decision_assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(schema=schema, candidate_spec=candidate_spec),
        catalog_assembler=ActionCatalogAssembler(),
    )
    cls = CountingDecisionOpener if counting else RuntimeLearningDecisionOpener
    return cls(decision_assembler)


def make_service(plan_by_robot, candidate_spec=None, counting_opener=False):
    plugin = ScriptedPlugin(plan_by_robot)
    decision_source = LearningCoordinationDecisionSource(plugin)
    decision_opener = make_decision_opener(candidate_spec, counting=counting_opener)
    step_allocator = EpisodeDecisionStepAllocator()
    episode_session = InMemoryAsynchronousLearningEpisodeSession(
        LearningTransitionAssembler(), InMemoryTrajectoryRecorder()
    )
    service = RuntimeLearningCaptureService(
        decision_source, decision_opener, step_allocator, episode_session
    )
    return service, plugin, decision_opener


def make_capture_input(
    robot_ids, candidates_by_robot=None, contexts_by_robot=None, critic_states_by_robot=None,
    ground_truth_by_robot=None, closing_outcomes_by_robot=None, time_s=0.0,
    geometry=None, candidate_spec=None, services=None,
) -> RuntimeCoordinationCaptureInput:
    geometry = geometry or make_geometry()
    return RuntimeCoordinationCaptureInput(
        request=make_request(robot_ids, candidates_by_robot, services=services),
        time_s=time_s,
        contexts_by_robot=contexts_by_robot or {},
        critic_states_by_robot=critic_states_by_robot or {},
        ground_truth_by_robot=ground_truth_by_robot,
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
            critic_states_by_robot={0: make_critic_state(0)},
        )
        result = service.capture_coordination_event(capture_input)

        assert result.newly_registered_robot_ids == (0,)
        assert result.replaced_robot_ids == ()
        assert result.unresolved == ()
        assert service.pending_robot_ids == (0,)


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
            critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
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
            critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
        )
        result = service.capture_coordination_event(capture_input)

        by_id = {item.robot_id: item for item in result.opened_decision.assigned}
        assert {by_id[0].decision_step, by_id[1].decision_step} == {0, 1}
        assert service.next_decision_step == 2

    def test_robot_order_preserved(self):
        geometry = make_geometry()
        c5 = make_candidate(target=(2.0, 2.0))
        c2 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({5: ("ASSIGNED", 0), 2: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [5, 2], {5: (c5,), 2: (c2,)},
            contexts_by_robot={5: make_context(5, geometry), 2: make_context(2, geometry, xy=(3.0, 3.0))},
            critic_states_by_robot={5: make_critic_state(0), 2: make_critic_state(1)},
        )
        result = service.capture_coordination_event(capture_input)

        assert result.newly_registered_robot_ids == (5, 2)
        assert result.opened_decision.assigned_robot_ids == (5, 2)

    def test_contexts_and_critic_exact(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        critic_a = make_critic_state(0, coverage=0.42)

        capture_input = make_capture_input(
            [0], {0: (c0,)},
            contexts_by_robot={0: make_context(0, geometry)},
            critic_states_by_robot={0: critic_a},
        )
        service.capture_coordination_event(capture_input)

        transition = service.complete_terminal_robot_decision(
            0, make_outcome("ep-capture", 0, 0, terminated=True)
        )
        assert transition.critic_state == critic_a

    def test_ground_truth_optional(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())

        capture_input = make_capture_input(
            [0], {0: (c0,)},
            contexts_by_robot={0: make_context(0, geometry)},
            critic_states_by_robot={0: make_critic_state(0)},
        )  # ground_truth_by_robot omitted entirely
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
            critic_states_by_robot={0: make_critic_state(0)},
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
            critic_states_by_robot={0: make_critic_state(0)},
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
            critic_states_by_robot={0: make_critic_state(0)},
        )
        service.capture_coordination_event(capture_input)
        assert plugin.assign_calls == 1
        assert opener.open_calls == 1


class TestReplacement:
    def _start_with_two_pending(self, service, geometry):
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service.start_episode(make_metadata())
        capture_input = make_capture_input(
            [0, 1], {0: (c0,), 1: (c1,)},
            contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
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
            critic_states_by_robot={1: make_critic_state(2)},
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
            critic_states_by_robot={1: make_critic_state(2)},
            closing_outcomes_by_robot={1: make_outcome("ep-capture", by_id[1].decision_step, 1)},
        )
        result = service.capture_coordination_event(capture_input)
        new_item = result.opened_decision.assigned[0]
        assert new_item.decision_step == 2

    def test_robot_1_replaces_its_previous_pending(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        by_id = self._start_with_two_pending(service, geometry)

        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        capture_input = make_capture_input(
            [1], {1: (c1_next,)},
            contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
            critic_states_by_robot={1: make_critic_state(2)},
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
            critic_states_by_robot={1: make_critic_state(2)},
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
            critic_states_by_robot={1: make_critic_state(2)},
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
                critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
            )
        )
        plugin._plan_by_robot = {1: ("ASSIGNED", 0)}
        c1_next = make_candidate(target=(7.0, 7.0))
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [1], {1: (c1_next,)},
                    contexts_by_robot={1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_states_by_robot={1: make_critic_state(2)},
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
                    critic_states_by_robot={0: make_critic_state(0)},
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
                critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
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
                    critic_states_by_robot={1: make_critic_state(2)},
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
                critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
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
                    critic_states_by_robot={1: make_critic_state(2)},
                    # Outcome keyed under robot 0 (not pending in this event at all).
                    closing_outcomes_by_robot={0: make_outcome("ep-capture", by_id[0].decision_step, 0)},
                )
            )

    def test_closing_outcome_with_wrong_step_fails_deeper(self):
        geometry = make_geometry()
        service, plugin, _ = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        c0 = make_candidate(target=(2.0, 2.0))
        result = service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_states_by_robot={0: make_critic_state(0)},
            )
        )
        pending_step = result.opened_decision.assigned[0].decision_step

        plugin._plan_by_robot = {0: ("ASSIGNED", 0)}
        c0_next = make_candidate(target=(9.0, 9.0))
        wrong_step_outcome = make_outcome("ep-capture", pending_step + 99, 0)
        with pytest.raises(ValueError):
            service.capture_coordination_event(
                make_capture_input(
                    [0], {0: (c0_next,)},
                    contexts_by_robot={0: make_context(0, geometry)},
                    critic_states_by_robot={0: make_critic_state(2)},
                    closing_outcomes_by_robot={0: wrong_step_outcome},
                )
            )


class TestSnapshotValidation:
    def test_missing_snapshot_for_assigned_robot_fails(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0, 1], {0: (c0,), 1: (c1,)},
                    contexts_by_robot={0: make_context(0, geometry)},  # robot 1 missing
                    critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
                )
            )

    def test_extra_snapshot_for_hold_robot_fails(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin, _ = make_service({0: ("ASSIGNED", 0), 1: ("HOLD", "no candidates")})
        service.start_episode(make_metadata())
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [0, 1], {0: (c0,), 1: (make_candidate(target=(9.0, 9.0)),)},
                    contexts_by_robot={0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                    critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
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
                critic_states_by_robot={0: make_critic_state(0)},
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
                critic_states_by_robot={0: make_critic_state(0)},
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
                critic_states_by_robot={0: make_critic_state(0), 1: make_critic_state(1)},
            )
        )
        # Now robot 1 goes HOLD while it still has a pending decision.
        plugin._plan_by_robot = {1: ("HOLD", "lost frontier")}
        with pytest.raises(RuntimeLearningCaptureConsistencyError):
            service.capture_coordination_event(
                make_capture_input(
                    [1], {1: ()},
                    contexts_by_robot={},
                    critic_states_by_robot={},
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
                critic_states_by_robot={0: make_critic_state(0)},
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
                critic_states_by_robot={0: make_critic_state(0)},
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
            decision_source, decision_opener, step_allocator, episode_session
        )
        service.start_episode(make_metadata())

        result = service.capture_coordination_event(
            make_capture_input(
                [0], {0: (low, high)},
                contexts_by_robot={0: make_context(0, geometry)},
                critic_states_by_robot={0: make_critic_state(0)},
            )
        )
        assert result.newly_registered_robot_ids == (0,)
        assert result.opened_decision.assigned[0].selections.selections[0].action_index == 1
