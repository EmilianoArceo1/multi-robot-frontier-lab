"""Privileged-boundary tests for RuntimeLearningCaptureService.

The boundary this module enforces has shifted from "the service never
builds CriticState/GroundTruthSnapshot" to: the service only *materializes*
step-bound CriticState/GroundTruthSnapshot contracts by calling the
CriticStateBuilder/GroundTruthSnapshotBuilder injected at construction, from
frozen, step-agnostic CriticStateCaptureSource/GroundTruthCaptureSource
objects -- it never computes a feature value itself, never derives ground
truth from critic state or vice versa, and never queries engine/live runtime
state. Ground truth still stays separate from every transition, the module
never imports privileged/forbidden dependencies, never writes files, and
never falls back silently for an incompatible plugin. Built from a real
episode with real components -- not just AST inspection."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
from pathlib import Path

import pytest

import robotics_sim.learning as learning_pkg
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
    GroundTruthSnapshot,
    HoldPolicy,
    TerminationReason,
    build_contract_manifest,
    compute_contract_bundle_hash,
)
from robotics_interfaces.learning.transitions import LearningTransition, RewardComponent
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.plugins import CandidateInputMode, PluginCapability, PluginMetadata
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig, build_feature_schema_v0
from robotics_sim.learning.action_catalog import ActionCatalogAssembler
from robotics_sim.learning.asynchronous_episode import InMemoryAsynchronousLearningEpisodeSession
from robotics_sim.learning.builders import CriticStateBuilder, GroundTruthSnapshotBuilder
from robotics_sim.learning.coordination_decision_source import (
    LearningCoordinationDecisionSource,
    LearningCoordinatorCompatibilityError,
)
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler
from robotics_sim.learning.decision_steps import EpisodeDecisionStepAllocator
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler
from robotics_sim.learning.recorder import InMemoryTrajectoryRecorder
from robotics_sim.learning.runtime_capture_service import (
    CriticStateCaptureSource,
    GroundTruthCaptureSource,
    RuntimeCoordinationCaptureInput,
    RuntimeLearningCaptureService,
)
from robotics_sim.learning.runtime_decision_opening import (
    RobotDecisionObservationContext,
    RuntimeLearningDecisionOpener,
)
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import RobotRewardOutcome, TransitionOutcomeBatch

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent
MODULE_PATH = LEARNING_DIR / "runtime_capture_service.py"
MODULE_SOURCE = MODULE_PATH.read_text(encoding="utf-8")

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


def make_metadata(episode_id="ep-cap-boundary") -> EpisodeMetadata:
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


def make_ground_truth_source(fire_x) -> GroundTruthCaptureSource:
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
        name="scripted-boundary-plugin", version="0.0.0", description="",
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
        return CoordinationResult(assignments=tuple(assignments), commands=tuple(commands), strategy="scripted")


def make_decision_opener(candidate_spec=None):
    candidate_spec = candidate_spec or make_candidate_spec()
    schema = build_feature_schema_v0()
    decision_assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(schema=schema, candidate_spec=candidate_spec),
        catalog_assembler=ActionCatalogAssembler(),
    )
    return RuntimeLearningDecisionOpener(decision_assembler)


def make_service(plan_by_robot):
    plugin = ScriptedPlugin(plan_by_robot)
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
    return service, plugin


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


def _run_single_robot_episode(episode_id, critic_source, ground_truth_source):
    geometry = make_geometry()
    c0 = make_candidate(target=(2.0, 2.0))
    service, plugin = make_service({0: ("ASSIGNED", 0)})
    service.start_episode(make_metadata(episode_id))
    service.capture_coordination_event(
        make_capture_input(
            [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: critic_source},
            ground_truth_sources_by_robot={0: ground_truth_source} if ground_truth_source is not None else None,
        )
    )
    transition = service.complete_terminal_robot_decision(
        0, make_outcome(episode_id, 0, 0, terminated=True)
    )
    record = service.finish_episode()
    return transition, record


class TestCriticStateFromOpenTime:
    def test_transition_critic_state_matches_the_source_captured_at_open(self):
        critic_source_a = make_critic_source(coverage=0.05)
        transition, _ = _run_single_robot_episode("ep-critic-a", critic_source_a, None)
        assert transition.critic_state.global_feature_names == ("coverage",)
        assert transition.critic_state.global_features == (0.05,)
        assert transition.critic_state.decision_step == 0

    def test_conceptual_later_critic_source_never_reaches_the_transition(self):
        critic_source_a = make_critic_source(coverage=0.05)
        critic_source_b_conceptual = make_critic_source(coverage=0.95)  # never passed anywhere
        assert critic_source_a.global_features != critic_source_b_conceptual.global_features

        transition, _ = _run_single_robot_episode("ep-critic-b", critic_source_a, None)
        assert transition.critic_state.global_features == (0.05,)
        assert transition.critic_state.global_features != critic_source_b_conceptual.global_features

    def test_two_episodes_open_with_same_critic_source_produce_equal_transitions(self):
        critic_source = make_critic_source(coverage=0.3)
        t_a, _ = _run_single_robot_episode("ep-equal", critic_source, make_ground_truth_source(1.0))
        t_b, _ = _run_single_robot_episode("ep-equal", critic_source, make_ground_truth_source(99.0))
        assert t_a == t_b  # differing ground truth never affects the transition


class TestGroundTruthSeparate:
    def test_ground_truth_lives_only_in_ground_truth_by_step(self):
        gt_source = make_ground_truth_source(fire_x=5.0)
        transition, record = _run_single_robot_episode(
            "ep-gt-separate", make_critic_source(), gt_source
        )

        for field in dataclasses.fields(transition):
            value = getattr(transition, field.name)
            assert not isinstance(value, GroundTruthSnapshot)
            if isinstance(value, dict):
                assert not any(isinstance(v, GroundTruthSnapshot) for v in value.values())

        assert len(record.ground_truth_by_step) == 1
        step, gt = record.ground_truth_by_step[0]
        assert step == 0
        assert gt.true_fire_locations == ((5.0, 5.0),)

    def test_changing_conceptual_ground_truth_without_changing_actor_inputs_does_not_change_transition(self):
        gt_source_a = make_ground_truth_source(fire_x=1.0)
        gt_source_b_conceptual = make_ground_truth_source(fire_x=123.0)  # never passed to the service
        assert gt_source_a.true_fire_locations != gt_source_b_conceptual.true_fire_locations

        transition, record = _run_single_robot_episode(
            "ep-gt-conceptual", make_critic_source(), gt_source_a
        )
        assert record.ground_truth_by_step[0][1].true_fire_locations == ((1.0, 1.0),)
        # The transition itself never carries ground truth at all, so it is
        # identical (same episode_id, same everything else) regardless of
        # which ground truth (if any) was captured.
        transition_without_gt, _ = _run_single_robot_episode(
            "ep-gt-conceptual", make_critic_source(), None
        )
        assert transition == transition_without_gt

    def test_ground_truth_never_enters_critic_state(self):
        # CriticStateBuilder rejects a GroundTruthSnapshot outright; proving
        # this at the builder level is the authoritative check that ground
        # truth cannot leak into CriticState through this service, since the
        # service's own materialization step delegates to exactly this
        # builder and passes it nothing but CriticStateCaptureSource fields.
        from robotics_sim.learning.source_models import CriticStateBuildInput

        gt = GroundTruthSnapshot(
            schema_version="0.1.0", decision_step=0, time_s=0.0, true_robot_poses={},
            true_occupancy=(), true_fire_locations=((1.0, 1.0),), global_coverage_fraction=0.0,
        )
        with pytest.raises(TypeError):
            CriticStateBuilder().build(
                CriticStateBuildInput(
                    decision_step=0, time_s=0.0, global_feature_names=(), global_features={},
                    per_robot_feature_names=(), per_robot_features={0: gt},
                )
            )


class TestDecisionStepAndTimeAlignment:
    def test_critic_and_ground_truth_decision_step_and_time_s_match_the_opened_decision(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        service, plugin = make_service({0: ("ASSIGNED", 0)})
        service.start_episode(make_metadata())
        result = service.capture_coordination_event(
            make_capture_input(
                [0], {0: (c0,)}, {0: make_context(0, geometry)}, {0: make_critic_source()},
                ground_truth_sources_by_robot={0: make_ground_truth_source(1.0)},
                time_s=12.5,
            )
        )
        opened_item = result.opened_decision.assigned[0]
        transition = service.complete_terminal_robot_decision(
            0, make_outcome("ep-cap-boundary", opened_item.decision_step, 0, terminated=True)
        )
        record = service.finish_episode()

        assert transition.critic_state.decision_step == opened_item.decision_step
        assert transition.critic_state.time_s == opened_item.time_s
        gt_step, gt = record.ground_truth_by_step[0]
        assert gt_step == opened_item.decision_step
        assert gt.decision_step == opened_item.decision_step
        assert gt.time_s == opened_item.time_s


class TestNoCriticOrGroundTruthDirectConstruction:
    def test_module_never_constructs_critic_state_or_ground_truth_directly(self):
        # runtime_capture_service.py must never call CriticState(...) or
        # GroundTruthSnapshot(...) itself -- both contracts are only ever
        # produced by the injected builders (CriticStateBuilder /
        # GroundTruthSnapshotBuilder), which live in a different module.
        tree = ast.parse(MODULE_SOURCE)
        constructor_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                constructor_names.add(node.func.id)
        assert "CriticState" not in constructor_names
        assert "GroundTruthSnapshot" not in constructor_names

    def test_module_delegates_materialization_to_injected_builders(self):
        # Positive half of the same boundary: the module *does* call
        # self._critic_state_builder.build(...) and
        # self._ground_truth_builder.build(...) -- it does not silently skip
        # materialization or inline the builder's logic.
        tree = ast.parse(MODULE_SOURCE)
        build_call_targets = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "build"
                and isinstance(node.func.value, ast.Attribute)
            ):
                build_call_targets.add(node.func.value.attr)
        assert "_critic_state_builder" in build_call_targets
        assert "_ground_truth_builder" in build_call_targets


class TestNoForbiddenImports:
    def test_no_forbidden_module_or_type_imports(self):
        tree = ast.parse(MODULE_SOURCE)
        forbidden_module_roots = {
            "os", "pathlib", "shutil", "tempfile", "io",
            "engine", "pandas", "torch", "numpy",
            "PyQt5", "PySide2", "PySide6", "PyQt6",
        }
        forbidden_module_prefixes = (
            "robotics_sim.app",
            "robotics_sim.simulation",
        )
        forbidden_names = {"HazardField", "FireSource"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in forbidden_module_roots
                    assert not alias.name.startswith(forbidden_module_prefixes)
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden_module_roots
                assert not node.module.startswith(forbidden_module_prefixes)
                for alias in node.names:
                    assert alias.name not in forbidden_names

    def test_no_filesystem_calls(self):
        tree = ast.parse(MODULE_SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "open"

    def test_full_episode_writes_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_single_robot_episode(
            "ep-no-files", make_critic_source(), make_ground_truth_source(fire_x=1.0)
        )
        assert os.listdir(tmp_path) == []

    def test_no_gui_or_ppo_symbols_referenced(self):
        tree = ast.parse(MODULE_SOURCE)
        forbidden_identifiers = {"QWidget", "QApplication", "PPO", "torch", "nn"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_identifiers:
                raise AssertionError(f"unexpected identifier {node.id!r} in runtime_capture_service.py")


class TestNoSilentPluginFallback:
    def test_incompatible_plugin_rejected_at_construction_not_swallowed(self):
        class PluginInternalPlugin:
            metadata = PluginMetadata(
                name="plugin-internal", version="0.0.0", description="",
                capabilities=(PluginCapability.COORDINATION,),
                candidate_input_mode=CandidateInputMode.PLUGIN_INTERNAL,
            )

            def assign(self, request):
                raise AssertionError("must never be called: incompatible plugins are rejected")

        with pytest.raises(LearningCoordinatorCompatibilityError):
            LearningCoordinationDecisionSource(PluginInternalPlugin())

    def test_service_never_wraps_or_retries_with_a_different_plugin(self):
        # RuntimeLearningCaptureService.__init__ requires already-built
        # collaborators for every dependency -- there is no fallback path,
        # no default plugin/builder, and no retry parameter anywhere in its
        # signature.
        params = inspect.signature(RuntimeLearningCaptureService.__init__).parameters
        assert set(params) == {
            "self", "decision_source", "decision_opener", "step_allocator", "episode_session",
            "critic_state_builder", "ground_truth_builder",
        }


class TestRealTransitionsNotJustAst:
    def test_real_multi_robot_episode_produces_genuine_transitions(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        service, plugin = make_service({0: ("ASSIGNED", 0), 1: ("ASSIGNED", 0)})
        service.start_episode(make_metadata("ep-real"))
        service.capture_coordination_event(
            make_capture_input(
                [0, 1], {0: (c0,), 1: (c1,)},
                {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                {0: make_critic_source(), 1: make_critic_source()},
                ground_truth_sources_by_robot={0: make_ground_truth_source(1.0), 1: make_ground_truth_source(2.0)},
            )
        )
        t0 = service.complete_terminal_robot_decision(0, make_outcome("ep-real", 0, 0, terminated=True))
        t1 = service.complete_terminal_robot_decision(1, make_outcome("ep-real", 1, 1, terminated=True))
        record = service.finish_episode()

        assert isinstance(t0, LearningTransition)
        assert isinstance(t1, LearningTransition)
        assert len(record.transitions) == 2
