"""Privileged-boundary tests for InMemoryAsynchronousLearningEpisodeSession:
CriticState and GroundTruthSnapshot are captured at register_opened_decisions()
time (never re-derived from later "world state" at completion time), ground
truth never leaks into a LearningTransition or CriticState, the module never
imports privileged/forbidden dependencies, and it never calls a plugin or a
candidate provider.  Built from real, assembled transitions -- not just AST
inspection."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
from pathlib import Path

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
from robotics_sim.learning.asynchronous_episode import InMemoryAsynchronousLearningEpisodeSession
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
)
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import (
    DecisionSelectionBatch,
    RobotActionSelection,
    RobotRewardOutcome,
    TransitionOutcomeBatch,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent
MODULE_PATH = LEARNING_DIR / "asynchronous_episode.py"
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


def make_candidate(target=(4.0, 6.0)) -> CandidateCaptureInput:
    return CandidateCaptureInput(
        candidate=ExplorationCandidate(target=target, source="frontier", information_gain=1.0),
        kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=True, reachable=True,
    )


def build_decision_capture(geometry, robot_id, decision_step, time_s, episode_id):
    candidate_spec = make_candidate_spec()
    robot_capture = RobotActorCaptureInput(
        robot=make_robot(robot_id), candidates=(make_candidate(),), graph_edges=(),
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


def make_opened_robot(geometry, robot_id, decision_step, time_s, episode_id):
    decision_capture = build_decision_capture(geometry, robot_id, decision_step, time_s, episode_id)
    selections = DecisionSelectionBatch(
        episode_id=episode_id, decision_step=decision_step,
        selections=(
            RobotActionSelection(robot_id=robot_id, action_index=0, issued_at_step=decision_step),
        ),
    )
    return OpenedRobotLearningDecision(
        robot_id=robot_id, decision_step=decision_step, time_s=time_s,
        decision_capture=decision_capture, selections=selections,
    )


def make_metadata(episode_id) -> EpisodeMetadata:
    bundle_hash = compute_contract_bundle_hash(build_contract_manifest())
    return EpisodeMetadata(
        episode_id=episode_id, seed=1, map_id="map-1", robot_count=1, fire_count=1,
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


def make_ground_truth(step, fire_x) -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        schema_version="0.1.0", decision_step=step, time_s=float(step), true_robot_poses={},
        true_occupancy=(), true_fire_locations=((fire_x, fire_x),),
        global_coverage_fraction=0.0,
    )


def make_outcome(episode_id, decision_step, robot_id) -> TransitionOutcomeBatch:
    return TransitionOutcomeBatch(
        episode_id=episode_id, decision_step=decision_step,
        rewards=(
            RobotRewardOutcome(
                robot_id=robot_id,
                components=(
                    RewardComponent(
                        name="new_coverage", raw_value=1.0, applied_weight=0.5, weighted_value=0.5
                    ),
                ),
            ),
        ),
        terminated=True, truncated=False, termination_reason=TerminationReason.MAX_STEPS,
    )


def make_session() -> InMemoryAsynchronousLearningEpisodeSession:
    return InMemoryAsynchronousLearningEpisodeSession(
        LearningTransitionAssembler(), InMemoryTrajectoryRecorder()
    )


def _run_one_robot_episode(episode_id, critic_state, ground_truth):
    geometry = make_geometry()
    session = make_session()
    session.start_episode(make_metadata(episode_id))
    r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0, episode_id=episode_id)
    session.register_opened_decisions(
        OpenedLearningDecision(episode_id=episode_id, time_s=0.0, assigned=(r0,), unresolved=()),
        {0: critic_state},
        {0: ground_truth} if ground_truth is not None else None,
    )
    transition = session.complete_robot_decision(
        robot_id=0, outcome=make_outcome(episode_id, 0, 0)
    )
    record = session.finish_episode()
    return transition, record


class TestCriticCapturedAtOpenNotAtCompletion:
    def test_critic_state_A_survives_conceptual_change_to_B(self):
        """1. Register with critic_state A.  2. Conceptually, a later
        global state B exists (built here, but never handed to the
        session).  3. Complete the action.  4. transition.critic_state is
        A, never B -- complete_robot_decision() has no parameter through
        which B could reach the transition at all."""
        episode_id = "ep-boundary-critic-ab"
        critic_a = make_critic_state(0, coverage=0.05)
        critic_b = make_critic_state(0, coverage=0.95)  # conceptual "later" state
        assert critic_a != critic_b

        transition, _ = _run_one_robot_episode(episode_id, critic_a, ground_truth=None)

        assert transition.critic_state == critic_a
        assert transition.critic_state != critic_b

    def test_two_sessions_differing_only_after_open_produce_equal_transitions(self):
        # Two independent episodes that open with the *same* critic_state
        # and ground_truth must produce equal transitions, since nothing
        # observed after opening can influence the assembled transition.
        episode_id = "ep-boundary-equal"
        critic = make_critic_state(0, coverage=0.3)
        t_a, _ = _run_one_robot_episode(episode_id, critic, make_ground_truth(0, fire_x=1.0))
        t_b, _ = _run_one_robot_episode(episode_id, critic, make_ground_truth(0, fire_x=99.0))
        assert t_a == t_b  # ground truth never affects the transition either


class TestGroundTruthCapturedAtOpenNotAtCompletion:
    def test_ground_truth_A_survives_conceptual_change(self):
        """1. Register ground truth A at open time.  2. Conceptually, the
        ground truth changes before completion (built here, but never
        handed to the session).  3. EpisodeRecord still stores A for that
        decision_step."""
        episode_id = "ep-boundary-gt-ab"
        gt_a = make_ground_truth(0, fire_x=1.0)
        gt_b_conceptual = make_ground_truth(0, fire_x=77.0)  # never passed to the session
        assert gt_a != gt_b_conceptual

        _, record = _run_one_robot_episode(episode_id, make_critic_state(0), gt_a)

        assert record.ground_truth_by_step == ((0, gt_a),)

    def test_ground_truth_lives_only_in_ground_truth_by_step(self):
        episode_id = "ep-boundary-separate"
        gt = make_ground_truth(0, fire_x=5.0)
        transition, record = _run_one_robot_episode(episode_id, make_critic_state(0), gt)

        for field in dataclasses.fields(transition):
            value = getattr(transition, field.name)
            assert not isinstance(value, GroundTruthSnapshot)
            if isinstance(value, dict):
                assert not any(isinstance(v, GroundTruthSnapshot) for v in value.values())

        assert record.ground_truth_by_step == ((0, gt),)


class TestNoForbiddenImports:
    def test_no_forbidden_module_or_type_imports(self):
        tree = ast.parse(MODULE_SOURCE)
        forbidden_module_roots = {
            "os", "pathlib", "shutil", "tempfile", "io",
            "engine", "pandas", "torch", "numpy",
            "PyQt5", "PySide2", "PySide6", "PyQt6",
        }
        forbidden_module_prefixes = ("robotics_sim.app", "robotics_sim.simulation")
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

    def test_full_flow_writes_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _run_one_robot_episode(
            "ep-boundary-no-files", make_critic_state(0), make_ground_truth(0, fire_x=1.0)
        )
        assert os.listdir(tmp_path) == []


class TestNoPluginOrCandidateProviderInteraction:
    def test_module_never_references_assign_or_candidate_metadata(self):
        tree = ast.parse(MODULE_SOURCE)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "assign"
                assert node.attr != "metadata"
            if isinstance(node, ast.Name):
                assert node.id not in {"plugin", "candidate_provider", "team_frontier_provider"}

    def test_session_api_has_no_plugin_or_provider_parameter(self):
        forbidden_param_fragments = ("plugin", "provider")
        for _, method in inspect.getmembers(
            InMemoryAsynchronousLearningEpisodeSession, predicate=inspect.isfunction
        ):
            for name in inspect.signature(method).parameters:
                lowered = name.lower()
                for fragment in forbidden_param_fragments:
                    assert fragment not in lowered, (
                        f"{method.__name__} accepts a suspicious parameter {name!r}"
                    )

    def test_real_transitions_assembled_without_any_plugin_object(self):
        # No plugin, coordination source, or candidate provider is ever
        # constructed or passed to the session -- the whole flow below only
        # uses already-opened decisions and the session's own public API,
        # yet still produces genuine, fully validated LearningTransition
        # objects (not a mock).
        transition, record = _run_one_robot_episode(
            "ep-boundary-no-plugin", make_critic_state(0), make_ground_truth(0, fire_x=2.0)
        )
        assert isinstance(transition, LearningTransition)
        assert len(record.transitions) == 1


class TestCriticStateNeverConstructedFromGroundTruth:
    def test_register_rejects_ground_truth_passed_as_critic_state(self):
        geometry = make_geometry()
        session = make_session()
        episode_id = "ep-boundary-no-mixing"
        session.start_episode(make_metadata(episode_id))
        r0 = make_opened_robot(geometry, robot_id=0, decision_step=0, time_s=0.0, episode_id=episode_id)
        gt = make_ground_truth(0, fire_x=1.0)
        try:
            session.register_opened_decisions(
                OpenedLearningDecision(
                    episode_id=episode_id, time_s=0.0, assigned=(r0,), unresolved=()
                ),
                {0: gt},  # a GroundTruthSnapshot where a CriticState is required
            )
            raised = False
        except TypeError:
            raised = True
        assert raised
        assert session.pending_count == 0
