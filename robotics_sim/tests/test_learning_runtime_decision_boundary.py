"""Boundary tests: RuntimeLearningDecisionOpener depends only on explicit
observable context (RobotDecisionObservationContext / HazardBeliefFrame),
never on ground truth; runtime_decision_opening.py imports nothing
privileged; no candidate.metadata read; no plugin/provider calls.

Field-name/AST inspection alone is not enough -- most tests below build two
real OpenedLearningDecision results under two different (conceptually)
hidden worlds and demonstrate they are identical, or that an observable
belief change does change thermal features."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Mapping

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.learning import (
    CandidateKind,
    CandidateSetSpec,
    CriticState,
    GroundTruthSnapshot,
    HoldPolicy,
)
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig
from robotics_sim.learning.action_catalog import ActionCatalogAssembler
from robotics_sim.learning.capture_inputs import CandidateCaptureInput
from robotics_sim.learning.coordination_decision_source import (
    ExplicitCandidatePool,
    PreparedLearningCoordinationDecision,
)
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler
from robotics_sim.learning.feature_schema_v0 import build_feature_schema_v0
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler
from robotics_sim.learning.runtime_decision_opening import (
    OpenedLearningDecision,
    OpenedRobotLearningDecision,
    RobotDecisionObservationContext,
    RuntimeDecisionOpeningInput,
    RuntimeLearningDecisionOpener,
    UnresolvedCoordinationDecision,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent
MODULE_NAME = "runtime_decision_opening.py"

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


def make_candidate(target=(4.0, 6.0), metadata=None) -> ExplorationCandidate:
    return ExplorationCandidate(
        target=target, source="test", information_gain=1.0,
        metadata=metadata if metadata is not None else {},
    )


def make_capture_input(candidate: ExplorationCandidate) -> CandidateCaptureInput:
    return CandidateCaptureInput(
        candidate=candidate, kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=True, reachable=True
    )


def make_prepared_decision(robot_id, candidate) -> PreparedLearningCoordinationDecision:
    candidate_pool = ExplicitCandidatePool(
        robot_ids=(robot_id,), candidates_by_robot={robot_id: (candidate,)}, source_name="test"
    )
    request = CoordinationRequest(
        robot_states=(make_robot(robot_id),), robots_to_assign=(robot_id,),
        proposals_by_robot={robot_id: (candidate,)},
    )
    result = CoordinationResult(
        targets=(), reasons=(), strategy="test",
        assignments=(
            CoordinationAssignment(
                robot_id=robot_id, status="ASSIGNED", target=candidate.target,
                proposal=candidate, reason="selected",
            ),
        ),
    )
    return PreparedLearningCoordinationDecision(
        request=request,
        candidate_pool=candidate_pool,
        result=result,
        capture_inputs_by_robot={robot_id: (make_capture_input(candidate),)},
        selected_candidate_index_by_robot={robot_id: 0},
    )


def make_context(robot_id, geometry, belief) -> RobotDecisionObservationContext:
    return RobotDecisionObservationContext(
        robot=make_robot(robot_id), hazard_belief=belief, graph_edges=(), visible_teammates=()
    )


def make_opening_input(prepared, decision_steps_by_robot, contexts_by_robot, geometry, episode_id="ep-boundary") -> RuntimeDecisionOpeningInput:
    return RuntimeDecisionOpeningInput(
        episode_id=episode_id, time_s=0.0,
        prepared_decision=prepared,
        decision_steps_by_robot=decision_steps_by_robot,
        contexts_by_robot=contexts_by_robot,
        grid_geometry=geometry, normalization=NORMALIZATION,
        candidate_spec=make_candidate_spec(),
    )


def make_opener() -> RuntimeLearningDecisionOpener:
    schema = build_feature_schema_v0()
    candidate_spec = make_candidate_spec()
    decision_assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(schema=schema, candidate_spec=candidate_spec),
        catalog_assembler=ActionCatalogAssembler(),
    )
    return RuntimeLearningDecisionOpener(decision_assembler)


def build_opened(geometry, belief_frame, robot_id=0, target=(4.0, 6.0)):
    candidate = make_candidate(target=target)
    prepared = make_prepared_decision(robot_id, candidate)
    opening_input = make_opening_input(
        prepared, {robot_id: 0}, {robot_id: make_context(robot_id, geometry, belief_frame)}, geometry
    )
    return make_opener().open(opening_input)


class TestGroundTruthDoesNotLeak:
    def test_two_hidden_worlds_same_belief_give_identical_opened_decision(self):
        geometry = make_geometry()

        # Two conceptually different hidden ground-truth worlds. Neither is
        # ever passed to RuntimeDecisionOpeningInput or
        # RuntimeLearningDecisionOpener -- there is no parameter that could
        # accept them.
        hidden_world_a = {"fires": (((8.0, 8.0), 0.9),), "occupancy_seed": 1}
        hidden_world_b = {"fires": (((1.0, 1.0), 0.2), ((5.0, 9.0), 1.0)), "occupancy_seed": 2}
        assert hidden_world_a != hidden_world_b

        def observed_belief():
            belief = HazardBelief(geometry)
            belief.observe_cells([6, 5, 0], [4, 4, 0], [0.3, 0.0, 0.0], robot_index=0)
            return belief.snapshot()

        opened_a = build_opened(geometry, observed_belief())
        opened_b = build_opened(geometry, observed_belief())
        assert opened_a == opened_b  # identical, despite the two hidden worlds

    def test_observable_belief_change_changes_thermal_features(self):
        geometry = make_geometry()
        belief = HazardBelief(geometry)
        belief.observe_cells([6], [4], [0.0], robot_index=0)
        opened_before = build_opened(geometry, belief.snapshot())

        belief.observe_cells([6], [4], [0.9], robot_index=0)  # fire now observed
        opened_after = build_opened(geometry, belief.snapshot())

        obs_before = opened_before.assigned[0].decision_capture.get_observation(0)
        obs_after = opened_after.assigned[0].decision_capture.get_observation(0)
        idx = obs_before.candidate_feature_names.index("fire_value_at_target")
        assert obs_before.candidate_features[0][idx] == 0.0
        assert obs_after.candidate_features[0][idx] == pytest.approx(0.9, rel=1e-6)
        assert opened_before != opened_after


class TestNoPrivilegedFieldAcceptsGroundTruthOrCritic:
    def test_context_hazard_belief_rejects_ground_truth_snapshot(self):
        snapshot = GroundTruthSnapshot(
            schema_version="0.1.0", decision_step=0, time_s=0.0, true_robot_poses={},
            true_occupancy=(), true_fire_locations=(), global_coverage_fraction=0.0,
        )
        with pytest.raises(TypeError):
            RobotDecisionObservationContext(
                robot=make_robot(0), hazard_belief=snapshot, graph_edges=(), visible_teammates=()
            )

    def test_context_hazard_belief_rejects_critic_state(self):
        critic_state = CriticState(
            schema_version="0.1.0", decision_step=0, time_s=0.0, global_feature_names=(),
            global_features=(), per_robot_feature_names=(), per_robot_features={},
        )
        with pytest.raises(TypeError):
            RobotDecisionObservationContext(
                robot=make_robot(0), hazard_belief=critic_state, graph_edges=(), visible_teammates=()
            )

    def test_no_field_named_ground_truth_or_critic(self):
        for cls in (
            RobotDecisionObservationContext,
            RuntimeDecisionOpeningInput,
            UnresolvedCoordinationDecision,
            OpenedRobotLearningDecision,
            OpenedLearningDecision,
        ):
            names = {f.name for f in dataclasses.fields(cls)}
            for forbidden in ("ground_truth", "true_fire", "true_occupancy", "critic_state", "metadata"):
                assert forbidden not in names, (cls.__name__, forbidden)


class _ExplodingMapping(Mapping):
    def __getitem__(self, key):  # pragma: no cover - failure path
        raise AssertionError("must not read candidate.metadata")

    def __iter__(self):  # pragma: no cover - failure path
        raise AssertionError("must not iterate candidate.metadata")

    def __len__(self):  # pragma: no cover - failure path
        raise AssertionError("must not measure candidate.metadata")


class TestMetadataNeverRead:
    def test_no_metadata_attribute_access_in_source(self):
        tree = ast.parse((LEARNING_DIR / MODULE_NAME).read_text(encoding="utf-8"))
        accessed = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "metadata"
        ]
        assert accessed == []

    def test_open_succeeds_with_exploding_metadata(self):
        geometry = make_geometry()
        candidate = make_candidate(target=(4.0, 6.0), metadata=_ExplodingMapping())
        prepared = make_prepared_decision(0, candidate)
        opening_input = make_opening_input(
            prepared, {0: 0}, {0: make_context(0, geometry, HazardBelief(geometry).snapshot())}, geometry
        )
        opened = make_opener().open(opening_input)
        assert opened.has_assigned_actions is True


class TestPrivilegedImports:
    FORBIDDEN_ROOTS = ("PyQt5", "PyQt6", "PySide2", "PySide6", "torch", "pandas")
    FORBIDDEN_MODULES = (
        "robotics_sim.simulation",
        "robotics_sim.app",
        "robotics_sim.environment.hazard_field",
        "robotics_sim.diagnostics",
    )
    FORBIDDEN_NAMES = (
        "HazardField",
        "FireSource",
        "GroundTruthSnapshot",
        "CriticState",
    )

    def _imports(self, filename: str):
        tree = ast.parse((LEARNING_DIR / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name, ""
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    yield node.module, alias.name

    def test_no_privileged_imports(self):
        for module, name in self._imports(MODULE_NAME):
            root = module.split(".")[0]
            assert root not in self.FORBIDDEN_ROOTS, (module,)
            assert not root.lower().startswith(("pyqt", "pyside")), (module,)
            for forbidden in self.FORBIDDEN_MODULES:
                assert not module.startswith(forbidden), (module,)
            assert name not in self.FORBIDDEN_NAMES, (module, name)


class TestNoPluginOrProviderCalls:
    def test_source_never_calls_assign_or_provider_methods(self):
        tree = ast.parse((LEARNING_DIR / MODULE_NAME).read_text(encoding="utf-8"))
        forbidden_calls = {"assign", "candidates_for_team", "candidates_for_robot"}
        called_methods = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden_calls
        }
        assert called_methods == set()

    def test_open_signature_has_no_plugin_or_services_parameter(self):
        # open() takes only a RuntimeDecisionOpeningInput -- there is no
        # plugin or CoordinationServices parameter anywhere in its
        # signature for it to call.
        import inspect

        params = inspect.signature(RuntimeLearningDecisionOpener.open).parameters
        assert set(params) == {"self", "opening_input"}

    def test_open_still_produces_a_result_from_only_the_prepared_decision(self):
        geometry = make_geometry()
        opened = build_opened(geometry, HazardBelief(geometry).snapshot())
        assert opened.has_assigned_actions is True
