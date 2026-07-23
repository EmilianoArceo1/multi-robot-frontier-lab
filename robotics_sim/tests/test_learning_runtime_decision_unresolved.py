"""Tests for HOLD/FAILED handling in RuntimeLearningDecisionOpener:
UnresolvedCoordinationDecision entries, never a LearningAction, never a
synthetic CandidateKind.HOLD, and never consuming a decision_step or
observable context for an unresolved robot."""

from __future__ import annotations

from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.learning import CandidateKind, CandidateSetSpec, HoldPolicy
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
    RobotDecisionObservationContext,
    RuntimeDecisionOpeningInput,
    RuntimeLearningDecisionOpener,
)

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


def make_candidate(target=(4.0, 6.0)) -> ExplorationCandidate:
    return ExplorationCandidate(target=target, source="test", information_gain=1.0)


def make_belief(geometry):
    return HazardBelief(geometry).snapshot()


def make_capture_input(candidate: ExplorationCandidate) -> CandidateCaptureInput:
    return CandidateCaptureInput(
        candidate=candidate, kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=True, reachable=True
    )


def _index_by_identity(seq, item):
    for i, x in enumerate(seq):
        if x is item:
            return i
    raise AssertionError("item not found by identity")


def make_prepared_decision(robots_spec) -> PreparedLearningCoordinationDecision:
    robot_ids = tuple(spec["robot_id"] for spec in robots_spec)
    candidates_by_robot = {spec["robot_id"]: spec["candidates"] for spec in robots_spec}
    candidate_pool = ExplicitCandidatePool(
        robot_ids=robot_ids, candidates_by_robot=candidates_by_robot, source_name="test"
    )

    request = CoordinationRequest(
        robot_states=tuple(make_robot(spec["robot_id"]) for spec in robots_spec),
        robots_to_assign=robot_ids,
        proposals_by_robot={rid: cands for rid, cands in candidates_by_robot.items() if cands},
    )

    assignments = []
    capture_inputs_by_robot = {}
    selected_index_by_robot = {}
    for spec in robots_spec:
        robot_id = spec["robot_id"]
        candidates = spec["candidates"]
        capture_inputs_by_robot[robot_id] = tuple(make_capture_input(c) for c in candidates)
        status = spec["status"]
        if status == "ASSIGNED":
            chosen = spec["chosen"]
            index = _index_by_identity(candidates, chosen)
            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id, status="ASSIGNED", target=chosen.target,
                    proposal=chosen, reason=spec.get("reason", "selected"),
                )
            )
        else:
            index = None
            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id, status=status, target=None, proposal=None,
                    reason=spec.get("reason", "no candidates"),
                )
            )
        selected_index_by_robot[robot_id] = index

    result = CoordinationResult(
        targets=(), reasons=(), strategy="test", assignments=tuple(assignments)
    )

    return PreparedLearningCoordinationDecision(
        request=request,
        candidate_pool=candidate_pool,
        result=result,
        capture_inputs_by_robot=capture_inputs_by_robot,
        selected_candidate_index_by_robot=selected_index_by_robot,
    )


def make_context(robot_id, geometry, xy=(1.0, 1.0)) -> RobotDecisionObservationContext:
    return RobotDecisionObservationContext(
        robot=make_robot(robot_id, xy),
        hazard_belief=make_belief(geometry),
        graph_edges=(),
        visible_teammates=(),
    )


def make_opening_input(
    prepared, decision_steps_by_robot, contexts_by_robot, geometry,
    episode_id="ep-unresolved", time_s=0.0,
) -> RuntimeDecisionOpeningInput:
    return RuntimeDecisionOpeningInput(
        episode_id=episode_id, time_s=time_s,
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


class TestSingleHold:
    def test_one_robot_hold(self):
        geometry = make_geometry()
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (), "status": "HOLD", "reason": "no candidates available"}]
        )
        # HOLD robots consume no decision_step and no observable context.
        opening_input = make_opening_input(prepared, {}, {}, geometry)

        opened = make_opener().open(opening_input)

        assert opened.has_assigned_actions is False
        assert opened.assigned == ()
        assert opened.unresolved_robot_ids == (0,)
        assert opened.unresolved[0].status == "HOLD"
        assert opened.unresolved[0].reason == "no candidates available"


class TestSingleFailed:
    def test_one_robot_failed(self):
        geometry = make_geometry()
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (), "status": "FAILED", "reason": "robot id not present"}]
        )
        opening_input = make_opening_input(prepared, {}, {}, geometry)

        opened = make_opener().open(opening_input)

        assert opened.assigned == ()
        assert opened.unresolved[0].status == "FAILED"
        assert opened.unresolved[0].reason == "robot id not present"


class TestMixedOutcomes:
    def test_assigned_and_hold(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (), "status": "HOLD", "reason": "no candidates"},
            ]
        )
        # Only robot 0 (ASSIGNED) needs a decision_step/context.
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)

        opened = make_opener().open(opening_input)

        assert opened.assigned_robot_ids == (0,)
        assert opened.unresolved_robot_ids == (1,)

    def test_assigned_and_failed(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (), "status": "FAILED", "reason": "unreachable"},
            ]
        )
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)

        opened = make_opener().open(opening_input)

        assert opened.assigned_robot_ids == (0,)
        assert opened.unresolved_robot_ids == (1,)
        assert opened.unresolved[0].status == "FAILED"


class TestAllUnresolved:
    def test_all_hold(self):
        geometry = make_geometry()
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (), "status": "HOLD", "reason": "no candidates"},
                {"robot_id": 1, "candidates": (), "status": "HOLD", "reason": "no candidates"},
            ]
        )
        opening_input = make_opening_input(prepared, {}, {}, geometry)

        opened = make_opener().open(opening_input)

        assert opened.assigned == ()
        assert opened.has_assigned_actions is False
        assert set(opened.unresolved_robot_ids) == {0, 1}

    def test_all_failed(self):
        geometry = make_geometry()
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (), "status": "FAILED", "reason": "planner error"},
                {"robot_id": 1, "candidates": (), "status": "FAILED", "reason": "planner error"},
            ]
        )
        opening_input = make_opening_input(prepared, {}, {}, geometry)

        opened = make_opener().open(opening_input)

        assert opened.assigned == ()
        assert set(opened.unresolved_robot_ids) == {0, 1}


class TestReasonAndCandidateCount:
    def test_reason_preserved(self):
        geometry = make_geometry()
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (), "status": "HOLD", "reason": "corridor reserved by teammate"}]
        )
        opening_input = make_opening_input(prepared, {}, {}, geometry)

        opened = make_opener().open(opening_input)
        assert opened.unresolved[0].reason == "corridor reserved by teammate"

    def test_candidate_count_correct_when_zero(self):
        geometry = make_geometry()
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (), "status": "HOLD", "reason": "no candidates"}]
        )
        opening_input = make_opening_input(prepared, {}, {}, geometry)

        opened = make_opener().open(opening_input)
        assert opened.unresolved[0].candidate_count == 0

    def test_candidate_count_correct_when_nonzero(self):
        # A robot can be HOLD even though real candidates existed for it
        # (e.g. the plugin rejected all of them) -- candidate_count reflects
        # the pool size, independent of the HOLD outcome.
        geometry = make_geometry()
        c0, c1 = make_candidate(target=(2.0, 2.0)), make_candidate(target=(3.0, 3.0))
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0, c1), "status": "HOLD", "reason": "all reserved"}]
        )
        opening_input = make_opening_input(prepared, {}, {}, geometry)

        opened = make_opener().open(opening_input)
        assert opened.unresolved[0].candidate_count == 2


class TestNoLearningActionAndNoSyntheticHold:
    def test_no_learning_action_field_on_unresolved(self):
        from dataclasses import fields

        from robotics_sim.learning.runtime_decision_opening import (
            UnresolvedCoordinationDecision,
        )

        field_names = {f.name for f in fields(UnresolvedCoordinationDecision)}
        assert field_names == {"robot_id", "status", "reason", "candidate_count"}

    def test_no_hold_kind_ever_constructed(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (), "status": "HOLD", "reason": "no candidates"},
            ]
        )
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)

        opened = make_opener().open(opening_input)
        obs = opened.assigned[0].decision_capture.get_observation(0)
        idx = obs.candidate_feature_names.index("hold_kind")
        assert obs.candidate_features[0][idx] == 0.0


class TestNoOverlapBetweenAssignedAndUnresolved:
    def test_robot_never_in_both(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (), "status": "HOLD", "reason": "no candidates"},
            ]
        )
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)

        opened = make_opener().open(opening_input)
        assert set(opened.assigned_robot_ids) & set(opened.unresolved_robot_ids) == set()
