"""Tests for RuntimeLearningDecisionOpener: assembling one independent
OpenedRobotLearningDecision per robot with an ASSIGNED coordination
outcome -- never a shared multi-robot batch."""

from __future__ import annotations

import pytest

from algorithms.independent_baseline.plugin import create_plugin as create_independent_baseline
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.learning import CandidateKind, CandidateSetSpec, HoldPolicy
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.plugins import CandidateInputMode, PluginMetadata
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig
from robotics_sim.learning.action_catalog import ActionCatalogAssembler
from robotics_sim.learning.capture_inputs import CandidateCaptureInput
from robotics_sim.learning.coordination_decision_source import (
    ExplicitCandidatePool,
    LearningCoordinationDecisionSource,
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


def make_candidate(target=(4.0, 6.0), heading_rad=None, information_gain=1.0) -> ExplorationCandidate:
    return ExplorationCandidate(
        target=target, source="test", information_gain=information_gain, heading_rad=heading_rad
    )


def make_belief(geometry, observations=()):
    belief = HazardBelief(geometry)
    for row, col, value in observations:
        belief.observe_cells([row], [col], [value], robot_index=0)
    return belief.snapshot()


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
    """robots_spec: list of dicts each with robot_id, candidates (tuple),
    status ("ASSIGNED"/"HOLD"/"FAILED"), chosen (ExplorationCandidate,
    required when status=="ASSIGNED"), reason (optional)."""

    robot_ids = tuple(spec["robot_id"] for spec in robots_spec)
    candidates_by_robot = {spec["robot_id"]: spec["candidates"] for spec in robots_spec}
    candidate_pool = ExplicitCandidatePool(
        robot_ids=robot_ids, candidates_by_robot=candidates_by_robot, source_name="test"
    )

    request = CoordinationRequest(
        robot_states=tuple(make_robot(spec["robot_id"]) for spec in robots_spec),
        robots_to_assign=robot_ids,
        proposals_by_robot=dict(candidates_by_robot),
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


def make_context(
    robot_id, geometry, xy=(1.0, 1.0), graph_edges=(), visible_teammates=(), belief=None
) -> RobotDecisionObservationContext:
    return RobotDecisionObservationContext(
        robot=make_robot(robot_id, xy),
        hazard_belief=belief if belief is not None else make_belief(geometry),
        graph_edges=graph_edges,
        visible_teammates=visible_teammates,
    )


def make_opening_input(
    prepared, decision_steps_by_robot, contexts_by_robot, geometry,
    episode_id="ep-open", time_s=0.0, candidate_spec=None,
) -> RuntimeDecisionOpeningInput:
    return RuntimeDecisionOpeningInput(
        episode_id=episode_id, time_s=time_s,
        prepared_decision=prepared,
        decision_steps_by_robot=decision_steps_by_robot,
        contexts_by_robot=contexts_by_robot,
        grid_geometry=geometry, normalization=NORMALIZATION,
        candidate_spec=candidate_spec or make_candidate_spec(),
    )


def make_opener(candidate_spec=None) -> RuntimeLearningDecisionOpener:
    schema = build_feature_schema_v0()
    candidate_spec = candidate_spec or make_candidate_spec()
    decision_assembler = DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(schema=schema, candidate_spec=candidate_spec),
        catalog_assembler=ActionCatalogAssembler(),
    )
    return RuntimeLearningDecisionOpener(decision_assembler)


class TestSingleRobotAssigned:
    def test_single_robot_assigned(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0), information_gain=5.0)
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0, c1), "status": "ASSIGNED", "chosen": c1}]
        )
        opening_input = make_opening_input(
            prepared, {0: 5}, {0: make_context(0, geometry)}, geometry
        )

        opened = make_opener().open(opening_input)

        assert opened.has_assigned_actions is True
        assert opened.assigned_robot_ids == (0,)
        assert opened.unresolved == ()
        entry = opened.assigned[0]
        assert entry.robot_id == 0
        assert entry.decision_step == 5
        assert entry.selections.selections[0].action_index == 1


class TestTwoRobotsIndependentBatches:
    def test_two_robots_produce_two_independent_decisions(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        opening_input = make_opening_input(
            prepared,
            {0: 0, 1: 1},
            {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
            geometry,
        )

        opened = make_opener().open(opening_input)

        assert set(opened.assigned_robot_ids) == {0, 1}
        assert len(opened.assigned) == 2
        # Each robot's DecisionCaptureBatch is its own independent object.
        by_id = {item.robot_id: item for item in opened.assigned}
        assert by_id[0].decision_capture is not by_id[1].decision_capture
        assert by_id[0].selections is not by_id[1].selections


class TestDifferentStepsPerRobot:
    def test_distinct_decision_steps(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        opening_input = make_opening_input(
            prepared,
            {0: 7, 1: 12},
            {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
            geometry,
        )

        opened = make_opener().open(opening_input)
        by_id = {item.robot_id: item for item in opened.assigned}
        assert by_id[0].decision_step == 7
        assert by_id[1].decision_step == 12

    def test_changing_one_robot_step_does_not_change_the_other(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        contexts = {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))}

        opened_a = make_opener().open(
            make_opening_input(prepared, {0: 1, 1: 100}, contexts, geometry)
        )
        opened_b = make_opener().open(
            make_opening_input(prepared, {0: 99, 1: 100}, contexts, geometry)
        )

        by_id_a = {item.robot_id: item for item in opened_a.assigned}
        by_id_b = {item.robot_id: item for item in opened_b.assigned}
        assert by_id_a[1].decision_step == by_id_b[1].decision_step == 100
        assert by_id_a[1].decision_capture == by_id_b[1].decision_capture
        assert by_id_a[0].decision_step != by_id_b[0].decision_step


class TestDuplicateStepsRejected:
    def test_duplicate_steps_across_robots_rejected(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        contexts = {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))}
        with pytest.raises(ValueError):
            make_opening_input(prepared, {0: 5, 1: 5}, contexts, geometry)


class TestDecisionStepKeysMustMatchAssigned:
    def test_missing_step_for_assigned_robot_rejected(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        contexts = {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))}
        with pytest.raises(ValueError):
            make_opening_input(prepared, {0: 5}, contexts, geometry)  # robot 1 missing

    def test_extra_step_for_unassigned_robot_rejected(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (), "status": "HOLD", "reason": "no candidates"},
            ]
        )
        contexts = {0: make_context(0, geometry)}
        with pytest.raises(ValueError):
            # robot 1 is HOLD -- must not appear in decision_steps_by_robot.
            make_opening_input(prepared, {0: 5, 1: 0}, contexts, geometry)


class TestContextKeysMustMatchAssigned:
    def test_missing_context_for_assigned_robot_rejected(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        with pytest.raises(ValueError):
            make_opening_input(prepared, {0: 0, 1: 1}, {0: make_context(0, geometry)}, geometry)

    def test_extra_context_for_hold_robot_rejected(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (), "status": "HOLD", "reason": "no candidates"},
            ]
        )
        with pytest.raises(ValueError):
            # robot 1 is HOLD -- must not appear in contexts_by_robot.
            make_opening_input(
                prepared, {0: 0},
                {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
                geometry,
            )


class TestOrderPreserved:
    def test_robot_order_preserved(self):
        geometry = make_geometry()
        c5 = make_candidate(target=(2.0, 2.0))
        c2 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 5, "candidates": (c5,), "status": "ASSIGNED", "chosen": c5},
                {"robot_id": 2, "candidates": (c2,), "status": "ASSIGNED", "chosen": c2},
            ]
        )
        opening_input = make_opening_input(
            prepared,
            {5: 0, 2: 1},
            {5: make_context(5, geometry), 2: make_context(2, geometry, xy=(3.0, 3.0))},
            geometry,
        )

        opened = make_opener().open(opening_input)
        assert opened.assigned_robot_ids == (5, 2)


class TestExactlyOneRobotPerBatch:
    def test_decision_capture_has_exactly_one_robot(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        opening_input = make_opening_input(
            prepared,
            {0: 0, 1: 1},
            {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
            geometry,
        )

        opened = make_opener().open(opening_input)
        for item in opened.assigned:
            assert len(item.decision_capture.actor_batch.observations) == 1
            assert len(item.selections.selections) == 1


class TestCandidateIdsUseOwnRobotStep:
    def test_candidate_ids_embed_each_robots_own_step(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        opening_input = make_opening_input(
            prepared,
            {0: 3, 1: 9},
            {0: make_context(0, geometry), 1: make_context(1, geometry, xy=(3.0, 3.0))},
            geometry,
        )

        opened = make_opener().open(opening_input)
        by_id = {item.robot_id: item for item in opened.assigned}
        obs0 = by_id[0].decision_capture.get_observation(0)
        obs1 = by_id[1].decision_capture.get_observation(1)
        assert obs0.candidate_ids[0] == "robot-0/step-3/candidate-0"
        assert obs1.candidate_ids[0] == "robot-1/step-9/candidate-0"


class TestHeadingPresence:
    def test_candidate_with_heading(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0), heading_rad=0.5)
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0}]
        )
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)

        opened = make_opener().open(opening_input)
        obs = opened.assigned[0].decision_capture.get_observation(0)
        idx = obs.candidate_feature_names.index("has_heading")
        assert obs.candidate_features[0][idx] == 1.0

    def test_candidate_without_heading(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0), heading_rad=None)
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0}]
        )
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)

        opened = make_opener().open(opening_input)
        obs = opened.assigned[0].decision_capture.get_observation(0)
        idx = obs.candidate_feature_names.index("has_heading")
        assert obs.candidate_features[0][idx] == 0.0


class TestActionMaskAligned:
    def test_action_mask_true_at_selected_index(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0}]
        )
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)

        opened = make_opener().open(opening_input)
        entry = opened.assigned[0]
        obs = entry.decision_capture.get_observation(0)
        selected_index = entry.selections.selections[0].action_index
        assert obs.action_mask[selected_index] is True


class TestGraphEdgesPreserved:
    def test_graph_edges_pass_through_unmodified(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        c1 = make_candidate(target=(6.0, 6.0))
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0, c1), "status": "ASSIGNED", "chosen": c0}]
        )
        opening_input = make_opening_input(
            prepared, {0: 0}, {0: make_context(0, geometry, graph_edges=((0, 1),))}, geometry
        )

        opened = make_opener().open(opening_input)
        obs = opened.assigned[0].decision_capture.get_observation(0)
        assert obs.graph_edges == ((0, 1),)


class TestVisibleTeammates:
    def test_teammates_visible(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        teammate = make_robot(robot_id=1, xy=(3.0, 3.0))
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0}]
        )
        opening_input = make_opening_input(
            prepared, {0: 0}, {0: make_context(0, geometry, visible_teammates=(teammate,))}, geometry
        )

        opened = make_opener().open(opening_input)
        obs = opened.assigned[0].decision_capture.get_observation(0)
        assert len(obs.visible_teammate_features) == 1


class TestDifferentBeliefsPerRobot:
    def test_beliefs_differ_by_robot(self):
        geometry = make_geometry()
        target = (4.0, 6.0)
        cell = geometry.world_to_grid(*target)
        c0 = make_candidate(target=target)
        c1 = make_candidate(target=target)
        belief_a = make_belief(geometry, observations=((cell.row, cell.col, 0.1),))
        belief_b = make_belief(geometry, observations=((cell.row, cell.col, 0.9),))

        prepared = make_prepared_decision(
            [
                {"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0},
                {"robot_id": 1, "candidates": (c1,), "status": "ASSIGNED", "chosen": c1},
            ]
        )
        opening_input = make_opening_input(
            prepared,
            {0: 0, 1: 1},
            {
                0: make_context(0, geometry, belief=belief_a),
                1: make_context(1, geometry, xy=(5.0, 5.0), belief=belief_b),
            },
            geometry,
        )

        opened = make_opener().open(opening_input)
        by_id = {item.robot_id: item for item in opened.assigned}
        obs0 = by_id[0].decision_capture.get_observation(0)
        obs1 = by_id[1].decision_capture.get_observation(1)
        idx = obs0.candidate_feature_names.index("fire_value_at_target")
        assert obs0.candidate_features[0][idx] != obs1.candidate_features[0][idx]


class TestInputsNotMutated:
    def test_prepared_and_opening_input_unchanged(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0}]
        )
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)

        before_pool = prepared.candidate_pool
        before_selected = dict(prepared.selected_candidate_index_by_robot)
        before_contexts = dict(opening_input.contexts_by_robot)
        before_steps = dict(opening_input.decision_steps_by_robot)

        make_opener().open(opening_input)

        assert prepared.candidate_pool is before_pool
        assert dict(prepared.selected_candidate_index_by_robot) == before_selected
        assert dict(opening_input.contexts_by_robot) == before_contexts
        assert dict(opening_input.decision_steps_by_robot) == before_steps
        assert opening_input.contexts_by_robot[0].robot.xy == (1.0, 1.0)


class TestPluginAndProvidersNotReinvoked:
    def test_plugin_not_called_again(self):
        class _CountingPlugin:
            metadata = PluginMetadata(
                name="counting-plugin", version="0.0.0", description="",
                capabilities=(), candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
            )

            def __init__(self, inner):
                self._inner = inner
                self.assign_calls = 0

            def assign(self, request):
                self.assign_calls += 1
                return self._inner.assign(request)

        counting_plugin = _CountingPlugin(create_independent_baseline())
        source = LearningCoordinationDecisionSource(counting_plugin)
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        request = CoordinationRequest(
            robot_states=(make_robot(0),), robots_to_assign=(0,),
            proposals_by_robot={0: (c0,)},
        )
        prepared = source.prepare_and_assign(request)
        assert counting_plugin.assign_calls == 1

        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)
        make_opener().open(opening_input)

        assert counting_plugin.assign_calls == 1  # unchanged by open()

    def test_providers_not_called(self):
        class _CountingTeamProvider:
            def __init__(self, candidates):
                self.candidates = candidates
                self.calls = 0

            def candidates_for_team(self, request):
                self.calls += 1
                return self.candidates

        from robotics_interfaces.services import CoordinationServices

        team_provider = _CountingTeamProvider({0: (make_candidate(target=(2.0, 2.0)),)})
        source = LearningCoordinationDecisionSource(create_independent_baseline())
        request = CoordinationRequest(
            robot_states=(make_robot(0),), robots_to_assign=(0,),
            services=CoordinationServices(team_frontier_provider=team_provider),
        )
        prepared = source.prepare_and_assign(request)
        calls_after_prepare = team_provider.calls

        geometry = make_geometry()
        opening_input = make_opening_input(prepared, {0: 0}, {0: make_context(0, geometry)}, geometry)
        make_opener().open(opening_input)

        assert team_provider.calls == calls_after_prepare  # unchanged by open()


class TestSmokeIndependentBaselinePlugin:
    def test_real_prepared_decision_end_to_end(self):
        geometry = make_geometry()
        low = make_candidate(target=(1.0, 1.0), information_gain=1.0)
        high = make_candidate(target=(5.0, 5.0), information_gain=4.0)

        source = LearningCoordinationDecisionSource(create_independent_baseline())
        request = CoordinationRequest(
            robot_states=(make_robot(0),), robots_to_assign=(0,),
            proposals_by_robot={0: (low, high)},
        )
        prepared = source.prepare_and_assign(request)

        opening_input = make_opening_input(prepared, {0: 4}, {0: make_context(0, geometry)}, geometry)
        opened = make_opener().open(opening_input)

        assert opened.has_assigned_actions is True
        assert opened.assigned[0].decision_step == 4
        assert opened.assigned[0].selections.selections[0].action_index == 1  # `high` wins
        assert opened.unresolved == ()


class TestNoSchemaField:
    def test_schema_is_not_a_field(self):
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(RuntimeDecisionOpeningInput)}
        assert "schema" not in field_names
        assert field_names == {
            "episode_id", "time_s", "prepared_decision", "decision_steps_by_robot",
            "contexts_by_robot", "grid_geometry", "normalization", "candidate_spec",
        }

    def test_constructing_with_schema_kwarg_fails(self):
        geometry = make_geometry()
        c0 = make_candidate(target=(2.0, 2.0))
        prepared = make_prepared_decision(
            [{"robot_id": 0, "candidates": (c0,), "status": "ASSIGNED", "chosen": c0}]
        )
        with pytest.raises(TypeError):
            RuntimeDecisionOpeningInput(
                episode_id="ep", time_s=0.0, prepared_decision=prepared,
                decision_steps_by_robot={0: 0},
                contexts_by_robot={0: make_context(0, geometry)},
                grid_geometry=geometry, normalization=NORMALIZATION,
                candidate_spec=make_candidate_spec(),
                schema=build_feature_schema_v0(),
            )
