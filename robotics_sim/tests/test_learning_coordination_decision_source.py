"""Tests for ExplicitCandidatePool, PreparedLearningCoordinationDecision and
LearningCoordinationDecisionSource."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Mapping

import pytest

import robotics_sim.learning as learning_pkg
from algorithms.independent_baseline.plugin import (
    IndependentBaselinePlugin,
    create_plugin as create_independent_baseline,
)
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.learning import CandidateKind
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_interfaces.services import CoordinationServices
from robotics_sim.learning.coordination_decision_source import (
    ExplicitCandidatePool,
    LearningCoordinationDecisionSource,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent


def make_robot(robot_id=0, xy=(0.0, 0.0), current_target=None) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=xy,
        safety_radius=0.3,
        sensor_range=3.0,
        vision_model="cone",
        theta=0.0,
        current_target=current_target,
        is_active=True,
    )


def make_candidate(target=(1.0, 1.0), source="test", information_gain=1.0, travel_cost=0.0) -> ExplorationCandidate:
    return ExplorationCandidate(
        target=target, source=source, information_gain=information_gain, travel_cost=travel_cost
    )


def make_world() -> WorldSnapshot:
    return WorldSnapshot(bounds=(0.0, 10.0, 0.0, 10.0), resolution=0.5)


def make_request(
    robot_ids,
    robots_to_assign=None,
    proposals_by_robot=None,
    services=None,
    world=None,
) -> CoordinationRequest:
    robot_states = tuple(make_robot(rid) for rid in robot_ids)
    return CoordinationRequest(
        robot_states=robot_states,
        robots_to_assign=tuple(robots_to_assign if robots_to_assign is not None else robot_ids),
        world=world,
        proposals_by_robot=proposals_by_robot or {},
        services=services,
    )


class _CountingTeamProvider:
    def __init__(self, candidates_by_robot: Mapping[int, tuple[ExplorationCandidate, ...]]):
        self.candidates_by_robot = dict(candidates_by_robot)
        self.calls: list = []

    def candidates_for_team(self, request):
        self.calls.append(request)
        return self.candidates_by_robot


class _CountingFrontierProvider:
    def __init__(self, candidates_by_robot: Mapping[int, tuple[ExplorationCandidate, ...]]):
        self.candidates_by_robot = dict(candidates_by_robot)
        self.calls: list = []

    def candidates_for_robot(self, robot, world, blocked_targets=()):
        self.calls.append((robot.robot_id, tuple(blocked_targets)))
        return self.candidates_by_robot.get(robot.robot_id, ())


def make_source() -> LearningCoordinationDecisionSource:
    return LearningCoordinationDecisionSource(create_independent_baseline())


class TestExplicitCandidatePoolValidation:
    def test_missing_key_rejected(self):
        with pytest.raises(ValueError):
            ExplicitCandidatePool(
                robot_ids=(0, 1),
                candidates_by_robot={0: (make_candidate(),)},  # robot 1 missing
                source_name="test",
            )

    def test_extra_key_rejected(self):
        with pytest.raises(ValueError):
            ExplicitCandidatePool(
                robot_ids=(0,),
                candidates_by_robot={0: (make_candidate(),), 1: ()},  # extra key
                source_name="test",
            )

    def test_duplicate_robot_id_rejected(self):
        with pytest.raises(ValueError):
            ExplicitCandidatePool(
                robot_ids=(0, 0),
                candidates_by_robot={0: ()},
                source_name="test",
            )

    def test_non_tuple_value_rejected(self):
        with pytest.raises(TypeError):
            ExplicitCandidatePool(
                robot_ids=(0,),
                candidates_by_robot={0: [make_candidate()]},  # list, not tuple
                source_name="test",
            )

    def test_non_candidate_element_rejected(self):
        with pytest.raises(TypeError):
            ExplicitCandidatePool(
                robot_ids=(0,),
                candidates_by_robot={0: ("not-a-candidate",)},
                source_name="test",
            )

    def test_empty_source_name_rejected(self):
        with pytest.raises(ValueError):
            ExplicitCandidatePool(robot_ids=(0,), candidates_by_robot={0: ()}, source_name="")

    def test_order_preserved(self):
        c0, c1, c2 = make_candidate(target=(0.0, 0.0)), make_candidate(target=(1.0, 1.0)), make_candidate(target=(2.0, 2.0))
        pool = ExplicitCandidatePool(
            robot_ids=(0,), candidates_by_robot={0: (c2, c0, c1)}, source_name="test"
        )
        assert pool.candidates_by_robot[0] == (c2, c0, c1)


class TestExplicitProposalsPriority:
    def test_explicit_proposals_take_priority_over_providers(self):
        source = make_source()
        c0 = make_candidate(target=(3.0, 3.0), information_gain=5.0)
        team_provider = _CountingTeamProvider({0: (make_candidate(target=(9.0, 9.0)),)})
        request = make_request(
            robot_ids=(0,),
            proposals_by_robot={0: (c0,)},
            services=CoordinationServices(team_frontier_provider=team_provider),
        )

        prepared = source.prepare_and_assign(request)

        assert prepared.candidate_pool.source_name == "request.proposals_by_robot"
        assert prepared.candidate_pool.candidates_by_robot[0] == (c0,)
        # The team provider must never be consulted when explicit proposals
        # already cover every requested robot.
        assert team_provider.calls == []

    def test_plugin_evaluates_exactly_the_explicit_proposals(self):
        source = make_source()
        c0 = make_candidate(target=(1.0, 1.0), information_gain=1.0)
        c1 = make_candidate(target=(2.0, 2.0), information_gain=9.0)
        request = make_request(robot_ids=(0,), proposals_by_robot={0: (c0, c1)})

        prepared = source.prepare_and_assign(request)

        # IndependentBaselinePlugin picks highest information_gain.
        assert prepared.selected_candidate_index_by_robot[0] == 1
        assert prepared.result.targets[0] == c1.target

    def test_providers_not_called_again_inside_the_plugin(self):
        source = make_source()
        c0 = make_candidate(target=(1.0, 1.0))
        team_provider = _CountingTeamProvider({0: (make_candidate(target=(8.0, 8.0)),)})
        frontier_provider = _CountingFrontierProvider({0: (make_candidate(target=(7.0, 7.0)),)})
        request = make_request(
            robot_ids=(0,),
            proposals_by_robot={0: (c0,)},
            services=CoordinationServices(
                team_frontier_provider=team_provider, frontier_provider=frontier_provider
            ),
            world=make_world(),
        )

        source.prepare_and_assign(request)

        assert team_provider.calls == []
        assert frontier_provider.calls == []


class TestTeamProviderFallback:
    def test_team_provider_used_exactly_once(self):
        source = make_source()
        c0 = make_candidate(target=(4.0, 4.0), information_gain=2.0)
        team_provider = _CountingTeamProvider({0: (c0,)})
        request = make_request(
            robot_ids=(0,), services=CoordinationServices(team_frontier_provider=team_provider)
        )

        prepared = source.prepare_and_assign(request)

        assert prepared.candidate_pool.source_name == "team_frontier_provider"
        assert prepared.candidate_pool.candidates_by_robot[0] == (c0,)
        assert len(team_provider.calls) == 1


class TestFrontierProviderFallback:
    def test_frontier_provider_used_when_no_team_provider(self):
        source = make_source()
        c0 = make_candidate(target=(5.0, 5.0), information_gain=3.0)
        frontier_provider = _CountingFrontierProvider({0: (c0,)})
        request = make_request(
            robot_ids=(0,),
            services=CoordinationServices(frontier_provider=frontier_provider),
            world=make_world(),
        )

        prepared = source.prepare_and_assign(request)

        assert prepared.candidate_pool.source_name == "frontier_provider"
        assert prepared.candidate_pool.candidates_by_robot[0] == (c0,)
        assert len(frontier_provider.calls) == 1


class TestRequestNotMutated:
    def test_original_request_untouched(self):
        source = make_source()
        team_provider = _CountingTeamProvider({0: (make_candidate(),)})
        services = CoordinationServices(team_frontier_provider=team_provider)
        request = make_request(robot_ids=(0,), services=services)

        source.prepare_and_assign(request)

        assert request.proposals_by_robot == {}
        assert request.services is services
        assert request.services.team_frontier_provider is team_provider


class TestPreparedRequestConsistency:
    def test_prepared_request_has_same_candidates_and_order(self):
        source = make_source()
        c0, c1 = make_candidate(target=(0.0, 0.0)), make_candidate(target=(1.0, 1.0))
        request = make_request(robot_ids=(0,), proposals_by_robot={0: (c0, c1)})

        prepared = source.prepare_and_assign(request)

        assert prepared.request.proposals_by_robot[0] == (c0, c1)
        assert prepared.request.proposals_by_robot[0] == prepared.candidate_pool.candidates_by_robot[0]
        for a, b in zip(prepared.request.proposals_by_robot[0], prepared.candidate_pool.candidates_by_robot[0]):
            assert a is b


class TestTwoRobots:
    def test_two_robots_both_captured(self):
        source = make_source()
        c0 = make_candidate(target=(0.0, 0.0), information_gain=1.0)
        c1 = make_candidate(target=(1.0, 1.0), information_gain=1.0)
        request = make_request(
            robot_ids=(0, 1), proposals_by_robot={0: (c0,), 1: (c1,)}
        )

        prepared = source.prepare_and_assign(request)

        assert set(prepared.candidate_pool.robot_ids) == {0, 1}
        assert prepared.selected_candidate_index_by_robot[0] == 0
        assert prepared.selected_candidate_index_by_robot[1] == 0
        assert len(prepared.capture_inputs_by_robot[0]) == 1
        assert len(prepared.capture_inputs_by_robot[1]) == 1


class TestZeroCandidates:
    def test_zero_candidates_produces_hold_and_no_index(self):
        source = make_source()
        request = make_request(robot_ids=(0,))  # no proposals, no services

        prepared = source.prepare_and_assign(request)

        assert prepared.candidate_pool.candidates_by_robot[0] == ()
        assert prepared.candidate_pool.source_name == "none"
        assert prepared.selected_candidate_index_by_robot[0] is None
        assert prepared.capture_inputs_by_robot[0] == ()


class TestCaptureInputSemantics:
    def test_kind_is_frontier_viewpoint(self):
        source = make_source()
        c0 = make_candidate()
        request = make_request(robot_ids=(0,), proposals_by_robot={0: (c0,)})
        prepared = source.prepare_and_assign(request)
        assert prepared.capture_inputs_by_robot[0][0].kind is CandidateKind.FRONTIER_VIEWPOINT

    def test_enabled_and_reachable_are_true(self):
        source = make_source()
        c0 = make_candidate()
        request = make_request(robot_ids=(0,), proposals_by_robot={0: (c0,)})
        prepared = source.prepare_and_assign(request)
        capture = prepared.capture_inputs_by_robot[0][0]
        assert capture.enabled is True
        assert capture.reachable is True
        assert capture.rejection_reasons == ()

    def test_candidate_object_preserved(self):
        source = make_source()
        c0 = make_candidate()
        request = make_request(robot_ids=(0,), proposals_by_robot={0: (c0,)})
        prepared = source.prepare_and_assign(request)
        assert prepared.capture_inputs_by_robot[0][0].candidate is c0


class _ExplodingMapping(Mapping):
    def __getitem__(self, key):  # pragma: no cover - failure path
        raise AssertionError("must not read candidate.metadata")

    def __iter__(self):  # pragma: no cover - failure path
        raise AssertionError("must not iterate candidate.metadata")

    def __len__(self):  # pragma: no cover - failure path
        raise AssertionError("must not measure candidate.metadata")


class TestMetadataNeverRead:
    def test_no_metadata_attribute_access_in_source(self):
        tree = ast.parse(
            (LEARNING_DIR / "coordination_decision_source.py").read_text(encoding="utf-8")
        )
        accessed = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "metadata"
        ]
        assert accessed == []

    def test_prepare_and_assign_succeeds_with_exploding_metadata(self):
        source = make_source()
        c0 = ExplorationCandidate(
            target=(1.0, 1.0), source="test", information_gain=1.0, metadata=_ExplodingMapping()
        )
        request = make_request(robot_ids=(0,), proposals_by_robot={0: (c0,)})
        prepared = source.prepare_and_assign(request)
        assert prepared.selected_candidate_index_by_robot[0] == 0


class TestIndependentCallsShareNoState:
    def test_two_calls_with_different_providers_do_not_leak(self):
        source = make_source()

        team_provider_a = _CountingTeamProvider({0: (make_candidate(target=(1.0, 1.0)),)})
        request_a = make_request(
            robot_ids=(0,), services=CoordinationServices(team_frontier_provider=team_provider_a)
        )
        prepared_a = source.prepare_and_assign(request_a)

        team_provider_b = _CountingTeamProvider({0: (make_candidate(target=(2.0, 2.0)),)})
        request_b = make_request(
            robot_ids=(0,), services=CoordinationServices(team_frontier_provider=team_provider_b)
        )
        prepared_b = source.prepare_and_assign(request_b)

        assert prepared_a.candidate_pool.candidates_by_robot[0][0].target == (1.0, 1.0)
        assert prepared_b.candidate_pool.candidates_by_robot[0][0].target == (2.0, 2.0)
        # Each provider was consulted only by its own call.
        assert len(team_provider_a.calls) == 1
        assert len(team_provider_b.calls) == 1

    def test_same_source_instance_reusable_across_independent_requests(self):
        source = make_source()
        c0 = make_candidate(target=(1.0, 1.0), information_gain=1.0)
        c1 = make_candidate(target=(2.0, 2.0), information_gain=1.0)

        prepared_first = source.prepare_and_assign(
            make_request(robot_ids=(0,), proposals_by_robot={0: (c0,)})
        )
        prepared_second = source.prepare_and_assign(
            make_request(robot_ids=(1,), proposals_by_robot={1: (c1,)})
        )

        assert prepared_first.candidate_pool.robot_ids == (0,)
        assert prepared_second.candidate_pool.robot_ids == (1,)
        assert prepared_first.candidate_pool.candidates_by_robot[0][0] is c0
        assert prepared_second.candidate_pool.candidates_by_robot[1][0] is c1


class TestSmokeIndependentBaselinePlugin:
    def test_real_plugin_end_to_end(self):
        plugin = create_independent_baseline()
        assert isinstance(plugin, IndependentBaselinePlugin)
        source = LearningCoordinationDecisionSource(plugin)

        # Distinct targets per robot so the plugin's duplicate-target
        # reservation logic (see IndependentBaselinePlugin._choose_candidate)
        # never has to break a tie -- each robot's own highest-information
        # candidate is unambiguously the expected pick.
        low_0 = make_candidate(target=(1.0, 1.0), information_gain=1.0, travel_cost=0.0)
        high_0 = make_candidate(target=(5.0, 5.0), information_gain=4.0, travel_cost=1.0)
        low_1 = make_candidate(target=(2.0, 2.0), information_gain=1.0, travel_cost=0.0)
        high_1 = make_candidate(target=(6.0, 6.0), information_gain=4.0, travel_cost=1.0)
        request = make_request(
            robot_ids=(0, 1),
            proposals_by_robot={0: (low_0, high_0), 1: (high_1, low_1)},
        )

        prepared = source.prepare_and_assign(request)

        assert prepared.selected_candidate_index_by_robot[0] == 1  # `high_0` is index 1 for robot 0
        assert prepared.selected_candidate_index_by_robot[1] == 0  # `high_1` is index 0 for robot 1
        assert prepared.result.targets[0] == high_0.target
        assert prepared.result.targets[1] == high_1.target
        assert dataclasses.is_dataclass(prepared)
