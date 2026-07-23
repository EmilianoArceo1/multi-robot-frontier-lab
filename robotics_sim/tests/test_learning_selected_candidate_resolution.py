"""Tests for resolve_selected_candidate_index."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import CoordinationAssignment
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.learning.coordination_decision_source import (
    LearningCoordinatorCompatibilityError,
    resolve_selected_candidate_index,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent


def make_candidate(
    target=(1.0, 1.0),
    source="test",
    information_gain=1.0,
    travel_cost=0.0,
    safety_cost=0.0,
    overlap_cost=0.0,
    heading_cost=0.0,
    heading_rad=None,
) -> ExplorationCandidate:
    return ExplorationCandidate(
        target=target,
        source=source,
        information_gain=information_gain,
        travel_cost=travel_cost,
        safety_cost=safety_cost,
        overlap_cost=overlap_cost,
        heading_cost=heading_cost,
        heading_rad=heading_rad,
    )


def make_assignment(
    robot_id=0, status="ASSIGNED", target=(1.0, 1.0), proposal=None, reason=""
) -> CoordinationAssignment:
    return CoordinationAssignment(
        robot_id=robot_id, status=status, target=target, proposal=proposal, reason=reason
    )


def make_command(robot_id=0, status="ASSIGNED", target=(1.0, 1.0), heading_rad=None) -> RobotCommand:
    return RobotCommand(robot_id=robot_id, status=status, target=target, heading_rad=heading_rad)


class TestIdentityMatch:
    def test_resolves_by_object_identity(self):
        c0 = make_candidate(target=(0.0, 0.0))
        c1 = make_candidate(target=(1.0, 1.0))
        c2 = make_candidate(target=(2.0, 2.0))
        candidates = (c0, c1, c2)
        assignment = make_assignment(target=c1.target, proposal=c1)

        index = resolve_selected_candidate_index(candidates, assignment, command=None)
        assert index == 1


class TestStructuralFallback:
    def test_unique_structural_match(self):
        c0 = make_candidate(target=(0.0, 0.0), information_gain=1.0)
        c1 = make_candidate(target=(1.0, 1.0), information_gain=2.0)
        candidates = (c0, c1)

        # A *different object*, but structurally identical to c1 -- e.g. the
        # plugin rebuilt the candidate instead of echoing the same object.
        rebuilt = make_candidate(target=(1.0, 1.0), information_gain=2.0)
        assert rebuilt is not c1
        assignment = make_assignment(target=rebuilt.target, proposal=rebuilt)

        index = resolve_selected_candidate_index(candidates, assignment, command=None)
        assert index == 1

    def test_candidate_with_heading(self):
        c0 = make_candidate(target=(1.0, 1.0), heading_rad=0.5)
        candidates = (c0,)
        rebuilt = make_candidate(target=(1.0, 1.0), heading_rad=0.5)
        assignment = make_assignment(target=rebuilt.target, proposal=rebuilt)

        index = resolve_selected_candidate_index(candidates, assignment, command=None)
        assert index == 0

    def test_candidate_without_heading(self):
        c0 = make_candidate(target=(1.0, 1.0), heading_rad=None)
        candidates = (c0,)
        rebuilt = make_candidate(target=(1.0, 1.0), heading_rad=None)
        assignment = make_assignment(target=rebuilt.target, proposal=rebuilt)

        index = resolve_selected_candidate_index(candidates, assignment, command=None)
        assert index == 0

    def test_two_candidates_same_target_different_heading_resolved_uniquely(self):
        c0 = make_candidate(target=(1.0, 1.0), heading_rad=0.0)
        c1 = make_candidate(target=(1.0, 1.0), heading_rad=1.5)
        candidates = (c0, c1)

        rebuilt = make_candidate(target=(1.0, 1.0), heading_rad=1.5)
        assignment = make_assignment(target=rebuilt.target, proposal=rebuilt)

        index = resolve_selected_candidate_index(candidates, assignment, command=None)
        assert index == 1

    def test_not_resolved_by_target_alone(self):
        # Two candidates share a target but differ in cost fields -- picking
        # "the one with this target" would be a silent wrong answer here.
        c0 = make_candidate(target=(1.0, 1.0), travel_cost=1.0)
        c1 = make_candidate(target=(1.0, 1.0), travel_cost=5.0)
        candidates = (c0, c1)

        rebuilt = make_candidate(target=(1.0, 1.0), travel_cost=5.0)
        assignment = make_assignment(target=rebuilt.target, proposal=rebuilt)

        index = resolve_selected_candidate_index(candidates, assignment, command=None)
        assert index == 1


class TestHoldAndFailed:
    def test_hold_produces_none(self):
        candidates = (make_candidate(),)
        assignment = make_assignment(status="HOLD", target=None, proposal=None)
        assert resolve_selected_candidate_index(candidates, assignment, command=None) is None

    def test_failed_produces_none(self):
        candidates = (make_candidate(),)
        assignment = make_assignment(status="FAILED", target=None, proposal=None)
        assert resolve_selected_candidate_index(candidates, assignment, command=None) is None

    def test_none_assignment_produces_none(self):
        candidates = (make_candidate(),)
        assert resolve_selected_candidate_index(candidates, assignment=None, command=None) is None


class TestContradictionsAndAmbiguity:
    def test_command_target_contradicts_resolved_candidate(self):
        c0 = make_candidate(target=(1.0, 1.0))
        candidates = (c0,)
        assignment = make_assignment(target=c0.target, proposal=c0)
        command = make_command(target=(9.0, 9.0))

        with pytest.raises(LearningCoordinatorCompatibilityError):
            resolve_selected_candidate_index(candidates, assignment, command)

    def test_no_match_raises(self):
        candidates = (make_candidate(target=(0.0, 0.0)),)
        rogue = make_candidate(target=(99.0, 99.0))
        assignment = make_assignment(target=rogue.target, proposal=rogue)

        with pytest.raises(LearningCoordinatorCompatibilityError):
            resolve_selected_candidate_index(candidates, assignment, command=None)

    def test_ambiguous_matches_raise(self):
        c0 = make_candidate(target=(1.0, 1.0), information_gain=2.0)
        c1 = make_candidate(target=(1.0, 1.0), information_gain=2.0)  # structurally identical
        candidates = (c0, c1)
        rebuilt = make_candidate(target=(1.0, 1.0), information_gain=2.0)
        assignment = make_assignment(target=rebuilt.target, proposal=rebuilt)

        with pytest.raises(LearningCoordinatorCompatibilityError):
            resolve_selected_candidate_index(candidates, assignment, command=None)

    def test_assigned_without_exploration_candidate_proposal_raises(self):
        candidates = (make_candidate(),)
        assignment = make_assignment(status="ASSIGNED", proposal=None)
        with pytest.raises(LearningCoordinatorCompatibilityError):
            resolve_selected_candidate_index(candidates, assignment, command=None)


class TestOrderIndependenceAndDuplicates:
    def test_assignments_out_of_order_still_resolve_correctly(self):
        # resolve_selected_candidate_index itself takes one assignment, but
        # this demonstrates it does not depend on candidates/assignment
        # arriving in any particular relative order -- callers index by
        # robot_id, not position.
        c0 = make_candidate(target=(0.0, 0.0))
        c1 = make_candidate(target=(1.0, 1.0))
        c2 = make_candidate(target=(2.0, 2.0))
        candidates = (c2, c0, c1)  # deliberately not target-sorted

        assignment = make_assignment(target=c1.target, proposal=c1)
        index = resolve_selected_candidate_index(candidates, assignment, command=None)
        assert index == 2

    def test_commands_out_of_order_do_not_affect_resolution(self):
        c0 = make_candidate(target=(1.0, 1.0), heading_rad=0.3)
        candidates = (c0,)
        assignment = make_assignment(target=c0.target, proposal=c0)
        # A command for a wholly different, unrelated robot passed by
        # mistake would not even reach this function (caller indexes by
        # robot_id) -- here we confirm a *matching* command's own field
        # order/content is what is checked, not position.
        command = make_command(target=c0.target, heading_rad=0.3)
        index = resolve_selected_candidate_index(candidates, assignment, command)
        assert index == 0


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

    def test_resolution_succeeds_with_exploding_metadata(self):
        class _ExplodingMapping(Mapping):
            def __getitem__(self, key):  # pragma: no cover - failure path
                raise AssertionError("must not read candidate.metadata")

            def __iter__(self):  # pragma: no cover - failure path
                raise AssertionError("must not iterate candidate.metadata")

            def __len__(self):  # pragma: no cover - failure path
                raise AssertionError("must not measure candidate.metadata")

        candidate = ExplorationCandidate(
            target=(1.0, 1.0), source="test", information_gain=1.0, metadata=_ExplodingMapping()
        )
        candidates = (candidate,)
        assignment = make_assignment(target=candidate.target, proposal=candidate)
        index = resolve_selected_candidate_index(candidates, assignment, command=None)
        assert index == 0
