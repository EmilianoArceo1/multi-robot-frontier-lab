"""Tests for EpisodeDecisionStepAllocator."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import robotics_sim.learning as learning_pkg
from robotics_sim.learning.decision_steps import (
    DecisionStepAllocatorStateError,
    EpisodeDecisionStepAllocator,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent


class TestAllocateRequiresActiveEpisode:
    def test_allocate_without_start_fails(self):
        allocator = EpisodeDecisionStepAllocator()
        with pytest.raises(DecisionStepAllocatorStateError):
            allocator.allocate(0)

    def test_allocate_many_without_start_fails(self):
        allocator = EpisodeDecisionStepAllocator()
        with pytest.raises(DecisionStepAllocatorStateError):
            allocator.allocate_many((0, 1))


class TestStartEpisode:
    def test_start_normal(self):
        allocator = EpisodeDecisionStepAllocator()
        assert allocator.is_active is False
        allocator.start_episode()
        assert allocator.is_active is True
        assert allocator.next_step == 0

    def test_start_step_nonzero(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode(start_step=100)
        assert allocator.next_step == 100
        assert allocator.allocate(0) == 100
        assert allocator.next_step == 101


class TestAllocateConsecutive:
    def test_allocate_consumes_consecutive_global_steps(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        assert allocator.allocate(0) == 0
        assert allocator.allocate(1) == 1
        assert allocator.allocate(0) == 2


class TestAllocateManyOrder:
    def test_allocate_many_preserves_order(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        result = allocator.allocate_many((4, 2))
        assert result == {4: 0, 2: 1}
        assert list(result) == [4, 2]

    def test_duplicate_ids_rejected(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        with pytest.raises(ValueError):
            allocator.allocate_many((1, 1))

    def test_duplicate_ids_rejected_before_consuming_any_step(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        with pytest.raises(ValueError):
            allocator.allocate_many((1, 2, 1))
        assert allocator.next_step == 0  # rejected call consumed nothing


class TestSameRobotMultipleGlobalSteps:
    def test_same_robot_can_receive_several_global_steps(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        allocator.allocate_many((4, 2))
        second = allocator.allocate(4)
        assert second == 2  # not 1 -- step 1 already belongs to robot 2

    def test_robot_id_does_not_determine_the_step(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        first = allocator.allocate(999)
        second = allocator.allocate(0)
        assert first == 0
        assert second == 1


class TestFinishAndAbort:
    def test_finish_clears_state(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        allocator.allocate(0)
        allocator.finish_episode()
        assert allocator.is_active is False
        with pytest.raises(DecisionStepAllocatorStateError):
            allocator.allocate(0)

    def test_abort_clears_state(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        allocator.allocate(0)
        allocator.abort_episode()
        assert allocator.is_active is False
        with pytest.raises(DecisionStepAllocatorStateError):
            allocator.allocate(0)

    def test_finish_without_active_episode_fails(self):
        allocator = EpisodeDecisionStepAllocator()
        with pytest.raises(DecisionStepAllocatorStateError):
            allocator.finish_episode()

    def test_abort_without_active_episode_fails(self):
        allocator = EpisodeDecisionStepAllocator()
        with pytest.raises(DecisionStepAllocatorStateError):
            allocator.abort_episode()


class TestDoubleStartAndFinish:
    def test_double_start_fails(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        with pytest.raises(DecisionStepAllocatorStateError):
            allocator.start_episode()

    def test_double_finish_fails(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        allocator.finish_episode()
        with pytest.raises(DecisionStepAllocatorStateError):
            allocator.finish_episode()


class TestBoolAndNegativeRejected:
    def test_start_step_bool_rejected(self):
        allocator = EpisodeDecisionStepAllocator()
        with pytest.raises(TypeError):
            allocator.start_episode(start_step=True)

    def test_start_step_negative_rejected(self):
        allocator = EpisodeDecisionStepAllocator()
        with pytest.raises(ValueError):
            allocator.start_episode(start_step=-1)

    def test_robot_id_bool_rejected(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        with pytest.raises(TypeError):
            allocator.allocate(True)

    def test_robot_id_negative_rejected(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        with pytest.raises(ValueError):
            allocator.allocate(-1)

    def test_allocate_many_robot_id_bool_rejected(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        with pytest.raises(TypeError):
            allocator.allocate_many((True, 2))

    def test_allocate_many_robot_id_negative_rejected(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        with pytest.raises(ValueError):
            allocator.allocate_many((-1, 2))


class TestSecondEpisodeRestartsFromRequestedStep:
    def test_second_episode_restarts_from_requested_start_step(self):
        allocator = EpisodeDecisionStepAllocator()
        allocator.start_episode()
        allocator.allocate(0)
        allocator.allocate(1)
        allocator.finish_episode()

        allocator.start_episode(start_step=50)
        assert allocator.next_step == 50
        assert allocator.allocate(0) == 50


class TestNoClockOrUuidDependency:
    def test_source_imports_nothing_time_or_uuid_related(self):
        tree = ast.parse((LEARNING_DIR / "decision_steps.py").read_text(encoding="utf-8"))
        forbidden_modules = {"time", "datetime", "uuid"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_modules
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_modules

    def test_repeated_runs_are_deterministic(self):
        allocator_a = EpisodeDecisionStepAllocator()
        allocator_a.start_episode()
        steps_a = [allocator_a.allocate(0) for _ in range(5)]

        allocator_b = EpisodeDecisionStepAllocator()
        allocator_b.start_episode()
        steps_b = [allocator_b.allocate(0) for _ in range(5)]

        assert steps_a == steps_b == [0, 1, 2, 3, 4]
