"""Episode-global decision_step allocation.

Invariant this module exists to enforce:

- decision_step is global within one episode, not a per-robot counter --
  there is one shared sequence for the whole episode;
- each trainable decision receives a unique value from that sequence;
- steps are assigned when an action is *opened* (this allocator's job),
  never when it closes;
- because the real multi-robot runtime is asynchronous, the order in which
  decisions finish can differ from the order in which their steps were
  assigned -- a step-8 decision may close before a step-7 decision does;
- exported artifacts are ordered by decision_step, not by arrival order
  (see robotics_sim.learning.recorder.InMemoryTrajectoryRecorder).

Pure and in-memory: no clock, no UUID, no per-robot counters -- one plain
integer counter for the whole episode.
"""

from __future__ import annotations

from typing import Mapping


class DecisionStepAllocatorError(RuntimeError):
    """Base class for EpisodeDecisionStepAllocator errors."""


class DecisionStepAllocatorStateError(DecisionStepAllocatorError):
    """Operation not valid in the allocator's current state."""


class EpisodeDecisionStepAllocator:
    """Assigns unique, episode-global decision_step values.

    One shared counter for the whole episode -- robot_id never influences
    which value is assigned; it only shapes the return value of
    allocate_many() (a per-robot mapping into that same global sequence).
    The same robot can receive several, non-adjacent global steps over the
    course of an episode; nothing here tracks "this robot's step count".
    """

    def __init__(self) -> None:
        self._next_step: int | None = None

    @property
    def is_active(self) -> bool:
        return self._next_step is not None

    @property
    def next_step(self) -> int:
        if self._next_step is None:
            raise DecisionStepAllocatorStateError(
                "no active episode: call start_episode() first"
            )
        return self._next_step

    def start_episode(self, start_step: int = 0) -> None:
        if self._next_step is not None:
            raise DecisionStepAllocatorStateError(
                "an episode is already active; call finish_episode() or abort_episode() "
                "first"
            )
        if isinstance(start_step, bool) or not isinstance(start_step, int):
            raise TypeError(f"start_step must be an int, got {type(start_step).__name__}")
        if start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {start_step}")
        self._next_step = start_step

    def allocate(self, robot_id: int) -> int:
        if self._next_step is None:
            raise DecisionStepAllocatorStateError("allocate() called with no active episode")
        if isinstance(robot_id, bool) or not isinstance(robot_id, int):
            raise TypeError(f"robot_id must be an int, got {type(robot_id).__name__}")
        if robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {robot_id}")

        step = self._next_step
        self._next_step += 1
        return step

    def allocate_many(self, robot_ids: tuple[int, ...]) -> Mapping[int, int]:
        if self._next_step is None:
            raise DecisionStepAllocatorStateError(
                "allocate_many() called with no active episode"
            )
        robot_ids = tuple(robot_ids)
        seen: set[int] = set()
        for robot_id in robot_ids:
            if isinstance(robot_id, bool) or not isinstance(robot_id, int):
                raise TypeError(f"robot_id must be an int, got {type(robot_id).__name__}")
            if robot_id < 0:
                raise ValueError(f"robot_id must be non-negative, got {robot_id}")
            if robot_id in seen:
                raise ValueError(f"robot_ids contains duplicate robot_id {robot_id}")
            seen.add(robot_id)

        # Validated up front (above) so a rejected call never partially
        # consumes the global sequence.
        assignment: dict[int, int] = {}
        for robot_id in robot_ids:
            assignment[robot_id] = self.allocate(robot_id)
        return assignment

    def finish_episode(self) -> None:
        if self._next_step is None:
            raise DecisionStepAllocatorStateError(
                "finish_episode() called with no active episode"
            )
        self._next_step = None

    def abort_episode(self) -> None:
        if self._next_step is None:
            raise DecisionStepAllocatorStateError(
                "abort_episode() called with no active episode"
            )
        self._next_step = None
