"""In-memory episode session: a small state machine composing
LearningTransitionAssembler and InMemoryTrajectoryRecorder to drive one
episode end to end, entirely in memory.

No filesystem, no NPZ/Parquet/JSON/ZIP, no GUI, no PPO, no runtime
integration, no reward computation, no candidate generation, no physical
action execution.  This class only sequences already-assembled pieces; it
does not capture, extract features, export, or train.

A decision step here is a policy decision (current ActorObservation ->
selected ExplorationCandidate -> accumulated reward -> next
ActorObservation), never a graphics frame, a controller tick, an A*
iteration, or an observed cell.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  No Qt, numpy, torch, pandas, robotics_sim.app
or engine imports.
"""

from __future__ import annotations

import enum

from robotics_interfaces.learning.export import EpisodeFireMetrics, EpisodeMetadata
from robotics_interfaces.learning.observations import CriticState, GroundTruthSnapshot
from robotics_interfaces.learning.transitions import LearningTransition
from robotics_sim.learning.decision_batch import DecisionCaptureBatch
from robotics_sim.learning.recorder import EpisodeRecord, InMemoryTrajectoryRecorder
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import (
    DecisionSelectionBatch,
    TransitionAssemblyInput,
    TransitionOutcomeBatch,
)


class SessionState(enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"
    TERMINATED = "terminated"


class SessionStateError(RuntimeError):
    """Operation not valid in the session's current state."""


class InMemoryLearningEpisodeSession:
    """Drives one episode: IDLE -> ACTIVE -> (repeat) -> TERMINATED -> IDLE.

    Composes a LearningTransitionAssembler and an InMemoryTrajectoryRecorder
    without duplicating either one's logic.
    """

    def __init__(
        self,
        transition_assembler: LearningTransitionAssembler,
        recorder: InMemoryTrajectoryRecorder,
    ) -> None:
        if not isinstance(transition_assembler, LearningTransitionAssembler):
            raise TypeError(
                f"transition_assembler must be a LearningTransitionAssembler, got "
                f"{type(transition_assembler).__name__}"
            )
        if not isinstance(recorder, InMemoryTrajectoryRecorder):
            raise TypeError(
                f"recorder must be an InMemoryTrajectoryRecorder, got "
                f"{type(recorder).__name__}"
            )
        self._transition_assembler = transition_assembler
        self._recorder = recorder
        self._state = SessionState.IDLE
        self._current_decision: DecisionCaptureBatch | None = None

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state is SessionState.ACTIVE

    @property
    def current_decision(self) -> DecisionCaptureBatch | None:
        return self._current_decision

    def start_episode(
        self, metadata: EpisodeMetadata, initial_decision: DecisionCaptureBatch
    ) -> None:
        if self._state is not SessionState.IDLE:
            raise SessionStateError(
                f"start_episode() requires IDLE, session is {self._state.value}"
            )
        if not isinstance(metadata, EpisodeMetadata):
            raise TypeError(f"metadata must be an EpisodeMetadata, got {type(metadata).__name__}")
        if not isinstance(initial_decision, DecisionCaptureBatch):
            raise TypeError(
                f"initial_decision must be a DecisionCaptureBatch, got "
                f"{type(initial_decision).__name__}"
            )
        if metadata.episode_id != initial_decision.actor_batch.episode_id:
            raise ValueError(
                f"metadata.episode_id={metadata.episode_id!r} does not match "
                f"initial_decision.episode_id={initial_decision.actor_batch.episode_id!r}"
            )

        self._recorder.start_episode(metadata)
        self._current_decision = initial_decision
        self._state = SessionState.ACTIVE

    def complete_current_decision(
        self,
        selections: DecisionSelectionBatch,
        outcome: TransitionOutcomeBatch,
        next_decision: DecisionCaptureBatch | None,
        critic_state: CriticState,
        ground_truth: GroundTruthSnapshot | None = None,
    ) -> LearningTransition:
        if self._state is not SessionState.ACTIVE:
            raise SessionStateError(
                f"complete_current_decision() requires ACTIVE, session is {self._state.value}"
            )

        build_input = TransitionAssemblyInput(
            current_decision=self._current_decision,
            selections=selections,
            outcome=outcome,
            next_decision=next_decision,
            critic_state=critic_state,
        )
        transition = self._transition_assembler.build(build_input)
        # ground_truth reaches the recorder directly -- it is never a field
        # of TransitionAssemblyInput or LearningTransition.
        self._recorder.append(transition, ground_truth=ground_truth)

        if outcome.terminated or outcome.truncated:
            self._current_decision = None
            self._state = SessionState.TERMINATED
        else:
            self._current_decision = next_decision

        return transition

    def set_fire_metrics(self, metrics: EpisodeFireMetrics) -> None:
        self._recorder.set_fire_metrics(metrics)

    def finish_episode(self) -> EpisodeRecord:
        if self._state is not SessionState.TERMINATED:
            raise SessionStateError(
                f"finish_episode() requires TERMINATED, session is {self._state.value}"
            )
        record = self._recorder.finish_episode()
        self._current_decision = None
        self._state = SessionState.IDLE
        return record

    def abort_episode(self) -> None:
        if self._state not in (SessionState.ACTIVE, SessionState.TERMINATED):
            raise SessionStateError(
                f"abort_episode() requires ACTIVE or TERMINATED, session is {self._state.value}"
            )
        self._recorder.abort_episode()
        self._current_decision = None
        self._state = SessionState.IDLE
