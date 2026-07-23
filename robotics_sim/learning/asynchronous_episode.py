"""In-memory asynchronous episode session: several robots' learning
decisions may be open (pending) at once, and each robot completes and
reopens its own decision independently of the others.

Unlike InMemoryLearningEpisodeSession (one shared "current decision" for
the whole episode), this session tracks one pending decision per robot_id,
together with the CriticState (and optional GroundTruthSnapshot) that was
true *when that decision was opened*.  A robot's decision closing never
requires, waits for, or is ordered against another robot's decision
closing -- e.g. a decision opened at step 1 may close before one opened at
step 0.  Ordering the resulting transitions is entirely
InMemoryTrajectoryRecorder.finish_episode()'s job (sorted ascending by
decision_step); this session never sorts or delays an append.

CriticState and GroundTruthSnapshot are captured at register_opened_decisions()
time, not at complete_robot_decision() time: the transition assembled when a
robot's decision *closes* must reflect the world as it was when that
decision's action was *chosen*, never information that only became
available afterwards.  complete_robot_decision() therefore no longer takes
a critic_state or ground_truth for the decision it is closing -- only for
the *next* decision it may open, which is captured and stored the same way.

The session also tracks every decision_step it has ever registered
(pending or already completed) for the life of one episode, and rejects
registering or reopening a step that collides with any of them -- pending
robot A and pending robot B can never end up sharing a decision_step, and a
robot cannot reopen a step that some transition has already used, even
though EpisodeDecisionStepAllocator (which actually allocates those values)
lives entirely outside this class.

No filesystem, no NPZ/Parquet/JSON/ZIP, no GUI, no PPO, no runtime
integration, no reward computation, no candidate generation, no plugin or
candidate-provider calls, and no CriticState/GroundTruthSnapshot
construction -- both are only ever accepted as already-built values from the
caller.  This class only sequences already-assembled pieces per robot; it
does not open decisions (RuntimeLearningDecisionOpener's job) or allocate
decision_step values (EpisodeDecisionStepAllocator's job).

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  No Qt, numpy, torch, pandas, robotics_sim.app
or engine imports.  GroundTruthSnapshot is imported only to type the
ground-truth values captured alongside a pending decision and forwarded,
unopened, straight to InMemoryTrajectoryRecorder.append() -- it never
becomes part of a LearningTransition or TransitionAssemblyInput built here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from robotics_interfaces.learning.export import EpisodeMetadata, EpisodeFireMetrics
from robotics_interfaces.learning.observations import CriticState, GroundTruthSnapshot
from robotics_interfaces.learning.transitions import LearningTransition
from robotics_sim.learning.recorder import EpisodeRecord, InMemoryTrajectoryRecorder
from robotics_sim.learning.runtime_decision_opening import (
    OpenedLearningDecision,
    OpenedRobotLearningDecision,
    UnresolvedCoordinationDecision,
)
from robotics_sim.learning.transition_assembler import LearningTransitionAssembler
from robotics_sim.learning.transition_inputs import TransitionAssemblyInput, TransitionOutcomeBatch


class AsynchronousEpisodeSessionError(RuntimeError):
    """Base class for InMemoryAsynchronousLearningEpisodeSession errors."""


class AsynchronousEpisodeSessionStateError(AsynchronousEpisodeSessionError):
    """Operation not valid in the session's current state (no active
    episode, an episode already active, or pending decisions remain at
    finish)."""


class PendingRobotDecisionError(AsynchronousEpisodeSessionError):
    """A decision-registration invariant was violated: a robot already had
    a pending decision, a robot had no pending decision to complete, a
    next_decision belonged to a different robot than the one being
    completed, or a decision_step collided with one already pending or
    completed this episode."""


@dataclass(frozen=True)
class _PendingRobotLearningDecision:
    """Internal, module-private: one robot's still-open decision together
    with the CriticState (and optional GroundTruthSnapshot) captured at the
    moment it was opened.

    Not exported from robotics_sim.learning -- callers only ever see the
    OpenedRobotLearningDecision they passed in and the LearningTransition
    that eventually comes back out.
    """

    opened: OpenedRobotLearningDecision
    critic_state: CriticState
    ground_truth: GroundTruthSnapshot | None


class InMemoryAsynchronousLearningEpisodeSession:
    """Tracks one pending decision per robot_id and assembles/records a
    LearningTransition whenever a robot's decision completes, independently
    of every other robot's pending decision.

    Composes a LearningTransitionAssembler and an InMemoryTrajectoryRecorder
    without duplicating either one's logic, and reuses OpenedLearningDecision/
    OpenedRobotLearningDecision as produced by RuntimeLearningDecisionOpener
    without redefining any of those contracts.
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
        self._metadata: EpisodeMetadata | None = None
        # Keyed by robot_id; insertion order is preserved by dict semantics,
        # and re-assigning an existing key (the replace-on-complete case)
        # keeps that key's original position -- this is what makes
        # pending_robot_ids order-preserving without any extra bookkeeping.
        self._pending: dict[int, _PendingRobotLearningDecision] = {}
        # Every decision_step registered this episode, pending or already
        # completed -- never removed until the episode ends, so a step can
        # never be reused by any robot for the life of one episode.
        self._seen_decision_steps: set[int] = set()

    @property
    def is_active(self) -> bool:
        return self._metadata is not None

    @property
    def pending_robot_ids(self) -> tuple[int, ...]:
        return tuple(self._pending.keys())

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def has_pending(self, robot_id: int) -> bool:
        return robot_id in self._pending

    @property
    def episode_id(self) -> str | None:
        return None if self._metadata is None else self._metadata.episode_id

    def start_episode(self, metadata: EpisodeMetadata) -> None:
        if self._metadata is not None:
            raise AsynchronousEpisodeSessionStateError(
                f"an episode is already active (episode_id="
                f"{self._metadata.episode_id!r}); call finish_episode() or "
                f"abort_episode() first"
            )
        if not isinstance(metadata, EpisodeMetadata):
            raise TypeError(f"metadata must be an EpisodeMetadata, got {type(metadata).__name__}")

        self._recorder.start_episode(metadata)
        self._metadata = metadata
        self._pending = {}
        self._seen_decision_steps = set()

    def register_opened_decisions(
        self,
        opened: OpenedLearningDecision,
        critic_states_by_robot: Mapping[int, CriticState],
        ground_truth_by_robot: Mapping[int, GroundTruthSnapshot] | None = None,
    ) -> tuple[UnresolvedCoordinationDecision, ...]:
        if self._metadata is None:
            raise AsynchronousEpisodeSessionStateError(
                "register_opened_decisions() called with no active episode"
            )
        if not isinstance(opened, OpenedLearningDecision):
            raise TypeError(
                f"opened must be an OpenedLearningDecision, got {type(opened).__name__}"
            )
        if opened.episode_id != self._metadata.episode_id:
            raise ValueError(
                f"opened.episode_id={opened.episode_id!r} does not match the active "
                f"episode {self._metadata.episode_id!r}"
            )

        assigned_robot_id_set = set(opened.assigned_robot_ids)

        critic_states_by_robot = dict(critic_states_by_robot)
        if set(critic_states_by_robot) != assigned_robot_id_set:
            raise ValueError(
                f"critic_states_by_robot keys {sorted(critic_states_by_robot)} do not match "
                f"the ASSIGNED robots {sorted(assigned_robot_id_set)}"
            )
        for robot_id, critic_state in critic_states_by_robot.items():
            if not isinstance(critic_state, CriticState):
                raise TypeError(
                    f"critic_states_by_robot[{robot_id}] must be a CriticState, got "
                    f"{type(critic_state).__name__}"
                )

        ground_truth_by_robot = (
            {} if ground_truth_by_robot is None else dict(ground_truth_by_robot)
        )
        extra_ground_truth_robots = set(ground_truth_by_robot) - assigned_robot_id_set
        if extra_ground_truth_robots:
            raise ValueError(
                f"ground_truth_by_robot contains robot id(s) {sorted(extra_ground_truth_robots)} "
                f"that are not ASSIGNED in opened"
            )
        for robot_id, ground_truth in ground_truth_by_robot.items():
            if not isinstance(ground_truth, GroundTruthSnapshot):
                raise TypeError(
                    f"ground_truth_by_robot[{robot_id}] must be a GroundTruthSnapshot, got "
                    f"{type(ground_truth).__name__}"
                )

        # Validate the whole batch before mutating any state, so a rejected
        # call never partially registers the rest of opened.assigned.
        for item in opened.assigned:
            if item.robot_id in self._pending:
                raise PendingRobotDecisionError(
                    f"robot {item.robot_id} already has a pending decision"
                )
            if item.decision_step in self._seen_decision_steps:
                raise PendingRobotDecisionError(
                    f"decision_step {item.decision_step} was already used by a pending or "
                    f"completed decision this episode"
                )

        for item in opened.assigned:
            self._pending[item.robot_id] = _PendingRobotLearningDecision(
                opened=item,
                critic_state=critic_states_by_robot[item.robot_id],
                ground_truth=ground_truth_by_robot.get(item.robot_id),
            )
            self._seen_decision_steps.add(item.decision_step)

        return opened.unresolved

    def complete_robot_decision(
        self,
        robot_id: int,
        outcome: TransitionOutcomeBatch,
        next_decision: OpenedRobotLearningDecision | None = None,
        next_critic_state: CriticState | None = None,
        next_ground_truth: GroundTruthSnapshot | None = None,
    ) -> LearningTransition:
        if self._metadata is None:
            raise AsynchronousEpisodeSessionStateError(
                "complete_robot_decision() called with no active episode"
            )
        if isinstance(robot_id, bool) or not isinstance(robot_id, int):
            raise TypeError(f"robot_id must be an int, got {type(robot_id).__name__}")
        if robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {robot_id}")

        pending = self._pending.get(robot_id)
        if pending is None:
            raise PendingRobotDecisionError(f"robot {robot_id} has no pending decision")

        if not isinstance(outcome, TransitionOutcomeBatch):
            raise TypeError(
                f"outcome must be a TransitionOutcomeBatch, got {type(outcome).__name__}"
            )

        is_terminal = outcome.terminated or outcome.truncated

        if is_terminal:
            if next_decision is not None:
                raise ValueError(
                    "a terminated or truncated outcome must not carry a next_decision"
                )
            if next_critic_state is not None:
                raise ValueError(
                    "a terminated or truncated outcome must not carry a next_critic_state"
                )
            if next_ground_truth is not None:
                raise ValueError(
                    "a terminated or truncated outcome must not carry a next_ground_truth"
                )
            next_decision_capture = None
            next_pending_entry = None
        else:
            if next_decision is None:
                raise ValueError("a non-terminal outcome requires a next_decision")
            if not isinstance(next_decision, OpenedRobotLearningDecision):
                raise TypeError(
                    f"next_decision must be an OpenedRobotLearningDecision, got "
                    f"{type(next_decision).__name__}"
                )
            if next_decision.robot_id != robot_id:
                raise PendingRobotDecisionError(
                    f"next_decision belongs to robot {next_decision.robot_id}, but robot "
                    f"{robot_id} is the one being completed"
                )
            if next_critic_state is None:
                raise ValueError("a non-terminal outcome requires a next_critic_state")
            if not isinstance(next_critic_state, CriticState):
                raise TypeError(
                    f"next_critic_state must be a CriticState, got "
                    f"{type(next_critic_state).__name__}"
                )
            if next_ground_truth is not None and not isinstance(
                next_ground_truth, GroundTruthSnapshot
            ):
                raise TypeError(
                    f"next_ground_truth must be a GroundTruthSnapshot or None, got "
                    f"{type(next_ground_truth).__name__}"
                )
            if next_decision.decision_step in self._seen_decision_steps:
                raise PendingRobotDecisionError(
                    f"decision_step {next_decision.decision_step} was already used by a "
                    f"pending or completed decision this episode"
                )
            next_decision_capture = next_decision.decision_capture
            next_pending_entry = _PendingRobotLearningDecision(
                opened=next_decision,
                critic_state=next_critic_state,
                ground_truth=next_ground_truth,
            )

        # Everything else -- outcome.episode_id/decision_step matching the
        # pending decision, outcome.rewards covering exactly this robot,
        # next_decision's episode_id/decision_step/time_s/robot-set -- is
        # already enforced by TransitionAssemblyInput; re-checking it here
        # would duplicate that contract.  critic_state always comes from the
        # pending entry (captured when this decision was opened), never
        # from the caller at completion time.
        build_input = TransitionAssemblyInput(
            current_decision=pending.opened.decision_capture,
            selections=pending.opened.selections,
            outcome=outcome,
            next_decision=next_decision_capture,
            critic_state=pending.critic_state,
        )
        transition = self._transition_assembler.build(build_input)
        # The *current* decision's ground truth reaches the recorder here --
        # next_ground_truth (if any) is only stored for the future
        # transition that will close next_decision, never this one.
        self._recorder.append(transition, ground_truth=pending.ground_truth)

        # Only mutate session state after a successful append, so a failed
        # validation or a failed recorder.append() never changes pending or
        # _seen_decision_steps.
        if is_terminal:
            del self._pending[robot_id]
        else:
            self._pending[robot_id] = next_pending_entry
            self._seen_decision_steps.add(next_decision.decision_step)

        return transition

    def set_fire_metrics(self, metrics: EpisodeFireMetrics) -> None:
        if self._metadata is None:
            raise AsynchronousEpisodeSessionStateError(
                "set_fire_metrics() called with no active episode"
            )
        self._recorder.set_fire_metrics(metrics)

    def finish_episode(self) -> EpisodeRecord:
        if self._metadata is None:
            raise AsynchronousEpisodeSessionStateError(
                "finish_episode() called with no active episode"
            )
        if self._pending:
            raise AsynchronousEpisodeSessionStateError(
                f"finish_episode() called with {len(self._pending)} pending decision(s) "
                f"for robot(s) {tuple(self._pending)}"
            )

        record = self._recorder.finish_episode()
        self._metadata = None
        self._pending = {}
        self._seen_decision_steps = set()
        return record

    def abort_episode(self) -> None:
        if self._metadata is None:
            raise AsynchronousEpisodeSessionStateError(
                "abort_episode() called with no active episode"
            )
        self._recorder.abort_episode()
        self._metadata = None
        self._pending = {}
        self._seen_decision_steps = set()
