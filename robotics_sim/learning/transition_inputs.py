"""Host-side inputs for assembling a LearningTransition.

These are *assembly* inputs, not the transition contract itself: they carry
a robot's chosen action_index (resolved against the real catalog only at
assembly time, never stored as a LearningAction here), per-robot reward
components, and terminal-state flags, all cross-checked against the current
DecisionCaptureBatch's actual robot set.

``TransitionAssemblyInput`` intentionally has no GroundTruthSnapshot field
and rejects one passed as ``critic_state``: ground truth only ever reaches
``InMemoryTrajectoryRecorder.append()`` directly, never through a
transition-assembly input.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  No Qt, numpy, torch, pandas, robotics_sim.app
or engine imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robotics_interfaces.learning.observations import CriticState, GroundTruthSnapshot
from robotics_interfaces.learning.termination import TerminationReason
from robotics_interfaces.learning.transitions import RewardComponent
from robotics_sim.learning.decision_batch import DecisionCaptureBatch


def _require_non_negative_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True)
class RobotActionSelection:
    """A policy's chosen action_index for one robot at one decision step.

    Deliberately does not store a LearningAction: resolving action_index to
    an executable action happens later, through
    ``DecisionCaptureBatch.resolve_action`` -- the real catalog's public
    API, not a value carried here.
    """

    robot_id: int
    action_index: int
    issued_at_step: int

    def __post_init__(self) -> None:
        _require_non_negative_int("robot_id", self.robot_id)
        _require_non_negative_int("action_index", self.action_index)
        _require_non_negative_int("issued_at_step", self.issued_at_step)


@dataclass(frozen=True)
class DecisionSelectionBatch:
    """All robots' action selections for one decision step.

    v0 requires exactly one selection per robot present in the current
    DecisionCaptureBatch (cross-checked in TransitionAssemblyInput, since
    this type alone has no reference to that batch); HOLD is never modeled
    as a LearningAction here -- a RuntimeActorFrame simply omits any robot
    that doesn't need a new decision.
    """

    episode_id: str
    decision_step: int
    selections: tuple[RobotActionSelection, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError(f"episode_id must be a non-empty string, got {self.episode_id!r}")
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")

        selections = tuple(self.selections)
        seen_robot_ids: set[int] = set()
        for i, selection in enumerate(selections):
            if not isinstance(selection, RobotActionSelection):
                raise TypeError(
                    f"selections[{i}] must be a RobotActionSelection, got "
                    f"{type(selection).__name__}"
                )
            if selection.robot_id in seen_robot_ids:
                raise ValueError(f"selections contains duplicate robot_id {selection.robot_id}")
            seen_robot_ids.add(selection.robot_id)
            if selection.issued_at_step != self.decision_step:
                raise ValueError(
                    f"selections[{i}].issued_at_step={selection.issued_at_step} does not match "
                    f"decision_step={self.decision_step}"
                )
        object.__setattr__(self, "selections", selections)


@dataclass(frozen=True)
class RobotRewardOutcome:
    """One robot's reward components for one decision step.

    Never receives reward_total directly: the total is always
    ``sum(component.weighted_value for component in components)``, computed
    by the assembler, not passed in here.
    """

    robot_id: int
    components: tuple[RewardComponent, ...]

    def __post_init__(self) -> None:
        _require_non_negative_int("robot_id", self.robot_id)

        components = tuple(self.components)
        seen_names: set[str] = set()
        for i, component in enumerate(components):
            if not isinstance(component, RewardComponent):
                raise TypeError(
                    f"components[{i}] must be a RewardComponent, got {type(component).__name__}"
                )
            if component.name in seen_names:
                raise ValueError(
                    f"components contains duplicate reward component name "
                    f"{component.name!r} for robot {self.robot_id}"
                )
            seen_names.add(component.name)
            for field_name in ("raw_value", "applied_weight", "weighted_value"):
                value = getattr(component, field_name)
                if isinstance(value, bool):
                    raise TypeError(
                        f"components[{i}].{field_name} must not be bool for robot "
                        f"{self.robot_id}"
                    )
                if not math.isfinite(value):
                    raise ValueError(
                        f"components[{i}].{field_name} must be finite, got {value!r}"
                    )
        object.__setattr__(self, "components", components)


@dataclass(frozen=True)
class TransitionOutcomeBatch:
    """Terminal-state flags and per-robot reward outcomes for one decision
    step.  Carries no next observation, no ground truth, no critic state,
    and no arbitrary metadata."""

    episode_id: str
    decision_step: int
    rewards: tuple[RobotRewardOutcome, ...]
    terminated: bool
    truncated: bool
    termination_reason: TerminationReason

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError(f"episode_id must be a non-empty string, got {self.episode_id!r}")
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")

        rewards = tuple(self.rewards)
        seen_robot_ids: set[int] = set()
        for i, reward in enumerate(rewards):
            if not isinstance(reward, RobotRewardOutcome):
                raise TypeError(
                    f"rewards[{i}] must be a RobotRewardOutcome, got {type(reward).__name__}"
                )
            if reward.robot_id in seen_robot_ids:
                raise ValueError(f"rewards contains duplicate robot_id {reward.robot_id}")
            seen_robot_ids.add(reward.robot_id)
        object.__setattr__(self, "rewards", rewards)

        if not isinstance(self.terminated, bool):
            raise TypeError(f"terminated must be bool, got {type(self.terminated).__name__}")
        if not isinstance(self.truncated, bool):
            raise TypeError(f"truncated must be bool, got {type(self.truncated).__name__}")
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated must not both be True")
        if not isinstance(self.termination_reason, TerminationReason):
            raise TypeError(
                f"termination_reason must be a TerminationReason, got "
                f"{type(self.termination_reason).__name__}"
            )

        is_terminal = self.terminated or self.truncated
        if not is_terminal and self.termination_reason is not TerminationReason.RUNNING:
            raise ValueError(
                "termination_reason must be RUNNING when terminated=False and truncated=False"
            )
        if is_terminal and self.termination_reason is TerminationReason.RUNNING:
            raise ValueError(
                "termination_reason must not be RUNNING when terminated or truncated is True"
            )


@dataclass(frozen=True)
class TransitionAssemblyInput:
    """Everything needed to assemble one LearningTransition from a decision
    step already captured by DecisionCaptureBatch.

    No GroundTruthSnapshot field.  ``critic_state`` is required: the current
    ObservationSpec contract requires LearningTransition.critic_state to be
    a real CriticState, so this input never advertises an optionality that
    downstream construction cannot honor.
    """

    current_decision: DecisionCaptureBatch
    selections: DecisionSelectionBatch
    outcome: TransitionOutcomeBatch
    next_decision: DecisionCaptureBatch | None
    critic_state: CriticState

    def __post_init__(self) -> None:
        if not isinstance(self.current_decision, DecisionCaptureBatch):
            raise TypeError(
                f"current_decision must be a DecisionCaptureBatch, got "
                f"{type(self.current_decision).__name__}"
            )
        if not isinstance(self.selections, DecisionSelectionBatch):
            raise TypeError(
                f"selections must be a DecisionSelectionBatch, got "
                f"{type(self.selections).__name__}"
            )
        if not isinstance(self.outcome, TransitionOutcomeBatch):
            raise TypeError(
                f"outcome must be a TransitionOutcomeBatch, got {type(self.outcome).__name__}"
            )
        if self.next_decision is not None and not isinstance(
            self.next_decision, DecisionCaptureBatch
        ):
            raise TypeError(
                f"next_decision must be a DecisionCaptureBatch or None, got "
                f"{type(self.next_decision).__name__}"
            )
        if isinstance(self.critic_state, GroundTruthSnapshot):
            raise TypeError("critic_state must not be a GroundTruthSnapshot")
        if not isinstance(self.critic_state, CriticState):
            raise TypeError(
                f"critic_state must be a CriticState, got "
                f"{type(self.critic_state).__name__}"
            )

        current_batch = self.current_decision.actor_batch
        current_episode_id = current_batch.episode_id
        current_decision_step = current_batch.decision_step
        current_time_s = current_batch.time_s

        if self.selections.episode_id != current_episode_id:
            raise ValueError(
                f"selections.episode_id={self.selections.episode_id!r} does not match "
                f"current_decision.episode_id={current_episode_id!r}"
            )
        if self.outcome.episode_id != current_episode_id:
            raise ValueError(
                f"outcome.episode_id={self.outcome.episode_id!r} does not match "
                f"current_decision.episode_id={current_episode_id!r}"
            )
        if self.selections.decision_step != current_decision_step:
            raise ValueError(
                f"selections.decision_step={self.selections.decision_step} does not match "
                f"current_decision.decision_step={current_decision_step}"
            )
        if self.outcome.decision_step != current_decision_step:
            raise ValueError(
                f"outcome.decision_step={self.outcome.decision_step} does not match "
                f"current_decision.decision_step={current_decision_step}"
            )

        current_robot_ids = tuple(o.robot_id for o in current_batch.observations)
        current_robot_id_set = set(current_robot_ids)

        selection_robot_ids = tuple(s.robot_id for s in self.selections.selections)
        if set(selection_robot_ids) != current_robot_id_set or len(selection_robot_ids) != len(
            current_robot_ids
        ):
            raise ValueError(
                f"selections must cover exactly the robots in current_decision "
                f"{current_robot_ids}, got {selection_robot_ids}"
            )

        reward_robot_ids = tuple(r.robot_id for r in self.outcome.rewards)
        if set(reward_robot_ids) != current_robot_id_set or len(reward_robot_ids) != len(
            current_robot_ids
        ):
            raise ValueError(
                f"outcome.rewards must cover exactly the robots in current_decision "
                f"{current_robot_ids}, got {reward_robot_ids}"
            )

        is_terminal = self.outcome.terminated or self.outcome.truncated
        if is_terminal and self.next_decision is not None:
            raise ValueError(
                "a terminal or truncated transition must not carry a next_decision"
            )
        if not is_terminal and self.next_decision is None:
            raise ValueError("a non-terminal transition requires a next_decision")

        if self.next_decision is not None:
            next_batch = self.next_decision.actor_batch
            if next_batch.episode_id != current_episode_id:
                raise ValueError(
                    f"next_decision.episode_id={next_batch.episode_id!r} does not match "
                    f"current_decision.episode_id={current_episode_id!r}"
                )
            if next_batch.decision_step <= current_decision_step:
                raise ValueError(
                    f"next_decision.decision_step={next_batch.decision_step} must be strictly "
                    f"greater than current_decision.decision_step={current_decision_step}"
                )
            if next_batch.time_s < current_time_s:
                raise ValueError(
                    f"next_decision.time_s={next_batch.time_s} must be >= "
                    f"current_decision.time_s={current_time_s}"
                )
            next_robot_ids = tuple(o.robot_id for o in next_batch.observations)
            if next_robot_ids != current_robot_ids:
                raise ValueError(
                    f"next_decision must have the same robot set and order as "
                    f"current_decision in v0: expected {current_robot_ids}, got "
                    f"{next_robot_ids}"
                )
