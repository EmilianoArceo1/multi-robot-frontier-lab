"""Transition contract for the learning pipeline.

A :class:`LearningTransition` deliberately excludes GroundTruthSnapshot:
ground truth is exported in a separate block so privileged data never
travels with the training transition itself.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from robotics_interfaces.learning.observations import (
    ActorObservation,
    CriticState,
    GroundTruthSnapshot,
)
from robotics_interfaces.learning.actions import LearningAction
from robotics_interfaces.learning.termination import TerminationReason


@dataclass(frozen=True)
class RewardComponent:
    """One named reward term contribution for one robot at one step."""

    name: str
    raw_value: float
    applied_weight: float
    weighted_value: float

    def __post_init__(self) -> None:
        for field_name in ("raw_value", "applied_weight", "weighted_value"):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value!r}")


@dataclass(frozen=True)
class LearningTransition:
    """One multi-robot decision-step transition.

    Ground truth is intentionally not part of this contract; it is exported
    separately (see TrajectoryExportSpec.include_ground_truth_separately).
    """

    schema_version: str
    episode_id: str
    decision_step: int
    actor_observations: Mapping[int, ActorObservation]
    critic_state: CriticState
    selected_actions: Mapping[int, LearningAction]
    reward_components_by_robot: Mapping[int, tuple[RewardComponent, ...]]
    reward_total_by_robot: Mapping[int, float]
    next_actor_observations: Mapping[int, ActorObservation]
    terminated: bool
    truncated: bool
    termination_reason: TerminationReason

    def __post_init__(self) -> None:
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")
        if isinstance(self.critic_state, GroundTruthSnapshot):
            raise TypeError("LearningTransition.critic_state must not be a GroundTruthSnapshot")
        if not isinstance(self.critic_state, CriticState):
            raise TypeError(
                f"critic_state must be a CriticState, got {type(self.critic_state).__name__}"
            )
        for label, mapping in (
            ("actor_observations", self.actor_observations),
            ("next_actor_observations", self.next_actor_observations),
        ):
            for robot_id, obs in mapping.items():
                if isinstance(obs, GroundTruthSnapshot):
                    raise TypeError(
                        f"{label}[{robot_id}] must not be a GroundTruthSnapshot"
                    )
                if not isinstance(obs, ActorObservation):
                    raise TypeError(
                        f"{label}[{robot_id}] must be an ActorObservation, got "
                        f"{type(obs).__name__}"
                    )
        for robot_id, action in self.selected_actions.items():
            if not isinstance(action, LearningAction):
                raise TypeError(
                    f"selected_actions[{robot_id}] must be a LearningAction, got "
                    f"{type(action).__name__}"
                )
        for robot_id, total in self.reward_total_by_robot.items():
            if not math.isfinite(total):
                raise ValueError(
                    f"reward_total_by_robot[{robot_id}] must be finite, got {total!r}"
                )
        if not isinstance(self.termination_reason, TerminationReason):
            raise TypeError(
                f"termination_reason must be a TerminationReason, got "
                f"{type(self.termination_reason).__name__}"
            )
