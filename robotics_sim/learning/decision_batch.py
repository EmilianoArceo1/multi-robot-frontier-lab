"""Pure host-side composition: RuntimeActorFrame -> DecisionCaptureBatch.

``DecisionCaptureBatch`` pairs the actor's observations with the executable
action catalog for the same decision step, and cross-checks that the two
stay aligned (same episode/step/time, same robot order, same candidate_ids
in the same order, ``action_mask`` matching ``ActionOption.enabled``).

``DecisionCaptureAssembler`` composes ``ActorObservationBatchAssembler`` and
``ActionCatalogAssembler`` without duplicating any feature-extraction logic.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  No Qt, numpy, torch, pandas, robotics_sim.app
or engine imports.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotics_interfaces.learning import LearningAction
from robotics_interfaces.learning.observations import ActorObservation
from robotics_sim.learning.action_catalog import (
    ActionCatalogAssembler,
    ActionCatalogBatch,
    RobotActionCatalog,
)
from robotics_sim.learning.capture_inputs import RuntimeActorFrame
from robotics_sim.learning.observation_batch import ActorObservationBatch, ActorObservationBatchAssembler


@dataclass(frozen=True)
class DecisionCaptureBatch:
    """One decision step's actor observations plus its executable action
    catalog, cross-checked for alignment."""

    actor_batch: ActorObservationBatch
    action_catalog_batch: ActionCatalogBatch

    def __post_init__(self) -> None:
        if not isinstance(self.actor_batch, ActorObservationBatch):
            raise TypeError(
                f"actor_batch must be an ActorObservationBatch, got "
                f"{type(self.actor_batch).__name__}"
            )
        if not isinstance(self.action_catalog_batch, ActionCatalogBatch):
            raise TypeError(
                f"action_catalog_batch must be an ActionCatalogBatch, got "
                f"{type(self.action_catalog_batch).__name__}"
            )

        if self.actor_batch.episode_id != self.action_catalog_batch.episode_id:
            raise ValueError(
                f"actor_batch.episode_id={self.actor_batch.episode_id!r} does not match "
                f"action_catalog_batch.episode_id={self.action_catalog_batch.episode_id!r}"
            )
        if self.actor_batch.decision_step != self.action_catalog_batch.decision_step:
            raise ValueError(
                f"actor_batch.decision_step={self.actor_batch.decision_step} does not match "
                f"action_catalog_batch.decision_step={self.action_catalog_batch.decision_step}"
            )
        if self.actor_batch.time_s != self.action_catalog_batch.time_s:
            raise ValueError(
                f"actor_batch.time_s={self.actor_batch.time_s} does not match "
                f"action_catalog_batch.time_s={self.action_catalog_batch.time_s}"
            )

        actor_robot_ids = tuple(o.robot_id for o in self.actor_batch.observations)
        catalog_robot_ids = tuple(c.robot_id for c in self.action_catalog_batch.catalogs)
        if actor_robot_ids != catalog_robot_ids:
            raise ValueError(
                f"actor_batch robot order {actor_robot_ids} does not match "
                f"action_catalog_batch robot order {catalog_robot_ids}"
            )

        for observation, catalog in zip(
            self.actor_batch.observations, self.action_catalog_batch.catalogs
        ):
            if len(observation.candidate_ids) != len(catalog.options):
                raise ValueError(
                    f"robot {observation.robot_id}: observation has "
                    f"{len(observation.candidate_ids)} candidate_ids but catalog has "
                    f"{len(catalog.options)} options"
                )
            for i, (candidate_id, option) in enumerate(
                zip(observation.candidate_ids, catalog.options)
            ):
                if candidate_id != option.candidate_id:
                    raise ValueError(
                        f"robot {observation.robot_id}: candidate_id mismatch at position "
                        f"{i}: observation has {candidate_id!r}, catalog has "
                        f"{option.candidate_id!r}"
                    )
            for i, (enabled_flag, option) in enumerate(
                zip(observation.action_mask, catalog.options)
            ):
                if enabled_flag != option.enabled:
                    raise ValueError(
                        f"robot {observation.robot_id}: action_mask[{i}]={enabled_flag} does "
                        f"not match option.enabled={option.enabled}"
                    )

    def get_observation(self, robot_id: int) -> ActorObservation:
        return self.actor_batch.get_for_robot(robot_id)

    def get_action_catalog(self, robot_id: int) -> RobotActionCatalog:
        return self.action_catalog_batch.get_for_robot(robot_id)

    def resolve_action(
        self, robot_id: int, action_index: int, issued_at_step: int
    ) -> LearningAction:
        """Resolve a policy-chosen ``action_index`` into an executable
        ``LearningAction``, delegating to ``ActionOption.to_learning_action``."""

        catalog = self.get_action_catalog(robot_id)
        option = catalog.get_by_action_index(action_index)
        return option.to_learning_action(issued_at_step)


class DecisionCaptureAssembler:
    """Composes ActorObservationBatchAssembler and ActionCatalogAssembler.

    RuntimeActorFrame -> ActorObservationBatchAssembler ->
    ActionCatalogAssembler -> DecisionCaptureBatch.  Never duplicates
    feature extraction: both assemblers are reused as-is.
    """

    def __init__(
        self,
        actor_assembler: ActorObservationBatchAssembler,
        catalog_assembler: ActionCatalogAssembler,
    ) -> None:
        if not isinstance(actor_assembler, ActorObservationBatchAssembler):
            raise TypeError(
                f"actor_assembler must be an ActorObservationBatchAssembler, got "
                f"{type(actor_assembler).__name__}"
            )
        if not isinstance(catalog_assembler, ActionCatalogAssembler):
            raise TypeError(
                f"catalog_assembler must be an ActionCatalogAssembler, got "
                f"{type(catalog_assembler).__name__}"
            )
        self._actor_assembler = actor_assembler
        self._catalog_assembler = catalog_assembler

    def build(self, frame: RuntimeActorFrame) -> DecisionCaptureBatch:
        actor_batch = self._actor_assembler.build(frame)
        action_catalog_batch = self._catalog_assembler.build(frame, actor_batch)
        return DecisionCaptureBatch(
            actor_batch=actor_batch, action_catalog_batch=action_catalog_batch
        )
