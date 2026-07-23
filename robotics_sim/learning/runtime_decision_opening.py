"""Pure host-side opening: PreparedLearningCoordinationDecision + explicit
observable context -> one independent DecisionCaptureBatch/
DecisionSelectionBatch pair per ASSIGNED robot, with HOLD/FAILED robots
split out as UnresolvedCoordinationDecision entries.

Per-robot, not multi-robot batches: the real multi-robot runtime is
asynchronous -- different robots receive a new action at different events.
A single multi-robot DecisionCaptureBatch with one shared decision_step
would force a later batch to reuse the exact same robot set and order,
which does not match that reality. Each ASSIGNED robot here gets its own
single-robot RuntimeActorFrame, built and assembled independently, so one
robot's decision_step can never influence another's.

decision_step is an episode-global identifier, not a per-robot counter: it
is one value drawn from a single sequence shared by the whole episode, not
"how many decisions this robot has made". Two robots opened from the same
event get two different global steps (e.g. 7 and 8) -- neither counts that
robot's own decisions. This module only consumes already-assigned,
already-unique step values; it never allocates or counts them itself (see
RuntimeDecisionOpeningInput for the exact uniqueness rules it does check).

This module never integrates with the runtime, never calls a plugin, never
regenerates candidates, never computes rewards, and never builds a
CriticState or GroundTruthSnapshot. It has no notion of episode/session
lifecycle -- a future RuntimeLearningCaptureService owns episode_id, the
allocation of that global decision_step sequence, pending transitions, the
recorder, and rewards. This class only opens one already-decided
coordination decision.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces(.learning). No Qt, pandas, torch, robotics_sim.app,
robotics_sim.simulation or engine imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from robotics_interfaces.learning import CandidateSetSpec
from robotics_interfaces.observations import RobotCoordinationState
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBeliefFrame
from robotics_sim.learning.capture_inputs import RobotActorCaptureInput, RuntimeActorFrame
from robotics_sim.learning.coordination_decision_source import (
    PreparedLearningCoordinationDecision,
)
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler, DecisionCaptureBatch
from robotics_sim.learning.feature_inputs import FeatureNormalizationConfig
from robotics_sim.learning.transition_inputs import DecisionSelectionBatch, RobotActionSelection

_UNRESOLVED_STATUSES = ("HOLD", "FAILED")


@dataclass(frozen=True)
class RobotDecisionObservationContext:
    """Explicit, observable-only context for one robot at one decision:
    its own state, the hazard belief it can see, the candidate graph edges
    for its pool, and exactly which teammates are visible to it.

    Never carries ground truth or arbitrary metadata -- there is no field
    for either.
    """

    robot: RobotCoordinationState
    hazard_belief: HazardBeliefFrame
    graph_edges: tuple[tuple[int, int], ...]
    visible_teammates: tuple[RobotCoordinationState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.robot, RobotCoordinationState):
            raise TypeError(
                f"robot must be a RobotCoordinationState, got {type(self.robot).__name__}"
            )
        if not isinstance(self.hazard_belief, HazardBeliefFrame):
            # Also rejects every ground-truth carrier (HazardField,
            # FireSource, GroundTruthSnapshot, ...): only the
            # discovered-only belief frame type is accepted, and this
            # module never imports any privileged type.
            raise TypeError(
                f"hazard_belief must be a HazardBeliefFrame, got "
                f"{type(self.hazard_belief).__name__}"
            )

        object.__setattr__(self, "graph_edges", tuple(tuple(edge) for edge in self.graph_edges))

        teammates = tuple(self.visible_teammates)
        seen_ids: set[int] = set()
        for i, teammate in enumerate(teammates):
            if not isinstance(teammate, RobotCoordinationState):
                raise TypeError(
                    f"visible_teammates[{i}] must be a RobotCoordinationState, got "
                    f"{type(teammate).__name__}"
                )
            if teammate is self.robot:
                raise ValueError(
                    "visible_teammates must not contain the observing robot's own object"
                )
            if teammate.robot_id == self.robot.robot_id:
                raise ValueError(
                    f"visible_teammates must not contain the observing robot's own robot_id "
                    f"{self.robot.robot_id}"
                )
            if teammate.robot_id in seen_ids:
                raise ValueError(
                    f"visible_teammates contains duplicate robot_id {teammate.robot_id}"
                )
            seen_ids.add(teammate.robot_id)
        object.__setattr__(self, "visible_teammates", teammates)


@dataclass(frozen=True)
class RuntimeDecisionOpeningInput:
    """Everything needed to open one already-decided coordination decision
    into learning captures: the prepared decision itself, one
    decision_steps_by_robot entry per ASSIGNED robot, explicit observable
    context for exactly those ASSIGNED robots, and the normalization/
    candidate configuration -- none of it defaulted or hidden.

    ``decision_steps_by_robot`` values are episode-global identifiers, not
    per-robot counters: every value comes from one shared sequence for the
    whole episode, not from "how many decisions this robot has made" --
    e.g. ``{0: 7, 1: 8}`` means the episode's global steps 7 and 8 landed on
    robots 0 and 1 respectively at this event, not that robot 0 has made 7
    decisions. Values are only required to be unique *within one opening
    call* here; the future RuntimeLearningCaptureService that allocates
    them is responsible for guaranteeing uniqueness across calls too, for
    the whole episode. Because robots finish their in-flight work at
    different real-world moments, the order in which robots' decisions
    close is not required to match the numeric order of their
    decision_step values.

    No ``schema`` field: the effective FeatureSchema belongs to the
    ActorObservationBatchAssembler/DecisionCaptureAssembler injected into
    RuntimeLearningDecisionOpener, and this input has no way to validate a
    second copy against it -- carrying one here would only be able to go
    silently stale.

    No ground truth, no critic state, no reward, no simulation config, no
    arbitrary metadata -- there is no field for any of them.
    """

    episode_id: str
    time_s: float
    prepared_decision: PreparedLearningCoordinationDecision
    decision_steps_by_robot: Mapping[int, int]
    contexts_by_robot: Mapping[int, RobotDecisionObservationContext]
    grid_geometry: GridGeometry
    normalization: FeatureNormalizationConfig
    candidate_spec: CandidateSetSpec

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError(f"episode_id must be a non-empty string, got {self.episode_id!r}")
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError(f"time_s must be finite and >= 0, got {self.time_s!r}")
        if not isinstance(self.prepared_decision, PreparedLearningCoordinationDecision):
            raise TypeError(
                f"prepared_decision must be a PreparedLearningCoordinationDecision, got "
                f"{type(self.prepared_decision).__name__}"
            )
        if not isinstance(self.grid_geometry, GridGeometry):
            raise TypeError(
                f"grid_geometry must be a GridGeometry, got {type(self.grid_geometry).__name__}"
            )
        if not isinstance(self.normalization, FeatureNormalizationConfig):
            raise TypeError(
                f"normalization must be a FeatureNormalizationConfig, got "
                f"{type(self.normalization).__name__}"
            )
        if not isinstance(self.candidate_spec, CandidateSetSpec):
            raise TypeError(
                f"candidate_spec must be a CandidateSetSpec, got "
                f"{type(self.candidate_spec).__name__}"
            )

        assigned_robot_ids = {
            robot_id
            for robot_id, index in self.prepared_decision.selected_candidate_index_by_robot.items()
            if index is not None
        }

        decision_steps_by_robot = dict(self.decision_steps_by_robot)
        if set(decision_steps_by_robot) != assigned_robot_ids:
            raise ValueError(
                f"decision_steps_by_robot keys {sorted(decision_steps_by_robot)} do not match "
                f"the ASSIGNED robots {sorted(assigned_robot_ids)} (HOLD/FAILED robots must not "
                f"appear here)"
            )
        seen_steps: dict[int, int] = {}
        for robot_id, step in decision_steps_by_robot.items():
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError(
                    f"decision_steps_by_robot[{robot_id}] must be an int, got "
                    f"{type(step).__name__}"
                )
            if step < 0:
                raise ValueError(
                    f"decision_steps_by_robot[{robot_id}]={step} must be non-negative"
                )
            if step in seen_steps:
                raise ValueError(
                    f"decision_steps_by_robot has duplicate step {step} for robots "
                    f"{seen_steps[step]} and {robot_id}"
                )
            seen_steps[step] = robot_id
        object.__setattr__(self, "decision_steps_by_robot", decision_steps_by_robot)

        contexts_by_robot = dict(self.contexts_by_robot)
        if set(contexts_by_robot) != assigned_robot_ids:
            raise ValueError(
                f"contexts_by_robot keys {sorted(contexts_by_robot)} do not match the ASSIGNED "
                f"robots {sorted(assigned_robot_ids)} (HOLD/FAILED robots need no observable "
                f"context: no ActorObservation is ever built for them)"
            )
        expected_shape = (self.grid_geometry.height, self.grid_geometry.width)
        for robot_id, context in contexts_by_robot.items():
            if not isinstance(context, RobotDecisionObservationContext):
                raise TypeError(
                    f"contexts_by_robot[{robot_id}] must be a RobotDecisionObservationContext, "
                    f"got {type(context).__name__}"
                )
            if context.robot.robot_id != robot_id:
                raise ValueError(
                    f"contexts_by_robot[{robot_id}].robot.robot_id={context.robot.robot_id} "
                    f"does not match its key {robot_id}"
                )
            if context.hazard_belief.observed.shape != expected_shape:
                raise ValueError(
                    f"contexts_by_robot[{robot_id}] hazard_belief shape "
                    f"{context.hazard_belief.observed.shape} does not match grid_geometry "
                    f"{expected_shape}"
                )
        object.__setattr__(self, "contexts_by_robot", contexts_by_robot)


@dataclass(frozen=True)
class UnresolvedCoordinationDecision:
    """A HOLD or FAILED coordination outcome for one robot.

    Never a selectable action: it is not converted to a LearningAction, and
    no synthetic candidate or CandidateKind.HOLD is invented for it. What to
    do about it (wait, terminate with NO_VALID_ACTION, retry, log a
    non-trainable event) is a decision for a future service, not this one.
    """

    robot_id: int
    status: str
    reason: str
    candidate_count: int

    def __post_init__(self) -> None:
        if isinstance(self.robot_id, bool) or not isinstance(self.robot_id, int):
            raise TypeError(f"robot_id must be an int, got {type(self.robot_id).__name__}")
        if self.robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {self.robot_id}")
        if self.status not in _UNRESOLVED_STATUSES:
            raise ValueError(f"status must be one of {_UNRESOLVED_STATUSES}, got {self.status!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(f"reason must be a non-empty string, got {self.reason!r}")
        if isinstance(self.candidate_count, bool) or not isinstance(self.candidate_count, int):
            raise TypeError(
                f"candidate_count must be an int, got {type(self.candidate_count).__name__}"
            )
        if self.candidate_count < 0:
            raise ValueError(f"candidate_count must be non-negative, got {self.candidate_count}")


@dataclass(frozen=True)
class OpenedRobotLearningDecision:
    """One ASSIGNED robot's independently-built decision capture: a
    single-robot DecisionCaptureBatch and its matching single-robot
    DecisionSelectionBatch, cross-checked for internal consistency.

    Carries no RuntimeActorFrame and no other live input -- only the
    finished, validated contracts.
    """

    robot_id: int
    decision_step: int
    time_s: float
    decision_capture: DecisionCaptureBatch
    selections: DecisionSelectionBatch

    def __post_init__(self) -> None:
        if isinstance(self.robot_id, bool) or not isinstance(self.robot_id, int):
            raise TypeError(f"robot_id must be an int, got {type(self.robot_id).__name__}")
        if self.robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {self.robot_id}")
        if isinstance(self.decision_step, bool) or not isinstance(self.decision_step, int):
            raise TypeError(
                f"decision_step must be an int, got {type(self.decision_step).__name__}"
            )
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError(f"time_s must be finite and >= 0, got {self.time_s!r}")
        if not isinstance(self.decision_capture, DecisionCaptureBatch):
            raise TypeError(
                f"decision_capture must be a DecisionCaptureBatch, got "
                f"{type(self.decision_capture).__name__}"
            )
        if not isinstance(self.selections, DecisionSelectionBatch):
            raise TypeError(
                f"selections must be a DecisionSelectionBatch, got "
                f"{type(self.selections).__name__}"
            )

        observations = self.decision_capture.actor_batch.observations
        if len(observations) != 1:
            raise ValueError(
                f"decision_capture must contain exactly one robot, got {len(observations)}"
            )
        selections = self.selections.selections
        if len(selections) != 1:
            raise ValueError(f"selections must contain exactly one selection, got {len(selections)}")

        observation = observations[0]
        selection = selections[0]

        if observation.robot_id != self.robot_id:
            raise ValueError(
                f"decision_capture robot_id={observation.robot_id} does not match "
                f"robot_id={self.robot_id}"
            )
        if selection.robot_id != self.robot_id:
            raise ValueError(
                f"selections robot_id={selection.robot_id} does not match robot_id={self.robot_id}"
            )
        if self.decision_capture.actor_batch.episode_id != self.selections.episode_id:
            raise ValueError(
                f"decision_capture.episode_id={self.decision_capture.actor_batch.episode_id!r} "
                f"does not match selections.episode_id={self.selections.episode_id!r}"
            )
        if self.decision_capture.actor_batch.decision_step != self.decision_step:
            raise ValueError(
                f"decision_capture.decision_step="
                f"{self.decision_capture.actor_batch.decision_step} does not match "
                f"decision_step={self.decision_step}"
            )
        if self.selections.decision_step != self.decision_step:
            raise ValueError(
                f"selections.decision_step={self.selections.decision_step} does not match "
                f"decision_step={self.decision_step}"
            )
        if self.decision_capture.actor_batch.time_s != self.time_s:
            raise ValueError(
                f"decision_capture.time_s={self.decision_capture.actor_batch.time_s} does not "
                f"match time_s={self.time_s}"
            )

        action_index = selection.action_index
        if not (0 <= action_index < len(observation.action_mask)):
            raise ValueError(
                f"selection.action_index={action_index} out of range for robot {self.robot_id}"
            )
        if not observation.action_mask[action_index]:
            raise ValueError(
                f"selected action_index={action_index} is not enabled for robot {self.robot_id}"
            )


@dataclass(frozen=True)
class OpenedLearningDecision:
    """Result of opening one coordination decision: one independent
    OpenedRobotLearningDecision per ASSIGNED robot (never a shared
    multi-robot batch), plus every unresolved robot."""

    episode_id: str
    time_s: float
    assigned: tuple[OpenedRobotLearningDecision, ...]
    unresolved: tuple[UnresolvedCoordinationDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError(f"episode_id must be a non-empty string, got {self.episode_id!r}")
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError(f"time_s must be finite and >= 0, got {self.time_s!r}")

        assigned = tuple(self.assigned)
        seen_assigned_ids: set[int] = set()
        seen_steps: dict[int, int] = {}
        for item in assigned:
            if not isinstance(item, OpenedRobotLearningDecision):
                raise TypeError(
                    f"assigned entries must be OpenedRobotLearningDecision, got "
                    f"{type(item).__name__}"
                )
            if item.robot_id in seen_assigned_ids:
                raise ValueError(f"assigned contains duplicate robot_id {item.robot_id}")
            seen_assigned_ids.add(item.robot_id)

            item_episode_id = item.decision_capture.actor_batch.episode_id
            if item_episode_id != self.episode_id:
                raise ValueError(
                    f"assigned entry for robot {item.robot_id} has episode_id "
                    f"{item_episode_id!r}, expected {self.episode_id!r}"
                )
            if item.time_s != self.time_s:
                raise ValueError(
                    f"assigned entry for robot {item.robot_id} has time_s {item.time_s!r}, "
                    f"expected {self.time_s!r}"
                )
            if item.decision_step in seen_steps:
                raise ValueError(
                    f"assigned contains duplicate decision_step {item.decision_step} for "
                    f"robots {seen_steps[item.decision_step]} and {item.robot_id}"
                )
            seen_steps[item.decision_step] = item.robot_id
        object.__setattr__(self, "assigned", assigned)

        unresolved = tuple(self.unresolved)
        seen_unresolved: set[int] = set()
        for item in unresolved:
            if not isinstance(item, UnresolvedCoordinationDecision):
                raise TypeError(
                    f"unresolved entries must be UnresolvedCoordinationDecision, got "
                    f"{type(item).__name__}"
                )
            if item.robot_id in seen_unresolved:
                raise ValueError(f"unresolved contains duplicate robot_id {item.robot_id}")
            seen_unresolved.add(item.robot_id)
        object.__setattr__(self, "unresolved", unresolved)

        overlap = seen_assigned_ids & seen_unresolved
        if overlap:
            raise ValueError(
                f"robot_id(s) {sorted(overlap)} appear in both assigned and unresolved"
            )

    @property
    def has_assigned_actions(self) -> bool:
        return len(self.assigned) > 0

    @property
    def assigned_robot_ids(self) -> tuple[int, ...]:
        return tuple(item.robot_id for item in self.assigned)

    @property
    def unresolved_robot_ids(self) -> tuple[int, ...]:
        return tuple(item.robot_id for item in self.unresolved)


class RuntimeLearningDecisionOpener:
    """Pure assembler: RuntimeDecisionOpeningInput -> OpenedLearningDecision.

    Builds one independent single-robot RuntimeActorFrame/DecisionCaptureBatch
    per ASSIGNED robot -- never a shared multi-robot batch -- so one robot's
    decision_step and observation can never depend on another's.

    Holds no state between calls -- no episode_id, no decision_step counter,
    no pending transitions, no recorder, no session, no rewards. A future
    RuntimeLearningCaptureService owns all of that; this class only opens
    one decision.
    """

    def __init__(self, decision_assembler: DecisionCaptureAssembler) -> None:
        if not isinstance(decision_assembler, DecisionCaptureAssembler):
            raise TypeError(
                f"decision_assembler must be a DecisionCaptureAssembler, got "
                f"{type(decision_assembler).__name__}"
            )
        self._decision_assembler = decision_assembler

    def open(self, opening_input: RuntimeDecisionOpeningInput) -> OpenedLearningDecision:
        if not isinstance(opening_input, RuntimeDecisionOpeningInput):
            raise TypeError(
                f"opening_input must be a RuntimeDecisionOpeningInput, got "
                f"{type(opening_input).__name__}"
            )

        prepared = opening_input.prepared_decision
        robot_ids = prepared.candidate_pool.robot_ids  # preserves ExplicitCandidatePool order

        assignments_by_robot = {a.robot_id: a for a in prepared.result.assignments}

        assigned: list[OpenedRobotLearningDecision] = []
        unresolved: list[UnresolvedCoordinationDecision] = []

        for robot_id in robot_ids:
            index = prepared.selected_candidate_index_by_robot[robot_id]
            assignment = assignments_by_robot.get(robot_id)
            capture_inputs = prepared.capture_inputs_by_robot[robot_id]

            if index is None:
                if assignment is None or assignment.status not in _UNRESOLVED_STATUSES:
                    raise ValueError(
                        f"robot {robot_id}: no selected candidate index but "
                        f"assignment.status={getattr(assignment, 'status', None)!r} (expected "
                        f"HOLD or FAILED)"
                    )
                reason = str(assignment.reason).strip() or "no reason provided"
                unresolved.append(
                    UnresolvedCoordinationDecision(
                        robot_id=robot_id,
                        status=str(assignment.status),
                        reason=reason,
                        candidate_count=len(capture_inputs),
                    )
                )
                continue

            if assignment is None or assignment.status != "ASSIGNED":
                raise ValueError(
                    f"robot {robot_id}: selected candidate index {index} but "
                    f"assignment.status={getattr(assignment, 'status', None)!r} (expected "
                    f"ASSIGNED)"
                )
            if not (0 <= index < len(capture_inputs)):
                raise ValueError(
                    f"robot {robot_id}: selected_candidate_index {index} out of range "
                    f"[0, {len(capture_inputs)})"
                )
            selected_capture = capture_inputs[index]
            if not selected_capture.enabled:
                raise ValueError(
                    f"robot {robot_id}: candidate at selected index {index} is not enabled"
                )
            if not selected_capture.reachable:
                raise ValueError(
                    f"robot {robot_id}: candidate at selected index {index} is not reachable"
                )

            context = opening_input.contexts_by_robot[robot_id]
            decision_step = opening_input.decision_steps_by_robot[robot_id]

            frame = RuntimeActorFrame(
                episode_id=opening_input.episode_id,
                decision_step=decision_step,
                time_s=opening_input.time_s,
                robots=(
                    RobotActorCaptureInput(
                        robot=context.robot,
                        candidates=capture_inputs,
                        graph_edges=context.graph_edges,
                        visible_teammates=context.visible_teammates,
                        hazard_belief=context.hazard_belief,
                    ),
                ),
                grid_geometry=opening_input.grid_geometry,
                normalization=opening_input.normalization,
                candidate_spec=opening_input.candidate_spec,
            )
            decision_capture = self._decision_assembler.build(frame)

            observation = decision_capture.get_observation(robot_id)
            if not (0 <= index < len(observation.candidate_ids)):
                raise ValueError(
                    f"robot {robot_id}: selected index {index} out of range for the built "
                    f"observation"
                )
            if not observation.action_mask[index]:
                raise ValueError(
                    f"robot {robot_id}: action_mask[{index}] is False for the selected "
                    f"candidate"
                )

            selections = DecisionSelectionBatch(
                episode_id=opening_input.episode_id,
                decision_step=decision_step,
                selections=(
                    RobotActionSelection(
                        robot_id=robot_id, action_index=index, issued_at_step=decision_step
                    ),
                ),
            )

            assigned.append(
                OpenedRobotLearningDecision(
                    robot_id=robot_id,
                    decision_step=decision_step,
                    time_s=opening_input.time_s,
                    decision_capture=decision_capture,
                    selections=selections,
                )
            )

        return OpenedLearningDecision(
            episode_id=opening_input.episode_id,
            time_s=opening_input.time_s,
            assigned=tuple(assigned),
            unresolved=tuple(unresolved),
        )
