"""Pure, in-memory session for capturing one coordination event's worth of
*human* robot/frontier selections, applied into the real
``robotics_interfaces.coordination.CoordinationResult`` contract.

This is the human counterpart to ``LearningCoordinationDecisionSource``
(coordination_decision_source.py): instead of a plugin choosing a
candidate, a human does, through ``select_candidate()``. Everything the
plugin-based source guarantees about the captured pool is preserved here
too -- the pool is copied and frozen exactly once, at construction, and
never touched again; no provider is ever consulted afterward;
``candidate.metadata`` is never read for any decision; and the candidate a
human picked is resolved by identity (candidate_id + candidate_index),
never by approximate coordinates.

Result type: ``robotics_sim.simulation.coordination.CoordinationResultBuilder
.assign_frontiers()`` accepts an optional ``request_executor:
Callable[[CoordinationRequest], CoordinationResult] | None`` and, when
supplied, requires ``isinstance(plugin_result, CoordinationResult)`` where
``CoordinationResult`` is ``robotics_interfaces.coordination.
CoordinationResult`` (see robotics_sim/simulation/coordination.py, read-only
reference, never imported or modified here). ``build_manual_coordination_
result()`` below returns exactly that type -- no second, parallel result
type is defined anywhere in this module.

The session does not compute a route, does not call a planner or a
plugin's ``assign()``, and does not write anything -- it only turns
explicit human input into a ``CoordinationResult`` plus one
``DemonstrationDecisionRecord`` per resolved robot.

Allowed dependency direction: robotics_sim.learning -> stdlib
(copy/dataclasses/enum/hashlib/math/types) + robotics_interfaces(.coordination,
.commands, .observations) + robotics_sim.learning.capture_inputs (the real
CandidateCaptureInput) + robotics_sim.learning.observation_batch
(build_candidate_id, called exactly once per candidate, at construction
only) + robotics_sim.learning.demonstration_episode
(DemonstrationEpisodeIdentity, DemonstrationDecisionRecord). No Qt,
robotics_sim.app, robotics_sim.simulation or engine imports.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import math
from types import MappingProxyType
from typing import Mapping

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import CoordinationAssignment, CoordinationResult
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.learning.capture_inputs import CandidateCaptureInput
from robotics_sim.learning.demonstration_episode import (
    DemonstrationDecisionRecord,
    DemonstrationEpisodeIdentity,
)
from robotics_sim.learning.observation_batch import build_candidate_id

_HASH_FIELDS: tuple[str, ...] = (
    "target",
    "source",
    "information_gain",
    "travel_cost",
    "safety_cost",
    "overlap_cost",
    "heading_cost",
    "heading_rad",
)


class ManualDemonstrationSessionError(RuntimeError):
    """Base class for ManualDemonstrationSelectionSession errors."""


class ManualDemonstrationStateError(ManualDemonstrationSessionError):
    """Operation not valid in the session's current state (already applied,
    already aborted, or not ready to apply)."""


class ManualDemonstrationSelectionError(ManualDemonstrationSessionError):
    """An invalid robot/candidate selection was attempted: unknown robot,
    out-of-range index, a candidate_id that does not match the one frozen
    into the pool at that index, or a disabled candidate."""


class ManualDemonstrationSessionState(enum.Enum):
    IDLE = "idle"
    WAITING_FOR_SELECTION = "waiting_for_selection"
    READY_TO_APPLY = "ready_to_apply"
    APPLIED = "applied"
    ABORTED = "aborted"


def _freeze_value(value: object) -> object:
    """Recursively convert a structure into an immutable, read-only
    equivalent: dict/Mapping -> MappingProxyType (values frozen too),
    list/tuple -> tuple, set -> frozenset. Anything else (str, int, float,
    bool, None, already-frozen dataclasses, ...) is returned as-is."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_candidate(candidate: ExplorationCandidate) -> ExplorationCandidate:
    """Return a new ExplorationCandidate whose ``metadata`` is a deeply
    frozen, independent copy. ``ExplorationCandidate`` is already a frozen
    dataclass (its own fields cannot be reassigned); the only mutable
    interior it carries is ``metadata`` (a plain dict by default), which is
    what this rebuilds as read-only. Never inspects metadata *contents* for
    any decision -- this only restructures it into an immutable shape."""

    return dataclasses.replace(candidate, metadata=_freeze_value(dict(candidate.metadata)))


def _hash_candidate_pool(candidates: tuple[CandidateCaptureInput, ...]) -> str:
    """Deterministic hash of one robot's shown candidate pool, in order.

    Never reads ``candidate.metadata`` -- only the same structural fields
    LearningCoordinationDecisionSource treats as identity-relevant, plus
    the host-side enabled/reachable/kind/rejection_reasons fields that
    ``CandidateCaptureInput`` itself adds.
    """

    digest = hashlib.sha256()
    for candidate_capture in candidates:
        candidate = candidate_capture.candidate
        parts = [repr(getattr(candidate, name)) for name in _HASH_FIELDS]
        parts.extend(
            [
                candidate_capture.kind.value,
                repr(candidate_capture.enabled),
                repr(candidate_capture.reachable),
                repr(candidate_capture.rejection_reasons),
            ]
        )
        digest.update("|".join(parts).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class FrozenCandidateSlot:
    """One candidate as frozen into the session at construction time.

    ``candidate_id`` is computed exactly once, when the pool is captured
    (via ``build_candidate_id()``), and stored here -- it is never
    recomputed later. ``candidate_capture`` wraps an already-deep-frozen
    ``ExplorationCandidate`` (see ``_freeze_candidate``); both
    ``CandidateCaptureInput`` and ``ExplorationCandidate`` are frozen
    dataclasses, so no field of this slot can be reassigned after
    construction.
    """

    candidate_id: str
    candidate_capture: CandidateCaptureInput

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError(f"candidate_id must be a non-empty string, got {self.candidate_id!r}")
        if not isinstance(self.candidate_capture, CandidateCaptureInput):
            raise TypeError(
                f"candidate_capture must be a CandidateCaptureInput, got "
                f"{type(self.candidate_capture).__name__}"
            )

    @property
    def candidate(self) -> ExplorationCandidate:
        return self.candidate_capture.candidate

    @property
    def enabled(self) -> bool:
        return self.candidate_capture.enabled


@dataclasses.dataclass(frozen=True)
class _RobotCandidateSnapshot:
    """Everything frozen for one robot when the pool was captured:
    its candidate slots (each already carrying its own frozen
    candidate_id), the decision_step it was opened at, and the pool hash.

    ``decision_step`` is read from ``decision_steps_by_robot`` exactly
    once, here, at freeze time -- select_candidate() never looks at the
    caller-supplied ``decision_steps_by_robot`` mapping again; it only
    reads this already-frozen snapshot.
    """

    robot_id: int
    decision_step: int
    candidates: tuple[FrozenCandidateSlot, ...]
    pool_hash: str


class ManualDemonstrationSelectionSession:
    """One coordination event's worth of human selections, one per pending
    robot, applied exactly once into a real ``CoordinationResult``.

    Holds no state beyond what was passed at construction plus the
    selections made through ``select_candidate()``; never consults a
    provider, never regenerates candidates, and freezes the candidate pool
    -- deeply, including every nested mutable structure inside
    ``candidate.metadata`` -- at construction, so later external mutation
    of the caller's objects can never reach this session, and no public
    accessor of the pool can be mutated by a caller either.
    """

    def __init__(
        self,
        identity: DemonstrationEpisodeIdentity,
        simulation_time_s: float,
        candidate_pool: Mapping[int, tuple[CandidateCaptureInput, ...]],
        robot_ids_pending: tuple[int, ...],
        decision_steps_by_robot: Mapping[int, int],
    ) -> None:
        if not isinstance(identity, DemonstrationEpisodeIdentity):
            raise TypeError(
                f"identity must be a DemonstrationEpisodeIdentity, got {type(identity).__name__}"
            )
        if isinstance(simulation_time_s, bool) or not isinstance(simulation_time_s, (int, float)):
            raise TypeError(
                f"simulation_time_s must be a real number, got {type(simulation_time_s).__name__}"
            )
        if not math.isfinite(simulation_time_s) or simulation_time_s < 0:
            raise ValueError(f"simulation_time_s must be finite and >= 0, got {simulation_time_s!r}")

        robot_ids_pending = tuple(robot_ids_pending)
        if len(set(robot_ids_pending)) != len(robot_ids_pending):
            raise ValueError(f"robot_ids_pending contains duplicates: {robot_ids_pending}")
        for robot_id in robot_ids_pending:
            if isinstance(robot_id, bool) or not isinstance(robot_id, int) or robot_id < 0:
                raise ValueError(f"robot_ids_pending entries must be non-negative ints, got {robot_id!r}")

        pending_set = set(robot_ids_pending)
        pool_keys = set(candidate_pool)
        if pool_keys != pending_set:
            raise ValueError(
                f"candidate_pool keys {sorted(pool_keys)} do not match robot_ids_pending "
                f"{sorted(pending_set)}"
            )
        step_keys = set(decision_steps_by_robot)
        if step_keys != pending_set:
            raise ValueError(
                f"decision_steps_by_robot keys {sorted(step_keys)} do not match "
                f"robot_ids_pending {sorted(pending_set)}"
            )
        steps = list(decision_steps_by_robot.values())
        if len(set(steps)) != len(steps):
            raise ValueError(f"decision_steps_by_robot has duplicate step values: {steps}")
        for step in steps:
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"decision_steps_by_robot values must be non-negative ints, got {step!r}")

        # -- The candidate pool is captured, deep-copied, deep-frozen, and
        # -- candidate_id is computed exactly once per candidate, all here.
        # -- Nothing below this point ever calls build_candidate_id() again
        # -- or reads decision_steps_by_robot (the parameter) again -- both
        # -- are only used to build the frozen snapshot below.
        snapshots: dict[int, _RobotCandidateSnapshot] = {}
        for robot_id in robot_ids_pending:
            raw_candidates = tuple(candidate_pool[robot_id])
            for i, candidate_capture in enumerate(raw_candidates):
                if not isinstance(candidate_capture, CandidateCaptureInput):
                    raise TypeError(
                        f"candidate_pool[{robot_id}][{i}] must be a CandidateCaptureInput, got "
                        f"{type(candidate_capture).__name__}"
                    )
            # copy.deepcopy first: an independent copy of every object
            # (including any mutable metadata dict) that the caller can no
            # longer reach, before any freezing happens.
            copied_candidates = copy.deepcopy(raw_candidates)
            decision_step = decision_steps_by_robot[robot_id]

            slots = []
            for index, candidate_capture in enumerate(copied_candidates):
                frozen_candidate = _freeze_candidate(candidate_capture.candidate)
                frozen_capture = dataclasses.replace(candidate_capture, candidate=frozen_candidate)
                candidate_id = build_candidate_id(robot_id, decision_step, index)
                slots.append(FrozenCandidateSlot(candidate_id=candidate_id, candidate_capture=frozen_capture))

            snapshots[robot_id] = _RobotCandidateSnapshot(
                robot_id=robot_id,
                decision_step=decision_step,
                candidates=tuple(slots),
                pool_hash=_hash_candidate_pool(tuple(slot.candidate_capture for slot in slots)),
            )

        self._identity = identity
        self._simulation_time_s = float(simulation_time_s)
        self._robot_ids_pending = robot_ids_pending
        self._pool: Mapping[int, _RobotCandidateSnapshot] = MappingProxyType(snapshots)
        self._public_candidate_pool: Mapping[int, tuple[FrozenCandidateSlot, ...]] = MappingProxyType(
            {robot_id: snapshot.candidates for robot_id, snapshot in snapshots.items()}
        )
        self._selections: dict[int, DemonstrationDecisionRecord] = {}
        self._focused_robot_id: int | None = None
        self._state = self._recompute_state()

    @property
    def state(self) -> ManualDemonstrationSessionState:
        return self._state

    @property
    def ready_to_apply(self) -> bool:
        return self._state is ManualDemonstrationSessionState.READY_TO_APPLY

    @property
    def robot_ids_pending(self) -> tuple[int, ...]:
        return self._robot_ids_pending

    @property
    def candidate_pool(self) -> Mapping[int, tuple[FrozenCandidateSlot, ...]]:
        """The full frozen candidate pool: a read-only mapping
        (types.MappingProxyType) of robot_id -> tuple of FrozenCandidateSlot.
        Assigning into the mapping, mutating a per-robot tuple, or
        reassigning a slot's fields all raise -- this is the same frozen
        snapshot built once at construction, never rebuilt."""

        return self._public_candidate_pool

    def candidates_for_robot(self, robot_id: int) -> tuple[FrozenCandidateSlot, ...]:
        self._require_pending_robot(robot_id)
        return self._pool[robot_id].candidates

    def decisions(self) -> tuple[DemonstrationDecisionRecord, ...]:
        """Currently-recorded decisions, in robot_ids_pending order. May be
        a strict subset of robot_ids_pending before ready_to_apply."""

        return tuple(
            self._selections[robot_id]
            for robot_id in self._robot_ids_pending
            if robot_id in self._selections
        )

    def _recompute_state(self) -> ManualDemonstrationSessionState:
        if not self._selections:
            return ManualDemonstrationSessionState.IDLE
        if set(self._selections) == set(self._robot_ids_pending):
            return ManualDemonstrationSessionState.READY_TO_APPLY
        return ManualDemonstrationSessionState.WAITING_FOR_SELECTION

    def _ensure_mutable(self) -> None:
        if self._state in (
            ManualDemonstrationSessionState.APPLIED,
            ManualDemonstrationSessionState.ABORTED,
        ):
            raise ManualDemonstrationStateError(
                f"session is {self._state.value}; no further selection is allowed"
            )

    def _require_pending_robot(self, robot_id: int) -> None:
        if robot_id not in self._robot_ids_pending:
            raise ManualDemonstrationSelectionError(
                f"robot_id {robot_id} is not one of the pending robots {self._robot_ids_pending}"
            )

    def select_robot(self, robot_id: int) -> None:
        """Set the UI-facing focused robot. Purely informational: it does
        not record a selection and does not require select_candidate() to
        be called immediately afterward."""

        self._ensure_mutable()
        self._require_pending_robot(robot_id)
        self._focused_robot_id = robot_id
        if self._state is ManualDemonstrationSessionState.IDLE:
            self._state = ManualDemonstrationSessionState.WAITING_FOR_SELECTION

    @property
    def focused_robot_id(self) -> int | None:
        return self._focused_robot_id

    def select_candidate(
        self,
        *,
        robot_id: int,
        candidate_index: int,
        candidate_id: str,
        human_response_time_s: float | None = None,
    ) -> None:
        """Record (or overwrite, if called again before Apply) one robot's
        chosen candidate.

        Validates, all at once: robot_id is pending, candidate_index is in
        range, ``candidate_id`` matches the id already frozen into the pool
        at that index (read from the frozen snapshot -- never recomputed,
        never a coordinate match), and the candidate is enabled.
        """

        self._ensure_mutable()
        self._require_pending_robot(robot_id)

        snapshot = self._pool[robot_id]
        candidates = snapshot.candidates
        if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
            raise TypeError(f"candidate_index must be an int, got {type(candidate_index).__name__}")
        if not (0 <= candidate_index < len(candidates)):
            raise ManualDemonstrationSelectionError(
                f"candidate_index={candidate_index} out of range [0, {len(candidates)}) for "
                f"robot {robot_id}"
            )

        slot = candidates[candidate_index]
        if candidate_id != slot.candidate_id:
            raise ManualDemonstrationSelectionError(
                f"candidate_id {candidate_id!r} does not match the id frozen at index "
                f"{candidate_index} for robot {robot_id} ({slot.candidate_id!r})"
            )
        if not slot.enabled:
            raise ManualDemonstrationSelectionError(
                f"candidate at index {candidate_index} for robot {robot_id} is not enabled"
            )

        if human_response_time_s is not None:
            if isinstance(human_response_time_s, bool) or not isinstance(
                human_response_time_s, (int, float)
            ):
                raise TypeError(
                    f"human_response_time_s must be a real number or None, got "
                    f"{type(human_response_time_s).__name__}"
                )
            if not math.isfinite(human_response_time_s) or human_response_time_s < 0:
                raise ValueError(
                    f"human_response_time_s must be finite and >= 0, got {human_response_time_s!r}"
                )
            human_response_time_s = float(human_response_time_s)

        record = DemonstrationDecisionRecord(
            episode_id=self._identity.episode_id,
            decision_step=snapshot.decision_step,
            robot_id=robot_id,
            candidate_pool=tuple(s.candidate_capture for s in candidates),
            selected_candidate_index=candidate_index,
            selected_candidate_id=candidate_id,
            target_xy=slot.candidate.target,
            candidate_pool_hash=snapshot.pool_hash,
            simulation_time_s=self._simulation_time_s,
            human_response_time_s=human_response_time_s,
        )
        self._selections[robot_id] = record
        self._state = self._recompute_state()

    def build_manual_coordination_result(self) -> CoordinationResult:
        """Apply every recorded human selection into one real
        ``robotics_interfaces.coordination.CoordinationResult`` --
        precisely the type ``CoordinationResultBuilder.assign_frontiers()``'s
        ``request_executor`` hook requires (see module docstring) --
        preserving robot_ids_pending order.

        Never computes a route, never invokes a planner or
        ``plugin.assign()``. May be called exactly once; a second call
        raises ManualDemonstrationStateError. Each selected candidate_id is
        also attached to its RobotCommand.metadata so the id shown at
        selection time is traceable all the way into the applied result.
        """

        if self._state is not ManualDemonstrationSessionState.READY_TO_APPLY:
            raise ManualDemonstrationStateError(
                f"build_manual_coordination_result() requires state READY_TO_APPLY, got "
                f"{self._state.value}"
            )

        targets = []
        reasons = []
        assignments = []
        commands = []
        for robot_id in self._robot_ids_pending:
            record = self._selections[robot_id]
            slot = self._pool[robot_id].candidates[record.selected_candidate_index]
            candidate = slot.candidate
            targets.append(candidate.target)
            reasons.append("human_selected")
            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=candidate.target,
                    reason="human_selected",
                    proposal=candidate,
                )
            )
            commands.append(
                RobotCommand(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=candidate.target,
                    heading_rad=candidate.heading_rad,
                    reason="human_selected",
                    metadata={"candidate_id": slot.candidate_id},
                )
            )

        result = CoordinationResult(
            targets=tuple(targets),
            reasons=tuple(reasons),
            strategy="manual_demonstration",
            assignments=tuple(assignments),
            commands=tuple(commands),
            debug={"candidate_id_by_robot": {rid: self._selections[rid].selected_candidate_id for rid in self._robot_ids_pending}},
        )
        self._state = ManualDemonstrationSessionState.APPLIED
        return result

    def abort(self) -> None:
        self._ensure_mutable()
        self._state = ManualDemonstrationSessionState.ABORTED
