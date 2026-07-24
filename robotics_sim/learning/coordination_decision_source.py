"""Host-side, learning-compatible coordination decision source.

Wraps one HOST_CANDIDATES coordination plugin so that:

1. the exact ``ExplorationCandidate`` pool a decision is evaluated against
   is captured exactly once, before the plugin runs;
2. the plugin is then invoked against a request built from that captured
   pool, with its own candidate-generating services disabled, so it cannot
   silently regenerate a different pool;
3. the candidate the plugin actually chose is resolved back to its position
   in the captured pool (never guessed, never silently skipped);
4. the pool is exposed as ``CandidateCaptureInput`` tuples ready to feed a
   future ``RuntimeActorFrame``.

This module does not integrate with the runtime, does not write episodes or
files, does not compute rewards, and does not build a CriticState. It has no
notion of episode_id or decision_step -- a future
``RuntimeLearningCaptureService`` owns that.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces (+ robotics_interfaces.learning for CandidateKind). No
Qt, pandas, torch, robotics_sim.app or engine imports.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.learning import CandidateKind
from robotics_interfaces.plugins import CandidateInputMode, PluginMetadata
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate
from robotics_interfaces.services import CoordinationServices
from robotics_sim.learning.capture_inputs import CandidateCaptureInput

_STRUCTURAL_FIELDS: tuple[str, ...] = (
    "target",
    "heading_rad",
    "source",
    "information_gain",
    "travel_cost",
    "safety_cost",
    "overlap_cost",
    "heading_cost",
)


class LearningCoordinatorCompatibilityError(ValueError):
    """A plugin or request cannot guarantee the captured pool is exactly
    the pool the plugin will evaluate.

    Raised instead of ever falling back silently: if compatibility or
    candidate resolution cannot be established with certainty, the caller
    must know immediately.
    """

    def __init__(self, plugin_name: str, reason: str) -> None:
        self.plugin_name = plugin_name
        self.reason = reason
        super().__init__(f"plugin {plugin_name!r} is not learning-compatible: {reason}")


@dataclass(frozen=True)
class LearningCoordinatorCompatibility:
    """Result of inspecting whether a plugin can be safely wrapped.

    ``candidate_input_mode`` is None when the plugin declares no mode at
    all, or when it has no usable metadata -- there is no real
    CandidateInputMode to report in that case, so the field is deliberately
    Optional rather than a required, possibly-fabricated value.
    """

    plugin_name: str
    candidate_input_mode: CandidateInputMode | None
    supported: bool
    reason: str


def _plugin_display_name(plugin: object) -> str:
    metadata = getattr(plugin, "metadata", None)
    name = getattr(metadata, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(plugin).__name__


def inspect_learning_compatibility(plugin: object) -> LearningCoordinatorCompatibility:
    """Inspect whether ``plugin`` can be wrapped by
    ``LearningCoordinationDecisionSource``.

    Only ``CandidateInputMode.HOST_CANDIDATES`` is supported today.
    Compatibility is never inferred from the plugin's name -- only from its
    declared ``metadata.candidate_input_mode`` plus the presence of a real
    ``assign`` method.
    """

    plugin_name = _plugin_display_name(plugin)

    metadata = getattr(plugin, "metadata", None)
    if not isinstance(metadata, PluginMetadata):
        return LearningCoordinatorCompatibility(
            plugin_name=plugin_name,
            candidate_input_mode=None,
            supported=False,
            reason="plugin has no PluginMetadata",
        )

    assign = getattr(plugin, "assign", None)
    if not callable(assign):
        return LearningCoordinatorCompatibility(
            plugin_name=metadata.name,
            candidate_input_mode=metadata.candidate_input_mode,
            supported=False,
            reason="plugin does not define assign(request)",
        )

    mode = metadata.candidate_input_mode
    if mode is None:
        return LearningCoordinatorCompatibility(
            plugin_name=metadata.name,
            candidate_input_mode=None,
            supported=False,
            reason="candidate_input_mode is not declared",
        )

    if mode is CandidateInputMode.HOST_CANDIDATES:
        return LearningCoordinatorCompatibility(
            plugin_name=metadata.name,
            candidate_input_mode=mode,
            supported=True,
            reason="HOST_CANDIDATES: plugin consumes host-provided candidates only",
        )

    if mode is CandidateInputMode.HYBRID:
        reason = "HYBRID: plugin may fall back to its own generation; not supported yet"
    elif mode is CandidateInputMode.PLUGIN_INTERNAL:
        reason = "PLUGIN_INTERNAL: plugin generates its own candidates internally"
    else:
        reason = f"{mode.value}: candidate_input_mode is not supported yet"

    return LearningCoordinatorCompatibility(
        plugin_name=metadata.name,
        candidate_input_mode=mode,
        supported=False,
        reason=reason,
    )


@dataclass(frozen=True)
class ExplicitCandidatePool:
    """The exact ExplorationCandidate pool captured for one decision, before
    any plugin has seen it."""

    robot_ids: tuple[int, ...]
    candidates_by_robot: Mapping[int, tuple[ExplorationCandidate, ...]]
    source_name: str

    def __post_init__(self) -> None:
        robot_ids = tuple(self.robot_ids)
        if len(set(robot_ids)) != len(robot_ids):
            raise ValueError(f"robot_ids contains duplicates: {robot_ids}")
        object.__setattr__(self, "robot_ids", robot_ids)

        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ValueError(f"source_name must be a non-empty string, got {self.source_name!r}")

        candidates_by_robot = dict(self.candidates_by_robot)
        if set(candidates_by_robot) != set(robot_ids):
            raise ValueError(
                f"candidates_by_robot keys {sorted(candidates_by_robot)} do not match "
                f"robot_ids {sorted(robot_ids)}"
            )

        normalized: dict[int, tuple[ExplorationCandidate, ...]] = {}
        for robot_id in robot_ids:
            value = candidates_by_robot[robot_id]
            if not isinstance(value, tuple):
                raise TypeError(
                    f"candidates_by_robot[{robot_id}] must be a tuple, got {type(value).__name__}"
                )
            for i, item in enumerate(value):
                if not isinstance(item, ExplorationCandidate):
                    raise TypeError(
                        f"candidates_by_robot[{robot_id}][{i}] must be an ExplorationCandidate, "
                        f"got {type(item).__name__}"
                    )
            normalized[robot_id] = value

        # Rebuild in robot_ids order -- never sorted, never reordered.
        object.__setattr__(
            self, "candidates_by_robot", {robot_id: normalized[robot_id] for robot_id in robot_ids}
        )


@dataclass(frozen=True)
class PreparedLearningCoordinationDecision:
    """One fully-prepared, cross-checked coordination decision: the request
    actually evaluated, the pool it was evaluated against, the plugin's raw
    result, and the learning-ready capture inputs derived from it."""

    request: CoordinationRequest
    candidate_pool: ExplicitCandidatePool
    result: CoordinationResult
    capture_inputs_by_robot: Mapping[int, tuple[CandidateCaptureInput, ...]]
    selected_candidate_index_by_robot: Mapping[int, int | None]

    def __post_init__(self) -> None:
        if not isinstance(self.request, CoordinationRequest):
            raise TypeError(
                f"request must be a CoordinationRequest, got {type(self.request).__name__}"
            )
        if not isinstance(self.candidate_pool, ExplicitCandidatePool):
            raise TypeError(
                f"candidate_pool must be an ExplicitCandidatePool, got "
                f"{type(self.candidate_pool).__name__}"
            )
        if not isinstance(self.result, CoordinationResult):
            raise TypeError(
                f"result must be a CoordinationResult, got {type(self.result).__name__}"
            )

        robot_ids = self.candidate_pool.robot_ids

        capture_keys = set(self.capture_inputs_by_robot)
        if capture_keys != set(robot_ids):
            raise ValueError(
                f"capture_inputs_by_robot keys {sorted(capture_keys)} do not match "
                f"candidate_pool.robot_ids {sorted(robot_ids)}"
            )
        selected_keys = set(self.selected_candidate_index_by_robot)
        if selected_keys != set(robot_ids):
            raise ValueError(
                f"selected_candidate_index_by_robot keys {sorted(selected_keys)} do not match "
                f"candidate_pool.robot_ids {sorted(robot_ids)}"
            )

        # request.proposals_by_robot must carry exactly the same objects, in
        # the same order, as candidate_pool.candidates_by_robot.
        for robot_id in robot_ids:
            pool_candidates = self.candidate_pool.candidates_by_robot[robot_id]
            request_candidates = tuple(self.request.proposals_by_robot.get(robot_id, ()))
            if len(request_candidates) != len(pool_candidates) or any(
                a is not b for a, b in zip(request_candidates, pool_candidates)
            ):
                raise ValueError(
                    f"request.proposals_by_robot[{robot_id}] is not exactly "
                    f"candidate_pool.candidates_by_robot[{robot_id}] (same objects, same order)"
                )

        assignments_by_robot = {a.robot_id: a for a in self.result.assignments}
        commands_by_robot = {c.robot_id: c for c in self.result.commands}

        for robot_id in robot_ids:
            n = len(self.candidate_pool.candidates_by_robot[robot_id])
            index = self.selected_candidate_index_by_robot[robot_id]
            if index is not None:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError(
                        f"selected_candidate_index_by_robot[{robot_id}] must be int or None, "
                        f"got {type(index).__name__}"
                    )
                if not (0 <= index < n):
                    raise ValueError(
                        f"selected_candidate_index_by_robot[{robot_id}]={index} out of range "
                        f"[0, {n})"
                    )

            assignment = assignments_by_robot.get(robot_id)
            if index is not None and assignment is not None and assignment.status != "ASSIGNED":
                raise ValueError(
                    f"robot {robot_id}: selected_candidate_index is not None but "
                    f"assignment.status={assignment.status!r} (expected ASSIGNED)"
                )
            if index is None and assignment is not None and assignment.status == "ASSIGNED":
                raise ValueError(
                    f"robot {robot_id}: assignment.status is ASSIGNED but "
                    f"selected_candidate_index is None"
                )

            if index is not None:
                candidate = self.candidate_pool.candidates_by_robot[robot_id][index]
                if (
                    assignment is not None
                    and assignment.target is not None
                    and candidate.target != assignment.target
                ):
                    raise ValueError(
                        f"robot {robot_id}: resolved candidate target {candidate.target!r} does "
                        f"not match assignment.target {assignment.target!r}"
                    )
                command = commands_by_robot.get(robot_id)
                if (
                    command is not None
                    and command.target is not None
                    and candidate.target != command.target
                ):
                    raise ValueError(
                        f"robot {robot_id}: resolved candidate target {candidate.target!r} does "
                        f"not match command.target {command.target!r}"
                    )
                if (
                    command is not None
                    and command.heading_rad is not None
                    and candidate.heading_rad != command.heading_rad
                ):
                    raise ValueError(
                        f"robot {robot_id}: resolved candidate heading_rad "
                        f"{candidate.heading_rad!r} does not match command.heading_rad "
                        f"{command.heading_rad!r}"
                    )


def _structural_key(candidate: ExplorationCandidate) -> tuple:
    return tuple(getattr(candidate, name) for name in _STRUCTURAL_FIELDS)


def resolve_selected_candidate_index(
    candidates: tuple[ExplorationCandidate, ...],
    assignment: CoordinationAssignment | None,
    command: RobotCommand | None,
) -> int | None:
    """Resolve which pool position ``assignment`` actually selected.

    Returns None when the plugin did not select a candidate (HOLD/FAILED, or
    no assignment at all). Raises ``LearningCoordinatorCompatibilityError``
    -- never guesses -- when an ASSIGNED assignment cannot be tied back to
    exactly one candidate in ``candidates``, or when ``command.target``
    contradicts the resolved candidate.
    """

    if assignment is None or assignment.status != "ASSIGNED":
        return None

    proposal = assignment.proposal
    if not isinstance(proposal, ExplorationCandidate):
        raise LearningCoordinatorCompatibilityError(
            "<candidate_resolution>",
            f"assignment.status is ASSIGNED but assignment.proposal is not an "
            f"ExplorationCandidate (got {type(proposal).__name__})",
        )

    index: int | None = None
    for i, candidate in enumerate(candidates):
        if candidate is proposal:
            index = i
            break

    if index is None:
        key = _structural_key(proposal)
        matches = [i for i, candidate in enumerate(candidates) if _structural_key(candidate) == key]
        if not matches:
            raise LearningCoordinatorCompatibilityError(
                "<candidate_resolution>",
                "assignment.proposal does not match any candidate in the pool, by identity or "
                "structure",
            )
        if len(matches) > 1:
            raise LearningCoordinatorCompatibilityError(
                "<candidate_resolution>",
                f"assignment.proposal matches {len(matches)} candidates structurally; ambiguous",
            )
        index = matches[0]

    if command is not None and command.target is not None and candidates[index].target != command.target:
        raise LearningCoordinatorCompatibilityError(
            "<candidate_resolution>",
            f"command.target {command.target!r} does not match resolved candidate target "
            f"{candidates[index].target!r}",
        )

    return index


def _normalize_candidate(value: object) -> ExplorationCandidate:
    if isinstance(value, ExplorationCandidate):
        return value
    if isinstance(value, CandidateProposal):
        return value.as_candidate(source="explicit_proposal")
    raise LearningCoordinatorCompatibilityError(
        "<candidate_capture>",
        f"proposals_by_robot entry is neither ExplorationCandidate nor CandidateProposal "
        f"(got {type(value).__name__})",
    )


def _index_unique(items, *, key) -> tuple[dict, set]:
    indexed: dict = {}
    duplicates: set = set()
    for item in items:
        item_key = key(item)
        if item_key in indexed:
            duplicates.add(item_key)
            continue
        indexed[item_key] = item
    return indexed, duplicates


def _build_capture_input(candidate: ExplorationCandidate) -> CandidateCaptureInput:
    """Wrap one pool candidate as a CandidateCaptureInput.

    Temporal semantics (this is a decision-time snapshot, not an outcome):
    - enabled=True means the candidate was presented to the plugin as an
      eligible action -- it does not mean it was ultimately chosen.
    - reachable=True means the candidate provider did not reject it before
      presenting it -- it is not an omniscient guarantee that a later A*
      planning attempt will succeed. A later planning failure is recorded
      as that action's *outcome*, not by revising this field after the
      fact.
    - kind is always FRONTIER_VIEWPOINT here because this source only ever
      wraps a frontier candidate provider -- the kind is known at the point
      of generation, never inferred from candidate.source.
    """

    return CandidateCaptureInput(
        candidate=candidate,
        kind=CandidateKind.FRONTIER_VIEWPOINT,
        enabled=True,
        reachable=True,
        rejection_reasons=(),
    )


class LearningCoordinationDecisionSource:
    """Wraps one HOST_CANDIDATES coordination plugin so a decision can be
    captured for learning without integrating with the runtime.

    Pure: prepare_and_assign() holds no state between calls, generates no
    decision_step, reads no system clock, and never touches the engine. It
    prepares one decision; a future RuntimeLearningCaptureService owns
    episode_id/decision_step.
    """

    def __init__(self, plugin: object) -> None:
        compatibility = inspect_learning_compatibility(plugin)
        if not compatibility.supported:
            raise LearningCoordinatorCompatibilityError(compatibility.plugin_name, compatibility.reason)
        self._plugin = plugin
        self._compatibility = compatibility

    @property
    def compatibility(self) -> LearningCoordinatorCompatibility:
        return self._compatibility

    def prepare_and_assign(self, request: CoordinationRequest) -> PreparedLearningCoordinationDecision:
        if not isinstance(request, CoordinationRequest):
            raise TypeError(
                f"request must be a CoordinationRequest, got {type(request).__name__}"
            )

        robot_ids = tuple(request.robots_to_assign)
        candidate_pool = self._obtain_pool(request, robot_ids)
        prepared_request = self._build_prepared_request(request, candidate_pool)

        result = self._plugin.assign(prepared_request)
        if not isinstance(result, CoordinationResult):
            raise LearningCoordinatorCompatibilityError(
                self._compatibility.plugin_name,
                f"assign() returned {type(result).__name__}, expected CoordinationResult",
            )

        assignments_by_robot, duplicate_assignments = _index_unique(
            result.assignments, key=lambda item: item.robot_id
        )
        if duplicate_assignments:
            raise LearningCoordinatorCompatibilityError(
                self._compatibility.plugin_name,
                f"duplicate CoordinationAssignment for robot_id(s) {sorted(duplicate_assignments)}",
            )
        commands_by_robot, duplicate_commands = _index_unique(
            result.commands, key=lambda item: item.robot_id
        )
        if duplicate_commands:
            raise LearningCoordinatorCompatibilityError(
                self._compatibility.plugin_name,
                f"duplicate RobotCommand for robot_id(s) {sorted(duplicate_commands)}",
            )

        capture_inputs_by_robot: dict[int, tuple[CandidateCaptureInput, ...]] = {}
        selected_index_by_robot: dict[int, int | None] = {}

        for robot_id in robot_ids:
            candidates = candidate_pool.candidates_by_robot[robot_id]
            assignment = assignments_by_robot.get(robot_id)
            if assignment is None:
                raise LearningCoordinatorCompatibilityError(
                    self._compatibility.plugin_name,
                    f"no CoordinationAssignment returned for requested robot_id {robot_id}",
                )
            command = commands_by_robot.get(robot_id)
            self._check_assignment_command_agreement(robot_id, assignment, command)

            try:
                index = resolve_selected_candidate_index(candidates, assignment, command)
            except LearningCoordinatorCompatibilityError as exc:
                raise LearningCoordinatorCompatibilityError(
                    self._compatibility.plugin_name, f"robot {robot_id}: {exc.reason}"
                ) from exc

            capture_inputs_by_robot[robot_id] = tuple(
                _build_capture_input(candidate) for candidate in candidates
            )
            selected_index_by_robot[robot_id] = index

        return PreparedLearningCoordinationDecision(
            request=prepared_request,
            candidate_pool=candidate_pool,
            result=result,
            capture_inputs_by_robot=capture_inputs_by_robot,
            selected_candidate_index_by_robot=selected_index_by_robot,
        )

    def _check_assignment_command_agreement(
        self,
        robot_id: int,
        assignment: CoordinationAssignment,
        command: RobotCommand | None,
    ) -> None:
        if command is None:
            return
        if str(assignment.status) != str(command.status):
            raise LearningCoordinatorCompatibilityError(
                self._compatibility.plugin_name,
                f"robot {robot_id}: assignment.status={assignment.status!r} does not match "
                f"command.status={command.status!r}",
            )
        if (
            assignment.target is not None
            and command.target is not None
            and assignment.target != command.target
        ):
            raise LearningCoordinatorCompatibilityError(
                self._compatibility.plugin_name,
                f"robot {robot_id}: assignment.target={assignment.target!r} does not match "
                f"command.target={command.target!r}",
            )

    def _obtain_pool(
        self, request: CoordinationRequest, robot_ids: tuple[int, ...]
    ) -> ExplicitCandidatePool:
        # Priority 1: fully explicit proposals, already present in the
        # original request -- only if every requested robot has a non-empty
        # tuple. Never mixed with another source within the same robot.
        if robot_ids and all(request.proposals_by_robot.get(robot_id) for robot_id in robot_ids):
            candidates_by_robot = {
                robot_id: tuple(
                    _normalize_candidate(item) for item in request.proposals_by_robot[robot_id]
                )
                for robot_id in robot_ids
            }
            return ExplicitCandidatePool(
                robot_ids=robot_ids,
                candidates_by_robot=candidates_by_robot,
                source_name="request.proposals_by_robot",
            )

        services = request.services

        # Priority 2: team frontier provider, called exactly once for the
        # whole decision.
        if services is not None and services.team_frontier_provider is not None:
            raw = services.team_frontier_provider.candidates_for_team(request)
            candidates_by_robot = {robot_id: tuple(raw.get(robot_id, ())) for robot_id in robot_ids}
            return ExplicitCandidatePool(
                robot_ids=robot_ids,
                candidates_by_robot=candidates_by_robot,
                source_name="team_frontier_provider",
            )

        # Priority 3: single-robot frontier provider, called once per robot.
        if services is not None and services.frontier_provider is not None and request.world is not None:
            robots_by_id = {robot.robot_id: robot for robot in request.robot_states}
            candidates_by_robot = {}
            for robot_id in robot_ids:
                robot = robots_by_id.get(robot_id)
                if robot is None:
                    candidates_by_robot[robot_id] = ()
                    continue
                blocked = tuple(request.blocked_targets_by_robot.get(robot_id, ()))
                candidates_by_robot[robot_id] = tuple(
                    services.frontier_provider.candidates_for_robot(
                        robot=robot, world=request.world, blocked_targets=blocked
                    )
                )
            return ExplicitCandidatePool(
                robot_ids=robot_ids,
                candidates_by_robot=candidates_by_robot,
                source_name="frontier_provider",
            )

        # No source available: a legitimate zero-candidate decision.
        return ExplicitCandidatePool(
            robot_ids=robot_ids,
            candidates_by_robot={robot_id: () for robot_id in robot_ids},
            source_name="none",
        )

    def _build_prepared_request(
        self, request: CoordinationRequest, pool: ExplicitCandidatePool
    ) -> CoordinationRequest:
        proposals_by_robot: Mapping[int, tuple[ExplorationCandidate, ...]] = dict(
            pool.candidates_by_robot
        )

        services = request.services
        if services is not None:
            # Disable the two candidate-generating services so the plugin
            # cannot regenerate a pool different from the one just
            # captured -- every other service field is preserved untouched.
            services = replace(services, team_frontier_provider=None, frontier_provider=None)

        return replace(request, proposals_by_robot=proposals_by_robot, services=services)
