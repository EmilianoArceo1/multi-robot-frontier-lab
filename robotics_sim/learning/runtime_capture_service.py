"""Host-side, in-memory coordinator: one coordination event -> zero or more
learning decisions opened/replaced/closed, wired end to end through the
existing pure components -- LearningCoordinationDecisionSource,
RuntimeLearningDecisionOpener, EpisodeDecisionStepAllocator, and
InMemoryAsynchronousLearningEpisodeSession -- without duplicating any of
their state or validation.

RuntimeLearningCaptureService does not integrate with the runtime yet (no
engine.py, no robotics_sim.simulation, no robotics_sim.app import) and does
not compute rewards. It *does* materialize CriticState/GroundTruthSnapshot,
but only from already-frozen, step-agnostic sources
(CriticStateCaptureSource / GroundTruthCaptureSource) supplied inside
RuntimeCoordinationCaptureInput, and only by calling the CriticStateBuilder/
GroundTruthSnapshotBuilder injected at construction -- it never computes a
feature value itself, never derives ground truth from critic state or vice
versa, and never reads engine/live runtime state. Those sources are frozen by
the (future) runtime adapter *before* plugin.assign() runs, i.e. before the
real decision_step values for this event are even known -- see
capture_coordination_event() for exactly when materialization happens.

Ownership boundaries (nothing here is duplicated):
- LearningCoordinationDecisionSource owns candidate-pool capture and the one
  plugin.assign() call.
- RuntimeLearningDecisionOpener owns turning a prepared decision + explicit
  per-robot context into OpenedRobotLearningDecision/UnresolvedCoordinationDecision
  objects.
- EpisodeDecisionStepAllocator owns the one episode-global decision_step
  counter.
- CriticStateBuilder/GroundTruthSnapshotBuilder own turning a
  CriticStateBuildInput/GroundTruthBuildInput (assembled by this service from
  a frozen source plus the just-allocated decision_step/time_s) into the
  actual CriticState/GroundTruthSnapshot contract; this service never
  vectorizes or validates a feature schema itself.
- InMemoryAsynchronousLearningEpisodeSession owns pending decisions, the
  captured-at-open CriticState/GroundTruthSnapshot, the episode-global
  decision_step uniqueness set, and the recorder.
This service only sequences calls into those components and classifies
each ASSIGNED robot in one coordination event as "new" (no prior pending
decision -- register it) or "replace" (had a pending decision -- close it and
open the next one in the same call).

No partial application, no rollback (see capture_coordination_event): every
validation this service can compute *without* querying hidden internal state
of episode_session (candidate-pool membership, ASSIGNED/unresolved split,
source key-sets, closing-outcome key-sets, terminal-state/episode_id
agreement) is checked, for every robot in the event, before
EpisodeDecisionStepAllocator.allocate_many() is ever called -- so those
failures never touch step_allocator or episode_session at all, and never
abort anything (there is nothing to abort yet).

Once allocate_many() has run, though, episode_session does not expose a
pending decision's stored decision_step through its public API (by design --
see asynchronous_episode.py), so an outcome whose decision_step is simply
wrong for the robot it names, a builder failure, a decision_opener.open()
failure, or a register_opened_decisions()/complete_robot_decision() failure
can all still occur. This service does not retry, does not roll back
robot-by-robot, and does not return the already-consumed decision_step
values to the allocator -- there is no API for any of that. Instead, any
failure after allocate_many() has succeeded aborts the *entire* episode
(both step_allocator and episode_session) and raises
RuntimeLearningCaptureConsistencyError with the original failure preserved
as __cause__ -- see _abort_after_post_allocation_failure(). Earlier robots
in the same event that were already registered/completed before the failure
are discarded along with everything else; the caller cannot resume or patch
up that episode and must start a new one.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces(.learning). No Qt, numpy, torch, pandas,
robotics_sim.app, robotics_sim.simulation, or engine imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.learning import CandidateSetSpec
from robotics_interfaces.learning.export import EpisodeFireMetrics, EpisodeMetadata
from robotics_interfaces.learning.observations import CriticState, GroundTruthSnapshot
from robotics_interfaces.learning.termination import TerminationReason
from robotics_interfaces.learning.transitions import LearningTransition
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.learning.asynchronous_episode import InMemoryAsynchronousLearningEpisodeSession
from robotics_sim.learning.builders import CriticStateBuilder, GroundTruthSnapshotBuilder
from robotics_sim.learning.coordination_decision_source import (
    LearningCoordinationDecisionSource,
    PreparedLearningCoordinationDecision,
)
from robotics_sim.learning.decision_steps import EpisodeDecisionStepAllocator
from robotics_sim.learning.feature_inputs import FeatureNormalizationConfig
from robotics_sim.learning.recorder import EpisodeRecord
from robotics_sim.learning.runtime_decision_opening import (
    OpenedLearningDecision,
    OpenedRobotLearningDecision,
    RobotDecisionObservationContext,
    RuntimeDecisionOpeningInput,
    RuntimeLearningDecisionOpener,
    UnresolvedCoordinationDecision,
)
from robotics_sim.learning.source_models import CriticStateBuildInput, GroundTruthBuildInput
from robotics_sim.learning.transition_inputs import TransitionOutcomeBatch


class RuntimeLearningCaptureError(RuntimeError):
    """Base class for RuntimeLearningCaptureService errors."""


class RuntimeLearningCaptureStateError(RuntimeLearningCaptureError):
    """Operation not valid in the service's current lifecycle state (no
    active episode, an episode already active, or finishing with pending
    decisions still open). Also raised -- instead of
    RuntimeLearningCaptureConsistencyError -- when a post-allocation failure
    inside capture_coordination_event() triggers an abort_episode() cleanup
    that itself fails: a double failure, chained to the cleanup exception
    (not the original one), since the service could not even be reset to a
    known-clean inactive state."""


class RuntimeLearningCaptureConsistencyError(RuntimeLearningCaptureError):
    """A contradiction between pending decisions, closing outcomes,
    ASSIGNED/unresolved assignments, or the supplied sources was detected --
    always before EpisodeDecisionStepAllocator.allocate_many() is called for
    the robot(s) involved, so step_allocator/episode_session are never
    touched for these.

    Also raised for *any* failure after allocate_many() has already run --
    materializing CriticState/GroundTruthSnapshot from a source, a
    decision_step/time_s mismatch, decision_opener.open() failing, or
    register_opened_decisions()/complete_robot_decision() failing -- but only
    after capture_coordination_event() has called abort_episode() to tear
    down the *entire* episode (see _abort_after_post_allocation_failure()).
    Those already-allocated global decision_step values are not returned to
    the allocator (there is no API for that) and no per-robot rollback is
    attempted; the whole episode is invalidated instead of leaving it active
    with steps consumed but no matching registered/completed decision. The
    original exception is preserved as __cause__. The caller must start a
    new episode; there is nothing to resume."""


def _require_robot_id_key(mapping_name: str, robot_id: object) -> None:
    if isinstance(robot_id, bool) or not isinstance(robot_id, int):
        raise TypeError(
            f"{mapping_name} keys must be int robot ids, got {type(robot_id).__name__}"
        )
    if robot_id < 0:
        raise ValueError(f"{mapping_name} keys must be non-negative, got {robot_id}")


def _validate_feature_group_names(group: str, names: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(names)
    for i, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{group}[{i}] must be a non-empty string, got {name!r}")
    return names


def _validate_feature_values(group: str, features: Mapping[str, float]) -> Mapping[str, float]:
    """Validate and copy a feature mapping into a fresh, read-only view.

    The returned MappingProxyType wraps a dict built here (not the caller's
    mapping), so later mutation of the caller's original mapping never
    reaches the copy, and mutation attempts through the returned mapping
    itself raise TypeError.
    """

    validated: dict[str, float] = {}
    for name, value in features.items():
        if isinstance(value, GroundTruthSnapshot):
            raise TypeError(f"{group}[{name!r}] must not be a GroundTruthSnapshot")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{group}[{name!r}] must be a real number, got {type(value).__name__}"
            )
        if not math.isfinite(value):
            raise ValueError(f"{group}[{name!r}] must be finite, got {value!r}")
        validated[name] = value
    return MappingProxyType(validated)


@dataclass(frozen=True)
class CriticStateCaptureSource:
    """Step-agnostic, frozen source for one robot's CriticState.

    Carries the same named feature groups as CriticStateBuildInput, minus
    decision_step and time_s: this source is captured/frozen *before*
    plugin.assign() runs, i.e. before the real decision_step for this event
    is even known (that only exists after EpisodeDecisionStepAllocator.
    allocate_many() has run -- see capture_coordination_event()). It also
    never carries a GroundTruthSnapshot, directly or nested in a feature
    value, and never carries metadata.

    Validation here is deliberately light: names are coerced to non-empty
    string tuples, feature values must be finite real numbers (not bool, not
    a GroundTruthSnapshot), and mappings are defensively copied. It does not
    duplicate FeatureSchema's exact-key-set matching or vector ordering --
    CriticStateBuilder (via CriticStateBuildInput) remains the sole authority
    for that.

    Deeply immutable: global_features and every per-robot feature mapping
    are copied into fresh dicts and exposed as types.MappingProxyType views,
    and per_robot_features itself is also a MappingProxyType. Neither
    mutating the caller's original mappings after construction nor
    assigning through source.global_features[...]/
    source.per_robot_features[...][...] can change this source -- both
    raise TypeError.
    """

    global_feature_names: tuple[str, ...]
    global_features: Mapping[str, float]
    per_robot_feature_names: tuple[str, ...]
    per_robot_features: Mapping[int, Mapping[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_feature_names",
            _validate_feature_group_names("global_feature_names", self.global_feature_names),
        )
        object.__setattr__(
            self,
            "per_robot_feature_names",
            _validate_feature_group_names(
                "per_robot_feature_names", self.per_robot_feature_names
            ),
        )
        object.__setattr__(
            self,
            "global_features",
            _validate_feature_values("global_features", self.global_features),
        )

        per_robot_features: dict[int, Mapping[str, float]] = {}
        for robot_id, features in self.per_robot_features.items():
            _require_robot_id_key("per_robot_features", robot_id)
            if isinstance(features, GroundTruthSnapshot):
                raise TypeError(
                    f"per_robot_features[{robot_id}] must not be a GroundTruthSnapshot"
                )
            per_robot_features[robot_id] = _validate_feature_values(
                f"per_robot_features[{robot_id}]", features
            )
        # Outer mapping is also a fresh, read-only view: neither
        # per_robot_features[robot_id][name] = x nor
        # per_robot_features[new_robot_id] = {...} can mutate this source,
        # and mutating the caller's original outer/inner mappings afterward
        # never reaches here either (both levels were copied above).
        object.__setattr__(self, "per_robot_features", MappingProxyType(per_robot_features))


@dataclass(frozen=True)
class GroundTruthCaptureSource:
    """Step-agnostic, frozen source for one robot's GroundTruthSnapshot.

    Carries the same privileged fields as GroundTruthBuildInput, minus
    decision_step and time_s (same reasoning as CriticStateCaptureSource:
    frozen before the real decision_step is known). Never carries a critic
    feature block or metadata. true_occupancy arrives already built by the
    (future) host adapter -- this class never derives it from
    mapped_obstacle_points or any other raw runtime state; it only makes a
    defensive, immutable copy of whatever occupancy grid it is given.

    Deeply immutable: true_robot_poses is copied into a fresh dict and
    exposed as a types.MappingProxyType (each pose itself a plain 3-tuple);
    true_occupancy and true_fire_locations are copied into tuples of tuples.
    Mutating the caller's original dict/lists after construction, or
    assigning through source.true_robot_poses[...], never changes this
    source -- the latter raises TypeError.
    """

    true_robot_poses: Mapping[int, tuple[float, float, float]]
    true_occupancy: tuple[tuple[int, ...], ...]
    true_fire_locations: tuple[tuple[float, float], ...]
    global_coverage_fraction: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.global_coverage_fraction):
            raise ValueError(
                f"global_coverage_fraction must be finite, got "
                f"{self.global_coverage_fraction!r}"
            )

        true_robot_poses: dict[int, tuple[float, float, float]] = {}
        for robot_id, pose in self.true_robot_poses.items():
            _require_robot_id_key("true_robot_poses", robot_id)
            pose_t = tuple(pose)
            if len(pose_t) != 3:
                raise ValueError(
                    f"true_robot_poses[{robot_id}] must be an (x, y, theta) triple, got "
                    f"{pose_t!r}"
                )
            for component in pose_t:
                if isinstance(component, bool) or not isinstance(component, (int, float)):
                    raise TypeError(
                        f"true_robot_poses[{robot_id}] components must be real numbers, got "
                        f"{type(component).__name__}"
                    )
                if not math.isfinite(component):
                    raise ValueError(
                        f"true_robot_poses[{robot_id}] must be finite, got {pose_t!r}"
                    )
            true_robot_poses[robot_id] = (float(pose_t[0]), float(pose_t[1]), float(pose_t[2]))
        object.__setattr__(self, "true_robot_poses", MappingProxyType(true_robot_poses))

        object.__setattr__(
            self,
            "true_occupancy",
            tuple(tuple(int(cell) for cell in row) for row in self.true_occupancy),
        )
        object.__setattr__(
            self,
            "true_fire_locations",
            tuple((float(x), float(y)) for x, y in self.true_fire_locations),
        )


@dataclass(frozen=True)
class RuntimeCoordinationCaptureInput:
    """Everything one coordination event needs, captured explicitly by the
    (future) runtime adapter -- this service never computes any of it.

    ``contexts_by_robot``/``critic_sources_by_robot`` are frozen,
    step-agnostic sources taken at the instant the *new* decision is opened;
    ``ground_truth_sources_by_robot`` is optional and corresponds to that
    same instant. ``closing_outcomes_by_robot`` describes each ASSIGNED
    robot's *previous* pending decision -- the one being closed and replaced
    by this event, not the one being opened.

    ``critic_sources_by_robot``/``ground_truth_sources_by_robot`` carry no
    decision_step and no time_s: the real decision_step for each ASSIGNED
    robot is only known after the plugin has run and
    EpisodeDecisionStepAllocator.allocate_many() has assigned it (which this
    dataclass cannot do -- see
    RuntimeLearningCaptureService.capture_coordination_event). The service
    materializes the step-bound CriticState/GroundTruthSnapshot contracts
    from these sources only after that allocation, via the injected
    CriticStateBuilder/GroundTruthSnapshotBuilder.

    Deliberately does not require its keys to match the ASSIGNED robots yet:
    that split is only known after LearningCoordinationDecisionSource has run
    the plugin, which this dataclass cannot do either.
    """

    request: CoordinationRequest
    time_s: float
    contexts_by_robot: Mapping[int, RobotDecisionObservationContext]
    critic_sources_by_robot: Mapping[int, CriticStateCaptureSource]
    ground_truth_sources_by_robot: Mapping[int, GroundTruthCaptureSource] | None
    closing_outcomes_by_robot: Mapping[int, TransitionOutcomeBatch]
    grid_geometry: GridGeometry
    normalization: FeatureNormalizationConfig
    candidate_spec: CandidateSetSpec

    def __post_init__(self) -> None:
        if not isinstance(self.request, CoordinationRequest):
            raise TypeError(
                f"request must be a CoordinationRequest, got {type(self.request).__name__}"
            )
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError(f"time_s must be finite and >= 0, got {self.time_s!r}")
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

        contexts_by_robot = dict(self.contexts_by_robot)
        for robot_id, context in contexts_by_robot.items():
            _require_robot_id_key("contexts_by_robot", robot_id)
            if not isinstance(context, RobotDecisionObservationContext):
                raise TypeError(
                    f"contexts_by_robot[{robot_id}] must be a RobotDecisionObservationContext, "
                    f"got {type(context).__name__}"
                )
        object.__setattr__(self, "contexts_by_robot", contexts_by_robot)

        critic_sources_by_robot = dict(self.critic_sources_by_robot)
        for robot_id, source in critic_sources_by_robot.items():
            _require_robot_id_key("critic_sources_by_robot", robot_id)
            if not isinstance(source, CriticStateCaptureSource):
                raise TypeError(
                    f"critic_sources_by_robot[{robot_id}] must be a CriticStateCaptureSource, "
                    f"got {type(source).__name__}"
                )
        object.__setattr__(self, "critic_sources_by_robot", critic_sources_by_robot)

        ground_truth_sources_by_robot = (
            None
            if self.ground_truth_sources_by_robot is None
            else dict(self.ground_truth_sources_by_robot)
        )
        if ground_truth_sources_by_robot is not None:
            for robot_id, source in ground_truth_sources_by_robot.items():
                _require_robot_id_key("ground_truth_sources_by_robot", robot_id)
                if not isinstance(source, GroundTruthCaptureSource):
                    raise TypeError(
                        f"ground_truth_sources_by_robot[{robot_id}] must be a "
                        f"GroundTruthCaptureSource, got {type(source).__name__}"
                    )
        object.__setattr__(
            self, "ground_truth_sources_by_robot", ground_truth_sources_by_robot
        )

        closing_outcomes_by_robot = dict(self.closing_outcomes_by_robot)
        for robot_id, outcome in closing_outcomes_by_robot.items():
            _require_robot_id_key("closing_outcomes_by_robot", robot_id)
            if not isinstance(outcome, TransitionOutcomeBatch):
                raise TypeError(
                    f"closing_outcomes_by_robot[{robot_id}] must be a TransitionOutcomeBatch, "
                    f"got {type(outcome).__name__}"
                )
        object.__setattr__(self, "closing_outcomes_by_robot", closing_outcomes_by_robot)


@dataclass(frozen=True)
class RuntimeCoordinationCaptureResult:
    """Everything capture_coordination_event() produced for one event.

    Stores the prepared decision and the (unmodified) opened decision so
    callers can inspect the candidate pool / selected indices / raw
    UnresolvedCoordinationDecision entries -- never a copy of
    RuntimeCoordinationCaptureInput, never a live robot reference, never
    ground truth directly.
    """

    prepared_decision: PreparedLearningCoordinationDecision
    opened_decision: OpenedLearningDecision
    completed_transitions: tuple[LearningTransition, ...]
    newly_registered_robot_ids: tuple[int, ...]
    replaced_robot_ids: tuple[int, ...]
    unresolved: tuple[UnresolvedCoordinationDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prepared_decision, PreparedLearningCoordinationDecision):
            raise TypeError(
                f"prepared_decision must be a PreparedLearningCoordinationDecision, got "
                f"{type(self.prepared_decision).__name__}"
            )
        if not isinstance(self.opened_decision, OpenedLearningDecision):
            raise TypeError(
                f"opened_decision must be an OpenedLearningDecision, got "
                f"{type(self.opened_decision).__name__}"
            )

        completed_transitions = tuple(self.completed_transitions)
        for i, transition in enumerate(completed_transitions):
            if not isinstance(transition, LearningTransition):
                raise TypeError(
                    f"completed_transitions[{i}] must be a LearningTransition, got "
                    f"{type(transition).__name__}"
                )
        object.__setattr__(self, "completed_transitions", completed_transitions)

        newly_registered_robot_ids = tuple(self.newly_registered_robot_ids)
        replaced_robot_ids = tuple(self.replaced_robot_ids)
        unresolved = tuple(self.unresolved)
        for i, item in enumerate(unresolved):
            if not isinstance(item, UnresolvedCoordinationDecision):
                raise TypeError(
                    f"unresolved[{i}] must be an UnresolvedCoordinationDecision, got "
                    f"{type(item).__name__}"
                )

        if len(set(newly_registered_robot_ids)) != len(newly_registered_robot_ids):
            raise ValueError(
                f"newly_registered_robot_ids contains duplicates: {newly_registered_robot_ids}"
            )
        if len(set(replaced_robot_ids)) != len(replaced_robot_ids):
            raise ValueError(f"replaced_robot_ids contains duplicates: {replaced_robot_ids}")

        new_set = set(newly_registered_robot_ids)
        replaced_set = set(replaced_robot_ids)
        overlap = new_set & replaced_set
        if overlap:
            raise ValueError(
                f"newly_registered_robot_ids and replaced_robot_ids share robot id(s) "
                f"{sorted(overlap)}"
            )

        unresolved_ids = tuple(item.robot_id for item in unresolved)
        unresolved_overlap = set(unresolved_ids) & (new_set | replaced_set)
        if unresolved_overlap:
            raise ValueError(
                f"unresolved shares robot id(s) {sorted(unresolved_overlap)} with "
                f"newly_registered_robot_ids/replaced_robot_ids"
            )

        if len(completed_transitions) != len(replaced_robot_ids):
            raise ValueError(
                f"completed_transitions has {len(completed_transitions)} entries but "
                f"replaced_robot_ids has {len(replaced_robot_ids)}"
            )
        for i, transition in enumerate(completed_transitions):
            transition_robot_ids = set(transition.actor_observations)
            if not transition_robot_ids.issubset(replaced_set):
                raise ValueError(
                    f"completed_transitions[{i}] involves robot id(s) "
                    f"{sorted(transition_robot_ids)}, which are not in replaced_robot_ids "
                    f"{replaced_robot_ids}"
                )

        pool_order = self.prepared_decision.candidate_pool.robot_ids
        for label, ids in (
            ("newly_registered_robot_ids", newly_registered_robot_ids),
            ("replaced_robot_ids", replaced_robot_ids),
            ("unresolved", unresolved_ids),
        ):
            id_set = set(ids)
            expected_order = tuple(robot_id for robot_id in pool_order if robot_id in id_set)
            if ids != expected_order:
                raise ValueError(
                    f"{label} must preserve candidate_pool.robot_ids order: expected "
                    f"{expected_order}, got {ids}"
                )

        object.__setattr__(self, "newly_registered_robot_ids", newly_registered_robot_ids)
        object.__setattr__(self, "replaced_robot_ids", replaced_robot_ids)
        object.__setattr__(self, "unresolved", unresolved)


class RuntimeLearningCaptureService:
    """Coordinates LearningCoordinationDecisionSource,
    RuntimeLearningDecisionOpener, EpisodeDecisionStepAllocator, and
    InMemoryAsynchronousLearningEpisodeSession into one host-side, in-memory
    capture service -- not yet wired to any runtime.

    Holds no pending decisions, seen steps, transitions, candidate pools,
    critic states, or ground truth of its own: all of that already belongs
    to episode_session (and, for candidate pools/critic/ground truth of a
    single event, to the objects passed straight through it). This class
    only sequences calls between the four injected components.
    """

    def __init__(
        self,
        decision_source: LearningCoordinationDecisionSource,
        decision_opener: RuntimeLearningDecisionOpener,
        step_allocator: EpisodeDecisionStepAllocator,
        episode_session: InMemoryAsynchronousLearningEpisodeSession,
        critic_state_builder: CriticStateBuilder,
        ground_truth_builder: GroundTruthSnapshotBuilder,
    ) -> None:
        if not isinstance(decision_source, LearningCoordinationDecisionSource):
            raise TypeError(
                f"decision_source must be a LearningCoordinationDecisionSource, got "
                f"{type(decision_source).__name__}"
            )
        if not isinstance(decision_opener, RuntimeLearningDecisionOpener):
            raise TypeError(
                f"decision_opener must be a RuntimeLearningDecisionOpener, got "
                f"{type(decision_opener).__name__}"
            )
        if not isinstance(step_allocator, EpisodeDecisionStepAllocator):
            raise TypeError(
                f"step_allocator must be an EpisodeDecisionStepAllocator, got "
                f"{type(step_allocator).__name__}"
            )
        if not isinstance(episode_session, InMemoryAsynchronousLearningEpisodeSession):
            raise TypeError(
                f"episode_session must be an InMemoryAsynchronousLearningEpisodeSession, got "
                f"{type(episode_session).__name__}"
            )
        if not isinstance(critic_state_builder, CriticStateBuilder):
            raise TypeError(
                f"critic_state_builder must be a CriticStateBuilder, got "
                f"{type(critic_state_builder).__name__}"
            )
        if not isinstance(ground_truth_builder, GroundTruthSnapshotBuilder):
            raise TypeError(
                f"ground_truth_builder must be a GroundTruthSnapshotBuilder, got "
                f"{type(ground_truth_builder).__name__}"
            )
        self._decision_source = decision_source
        self._decision_opener = decision_opener
        self._step_allocator = step_allocator
        self._episode_session = episode_session
        self._critic_state_builder = critic_state_builder
        self._ground_truth_builder = ground_truth_builder

    @property
    def is_active(self) -> bool:
        return self._episode_session.is_active

    @property
    def episode_id(self) -> str | None:
        return self._episode_session.episode_id

    @property
    def pending_robot_ids(self) -> tuple[int, ...]:
        return self._episode_session.pending_robot_ids

    @property
    def next_decision_step(self) -> int | None:
        return self._step_allocator.next_step if self._step_allocator.is_active else None

    def start_episode(self, metadata: EpisodeMetadata, start_step: int = 0) -> None:
        if self._step_allocator.is_active or self._episode_session.is_active:
            raise RuntimeLearningCaptureStateError(
                "start_episode() requires the allocator and the episode session to both be "
                "inactive"
            )
        if not isinstance(metadata, EpisodeMetadata):
            raise TypeError(f"metadata must be an EpisodeMetadata, got {type(metadata).__name__}")

        self._step_allocator.start_episode(start_step)
        try:
            self._episode_session.start_episode(metadata)
        except Exception:
            # Roll the allocator back so a failed start leaves the service
            # fully inactive again, not half-started.
            self._step_allocator.abort_episode()
            raise

    def capture_coordination_event(
        self, capture_input: RuntimeCoordinationCaptureInput
    ) -> RuntimeCoordinationCaptureResult:
        if not self._episode_session.is_active:
            raise RuntimeLearningCaptureStateError(
                "capture_coordination_event() called with no active episode"
            )
        if not isinstance(capture_input, RuntimeCoordinationCaptureInput):
            raise TypeError(
                f"capture_input must be a RuntimeCoordinationCaptureInput, got "
                f"{type(capture_input).__name__}"
            )

        episode_id = self._episode_session.episode_id

        # Run the plugin exactly once, via the wrapping decision source.
        prepared = self._decision_source.prepare_and_assign(capture_input.request)

        robot_ids = prepared.candidate_pool.robot_ids
        assigned_robot_ids = tuple(
            robot_id
            for robot_id in robot_ids
            if prepared.selected_candidate_index_by_robot[robot_id] is not None
        )
        assigned_set = set(assigned_robot_ids)
        unresolved_robot_ids = tuple(
            robot_id for robot_id in robot_ids if robot_id not in assigned_set
        )

        # -- Every check below is computable from capture_input + prepared
        # -- alone (no episode_session internals needed) -- so all of it
        # happens before step_allocator or episode_session are touched.
        contexts_keys = set(capture_input.contexts_by_robot)
        if contexts_keys != assigned_set:
            raise RuntimeLearningCaptureConsistencyError(
                f"contexts_by_robot keys {sorted(contexts_keys)} do not match the ASSIGNED "
                f"robots {sorted(assigned_set)}"
            )
        critic_source_keys = set(capture_input.critic_sources_by_robot)
        if critic_source_keys != assigned_set:
            raise RuntimeLearningCaptureConsistencyError(
                f"critic_sources_by_robot keys {sorted(critic_source_keys)} do not match the "
                f"ASSIGNED robots {sorted(assigned_set)}"
            )
        ground_truth_sources_by_robot = capture_input.ground_truth_sources_by_robot or {}
        extra_ground_truth = set(ground_truth_sources_by_robot) - assigned_set
        if extra_ground_truth:
            raise RuntimeLearningCaptureConsistencyError(
                f"ground_truth_sources_by_robot contains robot id(s) {sorted(extra_ground_truth)} "
                f"that are not ASSIGNED"
            )

        pending_before = set(self._episode_session.pending_robot_ids)
        robots_with_pending = tuple(
            robot_id for robot_id in assigned_robot_ids if robot_id in pending_before
        )
        robots_without_pending = tuple(
            robot_id for robot_id in assigned_robot_ids if robot_id not in pending_before
        )

        unresolved_with_pending = tuple(
            robot_id for robot_id in unresolved_robot_ids if robot_id in pending_before
        )
        if unresolved_with_pending:
            raise RuntimeLearningCaptureConsistencyError(
                f"robot id(s) {unresolved_with_pending} are HOLD/FAILED this event but already "
                f"have a pending decision; closing a pending decision for a HOLD/FAILED robot "
                f"is not decided by this service (no synthetic NO_VALID_ACTION, no silent "
                f"terminated/truncated inference) -- a future runtime outcome adapter must "
                f"supply an explicit outcome for it instead"
            )

        closing_outcomes = dict(capture_input.closing_outcomes_by_robot)
        expected_closing_keys = set(robots_with_pending)
        if set(closing_outcomes) != expected_closing_keys:
            raise RuntimeLearningCaptureConsistencyError(
                f"closing_outcomes_by_robot keys {sorted(closing_outcomes)} do not match the "
                f"ASSIGNED robots that already had a pending decision "
                f"{sorted(expected_closing_keys)}"
            )
        for robot_id in robots_with_pending:
            outcome = closing_outcomes[robot_id]
            if outcome.episode_id != episode_id:
                raise RuntimeLearningCaptureConsistencyError(
                    f"robot {robot_id}: closing outcome episode_id {outcome.episode_id!r} does "
                    f"not match the active episode {episode_id!r}"
                )
            is_terminal = outcome.terminated or outcome.truncated
            if is_terminal or outcome.termination_reason is not TerminationReason.RUNNING:
                raise RuntimeLearningCaptureConsistencyError(
                    f"robot {robot_id}: closing outcome must be non-terminal (terminated=False, "
                    f"truncated=False, termination_reason=RUNNING) because this event is "
                    f"replacing its pending decision with a new one; use "
                    f"complete_terminal_robot_decision() to close a decision without a "
                    f"replacement"
                )

        # -- All computable validation passed. Only now do we allocate steps
        # and mutate episode_session. --
        steps_by_robot = self._step_allocator.allocate_many(assigned_robot_ids)

        # -- Everything from here on runs after allocate_many() has already
        # consumed global decision_step values, so it is wrapped in one
        # try/except: materialization, opening, and registration/replacement
        # are not individually recoverable, and this service does not retry
        # or roll back robot-by-robot (see the module docstring). Any
        # failure in any of these steps invalidates the whole event, and
        # the whole episode is aborted rather than left with steps consumed
        # but no matching registered/completed decision -- see
        # _abort_after_post_allocation_failure().
        try:
            materialized_critic_by_robot: dict[int, CriticState] = {}
            materialized_ground_truth_by_robot: dict[int, GroundTruthSnapshot] = {}
            for robot_id in assigned_robot_ids:
                step = steps_by_robot[robot_id]
                critic_source = capture_input.critic_sources_by_robot[robot_id]
                critic_build_input = CriticStateBuildInput(
                    decision_step=step,
                    time_s=capture_input.time_s,
                    global_feature_names=critic_source.global_feature_names,
                    global_features=critic_source.global_features,
                    per_robot_feature_names=critic_source.per_robot_feature_names,
                    per_robot_features=critic_source.per_robot_features,
                )
                critic_state = self._critic_state_builder.build(critic_build_input)
                if critic_state.decision_step != step or critic_state.time_s != capture_input.time_s:
                    raise ValueError(
                        f"robot {robot_id}: built CriticState (decision_step="
                        f"{critic_state.decision_step}, time_s={critic_state.time_s}) does not "
                        f"match the opened decision (decision_step={step}, time_s="
                        f"{capture_input.time_s})"
                    )
                materialized_critic_by_robot[robot_id] = critic_state

                ground_truth_source = ground_truth_sources_by_robot.get(robot_id)
                if ground_truth_source is not None:
                    ground_truth_build_input = GroundTruthBuildInput(
                        decision_step=step,
                        time_s=capture_input.time_s,
                        true_robot_poses=ground_truth_source.true_robot_poses,
                        true_occupancy=ground_truth_source.true_occupancy,
                        true_fire_locations=ground_truth_source.true_fire_locations,
                        global_coverage_fraction=ground_truth_source.global_coverage_fraction,
                    )
                    ground_truth = self._ground_truth_builder.build(ground_truth_build_input)
                    if (
                        ground_truth.decision_step != step
                        or ground_truth.time_s != capture_input.time_s
                    ):
                        raise ValueError(
                            f"robot {robot_id}: built GroundTruthSnapshot (decision_step="
                            f"{ground_truth.decision_step}, time_s={ground_truth.time_s}) does "
                            f"not match the opened decision (decision_step={step}, time_s="
                            f"{capture_input.time_s})"
                        )
                    materialized_ground_truth_by_robot[robot_id] = ground_truth

            opening_input = RuntimeDecisionOpeningInput(
                episode_id=episode_id,
                time_s=capture_input.time_s,
                prepared_decision=prepared,
                decision_steps_by_robot=steps_by_robot,
                contexts_by_robot=capture_input.contexts_by_robot,
                grid_geometry=capture_input.grid_geometry,
                normalization=capture_input.normalization,
                candidate_spec=capture_input.candidate_spec,
            )
            opened = self._decision_opener.open(opening_input)

            without_pending_set = set(robots_without_pending)
            with_pending_set = set(robots_with_pending)
            new_items = tuple(
                item for item in opened.assigned if item.robot_id in without_pending_set
            )
            replace_items = tuple(
                item for item in opened.assigned if item.robot_id in with_pending_set
            )

            if new_items:
                reduced_opened = OpenedLearningDecision(
                    episode_id=episode_id,
                    time_s=capture_input.time_s,
                    assigned=new_items,
                    unresolved=(),
                )
                critic_for_new = {
                    item.robot_id: materialized_critic_by_robot[item.robot_id]
                    for item in new_items
                }
                ground_truth_for_new = {
                    item.robot_id: materialized_ground_truth_by_robot[item.robot_id]
                    for item in new_items
                    if item.robot_id in materialized_ground_truth_by_robot
                }
                self._episode_session.register_opened_decisions(
                    reduced_opened, critic_for_new, ground_truth_for_new or None
                )

            completed_transitions: list[LearningTransition] = []
            for item in replace_items:
                transition = self._episode_session.complete_robot_decision(
                    robot_id=item.robot_id,
                    outcome=closing_outcomes[item.robot_id],
                    next_decision=item,
                    next_critic_state=materialized_critic_by_robot[item.robot_id],
                    next_ground_truth=materialized_ground_truth_by_robot.get(item.robot_id),
                )
                completed_transitions.append(transition)

            result = RuntimeCoordinationCaptureResult(
                prepared_decision=prepared,
                opened_decision=opened,
                completed_transitions=tuple(completed_transitions),
                newly_registered_robot_ids=tuple(item.robot_id for item in new_items),
                replaced_robot_ids=tuple(item.robot_id for item in replace_items),
                unresolved=opened.unresolved,
            )
        except Exception as exc:
            self._abort_after_post_allocation_failure(exc)
            raise  # pragma: no cover -- the helper above always raises

        return result

    def _abort_after_post_allocation_failure(self, original_exc: BaseException) -> None:
        """Tear down the whole episode after a post-allocation failure.

        allocate_many() has already consumed global decision_step values by
        the time this is called, and this service never returns steps to
        the allocator or rolls back individual robots -- so the only
        consistent outcome is to invalidate the entire episode. Always
        raises: RuntimeLearningCaptureConsistencyError (chained to
        original_exc) if abort_episode() itself succeeds, or
        RuntimeLearningCaptureStateError (chained to the cleanup failure
        instead) if abort_episode() also fails.
        """

        try:
            self.abort_episode()
        except Exception as abort_exc:
            raise RuntimeLearningCaptureStateError(
                "capture_coordination_event() failed after decision_step allocation "
                f"({original_exc!r}) and the subsequent abort_episode() cleanup also "
                f"failed ({abort_exc!r}); the service could not be reset to a clean "
                "inactive state -- this is a double failure, not a single rejected event"
            ) from abort_exc

        raise RuntimeLearningCaptureConsistencyError(
            "capture_coordination_event() failed after decision_step allocation "
            "(materializing CriticState/GroundTruthSnapshot, opening the decision, or "
            "registering/replacing it); the episode has been aborted because "
            "allocate_many() already consumed global decision_step values that would "
            "otherwise leave an invisible, unregistered gap -- there is no rollback, "
            "start a new episode"
        ) from original_exc

    def complete_terminal_robot_decision(
        self, robot_id: int, outcome: TransitionOutcomeBatch
    ) -> LearningTransition:
        if not self._episode_session.is_active:
            raise RuntimeLearningCaptureStateError(
                "complete_terminal_robot_decision() called with no active episode"
            )
        if isinstance(robot_id, bool) or not isinstance(robot_id, int):
            raise TypeError(f"robot_id must be an int, got {type(robot_id).__name__}")
        if robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {robot_id}")
        if not self._episode_session.has_pending(robot_id):
            raise RuntimeLearningCaptureConsistencyError(
                f"robot {robot_id} has no pending decision to close"
            )
        if not isinstance(outcome, TransitionOutcomeBatch):
            raise TypeError(
                f"outcome must be a TransitionOutcomeBatch, got {type(outcome).__name__}"
            )
        if not (outcome.terminated or outcome.truncated):
            raise RuntimeLearningCaptureConsistencyError(
                "complete_terminal_robot_decision() requires a terminated or truncated "
                "outcome; a RUNNING outcome has no next_decision to open here -- use "
                "capture_coordination_event() instead"
            )

        return self._episode_session.complete_robot_decision(
            robot_id=robot_id,
            outcome=outcome,
            next_decision=None,
            next_critic_state=None,
            next_ground_truth=None,
        )

    def set_fire_metrics(self, metrics: EpisodeFireMetrics) -> None:
        if not self._episode_session.is_active:
            raise RuntimeLearningCaptureStateError(
                "set_fire_metrics() called with no active episode"
            )
        self._episode_session.set_fire_metrics(metrics)

    def finish_episode(self) -> EpisodeRecord:
        if not self._episode_session.is_active or not self._step_allocator.is_active:
            raise RuntimeLearningCaptureStateError(
                "finish_episode() requires both the episode session and the allocator to be "
                "active"
            )
        if self._episode_session.pending_count != 0:
            raise RuntimeLearningCaptureStateError(
                f"finish_episode() called with {self._episode_session.pending_count} pending "
                f"decision(s) for robot(s) {self._episode_session.pending_robot_ids}"
            )

        record = self._episode_session.finish_episode()
        self._step_allocator.finish_episode()
        return record

    def abort_episode(self) -> None:
        if not self._episode_session.is_active and not self._step_allocator.is_active:
            raise RuntimeLearningCaptureStateError("abort_episode() called with no active episode")

        session_error: Exception | None = None
        allocator_error: Exception | None = None

        if self._episode_session.is_active:
            try:
                self._episode_session.abort_episode()
            except Exception as exc:  # noqa: BLE001 -- must still try the allocator
                session_error = exc
        if self._step_allocator.is_active:
            try:
                self._step_allocator.abort_episode()
            except Exception as exc:  # noqa: BLE001 -- preserved and re-raised below
                allocator_error = exc

        if session_error is not None or allocator_error is not None:
            raise RuntimeLearningCaptureStateError(
                f"abort_episode() failed to cleanly abort every component: "
                f"session_error={session_error!r}, allocator_error={allocator_error!r}"
            ) from (session_error or allocator_error)
