"""Host-side capture inputs: real runtime state, before feature extraction.

These are *capture* inputs, not learning contracts: they describe what one
decision step actually observed (real robot state, real ExplorationCandidate
objects, an observed HazardBeliefFrame, and explicitly visible teammates)
before any feature vectorization happens.
``robotics_sim.learning.observation_batch`` turns a :class:`RuntimeActorFrame`
into an ``ActorObservationBatch``.

Boundary rules:
- CandidateCaptureInput never reads ``candidate.metadata`` and never carries
  ground truth; it does not duplicate target/heading/source/information_gain
  /costs -- those stay on the wrapped ExplorationCandidate.
- CandidateKind.HOLD is rejected here: HOLD v0 is a host-side fallback, never
  a policy-selectable candidate (see robotics_interfaces.learning.candidates
  HoldPolicy).
- RobotActorCaptureInput.hazard_belief must be a HazardBeliefFrame
  (discovered-only belief); no ground-truth carrier is accepted, and the
  observing robot may never appear inside its own visible_teammates.
- RuntimeActorFrame carries no CriticState, no GroundTruthSnapshot, no real
  map, no arbitrary metadata.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  No Qt, numpy, torch, pandas, robotics_sim.app
or engine imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robotics_interfaces.learning import CandidateKind, CandidateSetSpec
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBeliefFrame
from robotics_sim.learning.feature_inputs import FeatureNormalizationConfig


@dataclass(frozen=True)
class CandidateCaptureInput:
    """One real ExplorationCandidate plus its host-side decision metadata.

    Does not duplicate target/heading/source/information_gain/costs -- those
    stay on ``candidate``.  Never reads ``candidate.metadata``.
    """

    candidate: ExplorationCandidate
    kind: CandidateKind
    enabled: bool
    reachable: bool
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, ExplorationCandidate):
            raise TypeError(
                f"candidate must be an ExplorationCandidate, got "
                f"{type(self.candidate).__name__}"
            )
        if not isinstance(self.kind, CandidateKind):
            raise TypeError(f"kind must be a CandidateKind, got {type(self.kind).__name__}")
        if self.kind is CandidateKind.HOLD:
            raise ValueError(
                "CandidateCaptureInput must not carry CandidateKind.HOLD: HOLD v0 is a "
                "host-side fallback, never a policy-selectable candidate"
            )
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")
        if not isinstance(self.reachable, bool):
            raise TypeError(f"reachable must be bool, got {type(self.reachable).__name__}")
        if not self.reachable and self.enabled:
            raise ValueError("CandidateCaptureInput must not be enabled while unreachable")

        reasons = tuple(self.rejection_reasons)
        for i, reason in enumerate(reasons):
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"rejection_reasons[{i}] must be a non-empty string, got {reason!r}"
                )
        object.__setattr__(self, "rejection_reasons", reasons)


@dataclass(frozen=True)
class RobotActorCaptureInput:
    """One robot's real state plus its real candidates for one decision step.

    ``visible_teammates`` is explicit: this module never consults a global
    robot list, and the observing robot may never appear inside its own
    teammate tuple, whether by id or by identity.
    """

    robot: RobotCoordinationState
    candidates: tuple[CandidateCaptureInput, ...]
    graph_edges: tuple[tuple[int, int], ...]
    visible_teammates: tuple[RobotCoordinationState, ...]
    hazard_belief: HazardBeliefFrame

    def __post_init__(self) -> None:
        if not isinstance(self.robot, RobotCoordinationState):
            raise TypeError(
                f"robot must be a RobotCoordinationState, got {type(self.robot).__name__}"
            )

        candidates = tuple(self.candidates)
        for i, candidate_capture in enumerate(candidates):
            if not isinstance(candidate_capture, CandidateCaptureInput):
                raise TypeError(
                    f"candidates[{i}] must be a CandidateCaptureInput, got "
                    f"{type(candidate_capture).__name__}"
                )
        object.__setattr__(self, "candidates", candidates)

        object.__setattr__(
            self, "graph_edges", tuple(tuple(edge) for edge in self.graph_edges)
        )

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
                    f"visible_teammates must not contain the observing robot's own "
                    f"robot_id {self.robot.robot_id}"
                )
            if teammate.robot_id in seen_ids:
                raise ValueError(
                    f"visible_teammates contains duplicate robot_id {teammate.robot_id}"
                )
            seen_ids.add(teammate.robot_id)
        object.__setattr__(self, "visible_teammates", teammates)

        if not isinstance(self.hazard_belief, HazardBeliefFrame):
            # This also rejects every ground-truth carrier (HazardField,
            # FireSource, GroundTruthSnapshot, ...): only the discovered-only
            # belief frame type is accepted, and this module never imports
            # any privileged type.
            raise TypeError(
                f"hazard_belief must be a HazardBeliefFrame, got "
                f"{type(self.hazard_belief).__name__}"
            )


@dataclass(frozen=True)
class RuntimeActorFrame:
    """Everything captured from the runtime for one decision step, across all
    robots, before feature extraction.

    No CriticState, no GroundTruthSnapshot, no real map, no arbitrary
    metadata.

    v0 candidate/heading semantics: one ExplorationCandidate is one
    selectable action.  A candidate may carry a single optional
    ``heading_rad`` -- never more than one.  To represent the same viewpoint
    with different headings, the candidate generator must emit distinct
    ExplorationCandidate objects; this module and
    ``ActorObservationBatchAssembler`` never create or expand headings
    themselves.  ``CandidateSetSpec.max_headings_per_candidate`` is therefore
    not exceedable by any real candidate in v0 and is not checked here.
    """

    episode_id: str
    decision_step: int
    time_s: float
    robots: tuple[RobotActorCaptureInput, ...]
    grid_geometry: GridGeometry
    normalization: FeatureNormalizationConfig
    candidate_spec: CandidateSetSpec

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError(f"episode_id must be a non-empty string, got {self.episode_id!r}")
        if isinstance(self.decision_step, bool) or not isinstance(self.decision_step, int):
            raise TypeError(
                f"decision_step must be an int, got {type(self.decision_step).__name__}"
            )
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")
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

        robots = tuple(self.robots)
        expected_shape = (self.grid_geometry.height, self.grid_geometry.width)
        seen_robot_ids: set[int] = set()
        for i, robot_capture in enumerate(robots):
            if not isinstance(robot_capture, RobotActorCaptureInput):
                raise TypeError(
                    f"robots[{i}] must be a RobotActorCaptureInput, got "
                    f"{type(robot_capture).__name__}"
                )
            robot_id = robot_capture.robot.robot_id
            if robot_id in seen_robot_ids:
                raise ValueError(f"robots contains duplicate robot_id {robot_id}")
            seen_robot_ids.add(robot_id)

            if len(robot_capture.candidates) > self.candidate_spec.max_candidates:
                raise ValueError(
                    f"robot {robot_id} has {len(robot_capture.candidates)} candidates, "
                    f"exceeding candidate_spec.max_candidates="
                    f"{self.candidate_spec.max_candidates}"
                )

            if robot_capture.hazard_belief.observed.shape != expected_shape:
                raise ValueError(
                    f"robot {robot_id} hazard_belief shape "
                    f"{robot_capture.hazard_belief.observed.shape} does not match "
                    f"grid_geometry {expected_shape}"
                )
        object.__setattr__(self, "robots", robots)
