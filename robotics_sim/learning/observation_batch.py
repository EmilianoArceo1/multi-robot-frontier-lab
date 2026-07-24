"""Pure host-side assembly: RuntimeActorFrame -> ActorObservationBatch.

``ActorObservationBatchAssembler`` is a 1:1 transform, not a planner: it
never creates, sorts, prunes, or truncates candidates, never computes
rewards or picks actions, and never touches ``ExplorationCandidate.metadata``
or any ground-truth type.  Candidate order is preserved exactly as received;
deterministic candidate generation/ordering belongs to the future
CandidateGenerator, not here.

Candidate identity (:func:`build_candidate_id`) is positional and local to
one decision: ``robot-{robot_id}/step-{decision_step}/candidate-{index}``,
where ``index`` is the candidate's position in the tuple it was received in.
It is never a spatial or persistent identity across steps.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  No Qt, numpy, torch, pandas, robotics_sim.app
or engine imports.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotics_interfaces.learning import CandidateObservation, CandidateSetSpec
from robotics_interfaces.learning.observations import ActorObservation
from robotics_sim.learning.builders import ActorObservationBuilder
from robotics_sim.learning.capture_inputs import (
    CandidateCaptureInput,
    RobotActorCaptureInput,
    RuntimeActorFrame,
)
from robotics_sim.learning.feature_extractors import (
    CandidateFeatureExtractor,
    RobotFeatureExtractor,
    TeammateFeatureExtractor,
)
from robotics_sim.learning.feature_inputs import (
    CandidateFeatureExtractionInput,
    RobotFeatureExtractionInput,
    TeammateFeatureExtractionInput,
)
from robotics_sim.learning.source_models import (
    ActorObservationBuildInput,
    CandidateFeatureSource,
    FeatureSchema,
    TeammateFeatureSource,
)


def build_candidate_id(robot_id: int, decision_step: int, candidate_index: int) -> str:
    """Deterministic, position-based candidate id, local to one decision.

    Format: ``robot-{robot_id}/step-{decision_step}/candidate-{candidate_index}``.
    ``candidate_index`` is the candidate's position in the tuple it was
    received in -- never a spatial or persistent identity.  Never uses
    ``hash()``, floating-point coordinates, or ``candidate.metadata``.
    """

    for name, value in (
        ("robot_id", robot_id),
        ("decision_step", decision_step),
        ("candidate_index", candidate_index),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    return f"robot-{robot_id}/step-{decision_step}/candidate-{candidate_index}"


def _build_candidate_observation(
    candidate_capture: CandidateCaptureInput, candidate_id: str
) -> CandidateObservation:
    """Wrap one real ExplorationCandidate as a CandidateObservation.

    Never invents headings: ``heading_candidates`` is a single-entry tuple
    when ``candidate.heading_rad`` is set, empty otherwise.  ``kind`` comes
    only from ``candidate_capture.kind`` -- never inferred from
    ``candidate.source``.
    """

    candidate = candidate_capture.candidate
    heading_candidates = () if candidate.heading_rad is None else (candidate.heading_rad,)
    return CandidateObservation(
        candidate_id=candidate_id,
        kind=candidate_capture.kind,
        xy=candidate.target,
        heading_candidates=heading_candidates,
        source=candidate.source,
        reachable=candidate_capture.reachable,
        rejection_reasons=candidate_capture.rejection_reasons,
    )


@dataclass(frozen=True)
class ActorObservationBatch:
    """One ActorObservation per robot for one decision step.

    Carries only the finished, validated ActorObservation contracts -- no
    RuntimeActorFrame, no HazardBeliefFrame, no other runtime input, and no
    mutable objects.
    """

    episode_id: str
    decision_step: int
    time_s: float
    observations: tuple[ActorObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError(f"episode_id must be a non-empty string, got {self.episode_id!r}")
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")

        observations = tuple(self.observations)
        seen_robot_ids: set[int] = set()
        for i, observation in enumerate(observations):
            if not isinstance(observation, ActorObservation):
                raise TypeError(
                    f"observations[{i}] must be an ActorObservation, got "
                    f"{type(observation).__name__}"
                )
            if observation.decision_step != self.decision_step:
                raise ValueError(
                    f"observations[{i}].decision_step={observation.decision_step} does not "
                    f"match batch decision_step={self.decision_step}"
                )
            if observation.robot_id in seen_robot_ids:
                raise ValueError(
                    f"observations contains duplicate robot_id {observation.robot_id}"
                )
            seen_robot_ids.add(observation.robot_id)
        object.__setattr__(self, "observations", observations)

    def get_for_robot(self, robot_id: int) -> ActorObservation:
        """Return the observation for ``robot_id``.

        Raises explicitly -- never returns None -- when the robot is missing
        or (defensively; __post_init__ already forbids it) duplicated.
        """

        matches = tuple(o for o in self.observations if o.robot_id == robot_id)
        if not matches:
            raise KeyError(f"no ActorObservation for robot_id {robot_id}")
        if len(matches) > 1:
            raise ValueError(f"multiple ActorObservation entries for robot_id {robot_id}")
        return matches[0]


class ActorObservationBatchAssembler:
    """Pure host-side assembler: RuntimeActorFrame -> ActorObservationBatch.

    Reuses RobotFeatureExtractor, CandidateFeatureExtractor,
    TeammateFeatureExtractor and ActorObservationBuilder; it never re-derives
    their logic, never reads ``candidate.metadata``, never consults
    HazardField/FireSource/the engine, never searches for robots beyond what
    the frame provides, and never creates, sorts, or truncates candidates.
    """

    def __init__(self, schema: FeatureSchema, candidate_spec: CandidateSetSpec) -> None:
        if not isinstance(schema, FeatureSchema):
            raise TypeError(f"schema must be a FeatureSchema, got {type(schema).__name__}")
        if not isinstance(candidate_spec, CandidateSetSpec):
            raise TypeError(
                f"candidate_spec must be a CandidateSetSpec, got "
                f"{type(candidate_spec).__name__}"
            )
        self._schema = schema
        self._candidate_spec = candidate_spec
        self._robot_extractor = RobotFeatureExtractor()
        self._candidate_extractor = CandidateFeatureExtractor()
        self._teammate_extractor = TeammateFeatureExtractor()
        self._observation_builder = ActorObservationBuilder()

    def build(self, frame: RuntimeActorFrame) -> ActorObservationBatch:
        if not isinstance(frame, RuntimeActorFrame):
            raise TypeError(f"frame must be a RuntimeActorFrame, got {type(frame).__name__}")

        observations = tuple(self._build_one(frame, robot_capture) for robot_capture in frame.robots)

        return ActorObservationBatch(
            episode_id=frame.episode_id,
            decision_step=frame.decision_step,
            time_s=frame.time_s,
            observations=observations,
        )

    def _build_one(
        self, frame: RuntimeActorFrame, robot_capture: RobotActorCaptureInput
    ) -> ActorObservation:
        robot_id = robot_capture.robot.robot_id

        robot_features = self._robot_extractor.extract(
            RobotFeatureExtractionInput(robot=robot_capture.robot, normalization=frame.normalization)
        )

        candidate_sources: list[CandidateFeatureSource] = []
        for index, candidate_capture in enumerate(robot_capture.candidates):
            candidate_id = build_candidate_id(robot_id, frame.decision_step, index)
            candidate_observation = _build_candidate_observation(candidate_capture, candidate_id)
            features = self._candidate_extractor.extract(
                CandidateFeatureExtractionInput(
                    robot=robot_capture.robot,
                    candidate=candidate_capture.candidate,
                    candidate_observation=candidate_observation,
                    hazard_belief=robot_capture.hazard_belief,
                    grid_geometry=frame.grid_geometry,
                    normalization=frame.normalization,
                )
            )
            candidate_sources.append(
                CandidateFeatureSource(
                    candidate=candidate_observation,
                    features=features,
                    enabled=candidate_capture.enabled,
                )
            )

        teammate_sources = tuple(
            TeammateFeatureSource(
                robot_id=teammate.robot_id,
                features=self._teammate_extractor.extract(
                    TeammateFeatureExtractionInput(
                        observer=robot_capture.robot,
                        teammate=teammate,
                        normalization=frame.normalization,
                    )
                ),
            )
            for teammate in robot_capture.visible_teammates
        )

        build_input = ActorObservationBuildInput(
            schema=self._schema,
            robot_id=robot_id,
            decision_step=frame.decision_step,
            time_s=frame.time_s,
            robot_features=robot_features,
            candidates=tuple(candidate_sources),
            graph_edges=robot_capture.graph_edges,
            visible_teammates=teammate_sources,
        )
        return self._observation_builder.build(build_input)
