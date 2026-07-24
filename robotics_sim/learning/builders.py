"""Pure host-side builders from source models to learning contracts.

Three small, independent builders -- no mega-class, no recording, no I/O:

- :class:`ActorObservationBuilder`: names -> ordered vectors ->
  ``ActorObservation`` (built via the public constructor so contract
  validation always runs).  It never sees ground truth.
- :class:`CriticStateBuilder`: ``CriticStateBuildInput`` -> ``CriticState``.
- :class:`GroundTruthSnapshotBuilder`: ``GroundTruthBuildInput`` ->
  ``GroundTruthSnapshot``.

Candidate order is preserved exactly as received; deterministic candidate
generation/ordering is the future CandidateGenerator's job, not the
builder's.

No Qt, numpy, torch, pandas, robotics_sim.app or engine imports.
"""

from __future__ import annotations

from typing import Mapping

from robotics_interfaces.learning.candidates import validate_action_mask
from robotics_interfaces.learning.observations import (
    ActorObservation,
    CriticState,
    GroundTruthSnapshot,
)
from robotics_interfaces.learning.versioning import OBSERVATION_SPEC_VERSION
from robotics_sim.learning.source_models import (
    ActorObservationBuildInput,
    CriticStateBuildInput,
    GroundTruthBuildInput,
)


class BuilderError(ValueError):
    """Base class for builder-level validation errors."""


class FeatureSchemaMismatchError(BuilderError):
    """A feature mapping does not match the declared schema exactly."""

    def __init__(self, group: str, expected: tuple[str, ...], received: tuple[str, ...]):
        self.group = group
        self.expected = tuple(expected)
        self.received = tuple(received)
        missing = tuple(name for name in expected if name not in set(received))
        extra = tuple(sorted(set(received) - set(expected)))
        super().__init__(
            f"feature mismatch in group {group!r}: expected keys {list(expected)}, "
            f"received keys {sorted(received)}, missing {list(missing)}, "
            f"unexpected {list(extra)}"
        )


class DuplicateCandidateIdError(BuilderError):
    """Two candidate sources share the same candidate_id."""

    def __init__(self, candidate_id: str):
        self.candidate_id = candidate_id
        super().__init__(f"duplicate candidate_id {candidate_id!r} in build input")


class InvalidFeatureValueError(BuilderError):
    """A feature value is not a plain real number (e.g. a bool)."""

    def __init__(self, group: str, feature_name: str, value: object):
        self.group = group
        self.feature_name = feature_name
        self.value = value
        super().__init__(
            f"invalid feature value in group {group!r}: feature {feature_name!r} "
            f"received {value!r} of type {type(value).__name__}; expected int or "
            f"float (bool is not accepted as a numeric feature)"
        )


def _vectorize(
    group: str, names: tuple[str, ...], features: Mapping[str, float]
) -> tuple[float, ...]:
    """Order ``features`` by ``names``; require an exact key match.

    Extra keys are an error, never silently dropped.  Insertion order of
    the mapping is irrelevant: only the schema order matters.
    """

    received = tuple(features.keys())
    if set(received) != set(names) or len(received) != len(names):
        raise FeatureSchemaMismatchError(group, names, received)
    values: list[float] = []
    for name in names:
        value = features[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidFeatureValueError(group, name, value)
        values.append(float(value))
    return tuple(values)


class ActorObservationBuilder:
    """Builds ActorObservation from policy-visible inputs only.

    ``build`` deliberately takes no ground-truth argument of any kind.
    """

    def __init__(self, schema_version: str = OBSERVATION_SPEC_VERSION):
        self._schema_version = schema_version

    def build(self, build_input: ActorObservationBuildInput) -> ActorObservation:
        schema = build_input.schema

        robot_features = _vectorize(
            "robot_features", schema.robot_feature_names, build_input.robot_features
        )

        candidate_ids: list[str] = []
        seen_ids: set[str] = set()
        candidate_rows: list[tuple[float, ...]] = []
        action_mask: list[bool] = []
        for source in build_input.candidates:
            candidate_id = source.candidate.candidate_id
            if candidate_id in seen_ids:
                raise DuplicateCandidateIdError(candidate_id)
            seen_ids.add(candidate_id)
            candidate_ids.append(candidate_id)
            candidate_rows.append(
                _vectorize(
                    f"candidate_features[{candidate_id}]",
                    schema.candidate_feature_names,
                    source.features,
                )
            )
            action_mask.append(source.enabled)

        # Contractual HOLD restriction -- reuse the contract's validator,
        # do not duplicate its rules here.
        validate_action_mask(
            tuple(source.candidate for source in build_input.candidates),
            tuple(action_mask),
        )

        teammate_rows = tuple(
            _vectorize(
                f"visible_teammate_features[{source.robot_id}]",
                schema.teammate_feature_names,
                source.features,
            )
            for source in build_input.visible_teammates
        )

        return ActorObservation(
            schema_version=self._schema_version,
            robot_id=build_input.robot_id,
            decision_step=build_input.decision_step,
            time_s=build_input.time_s,
            robot_feature_names=schema.robot_feature_names,
            robot_features=robot_features,
            candidate_feature_names=schema.candidate_feature_names,
            candidate_features=tuple(candidate_rows),
            candidate_ids=tuple(candidate_ids),
            action_mask=tuple(action_mask),
            graph_edges=build_input.graph_edges,
            visible_teammate_feature_names=schema.teammate_feature_names,
            visible_teammate_features=teammate_rows,
        )


class CriticStateBuilder:
    """Builds CriticState from CriticStateBuildInput exclusively."""

    def __init__(self, schema_version: str = OBSERVATION_SPEC_VERSION):
        self._schema_version = schema_version

    def build(self, build_input: CriticStateBuildInput) -> CriticState:
        if isinstance(build_input, GroundTruthSnapshot):
            raise TypeError("CriticStateBuilder must not receive a GroundTruthSnapshot")
        if not isinstance(build_input, CriticStateBuildInput):
            raise TypeError(
                f"build_input must be a CriticStateBuildInput, got "
                f"{type(build_input).__name__}"
            )
        for robot_id, features in build_input.per_robot_features.items():
            if isinstance(features, GroundTruthSnapshot):
                raise TypeError(
                    f"per_robot_features[{robot_id}] must not be a GroundTruthSnapshot"
                )
        global_features = _vectorize(
            "global_features", build_input.global_feature_names, build_input.global_features
        )
        per_robot = {
            robot_id: _vectorize(
                f"per_robot_features[{robot_id}]",
                build_input.per_robot_feature_names,
                features,
            )
            for robot_id, features in build_input.per_robot_features.items()
        }
        return CriticState(
            schema_version=self._schema_version,
            decision_step=build_input.decision_step,
            time_s=build_input.time_s,
            global_feature_names=build_input.global_feature_names,
            global_features=global_features,
            per_robot_feature_names=build_input.per_robot_feature_names,
            per_robot_features=per_robot,
        )


class GroundTruthSnapshotBuilder:
    """Builds GroundTruthSnapshot independently of actor and critic."""

    def __init__(self, schema_version: str = OBSERVATION_SPEC_VERSION):
        self._schema_version = schema_version

    def build(self, build_input: GroundTruthBuildInput) -> GroundTruthSnapshot:
        if not isinstance(build_input, GroundTruthBuildInput):
            raise TypeError(
                f"build_input must be a GroundTruthBuildInput, got "
                f"{type(build_input).__name__}"
            )
        return GroundTruthSnapshot(
            schema_version=self._schema_version,
            decision_step=build_input.decision_step,
            time_s=build_input.time_s,
            true_robot_poses=dict(build_input.true_robot_poses),
            true_occupancy=build_input.true_occupancy,
            true_fire_locations=build_input.true_fire_locations,
            global_coverage_fraction=build_input.global_coverage_fraction,
        )
