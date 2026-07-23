"""Neutral, frozen input models for the host-side learning builders.

These are *builder inputs*, not contracts: they carry named feature
mappings plus a :class:`FeatureSchema` that fixes vector order.  The
builders in ``robotics_sim.learning.builders`` turn them into the frozen
contracts of ``robotics_interfaces.learning``.

Boundary rules:
- ActorObservationBuildInput carries only policy-visible data.  It has no
  ground truth, no true fire, no true occupancy, no critic state, no
  metadata bag, no hidden global state.
- Privileged data lives only in GroundTruthBuildInput; global training
  data lives only in CriticStateBuildInput.  The blocks stay separate.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  Nothing here may import Qt, numpy, torch,
pandas, robotics_sim.app or the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from robotics_interfaces.learning.candidates import CandidateObservation
from robotics_interfaces.learning.observations import Point2D, Pose2D


def _validate_name_group(group: str, names: tuple[str, ...]) -> tuple[str, ...]:
    names = tuple(names)
    seen: set[str] = set()
    for i, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{group}[{i}] must be a non-empty string, got {name!r}")
        if name in seen:
            raise ValueError(f"{group} contains duplicate name {name!r}")
        seen.add(name)
    return names


@dataclass(frozen=True)
class FeatureSchema:
    """Declares feature vector layout.  Declared order is preserved exactly;
    names are never sorted."""

    robot_feature_names: tuple[str, ...]
    candidate_feature_names: tuple[str, ...]
    teammate_feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "robot_feature_names",
            _validate_name_group("robot_feature_names", self.robot_feature_names),
        )
        object.__setattr__(
            self,
            "candidate_feature_names",
            _validate_name_group("candidate_feature_names", self.candidate_feature_names),
        )
        object.__setattr__(
            self,
            "teammate_feature_names",
            _validate_name_group("teammate_feature_names", self.teammate_feature_names),
        )


@dataclass(frozen=True)
class CandidateFeatureSource:
    """One candidate plus its named features and validity flag.

    Identity/geometry (candidate_id, kind, xy, headings) lives only inside
    the wrapped CandidateObservation and is not duplicated here.
    """

    candidate: CandidateObservation
    features: Mapping[str, float]
    enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateObservation):
            raise TypeError(
                f"candidate must be a CandidateObservation, got {type(self.candidate).__name__}"
            )
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")


@dataclass(frozen=True)
class TeammateFeatureSource:
    """Named features of one visible teammate."""

    robot_id: int
    features: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {self.robot_id}")


@dataclass(frozen=True)
class ActorObservationBuildInput:
    """Everything the actor builder is allowed to see.  Deliberately has no
    privileged fields and no metadata escape hatch."""

    schema: FeatureSchema
    robot_id: int
    decision_step: int
    time_s: float
    robot_features: Mapping[str, float]
    candidates: tuple[CandidateFeatureSource, ...]
    graph_edges: tuple[tuple[int, int], ...]
    visible_teammates: tuple[TeammateFeatureSource, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.schema, FeatureSchema):
            raise TypeError(f"schema must be a FeatureSchema, got {type(self.schema).__name__}")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(
            self, "graph_edges", tuple(tuple(edge) for edge in self.graph_edges)
        )
        object.__setattr__(self, "visible_teammates", tuple(self.visible_teammates))
        for i, source in enumerate(self.candidates):
            if not isinstance(source, CandidateFeatureSource):
                raise TypeError(
                    f"candidates[{i}] must be a CandidateFeatureSource, got "
                    f"{type(source).__name__}"
                )
        for i, source in enumerate(self.visible_teammates):
            if not isinstance(source, TeammateFeatureSource):
                raise TypeError(
                    f"visible_teammates[{i}] must be a TeammateFeatureSource, got "
                    f"{type(source).__name__}"
                )


@dataclass(frozen=True)
class CriticStateBuildInput:
    """Global training-time inputs matching the real CriticState signature.

    Feature values are named mappings; order comes from the name tuples.
    Must never contain a GroundTruthSnapshot -- ground truth stays in its
    own build input.
    """

    decision_step: int
    time_s: float
    global_feature_names: tuple[str, ...]
    global_features: Mapping[str, float]
    per_robot_feature_names: tuple[str, ...]
    per_robot_features: Mapping[int, Mapping[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_feature_names",
            _validate_name_group("global_feature_names", self.global_feature_names),
        )
        object.__setattr__(
            self,
            "per_robot_feature_names",
            _validate_name_group("per_robot_feature_names", self.per_robot_feature_names),
        )
        for robot_id, features in self.per_robot_features.items():
            if not isinstance(features, Mapping):
                raise TypeError(
                    f"per_robot_features[{robot_id}] must be a Mapping of feature name to "
                    f"value, got {type(features).__name__}"
                )


@dataclass(frozen=True)
class GroundTruthBuildInput:
    """Privileged inputs for GroundTruthSnapshot construction only.

    None of these fields may appear in ActorObservationBuildInput.
    """

    decision_step: int
    time_s: float
    true_robot_poses: Mapping[int, Pose2D]
    true_occupancy: tuple[tuple[int, ...], ...]
    true_fire_locations: tuple[Point2D, ...]
    global_coverage_fraction: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.global_coverage_fraction):
            raise ValueError(
                f"global_coverage_fraction must be finite, got "
                f"{self.global_coverage_fraction!r}"
            )
