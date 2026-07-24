"""Explicit inputs for the v0 feature extractors.

Only observable, non-privileged data may enter here.  The candidate input
carries a HazardBeliefFrame (discovered-only belief) -- never HazardField,
FireSource, HazardDebug, ground-truth occupancy/fire, or a
GroundTruthSnapshot.  The teammate input carries exactly one observer/
teammate pair: the caller is responsible for passing only visible or
communicated teammates, and extractors never consult a hidden global robot
list.

Normalization is fixed configuration, not computed from the current
episode: per-episode statistics would silently change a feature's meaning
between episodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robotics_interfaces.learning.candidates import CandidateObservation
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBeliefFrame
from robotics_sim.learning.builders import InvalidFeatureValueError

_SCALE_FIELDS: tuple[str, ...] = (
    "distance_scale",
    "information_gain_scale",
    "travel_cost_scale",
    "safety_cost_scale",
    "overlap_cost_scale",
    "heading_cost_scale",
    "sensor_range_scale",
    "safety_radius_scale",
)


def require_number(group: str, name: str, value: object) -> float:
    """Validate a numeric input: bool, NaN and infinity are rejected."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidFeatureValueError(group, name, value)
    if not math.isfinite(value):
        raise InvalidFeatureValueError(group, name, value)
    return float(value)


def normalize_by_scale(group: str, name: str, value: object, scale: float) -> float:
    """Divide a validated value by a validated positive scale.

    No silent clipping, no bool-to-float coercion; the result is guaranteed
    finite because the inputs are finite and the scale is positive.
    """

    numeric = require_number(group, name, value)
    result = numeric / scale
    if not math.isfinite(result):  # defensive; unreachable with valid config
        raise InvalidFeatureValueError(group, name, result)
    return result


@dataclass(frozen=True)
class FeatureNormalizationConfig:
    """Fixed, explicit normalization scales.  No scientific defaults are
    baked in: every experiment must state its scales explicitly."""

    distance_scale: float
    information_gain_scale: float
    travel_cost_scale: float
    safety_cost_scale: float
    overlap_cost_scale: float
    heading_cost_scale: float
    sensor_range_scale: float
    safety_radius_scale: float
    fire_window_radius_cells: int

    def __post_init__(self) -> None:
        for name in _SCALE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise InvalidFeatureValueError("FeatureNormalizationConfig", name, value)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(
                    f"FeatureNormalizationConfig.{name} must be finite and > 0, got {value!r}"
                )
            object.__setattr__(self, name, float(value))
        radius = self.fire_window_radius_cells
        if isinstance(radius, bool) or not isinstance(radius, int):
            raise InvalidFeatureValueError(
                "FeatureNormalizationConfig", "fire_window_radius_cells", radius
            )
        if radius < 0:
            raise ValueError(
                f"FeatureNormalizationConfig.fire_window_radius_cells must be >= 0, "
                f"got {radius}"
            )


@dataclass(frozen=True)
class RobotFeatureExtractionInput:
    robot: RobotCoordinationState
    normalization: FeatureNormalizationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.robot, RobotCoordinationState):
            raise TypeError(
                f"robot must be a RobotCoordinationState, got {type(self.robot).__name__}"
            )
        if not isinstance(self.normalization, FeatureNormalizationConfig):
            raise TypeError(
                f"normalization must be a FeatureNormalizationConfig, got "
                f"{type(self.normalization).__name__}"
            )


@dataclass(frozen=True)
class CandidateFeatureExtractionInput:
    """Observable inputs for one candidate's features.

    ``hazard_belief`` must be a HazardBeliefFrame (discovered-only belief).
    Ground-truth carriers of any kind are rejected at construction.
    """

    robot: RobotCoordinationState
    candidate: ExplorationCandidate
    candidate_observation: CandidateObservation
    hazard_belief: HazardBeliefFrame
    grid_geometry: GridGeometry
    normalization: FeatureNormalizationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.robot, RobotCoordinationState):
            raise TypeError(
                f"robot must be a RobotCoordinationState, got {type(self.robot).__name__}"
            )
        if not isinstance(self.candidate, ExplorationCandidate):
            raise TypeError(
                f"candidate must be an ExplorationCandidate, got "
                f"{type(self.candidate).__name__}"
            )
        if not isinstance(self.candidate_observation, CandidateObservation):
            raise TypeError(
                f"candidate_observation must be a CandidateObservation, got "
                f"{type(self.candidate_observation).__name__}"
            )
        if not isinstance(self.hazard_belief, HazardBeliefFrame):
            # This also rejects every ground-truth carrier (HazardField,
            # FireSource, GroundTruthSnapshot, ...) -- only the
            # discovered-only belief frame type is accepted, and this module
            # deliberately never imports any privileged type.
            raise TypeError(
                f"hazard_belief must be a HazardBeliefFrame, got "
                f"{type(self.hazard_belief).__name__}"
            )
        if not isinstance(self.grid_geometry, GridGeometry):
            raise TypeError(
                f"grid_geometry must be a GridGeometry, got "
                f"{type(self.grid_geometry).__name__}"
            )
        if not isinstance(self.normalization, FeatureNormalizationConfig):
            raise TypeError(
                f"normalization must be a FeatureNormalizationConfig, got "
                f"{type(self.normalization).__name__}"
            )
        if self.hazard_belief.observed.shape != (
            self.grid_geometry.height,
            self.grid_geometry.width,
        ):
            raise ValueError(
                f"hazard_belief shape {self.hazard_belief.observed.shape} does not match "
                f"grid geometry ({self.grid_geometry.height}, {self.grid_geometry.width})"
            )


@dataclass(frozen=True)
class TeammateFeatureExtractionInput:
    observer: RobotCoordinationState
    teammate: RobotCoordinationState
    normalization: FeatureNormalizationConfig

    def __post_init__(self) -> None:
        for name in ("observer", "teammate"):
            value = getattr(self, name)
            if not isinstance(value, RobotCoordinationState):
                raise TypeError(
                    f"{name} must be a RobotCoordinationState, got {type(value).__name__}"
                )
        if not isinstance(self.normalization, FeatureNormalizationConfig):
            raise TypeError(
                f"normalization must be a FeatureNormalizationConfig, got "
                f"{type(self.normalization).__name__}"
            )
