"""Pure v0 feature extractors: observable simulator data -> named floats.

Each extractor returns a mapping whose keys are exactly the corresponding
group of :mod:`robotics_sim.learning.feature_schema_v0`, in schema order.
All outputs are finite floats; bool, NaN and infinity inputs are rejected;
inputs are never mutated.

Privileged boundary: fire features come exclusively from the discovered-only
HazardBeliefFrame.  This module must never import or accept HazardField,
FireSource, HazardDebug, GroundTruthSnapshot, CriticState, or the engine.
An unobserved cell contributes 0.0 to fire values, but
``fire_observed_at_target``/``fire_local_observed_fraction`` let the network
distinguish "not observed" from "observed and safe" -- absence of
observation is never treated as confirmed absence of fire.

Candidate features never read ``ExplorationCandidate.metadata``.
"""

from __future__ import annotations

import math
from typing import Mapping

from robotics_interfaces.learning.candidates import CandidateKind
from robotics_sim.learning.feature_inputs import (
    CandidateFeatureExtractionInput,
    FeatureNormalizationConfig,
    RobotFeatureExtractionInput,
    TeammateFeatureExtractionInput,
    normalize_by_scale,
    require_number,
)


def _wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""

    wrapped = math.remainder(angle, math.tau)
    # math.remainder returns values in [-pi, pi]; keep the boundary stable.
    if wrapped <= -math.pi:
        wrapped += math.tau
    return wrapped


def _relative_and_distance(
    group: str,
    origin_xy: tuple[float, float],
    target_xy: tuple[float, float],
    normalization: FeatureNormalizationConfig,
) -> tuple[float, float, float]:
    ox = require_number(group, "origin_x", origin_xy[0])
    oy = require_number(group, "origin_y", origin_xy[1])
    tx = require_number(group, "target_x", target_xy[0])
    ty = require_number(group, "target_y", target_xy[1])
    dx, dy = tx - ox, ty - oy
    scale = normalization.distance_scale
    return dx / scale, dy / scale, math.hypot(dx, dy) / scale


class RobotFeatureExtractor:
    """Extracts the v0 robot feature group from RobotCoordinationState."""

    def extract(self, extraction_input: RobotFeatureExtractionInput) -> Mapping[str, float]:
        robot = extraction_input.robot
        cfg = extraction_input.normalization
        group = "robot_features"

        theta = require_number(group, "theta", robot.theta)
        has_target = robot.current_target is not None
        if has_target:
            _, _, target_distance_norm = _relative_and_distance(
                group, robot.xy, robot.current_target, cfg
            )
        else:
            target_distance_norm = 0.0

        return {
            "theta_sin": math.sin(theta),
            "theta_cos": math.cos(theta),
            "sensor_range_norm": normalize_by_scale(
                group, "sensor_range", robot.sensor_range, cfg.sensor_range_scale
            ),
            "safety_radius_norm": normalize_by_scale(
                group, "safety_radius", robot.safety_radius, cfg.safety_radius_scale
            ),
            "has_current_target": 1.0 if has_target else 0.0,
            "current_target_distance_norm": target_distance_norm,
        }


class CandidateFeatureExtractor:
    """Extracts the v0 candidate feature group.

    Fire features use only the observed hazard belief; the candidate kind
    one-hot comes only from CandidateObservation.kind (never inferred from
    ``candidate.source``).
    """

    def extract(
        self, extraction_input: CandidateFeatureExtractionInput
    ) -> Mapping[str, float]:
        robot = extraction_input.robot
        candidate = extraction_input.candidate
        observation = extraction_input.candidate_observation
        frame = extraction_input.hazard_belief
        geometry = extraction_input.grid_geometry
        cfg = extraction_input.normalization
        group = f"candidate_features[{observation.candidate_id}]"

        relative_x_norm, relative_y_norm, euclidean_distance_norm = _relative_and_distance(
            group, robot.xy, candidate.target, cfg
        )

        theta = require_number(group, "robot_theta", robot.theta)
        if candidate.heading_rad is not None:
            heading = require_number(group, "heading_rad", candidate.heading_rad)
            delta = _wrap_angle(heading - theta)
            has_heading, delta_sin, delta_cos = 1.0, math.sin(delta), math.cos(delta)
        else:
            has_heading, delta_sin, delta_cos = 0.0, 0.0, 0.0

        fire = self._fire_features(group, candidate.target, frame, geometry, cfg)

        kind = observation.kind
        return {
            "relative_x_norm": relative_x_norm,
            "relative_y_norm": relative_y_norm,
            "euclidean_distance_norm": euclidean_distance_norm,
            "information_gain_norm": normalize_by_scale(
                group, "information_gain", candidate.information_gain,
                cfg.information_gain_scale,
            ),
            "travel_cost_norm": normalize_by_scale(
                group, "travel_cost", candidate.travel_cost, cfg.travel_cost_scale
            ),
            "safety_cost_norm": normalize_by_scale(
                group, "safety_cost", candidate.safety_cost, cfg.safety_cost_scale
            ),
            "overlap_cost_norm": normalize_by_scale(
                group, "overlap_cost", candidate.overlap_cost, cfg.overlap_cost_scale
            ),
            "heading_cost_norm": normalize_by_scale(
                group, "heading_cost", candidate.heading_cost, cfg.heading_cost_scale
            ),
            "has_heading": has_heading,
            "heading_delta_sin": delta_sin,
            "heading_delta_cos": delta_cos,
            "fire_observed_at_target": fire[0],
            "fire_value_at_target": fire[1],
            "fire_local_observed_fraction": fire[2],
            "fire_local_max_value": fire[3],
            "fire_local_mean_value": fire[4],
            "frontier_kind": 1.0 if kind is CandidateKind.FRONTIER_VIEWPOINT else 0.0,
            "fire_information_kind": (
                1.0 if kind is CandidateKind.FIRE_INFORMATION_VIEWPOINT else 0.0
            ),
            "recovery_kind": 1.0 if kind is CandidateKind.RECOVERY_VIEWPOINT else 0.0,
            "hold_kind": 1.0 if kind is CandidateKind.HOLD else 0.0,
        }

    @staticmethod
    def _fire_features(
        group: str,
        target_xy: tuple[float, float],
        frame,
        geometry,
        cfg: FeatureNormalizationConfig,
    ) -> tuple[float, float, float, float, float]:
        tx = require_number(group, "target_x", target_xy[0])
        ty = require_number(group, "target_y", target_xy[1])
        cell = geometry.world_to_grid(tx, ty)
        if cell is None:
            # Target outside the grid: nothing about it has been observed.
            return 0.0, 0.0, 0.0, 0.0, 0.0

        observed_at_target = bool(frame.observed[cell.row, cell.col])
        value_at_target = (
            float(frame.values[cell.row, cell.col]) if observed_at_target else 0.0
        )

        radius = cfg.fire_window_radius_cells
        row_lo = max(cell.row - radius, 0)
        row_hi = min(cell.row + radius, geometry.height - 1)
        col_lo = max(cell.col - radius, 0)
        col_hi = min(cell.col + radius, geometry.width - 1)

        observed_window = frame.observed[row_lo : row_hi + 1, col_lo : col_hi + 1]
        values_window = frame.values[row_lo : row_hi + 1, col_lo : col_hi + 1]
        valid_cells = observed_window.size
        observed_cells = int(observed_window.sum())

        observed_fraction = observed_cells / valid_cells if valid_cells else 0.0
        if observed_cells:
            observed_values = values_window[observed_window]
            local_max = float(observed_values.max())
            local_mean = float(observed_values.mean())
        else:
            local_max = 0.0
            local_mean = 0.0

        return (
            1.0 if observed_at_target else 0.0,
            value_at_target,
            observed_fraction,
            local_max,
            local_mean,
        )


class TeammateFeatureExtractor:
    """Extracts the v0 teammate feature group for one observer/teammate pair.

    Never consults any robot beyond the explicit input.
    """

    def extract(
        self, extraction_input: TeammateFeatureExtractionInput
    ) -> Mapping[str, float]:
        observer = extraction_input.observer
        teammate = extraction_input.teammate
        cfg = extraction_input.normalization
        group = f"teammate_features[{teammate.robot_id}]"

        relative_x_norm, relative_y_norm, distance_norm = _relative_and_distance(
            group, observer.xy, teammate.xy, cfg
        )
        theta = require_number(group, "theta", teammate.theta)

        has_target = teammate.current_target is not None
        if has_target:
            target_relative_x_norm, target_relative_y_norm, _ = _relative_and_distance(
                group, observer.xy, teammate.current_target, cfg
            )
        else:
            target_relative_x_norm = 0.0
            target_relative_y_norm = 0.0

        return {
            "relative_x_norm": relative_x_norm,
            "relative_y_norm": relative_y_norm,
            "distance_norm": distance_norm,
            "theta_sin": math.sin(theta),
            "theta_cos": math.cos(theta),
            "has_current_target": 1.0 if has_target else 0.0,
            "target_relative_x_norm": target_relative_x_norm,
            "target_relative_y_norm": target_relative_y_norm,
        }
