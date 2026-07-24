"""FeatureSchema v0 for the fire_search_v0 task.

The exact names *and order* below are part of the learning contract: tensor
columns downstream are identified positionally, so this module uses literal
tuples only -- never mappings -- and must not be reordered without bumping
the schema version.

There is deliberately no ``fire_risk`` feature: fire is traversable and has
no navigation cost in fire_search_v0.  All fire features describe *observed
belief* (HazardBeliefFrame), never ground truth.
"""

from __future__ import annotations

from robotics_sim.learning.source_models import FeatureSchema

ROBOT_FEATURE_NAMES_V0: tuple[str, ...] = (
    "theta_sin",
    "theta_cos",
    "sensor_range_norm",
    "safety_radius_norm",
    "has_current_target",
    "current_target_distance_norm",
)

CANDIDATE_FEATURE_NAMES_V0: tuple[str, ...] = (
    "relative_x_norm",
    "relative_y_norm",
    "euclidean_distance_norm",
    "information_gain_norm",
    "travel_cost_norm",
    "safety_cost_norm",
    "overlap_cost_norm",
    "heading_cost_norm",
    "has_heading",
    "heading_delta_sin",
    "heading_delta_cos",
    "fire_observed_at_target",
    "fire_value_at_target",
    "fire_local_observed_fraction",
    "fire_local_max_value",
    "fire_local_mean_value",
    "frontier_kind",
    "fire_information_kind",
    "recovery_kind",
    "hold_kind",
)

TEAMMATE_FEATURE_NAMES_V0: tuple[str, ...] = (
    "relative_x_norm",
    "relative_y_norm",
    "distance_norm",
    "theta_sin",
    "theta_cos",
    "has_current_target",
    "target_relative_x_norm",
    "target_relative_y_norm",
)


def build_feature_schema_v0() -> FeatureSchema:
    """Return the canonical v0 FeatureSchema (deterministic, literal order)."""

    return FeatureSchema(
        robot_feature_names=ROBOT_FEATURE_NAMES_V0,
        candidate_feature_names=CANDIDATE_FEATURE_NAMES_V0,
        teammate_feature_names=TEAMMATE_FEATURE_NAMES_V0,
    )
