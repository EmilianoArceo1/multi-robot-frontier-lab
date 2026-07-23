"""Tests for the canonical v0 feature schema."""

from __future__ import annotations

from robotics_sim.learning import (
    CANDIDATE_FEATURE_NAMES_V0,
    ROBOT_FEATURE_NAMES_V0,
    TEAMMATE_FEATURE_NAMES_V0,
    build_feature_schema_v0,
)

EXPECTED_ROBOT = (
    "theta_sin",
    "theta_cos",
    "sensor_range_norm",
    "safety_radius_norm",
    "has_current_target",
    "current_target_distance_norm",
)

EXPECTED_CANDIDATE = (
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

EXPECTED_TEAMMATE = (
    "relative_x_norm",
    "relative_y_norm",
    "distance_norm",
    "theta_sin",
    "theta_cos",
    "has_current_target",
    "target_relative_x_norm",
    "target_relative_y_norm",
)


class TestFeatureSchemaV0:
    def test_exact_names_and_order(self):
        schema = build_feature_schema_v0()
        assert schema.robot_feature_names == EXPECTED_ROBOT
        assert schema.candidate_feature_names == EXPECTED_CANDIDATE
        assert schema.teammate_feature_names == EXPECTED_TEAMMATE

    def test_module_constants_match_schema(self):
        assert ROBOT_FEATURE_NAMES_V0 == EXPECTED_ROBOT
        assert CANDIDATE_FEATURE_NAMES_V0 == EXPECTED_CANDIDATE
        assert TEAMMATE_FEATURE_NAMES_V0 == EXPECTED_TEAMMATE

    def test_names_are_unique_within_each_group(self):
        schema = build_feature_schema_v0()
        for group in (
            schema.robot_feature_names,
            schema.candidate_feature_names,
            schema.teammate_feature_names,
        ):
            assert len(group) == len(set(group))

    def test_schema_is_deterministic(self):
        assert build_feature_schema_v0() == build_feature_schema_v0()

    def test_no_privileged_or_forbidden_feature_names(self):
        schema = build_feature_schema_v0()
        all_names = (
            schema.robot_feature_names
            + schema.candidate_feature_names
            + schema.teammate_feature_names
        )
        for forbidden in ("fire_risk", "ground_truth", "true_fire"):
            assert not any(forbidden in name for name in all_names), forbidden
