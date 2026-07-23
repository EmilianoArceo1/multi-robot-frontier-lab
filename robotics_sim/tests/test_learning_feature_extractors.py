"""Tests for the v0 feature extractors and the synthetic integration with
ActorObservationBuilder."""

from __future__ import annotations

import math

import pytest

from robotics_interfaces.learning import CandidateKind, CandidateObservation
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import (
    ActorObservationBuildInput,
    ActorObservationBuilder,
    CandidateFeatureExtractionInput,
    CandidateFeatureExtractor,
    CandidateFeatureSource,
    FeatureNormalizationConfig,
    InvalidFeatureValueError,
    RobotFeatureExtractionInput,
    RobotFeatureExtractor,
    TeammateFeatureExtractionInput,
    TeammateFeatureExtractor,
    TeammateFeatureSource,
    build_feature_schema_v0,
)

SCHEMA = build_feature_schema_v0()


def make_config(**overrides) -> FeatureNormalizationConfig:
    kwargs = dict(
        distance_scale=10.0,
        information_gain_scale=5.0,
        travel_cost_scale=20.0,
        safety_cost_scale=2.0,
        overlap_cost_scale=4.0,
        heading_cost_scale=1.0,
        sensor_range_scale=8.0,
        safety_radius_scale=1.0,
        fire_window_radius_cells=1,
    )
    kwargs.update(overrides)
    return FeatureNormalizationConfig(**kwargs)


def make_robot(**overrides) -> RobotCoordinationState:
    kwargs = dict(
        robot_id=0,
        xy=(1.0, 2.0),
        safety_radius=0.5,
        sensor_range=4.0,
        vision_model="cone",
        theta=math.pi / 2,
        current_target=(4.0, 6.0),
    )
    kwargs.update(overrides)
    return RobotCoordinationState(**kwargs)


def make_geometry() -> GridGeometry:
    return GridGeometry(bounds=(0.0, 10.0, 0.0, 10.0), resolution=1.0)


def make_candidate(**overrides) -> ExplorationCandidate:
    kwargs = dict(
        target=(4.0, 6.0),
        source="frontier",
        information_gain=2.5,
        travel_cost=10.0,
        safety_cost=1.0,
        overlap_cost=2.0,
        heading_cost=0.5,
        heading_rad=math.pi,
    )
    kwargs.update(overrides)
    return ExplorationCandidate(**kwargs)


def make_candidate_observation(
    kind: CandidateKind = CandidateKind.FRONTIER_VIEWPOINT,
    xy: tuple[float, float] = (4.0, 6.0),
) -> CandidateObservation:
    return CandidateObservation(
        candidate_id="c0",
        kind=kind,
        xy=xy,
        heading_candidates=(0.0,),
        source="frontier",
        reachable=True,
    )


def make_candidate_input(
    belief: HazardBelief | None = None,
    candidate: ExplorationCandidate | None = None,
    observation: CandidateObservation | None = None,
    config: FeatureNormalizationConfig | None = None,
    robot: RobotCoordinationState | None = None,
) -> CandidateFeatureExtractionInput:
    geometry = make_geometry()
    if belief is None:
        belief = HazardBelief(geometry)
    return CandidateFeatureExtractionInput(
        robot=robot if robot is not None else make_robot(),
        candidate=candidate if candidate is not None else make_candidate(),
        candidate_observation=observation if observation is not None else make_candidate_observation(),
        hazard_belief=belief.snapshot(),
        grid_geometry=geometry,
        normalization=config if config is not None else make_config(),
    )


class TestNormalizationConfig:
    def test_rejects_non_positive_and_non_finite_scales(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                make_config(distance_scale=bad)

    def test_rejects_bool_scale_and_bool_radius(self):
        with pytest.raises(InvalidFeatureValueError):
            make_config(travel_cost_scale=True)
        with pytest.raises(InvalidFeatureValueError):
            make_config(fire_window_radius_cells=True)

    def test_rejects_negative_or_float_radius(self):
        with pytest.raises(ValueError):
            make_config(fire_window_radius_cells=-1)
        with pytest.raises(InvalidFeatureValueError):
            make_config(fire_window_radius_cells=1.5)


class TestRobotFeatures:
    def test_values_and_order(self):
        features = RobotFeatureExtractor().extract(
            RobotFeatureExtractionInput(robot=make_robot(), normalization=make_config())
        )
        assert tuple(features.keys()) == SCHEMA.robot_feature_names
        assert features["theta_sin"] == pytest.approx(1.0)
        assert features["theta_cos"] == pytest.approx(0.0, abs=1e-12)
        assert features["sensor_range_norm"] == pytest.approx(0.5)
        assert features["safety_radius_norm"] == pytest.approx(0.5)
        assert features["has_current_target"] == 1.0
        # |(4,6) - (1,2)| = 5 -> / 10
        assert features["current_target_distance_norm"] == pytest.approx(0.5)

    def test_without_target(self):
        features = RobotFeatureExtractor().extract(
            RobotFeatureExtractionInput(
                robot=make_robot(current_target=None), normalization=make_config()
            )
        )
        assert features["has_current_target"] == 0.0
        assert features["current_target_distance_norm"] == 0.0

    def test_rejects_bool_theta(self):
        with pytest.raises(InvalidFeatureValueError):
            RobotFeatureExtractor().extract(
                RobotFeatureExtractionInput(
                    robot=make_robot(theta=True), normalization=make_config()
                )
            )

    def test_is_deterministic(self):
        extraction_input = RobotFeatureExtractionInput(
            robot=make_robot(), normalization=make_config()
        )
        extractor = RobotFeatureExtractor()
        assert extractor.extract(extraction_input) == extractor.extract(extraction_input)


class TestCandidateGeometryAndCosts:
    def test_relative_geometry_and_normalized_costs(self):
        features = CandidateFeatureExtractor().extract(make_candidate_input())
        assert tuple(features.keys()) == SCHEMA.candidate_feature_names
        assert features["relative_x_norm"] == pytest.approx(0.3)
        assert features["relative_y_norm"] == pytest.approx(0.4)
        assert features["euclidean_distance_norm"] == pytest.approx(0.5)
        assert features["information_gain_norm"] == pytest.approx(0.5)
        assert features["travel_cost_norm"] == pytest.approx(0.5)
        assert features["safety_cost_norm"] == pytest.approx(0.5)
        assert features["overlap_cost_norm"] == pytest.approx(0.5)
        assert features["heading_cost_norm"] == pytest.approx(0.5)

    def test_candidate_with_heading(self):
        features = CandidateFeatureExtractor().extract(make_candidate_input())
        # delta = pi - pi/2 = pi/2
        assert features["has_heading"] == 1.0
        assert features["heading_delta_sin"] == pytest.approx(1.0)
        assert features["heading_delta_cos"] == pytest.approx(0.0, abs=1e-12)

    def test_candidate_without_heading(self):
        features = CandidateFeatureExtractor().extract(
            make_candidate_input(candidate=make_candidate(heading_rad=None))
        )
        assert features["has_heading"] == 0.0
        assert features["heading_delta_sin"] == 0.0
        assert features["heading_delta_cos"] == 0.0

    def test_heading_delta_wraps_to_pi_range(self):
        features = CandidateFeatureExtractor().extract(
            make_candidate_input(
                candidate=make_candidate(heading_rad=math.pi * 1.5),
                robot=make_robot(theta=-math.pi * 0.75),
            )
        )
        # raw delta = 2.25*pi -> wrapped 0.25*pi
        assert features["heading_delta_sin"] == pytest.approx(math.sin(0.25 * math.pi))
        assert features["heading_delta_cos"] == pytest.approx(math.cos(0.25 * math.pi))

    @pytest.mark.parametrize(
        "kind, hot",
        [
            (CandidateKind.FRONTIER_VIEWPOINT, "frontier_kind"),
            (CandidateKind.FIRE_INFORMATION_VIEWPOINT, "fire_information_kind"),
            (CandidateKind.RECOVERY_VIEWPOINT, "recovery_kind"),
            (CandidateKind.HOLD, "hold_kind"),
        ],
    )
    def test_kind_one_hot(self, kind, hot):
        features = CandidateFeatureExtractor().extract(
            make_candidate_input(observation=make_candidate_observation(kind=kind))
        )
        one_hot = {name: features[name] for name in
                   ("frontier_kind", "fire_information_kind", "recovery_kind", "hold_kind")}
        assert one_hot.pop(hot) == 1.0
        assert all(v == 0.0 for v in one_hot.values())

    def test_rejects_nan_and_inf_costs_and_bool(self):
        with pytest.raises(InvalidFeatureValueError):
            CandidateFeatureExtractor().extract(
                make_candidate_input(candidate=make_candidate(travel_cost=float("nan")))
            )
        with pytest.raises(InvalidFeatureValueError):
            CandidateFeatureExtractor().extract(
                make_candidate_input(candidate=make_candidate(information_gain=float("inf")))
            )
        with pytest.raises(InvalidFeatureValueError):
            CandidateFeatureExtractor().extract(
                make_candidate_input(candidate=make_candidate(safety_cost=True))
            )

    def test_all_outputs_are_finite_floats(self):
        features = CandidateFeatureExtractor().extract(make_candidate_input())
        for name, value in features.items():
            assert isinstance(value, float) and not isinstance(value, bool), name
            assert math.isfinite(value), name


class TestCandidateFireFeatures:
    def test_no_cell_observed(self):
        features = CandidateFeatureExtractor().extract(make_candidate_input())
        assert features["fire_observed_at_target"] == 0.0
        assert features["fire_value_at_target"] == 0.0
        assert features["fire_local_observed_fraction"] == 0.0
        assert features["fire_local_max_value"] == 0.0
        assert features["fire_local_mean_value"] == 0.0

    def test_observed_safe_cells(self):
        belief = HazardBelief(make_geometry())
        # Target (4.0, 6.0) -> row 6, col 4; observe it and one neighbor, both safe.
        belief.observe_cells([6, 5], [4, 4], [0.0, 0.0], robot_index=0)
        features = CandidateFeatureExtractor().extract(make_candidate_input(belief=belief))
        assert features["fire_observed_at_target"] == 1.0
        assert features["fire_value_at_target"] == 0.0
        assert features["fire_local_observed_fraction"] == pytest.approx(2 / 9)
        assert features["fire_local_max_value"] == 0.0
        assert features["fire_local_mean_value"] == 0.0

    def test_fire_observed_in_centered_window(self):
        belief = HazardBelief(make_geometry())
        # Window (radius 1) around (6, 4): rows 5..7, cols 3..5 -> 9 cells.
        belief.observe_cells([6, 5, 7], [4, 3, 5], [0.8, 0.4, 0.0], robot_index=0)
        features = CandidateFeatureExtractor().extract(make_candidate_input(belief=belief))
        assert features["fire_observed_at_target"] == 1.0
        assert features["fire_value_at_target"] == pytest.approx(0.8, rel=1e-6)
        assert features["fire_local_observed_fraction"] == pytest.approx(3 / 9)
        assert features["fire_local_max_value"] == pytest.approx(0.8, rel=1e-6)
        assert features["fire_local_mean_value"] == pytest.approx(0.4, rel=1e-6)

    def test_window_clipped_at_grid_corner(self):
        belief = HazardBelief(make_geometry())
        # Candidate at (0.5, 0.5) -> cell (0, 0); radius-1 window clips to
        # rows 0..1, cols 0..1 -> 4 valid cells.
        belief.observe_cells([0], [0], [0.6], robot_index=0)
        features = CandidateFeatureExtractor().extract(
            make_candidate_input(
                belief=belief,
                candidate=make_candidate(target=(0.5, 0.5)),
                observation=make_candidate_observation(xy=(0.5, 0.5)),
            )
        )
        assert features["fire_observed_at_target"] == 1.0
        assert features["fire_local_observed_fraction"] == pytest.approx(1 / 4)
        assert features["fire_local_max_value"] == pytest.approx(0.6, rel=1e-6)

    def test_unobserved_cell_distinguishable_from_observed_safe(self):
        unobserved = CandidateFeatureExtractor().extract(make_candidate_input())
        safe_belief = HazardBelief(make_geometry())
        safe_belief.observe_cells([6], [4], [0.0], robot_index=0)
        observed_safe = CandidateFeatureExtractor().extract(
            make_candidate_input(belief=safe_belief)
        )
        # Same fire *value*, different observability flags.
        assert unobserved["fire_value_at_target"] == observed_safe["fire_value_at_target"]
        assert unobserved["fire_observed_at_target"] == 0.0
        assert observed_safe["fire_observed_at_target"] == 1.0


class TestTeammateFeatures:
    def test_with_target(self):
        features = TeammateFeatureExtractor().extract(
            TeammateFeatureExtractionInput(
                observer=make_robot(xy=(0.0, 0.0), current_target=None),
                teammate=make_robot(
                    robot_id=1, xy=(3.0, 4.0), theta=0.0, current_target=(6.0, 8.0)
                ),
                normalization=make_config(),
            )
        )
        assert tuple(features.keys()) == SCHEMA.teammate_feature_names
        assert features["relative_x_norm"] == pytest.approx(0.3)
        assert features["relative_y_norm"] == pytest.approx(0.4)
        assert features["distance_norm"] == pytest.approx(0.5)
        assert features["theta_sin"] == pytest.approx(0.0)
        assert features["theta_cos"] == pytest.approx(1.0)
        assert features["has_current_target"] == 1.0
        assert features["target_relative_x_norm"] == pytest.approx(0.6)
        assert features["target_relative_y_norm"] == pytest.approx(0.8)

    def test_without_target(self):
        features = TeammateFeatureExtractor().extract(
            TeammateFeatureExtractionInput(
                observer=make_robot(xy=(0.0, 0.0)),
                teammate=make_robot(robot_id=1, xy=(3.0, 4.0), current_target=None),
                normalization=make_config(),
            )
        )
        assert features["has_current_target"] == 0.0
        assert features["target_relative_x_norm"] == 0.0
        assert features["target_relative_y_norm"] == 0.0

    def test_rejects_nan_theta(self):
        with pytest.raises(InvalidFeatureValueError):
            TeammateFeatureExtractor().extract(
                TeammateFeatureExtractionInput(
                    observer=make_robot(),
                    teammate=make_robot(robot_id=1, theta=float("nan")),
                    normalization=make_config(),
                )
            )


class TestSyntheticBuilderIntegration:
    """Full observable pipeline: coordination state + candidates ->
    extractors -> ActorObservationBuilder -> valid ActorObservation.
    No engine, no GUI, no filesystem."""

    def test_extractors_feed_actor_observation_builder(self):
        geometry = make_geometry()
        belief = HazardBelief(geometry)
        belief.observe_cells([6], [4], [0.7], robot_index=0)
        frame = belief.snapshot()
        config = make_config()
        robot = make_robot()
        teammate = make_robot(robot_id=1, xy=(8.0, 8.0), current_target=None)

        candidates = (
            make_candidate(target=(4.0, 6.0)),
            make_candidate(target=(2.0, 2.0), heading_rad=None, travel_cost=4.0),
        )
        observations = (
            CandidateObservation(
                candidate_id="cand-a", kind=CandidateKind.FRONTIER_VIEWPOINT,
                xy=(4.0, 6.0), heading_candidates=(0.0,), source="frontier",
                reachable=True,
            ),
            CandidateObservation(
                candidate_id="cand-b", kind=CandidateKind.FIRE_INFORMATION_VIEWPOINT,
                xy=(2.0, 2.0), heading_candidates=(), source="fire_information",
                reachable=True,
            ),
        )

        candidate_extractor = CandidateFeatureExtractor()
        sources = tuple(
            CandidateFeatureSource(
                candidate=obs,
                features=candidate_extractor.extract(
                    CandidateFeatureExtractionInput(
                        robot=robot,
                        candidate=cand,
                        candidate_observation=obs,
                        hazard_belief=frame,
                        grid_geometry=geometry,
                        normalization=config,
                    )
                ),
                enabled=True,
            )
            for cand, obs in zip(candidates, observations)
        )

        build_input = ActorObservationBuildInput(
            schema=build_feature_schema_v0(),
            robot_id=robot.robot_id,
            decision_step=0,
            time_s=0.0,
            robot_features=RobotFeatureExtractor().extract(
                RobotFeatureExtractionInput(robot=robot, normalization=config)
            ),
            candidates=sources,
            graph_edges=((0, 1),),
            visible_teammates=(
                TeammateFeatureSource(
                    robot_id=teammate.robot_id,
                    features=TeammateFeatureExtractor().extract(
                        TeammateFeatureExtractionInput(
                            observer=robot, teammate=teammate, normalization=config
                        )
                    ),
                ),
            ),
        )
        actor_observation = ActorObservationBuilder().build(build_input)

        assert actor_observation.candidate_ids == ("cand-a", "cand-b")
        assert actor_observation.action_mask == (True, True)
        assert actor_observation.robot_feature_names == SCHEMA.robot_feature_names
        assert actor_observation.candidate_feature_names == SCHEMA.candidate_feature_names
        row_a = dict(zip(actor_observation.candidate_feature_names,
                         actor_observation.candidate_features[0]))
        assert row_a["fire_observed_at_target"] == 1.0
        assert row_a["fire_value_at_target"] == pytest.approx(0.7, rel=1e-6)
        assert row_a["frontier_kind"] == 1.0
        row_b = dict(zip(actor_observation.candidate_feature_names,
                         actor_observation.candidate_features[1]))
        assert row_b["has_heading"] == 0.0
        assert row_b["fire_information_kind"] == 1.0
