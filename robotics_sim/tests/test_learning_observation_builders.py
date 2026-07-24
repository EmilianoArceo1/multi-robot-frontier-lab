"""Tests for the host-side learning observation builders."""

from __future__ import annotations

import pytest

from robotics_interfaces.learning import CandidateKind, CandidateObservation
from robotics_sim.learning import (
    ActorObservationBuildInput,
    ActorObservationBuilder,
    CandidateFeatureSource,
    CriticStateBuildInput,
    CriticStateBuilder,
    DuplicateCandidateIdError,
    FeatureSchema,
    FeatureSchemaMismatchError,
    InvalidFeatureValueError,
    TeammateFeatureSource,
)

SCHEMA = FeatureSchema(
    robot_feature_names=("x", "y", "battery"),
    candidate_feature_names=("dist", "gain"),
    teammate_feature_names=("rel_x", "rel_y"),
)


def make_candidate(
    candidate_id: str, kind: CandidateKind = CandidateKind.FRONTIER_VIEWPOINT
) -> CandidateObservation:
    return CandidateObservation(
        candidate_id=candidate_id,
        kind=kind,
        xy=(1.0, 2.0),
        heading_candidates=(0.0,),
        source="frontier",
        reachable=True,
    )


def make_input(**overrides) -> ActorObservationBuildInput:
    kwargs = dict(
        schema=SCHEMA,
        robot_id=0,
        decision_step=1,
        time_s=0.5,
        robot_features={"x": 1.0, "y": 2.0, "battery": 0.9},
        candidates=(
            CandidateFeatureSource(
                candidate=make_candidate("c0"), features={"dist": 1.0, "gain": 0.5}, enabled=True
            ),
            CandidateFeatureSource(
                candidate=make_candidate("c1"), features={"dist": 2.0, "gain": 0.2}, enabled=False
            ),
        ),
        graph_edges=((0, 1),),
        visible_teammates=(
            TeammateFeatureSource(robot_id=1, features={"rel_x": 0.5, "rel_y": -0.5}),
        ),
    )
    kwargs.update(overrides)
    return ActorObservationBuildInput(**kwargs)


class TestFeatureOrdering:
    def test_vectors_follow_schema_order_not_mapping_order(self):
        obs = ActorObservationBuilder().build(
            make_input(robot_features={"battery": 0.9, "y": 2.0, "x": 1.0})
        )
        assert obs.robot_feature_names == ("x", "y", "battery")
        assert obs.robot_features == (1.0, 2.0, 0.9)

    def test_different_insertion_orders_give_same_observation(self):
        a = ActorObservationBuilder().build(
            make_input(robot_features={"x": 1.0, "y": 2.0, "battery": 0.9})
        )
        b = ActorObservationBuilder().build(
            make_input(robot_features={"battery": 0.9, "x": 1.0, "y": 2.0})
        )
        assert a == b

    def test_schema_rejects_duplicate_and_empty_names(self):
        with pytest.raises(ValueError):
            FeatureSchema(("x", "x"), ("d",), ("t",))
        with pytest.raises(ValueError):
            FeatureSchema(("x",), ("",), ("t",))

    def test_schema_preserves_declared_order(self):
        schema = FeatureSchema(("z", "a", "m"), ("d",), ("t",))
        assert schema.robot_feature_names == ("z", "a", "m")


class TestSchemaMismatch:
    def test_missing_robot_feature_key_raises(self):
        with pytest.raises(FeatureSchemaMismatchError) as excinfo:
            ActorObservationBuilder().build(make_input(robot_features={"x": 1.0, "y": 2.0}))
        assert excinfo.value.group == "robot_features"
        assert excinfo.value.expected == ("x", "y", "battery")
        assert set(excinfo.value.received) == {"x", "y"}

    def test_extra_robot_feature_key_raises(self):
        with pytest.raises(FeatureSchemaMismatchError):
            ActorObservationBuilder().build(
                make_input(
                    robot_features={"x": 1.0, "y": 2.0, "battery": 0.9, "sneaky": 3.0}
                )
            )

    def test_candidate_feature_mismatch_raises_with_group(self):
        bad = (
            CandidateFeatureSource(
                candidate=make_candidate("c0"), features={"dist": 1.0}, enabled=True
            ),
        )
        with pytest.raises(FeatureSchemaMismatchError) as excinfo:
            ActorObservationBuilder().build(make_input(candidates=bad, graph_edges=()))
        assert "c0" in excinfo.value.group

    def test_teammate_feature_mismatch_raises(self):
        bad = (TeammateFeatureSource(robot_id=1, features={"rel_x": 0.5}),)
        with pytest.raises(FeatureSchemaMismatchError):
            ActorObservationBuilder().build(make_input(visible_teammates=bad))


class TestCandidateHandling:
    def test_duplicate_candidate_id_raises(self):
        dupes = (
            CandidateFeatureSource(
                candidate=make_candidate("c0"), features={"dist": 1.0, "gain": 0.5}, enabled=True
            ),
            CandidateFeatureSource(
                candidate=make_candidate("c0"), features={"dist": 2.0, "gain": 0.2}, enabled=False
            ),
        )
        with pytest.raises(DuplicateCandidateIdError) as excinfo:
            ActorObservationBuilder().build(make_input(candidates=dupes))
        assert excinfo.value.candidate_id == "c0"

    def test_candidate_order_is_preserved(self):
        ordered = tuple(
            CandidateFeatureSource(
                candidate=make_candidate(f"c{i}"),
                features={"dist": float(i), "gain": 0.0},
                enabled=True,
            )
            for i in (3, 0, 2, 1)
        )
        obs = ActorObservationBuilder().build(
            make_input(candidates=ordered, graph_edges=())
        )
        assert obs.candidate_ids == ("c3", "c0", "c2", "c1")
        assert tuple(row[0] for row in obs.candidate_features) == (3.0, 0.0, 2.0, 1.0)

    def test_action_mask_reflects_enabled_flags(self):
        obs = ActorObservationBuilder().build(make_input())
        assert obs.action_mask == (True, False)


class TestHoldRestriction:
    def _hold_source(self, enabled: bool) -> CandidateFeatureSource:
        return CandidateFeatureSource(
            candidate=make_candidate("h0", kind=CandidateKind.HOLD),
            features={"dist": 0.0, "gain": 0.0},
            enabled=enabled,
        )

    def _frontier_source(self, enabled: bool) -> CandidateFeatureSource:
        return CandidateFeatureSource(
            candidate=make_candidate("c0"),
            features={"dist": 1.0, "gain": 0.5},
            enabled=enabled,
        )

    def test_hold_enabled_alongside_valid_action_raises(self):
        with pytest.raises(ValueError):
            ActorObservationBuilder().build(
                make_input(
                    candidates=(self._frontier_source(True), self._hold_source(True)),
                    graph_edges=(),
                )
            )

    def test_hold_alone_as_fallback_is_valid(self):
        obs = ActorObservationBuilder().build(
            make_input(
                candidates=(self._frontier_source(False), self._hold_source(True)),
                graph_edges=(),
            )
        )
        assert obs.action_mask == (False, True)


class TestContractValidationStillApplies:
    def test_invalid_graph_edges_rejected_by_actor_observation(self):
        with pytest.raises(ValueError):
            ActorObservationBuilder().build(make_input(graph_edges=((0, 5),)))

    def test_nan_feature_rejected_by_contract(self):
        with pytest.raises(ValueError):
            ActorObservationBuilder().build(
                make_input(robot_features={"x": float("nan"), "y": 2.0, "battery": 0.9})
            )


class TestTeammates:
    def test_zero_teammates(self):
        obs = ActorObservationBuilder().build(make_input(visible_teammates=()))
        assert obs.visible_teammate_features == ()
        assert obs.visible_teammate_feature_names == ("rel_x", "rel_y")

    def test_multiple_teammates(self):
        teammates = (
            TeammateFeatureSource(robot_id=1, features={"rel_x": 0.5, "rel_y": -0.5}),
            TeammateFeatureSource(robot_id=2, features={"rel_y": 1.5, "rel_x": -1.0}),
        )
        obs = ActorObservationBuilder().build(make_input(visible_teammates=teammates))
        assert obs.visible_teammate_features == ((0.5, -0.5), (-1.0, 1.5))


class TestBooleanFeatureRejection:
    def test_bool_in_robot_features_raises(self):
        with pytest.raises(InvalidFeatureValueError) as excinfo:
            ActorObservationBuilder().build(
                make_input(robot_features={"x": True, "y": 2.0, "battery": 0.9})
            )
        assert excinfo.value.group == "robot_features"
        assert excinfo.value.feature_name == "x"
        assert excinfo.value.value is True
        assert "bool" in str(excinfo.value)

    def test_bool_in_candidate_features_raises(self):
        bad = (
            CandidateFeatureSource(
                candidate=make_candidate("c0"),
                features={"dist": 1.0, "gain": False},
                enabled=True,
            ),
        )
        with pytest.raises(InvalidFeatureValueError) as excinfo:
            ActorObservationBuilder().build(make_input(candidates=bad, graph_edges=()))
        assert "c0" in excinfo.value.group
        assert excinfo.value.feature_name == "gain"

    def test_bool_in_teammate_features_raises(self):
        bad = (TeammateFeatureSource(robot_id=1, features={"rel_x": 0.5, "rel_y": True}),)
        with pytest.raises(InvalidFeatureValueError) as excinfo:
            ActorObservationBuilder().build(make_input(visible_teammates=bad))
        assert excinfo.value.feature_name == "rel_y"

    def test_plain_int_and_float_remain_valid(self):
        obs = ActorObservationBuilder().build(
            make_input(robot_features={"x": 1, "y": 2.5, "battery": 0})
        )
        assert obs.robot_features == (1.0, 2.5, 0.0)
        assert all(isinstance(v, float) for v in obs.robot_features)


class TestCriticStateBuilder:
    def test_builds_critic_state_with_ordered_vectors(self):
        state = CriticStateBuilder().build(
            CriticStateBuildInput(
                decision_step=1,
                time_s=0.5,
                global_feature_names=("coverage", "found_fires"),
                global_features={"found_fires": 1.0, "coverage": 0.4},
                per_robot_feature_names=("x", "y"),
                per_robot_features={0: {"y": 2.0, "x": 1.0}},
            )
        )
        assert state.global_features == (0.4, 1.0)
        assert state.per_robot_features[0] == (1.0, 2.0)

    def test_critic_feature_mismatch_raises(self):
        with pytest.raises(FeatureSchemaMismatchError):
            CriticStateBuilder().build(
                CriticStateBuildInput(
                    decision_step=1,
                    time_s=0.5,
                    global_feature_names=("coverage",),
                    global_features={"coverage": 0.4, "extra": 1.0},
                    per_robot_feature_names=(),
                    per_robot_features={},
                )
            )
