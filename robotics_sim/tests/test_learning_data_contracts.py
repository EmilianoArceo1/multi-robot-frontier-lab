"""Tests for the neutral learning data contracts (construction, validation,
primitives, warm-up, HOLD restriction, termination, export metadata)."""

from __future__ import annotations

import dataclasses
import enum
import math

import pytest

from robotics_interfaces.learning import (
    ActorObservation,
    CandidateKind,
    CandidateObservation,
    CandidateSetSpec,
    CriticState,
    EpisodeFireMetrics,
    EpisodeMetadata,
    GroundTruthSnapshot,
    HoldPolicy,
    HoldReason,
    LearningAction,
    LearningTransition,
    LinearWeightWarmup,
    RewardComponent,
    RewardPhase,
    RewardSpec,
    RewardTerm,
    RewardTermSpec,
    ReservationSpec,
    ReservationTieBreaker,
    RouteReservation,
    TerminationReason,
    TerminationSpec,
    TrajectoryExportSpec,
    UnsupportedPrimitiveError,
    build_contract_manifest,
    compute_contract_bundle_hash,
    to_primitive,
    validate_action_mask,
)


def make_actor_observation(**overrides) -> ActorObservation:
    kwargs = dict(
        schema_version="0.1.0",
        robot_id=0,
        decision_step=3,
        time_s=1.5,
        robot_feature_names=("x", "y", "battery"),
        robot_features=(1.0, 2.0, 0.9),
        candidate_feature_names=("dist", "gain"),
        candidate_features=((1.0, 0.5), (2.0, 0.25)),
        candidate_ids=("c0", "c1"),
        action_mask=(True, False),
        graph_edges=((0, 1),),
        visible_teammate_feature_names=("rel_x", "rel_y"),
        visible_teammate_features=((0.5, -0.5),),
    )
    kwargs.update(overrides)
    return ActorObservation(**kwargs)


def make_critic_state() -> CriticState:
    return CriticState(
        schema_version="0.1.0",
        decision_step=3,
        time_s=1.5,
        global_feature_names=("coverage",),
        global_features=(0.4,),
        per_robot_feature_names=("x", "y"),
        per_robot_features={0: (1.0, 2.0)},
    )


def make_ground_truth() -> GroundTruthSnapshot:
    return GroundTruthSnapshot(
        schema_version="0.1.0",
        decision_step=3,
        time_s=1.5,
        true_robot_poses={0: (1.0, 2.0, 0.5)},
        true_occupancy=((0, 1), (1, 0)),
        true_fire_locations=((3.0, 4.0),),
        global_coverage_fraction=0.4,
    )


class TestConstructionAndFrozen:
    def test_actor_observation_builds_and_is_frozen(self):
        obs = make_actor_observation()
        with pytest.raises(dataclasses.FrozenInstanceError):
            obs.robot_id = 5

    def test_critic_state_builds_and_is_frozen(self):
        state = make_critic_state()
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.decision_step = 9

    def test_ground_truth_builds_and_is_frozen(self):
        snapshot = make_ground_truth()
        with pytest.raises(dataclasses.FrozenInstanceError):
            snapshot.time_s = 0.0

    def test_other_contracts_build_and_are_frozen(self):
        action = LearningAction(
            robot_id=0, candidate_id="c0", candidate_index=0,
            heading_index=1, action_index=1, issued_at_step=3,
        )
        spec = TerminationSpec(
            schema_version="0.1.0", max_steps=500, require_coverage=True,
            require_all_fire_found=True, coverage_threshold=0.95,
        )
        reservation = RouteReservation(
            robot_id=1, route_id="r1", polyline=((0.0, 0.0), (1.0, 1.0)),
            start_time=0.0, estimated_end_time=4.0, safety_radius=0.5,
            priority=0, created_at=0.0, expires_at=10.0,
        )
        for frozen_obj, attr in ((action, "robot_id"), (spec, "max_steps"), (reservation, "priority")):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(frozen_obj, attr, 99)


class TestActorObservationValidation:
    def test_rejects_robot_feature_dim_mismatch(self):
        with pytest.raises(ValueError):
            make_actor_observation(robot_features=(1.0, 2.0))

    def test_rejects_inconsistent_candidate_width(self):
        with pytest.raises(ValueError):
            make_actor_observation(candidate_features=((1.0, 0.5), (2.0,)))

    def test_rejects_candidate_width_vs_names_mismatch(self):
        with pytest.raises(ValueError):
            make_actor_observation(candidate_features=((1.0,), (2.0,)))

    def test_rejects_action_mask_length_mismatch(self):
        with pytest.raises(ValueError):
            make_actor_observation(action_mask=(True,))

    def test_rejects_candidate_ids_length_mismatch(self):
        with pytest.raises(ValueError):
            make_actor_observation(candidate_ids=("c0",))

    def test_rejects_invalid_graph_edge_indices(self):
        with pytest.raises(ValueError):
            make_actor_observation(graph_edges=((0, 2),))
        with pytest.raises(ValueError):
            make_actor_observation(graph_edges=((-1, 0),))

    def test_rejects_nan_features(self):
        with pytest.raises(ValueError):
            make_actor_observation(robot_features=(1.0, float("nan"), 0.9))
        with pytest.raises(ValueError):
            make_actor_observation(candidate_features=((1.0, float("nan")), (2.0, 0.25)))

    def test_rejects_infinite_features(self):
        with pytest.raises(ValueError):
            make_actor_observation(robot_features=(1.0, float("inf"), 0.9))
        with pytest.raises(ValueError):
            make_actor_observation(
                visible_teammate_features=((float("-inf"), 0.0),)
            )

    def test_rejects_teammate_feature_width_mismatch(self):
        with pytest.raises(ValueError):
            make_actor_observation(visible_teammate_features=((0.5,),))


class TestToPrimitive:
    def test_scalars_pass_through(self):
        assert to_primitive(None) is None
        assert to_primitive(True) is True
        assert to_primitive(3) == 3
        assert to_primitive(1.5) == 1.5
        assert to_primitive("x") == "x"

    def test_serializes_tuples_and_mappings(self):
        assert to_primitive((1, 2, (3, 4))) == [1, 2, [3, 4]]
        assert to_primitive({"a": (1, 2)}) == {"a": [1, 2]}

    def test_serializes_enums(self):
        result = to_primitive(CandidateKind.HOLD)
        assert result["name"] == "HOLD"
        assert result["value"] == "hold"

    def test_serializes_dataclasses(self):
        warmup = LinearWeightWarmup(start_step=0, end_step=10, target_weight=1.0)
        assert to_primitive(warmup) == {
            "start_step": 0, "end_step": 10, "target_weight": 1.0,
        }

    def test_fails_for_unsupported_object(self):
        class Opaque:
            pass

        with pytest.raises(UnsupportedPrimitiveError):
            to_primitive(Opaque())

    def test_fails_for_non_str_mapping_keys(self):
        with pytest.raises(UnsupportedPrimitiveError):
            to_primitive({1: "a"})


class TestLinearWeightWarmup:
    def test_zero_before_start(self):
        warmup = LinearWeightWarmup(start_step=10, end_step=20, target_weight=2.0)
        assert warmup.weight_at(0) == 0.0
        assert warmup.weight_at(9) == 0.0

    def test_linear_interpolation(self):
        warmup = LinearWeightWarmup(start_step=10, end_step=20, target_weight=2.0)
        assert warmup.weight_at(15) == pytest.approx(1.0)
        assert warmup.weight_at(12) == pytest.approx(0.4)

    def test_target_after_end(self):
        warmup = LinearWeightWarmup(start_step=10, end_step=20, target_weight=2.0)
        assert warmup.weight_at(20) == 2.0
        assert warmup.weight_at(1000) == 2.0

    def test_rejects_invalid_range(self):
        with pytest.raises(ValueError):
            LinearWeightWarmup(start_step=20, end_step=10, target_weight=1.0)
        with pytest.raises(ValueError):
            LinearWeightWarmup(start_step=10, end_step=10, target_weight=1.0)

    def test_reward_spec_builds_with_phased_terms(self):
        spec = RewardSpec(
            schema_version="0.1.0",
            terms=(
                RewardTermSpec(
                    term=RewardTerm.NEW_COVERAGE,
                    phase_introduced=RewardPhase.COVERAGE,
                ),
                RewardTermSpec(
                    term=RewardTerm.FIRE_INFORMATION_GAIN,
                    phase_introduced=RewardPhase.FIRE_SEARCH,
                    warmup=LinearWeightWarmup(start_step=100, end_step=200, target_weight=1.0),
                ),
            ),
        )
        assert len(spec.terms) == 2

    def test_reward_spec_rejects_duplicate_terms(self):
        term = RewardTermSpec(
            term=RewardTerm.COLLISION, phase_introduced=RewardPhase.MULTI_ROBOT,
        )
        with pytest.raises(ValueError):
            RewardSpec(schema_version="0.1.0", terms=(term, term))


def _candidate(candidate_id: str, kind: CandidateKind) -> CandidateObservation:
    return CandidateObservation(
        candidate_id=candidate_id,
        kind=kind,
        xy=(1.0, 2.0),
        heading_candidates=(0.0, 1.57),
        source="frontier",
        reachable=True,
    )


class TestHoldRestriction:
    def test_hold_is_not_policy_selectable(self):
        policy = HoldPolicy()
        assert policy.policy_selectable is False
        assert policy.allow_when_non_hold_available is False
        assert policy.host_fallback_only is True

    def test_hold_policy_cannot_be_configured_away(self):
        with pytest.raises(ValueError):
            HoldPolicy(policy_selectable=True)
        with pytest.raises(ValueError):
            HoldPolicy(allow_when_non_hold_available=True)
        with pytest.raises(ValueError):
            HoldPolicy(host_fallback_only=False)

    def test_hold_cannot_be_enabled_with_other_valid_action(self):
        candidates = (
            _candidate("c0", CandidateKind.FRONTIER_VIEWPOINT),
            _candidate("h0", CandidateKind.HOLD),
        )
        with pytest.raises(ValueError):
            validate_action_mask(candidates, (True, True))

    def test_hold_allowed_as_fallback_when_no_valid_action(self):
        candidates = (
            _candidate("c0", CandidateKind.FRONTIER_VIEWPOINT),
            _candidate("h0", CandidateKind.HOLD),
        )
        validate_action_mask(candidates, (False, True))  # must not raise

    def test_hold_reasons_exist(self):
        assert {r.name for r in HoldReason} == {
            "NO_VALID_CANDIDATE", "WAITING_FOR_RESERVATION", "RECOVERY_COOLDOWN",
        }

    def test_candidate_set_spec_builds(self):
        spec = CandidateSetSpec(
            schema_version="0.1.0",
            max_candidates=32,
            max_headings_per_candidate=4,
            deterministic_ordering=True,
            deduplication_distance=0.5,
            hold_policy=HoldPolicy(),
        )
        assert spec.max_candidates == 32


class TestActionsAndTermination:
    def test_learning_action_rejects_negative_indices(self):
        for field_name in ("candidate_index", "heading_index", "action_index", "issued_at_step"):
            kwargs = dict(
                robot_id=0, candidate_id="c0", candidate_index=0,
                heading_index=0, action_index=0, issued_at_step=0,
            )
            kwargs[field_name] = -1
            with pytest.raises(ValueError):
                LearningAction(**kwargs)

    def test_termination_reasons_exist(self):
        expected = {
            "RUNNING", "COVERAGE_COMPLETE", "ALL_FIRE_FOUND",
            "COVERAGE_AND_FIRE_COMPLETE", "MAX_STEPS", "BUDGET_EXHAUSTED",
            "NO_VALID_ACTION", "COLLISION", "EXTERNAL_STOP", "ERROR",
        }
        assert expected <= {r.name for r in TerminationReason}

    def test_termination_spec_validates_threshold(self):
        with pytest.raises(ValueError):
            TerminationSpec(
                schema_version="0.1.0", max_steps=100, require_coverage=True,
                require_all_fire_found=False, coverage_threshold=1.5,
            )
        with pytest.raises(ValueError):
            TerminationSpec(
                schema_version="0.1.0", max_steps=100, require_coverage=True,
                require_all_fire_found=False, coverage_threshold=-0.1,
            )


class TestReservations:
    def test_reservation_spec_documents_known_bias(self):
        spec = ReservationSpec(
            schema_version="0.1.0",
            tie_breaker=ReservationTieBreaker.OLDEST_THEN_LOWEST_ROBOT_ID,
            ttl_s=30.0,
        )
        assert "starvation" in spec.known_bias
        with pytest.raises(ValueError):
            ReservationSpec(
                schema_version="0.1.0",
                tie_breaker=ReservationTieBreaker.OLDEST_THEN_LOWEST_ROBOT_ID,
                ttl_s=30.0,
                known_bias="   ",
            )


class TestTransitions:
    def test_learning_transition_builds(self):
        obs = make_actor_observation()
        transition = LearningTransition(
            schema_version="0.1.0",
            episode_id="ep-1",
            decision_step=3,
            actor_observations={0: obs},
            critic_state=make_critic_state(),
            selected_actions={
                0: LearningAction(
                    robot_id=0, candidate_id="c0", candidate_index=0,
                    heading_index=0, action_index=0, issued_at_step=3,
                )
            },
            reward_components_by_robot={
                0: (
                    RewardComponent(
                        name="new_coverage", raw_value=1.0,
                        applied_weight=0.5, weighted_value=0.5,
                    ),
                )
            },
            reward_total_by_robot={0: 0.5},
            next_actor_observations={0: make_actor_observation(decision_step=4)},
            terminated=False,
            truncated=False,
            termination_reason=TerminationReason.RUNNING,
        )
        assert transition.reward_total_by_robot[0] == 0.5

    def test_reward_component_rejects_non_finite(self):
        with pytest.raises(ValueError):
            RewardComponent(
                name="x", raw_value=float("nan"), applied_weight=1.0, weighted_value=0.0,
            )


class TestExportContracts:
    def make_metadata(self, **overrides) -> EpisodeMetadata:
        manifest = build_contract_manifest()
        kwargs = dict(
            episode_id="ep-1",
            seed=1234,
            map_id="house_3",
            robot_count=3,
            fire_count=2,
            sensor_range=5.0,
            field_of_view_deg=90.0,
            communication_range=10.0,
            max_steps=500,
            simulator_commit="abc1234",
            contract_versions={"ObservationSpec": "0.1.0"},
            contract_bundle_hash=compute_contract_bundle_hash(manifest),
        )
        kwargs.update(overrides)
        return EpisodeMetadata(**kwargs)

    def test_v0_invariants(self):
        metadata = self.make_metadata()
        assert metadata.fire_traversable is True
        assert metadata.fire_damage_model == "none"
        assert metadata.task_version == "fire_search_v0"
        assert metadata.contract_bundle_hash

    def test_v0_invariants_are_enforced(self):
        with pytest.raises(ValueError):
            self.make_metadata(task_version="other_task")
        with pytest.raises(ValueError):
            self.make_metadata(fire_traversable=False)
        with pytest.raises(ValueError):
            self.make_metadata(fire_damage_model="thermal")
        with pytest.raises(ValueError):
            self.make_metadata(contract_bundle_hash="")

    def test_fire_metrics_defaults_and_validation(self):
        metrics = EpisodeFireMetrics()
        assert metrics.fire_crossing_time_s == 0.0
        assert metrics.fire_overflight_distance == 0.0
        with pytest.raises(ValueError):
            EpisodeFireMetrics(fire_crossing_time_s=-1.0)

    def test_trajectory_export_spec_defaults(self):
        spec = TrajectoryExportSpec(schema_version="0.1.0")
        assert spec.tensor_format == "npz"
        assert spec.event_format == "parquet"
        assert spec.metadata_format == "json"
        assert spec.include_ground_truth_separately is True
