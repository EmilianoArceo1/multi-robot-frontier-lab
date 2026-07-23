"""Tests for ActorObservationBatch and ActorObservationBatchAssembler."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.learning import CandidateKind, CandidateSetSpec, HoldPolicy
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.learning import FeatureNormalizationConfig, build_feature_schema_v0
from robotics_sim.learning.capture_inputs import (
    CandidateCaptureInput,
    RobotActorCaptureInput,
    RuntimeActorFrame,
)
from robotics_sim.learning.observation_batch import (
    ActorObservationBatchAssembler,
    _build_candidate_observation,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent

NORMALIZATION = FeatureNormalizationConfig(
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


def make_geometry() -> GridGeometry:
    return GridGeometry(bounds=(0.0, 10.0, 0.0, 10.0), resolution=1.0)


def make_candidate_spec(max_candidates: int = 8, max_headings: int = 1) -> CandidateSetSpec:
    return CandidateSetSpec(
        schema_version="0.1.0",
        max_candidates=max_candidates,
        max_headings_per_candidate=max_headings,
        deterministic_ordering=True,
        deduplication_distance=0.5,
        hold_policy=HoldPolicy(),
    )


def make_robot(robot_id: int = 0, xy: tuple[float, float] = (1.0, 1.0)) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=xy,
        safety_radius=0.5,
        sensor_range=4.0,
        vision_model="cone",
        theta=0.0,
    )


def make_candidate(
    target=(4.0, 6.0),
    kind=CandidateKind.FRONTIER_VIEWPOINT,
    heading_rad=None,
    enabled=True,
    reachable=True,
    source="frontier",
) -> CandidateCaptureInput:
    return CandidateCaptureInput(
        candidate=ExplorationCandidate(
            target=target, source=source, information_gain=1.0, heading_rad=heading_rad
        ),
        kind=kind,
        enabled=enabled,
        reachable=reachable,
    )


def make_belief(geometry: GridGeometry, observations=()):
    belief = HazardBelief(geometry)
    for row, col, value in observations:
        belief.observe_cells([row], [col], [value], robot_index=0)
    return belief.snapshot()


def make_robot_capture(
    robot_id=0,
    candidates=(),
    graph_edges=(),
    visible_teammates=(),
    hazard_belief=None,
    geometry=None,
) -> RobotActorCaptureInput:
    geometry = geometry or make_geometry()
    return RobotActorCaptureInput(
        robot=make_robot(robot_id),
        candidates=candidates,
        graph_edges=graph_edges,
        visible_teammates=visible_teammates,
        hazard_belief=hazard_belief if hazard_belief is not None else make_belief(geometry),
    )


def make_frame(robots, geometry=None, decision_step=0, candidate_spec=None) -> RuntimeActorFrame:
    geometry = geometry or make_geometry()
    return RuntimeActorFrame(
        episode_id="ep-batch",
        decision_step=decision_step,
        time_s=0.0,
        robots=robots,
        grid_geometry=geometry,
        normalization=NORMALIZATION,
        candidate_spec=candidate_spec or make_candidate_spec(),
    )


def make_assembler(candidate_spec=None) -> ActorObservationBatchAssembler:
    return ActorObservationBatchAssembler(
        schema=build_feature_schema_v0(), candidate_spec=candidate_spec or make_candidate_spec()
    )


class TestRobotCount:
    def test_single_robot_produces_single_observation(self):
        geometry = make_geometry()
        robot_capture = make_robot_capture(
            robot_id=0, candidates=(make_candidate(),), geometry=geometry
        )
        batch = make_assembler().build(make_frame((robot_capture,), geometry=geometry))
        assert len(batch.observations) == 1
        assert batch.observations[0].robot_id == 0

    def test_two_robots_produce_two_observations(self):
        geometry = make_geometry()
        r0 = make_robot_capture(robot_id=0, candidates=(make_candidate(),), geometry=geometry)
        r1 = make_robot_capture(robot_id=1, candidates=(make_candidate(),), geometry=geometry)
        batch = make_assembler().build(make_frame((r0, r1), geometry=geometry))
        assert len(batch.observations) == 2
        assert {o.robot_id for o in batch.observations} == {0, 1}


class TestOrderingPreserved:
    def test_robot_order_preserved(self):
        geometry = make_geometry()
        r5 = make_robot_capture(robot_id=5, candidates=(make_candidate(),), geometry=geometry)
        r2 = make_robot_capture(robot_id=2, candidates=(make_candidate(),), geometry=geometry)
        batch = make_assembler().build(make_frame((r5, r2), geometry=geometry))
        assert [o.robot_id for o in batch.observations] == [5, 2]

    def test_candidate_order_preserved(self):
        geometry = make_geometry()
        candidates = (
            make_candidate(target=(2.0, 2.0)),
            make_candidate(target=(8.0, 8.0)),
            make_candidate(target=(5.0, 1.0)),
        )
        robot_capture = make_robot_capture(candidates=candidates, geometry=geometry)
        batch = make_assembler().build(make_frame((robot_capture,), geometry=geometry))
        assert batch.observations[0].candidate_ids == (
            "robot-0/step-0/candidate-0",
            "robot-0/step-0/candidate-1",
            "robot-0/step-0/candidate-2",
        )


class TestCandidateKindAndSource:
    def test_kind_is_explicit_not_inferred(self):
        candidate_capture = make_candidate(
            kind=CandidateKind.FIRE_INFORMATION_VIEWPOINT, source="frontier"
        )
        observation = _build_candidate_observation(candidate_capture, "robot-0/step-0/candidate-0")
        assert observation.kind is CandidateKind.FIRE_INFORMATION_VIEWPOINT

    def test_kind_reflected_in_one_hot_candidate_features(self):
        geometry = make_geometry()
        candidate_capture = make_candidate(kind=CandidateKind.RECOVERY_VIEWPOINT)
        robot_capture = make_robot_capture(candidates=(candidate_capture,), geometry=geometry)
        batch = make_assembler().build(make_frame((robot_capture,), geometry=geometry))
        obs = batch.observations[0]
        names = obs.candidate_feature_names
        row = obs.candidate_features[0]
        assert row[names.index("recovery_kind")] == 1.0
        assert row[names.index("frontier_kind")] == 0.0

    def test_source_preserved(self):
        candidate_capture = make_candidate(source="custom_planner_source")
        observation = _build_candidate_observation(candidate_capture, "robot-0/step-0/candidate-0")
        assert observation.source == "custom_planner_source"


class TestHeadings:
    def test_candidate_with_heading(self):
        candidate_capture = make_candidate(heading_rad=0.75)
        observation = _build_candidate_observation(candidate_capture, "id")
        assert observation.heading_candidates == (0.75,)

    def test_candidate_without_heading(self):
        candidate_capture = make_candidate(heading_rad=None)
        observation = _build_candidate_observation(candidate_capture, "id")
        assert observation.heading_candidates == ()

    def test_has_heading_feature_set_correctly(self):
        geometry = make_geometry()
        with_heading = make_candidate(target=(2.0, 2.0), heading_rad=0.5)
        without_heading = make_candidate(target=(7.0, 7.0), heading_rad=None)
        robot_capture = make_robot_capture(
            candidates=(with_heading, without_heading), geometry=geometry
        )
        batch = make_assembler().build(make_frame((robot_capture,), geometry=geometry))
        obs = batch.observations[0]
        idx = obs.candidate_feature_names.index("has_heading")
        assert obs.candidate_features[0][idx] == 1.0
        assert obs.candidate_features[1][idx] == 0.0


class TestTeammates:
    def test_visible_teammate_produces_teammate_features(self):
        geometry = make_geometry()
        teammate = make_robot(robot_id=1, xy=(3.0, 3.0))
        robot_capture = make_robot_capture(
            robot_id=0,
            candidates=(make_candidate(),),
            visible_teammates=(teammate,),
            geometry=geometry,
        )
        batch = make_assembler().build(make_frame((robot_capture,), geometry=geometry))
        obs = batch.observations[0]
        assert len(obs.visible_teammate_features) == 1
        assert len(obs.visible_teammate_feature_names) == 8

    def test_zero_teammates(self):
        geometry = make_geometry()
        robot_capture = make_robot_capture(candidates=(make_candidate(),), geometry=geometry)
        batch = make_assembler().build(make_frame((robot_capture,), geometry=geometry))
        assert batch.observations[0].visible_teammate_features == ()


class TestHazardBeliefPerRobot:
    def test_different_beliefs_produce_different_features(self):
        geometry = make_geometry()
        candidate = make_candidate(target=(4.0, 6.0))
        cell = geometry.world_to_grid(4.0, 6.0)
        belief_a = make_belief(geometry, observations=((cell.row, cell.col, 0.1),))
        belief_b = make_belief(geometry, observations=((cell.row, cell.col, 0.8),))

        r0 = make_robot_capture(
            robot_id=0, candidates=(candidate,), hazard_belief=belief_a, geometry=geometry
        )
        r1 = make_robot_capture(
            robot_id=1, candidates=(candidate,), hazard_belief=belief_b, geometry=geometry
        )
        batch = make_assembler().build(make_frame((r0, r1), geometry=geometry))

        obs0, obs1 = batch.observations
        idx = obs0.candidate_feature_names.index("fire_value_at_target")
        assert obs0.candidate_features[0][idx] != obs1.candidate_features[0][idx]

    def test_sharing_same_belief_object_is_valid(self):
        geometry = make_geometry()
        shared_belief = make_belief(geometry)
        candidate = make_candidate()
        r0 = make_robot_capture(
            robot_id=0, candidates=(candidate,), hazard_belief=shared_belief, geometry=geometry
        )
        r1 = make_robot_capture(
            robot_id=1, candidates=(candidate,), hazard_belief=shared_belief, geometry=geometry
        )
        batch = make_assembler().build(make_frame((r0, r1), geometry=geometry))
        assert len(batch.observations) == 2


class TestCandidateCaptureRejections:
    def test_unreachable_and_enabled_rejected(self):
        with pytest.raises(ValueError):
            CandidateCaptureInput(
                candidate=ExplorationCandidate(target=(1.0, 1.0)),
                kind=CandidateKind.FRONTIER_VIEWPOINT,
                enabled=True,
                reachable=False,
            )

    def test_hold_kind_rejected(self):
        with pytest.raises(ValueError):
            CandidateCaptureInput(
                candidate=ExplorationCandidate(target=(1.0, 1.0)),
                kind=CandidateKind.HOLD,
                enabled=False,
                reachable=True,
            )


class TestRuntimeActorFrameValidation:
    def test_candidates_exceeding_max_candidates_rejected(self):
        geometry = make_geometry()
        candidates = tuple(make_candidate(target=(float(i % 9), 1.0)) for i in range(3))
        robot_capture = make_robot_capture(candidates=candidates, geometry=geometry)
        with pytest.raises(ValueError):
            make_frame(
                (robot_capture,),
                geometry=geometry,
                candidate_spec=make_candidate_spec(max_candidates=2),
            )

    def test_single_heading_within_limit_is_accepted(self):
        geometry = make_geometry()
        candidate = make_candidate(heading_rad=0.2)
        robot_capture = make_robot_capture(candidates=(candidate,), geometry=geometry)
        frame = make_frame(
            (robot_capture,), geometry=geometry, candidate_spec=make_candidate_spec(max_headings=1)
        )
        assert frame.robots[0].candidates[0].candidate.heading_rad == 0.2

    def test_duplicate_robot_ids_rejected(self):
        geometry = make_geometry()
        r0a = make_robot_capture(robot_id=0, candidates=(make_candidate(),), geometry=geometry)
        r0b = make_robot_capture(robot_id=0, candidates=(make_candidate(),), geometry=geometry)
        with pytest.raises(ValueError):
            make_frame((r0a, r0b), geometry=geometry)


class TestTeammateCaptureRejections:
    def test_duplicate_teammate_ids_rejected(self):
        geometry = make_geometry()
        t1a = make_robot(robot_id=1, xy=(2.0, 2.0))
        t1b = make_robot(robot_id=1, xy=(3.0, 3.0))
        with pytest.raises(ValueError):
            make_robot_capture(
                robot_id=0,
                candidates=(make_candidate(),),
                visible_teammates=(t1a, t1b),
                geometry=geometry,
            )

    def test_robot_as_its_own_teammate_object_rejected(self):
        geometry = make_geometry()
        robot = make_robot(robot_id=0)
        with pytest.raises(ValueError):
            RobotActorCaptureInput(
                robot=robot,
                candidates=(make_candidate(),),
                graph_edges=(),
                visible_teammates=(robot,),
                hazard_belief=make_belief(geometry),
            )

    def test_robot_with_same_id_as_teammate_rejected(self):
        geometry = make_geometry()
        with pytest.raises(ValueError):
            RobotActorCaptureInput(
                robot=make_robot(robot_id=0),
                candidates=(make_candidate(),),
                graph_edges=(),
                visible_teammates=(make_robot(robot_id=0, xy=(9.0, 9.0)),),
                hazard_belief=make_belief(geometry),
            )


class TestGraphEdges:
    def test_invalid_graph_edge_index_rejected_by_actor_observation(self):
        geometry = make_geometry()
        robot_capture = make_robot_capture(
            candidates=(make_candidate(),), graph_edges=((0, 5),), geometry=geometry
        )
        with pytest.raises(ValueError):
            make_assembler().build(make_frame((robot_capture,), geometry=geometry))

    def test_zero_candidates_produces_empty_observation_without_synthetic_hold(self):
        geometry = make_geometry()
        robot_capture = make_robot_capture(candidates=(), geometry=geometry)
        batch = make_assembler().build(make_frame((robot_capture,), geometry=geometry))
        obs = batch.observations[0]
        assert obs.candidate_ids == ()
        assert obs.graph_edges == ()
        assert obs.action_mask == ()


class TestGetForRobot:
    def test_get_for_robot_returns_correct_observation(self):
        geometry = make_geometry()
        r0 = make_robot_capture(robot_id=0, candidates=(make_candidate(),), geometry=geometry)
        r1 = make_robot_capture(robot_id=1, candidates=(make_candidate(),), geometry=geometry)
        batch = make_assembler().build(make_frame((r0, r1), geometry=geometry))
        assert batch.get_for_robot(1).robot_id == 1

    def test_get_for_robot_missing_raises(self):
        geometry = make_geometry()
        r0 = make_robot_capture(robot_id=0, candidates=(make_candidate(),), geometry=geometry)
        batch = make_assembler().build(make_frame((r0,), geometry=geometry))
        with pytest.raises(KeyError):
            batch.get_for_robot(99)


class TestInputsNotMutated:
    def test_candidates_tuple_and_robot_unchanged_after_build(self):
        geometry = make_geometry()
        candidates = (make_candidate(target=(2.0, 2.0)), make_candidate(target=(7.0, 7.0)))
        robot = make_robot(robot_id=0)
        belief = make_belief(geometry)
        robot_capture = RobotActorCaptureInput(
            robot=robot,
            candidates=candidates,
            graph_edges=(),
            visible_teammates=(),
            hazard_belief=belief,
        )
        frame = make_frame((robot_capture,), geometry=geometry)
        before_candidates = robot_capture.candidates
        before_robot = robot_capture.robot

        make_assembler().build(frame)

        assert robot_capture.candidates == before_candidates
        assert robot_capture.robot is before_robot
        assert robot_capture.robot.xy == (1.0, 1.0)


class _ExplodingMapping(Mapping):
    """A metadata mapping that fails on any read access."""

    def __getitem__(self, key):  # pragma: no cover - failure path
        raise AssertionError("must not read candidate.metadata")

    def __iter__(self):  # pragma: no cover - failure path
        raise AssertionError("must not iterate candidate.metadata")

    def __len__(self):  # pragma: no cover - failure path
        raise AssertionError("must not measure candidate.metadata")


class TestMetadataNeverRead:
    @pytest.mark.parametrize("filename", ["capture_inputs.py", "observation_batch.py"])
    def test_no_metadata_attribute_access_in_source(self, filename):
        tree = ast.parse((LEARNING_DIR / filename).read_text(encoding="utf-8"))
        accessed = [
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "metadata"
        ]
        assert accessed == []

    def test_build_succeeds_with_exploding_metadata(self):
        geometry = make_geometry()
        candidate_capture = CandidateCaptureInput(
            candidate=ExplorationCandidate(
                target=(4.0, 6.0),
                information_gain=1.0,
                heading_rad=0.3,
                metadata=_ExplodingMapping(),
            ),
            kind=CandidateKind.FRONTIER_VIEWPOINT,
            enabled=True,
            reachable=True,
        )
        robot_capture = make_robot_capture(candidates=(candidate_capture,), geometry=geometry)
        batch = make_assembler().build(make_frame((robot_capture,), geometry=geometry))
        assert len(batch.observations[0].candidate_ids) == 1


class TestSmokeRealisticScenario:
    def test_two_robots_three_candidates_each_full_pipeline(self):
        geometry = make_geometry()
        candidate_spec = make_candidate_spec(max_candidates=3, max_headings=1)
        schema = build_feature_schema_v0()

        def robot_candidates() -> tuple[CandidateCaptureInput, ...]:
            return (
                make_candidate(
                    target=(2.0, 2.0), kind=CandidateKind.FRONTIER_VIEWPOINT, heading_rad=0.1
                ),
                make_candidate(target=(6.0, 6.0), kind=CandidateKind.FIRE_INFORMATION_VIEWPOINT),
                make_candidate(
                    target=(8.0, 1.0), kind=CandidateKind.RECOVERY_VIEWPOINT, heading_rad=1.2
                ),
            )

        shared_belief = make_belief(geometry, observations=((3, 3, 0.4),))

        robot0 = make_robot(robot_id=0, xy=(1.0, 1.0))
        robot1 = make_robot(robot_id=1, xy=(9.0, 9.0))

        r0_capture = RobotActorCaptureInput(
            robot=robot0,
            candidates=robot_candidates(),
            graph_edges=((0, 1), (1, 2)),
            visible_teammates=(robot1,),
            hazard_belief=shared_belief,
        )
        r1_capture = RobotActorCaptureInput(
            robot=robot1,
            candidates=robot_candidates(),
            graph_edges=((0, 1), (1, 2)),
            visible_teammates=(robot0,),
            hazard_belief=shared_belief,
        )

        frame = RuntimeActorFrame(
            episode_id="ep-smoke",
            decision_step=3,
            time_s=1.5,
            robots=(r0_capture, r1_capture),
            grid_geometry=geometry,
            normalization=NORMALIZATION,
            candidate_spec=candidate_spec,
        )

        batch = ActorObservationBatchAssembler(schema=schema, candidate_spec=candidate_spec).build(
            frame
        )

        assert len(batch.observations) == 2
        for obs in batch.observations:
            assert len(obs.candidate_ids) == 3
            assert len(obs.candidate_feature_names) == 20
            assert len(obs.visible_teammate_feature_names) == 8
            assert len(obs.visible_teammate_features) == 1

        assert batch.observations[0].candidate_ids == (
            "robot-0/step-3/candidate-0",
            "robot-0/step-3/candidate-1",
            "robot-0/step-3/candidate-2",
        )
        assert batch.observations[1].candidate_ids == (
            "robot-1/step-3/candidate-0",
            "robot-1/step-3/candidate-1",
            "robot-1/step-3/candidate-2",
        )

        # No ground truth anywhere in the batch: ActorObservation has no
        # metadata escape hatch and no privileged field by construction
        # (see robotics_interfaces.learning.observations.FORBIDDEN_ACTOR_FIELDS).
        for obs in batch.observations:
            for forbidden in (
                "ground_truth",
                "true_fire",
                "true_occupancy",
                "critic_state",
                "privileged",
                "metadata",
            ):
                assert not hasattr(obs, forbidden)
