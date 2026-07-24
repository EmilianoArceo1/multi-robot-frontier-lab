"""Tests for ActionOption, RobotActionCatalog, ActionCatalogBatch and
ActionCatalogAssembler."""

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
from robotics_sim.learning.action_catalog import (
    ActionCatalogAssembler,
    ActionCatalogBatch,
    ActionOption,
    RobotActionCatalog,
)
from robotics_sim.learning.capture_inputs import (
    CandidateCaptureInput,
    RobotActorCaptureInput,
    RuntimeActorFrame,
)
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler

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


def make_candidate_spec(max_candidates: int = 8) -> CandidateSetSpec:
    return CandidateSetSpec(
        schema_version="0.1.0",
        max_candidates=max_candidates,
        max_headings_per_candidate=1,
        deterministic_ordering=True,
        deduplication_distance=0.5,
        hold_policy=HoldPolicy(),
    )


def make_robot(robot_id: int = 0) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(1.0, 1.0),
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


def make_robot_capture(robot_id=0, candidates=(), geometry=None) -> RobotActorCaptureInput:
    geometry = geometry or make_geometry()
    return RobotActorCaptureInput(
        robot=make_robot(robot_id),
        candidates=candidates,
        graph_edges=(),
        visible_teammates=(),
        hazard_belief=HazardBelief(geometry).snapshot(),
    )


def make_frame(robots, geometry=None, decision_step=0, candidate_spec=None) -> RuntimeActorFrame:
    geometry = geometry or make_geometry()
    return RuntimeActorFrame(
        episode_id="ep-catalog",
        decision_step=decision_step,
        time_s=0.0,
        robots=robots,
        grid_geometry=geometry,
        normalization=NORMALIZATION,
        candidate_spec=candidate_spec or make_candidate_spec(),
    )


def build_catalog_batch(frame: RuntimeActorFrame) -> ActionCatalogBatch:
    actor_batch = ActorObservationBatchAssembler(
        schema=build_feature_schema_v0(), candidate_spec=make_candidate_spec()
    ).build(frame)
    return ActionCatalogAssembler().build(frame, actor_batch)


def make_option(
    robot_id=0,
    candidate_id="robot-0/step-0/candidate-0",
    candidate_index=0,
    action_index=0,
    target_xy=(4.0, 6.0),
    heading_rad=None,
    kind=CandidateKind.FRONTIER_VIEWPOINT,
    source="frontier",
    enabled=True,
    reachable=True,
) -> ActionOption:
    return ActionOption(
        robot_id=robot_id,
        candidate_id=candidate_id,
        candidate_index=candidate_index,
        action_index=action_index,
        target_xy=target_xy,
        heading_rad=heading_rad,
        kind=kind,
        source=source,
        enabled=enabled,
        reachable=reachable,
    )


class TestActionOptionHeadings:
    def test_option_with_heading(self):
        option = make_option(heading_rad=0.5)
        assert option.heading_rad == 0.5

    def test_option_without_heading(self):
        option = make_option(heading_rad=None)
        assert option.heading_rad is None


class TestToLearningAction:
    def test_correct_with_heading(self):
        option = make_option(
            robot_id=2,
            candidate_id="robot-2/step-5/candidate-3",
            candidate_index=3,
            action_index=3,
            heading_rad=0.75,
        )
        action = option.to_learning_action(issued_at_step=7)
        assert action.robot_id == 2
        assert action.candidate_id == "robot-2/step-5/candidate-3"
        assert action.candidate_index == 3
        assert action.action_index == 3
        assert action.heading_index == 0
        assert action.issued_at_step == 7

    def test_correct_without_heading(self):
        option = make_option(heading_rad=None)
        action = option.to_learning_action(issued_at_step=0)
        assert action.heading_index is None

    def test_disabled_option_not_executable(self):
        # enabled=False, reachable=True: disabled but not unreachable.
        option = make_option(enabled=False, reachable=True)
        with pytest.raises(ValueError):
            option.to_learning_action(issued_at_step=0)

    def test_unreachable_option_not_executable(self):
        # enabled=False, reachable=False: the only way to be unreachable,
        # since enabled=True requires reachable=True.
        option = make_option(enabled=False, reachable=False)
        with pytest.raises(ValueError):
            option.to_learning_action(issued_at_step=0)


class TestActionOptionValidation:
    def test_enabled_requires_reachable(self):
        with pytest.raises(ValueError):
            make_option(enabled=True, reachable=False)

    def test_hold_kind_rejected(self):
        with pytest.raises(ValueError):
            make_option(kind=CandidateKind.HOLD, enabled=False)

    def test_action_index_must_equal_candidate_index(self):
        with pytest.raises(ValueError):
            make_option(candidate_index=0, action_index=1)


class TestRobotActionCatalogValidation:
    def test_duplicate_action_and_candidate_index_rejected(self):
        # In v0, action_index == candidate_index always (ActionOption's own
        # invariant), so a duplicate candidate_index is inseparable from a
        # duplicate action_index -- both are caught by the same collision
        # check, since no real ActionOption can vary one without the other.
        o0 = make_option(candidate_id="c0", candidate_index=0, action_index=0)
        o1 = make_option(candidate_id="c1", candidate_index=0, action_index=0)
        with pytest.raises(ValueError):
            RobotActionCatalog(robot_id=0, decision_step=0, options=(o0, o1))

    def test_duplicate_candidate_id_rejected(self):
        o0 = make_option(candidate_id="dup", candidate_index=0, action_index=0)
        o1 = make_option(candidate_id="dup", candidate_index=1, action_index=1)
        with pytest.raises(ValueError):
            RobotActionCatalog(robot_id=0, decision_step=0, options=(o0, o1))

    def test_mismatched_robot_id_rejected(self):
        o0 = make_option(robot_id=0, candidate_id="c0")
        o1 = make_option(robot_id=1, candidate_id="c1", candidate_index=1, action_index=1)
        with pytest.raises(ValueError):
            RobotActionCatalog(robot_id=0, decision_step=0, options=(o0, o1))


class TestOrderingAndLookups:
    def test_order_preserved(self):
        options = tuple(
            make_option(candidate_id=f"c{i}", candidate_index=i, action_index=i, target_xy=(float(i), 0.0))
            for i in (2, 0, 1)
        )
        catalog = RobotActionCatalog(robot_id=0, decision_step=0, options=options)
        assert tuple(o.candidate_index for o in catalog.options) == (2, 0, 1)

    def test_get_by_action_index(self):
        options = tuple(
            make_option(candidate_id=f"c{i}", candidate_index=i, action_index=i) for i in range(3)
        )
        catalog = RobotActionCatalog(robot_id=0, decision_step=0, options=options)
        assert catalog.get_by_action_index(2).candidate_id == "c2"

    def test_get_by_candidate_id(self):
        options = tuple(
            make_option(candidate_id=f"c{i}", candidate_index=i, action_index=i) for i in range(3)
        )
        catalog = RobotActionCatalog(robot_id=0, decision_step=0, options=options)
        assert catalog.get_by_candidate_id("c1").action_index == 1

    def test_get_by_action_index_missing_raises(self):
        catalog = RobotActionCatalog(robot_id=0, decision_step=0, options=(make_option(),))
        with pytest.raises(KeyError):
            catalog.get_by_action_index(99)

    def test_get_by_candidate_id_missing_raises(self):
        catalog = RobotActionCatalog(robot_id=0, decision_step=0, options=(make_option(),))
        with pytest.raises(KeyError):
            catalog.get_by_candidate_id("missing")

    def test_enabled_options(self):
        options = (
            make_option(candidate_id="c0", candidate_index=0, action_index=0, enabled=True),
            make_option(candidate_id="c1", candidate_index=1, action_index=1, enabled=False),
        )
        catalog = RobotActionCatalog(robot_id=0, decision_step=0, options=options)
        assert tuple(o.candidate_id for o in catalog.enabled_options()) == ("c0",)


class TestActionCatalogAssembler:
    def test_single_robot_catalog_matches_candidates(self):
        geometry = make_geometry()
        candidates = (
            make_candidate(target=(2.0, 2.0), heading_rad=0.3),
            make_candidate(target=(7.0, 7.0), heading_rad=None),
        )
        robot_capture = make_robot_capture(candidates=candidates, geometry=geometry)
        frame = make_frame((robot_capture,), geometry=geometry)
        batch = build_catalog_batch(frame)

        catalog = batch.get_for_robot(0)
        assert [o.candidate_id for o in catalog.options] == [
            "robot-0/step-0/candidate-0",
            "robot-0/step-0/candidate-1",
        ]
        assert catalog.options[0].heading_rad == 0.3
        assert catalog.options[1].heading_rad is None
        assert catalog.options[0].target_xy == (2.0, 2.0)
        assert catalog.options[1].target_xy == (7.0, 7.0)

    def test_two_robots_order_preserved(self):
        geometry = make_geometry()
        r5 = make_robot_capture(robot_id=5, candidates=(make_candidate(),), geometry=geometry)
        r2 = make_robot_capture(robot_id=2, candidates=(make_candidate(),), geometry=geometry)
        frame = make_frame((r5, r2), geometry=geometry)
        batch = build_catalog_batch(frame)
        assert [c.robot_id for c in batch.catalogs] == [5, 2]


class _ExplodingMapping(Mapping):
    """A metadata mapping that fails on any read access."""

    def __getitem__(self, key):  # pragma: no cover - failure path
        raise AssertionError("must not read candidate.metadata")

    def __iter__(self):  # pragma: no cover - failure path
        raise AssertionError("must not iterate candidate.metadata")

    def __len__(self):  # pragma: no cover - failure path
        raise AssertionError("must not measure candidate.metadata")


class TestMetadataAndMutation:
    def test_no_metadata_attribute_access_in_source(self):
        tree = ast.parse((LEARNING_DIR / "action_catalog.py").read_text(encoding="utf-8"))
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
                target=(4.0, 6.0), information_gain=1.0, heading_rad=0.3,
                metadata=_ExplodingMapping(),
            ),
            kind=CandidateKind.FRONTIER_VIEWPOINT,
            enabled=True,
            reachable=True,
        )
        robot_capture = make_robot_capture(candidates=(candidate_capture,), geometry=geometry)
        frame = make_frame((robot_capture,), geometry=geometry)
        batch = build_catalog_batch(frame)
        assert len(batch.get_for_robot(0).options) == 1

    def test_inputs_not_mutated(self):
        geometry = make_geometry()
        candidates = (make_candidate(target=(2.0, 2.0)), make_candidate(target=(7.0, 7.0)))
        robot = make_robot(robot_id=0)
        robot_capture = RobotActorCaptureInput(
            robot=robot,
            candidates=candidates,
            graph_edges=(),
            visible_teammates=(),
            hazard_belief=HazardBelief(geometry).snapshot(),
        )
        frame = make_frame((robot_capture,), geometry=geometry)
        before_candidates = robot_capture.candidates

        build_catalog_batch(frame)

        assert robot_capture.candidates == before_candidates
        assert robot_capture.robot is robot
