"""Tests for DecisionCaptureBatch and DecisionCaptureAssembler."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.learning import (
    CandidateKind,
    CandidateSetSpec,
    GroundTruthSnapshot,
    HoldPolicy,
)
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
from robotics_sim.learning.decision_batch import DecisionCaptureAssembler, DecisionCaptureBatch
from robotics_sim.learning.observation_batch import ActorObservationBatchAssembler

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent
DECISION_MODULES = ("action_catalog.py", "decision_batch.py")

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


def make_robot(robot_id: int = 0, xy=(1.0, 1.0)) -> RobotCoordinationState:
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


def make_robot_capture(robot_id=0, candidates=(), visible_teammates=(), geometry=None):
    geometry = geometry or make_geometry()
    return RobotActorCaptureInput(
        robot=make_robot(robot_id),
        candidates=candidates,
        graph_edges=(),
        visible_teammates=visible_teammates,
        hazard_belief=HazardBelief(geometry).snapshot(),
    )


def make_frame(
    robots,
    geometry=None,
    episode_id="ep-decision",
    decision_step=0,
    time_s=0.0,
    candidate_spec=None,
) -> RuntimeActorFrame:
    geometry = geometry or make_geometry()
    return RuntimeActorFrame(
        episode_id=episode_id,
        decision_step=decision_step,
        time_s=time_s,
        robots=robots,
        grid_geometry=geometry,
        normalization=NORMALIZATION,
        candidate_spec=candidate_spec or make_candidate_spec(),
    )


def make_assembler() -> DecisionCaptureAssembler:
    return DecisionCaptureAssembler(
        actor_assembler=ActorObservationBatchAssembler(
            schema=build_feature_schema_v0(), candidate_spec=make_candidate_spec()
        ),
        catalog_assembler=ActionCatalogAssembler(),
    )


def build_pair(geometry, episode_id="ep-decision", decision_step=0, robots_spec=None):
    """Build an aligned (actor_batch, action_catalog_batch) pair.

    ``robots_spec`` is a tuple of (robot_id, candidates) pairs."""

    robots_spec = robots_spec or ((0, (make_candidate(),)),)
    robot_captures = tuple(
        make_robot_capture(robot_id=rid, candidates=candidates, geometry=geometry)
        for rid, candidates in robots_spec
    )
    frame = make_frame(
        robot_captures, geometry=geometry, episode_id=episode_id, decision_step=decision_step
    )
    actor_batch = ActorObservationBatchAssembler(
        schema=build_feature_schema_v0(), candidate_spec=make_candidate_spec()
    ).build(frame)
    action_catalog_batch = ActionCatalogAssembler().build(frame, actor_batch)
    return actor_batch, action_catalog_batch


class TestRobotCount:
    def test_single_robot(self):
        geometry = make_geometry()
        robot_capture = make_robot_capture(candidates=(make_candidate(),), geometry=geometry)
        frame = make_frame((robot_capture,), geometry=geometry)
        decision = make_assembler().build(frame)
        assert len(decision.actor_batch.observations) == 1
        assert len(decision.action_catalog_batch.catalogs) == 1

    def test_two_robots(self):
        geometry = make_geometry()
        r0 = make_robot_capture(robot_id=0, candidates=(make_candidate(),), geometry=geometry)
        r1 = make_robot_capture(robot_id=1, candidates=(make_candidate(),), geometry=geometry)
        frame = make_frame((r0, r1), geometry=geometry)
        decision = make_assembler().build(frame)
        assert {o.robot_id for o in decision.actor_batch.observations} == {0, 1}
        assert {c.robot_id for c in decision.action_catalog_batch.catalogs} == {0, 1}


class TestAlignment:
    def test_actor_and_catalog_aligned(self):
        geometry = make_geometry()
        candidates = (make_candidate(target=(2.0, 2.0)), make_candidate(target=(7.0, 7.0)))
        robot_capture = make_robot_capture(candidates=candidates, geometry=geometry)
        frame = make_frame((robot_capture,), geometry=geometry)
        decision = make_assembler().build(frame)

        obs = decision.get_observation(0)
        catalog = decision.get_action_catalog(0)
        assert len(obs.candidate_ids) == len(catalog.options)

    def test_candidate_id_matches_by_position(self):
        geometry = make_geometry()
        candidates = (make_candidate(target=(2.0, 2.0)), make_candidate(target=(7.0, 7.0)))
        robot_capture = make_robot_capture(candidates=candidates, geometry=geometry)
        frame = make_frame((robot_capture,), geometry=geometry)
        decision = make_assembler().build(frame)

        obs = decision.get_observation(0)
        catalog = decision.get_action_catalog(0)
        for candidate_id, option in zip(obs.candidate_ids, catalog.options):
            assert candidate_id == option.candidate_id

    def test_action_mask_matches_enabled(self):
        geometry = make_geometry()
        candidates = (
            make_candidate(target=(2.0, 2.0), enabled=True, reachable=True),
            make_candidate(target=(7.0, 7.0), enabled=False, reachable=True),
        )
        robot_capture = make_robot_capture(candidates=candidates, geometry=geometry)
        frame = make_frame((robot_capture,), geometry=geometry)
        decision = make_assembler().build(frame)

        obs = decision.get_observation(0)
        catalog = decision.get_action_catalog(0)
        for mask_flag, option in zip(obs.action_mask, catalog.options):
            assert mask_flag == option.enabled


class TestResolveAction:
    def test_resolve_action_with_heading(self):
        geometry = make_geometry()
        candidate = make_candidate(target=(2.0, 2.0), heading_rad=0.4)
        robot_capture = make_robot_capture(candidates=(candidate,), geometry=geometry)
        frame = make_frame((robot_capture,), geometry=geometry)
        decision = make_assembler().build(frame)

        action = decision.resolve_action(robot_id=0, action_index=0, issued_at_step=5)
        assert action.heading_index == 0
        assert action.candidate_index == 0
        assert action.issued_at_step == 5

    def test_resolve_action_without_heading(self):
        geometry = make_geometry()
        candidate = make_candidate(target=(2.0, 2.0), heading_rad=None)
        robot_capture = make_robot_capture(candidates=(candidate,), geometry=geometry)
        frame = make_frame((robot_capture,), geometry=geometry)
        decision = make_assembler().build(frame)

        action = decision.resolve_action(robot_id=0, action_index=0, issued_at_step=0)
        assert action.heading_index is None


class TestMismatchDetection:
    def test_episode_id_mismatch(self):
        geometry = make_geometry()
        actor_a, _ = build_pair(geometry, episode_id="ep-A")
        _, catalog_b = build_pair(geometry, episode_id="ep-B")
        with pytest.raises(ValueError):
            DecisionCaptureBatch(actor_batch=actor_a, action_catalog_batch=catalog_b)

    def test_decision_step_mismatch(self):
        geometry = make_geometry()
        actor_a, _ = build_pair(geometry, decision_step=0)
        _, catalog_b = build_pair(geometry, decision_step=1)
        with pytest.raises(ValueError):
            DecisionCaptureBatch(actor_batch=actor_a, action_catalog_batch=catalog_b)

    def test_robot_order_mismatch(self):
        geometry = make_geometry()
        spec_ab = ((0, (make_candidate(),)), (1, (make_candidate(),)))
        spec_ba = ((1, (make_candidate(),)), (0, (make_candidate(),)))
        actor_a, _ = build_pair(geometry, robots_spec=spec_ab)
        _, catalog_b = build_pair(geometry, robots_spec=spec_ba)
        with pytest.raises(ValueError):
            DecisionCaptureBatch(actor_batch=actor_a, action_catalog_batch=catalog_b)

    def test_candidate_ids_mismatch(self):
        geometry = make_geometry()
        actor_batch, catalog_batch = build_pair(geometry)
        bad_option = dataclasses.replace(
            catalog_batch.catalogs[0].options[0],
            candidate_id="robot-0/step-0/candidate-999",
        )
        bad_catalog = RobotActionCatalog(robot_id=0, decision_step=0, options=(bad_option,))
        bad_batch = ActionCatalogBatch(
            episode_id=catalog_batch.episode_id,
            decision_step=catalog_batch.decision_step,
            time_s=catalog_batch.time_s,
            catalogs=(bad_catalog,),
        )
        with pytest.raises(ValueError):
            DecisionCaptureBatch(actor_batch=actor_batch, action_catalog_batch=bad_batch)

    def test_count_mismatch(self):
        geometry = make_geometry()
        actor_batch, catalog_batch = build_pair(
            geometry,
            robots_spec=((0, (make_candidate(target=(2.0, 2.0)), make_candidate(target=(9.0, 9.0)))),),
        )
        truncated_catalog = RobotActionCatalog(
            robot_id=0, decision_step=0, options=(catalog_batch.catalogs[0].options[0],)
        )
        truncated_batch = ActionCatalogBatch(
            episode_id=catalog_batch.episode_id,
            decision_step=catalog_batch.decision_step,
            time_s=catalog_batch.time_s,
            catalogs=(truncated_catalog,),
        )
        with pytest.raises(ValueError):
            DecisionCaptureBatch(actor_batch=actor_batch, action_catalog_batch=truncated_batch)

    def test_enabled_mismatch(self):
        geometry = make_geometry()
        actor_batch, catalog_batch = build_pair(geometry)
        original_option = catalog_batch.catalogs[0].options[0]
        flipped_option = dataclasses.replace(
            original_option, enabled=not original_option.enabled, reachable=True
        )
        flipped_catalog = RobotActionCatalog(robot_id=0, decision_step=0, options=(flipped_option,))
        flipped_batch = ActionCatalogBatch(
            episode_id=catalog_batch.episode_id,
            decision_step=catalog_batch.decision_step,
            time_s=catalog_batch.time_s,
            catalogs=(flipped_catalog,),
        )
        with pytest.raises(ValueError):
            DecisionCaptureBatch(actor_batch=actor_batch, action_catalog_batch=flipped_batch)


class TestBoundary:
    def test_no_ground_truth_field_names(self):
        for cls in (ActionOption, RobotActionCatalog, ActionCatalogBatch, DecisionCaptureBatch):
            names = {f.name for f in dataclasses.fields(cls)}
            for forbidden in ("ground_truth", "true_fire", "true_occupancy", "critic_state", "metadata"):
                assert forbidden not in names, (cls.__name__, forbidden)

    def test_action_option_stores_no_candidate_or_metadata_object(self):
        names = {f.name for f in dataclasses.fields(ActionOption)}
        assert names == {
            "robot_id", "candidate_id", "candidate_index", "action_index",
            "target_xy", "heading_rad", "kind", "source", "enabled", "reachable",
        }

    def test_robot_action_catalog_rejects_ground_truth_snapshot_as_option(self):
        snapshot = GroundTruthSnapshot(
            schema_version="0.1.0",
            decision_step=0,
            time_s=0.0,
            true_robot_poses={},
            true_occupancy=(),
            true_fire_locations=(),
            global_coverage_fraction=0.0,
        )
        with pytest.raises(TypeError):
            RobotActionCatalog(robot_id=0, decision_step=0, options=(snapshot,))

    @pytest.mark.parametrize("filename", DECISION_MODULES)
    def test_no_privileged_imports(self, filename):
        forbidden_roots = ("PyQt5", "PyQt6", "PySide2", "PySide6", "torch", "pandas")
        forbidden_modules = (
            "robotics_sim.simulation.engine",
            "robotics_sim.app",
            "robotics_sim.environment.hazard_field",
            "robotics_sim.diagnostics",
        )
        forbidden_names = (
            "HazardField", "FireSource", "HazardDebug", "HazardSourceDebug",
            "GroundTruthSnapshot", "CriticState",
        )
        tree = ast.parse((LEARNING_DIR / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module, name = alias.name, ""
                    self._assert_allowed(module, name, forbidden_roots, forbidden_modules, forbidden_names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                for alias in node.names:
                    self._assert_allowed(
                        node.module, alias.name, forbidden_roots, forbidden_modules, forbidden_names
                    )

    @staticmethod
    def _assert_allowed(module, name, forbidden_roots, forbidden_modules, forbidden_names):
        root = module.split(".")[0]
        assert root not in forbidden_roots, module
        assert not root.lower().startswith(("pyqt", "pyside")), module
        for forbidden_module in forbidden_modules:
            assert not module.startswith(forbidden_module), module
        assert name not in forbidden_names, (module, name)


class TestSmokeRealisticScenario:
    def test_two_robots_three_candidates_each_full_pipeline(self):
        geometry = make_geometry()
        candidate_spec = make_candidate_spec(max_candidates=3)
        schema = build_feature_schema_v0()

        def robot_candidates():
            return (
                make_candidate(
                    target=(2.0, 2.0), kind=CandidateKind.FRONTIER_VIEWPOINT, heading_rad=0.1
                ),
                make_candidate(target=(6.0, 6.0), kind=CandidateKind.FIRE_INFORMATION_VIEWPOINT),
                make_candidate(
                    target=(8.0, 1.0), kind=CandidateKind.RECOVERY_VIEWPOINT, heading_rad=1.2
                ),
            )

        robot0 = make_robot(robot_id=0, xy=(1.0, 1.0))
        robot1 = make_robot(robot_id=1, xy=(9.0, 9.0))
        shared_belief = HazardBelief(geometry).snapshot()

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
            episode_id="ep-smoke-decision",
            decision_step=3,
            time_s=1.5,
            robots=(r0_capture, r1_capture),
            grid_geometry=geometry,
            normalization=NORMALIZATION,
            candidate_spec=candidate_spec,
        )

        assembler = DecisionCaptureAssembler(
            actor_assembler=ActorObservationBatchAssembler(
                schema=schema, candidate_spec=candidate_spec
            ),
            catalog_assembler=ActionCatalogAssembler(),
        )
        decision = assembler.build(frame)

        assert len(decision.actor_batch.observations) == 2
        assert len(decision.action_catalog_batch.catalogs) == 2

        for robot_id in (0, 1):
            obs = decision.get_observation(robot_id)
            catalog = decision.get_action_catalog(robot_id)
            assert len(obs.candidate_ids) == 3
            assert len(catalog.options) == 3
            assert obs.candidate_ids == tuple(o.candidate_id for o in catalog.options)

        action = decision.resolve_action(robot_id=1, action_index=2, issued_at_step=3)
        assert action.robot_id == 1
        assert action.candidate_index == 2
        assert action.heading_index == 0  # candidate 2 has heading_rad=1.2
