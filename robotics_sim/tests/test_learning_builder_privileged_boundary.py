"""Boundary tests: changing hidden ground truth must not change the actor
observation, and no privileged data can reach the actor builder."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

import robotics_sim.learning as learning_pkg
from robotics_interfaces.learning import CandidateKind, CandidateObservation, GroundTruthSnapshot
from robotics_sim.learning import (
    ActorObservationBuildInput,
    ActorObservationBuilder,
    CandidateFeatureSource,
    CriticStateBuildInput,
    CriticStateBuilder,
    FeatureSchema,
    GroundTruthBuildInput,
    GroundTruthSnapshotBuilder,
    TeammateFeatureSource,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent

SCHEMA = FeatureSchema(
    robot_feature_names=("x", "y"),
    candidate_feature_names=("dist",),
    teammate_feature_names=("rel_x",),
)


def visible_input() -> ActorObservationBuildInput:
    return ActorObservationBuildInput(
        schema=SCHEMA,
        robot_id=0,
        decision_step=2,
        time_s=1.0,
        robot_features={"x": 1.0, "y": 2.0},
        candidates=(
            CandidateFeatureSource(
                candidate=CandidateObservation(
                    candidate_id="c0",
                    kind=CandidateKind.FRONTIER_VIEWPOINT,
                    xy=(3.0, 4.0),
                    heading_candidates=(0.0,),
                    source="frontier",
                    reachable=True,
                ),
                features={"dist": 1.5},
                enabled=True,
            ),
        ),
        graph_edges=(),
        visible_teammates=(TeammateFeatureSource(robot_id=1, features={"rel_x": 0.5}),),
    )


def ground_truth_input(
    fire: tuple[tuple[float, float], ...],
    occupancy: tuple[tuple[int, ...], ...],
    coverage: float,
) -> GroundTruthBuildInput:
    return GroundTruthBuildInput(
        decision_step=2,
        time_s=1.0,
        true_robot_poses={0: (1.0, 2.0, 0.0)},
        true_occupancy=occupancy,
        true_fire_locations=fire,
        global_coverage_fraction=coverage,
    )


class TestHiddenWorldDoesNotLeakIntoActor:
    def test_same_visible_input_same_observation_despite_different_ground_truth(self):
        builder = ActorObservationBuilder()
        gt_builder = GroundTruthSnapshotBuilder()

        actor_a = builder.build(visible_input())
        ground_truth_a = gt_builder.build(
            ground_truth_input(
                fire=((10.0, 10.0),),
                occupancy=((0, 0), (0, 1)),
                coverage=0.2,
            )
        )

        # Completely different hidden world: fire moved, occupancy changed.
        actor_b = builder.build(visible_input())
        ground_truth_b = gt_builder.build(
            ground_truth_input(
                fire=((50.0, -3.0), (7.0, 8.0)),
                occupancy=((1, 1), (1, 0)),
                coverage=0.9,
            )
        )

        assert actor_a == actor_b
        assert ground_truth_a != ground_truth_b


class TestBuilderSignatures:
    def test_actor_build_takes_no_ground_truth_parameter(self):
        params = inspect.signature(ActorObservationBuilder.build).parameters
        assert set(params) == {"self", "build_input"}
        assert not any("ground" in name or "truth" in name for name in params)

    def test_actor_build_input_has_no_privileged_fields(self):
        field_names = {f.name for f in dataclasses.fields(ActorObservationBuildInput)}
        assert field_names == {
            "schema", "robot_id", "decision_step", "time_s", "robot_features",
            "candidates", "graph_edges", "visible_teammates",
        }
        for forbidden in ("ground_truth", "true_fire", "true_occupancy",
                          "critic_state", "privileged", "metadata"):
            assert forbidden not in field_names

    def test_critic_builder_rejects_nested_ground_truth(self):
        snapshot = GroundTruthSnapshotBuilder().build(
            ground_truth_input(fire=(), occupancy=(), coverage=0.0)
        )
        with pytest.raises(TypeError):
            CriticStateBuilder().build(snapshot)
        with pytest.raises(TypeError):
            CriticStateBuilder().build(
                CriticStateBuildInput(
                    decision_step=0,
                    time_s=0.0,
                    global_feature_names=(),
                    global_features={},
                    per_robot_feature_names=(),
                    per_robot_features={0: snapshot},
                )
            )


class TestImportHygiene:
    FORBIDDEN_ROOTS = ("PyQt5", "PyQt6", "PySide2", "PySide6", "numpy", "torch", "pandas")
    FORBIDDEN_MODULES = ("robotics_sim.app", "robotics_sim.simulation.engine")

    def _imports_of(self, filename: str):
        tree = ast.parse((LEARNING_DIR / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    yield node.module

    @pytest.mark.parametrize(
        "filename", ["__init__.py", "source_models.py", "builders.py", "recorder.py"]
    )
    def test_no_forbidden_imports(self, filename):
        imports = list(self._imports_of(filename))
        for module in imports:
            root = module.split(".")[0]
            assert root not in self.FORBIDDEN_ROOTS, f"{filename} imports {module}"
            assert not root.lower().startswith(("pyqt", "pyside")), f"{filename} imports {module}"
            for forbidden in self.FORBIDDEN_MODULES:
                assert not module.startswith(forbidden), f"{filename} imports {module}"

    def test_dependency_direction_only_towards_interfaces(self):
        for filename in ("source_models.py", "builders.py", "recorder.py"):
            for module in self._imports_of(filename):
                if module.startswith("robotics_interfaces"):
                    assert module.startswith("robotics_interfaces.learning"), (
                        f"{filename} must only use robotics_interfaces.learning, got {module}"
                    )
