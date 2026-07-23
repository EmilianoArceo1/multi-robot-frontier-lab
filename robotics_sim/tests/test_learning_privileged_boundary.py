"""Tests for the actor / critic / ground-truth privileged-information
boundary and the import hygiene of robotics_interfaces.learning."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

import robotics_interfaces.learning as learning_pkg
from robotics_interfaces.learning import (
    ActorObservation,
    CriticState,
    FORBIDDEN_ACTOR_FIELDS,
    GroundTruthSnapshot,
    LearningTransition,
)

LEARNING_DIR = Path(learning_pkg.__file__).resolve().parent

FORBIDDEN_IMPORT_ROOTS = (
    "robotics_sim",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "numpy",
    "torch",
    "pandas",
)


class TestActorObservationBoundary:
    def test_no_forbidden_fields(self):
        field_names = {f.name for f in dataclasses.fields(ActorObservation)}
        for forbidden in ("ground_truth", "true_fire", "true_occupancy",
                          "critic_state", "privileged", "metadata"):
            assert forbidden not in field_names, (
                f"ActorObservation must not expose field {forbidden!r}"
            )

    def test_forbidden_field_list_is_published(self):
        assert set(FORBIDDEN_ACTOR_FIELDS) >= {
            "ground_truth", "true_fire", "true_occupancy",
            "critic_state", "privileged", "metadata",
        }

    def test_no_attribute_gives_access_to_privileged_types(self):
        annotations = {
            f.name: str(f.type) for f in dataclasses.fields(ActorObservation)
        }
        for name, annotation in annotations.items():
            assert "CriticState" not in annotation, name
            assert "GroundTruthSnapshot" not in annotation, name


class TestTypeIndependence:
    def test_no_inheritance_between_the_three_blocks(self):
        types = (ActorObservation, CriticState, GroundTruthSnapshot)
        for a in types:
            for b in types:
                if a is not b:
                    assert not issubclass(a, b), f"{a.__name__} must not inherit {b.__name__}"

    def test_types_are_distinct(self):
        assert len({ActorObservation, CriticState, GroundTruthSnapshot}) == 3

    def test_critic_state_rejects_ground_truth_payload(self):
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
            CriticState(
                schema_version="0.1.0",
                decision_step=0,
                time_s=0.0,
                global_feature_names=(),
                global_features=(),
                per_robot_feature_names=(),
                per_robot_features={0: snapshot},
            )

    def test_learning_transition_has_no_ground_truth_field(self):
        field_names = {f.name for f in dataclasses.fields(LearningTransition)}
        assert "ground_truth" not in field_names
        assert not any("ground_truth" in name for name in field_names)


class TestImportHygiene:
    def _iter_import_roots(self, path: Path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    yield node.module.split(".")[0]

    def test_learning_modules_do_not_import_forbidden_packages(self):
        module_files = sorted(LEARNING_DIR.glob("*.py"))
        assert module_files, f"no modules found under {LEARNING_DIR}"
        for path in module_files:
            roots = set(self._iter_import_roots(path))
            for forbidden in FORBIDDEN_IMPORT_ROOTS:
                assert forbidden not in roots, (
                    f"{path.name} imports forbidden package {forbidden}"
                )
            # Qt can also sneak in via lowercase spellings.
            assert not any(r.lower().startswith(("pyqt", "pyside")) for r in roots), (
                f"{path.name} imports a Qt binding"
            )

    def test_learning_package_imports_cleanly_without_forbidden_modules(self):
        import subprocess
        import sys

        code = (
            "import sys\n"
            "import robotics_interfaces.learning\n"
            "bad = [m for m in sys.modules\n"
            "       if m.split('.')[0] in ("
            "'robotics_sim', 'numpy', 'torch', 'pandas',"
            " 'PyQt5', 'PyQt6', 'PySide2', 'PySide6')]\n"
            "assert not bad, f'forbidden modules loaded: {bad}'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(LEARNING_DIR.parent.parent),
        )
        assert result.returncode == 0, result.stderr
