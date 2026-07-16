"""
Contract-level tests for robotics_sim.diagnostics.

These do not exercise any producer (planner/collision/engine) -- they only
verify the neutral contract itself: no Qt/canvas/engine import anywhere in
the package, every public dataclass is frozen, and Maybe round-trips.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from robotics_sim.diagnostics.capture import NavigationDebugCapture, PlanDebugCapture
from robotics_sim.diagnostics.event_log import NavigationDebugEvent, NavigationDebugEventLog
from robotics_sim.diagnostics.navigation_snapshot import (
    ClearanceTerms,
    ControllerDebug,
    FrontierDebug,
    Maybe,
    NavigationDebugEventKind,
    NavigationDebugSnapshot,
    PathDebug,
    PlanningGridDebug,
    Pose,
    PredictedMotionDebug,
    RouteValidationDebug,
    SafetyDebug,
)

DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1] / "diagnostics"
FORBIDDEN_IMPORT_PREFIXES = (
    "PySide6",
    "PyQt5",
    "PyQt6",
    "qtpy",
    "robotics_sim.app",
    "robotics_sim.simulation",
)


# ---------------------------------------------------------------------------
# No Qt / canvas / engine import anywhere in the package.
# ---------------------------------------------------------------------------


def _imported_module_names(source: str) -> list[str]:
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("path", sorted(DIAGNOSTICS_DIR.glob("*.py")))
def test_diagnostics_module_imports_nothing_forbidden(path: Path):
    imported = _imported_module_names(path.read_text(encoding="utf-8"))
    offending = [
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)
    ]
    assert offending == [], f"{path.name} imports forbidden modules: {offending}"


def test_diagnostics_package_runtime_import_pulls_in_no_qt():
    import sys

    qt_mods_before = {m for m in sys.modules if "PySide" in m or "PyQt" in m}
    import robotics_sim.diagnostics.capture  # noqa: F401
    import robotics_sim.diagnostics.event_log  # noqa: F401
    import robotics_sim.diagnostics.navigation_snapshot  # noqa: F401

    qt_mods_after = {m for m in sys.modules if "PySide" in m or "PyQt" in m}
    assert qt_mods_after == qt_mods_before


# ---------------------------------------------------------------------------
# Every public contract dataclass is frozen.
# ---------------------------------------------------------------------------

FROZEN_TYPES = [
    Maybe,
    Pose,
    ClearanceTerms,
    PathDebug,
    RouteValidationDebug,
    PredictedMotionDebug,
    SafetyDebug,
    PlanningGridDebug,
    ControllerDebug,
    FrontierDebug,
    NavigationDebugSnapshot,
    NavigationDebugEvent,
]


@pytest.mark.parametrize("cls", FROZEN_TYPES)
def test_contract_dataclasses_are_frozen(cls):
    assert dataclasses.is_dataclass(cls)
    assert cls.__dataclass_params__.frozen is True


def test_frozen_instance_rejects_mutation():
    pose = Pose(x=1.0, y=2.0, theta=0.0, v=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pose.x = 5.0  # type: ignore[misc]


def test_capture_types_are_mutable_by_design():
    # Capture sinks are engine-internal plumbing, not the public contract --
    # they must stay mutable so producers can fill them in place.
    assert not PlanDebugCapture.__dataclass_params__.frozen
    assert not NavigationDebugCapture.__dataclass_params__.frozen


# ---------------------------------------------------------------------------
# Maybe round-trips.
# ---------------------------------------------------------------------------


def test_maybe_of_carries_value_and_is_available():
    m = Maybe.of(3.5)
    assert m.value == 3.5
    assert m.unavailable is False


def test_maybe_missing_carries_no_value_and_is_unavailable():
    m = Maybe.missing()
    assert m.value is None
    assert m.unavailable is True


# ---------------------------------------------------------------------------
# Event log ring buffer basics (deeper behavior tests live in
# test_navigation_debug_event_log.py once a real snapshot fixture exists).
# ---------------------------------------------------------------------------


def test_event_log_starts_empty():
    log = NavigationDebugEventLog(max_size=5)
    assert len(log) == 0
    assert log.latest() is None
    assert log.event_at(0) is None


def test_navigation_debug_event_kind_has_expected_members():
    expected = {
        "TICK",
        "PLAN_ACCEPTED",
        "PATH_SIMPLIFIED",
        "ROUTE_REJECTED",
        "SAFETY_REPLAN",
        "PREDICTED_COLLISION",
        "HOLD",
        "EXHAUSTED",
    }
    assert {member.value for member in NavigationDebugEventKind} == expected
