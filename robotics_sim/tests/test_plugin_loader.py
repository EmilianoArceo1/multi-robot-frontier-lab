from __future__ import annotations

import importlib
import sys

import pytest

from robotics_interfaces import (
    CoordinationRequest,
    CoordinationResult,
    PluginCapability,
    PluginContractError,
    RobotCoordinationState,
)
from robotics_sim.simulation.plugin_loader import (
    PluginLoadError,
    discover_coordination_plugins,
    list_coordination_plugin_names,
    load_coordination_plugin,
)


def _write_fake_plugin_package(tmp_path):
    package_root = tmp_path / "tmp_algorithms"
    plugin_root = package_root / "fake_coordination"
    plugin_root.mkdir(parents=True)

    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (plugin_root / "__init__.py").write_text("", encoding="utf-8")
    (plugin_root / "plugin.py").write_text(
        """
from robotics_interfaces import (
    CoordinationResult,
    PluginCapability,
    PluginMetadata,
)


class FakeCoordinationPlugin:
    metadata = PluginMetadata(
        name="fake_coordination",
        version="0.1.0",
        description="Fake plugin used by plugin loader tests.",
        capabilities=(PluginCapability.COORDINATION,),
        source="unit-test",
    )

    def assign(self, request):
        targets = tuple(None for _ in request.robot_states)
        reasons = tuple("fake hold" for _ in request.robot_states)
        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.metadata.name,
            debug={"robot_count": len(request.robot_states)},
        )


def create_plugin():
    return FakeCoordinationPlugin()
""",
        encoding="utf-8",
    )
    return package_root


def test_plugin_loader_discovers_plugin_by_create_plugin_convention(tmp_path, monkeypatch):
    _write_fake_plugin_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    plugins = discover_coordination_plugins(base_package="tmp_algorithms")

    assert "fake_coordination" in plugins
    assert plugins["fake_coordination"].metadata.name == "fake_coordination"
    assert PluginCapability.COORDINATION in plugins["fake_coordination"].metadata.capabilities


def test_loaded_plugin_can_assign_using_coordination_contract(tmp_path, monkeypatch):
    _write_fake_plugin_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    plugin = load_coordination_plugin(
        "fake_coordination",
        base_package="tmp_algorithms",
    )
    request = CoordinationRequest(
        robot_states=(
            RobotCoordinationState(
                robot_id=0,
                xy=(0.0, 0.0),
                safety_radius=0.3,
                sensor_range=4.0,
                vision_model="limited_fov",
            ),
        )
    )

    result = plugin.assign(request)

    assert isinstance(result, CoordinationResult)
    assert result.strategy == "fake_coordination"
    assert result.targets == (None,)
    assert result.reasons == ("fake hold",)
    assert result.debug["robot_count"] == 1


def test_plugin_loader_lists_plugin_names(tmp_path, monkeypatch):
    _write_fake_plugin_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    names = list_coordination_plugin_names(base_package="tmp_algorithms")

    assert names == ("fake_coordination",)


def test_loading_missing_plugin_reports_available_names(tmp_path, monkeypatch):
    _write_fake_plugin_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    with pytest.raises(PluginLoadError, match="fake_coordination"):
        load_coordination_plugin("missing", base_package="tmp_algorithms")


def test_invalid_plugin_contract_fails_fast(tmp_path, monkeypatch):
    package_root = tmp_path / "bad_algorithms"
    plugin_root = package_root / "bad_plugin"
    plugin_root.mkdir(parents=True)

    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (plugin_root / "__init__.py").write_text("", encoding="utf-8")
    (plugin_root / "plugin.py").write_text(
        """
class BadPlugin:
    pass


def create_plugin():
    return BadPlugin()
""",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    with pytest.raises(PluginContractError):
        discover_coordination_plugins(base_package="bad_algorithms")
