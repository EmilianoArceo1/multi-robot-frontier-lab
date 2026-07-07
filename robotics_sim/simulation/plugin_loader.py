from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from types import ModuleType
from typing import Iterable

from robotics_interfaces.plugins import (
    CoordinationPlugin,
    PluginContractError,
    validate_coordination_plugin,
)


class PluginLoadError(RuntimeError):
    """Raised when a plugin package cannot be imported or instantiated."""


@dataclass(frozen=True)
class PluginDiscoveryReport:
    """Optional debug data for plugin discovery."""

    base_package: str
    discovered_names: tuple[str, ...]
    skipped_modules: tuple[str, ...]


def discover_coordination_plugins(
    base_package: str = "algorithms",
) -> dict[str, CoordinationPlugin]:
    """Discover coordination plugins under a Python package.

    Discovery convention:
        algorithms/<plugin_name>/plugin.py
        plugin.py exposes create_plugin() -> CoordinationPlugin

    This is deliberately simple and debuggable. It does not use entry points,
    dynamic code execution, YAML, or reflection beyond importing plugin.py.
    """

    package = _import_package(base_package)
    plugin_modules = _iter_plugin_modules(package, base_package)

    plugins: dict[str, CoordinationPlugin] = {}
    for module in plugin_modules:
        plugin = _create_plugin_from_module(module)
        metadata_name = plugin.metadata.name
        if metadata_name in plugins:
            raise PluginLoadError(f"duplicate coordination plugin name: {metadata_name!r}")
        plugins[metadata_name] = plugin

    return plugins


def load_coordination_plugin(
    name: str,
    base_package: str = "algorithms",
) -> CoordinationPlugin:
    """Load one coordination plugin by metadata name."""

    plugins = discover_coordination_plugins(base_package=base_package)
    try:
        return plugins[name]
    except KeyError as exc:
        available = ", ".join(sorted(plugins)) or "<none>"
        raise PluginLoadError(
            f"coordination plugin {name!r} was not found. Available: {available}"
        ) from exc


def list_coordination_plugin_names(base_package: str = "algorithms") -> tuple[str, ...]:
    """Return discoverable coordination plugin names sorted alphabetically."""

    return tuple(sorted(discover_coordination_plugins(base_package=base_package)))


def _import_package(base_package: str) -> ModuleType:
    try:
        package = importlib.import_module(base_package)
    except ModuleNotFoundError as exc:
        raise PluginLoadError(f"plugin base package {base_package!r} was not found") from exc

    if not hasattr(package, "__path__"):
        raise PluginLoadError(f"plugin base package {base_package!r} is not a package")

    return package


def _iter_plugin_modules(package: ModuleType, base_package: str) -> Iterable[ModuleType]:
    for module_info in pkgutil.iter_modules(package.__path__):  # type: ignore[attr-defined]
        plugin_module_name = f"{base_package}.{module_info.name}.plugin"
        try:
            yield importlib.import_module(plugin_module_name)
        except ModuleNotFoundError as exc:
            # Skip packages that simply are not plugins. Do not swallow import
            # errors raised inside an existing plugin.py; those are real bugs.
            if exc.name == plugin_module_name:
                continue
            raise PluginLoadError(
                f"failed while importing plugin module {plugin_module_name!r}"
            ) from exc


def _create_plugin_from_module(module: ModuleType) -> CoordinationPlugin:
    factory = getattr(module, "create_plugin", None)
    if not callable(factory):
        raise PluginLoadError(f"{module.__name__} must expose create_plugin()")

    try:
        plugin = factory()
        return validate_coordination_plugin(plugin)
    except PluginContractError:
        raise
    except Exception as exc:  # pragma: no cover - defensive; tested via behavior above.
        raise PluginLoadError(f"failed to create plugin from {module.__name__}") from exc
