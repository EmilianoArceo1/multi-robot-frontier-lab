from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

from robotics_interfaces.coordination import CoordinationRequest, CoordinationResult


class PluginCapability(str, Enum):
    """Decision levels a plugin may implement.

    The first implemented plugin type is coordination, but the enum already
    names the levels we need for future papers: target generation, task
    allocation, path planning, local control, map updates, and full-stack
    policies.
    """

    TARGET_GENERATION = "target_generation"
    TASK_ALLOCATION = "task_allocation"
    COORDINATION = "coordination"
    PATH_PLANNING = "path_planning"
    CONTROL = "control"
    MAP_UPDATE = "map_update"
    PARAMETER_PATCH = "parameter_patch"
    FULL_STACK = "full_stack"


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    description: str
    capabilities: tuple[PluginCapability, ...]
    source: str = ""


@runtime_checkable
class CoordinationPlugin(Protocol):
    """Contract implemented by external coordination plugins."""

    metadata: PluginMetadata

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        ...


PluginFactory = Callable[[], CoordinationPlugin]


class PluginContractError(TypeError):
    """Raised when a discovered object does not satisfy the plugin contract."""


def validate_coordination_plugin(plugin: object) -> CoordinationPlugin:
    """Validate the runtime shape of a coordination plugin.

    We avoid relying only on isinstance(..., Protocol) because protocols with
    attributes can hide weak runtime checks. Explicit checks give clearer errors
    during plugin development.
    """

    metadata = getattr(plugin, "metadata", None)
    if not isinstance(metadata, PluginMetadata):
        raise PluginContractError("plugin.metadata must be a PluginMetadata instance")

    if not metadata.name or not isinstance(metadata.name, str):
        raise PluginContractError("plugin.metadata.name must be a non-empty string")

    if PluginCapability.COORDINATION not in metadata.capabilities:
        raise PluginContractError(
            f"plugin {metadata.name!r} must declare PluginCapability.COORDINATION"
        )

    assign = getattr(plugin, "assign", None)
    if not callable(assign):
        raise PluginContractError(f"plugin {metadata.name!r} must define assign(request)")

    return plugin  # type: ignore[return-value]
