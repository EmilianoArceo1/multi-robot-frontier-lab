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


def declares_capability(metadata: PluginMetadata, capability: PluginCapability) -> bool:
    """Runtime-clear query for whether a plugin owns a given responsibility.

    Hosts (e.g. robotics_sim.simulation.coordination) should use this instead of
    reaching into metadata.capabilities directly, so ownership checks stay
    readable and do not require importing GUI/engine internals.
    """

    return capability in metadata.capabilities


# Ownership-flavored alias of declares_capability. Kept as a separate name
# because PluginRuntimeProfile/build_runtime_profile read more naturally in
# "does this plugin own X" terms than in raw capability-membership terms.
plugin_owns = declares_capability


@dataclass(frozen=True)
class PluginRuntimeProfile:
    """What a coordination plugin actually controls at runtime.

    This is the operative counterpart to metadata.capabilities: the runtime
    (robotics_sim.simulation.coordination) and the GUI both read this instead
    of re-deriving ownership from raw capability tuples, so "MMPF owns target
    generation" is decided in exactly one place.
    """

    owns_target_generation: bool
    owns_task_allocation: bool
    owns_path_planning: bool
    owns_control: bool
    uses_legacy_frontier_service: bool = False
    uses_external_path_planner: bool = True
    uses_external_motion_controller: bool = True


def build_runtime_profile(metadata: PluginMetadata) -> PluginRuntimeProfile:
    """Derive a PluginRuntimeProfile from a plugin's declared capabilities.

    FULL_STACK is treated as owning target generation, task allocation, path
    planning, and control even if a plugin does not also redundantly list
    those four capabilities individually.
    """

    full_stack = plugin_owns(metadata, PluginCapability.FULL_STACK)
    owns_target_generation = full_stack or plugin_owns(metadata, PluginCapability.TARGET_GENERATION)
    owns_task_allocation = full_stack or plugin_owns(metadata, PluginCapability.TASK_ALLOCATION)
    owns_path_planning = full_stack or plugin_owns(metadata, PluginCapability.PATH_PLANNING)
    owns_control = full_stack or plugin_owns(metadata, PluginCapability.CONTROL)

    return PluginRuntimeProfile(
        owns_target_generation=owns_target_generation,
        owns_task_allocation=owns_task_allocation,
        owns_path_planning=owns_path_planning,
        owns_control=owns_control,
        uses_legacy_frontier_service=not owns_target_generation,
        uses_external_path_planner=not owns_path_planning,
        uses_external_motion_controller=not owns_control,
    )
