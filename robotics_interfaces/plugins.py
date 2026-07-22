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

    TARGET_GENERATION is semantically overloaded and DEPRECATED: several
    plugins declare it while only consuming candidates a host service already
    generated (see FRONTIER_DETECTION/TASK_GENERATION below for the split
    that replaces it). It is kept for backward compatibility with saved
    metadata, existing tests, and PluginRuntimeProfile.owns_target_generation
    -- new code should read CandidateInputMode + the new stage capabilities
    instead of TARGET_GENERATION, and no *new* plugin should declare it.
    """

    TARGET_GENERATION = "target_generation"
    TASK_ALLOCATION = "task_allocation"
    COORDINATION = "coordination"
    PATH_PLANNING = "path_planning"
    CONTROL = "control"
    MAP_UPDATE = "map_update"
    PARAMETER_PATCH = "parameter_patch"
    FULL_STACK = "full_stack"

    # Stage-accurate replacements for TARGET_GENERATION. A plugin declares
    # exactly the stages it actually performs:
    #   FRONTIER_DETECTION -- it computes frontiers/unknown-space boundaries
    #       itself instead of consuming a host-provided candidate/cluster
    #       pool (e.g. nav2d_wavefront's internal grid+BFS).
    #   TASK_GENERATION -- it turns raw candidates/clusters into a reduced
    #       set of tasks (e.g. frontier_cluster_hungarian's cluster-to-task
    #       reduction), as opposed to merely picking one of the candidates
    #       it was handed.
    FRONTIER_DETECTION = "frontier_detection"
    TASK_GENERATION = "task_generation"


class CandidateInputMode(str, Enum):
    """Where a plugin's exploration candidates actually come from.

    This is the explicit policy PluginMetadata declares instead of letting
    hosts/GUI/reasoning panels infer it from capabilities or plugin names.

    HOST_CANDIDATES        -- consumes flat ExplorationCandidate pools from
                              FrontierProvider/TeamFrontierProvider.
    HOST_FRONTIER_CLUSTERS -- consumes FrontierCluster objects from
                              FrontierInformationService (host detects
                              connected components; plugin reduces/allocates).
    PLUGIN_INTERNAL        -- the plugin detects/generates candidates itself
                              and must not depend on host frontier services.
    HYBRID                 -- can use host-provided input AND its own
                              generation/fallback; must export which source
                              was actually used for a given decision.
    LEGACY_INTEGRATED      -- a legacy adapter where detection/allocation are
                              not actually separated; must identify itself as
                              legacy rather than claim a separation that does
                              not exist.
    """

    HOST_CANDIDATES = "host_candidates"
    HOST_FRONTIER_CLUSTERS = "host_frontier_clusters"
    PLUGIN_INTERNAL = "plugin_internal"
    HYBRID = "hybrid"
    LEGACY_INTEGRATED = "legacy_integrated"


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    description: str
    capabilities: tuple[PluginCapability, ...]
    source: str = ""
    # None means "not yet migrated" -- see build_runtime_profile(), which
    # falls back to a best-effort classification derived from capabilities
    # for plugins that have not declared this explicitly yet.
    candidate_input_mode: "CandidateInputMode | None" = None


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
    of re-deriving ownership from raw capability tuples.

    detects_frontiers/generates_tasks/allocates_tasks/plans_paths/
    controls_motion/candidate_input_mode/supports_periodic_replan/
    uses_external_candidate_pipeline are the semantically-correct fields new
    code (gui_policy, coordination_scheduler, the reasoning panels) must
    read.

    owns_target_generation/owns_task_allocation/owns_path_planning/
    owns_control/uses_legacy_frontier_service are DEPRECATED compatibility
    fields kept for existing callers (logging in
    robotics_sim.simulation.coordination, older tests). They are computed
    from the legacy TARGET_GENERATION-or-FULL_STACK rule, which is why they
    can disagree with detects_frontiers/generates_tasks during the migration
    window: e.g. mmpf_explore still declares TARGET_GENERATION (so
    owns_target_generation is True) while candidate_input_mode is
    HOST_CANDIDATES and detects_frontiers/generates_tasks are both False,
    because MMPF only picks one of the candidates the host handed it. Do not
    use the deprecated fields in new code.
    """

    # --- Semantically-correct fields (use these in new code). ---
    detects_frontiers: bool
    generates_tasks: bool
    allocates_tasks: bool
    plans_paths: bool
    controls_motion: bool
    candidate_input_mode: CandidateInputMode
    supports_periodic_replan: bool = True
    uses_external_candidate_pipeline: bool = True
    uses_external_path_planner: bool = True
    uses_external_motion_controller: bool = True

    # --- Deprecated compatibility fields. Do not use in new code. ---
    owns_target_generation: bool = False
    owns_task_allocation: bool = False
    owns_path_planning: bool = False
    owns_control: bool = False
    uses_legacy_frontier_service: bool = False


def _fallback_candidate_input_mode(
    metadata: PluginMetadata,
    *,
    detects_frontiers: bool,
    owns_target_generation: bool,
) -> CandidateInputMode:
    """Best-effort CandidateInputMode for a plugin that has not declared one.

    This only matters until every shipped plugin sets
    PluginMetadata.candidate_input_mode explicitly (see Phase 5 of the
    exploration-pipeline-architecture refactor) -- it is a conservative guess
    from capabilities alone, not a substitute for the real classification a
    plugin author should provide. In particular it cannot tell
    HOST_FRONTIER_CLUSTERS apart from LEGACY_INTEGRATED for a plugin that
    only declares TASK_ALLOCATION (e.g. an unmigrated
    frontier_cluster_hungarian would fall through to LEGACY_INTEGRATED here
    even though it actually reads FrontierInformationService) -- Phase 5
    exists specifically to remove that ambiguity by declaring the mode
    explicitly instead of relying on this fallback.
    """

    if detects_frontiers:
        return CandidateInputMode.PLUGIN_INTERNAL
    if owns_target_generation:
        return CandidateInputMode.HOST_CANDIDATES
    return CandidateInputMode.LEGACY_INTEGRATED


def build_runtime_profile(metadata: PluginMetadata) -> PluginRuntimeProfile:
    """Derive a PluginRuntimeProfile from a plugin's declared capabilities.

    FULL_STACK is treated as owning every stage (frontier detection through
    control) even if a plugin does not also redundantly list those
    capabilities individually.
    """

    full_stack = plugin_owns(metadata, PluginCapability.FULL_STACK)

    detects_frontiers = full_stack or plugin_owns(metadata, PluginCapability.FRONTIER_DETECTION)
    generates_tasks = full_stack or plugin_owns(metadata, PluginCapability.TASK_GENERATION)
    allocates_tasks = full_stack or plugin_owns(metadata, PluginCapability.TASK_ALLOCATION)
    plans_paths = full_stack or plugin_owns(metadata, PluginCapability.PATH_PLANNING)
    controls_motion = full_stack or plugin_owns(metadata, PluginCapability.CONTROL)

    # Deprecated fields: computed from the legacy rule only, so they keep
    # producing exactly what they always have for existing plugins/tests
    # even as detects_frontiers/generates_tasks take over the real meaning.
    owns_target_generation = full_stack or plugin_owns(metadata, PluginCapability.TARGET_GENERATION)
    owns_task_allocation = allocates_tasks
    owns_path_planning = plans_paths
    owns_control = controls_motion

    candidate_input_mode = metadata.candidate_input_mode or _fallback_candidate_input_mode(
        metadata,
        detects_frontiers=detects_frontiers,
        owns_target_generation=owns_target_generation,
    )

    return PluginRuntimeProfile(
        detects_frontiers=detects_frontiers,
        generates_tasks=generates_tasks,
        allocates_tasks=allocates_tasks,
        plans_paths=plans_paths,
        controls_motion=controls_motion,
        candidate_input_mode=candidate_input_mode,
        supports_periodic_replan=True,
        uses_external_candidate_pipeline=not (detects_frontiers or generates_tasks),
        uses_external_path_planner=not plans_paths,
        uses_external_motion_controller=not controls_motion,
        owns_target_generation=owns_target_generation,
        owns_task_allocation=owns_task_allocation,
        owns_path_planning=owns_path_planning,
        owns_control=owns_control,
        uses_legacy_frontier_service=not owns_target_generation,
    )
