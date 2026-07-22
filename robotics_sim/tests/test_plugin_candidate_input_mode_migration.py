"""Locks in Phase 5's plugin metadata migration: every shipped algorithm now
declares an explicit CandidateInputMode instead of relying on the
best-effort fallback in build_runtime_profile(), and the new stage
capabilities (FRONTIER_DETECTION/TASK_GENERATION) are only added where the
plugin genuinely performs that stage. No plugin's scoring/allocation logic
changed in this migration -- only metadata.
"""

from __future__ import annotations

from algorithms.cqlite.plugin import CQLITE_COORDINATOR
from algorithms.frontier_cluster_hungarian.plugin import FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR
from algorithms.fuel_frontier_baseline.plugin import FUEL_FRONTIER_BASELINE_COORDINATOR
from algorithms.global_noic_legacy.plugin import NOIC_COORDINATOR
from algorithms.independent_baseline.plugin import INDEPENDENT_BASELINE_COORDINATOR
from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from algorithms.nav2d_wavefront.plugin import NAV2D_WAVEFRONT_COORDINATOR
from robotics_interfaces.plugins import CandidateInputMode, PluginCapability
from robotics_sim.simulation.plugin_loader import load_coordination_plugin


def _capabilities(name: str) -> tuple[PluginCapability, ...]:
    return load_coordination_plugin(name).metadata.capabilities


def _mode(name: str) -> CandidateInputMode | None:
    return load_coordination_plugin(name).metadata.candidate_input_mode


def test_global_noic_legacy_is_legacy_integrated_task_allocation_only():
    assert _mode(NOIC_COORDINATOR) is CandidateInputMode.LEGACY_INTEGRATED
    caps = _capabilities(NOIC_COORDINATOR)
    assert PluginCapability.TASK_ALLOCATION in caps
    assert PluginCapability.FRONTIER_DETECTION not in caps
    assert PluginCapability.TASK_GENERATION not in caps


def test_independent_baseline_is_host_candidates():
    assert _mode(INDEPENDENT_BASELINE_COORDINATOR) is CandidateInputMode.HOST_CANDIDATES
    caps = _capabilities(INDEPENDENT_BASELINE_COORDINATOR)
    assert PluginCapability.FRONTIER_DETECTION not in caps


def test_mmpf_explore_is_host_candidates():
    assert _mode(MMPF_COORDINATOR) is CandidateInputMode.HOST_CANDIDATES
    caps = _capabilities(MMPF_COORDINATOR)
    assert PluginCapability.FRONTIER_DETECTION not in caps


def test_cqlite_is_host_candidates_and_documents_simulated_communication():
    from robotics_interfaces.coordination import CoordinationRequest
    from robotics_interfaces.observations import RobotCoordinationState

    assert _mode(CQLITE_COORDINATOR) is CandidateInputMode.HOST_CANDIDATES
    caps = _capabilities(CQLITE_COORDINATOR)
    assert PluginCapability.FRONTIER_DETECTION not in caps

    plugin = load_coordination_plugin(CQLITE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            RobotCoordinationState(
                robot_id=0, xy=(0.0, 0.0), safety_radius=0.35, sensor_range=2.5, vision_model="Camera / FoV"
            ),
        ),
        robots_to_assign=(0,),
    )
    result = plugin.assign(request)
    assert result.debug["communication"]["simulated"] is True


def test_fuel_frontier_baseline_is_hybrid_with_task_generation():
    assert _mode(FUEL_FRONTIER_BASELINE_COORDINATOR) is CandidateInputMode.HYBRID
    caps = _capabilities(FUEL_FRONTIER_BASELINE_COORDINATOR)
    assert PluginCapability.TASK_GENERATION in caps
    assert PluginCapability.FRONTIER_DETECTION not in caps


def test_frontier_cluster_hungarian_is_host_frontier_clusters_with_task_generation():
    assert _mode(FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR) is CandidateInputMode.HOST_FRONTIER_CLUSTERS
    caps = _capabilities(FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR)
    assert PluginCapability.TASK_GENERATION in caps
    assert PluginCapability.FRONTIER_DETECTION not in caps
    assert PluginCapability.TARGET_GENERATION not in caps  # never claimed detection


def test_nav2d_wavefront_is_plugin_internal_with_frontier_detection_and_task_generation():
    assert _mode(NAV2D_WAVEFRONT_COORDINATOR) is CandidateInputMode.PLUGIN_INTERNAL
    caps = _capabilities(NAV2D_WAVEFRONT_COORDINATOR)
    assert PluginCapability.FRONTIER_DETECTION in caps
    assert PluginCapability.TASK_GENERATION in caps


def test_every_shipped_plugin_now_declares_an_explicit_candidate_input_mode():
    """No shipped plugin should still rely on build_runtime_profile()'s
    best-effort fallback after Phase 5 -- only third-party/unmigrated
    plugins should ever hit that fallback."""
    for name in (
        NOIC_COORDINATOR,
        INDEPENDENT_BASELINE_COORDINATOR,
        MMPF_COORDINATOR,
        CQLITE_COORDINATOR,
        FUEL_FRONTIER_BASELINE_COORDINATOR,
        FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR,
        NAV2D_WAVEFRONT_COORDINATOR,
    ):
        assert _mode(name) is not None, f"{name} should declare candidate_input_mode explicitly"
