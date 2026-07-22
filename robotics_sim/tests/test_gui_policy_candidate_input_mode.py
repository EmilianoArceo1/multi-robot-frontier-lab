"""Tests for gui_policy.py's migration to CandidateInputMode (Phase 9 of the
exploration-pipeline-architecture refactor). Covers all five
CandidateInputMode branches end to end through real, migrated plugins (see
test_plugin_candidate_input_mode_migration.py for the metadata migration
itself) plus the LEGACY_INTEGRATED warning banner.
"""

from __future__ import annotations

from algorithms.cqlite.plugin import CQLITE_COORDINATOR
from algorithms.frontier_cluster_hungarian.plugin import FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR
from algorithms.fuel_frontier_baseline.plugin import FUEL_FRONTIER_BASELINE_COORDINATOR
from algorithms.global_noic_legacy.plugin import NOIC_COORDINATOR
from algorithms.independent_baseline.plugin import INDEPENDENT_BASELINE_COORDINATOR
from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from algorithms.nav2d_wavefront.plugin import NAV2D_WAVEFRONT_COORDINATOR
from robotics_interfaces.plugins import CandidateInputMode
from robotics_sim.simulation.coordination import runtime_profile_for_strategy
from robotics_sim.simulation.gui_policy import compute_gui_control_policy


def _policy(name: str):
    return compute_gui_control_policy(runtime_profile_for_strategy(name))


def test_host_candidates_disables_exploration_planner_for_mmpf_independent_and_cqlite():
    for name in (MMPF_COORDINATOR, INDEPENDENT_BASELINE_COORDINATOR, CQLITE_COORDINATOR):
        policy = _policy(name)
        assert policy.candidate_input_mode is CandidateInputMode.HOST_CANDIDATES
        assert policy.exploration_planner_enabled is False
        assert "host provides candidates" in policy.exploration_planner_reason.lower()
        assert policy.legacy_pipeline_warning is None


def test_host_frontier_clusters_disables_exploration_planner_for_hungarian():
    policy = _policy(FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR)

    assert policy.candidate_input_mode is CandidateInputMode.HOST_FRONTIER_CLUSTERS
    assert policy.exploration_planner_enabled is False
    assert "cluster" in policy.exploration_planner_reason.lower()
    assert policy.legacy_pipeline_warning is None


def test_plugin_internal_disables_exploration_planner_for_nav2d():
    policy = _policy(NAV2D_WAVEFRONT_COORDINATOR)

    assert policy.candidate_input_mode is CandidateInputMode.PLUGIN_INTERNAL
    assert policy.exploration_planner_enabled is False
    assert "internal to this algorithm" in policy.exploration_planner_reason
    assert policy.legacy_pipeline_warning is None


def test_hybrid_disables_exploration_planner_for_fuel_and_explains_fallback():
    policy = _policy(FUEL_FRONTIER_BASELINE_COORDINATOR)

    assert policy.candidate_input_mode is CandidateInputMode.HYBRID
    assert policy.exploration_planner_enabled is False
    assert "fallback" in policy.exploration_planner_reason.lower()
    assert policy.legacy_pipeline_warning is None


def test_legacy_integrated_keeps_exploration_planner_enabled_with_a_warning_for_noic():
    policy = _policy(NOIC_COORDINATOR)

    assert policy.candidate_input_mode is CandidateInputMode.LEGACY_INTEGRATED
    assert policy.exploration_planner_enabled is True
    assert policy.legacy_pipeline_warning is not None
    assert "legacy" in policy.legacy_pipeline_warning.lower()


def test_only_legacy_integrated_ever_sets_a_pipeline_warning():
    for name in (
        MMPF_COORDINATOR,
        INDEPENDENT_BASELINE_COORDINATOR,
        CQLITE_COORDINATOR,
        FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR,
        NAV2D_WAVEFRONT_COORDINATOR,
        FUEL_FRONTIER_BASELINE_COORDINATOR,
    ):
        assert _policy(name).legacy_pipeline_warning is None
    assert _policy(NOIC_COORDINATOR).legacy_pipeline_warning is not None
