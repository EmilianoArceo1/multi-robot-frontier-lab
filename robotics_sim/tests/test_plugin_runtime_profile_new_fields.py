"""Tests for the semantically-correct PluginRuntimeProfile fields added in
Phase 4 of the exploration-pipeline-architecture refactor, alongside the
still-supported deprecated owns_* fields.

See robotics_interfaces/plugins.py: PluginRuntimeProfile now has both an
authoritative set (detects_frontiers/generates_tasks/allocates_tasks/
plans_paths/controls_motion/candidate_input_mode/...) and a deprecated
compatibility set (owns_target_generation/...), computed independently so
existing behavior for unmigrated plugins does not change.
"""

from __future__ import annotations

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from robotics_interfaces.plugins import (
    CandidateInputMode,
    PluginCapability,
    PluginMetadata,
    build_runtime_profile,
)
from robotics_sim.simulation.plugin_loader import load_coordination_plugin


def _metadata(capabilities, candidate_input_mode=None) -> PluginMetadata:
    return PluginMetadata(
        name="test plugin",
        version="0.0.0",
        description="",
        capabilities=tuple(capabilities),
        candidate_input_mode=candidate_input_mode,
    )


def test_full_stack_owns_every_new_stage_field():
    profile = build_runtime_profile(_metadata([PluginCapability.COORDINATION, PluginCapability.FULL_STACK]))

    assert profile.detects_frontiers is True
    assert profile.generates_tasks is True
    assert profile.allocates_tasks is True
    assert profile.plans_paths is True
    assert profile.controls_motion is True


def test_frontier_detection_capability_sets_detects_frontiers_only():
    profile = build_runtime_profile(
        _metadata([PluginCapability.COORDINATION, PluginCapability.FRONTIER_DETECTION])
    )

    assert profile.detects_frontiers is True
    assert profile.generates_tasks is False
    assert profile.allocates_tasks is False


def test_explicit_candidate_input_mode_is_used_verbatim():
    profile = build_runtime_profile(
        _metadata(
            [PluginCapability.COORDINATION, PluginCapability.TASK_ALLOCATION],
            candidate_input_mode=CandidateInputMode.HOST_FRONTIER_CLUSTERS,
        )
    )

    assert profile.candidate_input_mode is CandidateInputMode.HOST_FRONTIER_CLUSTERS


def test_fallback_candidate_input_mode_for_frontier_detecting_plugin_is_plugin_internal():
    profile = build_runtime_profile(
        _metadata([PluginCapability.COORDINATION, PluginCapability.FRONTIER_DETECTION])
    )
    assert profile.candidate_input_mode is CandidateInputMode.PLUGIN_INTERNAL


def test_fallback_candidate_input_mode_for_legacy_target_generation_is_host_candidates():
    profile = build_runtime_profile(
        _metadata(
            [PluginCapability.COORDINATION, PluginCapability.TARGET_GENERATION, PluginCapability.TASK_ALLOCATION]
        )
    )
    assert profile.candidate_input_mode is CandidateInputMode.HOST_CANDIDATES


def test_fallback_candidate_input_mode_for_task_allocation_only_is_legacy_integrated():
    profile = build_runtime_profile(
        _metadata([PluginCapability.COORDINATION, PluginCapability.TASK_ALLOCATION])
    )
    assert profile.candidate_input_mode is CandidateInputMode.LEGACY_INTEGRATED


def test_uses_external_candidate_pipeline_is_true_unless_the_plugin_generates_its_own():
    consumer_profile = build_runtime_profile(
        _metadata([PluginCapability.COORDINATION, PluginCapability.TASK_ALLOCATION])
    )
    generator_profile = build_runtime_profile(
        _metadata([PluginCapability.COORDINATION, PluginCapability.FRONTIER_DETECTION])
    )

    assert consumer_profile.uses_external_candidate_pipeline is True
    assert generator_profile.uses_external_candidate_pipeline is False


def test_mmpf_deprecated_and_new_fields_intentionally_disagree_during_migration():
    """MMPF still declares TARGET_GENERATION (deprecated field stays True for
    compatibility) but does not detect frontiers or generate tasks itself --
    it only ranks candidates the host handed it. This divergence is the
    documented, intentional state until MMPF's own metadata is migrated in
    Phase 5; this test exists so that migration has a concrete before/after
    to diff against."""
    plugin = load_coordination_plugin(MMPF_COORDINATOR)
    profile = build_runtime_profile(plugin.metadata)

    assert profile.owns_target_generation is True  # deprecated field, legacy rule
    assert profile.detects_frontiers is False  # new field, real behavior
    assert profile.generates_tasks is False  # new field, real behavior


def test_deprecated_fields_default_safely_when_not_supplied_directly():
    """PluginRuntimeProfile can still be constructed with only the new
    required fields (e.g. by a future test double); the deprecated fields
    must not become mandatory."""
    from robotics_interfaces.plugins import PluginRuntimeProfile

    profile = PluginRuntimeProfile(
        detects_frontiers=False,
        generates_tasks=False,
        allocates_tasks=True,
        plans_paths=False,
        controls_motion=False,
        candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
    )

    assert profile.owns_target_generation is False
    assert profile.uses_legacy_frontier_service is False
    assert profile.supports_periodic_replan is True
