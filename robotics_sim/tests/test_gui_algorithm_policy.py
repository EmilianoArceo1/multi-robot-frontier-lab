from __future__ import annotations

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from algorithms.global_noic_legacy.plugin import NOIC_COORDINATOR
from robotics_interfaces.plugins import (
    PluginCapability,
    PluginMetadata,
    build_runtime_profile,
)
from robotics_sim.simulation.coordination import runtime_profile_for_strategy
from robotics_sim.simulation.gui_policy import compute_gui_control_policy


def test_mmpf_gui_policy_disables_exploration_planner():
    profile = runtime_profile_for_strategy(MMPF_COORDINATOR)
    policy = compute_gui_control_policy(profile)

    assert policy.exploration_planner_enabled is False
    assert "algorithm" in policy.exploration_planner_reason


def test_mmpf_gui_policy_keeps_path_planner_enabled():
    profile = runtime_profile_for_strategy(MMPF_COORDINATOR)
    policy = compute_gui_control_policy(profile)

    assert policy.path_planner_enabled is True
    assert policy.path_simplifier_enabled is True
    assert policy.control_enabled is True


def test_noic_legacy_policy_keeps_legacy_controls_available():
    profile = runtime_profile_for_strategy(NOIC_COORDINATOR)
    policy = compute_gui_control_policy(profile)

    assert policy.exploration_planner_enabled is True
    assert policy.path_planner_enabled is True
    assert policy.control_enabled is True


def test_full_stack_gui_policy_disables_exploration_path_and_control():
    metadata = PluginMetadata(
        name="synthetic full stack",
        version="0.0.0",
        description="test-only full-stack plugin",
        capabilities=(PluginCapability.COORDINATION, PluginCapability.FULL_STACK),
    )
    profile = build_runtime_profile(metadata)
    policy = compute_gui_control_policy(profile)

    assert policy.exploration_planner_enabled is False
    assert policy.path_planner_enabled is False
    assert policy.path_simplifier_enabled is False
    assert policy.control_enabled is False
