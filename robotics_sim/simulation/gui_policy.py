"""Pure GUI enablement policy derived from a plugin's runtime profile.

This module has no Qt dependency on purpose: robotics_sim/app/*.py (the Qt
layer) calls compute_gui_control_policy() and only then touches widgets. That
keeps "which controls make sense together" testable without a QApplication
and reusable if the GUI is ever replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotics_interfaces.plugins import PluginRuntimeProfile


@dataclass(frozen=True)
class GuiControlPolicy:
    """Which algorithm-selection controls should be enabled, and why.

    exploration_planner_enabled=False does not mean exploration stops; it
    means the legacy exploration planner combo is no longer the authoritative
    source and should be shown as a fallback/service, not the active
    algorithm, when the selected plugin owns target generation.
    """

    exploration_planner_enabled: bool
    exploration_planner_reason: str
    path_planner_enabled: bool
    path_simplifier_enabled: bool
    control_enabled: bool


def compute_gui_control_policy(profile: PluginRuntimeProfile) -> GuiControlPolicy:
    """Derive GUI enablement from what the selected plugin actually owns."""

    exploration_planner_enabled = not profile.owns_target_generation
    exploration_planner_reason = (
        "primary exploration source"
        if exploration_planner_enabled
        else "provided by algorithm; legacy planner is a service/fallback only"
    )

    path_planner_enabled = not profile.owns_path_planning
    # The path simplifier post-processes the external path planner's output;
    # it has nothing to simplify once the plugin owns path planning itself.
    path_simplifier_enabled = path_planner_enabled

    control_enabled = not profile.owns_control

    return GuiControlPolicy(
        exploration_planner_enabled=exploration_planner_enabled,
        exploration_planner_reason=exploration_planner_reason,
        path_planner_enabled=path_planner_enabled,
        path_simplifier_enabled=path_simplifier_enabled,
        control_enabled=control_enabled,
    )
