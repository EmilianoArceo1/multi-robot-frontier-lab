"""Pure GUI enablement policy derived from a plugin's runtime profile.

This module has no Qt dependency on purpose: robotics_sim/app/*.py (the Qt
layer) calls compute_gui_control_policy() and only then touches widgets. That
keeps "which controls make sense together" testable without a QApplication
and reusable if the GUI is ever replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotics_interfaces.plugins import CandidateInputMode, PluginRuntimeProfile

# (exploration_planner_enabled, exploration_planner_reason) per
# CandidateInputMode. This is the real ownership signal now -- it does NOT
# read the deprecated owns_target_generation, so a plugin whose
# TARGET_GENERATION capability is eventually dropped (see
# PluginCapability's deprecation note) keeps exactly the same GUI behavior,
# because that behavior was never derived from TARGET_GENERATION in the
# first place.
_EXPLORATION_PLANNER_POLICY: dict[CandidateInputMode, tuple[bool, str]] = {
    CandidateInputMode.LEGACY_INTEGRATED: (
        True,
        "Legacy integrated pipeline: the exploration planner combo still "
        "selects the scoring formula this coordinator uses internally.",
    ),
    CandidateInputMode.PLUGIN_INTERNAL: (
        False,
        "Frontier/task generation is internal to this algorithm; the legacy "
        "exploration planner combo has no effect.",
    ),
    CandidateInputMode.HOST_FRONTIER_CLUSTERS: (
        False,
        "Host detects frontier clusters for this algorithm; the legacy "
        "exploration planner combo has no effect on cluster detection.",
    ),
    CandidateInputMode.HOST_CANDIDATES: (
        False,
        "Host provides candidates for this algorithm; the legacy "
        "exploration planner combo has no effect on candidate generation.",
    ),
    CandidateInputMode.HYBRID: (
        False,
        "Host provides candidates, but this algorithm may also generate its "
        "own fallback candidates; see Frontier Reasoning for the source "
        "actually used in a given decision.",
    ),
}

_LEGACY_PIPELINE_WARNING = (
    "This coordinator is a legacy integrated pipeline: frontier detection "
    "and task allocation are not actually separated inside it."
)


@dataclass(frozen=True)
class GuiControlPolicy:
    """Which algorithm-selection controls should be enabled, and why.

    exploration_planner_enabled=False does not mean exploration stops; it
    means the legacy exploration planner combo is not the authoritative
    source for this algorithm's candidates and should be shown as a
    fallback/inactive control, not the active algorithm.

    candidate_input_mode/legacy_pipeline_warning exist so the GUI/reasoning
    panels can show the real source instead of a control that merely looks
    disabled with no explanation -- legacy_pipeline_warning is set only for
    CandidateInputMode.LEGACY_INTEGRATED.
    """

    exploration_planner_enabled: bool
    exploration_planner_reason: str
    path_planner_enabled: bool
    path_simplifier_enabled: bool
    control_enabled: bool
    candidate_input_mode: CandidateInputMode
    legacy_pipeline_warning: str | None = None


def compute_gui_control_policy(profile: PluginRuntimeProfile) -> GuiControlPolicy:
    """Derive GUI enablement from what the selected plugin actually does.

    exploration_planner_enabled/reason come from profile.candidate_input_mode
    (not the deprecated owns_target_generation). path_planner_enabled/
    path_simplifier_enabled/control_enabled read plans_paths/controls_motion,
    the semantically-correct successors of owns_path_planning/owns_control
    (same values for every plugin migrated so far -- only the field read
    changed, not the computation).
    """

    exploration_planner_enabled, exploration_planner_reason = _EXPLORATION_PLANNER_POLICY[
        profile.candidate_input_mode
    ]

    path_planner_enabled = not profile.plans_paths
    # The path simplifier post-processes the external path planner's output;
    # it has nothing to simplify once the plugin owns path planning itself.
    path_simplifier_enabled = path_planner_enabled

    control_enabled = not profile.controls_motion

    legacy_pipeline_warning = (
        _LEGACY_PIPELINE_WARNING
        if profile.candidate_input_mode is CandidateInputMode.LEGACY_INTEGRATED
        else None
    )

    return GuiControlPolicy(
        exploration_planner_enabled=exploration_planner_enabled,
        exploration_planner_reason=exploration_planner_reason,
        path_planner_enabled=path_planner_enabled,
        path_simplifier_enabled=path_simplifier_enabled,
        control_enabled=control_enabled,
        candidate_input_mode=profile.candidate_input_mode,
        legacy_pipeline_warning=legacy_pipeline_warning,
    )
