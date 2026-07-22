"""Tests for CoordinatorReasoningPanel's structured provenance rendering
(Phase 10 of the exploration-pipeline-architecture refactor): decision
trigger/scope, robots actually updated, candidate_input_mode, actual
candidate source, stage ownership, host services called, and contract
warnings -- none of it inferred from the coordinator's name or the
deprecated owns_target_generation field.
"""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from robotics_interfaces.decision_context import (
    CoordinationDecisionContext,
    CoordinationScope,
    CoordinationTrigger,
)
from robotics_interfaces.plugins import CandidateInputMode, PluginRuntimeProfile
from robotics_sim.app.coordinator_reasoning_panel import CoordinatorReasoningPanel
from robotics_sim.simulation.coordination_result_applier import ApplyReport
from robotics_sim.simulation.coordination_service_audit import ServiceAuditReport


_app = QApplication.instance() or QApplication([])


def _profile() -> PluginRuntimeProfile:
    return PluginRuntimeProfile(
        detects_frontiers=False,
        generates_tasks=True,
        allocates_tasks=True,
        plans_paths=False,
        controls_motion=False,
        candidate_input_mode=CandidateInputMode.HOST_FRONTIER_CLUSTERS,
    )


def _result():
    return SimpleNamespace(
        targets=((1.0, 1.0), (2.0, 2.0)),
        reasons=("assigned", "assigned"),
        assignments=(
            SimpleNamespace(
                robot_id=0,
                status="ASSIGNED",
                proposal=SimpleNamespace(source="host_frontier_candidate_pipeline", metadata={}),
            ),
        ),
        debug={"robots_to_assign": (0, 1)},
    )


def test_provenance_shows_not_available_when_no_decision_context_is_attached():
    panel = CoordinatorReasoningPanel()

    panel.update_coordination(planner="Nearest frontier", coordinator="test", result=_result(), time_s=1.0)

    assert "not available" in panel.provenance.text()


def test_provenance_renders_trigger_scope_and_requesting_ids():
    panel = CoordinatorReasoningPanel()

    context = CoordinationDecisionContext(
        trigger=CoordinationTrigger.TARGET_REACHED,
        scope=CoordinationScope.REQUESTED_ROBOTS,
        requesting_robot_ids=(1,),
        requesting_robot_id=1,
    )
    panel.update_coordination(
        planner="Nearest frontier",
        coordinator="test",
        result=_result(),
        time_s=1.0,
        decision_context=context,
    )

    text = panel.provenance.text()
    assert "target_reached" in text.lower()
    assert "requested_robots" in text.lower()
    assert "[1]" in text


def test_provenance_renders_apply_report_updated_preserved_cleared():
    panel = CoordinatorReasoningPanel()

    context = CoordinationDecisionContext(
        trigger=CoordinationTrigger.PERIODIC_TEAM_REPLAN,
        scope=CoordinationScope.FULL_TEAM,
        requesting_robot_ids=(0, 1),
    )
    apply_report = ApplyReport(
        updated_robot_ids=(0,),
        preserved_robot_ids=(1,),
        cleared_robot_ids=(2,),
        rejected_robot_ids=(99,),
    )
    panel.update_coordination(
        planner="Nearest frontier",
        coordinator="test",
        result=_result(),
        time_s=1.0,
        decision_context=context,
        apply_report=apply_report,
    )

    text = panel.provenance.text()
    assert "[0]" in text  # updated
    assert "[1]" in text  # preserved
    assert "[2]" in text  # cleared
    assert "99" in text  # rejected


def test_candidate_source_reads_profile_mode_and_proposal_source_not_the_name():
    panel = CoordinatorReasoningPanel()

    panel.update_coordination(
        planner="Nearest frontier",
        coordinator="Totally Unrelated Coordinator Name",
        result=_result(),
        time_s=1.0,
        runtime_profile=_profile(),
    )

    text = panel.candidate_source.text()
    assert "host_frontier_clusters" in text.lower()
    assert "host_frontier_candidate_pipeline" in text


def test_stage_ownership_reads_new_profile_fields_not_owns_target_generation():
    panel = CoordinatorReasoningPanel()

    panel.update_coordination(
        planner="Nearest frontier",
        coordinator="test",
        result=_result(),
        time_s=1.0,
        runtime_profile=_profile(),
    )

    text = panel.ownership.text()
    assert "frontier detection: False" in text
    assert "task generation: True" in text
    assert "task allocation: True" in text


def test_stage_ownership_falls_back_gracefully_for_a_bare_legacy_profile_double():
    """An older caller might still pass a SimpleNamespace with only the
    deprecated fields (as test_hungarian_coordination_is_auditable_per_robot
    does) -- this must not crash, and should show what it can."""
    panel = CoordinatorReasoningPanel()
    legacy_profile = SimpleNamespace(
        owns_target_generation=True,
        owns_task_allocation=True,
        owns_path_planning=False,
        owns_control=False,
    )

    panel.update_coordination(
        planner="Nearest frontier",
        coordinator="test",
        result=_result(),
        time_s=1.0,
        runtime_profile=legacy_profile,
    )

    text = panel.ownership.text()
    assert "task allocation: True" in text
    assert "path planning: False" in text


def test_host_services_and_warnings_render_not_available_without_service_audit():
    panel = CoordinatorReasoningPanel()

    panel.update_coordination(planner="Nearest frontier", coordinator="test", result=_result(), time_s=1.0)

    assert "Not instrumented" in panel.host_services.text()
    assert "No contract audit" in panel.contract_warnings.text()


def test_host_services_and_warnings_render_from_a_real_service_audit_report():
    panel = CoordinatorReasoningPanel()
    audit = ServiceAuditReport(
        plugin_name="test plugin",
        candidate_input_mode=CandidateInputMode.HOST_FRONTIER_CLUSTERS,
        call_counts_by_service={"frontier_information_service": {"get_frontier_clusters": 1}},
        warnings=("test plugin declares HOST_CANDIDATES but called frontier_information_service",),
    )

    panel.update_coordination(
        planner="Nearest frontier",
        coordinator="test",
        result=_result(),
        time_s=1.0,
        service_audit=audit,
    )

    assert "frontier_information_service" in panel.host_services.text()
    assert "declares HOST_CANDIDATES" in panel.contract_warnings.text()


def test_clear_resets_the_new_provenance_cards_too():
    panel = CoordinatorReasoningPanel()
    context = CoordinationDecisionContext(
        trigger=CoordinationTrigger.MISSING_TARGET, scope=CoordinationScope.REQUESTED_ROBOTS
    )
    panel.update_coordination(
        planner="Nearest frontier", coordinator="test", result=_result(), time_s=1.0, decision_context=context
    )

    panel.clear()

    assert panel.provenance.text() == "—"
    assert panel.candidate_source.text() == "—"
    assert panel.host_services.text() == "—"
    assert panel.contract_warnings.text() == "—"
