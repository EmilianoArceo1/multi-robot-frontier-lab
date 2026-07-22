"""Tests for robotics_sim/simulation/coordination_service_audit.py."""

from __future__ import annotations

import pytest

from robotics_interfaces.plugins import CandidateInputMode, PluginCapability, PluginMetadata
from robotics_interfaces.services import CoordinationServices
from robotics_sim.simulation.coordination_service_audit import (
    CoordinationContractError,
    CoordinationServiceAuditor,
)


class _FakeFrontierProvider:
    def candidates_for_robot(self, robot, world, blocked_targets=()):
        return ("candidate-result",)


class _FakeFrontierInformationService:
    def get_frontier_clusters(self, robot_id=None):
        return ("cluster-result",)


class _FakeMetricsService:
    def record_event(self, event):
        return None


def _metadata(mode: CandidateInputMode | None, capabilities=(PluginCapability.COORDINATION,)) -> PluginMetadata:
    return PluginMetadata(
        name="test plugin",
        version="0.0.0",
        description="",
        capabilities=tuple(capabilities),
        candidate_input_mode=mode,
    )


def test_instrumented_service_returns_the_exact_same_result_unmodified():
    services = CoordinationServices(frontier_provider=_FakeFrontierProvider())
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.HOST_CANDIDATES))

    instrumented = auditor.instrument(services)
    result = instrumented.frontier_provider.candidates_for_robot(None, None)

    assert result == ("candidate-result",)


def test_instrumentation_records_call_counts_per_method():
    services = CoordinationServices(frontier_provider=_FakeFrontierProvider())
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.HOST_CANDIDATES))

    instrumented = auditor.instrument(services)
    instrumented.frontier_provider.candidates_for_robot(None, None)
    instrumented.frontier_provider.candidates_for_robot(None, None, blocked_targets=((1.0, 1.0),))

    report = auditor.report()
    assert report.call_counts_by_service["frontier_provider"]["candidates_for_robot"] == 2


def test_instrument_with_no_services_returns_none():
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.HOST_CANDIDATES))
    assert auditor.instrument(None) is None


def test_plugin_internal_calling_frontier_provider_warns_in_normal_mode():
    services = CoordinationServices(frontier_provider=_FakeFrontierProvider())
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.PLUGIN_INTERNAL), strict=False)
    instrumented = auditor.instrument(services)
    instrumented.frontier_provider.candidates_for_robot(None, None)

    auditor.check_contract()

    report = auditor.report()
    assert report.warnings
    assert "PLUGIN_INTERNAL" in report.warnings[0]


def test_plugin_internal_calling_frontier_provider_raises_in_strict_mode():
    services = CoordinationServices(frontier_provider=_FakeFrontierProvider())
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.PLUGIN_INTERNAL), strict=True)
    instrumented = auditor.instrument(services)
    instrumented.frontier_provider.candidates_for_robot(None, None)

    with pytest.raises(CoordinationContractError):
        auditor.check_contract()


def test_plugin_internal_not_calling_any_host_service_has_no_warnings():
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.PLUGIN_INTERNAL))
    auditor.instrument(CoordinationServices())

    auditor.check_contract()

    assert auditor.report().warnings == ()


def test_host_frontier_clusters_calling_frontier_information_service_is_allowed():
    services = CoordinationServices(frontier_information_service=_FakeFrontierInformationService())
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.HOST_FRONTIER_CLUSTERS))
    instrumented = auditor.instrument(services)
    instrumented.frontier_information_service.get_frontier_clusters()

    auditor.check_contract()

    assert auditor.report().warnings == ()


def test_host_candidates_calling_frontier_information_service_warns():
    services = CoordinationServices(frontier_information_service=_FakeFrontierInformationService())
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.HOST_CANDIDATES))
    instrumented = auditor.instrument(services)
    instrumented.frontier_information_service.get_frontier_clusters()

    auditor.check_contract()

    assert auditor.report().warnings


def test_host_candidates_declaring_frontier_detection_capability_warns_even_without_any_calls():
    auditor = CoordinationServiceAuditor(
        metadata=_metadata(
            CandidateInputMode.HOST_CANDIDATES,
            capabilities=(PluginCapability.COORDINATION, PluginCapability.FRONTIER_DETECTION),
        )
    )
    auditor.instrument(CoordinationServices())

    auditor.check_contract()

    assert auditor.report().warnings


def test_metrics_service_calls_are_tracked_without_affecting_the_contract():
    services = CoordinationServices(metrics_service=_FakeMetricsService())
    auditor = CoordinationServiceAuditor(metadata=_metadata(CandidateInputMode.HOST_CANDIDATES))
    instrumented = auditor.instrument(services)
    instrumented.metrics_service.record_event("anything")

    auditor.check_contract()
    report = auditor.report()

    assert report.call_counts_by_service["metrics_service"]["record_event"] == 1
    assert report.warnings == ()
