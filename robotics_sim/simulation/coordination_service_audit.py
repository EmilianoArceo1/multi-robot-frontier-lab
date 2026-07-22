"""Instruments CoordinationServices with call-count tracking and optional
contract enforcement against a plugin's declared CandidateInputMode.

Every wrapped service method calls straight through to the real
implementation and returns its value completely unmodified -- this module
never changes what a plugin receives, only records that it asked. In normal
(non-strict) mode, a detected contract violation is logged as a warning; in
strict mode (SimulationConfig.coordination_strict_contracts / tests and
experiments) it raises CoordinationContractError instead.

Enforced rules (the ones from the refactor brief that are actually
mechanically checkable from service call counts + declared metadata):
  - PLUGIN_INTERNAL must not call any host frontier service
    (frontier_provider/team_frontier_provider/frontier_information_service).
  - HOST_CANDIDATES must not declare PluginCapability.FRONTIER_DETECTION,
    and must not call frontier_information_service (that is
    HOST_FRONTIER_CLUSTERS' service).
  - HOST_FRONTIER_CLUSTERS calling frontier_information_service is
    explicitly allowed (not checked/warned).

HYBRID exporting which source it actually used, and LEGACY_INTEGRATED
identifying itself as legacy, are debug/diagnostics-content contracts (see
CandidateGenerationResult.source_name/diagnostics) -- this auditor only
watches service call counts, so it does not and cannot enforce those two;
that is a known, documented limitation, not an oversight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from robotics_interfaces.plugins import CandidateInputMode, PluginCapability, PluginMetadata
from robotics_interfaces.services import CoordinationServices

_LOGGER = logging.getLogger(__name__)

_HOST_FRONTIER_SERVICE_NAMES = (
    "frontier_provider",
    "team_frontier_provider",
    "frontier_information_service",
)

_INSTRUMENTED_SERVICE_NAMES = (
    "frontier_provider",
    "team_frontier_provider",
    "path_planning_service",
    "collision_checking_service",
    "map_query_service",
    "metrics_service",
    "frontier_information_service",
    "region_decomposition_service",
    "coverage_path_service",
)


class CoordinationContractError(RuntimeError):
    """Raised in strict mode when observed service usage (or declared
    capabilities) contradicts a plugin's declared CandidateInputMode."""


@dataclass
class ServiceCallAudit:
    """Mutable per-service call-count record."""

    service_name: str
    call_counts: dict[str, int] = field(default_factory=dict)

    def record(self, method_name: str) -> None:
        self.call_counts[method_name] = self.call_counts.get(method_name, 0) + 1

    @property
    def total_calls(self) -> int:
        return sum(self.call_counts.values())


@dataclass(frozen=True)
class ServiceAuditReport:
    """Read-only summary handed to reasoning panels/logs."""

    plugin_name: str
    candidate_input_mode: CandidateInputMode | None
    call_counts_by_service: Mapping[str, Mapping[str, int]]
    warnings: tuple[str, ...] = ()


class _AuditingProxy:
    """Call-counting proxy for one service instance.

    __getattr__ wraps every callable attribute so a call is counted before
    delegating, unmodified, to the real implementation. Non-callable
    attributes pass through untouched.
    """

    def __init__(self, wrapped: Any, audit: ServiceCallAudit):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_audit", audit)

    def __getattr__(self, name: str) -> Any:
        target = getattr(object.__getattribute__(self, "_wrapped"), name)
        if not callable(target):
            return target

        audit = object.__getattribute__(self, "_audit")

        def _call(*args: Any, **kwargs: Any) -> Any:
            audit.record(name)
            return target(*args, **kwargs)

        return _call


@dataclass
class CoordinationServiceAuditor:
    """Wraps one plugin decision's CoordinationServices and checks its usage
    against metadata.candidate_input_mode once the decision is done.

    Usage:
        auditor = CoordinationServiceAuditor(metadata=plugin.metadata, strict=False)
        instrumented_services = auditor.instrument(request.services)
        ... call plugin.assign(replace(request, services=instrumented_services)) ...
        auditor.check_contract()  # warns or raises
        report = auditor.report()
    """

    metadata: PluginMetadata
    strict: bool = False
    _audits: dict[str, ServiceCallAudit] = field(default_factory=dict, repr=False)
    _warnings: list[str] = field(default_factory=list, repr=False)

    def instrument(self, services: CoordinationServices | None) -> CoordinationServices | None:
        if services is None:
            return None

        wrapped_fields: dict[str, Any] = {}
        for service_name in _INSTRUMENTED_SERVICE_NAMES:
            instance = getattr(services, service_name, None)
            if instance is None:
                continue
            audit = self._audits.setdefault(service_name, ServiceCallAudit(service_name=service_name))
            wrapped_fields[service_name] = _AuditingProxy(instance, audit)

        if not wrapped_fields:
            return services
        return replace(services, **wrapped_fields)

    def check_contract(self) -> None:
        """Evaluate recorded call counts + declared capabilities against
        candidate_input_mode. Logs a warning (default) or raises
        CoordinationContractError (strict) per violation found."""

        mode = self.metadata.candidate_input_mode
        host_frontier_calls = {
            name: self._audits[name].total_calls
            for name in _HOST_FRONTIER_SERVICE_NAMES
            if name in self._audits and self._audits[name].total_calls > 0
        }

        if mode is CandidateInputMode.PLUGIN_INTERNAL and host_frontier_calls:
            self._violate(
                f"{self.metadata.name!r} declares CandidateInputMode.PLUGIN_INTERNAL but "
                f"called host frontier service(s): {sorted(host_frontier_calls)}"
            )

        if mode is CandidateInputMode.HOST_CANDIDATES:
            if PluginCapability.FRONTIER_DETECTION in self.metadata.capabilities:
                self._violate(
                    f"{self.metadata.name!r} declares CandidateInputMode.HOST_CANDIDATES but "
                    "also declares PluginCapability.FRONTIER_DETECTION"
                )
            if host_frontier_calls.get("frontier_information_service", 0):
                self._violate(
                    f"{self.metadata.name!r} declares CandidateInputMode.HOST_CANDIDATES but "
                    "called frontier_information_service (that is HOST_FRONTIER_CLUSTERS' service)"
                )

    def _violate(self, message: str) -> None:
        if self.strict:
            raise CoordinationContractError(message)
        self._warnings.append(message)
        _LOGGER.warning("coordination contract warning: %s", message)

    def report(self) -> ServiceAuditReport:
        return ServiceAuditReport(
            plugin_name=self.metadata.name,
            candidate_input_mode=self.metadata.candidate_input_mode,
            call_counts_by_service={
                name: dict(audit.call_counts) for name, audit in self._audits.items()
            },
            warnings=tuple(self._warnings),
        )
