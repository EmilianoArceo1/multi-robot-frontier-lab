"""Small, neutral candidate-generation contract.

This exists so a frontier/candidate source (host pipeline today, a real
CandidateGenerator plugin composition later) can be swapped without every
coordination plugin depending on a concrete implementation. It is
deliberately separate from CoordinationServices: a CandidateGenerator
answers "what candidates/clusters exist for these robots right now", while
CoordinationServices is the broader simulator-capability bundle (path
planning, collision checking, metrics, ...).

No algorithm is required to use this yet -- see
robotics_sim.simulation.coordination_services for the host adapter that
wraps the existing detect_connected_frontier_components/
detect_global_frontier_candidates pipeline, which RuntimeTeamFrontierProvider
and RuntimeFrontierInformationService may reuse where that is a safe,
behavior-preserving delegation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from robotics_interfaces.frontiers import FrontierCluster
from robotics_interfaces.observations import Point2D, RobotCoordinationState, WorldSnapshot
from robotics_interfaces.proposals import ExplorationCandidate


@dataclass(frozen=True)
class CandidateGenerationRequest:
    """Input to a CandidateGenerator: who needs candidates, over what world."""

    robot_states: tuple[RobotCoordinationState, ...]
    robot_ids: tuple[int, ...] = ()
    world: WorldSnapshot | None = None
    blocked_targets_by_robot: Mapping[int, tuple[Point2D, ...]] = field(default_factory=dict)
    existing_targets_by_robot: Mapping[int, Point2D | None] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    time_s: float = 0.0


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Output of a CandidateGenerator.

    frontier_clusters is optional: a generator that only produces a flat
    candidate pool (no connected-component clustering) leaves it None rather
    than fabricating clusters. source_name identifies which generator
    actually produced this result, for reasoning-panel provenance.
    """

    candidates_by_robot: Mapping[int, tuple[ExplorationCandidate, ...]] = field(default_factory=dict)
    frontier_clusters: tuple[FrontierCluster, ...] | None = None
    source_name: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    generated_at_s: float = 0.0


@runtime_checkable
class CandidateGenerator(Protocol):
    """Protocol for anything that can answer a CandidateGenerationRequest.

    This is intentionally minimal -- one method -- so both a host-side
    adapter and, later, a real composed generator (e.g. Nav2D wavefront
    feeding CQLite-style allocation) can implement it without depending on
    each other.
    """

    def generate(self, request: CandidateGenerationRequest) -> CandidateGenerationResult:
        ...
