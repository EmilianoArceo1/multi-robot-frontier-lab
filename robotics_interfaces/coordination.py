from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Literal

from robotics_interfaces.observations import (
    Point2D,
    RobotCoordinationState,
    RobotTeamSnapshot,
    WorldBounds,
    WorldSnapshot,
)
from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.decision_context import CoordinationDecisionContext
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate
from robotics_interfaces.services import CoordinationServices

AssignmentStatus = Literal["ASSIGNED", "HOLD", "FAILED", "CLEAR"]


@dataclass(frozen=True)
class CoordinationRequest:
    """Input contract for coordination plugins.

    The request supports three levels of algorithm independence:
    1. use explicit proposals_by_robot,
    2. ask services.frontier_provider for candidates,
    3. use world + robot_states to generate candidates internally.

    shared remains only as a compatibility escape hatch for legacy adapters.
    """

    robot_states: tuple[RobotCoordinationState, ...]
    robots_to_assign: tuple[int, ...] = ()
    world: WorldSnapshot | None = None
    proposals_by_robot: Mapping[
        int,
        tuple[ExplorationCandidate | CandidateProposal, ...],
    ] = field(default_factory=dict)
    existing_targets_by_robot: Mapping[int, Point2D | None] = field(default_factory=dict)
    blocked_targets_by_robot: Mapping[int, tuple[Point2D, ...]] = field(default_factory=dict)
    route_points_by_robot: tuple[tuple[Point2D, ...], ...] = ()
    services: CoordinationServices | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    shared: Mapping[str, Any] = field(default_factory=dict)
    time_s: float = 0.0
    # Optional structured "why was this decision run" contract. None is the
    # backward-compatible default: existing construction sites (engine.py,
    # tests, experiments) do not need to supply it. See
    # robotics_sim.simulation.coordination_scheduler for the component that
    # will populate it.
    decision_context: CoordinationDecisionContext | None = None


@dataclass(frozen=True)
class CoordinationAssignment:
    robot_id: int
    status: AssignmentStatus
    target: Point2D | None
    reason: str = ""
    proposal: CandidateProposal | ExplorationCandidate | None = None


@dataclass(frozen=True)
class CoordinationResult:
    """Output contract from coordination plugins.

    targets/reasons are kept for current runtime adapters. assignments is the
    richer representation that new code should use. commands is optional and
    only meaningful once a runtime consumer is ready to read path/control_xy
    (see robotics_interfaces.plugins.PluginRuntimeProfile) — a plugin that only
    owns target generation/task allocation can leave path/control_xy unset.
    """

    targets: tuple[Point2D | None, ...] = ()
    reasons: tuple[str, ...] = ()
    strategy: str = ""
    assignments: tuple[CoordinationAssignment, ...] = ()
    debug: Mapping[str, Any] = field(default_factory=dict)
    commands: tuple[RobotCommand, ...] = ()
