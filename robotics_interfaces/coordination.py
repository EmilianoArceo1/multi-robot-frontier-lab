from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Literal

Point2D = tuple[float, float]
AssignmentStatus = Literal["ASSIGNED", "HOLD", "FAILED"]


@dataclass(frozen=True)
class RobotCoordinationState:
    """Simulator-independent robot state exposed to coordination plugins.

    This mirrors the minimum data the current simulator coordinator already
    needs, but it stays outside robotics_sim so future algorithms can be tested
    without importing engine, Qt, canvas, RobotAgent, or GUI objects.
    """

    robot_id: int
    xy: Point2D
    safety_radius: float
    sensor_range: float
    vision_model: str
    theta: float = 0.0
    current_target: Point2D | None = None
    is_active: bool = True


@dataclass(frozen=True)
class CandidateProposal:
    """Candidate target/viewpoint proposed before team-level coordination.

    A plugin may receive proposals from the simulator, generate its own, or use
    this type internally to expose debug information and unit-test behavior.
    """

    robot_id: int
    target: Point2D
    score: float
    information_gain: float = 0.0
    travel_cost: float = 0.0
    overlap_cost: float = 0.0
    safety_cost: float = 0.0
    heading_cost: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class CoordinationRequest:
    """Input contract sent by the simulator host to a coordination plugin.

    Keep this as a data snapshot. Do not pass mutable simulator objects such as
    RobotAgent, MainWindow, canvas items, engine instances, or Qt objects.
    
    The `shared` mapping is intentionally available as an escape hatch while
    the simulator is being refactored. It lets the current runtime pass legacy
    read-only objects during migration without expanding the core contract too
    aggressively.
    """

    robot_states: tuple[RobotCoordinationState, ...]
    robots_to_assign: tuple[int, ...] = ()
    proposals_by_robot: Mapping[int, tuple[CandidateProposal, ...]] = field(default_factory=dict)
    existing_targets_by_robot: Mapping[int, Point2D | None] = field(default_factory=dict)
    blocked_targets_by_robot: Mapping[int, tuple[Point2D, ...]] = field(default_factory=dict)
    route_points_by_robot: tuple[tuple[Point2D, ...], ...] = ()
    time_s: float = 0.0
    shared: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoordinationAssignment:
    """One plugin decision for one robot."""

    robot_id: int
    status: AssignmentStatus
    target: Point2D | None
    reason: str = ""
    proposal: CandidateProposal | None = None


@dataclass(frozen=True)
class CoordinationResult:
    """Output contract returned by coordination plugins.

    `targets` and `reasons` intentionally match the shape expected by the
    current simulator runtime, making migration from the legacy coordinator
    incremental.
    """

    targets: tuple[Point2D | None, ...]
    reasons: tuple[str, ...]
    strategy: str
    assignments: tuple[CoordinationAssignment, ...] = ()
    debug: Mapping[str, Any] = field(default_factory=dict)
