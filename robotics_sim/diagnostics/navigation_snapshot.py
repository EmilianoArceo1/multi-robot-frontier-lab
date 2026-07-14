"""
Immutable navigation-debug contract.

Every dataclass here is frozen. Producers (planning/navigation/safety/
control/simulation code) build these once per relevant event from values
they already computed; nothing in this module recomputes anything. A field
is `Maybe.missing()` when the current implementation genuinely does not
carry that value out of the algorithm that computed it yet -- never a
guessed or recalculated stand-in.

Zero Qt/canvas/engine imports here -- enforced by
robotics_sim/tests/test_navigation_debug_contract.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

from robotics_sim.environment.grid_geometry import GridCell

Point2D = tuple[float, float]

T = TypeVar("T")


@dataclass(frozen=True)
class Maybe(Generic[T]):
    """A diagnostic value that may not exist in the current implementation.

    unavailable=True means the producing algorithm does not surface this
    value today (would need a contract change, not just reading an
    attribute) -- distinct from a field simply being legitimately empty
    (e.g. an empty pending_path because none was requested), which is
    represented by an ordinary, non-Maybe field instead.
    """

    value: T | None
    unavailable: bool = False

    @classmethod
    def of(cls, value: T) -> "Maybe[T]":
        return cls(value=value, unavailable=False)

    @classmethod
    def missing(cls) -> "Maybe[T]":
        return cls(value=None, unavailable=True)


@dataclass(frozen=True)
class Pose:
    """Frozen pose snapshot. A local equivalent of core.state.RobotState,
    which is a mutable dataclass and therefore unsuitable for embedding in
    an immutable contract -- this avoids aliasing a live, mutating object."""

    x: float
    y: float
    theta: float
    v: float


@dataclass(frozen=True)
class ClearanceTerms:
    """The exact boolean safety condition a checker evaluated, with the real
    terms it used -- not a generic distance-vs-radius formula invented for
    display. `distance` is Maybe.missing() when the checker that produced
    this result does not compute a scalar distance for a clear (non-blocked)
    outcome (see CollisionReport.distance's docstring)."""

    checker: str
    distance: Maybe[float]
    required_clearance: float
    blocked: bool
    blocking_point: Point2D | None
    reason: str


@dataclass(frozen=True)
class PathDebug:
    raw_path: Maybe[tuple[Point2D, ...]]
    simplified_path: Maybe[tuple[Point2D, ...]]
    active_path: tuple[Point2D, ...]
    pending_path: tuple[Point2D, ...]
    active_segment: tuple[Point2D, Point2D] | None
    active_waypoint_index: int | None
    planner_name: Maybe[str]
    simplifier_name: Maybe[str]


@dataclass(frozen=True)
class RouteValidationDebug:
    first_segment: Maybe[ClearanceTerms]
    endpoint_reaches_goal: bool | None


@dataclass(frozen=True)
class PredictedMotionDebug:
    trajectory: Maybe[tuple[Point2D, ...]]
    collision: Maybe[ClearanceTerms]


@dataclass(frozen=True)
class SafetyDebug:
    robot_radius: float
    safety_radius: float
    active_segment: Maybe[ClearanceTerms]


@dataclass(frozen=True)
class PlanningGridDebug:
    start_cell: Maybe[GridCell]
    start_cell_world: Maybe[Point2D]
    first_waypoint_cell: Maybe[GridCell]
    first_waypoint_world: Maybe[Point2D]
    unknown_is_traversable: Maybe[bool]
    start_cell_cleared: Maybe[bool]


@dataclass(frozen=True)
class ControllerDebug:
    v: float
    omega: float
    acceleration: float
    heading_error: Maybe[float]
    distance_to_goal: Maybe[float]
    # theta_target: the desired heading goal_metrics() computed toward the
    # active target -- distinct from robot_pose.theta (current heading).
    desired_heading: Maybe[float] = field(default_factory=Maybe.missing)
    # The controller's real return shape is [acceleration, angular_velocity]
    # -- there is no separate "desired speed" term in the active
    # implementation, so these are exactly that pair, pre/post clip_control().
    nominal_control: Maybe[tuple[float, float]] = field(default_factory=Maybe.missing)
    applied_control: Maybe[tuple[float, float]] = field(default_factory=Maybe.missing)


@dataclass(frozen=True)
class FrontierDebug:
    """Placeholder shape for a future round. No producer populates this in
    the MVP -- exploration_planners.py / frontier scoring are out of scope
    for this branch. Always Maybe.missing() fields today."""

    candidate_count: Maybe[int]
    selected_target: Maybe[Point2D]
    selected_score: Maybe[float]
    reason: Maybe[str]


class NavigationDebugEventKind(Enum):
    """Event vocabulary for the diagnostic ring buffer.

    This is new vocabulary -- no equivalent enum exists elsewhere in the
    repo (see plan notes). navigation_state/decision_kind on
    NavigationDebugSnapshot deliberately reuse the real existing vocabulary
    (RobotStatus / NavigationDecisionKind values as plain strings) instead;
    this enum only covers the route-acceptance/event-log concepts that have
    no first-class representation today.
    """

    TICK = "TICK"
    PLAN_ACCEPTED = "PLAN_ACCEPTED"
    PATH_SIMPLIFIED = "PATH_SIMPLIFIED"
    ROUTE_REJECTED = "ROUTE_REJECTED"
    SAFETY_REPLAN = "SAFETY_REPLAN"
    PREDICTED_COLLISION = "PREDICTED_COLLISION"
    HOLD = "HOLD"
    EXHAUSTED = "EXHAUSTED"


@dataclass(frozen=True)
class NavigationDebugSnapshot:
    """Everything the debug overlay needs to explain one navigation decision,
    assembled once by robotics_sim.simulation.engine from values already
    computed elsewhere this tick. The GUI must treat every field as final --
    it never re-derives, re-plans, or re-checks anything from this object."""

    snapshot_id: int
    simulation_time: float
    robot_id: str
    navigation_state: str
    decision_kind: str
    decision_reason: str
    robot_pose: Pose
    path: PathDebug
    route: RouteValidationDebug
    predicted_motion: PredictedMotionDebug
    safety: SafetyDebug
    planning_grid: PlanningGridDebug
    controller: ControllerDebug
    frontier: FrontierDebug
    # Low-level tracking FSM mode (RobotMode.value: ROTATE/TRACK/STOP/IDLE/
    # BLOCKED/FAILED) -- distinct from navigation_state (RobotStatus, the
    # higher-level idle/planning/moving/... concept) and decision_kind (the
    # per-tick NavigationDecision). Read directly off robot.mode, already
    # computed by TrackingStateMachine.update() before this tick's control
    # call -- not recomputed here.
    tracking_mode: str = ""
    # Whichever of TrackingStateMachine's two hysteresis thresholds
    # (rotate_to_track_threshold / track_to_rotate_threshold) governs the
    # current tracking_mode's ROTATE<->TRACK transition -- read directly off
    # robot.state_machine, not recomputed.
    rotate_threshold: Maybe[float] = field(default_factory=Maybe.missing)
    # One-line human explanation, built from the real fields on this same
    # snapshot (never a separate inference) -- see
    # engine._navigation_debug_explanation().
    explanation: str = ""
    # Count of physical obstacle samples currently stored by the runtime.
    # Kept in the immutable snapshot so historical views do not mix the
    # current map count with an older robot/control state.
    mapped_obstacle_points_count: int = 0
