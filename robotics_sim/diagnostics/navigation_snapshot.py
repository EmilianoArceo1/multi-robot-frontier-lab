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
class SensorDebug:
    """Sensor state frozen for replay.

    The exact occlusion-aware FoV polygon is stored as compressed float32
    coordinates. Keeping it in the snapshot prevents the replayed robot pose
    from separating from a live/recomputed FoV.
    """

    vision_range: float = 0.0
    visible_polygon_count: int = 0
    visible_polygon_f32_zlib: bytes = b""


@dataclass(frozen=True)
class BeliefMapDebug:
    """Compact immutable belief/exploration frame for in-memory replay.

    Arrays are compressed bytes rather than live NumPy references, so a
    historical snapshot cannot be changed by later mapping updates. Multiple
    navigation snapshots may share the same BeliefMapDebug object when the
    map revision did not change -- grid_zlib/explored_packbits_zlib only
    change with revision, so this sharing is safe for them. visit_count_zlib/
    last_seen_zlib do NOT follow revision (BeliefMap.revision explicitly
    excludes visit-count/last-seen changes, see its own docstring), so the
    producer must compress those two fresh on every capture rather than
    reusing a revision-keyed cache, or a later tick's revisit would silently
    restore an earlier tick's visit_count/last_seen.
    """

    revision: int
    resolution: float
    bounds: tuple[float, float, float, float]
    grid_shape: tuple[int, int]
    grid_zlib: bytes
    explored_shape: tuple[int, int, int]
    explored_packbits_zlib: bytes
    # uint16 [height, width] / float32 [height, width] -- same shape as
    # grid_shape. Empty bytes for a frame captured before this field existed;
    # restore falls back to BeliefMap's own zero-state defaults for that case
    # (see restore_navigation_debug_snapshot()).
    visit_count_zlib: bytes = b""
    last_seen_zlib: bytes = b""


@dataclass(frozen=True)
class HazardSourceDebug:
    """One frozen FireSource, mirroring robotics_sim.environment.hazard_
    field.FireSource without importing it (this package stays Qt/engine-
    import-free; see the module docstring)."""

    fire_id: int
    position: Point2D
    intensity: float
    radius: float


@dataclass(frozen=True)
class HazardBeliefDebug:
    """Compact immutable Team HazardBelief frame for in-memory replay --
    the discovered-only counterpart to BeliefMapDebug/HazardDebug.

    Deliberately its own dataclass, not folded into HazardDebug: HazardDebug
    represents the omniscient ground-truth FireSource set, while this
    represents only what the team has actually observed (see
    environment.hazard_belief.HazardBelief's own module docstring) --
    keeping them separate means a restore path can never accidentally mix
    the two concepts by touching one field that means both things.

    Arrays are compressed bytes rather than live NumPy references (same
    zlib/packbits pattern as BeliefMapDebug), so a historical snapshot
    cannot be changed by a later observation. Multiple navigation snapshots
    may share the same HazardBeliefDebug object when the belief revision
    did not change (see engine._navigation_debug_hazard_belief_frame()'s
    cache). bounds/resolution are intentionally NOT duplicated here --
    HazardBelief always shares one GridGeometry with the same tick's
    BeliefMap (see hazard_service.RuntimeHazardService), so a consumer
    reads those from the snapshot's own belief_map field instead.

    values_zlib/observed_packbits_zlib/observed_by_robot_packbits_zlib are
    empty bytes for a frame captured before this field existed -- restore
    falls back to an empty HazardBelief for that case (see
    engine.restore_navigation_debug_snapshot()'s docstring for why an empty
    belief, never HazardField, is the only safe fallback).
    """

    shape: tuple[int, int]
    robot_count: int
    revision: int
    values_zlib: bytes = b""
    observed_packbits_zlib: bytes = b""
    observed_by_robot_packbits_zlib: bytes = b""


@dataclass(frozen=True)
class HazardDebug:
    """Frozen hazard-field state for restore.

    Occupancy (BeliefMap) and the continuous hazard layer are deliberately
    separate (see HazardField's module docstring) -- restoring hazards from
    this never touches BeliefMapDebug/belief_map.grid, and vice versa.
    `next_fire_id` is captured (not just the sources) so a fire added right
    after a restore gets the id it would have received at that point in
    time, rather than colliding with or skipping ahead of ids only the
    discarded future ever saw. `version` is HazardField's own change
    counter, carried through for parity with BeliefMapDebug.revision even
    though restore does not currently dedupe on it (fire counts are small
    enough that rebuilding every tick is cheap, unlike the belief grid).
    """

    version: int
    next_fire_id: int
    sources: tuple[HazardSourceDebug, ...]


@dataclass(frozen=True)
class AgentStateDebug:
    """Explicit RobotAgent state for restore -- never inferred from
    path.active_path[-1] (which is not always the same point: exploration
    hysteresis can keep active_path_goal_xy pointing at a target the
    simplified/active path does not literally end on).

    Recovery-policy memory (recent_safe_positions, recent_recovery_targets,
    failed_exploration_targets, exploration_exhausted_map_signature) and
    replan-throttle timestamps (last_safety_replan_time/_signature,
    last_replan_time, last_prefetch_time, narrow_passage_slowdown_until_
    time) are deliberately NOT included: restoring a throttle timestamp
    forward of the rewound simulation_time could suppress a legitimate
    replan that "just happened" in a future which no longer exists. `status`
    is not duplicated here either -- it is exactly
    NavigationDebugSnapshot.navigation_state (str(agent.status) at capture
    time); restore uses that field directly.
    """

    final_goal_xy: Point2D | None
    exploration_target_xy: Point2D | None
    active_path_goal_xy: Point2D | None
    active_path_mode: str | None
    route_generation: int
    route_affected_replan_count: int
    first_segment_blocked_count: int
    last_frontier_candidate_count: int
    prefetch_success_count: int
    prefetch_fail_count: int
    safety_replan_count: int
    target_switch_count: int


@dataclass(frozen=True)
class RuntimeMetricsDebug:
    """Engine-level cumulative counters, frozen alongside the rest of the
    snapshot so restore never leaves simulation_time behind while these
    keep reading a later run's totals (e.g. simulation_time=15s next to
    route_result_count from a run that had already reached t=42s)."""

    total_distance_traveled: float
    route_request_count: int
    route_result_count: int
    route_failure_count: int
    sensor_update_count: int
    mapping_update_count: int
    safety_replan_count: int
    exploration_replan_count: int
    planner_jobs_started: int
    planner_jobs_completed: int


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
    it never re-derives, re-plans, or re-checks anything from this object.

    Restore contract (single-robot only -- see engine.restore_navigation_
    debug_snapshot()): a snapshot is sufficient to roll the live simulation
    back to the moment it was captured using exactly these fields --
    simulation_time, robot_pose (+ .v for kinematic state), navigation_state
    (RobotAgent.status) and tracking_mode, path.active_path /
    path.active_waypoint_index (the route + current target), belief_map
    (occupancy grid + per-robot explored mask), hazard (FireSource set +
    next_fire_id, a layer fully separate from occupancy), agent_state
    (final/exploration/active-path goals, active_path_mode, route_
    generation, and the agent's own cumulative counters -- explicit, never
    inferred from path.active_path[-1]), and metrics (engine-level
    cumulative counters, so they never read ahead of simulation_time after a
    restore). mapped_obstacle_points_count is also load-bearing: the
    engine's live mapped_obstacle_points list is append-only, so truncating
    it to this count reproduces the exact set of boundary samples known at
    capture time without storing the points themselves here.

    Known gaps (not restorable from this contract, by design -- see the
    restore method's docstring): the executed-path trail (path_points) has
    no authoritative source to rebuild from and resets to the restored
    point. The visible explored-area *coverage* does NOT regress despite
    the bounded explored_area_polygons sweep-history list being cleared --
    belief_map.explored_by_robot (restored exactly) is the authoritative
    "explored" state, and the canvas is reseeded from it directly (see
    canvas.set_explored_area_seed()). RobotAgent's recovery-policy memory
    and replan-throttle timestamps are also excluded (see AgentStateDebug's
    docstring for why that is deliberate, not an oversight).
    """

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
    # Exact sensor footprint and compact map state for coherent history replay.
    sensor: SensorDebug = field(default_factory=SensorDebug)
    belief_map: Maybe[BeliefMapDebug] = field(default_factory=Maybe.missing)
    # Restore-only fields -- see the class docstring's "Restore contract"
    # paragraph and each dataclass's own docstring.
    hazard: Maybe[HazardDebug] = field(default_factory=Maybe.missing)
    # Team HazardBelief -- discovered-only, deliberately separate from
    # `hazard` (ground-truth FireSource set) above. Missing/unavailable for
    # a snapshot captured before this field existed; restore treats that
    # exactly like an empty belief (see HazardBeliefDebug's docstring).
    hazard_belief: Maybe[HazardBeliefDebug] = field(default_factory=Maybe.missing)
    agent_state: Maybe[AgentStateDebug] = field(default_factory=Maybe.missing)
    metrics: Maybe[RuntimeMetricsDebug] = field(default_factory=Maybe.missing)
