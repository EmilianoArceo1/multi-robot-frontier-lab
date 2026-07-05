"""
RobotObservation — snapshot built by the engine and passed to RobotAgent.step().

The engine constructs one per robot per simulation step, packaging everything
an agent or future algorithm host needs to make a decision. This removes the
need for decision code to reach directly into engine attributes, Qt widgets,
canvas objects, or concrete robot physics internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_sim.environment.belief_map import BeliefMap
    from robotics_sim.environment.occupancy_grid import OccupancyGrid


Point2D = tuple[float, float]


@dataclass(frozen=True)
class NeighborRobotSnapshot:
    """Plain snapshot of another robot visible to coordination logic.

    This is deliberately smaller than RobotAgent/Robot. It is safe to expose to
    internal algorithm contracts because it contains values, not mutable engine
    objects.
    """

    robot_id: int | None
    xy: Point2D
    heading: float = 0.0
    radius: float = 0.0
    active_target_xy: Point2D | None = None
    is_active: bool = True


@dataclass(frozen=True)
class CommunicationSnapshot:
    """Communication assumptions available to future decentralized algorithms."""

    mode: str = "perfect"
    connected_robot_ids: tuple[int, ...] = ()
    bandwidth_kbps: float | None = None
    range_m: float | None = None
    latency_ms: float | None = None
    packet_loss: float | None = None


@dataclass(frozen=True)
class RuntimeParameterSnapshot:
    """Runtime values that algorithms may read but should not mutate directly."""

    values: dict[str, Any] = field(default_factory=dict)

    def get_float(self, key: str, default: float) -> float:
        try:
            return float(self.values.get(key, default))
        except (TypeError, ValueError):
            return float(default)


@dataclass
class RobotObservation:
    """
    Immutable-by-convention snapshot of the world as seen by one robot.

    Built by SimulationControllerMixin.build_observation() each step and passed
    to RobotAgent.step() so the agent never reads engine state directly.

    Existing runtime fields
    -----------------------
    robot_xy / robot_heading / robot_radius:
        Current pose of this robot.
    belief_map:
        Shared or per-robot logical occupancy map. FREE/UNKNOWN/OCCUPIED.
    planning_grid:
        Inflated grid ready for A*/Dijkstra, or None if not yet built.
    mapped_obstacle_points:
        Dense boundary samples from the partial obstacle map.
    dynamic_obstacles:
        Other live robots modelled as (cx, cy, radius) disks.
    active_segment_blocked / predicted_collision:
        Safety flags computed by the engine before calling step().
    current_time:
        Simulation time in seconds, not wall-clock time.
    grid_resolution / goal_tolerance / sensor_range:
        Config parameters the agent needs for thresholds and planning.
    final_goal_xy:
        GUI mission goal G. Executable only in "Goal seeking" mode.
    vision_model / ipp_distance_penalty:
        Passed through to exploration planners.
    excluded_targets:
        Frontiers already claimed by other robots.
    route_points_by_robot:
        Active routes of all robots for coordination.

    Forward-compatible fields
    -------------------------
    These fields let future internal algorithm contracts reason about physical
    limits, sensing, communication, runtime parameters, and current plans
    without depending on Robot, RobotAgent, engine, Qt, or canvas.
    """

    # Robot pose
    robot_xy: Point2D
    robot_heading: float
    robot_radius: float

    # World knowledge
    belief_map: "BeliefMap"
    planning_grid: "OccupancyGrid | None"
    mapped_obstacle_points: list[Point2D]
    dynamic_obstacles: list[tuple[float, float, float]]  # (cx, cy, radius)

    # Safety flags computed by the engine before calling step()
    active_segment_blocked: bool
    predicted_collision: bool

    # Simulation parameters
    current_time: float
    grid_resolution: float
    goal_tolerance: float
    sensor_range: float

    # Goal context
    final_goal_xy: Point2D | None
    vision_model: str = ""
    ipp_distance_penalty: float = 0.0

    # Multi-robot coordination
    excluded_targets: list[Point2D] = field(default_factory=list)
    route_points_by_robot: list[list[Point2D]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Forward-compatible internal contract fields.
    # Existing engine code can ignore all of these safely.
    # ------------------------------------------------------------------

    robot_id: int | None = None
    robot_velocity_xy: Point2D | None = None
    robot_linear_speed: float | None = None
    robot_angular_speed: float | None = None

    max_speed: float | None = None
    max_acceleration: float | None = None
    max_angular_speed: float | None = None

    sensor_fov_rad: float | None = None
    sensor_model: str | None = None

    planner_mode: str = ""
    control_mode: str = ""
    active_target_xy: Point2D | None = None
    active_waypoint_xy: Point2D | None = None
    active_path_goal_xy: Point2D | None = None
    current_path: list[Point2D] = field(default_factory=list)

    neighbors: list[NeighborRobotSnapshot] = field(default_factory=list)
    communication: CommunicationSnapshot = field(default_factory=CommunicationSnapshot)

    runtime_parameters: RuntimeParameterSnapshot = field(
        default_factory=RuntimeParameterSnapshot
    )
    map_metadata: dict[str, Any] = field(default_factory=dict)
    metrics_snapshot: dict[str, float | int] = field(default_factory=dict)
    algorithm_context: dict[str, Any] = field(default_factory=dict)

    def runtime_value(self, key: str, default: Any = None) -> Any:
        """Read a runtime parameter without exposing engine internals."""
        return self.runtime_parameters.values.get(key, default)
