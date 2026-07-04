"""
RobotObservation — snapshot built by the engine and passed to RobotAgent.step().

The engine constructs one per robot per simulation step, packaging everything
the agent needs to make a navigation decision. This removes the need for the
agent to reach directly into engine attributes or Qt widgets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_sim.environment.belief_map import BeliefMap
    from robotics_sim.environment.occupancy_grid import OccupancyGrid


@dataclass
class RobotObservation:
    """
    Immutable-by-convention snapshot of the world as seen by one robot.

    Built by SimulationControllerMixin.build_observation() each step.
    Passed to RobotAgent.step() so the agent never reads engine state directly.

    robot_xy / robot_heading / robot_radius:
        Current pose of this robot.

    belief_map:
        Shared (or per-robot) logical occupancy map.  FREE/UNKNOWN/OCCUPIED.

    planning_grid:
        Inflated grid ready for A*/Dijkstra, or None if not yet built.
        Build lazily via engine.build_planning_grid_for_robot() when needed.

    mapped_obstacle_points:
        Dense boundary samples from the partial obstacle map.

    dynamic_obstacles:
        Other live robots modelled as (cx, cy, radius) disks.

    active_segment_blocked:
        True when the engine's collision checker found that the segment
        robot -> active_waypoint is blocked by known obstacle points.

    predicted_collision:
        True when the nominal control would enter an obstacle region.

    current_time:
        Simulation time in seconds (not wall-clock time).

    grid_resolution / goal_tolerance / sensor_range:
        Config parameters the agent needs for distance thresholds.

    final_goal_xy:
        GUI mission goal G.  Executable only in "Goal seeking" mode.

    vision_model / ipp_distance_penalty:
        Passed through to exploration planners.

    excluded_targets:
        Frontiers already claimed by other robots (multi-robot use).

    route_points_by_robot:
        Active routes of all robots for coordination (multi-robot use).
    """

    # Robot pose
    robot_xy: tuple[float, float]
    robot_heading: float
    robot_radius: float

    # World knowledge
    belief_map: "BeliefMap"
    planning_grid: "OccupancyGrid | None"
    mapped_obstacle_points: list[tuple[float, float]]
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
    final_goal_xy: tuple[float, float] | None
    vision_model: str = ""
    ipp_distance_penalty: float = 0.0

    # Multi-robot coordination
    excluded_targets: list[tuple[float, float]] = field(default_factory=list)
    route_points_by_robot: list[list[tuple[float, float]]] = field(default_factory=list)
