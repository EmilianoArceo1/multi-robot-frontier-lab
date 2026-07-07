from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

Point2D = tuple[float, float]
WorldBounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class WorldSnapshot:
    """Simulator-agnostic world observation for coordination plugins.

    This contract intentionally uses plain Python data only.  External
    algorithms should be able to reason about the explored map without
    importing robotics_sim, Qt, the canvas, or engine internals.
    """

    explored_points: tuple[Point2D, ...] = ()
    mapped_obstacle_points: tuple[Point2D, ...] = ()
    bounds: WorldBounds | None = None
    resolution: float = 0.5
    final_goal_xy: Point2D | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotCoordinationState:
    """Simulator-agnostic robot state used by coordination plugins."""

    robot_id: int
    xy: Point2D
    safety_radius: float
    sensor_range: float
    vision_model: str
    theta: float = 0.0
    current_target: Point2D | None = None
    is_active: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotTeamSnapshot:
    """Optional grouped team snapshot for algorithms that need team context."""

    robots: tuple[RobotCoordinationState, ...]
    time_s: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
