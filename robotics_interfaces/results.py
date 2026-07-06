"""Small request/response dataclasses used by robotics_interfaces.services.

Kept separate from services.py so the protocols file stays focused on
behavior (what an algorithm can ask for) while this file stays focused on
plain data (what gets passed back and forth). Everything here is simulator-
independent plain data: no Qt, no robotics_sim, no engine internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from robotics_interfaces.observations import Point2D, WorldBounds


@dataclass(frozen=True)
class PathPlanningRequest:
    """Inputs needed to plan a path between two points.

    robot_id and metadata are optional context an algorithm may attach (e.g.
    for per-robot planner tuning or logging); the service itself only needs
    start/goal/robot_radius/bounds/resolution to produce a path.
    """

    start: Point2D
    goal: Point2D
    robot_radius: float
    bounds: WorldBounds
    resolution: float
    obstacle_points: tuple[Point2D, ...] = ()
    planner_type: str = "A*"
    robot_id: int = -1
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathPlanningResponse:
    """Result of a path planning request."""

    success: bool
    waypoints: tuple[Point2D, ...] = ()
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CollisionCheckResult:
    """Result of validating a path/corridor against the environment or team."""

    is_safe: bool
    reason_code: str = ""
    detail: str = ""
    conflict_robot_id: int | None = None
    min_distance: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MapQuerySnapshot:
    """Minimal read-only view of the shared map, for algorithms that only
    need to ask "what do we know" without depending on WorldSnapshot's
    coordination-specific fields."""

    explored_points: tuple[Point2D, ...] = ()
    mapped_obstacle_points: tuple[Point2D, ...] = ()
    bounds: WorldBounds | None = None
    resolution: float = 0.5


@dataclass(frozen=True)
class MetricsEvent:
    """One named metrics/telemetry event an algorithm wants recorded."""

    name: str
    data: Mapping[str, Any] = field(default_factory=dict)
