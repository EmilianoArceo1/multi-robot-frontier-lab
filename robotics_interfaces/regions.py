"""Region/task decomposition and coverage path contracts.

These support algorithms that split exploration into region-level tasks
(closer to a CVRP-style allocation) instead of single frontier targets, and
that plan a conceptual coverage route over those regions before handing off
to PATH_PLANNING/CONTROL. Nothing here is a trajectory: no time
parameterization, no dynamics, no B-splines/ESDF/trajectory optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from robotics_interfaces.observations import Point2D


@dataclass(frozen=True)
class RegionTask:
    """A region/cell block of unknown space that can be assigned to a robot."""

    region_id: str
    centroid: Point2D
    unknown_cell_count: int = 0
    assigned_robot_id: int | None = None
    cells: tuple[Point2D, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoveragePath:
    """A conceptual waypoint route over one or more regions for one robot."""

    robot_id: int
    waypoints: tuple[Point2D, ...]
    region_ids: tuple[str, ...] = ()
    estimated_cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
