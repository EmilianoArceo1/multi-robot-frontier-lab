"""Planning projection helpers for continuous dynamic hazards.

The path planners continue to consume the existing binary ``OccupancyGrid``.
This module projects a separate continuous hazard field into that derived grid
without modifying the logical occupancy belief.
"""

from __future__ import annotations

from robotics_sim.environment.hazard_field import HazardField
from robotics_sim.environment.occupancy_grid import OccupancyGrid


def apply_hazard_to_planning_grid(
    planning_grid: OccupancyGrid,
    hazard_field: HazardField | None,
    *,
    block_threshold: float,
    inflate_radius: float = 0.0,
) -> OccupancyGrid:
    """Mark thresholded hazard cells occupied in a derived planning grid.

    ``planning_grid`` is intentionally mutated because it is already a fresh
    projection created for one planning request. ``BeliefMap.grid`` and the
    ``HazardField`` remain untouched.
    """

    if hazard_field is None:
        return planning_grid

    if hazard_field.shape != planning_grid.data.shape:
        raise ValueError(
            "Hazard/planning grid shape mismatch: "
            f"hazard={hazard_field.shape}, planning={planning_grid.data.shape}."
        )
    if abs(hazard_field.resolution - planning_grid.resolution) > 1e-9:
        raise ValueError(
            "Hazard/planning grid resolution mismatch: "
            f"hazard={hazard_field.resolution}, planning={planning_grid.resolution}."
        )

    blocked_points = hazard_field.blocked_world_points(float(block_threshold))
    if blocked_points:
        planning_grid.add_obstacle_points(
            blocked_points,
            padding=max(0.0, float(inflate_radius)),
        )
    return planning_grid
