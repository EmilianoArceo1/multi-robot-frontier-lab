"""Planning projection helpers for continuous dynamic hazards.

The path planners continue to consume the existing binary ``OccupancyGrid``.
This module projects a separate continuous hazard layer into that derived
grid without modifying the logical occupancy belief.

Two independent projections live here:
    - ``apply_hazard_to_planning_grid()``        ground-truth HazardField.
      Kept exactly as-is for its own legacy contract/tests -- no longer
      called by the runtime planner (see apply_hazard_belief_to_planning_
      grid() below), but still a valid, correct projection in its own right.
    - ``apply_hazard_belief_to_planning_grid()``  discovered-only Team
      HazardBelief. This is what the runtime planner actually uses: a cell
      is never blocked unless the team has actually observed it.
"""

from __future__ import annotations

from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.environment.hazard_belief import HazardBelief
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


def apply_hazard_belief_to_planning_grid(
    planning_grid: OccupancyGrid,
    hazard_belief: HazardBelief | None,
    *,
    block_threshold: float,
    inflate_radius: float = 0.0,
) -> OccupancyGrid:
    """Mark only OBSERVED, thresholded hazard-belief cells occupied.

    Unlike ``apply_hazard_to_planning_grid()`` (ground truth), a cell here is
    never blocked unless ``hazard_belief.observed`` is True for it -- no
    matter how hot the ground-truth field actually is at that cell. This is
    what the runtime planner uses: planning must never be omniscient.

    ``planning_grid`` is mutated (same contract as ``apply_hazard_to_
    planning_grid()``); ``hazard_belief`` is only read via its narrow
    ``blocked_cells()`` query -- never ``snapshot()`` (an O(H*W) full-grid
    copy this hot path, called once per robot per sensor update/planning
    request, does not need) and never mutated.
    """

    if hazard_belief is None:
        return planning_grid

    if hazard_belief.shape != planning_grid.data.shape:
        raise ValueError(
            "HazardBelief/planning grid shape mismatch: "
            f"belief={hazard_belief.shape}, planning={planning_grid.data.shape}."
        )
    if abs(hazard_belief.geometry.resolution - planning_grid.resolution) > 1e-9:
        raise ValueError(
            "HazardBelief/planning grid resolution mismatch: "
            f"belief={hazard_belief.geometry.resolution}, planning={planning_grid.resolution}."
        )

    rows, cols = hazard_belief.blocked_cells(float(block_threshold))
    if rows.size == 0:
        return planning_grid

    geometry = hazard_belief.geometry
    blocked_points = [
        geometry.grid_to_world(GridCell(row=int(row), col=int(col)))
        for row, col in zip(rows, cols)
    ]
    planning_grid.add_obstacle_points(
        blocked_points,
        padding=max(0.0, float(inflate_radius)),
    )
    return planning_grid
