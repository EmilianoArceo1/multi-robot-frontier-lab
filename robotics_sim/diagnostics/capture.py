"""
Mutable capture sinks -- engine-internal plumbing, never exposed to the
canvas.

These exist so an algorithm (compute_planned_waypoints, the collision
checker call sites in engine.py, the tracking controller) can optionally
hand back the intermediate values it already computes locally, without
changing its return type or behavior for the (default) case where no
capture is supplied. robotics_sim.simulation.engine freezes these into a
NavigationDebugSnapshot at the end of a tick; nothing outside engine.py
should ever hold a live capture object.
"""
from __future__ import annotations

from dataclasses import dataclass

from robotics_sim.diagnostics.navigation_snapshot import ClearanceTerms, Maybe
from robotics_sim.environment.collision_checker import CollisionReport
from robotics_sim.environment.grid_geometry import GridCell

Point2D = tuple[float, float]


@dataclass
class PlanDebugCapture:
    """Outparam compute_planned_waypoints() fills in place on its success
    path when provided, using local variables it already has (result.
    grid_path, simplified_grid_path, start_cell, ...) before they would
    otherwise be discarded. Omitted (None, the default) by every existing
    caller -- zero cost, zero behavior change."""

    planner_name: str | None = None
    simplifier_name: str | None = None
    raw_world_path: tuple[Point2D, ...] | None = None
    simplified_world_path: tuple[Point2D, ...] | None = None
    start_cell: GridCell | None = None
    start_cell_world: Point2D | None = None
    first_waypoint_cell: GridCell | None = None
    first_waypoint_world: Point2D | None = None
    unknown_is_traversable: bool | None = None
    start_cell_cleared: bool | None = None
    total_cost: float | None = None
    expanded_nodes: int | None = None
    goal_cell: GridCell | None = None
    grid_resolution: float | None = None


@dataclass
class NavigationDebugCapture:
    """One of these is created per simulation tick, only when
    navigation_debug_enabled. Threaded through build_observation(),
    predicted_motion_report(), and the tracking controller call so each can
    stash the CollisionReport/metrics it already computed; consumed exactly
    once by engine._finalize_navigation_debug_snapshot()."""

    active_segment: ClearanceTerms | None = None
    predicted_trajectory: tuple[Point2D, ...] | None = None
    predicted_collision: ClearanceTerms | None = None
    first_segment: ClearanceTerms | None = None
    endpoint_reaches_goal: bool | None = None
    heading_error: float | None = None
    distance_to_goal: float | None = None
    desired_heading: float | None = None
    # Controller output as actually returned: [acceleration, angular_velocity]
    # -- the real control law has no separate "desired speed" return value,
    # so these are the exact (a, omega) pair TrackingController.compute_
    # control() computes, before (nominal) and after (applied) clip_control().
    nominal_control: tuple[float, float] | None = None
    applied_control: tuple[float, float] | None = None
    plan: PlanDebugCapture | None = None


def clearance_terms_from_report(
    report: CollisionReport,
    *,
    checker: str,
    required_clearance: float,
) -> ClearanceTerms:
    """Build the immutable ClearanceTerms the snapshot carries from a
    CollisionReport the checker already produced -- never a second
    collision computation."""
    distance = Maybe.missing() if report.distance is None else Maybe.of(float(report.distance))
    return ClearanceTerms(
        checker=checker,
        distance=distance,
        required_clearance=float(required_clearance),
        blocked=bool(report.collision),
        blocking_point=report.point,
        reason=str(report.reason),
    )
