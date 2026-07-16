"""
Diagnostic (not a behavior fix) for the "robot avoids/fails at narrow
passages" symptom, run after route-repair replanning was confirmed to
correctly preserve the active path_goal (see
test_route_affected_replan_preserves_goal.py). With that state-machine bug
fixed, manual Office.sim still shows repeated failures repairing toward the
SAME preserved path_goal:

    Planner failed in exploration mode: no path found
    [ROUTE fail] attempted=(2.25,3.25) reason=no_path

    path found with A*
    goal adjusted to nearest traversable cell
    rejected: final waypoint does not reach path goal

This points at the planning grid / clearance / rasterization itself, not
the navigation state machine. This module adds a small, pure, isolated
diagnostic to tell apart four possible causes of a narrow-passage failure:

    a) valid safety clearance -- the corridor is genuinely narrower than
       2 * effective_planning_clearance and SHOULD be blocked, at any
       resolution. Not a bug.
    b) coarse grid rasterization -- the corridor is geometrically wider
       than 2 * effective_planning_clearance (so a continuous-space
       point-robot planner could pass through) but grid quantization at
       the configured resolution still blocks it.
    c) candidate endpoint placement near obstacles -- a candidate/target
       sits close enough to a wall that A* only reaches it by adjusting
       the goal to the nearest traversable cell, which
       candidate_reachable_on_planning_grid() / route endpoint validation
       already correctly reject as "not reachable" (fixed a few rounds
       ago) -- a distinct failure mode from (b), not a bug either.
    d) actual no-path due to mapped obstacles -- there genuinely is no
       route, independent of grid resolution.

Finding (see test 2/3 below): at grid_resolution=0.50 m/cell and
safety_radius=0.35 m (so effective_planning_clearance=0.35 m, nominal
required corridor width = 2*0.35 = 0.70 m), a corridor has to be
substantially WIDER than the nominal 0.70 m before A* can actually route
through it at 0.50 m resolution -- empirically, a 1.45 m wide corridor
(more than double the nominal requirement) was still blocked at 0.50 m/cell,
while the SAME corridor was passable at 0.25 m/cell. This is because
obstacle rasterization (OccupancyGrid.add_rectangular_obstacles() /
set_obstacle_rect_world()) marks a WHOLE grid cell occupied if the
(clearance-inflated) obstacle rectangle overlaps that cell AT ALL, not
based on whether the cell's center falls inside it -- so a coarse grid can
add up to an extra ~1 cell-width of effective inflation beyond the
intended clearance on each side of a passage, on top of the nominal
clearance itself. At grid_resolution=0.50 and safety_radius=0.35, passages
near the clearance limit may therefore be conservatively blocked by
rasterization alone, well before the true safety-clearance limit is reached.

No runtime behavior is changed in this round: this file only adds tests
that call existing, already-correct helpers
(engine.effective_planning_clearance(), engine.candidate_reachable_on_planning_grid(),
planner_registry.compute_planned_waypoints()) with different inputs to
characterize their behavior. Two small, local, test-only helper functions
(_required_corridor_width(), _rasterized_corridor_connectivity()) exist
only in this file -- they are diagnostic scaffolding, not production code,
and nothing outside this file references them.
"""
from __future__ import annotations

from robotics_sim.environment.occupancy_grid import FREE, OccupancyGrid
from robotics_sim.planning.planner_registry import compute_planned_waypoints
from robotics_sim.simulation.engine import (
    candidate_reachable_on_planning_grid,
    effective_planning_clearance,
)


# ---------------------------------------------------------------------------
# Local diagnostic helpers (test-only scaffolding, not production code).
# ---------------------------------------------------------------------------


def _required_corridor_width(clearance: float) -> float:
    """Nominal continuous-space corridor width a point-robot planner needs:
    each wall inflates inward by clearance, so both walls together consume
    2*clearance before any free space remains."""
    return 2.0 * float(clearance)


def _corridor_walls(gap_width: float) -> list[tuple[float, float, float, float]]:
    """Two horizontal walls running along x in [0, 10], leaving a
    gap_width-wide free corridor centered on y=0 (before inflation)."""
    half_gap = gap_width / 2.0
    return [
        (0.0, -5.0, 10.0, 5.0 - half_gap),
        (0.0, half_gap, 10.0, 5.0 - half_gap),
    ]


def _rasterized_corridor_connectivity(
    gap_width: float, resolution: float, clearance: float
) -> tuple[bool, str]:
    """Whether A* finds a route start->goal through a synthetic corridor of
    the given (pre-inflation, continuous) gap_width, at the given grid
    resolution and clearance. Returns (success, reason)."""
    bounds = (0.0, 10.0, -5.0, 5.0)
    start_xy = (0.5, 0.0)
    goal_xy = (9.5, 0.0)
    success, reason, waypoints = compute_planned_waypoints(
        planner_type="A*",
        start_xy=start_xy,
        goal_xy=goal_xy,
        obstacles=_corridor_walls(gap_width),
        bounds=bounds,
        resolution=resolution,
        robot_radius=clearance,
        unknown_is_traversable=True,
    )
    return bool(success and waypoints), reason


ROBOT_RADIUS = 0.20
SAFETY_RADIUS = 0.35


# ---------------------------------------------------------------------------
# 1. Protect the already-established contract: effective clearance is the
#    TOTAL center clearance, not robot_radius + safety_radius.
# ---------------------------------------------------------------------------


def test_effective_clearance_is_total_center_clearance_not_extra_margin():
    clearance = effective_planning_clearance(ROBOT_RADIUS, SAFETY_RADIUS)
    assert clearance == SAFETY_RADIUS
    assert clearance != ROBOT_RADIUS + SAFETY_RADIUS


# ---------------------------------------------------------------------------
# 2/3. Category (b): a corridor geometrically wider than the nominal
# required width can still be blocked at a coarse grid resolution, and the
# SAME corridor can regain connectivity at a finer resolution.
# ---------------------------------------------------------------------------


def test_corridor_wider_than_continuous_clearance_can_be_blocked_by_coarse_grid():
    clearance = effective_planning_clearance(ROBOT_RADIUS, SAFETY_RADIUS)
    required = _required_corridor_width(clearance)  # 0.70 m nominal
    # More than double the nominal requirement -- comfortably "wider than
    # 2*effective_clearance" in continuous space.
    gap_width = required + 0.75  # 1.45 m

    connected, reason = _rasterized_corridor_connectivity(gap_width, resolution=0.50, clearance=clearance)

    # This asserts the ACTUAL observed behavior (verified empirically, not
    # assumed): at 0.50 m/cell this geometrically-generous corridor is
    # still blocked. If a future change to rasterization/inflation makes
    # this pass, that is worth knowing -- update this assertion
    # deliberately, not by loosening it silently.
    assert connected is False, (
        f"expected coarse-grid rasterization to still block a {gap_width:.2f} m corridor "
        f"(required={required:.2f} m) at resolution=0.50; got connected=True, reason={reason!r}. "
        "If this now passes, rasterization no longer over-inflates as documented above -- "
        "update this test's numbers rather than just flipping the assertion."
    )


def test_same_corridor_with_finer_grid_preserves_connectivity_if_geometrically_valid():
    clearance = effective_planning_clearance(ROBOT_RADIUS, SAFETY_RADIUS)
    required = _required_corridor_width(clearance)
    gap_width = required + 0.75  # identical corridor to the test above

    coarse_connected, _ = _rasterized_corridor_connectivity(gap_width, resolution=0.50, clearance=clearance)
    fine_connected, fine_reason = _rasterized_corridor_connectivity(gap_width, resolution=0.25, clearance=clearance)

    assert fine_connected is True, (
        f"a corridor this much wider than the nominal requirement ({gap_width:.2f} m vs "
        f"required={required:.2f} m) should be plannable at the finer 0.25 m resolution; "
        f"got connected=False, reason={fine_reason!r}"
    )
    assert coarse_connected is False and fine_connected is True, (
        "the finer grid must preserve connectivity that the coarser grid lost for the exact "
        "same geometrically-valid corridor -- this is the rasterization effect this diagnostic exists to show"
    )


# ---------------------------------------------------------------------------
# 4. Category (a): a corridor genuinely below the safety clearance must be
#    blocked regardless of resolution -- this is correct, not a bug to fix.
# ---------------------------------------------------------------------------


def test_corridor_below_clearance_is_blocked_for_both_resolutions():
    clearance = effective_planning_clearance(ROBOT_RADIUS, SAFETY_RADIUS)
    required = _required_corridor_width(clearance)
    gap_width = required - 0.10  # genuinely narrower than the safety clearance allows

    for resolution in (0.50, 0.25):
        connected, reason = _rasterized_corridor_connectivity(gap_width, resolution=resolution, clearance=clearance)
        assert connected is False, (
            f"a corridor narrower than the required clearance ({gap_width:.2f} m < "
            f"{required:.2f} m) must be blocked at resolution={resolution}, regardless of grid "
            f"fineness -- got connected=True, reason={reason!r}"
        )


# ---------------------------------------------------------------------------
# 5. Category (c): a candidate near a wall that can only be reached via an
# adjusted goal cell is correctly rejected by endpoint validation -- a
# distinct failure mode from (b), already fixed and not touched here.
# ---------------------------------------------------------------------------


def test_endpoint_near_wall_is_not_considered_reachable_if_adjusted_goal_is_required():
    bounds = (-5.0, 5.0, -5.0, 5.0)
    resolution = 0.5
    robot_radius = ROBOT_RADIUS
    start_xy = (0.0, 0.0)
    candidate_xy = (2.25, 0.25)  # cell-center-aligned, see test_exploration_candidate_reachability.py

    grid = OccupancyGrid.from_bounds(
        x_min=bounds[0], x_max=bounds[1], y_min=bounds[2], y_max=bounds[3],
        resolution=resolution, initial_value=FREE, unknown_is_traversable=True,
    )
    # Occupy exactly the candidate's own cell, forcing compute_planned_waypoints()
    # to adjust the goal to the nearest traversable cell instead of reaching
    # the candidate itself -- category (c), not (b): this has nothing to do
    # with corridor width or grid resolution.
    grid.add_obstacle_points([candidate_xy], padding=0.1)

    reachable = candidate_reachable_on_planning_grid(
        grid, "A*", start_xy, candidate_xy,
        bounds=bounds, resolution=resolution, robot_radius=robot_radius, goal_tolerance=0.25,
    )

    assert reachable is False, (
        "a candidate that can only be reached via an adjusted nearest-traversable-cell goal "
        "must not be considered reachable, independent of any corridor-width/resolution effect"
    )
