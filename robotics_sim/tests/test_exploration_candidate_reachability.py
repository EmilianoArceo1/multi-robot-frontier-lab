"""
Regression tests for filtering unreachable exploration candidates.

Manual Office.sim logs showed the FoV-aware exploration planner repeatedly
selecting targets the real A* navigation stack immediately rejected:

    attempted_target=(3.25, 0.75)
    attempted_target=(1.25, -0.75)

Root cause: FoVAwareDirectionalFrontierPlanner._score_candidate() scores
candidates -- including checking they have *some* path -- using an internal
planning grid built from belief.to_planning_grid(unknown_is_traversable=True,
inflate_radius=robot_radius+safety_margin) only. The real single-robot
navigation grid (engine.build_planning_grid_for_robot()) additionally
overlays dense mapped-obstacle-point samples via
planning_grid.add_obstacle_points(obstacle_points, padding=robot_radius) --
samples the belief-only grid never sees. A candidate the exploration
scorer considers reachable can therefore still come back "no path found"
from the real planner.

Fix: an optional is_candidate_reachable(xy) -> bool callback, threaded
through PlannerServices -> TargetSelectionRequest -> select_exploration_goal
-> FoVAwareDirectionalFrontierPlanner.select_goal(). When supplied, scored
candidates it rejects are dropped before final selection. Left unset
(the default), behavior is unchanged. In the runtime path,
engine.ensure_planner_services() refreshes this callback every tick via
engine.make_exploration_reachability_check(), built from the exact same
planning grid (belief + dense mapped-obstacle-point padding) real
single-robot A* uses -- see engine.candidate_reachable_on_planning_grid(),
a module-level, Qt-free function usable without a full engine instance.

These tests exercise exploration_planners.py and the standalone
engine.candidate_reachable_on_planning_grid() helper directly -- no Qt, no
canvas, no full engine/GUI instantiation.

Part A -- reachability/endpoint-validation consistency
--------------------------------------------------------
A later Office.sim run (after the fix above landed) still showed repeated
route endpoint mismatches:

    [PREFETCH] requested target=(0.25, -3.75)
    [PREFETCH] rejected: final waypoint does not reach target;
    path found with A*; goal adjusted to nearest traversable cell

    Planner failed in exploration mode:
    path found with A*; goal adjusted to nearest traversable cell;
    rejected: final waypoint does not reach path goal

Root cause: candidate_reachable_on_planning_grid() only checked
`success and waypoints` -- it never checked whether the route's final
waypoint actually reached the requested candidate. compute_planned_waypoints()
can return success=True after silently relocating an occupied goal cell to
the "nearest traversable cell" (see planner_registry._nearest_traversable_cell()),
which is exactly the kind of route apply_route_result()/on_prefetch_route_ready()
correctly reject via NavigationSupervisor.validate_route_endpoint() (added a
few rounds ago for the same reason). The two checks disagreed: reachability
filtering said "yes, reachable" for a candidate the endpoint validator would
reject moments later, once a real route was actually requested for it --
wasting a REQUEST_PLAN/PREFETCH cycle and, in narrow passages where the
adjusted cell can be well outside goal_tolerance, effectively bouncing off
without ever finding a genuinely reachable candidate on the other side.

Fix: candidate_reachable_on_planning_grid() now also calls
NavigationSupervisor.validate_route_endpoint(waypoints, candidate_xy,
goal_tolerance) -- the exact same check apply_route_result() and
on_prefetch_route_ready() already used -- so "reachable" means the same
thing everywhere: a route whose final waypoint actually reaches the
requested point within tolerance, not merely "A* found a path to
*something*". This required adding a goal_tolerance parameter, threaded
through from engine.make_exploration_reachability_check().

Separately, on_prefetch_route_ready()'s endpoint-mismatch rejection now
marks the rejected pending target as a failed exploration target (see
test_prefetch_unreachable_target_memory.py), so the same target is not
immediately re-proposed on the next prefetch/REQUEST_PLAN cycle.

Part B -- obstacle inflation / narrow-corridor clearance
-----------------------------------------------------------
Suspected (but NOT confirmed) root cause: obstacle inflation using
robot_radius + safety_radius (double-counting) instead of the intended
"safety_radius already IS the total clearance from the robot's center"
semantics.

Investigation finding: engine.safety_radius_for_robot() already uses
max(config.safety_radius, body_radius), not addition -- confirmed as the
INTENDED semantic by every other use site in the codebase (config.py's own
max(...) clamp, main_window.py's radius-consistency enforcement,
simulation_canvas.py's "Safety Radius r" circle rendering, and the GUI
slider label itself, "Safety Radius r (m)" -- a single total-clearance
value, not a margin layered on top of the body). engine.py always passes a
pre-built planning_grid into compute_planned_waypoints() for the live
single-robot flow, which bypasses planner_registry._build_planning_grid()'s
own (separate, unrelated) robot_radius + safety_margin formula entirely --
that additive path exists but is never reached by this flow, since
safety_margin is never passed as non-zero here.

Conclusion: NOT double-counted in the runtime path this round touches. No
behavior change was made to the clearance value itself. A named helper,
engine.effective_planning_clearance(robot_radius, safety_radius), was
added purely to replace the scattered max(...) expression with one
documented, testable definition -- safety_radius_for_robot() now calls it
instead of inlining max(...).
"""
from __future__ import annotations

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.occupancy_grid import FREE, OccupancyGrid
from robotics_sim.planning.exploration_planners import select_exploration_goal
from robotics_sim.planning.planner_registry import compute_planned_waypoints
from robotics_sim.simulation.engine import (
    candidate_reachable_on_planning_grid,
    effective_planning_clearance,
)


def _belief_with_two_frontier_regions() -> BeliefMap:
    """A belief map with two well-separated frontier clusters:

    - a large region directly ahead of the robot (should score highest
      under FoV-aware weighting: close, big, well-aligned with heading), and
    - a small, far, misaligned region (should score much lower).
    """
    belief = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=1.0, robot_count=1)

    # Large region ahead of the robot (heading = 0, i.e. +x direction).
    for x in range(1, 6):
        for y in range(-2, 3):
            cell = belief.world_to_cell((float(x), float(y)))
            if cell is not None:
                belief.mark_free_cell(cell)

    # Small, far, unaligned region (behind and to the side).
    small_cell = belief.world_to_cell((-8.0, 8.0))
    if small_cell is not None:
        belief.mark_free_cell(small_cell)

    belief.force_free_point((0.0, 0.0))
    return belief


def _select(belief: BeliefMap, *, is_candidate_reachable=None):
    return select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_count=1,
        robot_radius=0.2,
        sensor_range=6.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
        is_candidate_reachable=is_candidate_reachable,
    )


# ---------------------------------------------------------------------------
# 1. A lower-scored but reachable candidate must be preferred over a
#    higher-scored one the reachability check rejects.
# ---------------------------------------------------------------------------


def test_fov_planner_prefers_reachable_candidate_over_unreachable_high_score_candidate():
    belief = _belief_with_two_frontier_regions()

    baseline = _select(belief)
    assert baseline.success
    assert len(baseline.candidates) >= 2, "scenario must produce at least two scored candidates"

    top_choice = baseline.candidates[0].target
    runner_up = baseline.candidates[1].target
    assert top_choice != runner_up
    assert baseline.target == top_choice

    def is_candidate_reachable(xy) -> bool:
        # Reject only the naturally-highest-scoring candidate (by rounded
        # target match), mirroring a real navigation grid that rejects one
        # specific frontier while leaving others reachable.
        return (round(xy[0], 3), round(xy[1], 3)) != (round(top_choice[0], 3), round(top_choice[1], 3))

    filtered = _select(belief, is_candidate_reachable=is_candidate_reachable)

    assert filtered.success
    assert filtered.target != top_choice, (
        "the higher-scored but unreachable candidate must not be selected"
    )
    assert filtered.target == runner_up, (
        "the next-best reachable candidate must be selected instead"
    )


# ---------------------------------------------------------------------------
# 2. When every scored candidate is rejected, selection must fail cleanly
#    with a distinguishable reason -- never fall back to the best-scored
#    but unreachable target.
# ---------------------------------------------------------------------------


def test_fov_planner_reports_no_reachable_candidates_when_all_candidates_unreachable():
    belief = _belief_with_two_frontier_regions()

    baseline = _select(belief)
    assert baseline.success
    best_target = baseline.target

    result = _select(belief, is_candidate_reachable=lambda xy: False)

    assert not result.success
    assert result.target is None, (
        "no candidate may be selected when the reachability check rejects everything, "
        "even the best-scored one"
    )
    assert result.target != best_target
    assert "no reachable frontier candidates" in result.reason


# ---------------------------------------------------------------------------
# 3. The reachability helper must use the same dense-obstacle-point padding
#    the real navigation grid uses, not the exploration planner's own
#    looser internal grid.
# ---------------------------------------------------------------------------


def test_reachability_filter_uses_same_obstacle_padding_as_navigation_grid_if_feasible():
    bounds = (-5.0, 5.0, -5.0, 5.0)
    resolution = 0.5
    robot_radius = 0.3
    start_xy = (0.0, 0.0)
    # Cell-center-aligned for resolution=0.5/bounds starting at -5.0 (real
    # frontier candidates are always cell centers, via belief_map.grid_to_world());
    # an arbitrary non-aligned point like (2.0, 0.0) sits exactly on a cell
    # boundary and gets a route endpoint quantized to a different cell's
    # center, up to resolution*sqrt(2)/2 away -- a grid-quantization
    # artifact, not an endpoint-mismatch bug.
    candidate_xy = (2.25, 0.25)

    def _build_grid(obstacle_points):
        grid = OccupancyGrid.from_bounds(
            x_min=bounds[0], x_max=bounds[1], y_min=bounds[2], y_max=bounds[3],
            resolution=resolution, initial_value=FREE, unknown_is_traversable=True,
        )
        if obstacle_points:
            grid.add_obstacle_points(obstacle_points, padding=robot_radius)
        return grid

    loose_grid = _build_grid(None)
    assert candidate_reachable_on_planning_grid(
        loose_grid, "A*", start_xy, candidate_xy,
        bounds=bounds, resolution=resolution, robot_radius=robot_radius, goal_tolerance=0.25,
    ), "candidate must be reachable on a grid with no dense obstacle samples"

    # A dense wall of mapped-obstacle-point samples, one per grid cell
    # along the column between start and candidate -- exactly the kind of
    # sample build_planning_grid_for_robot() overlays on top of the belief
    # grid, which the exploration planner's own internal scoring grid
    # never sees.
    wall_points = [(1.0, -5.0 + resolution * i) for i in range(21)]
    strict_grid = _build_grid(wall_points)

    assert not candidate_reachable_on_planning_grid(
        strict_grid, "A*", start_xy, candidate_xy,
        bounds=bounds, resolution=resolution, robot_radius=robot_radius, goal_tolerance=0.25,
    ), (
        "candidate must be rejected once dense mapped-obstacle-point padding "
        "blocks the only path, even though the same candidate is reachable "
        "on the loose grid"
    )


# ---------------------------------------------------------------------------
# 4-5 (Part A). candidate_reachable_on_planning_grid() must agree with the
# same endpoint-reaches-goal rule apply_route_result()/on_prefetch_route_ready()
# already enforce -- "a path was found" is not the same as "the requested
# candidate was actually reached".
# ---------------------------------------------------------------------------


def test_reachability_rejects_candidate_when_planner_adjusts_goal_endpoint():
    bounds = (-5.0, 5.0, -5.0, 5.0)
    resolution = 0.5
    robot_radius = 0.2
    start_xy = (0.0, 0.0)
    # Cell-center-aligned for resolution=0.5/bounds starting at -5.0 (real
    # frontier candidates are always cell centers, via belief_map.grid_to_world());
    # an arbitrary non-aligned point like (2.0, 0.0) sits exactly on a cell
    # boundary and gets a route endpoint quantized to a different cell's
    # center, up to resolution*sqrt(2)/2 away -- a grid-quantization
    # artifact, not an endpoint-mismatch bug.
    candidate_xy = (2.25, 0.25)

    grid = OccupancyGrid.from_bounds(
        x_min=bounds[0], x_max=bounds[1], y_min=bounds[2], y_max=bounds[3],
        resolution=resolution, initial_value=FREE, unknown_is_traversable=True,
    )
    # Occupy exactly the candidate's own cell (and a small margin around
    # it), leaving the surrounding area free -- compute_planned_waypoints()
    # will find a path, but only by adjusting the goal to the nearest
    # traversable cell, not the candidate itself. Confirm that premise
    # directly before asserting on the reachability check built on top of it.
    grid.add_obstacle_points([candidate_xy], padding=0.1)
    success, reason, waypoints = compute_planned_waypoints(
        planner_type="A*", start_xy=start_xy, goal_xy=candidate_xy,
        planning_grid=grid.copy(), bounds=bounds, resolution=resolution,
        robot_radius=robot_radius, unknown_is_traversable=True, obstacle_points=[],
    )
    assert success and waypoints, "sanity check: the scenario must actually adjust-and-succeed, not just fail"
    assert "goal adjusted" in reason
    final_wp = waypoints[-1]
    assert abs(final_wp[0] - candidate_xy[0]) > 0.25 or abs(final_wp[1] - candidate_xy[1]) > 0.25, (
        "sanity check: the adjusted endpoint must actually miss goal_tolerance, or this test proves nothing"
    )

    reachable = candidate_reachable_on_planning_grid(
        grid, "A*", start_xy, candidate_xy,
        bounds=bounds, resolution=resolution, robot_radius=robot_radius, goal_tolerance=0.25,
    )
    assert reachable is False, (
        "a route that only reaches an ADJUSTED nearest-traversable-cell goal must not "
        "count as the candidate being reachable"
    )


def test_reachability_accepts_candidate_only_when_route_endpoint_reaches_requested_target():
    bounds = (-5.0, 5.0, -5.0, 5.0)
    resolution = 0.5
    robot_radius = 0.2
    start_xy = (0.0, 0.0)
    # Cell-center-aligned for resolution=0.5/bounds starting at -5.0 (real
    # frontier candidates are always cell centers, via belief_map.grid_to_world());
    # an arbitrary non-aligned point like (2.0, 0.0) sits exactly on a cell
    # boundary and gets a route endpoint quantized to a different cell's
    # center, up to resolution*sqrt(2)/2 away -- a grid-quantization
    # artifact, not an endpoint-mismatch bug.
    candidate_xy = (2.25, 0.25)

    grid = OccupancyGrid.from_bounds(
        x_min=bounds[0], x_max=bounds[1], y_min=bounds[2], y_max=bounds[3],
        resolution=resolution, initial_value=FREE, unknown_is_traversable=True,
    )  # no obstacles at all -- the planner reaches the candidate exactly

    reachable = candidate_reachable_on_planning_grid(
        grid, "A*", start_xy, candidate_xy,
        bounds=bounds, resolution=resolution, robot_radius=robot_radius, goal_tolerance=0.25,
    )
    assert reachable is True


# ---------------------------------------------------------------------------
# 6-8 (Part B). Effective planning clearance: document and lock in the
# max(robot_radius, safety_radius) semantic (config.safety_radius is the
# TOTAL clearance radius from the robot's center, already inclusive of the
# body -- not an extra margin added on top of it), and prove a synthetic
# corridor plans/blocks at the expected clearance threshold.
# ---------------------------------------------------------------------------


def test_planning_inflation_does_not_double_count_safety_radius():
    # config.py, main_window.py, and simulation_canvas.py all clamp
    # safety_radius to be at least body_radius via max(...) -- never
    # body_radius + safety_radius. The GUI slider itself is labeled
    # "Safety Radius r (m)": a single total-clearance value, not a margin
    # layered on top of the robot's own body. effective_planning_clearance()
    # encodes that same, already-intended semantic under one tested name.
    assert effective_planning_clearance(robot_radius=0.20, safety_radius=0.35) == 0.35
    # safety_radius must never be allowed to shrink the robot's effective
    # footprint below its own physical body radius.
    assert effective_planning_clearance(robot_radius=0.20, safety_radius=0.05) == 0.20
    # Explicitly NOT robot_radius + safety_radius (0.55) -- that would
    # double-count the body radius safety_radius already includes.
    assert effective_planning_clearance(robot_radius=0.20, safety_radius=0.35) != 0.20 + 0.35


def _corridor_walls(gap_width: float) -> list[tuple[float, float, float, float]]:
    """Two horizontal walls running along x in [0, 10], leaving a
    gap_width-wide free corridor centered on y=0."""
    half_gap = gap_width / 2.0
    return [
        (0.0, -5.0, 10.0, 5.0 - half_gap),
        (0.0, half_gap, 10.0, 5.0 - half_gap),
    ]


def test_corridor_above_required_clearance_is_plannable():
    clearance = effective_planning_clearance(robot_radius=0.20, safety_radius=0.35)
    bounds = (0.0, 10.0, -5.0, 5.0)
    resolution = 0.25
    start_xy = (0.5, 0.0)
    goal_xy = (9.5, 0.0)

    # Gap comfortably wider than the 2*clearance a robot needs to fit
    # through with both walls inflated inward.
    gap_width = 2.0 * clearance + 0.8
    success, _reason, waypoints = compute_planned_waypoints(
        planner_type="A*", start_xy=start_xy, goal_xy=goal_xy,
        obstacles=_corridor_walls(gap_width), bounds=bounds, resolution=resolution,
        robot_radius=clearance, unknown_is_traversable=True,
    )

    assert success and waypoints, "a corridor above the required clearance must be plannable"


def test_corridor_below_required_clearance_is_blocked():
    clearance = effective_planning_clearance(robot_radius=0.20, safety_radius=0.35)
    bounds = (0.0, 10.0, -5.0, 5.0)
    resolution = 0.25
    start_xy = (0.5, 0.0)
    goal_xy = (9.5, 0.0)

    # Gap narrower than 2*clearance: after both walls inflate inward, no
    # free cell remains anywhere in the corridor.
    gap_width = 2.0 * clearance - 0.2
    success, _reason, waypoints = compute_planned_waypoints(
        planner_type="A*", start_xy=start_xy, goal_xy=goal_xy,
        obstacles=_corridor_walls(gap_width), bounds=bounds, resolution=resolution,
        robot_radius=clearance, unknown_is_traversable=True,
    )

    assert not success or not waypoints, "a corridor below the required clearance must be rejected, not squeezed through"
