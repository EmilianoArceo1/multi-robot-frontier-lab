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
"""
from __future__ import annotations

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.occupancy_grid import FREE, OccupancyGrid
from robotics_sim.planning.exploration_planners import select_exploration_goal
from robotics_sim.simulation.engine import candidate_reachable_on_planning_grid


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
    candidate_xy = (2.0, 0.0)

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
        bounds=bounds, resolution=resolution, robot_radius=robot_radius,
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
        bounds=bounds, resolution=resolution, robot_radius=robot_radius,
    ), (
        "candidate must be rejected once dense mapped-obstacle-point padding "
        "blocks the only path, even though the same candidate is reachable "
        "on the loose grid"
    )
