"""
Regression tests for the NOIC / coordinated frontier assignment deadlock.

Run from the project root:
    python -m pytest tests/test_noic_frontier_regressions.py -q

These tests are intentionally behavioral. They are designed to fail on the
current deadlocking implementation and pass after the coordinator is changed to
prefer progress over HOLD when usable frontier candidates still exist.
"""

from __future__ import annotations

from robotics_sim.planning import coordinated_frontier_planner as cfp
from robotics_sim.planning.exploration_planners import FrontierCandidate
from robotics_sim.simulation.coordination import RobotCoordinationState


def _state(x: float, y: float) -> RobotCoordinationState:
    return RobotCoordinationState(
        xy=(x, y),
        safety_radius=0.35,
        sensor_range=2.50,
        vision_model="Camera / FoV",
        theta=0.0,
    )


def _candidate(x: float, y: float, gain: float = 4.0) -> FrontierCandidate:
    return FrontierCandidate(
        target=(x, y),
        size=1,
        distance_from_robot=0.0,
        information_gain=float(gain),
        score=float(gain),
        reason=f"test candidate gain={gain}",
    )


def test_noic_assigns_reachable_candidate_even_when_score_is_negative(monkeypatch):
    """A reachable frontier with nonzero information gain must beat HOLD.

    Current failure mode:
    - DFS has a HOLD branch with penalty -2.
    - A candidate with score -9 is considered worse than HOLD.
    - The robot receives None and stays in HOLD forever.

    For exploration, this is wrong: a non-catastrophic negative score is still
    progress. HOLD should be reserved for genuinely impossible candidates, not
    merely suboptimal ones.
    """

    monkeypatch.setattr(
        cfp,
        "_detect_global_frontier_viewpoints",
        lambda **kwargs: [_candidate(10.0, 0.0, gain=1.0)],
    )

    result = cfp.assign_frontier_viewpoints(
        robot_states=[_state(0.0, 0.0)],
        existing_targets=[None],
        robots_to_assign=[0],
        invalidated_targets_by_robot=[[]],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-20.0, 20.0, -20.0, 20.0),
        resolution=0.5,
        final_goal_xy=(8.0, 6.0),
        ipp_distance_penalty=1.0,  # makes score = 1 - 10 = -9, still usable
        target_exclusion_radius=1.0,
        dynamic_obstacle_margin=0.25,
        route_points_by_robot=[[]],
        explored_points_by_robot=[[(0.0, 0.0)]],
    )

    assert result.targets[0] == (10.0, 0.0)
    assert result.assignments[0] is not None


def test_invalidated_frontier_does_not_block_neighboring_shifted_frontier(monkeypatch):
    """Invalidating a reached cell must not blacklist the whole frontier region.

    Current failure mode:
    - target_exclusion_radius is also used for invalidated targets.
    - With resolution=0.5 and exclusion radius=1.0, reaching one target blocks
      nearby candidate cells that represent the frontier moving forward.
    - The robot holds even though a neighboring informative candidate exists.
    """

    monkeypatch.setattr(
        cfp,
        "_detect_global_frontier_viewpoints",
        lambda **kwargs: [_candidate(1.5, 0.0, gain=8.0)],
    )

    result = cfp.assign_frontier_viewpoints(
        robot_states=[_state(0.0, 0.0)],
        existing_targets=[None],
        robots_to_assign=[0],
        invalidated_targets_by_robot=[[(1.0, 0.0)]],
        explored_points=[(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(8.0, 6.0),
        ipp_distance_penalty=0.2,
        target_exclusion_radius=1.0,
        dynamic_obstacle_margin=0.25,
        route_points_by_robot=[[]],
        explored_points_by_robot=[[(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]],
    )

    assert result.targets[0] == (1.5, 0.0)


def test_large_frontier_cluster_produces_multiple_candidate_viewpoints():
    """A large connected frontier should expose more than one target candidate.

    Current failure mode:
    - _detect_global_frontier_viewpoints returns exactly one centroid-like target
      per connected frontier cluster.
    - In multi-robot mode, one large connected frontier can starve two robots.
    """

    # A long explored strip. Cells on its top and bottom are frontier cells
    # because they border unknown space. This is one connected frontier cluster.
    explored_points = [(x * 0.5, 0.0) for x in range(-8, 9)]

    candidates = cfp._detect_global_frontier_viewpoints(
        explored_points=explored_points,
        mapped_obstacle_points=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert len(candidates) >= 3


def test_initial_fov_overlap_still_assigns_targets_to_multiple_robots(monkeypatch):
    """Initial FoV overlap should not produce a total HOLD on startup."""

    monkeypatch.setattr(
        cfp,
        "_detect_global_frontier_viewpoints",
        lambda **kwargs: [
            _candidate(4.0, 0.0, gain=6.0),
            _candidate(4.0, 1.5, gain=6.0),
            _candidate(4.0, -1.5, gain=6.0),
        ],
    )

    result = cfp.assign_frontier_viewpoints(
        robot_states=[_state(0.0, 0.0), _state(0.2, 0.0), _state(0.0, 0.2)],
        existing_targets=[None, None, None],
        robots_to_assign=[0, 1, 2],
        invalidated_targets_by_robot=[[], [], []],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        final_goal_xy=(8.0, 6.0),
        ipp_distance_penalty=0.0,
        target_exclusion_radius=0.8,
        dynamic_obstacle_margin=0.25,
        route_points_by_robot=[[], [], []],
        explored_points_by_robot=[[(4.0, 0.0)], [(4.0, 0.0)], [(4.0, 0.0)]],
    )

    assigned = [target is not None for target in result.targets]
    assert sum(assigned) >= 2


def test_nearby_robots_get_divergent_targets_when_candidates_are_separated(monkeypatch):
    """Two robots that start close should receive different frontiers when available."""

    monkeypatch.setattr(
        cfp,
        "_detect_global_frontier_viewpoints",
        lambda **kwargs: [_candidate(4.0, 0.0, gain=7.0), _candidate(-4.0, 0.0, gain=7.0)],
    )

    result = cfp.assign_frontier_viewpoints(
        robot_states=[_state(0.0, 0.0), _state(0.15, 0.0)],
        existing_targets=[None, None],
        robots_to_assign=[0, 1],
        invalidated_targets_by_robot=[[], []],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        final_goal_xy=(8.0, 6.0),
        ipp_distance_penalty=0.0,
        target_exclusion_radius=0.8,
        dynamic_obstacle_margin=0.25,
        route_points_by_robot=[[], []],
        explored_points_by_robot=[[], []],
    )

    assert result.targets[0] != result.targets[1]
    assert result.targets[0] is not None
    assert result.targets[1] is not None


def test_route_overlap_fallback_assigns_positive_gain_candidate(monkeypatch):
    """Severe overlap should be treated as a soft penalty and fall back to a useful candidate."""

    monkeypatch.setattr(
        cfp,
        "_detect_global_frontier_viewpoints",
        lambda **kwargs: [_candidate(3.0, 0.0, gain=5.0)],
    )

    result = cfp.assign_frontier_viewpoints(
        robot_states=[_state(0.0, 0.0)],
        existing_targets=[None],
        robots_to_assign=[0],
        invalidated_targets_by_robot=[[]],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        final_goal_xy=(8.0, 6.0),
        ipp_distance_penalty=0.0,
        target_exclusion_radius=0.8,
        dynamic_obstacle_margin=0.25,
        route_points_by_robot=[[(-1.0, 0.0), (3.0, 0.0)]],
        explored_points_by_robot=[[(0.0, 0.0)]],
    )

    assert result.targets[0] == (3.0, 0.0)
    assert result.assignments[0] is not None


def test_fov_overlap_penalizes_but_does_not_block_assignment(monkeypatch):
    """FoV overlap should penalize duplication, not veto the whole assignment."""

    monkeypatch.setattr(
        cfp,
        "_detect_global_frontier_viewpoints",
        lambda **kwargs: [_candidate(2.5, 0.0, gain=4.0)],
    )

    result = cfp.assign_frontier_viewpoints(
        robot_states=[_state(0.0, 0.0), _state(0.1, 0.0)],
        existing_targets=[None, None],
        robots_to_assign=[0],
        invalidated_targets_by_robot=[[], []],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        final_goal_xy=(8.0, 6.0),
        ipp_distance_penalty=0.0,
        target_exclusion_radius=0.8,
        dynamic_obstacle_margin=0.25,
        route_points_by_robot=[[], []],
        explored_points_by_robot=[[(2.5, 0.0)], [(2.5, 0.0)]],
    )

    assert result.targets[0] == (2.5, 0.0)


def test_safety_disk_crossing_rejects_only_the_bad_candidate(monkeypatch):
    """A candidate that crosses a teammate safety disk should be rejected, not all options."""

    monkeypatch.setattr(
        cfp,
        "_detect_global_frontier_viewpoints",
        lambda **kwargs: [_candidate(0.6, 0.0, gain=4.0), _candidate(3.0, 0.0, gain=4.0)],
    )

    result = cfp.assign_frontier_viewpoints(
        robot_states=[_state(0.0, 0.0), _state(0.2, 0.0)],
        existing_targets=[None, None],
        robots_to_assign=[0],
        invalidated_targets_by_robot=[[], []],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        final_goal_xy=(8.0, 6.0),
        ipp_distance_penalty=0.0,
        target_exclusion_radius=0.8,
        dynamic_obstacle_margin=0.25,
        route_points_by_robot=[[], []],
        explored_points_by_robot=[[], []],
    )

    assert result.targets[0] == (3.0, 0.0)
