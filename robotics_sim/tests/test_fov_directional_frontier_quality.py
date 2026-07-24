"""Quality regressions for the FoV/hazard-aware frontier planner.

These tests stay below the engine/coordinator boundary.  They verify that the
single-robot scorer actually receives spatially diverse viewpoints, that its
candidate budget is not monopolized by component size, and that the published
score is applied without hidden directional overrides.
"""
from __future__ import annotations

import math

import pytest

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.planning.exploration_planners import (
    FoVAwareHazardFrontierPlanner,
    NearestFrontierPlanner,
    _InternalCandidate,
    _frontier_candidates,
    _sample_frontier_viewpoints,
    _score_candidate,
)


def _belief(*, bounds=(-5.0, 5.0, -5.0, 5.0), resolution=1.0) -> BeliefMap:
    return BeliefMap(bounds=bounds, resolution=resolution, robot_count=1)


def _fill_free(belief: BeliefMap) -> None:
    for row in range(belief.height):
        for col in range(belief.width):
            belief.mark_free_cell((row, col))


def test_connected_frontier_exposes_aligned_and_spatially_diverse_viewpoints():
    belief = _belief(bounds=(0.0, 11.0, 0.0, 11.0))
    # A known square surrounded by UNKNOWN produces one connected frontier
    # ring.  Its old centroid-nearest representative was tied across much of
    # the ring and therefore effectively arbitrary.
    for row in range(3, 8):
        for col in range(3, 8):
            belief.mark_free_cell((row, col))

    candidates = _frontier_candidates(
        belief=belief,
        reserved_targets=[],
        dynamic_obstacles=[],
        target_exclusion_radius=1.0,
        robot_radius=0.2,
        dynamic_obstacle_margin=0.25,
        clusters=FoVAwareHazardFrontierPlanner().cluster_frontiers(belief),
    )
    assert len(candidates) == 1

    robot_xy = belief.cell_to_world((5, 5))
    viewpoints = _sample_frontier_viewpoints(
        candidates[0],
        belief=belief,
        robot_xy=robot_xy,
        robot_heading=0.0,
        limit=5,
    )

    assert len(viewpoints) == 5
    assert len({item.target_cell for item in viewpoints}) == 5
    bearings = [math.atan2(item.target[1] - robot_xy[1], item.target[0] - robot_xy[0]) for item in viewpoints]
    assert min(abs(angle) for angle in bearings) <= 1e-9, (
        "an east-facing robot must be offered an east-aligned cell from the "
        "ring, not only one arbitrary centroid representative"
    )
    assert max(item.target[0] for item in viewpoints) - min(item.target[0] for item in viewpoints) >= 4.0


def test_fov_clustering_does_not_merge_diagonal_doorway_faces():
    belief = _belief()
    belief.mark_free_cell((4, 4))
    belief.mark_free_cell((5, 5))

    generic = NearestFrontierPlanner().cluster_frontiers(belief)
    fov = FoVAwareHazardFrontierPlanner().cluster_frontiers(belief)

    assert len(generic) == 1
    assert len(fov) == 2


def test_failed_viewpoint_does_not_exclude_its_entire_frontier_component():
    belief = _belief(bounds=(0.0, 11.0, 0.0, 11.0))
    for row in range(3, 8):
        for col in range(3, 8):
            belief.mark_free_cell((row, col))

    original = _frontier_candidates(
        belief=belief,
        reserved_targets=[],
        dynamic_obstacles=[],
        target_exclusion_radius=0.1,
        robot_radius=0.2,
        dynamic_obstacle_margin=0.25,
        clusters=FoVAwareHazardFrontierPlanner().cluster_frontiers(belief),
    )
    assert len(original) == 1

    alternatives = _frontier_candidates(
        belief=belief,
        reserved_targets=[original[0].target],
        dynamic_obstacles=[],
        target_exclusion_radius=0.1,
        robot_radius=0.2,
        dynamic_obstacle_margin=0.25,
        viewpoints_per_cluster=5,
        robot_xy=belief.cell_to_world((5, 5)),
        robot_heading=0.0,
        clusters=FoVAwareHazardFrontierPlanner().cluster_frontiers(belief),
    )

    assert alternatives, "one failed cell must not suppress a long, otherwise-valid frontier"
    assert all(item.target != original[0].target for item in alternatives)


def test_preselection_balances_size_distance_and_heading_rankings():
    planner = FoVAwareHazardFrontierPlanner()
    robot_xy = (0.0, 0.0)

    largest_behind = _InternalCandidate((0, 0), (-9.0, 0.0), 100, "frontier")
    nearest_side = _InternalCandidate((1, 1), (0.0, 1.0), 1, "frontier")
    aligned_ahead = _InternalCandidate((2, 2), (5.0, 0.0), 1, "frontier")
    fillers = [
        _InternalCandidate((10 + index, 10), (-8.0 + index, 4.0), 90 - index, "frontier")
        for index in range(6)
    ]

    chosen = planner._preselect(
        [largest_behind, nearest_side, aligned_ahead, *fillers],
        robot_xy,
        max_candidates=3,
        robot_heading=0.0,
    )

    cells = {item.target_cell for item in chosen}
    assert largest_behind.target_cell in cells
    assert nearest_side.target_cell in cells
    assert aligned_ahead.target_cell in cells


def test_scoring_uses_exact_hazard_gaussian_and_published_weights():
    belief = _belief()
    _fill_free(belief)
    robot_xy = belief.cell_to_world((5, 5))
    ahead_cell = (5, 7)
    hazard = belief.cell_to_world(ahead_cell)

    weights = {
        "information": 0.0,
        "frontier": 0.0,
        "alignment": 0.0,
        "hazard": 4.0,
        "length": 0.0,
        "repetition": 0.0,
        "turn": 0.0,
        "multi_robot": 0.0,
    }
    common = dict(
        belief=belief,
        robot_xy=robot_xy,
        robot_heading=0.0,
        reserved_targets=[],
        known_hazards=[hazard],
        hazard_sigma=4.0,
        sensor_range=2.5,
        fov_angle=math.radians(120.0),
        fov_stride_cells=2,
        use_occlusion=True,
        seen_saturation=5.0,
        max_frontier_size=1,
        target_exclusion_radius=1.0,
        weights=weights,
    )

    ahead = _score_candidate(
        candidate=_InternalCandidate(
            ahead_cell, belief.cell_to_world(ahead_cell), 1, "frontier", (ahead_cell,)
        ),
        planning_grid=belief.to_planning_grid(unknown_is_traversable=True),
        **common,
    )
    one_sigma_cell = (5, 3)
    one_sigma = _score_candidate(
        candidate=_InternalCandidate(
            one_sigma_cell,
            belief.cell_to_world(one_sigma_cell),
            1,
            "frontier",
            (one_sigma_cell,),
        ),
        planning_grid=belief.to_planning_grid(unknown_is_traversable=True),
        **common,
    )

    assert ahead is not None and one_sigma is not None
    assert ahead.hazard_gain == pytest.approx(1.0)
    assert ahead.score == pytest.approx(4.0)
    assert one_sigma.hazard_gain == pytest.approx(math.exp(-0.5))
    assert one_sigma.score == pytest.approx(4.0 * math.exp(-0.5))
    assert "hazard=1.000000" in ahead.reason
    assert "backtrack=" not in ahead.reason
