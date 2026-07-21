"""Contract tests for the pure connected-frontier-component detector.

detect_connected_frontier_components() (robotics_sim/planning/
coordinated_frontier_planner.py) is now the single source of truth for
connected frontier geometry: real world-space cells per component, a real
centroid, and one or more sampled viewpoints. Everything downstream
(the legacy detect_global_frontier_candidates()/_detect_global_frontier_viewpoints()
flat-candidate view, RuntimeFrontierInformationService) derives from it
instead of re-detecting or re-clustering.

These tests exercise the real detector directly (white-box, same pattern as
test_planning_map_characterization.py) -- no engine, no Qt, no MainWindow,
no global mutable state.
"""
from __future__ import annotations

from robotics_interfaces.frontiers import FrontierCluster
from robotics_sim.planning.coordinated_frontier_planner import (
    _cell_key,
    _inside_bounds,
    _occupied_cells_from_points,
    detect_connected_frontier_components,
    detect_global_frontier_candidates,
)

RESOLUTION = 0.5
BOUNDS = (-5.0, 15.0, -5.0, 15.0)
STRIP_BOUNDS = (-5.0, 5.0, -5.0, 5.0)


def _two_disconnected_blob_points() -> list[tuple[float, float]]:
    """Two isolated 2x2 explored blobs, far enough apart that they can never
    be 8-connected to each other."""
    return [
        (0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5),
        (10.0, 10.0), (10.5, 10.0), (10.0, 10.5), (10.5, 10.5),
    ]


def _large_strip_points() -> list[tuple[float, float]]:
    """One long explored strip along y=0 -- both edges border unknown
    space, so this is one large connected frontier component (the same
    fixture test_noic_frontier_regressions.py's
    test_large_frontier_cluster_produces_multiple_candidate_viewpoints
    uses)."""
    return [(x * 0.5, 0.0) for x in range(-8, 9)]


# ---------------------------------------------------------------------------
# Characterization: pins down the detector's actual output for the large-
# frontier fixture. If this ever changes, it must be a deliberate decision,
# not an accidental side effect of a refactor.
# ---------------------------------------------------------------------------


def test_characterization_large_frontier_strip_flattens_to_three_candidates():
    candidates = detect_global_frontier_candidates(
        explored_points=_large_strip_points(),
        mapped_obstacle_points=[],
        bounds=STRIP_BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert [c.target for c in candidates] == [(-2.0, 0.0), (0.0, 0.0), (1.5, 0.0)]
    assert [c.size for c in candidates] == [5, 6, 6]
    assert [c.information_gain for c in candidates] == [71.0, 70.0, 70.0]
    for candidate in candidates:
        assert candidate.score == candidate.information_gain
        assert candidate.distance_from_robot == 0.0
        assert candidate.reason == f"frontier size={candidate.size}, info_gain={candidate.information_gain:.1f}"


# ---------------------------------------------------------------------------
# 1. Two disconnected components.
# ---------------------------------------------------------------------------


def test_two_disconnected_frontiers_produce_two_clusters():
    clusters = detect_connected_frontier_components(
        explored_points=_two_disconnected_blob_points(),
        mapped_obstacle_points=[],
        bounds=BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert len(clusters) == 2
    for cluster in clusters:
        assert isinstance(cluster, FrontierCluster)
        assert len(cluster.cells) > 0
    ids = [cluster.cluster_id for cluster in clusters]
    assert len(set(ids)) == 2


# ---------------------------------------------------------------------------
# 2. Every returned cell is a genuine frontier cell: explored, free (not
#    occupied), and has at least one cardinal (4-connected) unknown
#    neighbor.
# ---------------------------------------------------------------------------


def test_every_returned_cell_is_a_genuine_frontier_cell():
    explored_points = _two_disconnected_blob_points()
    mapped_obstacle_points: list[tuple[float, float]] = [(9.5, 9.5)]
    explored = {
        _cell_key(point, RESOLUTION) for point in explored_points if _inside_bounds(point, BOUNDS)
    }
    occupied = _occupied_cells_from_points(mapped_obstacle_points, RESOLUTION, 0.35)

    clusters = detect_connected_frontier_components(
        explored_points=explored_points,
        mapped_obstacle_points=mapped_obstacle_points,
        bounds=BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert clusters  # sanity: this fixture must produce something to check
    for cluster in clusters:
        for cell in cluster.cells:
            cell_key = _cell_key(cell, RESOLUTION)
            assert cell_key in explored
            assert cell_key not in occupied
            has_unknown_neighbor = any(
                (cell_key[0] + dx, cell_key[1] + dy) not in explored
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            assert has_unknown_neighbor


# ---------------------------------------------------------------------------
# 3. Determinism: reordering explored_points/mapped_obstacle_points never
#    changes the result.
# ---------------------------------------------------------------------------


def test_input_point_order_does_not_change_the_result():
    explored_points = _two_disconnected_blob_points()
    obstacle_points = [(3.0, 3.0), (3.5, 3.0), (11.0, 12.0)]

    forward = detect_connected_frontier_components(
        explored_points=explored_points,
        mapped_obstacle_points=obstacle_points,
        bounds=BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )
    reordered = detect_connected_frontier_components(
        explored_points=list(reversed(explored_points)),
        mapped_obstacle_points=list(reversed(obstacle_points)),
        bounds=BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert forward == reordered
    assert len(forward) == 2


# ---------------------------------------------------------------------------
# 4. Order and IDs: cells sorted, components ordered, cluster_id sequential.
# ---------------------------------------------------------------------------


def test_cells_are_sorted_and_ids_are_sequential():
    clusters = detect_connected_frontier_components(
        explored_points=_two_disconnected_blob_points(),
        mapped_obstacle_points=[],
        bounds=BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert [cluster.cluster_id for cluster in clusters] == [
        "frontier-component-0000",
        "frontier-component-0001",
    ]
    for cluster in clusters:
        discrete = [_cell_key(cell, RESOLUTION) for cell in cluster.cells]
        assert discrete == sorted(discrete)
    # The lower-numbered component's cells must sort before the other's --
    # components are ordered by their own lexicographically-smallest cell.
    assert clusters[0].cells[0] < clusters[1].cells[0]


# ---------------------------------------------------------------------------
# 5. Centroid is the arithmetic mean of the WHOLE component's cells, not a
#    slice.
# ---------------------------------------------------------------------------


def test_centroid_is_the_mean_of_all_component_cells():
    clusters = detect_connected_frontier_components(
        explored_points=_two_disconnected_blob_points(),
        mapped_obstacle_points=[],
        bounds=BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert clusters
    for cluster in clusters:
        expected_x = sum(cell[0] for cell in cluster.cells) / len(cluster.cells)
        expected_y = sum(cell[1] for cell in cluster.cells) / len(cluster.cells)
        assert cluster.centroid == (expected_x, expected_y)


# ---------------------------------------------------------------------------
# 6. A large connected frontier stays ONE cluster with multiple viewpoints
#    -- its slices never become separate clusters.
# ---------------------------------------------------------------------------


def test_large_frontier_is_one_cluster_with_multiple_viewpoints():
    clusters = detect_connected_frontier_components(
        explored_points=_large_strip_points(),
        mapped_obstacle_points=[],
        bounds=STRIP_BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert len(clusters) == 1
    assert len(clusters[0].viewpoints) > 1
    assert clusters[0].information_gain == max(vp.information_gain for vp in clusters[0].viewpoints)
    assert clusters[0].valid is True


# ---------------------------------------------------------------------------
# 7. The flattened legacy candidate view stays equivalent to the
#    characterized fixture (targets/sizes/information_gain).
# ---------------------------------------------------------------------------


def test_flatten_legacy_candidates_match_characterized_fixture():
    candidates = detect_global_frontier_candidates(
        explored_points=_large_strip_points(),
        mapped_obstacle_points=[],
        bounds=STRIP_BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert [c.target for c in candidates] == [(-2.0, 0.0), (0.0, 0.0), (1.5, 0.0)]
    assert [c.size for c in candidates] == [5, 6, 6]
    assert [c.information_gain for c in candidates] == [71.0, 70.0, 70.0]


# ---------------------------------------------------------------------------
# No frontier data -> empty tuple, not an exception.
# ---------------------------------------------------------------------------


def test_no_explored_points_returns_empty_tuple():
    clusters = detect_connected_frontier_components(
        explored_points=[],
        mapped_obstacle_points=[],
        bounds=BOUNDS,
        resolution=RESOLUTION,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    assert clusters == ()
