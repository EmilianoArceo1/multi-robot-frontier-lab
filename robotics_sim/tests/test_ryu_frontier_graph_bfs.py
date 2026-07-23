"""Regression tests for the cited Ryu frontier-graph BFS selector."""

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.planning.exploration_planners import (
    EXPLORATION_PLANNER_OPTIONS,
    select_exploration_goal,
)
from robotics_sim.planning.ryu_frontier_graph_bfs import (
    RYU_FRONTIER_GRAPH_BFS,
    RYU_FRONTIER_GRAPH_BFS_CITATION,
    bfs_frontier_nodes,
)
from robotics_sim.simulation.config import FRONTIER_ALGORITHM_DETECTOR_OPTIONS
from robotics_sim.simulation.planner_services import PlannerServices


def _corridor_belief() -> tuple[BeliefMap, tuple[float, float]]:
    belief = BeliefMap(bounds=(-3.0, 3.0, -3.0, 3.0), resolution=1.0)
    for cell in ((3, 1), (3, 2), (3, 3), (3, 4), (2, 3), (1, 3)):
        belief.mark_free_cell(cell)
    return belief, belief.cell_to_world((3, 1))


def test_cited_bfs_is_exposed_in_frontier_detector_selector():
    assert RYU_FRONTIER_GRAPH_BFS in EXPLORATION_PLANNER_OPTIONS
    assert RYU_FRONTIER_GRAPH_BFS in FRONTIER_ALGORITHM_DETECTOR_OPTIONS
    assert "10.3390/s20216270" in RYU_FRONTIER_GRAPH_BFS_CITATION


def test_bfs_orders_reachable_frontier_nodes_by_graph_depth():
    belief, robot_xy = _corridor_belief()
    nodes = bfs_frontier_nodes(
        belief,
        robot_xy,
        dbscan_clusters=(((3, 4),), ((1, 3),)),
    )

    assert [node.representative for node in nodes] == [(3, 4), (1, 3)]
    assert [node.bfs_depth for node in nodes] == [3, 4]


def test_bfs_uses_paper_ccl_fallback_when_dbscan_is_unavailable():
    belief, robot_xy = _corridor_belief()
    services = PlannerServices()
    services.clustering_algorithm = "missing DBSCAN implementation"

    result = services.select_exploration_target(
        planner_name=RYU_FRONTIER_GRAPH_BFS,
        belief_map=belief,
        robot_xy=robot_xy,
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_radius=0.2,
        sensor_range=2.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
    )

    assert result.success is True
    assert result.target is not None
    assert "8-connected CCL fallback" in result.reason
    assert "10.3390/s20216270" in result.reason


def test_bfs_accepts_dbscan_frontier_nodes_without_using_fallback():
    belief, robot_xy = _corridor_belief()
    result = select_exploration_goal(
        RYU_FRONTIER_GRAPH_BFS,
        belief_map=belief,
        robot_xy=robot_xy,
        frontier_clusters=(((3, 4),),),
        clustering_algorithm="DBSCAN",
    )

    assert result.success is True
    assert result.target == belief.cell_to_world((3, 4))
    assert "selected DBSCAN frontier nodes" in result.reason
