from __future__ import annotations

from robotics_interfaces.coordination import CoordinationResult
from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.frontiers import FrontierCluster, ViewpointCandidate
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_interfaces.regions import CoveragePath, RegionTask
from robotics_interfaces.services import (
    CollisionCheckingService,
    CoordinationServices,
    CoveragePathService,
    FrontierInformationService,
    FrontierProvider,
    MapQueryService,
    MetricsService,
    PathPlanningService,
    RegionDecompositionService,
    TeamFrontierProvider,
)
from robotics_sim.simulation.runtime_services import RuntimeFrontierInformationService


def test_viewpoint_candidate_can_convert_to_exploration_candidate():
    viewpoint = ViewpointCandidate(
        xy=(2.0, 1.0),
        heading_rad=0.5,
        information_gain=6.0,
        travel_cost=1.0,
        coverage_fraction=0.4,
        visible_cell_count=12,
    )

    candidate = viewpoint.as_exploration_candidate(source="fuel_style_viewpoint")

    assert isinstance(candidate, ExplorationCandidate)
    assert candidate.target == (2.0, 1.0)
    assert candidate.heading_rad == 0.5
    assert candidate.information_gain == 6.0
    assert candidate.travel_cost == 1.0
    assert candidate.source == "fuel_style_viewpoint"
    assert candidate.metadata["coverage_fraction"] == 0.4
    assert candidate.metadata["visible_cell_count"] == 12


def test_frontier_cluster_exposes_best_viewpoint():
    low = ViewpointCandidate(xy=(1.0, 0.0), information_gain=2.0)
    high = ViewpointCandidate(xy=(3.0, 0.0), information_gain=9.0, travel_cost=1.0)
    cluster = FrontierCluster(
        cluster_id="cluster-0",
        centroid=(2.0, 0.0),
        viewpoints=(low, high),
        information_gain=9.0,
    )

    assert cluster.best_viewpoint is high

    candidate = cluster.as_exploration_candidate()
    assert candidate is not None
    assert candidate.target == (3.0, 0.0)

    empty_cluster = FrontierCluster(cluster_id="empty")
    assert empty_cluster.best_viewpoint is None
    assert empty_cluster.as_exploration_candidate() is None


def test_region_task_represents_unknown_workload():
    region = RegionTask(
        region_id="region-a",
        centroid=(4.0, 4.0),
        unknown_cell_count=25,
        cells=((3.5, 3.5), (4.5, 4.5)),
    )

    assert region.assigned_robot_id is None
    assert region.unknown_cell_count == 25
    assert len(region.cells) == 2

    assigned = RegionTask(
        region_id="region-b",
        centroid=(0.0, 0.0),
        assigned_robot_id=1,
    )
    assert assigned.assigned_robot_id == 1


def test_coverage_path_represents_region_sequence():
    path = CoveragePath(
        robot_id=0,
        waypoints=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0)),
        region_ids=("region-a", "region-b"),
        estimated_cost=4.0,
    )

    assert path.robot_id == 0
    assert len(path.waypoints) == 3
    assert path.region_ids == ("region-a", "region-b")
    assert path.estimated_cost == 4.0


def test_coordination_services_accept_frontier_region_coverage_services():
    services = CoordinationServices()
    assert services.frontier_information_service is None
    assert services.region_decomposition_service is None
    assert services.coverage_path_service is None

    for protocol in (
        FrontierProvider,
        TeamFrontierProvider,
        PathPlanningService,
        CollisionCheckingService,
        MapQueryService,
        MetricsService,
        FrontierInformationService,
        RegionDecompositionService,
        CoveragePathService,
    ):
        assert getattr(protocol, "_is_protocol", False) is True


def test_runtime_frontier_information_service_is_invocable():
    explored_points = tuple((x * 0.5, 0.0) for x in range(-8, 9))
    service = RuntimeFrontierInformationService(
        explored_points=explored_points,
        mapped_obstacle_points=(),
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    clusters = service.get_frontier_clusters()

    assert len(clusters) >= 1
    for cluster in clusters:
        assert isinstance(cluster, FrontierCluster)
        assert cluster.best_viewpoint is not None

    # No map data -> empty tuple, not an exception.
    empty_service = RuntimeFrontierInformationService()
    assert empty_service.get_frontier_clusters() == ()


def test_robot_command_preserves_heading_rad():
    command = RobotCommand(
        robot_id=0,
        status="ASSIGNED",
        target=(3.0, 0.0),
        heading_rad=1.2,
    )
    result = CoordinationResult(
        targets=(command.target,),
        reasons=("viewpoint with yaw",),
        strategy="test",
        commands=(command,),
    )

    assert result.commands[0].heading_rad == 1.2
