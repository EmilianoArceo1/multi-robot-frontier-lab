from __future__ import annotations

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from robotics_interfaces.plugins import PluginCapability
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_interfaces.results import PathPlanningRequest
from robotics_interfaces.services import (
    CollisionCheckingService,
    FrontierProvider,
    MapQueryService,
    MetricsService,
    PathPlanningService,
    TeamFrontierProvider,
)
from robotics_sim.simulation.coordination import MultiRobotCoordinator, RobotCoordinationState
from robotics_sim.simulation.runtime_services import (
    RuntimeCollisionCheckingService,
    RuntimeFrontierInformationService,
    RuntimeMapQueryService,
    RuntimeMetricsService,
    RuntimePathPlanningService,
    frontier_clusters_from_candidates,
)


def _build_request(coordinator: MultiRobotCoordinator):
    return coordinator._build_plugin_request(
        planner_name="test planner",
        robot_states=[
            RobotCoordinationState(xy=(0.0, 0.0), safety_radius=0.35, sensor_range=2.5, vision_model="LiDAR"),
            RobotCoordinationState(xy=(2.0, 0.0), safety_radius=0.35, sensor_range=2.5, vision_model="LiDAR"),
        ],
        existing_targets=[None, None],
        robots_to_assign=[0, 1],
        invalidated_targets_by_robot=[[], []],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
        ipp_distance_penalty=0.5,
        target_exclusion_radius=1.5,
        dynamic_obstacle_margin=0.25,
        route_points_by_robot=[[], []],
        explored_points_by_robot=[[], []],
        goal_tolerance=0.25,
    )


def test_runtime_services_are_interface_protocols():
    for protocol in (
        FrontierProvider,
        TeamFrontierProvider,
        PathPlanningService,
        CollisionCheckingService,
        MapQueryService,
        MetricsService,
    ):
        assert getattr(protocol, "_is_protocol", False) is True

    # Sanity: capability enum should stay usable as the vocabulary these
    # services are gated by, even though the protocols themselves are generic.
    assert PluginCapability.PATH_PLANNING.value == "path_planning"


def test_algorithm_can_request_path_without_importing_robotics_sim():
    """An algorithm module only needs robotics_interfaces to call a
    PathPlanningService -- it should never need to import robotics_sim
    to build the request or read the response."""

    class FakeAlgorithmPathPlanningService:
        """Duck-typed like an algorithm-side test double would be: no
        robotics_sim import anywhere in this class."""

        def plan_path(self, request: PathPlanningRequest):
            from robotics_interfaces.results import PathPlanningResponse

            return PathPlanningResponse(
                success=True,
                waypoints=(request.start, request.goal),
                reason="fake straight line",
            )

    service: PathPlanningService = FakeAlgorithmPathPlanningService()
    request = PathPlanningRequest(
        start=(0.0, 0.0),
        goal=(4.0, 0.0),
        robot_radius=0.35,
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
    )

    response = service.plan_path(request)

    assert isinstance(service, PathPlanningService)
    assert response.success is True
    assert response.waypoints == ((0.0, 0.0), (4.0, 0.0))


def test_runtime_path_planning_service_uses_planner_registry():
    service = RuntimePathPlanningService()
    request = PathPlanningRequest(
        start=(0.0, 0.0),
        goal=(4.0, 0.0),
        robot_radius=0.35,
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        planner_type="Direct",
    )

    response = service.plan_path(request)

    assert response.success is True
    assert response.waypoints == ((4.0, 0.0),)


def test_collision_service_rejects_path_crossing_robot_safety_zone():
    service = RuntimeCollisionCheckingService(
        other_robot_disks_by_id={1: (2.0, 0.0, 0.35)},
    )

    result = service.is_path_safe([(0.0, 0.0), (4.0, 0.0)], robot_id=0, safety_radius=0.35)

    assert result.is_safe is False
    assert result.reason_code == "route_conflict_with_robot_safety_zone"


def test_collision_service_accepts_path_without_conflict():
    service = RuntimeCollisionCheckingService(
        other_robot_disks_by_id={1: (2.0, 5.0, 0.35)},
    )

    result = service.is_path_safe([(0.0, 0.0), (4.0, 0.0)], robot_id=0, safety_radius=0.35)

    assert result.is_safe is True


def test_coordination_request_receives_path_planning_service():
    coordinator = MultiRobotCoordinator(strategy=MMPF_COORDINATOR)
    request = _build_request(coordinator)

    service = request.services.path_planning_service
    assert isinstance(service, RuntimePathPlanningService)

    response = service.plan_path(
        PathPlanningRequest(
            start=(0.0, 0.0),
            goal=(4.0, 0.0),
            robot_radius=0.35,
            bounds=(-5.0, 5.0, -5.0, 5.0),
            resolution=0.5,
            planner_type="Direct",
        )
    )
    assert response.success is True
    assert response.waypoints == ((4.0, 0.0),)


def test_coordination_request_receives_collision_checking_service():
    coordinator = MultiRobotCoordinator(strategy=MMPF_COORDINATOR)
    request = _build_request(coordinator)

    service = request.services.collision_checking_service
    assert isinstance(service, RuntimeCollisionCheckingService)

    # Robot 1 sits at (2.0, 0.0); robot 0's straight corridor to (4.0, 0.0)
    # must cross robot 1's safety zone.
    result = service.is_path_safe([(0.0, 0.0), (4.0, 0.0)], robot_id=0, safety_radius=0.35)

    assert result.is_safe is False
    assert result.reason_code == "route_conflict_with_robot_safety_zone"
    assert result.conflict_robot_id == 1


def test_coordination_request_receives_map_query_and_metrics_services():
    coordinator = MultiRobotCoordinator(strategy=MMPF_COORDINATOR)
    request = _build_request(coordinator)

    assert isinstance(request.services.map_query_service, RuntimeMapQueryService)
    assert isinstance(request.services.metrics_service, RuntimeMetricsService)

    snapshot = request.services.map_query_service.map_snapshot()
    assert snapshot.bounds == (-5.0, 5.0, -5.0, 5.0)
    assert snapshot.resolution == 0.5


def test_coordination_request_receives_frontier_information_service():
    coordinator = MultiRobotCoordinator(strategy=MMPF_COORDINATOR)
    request = _build_request(coordinator)

    assert isinstance(request.services.frontier_information_service, RuntimeFrontierInformationService)


def test_runtime_frontier_information_service_can_convert_candidates_to_clusters():
    """When map-based frontier detection finds nothing (e.g. not enough
    explored_points yet), the service must fall back to converting whatever
    legacy candidates (team/single-robot frontier providers) it was given,
    instead of returning empty."""
    candidates = (
        ExplorationCandidate(target=(2.0, 0.0), source="team_frontier", information_gain=5.0),
        ExplorationCandidate(target=(0.0, 3.0), source="team_frontier", information_gain=7.0, heading_rad=0.3),
    )

    clusters = frontier_clusters_from_candidates(candidates, id_prefix="legacy")
    assert len(clusters) == 2
    assert clusters[0].centroid == (2.0, 0.0)
    assert clusters[0].best_viewpoint.information_gain == 5.0
    assert clusters[1].best_viewpoint.heading_rad == 0.3

    service = RuntimeFrontierInformationService(legacy_candidates_by_robot={0: candidates})

    per_robot = service.get_frontier_clusters(robot_id=0)
    assert len(per_robot) == 2

    all_robots = service.get_frontier_clusters()
    assert len(all_robots) == 2

    assert RuntimeFrontierInformationService().get_frontier_clusters() == ()


def test_runtime_frontier_information_service_returns_real_connected_components():
    """With enough map data, get_frontier_clusters() must return real
    detect_connected_frontier_components() output -- non-empty cells, the
    connected-component source tag, and valid=True -- not the old
    one-candidate-per-pseudo-cluster (cells=()) shape."""
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

    assert len(clusters) == 1
    cluster = clusters[0]
    assert len(cluster.cells) > 0
    assert cluster.metadata["source"] == "connected_frontier_component"
    assert cluster.valid is True
    assert len(cluster.viewpoints) > 1


def test_runtime_frontier_information_service_robot_id_does_not_change_geometry():
    """robot_id is only allowed to annotate metadata with
    requested_for_robot_id -- it must never filter or regenerate the
    detected components (cluster_id/cells/centroid/viewpoints/
    information_gain/valid must stay identical)."""
    explored_points = tuple((x * 0.5, 0.0) for x in range(-8, 9))
    service = RuntimeFrontierInformationService(
        explored_points=explored_points,
        mapped_obstacle_points=(),
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        robot_radius=0.35,
        sensor_range=2.5,
    )

    unscoped = service.get_frontier_clusters()
    scoped = service.get_frontier_clusters(robot_id=0)

    assert len(unscoped) == len(scoped)
    for plain, per_robot in zip(unscoped, scoped):
        assert plain.cluster_id == per_robot.cluster_id
        assert plain.cells == per_robot.cells
        assert plain.centroid == per_robot.centroid
        assert plain.viewpoints == per_robot.viewpoints
        assert plain.information_gain == per_robot.information_gain
        assert plain.valid == per_robot.valid
        assert "requested_for_robot_id" not in plain.metadata
        assert per_robot.metadata["requested_for_robot_id"] == 0
        # Every other metadata key must survive untouched.
        for key, value in plain.metadata.items():
            assert per_robot.metadata[key] == value


def test_runtime_frontier_information_service_legacy_fallback_when_map_is_insufficient():
    """Without enough map data, get_frontier_clusters() must fall back to
    legacy_candidates_by_robot -- adapters with cells=() and
    legacy_adapter=True in metadata -- and never mix them with real
    components (there are none to mix with here)."""
    candidates = (
        ExplorationCandidate(target=(2.0, 0.0), source="team_frontier", information_gain=5.0),
        ExplorationCandidate(target=(0.0, 3.0), source="team_frontier", information_gain=7.0),
    )
    service = RuntimeFrontierInformationService(legacy_candidates_by_robot={0: candidates})

    clusters = service.get_frontier_clusters(robot_id=0)

    assert len(clusters) == 2
    targets = [cluster.centroid for cluster in clusters]
    assert targets == [(2.0, 0.0), (0.0, 3.0)]
    for cluster in clusters:
        assert cluster.cells == ()
        assert cluster.valid is True
        assert cluster.metadata["legacy_adapter"] is True
