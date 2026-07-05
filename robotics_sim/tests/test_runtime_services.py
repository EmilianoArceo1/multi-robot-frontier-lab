from __future__ import annotations

from robotics_interfaces.plugins import PluginCapability
from robotics_interfaces.results import PathPlanningRequest
from robotics_interfaces.services import (
    CollisionCheckingService,
    FrontierProvider,
    MapQueryService,
    MetricsService,
    PathPlanningService,
    TeamFrontierProvider,
)
from robotics_sim.simulation.runtime_services import (
    RuntimeCollisionCheckingService,
    RuntimePathPlanningService,
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
