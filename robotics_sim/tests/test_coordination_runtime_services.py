from __future__ import annotations

from dataclasses import dataclass

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_sim.simulation.coordination_services import (
    RuntimeFrontierProvider,
    RuntimeTeamFrontierProvider,
)


@dataclass(frozen=True)
class FakePlannerAssignment:
    target: tuple[float, float]
    reason: str = "fake frontier"
    information_gain: float = 7.0
    distance: float = 2.0
    score: float = 6.0
    route_overlap_ratio: float = 0.0


@dataclass(frozen=True)
class FakePlannerResult:
    targets: tuple[tuple[float, float] | None, ...]
    reasons: tuple[str, ...]
    assignments: tuple[FakePlannerAssignment | None, ...]


def _robot(robot_id: int = 0, xy: tuple[float, float] = (0.0, 0.0)) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=xy,
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="LiDAR",
    )


def _world() -> WorldSnapshot:
    return WorldSnapshot(
        explored_points=((0.0, 0.0),),
        mapped_obstacle_points=(),
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
    )


def test_runtime_frontier_provider_adapts_legacy_planner(monkeypatch):
    from robotics_sim.simulation import coordination_services as services_module

    calls = []

    def fake_assign_frontier_viewpoints(**kwargs):
        calls.append(kwargs)
        return FakePlannerResult(
            targets=((2.0, 0.0),),
            reasons=("fake reason",),
            assignments=(FakePlannerAssignment(target=(2.0, 0.0)),),
        )

    monkeypatch.setattr(
        services_module,
        "assign_frontier_viewpoints",
        fake_assign_frontier_viewpoints,
    )

    provider = RuntimeFrontierProvider(ipp_distance_penalty=0.25)

    candidates = provider.candidates_for_robot(
        robot=_robot(),
        world=_world(),
        blocked_targets=((1.0, 1.0),),
    )

    assert len(candidates) == 1
    assert candidates[0].target == (2.0, 0.0)
    assert candidates[0].information_gain == 7.0
    assert candidates[0].travel_cost == 0.5
    assert calls[0]["robots_to_assign"] == (0,)
    assert calls[0]["invalidated_targets_by_robot"] == (((1.0, 1.0),),)


def test_runtime_team_frontier_provider_exposes_raw_candidate_pool(monkeypatch):
    from robotics_sim.simulation import coordination_services as services_module
    from robotics_sim.planning.exploration_planners import FrontierCandidate

    calls = []

    def fake_detect_global_frontier_candidates(**kwargs):
        calls.append(kwargs)
        return (
            FrontierCandidate(
                target=(2.0, 0.0),
                size=3,
                distance_from_robot=0.0,
                score=8.0,
                information_gain=7.0,
                reason="fake frontier east",
            ),
            FrontierCandidate(
                target=(0.0, 2.0),
                size=2,
                distance_from_robot=0.0,
                score=7.0,
                information_gain=6.0,
                reason="fake frontier north",
            ),
        )

    def forbidden_assign_frontier_viewpoints(**kwargs):
        raise AssertionError(
            "RuntimeTeamFrontierProvider must not call assign_frontier_viewpoints(); "
            "providers expose candidates, coordinators allocate targets."
        )

    monkeypatch.setattr(
        services_module,
        "detect_global_frontier_candidates",
        fake_detect_global_frontier_candidates,
    )
    monkeypatch.setattr(
        services_module,
        "assign_frontier_viewpoints",
        forbidden_assign_frontier_viewpoints,
    )

    request = CoordinationRequest(
        robot_states=(
            _robot(0, (0.0, 0.0)),
            _robot(1, (0.0, 1.0)),
        ),
        robots_to_assign=(0, 1),
        world=_world(),
        existing_targets_by_robot={0: None, 1: None},
        blocked_targets_by_robot={0: ((1.0, 1.0),), 1: ()},
        route_points_by_robot=((), ()),
        shared={"explored_points_by_robot": (((0.0, 0.0),), ((0.0, 1.0),))},
    )

    provider = RuntimeTeamFrontierProvider(ipp_distance_penalty=0.25)
    candidates = provider.candidates_for_team(request)

    assert len(calls) == 1
    assert calls[0]["explored_points"] == ((0.0, 0.0),)
    assert calls[0]["mapped_obstacle_points"] == ()
    assert calls[0]["bounds"] == (-5.0, 5.0, -5.0, 5.0)
    assert calls[0]["resolution"] == 0.5

    assert set(candidates) == {0, 1}

    for robot_id in (0, 1):
        assert len(candidates[robot_id]) == 2
        assert {candidate.target for candidate in candidates[robot_id]} == {
            (2.0, 0.0),
            (0.0, 2.0),
        }
        assert all(
            candidate.source == "runtime_team_frontier_provider"
            for candidate in candidates[robot_id]
        )
        assert all(
            candidate.metadata["provider"] == "RuntimeTeamFrontierProvider"
            for candidate in candidates[robot_id]
        )
        assert all(
            candidate.metadata["team_synchronized"] is True
            for candidate in candidates[robot_id]
        )

def test_runtime_passes_min_frontier_distance_to_plugin():
    from robotics_sim.simulation import coordination as sim_coord

    coordinator = sim_coord.MultiRobotCoordinator(strategy=MMPF_COORDINATOR)
    request = coordinator._build_plugin_request(
        planner_name="test planner",
        robot_states=[
            sim_coord.RobotCoordinationState(
                xy=(-1.0, -0.6),
                safety_radius=0.35,
                sensor_range=2.5,
                vision_model="LiDAR",
            ),
        ],
        existing_targets=[None],
        robots_to_assign=[0],
        invalidated_targets_by_robot=[[]],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.50,
        final_goal_xy=(5.0, 5.0),
        ipp_distance_penalty=0.5,
        target_exclusion_radius=1.5,
        dynamic_obstacle_margin=0.5,
        route_points_by_robot=[[]],
        explored_points_by_robot=[[]],
        goal_tolerance=0.25,
    )

    assert request.parameters["grid_resolution"] == 0.50
    assert request.parameters["goal_tolerance"] == 0.25
    assert request.parameters["min_frontier_travel_distance"] == 1.0


def test_multi_robot_coordinator_injects_world_and_team_runtime_services(monkeypatch):
    from robotics_sim.simulation import coordination as sim_coord
    from robotics_sim.simulation import coordination_services as services_module
    from robotics_sim.planning.exploration_planners import FrontierCandidate

    calls = []

    def fake_detect_global_frontier_candidates(**kwargs):
        calls.append(kwargs)
        return (
            FrontierCandidate(
                target=(3.0, 0.0),
                size=4,
                distance_from_robot=0.0,
                score=8.0,
                information_gain=7.0,
                reason="fake runtime frontier east",
            ),
            FrontierCandidate(
                target=(0.0, 3.0),
                size=3,
                distance_from_robot=0.0,
                score=7.0,
                information_gain=6.0,
                reason="fake runtime frontier north",
            ),
        )

    def forbidden_assign_frontier_viewpoints(**kwargs):
        raise AssertionError(
            "RuntimeTeamFrontierProvider must not call assign_frontier_viewpoints() "
            "during team candidate generation."
        )

    monkeypatch.setattr(
        services_module,
        "detect_global_frontier_candidates",
        fake_detect_global_frontier_candidates,
    )
    monkeypatch.setattr(
        services_module,
        "assign_frontier_viewpoints",
        forbidden_assign_frontier_viewpoints,
    )

    coordinator = sim_coord.MultiRobotCoordinator(strategy=MMPF_COORDINATOR)
    result = coordinator.assign_frontiers(
        planner_name="test planner",
        robot_states=[
            sim_coord.RobotCoordinationState(
                xy=(0.0, 0.0),
                safety_radius=0.35,
                sensor_range=2.5,
                vision_model="LiDAR",
            ),
            sim_coord.RobotCoordinationState(
                xy=(0.0, 1.0),
                safety_radius=0.35,
                sensor_range=2.5,
                vision_model="LiDAR",
            ),
        ],
        existing_targets=[None, None],
        robots_to_assign=[0, 1],
        invalidated_targets_by_robot=[[], []],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
        route_points_by_robot=[[], []],
        explored_points_by_robot=[[(0.0, 0.0)], [(0.0, 1.0)]],
    )

    assert len(calls) == 1
    assert calls[0]["explored_points"] == ((0.0, 0.0),)

    assert result.strategy == MMPF_COORDINATOR
    assert result.debug["source"] == "team_frontier_provider"
    assert set(result.targets) == {(3.0, 0.0), (0.0, 3.0)}
    assert result.reasons[0].startswith("selected by MMPF explore coordinator")
    assert result.reasons[1].startswith("selected by MMPF explore coordinator")