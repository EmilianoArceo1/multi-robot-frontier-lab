from __future__ import annotations

from dataclasses import dataclass

from algorithms.global_noic_legacy.plugin import NOIC_COORDINATOR
from robotics_interfaces import CoordinationRequest, RobotCoordinationState
from robotics_sim.simulation.plugin_loader import load_coordination_plugin


@dataclass(frozen=True)
class FakePlannerAssignment:
    target: tuple[float, float]
    reason: str = "fake planner selected target"
    information_gain: float = 5.0
    distance: float = 2.0
    other_map_ratio: float = 0.1
    route_overlap_ratio: float = 0.2


@dataclass(frozen=True)
class FakePlannerResult:
    targets: tuple[tuple[float, float] | None, ...]
    reasons: tuple[str, ...]
    assignments: tuple[FakePlannerAssignment | None, ...]


def _state(robot_id: int, x: float, y: float) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
        theta=0.0,
    )


def test_global_noic_legacy_plugin_is_discoverable():
    plugin = load_coordination_plugin(NOIC_COORDINATOR)

    assert plugin.metadata.name == NOIC_COORDINATOR


def test_global_noic_legacy_plugin_calls_injected_legacy_planner():
    calls = []

    def fake_legacy_assign(**kwargs):
        calls.append(kwargs)
        return FakePlannerResult(
            targets=((3.0, 4.0),),
            reasons=("fake assigned",),
            assignments=(FakePlannerAssignment(target=(3.0, 4.0)),),
        )

    plugin = load_coordination_plugin(NOIC_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_state(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        existing_targets_by_robot={0: None},
        blocked_targets_by_robot={0: ()},
        route_points_by_robot=((),),
        shared={
            "legacy_assign_frontier_viewpoints": fake_legacy_assign,
            "explored_points": ((0.0, 0.0),),
            "mapped_obstacle_points": (),
            "bounds": (-5.0, 5.0, -5.0, 5.0),
            "resolution": 0.5,
            "final_goal_xy": (5.0, 5.0),
            "ipp_distance_penalty": 0.2,
            "target_exclusion_radius": 1.0,
            "dynamic_obstacle_margin": 0.25,
            "explored_points_by_robot": (((0.0, 0.0),),),
        },
    )

    result = plugin.assign(request)

    assert len(calls) == 1
    assert calls[0]["robots_to_assign"] == [0]
    assert result.strategy == NOIC_COORDINATOR
    assert result.targets == ((3.0, 4.0),)
    assert result.assignments[0].status == "ASSIGNED"
    assert "info_gain=5.0" in result.reasons[0]


def test_multi_robot_coordinator_loads_noic_as_dynamic_plugin(monkeypatch):
    from robotics_sim.simulation import coordination as sim_coord

    calls = []

    def fake_legacy_assign(**kwargs):
        calls.append(kwargs)
        return FakePlannerResult(
            targets=((1.0, 2.0),),
            reasons=("fake host assigned",),
            assignments=(FakePlannerAssignment(target=(1.0, 2.0)),),
        )

    monkeypatch.setattr(sim_coord, "assign_frontier_viewpoints", fake_legacy_assign)

    coordinator = sim_coord.MultiRobotCoordinator(strategy=sim_coord.NOIC_COORDINATOR)
    result = coordinator.assign_frontiers(
        planner_name="test planner",
        robot_states=[
            sim_coord.RobotCoordinationState(
                xy=(0.0, 0.0),
                safety_radius=0.35,
                sensor_range=2.5,
                vision_model="Camera / FoV",
            )
        ],
        existing_targets=[None],
        robots_to_assign=[0],
        invalidated_targets_by_robot=[[]],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
        route_points_by_robot=[[]],
        explored_points_by_robot=[[(0.0, 0.0)]],
    )

    assert len(calls) == 1
    assert result.strategy == sim_coord.NOIC_COORDINATOR
    assert result.targets == ((1.0, 2.0),)
    assert "info_gain=5.0" in result.reasons[0]
