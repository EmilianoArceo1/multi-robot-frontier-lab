from __future__ import annotations

from pathlib import Path

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate
from robotics_interfaces.services import CoordinationServices
from robotics_sim.simulation.plugin_loader import load_coordination_plugin


def _robot(robot_id: int, x: float, y: float) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
    )


class FakeFrontierProvider:
    def __init__(self):
        self.calls = []

    def candidates_for_robot(self, robot, world, blocked_targets=()):
        self.calls.append((robot.robot_id, world, blocked_targets))
        return (
            ExplorationCandidate(
                target=(robot.xy[0] + 2.0, robot.xy[1]),
                source="fake_frontier_provider",
                information_gain=8.0,
                travel_cost=1.0,
            ),
        )


def test_mmpf_plugin_is_discoverable():
    plugin = load_coordination_plugin(MMPF_COORDINATOR)

    assert plugin.metadata.name == MMPF_COORDINATOR


def test_mmpf_can_assign_from_explicit_candidates():
    plugin = load_coordination_plugin(MMPF_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={
            0: (
                CandidateProposal(robot_id=0, target=(1.0, 0.0), score=1.0),
                CandidateProposal(robot_id=0, target=(2.0, 0.0), score=5.0),
            )
        },
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 0.0),)
    assert result.assignments[0].status == "ASSIGNED"
    assert result.assignments[0].target == (2.0, 0.0)


def test_mmpf_can_request_candidates_from_frontier_provider():
    plugin = load_coordination_plugin(MMPF_COORDINATOR)
    provider = FakeFrontierProvider()
    world = WorldSnapshot(
        explored_points=((0.0, 0.0),),
        mapped_obstacle_points=(),
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
    )
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        world=world,
        services=CoordinationServices(frontier_provider=provider),
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 0.0),)
    assert provider.calls
    assert provider.calls[0][0] == 0


def test_mmpf_returns_hold_when_no_candidates_and_no_provider():
    plugin = load_coordination_plugin(MMPF_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
    )

    result = plugin.assign(request)

    assert result.targets == (None,)
    assert result.assignments[0].status == "HOLD"
    assert "no candidates" in result.assignments[0].reason


def test_mmpf_avoids_duplicate_targets_between_robots():
    plugin = load_coordination_plugin(MMPF_COORDINATOR)
    shared_target = CandidateProposal(robot_id=0, target=(1.0, 1.0), score=10.0)
    request = CoordinationRequest(
        robot_states=(
            _robot(0, 0.0, 0.0),
            _robot(1, 0.0, 1.0),
        ),
        robots_to_assign=(0, 1),
        proposals_by_robot={
            0: (shared_target,),
            1: (
                CandidateProposal(robot_id=1, target=(1.0, 1.0), score=10.0),
                CandidateProposal(robot_id=1, target=(2.0, 1.0), score=5.0),
            ),
        },
    )

    result = plugin.assign(request)

    assert result.targets == ((1.0, 1.0), (2.0, 1.0))


def test_mmpf_plugin_does_not_import_robotics_sim():
    source = Path("algorithms/mmpf_explore/plugin.py").read_text(encoding="utf-8")

    assert "robotics_sim" not in source
