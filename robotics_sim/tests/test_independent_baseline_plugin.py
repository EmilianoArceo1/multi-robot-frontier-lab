from __future__ import annotations

from pathlib import Path

from algorithms.independent_baseline.plugin import INDEPENDENT_BASELINE_COORDINATOR
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.plugins import PluginCapability
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate
from robotics_interfaces.services import CoordinationServices
from robotics_sim.simulation.plugin_loader import list_coordination_plugin_names, load_coordination_plugin


def _robot(robot_id: int, x: float, y: float, current_target=None) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
        current_target=current_target,
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


def _world() -> WorldSnapshot:
    return WorldSnapshot(
        explored_points=((0.0, 0.0),),
        mapped_obstacle_points=(),
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
    )


def test_independent_baseline_does_not_import_robotics_sim():
    source = Path("algorithms/independent_baseline/plugin.py").read_text(encoding="utf-8")
    assert "robotics_sim" not in source


def test_plugin_loader_discovers_independent_baseline():
    names = list_coordination_plugin_names()
    assert INDEPENDENT_BASELINE_COORDINATOR in names

    plugin = load_coordination_plugin(INDEPENDENT_BASELINE_COORDINATOR)
    assert plugin.metadata.name == INDEPENDENT_BASELINE_COORDINATOR
    assert PluginCapability.TARGET_GENERATION in plugin.metadata.capabilities
    assert PluginCapability.TASK_ALLOCATION in plugin.metadata.capabilities
    assert PluginCapability.PATH_PLANNING not in plugin.metadata.capabilities
    assert PluginCapability.CONTROL not in plugin.metadata.capabilities


def test_independent_baseline_returns_robot_commands():
    plugin = load_coordination_plugin(INDEPENDENT_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={
            0: (
                CandidateProposal(robot_id=0, target=(1.0, 0.0), score=1.0, information_gain=1.0),
                CandidateProposal(robot_id=0, target=(2.0, 0.0), score=5.0, information_gain=5.0),
            ),
        },
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 0.0),)
    assert len(result.commands) == 1
    command = result.commands[0]
    assert command.robot_id == 0
    assert command.status == "ASSIGNED"
    assert command.target == (2.0, 0.0)


def test_independent_baseline_uses_frontier_service_when_available():
    plugin = load_coordination_plugin(INDEPENDENT_BASELINE_COORDINATOR)
    provider = FakeFrontierProvider()
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        world=_world(),
        services=CoordinationServices(frontier_provider=provider),
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 0.0),)
    assert provider.calls
    assert provider.calls[0][0] == 0


def test_independent_baseline_preserves_existing_target_when_not_reassigning():
    plugin = load_coordination_plugin(INDEPENDENT_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            _robot(0, 0.0, 0.0, current_target=(9.0, 9.0)),
            _robot(1, 5.0, 0.0),
        ),
        robots_to_assign=(1,),
        proposals_by_robot={
            1: (CandidateProposal(robot_id=1, target=(6.0, 0.0), score=4.0, information_gain=4.0),),
        },
    )

    result = plugin.assign(request)

    assert result.targets[0] == (9.0, 9.0)
    assert result.targets[1] == (6.0, 0.0)
    assert result.reasons[0] == "kept existing target"
