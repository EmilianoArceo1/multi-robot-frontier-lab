from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from algorithms.fuel_frontier_baseline.plugin import FUEL_FRONTIER_BASELINE_COORDINATOR
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.plugins import PluginCapability
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_interfaces.services import CoordinationServices
from robotics_sim.simulation.plugin_loader import list_coordination_plugin_names, load_coordination_plugin


def _robot(robot_id: int, x: float, y: float, theta: float = 0.0, current_target=None):
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        theta=theta,
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
        current_target=current_target,
    )


def _world():
    return WorldSnapshot(
        explored_points=((0.0, 0.0),),
        mapped_obstacle_points=(),
        bounds=(-8.0, 8.0, -8.0, 8.0),
        resolution=0.5,
    )


@dataclass(frozen=True)
class FakeViewpoint:
    xy: tuple[float, float]
    heading_rad: float | None = None
    information_gain: float = 0.0
    coverage_fraction: float = 0.0
    visible_cell_count: int = 0
    travel_cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FakeFrontierCluster:
    cluster_id: str
    centroid: tuple[float, float]
    viewpoints: tuple[FakeViewpoint, ...] = ()
    information_gain: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FakeFrontierInformationService:
    def __init__(self, clusters):
        self.clusters = tuple(clusters)
        self.calls = []

    def get_frontier_clusters(self, robot_id=None):
        self.calls.append(robot_id)
        return self.clusters


def test_fuel_frontier_baseline_does_not_import_robotics_sim():
    source = Path("algorithms/fuel_frontier_baseline/plugin.py").read_text(encoding="utf-8")
    assert "robotics_sim" not in source
    assert "ros" not in source.lower()


def test_plugin_loader_discovers_fuel_frontier_baseline():
    names = list_coordination_plugin_names()
    assert FUEL_FRONTIER_BASELINE_COORDINATOR in names

    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    assert plugin.metadata.name == FUEL_FRONTIER_BASELINE_COORDINATOR
    assert PluginCapability.COORDINATION in plugin.metadata.capabilities
    assert PluginCapability.TARGET_GENERATION in plugin.metadata.capabilities
    assert PluginCapability.TASK_ALLOCATION in plugin.metadata.capabilities
    assert PluginCapability.PATH_PLANNING not in plugin.metadata.capabilities
    assert PluginCapability.CONTROL not in plugin.metadata.capabilities


def test_fuel_frontier_baseline_returns_robot_command_with_heading():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0, theta=0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={
            0: (
                ExplorationCandidate(
                    target=(2.0, 0.0),
                    source="test_viewpoint",
                    information_gain=6.0,
                    heading_rad=0.25,
                    metadata={"cluster_id": "frontier-a"},
                ),
            ),
        },
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 0.0),)
    assert len(result.commands) == 1
    command = result.commands[0]
    assert command.status == "ASSIGNED"
    assert command.target == (2.0, 0.0)
    assert command.heading_rad == 0.25
    assert command.metadata["cluster_id"] == "frontier-a"


def test_fuel_frontier_baseline_uses_frontier_information_service_viewpoints():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    service = FakeFrontierInformationService(
        clusters=(
            FakeFrontierCluster(
                cluster_id="cluster-a",
                centroid=(2.0, 0.0),
                viewpoints=(
                    FakeViewpoint(
                        xy=(2.0, 0.0),
                        heading_rad=0.5,
                        information_gain=3.0,
                        coverage_fraction=0.3,
                    ),
                    FakeViewpoint(
                        xy=(3.0, 0.0),
                        heading_rad=0.1,
                        information_gain=8.0,
                        coverage_fraction=0.8,
                    ),
                ),
            ),
        )
    )
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        world=_world(),
        services=SimpleNamespace(frontier_information_service=service),
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert service.calls == [0]
    assert result.targets == ((3.0, 0.0),)
    assert result.commands[0].heading_rad == 0.1
    assert result.commands[0].metadata["cluster_id"] == "cluster-a"


def test_fuel_frontier_baseline_keeps_distinct_clusters_for_multiple_robots():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            _robot(0, 0.0, 0.0),
            _robot(1, 0.0, 1.0),
        ),
        robots_to_assign=(0, 1),
        proposals_by_robot={
            0: (
                ExplorationCandidate(
                    target=(3.0, 0.0),
                    information_gain=10.0,
                    metadata={"cluster_id": "shared"},
                ),
                ExplorationCandidate(
                    target=(0.0, 3.0),
                    information_gain=7.0,
                    metadata={"cluster_id": "north"},
                ),
            ),
            1: (
                ExplorationCandidate(
                    target=(3.0, 0.0),
                    information_gain=10.0,
                    metadata={"cluster_id": "shared"},
                ),
                ExplorationCandidate(
                    target=(0.0, 4.0),
                    information_gain=7.5,
                    metadata={"cluster_id": "north-2"},
                ),
            ),
        },
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert result.targets[0] == (3.0, 0.0)
    assert result.targets[1] == (0.0, 4.0)
    assert result.debug["per_robot"][1]["rejected"]["target_reservation_conflict"] == 1


def test_fuel_frontier_baseline_rejects_near_frontier():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={
            0: (
                ExplorationCandidate(target=(0.25, 0.0), information_gain=100.0),
                ExplorationCandidate(target=(2.0, 0.0), information_gain=5.0),
            ),
        },
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 0.0),)
    assert result.debug["per_robot"][0]["rejected"]["too_close_to_robot"] == 1


def test_fuel_frontier_baseline_preserves_existing_target_when_not_reassigning():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            _robot(0, 0.0, 0.0, current_target=(8.0, 8.0)),
            _robot(1, 0.0, 1.0),
        ),
        robots_to_assign=(1,),
        proposals_by_robot={
            1: (
                ExplorationCandidate(target=(3.0, 1.0), information_gain=4.0),
            ),
        },
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert result.targets[0] == (8.0, 8.0)
    assert result.targets[1] == (3.0, 1.0)
    assert result.reasons[0] == "kept existing target"
