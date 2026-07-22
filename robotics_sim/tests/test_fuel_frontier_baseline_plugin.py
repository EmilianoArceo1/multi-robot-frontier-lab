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


class FakeTeamFrontierProvider:
    def __init__(self, candidates_by_robot):
        self.candidates_by_robot = dict(candidates_by_robot)
        self.calls = 0

    def candidates_for_team(self, request):
        self.calls += 1
        return self.candidates_by_robot


class FakeFrontierProvider:
    def __init__(self, candidates_by_robot):
        self.candidates_by_robot = dict(candidates_by_robot)
        self.calls = []

    def candidates_for_robot(self, robot, world, blocked_targets=()):
        self.calls.append(robot.robot_id)
        return self.candidates_by_robot.get(robot.robot_id, ())


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


def test_fuel_uses_frontier_information_service_when_available():
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


def test_fuel_falls_back_to_team_frontier_provider_when_no_clusters():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    empty_frontier_service = FakeFrontierInformationService(clusters=())
    team_provider = FakeTeamFrontierProvider(
        candidates_by_robot={
            0: (ExplorationCandidate(target=(3.0, 0.0), source="team", information_gain=6.0),),
        }
    )
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        world=_world(),
        services=SimpleNamespace(
            frontier_information_service=empty_frontier_service,
            team_frontier_provider=team_provider,
        ),
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert empty_frontier_service.calls == [0]
    assert team_provider.calls == 1
    assert result.targets == ((3.0, 0.0),)
    assert result.debug["per_robot"][0]["candidate_source"] == "team_frontier_provider"


def test_fuel_falls_back_to_frontier_provider_when_no_team_candidates():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    team_provider = FakeTeamFrontierProvider(candidates_by_robot={})
    single_provider = FakeFrontierProvider(
        candidates_by_robot={
            0: (ExplorationCandidate(target=(2.0, 1.0), source="single", information_gain=4.0),),
        }
    )
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        world=_world(),
        services=SimpleNamespace(
            frontier_information_service=None,
            team_frontier_provider=team_provider,
            frontier_provider=single_provider,
        ),
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert team_provider.calls == 1
    assert single_provider.calls == [0]
    assert result.targets == ((2.0, 1.0),)
    assert result.debug["per_robot"][0]["candidate_source"] == "frontier_provider"


def test_fuel_generates_bootstrap_targets_when_no_frontiers_exist():
    """No proposals, no services, no map data (t=0) -- FUEL must still move
    the team instead of holding forever, or it never senses anything new."""
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            _robot(0, 0.0, 0.0),
            _robot(1, 0.5, 0.0),
            _robot(2, 0.0, 0.5),
        ),
        robots_to_assign=(0, 1, 2),
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert all(status.status == "ASSIGNED" for status in result.assignments)
    assert all(
        reason == "bootstrap exploration target while waiting for frontier clusters"
        for reason in result.reasons
    )
    assert len(set(result.targets)) == 3  # distinct targets, no collapse
    for robot_id, debug in result.debug["per_robot"].items():
        assert debug["candidate_source"] == "bootstrap"

    for robot_id, target in zip((0, 1, 2), result.targets):
        start = {0: (0.0, 0.0), 1: (0.5, 0.0), 2: (0.0, 0.5)}[robot_id]
        distance = ((target[0] - start[0]) ** 2 + (target[1] - start[1]) ** 2) ** 0.5
        assert distance >= max(1.5, 0.75 * 2.5) - 1e-6


def test_fuel_returns_commands_for_all_active_robots():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            _robot(0, 0.0, 0.0),
            _robot(1, 3.0, 0.0),
            _robot(2, 0.0, 3.0),
        ),
        # No robots_to_assign given -> every active robot must be handled.
        parameters={"grid_resolution": 0.5, "goal_tolerance": 0.25},
    )

    result = plugin.assign(request)

    assert len(result.commands) == 3
    assert {command.robot_id for command in result.commands} == {0, 1, 2}
    assert all(command.status == "ASSIGNED" for command in result.commands)


def test_fuel_hold_only_when_all_candidate_sources_fail_and_bootstrap_disabled():
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        parameters={
            "grid_resolution": 0.5,
            "goal_tolerance": 0.25,
            "fuel_enable_bootstrap": False,
        },
    )

    result = plugin.assign(request)

    assert result.targets == (None,)
    assert result.assignments[0].status == "HOLD"
    assert result.debug["per_robot"][0]["candidate_source"] == "none"
    assert "candidate_source=none" in result.assignments[0].reason


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


def test_fuel_cell_count_gain_does_not_overpower_large_extra_travel():
    """One extra UNKNOWN cell used to offset five metres at lambda=0.2.

    Runtime information gain is a cell count (roughly up to 80 cells for the
    default sensor/grid), so raw gain and metre cost are not commensurate.
    Ten additional cells must not justify travelling six extra metres when a
    strong nearby frontier is available.
    """
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={
            0: (
                ExplorationCandidate(
                    target=(2.0, 0.0),
                    information_gain=60.0,
                    metadata={"cluster_id": "near"},
                ),
                ExplorationCandidate(
                    target=(8.0, 0.0),
                    information_gain=70.0,
                    metadata={"cluster_id": "far"},
                ),
            ),
        },
        parameters={
            "grid_resolution": 0.5,
            "goal_tolerance": 0.25,
            "ipp_distance_penalty": 0.2,
        },
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 0.0),)
    assert result.commands[0].metadata["fuel_score"] > 0.0


def test_fuel_selects_viewpoint_with_robot_aware_cost_inside_cluster():
    """All viewpoints must reach `_fuel_score` before one per cluster wins."""
    plugin = load_coordination_plugin(FUEL_FRONTIER_BASELINE_COORDINATOR)
    service = FakeFrontierInformationService(
        clusters=(
            FakeFrontierCluster(
                cluster_id="long-frontier",
                centroid=(5.0, 0.0),
                viewpoints=(
                    FakeViewpoint(xy=(2.0, 0.0), information_gain=60.0),
                    FakeViewpoint(xy=(8.0, 0.0), information_gain=70.0),
                ),
            ),
        )
    )
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        world=_world(),
        services=SimpleNamespace(frontier_information_service=service),
        parameters={
            "grid_resolution": 0.5,
            "goal_tolerance": 0.25,
            "ipp_distance_penalty": 0.2,
        },
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 0.0),)
    assert result.debug["per_robot"][0]["raw_candidates"] == 2
    assert result.debug["per_robot"][0]["clustered_candidates"] == 1
    assert result.commands[0].metadata["cluster_id"] == "long-frontier"
