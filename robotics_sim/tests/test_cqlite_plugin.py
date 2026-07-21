from __future__ import annotations

import json
from pathlib import Path

import pytest

from algorithms.cqlite.plugin import CQLITE_COORDINATOR, CQLitePlugin
from experiments.run_cqlite_experiments import run_matrix
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.plugins import PluginCapability
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.simulation.config import config_from_sim_payload, config_to_sim_payload
from robotics_sim.simulation.plugin_loader import list_coordination_plugin_names, load_coordination_plugin


ROOT = Path(__file__).resolve().parents[2]
PRESETS = (
    ROOT / "examples" / "cqlite_house_3.sim",
    ROOT / "examples" / "cqlite_bookstore_3.sim",
    ROOT / "examples" / "cqlite_bookstore_6.sim",
)


def _robot(robot_id: int, x: float, y: float, current_target=None) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        safety_radius=0.22,
        sensor_range=15.0,
        vision_model="LiDAR",
        current_target=current_target,
    )


def _candidate(x: float, y: float, gain: float = 1.0) -> ExplorationCandidate:
    return ExplorationCandidate(target=(x, y), information_gain=gain)


def _parameters(**overrides):
    values = {
        "grid_resolution": 0.5,
        "min_frontier_travel_distance": 0.1,
        "target_exclusion_radius": 0.25,
        "cqlite_alpha": 0.6,
        "cqlite_gamma": 0.95,
        "cqlite_step_cost": 2.0,
        "cqlite_overlap_radius": 1.0,
        "cqlite_communication_range": 50.0,
        "cqlite_nominal_speed": 0.5,
        "cqlite_rho": 2.0,
        "cqlite_sigma": 0.01,
        "cqlite_information_weight": 0.0,
    }
    values.update(overrides)
    return values


def test_plugin_is_discoverable_and_stays_outside_simulator_package() -> None:
    assert CQLITE_COORDINATOR in list_coordination_plugin_names()
    plugin = load_coordination_plugin(CQLITE_COORDINATOR)
    assert PluginCapability.COORDINATION in plugin.metadata.capabilities
    assert PluginCapability.TASK_ALLOCATION in plugin.metadata.capabilities
    assert PluginCapability.PATH_PLANNING not in plugin.metadata.capabilities
    source = (ROOT / "algorithms" / "cqlite" / "plugin.py").read_text(encoding="utf-8")
    assert "import robotics_sim" not in source
    assert "from robotics_sim" not in source


def test_first_q_update_matches_paper_equations_one_and_seven() -> None:
    plugin = CQLitePlugin()
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={0: (_candidate(1.0, 0.0),)},
        parameters=_parameters(),
        time_s=1.0,
    )

    result = plugin.assign(request)

    # reward = lambda - Q + rho*(1-overlap) + sigma*rc
    #        = 2 - 0 + 2*(1-0) + .01*50 = 4.5
    # Q' = .4*0 + .6*(4.5 + .95*0) = 2.7
    command = result.commands[0]
    assert command.metadata["cqlite_reward"] == pytest.approx(4.5)
    assert command.metadata["cqlite_q_after"] == pytest.approx(2.7)


def test_travel_time_priority_prefers_near_frontier_when_q_values_match() -> None:
    plugin = CQLitePlugin()
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={0: (_candidate(1.0, 0.0), _candidate(6.0, 0.0))},
        parameters=_parameters(),
        time_s=1.0,
    )

    result = plugin.assign(request)

    assert result.targets == ((1.0, 0.0),)
    assert result.commands[0].metadata["cqlite_travel_time"] == pytest.approx(2.0)


def test_voronoi_allocation_gives_distinct_local_targets_to_team() -> None:
    plugin = CQLitePlugin()
    robots = (_robot(0, 0.0, 0.0), _robot(1, 4.0, 0.0))
    candidates = (_candidate(1.0, 0.0), _candidate(5.0, 0.0))
    request = CoordinationRequest(
        robot_states=robots,
        robots_to_assign=(0, 1),
        proposals_by_robot={0: candidates, 1: candidates},
        parameters=_parameters(),
        time_s=1.0,
    )

    result = plugin.assign(request)

    assert result.targets == ((1.0, 0.0), (5.0, 0.0))
    assert len(set(result.targets)) == 2
    assert all(command.metadata["cqlite_in_voronoi_region"] for command in result.commands)


def test_lite_messages_only_follow_communication_graph_edges() -> None:
    plugin = CQLitePlugin()
    robots = (_robot(0, 0.0, 0.0), _robot(1, 2.0, 0.0), _robot(2, 20.0, 0.0))
    request = CoordinationRequest(
        robot_states=robots,
        robots_to_assign=(0, 1, 2),
        proposals_by_robot={
            0: (_candidate(-1.0, 0.0),),
            1: (_candidate(3.0, 0.0),),
            2: (_candidate(21.0, 0.0),),
        },
        parameters=_parameters(cqlite_communication_range=3.0),
        time_s=1.0,
    )

    result = plugin.assign(request)

    assert result.debug["network"]["undirected_edge_count"] == 1
    assert result.debug["network"]["neighbors_by_robot"] == {"0": [1], "1": [0], "2": []}
    assert result.debug["communication"]["messages_this_decision"] == 2
    assert result.debug["communication"]["payload_bytes_this_decision"] == 80


def test_completed_frontier_is_learned_and_not_selected_again() -> None:
    plugin = CQLitePlugin()
    first = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={0: (_candidate(1.0, 0.0),)},
        parameters=_parameters(),
        time_s=1.0,
    )
    assert plugin.assign(first).targets == ((1.0, 0.0),)

    second = CoordinationRequest(
        robot_states=(_robot(0, 1.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={0: (_candidate(1.0, 0.0), _candidate(2.0, 0.0))},
        parameters=_parameters(),
        time_s=2.0,
    )
    result = plugin.assign(second)

    assert result.targets == ((2.0, 0.0),)
    assert result.debug["per_robot"]["0"]["rejected"]["already_explored"] == 1
    assert result.debug["q_updates"]["0"] == 3


def test_paper_presets_pin_parameters_and_round_trip_coordination_settings() -> None:
    expected_counts = [3, 3, 6]
    for preset, expected_count in zip(PRESETS, expected_counts):
        payload = json.loads(preset.read_text(encoding="utf-8"))
        assert payload["multi_robot"]["robot_count"] == expected_count
        assert payload["sensor"]["range"] == 15.0
        assert payload["coordination"]["strategy"] == CQLITE_COORDINATOR
        params = payload["coordination"]["parameters"]
        assert params["cqlite_alpha"] == 0.6
        assert params["cqlite_gamma"] == 0.95
        assert params["cqlite_step_cost"] == 2.0
        assert params["cqlite_overlap_radius"] == 1.0
        config = config_from_sim_payload(payload)
        assert config.coordination_parameters == params
        assert config_to_sim_payload(config)["coordination"]["parameters"] == params


def test_native_proxy_matrix_is_deterministic_and_separates_published_results() -> None:
    first = run_matrix(trial_count=1, seed_base=23)
    second = run_matrix(trial_count=1, seed_base=23)

    assert first == second
    assert first["fidelity"] == "decision_level_native_proxy_not_gazebo_slam"
    assert set(first["native_results"]) == {"house_3", "bookstore_3", "bookstore_6"}
    assert first["published_table_i_reference"]["bookstore_6"]["CQLite"]["communication_mb"] == [0.2, 0.01]
    for scenario in first["native_results"].values():
        assert scenario["aggregate"]["exploration_percent"]["mean"] == pytest.approx(100.0)
