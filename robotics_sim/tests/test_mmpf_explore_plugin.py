from __future__ import annotations

from pathlib import Path

from algorithms.mmpf_explore.plugin import MMPF_EXPLORE_COORDINATOR
from robotics_interfaces import (
    CandidateProposal,
    CoordinationRequest,
    PluginCapability,
    RobotCoordinationState,
)
from robotics_sim.simulation.plugin_loader import load_coordination_plugin


def _state(robot_id: int, x: float, y: float) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
    )


def _proposal(
    robot_id: int,
    target: tuple[float, float],
    score: float,
    information_gain: float = 0.0,
    travel_cost: float = 0.0,
) -> CandidateProposal:
    return CandidateProposal(
        robot_id=robot_id,
        target=target,
        score=score,
        information_gain=information_gain,
        travel_cost=travel_cost,
        reason="unit test proposal",
    )


def test_mmpf_explore_plugin_is_discoverable():
    plugin = load_coordination_plugin(MMPF_EXPLORE_COORDINATOR)

    assert plugin.metadata.name == MMPF_EXPLORE_COORDINATOR
    assert PluginCapability.COORDINATION in plugin.metadata.capabilities
    assert PluginCapability.TASK_ALLOCATION in plugin.metadata.capabilities


def test_mmpf_explore_plugin_does_not_import_simulator_runtime():
    source = Path("algorithms/mmpf_explore/plugin.py").read_text(encoding="utf-8")

    assert "from robotics_sim" not in source
    assert "import robotics_sim" not in source
    assert "Qt" not in source
    assert "MainWindow" not in source
    assert "engine" not in source


def test_mmpf_explore_assigns_unique_targets_from_proposals():
    plugin = load_coordination_plugin(MMPF_EXPLORE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            _state(0, 0.0, 0.0),
            _state(1, 1.0, 0.0),
        ),
        robots_to_assign=(0, 1),
        proposals_by_robot={
            0: (
                _proposal(0, (2.0, 2.0), score=10.0, information_gain=4.0),
                _proposal(0, (4.0, 4.0), score=2.0),
            ),
            1: (
                _proposal(1, (2.0, 2.0), score=12.0),
                _proposal(1, (3.0, 3.0), score=8.0),
            ),
        },
    )

    result = plugin.assign(request)

    assert result.strategy == MMPF_EXPLORE_COORDINATOR
    assert result.targets == ((2.0, 2.0), (3.0, 3.0))
    assert tuple(assignment.status for assignment in result.assignments) == (
        "ASSIGNED",
        "ASSIGNED",
    )
    assert result.assignments[0].proposal is not None
    assert result.assignments[1].proposal is not None


def test_mmpf_explore_respects_existing_target_for_unselected_robot():
    plugin = load_coordination_plugin(MMPF_EXPLORE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            _state(0, 0.0, 0.0),
            _state(1, 1.0, 0.0),
        ),
        robots_to_assign=(0,),
        existing_targets_by_robot={1: (8.0, 8.0)},
        proposals_by_robot={
            0: (
                _proposal(0, (2.0, 2.0), score=10.0),
            ),
        },
    )

    result = plugin.assign(request)

    assert result.targets == ((2.0, 2.0), (8.0, 8.0))
    assert result.assignments[0].status == "ASSIGNED"
    assert result.assignments[1].status == "ASSIGNED"
    assert result.assignments[1].reason == "kept existing target"


def test_mmpf_explore_holds_when_no_feasible_proposal_exists():
    plugin = load_coordination_plugin(MMPF_EXPLORE_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(
            _state(0, 0.0, 0.0),
        ),
        robots_to_assign=(0,),
        blocked_targets_by_robot={0: ((2.0, 2.0),)},
        proposals_by_robot={
            0: (
                _proposal(0, (2.0, 2.0), score=10.0),
            ),
        },
    )

    result = plugin.assign(request)

    assert result.targets == (None,)
    assert result.assignments[0].status == "HOLD"
    assert "no feasible proposal" in result.reasons[0]
