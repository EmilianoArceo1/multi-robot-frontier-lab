from pathlib import Path
from types import SimpleNamespace

import pytest

from algorithms.marvel.backend import (
    NUM_ANGLES_BIN,
    NUM_HEADING_CANDIDATES,
    MarvelInferenceBackend,
)
from algorithms.marvel.plugin import MARVEL_COORDINATOR, MarvelPlugin
from algorithms.marvel.runtime import (
    MARVEL_WEIGHTS_ENV,
    MarvelRuntimeConfiguration,
)
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_sim.simulation.approach_profiles import (
    APPROACH_CATEGORY_OPTIONS,
    approach_profile_for_task_assignment,
)
from robotics_sim.simulation.mapping_architecture import MappingArchitecture
from robotics_sim.simulation.plugin_loader import list_coordination_plugin_names


def test_marvel_is_discoverable_without_importing_or_loading_torch():
    assert MARVEL_COORDINATOR in list_coordination_plugin_names()


def test_marvel_profile_preserves_ctde_shared_map_assumptions():
    profile = approach_profile_for_task_assignment(MARVEL_COORDINATOR)

    assert profile.architecture_label == "Decentralized execution (CTDE)"
    assert profile.mapping_architecture is MappingArchitecture.CENTRALIZED
    assert tuple(badge.label for badge in profile.badges) == (
        "Learning-based",
        "Goal-level",
        "Unconstrained",
    )


def test_approach_taxonomy_has_three_binary_categories():
    assert APPROACH_CATEGORY_OPTIONS == {
        "Paradigm": ("Conventional", "Learning-based"),
        "Decision": ("Goal-level", "Action-level"),
        "Communication": ("Unconstrained", "Constrained"),
    }


def test_marvel_weight_path_is_interchangeable_through_environment(
    monkeypatch,
    tmp_path: Path,
):
    checkpoint = tmp_path / "official.pth"
    monkeypatch.setenv(MARVEL_WEIGHTS_ENV, str(checkpoint))

    runtime = MarvelRuntimeConfiguration.from_environment()

    assert runtime.checkpoint_path == checkpoint
    assert "checkpoint not found" in str(runtime.readiness_error())


def test_missing_official_weights_hold_instead_of_using_a_fallback(monkeypatch):
    monkeypatch.setenv(MARVEL_WEIGHTS_ENV, "missing-marvel-checkpoint.pth")
    plugin = MarvelPlugin()
    request = CoordinationRequest(
        robot_states=(
            RobotCoordinationState(
                robot_id=0,
                xy=(0.0, 0.0),
                safety_radius=0.3,
                sensor_range=10.0,
                vision_model="Camera / FoV",
            ),
        ),
    )

    result = plugin.assign(request)

    assert result.strategy == MARVEL_COORDINATOR
    assert result.assignments[0].status == "HOLD"
    assert "checkpoint not found" in result.assignments[0].reason
    assert result.debug["ready"] is False


def _known_square_world() -> WorldSnapshot:
    resolution = 0.5
    bounds = (-12.0, 12.0, -12.0, 12.0)
    explored = []
    for row in range(8, 40):
        for col in range(8, 40):
            explored.append(
                (
                    bounds[0] + (col + 0.5) * resolution,
                    bounds[2] + (row + 0.5) * resolution,
                )
            )
    return WorldSnapshot(
        explored_points=tuple(explored),
        bounds=bounds,
        resolution=resolution,
        metadata={"mapping_architecture": "centralized"},
    )


def test_marvel_backend_builds_authors_observation_and_decodes_policy_action():
    torch = pytest.importorskip("torch")
    captured_shapes = []

    class FakePolicy:
        def __call__(self, *observation):
            captured_shapes.extend(tuple(tensor.shape) for tensor in observation)
            edge_count = observation[4].shape[1]
            # The backend sorts policy logits and must skip the masked self
            # action before returning the next highest-ranked graph action.
            return torch.arange(
                edge_count * NUM_HEADING_CANDIDATES,
                dtype=torch.float32,
            ).reshape(1, -1)

    robot = RobotCoordinationState(
        robot_id=0,
        xy=(0.0, 0.0),
        safety_radius=0.35,
        sensor_range=10.0,
        vision_model="Camera / FoV",
        theta=0.0,
    )
    request = CoordinationRequest(
        robot_states=(robot,),
        robots_to_assign=(0,),
        world=_known_square_world(),
        parameters={
            "target_exclusion_radius": 1.5,
            "min_frontier_travel_distance": 0.75,
            "marvel_fov_degrees": 120.0,
        },
        shared={"mapping_architecture": "centralized"},
    )

    result = MarvelInferenceBackend().assign(request, FakePolicy())

    assert result.assignments[0].status == "ASSIGNED"
    assert result.targets[0] is not None
    assert result.commands[0].heading_rad is not None
    assert result.debug["ready"] is True
    assert captured_shapes[0][0] == 1
    assert captured_shapes[0][2] == 6
    assert captured_shapes[6][2] == NUM_ANGLES_BIN
    assert captured_shapes[7][2] == NUM_ANGLES_BIN
    assert captured_shapes[8][2:] == (
        NUM_HEADING_CANDIDATES,
        NUM_ANGLES_BIN,
    )


def test_marvel_backend_rejects_per_robot_maps_not_supported_by_paper():
    robot = RobotCoordinationState(
        robot_id=0,
        xy=(0.0, 0.0),
        safety_radius=0.35,
        sensor_range=10.0,
        vision_model="Camera / FoV",
    )
    request = CoordinationRequest(
        robot_states=(robot,),
        robots_to_assign=(0,),
        world=_known_square_world(),
        shared={"mapping_architecture": "decentralized_slam"},
    )

    result = MarvelInferenceBackend().assign(
        request,
        SimpleNamespace(),
    )

    assert result.assignments[0].status == "HOLD"
    assert "shared centralized belief map" in result.assignments[0].reason
