from pathlib import Path

from algorithms.marvel.plugin import MARVEL_COORDINATOR, MarvelPlugin
from algorithms.marvel.runtime import (
    MARVEL_WEIGHTS_ENV,
    MarvelRuntimeConfiguration,
)
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState
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
