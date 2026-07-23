from pathlib import Path
from types import SimpleNamespace

import pytest

from algorithms.marvel.backend import (
    FRONTIER_CELL_SIZE,
    NODE_RESOLUTION,
    NUM_ANGLES_BIN,
    NUM_HEADING_CANDIDATES,
    PAPER_SENSOR_RANGE,
    SCALED_SPATIAL_MODE,
    UPDATING_MAP_SIZE,
    MarvelSpatialConfiguration,
    MarvelInferenceBackend,
)
from algorithms.marvel.plugin import MARVEL_COORDINATOR, MarvelPlugin
from algorithms.marvel_scaled.plugin import (
    MARVEL_SCALED_COORDINATOR,
    MarvelScaledPlugin,
)
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
from robotics_sim.simulation.algorithm_pipeline_profiles import (
    task_assignment_pipeline_profile,
)
from robotics_sim.simulation.mapping_architecture import MappingArchitecture
from robotics_sim.simulation.plugin_loader import list_coordination_plugin_names
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.config import (
    load_sim_file,
    normalized_robot_start_configs,
    sensor_visible_polygon_world,
)


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_PRESET = ROOT / "examples" / "marvel_original_scale.sim"


def test_marvel_is_discoverable_without_importing_or_loading_torch():
    available = list_coordination_plugin_names()
    assert MARVEL_COORDINATOR in available
    assert MARVEL_SCALED_COORDINATOR in available


def test_marvel_profile_preserves_ctde_shared_map_assumptions():
    profile = approach_profile_for_task_assignment(MARVEL_COORDINATOR)

    assert profile.architecture_label == "Decentralized execution (CTDE)"
    assert profile.mapping_architecture is MappingArchitecture.CENTRALIZED
    assert tuple(badge.label for badge in profile.badges) == (
        "Learning-based",
        "Goal-level",
        "Unconstrained",
    )
    pipeline = task_assignment_pipeline_profile(MARVEL_COORDINATOR)
    assert pipeline is not None
    assert pipeline.default_vision_model == "Camera / FoV"
    assert pipeline.default_sensor_range == PAPER_SENSOR_RANGE
    assert pipeline.default_camera_fov_degrees == 120.0
    assert pipeline.lock_vision_model is True
    assert pipeline.bootstrap_panorama is True

    scaled = task_assignment_pipeline_profile(MARVEL_SCALED_COORDINATOR)
    assert scaled is not None
    assert scaled.default_vision_model == "Camera / FoV"
    assert scaled.default_sensor_range == 3.0
    assert scaled.default_camera_fov_degrees == 120.0
    assert scaled.bootstrap_panorama is True

    scaled_approach = approach_profile_for_task_assignment(
        MARVEL_SCALED_COORDINATOR
    )
    assert scaled_approach.mapping_architecture is MappingArchitecture.CENTRALIZED
    assert "scaled" in scaled_approach.architecture_label.lower()


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


def test_scaled_plugin_missing_weights_reports_its_own_selector_name(monkeypatch):
    monkeypatch.setenv(MARVEL_WEIGHTS_ENV, "missing-marvel-checkpoint.pth")
    plugin = MarvelScaledPlugin()
    request = CoordinationRequest(
        robot_states=(
            RobotCoordinationState(
                robot_id=0,
                xy=(0.0, 0.0),
                safety_radius=0.3,
                sensor_range=3.0,
                vision_model="Camera / FoV",
            ),
        ),
    )

    result = plugin.assign(request)

    assert result.strategy == MARVEL_SCALED_COORDINATOR
    assert result.assignments[0].status == "HOLD"
    assert "checkpoint not found" in result.assignments[0].reason


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


def test_marvel_backend_rejects_non_directional_sensor_model():
    robot = RobotCoordinationState(
        robot_id=0,
        xy=(0.0, 0.0),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="LiDAR",
    )
    request = CoordinationRequest(
        robot_states=(robot,),
        robots_to_assign=(0,),
        world=_known_square_world(),
        shared={"mapping_architecture": "centralized"},
    )

    result = MarvelInferenceBackend().assign(request, SimpleNamespace())

    assert result.assignments[0].status == "HOLD"
    assert "requires a directional Camera / FoV observation" in (
        result.assignments[0].reason
    )


def test_marvel_backend_accepts_adjusted_range_and_fov():
    torch = pytest.importorskip("torch")

    class FakePolicy:
        def __call__(self, *observation):
            edge_count = observation[4].shape[1]
            return torch.arange(
                edge_count * NUM_HEADING_CANDIDATES,
                dtype=torch.float32,
            ).reshape(1, -1)

    robot = RobotCoordinationState(
        robot_id=0,
        xy=(0.0, 0.0),
        safety_radius=0.35,
        sensor_range=6.0,
        vision_model="Camera / FoV",
    )
    request = CoordinationRequest(
        robot_states=(robot,),
        robots_to_assign=(0,),
        world=_known_square_world(),
        parameters={
            "target_exclusion_radius": 1.5,
            "min_frontier_travel_distance": 0.75,
            "marvel_fov_degrees": 90.0,
        },
        shared={"mapping_architecture": "centralized"},
    )

    result = MarvelInferenceBackend().assign(request, FakePolicy())

    assert result.assignments[0].status == "ASSIGNED"
    observation_debug = result.debug["observation"]
    assert observation_debug["camera_fov_degrees"] == 90.0
    assert observation_debug["sensor_ranges_m"] == (6.0,)
    assert observation_debug["paper_defaults"] == {
        "sensor_range_m": 10.0,
        "camera_fov_degrees": 120.0,
    }


def test_scaled_marvel_preserves_paper_spatial_ratios_at_three_meters():
    torch = pytest.importorskip("torch")

    class FakePolicy:
        def __call__(self, *observation):
            edge_count = observation[4].shape[1]
            return torch.arange(
                edge_count * NUM_HEADING_CANDIDATES,
                dtype=torch.float32,
            ).reshape(1, -1)

    robot = RobotCoordinationState(
        robot_id=0,
        xy=(0.0, 0.0),
        safety_radius=0.35,
        sensor_range=3.0,
        vision_model="Camera / FoV",
    )
    request = CoordinationRequest(
        robot_states=(robot,),
        robots_to_assign=(0,),
        world=_known_square_world(),
        parameters={
            "target_exclusion_radius": 0.5,
            "min_frontier_travel_distance": 0.5,
            "marvel_fov_degrees": 120.0,
        },
        shared={"mapping_architecture": "centralized"},
    )

    result = MarvelInferenceBackend(
        strategy_name=MARVEL_SCALED_COORDINATOR,
        spatial_mode=SCALED_SPATIAL_MODE,
    ).assign(request, FakePolicy())

    assert result.strategy == MARVEL_SCALED_COORDINATOR
    assert result.assignments[0].status == "ASSIGNED"
    observation = result.debug["observation"]
    assert observation["spatial_mode"] == SCALED_SPATIAL_MODE
    assert observation["scale_factor"] == pytest.approx(0.3)
    assert observation["node_resolution_m"] == pytest.approx(1.2)
    assert observation["updating_map_size_m"] == pytest.approx(18.0)
    # The synthetic belief uses a 0.5 m grid, so the requested 0.24 m
    # frontier voxel is clamped to one representable host cell.
    assert observation["frontier_cell_size_m"] == pytest.approx(0.5)


def test_spatial_scale_retains_dimensionless_paper_geometry():
    scaled = MarvelSpatialConfiguration.scaled(
        sensor_range_m=3.0,
        grid_resolution_m=0.2,
    )

    assert scaled.node_resolution_m / scaled.reference_sensor_range_m == pytest.approx(
        NODE_RESOLUTION / PAPER_SENSOR_RANGE
    )
    assert (
        scaled.updating_map_size_m / scaled.reference_sensor_range_m
        == pytest.approx(
        UPDATING_MAP_SIZE / PAPER_SENSOR_RANGE
        )
    )
    assert (
        scaled.frontier_cell_size_m / scaled.reference_sensor_range_m
        == pytest.approx(
        FRONTIER_CELL_SIZE / PAPER_SENSOR_RANGE
        )
    )


def test_original_marvel_preset_loads_published_sensor_and_graph_scale():
    config = load_sim_file(str(ORIGINAL_PRESET))
    starts = normalized_robot_start_configs(config)

    assert config.coordinator_type == MARVEL_COORDINATOR
    assert config.agent_mode == "Multiple Robot Mode"
    assert config.vision_model == "Camera / FoV"
    assert config.vision == PAPER_SENSOR_RANGE
    assert config.camera_fov_degrees == 120.0
    assert config.grid_resolution == 0.4
    assert len(starts) == 4
    assert all(start.vision == PAPER_SENSOR_RANGE for start in starts)
    assert all(
        start.x % NODE_RESOLUTION == 0.0
        and start.y % NODE_RESOLUTION == 0.0
        for start in starts
    )
    published = config.experiment["published_parameters"]
    assert published["training_environment_width_m"] == 90.0
    assert published["node_resolution_m"] == NODE_RESOLUTION
    assert "compact" in config.experiment["host_assumptions"]["geometry"]


def test_marvel_starting_belief_bootstraps_panorama_then_restores_camera_fov():
    robot_1 = SimpleNamespace(name="R1")
    robot_2 = SimpleNamespace(name="R2")
    observations = []
    obstacle_updates = []
    forced_free = []
    messages = []
    fake = SimpleNamespace(
        config=SimpleNamespace(
            coordinator_type=MARVEL_SCALED_COORDINATOR,
            camera_fov_degrees=120.0,
        ),
        robots=[robot_1, robot_2],
        robot=robot_1,
        record_explored_area=lambda force, robot_index: observations.append(
            (
                fake.robot.name,
                robot_index,
                fake.config.camera_fov_degrees,
                force,
            )
        ),
        update_sensed_obstacles=lambda force_status: obstacle_updates.append(
            (fake.robot.name, fake.config.camera_fov_degrees, force_status)
        ),
        force_robot_pose_free_in_belief=forced_free.append,
        log_console_message=messages.append,
    )

    SimulationControllerMixin.initialize_multi_robot_starting_belief(fake)

    assert observations == [
        ("R1", 0, 360.0, True),
        ("R2", 1, 360.0, True),
        ("R1", 0, 120.0, True),
        ("R2", 1, 120.0, True),
    ]
    assert obstacle_updates[0] == ("R1", 360.0, False)
    assert obstacle_updates[1] == ("R2", 360.0, False)
    assert forced_free == [0, 1, 0, 1]
    assert fake.config.camera_fov_degrees == 120.0
    assert fake.robot is robot_1
    assert "360° observation at each robot start" in messages[0]


def test_sensor_polygon_accepts_marvel_paper_fov():
    polygon = sensor_visible_polygon_world(
        origin=(0.0, 0.0),
        theta=0.0,
        vision=10.0,
        vision_model="Camera / FoV",
        obstacles=[],
        ray_count=3,
        camera_fov_degrees=120.0,
    )

    assert polygon[0] == (0.0, 0.0)
    assert polygon[1][0] == pytest.approx(5.0)
    assert polygon[1][1] == pytest.approx(-10.0 * 3**0.5 / 2.0)
    assert polygon[-1][0] == pytest.approx(5.0)
    assert polygon[-1][1] == pytest.approx(10.0 * 3**0.5 / 2.0)
