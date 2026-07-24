"""Contract tests for the smoke_v0 map corpus and its pure validator
(robotics_sim/environment/sim_map_validator.py).

Same pattern as test_connected_frontier_components.py / test_marvel_
architecture.py: real files, real parser, no engine, no Qt, no MainWindow.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from robotics_sim.environment.sim_map_validator import (
    MIN_CONNECTED_FREE_FRACTION,
    MapValidationReport,
    validate_manifest,
    validate_sim_map,
    validate_sim_map_file,
)
from robotics_sim.simulation.config import RobotStartConfig, SimulationConfig

ROOT = Path(__file__).resolve().parents[2]
MAPS_DIR = ROOT / "experiments" / "maps" / "smoke_v0"
MANIFEST_PATH = MAPS_DIR / "manifest.json"

EXPECTED_FAMILIES = {
    "smoke_v0_01_open": "open",
    "smoke_v0_02_office": "office",
    "smoke_v0_03_corridors": "corridors",
    "smoke_v0_04_loops": "loops",
    "smoke_v0_05_bottleneck": "bottleneck",
    "smoke_v0_06_mixed": "mixed",
}


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def reports(manifest) -> dict[str, MapValidationReport]:
    return validate_manifest(manifest, maps_dir=str(MAPS_DIR))


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


def test_manifest_has_required_top_level_fields(manifest):
    assert manifest["corpus_id"] == "smoke_v0"
    assert manifest["schema_version"] == 1
    assert manifest["split"] == "smoke"
    assert manifest["base_robot_count"] == 4
    assert len(manifest["maps"]) == 6


def test_manifest_entries_have_required_per_map_fields(manifest):
    required = {
        "map_id", "filename", "family", "difficulty", "geometry_seed",
        "robot_start_positions", "fire_scenarios", "tags", "expected_properties",
    }
    for entry in manifest["maps"]:
        missing = required - entry.keys()
        assert not missing, f"{entry.get('map_id')} is missing fields: {missing}"
        assert entry["difficulty"] == "smoke"
        assert entry["geometry_seed"] is None
        assert len(entry["fire_scenarios"]) >= 2


def test_manifest_map_ids_are_unique(manifest):
    map_ids = [entry["map_id"] for entry in manifest["maps"]]
    assert len(map_ids) == len(set(map_ids))


def test_manifest_filenames_are_unique(manifest):
    filenames = [entry["filename"] for entry in manifest["maps"]]
    assert len(filenames) == len(set(filenames))


def test_manifest_families_are_unique(manifest):
    families = [entry["family"] for entry in manifest["maps"]]
    assert len(families) == len(set(families))


def test_manifest_families_match_expected(manifest):
    families = {entry["map_id"]: entry["family"] for entry in manifest["maps"]}
    assert families == EXPECTED_FAMILIES


# ---------------------------------------------------------------------------
# The six real .sim files load and validate
# ---------------------------------------------------------------------------


def test_all_six_sim_files_exist_and_load(manifest):
    from robotics_sim.simulation.config import load_sim_file

    for entry in manifest["maps"]:
        path = MAPS_DIR / entry["filename"]
        assert path.is_file(), f"missing {path}"
        config = load_sim_file(str(path))
        assert isinstance(config, SimulationConfig)


def test_all_six_maps_pass_validation(reports):
    for map_id, report in reports.items():
        assert report.valid, f"{map_id} failed: {report.errors}"


def test_all_six_maps_share_bounds_and_resolution(reports):
    widths = {report.width for report in reports.values()}
    heights = {report.height for report in reports.values()}
    resolutions = {report.resolution for report in reports.values()}
    assert len(widths) == 1
    assert len(heights) == 1
    assert len(resolutions) == 1


def test_all_six_maps_meet_connectivity_threshold(reports):
    for map_id, report in reports.items():
        assert report.connected_free_fraction >= MIN_CONNECTED_FREE_FRACTION, map_id


def test_all_six_maps_have_valid_start_positions(reports):
    for map_id, report in reports.items():
        assert report.start_positions_valid, map_id


def test_all_six_maps_have_valid_fire_scenarios(reports):
    for map_id, report in reports.items():
        assert report.fire_positions_valid, map_id


def test_fires_are_outside_initial_region_observation(manifest):
    """Every fire in the manifest must fail to be "visible" from the initial
    region by itself -- i.e. removing it from its scenario should not be
    what makes the map valid. Re-validate each fire alone against its map
    and confirm fire_positions_valid stays True (the dedicated failure mode
    is exercised directly in test_fire_inside_obstacle_fails /
    test_fire_visible_from_start_fails below with synthetic maps)."""
    from robotics_sim.simulation.config import load_sim_file

    for entry in manifest["maps"]:
        config = load_sim_file(str(MAPS_DIR / entry["filename"]))
        for scenario in entry["fire_scenarios"]:
            fires = [(fire["x"], fire["y"]) for fire in scenario["fires"]]
            report = validate_sim_map(
                map_id=entry["map_id"], config=config, fire_scenarios=[fires]
            )
            assert report.fire_positions_valid, (entry["map_id"], scenario["scenario_id"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_validation_is_deterministic(manifest):
    entry = manifest["maps"][0]
    path = str(MAPS_DIR / entry["filename"])
    fire_scenarios = [
        [(fire["x"], fire["y"]) for fire in scenario["fires"]]
        for scenario in entry["fire_scenarios"]
    ]

    report_a = validate_sim_map_file(
        path, map_id=entry["map_id"], fire_scenarios=fire_scenarios, expected_robot_count=4
    )
    report_b = validate_sim_map_file(
        path, map_id=entry["map_id"], fire_scenarios=fire_scenarios, expected_robot_count=4
    )
    assert report_a == report_b


def test_manifest_map_order_is_stable(manifest):
    map_ids = [entry["map_id"] for entry in manifest["maps"]]
    assert map_ids == list(EXPECTED_FAMILIES.keys())


# ---------------------------------------------------------------------------
# Synthetic failure-mode contract tests (pure SimulationConfig, no files)
# ---------------------------------------------------------------------------

_BASE_ROBOTS = [
    RobotStartConfig(x=-1.0, y=-1.0, theta=0.0, v=0.0),
    RobotStartConfig(x=1.0, y=-1.0, theta=0.0, v=0.0),
    RobotStartConfig(x=-1.0, y=1.0, theta=0.0, v=0.0),
    RobotStartConfig(x=1.0, y=1.0, theta=0.0, v=0.0),
]


def _synthetic_config(obstacles, robots=None) -> SimulationConfig:
    return SimulationConfig(
        grid_resolution=0.5,
        obstacles=obstacles,
        robot_count=4,
        same_robot_configuration=False,
        robots=list(robots) if robots is not None else list(_BASE_ROBOTS),
    )


def test_disconnected_free_space_fails():
    # A full-height wall across the middle of the world, with no gap, splits
    # free space into two disconnected halves -- only the half containing
    # the robot starts is reachable.
    config = _synthetic_config(obstacles=[(4.0, -8.0, 0.5, 16.0)])
    report = validate_sim_map(map_id="disconnected", config=config, expected_robot_count=4)
    assert not report.valid
    assert report.connected_free_fraction < MIN_CONNECTED_FREE_FRACTION
    assert any("connected_free_fraction" in e for e in report.errors)


def test_start_inside_obstacle_fails():
    robots = [
        RobotStartConfig(x=0.0, y=0.0, theta=0.0, v=0.0),  # inside the obstacle below
        RobotStartConfig(x=3.0, y=-1.0, theta=0.0, v=0.0),
        RobotStartConfig(x=-3.0, y=1.0, theta=0.0, v=0.0),
        RobotStartConfig(x=3.0, y=1.0, theta=0.0, v=0.0),
    ]
    config = _synthetic_config(obstacles=[(-1.0, -1.0, 2.0, 2.0)], robots=robots)
    report = validate_sim_map(map_id="start_in_obstacle", config=config, expected_robot_count=4)
    assert not report.valid
    assert not report.start_positions_valid
    assert any("collides with obstacle" in e for e in report.errors)


def test_fire_inside_obstacle_fails():
    config = _synthetic_config(obstacles=[(-1.0, -1.0, 2.0, 2.0)])
    report = validate_sim_map(
        map_id="fire_in_obstacle",
        config=config,
        fire_scenarios=[[(0.0, 0.0)]],
        expected_robot_count=4,
    )
    assert not report.valid
    assert not report.fire_positions_valid
    assert any("is inside obstacle" in e for e in report.errors)


def test_fire_visible_from_start_fails():
    # No obstacles, fire placed well within every robot's sensor range of a
    # start -- nothing blocks line of sight, so it must be flagged visible.
    config = _synthetic_config(obstacles=[])
    fire = (-1.0 + config.vision * 0.5, -1.0)
    report = validate_sim_map(
        map_id="fire_visible",
        config=config,
        fire_scenarios=[[fire]],
        expected_robot_count=4,
    )
    assert not report.fire_positions_valid
    assert any("is visible from initial region" in e for e in report.errors)


def test_geometry_outside_bounds_fails():
    from robotics_sim.simulation.config import WORLD_X_MAX

    config = _synthetic_config(obstacles=[(WORLD_X_MAX - 0.5, -1.0, 5.0, 2.0)])
    report = validate_sim_map(map_id="geometry_oob", config=config, expected_robot_count=4)
    assert not report.valid
    assert any("extends outside world bounds" in e for e in report.errors)


def test_robots_overlapping_at_start_fails():
    robots = [
        RobotStartConfig(x=0.0, y=0.0, theta=0.0, v=0.0),
        RobotStartConfig(x=0.05, y=0.0, theta=0.0, v=0.0),  # overlaps robot 0
        RobotStartConfig(x=-3.0, y=1.0, theta=0.0, v=0.0),
        RobotStartConfig(x=3.0, y=1.0, theta=0.0, v=0.0),
    ]
    config = _synthetic_config(obstacles=[], robots=robots)
    report = validate_sim_map(map_id="overlap", config=config, expected_robot_count=4)
    assert not report.valid
    assert not report.start_positions_valid
    assert any("overlap" in e for e in report.errors)


def test_wrong_robot_count_fails():
    config = _synthetic_config(obstacles=[], robots=_BASE_ROBOTS[:2])
    config.robot_count = 2
    report = validate_sim_map(map_id="wrong_count", config=config, expected_robot_count=4)
    assert not report.valid
    assert not report.start_positions_valid
    assert any("expected 4 robot start positions" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Isolation: the validator module must not import Qt, the app package, the
# runtime engine, learning, or algorithms directly.
# ---------------------------------------------------------------------------


def test_validator_module_has_no_forbidden_imports():
    source_path = ROOT / "robotics_sim" / "environment" / "sim_map_validator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    forbidden_roots = {"PySide6", "PyQt5", "PyQt6", "algorithms", "learning"}
    assert not (imported_roots & forbidden_roots), imported_roots

    forbidden_submodules = {"robotics_sim.app", "robotics_sim.simulation.engine", "robotics_sim.learning"}
    seen_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            seen_modules.add(node.module)
    assert not (seen_modules & forbidden_submodules), seen_modules
