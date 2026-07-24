"""Contract tests for the headless smoke episode runner
(robotics_sim/simulation/headless_episode_runner.py) and its CLI
(experiments/run_smoke_episode.py).

Termination-logic tests use a small fake HeadlessSimulation to isolate the
episode loop from the real engine (explicitly allowed by the task spec).
At least one test below drives the real engine end to end.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from robotics_sim.simulation.config import load_sim_file
from robotics_sim.simulation.headless_episode_runner import (
    DEFAULT_FIRE_DETECTION_THRESHOLD,
    TERMINATION_ALL_FIRES_DETECTED,
    TERMINATION_COVERAGE_REACHED,
    TERMINATION_STEP_LIMIT,
    TERMINATION_TIME_LIMIT,
    TICK_DT_S,
    EngineHeadlessSimulation,
    HeadlessEpisodeError,
    HeadlessSmokeEpisodeRunner,
    SmokeEpisodeResult,
    SmokeScenario,
    load_smoke_scenario,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "experiments" / "maps" / "smoke_v0" / "manifest.json"
CLI_PATH = ROOT / "experiments" / "run_smoke_episode.py"


# ---------------------------------------------------------------------------
# Fake simulation for isolating the episode loop
# ---------------------------------------------------------------------------


class _FakeSimulation:
    """Programmable stand-in for HeadlessSimulation. Each property/method
    reads the CURRENT tick index (advanced by tick()), so a test can script
    exactly which tick each metric crosses its threshold on."""

    def __init__(self, *, coverage_by_tick=None, fires_detected_by_tick=None, distance_per_tick=0.1):
        self._tick_index = 0
        self._time = 0.0
        self._coverage_by_tick = coverage_by_tick or {}
        self._fires_detected_by_tick = fires_detected_by_tick or {}
        self._distance_per_tick = distance_per_tick
        self._distance = 0.0
        self._decisions = 0
        self.installed_fires: list[tuple[float, float]] = []
        self.started = False

    def install_fire(self, x: float, y: float) -> None:
        self.installed_fires.append((x, y))

    def start(self) -> None:
        self.started = True

    def tick(self, dt: float) -> None:
        self._tick_index += 1
        self._time += dt
        self._distance += self._distance_per_tick
        self._decisions += 1

    def fires_detected_count(self, fire_positions, detection_threshold) -> int:
        return self._fires_detected_by_tick.get(self._tick_index, 0)

    @property
    def simulation_time(self) -> float:
        return self._time

    @property
    def coverage_fraction(self) -> float:
        return self._coverage_by_tick.get(self._tick_index, 0.0)

    @property
    def total_distance_traveled(self) -> float:
        return self._distance

    @property
    def decision_count(self) -> int:
        return self._decisions


def _fake_scenario(sim_path: Path, *, fire_positions=(), max_time_s=100.0, max_steps=None) -> SmokeScenario:
    return SmokeScenario(
        corpus_id="smoke_v0",
        map_id=sim_path.stem,
        scenario_id="single_fire",
        sim_path=sim_path,
        seed=1,
        fire_positions=fire_positions,
        max_time_s=max_time_s,
        max_steps=max_steps,
    )


REAL_SIM_PATH = ROOT / "experiments" / "maps" / "smoke_v0" / "01_open.sim"


# ---------------------------------------------------------------------------
# SmokeScenario / manifest loader
# ---------------------------------------------------------------------------


def test_load_scenario_01_open_single_fire():
    scenario = load_smoke_scenario(
        MANIFEST_PATH, map_id="01_open", scenario_id="single_fire", seed=1, max_time_s=120.0
    )
    assert scenario.corpus_id == "smoke_v0"
    assert scenario.map_id == "01_open"
    assert scenario.scenario_id == "single_fire"
    assert scenario.sim_path.is_file()
    assert len(scenario.fire_positions) == 1
    assert scenario.seed == 1
    assert scenario.max_time_s == 120.0


def test_load_scenario_double_fire_preserves_order():
    scenario = load_smoke_scenario(
        MANIFEST_PATH, map_id="03_corridors", scenario_id="double_fire", seed=1, max_time_s=120.0
    )
    assert len(scenario.fire_positions) == 2
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entry = next(e for e in manifest["maps"] if e["map_id"] == "smoke_v0_03_corridors")
    scenario_entry = next(s for s in entry["fire_scenarios"] if s["scenario_id"] == "double_fire")
    expected = tuple((f["x"], f["y"]) for f in scenario_entry["fires"])
    assert scenario.fire_positions == expected


def test_load_scenario_missing_map_raises():
    with pytest.raises(HeadlessEpisodeError):
        load_smoke_scenario(
            MANIFEST_PATH, map_id="99_does_not_exist", scenario_id="single_fire", seed=1, max_time_s=10.0
        )


def test_load_scenario_missing_scenario_raises():
    with pytest.raises(HeadlessEpisodeError):
        load_smoke_scenario(
            MANIFEST_PATH, map_id="01_open", scenario_id="triple_fire", seed=1, max_time_s=10.0
        )


def test_load_scenario_missing_manifest_raises(tmp_path):
    with pytest.raises(HeadlessEpisodeError):
        load_smoke_scenario(
            tmp_path / "no_such_manifest.json",
            map_id="01_open", scenario_id="single_fire", seed=1, max_time_s=10.0,
        )


def test_scenario_missing_sim_file_raises(tmp_path):
    with pytest.raises(HeadlessEpisodeError):
        SmokeScenario(
            corpus_id="smoke_v0", map_id="ghost", scenario_id="single_fire",
            sim_path=tmp_path / "ghost.sim", seed=1, fire_positions=(), max_time_s=10.0,
        )


def test_seed_bool_rejected():
    with pytest.raises(HeadlessEpisodeError):
        SmokeScenario(
            corpus_id="smoke_v0", map_id=REAL_SIM_PATH.stem, scenario_id="single_fire",
            sim_path=REAL_SIM_PATH, seed=True, fire_positions=(), max_time_s=10.0,
        )


@pytest.mark.parametrize("bad_max_time", [0.0, -5.0, float("inf"), float("nan")])
def test_max_time_invalid_rejected(bad_max_time):
    with pytest.raises(HeadlessEpisodeError):
        SmokeScenario(
            corpus_id="smoke_v0", map_id=REAL_SIM_PATH.stem, scenario_id="single_fire",
            sim_path=REAL_SIM_PATH, seed=1, fire_positions=(), max_time_s=bad_max_time,
        )


def test_fire_position_not_finite_rejected():
    with pytest.raises(HeadlessEpisodeError):
        SmokeScenario(
            corpus_id="smoke_v0", map_id=REAL_SIM_PATH.stem, scenario_id="single_fire",
            sim_path=REAL_SIM_PATH, seed=1, fire_positions=((float("nan"), 0.0),), max_time_s=10.0,
        )


def test_scenario_map_id_must_match_sim_path_stem():
    with pytest.raises(HeadlessEpisodeError):
        SmokeScenario(
            corpus_id="smoke_v0", map_id="not_the_real_stem", scenario_id="single_fire",
            sim_path=REAL_SIM_PATH, seed=1, fire_positions=(), max_time_s=10.0,
        )


# ---------------------------------------------------------------------------
# No GUI
# ---------------------------------------------------------------------------


def test_runner_module_does_not_import_mainwindow_or_canvas():
    source_path = ROOT / "robotics_sim" / "simulation" / "headless_episode_runner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    seen_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            seen_modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                seen_modules.add(alias.name)

    forbidden = {"robotics_sim.app", "robotics_sim.app.main_window", "robotics_sim.app.simulation_canvas"}
    assert not (seen_modules & forbidden), seen_modules
    assert not any(m.startswith("robotics_sim.app") for m in seen_modules), seen_modules


def test_runner_module_references_no_qapplication_symbol():
    """Checks actual code identifiers, not the module's own prose docstring
    (which legitimately explains that no QApplication is created)."""
    source_path = ROOT / "robotics_sim" / "simulation" / "headless_episode_runner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name)

    assert "QApplication" not in names


def test_real_episode_never_creates_a_qapplication():
    from PySide6.QtWidgets import QApplication

    assert QApplication.instance() is None, "a QApplication already exists before this test ran"

    scenario = _fake_scenario(REAL_SIM_PATH, max_time_s=2.0)
    HeadlessSmokeEpisodeRunner().run(scenario)

    assert QApplication.instance() is None


# ---------------------------------------------------------------------------
# Termination logic (fake simulation)
# ---------------------------------------------------------------------------


def test_episode_terminates_on_time_limit():
    sim = _FakeSimulation()
    scenario = _fake_scenario(REAL_SIM_PATH, max_time_s=1.0)  # 20 ticks at TICK_DT_S
    result = HeadlessSmokeEpisodeRunner(simulation_factory=lambda cfg, seed: sim).run(scenario)
    assert result.termination_reason == TERMINATION_TIME_LIMIT
    assert result.success is False
    assert result.tick_count == pytest.approx(1.0 / TICK_DT_S, abs=1)


def test_episode_terminates_on_step_limit():
    sim = _FakeSimulation()
    scenario = _fake_scenario(REAL_SIM_PATH, max_time_s=1000.0, max_steps=5)
    result = HeadlessSmokeEpisodeRunner(simulation_factory=lambda cfg, seed: sim).run(scenario)
    assert result.termination_reason == TERMINATION_STEP_LIMIT
    assert result.tick_count == 5
    assert result.success is False


def test_episode_terminates_on_all_fires_detected():
    sim = _FakeSimulation(fires_detected_by_tick={1: 0, 2: 1})
    scenario = _fake_scenario(REAL_SIM_PATH, fire_positions=((1.0, 1.0),), max_time_s=1000.0, max_steps=100)
    result = HeadlessSmokeEpisodeRunner(simulation_factory=lambda cfg, seed: sim).run(scenario)
    assert result.termination_reason == TERMINATION_ALL_FIRES_DETECTED
    assert result.tick_count == 2
    assert result.success is True
    assert result.fires_detected == 1
    assert result.fires_total == 1


def test_episode_terminates_on_coverage_reached():
    sim = _FakeSimulation(coverage_by_tick={1: 0.10, 2: 0.95})
    scenario = _fake_scenario(REAL_SIM_PATH, max_time_s=1000.0, max_steps=100)
    result = HeadlessSmokeEpisodeRunner(simulation_factory=lambda cfg, seed: sim).run(
        scenario, coverage_threshold=0.90
    )
    assert result.termination_reason == TERMINATION_COVERAGE_REACHED
    assert result.tick_count == 2
    assert result.success is True
    assert result.coverage_fraction == pytest.approx(0.95)


def test_termination_priority_fires_beats_coverage():
    # Both conditions true on the same tick -- ALL_FIRES_DETECTED must win.
    sim = _FakeSimulation(coverage_by_tick={1: 0.95}, fires_detected_by_tick={1: 1})
    scenario = _fake_scenario(REAL_SIM_PATH, fire_positions=((1.0, 1.0),), max_time_s=1000.0, max_steps=100)
    result = HeadlessSmokeEpisodeRunner(simulation_factory=lambda cfg, seed: sim).run(
        scenario, coverage_threshold=0.90
    )
    assert result.termination_reason == TERMINATION_ALL_FIRES_DETECTED


def test_termination_priority_coverage_beats_time_limit():
    # Coverage threshold reached on the very last allowed tick, exactly
    # when TIME_LIMIT would also fire -- COVERAGE_REACHED must win.
    ticks_at_limit = round(1.0 / TICK_DT_S)
    sim = _FakeSimulation(coverage_by_tick={ticks_at_limit: 0.95})
    scenario = _fake_scenario(REAL_SIM_PATH, max_time_s=1.0)
    result = HeadlessSmokeEpisodeRunner(simulation_factory=lambda cfg, seed: sim).run(
        scenario, coverage_threshold=0.90
    )
    assert result.termination_reason == TERMINATION_COVERAGE_REACHED


def test_termination_priority_time_beats_step_limit():
    # TIME_LIMIT and STEP_LIMIT would both fire on the same tick --
    # TIME_LIMIT must win (checked before STEP_LIMIT).
    ticks_at_limit = round(1.0 / TICK_DT_S)
    sim = _FakeSimulation()
    scenario = _fake_scenario(REAL_SIM_PATH, max_time_s=1.0, max_steps=ticks_at_limit)
    result = HeadlessSmokeEpisodeRunner(simulation_factory=lambda cfg, seed: sim).run(scenario)
    assert result.termination_reason == TERMINATION_TIME_LIMIT


def test_episode_error_is_not_silently_swallowed():
    class _ExplodingSimulation(_FakeSimulation):
        def tick(self, dt: float) -> None:
            raise RuntimeError("boom")

    scenario = _fake_scenario(REAL_SIM_PATH, max_time_s=10.0)
    with pytest.raises(HeadlessEpisodeError):
        HeadlessSmokeEpisodeRunner(simulation_factory=lambda cfg, seed: _ExplodingSimulation()).run(scenario)


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------


def test_to_dict_is_stable_and_complete():
    result = SmokeEpisodeResult(
        corpus_id="smoke_v0", map_id="01_open", scenario_id="single_fire", seed=1,
        termination_reason=TERMINATION_TIME_LIMIT, success=False, simulation_time_s=10.0,
        tick_count=200, decision_count=5, coverage_fraction=0.22, distance_traveled=15.8,
        fires_total=1, fires_detected=0,
    )
    expected_keys = {
        "corpus_id", "map_id", "scenario_id", "seed", "termination_reason", "success",
        "simulation_time_s", "tick_count", "decision_count", "coverage_fraction",
        "distance_traveled", "fires_total", "fires_detected",
    }
    d1 = result.to_dict()
    d2 = result.to_dict()
    assert d1 == d2
    assert set(d1.keys()) == expected_keys
    assert list(d1.keys()) == list(d2.keys())


# ---------------------------------------------------------------------------
# Real runtime: determinism + fire installation (smoke tests)
# ---------------------------------------------------------------------------


def test_real_engine_episode_runs_end_to_end():
    scenario = load_smoke_scenario(
        MANIFEST_PATH, map_id="01_open", scenario_id="single_fire", seed=1, max_time_s=3.0
    )
    result = HeadlessSmokeEpisodeRunner().run(scenario)
    assert result.termination_reason in {
        TERMINATION_TIME_LIMIT, TERMINATION_STEP_LIMIT, TERMINATION_COVERAGE_REACHED, TERMINATION_ALL_FIRES_DETECTED,
    }
    assert result.tick_count > 0
    assert result.simulation_time_s > 0.0
    assert 0.0 <= result.coverage_fraction <= 1.0
    assert result.distance_traveled >= 0.0


def test_two_identical_real_runs_are_deterministic():
    scenario = load_smoke_scenario(
        MANIFEST_PATH, map_id="01_open", scenario_id="single_fire", seed=1, max_time_s=3.0
    )
    result_a = HeadlessSmokeEpisodeRunner().run(scenario)
    result_b = HeadlessSmokeEpisodeRunner().run(scenario)

    assert result_a.termination_reason == result_b.termination_reason
    assert result_a.tick_count == result_b.tick_count
    assert result_a.decision_count == result_b.decision_count
    assert result_a.fires_detected == result_b.fires_detected
    assert result_a.coverage_fraction == pytest.approx(result_b.coverage_fraction, abs=1e-9)
    assert result_a.distance_traveled == pytest.approx(result_b.distance_traveled, abs=1e-9)


def test_fire_installed_at_manifest_positions():
    scenario = load_smoke_scenario(
        MANIFEST_PATH, map_id="01_open", scenario_id="double_fire", seed=1, max_time_s=1.0
    )
    config = load_sim_file(str(scenario.sim_path))
    simulation = EngineHeadlessSimulation(config, scenario.seed)
    for x, y in scenario.fire_positions:
        simulation.install_fire(x, y)

    installed_positions = tuple(source.position for source in simulation.hazard_service.sources())
    assert set(installed_positions) == set(scenario.fire_positions)


def test_fire_does_not_modify_occupancy():
    """Ground-truth occupancy (point_inside_ground_truth_obstacle, the
    OccupancyGrid rasterization, and CollisionChecker) are all pure
    functions of SimulationConfig.obstacles -- confirming that list is
    unchanged after install_fire() is a complete proof that fire placement
    never touches occupancy, without reaching into any private state."""
    scenario = load_smoke_scenario(
        MANIFEST_PATH, map_id="01_open", scenario_id="single_fire", seed=1, max_time_s=1.0
    )
    config = load_sim_file(str(scenario.sim_path))
    simulation = EngineHeadlessSimulation(config, scenario.seed)
    obstacles_before = list(simulation.obstacles)

    for x, y in scenario.fire_positions:
        simulation.install_fire(x, y)

    assert simulation.obstacles == obstacles_before


# ---------------------------------------------------------------------------
# fires_detected_count(): per-position HazardBelief.read_cells(), explicit
# detection_threshold -- never blocked_cells()/observed_blocked_world_points()
# ---------------------------------------------------------------------------


def _real_headless_simulation() -> EngineHeadlessSimulation:
    config = load_sim_file(str(REAL_SIM_PATH))
    return EngineHeadlessSimulation(config, seed=1)


def test_fires_detected_count_counts_an_observed_fire():
    simulation = _real_headless_simulation()
    fire_xy = (0.0, 0.0)  # open space in 01_open.sim
    simulation.install_fire(*fire_xy)

    belief = simulation.hazard_service.belief
    cell = belief.geometry.world_to_grid(*fire_xy, clamp=True)
    belief.observe_cells([cell.row], [cell.col], [0.90], robot_index=0)

    count = simulation.fires_detected_count((fire_xy,), DEFAULT_FIRE_DETECTION_THRESHOLD)
    assert count == 1


def test_fires_detected_count_ignores_an_unobserved_fire():
    simulation = _real_headless_simulation()
    fire_xy = (0.0, 0.0)
    simulation.install_fire(*fire_xy)
    # No observe_cells() call -- the fire's cell stays observed=False.

    count = simulation.fires_detected_count((fire_xy,), DEFAULT_FIRE_DETECTION_THRESHOLD)
    assert count == 0


def test_fires_detected_count_ignores_a_blocked_cell_with_no_matching_fire():
    """A cell elsewhere in the belief being observed+blocked must not
    inflate the count for a fire position that itself was never observed --
    proves the method checks each fire's OWN cell, not "is anything in the
    whole map currently blocked" (which blocked_cells()/
    observed_blocked_world_points() would answer instead)."""
    simulation = _real_headless_simulation()
    fire_xy = (0.0, 0.0)
    simulation.install_fire(*fire_xy)

    belief = simulation.hazard_service.belief
    unrelated_cell = belief.geometry.world_to_grid(8.0, 6.0, clamp=True)
    belief.observe_cells([unrelated_cell.row], [unrelated_cell.col], [1.0], robot_index=0)

    count = simulation.fires_detected_count((fire_xy,), DEFAULT_FIRE_DETECTION_THRESHOLD)
    assert count == 0


def test_fires_detected_count_uses_the_explicit_threshold_not_block_threshold():
    """A value that clears a low, explicitly-passed detection_threshold but
    sits below hazard_service.block_threshold must still count -- proving
    the comparison uses the caller-supplied threshold, not
    self._host.hazard_service.block_threshold."""
    simulation = _real_headless_simulation()
    fire_xy = (0.0, 0.0)
    simulation.install_fire(*fire_xy)

    block_threshold = simulation.hazard_service.block_threshold
    low_value = block_threshold / 2.0
    assert low_value < block_threshold  # sanity check on the fixture itself

    belief = simulation.hazard_service.belief
    cell = belief.geometry.world_to_grid(*fire_xy, clamp=True)
    belief.observe_cells([cell.row], [cell.col], [low_value], robot_index=0)

    assert simulation.fires_detected_count((fire_xy,), block_threshold) == 0
    assert simulation.fires_detected_count((fire_xy,), low_value) == 1


# ---------------------------------------------------------------------------
# time.perf_counter patch restoration
# ---------------------------------------------------------------------------


def test_perf_counter_restored_after_exception_during_tick():
    original_perf_counter = time.perf_counter

    simulation = _real_headless_simulation()
    simulation.install_fire(0.0, 0.0)
    simulation.start()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    simulation._host.simulation_step_multi = _boom

    with pytest.raises(RuntimeError, match="boom"):
        simulation.tick(TICK_DT_S)

    assert time.perf_counter is original_perf_counter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_cli_returns_valid_json_and_exit_zero():
    proc = _run_cli(["--map", "01_open", "--scenario", "single_fire", "--seed", "1", "--max-time-s", "3"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["map_id"] == "01_open"
    assert payload["scenario_id"] == "single_fire"
    assert payload["seed"] == 1


def test_cli_invalid_map_returns_nonzero_exit_code():
    proc = _run_cli(["--map", "99_does_not_exist", "--scenario", "single_fire", "--seed", "1", "--max-time-s", "3"])
    assert proc.returncode != 0
