"""
Headless episode runner for the smoke_v0 map corpus.

Runs one reproducible exploration episode against a real .sim map with no
GUI: no MainWindow, no SimulationCanvas, no QApplication, no Qt event loop.

Why this is possible without touching engine.py
-------------------------------------------------
robotics_sim.simulation.engine.SimulationControllerMixin is a plain mixin
with no __init__ of its own -- MainWindow is the only class that mixes it
into a QMainWindow. Every method on the mixin itself reads/writes plain
Python/numpy state (self.config, self.robots, self.belief_map,
self.hazard_service, ...) except request_route_async()'s non-"Direct"
planner branch, which spins up a QRunnable on a QThreadPool. All six
smoke_v0 .sim files use planner_type="Direct" (config.py's own default),
which takes a fully synchronous, Qt-free branch inside that same method
(see request_route_async(), engine.py) -- so the QRunnable/QThreadPool path
is simply never reached here.

This module reuses the exact technique robotics_sim/tests/test_fov_
costmap_integration.py's _make_fake_engine() and friends already use: bind
the real, unmodified SimulationControllerMixin functions onto a small
non-Qt host object (instead of onto a MainWindow instance), and duck-type
stub out only the GUI leaves the mixin calls into (self.canvas's
set_*/append_*/invalidate_* setters, self.top_bar.set_status,
self.start_button/self.speed_button.setText/.setIcon) -- all one-way
setters the mixin never reads back. self.telemetry uses the real
TelemetryLogger (robotics_sim/simulation/telemetry.py is itself Qt-free)
with its default no-op sink, so nothing is lost by not stubbing it too.

Two GUI-only methods the mixin's own start_multi_robot_simulation() calls
-- ensure_multi_robot_configs() and set_configuration_locked() -- are
defined on MainWindow itself (main_window.py), not on the mixin, and read
real Qt widgets (a spinbox, a switch). This runner does not call
start_multi_robot_simulation() for that reason; it reimplements only the
plain-Python state-setup lines that method performs (robot construction,
belief/hazard reset, initial belief scan, initial route assignment),
calling the real bound mixin methods for each step, exactly as
_make_fake_engine() does for narrower slices of the same mixin.

Determinism
-----------
simulation_step_multi() reads time.perf_counter() once per call (via
should_run_sensor_update()) to throttle sensor updates at ~10 Hz *or*
sooner if the robot moved/rotated enough. Left alone, this ties sensor-
update timing to real wall-clock jitter, which is not reproducible between
runs. This runner never lets that happen: every call into the engine
temporarily replaces the process-wide time.perf_counter with a synthetic
clock that only advances by the exact `dt` this runner itself chose,
before that dt is fed to the tick -- so "now" inside the engine is a pure,
deterministic function of accumulated simulated time, never of real
wall-clock speed. (Confirmed via repo-wide search: no other code path
touched by a headless episode reads a wall clock or `random`/`numpy.random`
in a way that affects planning/control decisions.)

The patch itself (EngineHeadlessSimulation._patched_clock()) is a
@contextmanager built on a plain try/finally: the original
time.perf_counter is saved once, the synthetic one is installed, and the
original is always put back in the `finally` block -- including when the
wrapped call (e.g. simulation_step_multi()) raises. This is verified
directly by test_perf_counter_restored_after_exception_during_tick() in
robotics_sim/tests/test_headless_smoke_episode.py, which forces an
exception mid-tick and asserts time.perf_counter is back to the exact
original function object afterward.
"""

from __future__ import annotations

import inspect
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.simulation.config import SimulationConfig, SpatialObstacleIndex, load_sim_file
from robotics_sim.simulation.config import normalized_robot_start_configs
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.telemetry import QUIET, TelemetryLogger

# Matches engine.py's own `dt = min(real_dt, 0.05) * simulation_speed` clamp
# ceiling (simulation_step_multi()) -- passing exactly this value means the
# clamp never further alters what this runner declared, so the runner's
# own dt bookkeeping (max_steps derivation, the synthetic clock) always
# matches what the engine actually integrated.
TICK_DT_S = 0.05

# Explicit fire-detection threshold, deliberately independent of
# RuntimeHazardService.block_threshold (a different concern: whether a cell
# should block navigation/replanning). Same numeric default as
# SimulationConfig.hazard_block_threshold (config.py) -- not read from a
# live hazard_service instance, so it never silently drifts if a scenario
# configures its own block_threshold. See EngineHeadlessSimulation.
# fires_detected_count() and HeadlessSmokeEpisodeRunner.run()'s
# fire_detection_threshold argument.
DEFAULT_FIRE_DETECTION_THRESHOLD = 0.55

TERMINATION_ALL_FIRES_DETECTED = "ALL_FIRES_DETECTED"
TERMINATION_COVERAGE_REACHED = "COVERAGE_REACHED"
TERMINATION_TIME_LIMIT = "TIME_LIMIT"
TERMINATION_STEP_LIMIT = "STEP_LIMIT"

_SUCCESS_TERMINATIONS = frozenset({TERMINATION_ALL_FIRES_DETECTED, TERMINATION_COVERAGE_REACHED})


class HeadlessEpisodeError(Exception):
    """Raised for invalid scenario configuration or an unexpected failure
    while running an episode. Exceptions from the underlying simulation are
    never silently swallowed into a "successful" result -- they surface as
    this (chained) exception instead."""


# ---------------------------------------------------------------------------
# SmokeScenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeScenario:
    """Everything needed to run one reproducible smoke episode.

    Deliberately holds no live engine/simulation object -- only plain,
    picklable/loggable data. The .sim file is re-loaded fresh by the runner
    each time from `sim_path`.
    """

    corpus_id: str
    map_id: str
    scenario_id: str
    sim_path: Path
    seed: int
    fire_positions: tuple[tuple[float, float], ...]
    max_time_s: float
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sim_path, Path):
            object.__setattr__(self, "sim_path", Path(self.sim_path))
        if not self.sim_path.is_file():
            raise HeadlessEpisodeError(f"sim file does not exist: {self.sim_path}")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise HeadlessEpisodeError(f"seed must be a non-bool int, got {self.seed!r}")

        if not math.isfinite(self.max_time_s) or self.max_time_s <= 0:
            raise HeadlessEpisodeError(f"max_time_s must be finite and > 0, got {self.max_time_s!r}")

        if self.max_steps is not None:
            if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
                raise HeadlessEpisodeError(f"max_steps must be a non-bool int or None, got {self.max_steps!r}")
            if self.max_steps <= 0:
                raise HeadlessEpisodeError(f"max_steps must be > 0, got {self.max_steps!r}")

        for point in self.fire_positions:
            x, y = point
            if not (math.isfinite(x) and math.isfinite(y)):
                raise HeadlessEpisodeError(f"fire position is not finite: {point!r}")

        # Structural "this scenario belongs to this map" check -- the
        # loader canonicalizes map_id to the .sim filename's stem (see
        # load_smoke_scenario()), so any SmokeScenario built by hand must
        # keep that same invariant.
        if self.sim_path.stem != self.map_id:
            raise HeadlessEpisodeError(
                f"scenario map_id {self.map_id!r} does not match sim_path {self.sim_path!r}"
            )


# ---------------------------------------------------------------------------
# SmokeEpisodeResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeEpisodeResult:
    corpus_id: str
    map_id: str
    scenario_id: str
    seed: int
    termination_reason: str
    success: bool
    simulation_time_s: float
    tick_count: int
    decision_count: int
    coverage_fraction: float
    distance_traveled: float
    fires_total: int
    fires_detected: int

    def to_dict(self) -> dict[str, object]:
        return {
            "corpus_id": self.corpus_id,
            "map_id": self.map_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "termination_reason": self.termination_reason,
            "success": self.success,
            "simulation_time_s": self.simulation_time_s,
            "tick_count": self.tick_count,
            "decision_count": self.decision_count,
            "coverage_fraction": self.coverage_fraction,
            "distance_traveled": self.distance_traveled,
            "fires_total": self.fires_total,
            "fires_detected": self.fires_detected,
        }


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------


def load_smoke_scenario(
    manifest_path: Path,
    *,
    map_id: str,
    scenario_id: str,
    seed: int,
    max_time_s: float,
    max_steps: int | None = None,
) -> SmokeScenario:
    """Pure loader: reads manifest.json and the requested map's fire scenario.

    `map_id` matches either the manifest's full map_id (e.g.
    "smoke_v0_01_open") or the .sim filename's stem (e.g. "01_open", the
    form used by the CLI) -- the returned SmokeScenario.map_id is always
    canonicalized to the filename stem.

    Does not mutate the manifest dict it reads, and never writes a file.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise HeadlessEpisodeError(f"manifest does not exist: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    corpus_id = str(manifest.get("corpus_id", ""))
    maps_dir = manifest_path.resolve().parent

    entry = None
    for candidate in manifest.get("maps", []):
        stem = Path(candidate["filename"]).stem
        if candidate.get("map_id") == map_id or stem == map_id:
            entry = candidate
            break
    if entry is None:
        raise HeadlessEpisodeError(f"map_id {map_id!r} not found in manifest {manifest_path}")

    canonical_map_id = Path(entry["filename"]).stem

    scenario_entry = None
    for candidate in entry.get("fire_scenarios", []):
        if candidate.get("scenario_id") == scenario_id:
            scenario_entry = candidate
            break
    if scenario_entry is None:
        raise HeadlessEpisodeError(
            f"scenario_id {scenario_id!r} not found for map {map_id!r} in manifest {manifest_path}"
        )

    fire_positions = tuple(
        (float(fire["x"]), float(fire["y"])) for fire in scenario_entry.get("fires", [])
    )

    sim_path = maps_dir / entry["filename"]

    return SmokeScenario(
        corpus_id=corpus_id,
        map_id=canonical_map_id,
        scenario_id=scenario_id,
        sim_path=sim_path,
        seed=seed,
        fire_positions=fire_positions,
        max_time_s=max_time_s,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# HeadlessSimulation protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class HeadlessSimulation(Protocol):
    """Minimal surface HeadlessSmokeEpisodeRunner needs from a running
    simulation. EngineHeadlessSimulation (below) is the real implementation;
    tests may implement this with a small fake to isolate the episode loop
    from the real engine."""

    def install_fire(self, x: float, y: float) -> None: ...

    def start(self) -> None: ...

    def tick(self, dt: float) -> None: ...

    def fires_detected_count(
        self,
        fire_positions: tuple[tuple[float, float], ...],
        detection_threshold: float,
    ) -> int: ...

    @property
    def simulation_time(self) -> float: ...

    @property
    def coverage_fraction(self) -> float: ...

    @property
    def total_distance_traveled(self) -> float: ...

    @property
    def decision_count(self) -> int: ...


# ---------------------------------------------------------------------------
# Real engine-backed HeadlessSimulation
# ---------------------------------------------------------------------------

# Every self.canvas.<name>(...) call reachable from the mixin methods this
# runner exercises (start_multi_robot_simulation's non-Qt lines,
# initialize_multi_robot_starting_belief, assign_route_to_multi_robot,
# simulation_step_multi, add_fire, reset_belief_map) -- confirmed by
# grepping every `self.canvas.` call site in engine.py. All are one-way
# setters/appenders the mixin never reads a return value from.
_CANVAS_METHODS = (
    "append_explored_area_polygon", "append_mapped_obstacle_points",
    "clear_explored_area_geometry", "invalidate_explored_area_cache",
    "invalidate_sensor_cache", "repaint", "set_exploration_target",
    "set_explored_area_polygons", "set_explored_area_seed",
    "set_frontier_reasoning_decision", "set_frontier_reasoning_simulation_paused",
    "set_grid_overlay_snapshot", "set_known_obstacles", "set_last_control",
    "set_mapped_obstacle_points", "set_multi_exploration_targets",
    "set_multi_robots", "set_multi_runtime_state",
    "set_navigation_debug_history_position", "set_navigation_debug_last_event",
    "set_navigation_debug_snapshot", "set_path", "set_planned_path",
    "set_preview_config", "set_robot", "set_runtime_state",
    "set_simulation_metrics", "set_simulation_running_for_perf", "set_status",
)


def _noop(*_args, **_kwargs) -> None:
    return None


class _NullCanvas:
    """Duck-typed stand-in for SimulationCanvas -- never a QWidget, never
    imported from robotics_sim.app. `fps` is read (not called) by one perf
    logging line, so it needs a plain attribute, not a method."""

    fps = 0.0


for _name in _CANVAS_METHODS:
    setattr(_NullCanvas, _name, _noop)


class _NullTopBar:
    set_status = staticmethod(_noop)


class _NullButton:
    def setText(self, *_a, **_k) -> None:
        return None

    def setIcon(self, *_a, **_k) -> None:
        return None


def _build_fake_host(config: SimulationConfig) -> SimpleNamespace:
    """Bind every real SimulationControllerMixin function/constant onto a
    plain SimpleNamespace host -- same idiom as _make_fake_engine() in
    robotics_sim/tests/test_fov_costmap_integration.py, scaled to cover the
    whole mixin instead of a hand-picked subset, since a full tick loop
    (simulation_step_multi) transitively touches dozens of mixin methods.
    """
    host = SimpleNamespace()
    host.config = config
    host.canvas = _NullCanvas()
    host.telemetry = TelemetryLogger(level=QUIET)
    host.top_bar = _NullTopBar()
    host.start_button = _NullButton()
    host.speed_button = _NullButton()
    host.spatial_index = SpatialObstacleIndex()
    host.collision_checker = CollisionChecker()

    host.running = False
    host.paused = False
    host.simulation_speed = 1.0
    host.selected_robot_index = 0
    host.editor_mode = False
    host.multi_robot_commands_by_id = {}

    for name, member in vars(SimulationControllerMixin).items():
        if name.startswith("__"):
            continue
        if inspect.isfunction(member):
            setattr(host, name, member.__get__(host))
        elif isinstance(member, staticmethod):
            setattr(host, name, member.__func__)
        elif isinstance(member, property):
            continue  # e.g. `telemetry` -- shadowed by the plain instance attribute set above
        else:
            setattr(host, name, member)

    return host


class EngineHeadlessSimulation:
    """Real SimulationControllerMixin, ticked headlessly.

    Construction only builds robots + belief map + hazard service (no
    initial sensor scan, no initial routes yet) so fires can be installed
    via install_fire() before start() runs the first sensor observation --
    matching the episode-loop ordering in headless_episode_runner's
    module docstring / the task spec ("aplicar el escenario de fuego...
    antes del primer tick").
    """

    def __init__(self, config: SimulationConfig, seed: int) -> None:
        self._seed = seed
        self._clock_t = 0.0
        host = _build_fake_host(config)
        self._host = host

        with self._patched_clock():
            host.spatial_index.rebuild(config.obstacles)
            host.planning_in_progress = False
            host.route_request_id = 0
            host.active_planner_workers = {}

            robot_starts = normalized_robot_start_configs(config)
            host.robots = [host.create_robot_instance(sc) for sc in robot_starts]
            host.robot = host.robots[0] if host.robots else None
            host.sync_runtime_robot_agents()

            host.known_obstacles = []
            host.explored_area_polygons = []
            host.reset_belief_map(robot_count=len(host.robots) or 1, preserve_hazards=True)

            host.current_exploration_target = None
            host.multi_exploration_targets = [None for _ in host.robots]
            host._multi_robot_coordinator = None
            host.last_coordination_debug = {}
            host.multi_invalidated_exploration_targets = [[] for _ in host.robots]
            host.last_exploration_replan_sim_time = -1.0e9
            host.last_exploration_gate_message_time = -1.0e9
            host.last_goal_selection_reason = "headless episode runner"
            host.route_request_count = 0
            host.route_result_count = 0
            host.sensor_update_count = 0
            host.mapping_update_count = 0
            host.safety_replan_count = 0
            host.exploration_replan_count = 0
            host.total_distance_traveled = 0.0
            host.last_explored_pose = None
            host.multi_last_explored_poses = {}
            host.last_visible_sensor_polygon = None
            host.multi_visible_sensor_polygons = {}
            host.last_sensor_update_time = 0.0
            host.last_sensor_update_pose = None

            host.multi_path_points = [[(float(r.x), float(r.y))] for r in host.robots]
            host.multi_robot_commands_by_id = {}
            host.multi_planned_path_points = [[] for _ in host.robots]
            host.multi_last_controls = [np.array([[0.0], [0.0]], dtype=float) for _ in host.robots]
            host.multi_route_states = [host.ROUTE_STATE_ACTIVE for _ in host.robots]
            host.multi_route_state_reasons = [""] * len(host.robots)
            host.multi_last_route_state_log_times = [-1.0e9] * len(host.robots)
            host.path_points = host.multi_path_points[0] if host.multi_path_points else []
            host.last_control = (
                host.multi_last_controls[0] if host.multi_last_controls else np.array([[0.0], [0.0]], dtype=float)
            )
            host.last_motion_log_time = -1.0e9
            host.multi_last_motion_log_times = {}
            host.simulation_time = 0.0

        self._started = False

    @contextmanager
    def _patched_clock(self):
        """Replace the process-wide time.perf_counter with a synthetic clock
        tied to this simulation's own accumulated dt (see module docstring
        "Determinism"). Scoped narrowly to each call into the engine.

        Plain try/finally, not just a context-manager wrapper around one:
        the original function is saved once, installed, and *always* put
        back in `finally` -- including when the wrapped call (e.g.
        simulation_step_multi()) raises. See
        test_perf_counter_restored_after_exception_during_tick() in
        test_headless_smoke_episode.py for the regression test."""
        original_perf_counter = time.perf_counter
        time.perf_counter = lambda: self._clock_t
        try:
            yield
        finally:
            time.perf_counter = original_perf_counter

    def install_fire(self, x: float, y: float) -> None:
        host = self._host
        with self._patched_clock():
            ok = host.add_fire(float(x), float(y))
        if not ok:
            raise HeadlessEpisodeError(f"failed to place fire at ({x}, {y}) -- out of bounds or invalid")

    def start(self) -> None:
        if self._started:
            return
        host = self._host
        with self._patched_clock():
            host.initialize_multi_robot_starting_belief()
            for index in range(len(host.robots)):
                host.assign_route_to_multi_robot(index, reason="Initial multi-robot route")
            host.running = True
            host.paused = False
        self._started = True

    def tick(self, dt: float) -> None:
        if not self._started:
            raise HeadlessEpisodeError("tick() called before start()")
        self._clock_t += float(dt)
        with self._patched_clock():
            self._host.simulation_step_multi(float(dt))

    def fires_detected_count(
        self,
        fire_positions: tuple[tuple[float, float], ...],
        detection_threshold: float = DEFAULT_FIRE_DETECTION_THRESHOLD,
    ) -> int:
        """Count how many of `fire_positions` the team's HazardBelief has
        detected, checking each position independently and exactly once.

        Deliberately does NOT call HazardBelief.blocked_cells() or
        RuntimeHazardService.observed_blocked_world_points() -- both return
        *every* observed-and-hazardous cell anywhere in the belief, which
        would silently count unrelated hazardous cells as "detected fires"
        and would not distinguish one fire from another sharing a
        threshold-crossing neighborhood. Instead, per fire position:

            1. convert it to its own grid cell via the belief's real
               GridGeometry (world_to_grid, clamped to bounds -- the same
               conversion RuntimeHazardService itself uses);
            2. read exactly that cell's (value, observed) pair via
               HazardBelief.read_cells() -- O(1), not a whole-grid scan;
            3. count it once iff observed is True AND value >=
               detection_threshold.

        detection_threshold is an explicit, caller-supplied value (see
        DEFAULT_FIRE_DETECTION_THRESHOLD / HeadlessSmokeEpisodeRunner.run()'s
        fire_detection_threshold argument) -- deliberately NOT
        self._host.hazard_service.block_threshold, which governs a
        different concern (when a cell should block navigation/replanning),
        not "was this fire detected."
        """
        belief = self._host.hazard_service.belief
        geometry = belief.geometry
        detected = 0
        for x, y in fire_positions:
            cell = geometry.world_to_grid(float(x), float(y), clamp=True)
            values, observed = belief.read_cells([cell.row], [cell.col])
            if bool(observed[0]) and float(values[0]) >= detection_threshold:
                detected += 1
        return detected

    @property
    def simulation_time(self) -> float:
        return float(self._host.simulation_time)

    @property
    def coverage_fraction(self) -> float:
        # estimated_free_space_coverage_percent() = explored free cells /
        # reachable (obstacle-excluded) free cells, NOT / all cells
        # including obstacle interiors -- see its own docstring in engine.py.
        with self._patched_clock():
            return float(self._host.estimated_free_space_coverage_percent()) / 100.0

    @property
    def total_distance_traveled(self) -> float:
        # Team-wide: one scalar accumulator summing every robot's
        # incremental travel (see _append_multi_executed_path_point in
        # engine.py) -- there is no per-robot accumulator in the runtime.
        return float(self._host.total_distance_traveled)

    @property
    def decision_count(self) -> int:
        # exploration_replan_count: incremented only when a robot is
        # assigned a genuinely NEW exploration target (frontier reached, or
        # a blocked/invalidated target forces a fresh one) -- never
        # incremented for a same-target safety/repair replan
        # (safety_replan_count is the separate counter for those, per
        # engine.py's replan_after_new_information()/simulation_step_multi()).
        # KNOWN GAP: does not count the very first target assignment made
        # in start() (start_multi_robot_simulation() doesn't touch this
        # counter for the initial assignment either -- this matches
        # upstream behavior, not a shortcut taken here).
        return int(self._host.exploration_replan_count)

    # -- read-only debug/introspection accessors (tests only) -------------

    @property
    def hazard_service(self):
        return self._host.hazard_service

    @property
    def obstacles(self) -> list[tuple[float, float, float, float]]:
        return list(self._host.config.obstacles)

    @property
    def robots(self):
        return list(self._host.robots)


def build_engine_headless_simulation(config: SimulationConfig, seed: int) -> EngineHeadlessSimulation:
    """Default simulation_factory for HeadlessSmokeEpisodeRunner."""
    return EngineHeadlessSimulation(config, seed)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class HeadlessSmokeEpisodeRunner:
    """Drives one deterministic episode: build -> install fires -> initial
    sensor update -> tick until a terminal condition -> SmokeEpisodeResult.

    simulation_factory is injectable so termination-logic tests can swap in
    a small fake HeadlessSimulation and isolate the loop from the real
    engine (see robotics_sim/tests/test_headless_smoke_episode.py).
    """

    def __init__(
        self,
        simulation_factory: Callable[[SimulationConfig, int], HeadlessSimulation] = build_engine_headless_simulation,
    ) -> None:
        self._simulation_factory = simulation_factory

    def run(
        self,
        scenario: SmokeScenario,
        *,
        coverage_threshold: float = 0.90,
        fire_detection_threshold: float = DEFAULT_FIRE_DETECTION_THRESHOLD,
    ) -> SmokeEpisodeResult:
        if not math.isfinite(coverage_threshold) or not (0.0 < coverage_threshold <= 1.0):
            raise HeadlessEpisodeError(f"coverage_threshold must be in (0, 1], got {coverage_threshold!r}")
        if not math.isfinite(fire_detection_threshold):
            raise HeadlessEpisodeError(
                f"fire_detection_threshold must be finite, got {fire_detection_threshold!r}"
            )

        random.seed(scenario.seed)
        np.random.seed(scenario.seed)

        try:
            config = load_sim_file(str(scenario.sim_path))
            simulation = self._simulation_factory(config, scenario.seed)

            for x, y in scenario.fire_positions:
                simulation.install_fire(x, y)

            simulation.start()

            max_steps = (
                scenario.max_steps
                if scenario.max_steps is not None
                else max(1, math.ceil(scenario.max_time_s / TICK_DT_S))
            )

            tick_count = 0
            coverage_fraction = simulation.coverage_fraction
            fires_detected = simulation.fires_detected_count(scenario.fire_positions, fire_detection_threshold)
            termination_reason: str | None = None

            while termination_reason is None:
                simulation.tick(TICK_DT_S)
                tick_count += 1

                coverage_fraction = simulation.coverage_fraction
                fires_detected = simulation.fires_detected_count(scenario.fire_positions, fire_detection_threshold)

                # Stable priority: ALL_FIRES_DETECTED > COVERAGE_REACHED >
                # TIME_LIMIT > STEP_LIMIT (see headless_episode_runner
                # module docstring / task spec section 8).
                if scenario.fire_positions and fires_detected >= len(scenario.fire_positions):
                    termination_reason = TERMINATION_ALL_FIRES_DETECTED
                elif coverage_fraction >= coverage_threshold:
                    termination_reason = TERMINATION_COVERAGE_REACHED
                elif simulation.simulation_time >= scenario.max_time_s:
                    termination_reason = TERMINATION_TIME_LIMIT
                elif tick_count >= max_steps:
                    termination_reason = TERMINATION_STEP_LIMIT

            return SmokeEpisodeResult(
                corpus_id=scenario.corpus_id,
                map_id=scenario.map_id,
                scenario_id=scenario.scenario_id,
                seed=scenario.seed,
                termination_reason=termination_reason,
                success=termination_reason in _SUCCESS_TERMINATIONS,
                simulation_time_s=simulation.simulation_time,
                tick_count=tick_count,
                decision_count=simulation.decision_count,
                coverage_fraction=coverage_fraction,
                distance_traveled=simulation.total_distance_traveled,
                fires_total=len(scenario.fire_positions),
                fires_detected=fires_detected,
            )
        except HeadlessEpisodeError:
            raise
        except Exception as exc:  # noqa: BLE001 -- re-raised with context, never swallowed
            raise HeadlessEpisodeError(
                f"episode failed for map_id={scenario.map_id!r} scenario_id={scenario.scenario_id!r}: {exc}"
            ) from exc
