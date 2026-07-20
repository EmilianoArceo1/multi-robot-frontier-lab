"""
Audit characterization: does every REAL production path that can turn a
physical obstacle into BeliefMap.OCCUPIED also produce corresponding
observed geometry (mapped_obstacle_points / observed_obstacle_snapshot())?

This matters because PlanningCostmapBuilder (robotics_sim/planning/
planning_costmap_builder.py) now deliberately does NOT treat a legacy
BeliefMap.OCCUPIED cell as physical occupancy on its own -- only
ObservedObstacleSnapshot.points (and observed hazard cells) block a cell
(see that module's docstring, "Legacy belief occupancy vs. observed
obstacle geometry"). Before wiring the builder to runtime, every production
path that writes OCCUPIED must be shown to ALSO write the matching observed
geometry -- otherwise connecting the builder would silently stop blocking
routes the current runtime still blocks.

Call-site inventory (git grep "mark_occupied_cell" / "mark_occupied_points",
July 2026) and classification:

    robotics_sim/environment/belief_map.py:396
        Inside mark_occupied_points() itself (self.mark_occupied_cell(...)
        per point). Not an independent call site -- inherits whatever
        classification its only caller has.

    robotics_sim/simulation/engine.py:6612
        update_sensed_obstacles() -> belief.mark_occupied_points(newly_mapped,
        ...), called with the EXACT SAME newly_mapped list that is appended
        to self.mapped_obstacle_points immediately above (line 6605).
        Classification: (1) sensor-derived obstacle observation. Coverage is
        guaranteed BY CONSTRUCTION (same list feeds both), not by
        coincidence -- see test_update_sensed_obstacles_produces_observed_
        geometry_matching_mapped_points below.

    robotics_sim/planning/exploration_planners.py:166
        _belief_from_kwargs() -> belief.mark_occupied_cell(cell) per point
        in kwargs["mapped_obstacle_points"] -- an EPHEMERAL, per-call
        BeliefMap used only by FrontierExplorationPlanner/GoalSeekingPlanner
        for frontier-goal selection. Classification: (4) legacy
        rasterization. It never invents occupancy: its OCCUPIED cells are
        derived 1:1 from points the caller already had. It is also fully
        disconnected from PlanningCostmapBuilder -- grep for ".snapshot()"
        in exploration_planners.py finds no matches, so this ephemeral
        belief never becomes an ExplorationMapSnapshot. See
        test_frontier_planner_ephemeral_belief_only_rasterizes_already_
        observed_points below.

    (all other matches are in test files -- classification (3)
    test/manual, out of scope for production-path risk.)

Other grep()s required by this audit and what they showed:

    "restore_grid_state"  -- engine.py:4725, inside the navigation-debug
        snapshot restore path. Classification: (2) restore/debug state.
        Always paired, in the SAME restore function, with
        self._truncate_mapped_obstacle_points(snapshot.mapped_obstacle_
        points_count) (engine.py:4830) -- both layers are rolled back
        together from the same historical capture. See
        test_map_snapshot_producers.py for the producer-level coverage of
        restore_grid_state()'s own revision policy; this file's Case 6
        below exercises _truncate_mapped_obstacle_points() itself, the
        piece that keeps mapped_obstacle_points in step with a restored
        belief.

    "record_explored_area" -- engine.py:6487. Marks FREE cells only (via
        update_explored_free_points_from_polygon()/mark_visible_polygon()
        -> mark_free_cell()), never OCCUPIED. Out of scope for this audit.

    "mapped_obstacle_points ="  -- besides update_sensed_obstacles()'s own
        lazy-init guard and reset_belief_map()'s coordinated reset (see
        Case 5), the one surprising hit is robotics_sim/app/simulation_
        canvas.py:955 (SimulationCanvas.mapped_obstacle_points) -- a
        DIFFERENT attribute on the Qt canvas object, a rendering-cache
        mirror fed downstream from update_sensed_obstacles()'s own output
        (self.canvas.append_mapped_obstacle_points(newly_mapped)). It never
        writes BeliefMap and is not a production source of occupancy.
        Classification: (6) otro / GUI-only.

Fakes below are lightweight duck-typed SimpleNamespace objects binding the
REAL SimulationControllerMixin methods under test (update_sensed_obstacles,
observed_obstacle_snapshot, reset_belief_map, _truncate_mapped_obstacle_
points, sync_legacy_map_views_from_belief) -- the same convention already
used by test_map_snapshot_producers.py / test_planning_map_
characterization.py / test_planning_costmap_builder.py. Sensor-geometry
collaborators (visible_candidate_obstacles/sample_obstacle_boundary_points/
point_visible_from_robot/quantize_map_point) are simple, controllable
stubs -- boundary sampling and visibility have their own coverage
elsewhere; this file is about whether sensing's OUTPUT reaches
mapped_obstacle_points/observed_obstacle_snapshot()/the builder, not about
re-deriving sampling geometry. No production code is modified or
reimplemented here: mark_occupied_cell/mark_occupied_points/
_truncate_mapped_obstacle_points/reset_belief_map/update_sensed_obstacles
are always the REAL bound methods.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.map_snapshots import ObservedObstacleSnapshot
from robotics_sim.environment.occupancy_grid import FREE as OG_FREE
from robotics_sim.environment.occupancy_grid import OCCUPIED as OG_OCCUPIED
from robotics_sim.environment.occupancy_grid import OccupancyGrid
from robotics_sim.planning.costmap_snapshot import OCCUPIED as COSTMAP_OCCUPIED
from robotics_sim.planning.exploration_planners import _belief_from_kwargs
from robotics_sim.planning.planning_costmap_builder import (
    PlanningCostmapBuilder,
    PlanningCostmapPolicy,
)
from robotics_sim.simulation.engine import SimulationControllerMixin

BOUNDS = (0.0, 10.0, 0.0, 10.0)
RESOLUTION = 1.0
ROBOT_RADIUS = 0.3


def _make_fake_engine(*, obstacles: list | None = None) -> SimpleNamespace:
    config = SimpleNamespace(
        grid_resolution=RESOLUTION,
        mapping_point_spacing=0.5,
        default_fire_intensity=1.0,
        default_fire_radius=2.0,
        fire_selection_radius=0.6,
        hazard_block_threshold=0.55,
        obstacles=list(obstacles or []),
    )
    robot = SimpleNamespace(x=0.0, y=0.0, theta=0.0, vision=3.0)
    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        config=config,
        canvas=SimpleNamespace(
            append_mapped_obstacle_points=lambda points: None,
            set_status=lambda message: None,
        ),
    )
    fake.visible_candidate_obstacles = lambda: [(5.0, 5.0, 1.0, 1.0)]
    fake.sample_obstacle_boundary_points = lambda obstacle, spacing: [(5.5, 5.0), (6.0, 5.0)]
    fake.point_visible_from_robot = lambda point, candidate_obstacles=None: True
    fake.quantize_map_point = lambda point, resolution: (round(float(point[0]), 3), round(float(point[1]), 3))
    fake.force_all_robot_poses_free_in_belief = lambda: 0

    for name in (
        "reset_belief_map",
        "ensure_belief_map",
        "sync_legacy_map_views_from_belief",
        "update_sensed_obstacles",
        "observed_obstacle_snapshot",
        "_truncate_mapped_obstacle_points",
        "push_discovered_hazard_frame",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    fake.reset_belief_map()
    return fake


# ---------------------------------------------------------------------------
# Case 1: sensor mapping produces observed geometry.
# ---------------------------------------------------------------------------


def test_update_sensed_obstacles_produces_observed_geometry_matching_mapped_points():
    fake = _make_fake_engine()
    snapshot_before = fake.observed_obstacle_snapshot()
    assert snapshot_before.points == ()
    assert snapshot_before.revision == 0

    newly_mapped = fake.update_sensed_obstacles(force_status=False)

    assert newly_mapped, "sanity: the stubbed sensor pipeline found new points"
    assert list(fake.mapped_obstacle_points) == list(newly_mapped)
    assert fake.mapped_obstacle_revision > 0

    snapshot_after = fake.observed_obstacle_snapshot()
    assert set(snapshot_after.points) == set(newly_mapped)
    assert snapshot_after.revision > snapshot_before.revision

    # The snapshot taken BEFORE update_sensed_obstacles() ran must remain
    # exactly what it was -- snapshots are immutable, not live views.
    assert snapshot_before.points == ()
    assert snapshot_before.revision == 0


# ---------------------------------------------------------------------------
# Case 2: observed geometry blocks the new builder.
# ---------------------------------------------------------------------------


def test_observed_geometry_from_sensing_blocks_the_new_builder():
    fake = _make_fake_engine()
    newly_mapped = fake.update_sensed_obstacles(force_status=False)
    observed = fake.observed_obstacle_snapshot()
    exploration = fake.belief_map.snapshot()

    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=ROBOT_RADIUS)
    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
    )

    reference_grid = OccupancyGrid.from_bounds(
        x_min=exploration.bounds[0], x_max=exploration.bounds[1],
        y_min=exploration.bounds[2], y_max=exploration.bounds[3],
        resolution=exploration.resolution, initial_value=OG_FREE, unknown_is_traversable=True,
    )
    reference_grid.add_obstacle_points(newly_mapped, padding=ROBOT_RADIUS)

    assert (result.grid == COSTMAP_OCCUPIED).any(), "sanity: something is actually occupied"
    assert np.array_equal(result.grid == COSTMAP_OCCUPIED, reference_grid.data == OG_OCCUPIED), (
        "the builder's footprint for real sensor-observed geometry must match a raw "
        "OccupancyGrid.add_obstacle_points() call with the same points/padding"
    )


# ---------------------------------------------------------------------------
# Case 3: belief occupancy alone does not block.
# ---------------------------------------------------------------------------


def test_belief_occupancy_alone_does_not_block_the_new_builder():
    fake = _make_fake_engine()
    newly_mapped = fake.update_sensed_obstacles(force_status=False)
    assert newly_mapped

    exploration = fake.belief_map.snapshot()
    assert (exploration.grid == 1).any(), "sanity: real, sensor-caused legacy-OCCUPIED cells exist"

    # Same BeliefMap/exploration as above, but the observed-obstacle layer
    # is discarded -- exactly the "occupancy without geometry" shape this
    # audit is checking every production path against.
    empty_observed = ObservedObstacleSnapshot(
        points=(), bounds=exploration.bounds, resolution=exploration.resolution, revision=0,
    )
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=ROBOT_RADIUS)

    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=empty_observed, policy=policy,
    )

    assert not (result.grid == COSTMAP_OCCUPIED).any(), (
        "PlanningCostmapBuilder must not block on legacy BeliefMap.OCCUPIED cells alone "
        "-- this is the deliberate new contract, see planning_costmap_builder.py's module "
        "docstring, 'Legacy belief occupancy vs. observed obstacle geometry'"
    )


# ---------------------------------------------------------------------------
# Case 4: the other production OCCUPIED-writing call site
# (exploration_planners._belief_from_kwargs) never invents occupancy and
# never reaches the builder.
# ---------------------------------------------------------------------------


def test_frontier_planner_ephemeral_belief_only_rasterizes_already_observed_points():
    """_belief_from_kwargs() is the only OTHER production call site of
    mark_occupied_cell/mark_occupied_points (besides update_sensed_
    obstacles()). It builds a throwaway BeliefMap, used only for frontier-
    goal selection, from an ALREADY-computed mapped_obstacle_points list the
    caller passes in -- proving its OCCUPIED cells never exceed the input
    points, so it cannot be an occupancy-only path even though it calls
    mark_occupied_cell() directly."""
    mapped_points = [(5.5, 5.5), (6.5, 5.5)]
    kwargs = {
        "bounds": BOUNDS,
        "resolution": RESOLUTION,
        "mapped_obstacle_points": mapped_points,
        "explored_points": [],
    }

    ephemeral_belief = _belief_from_kwargs(kwargs)

    assert isinstance(ephemeral_belief, BeliefMap)
    occupied_cells = {
        ephemeral_belief.world_to_cell(p) for p in ephemeral_belief.occupied_points()
    }
    expected_cells = {ephemeral_belief.world_to_cell(p) for p in mapped_points}
    assert occupied_cells == expected_cells, (
        "the ephemeral frontier-planner belief's OCCUPIED cells must derive exactly from "
        "the caller-supplied mapped_obstacle_points -- no cell becomes OCCUPIED without a "
        "corresponding input point"
    )


# ---------------------------------------------------------------------------
# Whole-package AST inventory for mark_occupied_cell()/mark_occupied_points()
# call sites -- everything below backs
# test_all_production_mark_occupied_call_sites_are_the_audited_set().
#
# A per-module (module_name, called_method) set (the previous version of
# this test) cannot detect: (1) a new call in a DIFFERENT function of an
# already-known module -- the pair is identical either way; (2) a SECOND
# call inside the SAME function -- a set collapses duplicates; (3) a new
# call site in any robotics_sim module other than the three already on the
# hardcoded import list -- nothing outside that list was ever scanned. The
# Counter over (module, enclosing_function, called_method) below, walked
# across every production .py file under robotics_sim/, fixes all three:
# a second call in the same function increments that key's count past 1
# (caught by Counter equality), and every module in the package is scanned,
# not a hardcoded three.
# ---------------------------------------------------------------------------

_TRACKED_OCCUPANCY_METHODS = ("mark_occupied_cell", "mark_occupied_points")
_EXCLUDED_DIR_NAMES = {"tests", "__pycache__"}
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # .../robotics_sim
_REPO_ROOT = _PACKAGE_ROOT.parent


class _MarkOccupiedCallVisitor(ast.NodeVisitor):
    """Records every call to .mark_occupied_cell(...)/.mark_occupied_points(...)
    in one module's AST, together with the name of its nearest enclosing
    def/async def ("<module>" for a call made at module or class-body scope,
    outside any function) and its line number. Two calls inside the SAME
    enclosing function are both recorded as separate entries here -- the
    caller's Counter is what turns a second one into a count of 2, not this
    visitor collapsing them.
    """

    def __init__(self) -> None:
        self._function_stack: list[str] = []
        self.calls: list[tuple[str, str, int]] = []  # (enclosing_function, called_method, lineno)

    def _visit_function(self, node) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in _TRACKED_OCCUPANCY_METHODS:
            enclosing_function = self._function_stack[-1] if self._function_stack else "<module>"
            self.calls.append((enclosing_function, node.func.attr, node.lineno))
        self.generic_visit(node)


def _iter_production_python_files():
    """Every .py file under robotics_sim/, excluding robotics_sim/tests/ and
    __pycache__/ (no other generated files exist in this tree today; the
    exclusion is expressed as a directory-name set, not a hardcoded file
    list, so it still applies if either directory reappears deeper in the
    tree)."""
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        relative_parts = path.relative_to(_PACKAGE_ROOT).parts
        if _EXCLUDED_DIR_NAMES.intersection(relative_parts):
            continue
        yield path


def _module_name_for_path(path: Path) -> str:
    parts = path.relative_to(_REPO_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _find_mark_occupied_calls(path: Path) -> list[tuple[str, str, int]]:
    """AST-parses one production file's own source text (never a text/
    substring search) and returns every (enclosing_function, called_method,
    lineno) for a call to mark_occupied_cell()/mark_occupied_points() found
    in it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _MarkOccupiedCallVisitor()
    visitor.visit(tree)
    return visitor.calls


def test_all_production_mark_occupied_call_sites_are_the_audited_set():
    """Regression tripwire for this audit's call-site inventory (see module
    docstring). Walks every production .py file under robotics_sim/ (never
    just the three modules known to call these methods today) and tallies a
    Counter over (module, enclosing_function, called_method). A brand-new
    call site anywhere in robotics_sim/, a new call in a different function
    of an already-known module, or a second call inside an already-known
    function all change this Counter and fail the test -- forcing that
    new/changed call site to be classified (sensor-derived / restore / test
    / legacy rasterization / ground-truth leak / other) before anyone
    assumes PlanningCostmapBuilder integration is still safe.
    """
    found: Counter[tuple[str, str, str]] = Counter()
    call_lines: dict[tuple[str, str, str], list[int]] = {}

    for path in _iter_production_python_files():
        module_name = _module_name_for_path(path)
        for enclosing_function, called_method, lineno in _find_mark_occupied_calls(path):
            key = (module_name, enclosing_function, called_method)
            found[key] += 1
            call_lines.setdefault(key, []).append(lineno)

    # Confirmed by this same AST walk (not assumed): belief_map.py's
    # mark_occupied_points() calls self.mark_occupied_cell() exactly once,
    # engine.py's update_sensed_obstacles() calls belief.mark_occupied_
    # points() exactly once, and exploration_planners.py's
    # _belief_from_kwargs() calls belief.mark_occupied_cell() exactly once
    # (inside a `for point in ...` loop, but that is ONE Call node in the
    # AST -- the loop's per-point repetition happens at runtime, not as
    # multiple call sites in source).
    audited = Counter(
        {
            ("robotics_sim.environment.belief_map", "mark_occupied_points", "mark_occupied_cell"): 1,
            ("robotics_sim.simulation.engine", "update_sensed_obstacles", "mark_occupied_points"): 1,
            ("robotics_sim.planning.exploration_planners", "_belief_from_kwargs", "mark_occupied_cell"): 1,
        }
    )

    if found != audited:
        all_keys = sorted(set(found) | set(audited))
        new_sites = {key: found[key] for key in all_keys if found[key] and not audited[key]}
        missing_sites = {key: audited[key] for key in all_keys if audited[key] and not found[key]}
        differing_counts = {
            key: {"expected": audited[key], "found": found[key]}
            for key in all_keys
            if found[key] != audited[key] and key not in new_sites and key not in missing_sites
        }
        lines_by_site = {key: call_lines.get(key, []) for key in all_keys}
        raise AssertionError(
            "unaudited BeliefMap.mark_occupied_cell()/mark_occupied_points() call site(s) "
            "detected under robotics_sim/ (production code only, robotics_sim/tests/ and "
            "__pycache__/ excluded).\n"
            f"new call sites (module, enclosing_function, called_method) -> count: {new_sites!r}\n"
            f"missing call sites (expected but not found): {missing_sites!r}\n"
            f"differing counts (expected vs. found): {differing_counts!r}\n"
            f"line numbers found per call site (for diagnosis): {lines_by_site!r}\n"
            "Classify any new/changed call site (sensor-derived / restore / test / legacy "
            "rasterization / ground-truth leak / other) before assuming PlanningCostmapBuilder "
            "integration is still safe."
        )


# ---------------------------------------------------------------------------
# Case 5: reset.
# ---------------------------------------------------------------------------


def test_reset_belief_map_clears_observed_points_and_keeps_layers_independent():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    assert fake.mapped_obstacle_points
    revision_before = fake.mapped_obstacle_revision
    belief_before = fake.belief_map

    fake.reset_belief_map()

    assert fake.mapped_obstacle_points == []
    assert fake.mapped_obstacle_revision > revision_before

    snapshot_after = fake.observed_obstacle_snapshot()
    assert snapshot_after.points == ()

    # Independent layers: a brand-new BeliefMap object with its OWN revision
    # counter continuing forward -- neither counter is derived from, or
    # reset by, the other.
    assert fake.belief_map is not belief_before
    assert fake.belief_map.revision > belief_before.revision


# ---------------------------------------------------------------------------
# Case 6: navigation snapshot rollback.
# ---------------------------------------------------------------------------


def test_truncate_mapped_obstacle_points_rolls_back_observed_geometry():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    assert len(fake.mapped_obstacle_points) == 2  # matches the stub's 2 sampled points

    snapshot_before_truncate = fake.observed_obstacle_snapshot()
    revision_before = fake.mapped_obstacle_revision

    changed = fake._truncate_mapped_obstacle_points(1)

    assert changed is True
    assert len(fake.mapped_obstacle_points) == 1
    assert fake.mapped_obstacle_points == list(snapshot_before_truncate.points[:1])
    assert fake.mapped_obstacle_revision > revision_before

    snapshot_after_truncate = fake.observed_obstacle_snapshot()
    assert snapshot_after_truncate.points == snapshot_before_truncate.points[:1]

    # The snapshot taken BEFORE truncation keeps its full, untruncated set.
    assert len(snapshot_before_truncate.points) == 2


# ---------------------------------------------------------------------------
# Case 7: hazard independence.
# ---------------------------------------------------------------------------


def test_hazard_observation_does_not_touch_mapped_obstacle_points_but_still_blocks():
    fake = _make_fake_engine()
    exploration = fake.belief_map.snapshot()

    hazard_row, hazard_col = 4, 4
    fake.hazard_service.belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)
    hazard_frame = fake.hazard_service.belief.snapshot()
    hazard_geometry = fake.hazard_service.belief.geometry

    observed = fake.observed_obstacle_snapshot()
    assert observed.points == (), "no obstacle points were ever mapped in this test"

    policy = PlanningCostmapPolicy(
        unknown_is_traversable=True, obstacle_padding=0.0,
        hazard_block_threshold=fake.hazard_service.block_threshold,
    )
    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
        hazard_belief=hazard_frame, hazard_geometry=hazard_geometry,
    )

    assert result.grid[hazard_row, hazard_col] == COSTMAP_OCCUPIED
    assert ("hazard", hazard_frame.revision) in result.source_revisions
    assert dict(result.source_revisions)["observed_obstacles"] == observed.revision
    assert observed.points == (), "the hazard build() call must never populate observed obstacles"
    assert fake.mapped_obstacle_points == [], "hazard observation must never touch mapped_obstacle_points"


# ---------------------------------------------------------------------------
# Case 8: ground truth exclusion.
# ---------------------------------------------------------------------------


def test_ground_truth_obstacles_never_enter_observed_snapshot_without_sensing():
    ground_truth_rect = (4.0, 4.0, 2.0, 2.0)  # x, y, width, height
    fake = _make_fake_engine(obstacles=[ground_truth_rect])

    observed = fake.observed_obstacle_snapshot()
    assert observed.points == (), (
        "config.obstacles must never leak into observed_obstacle_snapshot() -- it only "
        "ever reads mapped_obstacle_points, which stays empty until sensing runs"
    )

    exploration = fake.belief_map.snapshot()
    policy = PlanningCostmapPolicy(unknown_is_traversable=True, obstacle_padding=0.0)
    result = PlanningCostmapBuilder().build(
        exploration=exploration, observed_obstacles=observed, policy=policy,
    )

    # Proven by EXECUTION, not by reading source: densely sample the
    # ground-truth rectangle's own world-space footprint and confirm none of
    # the cells it maps to are occupied in the builder's actual output.
    x, y, w, h = ground_truth_rect
    geometry = GridGeometry(exploration.bounds, exploration.resolution)
    rect_cells = set()
    steps = 5
    for i in range(steps + 1):
        for j in range(steps + 1):
            cell = geometry.world_to_grid(x + w * i / steps, y + h * j / steps)
            if cell is not None:
                rect_cells.add((cell.row, cell.col))

    assert rect_cells, "sanity: the rectangle actually maps to real grid cells"
    for row, col in rect_cells:
        assert result.grid[row, col] != COSTMAP_OCCUPIED, (
            f"cell ({row},{col}) inside ground-truth rectangle {ground_truth_rect} must not "
            "be occupied -- PlanningCostmapBuilder has no ground-truth parameter at all (see "
            "test_build_signature_has_no_ground_truth_parameter in "
            "test_planning_costmap_builder.py) and observed_obstacle_snapshot() never reads "
            "config.obstacles either"
        )
    assert not (result.grid == COSTMAP_OCCUPIED).any(), (
        "with no sensing run and no hazard, the entire builder output must be free of occupancy"
    )
