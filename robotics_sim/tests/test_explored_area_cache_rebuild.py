"""
Tests for reconstructing the visible explored-area layer after a pixmap
cache invalidation (theme toggle / resize / pan-zoom / a fresh-run
BeliefMap replacement) -- see simulation_canvas.SimulationCanvas.
rebuild_explored_area_cache() / set_explored_area_seed() / append_explored_
area_polygon(), and engine.SimulationControllerMixin._publish_explored_
area_source_to_canvas().

The bug this guards against: previously, only a *restored snapshot* ever
seeded _explored_area_seed_mask. A normal/live run relied solely on the
bounded explored_area_polygons history (EXPLORED_POLYGON_HISTORY_LIMIT), so
any cache invalidation (theme toggle, resize, pan/zoom) rebuilt from that
bounded history alone -- coverage recorded more than
EXPLORED_POLYGON_HISTORY_LIMIT sensor sweeps ago silently vanished. Multi-
robot made it worse: invalidate_explored_area_cache() drops
_explored_area_caches_by_robot entirely, and draw_explored_area_trace() used
to check that dict for emptiness *before* rebuilding it, so the first
append_explored_area_polygon(robot_index=...) after an invalidation created
only its own cache and hid every other robot's still-uncached coverage.

The fix: engine.py now publishes the LIVE belief_map.explored_by_robot mask
to the canvas once per fresh run (not just on snapshot restore), and the
canvas rebuilds its cache(s) directly from that authoritative mask instead
of the bounded polygon history.

The first test below (test_theme_change_does_not_mutate_belief_or_frontiers)
is the characterization test required before touching production code: it
proves a theme change is purely cosmetic and never mutates the BeliefMap it
was seeded from, or the frontier cells computed from that belief.
"""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.app.theme import ThemeMode
from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.planning.exploration_planners import _frontier_cells
from robotics_sim.simulation.config import EXPLORED_POLYGON_HISTORY_LIMIT
from robotics_sim.simulation.engine import SimulationControllerMixin

_app = QApplication.instance() or QApplication([])

_BOUNDS = (-2.0, 2.0, -2.0, 2.0)
_RESOLUTION = 1.0
_GRID_SHAPE = (4, 4)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_canvas() -> SimulationCanvas:
    canvas = SimulationCanvas()
    canvas.resize(300, 240)
    canvas.navigation_debug_enabled = False  # LIVE view, not history replay
    return canvas


def _paint_once(canvas: SimulationCanvas, method_name: str = "draw_explored_area_trace") -> None:
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    getattr(canvas, method_name)(painter)
    painter.end()


def _cell_center(row: int, col: int) -> tuple[float, float]:
    x_min, _x_max, y_min, _y_max = _BOUNDS
    return (x_min + (col + 0.5) * _RESOLUTION, y_min + (row + 0.5) * _RESOLUTION)


def _square_polygon(cx: float, cy: float, half: float = 0.4) -> list[tuple[float, float]]:
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def _sample_alpha(canvas: SimulationCanvas, cache: QPixmap | None, world_xy: tuple[float, float]) -> int:
    """Alpha channel at world_xy, read back from a rendered cache -- proof a
    cell is actually painted, not just that a cache object exists.

    Uses QImage.pixelColor(), not QColor(image.pixel(x, y)) -- the latter
    does not unpack the alpha channel from a packed ARGB32_Premultiplied
    value the way one might expect (it reads back 255 even for a fully
    transparent pixel), which would make every alpha assertion here
    vacuously true.
    """
    if cache is None:
        return 0
    sx, sy = canvas.world_to_screen(*world_xy)
    image = cache.toImage()
    x, y = int(sx), int(sy)
    if x < 0 or y < 0 or x >= image.width() or y >= image.height():
        return 0
    return image.pixelColor(x, y).alpha()


def _make_belief(robot_count: int = 1) -> BeliefMap:
    return BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=robot_count)


# ---------------------------------------------------------------------------
# 1. Theme change must be purely cosmetic: the BeliefMap it was seeded from,
# and the frontier cells computed from that belief, must be byte/value
# identical before and after a LIGHT -> DARK -> LIGHT round trip that forces
# a real explored-area cache rebuild in each mode.
# ---------------------------------------------------------------------------


def test_theme_change_does_not_mutate_belief_or_frontiers():
    belief = _make_belief(robot_count=1)
    # Free cells adjacent to UNKNOWN neighbors so a real frontier exists;
    # one occupied cell so belief.grid has more than one distinct value.
    belief.mark_free_cell((0, 0), robot_index=0, time_s=1.0)
    belief.mark_free_cell((0, 1), robot_index=0, time_s=1.0)
    belief.mark_occupied_cell((2, 2), time_s=1.0)

    canvas = _make_canvas()
    canvas.set_explored_area_seed(belief.explored_by_robot, belief.resolution, belief.bounds)
    _paint_once(canvas)

    grid_before = belief.grid.copy()
    explored_before = belief.explored_by_robot.copy()
    visit_before = belief.visit_count.copy()
    last_seen_before = belief.last_seen.copy()
    revision_before = belief.revision
    frontier_before = _frontier_cells(belief)
    assert frontier_before, "test setup must produce at least one real frontier cell"

    canvas.set_theme_mode(ThemeMode.DARK)
    canvas.rebuild_explored_area_cache()  # force a real rebuild under DARK
    _paint_once(canvas)

    canvas.set_theme_mode(ThemeMode.LIGHT)
    canvas.rebuild_explored_area_cache()  # force a real rebuild back under LIGHT
    _paint_once(canvas)

    assert np.array_equal(belief.grid, grid_before)
    assert np.array_equal(belief.explored_by_robot, explored_before)
    assert np.array_equal(belief.visit_count, visit_before)
    assert np.array_equal(belief.last_seen, last_seen_before)
    assert belief.revision == revision_before
    assert _frontier_cells(belief) == frontier_before

    # The canvas must keep the exact same ndarray reference throughout --
    # never a copy, never a different array from a "fresh" belief.
    assert canvas._explored_area_seed_mask is belief.explored_by_robot


# ---------------------------------------------------------------------------
# 2. Single robot: coverage older than EXPLORED_POLYGON_HISTORY_LIMIT sweeps
# must survive a cache rebuild because it comes from the authoritative mask,
# not from the bounded polygon history.
# ---------------------------------------------------------------------------


def test_single_robot_history_beyond_limit_stays_visible_after_rebuild():
    belief = _make_belief(robot_count=1)
    old_cell = (0, 0)
    old_world = _cell_center(*old_cell)
    belief.mark_free_cell(old_cell, robot_index=0, time_s=0.0)

    canvas = _make_canvas()
    canvas.set_explored_area_seed(belief.explored_by_robot, belief.resolution, belief.bounds)

    old_polygon = _square_polygon(*old_world)
    canvas.append_explored_area_polygon(old_polygon, robot_index=None)

    for _ in range(EXPLORED_POLYGON_HISTORY_LIMIT + 5):
        canvas.append_explored_area_polygon(_square_polygon(1.5, 1.5), robot_index=None)

    assert old_polygon not in canvas.explored_area_polygons
    assert len(canvas.explored_area_polygons) <= EXPLORED_POLYGON_HISTORY_LIMIT

    assert _sample_alpha(canvas, canvas._explored_area_cache, old_world) > 0

    # Theme toggle invalidates and forces a rebuild -- coverage must survive.
    canvas.set_theme_mode(ThemeMode.DARK)
    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, old_world) > 0


# ---------------------------------------------------------------------------
# 3. Multi-robot: history beyond the limit for BOTH robots must survive a
# rebuild, each still attributed to its own per-robot cache.
# ---------------------------------------------------------------------------


def test_multi_robot_history_beyond_limit_stays_visible_and_attributed():
    belief = _make_belief(robot_count=2)
    old_cell_r0 = (0, 0)
    old_cell_r1 = (3, 3)
    old_world_r0 = _cell_center(*old_cell_r0)
    old_world_r1 = _cell_center(*old_cell_r1)
    belief.mark_free_cell(old_cell_r0, robot_index=0, time_s=0.0)
    belief.mark_free_cell(old_cell_r1, robot_index=1, time_s=0.0)

    canvas = _make_canvas()
    canvas.set_explored_area_seed(belief.explored_by_robot, belief.resolution, belief.bounds)

    # Force the initial mask-based rebuild (mirrors what the real paint
    # loop does via draw_explored_area_trace()/ensure_explored_area_cache()).
    _paint_once(canvas)
    assert set(canvas._explored_area_caches_by_robot) == {0, 1}

    for _ in range(EXPLORED_POLYGON_HISTORY_LIMIT + 5):
        canvas.append_explored_area_polygon(_square_polygon(-0.5, -0.5), robot_index=0)
        canvas.append_explored_area_polygon(_square_polygon(0.5, 0.5), robot_index=1)

    assert len(canvas.explored_area_polygons) <= EXPLORED_POLYGON_HISTORY_LIMIT

    canvas.set_theme_mode(ThemeMode.DARK)
    _paint_once(canvas)

    cache0 = canvas._explored_area_caches_by_robot.get(0)
    cache1 = canvas._explored_area_caches_by_robot.get(1)
    assert cache0 is not None and cache1 is not None
    assert _sample_alpha(canvas, cache0, old_world_r0) > 0
    assert _sample_alpha(canvas, cache1, old_world_r1) > 0
    # Attribution: robot 0's old zone must not bleed into robot 1's cache.
    assert _sample_alpha(canvas, cache1, old_world_r0) == 0
    assert _sample_alpha(canvas, cache0, old_world_r1) == 0


# ---------------------------------------------------------------------------
# 4. Multi-robot: appending a NEW polygon for one robot right after a
# theme-toggle rebuild must not hide the other (already-cached-from-mask)
# robot's historical coverage.
# ---------------------------------------------------------------------------


def test_multi_robot_append_after_rebuild_keeps_other_robot_visible():
    belief = _make_belief(robot_count=2)
    old_cell_r0 = (0, 0)
    old_cell_r1 = (3, 3)
    old_world_r0 = _cell_center(*old_cell_r0)
    old_world_r1 = _cell_center(*old_cell_r1)
    belief.mark_free_cell(old_cell_r0, robot_index=0, time_s=0.0)
    belief.mark_free_cell(old_cell_r1, robot_index=1, time_s=0.0)

    canvas = _make_canvas()
    canvas.set_explored_area_seed(belief.explored_by_robot, belief.resolution, belief.bounds)
    _paint_once(canvas)
    assert set(canvas._explored_area_caches_by_robot) == {0, 1}

    # Theme toggle invalidates everything, including per-robot caches.
    canvas.set_theme_mode(ThemeMode.DARK)
    assert canvas._explored_area_caches_by_robot == {}

    # A single new sweep for robot 0 only -- no full repaint/ensure call in
    # between, exactly like the live engine calling record_explored_area()
    # for one robot at a time.
    canvas.append_explored_area_polygon(_square_polygon(-1.9, -1.9, half=0.05), robot_index=0)

    _paint_once(canvas)

    cache0 = canvas._explored_area_caches_by_robot.get(0)
    cache1 = canvas._explored_area_caches_by_robot.get(1)
    assert cache0 is not None
    assert cache1 is not None, (
        "robot 1's cache must have been rebuilt from the mask too, not "
        "dropped just because only robot 0 appended a new sweep"
    )
    assert _sample_alpha(canvas, cache0, old_world_r0) > 0
    assert _sample_alpha(canvas, cache1, old_world_r1) > 0


# ---------------------------------------------------------------------------
# 5. Resize / explicit cache invalidation: full coverage must be rebuilt
# from the mask, not from the last EXPLORED_POLYGON_HISTORY_LIMIT polygons.
# ---------------------------------------------------------------------------


def test_resize_invalidation_rebuilds_full_coverage_from_mask_not_polygons():
    belief = _make_belief(robot_count=1)
    old_cell = (0, 0)
    old_world = _cell_center(*old_cell)
    belief.mark_free_cell(old_cell, robot_index=0, time_s=0.0)

    canvas = _make_canvas()
    canvas.set_explored_area_seed(belief.explored_by_robot, belief.resolution, belief.bounds)
    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, old_world) > 0

    # No polygons at all recorded -- if the rebuild depended on
    # explored_area_polygons, coverage would now be empty.
    assert canvas.explored_area_polygons == []

    canvas.invalidate_explored_area_cache()
    assert canvas._explored_area_cache is None

    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, old_world) > 0


# ---------------------------------------------------------------------------
# 6. Fresh-run source replacement: engine._publish_explored_area_source_to_
# canvas() must point the canvas at the NEW BeliefMap's mask, never leaving
# it referencing a previous (replaced) run's BeliefMap.
# ---------------------------------------------------------------------------


def test_publish_explored_area_source_replaces_previous_run_mask():
    canvas = _make_canvas()

    belief_a = _make_belief(robot_count=1)
    belief_a.mark_free_cell((0, 0), robot_index=0, time_s=0.0)
    world_a = _cell_center(0, 0)

    host = SimpleNamespace(belief_map=belief_a, canvas=canvas)
    SimulationControllerMixin._publish_explored_area_source_to_canvas(host)

    assert canvas._explored_area_seed_mask is belief_a.explored_by_robot
    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, world_a) > 0

    # A fresh run replaces the BeliefMap instance outright (see
    # engine.reset_belief_map()) -- belief_b starts empty.
    belief_b = _make_belief(robot_count=1)
    host.belief_map = belief_b
    SimulationControllerMixin._publish_explored_area_source_to_canvas(host)

    assert canvas._explored_area_seed_mask is belief_b.explored_by_robot
    assert canvas._explored_area_seed_mask is not belief_a.explored_by_robot

    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, world_a) == 0


# ---------------------------------------------------------------------------
# 7. Restore compatibility: set_explored_area_seed() with a restored
# snapshot's mask still reconstructs coverage correctly.
# ---------------------------------------------------------------------------


def test_restored_snapshot_seed_still_reconstructs_coverage():
    canvas = _make_canvas()
    mask = np.zeros((1,) + _GRID_SHAPE, dtype=bool)
    mask[0, 1, 1] = True
    restored_world = _cell_center(1, 1)

    canvas.set_explored_area_seed(mask, _RESOLUTION, _BOUNDS)
    _paint_once(canvas)

    assert canvas._explored_area_cache is not None
    assert _sample_alpha(canvas, canvas._explored_area_cache, restored_world) > 0


# ---------------------------------------------------------------------------
# 8. Polygon history stays bounded -- the fix must not switch to storing
# polygons without limit.
# ---------------------------------------------------------------------------


def test_polygon_history_stays_bounded():
    canvas = _make_canvas()
    for _ in range(EXPLORED_POLYGON_HISTORY_LIMIT * 3):
        canvas.append_explored_area_polygon(_square_polygon(0.0, 0.0), robot_index=None)
    assert len(canvas.explored_area_polygons) <= EXPLORED_POLYGON_HISTORY_LIMIT


# ---------------------------------------------------------------------------
# 9. The canvas never mutates the source mask -- rebuild/theme-toggle/
# append/draw must all leave it byte-identical.
# ---------------------------------------------------------------------------


def test_canvas_never_mutates_source_mask():
    belief = _make_belief(robot_count=2)
    belief.mark_free_cell((0, 0), robot_index=0, time_s=0.0)
    belief.mark_free_cell((3, 3), robot_index=1, time_s=0.0)
    mask = belief.explored_by_robot
    snapshot = mask.copy()

    canvas = _make_canvas()
    canvas.set_explored_area_seed(mask, belief.resolution, belief.bounds)
    assert np.array_equal(mask, snapshot)

    canvas.rebuild_explored_area_cache()
    assert np.array_equal(mask, snapshot)

    canvas.set_theme_mode(ThemeMode.DARK)
    assert np.array_equal(mask, snapshot)

    canvas.append_explored_area_polygon(_square_polygon(0.0, 0.0), robot_index=0)
    assert np.array_equal(mask, snapshot)

    _paint_once(canvas)
    assert np.array_equal(mask, snapshot)

    # Still the SAME object -- a read-only reference, never a defensive
    # per-tick copy.
    assert canvas._explored_area_seed_mask is mask


# ---------------------------------------------------------------------------
# 10. No frontend/frontier coupling: simulation_canvas.py must not import
# exploration/frontier/engine/BeliefMap internals -- it only ever receives
# plain ndarray/resolution/bounds data.
# ---------------------------------------------------------------------------


def test_canvas_module_does_not_import_frontier_or_engine_internals():
    module_path = inspect.getfile(SimulationCanvas)
    with open(module_path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=module_path)

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    forbidden_substrings = (
        "exploration_planners",
        "coordinated_frontier_planner",
        "simulation.engine",
        "environment.belief_map",
    )
    for module in imported_modules:
        for forbidden in forbidden_substrings:
            assert forbidden not in module, (module, forbidden)
