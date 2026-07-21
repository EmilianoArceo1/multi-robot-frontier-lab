"""
Tests for the continuous, world-coordinate explored-area geometry
(SimulationCanvas._explored_area_paths_by_robot) that rebuild_explored_
area_cache() now prefers over the discrete belief_map.explored_by_robot
seed mask (see fix/explored-area-cache-rebuild's 31a1e4b).

Problem this fixes: 31a1e4b correctly stopped coverage from vanishing after
a cache invalidation (theme toggle/resize/pan/zoom), but its rebuild
replayed the DISCRETE seed mask -- one QPainter.drawRect() per True belief
cell. During a live run the canvas actually paints smooth, continuous FoV
polygons (diagonal edges, rounded/angled sensor footprints); after any
invalidation, the mask-based rebuild replaced that smooth silhouette with
grid-cell-quantized rectangles -- visually "melting" into a staircase every
time the user toggled the theme.

Fix: SimulationCanvas.append_explored_area_polygon() now ALSO accumulates
each polygon into a per-robot, world-coordinate QPainterPath (_explored_
area_paths_by_robot) -- cheap, O(1) per sweep (just one more closed
subpath, never united()/simplified()/rebuilt). rebuild_explored_area_cache()
replays that continuous path (transformed to screen space with the current
view transform) whenever it exists for a robot, falling back to the
discrete mask only for a robot that has no continuous path of its own yet
(e.g. right after a snapshot restore, which does not store continuous
geometry -- see engine.restore_navigation_debug_snapshot()'s updated
docstring). The two are never combined for the same robot in the same
rebuild.

BeliefMap remains the sole discrete/logical authority (UNKNOWN/FREE/
OCCUPIED, explored_by_robot, frontiers, metrics, planning) -- this
geometry is exclusively additive and visual: it never feeds back into
frontiers, coverage logic, planning, or the belief map (see test 3's "no
visual coupling"-equivalent framing throughout: every assertion here reads
canvas-only state).
"""
from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.app.theme import ThemeMode
from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.simulation.engine import SimulationControllerMixin

_app = QApplication.instance() or QApplication([])


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


def _sample_alpha(canvas: SimulationCanvas, cache: QPixmap | None, world_xy: tuple[float, float]) -> int:
    """Alpha channel at world_xy, read back from a rendered cache.

    Uses QImage.pixelColor(), not QColor(image.pixel(x, y)) -- the latter
    does not unpack the alpha channel from a packed ARGB32_Premultiplied
    value the way one might expect (reads back 255 even for a fully
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


def _painted_world_points(canvas: SimulationCanvas, cache: QPixmap, points: list[tuple[float, float]]) -> frozenset:
    """The subset of `points` (world coordinates) that render as painted
    (alpha > 0) in `cache` -- an "alpha mask", expressed in world
    coordinates so it is comparable across a resize/pan/zoom that changes
    the screen-space projection but not the underlying geometry."""
    return frozenset(p for p in points if _sample_alpha(canvas, cache, p) > 0)


def _diagonal_triangle_polygon() -> list[tuple[float, float]]:
    """A right triangle with a diagonal hypotenuse along x + y = 3, not
    aligned to any grid cell boundary -- the shape under test throughout
    this file."""
    return [(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)]


def _path_world_points(path: QPainterPath) -> list[tuple[float, float]]:
    return [(path.elementAt(i).x, path.elementAt(i).y) for i in range(path.elementCount())]


def _publish_source(canvas: SimulationCanvas, belief: BeliefMap) -> None:
    """Simulate engine._publish_explored_area_source_to_canvas() without a
    full engine -- same duck-typed-fake pattern as test_explored_area_
    cache_rebuild.py."""
    host = SimpleNamespace(belief_map=belief, canvas=canvas)
    SimulationControllerMixin._publish_explored_area_source_to_canvas(host)


# ---------------------------------------------------------------------------
# 1. Continuous geometry survives a LIGHT -> DARK -> LIGHT round trip:
# extent, silhouette, and diagonal precision (no grid staircase) all
# survive. RGB is not compared -- the palette legitimately changes.
# ---------------------------------------------------------------------------


def test_continuous_geometry_survives_theme_round_trip():
    canvas = _make_canvas()
    triangle = _diagonal_triangle_polygon()
    canvas.append_explored_area_polygon(triangle, robot_index=None)
    _paint_once(canvas)
    cache_before = canvas._explored_area_cache
    assert cache_before is not None

    # A coarse grid of probe points across the triangle's bounding box,
    # expressed in world coordinates so the same probes are meaningful
    # both before and after.
    probes = [
        (x / 10.0, y / 10.0)
        for x in range(-5, 35)
        for y in range(-5, 35)
    ]
    painted_before = _painted_world_points(canvas, cache_before, probes)
    assert painted_before, "test setup must produce at least some painted area"

    # Sub-cell diagonal precision, BEFORE toggling: two points inside the
    # same 1x1 grid cell (x,y in [1,2]) but on opposite sides of the
    # x+y=3 hypotenuse must be classified differently. A discrete,
    # cell-quantized renderer could never do this (a whole cell is either
    # fully painted or not); only a true continuous edge can.
    inside_point = (1.2, 1.2)   # 1.2 + 1.2 = 2.4 < 3 -> inside the triangle
    outside_point = (1.8, 1.8)  # 1.8 + 1.8 = 3.6 > 3 -> outside the triangle
    assert _sample_alpha(canvas, cache_before, inside_point) > 0
    assert _sample_alpha(canvas, cache_before, outside_point) == 0

    canvas.set_theme_mode(ThemeMode.DARK)
    _paint_once(canvas)
    cache_dark = canvas._explored_area_cache
    assert cache_dark is not None
    assert cache_dark is not cache_before  # invalidated -> a real rebuild happened

    canvas.set_theme_mode(ThemeMode.LIGHT)
    _paint_once(canvas)
    cache_after = canvas._explored_area_cache
    assert cache_after is not None

    painted_after = _painted_world_points(canvas, cache_after, probes)

    # Extension/silhouette preserved exactly (same geometry, same view
    # transform -- alpha VALUES may differ by theme, the painted/unpainted
    # boolean set must not).
    assert painted_after == painted_before

    # Diagonal precision preserved -- no "melting" into grid steps.
    assert _sample_alpha(canvas, cache_after, inside_point) > 0
    assert _sample_alpha(canvas, cache_after, outside_point) == 0

    # The continuous path itself must never be touched by a theme change.
    assert canvas._explored_area_paths_by_robot.get(None) is not None
    assert not canvas._explored_area_paths_by_robot[None].isEmpty()


# ---------------------------------------------------------------------------
# 2. Continuous geometry is preferred over a discrete seed mask covering a
# different area -- a rebuild must reflect the path, not the mask.
# ---------------------------------------------------------------------------


def test_continuous_geometry_preferred_over_discrete_seed_mask():
    belief = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=1.0, robot_count=1)
    # A big square block, entirely disjoint from the triangle below.
    for x in range(-8, -3):
        for y in range(-8, -3):
            cell = belief.world_to_cell((float(x), float(y)))
            if cell is not None:
                belief.mark_free_cell(cell)
    mask_point = (-6.0, -6.0)  # deep inside the mask's square

    canvas = _make_canvas()
    canvas.set_explored_area_seed(belief.explored_by_robot, belief.resolution, belief.bounds)

    triangle = _diagonal_triangle_polygon()
    canvas.append_explored_area_polygon(triangle, robot_index=None)
    _paint_once(canvas)

    cache = canvas._explored_area_cache
    assert cache is not None
    # The continuous path's own area is painted...
    assert _sample_alpha(canvas, cache, (1.0, 1.0)) > 0
    # ...but the discrete mask's disjoint square is NOT -- once a
    # continuous path exists for this robot, the mask is not also painted.
    assert _sample_alpha(canvas, cache, mask_point) == 0


# ---------------------------------------------------------------------------
# 3. Seed fallback preserved: with no continuous path at all, a discrete
# seed mask alone still reconstructs coverage on rebuild -- 31a1e4b's
# behavior is unchanged for this case.
# ---------------------------------------------------------------------------


def test_seed_mask_fallback_preserved_without_a_continuous_path():
    belief = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=1.0, robot_count=1)
    cell = belief.world_to_cell((2.0, 2.0))
    belief.mark_free_cell(cell, robot_index=0, time_s=0.0)
    mask_world = belief.cell_to_world(cell)

    canvas = _make_canvas()
    canvas.set_explored_area_seed(belief.explored_by_robot, belief.resolution, belief.bounds)
    assert canvas._explored_area_paths_by_robot == {}

    _paint_once(canvas)

    cache = canvas._explored_area_cache
    assert cache is not None
    assert _sample_alpha(canvas, cache, mask_world) > 0

    # Survives an invalidation (theme/resize/pan/zoom) too, exactly like
    # before this geometry existed.
    canvas.set_theme_mode(ThemeMode.DARK)
    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, mask_world) > 0


# ---------------------------------------------------------------------------
# 4. Multi-robot: each robot's continuous path survives a theme toggle
# independently, with correct attribution.
# ---------------------------------------------------------------------------


def test_multi_robot_continuous_paths_survive_theme_toggle():
    canvas = _make_canvas()
    triangle_r0 = _diagonal_triangle_polygon()
    triangle_r1 = [(x - 6.0, y - 6.0) for x, y in _diagonal_triangle_polygon()]
    point_r0 = (1.0, 1.0)
    point_r1 = (-5.0, -5.0)

    canvas.append_explored_area_polygon(triangle_r0, robot_index=0)
    canvas.append_explored_area_polygon(triangle_r1, robot_index=1)
    _paint_once(canvas)
    assert set(canvas._explored_area_caches_by_robot) == {0, 1}

    canvas.set_theme_mode(ThemeMode.DARK)
    _paint_once(canvas)

    cache0 = canvas._explored_area_caches_by_robot.get(0)
    cache1 = canvas._explored_area_caches_by_robot.get(1)
    assert cache0 is not None and cache1 is not None

    assert _sample_alpha(canvas, cache0, point_r0) > 0
    assert _sample_alpha(canvas, cache1, point_r1) > 0
    # Each cache holds ONLY its own robot's geometry.
    assert _sample_alpha(canvas, cache0, point_r1) == 0
    assert _sample_alpha(canvas, cache1, point_r0) == 0

    # Both continuous paths themselves are untouched by the toggle.
    assert not canvas._explored_area_paths_by_robot[0].isEmpty()
    assert not canvas._explored_area_paths_by_robot[1].isEmpty()


# ---------------------------------------------------------------------------
# 5. Append after rebuild: robot 0 keeps old + new geometry; robot 1 keeps
# its full geometry untouched.
# ---------------------------------------------------------------------------


def test_append_after_rebuild_keeps_old_and_new_geometry_per_robot():
    canvas = _make_canvas()
    old_point_r0 = (1.0, 1.0)
    old_point_r1 = (-5.0, -5.0)
    triangle_r0 = _diagonal_triangle_polygon()
    triangle_r1 = [(x - 6.0, y - 6.0) for x, y in _diagonal_triangle_polygon()]

    canvas.append_explored_area_polygon(triangle_r0, robot_index=0)
    canvas.append_explored_area_polygon(triangle_r1, robot_index=1)
    _paint_once(canvas)
    assert set(canvas._explored_area_caches_by_robot) == {0, 1}

    canvas.invalidate_explored_area_cache()
    assert canvas._explored_area_caches_by_robot == {}

    new_point_r0 = (7.0, 7.0)  # (7-6)+(7-6) = 2 < 3 -> inside the shifted triangle
    new_triangle_r0 = [(x + 6.0, y + 6.0) for x, y in _diagonal_triangle_polygon()]
    canvas.append_explored_area_polygon(new_triangle_r0, robot_index=0)

    _paint_once(canvas)

    cache0 = canvas._explored_area_caches_by_robot.get(0)
    cache1 = canvas._explored_area_caches_by_robot.get(1)
    assert cache0 is not None
    assert cache1 is not None

    assert _sample_alpha(canvas, cache0, old_point_r0) > 0, "robot 0 keeps its old geometry"
    assert _sample_alpha(canvas, cache0, new_point_r0) > 0, "robot 0 gets the new geometry too"
    assert _sample_alpha(canvas, cache1, old_point_r1) > 0, "robot 1's geometry is fully untouched"


# ---------------------------------------------------------------------------
# 6. Resize/pan/zoom: the world-coordinate path itself never changes; only
# the screen-space projection does.
# ---------------------------------------------------------------------------


def test_resize_changes_projection_not_world_path():
    """Note: canvas.resize(...) does not reliably change canvas.size() in
    this offscreen/headless test harness (no parent layout, no show()) --
    the same reason every fixture in this file/its sibling calls resize()
    once at construction only as a best-effort hint, never asserts on it
    taking effect, and drives real invalidation explicitly instead (see
    test_explored_area_cache_rebuild.py's own resize/pan/zoom test). This
    test does the same: it invalidates explicitly (the real callback any
    of resize/pan/zoom ultimately triggers -- see resizeEvent()/
    invalidate_view_transform_caches()) and exercises an actual pan+zoom
    via the camera config fields, which reliably changes world_to_screen()
    output regardless of widget size.
    """
    canvas = _make_canvas()
    triangle = _diagonal_triangle_polygon()
    canvas.append_explored_area_polygon(triangle, robot_index=None)
    _paint_once(canvas)

    world_path = canvas._explored_area_paths_by_robot[None]
    points_before = _path_world_points(world_path)
    probe_world = (1.0, 1.0)
    screen_before = canvas.world_to_screen(*probe_world)
    assert _sample_alpha(canvas, canvas._explored_area_cache, probe_world) > 0

    # Simulate what resizeEvent()/a pan/zoom does: invalidate the render
    # cache outright -- the world path itself must be untouched.
    canvas.invalidate_explored_area_cache()
    assert canvas._explored_area_cache is None

    world_path_after = canvas._explored_area_paths_by_robot[None]
    points_after = _path_world_points(world_path_after)
    assert points_after == points_before, "the world-coordinate path must never change on invalidation"

    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, probe_world) > 0

    # A real pan+zoom via the camera config -- changes world_to_screen()'s
    # mapping for the exact same world point, without touching the path.
    canvas.config.camera_center_x += 2.0
    canvas.config.camera_width *= 0.5
    canvas.invalidate_view_transform_caches()
    assert canvas._explored_area_cache is None

    world_path_after_pan = canvas._explored_area_paths_by_robot[None]
    assert _path_world_points(world_path_after_pan) == points_before

    screen_after_pan = canvas.world_to_screen(*probe_world)
    assert screen_after_pan != screen_before, "pan/zoom must actually change the projection"

    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, probe_world) > 0


# ---------------------------------------------------------------------------
# Extra: fresh-run replacement clears geometry (clear_explored_area_
# geometry(), called by engine._publish_explored_area_source_to_canvas())
# and never leaves a previous run's continuous path visible.
# ---------------------------------------------------------------------------


def test_fresh_run_clears_previous_continuous_geometry():
    canvas = _make_canvas()
    belief_a = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=1.0, robot_count=1)
    _publish_source(canvas, belief_a)

    triangle = _diagonal_triangle_polygon()
    canvas.append_explored_area_polygon(triangle, robot_index=None)
    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, (1.0, 1.0)) > 0
    assert not canvas._explored_area_paths_by_robot.get(None).isEmpty()

    # A fresh run replaces the BeliefMap outright and re-publishes -- this
    # must never leave the previous run's smooth geometry on screen.
    belief_b = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=1.0, robot_count=1)
    _publish_source(canvas, belief_b)

    assert canvas._explored_area_paths_by_robot == {}
    assert canvas.explored_area_polygons == []
    assert canvas._explored_area_seed_mask is belief_b.explored_by_robot

    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, (1.0, 1.0)) == 0
