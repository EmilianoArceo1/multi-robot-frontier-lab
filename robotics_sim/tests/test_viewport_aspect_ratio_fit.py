"""
Tests for the uniform-scale viewport aspect-ratio fit
(simulation_canvas._fit_world_span_to_plot_aspect() /
SimulationCanvas.render_view_span_world()/render_view_bounds_world()).

Bug this fixes: world_to_screen()/screen_to_world() used
rect.width()/span_x and rect.height()/span_y as independent X/Y scale
factors. Those are only equal when the configured viewport's aspect ratio
(camera_width/camera_height) happens to match the canvas's plot_rect aspect
ratio -- any mismatch silently stretched every rendered shape: circles
became ellipses, squares became rectangles, FoV footprints and obstacles
distorted.

Fix: two viewport concepts, kept explicitly distinct:

- Logical viewport (logical_view_span_world()/logical_view_bounds_world(),
  plain aliases of the pre-existing active_view_span_world()/
  active_view_bounds_world()): exactly camera_center_x/y +/-
  camera_width/height/2 in simulation mode (or the editor pan/zoom
  rectangle in editor mode). This is what persists in SimulationConfig/
  .sim files, what the editable viewport frame (camera_bounds_world(),
  untouched by this fix) draws, and what the exploration-coverage metric's
  ROI is built from. It never changes on resize/theme/pan-zoom-driven
  aspect correction.
- Render viewport (render_view_span_world()/render_view_bounds_world()):
  the logical viewport expanded -- via _fit_world_span_to_plot_aspect(),
  symmetrically, on whichever single axis is needed, never cropped, never
  stretched independently -- to match plot_rect()'s aspect ratio. This,
  not the logical viewport, is what world_to_screen()/screen_to_world()
  and every render/culling call site actually use.

Also fixes the exploration-coverage metric (engine.
estimated_explored_percent(), the "Belief coverage of rectangle" metrics-
panel row and the explored_percent value recorded into telemetry/
navigation-debug snapshots): it used to divide by belief_map.grid's full
WORLD_X/Y extent (via BeliefMapStats.coverage_percent), completely
ignoring the configured camera viewport. It now restricts both numerator
and denominator to belief cells whose centers fall inside the logical
viewport (see engine.logical_exploration_viewport_bounds()) -- so it
reacts to an explicitly configured camera_width/camera_height, but never
to resize/theme/the render viewport's automatic aspect-ratio expansion.
"""
from __future__ import annotations

import ast
import inspect
import math
from types import MethodType, SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas, _fit_world_span_to_plot_aspect
from robotics_sim.app.theme import ThemeMode
from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.simulation.config import SimulationConfig
from robotics_sim.simulation.engine import SimulationControllerMixin

_app = QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_canvas() -> SimulationCanvas:
    canvas = SimulationCanvas()
    canvas.resize(300, 240)
    canvas.navigation_debug_enabled = False
    return canvas


def _paint_once(canvas: SimulationCanvas, method_name: str = "draw_explored_area_trace") -> None:
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    getattr(canvas, method_name)(painter)
    painter.end()


def _sample_alpha(canvas: SimulationCanvas, cache: QPixmap | None, world_xy: tuple[float, float]) -> int:
    """See test_smooth_explored_area_rebuild.py's identical helper for why
    QImage.pixelColor(), not QColor(image.pixel(x, y)), is required here."""
    if cache is None:
        return 0
    sx, sy = canvas.world_to_screen(*world_xy)
    image = cache.toImage()
    x, y = int(sx), int(sy)
    if x < 0 or y < 0 or x >= image.width() or y >= image.height():
        return 0
    return image.pixelColor(x, y).alpha()


def _make_metric_host(
    belief: BeliefMap,
    *,
    camera_center_x: float = 0.0,
    camera_center_y: float = 0.0,
    camera_width: float | None = None,
    camera_height: float | None = None,
) -> SimpleNamespace:
    """Minimal duck-typed SimulationControllerMixin host for
    estimated_explored_percent()/logical_exploration_viewport_bounds() --
    same pattern as the other engine.py unit tests in this suite."""
    config = SimulationConfig(
        camera_center_x=camera_center_x,
        camera_center_y=camera_center_y,
        camera_width=camera_width if camera_width is not None else (belief.x_max - belief.x_min),
        camera_height=camera_height if camera_height is not None else (belief.y_max - belief.y_min),
    )
    host = SimpleNamespace(config=config, belief_map=belief, ensure_belief_map=lambda: belief)
    host.logical_exploration_viewport_bounds = MethodType(
        SimulationControllerMixin.logical_exploration_viewport_bounds, host
    )
    return host


# ---------------------------------------------------------------------------
# Direct unit test of the pure helper.
# ---------------------------------------------------------------------------


def test_fit_world_span_to_plot_aspect_direct():
    # Plot wider than logical -> expand X, keep Y.
    render_x, render_y = _fit_world_span_to_plot_aspect(4.0, 4.0, 550.0, 370.0)
    assert render_y == pytest.approx(4.0)
    assert render_x == pytest.approx(4.0 * (550.0 / 370.0))

    # Plot taller (relatively) than logical -> expand Y, keep X.
    render_x2, render_y2 = _fit_world_span_to_plot_aspect(10.0, 2.0, 370.0, 550.0)
    assert render_x2 == pytest.approx(10.0)
    assert render_y2 == pytest.approx(10.0 / (370.0 / 550.0))

    # Invalid plot dimensions fall back to the logical span unchanged.
    assert _fit_world_span_to_plot_aspect(3.0, 5.0, 0.0, 100.0) == (3.0, 5.0)
    assert _fit_world_span_to_plot_aspect(3.0, 5.0, 100.0, -1.0) == (3.0, 5.0)


# ---------------------------------------------------------------------------
# 1. Uniform scale.
# ---------------------------------------------------------------------------


def test_uniform_scale():
    canvas = _make_canvas()
    canvas.config.camera_center_x = 0.0
    canvas.config.camera_center_y = 0.0
    canvas.config.camera_width = 4.0
    canvas.config.camera_height = 10.0  # deliberately mismatched vs. plot_rect
    canvas.invalidate_view_transform_caches()

    sx0, sy0 = canvas.world_to_screen(0.0, 0.0)
    sx1, sy1 = canvas.world_to_screen(1.0, 0.0)
    sx2, sy2 = canvas.world_to_screen(0.0, 1.0)
    pixels_per_unit_x = abs(sx1 - sx0)
    pixels_per_unit_y = abs(sy2 - sy0)
    assert pixels_per_unit_x == pytest.approx(pixels_per_unit_y, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Square stays square.
# ---------------------------------------------------------------------------


def test_square_stays_square():
    canvas = _make_canvas()
    canvas.config.camera_width = 3.0
    canvas.config.camera_height = 9.0
    canvas.invalidate_view_transform_caches()

    x0, y0 = canvas.world_to_screen(-1.0, -1.0)
    x1, y1 = canvas.world_to_screen(1.0, 1.0)
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    assert width == pytest.approx(height, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Circle/FoV does not stretch.
# ---------------------------------------------------------------------------


def test_circle_does_not_stretch():
    canvas = _make_canvas()
    canvas.config.camera_width = 6.0
    canvas.config.camera_height = 2.0
    canvas.invalidate_view_transform_caches()

    cx, cy = canvas.world_to_screen(0.0, 0.0)
    radius = 1.0
    east = canvas.world_to_screen(radius, 0.0)
    west = canvas.world_to_screen(-radius, 0.0)
    north = canvas.world_to_screen(0.0, radius)
    south = canvas.world_to_screen(0.0, -radius)

    radius_x = (math.hypot(east[0] - cx, east[1] - cy) + math.hypot(west[0] - cx, west[1] - cy)) / 2.0
    radius_y = (math.hypot(north[0] - cx, north[1] - cy) + math.hypot(south[0] - cx, south[1] - cy)) / 2.0
    assert radius_x == pytest.approx(radius_y, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Wider canvas (relative to the logical viewport) expands X.
# ---------------------------------------------------------------------------


def test_wider_canvas_expands_x():
    canvas = _make_canvas()
    plot = canvas.plot_rect()
    plot_aspect = plot.width() / plot.height()

    logical_height = 4.0
    logical_width = logical_height * (plot_aspect / 3.0)  # << plot_aspect
    canvas.config.camera_width = logical_width
    canvas.config.camera_height = logical_height
    canvas.invalidate_view_transform_caches()
    assert (logical_width / logical_height) < plot_aspect, "test setup must make the canvas relatively wider"

    render_width, render_height = canvas.render_view_span_world()
    assert render_height == pytest.approx(logical_height)
    assert render_width > logical_width

    logical_left, logical_right, _, _ = canvas.logical_view_bounds_world()
    render_left, render_right, _, _ = canvas.render_view_bounds_world()
    left_extra = logical_left - render_left
    right_extra = render_right - logical_right
    assert left_extra == pytest.approx(right_extra, rel=1e-9)
    assert left_extra > 0.0


# ---------------------------------------------------------------------------
# 5. Taller canvas (relative to the logical viewport) expands Y.
# ---------------------------------------------------------------------------


def test_taller_canvas_expands_y():
    canvas = _make_canvas()
    plot = canvas.plot_rect()
    plot_aspect = plot.width() / plot.height()

    logical_width = 4.0
    logical_height = logical_width / (plot_aspect * 3.0)  # logical_aspect >> plot_aspect
    canvas.config.camera_width = logical_width
    canvas.config.camera_height = logical_height
    canvas.invalidate_view_transform_caches()
    assert (logical_width / logical_height) > plot_aspect, "test setup must make the canvas relatively taller"

    render_width, render_height = canvas.render_view_span_world()
    assert render_width == pytest.approx(logical_width)
    assert render_height > logical_height

    _, _, logical_bottom, logical_top = canvas.logical_view_bounds_world()
    _, _, render_bottom, render_top = canvas.render_view_bounds_world()
    bottom_extra = logical_bottom - render_bottom
    top_extra = render_top - logical_top
    assert bottom_extra == pytest.approx(top_extra, rel=1e-9)
    assert bottom_extra > 0.0


# ---------------------------------------------------------------------------
# 6. No crop: the four logical bounds are always contained in render bounds.
# ---------------------------------------------------------------------------


def test_no_crop():
    canvas = _make_canvas()
    for width, height in ((3.0, 11.0), (11.0, 3.0), (5.0, 5.0)):
        canvas.config.camera_width = width
        canvas.config.camera_height = height
        canvas.invalidate_view_transform_caches()

        l_left, l_right, l_bottom, l_top = canvas.logical_view_bounds_world()
        r_left, r_right, r_bottom, r_top = canvas.render_view_bounds_world()
        assert r_left <= l_left
        assert r_right >= l_right
        assert r_bottom <= l_bottom
        assert r_top >= l_top


# ---------------------------------------------------------------------------
# 7. Inverse transform.
# ---------------------------------------------------------------------------


def test_inverse_transform():
    canvas = _make_canvas()
    canvas.config.camera_center_x = -1.5
    canvas.config.camera_center_y = 2.0
    canvas.config.camera_width = 7.0
    canvas.config.camera_height = 2.5
    canvas.invalidate_view_transform_caches()

    points = [(0.0, 0.0), (1.5, -1.0), (-2.0, 2.0), (-1.5, 2.0), (3.0, 3.5)]
    for point in points:
        screen = canvas.world_to_screen(*point)
        back = canvas.screen_to_world(*screen)
        assert back[0] == pytest.approx(point[0], abs=1e-6)
        assert back[1] == pytest.approx(point[1], abs=1e-6)


# ---------------------------------------------------------------------------
# 8. Resize preserves the logical viewport; only render bounds recompute.
# ---------------------------------------------------------------------------


def test_resize_preserves_logical_viewport():
    canvas = _make_canvas()
    canvas.config.camera_center_x = 1.0
    canvas.config.camera_center_y = -2.0
    canvas.config.camera_width = 5.0
    canvas.config.camera_height = 5.0
    canvas.invalidate_view_transform_caches()

    logical_before = canvas.logical_view_bounds_world()
    render_before = canvas.render_view_bounds_world()

    # canvas.resize() does not reliably change canvas.size() in this
    # offscreen/headless harness (no parent layout, never shown) -- same
    # finding already documented in test_smooth_explored_area_rebuild.py's
    # own resize test. Adjusting the plot margins changes plot_rect() just
    # as resizeEvent() would (it also only invalidates caches -- see
    # invalidate_view_transform_caches() -- and never touches camera_*).
    canvas.plot_margin_right += 150
    canvas.invalidate_view_transform_caches()

    assert canvas.config.camera_center_x == 1.0
    assert canvas.config.camera_center_y == -2.0
    assert canvas.config.camera_width == 5.0
    assert canvas.config.camera_height == 5.0
    assert canvas.logical_view_bounds_world() == logical_before

    render_after = canvas.render_view_bounds_world()
    assert render_after != render_before

    # Geometry is still undistorted after the "resize".
    sx0, sy0 = canvas.world_to_screen(1.0, -2.0)
    sx1, sy1 = canvas.world_to_screen(2.0, -2.0)
    sx2, sy2 = canvas.world_to_screen(1.0, -1.0)
    assert abs(sx1 - sx0) == pytest.approx(abs(sy2 - sy0), rel=1e-6)


# ---------------------------------------------------------------------------
# 9. Editor frame remains logical (never the render-expanded bounds).
# ---------------------------------------------------------------------------


def test_editor_frame_remains_logical():
    canvas = _make_canvas()
    canvas.config.camera_center_x = 2.0
    canvas.config.camera_center_y = 3.0
    canvas.config.camera_width = 4.0
    canvas.config.camera_height = 20.0
    canvas.invalidate_view_transform_caches()

    frame_left, frame_right, frame_bottom, frame_top = canvas.camera_bounds_world()
    assert frame_left == pytest.approx(0.0)
    assert frame_right == pytest.approx(4.0)
    assert frame_bottom == pytest.approx(-7.0)
    assert frame_top == pytest.approx(13.0)

    render_bounds = canvas.render_view_bounds_world()
    assert render_bounds != (frame_left, frame_right, frame_bottom, frame_top)
    assert canvas.logical_view_bounds_world() == pytest.approx(
        (frame_left, frame_right, frame_bottom, frame_top)
    )


# ---------------------------------------------------------------------------
# 10. Metric independent from render fit (canvas wide/tall/square).
# ---------------------------------------------------------------------------


def test_metric_independent_from_render_fit():
    belief = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=1.0, robot_count=1)
    for x in range(-2, 2):
        for y in range(-2, 2):
            cell = belief.world_to_cell((float(x), float(y)))
            if cell is not None:
                belief.mark_free_cell(cell, robot_index=0, time_s=0.0)

    host = _make_metric_host(belief, camera_width=6.0, camera_height=6.0)

    # Increasing plot_margin_bottom shrinks plot_rect's HEIGHT (wider
    # aspect); increasing plot_margin_right shrinks its WIDTH (taller
    # aspect) -- see plot_rect()'s .adjusted() call.
    wide_canvas = _make_canvas()
    wide_canvas.plot_margin_bottom += 300
    tall_canvas = _make_canvas()
    tall_canvas.plot_margin_right += 200
    square_canvas = _make_canvas()

    values = []
    for canvas in (wide_canvas, tall_canvas, square_canvas):
        canvas.config.camera_width = 6.0
        canvas.config.camera_height = 6.0
        canvas.invalidate_view_transform_caches()
        host.canvas = canvas  # not read by the metric -- proves it truly is not
        values.append(SimulationControllerMixin.estimated_explored_percent(host))

    assert values[0] == values[1] == values[2]


# ---------------------------------------------------------------------------
# 11. Metric uses the logical ROI, not the render viewport's extra margin.
# ---------------------------------------------------------------------------


def test_metric_uses_logical_roi_not_render_margin():
    belief = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=1.0, robot_count=1)
    camera_width, camera_height = 4.0, 4.0
    host = _make_metric_host(belief, camera_width=camera_width, camera_height=camera_height)

    canvas = _make_canvas()
    canvas.config.camera_width = camera_width
    canvas.config.camera_height = camera_height
    # Shrinks plot_rect's height, making it relatively much wider, so the
    # render viewport expands on X -- see plot_rect()'s .adjusted() call.
    canvas.plot_margin_bottom += 300
    canvas.invalidate_view_transform_caches()

    logical_left, logical_right, _, _ = canvas.logical_view_bounds_world()
    render_left, render_right, _, _ = canvas.render_view_bounds_world()
    assert render_right > logical_right, "test setup must actually expand the render viewport in X"

    inside_cell = belief.world_to_cell((0.5, 0.5), clamp=True)
    margin_world_x = (logical_right + render_right) / 2.0  # inside the extra margin only
    margin_cell = belief.world_to_cell((margin_world_x, 0.0), clamp=True)

    belief.mark_free_cell(inside_cell, robot_index=0, time_s=0.0)
    metric_after_inside = SimulationControllerMixin.estimated_explored_percent(host)
    assert metric_after_inside > 0.0

    belief.mark_free_cell(margin_cell, robot_index=0, time_s=0.0)
    metric_after_margin = SimulationControllerMixin.estimated_explored_percent(host)

    assert metric_after_margin == metric_after_inside, (
        "a cell visible only in the render viewport's extra margin must not change the metric"
    )


# ---------------------------------------------------------------------------
# 12. Explicitly configured ROI (camera_width/camera_height) changes the metric.
# ---------------------------------------------------------------------------


def test_configured_roi_changes_metric():
    belief = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=1.0, robot_count=1)
    for x in range(-8, 8):
        for y in range(-8, 8):
            cell = belief.world_to_cell((float(x), float(y)))
            if cell is not None:
                belief.mark_free_cell(cell, robot_index=0, time_s=0.0)

    host = _make_metric_host(belief, camera_width=4.0, camera_height=4.0)
    small_viewport_metric = SimulationControllerMixin.estimated_explored_percent(host)

    host.config.camera_width = 20.0
    host.config.camera_height = 20.0
    large_viewport_metric = SimulationControllerMixin.estimated_explored_percent(host)

    assert small_viewport_metric == pytest.approx(100.0)
    assert large_viewport_metric < small_viewport_metric


# ---------------------------------------------------------------------------
# 13. Existing smooth coverage survives resize/theme with the uniform transform.
# ---------------------------------------------------------------------------


def test_smooth_coverage_survives_with_uniform_transform():
    canvas = _make_canvas()
    canvas.config.camera_width = 3.0
    canvas.config.camera_height = 11.0  # forces X-axis aspect-fit expansion
    canvas.invalidate_view_transform_caches()

    triangle = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]  # diagonal hypotenuse
    canvas.append_explored_area_polygon(triangle, robot_index=None)
    _paint_once(canvas)

    inside_point = (0.2, 0.2)
    outside_point = (0.6, 0.6)
    assert _sample_alpha(canvas, canvas._explored_area_cache, inside_point) > 0
    assert _sample_alpha(canvas, canvas._explored_area_cache, outside_point) == 0

    # Resize-equivalent invalidation (see test 8's note on canvas.resize()).
    canvas.invalidate_explored_area_cache()
    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, inside_point) > 0
    assert _sample_alpha(canvas, canvas._explored_area_cache, outside_point) == 0

    canvas.set_theme_mode(ThemeMode.DARK)
    _paint_once(canvas)
    canvas.set_theme_mode(ThemeMode.LIGHT)
    _paint_once(canvas)
    assert _sample_alpha(canvas, canvas._explored_area_cache, inside_point) > 0
    assert _sample_alpha(canvas, canvas._explored_area_cache, outside_point) == 0


# ---------------------------------------------------------------------------
# 14. No production coupling: BeliefMap/frontiers/costmaps/planners/
# coordinators were not touched by this fix.
# ---------------------------------------------------------------------------


def test_no_production_coupling():
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
        "planning_costmap_builder",
        "simulation.coordination",
    )
    for module in imported_modules:
        for forbidden in forbidden_substrings:
            assert forbidden not in module, (module, forbidden)

    # The new metric helpers in engine.py must only ever reference belief
    # grid geometry and SimulationConfig's camera_* fields -- never
    # frontier/planning/coordinator/costmap internals.
    source = "".join((
        inspect.getsource(SimulationControllerMixin.estimated_explored_percent),
        inspect.getsource(SimulationControllerMixin.logical_exploration_viewport_bounds),
    )).lower()
    for term in ("frontier", "costmap", "coordinator", "planner"):
        assert term not in source, term
