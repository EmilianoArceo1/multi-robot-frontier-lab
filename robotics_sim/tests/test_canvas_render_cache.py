"""
Tests for SimulationCanvas's static/semi-static render layer caching.

Manual Office.sim evidence: render_ms often 35-58ms with FPS dropping to
~14-23. The caching infrastructure this exercises (static background/grid,
ground-truth obstacles, mapped-obstacle points, explored area, grid
overlay) already existed before this round and turned out to already be
handling the required invalidation triggers correctly (viewport/zoom/pan
via invalidate_view_transform_caches(), runtime map growth via the
incremental append path, grid resolution/snapshot changes via
draw_grid_overlay()'s own cache-key comparison) -- these tests capture
that contract as a regression guard, since it was previously unverified by
any test. Robot/FOV/route overlays are intentionally never cached (they
change every tick) and are not exercised here.

Same testing approach as test_grid_resolution_preview.py: a real
SimulationCanvas instance (needs a QApplication), but never .show()'d --
only cache/state is asserted on, never pixel output.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.simulation.config import OBSTACLE_VISUAL_REFRESH_POINT_STEP

_app = QApplication.instance() or QApplication([])


def _make_canvas(width: int = 400, height: int = 300) -> SimulationCanvas:
    canvas = SimulationCanvas()
    canvas.resize(width, height)
    return canvas


def _draw_planned_route_once(canvas: SimulationCanvas) -> None:
    """Exercise draw_planned_route() directly against a throwaway QPixmap,
    mirroring test_grid_resolution_preview.py's _draw_overlay_once() -- only
    cache state is asserted on, never pixel output."""
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    canvas.draw_planned_route(painter)
    painter.end()


def _draw_executed_path_once(canvas: SimulationCanvas) -> None:
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    canvas.draw_executed_path(painter)
    painter.end()


# ---------------------------------------------------------------------------
# test_canvas_cache_reuses_static_layer_without_changes
# ---------------------------------------------------------------------------


def test_canvas_cache_reuses_static_layer_without_changes():
    canvas = _make_canvas()

    canvas.ensure_static_plot_cache()
    first_cache = canvas._static_plot_cache
    assert first_cache is not None

    # Nothing changed (same size, same view) -- must reuse the same pixmap
    # object, not rebuild it.
    canvas.ensure_static_plot_cache()
    second_cache = canvas._static_plot_cache
    assert second_cache is first_cache


# ---------------------------------------------------------------------------
# test_canvas_cache_invalidates_on_view_change
# ---------------------------------------------------------------------------


def test_canvas_cache_invalidates_on_view_change():
    canvas = _make_canvas()

    canvas.ensure_static_plot_cache()
    first_cache = canvas._static_plot_cache
    assert first_cache is not None

    # Mirrors what wheelEvent() does after changing editor_zoom/
    # editor_pan_offset (the only way the world_to_screen() transform
    # changes without a widget resize).
    canvas.editor_zoom = 2.0
    canvas.invalidate_view_transform_caches()
    assert canvas._static_plot_cache is None, "a view/zoom change must invalidate the background cache"

    canvas.ensure_static_plot_cache()
    second_cache = canvas._static_plot_cache
    assert second_cache is not None
    assert second_cache is not first_cache, "must be a freshly rebuilt pixmap, not the stale one"


# ---------------------------------------------------------------------------
# test_canvas_cache_invalidates_on_runtime_map_change
# ---------------------------------------------------------------------------


def test_canvas_cache_invalidates_on_runtime_map_change():
    canvas = _make_canvas()

    canvas.set_mapped_obstacle_points([(1.0, 1.0)])
    canvas.ensure_obstacles_cache()
    first_obstacles_cache = canvas._obstacles_cache
    assert first_obstacles_cache is not None

    # A small runtime map update (below OBSTACLE_VISUAL_REFRESH_POINT_STEP)
    # must NOT invalidate the obstacle opacity cache -- this is the exact
    # optimization append_mapped_obstacle_points()'s own docstring
    # describes ("do not rebuild ... after every single sensor point").
    canvas.append_mapped_obstacle_points([(2.0, 2.0)])
    assert canvas._obstacles_cache is first_obstacles_cache, (
        "a small map update below the refresh threshold must not invalidate the cache"
    )

    # Crossing the refresh threshold's worth of new points must invalidate
    # it (obstacle-completion opacity depends on mapped-point density).
    many_new_points = [(float(i), 3.0) for i in range(OBSTACLE_VISUAL_REFRESH_POINT_STEP)]
    canvas.append_mapped_obstacle_points(many_new_points)
    assert canvas._obstacles_cache is None, (
        "crossing OBSTACLE_VISUAL_REFRESH_POINT_STEP worth of new points must invalidate the obstacle cache"
    )

    canvas.ensure_obstacles_cache()
    assert canvas._obstacles_cache is not None
    assert canvas._obstacles_cache is not first_obstacles_cache


def test_canvas_mapped_points_cache_reused_across_small_incremental_updates():
    """The mapped-obstacle-points pixmap itself is painted onto
    incrementally (paint_mapped_points_to_cache()), not rebuilt, for a
    normal per-tick sensor update -- the SAME pixmap object should be
    reused across several small appends."""
    canvas = _make_canvas()

    canvas.set_mapped_obstacle_points([(1.0, 1.0)])
    canvas.ensure_mapped_points_cache()
    first_cache = canvas._mapped_points_cache
    assert first_cache is not None

    canvas.append_mapped_obstacle_points([(2.0, 2.0)])
    assert canvas._mapped_points_cache is first_cache

    canvas.append_mapped_obstacle_points([(3.0, 3.0)])
    assert canvas._mapped_points_cache is first_cache


# ---------------------------------------------------------------------------
# Route/FOV render-path caching (robot_layer_ms breakdown round).
#
# Manual Office.sim render-detail evidence found robot_layer_ms dominating
# steady-state paint time (13.5-17.7ms) while background/map/overlays stayed
# low. draw_executed_path()/draw_planned_route() were redrawing their
# polyline with one drawLine() call per segment every frame -- now cached as
# a single QPainterPath, rebuilt only when the underlying points or the view
# transform actually change. sensor_polygon_for_pose() already cached the
# expensive FOV raycasting before this round; these tests just confirm that
# contract still holds now that it is timed as its own robot_fov_ms bucket.
# ---------------------------------------------------------------------------


def test_route_path_cache_reused_until_path_changes():
    canvas = _make_canvas()
    canvas.set_planned_path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    _draw_planned_route_once(canvas)
    first_cache = canvas._planned_route_cache
    assert first_cache is not None

    # Same points, same view -- must reuse the same QPainterPath object.
    _draw_planned_route_once(canvas)
    assert canvas._planned_route_cache is first_cache

    # The planned route itself changes -- must rebuild.
    canvas.set_planned_path([(0.0, 0.0), (1.0, 0.0), (2.0, 1.0)])
    _draw_planned_route_once(canvas)
    assert canvas._planned_route_cache is not first_cache


def test_route_path_cache_invalidates_on_view_change():
    canvas = _make_canvas()
    canvas.set_editor_mode(True)
    canvas.set_planned_path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    _draw_planned_route_once(canvas)
    first_cache = canvas._planned_route_cache
    assert first_cache is not None

    canvas.editor_zoom = 2.0
    canvas.invalidate_view_transform_caches()
    _draw_planned_route_once(canvas)
    assert canvas._planned_route_cache is not first_cache, (
        "a view/zoom change must invalidate the cached route path, or the drawn "
        "polyline would be stale relative to the new world_to_screen() transform"
    )


def test_fov_cache_reused_when_pose_unchanged():
    canvas = _make_canvas()

    first = canvas.sensor_polygon_for_pose(0, 1.0, 1.0, 0.0, 3.0)
    assert first is not None

    # Same pose (well below SENSOR_DRAW_RECOMPUTE_DISTANCE/ROTATION), same
    # vision/model/obstacles signature -- must return the identical cached
    # polygon object, not recompute the raycast.
    second = canvas.sensor_polygon_for_pose(0, 1.0, 1.0, 0.0, 3.0)
    assert second is first


def test_fov_cache_invalidates_when_pose_changes():
    canvas = _make_canvas()

    first = canvas.sensor_polygon_for_pose(0, 1.0, 1.0, 0.0, 3.0)
    assert first is not None

    # Pose moved well beyond SENSOR_DRAW_RECOMPUTE_DISTANCE -- must not
    # return the stale polygon computed for the old pose.
    second = canvas.sensor_polygon_for_pose(0, 5.0, 5.0, 0.0, 3.0)
    assert second is not first


def test_planned_route_cache_still_invalidates_on_path_change():
    """Regression guard for the executed-trail rewrite in this round:
    confirms draw_planned_route()'s own (separate) QPainterPath cache
    still invalidates correctly on a path change and was not disturbed
    by switching the executed trail to a pixmap-based cache."""
    canvas = _make_canvas()
    canvas.set_planned_path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    _draw_planned_route_once(canvas)
    first_cache = canvas._planned_route_cache
    assert first_cache is not None

    canvas.set_planned_path([(0.0, 0.0), (1.0, 0.0), (2.0, 2.0)])
    _draw_planned_route_once(canvas)
    assert canvas._planned_route_cache is not first_cache


# ---------------------------------------------------------------------------
# Executed-trail incremental pixmap cache.
#
# Real Office.sim evidence showed route_path_ms growing unboundedly (17ms up
# to 431ms) over a long run even with the QPainterPath cache from the
# previous round in place: rebuilding the path object was avoided, but
# painter.drawPath() still rasterizes the WHOLE accumulated path every
# single frame, so per-frame paint cost grew with total trail length. The
# fix paints the trail into a persistent QPixmap instead -- new points are
# painted into it once, and every frame just blits the pixmap.
# ---------------------------------------------------------------------------


def test_executed_trail_cache_paints_only_new_segments():
    canvas = _make_canvas()
    canvas.set_path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    _draw_executed_path_once(canvas)
    assert canvas._executed_trail_pixmap is not None
    # Building from scratch: 3 points -> 2 segments.
    assert canvas._route_detail["executed_trail_segments_painted"] == 2
    first_pixmap = canvas._executed_trail_pixmap

    # Normal growth: append in place (same list object), matching how the
    # engine grows the executed trail tick-by-tick.
    canvas.path_points.append((3.0, 0.0))
    canvas.path_points.append((4.0, 0.0))
    _draw_executed_path_once(canvas)

    # Must still be the SAME pixmap object (painted onto, not rebuilt) --
    # and only the 2 new segments should have been painted this frame, not
    # all 4.
    assert canvas._executed_trail_pixmap is first_pixmap
    assert canvas._route_detail["executed_trail_segments_painted"] == 2


def test_executed_trail_cache_reused_without_new_points():
    canvas = _make_canvas()
    canvas.set_path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    _draw_executed_path_once(canvas)
    first_pixmap = canvas._executed_trail_pixmap
    assert first_pixmap is not None

    # No new points appended -- must reuse the same pixmap and paint
    # nothing new.
    _draw_executed_path_once(canvas)
    assert canvas._executed_trail_pixmap is first_pixmap
    assert canvas._route_detail["executed_trail_segments_painted"] == 0
    assert canvas._route_detail["executed_trail_cache_hit"] is True


def test_executed_trail_cache_invalidates_on_view_change():
    canvas = _make_canvas()
    canvas.set_editor_mode(True)
    canvas.set_path([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])

    _draw_executed_path_once(canvas)
    first_pixmap = canvas._executed_trail_pixmap
    assert first_pixmap is not None

    canvas.editor_zoom = 2.0
    canvas.invalidate_view_transform_caches()
    _draw_executed_path_once(canvas)
    assert canvas._executed_trail_pixmap is not first_pixmap, (
        "a view/zoom change must invalidate the cached trail pixmap, or the blitted "
        "trail would be stale relative to the new world_to_screen() transform"
    )


def test_executed_trail_cache_rebuilds_after_path_truncation():
    canvas = _make_canvas()
    canvas.set_path([(float(i), 0.0) for i in range(10)])

    _draw_executed_path_once(canvas)
    first_pixmap = canvas._executed_trail_pixmap
    assert first_pixmap is not None
    assert canvas._executed_trail_pixmap_count == 10

    # Truncation replaces path_points with a NEW (shorter) list object --
    # mirrors how the engine slides a capped trail window. Same length
    # comparison alone would miss this; the cache must rebuild from the
    # new list, not keep appending relative to the old count.
    canvas.set_path([(float(i), 0.0) for i in range(5, 10)])
    _draw_executed_path_once(canvas)

    assert canvas._executed_trail_pixmap is not first_pixmap
    assert canvas._executed_trail_pixmap_count == 5
