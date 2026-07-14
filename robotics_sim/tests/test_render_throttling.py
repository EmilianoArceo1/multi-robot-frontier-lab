"""
Tests for RenderThrottler (simulation_canvas.py): coalesces the
high-frequency, simulation-driven repaint requests
(set_runtime_state()/set_multi_runtime_state()) to a target FPS, without
touching any other self.update() call (mouse/editor interactions stay
immediate).

Pure/Qt-free class -- these tests drive it directly with explicit `now`
timestamps, no QApplication/paintEvent needed.
"""
from __future__ import annotations

from robotics_sim.app.render_perf import RenderDetailLogger, format_render_detail_line
from robotics_sim.app.simulation_canvas import DEFAULT_RENDER_THROTTLE_FPS, RenderThrottler


# ---------------------------------------------------------------------------
# F. Repaint requests faster than the target FPS are coalesced (skipped).
# ---------------------------------------------------------------------------


def test_render_throttler_coalesces_repaint_requests():
    throttler = RenderThrottler(target_fps=30.0)  # ~0.0333s minimum interval

    assert throttler.should_render(now=0.0) is True
    assert throttler.should_render(now=0.01) is False
    assert throttler.should_render(now=0.02) is False
    assert throttler.should_render(now=0.03) is False
    assert throttler.should_render(now=0.04) is True, "once the interval elapses, a repaint is allowed again"
    assert throttler.should_render(now=0.045) is False


def test_render_throttler_default_target_fps_is_30():
    assert DEFAULT_RENDER_THROTTLE_FPS == 30.0


# ---------------------------------------------------------------------------
# G. force=True always renders immediately, regardless of the throttle
#    window -- used for paused/user-interaction repaints.
# ---------------------------------------------------------------------------


def test_render_throttler_allows_immediate_paint_when_paused_or_forced():
    throttler = RenderThrottler(target_fps=30.0)

    assert throttler.should_render(now=0.0) is True
    # Well within the throttle window, but forced -- must still render.
    assert throttler.should_render(now=0.005, force=True) is True
    # Normal throttling resumes immediately after a forced render.
    assert throttler.should_render(now=0.006) is False
    assert throttler.should_render(now=0.04) is True


# ---------------------------------------------------------------------------
# C. SIM_RENDER_FPS env var overrides the default target FPS.
# ---------------------------------------------------------------------------


def test_render_throttler_uses_env_target_fps():
    throttler = RenderThrottler(env={"SIM_RENDER_FPS": "20"})
    assert throttler.target_fps == 20.0

    # ~0.05s minimum interval at 20 FPS -- stricter than the 30 FPS default.
    assert throttler.should_render(now=0.0) is True
    assert throttler.should_render(now=0.04) is False, "still within the 20 FPS window"
    assert throttler.should_render(now=0.05) is True


def test_render_throttler_defaults_to_30_fps_when_env_unset():
    throttler = RenderThrottler(env={})
    assert throttler.target_fps == DEFAULT_RENDER_THROTTLE_FPS


def test_render_throttler_default_fps():
    """Exact name requested this round -- default target FPS is 30,
    whether read from DEFAULT_RENDER_THROTTLE_FPS directly or via a bare
    RenderThrottler() construction with no env override."""
    assert DEFAULT_RENDER_THROTTLE_FPS == 30.0
    assert RenderThrottler(env={}).target_fps == 30.0


# ---------------------------------------------------------------------------
# SIM_RENDER_DETAIL_LOG=1 optional per-layer paint breakdown -- disabled by
# default, never spams even when enabled.
# ---------------------------------------------------------------------------


def test_render_detail_log_disabled_by_default(capsys):
    logger = RenderDetailLogger(env={})

    emitted = logger.maybe_log(
        total_ms=42.1, background_ms=2.0, map_layer_ms=24.0,
        robot_body_ms=1.0, robot_fov_ms=0.5, route_path_ms=1.2,
        sensor_debug_overlay_ms=0.4, overlays_ms=5.0, cache_hit=True,
    )

    assert emitted is False
    captured = capsys.readouterr()
    assert captured.out == ""


def test_render_detail_log_enabled_reports_line(capsys):
    logger = RenderDetailLogger(env={"SIM_RENDER_DETAIL_LOG": "1"})

    emitted = logger.maybe_log(
        total_ms=42.1, background_ms=2.0, map_layer_ms=24.0,
        robot_body_ms=1.0, robot_fov_ms=0.5, route_path_ms=1.2,
        sensor_debug_overlay_ms=0.4, overlays_ms=5.0, cache_hit=True, now=0.0,
    )

    assert emitted is True
    captured = capsys.readouterr()
    assert captured.out.strip() == (
        "[RENDER] total_ms=42.1 background_ms=2.0 map_layer_ms=24.0 "
        "grid_overlay_ms=0.0 grid_overlay_cache_status=n/a grid_overlay_visible_cells=0 "
        "grid_overlay_rebuild_ms=0.0 grid_overlay_blit_ms=0.0 grid_overlay_cells_ms=0.0 "
        "grid_overlay_lines_ms=0.0 explored_area_ms=0.0 ground_truth_obstacles_ms=0.0 "
        "mapped_obstacle_points_ms=0.0 robot_body_ms=1.0 "
        "robot_fov_ms=0.5 robot_fov_cache_hit=True robot_fov_compute_ms=0.0 "
        "robot_fov_paint_ms=0.0 route_path_ms=1.2 planned_route_build_ms=0.0 "
        "planned_route_paint_ms=0.0 executed_trail_build_ms=0.0 executed_trail_paint_ms=0.0 "
        "executed_trail_points=0 executed_trail_segments_painted=0 "
        "executed_trail_cache_hit=False sensor_debug_overlay_ms=0.4 "
        "overlays_ms=5.0 editor_overlays_ms=0.0 grid_preview_ms=0.0 plot_border_ms=0.0 "
        "card_ms=0.0 title_ms=0.0 telemetry_ms=0.0 cache_hit=True"
    )

    # Throttled: at most once every 2 seconds, even while enabled.
    again = logger.maybe_log(
        total_ms=1.0, background_ms=0.0, map_layer_ms=0.0,
        robot_body_ms=0.0, robot_fov_ms=0.0, route_path_ms=0.0,
        sensor_debug_overlay_ms=0.0, overlays_ms=0.0, cache_hit=False, now=0.5,
    )
    assert again is False
    captured = capsys.readouterr()
    assert captured.out == ""


def test_render_detail_breaks_down_robot_layer(capsys):
    """This round replaces the old single robot_layer_ms bucket with four
    dedicated sub-buckets so the previously-dominant robot_layer_ms figure
    (13.5-17.7ms in manual Office.sim evidence) can be attributed to a
    specific drawing routine instead of one opaque total."""
    logger = RenderDetailLogger(env={"SIM_RENDER_DETAIL_LOG": "1"})

    emitted = logger.maybe_log(
        total_ms=30.0, background_ms=1.0, map_layer_ms=4.0,
        robot_body_ms=2.1, robot_fov_ms=1.3, route_path_ms=3.4,
        sensor_debug_overlay_ms=0.6, overlays_ms=3.0, cache_hit=True, now=0.0,
    )

    assert emitted is True
    captured = capsys.readouterr()
    assert "robot_body_ms=2.1" in captured.out
    assert "robot_fov_ms=1.3" in captured.out
    assert "route_path_ms=3.4" in captured.out
    assert "sensor_debug_overlay_ms=0.6" in captured.out
    assert "robot_layer_ms=" not in captured.out


def test_route_detail_reports_executed_trail_paint_metrics(capsys):
    """route_path_ms alone hid a real bug: real Office.sim evidence showed
    it growing unboundedly (17ms up to 431ms) as the executed trail
    accumulated points, even though a QPainterPath cache was already in
    place -- because drawPath() still rasterizes the WHOLE path every
    frame. These finer-grained fields let a steady low
    executed_trail_paint_ms be distinguished from a low but silently-
    growing route_path_ms."""
    logger = RenderDetailLogger(env={"SIM_RENDER_DETAIL_LOG": "1"})

    emitted = logger.maybe_log(
        total_ms=12.0, background_ms=1.0, map_layer_ms=2.0,
        robot_body_ms=0.1, robot_fov_ms=1.0, route_path_ms=1.6,
        planned_route_build_ms=0.1, planned_route_paint_ms=0.2,
        executed_trail_build_ms=0.2, executed_trail_paint_ms=1.1,
        executed_trail_points=3400, executed_trail_segments_painted=3,
        executed_trail_cache_hit=True,
        sensor_debug_overlay_ms=0.2, overlays_ms=3.0, cache_hit=True, now=0.0,
    )

    assert emitted is True
    captured = capsys.readouterr()
    assert "planned_route_build_ms=0.1" in captured.out
    assert "planned_route_paint_ms=0.2" in captured.out
    assert "executed_trail_build_ms=0.2" in captured.out
    assert "executed_trail_paint_ms=1.1" in captured.out
    assert "executed_trail_points=3400" in captured.out
    assert "executed_trail_segments_painted=3" in captured.out
    assert "executed_trail_cache_hit=True" in captured.out


# ---------------------------------------------------------------------------
# Fine-grained map_layer_ms/overlays_ms/robot_fov_ms instrumentation.
#
# Diagnosis-only round: map_layer_ms was measured as the combined total of
# draw_grid_overlay()/draw_explored_area_trace()/draw_ground_truth_obstacles()/
# draw_mapped_obstacle_points(), overlays_ms mixed editor preview/selection/
# camera-frame, the grid-resolution preview, the plot border, and the
# card/title/telemetry chrome, and robot_fov_ms measured draw_sensor_range()
# as one figure with no visibility into cache hit/miss, polygon compute, or
# paint. These tests confirm the new sub-fields appear on the [RENDER] line
# and that the aggregate fields still equal their sum (see
# test_canvas_render_cache.py for the actual sum-matches-real-draw-plot()
# regression, driven through a real paintEvent-style pixmap).
# ---------------------------------------------------------------------------


def test_render_detail_line_includes_all_new_instrumentation_fields():
    line = format_render_detail_line(total_ms=1.0, background_ms=0.0, map_layer_ms=0.0)
    for field in (
        "grid_overlay_ms", "grid_overlay_cache_status", "grid_overlay_visible_cells",
        "grid_overlay_rebuild_ms", "grid_overlay_blit_ms", "grid_overlay_cells_ms",
        "grid_overlay_lines_ms", "explored_area_ms", "ground_truth_obstacles_ms",
        "mapped_obstacle_points_ms", "robot_fov_cache_hit", "robot_fov_compute_ms",
        "robot_fov_paint_ms", "editor_overlays_ms", "grid_preview_ms", "plot_border_ms",
        "card_ms", "title_ms", "telemetry_ms",
    ):
        assert f"{field}=" in line, f"[RENDER] line is missing the {field} field"


def test_render_detail_map_layer_subfields_reported(capsys):
    logger = RenderDetailLogger(env={"SIM_RENDER_DETAIL_LOG": "1"})

    emitted = logger.maybe_log(
        total_ms=20.0, background_ms=1.0, map_layer_ms=9.0,
        grid_overlay_ms=5.0, grid_overlay_cache_status="rebuild",
        grid_overlay_visible_cells=1200, grid_overlay_rebuild_ms=4.7,
        grid_overlay_blit_ms=0.2, grid_overlay_cells_ms=4.0, grid_overlay_lines_ms=0.5,
        explored_area_ms=1.5, ground_truth_obstacles_ms=1.0, mapped_obstacle_points_ms=1.5,
        now=0.0,
    )

    assert emitted is True
    captured = capsys.readouterr()
    assert "grid_overlay_ms=5.0" in captured.out
    assert "grid_overlay_cache_status=rebuild" in captured.out
    assert "grid_overlay_visible_cells=1200" in captured.out
    assert "grid_overlay_rebuild_ms=4.7" in captured.out
    assert "grid_overlay_blit_ms=0.2" in captured.out
    assert "grid_overlay_cells_ms=4.0" in captured.out
    assert "grid_overlay_lines_ms=0.5" in captured.out
    assert "explored_area_ms=1.5" in captured.out
    assert "ground_truth_obstacles_ms=1.0" in captured.out
    assert "mapped_obstacle_points_ms=1.5" in captured.out


def test_render_detail_overlays_subfields_reported(capsys):
    logger = RenderDetailLogger(env={"SIM_RENDER_DETAIL_LOG": "1"})

    emitted = logger.maybe_log(
        total_ms=10.0, background_ms=0.5, map_layer_ms=0.5,
        overlays_ms=6.0, editor_overlays_ms=0.2, grid_preview_ms=0.1,
        plot_border_ms=0.1, card_ms=2.0, title_ms=0.6, telemetry_ms=3.0,
        now=0.0,
    )

    assert emitted is True
    captured = capsys.readouterr()
    assert "editor_overlays_ms=0.2" in captured.out
    assert "grid_preview_ms=0.1" in captured.out
    assert "plot_border_ms=0.1" in captured.out
    assert "card_ms=2.0" in captured.out
    assert "title_ms=0.6" in captured.out
    assert "telemetry_ms=3.0" in captured.out


def test_render_detail_fov_compute_paint_cache_hit_reported(capsys):
    logger = RenderDetailLogger(env={"SIM_RENDER_DETAIL_LOG": "1"})

    emitted = logger.maybe_log(
        total_ms=5.0, background_ms=0.5, map_layer_ms=0.5,
        robot_fov_ms=1.4, robot_fov_cache_hit=False,
        robot_fov_compute_ms=1.1, robot_fov_paint_ms=0.3,
        now=0.0,
    )

    assert emitted is True
    captured = capsys.readouterr()
    assert "robot_fov_cache_hit=False" in captured.out
    assert "robot_fov_compute_ms=1.1" in captured.out
    assert "robot_fov_paint_ms=0.3" in captured.out


def test_render_detail_disabled_by_default_ignores_new_fields_too(capsys):
    """The new fields must not change RenderDetailLogger's disabled-by-
    default behavior: still silent, still no print, regardless of what
    values are passed for them."""
    logger = RenderDetailLogger(env={})

    emitted = logger.maybe_log(
        total_ms=1.0, background_ms=0.0, map_layer_ms=0.0,
        grid_overlay_cache_status="rebuild", grid_overlay_visible_cells=99999,
        robot_fov_cache_hit=False, robot_fov_compute_ms=999.0,
    )

    assert emitted is False
    captured = capsys.readouterr()
    assert captured.out == ""
