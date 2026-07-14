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

from robotics_sim.app.render_perf import RenderDetailLogger
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
        "[RENDER] total_ms=42.1 background_ms=2.0 map_layer_ms=24.0 robot_body_ms=1.0 "
        "robot_fov_ms=0.5 route_path_ms=1.2 planned_route_build_ms=0.0 "
        "planned_route_paint_ms=0.0 executed_trail_build_ms=0.0 executed_trail_paint_ms=0.0 "
        "executed_trail_points=0 executed_trail_segments_painted=0 "
        "executed_trail_cache_hit=False sensor_debug_overlay_ms=0.4 "
        "overlays_ms=5.0 cache_hit=True"
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
