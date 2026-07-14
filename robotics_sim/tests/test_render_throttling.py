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
