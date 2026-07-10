"""
Tests for the grid_resolution slider, the "Show Grid" persistent overlay,
and the temporary red grid preview on SimulationCanvas.

Context
-------
grid_resolution was previously exposed via a NumericStepper (see git
history). This round replaces it with a discrete-step horizontal slider
(SteppedSliderRow) so 0.10..1.00 m/cell can be picked in 0.05 steps, and
adds a "Show Grid" toggle that keeps a grid overlay visible persistently
(instead of it always auto-hiding), optionally colored by occupied/free/
unknown cell state while the simulation is running.

This is a rendering/config-exposure feature, not a planning change: A*,
the planning grid builders, reachability, and navigation/exploration/
recovery logic are all untouched. The slider still writes into the same
SimulationConfig.grid_resolution the runtime already reads (engine.py's
read_config()/simulation_start_summary()); the "Show Grid" toggle and
overlay never touch SimulationConfig at all, and the occupancy snapshot
they can display is a read-only copy (occupancy_grid_snapshot_from_belief)
of BeliefMap.grid, never a live reference -- nothing here can mutate
occupancy state, routes, or metrics.

Testing approach
-----------------
Same approach as before: no full MainWindow instantiation. These tests
instantiate only the focused pieces of UI directly (SteppedSliderRow,
ToggleSwitch, SimulationCanvas) plus, for the snapshot function, a plain
BeliefMap -- no engine/mixin instantiation needed since
occupancy_grid_snapshot_from_belief is a pure function of a BeliefMap.

Test 7 from the spec ("grid overlay can be visible while running even
when resolution control is locked") is expressed here as the same
disable-widget mechanism set_configuration_locked() uses
(locked_during_run_widgets), applied directly to a SteppedSliderRow and a
ToggleSwitch, without instantiating MainWindow -- config_panel.py wires
grid_resolution_input into locked_during_run_widgets and deliberately
leaves grid_overlay_toggle out of it, verified by inspection there.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.render_perf import (
    PerfGuiWarningGate,
    RenderPerfMonitor,
    format_gui_perf_warning,
    format_perf_line,
    format_route_plan_perf_line,
)
from robotics_sim.app.simulation_canvas import MAX_GRID_OVERLAY_CELLS, SimulationCanvas
from robotics_sim.app.widgets import (
    GRID_RESOLUTION_MAX,
    GRID_RESOLUTION_MIN,
    GRID_RESOLUTION_STEP,
    SteppedSliderRow,
    ToggleSwitch,
    grid_resolution_from_slider,
    slider_value_from_grid_resolution,
)
from robotics_sim.environment.belief_map import FREE, OCCUPIED, BeliefMap
from robotics_sim.simulation.config import SimulationConfig
from robotics_sim.simulation.engine import (
    PlannerWorker,
    SimulationControllerMixin,
    occupancy_grid_snapshot_from_belief,
)

_app = QApplication.instance() or QApplication([])


def _draw_overlay_once(canvas: SimulationCanvas) -> None:
    """Exercise draw_grid_overlay() directly against a throwaway QPixmap,
    without a full paintEvent/show() cycle. Only cache/degradation *state*
    is asserted on in tests -- never pixel output -- so this stays a state
    test, not a pixel-perfect rendering test."""
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    canvas.draw_grid_overlay(painter, canvas.plot_rect())
    painter.end()


def _grid_resolution_slider(value: float = 0.50) -> SteppedSliderRow:
    return SteppedSliderRow(
        "Grid resolution",
        value,
        GRID_RESOLUTION_MIN,
        GRID_RESOLUTION_MAX,
        GRID_RESOLUTION_STEP,
        unit_suffix="m/cell",
    )


# ---------------------------------------------------------------------------
# 0. Default remains 0.50 (unchanged from before).
# ---------------------------------------------------------------------------


def test_grid_resolution_default_remains_050():
    assert SimulationConfig().grid_resolution == 0.50


# ---------------------------------------------------------------------------
# 1. Slider tick <-> grid_resolution mapping.
# ---------------------------------------------------------------------------


def test_grid_resolution_slider_maps_to_expected_values():
    assert grid_resolution_from_slider(slider_value_from_grid_resolution(0.50)) == 0.50
    assert grid_resolution_from_slider(slider_value_from_grid_resolution(0.25)) == 0.25
    assert grid_resolution_from_slider(0) == 0.10
    assert grid_resolution_from_slider(18) == 1.00


# ---------------------------------------------------------------------------
# 2. Moving the slider updates the value read into SimulationConfig.
# ---------------------------------------------------------------------------


def test_grid_resolution_slider_updates_config():
    control = _grid_resolution_slider(0.50)
    assert control.value() == 0.50

    control.setValue(0.25)

    # Mirrors read_config()'s grid_resolution=max(0.10, float(control.value())).
    config = SimulationConfig(grid_resolution=max(0.10, float(control.value())))
    assert config.grid_resolution == 0.25


# ---------------------------------------------------------------------------
# 3. The visible label updates with the slider.
# ---------------------------------------------------------------------------


def test_grid_resolution_value_label_updates():
    control = _grid_resolution_slider(0.50)

    control.setValue(0.25)

    assert "0.25" in control.label.text()


# ---------------------------------------------------------------------------
# 4. The temporary red preview still activates when the slider changes.
# ---------------------------------------------------------------------------


def test_temporary_preview_still_activates_when_slider_changes():
    canvas = SimulationCanvas()
    control = _grid_resolution_slider(0.50)
    control.valueChanged.connect(lambda value: canvas.show_grid_resolution_preview(value))

    control.setValue(0.25)

    assert canvas.is_grid_resolution_preview_active() is True
    assert canvas.grid_resolution_preview_value() == 0.25


def test_grid_resolution_preview_auto_hide_timer_is_armed():
    canvas = SimulationCanvas()

    canvas.show_grid_resolution_preview(0.25)

    assert canvas._grid_resolution_preview_timer.isActive(), (
        "showing the preview must arm the auto-hide timer when the persistent "
        "overlay is off, not leave it visible forever"
    )
    assert 700 <= canvas._grid_resolution_preview_timer.interval() <= 1000


def test_preview_auto_hide_timer_is_not_armed_when_overlay_enabled():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)

    canvas.show_grid_resolution_preview(0.25)

    assert not canvas._grid_resolution_preview_timer.isActive(), (
        "the persistent overlay already keeps a grid visible -- the temporary "
        "preview's auto-hide timer should not also be armed"
    )


# ---------------------------------------------------------------------------
# 5. "Show Grid" toggle enables the persistent overlay.
# ---------------------------------------------------------------------------


def test_grid_overlay_toggle_enables_persistent_overlay():
    canvas = SimulationCanvas()
    assert canvas.is_grid_overlay_enabled() is False

    canvas.set_grid_overlay_enabled(True)

    assert canvas.is_grid_overlay_enabled() is True


# ---------------------------------------------------------------------------
# 6. Toggling the overlay must not mutate SimulationConfig.
# ---------------------------------------------------------------------------


def test_grid_overlay_toggle_does_not_mutate_config():
    canvas = SimulationCanvas()
    original_config = canvas.config
    assert original_config.grid_resolution == 0.50

    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.25)

    assert canvas.config is original_config
    assert canvas.config.grid_resolution == 0.50, (
        "the overlay is purely visual -- enabling it or changing its display "
        "resolution must not silently mutate the canvas's real config"
    )


# ---------------------------------------------------------------------------
# 7. The overlay toggle stays interactive while grid_resolution is locked.
#
# Mirrors config_panel.py's real wiring: grid_resolution_input is one of
# the locked_during_run_widgets (disabled while running, via
# set_configuration_locked()); grid_overlay_toggle is deliberately not in
# that list, since it is rendering-only and safe to change mid-run.
# ---------------------------------------------------------------------------


def test_grid_overlay_can_be_visible_while_running_even_when_resolution_control_locked():
    control = _grid_resolution_slider(0.50)
    toggle = ToggleSwitch(False)

    locked_during_run_widgets = [control]  # grid_overlay_toggle intentionally excluded
    for widget in locked_during_run_widgets:
        widget.setEnabled(False)

    assert control.isEnabled() is False, "grid_resolution control must lock while running"

    toggle.setChecked(True)

    assert toggle.isEnabled() is True
    assert toggle.isChecked() is True, (
        "the Show Grid toggle must remain changeable while the simulation is running"
    )


# ---------------------------------------------------------------------------
# 8. Neither the temporary preview nor the persistent overlay changes
#    simulation state.
# ---------------------------------------------------------------------------


def test_grid_overlay_preview_does_not_change_simulation_state():
    canvas = SimulationCanvas()
    original_config = canvas.config
    assert original_config.grid_resolution == 0.50

    # Preview and overlay a DIFFERENT resolution than the canvas's actual config.
    canvas.show_grid_resolution_preview(0.25)
    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.25)

    assert canvas.config is original_config
    assert canvas.config.grid_resolution == 0.50, (
        "previewing or overlaying a different resolution must not silently "
        "mutate the canvas's real config"
    )
    assert canvas.grid_resolution_preview_value() == 0.25


# ---------------------------------------------------------------------------
# Occupancy snapshot: read-only, never a live view into BeliefMap.grid.
# ---------------------------------------------------------------------------


def test_occupancy_grid_snapshot_is_read_only_copy():
    belief = BeliefMap(bounds=(0.0, 5.0, 0.0, 5.0), resolution=0.5)
    belief.grid[0, 0] = OCCUPIED

    snapshot = occupancy_grid_snapshot_from_belief(belief)

    assert snapshot["resolution"] == 0.5
    assert snapshot["bounds"] == belief.bounds
    assert snapshot["grid"][0, 0] == OCCUPIED

    snapshot["grid"][0, 0] = FREE

    assert belief.grid[0, 0] == OCCUPIED, "snapshot must be a copy, not a live view"


def test_occupancy_grid_snapshot_is_none_without_belief_map():
    assert occupancy_grid_snapshot_from_belief(None) is None


def test_grid_overlay_snapshot_setter_does_not_mutate_config():
    canvas = SimulationCanvas()
    belief = BeliefMap(bounds=(0.0, 5.0, 0.0, 5.0), resolution=0.5)
    snapshot = occupancy_grid_snapshot_from_belief(belief)

    canvas.set_grid_overlay_snapshot(snapshot)

    assert canvas.config.grid_resolution == 0.50


# ---------------------------------------------------------------------------
# Part A: render FPS telemetry (RenderPerfMonitor / format_perf_line).
#
# Pure, Qt-free, no real wall-clock waits -- record_frame() accepts an
# explicit `now` so throttling can be driven deterministically.
#
# Field names are paint_fps/paint_ms, not render_fps/frame_ms: what is
# actually measured is paint EVENT frequency and time spent inside
# paintEvent, not a theoretical FPS independently derived from frame_ms --
# those two can disagree once something else (the simulation tick, route
# planning) delays paintEvent calls without directly costing paint time.
# ---------------------------------------------------------------------------


def test_perf_status_uses_paint_fps_and_paint_ms_names():
    monitor = RenderPerfMonitor(log_interval_s=0.0)

    line = monitor.record_frame(
        paint_ms=20.0, overlay_enabled=True, grid_resolution=0.10, now=0.0
    )

    assert line is not None
    assert "paint_fps=" in line
    assert "paint_ms=" in line
    assert "render_fps=" not in line
    assert "frame_ms=" not in line


def test_fps_monitor_reports_periodically():
    monitor = RenderPerfMonitor(log_interval_s=1.0)
    t = 0.0
    emitted_at = []

    for _ in range(120):  # ~2s of frames at 60fps
        t += 1.0 / 60.0
        line = monitor.record_frame(
            paint_ms=16.0, overlay_enabled=False, grid_resolution=0.5, now=t
        )
        if line is not None:
            emitted_at.append(t)

    assert len(emitted_at) >= 1, "must eventually emit a [PERF] line"
    assert len(emitted_at) <= 3, "must emit roughly once per second, not on most frames"


def test_fps_monitor_does_not_emit_every_frame():
    monitor = RenderPerfMonitor(log_interval_s=1.0)
    t = 0.0
    total_frames = 60
    emitted = 0

    for _ in range(total_frames):
        t += 1.0 / 60.0
        line = monitor.record_frame(
            paint_ms=16.0, overlay_enabled=False, grid_resolution=0.5, now=t
        )
        if line is not None:
            emitted += 1

    assert emitted < total_frames


def test_fps_log_includes_overlay_state_and_resolution():
    monitor = RenderPerfMonitor(log_interval_s=0.0)

    line = monitor.record_frame(
        paint_ms=20.0, overlay_enabled=True, grid_resolution=0.10, now=0.0
    )

    assert line is not None
    assert "overlay=on" in line
    assert "grid_res=0.10" in line


def test_perf_line_reports_overlay_off_cleanly():
    monitor = RenderPerfMonitor(log_interval_s=0.0)

    line = monitor.record_frame(
        paint_ms=8.0,
        overlay_enabled=False,
        grid_resolution=0.50,
        visible_cells=None,
        cache_status="off",
        now=0.0,
    )

    assert line is not None
    assert "overlay=off" in line
    assert "visible_cells=n/a" in line
    assert "cache=off" in line


def test_fps_status_formatter_includes_required_fields():
    line = format_perf_line(
        paint_fps=42.7,
        paint_ms=23.4,
        overlay_enabled=True,
        grid_resolution=0.10,
        visible_cells=39200,
        cache_status="hit",
    )

    for field in ("paint_fps", "paint_ms", "overlay", "grid_res", "visible_cells", "cache"):
        assert field in line


def test_route_plan_timing_formatter_if_route_timing_added():
    line = format_route_plan_perf_line(
        route_plan_ms=184.2,
        reason="recovered_after_failure",
        grid_resolution=0.10,
        mapped_obs=716,
        result="ok",
    )

    assert "route_plan_ms=184.2" in line
    assert "reason=recovered_after_failure" in line
    assert "grid_res=0.10" in line
    assert "mapped_obs=716" in line
    assert "result=ok" in line


# ---------------------------------------------------------------------------
# Part B: grid overlay caching, degradation, and snapshot throttling.
# ---------------------------------------------------------------------------


def test_grid_overlay_cache_reused_when_view_and_snapshot_unchanged():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.5)
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.5)
    canvas.set_grid_overlay_snapshot(occupancy_grid_snapshot_from_belief(belief))

    _draw_overlay_once(canvas)
    assert canvas.grid_overlay_cache_status() == "rebuild"

    _draw_overlay_once(canvas)
    assert canvas.grid_overlay_cache_status() == "hit", (
        "same resolution/view/snapshot -- the second draw must reuse the cache"
    )


def test_grid_overlay_cache_invalidates_when_resolution_changes():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.5)
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.5)
    canvas.set_grid_overlay_snapshot(occupancy_grid_snapshot_from_belief(belief))

    _draw_overlay_once(canvas)
    _draw_overlay_once(canvas)
    assert canvas.grid_overlay_cache_status() == "hit"

    canvas.set_grid_overlay_resolution(0.25)
    _draw_overlay_once(canvas)

    assert canvas.grid_overlay_cache_status() == "rebuild"


def test_grid_overlay_cache_invalidates_when_snapshot_changes():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.5)
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.5)
    canvas.set_grid_overlay_snapshot(occupancy_grid_snapshot_from_belief(belief))

    _draw_overlay_once(canvas)
    _draw_overlay_once(canvas)
    assert canvas.grid_overlay_cache_status() == "hit"

    # A genuinely new snapshot push (even with identical content) must bump
    # the version and invalidate the cache -- diffing array contents would
    # be as expensive as the rendering work the cache exists to avoid.
    canvas.set_grid_overlay_snapshot(occupancy_grid_snapshot_from_belief(belief))
    _draw_overlay_once(canvas)

    assert canvas.grid_overlay_cache_status() == "rebuild"


def test_grid_overlay_degrades_when_visible_cell_count_exceeds_cap():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.10)
    # Canvas's default view spans 20m x 16m -- at 0.10 m/cell that is 32000
    # cells, comfortably over MAX_GRID_OVERLAY_CELLS.
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.10)
    canvas.set_grid_overlay_snapshot(occupancy_grid_snapshot_from_belief(belief))

    _draw_overlay_once(canvas)

    assert canvas.grid_overlay_visible_cell_count() > MAX_GRID_OVERLAY_CELLS
    assert canvas.is_grid_overlay_degraded() is True
    assert canvas.grid_overlay_cache_status() == "degraded", (
        "must not attempt a full per-cell draw once the cap is exceeded"
    )


def test_overlay_snapshot_updates_are_throttled():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.5)

    fake_engine = SimpleNamespace(canvas=canvas, belief_map=belief)
    fake_engine.occupancy_grid_snapshot = lambda: SimulationControllerMixin.occupancy_grid_snapshot(fake_engine)

    # Nothing pushed yet -- the first call must push immediately.
    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)
    assert canvas._grid_overlay_snapshot_version == 1

    # Calling again right away is well within the throttle interval.
    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)
    assert canvas._grid_overlay_snapshot_version == 1, (
        "must not copy/push the belief grid again inside the throttle interval"
    )

    # Simulate enough wall time having passed since the last push.
    fake_engine._grid_overlay_snapshot_last_push_time -= 1.0
    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)

    assert canvas._grid_overlay_snapshot_version == 2


# ---------------------------------------------------------------------------
# Part B (this round): PERF diagnostics must NEVER print to the terminal.
# Routine samples are only ever stored in-memory (latest_perf_status) for
# an optional in-app "Show FPS" display; only a rare, heavily throttled
# warning is allowed to reach the GUI console. Nothing reaches stdout.
# ---------------------------------------------------------------------------


def test_perf_does_not_print_to_stdout_by_default(capsys):
    canvas = SimulationCanvas()

    for _ in range(10):
        canvas._report_render_perf(time.perf_counter())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[PERF]" not in captured.out


def test_route_plan_perf_does_not_print_to_stdout_by_default(capsys):
    kwargs = dict(
        planner_type="A*",
        start_xy=(0.0, 0.0),
        goal_xy=(2.0, 2.0),
        obstacles=[],
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        robot_radius=0.2,
        obstacle_points=[(1.0, 1.0)],
        __perf_reason__="test_reason",
    )
    worker = PlannerWorker(request_id=1, planner_kwargs=kwargs, path_simplifier="none")

    worker.run()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[PERF]" not in captured.out
    # route_plan_ms is still measured and stored -- just never printed or
    # appended to any GUI widget from this background-thread worker.
    assert worker.route_plan_ms >= 0.0
    assert worker.route_plan_result == "ok"
    assert "route_plan_ms=" in worker.route_plan_perf_line


def test_routine_perf_samples_do_not_spam_gui_console(capsys):
    canvas = SimulationCanvas()
    before = len(canvas.status_history_lines())

    for _ in range(50):
        canvas._report_render_perf(time.perf_counter())

    captured = capsys.readouterr()
    assert captured.out == ""
    # A near-instant synthetic paint keeps paint_fps healthy, so at most the
    # rare warning path could fire once -- routine samples themselves must
    # never accumulate in the console.
    assert len(canvas.status_history_lines()) <= before + 1


def test_latest_perf_status_is_stored_for_ui(capsys):
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.10)

    canvas._report_render_perf(time.perf_counter())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert canvas.latest_perf_status is not None
    assert "paint_fps" in canvas.latest_perf_status
    assert "paint_ms" in canvas.latest_perf_status
    assert canvas.latest_perf_status["overlay_enabled"] is True
    assert canvas.latest_perf_status["grid_resolution"] == 0.10


def test_perf_warning_can_go_to_gui_console_throttled():
    canvas = SimulationCanvas()
    canvas.set_simulation_running_for_perf(True)
    canvas.set_grid_overlay_enabled(True)
    before = len(canvas.status_history_lines())

    # Force a severely low paint_fps reading directly so the warning gate
    # has something to react to, without needing real slow paint calls.
    canvas._render_perf_monitor.paint_fps = 5.8
    canvas._maybe_emit_perf_gui_warning()
    assert len(canvas.status_history_lines()) == before + 1

    # Immediately again -- still within the throttle interval.
    canvas._maybe_emit_perf_gui_warning()
    assert len(canvas.status_history_lines()) == before + 1, (
        "must not warn again inside the throttle interval"
    )


# ---------------------------------------------------------------------------
# Part C (this round): a low paint_fps is only ever meaningful -- and only
# ever worth a GUI-console line -- while the simulation is actually running
# AND the grid overlay is enabled. Idle/setup/reset and overlay=off must
# never produce a [PERF WARN], even if paint_fps happens to read low.
# ---------------------------------------------------------------------------


def test_perf_warning_not_emitted_before_simulation_running():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    # simulation_running_for_perf defaults to False -- as it is before Start.
    before = len(canvas.status_history_lines())

    canvas._render_perf_monitor.paint_fps = 0.0
    canvas._maybe_emit_perf_gui_warning()

    assert len(canvas.status_history_lines()) == before, (
        "a low paint_fps during setup/load/reset is not meaningful and must not warn"
    )


def test_perf_warning_not_emitted_when_overlay_off():
    canvas = SimulationCanvas()
    canvas.set_simulation_running_for_perf(True)
    # grid_overlay_enabled defaults to False.
    before = len(canvas.status_history_lines())

    canvas._render_perf_monitor.paint_fps = 0.2
    canvas._maybe_emit_perf_gui_warning()

    assert len(canvas.status_history_lines()) == before, (
        "with overlay off, Show Grid cannot be the cause -- must not warn"
    )


def test_perf_warning_emitted_when_simulation_running_and_overlay_on():
    canvas = SimulationCanvas()
    canvas.set_simulation_running_for_perf(True)
    canvas.set_grid_overlay_enabled(True)
    before = len(canvas.status_history_lines())

    canvas._render_perf_monitor.paint_fps = 5.8
    canvas._maybe_emit_perf_gui_warning()

    assert len(canvas.status_history_lines()) == before + 1, (
        "with the simulation running and the overlay on, a genuinely low "
        "paint_fps may produce one throttled warning"
    )


def test_grid_overlay_degraded_notice_not_appended_before_simulation_running():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.10)
    # simulation_running_for_perf defaults to False -- as it is during
    # setup/load, e.g. previewing Show Grid before pressing Start.
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.10)
    canvas.set_grid_overlay_snapshot(occupancy_grid_snapshot_from_belief(belief))
    before = len(canvas.status_history_lines())

    _draw_overlay_once(canvas)

    assert canvas.is_grid_overlay_degraded() is True, "the overlay itself still degrades correctly"
    assert len(canvas.status_history_lines()) == before, (
        "the degraded notice must not appear before the simulation is running"
    )


def test_latest_perf_status_still_updates_even_when_console_warnings_suppressed():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    canvas.set_grid_overlay_resolution(0.10)
    # Not running and/or overlay could be off -- either way, console
    # warnings are suppressed, but in-memory diagnostics must still work.
    before = len(canvas.status_history_lines())

    canvas._report_render_perf(time.perf_counter())

    assert len(canvas.status_history_lines()) == before, (
        "no console warning should appear while idle"
    )
    assert canvas.latest_perf_status is not None
    assert "paint_fps" in canvas.latest_perf_status
    assert canvas.latest_perf_status["overlay_enabled"] is True
    assert canvas.latest_perf_status["grid_resolution"] == 0.10


def test_perf_gui_warning_is_throttled():
    gate = PerfGuiWarningGate(fps_threshold=20.0, interval_s=5.0)

    assert gate.should_warn(5.8, now=0.0) is True
    assert gate.should_warn(5.8, now=0.5) is False, "must not re-warn inside the throttle interval"
    assert gate.should_warn(5.8, now=1.0) is False
    assert gate.should_warn(5.8, now=6.0) is True, "must warn again once the interval has elapsed"

    assert gate.should_warn(42.0, now=100.0) is False, "a healthy fps must never trigger a warning"


def test_perf_gui_warning_format_includes_fps_and_overlay_state():
    line = format_gui_perf_warning(paint_fps=5.8, overlay_enabled=True, grid_resolution=0.10)

    assert "PERF WARN" in line
    assert "5.8" in line
    assert "overlay=on" in line


# ---------------------------------------------------------------------------
# Part C: skip/slow occupancy snapshot pushes while the overlay is degraded
# (grid-lines-only -- the snapshot's cell colors are never drawn then), and
# resume the normal rate once no longer degraded.
# ---------------------------------------------------------------------------


def _fake_engine_with_degraded(canvas: SimulationCanvas, belief: BeliefMap, degraded: bool) -> SimpleNamespace:
    fake_engine = SimpleNamespace(canvas=canvas, belief_map=belief)
    fake_engine.occupancy_grid_snapshot = lambda: SimulationControllerMixin.occupancy_grid_snapshot(fake_engine)
    canvas._grid_overlay_degraded = degraded
    return fake_engine


def test_overlay_degraded_mode_skips_or_slows_snapshot_pushes():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.5)
    fake_engine = _fake_engine_with_degraded(canvas, belief, degraded=True)

    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)
    assert canvas._grid_overlay_snapshot_version == 1

    # Well past the normal 10 Hz interval (0.1s) but still inside the
    # degraded-mode 1 Hz interval -- must not push again yet.
    fake_engine._grid_overlay_snapshot_last_push_time -= 0.3
    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)
    assert canvas._grid_overlay_snapshot_version == 1, (
        "degraded overlay must not copy/push the belief grid at the normal 10 Hz rate"
    )


def test_overlay_degraded_mode_reduces_snapshot_frequency():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.5)
    fake_engine = _fake_engine_with_degraded(canvas, belief, degraded=True)

    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)
    version_after_first = canvas._grid_overlay_snapshot_version

    # Past the degraded-mode 1 Hz interval -- must push again.
    fake_engine._grid_overlay_snapshot_last_push_time -= 1.5
    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)

    assert canvas._grid_overlay_snapshot_version == version_after_first + 1


def test_overlay_snapshot_push_resumes_when_visible_cells_under_cap():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.5)
    fake_engine = _fake_engine_with_degraded(canvas, belief, degraded=True)

    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)
    version_after_first = canvas._grid_overlay_snapshot_version

    # Zoomed in (or otherwise no longer degraded) -- the fast 10 Hz interval
    # applies again immediately, without any extra state to reset.
    canvas._grid_overlay_degraded = False
    fake_engine._grid_overlay_snapshot_last_push_time -= 0.15
    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)

    assert canvas._grid_overlay_snapshot_version == version_after_first + 1


def test_snapshot_frequency_returns_to_normal_when_not_degraded():
    canvas = SimulationCanvas()
    canvas.set_grid_overlay_enabled(True)
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=0.5)
    fake_engine = _fake_engine_with_degraded(canvas, belief, degraded=False)

    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)
    version_after_first = canvas._grid_overlay_snapshot_version

    # Only just past the normal 10 Hz interval -- should already push again
    # (proves the interval is 0.1s, not the slower degraded 1.0s, when not
    # degraded).
    fake_engine._grid_overlay_snapshot_last_push_time -= 0.15
    SimulationControllerMixin.push_grid_overlay_snapshot_if_due(fake_engine)

    assert canvas._grid_overlay_snapshot_version == version_after_first + 1
