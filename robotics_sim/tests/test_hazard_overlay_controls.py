"""
Tests for two independent rendering-only ground-truth DEBUG overlay toggles
on SimulationCanvas (canvas.show_hazard_map / canvas.show_fire_markers,
default OFF) and the minimalist vectorial source beacon they gate:

    - The discovered hazard belief (draw_discovered_hazard()) and its
      discovered fire sources (draw_fire_markers()) are ALWAYS visible.
      Neither toggle can hide them -- both only ever ADD ground-truth debug
      information on top.
    - show_hazard_map=True additionally draws the full ground-truth
      HazardField as a semi-transparent BLUE heatmap (draw_ground_truth_
      hazard_map()), BELOW the warm discovered layer.
    - show_fire_markers=True additionally draws every UNDISCOVERED source
      (tenue blue ring/core); a discovered source is never duplicated --
      it is drawn exactly once, with the "discovered" style.
    - _visible_fire_sources() is the pure anti-omniscience filter deciding
      "discovered": only a source's own CENTER cell being observed=True
      matters, never its radius.
    - _current_fire_marker_context() resolves LIVE vs HISTORY from exactly
      one source of truth each, never mixed.
    - theme.py's hazard_map_*/discovered_hazard_*/fire_* tokens (LIGHT/
      DARK) drive both heatmaps' and the beacon's colors; ThemeMode is part
      of every hazard pixmap cache key, so a theme switch never reuses a
      pixmap built with the other palette.

Same testing approach as test_discovered_hazard_rendering.py: a real
SimulationCanvas instance (needs a QApplication), never .show()'d.
"""
from __future__ import annotations

import inspect
import zlib
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app import simulation_canvas as simulation_canvas_module
from robotics_sim.app.simulation_canvas import SimulationCanvas, _visible_fire_sources
from robotics_sim.app.theme import ThemeMode, theme_colors
from robotics_sim.diagnostics.navigation_snapshot import (
    BeliefMapDebug,
    ControllerDebug,
    FrontierDebug,
    HazardBeliefDebug,
    HazardDebug,
    HazardSourceDebug,
    Maybe,
    NavigationDebugSnapshot,
    PathDebug,
    PlanningGridDebug,
    Pose,
    PredictedMotionDebug,
    RouteValidationDebug,
    SafetyDebug,
    SensorDebug,
)
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief

_app = QApplication.instance() or QApplication([])

_BOUNDS = (0.0, 5.0, 0.0, 5.0)
_RESOLUTION = 1.0  # -> 5x5 grid, cell (row, col) centers at integers + 0.5


def _make_canvas(width: int = 200, height: int = 200) -> SimulationCanvas:
    canvas = SimulationCanvas()
    canvas.resize(width, height)
    return canvas


def _make_belief(robot_count: int = 1) -> HazardBelief:
    return HazardBelief(GridGeometry(_BOUNDS, _RESOLUTION), robot_count=robot_count)


def _payload(belief: HazardBelief) -> dict:
    return {"frame": belief.snapshot(), "bounds": _BOUNDS, "resolution": _RESOLUTION}


def _uniform_grid(value: float) -> np.ndarray:
    return np.full((5, 5), value, dtype=np.float32)


def _hazard_snapshot(sources=(), grid=None, *, version: int = 1) -> dict:
    return {"sources": tuple(sources), "bounds": _BOUNDS, "resolution": _RESOLUTION, "grid": grid, "version": version}


def _fire_source(x: float, y: float, *, fire_id: int = 1, radius: float = 1.0):
    return SimpleNamespace(fire_id=fire_id, position=(x, y), intensity=1.0, radius=radius)


def _draw_once(canvas: SimulationCanvas, method_name: str) -> None:
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    getattr(canvas, method_name)(painter)
    painter.end()


def _pixel_rgb(pixmap: QPixmap, row: int, col: int) -> tuple[int, int, int]:
    """Grid row 0 is the world's lower edge, flipped to the image's bottom
    row by the pixmap builders -- mirror that here."""
    image = pixmap.toImage()
    color = image.pixelColor(col, image.height() - 1 - row)
    return (color.red(), color.green(), color.blue())


def _capture_beacon_calls(monkeypatch) -> list:
    calls: list[dict] = []
    monkeypatch.setattr(
        simulation_canvas_module,
        "_draw_fire_beacon",
        lambda *args, **kwargs: calls.append({"args": args, "discovered": kwargs.get("discovered")}),
    )
    return calls


# ---------------------------------------------------------------------------
# History-snapshot builder -- mirrors test_discovered_hazard_rendering.py's
# _make_history_snapshot(), extended with ground-truth `hazard` sources.
# ---------------------------------------------------------------------------


def _make_belief_map_debug() -> BeliefMapDebug:
    grid = np.full((5, 5), -1, dtype=np.int8)  # UNKNOWN
    explored = np.zeros((1, 5, 5), dtype=np.uint8)
    packed = np.packbits(explored.reshape(-1), bitorder="little")
    return BeliefMapDebug(
        revision=1,
        resolution=_RESOLUTION,
        bounds=_BOUNDS,
        grid_shape=(5, 5),
        grid_zlib=zlib.compress(grid.tobytes(order="C"), level=1),
        explored_shape=(1, 5, 5),
        explored_packbits_zlib=zlib.compress(packed.tobytes(), level=1),
    )


def _make_hazard_belief_debug(belief: HazardBelief) -> HazardBeliefDebug:
    frame = belief.snapshot()
    values = np.ascontiguousarray(frame.values, dtype=np.float32)
    observed = np.ascontiguousarray(frame.observed, dtype=bool)
    observed_by_robot = np.ascontiguousarray(frame.observed_by_robot, dtype=bool)
    packed_observed = np.packbits(observed.reshape(-1), bitorder="little")
    packed_observed_by_robot = np.packbits(observed_by_robot.reshape(-1), bitorder="little")
    return HazardBeliefDebug(
        shape=(int(values.shape[0]), int(values.shape[1])),
        robot_count=belief.robot_count,
        revision=frame.revision,
        values_zlib=zlib.compress(values.tobytes(order="C"), level=1),
        observed_packbits_zlib=zlib.compress(packed_observed.tobytes(), level=1),
        observed_by_robot_packbits_zlib=zlib.compress(packed_observed_by_robot.tobytes(), level=1),
    )


def _make_history_snapshot(
    *,
    snapshot_id: int = 1,
    belief: HazardBelief | None = None,
    hazard_sources: tuple[HazardSourceDebug, ...] | None = None,
    include_hazard_belief: bool = True,
) -> NavigationDebugSnapshot:
    hazard_belief_maybe = (
        Maybe.of(_make_hazard_belief_debug(belief))
        if include_hazard_belief and belief is not None
        else Maybe.missing()
    )
    hazard_maybe = (
        Maybe.of(HazardDebug(version=1, next_fire_id=len(hazard_sources) + 1, sources=hazard_sources))
        if hazard_sources is not None
        else Maybe.missing()
    )
    return NavigationDebugSnapshot(
        snapshot_id=snapshot_id,
        simulation_time=1.0,
        robot_id="R1",
        navigation_state="moving",
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        robot_pose=Pose(x=0.0, y=0.0, theta=0.0, v=0.0),
        path=PathDebug(
            raw_path=Maybe.missing(),
            simplified_path=Maybe.missing(),
            active_path=(),
            pending_path=(),
            active_segment=None,
            active_waypoint_index=None,
            planner_name=Maybe.missing(),
            simplifier_name=Maybe.missing(),
        ),
        route=RouteValidationDebug(first_segment=Maybe.missing(), endpoint_reaches_goal=None),
        predicted_motion=PredictedMotionDebug(trajectory=Maybe.missing(), collision=Maybe.missing()),
        safety=SafetyDebug(robot_radius=0.2, safety_radius=0.3, active_segment=Maybe.missing()),
        planning_grid=PlanningGridDebug(
            start_cell=Maybe.missing(),
            start_cell_world=Maybe.missing(),
            first_waypoint_cell=Maybe.missing(),
            first_waypoint_world=Maybe.missing(),
            unknown_is_traversable=Maybe.missing(),
            start_cell_cleared=Maybe.missing(),
        ),
        controller=ControllerDebug(
            v=0.0, omega=0.0, acceleration=0.0, heading_error=Maybe.missing(), distance_to_goal=Maybe.missing()
        ),
        frontier=FrontierDebug(
            candidate_count=Maybe.missing(),
            selected_target=Maybe.missing(),
            selected_score=Maybe.missing(),
            reason=Maybe.missing(),
        ),
        sensor=SensorDebug(),
        belief_map=Maybe.of(_make_belief_map_debug()),
        hazard=hazard_maybe,
        hazard_belief=hazard_belief_maybe,
    )


def _enter_history(canvas: SimulationCanvas, snapshot, *, position: int = 2, total: int = 10) -> None:
    canvas.navigation_debug_enabled = True
    canvas.set_navigation_debug_snapshot(snapshot)
    canvas.set_navigation_debug_history_position(position, total)


# ---------------------------------------------------------------------------
# 1-2: defaults -- both OFF.
# ---------------------------------------------------------------------------


def test_hazard_map_default_is_false():
    canvas = _make_canvas()
    assert canvas.show_hazard_map is False
    assert canvas.is_hazard_map_enabled() is False


def test_fire_markers_default_is_false():
    canvas = _make_canvas()
    assert canvas.show_fire_markers is False
    assert canvas.is_fire_markers_enabled() is False


# ---------------------------------------------------------------------------
# 3-4: OFF never hides discovered information.
# ---------------------------------------------------------------------------


def test_hazard_map_off_does_not_hide_discovered_hazard():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    assert canvas.show_hazard_map is False

    _draw_once(canvas, "draw_discovered_hazard")

    assert canvas._discovered_hazard_pixmap_cache is not None


def test_fire_markers_off_does_not_hide_discovered_sources(monkeypatch):
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)  # discovered: (2, 2)
    canvas.set_discovered_hazard_frame(_payload(belief))
    discovered_source = _fire_source(2.5, 2.5, fire_id=1)
    undiscovered_source = _fire_source(4.5, 4.5, fire_id=2)  # (4, 4) never observed
    canvas.set_hazard_snapshot(_hazard_snapshot((discovered_source, undiscovered_source)))
    assert canvas.show_fire_markers is False

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert len(calls) == 1
    assert calls[0]["discovered"] is True


# ---------------------------------------------------------------------------
# 5-6: ON only ADDS ground-truth debug information.
# ---------------------------------------------------------------------------


def test_hazard_map_on_adds_the_full_blue_map():
    canvas_off = _make_canvas()
    canvas_off.set_hazard_snapshot(_hazard_snapshot(grid=_uniform_grid(0.6)))
    _draw_once(canvas_off, "draw_ground_truth_hazard_map")
    assert canvas_off._ground_truth_hazard_pixmap_cache is None

    canvas_on = _make_canvas()
    canvas_on.set_hazard_snapshot(_hazard_snapshot(grid=_uniform_grid(0.6)))
    canvas_on.set_hazard_map_enabled(True)
    _draw_once(canvas_on, "draw_ground_truth_hazard_map")
    assert canvas_on._ground_truth_hazard_pixmap_cache is not None


def test_fire_markers_on_adds_undiscovered_sources(monkeypatch):
    canvas = _make_canvas()
    belief = _make_belief()  # nothing observed at all
    canvas.set_discovered_hazard_frame(_payload(belief))
    canvas.set_hazard_snapshot(_hazard_snapshot((_fire_source(2.5, 2.5),)))
    canvas.set_fire_markers_enabled(True)

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert len(calls) == 1
    assert calls[0]["discovered"] is False


# ---------------------------------------------------------------------------
# 7-11: discovered layer always draws; undiscovered gated correctly; no
# duplication.
# ---------------------------------------------------------------------------


def test_discovered_hazard_draws_with_both_toggles_off():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    assert canvas.show_hazard_map is False
    assert canvas.show_fire_markers is False

    _draw_once(canvas, "draw_discovered_hazard")

    assert canvas._discovered_hazard_pixmap_cache is not None


def test_discovered_marker_draws_with_fire_markers_off(monkeypatch):
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    canvas.set_hazard_snapshot(_hazard_snapshot((_fire_source(2.5, 2.5),)))

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert len(calls) == 1
    assert calls[0]["discovered"] is True


def test_undiscovered_marker_does_not_draw_with_fire_markers_off(monkeypatch):
    canvas = _make_canvas()
    belief = _make_belief()  # nothing observed
    canvas.set_discovered_hazard_frame(_payload(belief))
    canvas.set_hazard_snapshot(_hazard_snapshot((_fire_source(2.5, 2.5),)))
    assert canvas.show_fire_markers is False

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert calls == []


def test_undiscovered_marker_does_draw_with_fire_markers_on(monkeypatch):
    canvas = _make_canvas()
    belief = _make_belief()  # nothing observed
    canvas.set_discovered_hazard_frame(_payload(belief))
    canvas.set_hazard_snapshot(_hazard_snapshot((_fire_source(2.5, 2.5),)))
    canvas.set_fire_markers_enabled(True)

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert len(calls) == 1
    assert calls[0]["discovered"] is False


def test_a_discovered_source_is_not_duplicated_with_fire_markers_on(monkeypatch):
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    canvas.set_hazard_snapshot(_hazard_snapshot((_fire_source(2.5, 2.5),)))
    canvas.set_fire_markers_enabled(True)

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert len(calls) == 1, "a discovered source must be drawn exactly once, never twice"
    assert calls[0]["discovered"] is True


# ---------------------------------------------------------------------------
# 6, 17: _visible_fire_sources() -- pure anti-omniscience filter (decides
# "discovered", independent of the toggle logic above).
# ---------------------------------------------------------------------------


def test_unobserved_source_is_not_discovered():
    observed = np.zeros((5, 5), dtype=bool)
    source = _fire_source(2.5, 2.5)

    assert _visible_fire_sources([source], observed, bounds=_BOUNDS, resolution=_RESOLUTION) == []


def test_observed_center_is_discovered():
    observed = np.zeros((5, 5), dtype=bool)
    observed[2, 2] = True
    source = _fire_source(2.5, 2.5)

    assert _visible_fire_sources([source], observed, bounds=_BOUNDS, resolution=_RESOLUTION) == [source]


def test_marker_discovery_does_not_depend_on_radius():
    observed = np.zeros((5, 5), dtype=bool)
    observed[2, 2] = True
    small = _fire_source(2.5, 2.5, fire_id=1, radius=0.1)
    huge = _fire_source(2.5, 2.5, fire_id=2, radius=50.0)

    assert _visible_fire_sources([small], observed, bounds=_BOUNDS, resolution=_RESOLUTION) == [small]
    assert _visible_fire_sources([huge], observed, bounds=_BOUNDS, resolution=_RESOLUTION) == [huge]

    # A huge radius must not rescue a source whose own center is
    # unobserved -- only a neighboring "thermal edge" cell was seen.
    observed_edge_only = np.zeros((5, 5), dtype=bool)
    observed_edge_only[1, 1] = True
    assert _visible_fire_sources([huge], observed_edge_only, bounds=_BOUNDS, resolution=_RESOLUTION) == []
    assert _visible_fire_sources([small], observed_edge_only, bounds=_BOUNDS, resolution=_RESOLUTION) == []


# ---------------------------------------------------------------------------
# 12-13: palettes are actually blue / warm.
# ---------------------------------------------------------------------------


def test_full_hazard_map_uses_a_blue_palette():
    canvas = _make_canvas()
    canvas.set_hazard_snapshot(_hazard_snapshot(grid=_uniform_grid(1.0)))
    canvas.set_hazard_map_enabled(True)

    _draw_once(canvas, "draw_ground_truth_hazard_map")

    pixmap = canvas._ground_truth_hazard_pixmap_cache
    assert pixmap is not None
    r, g, b = _pixel_rgb(pixmap, 2, 2)
    assert b > r, "the full ground-truth map must read blue, never warm red/orange"


def test_discovered_hazard_uses_a_warm_palette():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [1.0], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))

    _draw_once(canvas, "draw_discovered_hazard")

    pixmap = canvas._discovered_hazard_pixmap_cache
    assert pixmap is not None
    r, g, b = _pixel_rgb(pixmap, 2, 2)
    assert r > b, "the discovered heatmap must read warm, never cold blue"


# ---------------------------------------------------------------------------
# 14: paint order -- full ground-truth map below the discovered map.
# ---------------------------------------------------------------------------


def test_full_map_draws_below_discovered_map():
    source = inspect.getsource(SimulationCanvas.draw_plot)
    ground_truth_pos = source.index("self.draw_ground_truth_hazard_map(painter)")
    discovered_pos = source.index("self.draw_discovered_hazard(painter)")
    assert ground_truth_pos < discovered_pos


# ---------------------------------------------------------------------------
# 15-16: the beacon shape -- circles/rings only, no flame, no images.
# ---------------------------------------------------------------------------


def test_marker_shape_does_not_use_a_flame_path():
    assert not hasattr(simulation_canvas_module, "_fire_marker_flame_path")
    assert not hasattr(simulation_canvas_module, "_draw_fire_marker")
    source = inspect.getsource(simulation_canvas_module._draw_fire_beacon)
    # Checked as an actual code pattern (constructor call), not a bare
    # substring -- the function's own docstring mentions "QPainterPath" in
    # prose (explaining what must NOT happen) without violating the rule.
    assert "QPainterPath(" not in source
    assert "cubicTo" not in source


def test_marker_uses_circles_and_rings_not_emoji_or_images():
    source = inspect.getsource(simulation_canvas_module._draw_fire_beacon)
    assert source.count("drawEllipse") >= 3  # halo, outer ring, core
    for forbidden in ("QPixmap(", "QImage(", "QTimer(", ".png", ".svg", ".jpg"):
        assert forbidden not in source
    assert source.isascii(), "no emoji/unicode pictograph literals"


# ---------------------------------------------------------------------------
# 17-18: theme tokens and cache invalidation.
# ---------------------------------------------------------------------------


def test_light_and_dark_produce_different_palettes():
    light = theme_colors(ThemeMode.LIGHT)
    dark = theme_colors(ThemeMode.DARK)
    fields = (
        "hazard_map_low", "hazard_map_mid", "hazard_map_high",
        "discovered_hazard_low", "discovered_hazard_mid", "discovered_hazard_high", "discovered_hazard_core",
        "fire_discovered_ring", "fire_discovered_core", "fire_undiscovered_ring", "fire_undiscovered_core",
    )
    for field in fields:
        assert getattr(light, field) != getattr(dark, field), field


def test_theme_change_correctly_invalidates_hazard_pixmaps():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    canvas.set_hazard_snapshot(_hazard_snapshot(grid=_uniform_grid(0.8)))
    canvas.set_hazard_map_enabled(True)

    _draw_once(canvas, "draw_discovered_hazard")
    _draw_once(canvas, "draw_ground_truth_hazard_map")
    light_discovered = canvas._discovered_hazard_pixmap_cache
    light_ground_truth = canvas._ground_truth_hazard_pixmap_cache
    light_discovered_rgb = _pixel_rgb(light_discovered, 2, 2)
    light_ground_truth_rgb = _pixel_rgb(light_ground_truth, 2, 2)

    canvas.set_theme_mode(ThemeMode.DARK)
    _draw_once(canvas, "draw_discovered_hazard")
    _draw_once(canvas, "draw_ground_truth_hazard_map")

    assert canvas._discovered_hazard_pixmap_cache is not light_discovered
    assert canvas._ground_truth_hazard_pixmap_cache is not light_ground_truth
    assert _pixel_rgb(canvas._discovered_hazard_pixmap_cache, 2, 2) != light_discovered_rgb
    assert _pixel_rgb(canvas._ground_truth_hazard_pixmap_cache, 2, 2) != light_ground_truth_rgb

    # And switching back to LIGHT must not reuse a stale DARK pixmap either.
    canvas.set_theme_mode(ThemeMode.LIGHT)
    _draw_once(canvas, "draw_discovered_hazard")
    assert _pixel_rgb(canvas._discovered_hazard_pixmap_cache, 2, 2) == light_discovered_rgb


# ---------------------------------------------------------------------------
# 19-20: other canvas state stays decoupled.
# ---------------------------------------------------------------------------


def test_grid_toggle_remains_independent():
    canvas = _make_canvas()
    canvas.set_hazard_map_enabled(True)
    canvas.set_fire_markers_enabled(True)
    assert canvas.grid_overlay_enabled is False

    canvas.set_grid_overlay_enabled(True)

    assert canvas.show_hazard_map is True
    assert canvas.show_fire_markers is True


def test_toggles_do_not_change_belief_planning_or_ground_truth():
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    payload = _payload(belief)
    canvas.set_discovered_hazard_frame(payload)
    hazard_snapshot = _hazard_snapshot((_fire_source(2.5, 2.5),), grid=_uniform_grid(0.5))
    canvas.set_hazard_snapshot(hazard_snapshot)

    frame_before = canvas._discovered_hazard_frame
    snapshot_before = canvas._hazard_snapshot

    canvas.set_hazard_map_enabled(True)
    canvas.set_fire_markers_enabled(True)
    canvas.set_hazard_map_enabled(False)
    canvas.set_fire_markers_enabled(False)

    assert canvas._discovered_hazard_frame is frame_before  # same object -- never rebuilt/reset
    assert canvas._hazard_snapshot is snapshot_before

    for setter in (SimulationCanvas.set_hazard_map_enabled, SimulationCanvas.set_fire_markers_enabled):
        source = inspect.getsource(setter)
        for forbidden in ("HazardBelief", "hazard_service", "planning_costmap", "observe_", "FireSource("):
            assert forbidden not in source


# ---------------------------------------------------------------------------
# LIVE vs HISTORY context resolution -- unchanged architecture invariant,
# still exercised end-to-end through the new draw_fire_markers() logic.
# ---------------------------------------------------------------------------


def test_history_uses_historical_source_and_belief(monkeypatch):
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    source = HazardSourceDebug(fire_id=1, position=(2.5, 2.5), intensity=1.0, radius=1.0)
    snapshot = _make_history_snapshot(belief=belief, hazard_sources=(source,))
    _enter_history(canvas, snapshot)

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert len(calls) == 1
    assert calls[0]["discovered"] is True
    expected_sx, expected_sy = canvas.world_to_screen(2.5, 2.5)
    _painter, sx, sy, _colors = calls[0]["args"]
    assert (sx, sy) == (expected_sx, expected_sy)


def test_history_without_belief_draws_no_marker(monkeypatch):
    """Ground truth IS present (a real HazardDebug with a source) but the
    same historical snapshot has no HazardBeliefDebug -- must draw nothing,
    never fall back to ground truth alone (even with Fire Markers ON)."""
    canvas = _make_canvas()
    canvas.set_fire_markers_enabled(True)
    source = HazardSourceDebug(fire_id=1, position=(2.5, 2.5), intensity=1.0, radius=1.0)
    snapshot = _make_history_snapshot(belief=None, hazard_sources=(source,), include_hazard_belief=False)
    _enter_history(canvas, snapshot)

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert calls == []
    assert canvas._current_fire_marker_context() is None


def test_live_does_not_use_a_historical_source(monkeypatch):
    """A LIVE source at one cell and a DIFFERENT historical source at
    another cell are both wired up; while the view is LIVE (no history
    selected), only the live source may produce a marker."""
    canvas = _make_canvas()
    live_belief = _make_belief()
    live_belief.observe_cells([0], [0], [0.9], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(live_belief))
    canvas.set_hazard_snapshot(_hazard_snapshot((_fire_source(0.5, 0.5, fire_id=1),)))

    history_belief = _make_belief()
    history_belief.observe_cells([4], [4], [0.5], robot_index=0)
    history_source = HazardSourceDebug(fire_id=2, position=(4.5, 4.5), intensity=1.0, radius=1.0)
    canvas.set_navigation_debug_snapshot(
        _make_history_snapshot(belief=history_belief, hazard_sources=(history_source,))
    )
    assert canvas.navigation_debug_enabled is False

    calls = _capture_beacon_calls(monkeypatch)
    _draw_once(canvas, "draw_fire_markers")

    assert len(calls) == 1
    expected_sx, expected_sy = canvas.world_to_screen(0.5, 0.5)
    _painter, sx, sy, _colors = calls[0]["args"]
    assert (sx, sy) == (expected_sx, expected_sy)


def test_fire_beacon_paints_without_crashing_end_to_end():
    """One real, undoctored draw_fire_markers() call -- proof the beacon
    layers actually render through real QPainter calls, not just that the
    monkeypatched unit tests above wire correctly."""
    canvas = _make_canvas()
    belief = _make_belief()
    belief.observe_cells([2], [2], [0.8], robot_index=0)
    canvas.set_discovered_hazard_frame(_payload(belief))
    canvas.set_hazard_snapshot(_hazard_snapshot((_fire_source(2.5, 2.5), _fire_source(4.5, 4.5, fire_id=2))))
    canvas.set_fire_markers_enabled(True)
    canvas.set_hazard_map_enabled(True)

    _draw_once(canvas, "draw_ground_truth_hazard_map")  # must not raise
    _draw_once(canvas, "draw_discovered_hazard")  # must not raise
    _draw_once(canvas, "draw_fire_markers")  # must not raise
