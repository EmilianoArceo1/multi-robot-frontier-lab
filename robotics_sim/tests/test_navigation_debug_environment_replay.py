"""
Tests for the canvas's decoded historical-environment replay pipeline:
SimulationCanvas._decoded_navigation_debug_environment() (the compressed
BeliefMapDebug -> numpy grid/explored-mask decode, revision-keyed cache) and
its two consumers, draw_explored_area_trace()/_draw_historical_explored_area()
and draw_sensor_range()'s sensor-polygon replay branch.

These exercise only the canvas: a real SimulationCanvas instance (needs a
QApplication, never .show()'d), pushed a fabricated NavigationDebugSnapshot --
same approach as test_navigation_debug_canvas_wiring.py. No engine, no Robot,
no restore: this is "does the canvas correctly render what a snapshot says",
independent of whether that snapshot came from live capture or a restore.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.diagnostics.navigation_snapshot import (
    BeliefMapDebug,
    ControllerDebug,
    FrontierDebug,
    HazardBeliefDebug,
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

_app = QApplication.instance() or QApplication([])

_BOUNDS = (-2.0, 2.0, -2.0, 2.0)
_RESOLUTION = 1.0
_GRID_SHAPE = (4, 4)


def _make_belief_frame(*, revision: int, occupied_cell: tuple[int, int]) -> BeliefMapDebug:
    grid = np.full(_GRID_SHAPE, -1, dtype=np.int8)  # UNKNOWN
    grid[occupied_cell] = 1  # OCCUPIED
    explored = np.zeros((1,) + _GRID_SHAPE, dtype=np.uint8)
    explored[0, occupied_cell[0], occupied_cell[1]] = 1
    packed = np.packbits(explored.reshape(-1), bitorder="little")
    return BeliefMapDebug(
        revision=revision,
        resolution=_RESOLUTION,
        bounds=_BOUNDS,
        grid_shape=_GRID_SHAPE,
        grid_zlib=zlib.compress(grid.tobytes(order="C"), level=1),
        explored_shape=(1,) + _GRID_SHAPE,
        explored_packbits_zlib=zlib.compress(packed.tobytes(), level=1),
    )


def _make_hazard_belief_frame(*, revision: int, hot_cell: tuple[int, int], hot_value: float = 0.8) -> HazardBeliefDebug:
    values = np.zeros(_GRID_SHAPE, dtype=np.float32)
    values[hot_cell] = hot_value
    observed = np.zeros(_GRID_SHAPE, dtype=bool)
    observed[hot_cell] = True
    observed_by_robot = observed.reshape((1,) + _GRID_SHAPE)
    packed_observed = np.packbits(observed.reshape(-1), bitorder="little")
    packed_observed_by_robot = np.packbits(observed_by_robot.reshape(-1), bitorder="little")
    return HazardBeliefDebug(
        shape=_GRID_SHAPE,
        robot_count=1,
        revision=revision,
        values_zlib=zlib.compress(values.tobytes(order="C"), level=1),
        observed_packbits_zlib=zlib.compress(packed_observed.tobytes(), level=1),
        observed_by_robot_packbits_zlib=zlib.compress(packed_observed_by_robot.tobytes(), level=1),
    )


def _make_sensor(polygon: list[tuple[float, float]] | None) -> SensorDebug:
    if not polygon:
        return SensorDebug()
    points = np.asarray(polygon, dtype=np.float32)
    return SensorDebug(
        vision_range=5.0,
        visible_polygon_count=len(polygon),
        visible_polygon_f32_zlib=zlib.compress(points.tobytes(order="C"), level=1),
    )


def _make_snapshot(
    *,
    snapshot_id: int,
    belief_frame: BeliefMapDebug | None,
    polygon: list[tuple[float, float]] | None = None,
    hazard_belief_frame: HazardBeliefDebug | None = None,
) -> NavigationDebugSnapshot:
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
        sensor=_make_sensor(polygon),
        belief_map=Maybe.of(belief_frame) if belief_frame is not None else Maybe.missing(),
        hazard_belief=Maybe.of(hazard_belief_frame) if hazard_belief_frame is not None else Maybe.missing(),
    )


def _make_canvas() -> SimulationCanvas:
    canvas = SimulationCanvas()
    canvas.resize(300, 240)
    canvas.navigation_debug_enabled = True
    return canvas


def _select_history(canvas: SimulationCanvas, snapshot, *, position: int = 1, total: int = 1) -> None:
    canvas.set_navigation_debug_snapshot(snapshot)
    canvas.set_navigation_debug_history_position(position, total)


# ---------------------------------------------------------------------------
# _decoded_navigation_debug_environment()
# ---------------------------------------------------------------------------


def test_returns_none_in_live_view():
    canvas = _make_canvas()
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)))
    canvas.set_navigation_debug_snapshot(snapshot)
    canvas.set_navigation_debug_history_position(None, 1)  # LIVE, not HISTORY

    assert canvas._decoded_navigation_debug_environment() is None


def test_returns_none_when_navigation_debug_disabled():
    canvas = _make_canvas()
    canvas.navigation_debug_enabled = False
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)))
    _select_history(canvas, snapshot)

    assert canvas._decoded_navigation_debug_environment() is None


def test_returns_none_when_snapshot_has_no_belief_map():
    canvas = _make_canvas()
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=None)
    _select_history(canvas, snapshot)

    assert canvas._decoded_navigation_debug_environment() is None


def test_decodes_grid_and_explored_mask_correctly():
    canvas = _make_canvas()
    frame = _make_belief_frame(revision=7, occupied_cell=(2, 3))
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=frame)
    _select_history(canvas, snapshot)

    decoded = canvas._decoded_navigation_debug_environment()

    assert decoded is not None
    assert decoded["grid"].shape == _GRID_SHAPE
    assert decoded["grid"][2, 3] == 1  # OCCUPIED
    assert decoded["grid"][0, 0] == -1  # UNKNOWN elsewhere
    assert decoded["explored_by_robot"].shape == (1,) + _GRID_SHAPE
    assert bool(decoded["explored_by_robot"][0, 2, 3]) is True
    assert decoded["resolution"] == _RESOLUTION
    assert decoded["bounds"] == _BOUNDS


def test_cache_reused_for_the_same_frame_revision():
    canvas = _make_canvas()
    frame = _make_belief_frame(revision=1, occupied_cell=(0, 0))
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=frame)
    _select_history(canvas, snapshot)

    first = canvas._decoded_navigation_debug_environment()
    second = canvas._decoded_navigation_debug_environment()

    assert second is first, "identical (frame identity, revision) must not re-decompress/re-decode"


def test_cache_rebuilds_on_new_revision():
    canvas = _make_canvas()
    snapshot_a = _make_snapshot(snapshot_id=1, belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)))
    _select_history(canvas, snapshot_a)
    first = canvas._decoded_navigation_debug_environment()

    snapshot_b = _make_snapshot(snapshot_id=2, belief_frame=_make_belief_frame(revision=2, occupied_cell=(1, 1)))
    _select_history(canvas, snapshot_b)
    second = canvas._decoded_navigation_debug_environment()

    assert second is not first
    assert second["grid"][1, 1] == 1


def test_decode_key_cleared_when_snapshot_reset_to_none():
    canvas = _make_canvas()
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)))
    _select_history(canvas, snapshot)
    assert canvas._decoded_navigation_debug_environment() is not None

    canvas.set_navigation_debug_snapshot(None)

    assert canvas._nav_debug_environment_decode_key is None
    assert canvas._nav_debug_environment_decoded is None


# ---------------------------------------------------------------------------
# _decoded_navigation_debug_hazard_belief() -- Team HazardBelief's own
# decode/cache pipeline, kept entirely separate from _decoded_navigation_
# debug_environment() above (BeliefMapDebug) even though both read from the
# same selected historical snapshot.
# ---------------------------------------------------------------------------


def test_hazard_belief_returns_none_in_live_view():
    canvas = _make_canvas()
    snapshot = _make_snapshot(
        snapshot_id=1,
        belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)),
        hazard_belief_frame=_make_hazard_belief_frame(revision=1, hot_cell=(1, 1)),
    )
    canvas.set_navigation_debug_snapshot(snapshot)
    canvas.set_navigation_debug_history_position(None, 1)  # LIVE, not HISTORY

    assert canvas._decoded_navigation_debug_hazard_belief() is None


def test_hazard_belief_returns_none_when_navigation_debug_disabled():
    canvas = _make_canvas()
    canvas.navigation_debug_enabled = False
    snapshot = _make_snapshot(
        snapshot_id=1,
        belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)),
        hazard_belief_frame=_make_hazard_belief_frame(revision=1, hot_cell=(1, 1)),
    )
    _select_history(canvas, snapshot)

    assert canvas._decoded_navigation_debug_hazard_belief() is None


def test_hazard_belief_returns_none_when_snapshot_has_no_hazard_belief_field():
    """An old snapshot captured before HazardBeliefDebug existed -- must hide
    the layer, never fall back to the live frame or ground truth (see
    draw_discovered_hazard()'s own history branch)."""
    canvas = _make_canvas()
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)))
    _select_history(canvas, snapshot)

    assert canvas._decoded_navigation_debug_hazard_belief() is None


def test_hazard_belief_returns_none_when_snapshot_has_no_belief_map():
    """hazard_belief present but belief_map missing -- an inconsistent
    capture that should never happen in practice (both are captured from
    the same tick), but bounds/resolution come from belief_map alone (see
    the method's own docstring), so this must degrade to "hide", not crash."""
    canvas = _make_canvas()
    snapshot = _make_snapshot(
        snapshot_id=1, belief_frame=None, hazard_belief_frame=_make_hazard_belief_frame(revision=1, hot_cell=(1, 1))
    )
    _select_history(canvas, snapshot)

    assert canvas._decoded_navigation_debug_hazard_belief() is None


def test_hazard_belief_decodes_values_and_observed_correctly():
    canvas = _make_canvas()
    frame = _make_hazard_belief_frame(revision=1, hot_cell=(2, 3), hot_value=0.6)
    snapshot = _make_snapshot(
        snapshot_id=1, belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)), hazard_belief_frame=frame
    )
    _select_history(canvas, snapshot)

    decoded = canvas._decoded_navigation_debug_hazard_belief()

    assert decoded is not None
    assert decoded["values"].shape == _GRID_SHAPE
    assert decoded["values"].dtype == np.float32
    assert decoded["values"][2, 3] == pytest.approx(0.6)
    assert decoded["observed"][2, 3] == True  # noqa: E712
    assert decoded["observed"][0, 0] == False  # noqa: E712
    assert decoded["resolution"] == _RESOLUTION
    assert decoded["bounds"] == _BOUNDS


def test_hazard_belief_cache_reused_for_the_same_frame_revision():
    canvas = _make_canvas()
    frame = _make_hazard_belief_frame(revision=1, hot_cell=(0, 0))
    snapshot = _make_snapshot(
        snapshot_id=1, belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)), hazard_belief_frame=frame
    )
    _select_history(canvas, snapshot)

    first = canvas._decoded_navigation_debug_hazard_belief()
    second = canvas._decoded_navigation_debug_hazard_belief()

    assert second is first, "identical (frame identity, revision) must not re-decompress/re-decode"


def test_hazard_belief_cache_rebuilds_on_new_revision():
    canvas = _make_canvas()
    snapshot_a = _make_snapshot(
        snapshot_id=1,
        belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)),
        hazard_belief_frame=_make_hazard_belief_frame(revision=1, hot_cell=(0, 0)),
    )
    _select_history(canvas, snapshot_a)
    first = canvas._decoded_navigation_debug_hazard_belief()

    snapshot_b = _make_snapshot(
        snapshot_id=2,
        belief_frame=_make_belief_frame(revision=2, occupied_cell=(1, 1)),
        hazard_belief_frame=_make_hazard_belief_frame(revision=2, hot_cell=(1, 1)),
    )
    _select_history(canvas, snapshot_b)
    second = canvas._decoded_navigation_debug_hazard_belief()

    assert second is not first
    assert second["observed"][1, 1] == True  # noqa: E712


def test_hazard_belief_decode_key_cleared_when_snapshot_reset_to_none():
    canvas = _make_canvas()
    snapshot = _make_snapshot(
        snapshot_id=1,
        belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)),
        hazard_belief_frame=_make_hazard_belief_frame(revision=1, hot_cell=(0, 0)),
    )
    _select_history(canvas, snapshot)
    assert canvas._decoded_navigation_debug_hazard_belief() is not None

    canvas.set_navigation_debug_snapshot(None)

    assert canvas._nav_debug_hazard_belief_decode_key is None
    assert canvas._nav_debug_hazard_belief_decoded is None


# ---------------------------------------------------------------------------
# Consumers: explored-area raster + sensor-polygon replay must read the
# decoded historical environment while a HISTORY frame is selected, and fall
# back to the live path otherwise (covered by test_navigation_debug_canvas_
# wiring.py / test_canvas_render_cache.py for the LIVE side).
# ---------------------------------------------------------------------------


def _paint_once(canvas: SimulationCanvas, method_name: str) -> None:
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    getattr(canvas, method_name)(painter)
    painter.end()


def test_draw_explored_area_trace_rasterizes_the_historical_mask_without_crashing():
    canvas = _make_canvas()
    frame = _make_belief_frame(revision=1, occupied_cell=(0, 0))
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=frame)
    _select_history(canvas, snapshot)

    _paint_once(canvas, "draw_explored_area_trace")

    assert canvas._nav_debug_explored_cache is not None


def test_draw_explored_area_trace_cache_keyed_on_frame_revision():
    canvas = _make_canvas()
    frame = _make_belief_frame(revision=3, occupied_cell=(0, 0))
    snapshot = _make_snapshot(snapshot_id=1, belief_frame=frame)
    _select_history(canvas, snapshot)
    _paint_once(canvas, "draw_explored_area_trace")
    first_cache = canvas._nav_debug_explored_cache

    _paint_once(canvas, "draw_explored_area_trace")

    assert canvas._nav_debug_explored_cache is first_cache, "same revision + view must not re-rasterize"


def test_sensor_range_replays_the_exact_captured_polygon_in_history():
    canvas = _make_canvas()
    canvas.config.show_vision = True
    polygon = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    snapshot = _make_snapshot(
        snapshot_id=1,
        belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)),
        polygon=polygon,
    )
    _select_history(canvas, snapshot)

    captured = {}
    canvas.draw_sensor_polygon = lambda painter, poly, color: captured.setdefault("polygon", poly)
    _paint_once(canvas, "draw_sensor_range")

    assert captured["polygon"] == polygon


def test_sensor_range_falls_back_to_recomputed_polygon_when_snapshot_has_none():
    canvas = _make_canvas()
    canvas.config.show_vision = True
    snapshot = _make_snapshot(
        snapshot_id=1,
        belief_frame=_make_belief_frame(revision=1, occupied_cell=(0, 0)),
        polygon=None,
    )
    _select_history(canvas, snapshot)

    captured = {}
    canvas.draw_sensor_polygon = lambda painter, poly, color: captured.setdefault("polygon", poly)
    _paint_once(canvas, "draw_sensor_range")

    # No compressed polygon on the snapshot -> must not crash, and must not
    # silently draw nothing either.
    assert "polygon" in captured


# ---------------------------------------------------------------------------
# set_explored_area_seed() -- restored-belief coverage reseeding the LIVE
# explored-area cache (see engine.restore_navigation_debug_snapshot() /
# canvas.rebuild_explored_area_cache()). This is what the canvas shows once
# the view has returned to LIVE after a restore -- independent of the
# history-replay pipeline exercised above.
# ---------------------------------------------------------------------------


def _make_mask(true_cell: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((1,) + _GRID_SHAPE, dtype=bool)
    mask[0, true_cell[0], true_cell[1]] = True
    return mask


def _make_live_canvas() -> SimulationCanvas:
    canvas = SimulationCanvas()
    canvas.resize(300, 240)
    canvas.navigation_debug_enabled = False  # LIVE view, not history replay
    return canvas


def test_seed_paints_into_the_live_cache_immediately():
    canvas = _make_live_canvas()
    canvas.set_explored_area_seed(_make_mask((2, 3)), _RESOLUTION, _BOUNDS)

    _paint_once(canvas, "draw_explored_area_trace")

    assert canvas._explored_area_cache is not None


def test_seed_survives_a_cache_rebuild():
    """The actual bug this fixes: previously, once the bounded sensor-sweep
    polygon list was empty (as it always is right after a restore, or
    whenever a resize/pan/zoom forces a rebuild with few recent sweeps),
    nothing survived to show the already-explored coverage. The seed must
    be replayed on every rebuild, not just painted once."""
    canvas = _make_live_canvas()
    canvas.set_explored_area_seed(_make_mask((1, 1)), _RESOLUTION, _BOUNDS)
    _paint_once(canvas, "draw_explored_area_trace")
    assert canvas._explored_area_cache is not None

    # Simulate what a resize/pan/zoom does: invalidate the cache outright.
    canvas.invalidate_explored_area_cache()
    assert canvas._explored_area_cache is None

    _paint_once(canvas, "draw_explored_area_trace")

    assert canvas._explored_area_cache is not None


def test_new_live_sweep_polygon_paints_on_top_of_the_seed_without_erasing_it():
    canvas = _make_live_canvas()
    canvas.set_explored_area_seed(_make_mask((0, 0)), _RESOLUTION, _BOUNDS)
    _paint_once(canvas, "draw_explored_area_trace")
    seeded_cache = canvas._explored_area_cache
    assert seeded_cache is not None

    canvas.append_explored_area_polygon([(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)])

    # Painted incrementally onto the existing (seeded) cache object, not a
    # freshly rebuilt one that would have dropped the seed.
    assert canvas._explored_area_cache is seeded_cache


def test_clear_explored_area_seed_removes_it_from_the_next_rebuild():
    canvas = _make_live_canvas()
    canvas.set_explored_area_seed(_make_mask((0, 0)), _RESOLUTION, _BOUNDS)

    canvas.clear_explored_area_seed()

    assert canvas._explored_area_seed_mask is None
    canvas.invalidate_explored_area_cache()
    _paint_once(canvas, "draw_explored_area_trace")  # must not crash with no seed and no polygons
    assert canvas._explored_area_cache is None  # nothing to draw -- show_explored_area path exits early
