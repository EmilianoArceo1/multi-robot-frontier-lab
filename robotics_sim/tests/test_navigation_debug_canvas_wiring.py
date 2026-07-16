"""
Tests for the navigation-debug canvas wiring: cache reuse keyed on
snapshot_id, the toggle mutating no simulation state, the layer never
recomputing a decision, and simulation_canvas.py never importing concrete
planning/collision-checking code (the anti-pattern the architecture
explicitly forbids).

Same testing approach as test_canvas_render_cache.py / test_grid_resolution_
preview.py: a real SimulationCanvas instance (needs a QApplication), never
.show()'d -- only cache/state is asserted on, never pixel output.
"""
from __future__ import annotations

import ast
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.diagnostics.navigation_snapshot import (
    ControllerDebug,
    FrontierDebug,
    Maybe,
    NavigationDebugSnapshot,
    PathDebug,
    PlanningGridDebug,
    Pose,
    PredictedMotionDebug,
    RouteValidationDebug,
    SafetyDebug,
)

_app = QApplication.instance() or QApplication([])


def _make_canvas(width: int = 400, height: int = 300) -> SimulationCanvas:
    canvas = SimulationCanvas()
    canvas.resize(width, height)
    return canvas


def _make_snapshot(snapshot_id: int) -> NavigationDebugSnapshot:
    return NavigationDebugSnapshot(
        snapshot_id=snapshot_id,
        simulation_time=1.0,
        robot_id="R1",
        navigation_state="moving",
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        robot_pose=Pose(x=0.0, y=0.0, theta=0.0, v=0.3),
        path=PathDebug(
            raw_path=Maybe.of(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            simplified_path=Maybe.of(((0.0, 0.0), (2.0, 0.0))),
            active_path=((0.0, 0.0), (2.0, 0.0)),
            pending_path=(),
            active_segment=((0.0, 0.0), (2.0, 0.0)),
            active_waypoint_index=0,
            planner_name=Maybe.of("A*"),
            simplifier_name=Maybe.of("Direction changes"),
        ),
        route=RouteValidationDebug(first_segment=Maybe.missing(), endpoint_reaches_goal=None),
        predicted_motion=PredictedMotionDebug(trajectory=Maybe.missing(), collision=Maybe.missing()),
        safety=SafetyDebug(robot_radius=0.2, safety_radius=0.35, active_segment=Maybe.missing()),
        planning_grid=PlanningGridDebug(
            start_cell=Maybe.missing(),
            start_cell_world=Maybe.of((0.0, 0.0)),
            first_waypoint_cell=Maybe.missing(),
            first_waypoint_world=Maybe.missing(),
            unknown_is_traversable=Maybe.of(True),
            start_cell_cleared=Maybe.of(False),
        ),
        controller=ControllerDebug(
            v=0.3, omega=0.0, acceleration=0.0, heading_error=Maybe.of(0.0), distance_to_goal=Maybe.of(2.0)
        ),
        frontier=FrontierDebug(
            candidate_count=Maybe.missing(),
            selected_target=Maybe.missing(),
            selected_score=Maybe.missing(),
            reason=Maybe.missing(),
        ),
    )


def _draw_navigation_debug_overlay_once(canvas: SimulationCanvas) -> None:
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    canvas.draw_navigation_debug_overlay(painter)
    painter.end()


# ---------------------------------------------------------------------------
# Cache reuse keyed on (snapshot_id, view transform).
# ---------------------------------------------------------------------------


def test_second_render_of_same_snapshot_reuses_cache():
    canvas = _make_canvas()
    canvas.set_navigation_debug_snapshot(_make_snapshot(1))

    _draw_navigation_debug_overlay_once(canvas)
    first_cache = canvas._nav_debug_overlay_cache
    assert first_cache is not None

    _draw_navigation_debug_overlay_once(canvas)
    second_cache = canvas._nav_debug_overlay_cache
    assert second_cache is first_cache, "identical snapshot_id + view must not rebuild the overlay cache"


def test_new_snapshot_id_rebuilds_cache():
    canvas = _make_canvas()
    canvas.set_navigation_debug_snapshot(_make_snapshot(1))
    _draw_navigation_debug_overlay_once(canvas)
    first_cache = canvas._nav_debug_overlay_cache

    canvas.set_navigation_debug_snapshot(_make_snapshot(2))
    _draw_navigation_debug_overlay_once(canvas)
    second_cache = canvas._nav_debug_overlay_cache

    assert second_cache is not first_cache


# ---------------------------------------------------------------------------
# Pausing == no new set_navigation_debug_snapshot() calls. The canvas must
# never clear _nav_debug_snapshot on its own (no idle-frame path exists that
# would), so the last relevant snapshot survives any number of repaints.
# ---------------------------------------------------------------------------


def test_snapshot_survives_repeated_draws_with_no_new_push():
    canvas = _make_canvas()
    canvas.navigation_debug_enabled = True
    canvas.set_navigation_debug_snapshot(_make_snapshot(5))

    for _ in range(5):
        canvas.paintEvent(None)

    assert canvas.navigation_debug_snapshot().snapshot_id == 5


# ---------------------------------------------------------------------------
# Toggling the layer changes no simulation-facing state.
# ---------------------------------------------------------------------------


def test_toggling_navigation_debug_mutates_no_simulation_state():
    canvas = _make_canvas()
    canvas.robot = None
    config_before = canvas.config
    robot_before = canvas.robot

    canvas.set_navigation_debug_enabled(True)
    canvas.set_navigation_debug_enabled(False)

    assert canvas.config is config_before
    assert canvas.robot is robot_before


# ---------------------------------------------------------------------------
# Disabled layer: paintEvent must not even call draw_navigation_debug_overlay.
# ---------------------------------------------------------------------------


class _RaisingStub:
    def __init__(self, message: str):
        self._message = message

    def __call__(self, *args, **kwargs):
        raise AssertionError(self._message)


def test_paint_event_skips_overlay_when_disabled():
    canvas = _make_canvas()
    canvas.navigation_debug_enabled = False
    canvas.set_navigation_debug_snapshot(_make_snapshot(1))
    canvas.draw_navigation_debug_overlay = _RaisingStub("overlay must not be drawn while disabled")

    canvas.paintEvent(None)  # would raise via the stub above if reached


def test_full_paint_event_with_navigation_debug_enabled_does_not_crash():
    canvas = _make_canvas()
    canvas.navigation_debug_enabled = True
    canvas.set_navigation_debug_snapshot(_make_snapshot(1))

    canvas.paintEvent(None)  # must not raise


# ---------------------------------------------------------------------------
# Anti-pattern regression guard: the canvas must never import concrete
# planning/collision-checking algorithms -- it only ever consumes plain
# pushed data / the immutable NavigationDebugSnapshot contract.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# History stepping has exactly one control now: main_window's
# navigation_snapshot_bar, docked above the canvas (see
# _build_navigation_snapshot_bar() / test_navigation_panel_controls.py).
# The canvas itself no longer owns any `<`/`>` step buttons -- there used to
# be a second, redundant pair of real Qt child widgets here, which meant two
# independent controls could drive the same engine history state.
# ---------------------------------------------------------------------------


def test_canvas_has_no_history_step_buttons_of_its_own():
    canvas = _make_canvas()

    assert not hasattr(canvas, "navigation_debug_step_back_button")
    assert not hasattr(canvas, "navigation_debug_step_forward_button")


def test_simulation_canvas_imports_no_concrete_algorithms():
    module_path = Path(__file__).resolve().parents[1] / "app" / "simulation_canvas.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)

    forbidden = {"AStarPlanner", "DijkstraPlanner", "CollisionChecker", "compute_planned_waypoints", "RobotAgent"}
    offending = imported_names & forbidden
    assert offending == set(), f"simulation_canvas.py must not import: {offending}"
