"""Regression tests for multi-robot planned-route rendering."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas


_app = QApplication.instance() or QApplication([])


def _robot(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(
        x=x,
        y=y,
        theta=0.0,
        _sim_body_radius=0.20,
        waypoints=SimpleNamespace(current_index=0),
    )


def _alpha_at_world(canvas: SimulationCanvas, pixmap: QPixmap, point: tuple[float, float]) -> int:
    sx, sy = canvas.world_to_screen(*point)
    return pixmap.toImage().pixelColor(int(sx), int(sy)).alpha()


def test_second_multi_robot_route_does_not_fill_area_enclosed_by_polyline():
    """A previous F marker's purple brush must not fill the next route.

    Qt implicitly closes an open QPainterPath for filling.  The first route's
    endpoint marker leaves a purple brush on the painter, so the second bent
    route used to render as a solid triangular wedge from robot to frontier.
    """
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.robots = [_robot(-8.0, 4.0), _robot(-4.0, -3.0)]
    canvas.multi_planned_path_points = [
        [(-8.0, 4.0), (-7.0, 4.0), (-7.0, 3.0)],
        [(-4.0, -3.0), (4.0, -3.0), (4.0, 3.0)],
    ]

    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    canvas.draw_multi_planned_routes(painter)
    painter.end()

    # Strictly inside the triangle that Qt creates by closing the second
    # route, but far from its route strokes and waypoint markers.
    assert _alpha_at_world(canvas, pixmap, (2.0, -1.0)) == 0
    # The route itself is still present: only the accidental fill is removed.
    assert _alpha_at_world(canvas, pixmap, (0.0, -3.0)) > 0


def test_multi_frontier_marker_never_exceeds_owning_robot_body():
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.robots = [_robot(0.0, 0.0)]

    body_px = canvas._multi_robot_body_radius_px(0)
    frontier_px = canvas._multi_frontier_marker_radius_px(0)

    assert 5.0 <= frontier_px <= 7.0
    assert frontier_px <= body_px


def test_unrouted_multi_frontier_uses_compact_marker_radius(monkeypatch):
    """The fallback F marker must use the same sizing policy as endpoints."""
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.config.agent_mode = "Multiple Robot Mode"
    canvas.robots = [_robot(-5.0, -4.0)]
    canvas.multi_exploration_targets = [(0.0, 0.0)]
    canvas.multi_planned_path_points = [[]]
    monkeypatch.setattr(canvas, "_multi_frontier_marker_radius_px", lambda _index: 4.0)

    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    canvas.draw_goal_and_robot(painter)
    painter.end()

    assert _alpha_at_world(canvas, pixmap, (0.0, 0.0)) > 0
    # One screen pixel outside the forced 4 px radius (plus its 1 px pen)
    # stays transparent, proving the old hard-coded 11 px disk is gone.
    sx, sy = canvas.world_to_screen(0.0, 0.0)
    assert pixmap.toImage().pixelColor(int(sx + 6), int(sy)).alpha() == 0
