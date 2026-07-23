"""Regression coverage for route and custom-discovery presentation controls."""

from __future__ import annotations

from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import QApplication

from robotics_sim.app.simulation_canvas import SimulationCanvas
from robotics_sim.simulation.config import (
    SimulationConfig,
    config_from_sim_payload,
    config_to_sim_payload,
)


_app = QApplication.instance() or QApplication([])


def _transparent_canvas_pixmap(canvas: SimulationCanvas) -> QPixmap:
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.transparent)
    return pixmap


def _alpha_at_world(canvas: SimulationCanvas, pixmap: QPixmap, x: float, y: float) -> int:
    sx, sy = canvas.world_to_screen(x, y)
    return pixmap.toImage().pixelColor(round(sx), round(sy)).alpha()


def test_visual_style_fields_round_trip_through_sim_payload():
    config = SimulationConfig(
        map_visualization="Custom Discovery",
        custom_unexplored_color="#112233",
        custom_explored_color="#445566",
        custom_obstacle_color="#778899",
        custom_explored_opacity=0.35,
        mapped_obstacle_line_width=3.25,
        show_path=False,
        show_traveled_path=True,
    )

    restored = config_from_sim_payload(config_to_sim_payload(config))

    assert restored.custom_obstacle_color == "#778899"
    assert restored.custom_explored_opacity == 0.35
    assert restored.mapped_obstacle_line_width == 3.25
    assert restored.show_path is False
    assert restored.show_traveled_path is True


def test_visual_numeric_fields_are_clamped_when_loading_payload():
    payload = config_to_sim_payload(SimulationConfig())
    payload["simulation"]["custom_explored_opacity"] = 8.0
    payload["simulation"]["mapped_obstacle_line_width"] = -4.0

    restored = config_from_sim_payload(payload)

    assert restored.custom_explored_opacity == 1.0
    assert restored.mapped_obstacle_line_width == 0.25


def test_traveled_route_toggle_controls_executed_line():
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.path_points = [(-2.0, 0.0), (2.0, 0.0)]

    hidden = _transparent_canvas_pixmap(canvas)
    painter = QPainter(hidden)
    canvas.config.show_traveled_path = False
    canvas.draw_executed_path(painter)
    painter.end()
    assert _alpha_at_world(canvas, hidden, 0.0, 0.0) == 0

    visible = _transparent_canvas_pixmap(canvas)
    painter = QPainter(visible)
    canvas.config.show_traveled_path = True
    canvas.draw_executed_path(painter)
    painter.end()
    assert _alpha_at_world(canvas, visible, 0.0, 0.0) > 0


def test_planned_route_toggle_also_controls_frontier_marker():
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.exploration_target_xy = (3.0, 2.0)

    hidden = _transparent_canvas_pixmap(canvas)
    painter = QPainter(hidden)
    canvas.config.show_path = False
    canvas.draw_goal_and_robot(painter)
    painter.end()
    assert _alpha_at_world(canvas, hidden, 3.0, 2.0) == 0

    visible = _transparent_canvas_pixmap(canvas)
    painter = QPainter(visible)
    canvas.config.show_path = True
    canvas.draw_goal_and_robot(painter)
    painter.end()
    assert _alpha_at_world(canvas, visible, 3.0, 2.0) > 0


def test_custom_exploration_opacity_controls_explored_alpha():
    canvas = SimulationCanvas()
    canvas.config.map_visualization = "Custom Discovery"
    canvas.config.custom_explored_opacity = 0.40

    assert canvas._explored_area_alpha(24) == 102


def _mapped_point_nontransparent_pixels(width: float) -> tuple[int, object]:
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.config.map_visualization = "Custom Discovery"
    canvas.config.custom_obstacle_color = "#00FF00"
    canvas.config.mapped_obstacle_line_width = width
    canvas.mapped_obstacle_points = [(0.0, 0.0)]
    canvas.rebuild_mapped_points_cache()
    image = canvas._mapped_points_cache.toImage()
    sx, sy = canvas.world_to_screen(0.0, 0.0)
    center = image.pixelColor(round(sx), round(sy))
    count = 0
    for px in range(round(sx) - 8, round(sx) + 9):
        for py in range(round(sy) - 8, round(sy) + 9):
            if image.pixelColor(px, py).alpha() > 0:
                count += 1
    return count, center


def test_mapped_width_preserves_the_dedicated_mapping_color():
    thin_count, thin_center = _mapped_point_nontransparent_pixels(0.5)
    thick_count, thick_center = _mapped_point_nontransparent_pixels(5.0)

    assert thin_center.red() > thin_center.green()
    assert thick_center.red() > thick_center.green()
    assert thick_count > thin_count


def test_custom_obstacle_color_is_used_by_ground_truth_obstacle_layer():
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.config = SimulationConfig(
        map_visualization="Custom Discovery",
        custom_obstacle_color="#00FF00",
        obstacles=[(-1.0, 1.0, 2.0, 2.0)],
    )

    pixmap = _transparent_canvas_pixmap(canvas)
    painter = QPainter(pixmap)
    canvas.draw_ground_truth_obstacles(painter)
    painter.end()

    sx, sy = canvas.world_to_screen(0.0, 2.0)
    center = pixmap.toImage().pixelColor(round(sx), round(sy))
    assert center.alpha() > 0
    assert center.green() > center.red()


def test_custom_discovery_opacity_is_uniform_across_robot_overlap():
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.config.map_visualization = "Custom Discovery"
    canvas.config.custom_explored_opacity = 0.40
    canvas.append_explored_area_polygon(
        [(-3.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-3.0, 1.0)],
        robot_index=0,
    )
    canvas.append_explored_area_polygon(
        [(-1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (-1.0, 1.0)],
        robot_index=1,
    )

    cache = canvas._explored_area_cache
    assert cache is not None
    assert canvas._explored_area_caches_by_robot == {}
    alphas = []
    for x in (-2.0, 0.0, 2.0):
        sx, sy = canvas.world_to_screen(x, 0.0)
        alphas.append(cache.toImage().pixelColor(round(sx), round(sy)).alpha())

    assert alphas == [102, 102, 102]

    canvas.invalidate_explored_area_cache()
    canvas.rebuild_explored_area_cache()
    rebuilt = canvas._explored_area_cache.toImage()
    rebuilt_alphas = []
    for x in (-2.0, 0.0, 2.0):
        sx, sy = canvas.world_to_screen(x, 0.0)
        rebuilt_alphas.append(rebuilt.pixelColor(round(sx), round(sy)).alpha())
    assert rebuilt_alphas == [102, 102, 102]


def test_custom_discovery_fov_keeps_outline_without_extra_fill():
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.config.map_visualization = "Custom Discovery"
    polygon = [(-1.0, -1.0), (2.0, 0.0), (-1.0, 1.0)]

    pixmap = _transparent_canvas_pixmap(canvas)
    painter = QPainter(pixmap)
    canvas.draw_sensor_polygon(painter, polygon, canvas.sensor_display_color(0))
    painter.end()

    assert _alpha_at_world(canvas, pixmap, 0.0, 0.0) == 0
    assert _alpha_at_world(canvas, pixmap, -1.0, 0.0) > 0


def test_multi_robot_traveled_routes_use_all_recorded_histories():
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.config.agent_mode = "Multiple Robot Mode"
    canvas.config.show_traveled_path = True
    canvas.robots = [SimpleNamespace(x=1.0, y=0.0), SimpleNamespace(x=1.0, y=2.0)]
    canvas.multi_path_points = [
        [(-2.0, 0.0), (2.0, 0.0)],
        [(-2.0, 2.0), (2.0, 2.0)],
    ]

    pixmap = _transparent_canvas_pixmap(canvas)
    painter = QPainter(pixmap)
    canvas.draw_multi_executed_paths(painter)
    painter.end()

    assert _alpha_at_world(canvas, pixmap, 0.0, 0.0) > 0
    assert _alpha_at_world(canvas, pixmap, 0.0, 2.0) > 0


def test_multi_robot_traveled_route_cache_paints_only_appended_segments():
    canvas = SimulationCanvas()
    canvas.resize(900, 700)
    canvas.config.agent_mode = "Multiple Robot Mode"
    canvas.config.show_traveled_path = True
    paths = [
        [(-2.0, 0.0), (-1.0, 0.0)],
        [(-2.0, 2.0), (-1.0, 2.0)],
    ]
    canvas.set_multi_runtime_state(path_points=paths)

    first_frame = _transparent_canvas_pixmap(canvas)
    painter = QPainter(first_frame)
    canvas.draw_multi_executed_paths(painter)
    painter.end()
    first_cache = canvas._multi_executed_trail_pixmap

    paths[0].append((0.0, 0.0))
    paths[1].append((0.0, 2.0))
    canvas.set_multi_runtime_state(path_points=paths)
    second_frame = _transparent_canvas_pixmap(canvas)
    painter = QPainter(second_frame)
    canvas.draw_multi_executed_paths(painter)
    painter.end()

    assert canvas._multi_executed_trail_pixmap is first_cache
    assert canvas._route_detail["executed_trail_segments_painted"] == 2
    assert canvas._route_detail["executed_trail_cache_hit"] is True
    assert _alpha_at_world(canvas, second_frame, -1.5, 0.0) > 0
    assert _alpha_at_world(canvas, second_frame, -1.5, 2.0) > 0


def test_multi_runtime_state_preserves_inner_path_identity_for_incremental_cache():
    canvas = SimulationCanvas()
    paths = [[(0.0, 0.0)], [(1.0, 1.0)]]

    canvas.set_multi_runtime_state(path_points=paths)

    assert canvas.multi_path_points is not paths
    assert canvas.multi_path_points[0] is paths[0]
    assert canvas.multi_path_points[1] is paths[1]
