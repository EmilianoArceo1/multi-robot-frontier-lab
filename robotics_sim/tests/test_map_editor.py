from __future__ import annotations

from robotics_sim.app.map_editor import (
    create_free_draw_obstacles_from_path,
    create_rect_obstacle_from_drag,
    create_square_obstacle_from_drag,
    find_obstacle_at,
    merge_obstacles,
    remove_obstacle_at,
)
from robotics_sim.simulation.config import SimulationConfig


def test_drag_from_bottom_left_to_top_right_creates_valid_rect_obstacle():
    obstacle = create_rect_obstacle_from_drag((1.0, 2.0), (4.0, 5.0))

    assert obstacle == (1.0, 2.0, 3.0, 3.0)


def test_drag_from_top_right_to_bottom_left_creates_same_normalized_obstacle():
    obstacle = create_rect_obstacle_from_drag((4.0, 5.0), (1.0, 2.0))

    assert obstacle == (1.0, 2.0, 3.0, 3.0)


def test_tiny_drag_is_ignored():
    obstacle = create_rect_obstacle_from_drag((0.0, 0.0), (0.08, 0.02), min_size=0.2)

    assert obstacle is None


def test_find_obstacle_at_returns_index_for_point_inside_obstacle():
    obstacles = [(0.0, 0.0, 2.0, 2.0)]

    assert find_obstacle_at(obstacles, (1.0, 1.0)) == 0


def test_clicking_outside_obstacles_does_not_remove_anything():
    obstacles = [(0.0, 0.0, 2.0, 2.0)]

    removed = remove_obstacle_at(obstacles, (3.0, 3.0))

    assert removed is False
    assert obstacles == [(0.0, 0.0, 2.0, 2.0)]


def test_removing_an_obstacle_preserves_order_of_the_remaining_obstacles():
    obstacles = [(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 1.0, 1.0), (4.0, 4.0, 1.0, 1.0)]

    removed = remove_obstacle_at(obstacles, (2.2, 2.2))

    assert removed is True
    assert obstacles == [(0.0, 0.0, 1.0, 1.0), (4.0, 4.0, 1.0, 1.0)]


def test_map_editor_uses_existing_obstacle_data_structure_from_config():
    config = SimulationConfig()
    obstacle = create_rect_obstacle_from_drag((1.0, 1.0), (3.0, 4.0))
    config.obstacles.append(obstacle)

    assert isinstance(obstacle, tuple)
    assert len(obstacle) == 4
    assert config.obstacles[-1] == obstacle
    assert isinstance(config.obstacles, list)


def test_free_draw_path_creates_single_stroked_obstacle():
    obstacles = create_free_draw_obstacles_from_path([(0.0, 0.0), (0.4, 0.1), (0.8, 0.2)], brush_size=0.2)

    assert len(obstacles) == 1
    left, bottom, width, height = obstacles[0]
    assert left == -0.1
    assert bottom == -0.1
    assert width == 1.0
    assert height == 0.4


def test_square_tool_creates_square_obstacle_from_drag():
    obstacle = create_square_obstacle_from_drag((1.0, 2.0), (4.5, 5.0))

    assert obstacle == (1.0, 2.0, 3.5, 3.5)


def test_overlapping_obstacles_are_merged_into_one_rectangle():
    merged = merge_obstacles([(0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 2.0, 2.0)])

    assert merged == [(0.0, 0.0, 3.0, 3.0)]
