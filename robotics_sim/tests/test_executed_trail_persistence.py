"""Regression tests for complete executed-trajectory retention.

The visual trail used to be a sliding window: single-robot history was
trimmed after 1,400 samples and each multi-robot history after 900.  The
canvas now renders new segments into persistent pixmaps, so the engine can
retain the complete world-space trajectory until an explicit restart.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from robotics_sim.simulation.engine import SimulationControllerMixin


def _make_single_fake() -> SimpleNamespace:
    fake = SimpleNamespace(path_points=[], total_distance_traveled=0.0)
    fake._append_executed_path_point = (
        SimulationControllerMixin._append_executed_path_point.__get__(fake)
    )
    return fake


def _make_multi_fake(robot_count: int = 2) -> SimpleNamespace:
    fake = SimpleNamespace(
        multi_path_points=[[] for _ in range(robot_count)],
        total_distance_traveled=0.0,
    )
    fake._append_multi_executed_path_point = (
        SimulationControllerMixin._append_multi_executed_path_point.__get__(fake)
    )
    return fake


def test_single_trajectory_persists_beyond_previous_trim_threshold():
    fake = _make_single_fake()
    original_path = fake.path_points
    sample_count = 2_500

    for index in range(sample_count):
        fake._append_executed_path_point((float(index), 0.0))

    assert fake.path_points is original_path
    assert len(fake.path_points) == sample_count
    assert fake.path_points[0] == (0.0, 0.0)
    assert fake.path_points[-1] == (float(sample_count - 1), 0.0)
    assert math.isclose(fake.total_distance_traveled, sample_count - 1)


def test_each_multi_robot_trajectory_persists_beyond_previous_900_point_cap():
    fake = _make_multi_fake()
    original_paths = tuple(fake.multi_path_points)
    sample_count = 1_500

    for index in range(sample_count):
        fake._append_multi_executed_path_point(0, (float(index), 0.0))
        fake._append_multi_executed_path_point(1, (0.0, float(index)))

    assert fake.multi_path_points[0] is original_paths[0]
    assert fake.multi_path_points[1] is original_paths[1]
    assert [len(path) for path in fake.multi_path_points] == [sample_count, sample_count]
    assert fake.multi_path_points[0][0] == (0.0, 0.0)
    assert fake.multi_path_points[1][0] == (0.0, 0.0)
    assert fake.multi_path_points[0][-1] == (float(sample_count - 1), 0.0)
    assert fake.multi_path_points[1][-1] == (0.0, float(sample_count - 1))
    assert math.isclose(fake.total_distance_traveled, 2 * (sample_count - 1))


def test_multi_append_initializes_a_missing_robot_history_without_replacing_others():
    fake = _make_multi_fake(robot_count=1)
    first_robot_path = fake.multi_path_points[0]

    fake._append_multi_executed_path_point(2, (3.0, 4.0))

    assert fake.multi_path_points[0] is first_robot_path
    assert fake.multi_path_points[1] == []
    assert fake.multi_path_points[2] == [(3.0, 4.0)]
