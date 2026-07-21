"""Regressions for the circular robot vs rectangular obstacle geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robotics_sim.environment.collision_checker import (
    CollisionChecker,
    RobotSnapshot,
    distance_segment_to_rect,
    point_inside_expanded_rect,
)


@pytest.mark.parametrize(
    ("start", "goal", "obstacle", "expected_clearance"),
    [
        (
            (4.5978, 2.7077),
            (3.75, 3.25),
            (1.6306966050508134, 2.1557331677126292, 2.6086209787505634, 0.23514812740183677),
            0.46005654,
        ),
        (
            (-6.9420, -3.2710),
            (-7.75, -2.75),
            (-7.5063056713058725, -5.7511226161741416, 0.2048655218914046, 2.1751201784669925),
            0.45111999,
        ),
    ],
)
def test_office_corner_routes_do_not_trigger_square_cap_false_collision(
    start,
    goal,
    obstacle,
    expected_clearance,
):
    """The exact R2/R3 routes remain outside their 0.35 m circular envelope."""
    checker = CollisionChecker()
    safety_radius = 0.35

    assert distance_segment_to_rect(start, goal, obstacle) == pytest.approx(expected_clearance, abs=1e-7)
    assert not checker.check_position(start, [obstacle], safety_radius).collision
    assert not checker.check_segment(start, goal, [obstacle], safety_radius).collision

    heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    snapshot = RobotSnapshot(
        x=start[0],
        y=start[1],
        theta=heading,
        v=0.4,
        max_speed=1.2,
        max_acceleration=2.0,
        max_angular_speed=2.5,
    )
    report = checker.check_predicted_motion(
        snapshot,
        np.array([[0.0], [0.0]]),
        dt=0.05,
        steps=10,
        obstacles=[obstacle],
        robot_radius=safety_radius,
    )
    assert not report.collision


def test_rounded_expansion_preserves_real_corner_and_edge_collisions():
    checker = CollisionChecker()
    obstacle = (0.0, 0.0, 1.0, 1.0)
    safety_radius = 0.35

    # 0.20 m in both axes from the corner is only 0.283 m away: collision.
    assert point_inside_expanded_rect((1.2, 1.2), obstacle, safety_radius)
    # 0.30 m in both axes is inside the old square cap but 0.424 m from the
    # real corner, so a circular robot is clear.
    assert not point_inside_expanded_rect((1.3, 1.3), obstacle, safety_radius)
    # Straight-edge clearance remains conservative and unchanged.
    assert checker.check_position((1.3, 0.5), [obstacle], safety_radius).collision
    # A segment that genuinely crosses the rounded safety envelope is blocked.
    assert checker.check_segment((1.3, 1.3), (0.9, 0.9), [obstacle], safety_radius).collision

