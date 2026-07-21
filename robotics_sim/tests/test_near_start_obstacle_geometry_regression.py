"""Regression coverage for near-start obstacle geometry consistency."""

from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.occupancy_grid import OCCUPIED
from robotics_sim.planning.grid_planners import AStarPlanner
from robotics_sim.simulation.engine import SimulationControllerMixin


def _engine_at(x: float, y: float) -> SimpleNamespace:
    resolution = 0.5
    radius = 0.35
    belief = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=resolution, robot_count=1)
    robot = SimpleNamespace(x=x, y=y, theta=0.0)
    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        belief_map=belief,
        mapped_obstacle_points=[],
        hazard_service=None,
        config=SimpleNamespace(
            grid_resolution=resolution,
            body_radius=radius,
            safety_radius=radius,
        ),
    )
    fake.ensure_belief_map = lambda: belief
    for name in (
        "safety_radius_for_robot",
        "sanitize_planner_obstacle_points",
        "build_planning_grid_for_robot",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


def test_office_corner_inside_safety_envelope_is_not_erased_for_astar():
    """The reproduced Office pose must not acquire a synthetic free bubble."""
    start = (5.74, -4.14)
    obstacle_corner = (5.8919, -3.9818)  # about 0.219 m away; safety radius is 0.35 m
    fake = _engine_at(*start)

    points, removed = fake.sanitize_planner_obstacle_points(
        [obstacle_corner],
        start_xy=start,
        robot_radius=0.35,
        resolution=0.5,
    )

    assert points == [obstacle_corner]
    assert removed == 0

    grid = fake.build_planning_grid_for_robot(
        fake.robot,
        obstacle_points=points,
        robot_radius=0.35,
    )
    assert grid.get_value(grid.world_to_grid(*start)) == OCCUPIED

    result = AStarPlanner(allow_unknown=True).plan(grid, start, (7.0, -5.0))
    assert result.success is False
    assert result.reason == "start cell is not traversable"


def test_sanitizer_preserves_near_and_far_finite_geometry_but_drops_nonfinite_samples():
    fake = _engine_at(0.0, 0.0)
    points, removed = fake.sanitize_planner_obstacle_points(
        [(0.02, 0.01), (2.0, 0.0), (float("nan"), 1.0)],
        start_xy=(0.0, 0.0),
        robot_radius=0.35,
        resolution=0.5,
    )

    assert points == [(0.02, 0.01), (2.0, 0.0)]
    assert removed == 0
