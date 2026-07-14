"""
Tests for fire hazard placement: add/remove via engine methods (the canvas
click-routing itself is a thin Qt signal, exercised only implicitly here),
and the belief-map/mapped_obstacle_points effect.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.navigation_modes import is_goal_seeking_planner


def _build_fake_engine() -> SimpleNamespace:
    belief = BeliefMap(bounds=(-10.0, 10.0, -10.0, 10.0), resolution=0.5, robot_count=1)
    fake = SimpleNamespace(
        config=SimpleNamespace(grid_resolution=0.5),
        mapped_obstacle_points=[],
        mapped_obstacle_point_keys=set(),
        fires=[],
        fire_points_by_fire=[],
        simulation_time=1.0,
        belief_map=belief,
        status_messages=[],
    )
    fake.ensure_belief_map = lambda: belief
    fake.canvas = SimpleNamespace(
        set_fires=lambda fires: fake.status_messages.append(("fires", list(fires))),
        set_mapped_obstacle_points=lambda pts: None,
        invalidate_mapped_points_cache=lambda: None,
        set_status=lambda msg: fake.status_messages.append(("status", msg)),
    )
    for name in (
        "_fire_obstacle_cluster_points",
        "add_fire",
        "remove_fire_near",
        "on_fire_toggle_requested",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    fake.FIRE_HIT_RADIUS_M = SimulationControllerMixin.FIRE_HIT_RADIUS_M
    return fake


def test_add_fire_marks_belief_map_occupied_and_records_center():
    fake = _build_fake_engine()

    fake.add_fire(2.0, 3.0)

    assert fake.fires == [(2.0, 3.0)]
    assert len(fake.mapped_obstacle_points) > 0
    # The belief cell at the fire's own center must now read OCCUPIED --
    # the same mechanism a real sensed obstacle uses.
    assert fake.belief_map.cell_state((2.0, 3.0)) == fake.belief_map.grid[
        fake.belief_map.world_to_cell((2.0, 3.0))
    ]


def test_remove_fire_near_clears_points_and_frees_belief_cells():
    fake = _build_fake_engine()
    fake.add_fire(2.0, 3.0)
    assert fake.mapped_obstacle_points

    removed = fake.remove_fire_near(2.05, 2.97)  # close enough to hit it

    assert removed is True
    assert fake.fires == []
    assert fake.mapped_obstacle_points == []
    assert fake.mapped_obstacle_point_keys == set()


def test_remove_fire_near_returns_false_when_nothing_within_hit_radius():
    fake = _build_fake_engine()
    fake.add_fire(2.0, 3.0)

    removed = fake.remove_fire_near(8.0, 8.0)

    assert removed is False
    assert len(fake.fires) == 1


def test_toggle_adds_then_removes_same_spot():
    fake = _build_fake_engine()

    fake.on_fire_toggle_requested(1.0, 1.0)
    assert len(fake.fires) == 1

    fake.on_fire_toggle_requested(1.0, 1.0)
    assert fake.fires == []


def test_two_fires_are_independently_removable():
    fake = _build_fake_engine()
    fake.add_fire(1.0, 1.0)
    fake.add_fire(5.0, 5.0)

    assert fake.remove_fire_near(1.0, 1.0) is True
    assert fake.fires == [(5.0, 5.0)]
    assert fake.remove_fire_near(5.0, 5.0) is True
    assert fake.fires == []


# ---------------------------------------------------------------------------
# Click routing: exploration mode -> fire, goal seeking -> goal (pure logic
# the canvas branches on -- see simulation_canvas.py's mousePressEvent).
# ---------------------------------------------------------------------------


def test_click_routes_to_goal_in_goal_seeking_mode():
    assert is_goal_seeking_planner("Goal seeking") is True


def test_click_routes_to_fire_in_exploration_mode():
    assert is_goal_seeking_planner("FoV-aware directional frontier") is False
