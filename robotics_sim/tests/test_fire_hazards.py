"""
Tests for fire hazard placement through the engine's public entry points --
add_fire()/remove_fire_near()/on_fire_toggle_requested() (SimulationController
Mixin) -- against the current layered architecture:

    BeliefMap.grid        logical UNKNOWN/FREE/OCCUPIED occupancy
    HazardField            a separate continuous thermal layer, rasterized
                            from FireSource objects, that never writes to
                            BeliefMap.grid (see HazardField's module
                            docstring)

The previous version of this file tested an obsolete contract where adding a
fire wrote OCCUPIED cells into the belief map via mapped_obstacle_points /
_fire_obstacle_cluster_points -- that mechanism no longer exists. Do not
reintroduce it: hazards are a rendering/planning-time overlay, not sensed
occupancy.

Pure-HazardField mechanics (sources()/values(), independent removal,
restore_sources()) live in test_hazard_field.py. Hazard-vs-planning-grid
projection lives in test_planning_costmap_hazard.py. Hazard state surviving
navigation-debug snapshot restore lives in test_navigation_snapshot_restore.py.
This file covers only the engine-level add/remove/toggle entry points and
their side effects (or deliberate lack thereof) on occupancy, the canvas
push, and frontier detection.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.environment.belief_map import BeliefMap, FREE, OCCUPIED, UNKNOWN
from robotics_sim.planning.exploration_planners import _frontier_cells
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.navigation_modes import is_goal_seeking_planner

_BOUNDS = (-10.0, 10.0, -10.0, 10.0)
_RESOLUTION = 0.5


def _build_fake_engine(*, running: bool = False) -> SimpleNamespace:
    belief = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1)
    fake = SimpleNamespace(
        config=SimpleNamespace(
            grid_resolution=_RESOLUTION,
            agent_mode="Single Robot Mode",
            planner_type="Direct",
        ),
        belief_map=belief,
        running=running,
        robots=[],
        robot=None,
        status_messages=[],
        hazard_snapshots=[],
    )
    fake.canvas = SimpleNamespace(
        set_status=lambda msg: fake.status_messages.append(msg),
        set_hazard_snapshot=lambda snapshot: fake.hazard_snapshots.append(snapshot),
    )
    for name in (
        "ensure_belief_map",
        "ensure_hazard_service",
        "push_hazard_snapshot",
        "add_fire",
        "remove_fire_near",
        "on_fire_toggle_requested",
        "_replan_routes_affected_by_hazard",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


# ---------------------------------------------------------------------------
# 1-2. Occupancy is never touched by add/remove -- hazards are a separate
# layer (see HazardField's module docstring).
# ---------------------------------------------------------------------------


def test_add_fire_does_not_modify_belief_map_grid():
    fake = _build_fake_engine()
    grid_before = fake.belief_map.grid.copy()

    ok = fake.add_fire(2.0, 3.0)

    assert ok is True
    assert (fake.belief_map.grid == grid_before).all()


def test_remove_fire_does_not_modify_belief_map_grid():
    fake = _build_fake_engine()
    fake.add_fire(2.0, 3.0)
    grid_before = fake.belief_map.grid.copy()

    removed = fake.remove_fire_near(2.0, 3.0)

    assert removed is True
    assert (fake.belief_map.grid == grid_before).all()


# ---------------------------------------------------------------------------
# 3-4. A cell's occupancy state (whatever it already was) is unaffected by
# a fire existing on top of it, before or after removal.
# ---------------------------------------------------------------------------


def test_unknown_cell_stays_unknown_after_fire_add_and_remove():
    fake = _build_fake_engine()
    point = (2.0, 3.0)
    assert fake.belief_map.cell_state(point) == UNKNOWN

    fake.add_fire(*point)
    assert fake.belief_map.cell_state(point) == UNKNOWN

    fake.remove_fire_near(*point)
    assert fake.belief_map.cell_state(point) == UNKNOWN


def test_real_occupied_cell_stays_occupied_after_fire_add_and_remove():
    fake = _build_fake_engine()
    point = (2.0, 3.0)
    row, col = fake.belief_map.world_to_cell(point)
    fake.belief_map.grid[row, col] = OCCUPIED
    assert fake.belief_map.cell_state(point) == OCCUPIED

    fake.add_fire(*point)
    assert fake.belief_map.cell_state(point) == OCCUPIED

    fake.remove_fire_near(*point)
    assert fake.belief_map.cell_state(point) == OCCUPIED


# ---------------------------------------------------------------------------
# 9. Two fires are independently removable through the engine entry point
# (see test_hazard_field.py for the equivalent pure-HazardField case).
# ---------------------------------------------------------------------------


def test_two_fires_are_independently_removable_through_the_engine():
    fake = _build_fake_engine()
    fake.add_fire(1.0, 1.0)
    fake.add_fire(5.0, 5.0)
    service = fake.ensure_hazard_service()
    assert len(service.field.sources()) == 2

    assert fake.remove_fire_near(1.0, 1.0) is True
    remaining = [s.position for s in service.field.sources()]
    assert remaining == [(5.0, 5.0)]

    assert fake.remove_fire_near(5.0, 5.0) is True
    assert service.field.sources() == ()


def test_toggle_adds_then_removes_the_same_spot():
    fake = _build_fake_engine()
    service = fake.ensure_hazard_service()

    fake.on_fire_toggle_requested(1.0, 1.0)
    assert len(service.field.sources()) == 1

    fake.on_fire_toggle_requested(1.0, 1.0)
    assert service.field.sources() == ()


# ---------------------------------------------------------------------------
# 10. The canvas receives an updated thermal snapshot on every change.
# ---------------------------------------------------------------------------


def test_canvas_receives_the_updated_hazard_snapshot_on_add_and_remove():
    fake = _build_fake_engine()

    fake.add_fire(2.0, 3.0)
    assert fake.hazard_snapshots, "push_hazard_snapshot() must run after add_fire()"
    latest = fake.hazard_snapshots[-1]
    assert len(latest["sources"]) == 1
    assert latest["sources"][0].position == (2.0, 3.0)
    version_after_add = latest["version"]

    fake.remove_fire_near(2.0, 3.0)
    latest = fake.hazard_snapshots[-1]
    assert latest["sources"] == ()
    assert latest["version"] > version_after_add


# ---------------------------------------------------------------------------
# 12. Frontier detection reads only BeliefMap -- a hazard change alone must
# never change its result. _frontier_cells() is the real production
# function (robotics_sim.planning.exploration_planners), imported read-only
# here, not reimplemented.
# ---------------------------------------------------------------------------


def test_frontier_detection_is_unaffected_by_a_hazard_change():
    fake = _build_fake_engine()
    belief = fake.belief_map
    # Build one FREE cell next to UNKNOWN neighbors -- a frontier cell.
    free_point = (0.0, 0.0)
    row, col = belief.world_to_cell(free_point)
    belief.grid[row, col] = FREE

    frontiers_before = _frontier_cells(belief)
    assert (row, col) in frontiers_before

    fake.add_fire(*free_point)
    frontiers_with_fire = _frontier_cells(belief)
    assert frontiers_with_fire == frontiers_before

    fake.remove_fire_near(*free_point)
    frontiers_after_remove = _frontier_cells(belief)
    assert frontiers_after_remove == frontiers_before


# ---------------------------------------------------------------------------
# Click routing: exploration mode -> fire, goal seeking -> goal (pure logic
# the canvas branches on -- see simulation_canvas.py's mousePressEvent).
# Unrelated to the occupancy-vs-hazard contract, kept as-is.
# ---------------------------------------------------------------------------


def test_click_routes_to_goal_in_goal_seeking_mode():
    assert is_goal_seeking_planner("Goal seeking") is True


def test_click_routes_to_fire_in_exploration_mode():
    assert is_goal_seeking_planner("FoV-aware directional frontier") is False
