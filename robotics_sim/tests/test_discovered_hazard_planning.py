"""
Phase 3: planning costmap and route invalidation must use only the team's
DISCOVERED HazardBelief -- never the omniscient ground-truth HazardField.

    - apply_hazard_belief_to_planning_grid() (planning_costmap.py) blocks a
      cell only when observed=True and value >= threshold. The legacy
      apply_hazard_to_planning_grid() (ground truth) is kept intact for its
      own contract/tests -- see test_planning_costmap_hazard.py -- and is no
      longer called by the runtime planner.
    - RuntimeHazardService.observed_blocked_world_points() is the discovered
      counterpart to blocked_world_points() (ground truth, also kept intact).
    - Creating/removing a FireSource never triggers replanning by itself
      anymore; route repair is gated on HazardObservationResult.
      newly_blocked_cells > 0 from an actual sensor observation.
"""
from __future__ import annotations

from types import SimpleNamespace

from robot import Robot
from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.environment.belief_map import BeliefMap, FREE, OCCUPIED
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.environment.hazard_field import HazardField
from robotics_sim.environment.occupancy_grid import OCCUPIED as OG_OCCUPIED, UNKNOWN as OG_UNKNOWN, OccupancyGrid
from robotics_sim.planning.exploration_planners import _frontier_cells
from robotics_sim.planning.planning_costmap import (
    apply_hazard_belief_to_planning_grid,
    apply_hazard_to_planning_grid,
)
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.hazard_service import RuntimeHazardService
from robotics_sim.simulation.telemetry import TelemetryLogger

_BOUNDS = (0.0, 10.0, 0.0, 10.0)
_RESOLUTION = 1.0  # -> 10x10 grid, cell centers at 0.5, 1.5, ..., 9.5


def _make_belief(robot_count: int = 1) -> HazardBelief:
    return HazardBelief(GridGeometry(_BOUNDS, _RESOLUTION), robot_count=robot_count)


def _make_planning_grid() -> OccupancyGrid:
    return OccupancyGrid.from_bounds(*_BOUNDS, _RESOLUTION, initial_value=OG_UNKNOWN)


def _square_polygon(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ---------------------------------------------------------------------------
# COSTMAP 1-6
# ---------------------------------------------------------------------------


def test_costmap_ignores_unobserved_hot_ground_truth():
    """1. HazardField caliente pero HazardBelief no observado: transitable."""
    field = HazardField(bounds=_BOUNDS, resolution=_RESOLUTION)
    field.add_fire((5.5, 5.5), intensity=1.0, radius=2.0)
    belief = _make_belief()  # never observed
    planning_grid = _make_planning_grid()

    apply_hazard_belief_to_planning_grid(planning_grid, belief, block_threshold=0.5)

    cell = planning_grid.world_to_grid(5.5, 5.5)
    assert planning_grid.get_value(cell) == OG_UNKNOWN
    assert field.values(copy=False)[5, 5] > 0.0  # sanity: ground truth really is hot there


def test_costmap_leaves_observed_safe_cell_traversable():
    """2. Celda observada segura: sigue transitable."""
    belief = _make_belief()
    belief.observe_cells([5], [5], [0.0], robot_index=0)
    planning_grid = _make_planning_grid()

    apply_hazard_belief_to_planning_grid(planning_grid, belief, block_threshold=0.5)

    cell = planning_grid.world_to_grid(5.5, 5.5)
    assert planning_grid.get_value(cell) == OG_UNKNOWN


def test_costmap_blocks_observed_cell_at_or_above_threshold():
    """3. Celda observada caliente sobre threshold: queda bloqueada."""
    belief = _make_belief()
    belief.observe_cells([5], [5], [0.6], robot_index=0)
    planning_grid = _make_planning_grid()

    apply_hazard_belief_to_planning_grid(planning_grid, belief, block_threshold=0.6)

    cell = planning_grid.world_to_grid(5.5, 5.5)
    assert planning_grid.get_value(cell) == OG_OCCUPIED


def test_costmap_leaves_observed_cell_below_threshold_traversable():
    """4. Celda observada caliente debajo del threshold: sigue transitable."""
    belief = _make_belief()
    belief.observe_cells([5], [5], [0.4], robot_index=0)
    planning_grid = _make_planning_grid()

    apply_hazard_belief_to_planning_grid(planning_grid, belief, block_threshold=0.6)

    cell = planning_grid.world_to_grid(5.5, 5.5)
    assert planning_grid.get_value(cell) == OG_UNKNOWN


def test_high_value_unobserved_cell_never_blocks_even_next_to_an_observed_hot_one():
    """5. Celda con valor alto pero observed=False: nunca se bloquea, incluso
    junto a una celda realmente observada y bloqueada."""
    belief = _make_belief()
    belief.observe_cells([5], [5], [1.0], robot_index=0)  # really observed, really hot
    planning_grid = _make_planning_grid()
    # A distinct cell that ground truth would call hot, but the belief never
    # observed -- there is no way to feed a "high value" into the belief
    # without observe_cells(), which is exactly the point: it stays UNKNOWN.

    apply_hazard_belief_to_planning_grid(planning_grid, belief, block_threshold=0.5)

    observed_cell = planning_grid.world_to_grid(5.5, 5.5)
    unobserved_cell = planning_grid.world_to_grid(2.5, 2.5)
    assert planning_grid.get_value(observed_cell) == OG_OCCUPIED
    assert planning_grid.get_value(unobserved_cell) == OG_UNKNOWN


def test_legacy_apply_hazard_to_planning_grid_is_unchanged():
    """6/23. The ground-truth legacy function keeps blocking directly from
    HazardField, completely independent of any HazardBelief -- see
    test_planning_costmap_hazard.py for its full existing regression suite,
    kept untouched and still passing."""
    field = HazardField(bounds=_BOUNDS, resolution=_RESOLUTION)
    field.add_fire((5.5, 5.5), intensity=1.0, radius=2.0)
    planning_grid = _make_planning_grid()

    apply_hazard_to_planning_grid(planning_grid, field, block_threshold=0.5)

    cell = planning_grid.world_to_grid(5.5, 5.5)
    assert planning_grid.get_value(cell) == OG_OCCUPIED


def test_apply_hazard_belief_to_planning_grid_never_mutates_the_belief():
    belief = _make_belief()
    belief.observe_cells([5], [5], [0.8], robot_index=0)
    frame_before = belief.snapshot()
    planning_grid = _make_planning_grid()

    apply_hazard_belief_to_planning_grid(planning_grid, belief, block_threshold=0.5)

    frame_after = belief.snapshot()
    assert (frame_after.values == frame_before.values).all()
    assert (frame_after.observed == frame_before.observed).all()
    assert frame_after.revision == frame_before.revision


def test_apply_hazard_belief_to_planning_grid_rejects_shape_mismatch():
    import pytest

    belief = HazardBelief(GridGeometry((0.0, 3.0, 0.0, 3.0), _RESOLUTION), robot_count=1)  # 3x3
    planning_grid = _make_planning_grid()  # 10x10

    with pytest.raises(ValueError):
        apply_hazard_belief_to_planning_grid(planning_grid, belief, block_threshold=0.5)


# ---------------------------------------------------------------------------
# Engine-level fixture for REPLANNING tests (7-15) and REGRESIONES (20).
# ---------------------------------------------------------------------------


def _build_fake_engine(*, robot_xy=(0.5, 0.5), planner_type: str = "A*", running: bool = True) -> SimpleNamespace:
    robot = Robot(x=robot_xy[0], y=robot_xy[1], theta=0.0, v=0.0)
    agent = RobotAgent(robot_id=0, position=robot_xy, planner_mode="Goal seeking")

    belief_map = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1)
    hazard_service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1)

    fake = SimpleNamespace(
        running=running,
        robot=robot,
        robots=[],
        agent=agent,
        belief_map=belief_map,
        hazard_service=hazard_service,
        collision_checker=CollisionChecker(),
        config=SimpleNamespace(agent_mode="Single Robot Mode", planner_type=planner_type),
        console_logs=[],
        simulation_time=1.0,
        explored_free_points=set(),
        replan_calls=[],
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.canvas = SimpleNamespace(set_status=lambda message: None)
    fake.runtime_agent = lambda robot_index=None: fake.agent
    fake.ensure_belief_map = lambda: fake.belief_map
    fake.ensure_hazard_service = lambda: fake.hazard_service
    fake.safety_radius_for_robot = lambda robot_obj=None: 0.2
    fake.replan_after_new_information = lambda reason: fake.replan_calls.append(reason)

    for name in (
        "update_explored_free_points_from_polygon",
        "_replan_routes_affected_by_hazard",
        "_route_intersects_hazard_points",
        "current_route_points",
        "add_fire",
        "remove_fire_near",
        "on_fire_toggle_requested",
        "push_hazard_snapshot",
        "push_discovered_hazard_frame",
        "_invalidate_prefetch_request",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    return fake


# ---------------------------------------------------------------------------
# REPLANNING 7-9: creating/removing ground-truth FireSource never replans.
# ---------------------------------------------------------------------------


def test_add_fire_outside_fov_does_not_request_replan():
    fake = _build_fake_engine()
    fake.robot.set_waypoints([(9.5, 0.5)])

    assert fake.add_fire(5.5, 5.5) is True
    assert fake.replan_calls == []


def test_toggle_fire_outside_fov_does_not_request_replan():
    fake = _build_fake_engine()

    fake.on_fire_toggle_requested(5.5, 5.5)

    assert fake.replan_calls == []


def test_remove_fire_outside_fov_does_not_request_replan():
    fake = _build_fake_engine()
    fake.add_fire(5.5, 5.5)
    fake.replan_calls.clear()

    assert fake.remove_fire_near(5.5, 5.5) is True
    assert fake.replan_calls == []


# ---------------------------------------------------------------------------
# REPLANNING 10-12: observing must not over-trigger.
# ---------------------------------------------------------------------------


def test_observing_only_safe_cells_does_not_request_replan():
    fake = _build_fake_engine()
    fake.robot.set_waypoints([(9.5, 9.5)])
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)  # no fire anywhere

    fake.update_explored_free_points_from_polygon(polygon, robot_index=0)

    assert fake.replan_calls == []


def test_repeating_identical_observation_does_not_request_replan():
    fake = _build_fake_engine()
    fake.hazard_service.add_fire((5.5, 5.5))
    fake.robot.set_waypoints([(9.5, 9.5)])  # crosses (5.5, 5.5)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    fake.update_explored_free_points_from_polygon(polygon, robot_index=0)
    assert len(fake.replan_calls) == 1  # sanity: the first observation does trigger
    fake.replan_calls.clear()

    fake.update_explored_free_points_from_polygon(polygon, robot_index=0)  # identical repeat

    assert fake.replan_calls == []


def test_second_robot_attributing_an_already_blocked_cell_does_not_request_second_replan():
    fake = _build_fake_engine()
    fake.hazard_service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=2)
    fake.hazard_service.add_fire((5.5, 5.5))
    fake.robot.set_waypoints([(9.5, 9.5)])
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    fake.update_explored_free_points_from_polygon(polygon, robot_index=0)
    assert len(fake.replan_calls) == 1
    fake.replan_calls.clear()

    fake.update_explored_free_points_from_polygon(polygon, robot_index=1)  # same cells, robot 1

    assert fake.replan_calls == []


# ---------------------------------------------------------------------------
# REPLANNING 13-15: a real threshold crossing does drive the existing
# repair flow, and only for routes that actually cross it.
# ---------------------------------------------------------------------------


def test_first_observation_crossing_threshold_triggers_route_review():
    fake = _build_fake_engine()
    fake.hazard_service.add_fire((5.5, 5.5))
    fake.robot.set_waypoints([(9.5, 9.5)])  # diagonal route passes through (5.5, 5.5)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    fake.update_explored_free_points_from_polygon(polygon, robot_index=0)

    assert fake.replan_calls == ["Dynamic fire hazard affects current route."]


def test_route_not_crossing_observed_hazard_is_not_modified():
    fake = _build_fake_engine()
    fake.hazard_service.add_fire((5.5, 5.5))
    fake.robot.set_waypoints([(1.0, 9.0)])  # nowhere near (5.5, 5.5)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    fake.update_explored_free_points_from_polygon(polygon, robot_index=0)

    assert fake.replan_calls == []


def test_route_crossing_observed_hazard_enters_existing_repair_flow():
    """The existing repair algorithm (invalidate_pending_path() + replan_
    after_new_information()) actually runs, driven by observed_blocked_
    world_points() -- not a reimplementation."""
    fake = _build_fake_engine()
    fake.hazard_service.add_fire((5.5, 5.5))
    fake.agent.mark_pending_path_requested((3.0, 3.0))
    fake.agent.pending_path = [(2.0, 2.0), (3.0, 3.0)]
    fake.robot.set_waypoints([(9.5, 9.5)])
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    fake.update_explored_free_points_from_polygon(polygon, robot_index=0)

    assert fake.agent.pending_path is None  # existing repair flow discarded the stale prefetch
    assert fake.replan_calls == ["Dynamic fire hazard affects current route."]


# ---------------------------------------------------------------------------
# 16-19: observed_blocked_world_points() -- the discovered costmap surface.
# ---------------------------------------------------------------------------


def test_unobserved_hazard_never_appears_in_observed_blocked_world_points():
    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    service.add_fire((5.5, 5.5))  # ground truth hot, never observed

    assert service.observed_blocked_world_points() == ()


def test_observed_hazard_appears_in_observed_blocked_world_points():
    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    service.add_fire((5.5, 5.5))
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    service.observe_visible_polygon(polygon, robot_index=0)

    assert (5.5, 5.5) in service.observed_blocked_world_points()


def test_removing_fire_without_reobserving_keeps_cell_in_observed_blocked_points():
    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    service.add_fire((5.5, 5.5))
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    service.observe_visible_polygon(polygon, robot_index=0)
    assert (5.5, 5.5) in service.observed_blocked_world_points()

    service.remove_fire_near((5.5, 5.5))

    assert (5.5, 5.5) in service.observed_blocked_world_points()


def test_reobserving_after_removal_clears_cell_from_observed_blocked_points():
    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    service.add_fire((5.5, 5.5))
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    service.observe_visible_polygon(polygon, robot_index=0)
    service.remove_fire_near((5.5, 5.5))

    service.observe_visible_polygon(polygon, robot_index=0)  # re-observe

    assert (5.5, 5.5) not in service.observed_blocked_world_points()


# ---------------------------------------------------------------------------
# REGRESIONES 20-23.
# ---------------------------------------------------------------------------


def test_occupancy_grid_never_modified_by_replanning_or_costmap_functions():
    """20. Neither observed_blocked_world_points(), _replan_routes_affected_
    by_hazard(), nor apply_hazard_belief_to_planning_grid() ever touch
    BeliefMap.grid."""
    fake = _build_fake_engine()
    fake.belief_map.grid[3, 3] = OCCUPIED
    fake.belief_map.grid[1, 1] = FREE
    grid_before = fake.belief_map.grid.copy()
    fake.hazard_service.add_fire((5.5, 5.5))
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    fake.hazard_service.observe_visible_polygon(polygon, robot_index=0)
    fake.robot.set_waypoints([(9.5, 9.5)])

    fake._replan_routes_affected_by_hazard()
    apply_hazard_belief_to_planning_grid(_make_planning_grid(), fake.hazard_service.belief, block_threshold=0.55)

    assert (fake.belief_map.grid == grid_before).all()


def test_frontier_selection_unaffected_by_hazard_observation_and_replanning():
    """21. _frontier_cells() reads only BeliefMap -- hazard observation and
    the resulting replan decision must never change its result."""
    belief = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1)
    free_point = (2.0, 2.0)
    row, col = belief.world_to_cell(free_point)
    belief.grid[row, col] = FREE
    frontiers_before = _frontier_cells(belief)
    assert (row, col) in frontiers_before

    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    service.add_fire(free_point)
    polygon = _square_polygon(1.0, 1.0, 3.0, 3.0)
    service.observe_visible_polygon(polygon, robot_index=0)

    frontiers_after = _frontier_cells(belief)
    assert frontiers_after == frontiers_before


def test_hazard_field_legacy_contract_is_unchanged():
    """22. HazardField's own add/remove/values contract is untouched by
    Phase 3 -- only what consumes it (planning/route validation) changed."""
    field = HazardField(bounds=_BOUNDS, resolution=_RESOLUTION)
    source = field.add_fire((5.5, 5.5), intensity=0.8, radius=1.5)

    assert field.sources() == (source,)
    assert field.values(copy=False)[5, 5] > 0.0

    field.remove_fire(source.fire_id)
    assert field.sources() == ()
    assert not field.values(copy=False).any()


# ---------------------------------------------------------------------------
# Hot path: HazardBelief.snapshot() is an O(height*width) full-grid copy.
# observe_visible_polygon() runs once per robot per sensor update, and
# observed_blocked_world_points()/apply_hazard_belief_to_planning_grid() run
# once per planning request -- none of the three may use snapshot() anymore
# (see read_cells()/blocked_cells(), the narrow O(len(cells)) replacements).
# Monkeypatching snapshot() to raise proves it structurally, not just by
# coincidence of the current implementation.
# ---------------------------------------------------------------------------


def _forbid_snapshot(belief: HazardBelief, monkeypatch) -> None:
    def _raise():
        raise AssertionError("snapshot() must not be called on this hot path")

    monkeypatch.setattr(belief, "snapshot", _raise)


def test_observe_visible_polygon_does_not_call_belief_snapshot(monkeypatch):
    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    service.add_fire((5.5, 5.5))
    _forbid_snapshot(service.belief, monkeypatch)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    result = service.observe_visible_polygon(polygon, robot_index=0)

    assert result.changed is True
    assert result.newly_blocked_cells == 1


def test_observed_blocked_world_points_does_not_call_belief_snapshot(monkeypatch):
    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    service.add_fire((5.5, 5.5))
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    service.observe_visible_polygon(polygon, robot_index=0)
    _forbid_snapshot(service.belief, monkeypatch)

    points = service.observed_blocked_world_points()

    assert (5.5, 5.5) in points


def test_apply_hazard_belief_to_planning_grid_does_not_call_belief_snapshot(monkeypatch):
    belief = _make_belief()
    belief.observe_cells([5], [5], [0.8], robot_index=0)
    _forbid_snapshot(belief, monkeypatch)
    planning_grid = _make_planning_grid()

    apply_hazard_belief_to_planning_grid(planning_grid, belief, block_threshold=0.5)

    cell = planning_grid.world_to_grid(5.5, 5.5)
    assert planning_grid.get_value(cell) == OG_OCCUPIED
