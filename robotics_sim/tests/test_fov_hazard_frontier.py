"""End-to-end contracts for the replacement FoV/hazard frontier planner."""
from __future__ import annotations

import pytest

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.environment.occupancy_grid import OCCUPIED
from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import FoVAwareHazardFrontierPlanner
from robotics_sim.simulation.hazard_service import RuntimeHazardService
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.planner_services import PlannerServices


def _free_belief() -> BeliefMap:
    belief = BeliefMap(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        robot_count=1,
    )
    for row in range(belief.height):
        for col in range(belief.width):
            belief.mark_free_cell((row, col))
    return belief


def test_fire_is_discovered_when_its_edge_enters_fov_before_centre():
    service = RuntimeHazardService(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        default_radius=2.0,
    )
    source = service.add_fire((7.5, 5.5)).source
    assert source is not None
    assert service.discovered_sources() == ()

    # Cell centre (6.5, 5.5) is inside the fire footprint and polygon, while
    # the fire centre (7.5, 5.5) remains outside the FoV.
    polygon = [(6.0, 5.0), (7.0, 5.0), (7.0, 6.0), (6.0, 6.0)]
    result = service.observe_visible_polygon(polygon, robot_index=0)

    centre_cell = service.field.geometry.world_to_grid(*source.position)
    assert centre_cell is not None
    frame = service.belief.snapshot()
    assert bool(frame.observed[centre_cell.row, centre_cell.col]) is False
    assert result.newly_discovered_sources == (source,)
    assert service.discovered_sources() == (source,)


def test_reobserving_same_fire_edge_does_not_emit_duplicate_discovery():
    service = RuntimeHazardService(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        default_radius=2.0,
    )
    source = service.add_fire((7.5, 5.5)).source
    assert source is not None
    polygon = [(6.0, 5.0), (7.0, 5.0), (7.0, 6.0), (6.0, 6.0)]

    first = service.observe_visible_polygon(polygon, robot_index=0)
    second = service.observe_visible_polygon(polygon, robot_index=0)

    assert first.newly_discovered_sources == (source,)
    assert second.newly_discovered_sources == ()


def test_observed_safe_centre_does_not_expose_stale_ground_truth_source():
    service = RuntimeHazardService(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
    )
    source = service.add_fire((7.25, 5.25)).source
    assert source is not None
    cell = service.field.geometry.world_to_grid(*source.position)
    assert cell is not None

    service.belief.observe_cells([cell.row], [cell.col], [0.0], robot_index=0)

    assert service.discovered_sources() == ()


def test_discovered_fire_centre_is_a_direct_target_without_any_frontier():
    belief = _free_belief()
    fire_position = (7.25, 5.25)

    result = FoVAwareHazardFrontierPlanner().select_goal(
        belief_map=belief,
        robot_xy=(2.5, 5.5),
        robot_heading=0.0,
        robot_radius=0.2,
        sensor_range=2.5,
        known_hazards=[fire_position],
    )

    assert result.success is True
    assert result.target == fire_position
    assert len(result.candidates) == 1
    assert "kind=hazard" in result.candidates[0].reason
    assert result.candidates[0].hazard_gain == 1.0
    assert "selected newly detected hazard centre" in result.reason


def test_fire_keeps_its_exact_position_when_it_shares_a_frontier_cell():
    belief = BeliefMap(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        robot_count=1,
    )
    for col in range(2, 8):
        belief.mark_free_cell((5, col))
    fire_position = (7.25, 5.25)

    result = FoVAwareHazardFrontierPlanner().select_goal(
        belief_map=belief,
        robot_xy=(2.5, 5.5),
        robot_heading=0.0,
        robot_radius=0.2,
        sensor_range=2.5,
        known_hazards=[fire_position],
    )

    assert result.success is True
    assert result.target == fire_position
    assert any(
        candidate.target == fire_position and "kind=hazard" in candidate.reason
        for candidate in result.candidates
    )
    assert any(candidate.kind != "hazard" for candidate in result.candidates)


def test_unreachable_fire_centre_falls_back_to_frontier_exploration():
    belief = BeliefMap(
        bounds=(0.0, 10.0, 0.0, 10.0),
        resolution=1.0,
        robot_count=1,
    )
    for col in range(2, 8):
        belief.mark_free_cell((5, col))
    fire_position = (7.25, 5.25)
    fire_cell = belief.world_to_cell(fire_position)
    assert fire_cell is not None
    planning_grid = belief.to_planning_grid(unknown_is_traversable=True)
    planning_grid.set_value(GridCell(*fire_cell), OCCUPIED)

    result = FoVAwareHazardFrontierPlanner().select_goal(
        belief_map=belief,
        robot_xy=(2.5, 5.5),
        robot_heading=0.0,
        robot_radius=0.2,
        sensor_range=2.5,
        known_hazards=[fire_position],
        planning_grid=planning_grid,
    )

    assert result.success is True
    assert result.target != fire_position
    assert all(candidate.kind != "hazard" for candidate in result.candidates)


def test_planner_does_not_infer_an_undiscovered_fire():
    result = FoVAwareHazardFrontierPlanner().select_goal(
        belief_map=_free_belief(),
        robot_xy=(2.5, 5.5),
        robot_heading=0.0,
        robot_radius=0.2,
        sensor_range=2.5,
        known_hazards=[],
    )

    assert result.success is False
    assert result.target is None
    assert "no frontier, forward-recovery, or discovered-hazard" in result.reason


def test_planner_services_forwards_discovered_hazards_to_the_selected_planner():
    services = PlannerServices()
    services.known_hazards = ((7.25, 5.25),)

    result = services.select_exploration_target(
        planner_name="FoV-aware directional frontier",
        belief_map=_free_belief(),
        robot_xy=(2.5, 5.5),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_radius=0.2,
        sensor_range=2.5,
        vision_model="LiDAR",
        ipp_distance_penalty=1.0,
    )

    assert result.success is True
    assert result.target == (7.25, 5.25)


def test_multi_robot_term_uses_only_other_robot_reservations():
    belief = _free_belief()
    fire_position = (7.25, 5.25)
    planner = FoVAwareHazardFrontierPlanner()

    navigation_exclusion = planner.select_goal(
        belief_map=belief,
        robot_xy=(2.5, 5.5),
        robot_heading=0.0,
        robot_radius=0.2,
        sensor_range=2.5,
        known_hazards=[fire_position],
        excluded_targets=[fire_position],
        target_exclusion_radius=1.0,
    )
    other_robot_reservation = planner.select_goal(
        belief_map=belief,
        robot_xy=(2.5, 5.5),
        robot_heading=0.0,
        robot_radius=0.2,
        sensor_range=2.5,
        known_hazards=[fire_position],
        reserved_targets=[fire_position],
        target_exclusion_radius=1.0,
    )

    assert "multi=0.000000" in navigation_exclusion.candidates[0].reason
    assert "multi=1.000000" in other_robot_reservation.candidates[0].reason
    assert other_robot_reservation.candidates[0].score == pytest.approx(
        navigation_exclusion.candidates[0].score - 1.2
    )


def test_newly_discovered_fire_interrupts_active_frontier_route_once():
    belief = _free_belief()
    fire_position = (7.25, 5.25)
    services = PlannerServices()
    services.known_hazards = (fire_position,)

    agent = RobotAgent(
        robot_id=0,
        position=(2.5, 5.5),
        planner_mode="FoV-aware directional frontier",
    )
    old_frontier = (4.5, 5.5)
    agent.exploration_target_xy = old_frontier
    agent.assign_path(target=old_frontier, waypoints=[old_frontier])

    observation = RobotObservation(
        robot_xy=agent.position,
        robot_heading=agent.heading,
        robot_radius=agent.radius,
        belief_map=belief,
        planning_grid=None,
        mapped_obstacle_points=[],
        dynamic_obstacles=[],
        active_segment_blocked=False,
        predicted_collision=False,
        current_time=1.0,
        grid_resolution=1.0,
        goal_tolerance=0.25,
        sensor_range=2.5,
        final_goal_xy=None,
        vision_model="LiDAR",
        ipp_distance_penalty=0.0,
    )

    behavior = ExplorationBehavior()
    decision = behavior.update(agent, observation, services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == fire_position
    assert decision.brake is True
    assert decision.force_new_target is True
    assert "fire footprint entered FoV" in decision.reason

    # The same source remains known, but it is no longer a new mission event.
    # Even before the engine applies the first request, a second frame must not
    # create a replan storm for the same fire.
    second = behavior.update(agent, observation, services)
    assert not (second.kind == "REQUEST_PLAN" and second.target == fire_position)
