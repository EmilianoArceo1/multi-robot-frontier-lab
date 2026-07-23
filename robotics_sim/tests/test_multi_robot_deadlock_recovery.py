from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot import Robot
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.simulation.engine import SimulationControllerMixin


def _bind(fake, *names: str) -> None:
    for name in names:
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))


def test_reserved_frontier_exactly_at_minimum_separation_is_valid():
    fake = SimpleNamespace(
        config=SimpleNamespace(grid_resolution=0.5),
        multi_exploration_targets=[(0.0, 0.0), (1.0, 0.0)],
    )
    _bind(fake, "multi_frontier_exclusion_radius", "target_is_clear_of_reserved_frontiers")

    valid, reason = fake.target_is_clear_of_reserved_frontiers(0, (0.0, 0.0))

    assert valid is True
    assert reason == "target clear of reserved frontiers"


def test_invalid_coordinator_target_is_blacklisted_and_reselected_immediately():
    ego = SimpleNamespace(x=0.0, y=0.0)
    teammate = SimpleNamespace(x=0.5, y=0.0)
    fake = SimpleNamespace(
        robots=[ego, teammate],
        config=SimpleNamespace(
            exploration_planner="Fov-aware directional frontier",
            coordinator_type="FUEL Frontier Baseline",
            grid_resolution=0.5,
        ),
        multi_exploration_targets=[None, None],
        multi_invalidated_exploration_targets=[[], []],
        MAX_TARGET_RESELECTION_ATTEMPTS=4,
        logs=[],
    )
    fake.final_goal_xy = lambda: (9.0, 9.0)
    fake.safety_radius_for_robot = lambda robot: 0.25
    fake.publish_multi_exploration_targets = lambda: None
    fake.temporary_separation_target_for_robot = lambda index: None
    fake.log_console_message = lambda message, **kwargs: fake.logs.append(message)

    proposals = [(0.5, 0.0), (3.0, 0.0)]

    def synchronize(*, requesting_robot_index, force_new_target):
        fake.multi_exploration_targets[requesting_robot_index] = proposals.pop(0)

    fake.synchronize_multi_frontier_targets = synchronize
    _bind(
        fake,
        "ensure_multi_exploration_target_slots",
        "multi_frontier_exclusion_radius",
        "multi_dynamic_target_margin",
        "target_is_clear_of_reserved_frontiers",
        "target_is_clear_of_dynamic_robots",
        "multi_exploration_target_is_valid",
        "invalidate_current_multi_frontier",
        "select_navigation_goal_for_multi_robot",
    )

    target, reason = fake.select_navigation_goal_for_multi_robot(0, (0.0, 0.0))

    assert target == (3.0, 0.0)
    assert "frontier assigned" in reason
    assert (0.5, 0.0) in fake.multi_invalidated_exploration_targets[0]
    assert proposals == []
    assert any("trying alternative" in message for message in fake.logs)


def test_only_teammate_blocked_targets_request_retryable_wait():
    ego = SimpleNamespace(x=0.0, y=0.0)
    teammate = SimpleNamespace(x=1.0, y=0.0)
    fake = SimpleNamespace(
        robots=[ego, teammate],
        config=SimpleNamespace(
            exploration_planner="Fov-aware directional frontier",
            coordinator_type="MARVEL CTDE graph-attention policy (scaled environment)",
            grid_resolution=0.5,
        ),
        multi_exploration_targets=[None, None],
        multi_invalidated_exploration_targets=[[], []],
        MAX_TARGET_RESELECTION_ATTEMPTS=2,
        ROUTE_STATE_WAITING_FOR_CORRIDOR=(
            SimulationControllerMixin.ROUTE_STATE_WAITING_FOR_CORRIDOR
        ),
        logs=[],
    )
    fake.final_goal_xy = lambda: (9.0, 9.0)
    fake.safety_radius_for_robot = lambda robot: 0.35
    fake.publish_multi_exploration_targets = lambda: None
    fake.temporary_separation_target_for_robot = lambda index: None
    fake.log_console_message = lambda message, **kwargs: fake.logs.append(message)
    proposals = [(1.0, 0.0), (1.2, 0.0)]

    def synchronize(*, requesting_robot_index, force_new_target):
        fake.multi_exploration_targets[requesting_robot_index] = proposals.pop(0)

    fake.synchronize_multi_frontier_targets = synchronize
    _bind(
        fake,
        "ensure_multi_exploration_target_slots",
        "multi_frontier_exclusion_radius",
        "multi_dynamic_target_margin",
        "target_is_clear_of_reserved_frontiers",
        "target_is_clear_of_dynamic_robots",
        "multi_exploration_target_is_valid",
        "invalidate_current_multi_frontier",
        "select_navigation_goal_for_multi_robot",
    )

    target, reason = fake.select_navigation_goal_for_multi_robot(0, (0.0, 0.0))

    assert target == (0.0, 0.0)
    assert "no valid frontier assigned" in reason
    assert fake._multi_goal_hold_states_by_robot[0] == (
        SimulationControllerMixin.ROUTE_STATE_WAITING_FOR_CORRIDOR
    )


def test_safe_formation_does_not_receive_host_generated_separation_targets():
    """A coordinator HOLD must remain HOLD when pairwise clearance is valid."""
    ego = SimpleNamespace(x=0.0, y=0.0)
    teammate = SimpleNamespace(x=0.0, y=1.23)
    fake = SimpleNamespace(
        robots=[ego, teammate],
        config=SimpleNamespace(grid_resolution=0.5),
    )
    fake.safety_radius_for_robot = lambda robot: 0.35
    _bind(fake, "multi_dynamic_target_margin", "temporary_separation_target_for_robot")

    assert fake.temporary_separation_target_for_robot(0) is None
    assert fake.temporary_separation_target_for_robot(1) is None


def test_actual_clearance_violation_can_receive_emergency_separation_target():
    ego = SimpleNamespace(x=0.0, y=0.0)
    teammate = SimpleNamespace(x=0.0, y=0.5)
    fake = SimpleNamespace(
        robots=[ego, teammate],
        config=SimpleNamespace(grid_resolution=0.5),
    )
    fake.safety_radius_for_robot = lambda robot: 0.35
    _bind(fake, "multi_dynamic_target_margin", "temporary_separation_target_for_robot")

    target = fake.temporary_separation_target_for_robot(0)

    assert target is not None
    assert target[1] < ego.y


def test_coordinator_hold_is_not_replaced_by_host_motion_in_safe_formation():
    ego = SimpleNamespace(x=0.0, y=0.0)
    teammate = SimpleNamespace(x=0.0, y=1.23)
    fake = SimpleNamespace(
        robots=[ego, teammate],
        config=SimpleNamespace(
            exploration_planner="FoV-aware directional frontier",
            coordinator_type="MARVEL CTDE graph-attention policy",
            grid_resolution=0.5,
        ),
        multi_exploration_targets=[None, None],
        multi_invalidated_exploration_targets=[[], []],
        MAX_TARGET_RESELECTION_ATTEMPTS=4,
        last_goal_selection_reason="",
    )
    fake.final_goal_xy = lambda: (9.0, 9.0)
    fake.safety_radius_for_robot = lambda robot: 0.35
    fake.publish_multi_exploration_targets = lambda: None
    def coordinator_hold(**kwargs):
        fake.last_goal_selection_reason = (
            "R1: MARVEL checkpoint not found at algorithms/marvel/weights/"
            "checkpoint.pth [MARVEL CTDE graph-attention policy]"
        )

    fake.synchronize_multi_frontier_targets = coordinator_hold
    _bind(
        fake,
        "ensure_multi_exploration_target_slots",
        "multi_dynamic_target_margin",
        "temporary_separation_target_for_robot",
        "select_navigation_goal_for_multi_robot",
    )

    target, reason = fake.select_navigation_goal_for_multi_robot(0, (0.0, 0.0))

    assert target == (0.0, 0.0)
    assert "no valid frontier assigned" in reason
    assert "holding position" in reason
    assert "MARVEL checkpoint not found" in reason
    assert fake.multi_exploration_targets == [None, None]


def test_emergency_separation_never_enters_task_assignment_target_slots():
    ego = SimpleNamespace(x=0.0, y=0.0)
    teammate = SimpleNamespace(x=0.0, y=0.5)
    fake = SimpleNamespace(
        robots=[ego, teammate],
        config=SimpleNamespace(
            exploration_planner="FoV-aware directional frontier",
            coordinator_type="test coordinator",
            grid_resolution=0.5,
        ),
        multi_exploration_targets=[None, None],
        multi_invalidated_exploration_targets=[[], []],
        MAX_TARGET_RESELECTION_ATTEMPTS=4,
        last_goal_selection_reason="",
    )
    fake.final_goal_xy = lambda: (9.0, 9.0)
    fake.safety_radius_for_robot = lambda robot: 0.35
    fake.publish_multi_exploration_targets = lambda: None
    fake.synchronize_multi_frontier_targets = lambda **kwargs: None
    fake.multi_exploration_target_is_valid = lambda index, target: (True, "valid")
    _bind(
        fake,
        "ensure_multi_exploration_target_slots",
        "multi_dynamic_target_margin",
        "temporary_separation_target_for_robot",
        "select_navigation_goal_for_multi_robot",
    )

    target, reason = fake.select_navigation_goal_for_multi_robot(0, (0.0, 0.0))

    assert target != (0.0, 0.0)
    assert "temporary separation target" in reason
    assert fake.multi_exploration_targets == [None, None]


def test_predicted_tracking_collision_uses_safe_rotation_without_retargeting():
    robot = Robot(
        x=0.0,
        y=0.0,
        theta=0.0,
        v=0.6,
        goal=(0.0, 2.0),
        max_speed=1.5,
        max_acceleration=2.0,
        max_angular_speed=2.5,
        robot_radius=0.35,
    )
    fake = SimpleNamespace(
        robots=[robot],
        robot=robot,
        collision_checker=CollisionChecker(),
        config=SimpleNamespace(
            obstacles=[],
            max_speed=1.5,
            max_acceleration=2.0,
            max_angular_speed=2.5,
        ),
    )
    _bind(fake, "robot_snapshot", "predicted_motion_report", "multi_rotation_escape_control")

    control = fake.multi_rotation_escape_control(
        robot_index=0,
        target=(0.0, 2.0),
        dt=0.05,
        robot_radius=0.35,
        known_obstacle_points=[],
    )

    assert control is not None
    assert np.asarray(control).shape == (2, 1)
    assert float(control[0, 0]) == 0.0
    assert float(control[1, 0]) > 0.0
    assert robot.v == 0.0
    assert np.allclose(robot.active_waypoint(), (0.0, 2.0))


def test_predicted_collision_hard_stops_before_replan_when_already_aligned():
    robot = Robot(x=0.0, y=0.0, theta=0.0, v=0.6, goal=(2.0, 0.0))
    fake = SimpleNamespace(
        robots=[robot],
        robot=robot,
        collision_checker=CollisionChecker(),
        config=SimpleNamespace(
            obstacles=[],
            max_speed=1.0,
            max_acceleration=1.0,
            max_angular_speed=1.0,
        ),
    )
    _bind(fake, "robot_snapshot", "predicted_motion_report", "multi_rotation_escape_control")

    control = fake.multi_rotation_escape_control(
        robot_index=0,
        target=(2.0, 0.0),
        dt=0.05,
        robot_radius=0.2,
        known_obstacle_points=[],
    )

    assert control is None
    assert robot.v == 0.0
    assert np.allclose(robot.active_waypoint(), (2.0, 0.0))


def test_identical_static_safety_repairs_are_bounded_and_throttled_frames_do_not_count():
    robot = SimpleNamespace(x=4.5978, y=2.7077)
    fake = SimpleNamespace(
        robots=[robot],
        config=SimpleNamespace(exploration_replan_cooldown=1.0, grid_resolution=0.5),
        simulation_time=0.0,
        MAX_SAME_TARGET_STATIC_SAFETY_REPAIRS=2,
    )
    _bind(
        fake,
        "ensure_multi_replan_guard_slots",
        "safety_replan_cooldown_seconds",
        "multi_safety_replan_allowed",
        "repeated_multi_safety_replan_requires_new_target",
        "reset_multi_safety_replan_streak",
    )

    target = (3.75, 3.25)
    reason = "Predicted collision before motion update"

    assert fake.multi_safety_replan_allowed(0, reason, target)
    assert not fake.repeated_multi_safety_replan_requires_new_target(0)

    fake.simulation_time = 0.1
    assert not fake.multi_safety_replan_allowed(0, reason, target)
    assert fake.multi_safety_replan_streaks == [1]

    fake.simulation_time = 1.0
    assert fake.multi_safety_replan_allowed(0, reason, target)
    assert not fake.repeated_multi_safety_replan_requires_new_target(0)

    fake.simulation_time = 2.0
    assert fake.multi_safety_replan_allowed(0, reason, target)
    assert fake.repeated_multi_safety_replan_requires_new_target(0)

    fake.reset_multi_safety_replan_streak(0)
    assert fake.multi_safety_replan_streaks == [0]
    assert not fake.repeated_multi_safety_replan_requires_new_target(0)


def test_safety_repair_streak_resets_after_real_position_progress():
    robot = SimpleNamespace(x=0.0, y=0.0)
    fake = SimpleNamespace(
        robots=[robot],
        config=SimpleNamespace(exploration_replan_cooldown=1.0, grid_resolution=0.5),
        simulation_time=0.0,
        MAX_SAME_TARGET_STATIC_SAFETY_REPAIRS=2,
    )
    _bind(
        fake,
        "ensure_multi_replan_guard_slots",
        "safety_replan_cooldown_seconds",
        "multi_safety_replan_allowed",
        "repeated_multi_safety_replan_requires_new_target",
    )

    assert fake.multi_safety_replan_allowed(0, "static veto", (2.0, 0.0))
    fake.simulation_time = 1.0
    robot.x = 0.2
    assert fake.multi_safety_replan_allowed(0, "static veto", (2.0, 0.0))

    assert fake.multi_safety_replan_streaks == [1]
    assert not fake.repeated_multi_safety_replan_requires_new_target(0)
