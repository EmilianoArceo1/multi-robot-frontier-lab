"""
Regression tests distinguishing the current controller waypoint from the
final exploration target / path goal.

Manual Office.sim log showed:

    R1 route assigned: target=(7.25, 3.75), waypoints=6
    ...
    [NAV] kind=HOLD reason='frontier reached; no valid next frontier available'
    active_target=(5.25, -0.25) path_goal=(7.25, 3.75)

The robot was declared to have "reached the frontier" while sitting near an
INTERMEDIATE waypoint of its route, 4+ meters from the actual exploration
target (7.25, 3.75). This stopped exploration prematurely, mid-route.

Root cause: ExplorationBehavior.update() step 3 ("frontier reached") used
agent.active_target() / agent.distance_to_active_target() -- the current
controller waypoint from WaypointManager -- to decide whether the frontier
was reached. A route can be reassigned mid-flight (safety replan, prefetch
accept) with a fresh, shorter waypoint list starting near the robot's
current position; active_target() then points at a waypoint close to the
robot even though the true exploration target (active_path_goal_xy) is
still far away.

Fix: step 3 now checks agent.active_path_goal_xy /
agent.distance_to_active_path_goal() -- the FINAL destination of the active
route -- instead. active_target() is still used by step 5 (FOLLOW_PATH) for
the low-level controller, and should_prefetch_next_target() (step 4) was
already correctly using distance_to_active_path_goal() -- tests 3 and 4
below confirm that remains true.

These tests exercise RobotAgent and ExplorationBehavior directly (no Qt, no
canvas, no full engine/GUI) using the same _FakePlannerServices stub pattern
as the other exploration-behavior regression tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.observation import RobotObservation


FINAL_PATH_GOAL = (7.25, 3.75)
FIRST_WAYPOINT = (5.25, -0.25)
ALTERNATE_TARGET = (2.0, 3.0)


@dataclass
class _FakePlannerServices:
    """Stand-in for PlannerServices.select_exploration_target()."""

    target: tuple[float, float] | None
    calls: list[dict] = field(default_factory=list)

    def select_exploration_target(self, **kwargs) -> ExplorationPlannerResult:
        self.calls.append(kwargs)
        if self.target is None:
            return ExplorationPlannerResult(False, None, "no valid frontier candidates found")
        return ExplorationPlannerResult(True, self.target, "fake planner: selected target")


def _make_agent(position) -> RobotAgent:
    return RobotAgent(
        robot_id=0,
        position=position,
        planner_mode="FoV-aware directional frontier",
    )


def _make_observation(**overrides) -> RobotObservation:
    defaults = dict(
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        robot_radius=0.2,
        belief_map=None,
        planning_grid=None,
        mapped_obstacle_points=[],
        dynamic_obstacles=[],
        active_segment_blocked=False,
        predicted_collision=False,
        current_time=0.0,
        grid_resolution=0.5,
        goal_tolerance=0.25,
        sensor_range=2.5,
        final_goal_xy=None,
    )
    defaults.update(overrides)
    return RobotObservation(**defaults)


# ---------------------------------------------------------------------------
# 1. Reaching an intermediate waypoint must not count as "frontier reached".
# ---------------------------------------------------------------------------


def test_intermediate_waypoint_does_not_count_as_frontier_reached():
    # Robot sits exactly on the first waypoint of a multi-waypoint route
    # whose final destination is still far away -- mirrors the log evidence
    # (active_target=(5.25, -0.25), path_goal=(7.25, 3.75)).
    agent = _make_agent(position=FIRST_WAYPOINT)
    agent.assign_path(
        target=FINAL_PATH_GOAL,
        waypoints=[FIRST_WAYPOINT, (6.0, 1.0), FINAL_PATH_GOAL],
        planner_reason="test route",
    )

    observation = _make_observation(robot_xy=agent.position, goal_tolerance=0.25, grid_resolution=0.5)

    assert agent.distance_to_active_target() <= observation.goal_tolerance, (
        "test setup: robot must be within tolerance of the CURRENT waypoint"
    )
    assert agent.distance_to_active_path_goal() > 4.0, (
        "test setup: the FINAL path goal must still be far away"
    )

    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    behavior = ExplorationBehavior()
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind != "REQUEST_PLAN", (
        f"reaching an intermediate waypoint must not request a new frontier plan; decision={decision!r}"
    )
    assert "frontier reached" not in decision.reason
    assert len(fake_services.calls) == 0, (
        "the frontier planner must not even be consulted for an intermediate waypoint"
    )
    # Waypoint progression continues: the robot keeps tracking its current
    # waypoint.
    assert decision.kind == "FOLLOW_PATH"
    assert decision.target == FIRST_WAYPOINT


# ---------------------------------------------------------------------------
# 2. Reaching the actual final path goal DOES count as "frontier reached".
# ---------------------------------------------------------------------------


def test_frontier_reached_uses_final_path_goal_not_current_waypoint():
    agent = _make_agent(position=(7.2, 3.7))  # close to FINAL_PATH_GOAL
    agent.assign_path(
        target=FINAL_PATH_GOAL,
        waypoints=[FINAL_PATH_GOAL],
        planner_reason="final leg",
    )

    observation = _make_observation(
        robot_xy=agent.position, goal_tolerance=0.25, grid_resolution=0.5, current_time=10.0,
    )
    assert agent.distance_to_active_path_goal() <= observation.goal_tolerance

    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    behavior = ExplorationBehavior()
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.reason == "frontier reached; requesting next frontier"
    assert decision.target == ALTERNATE_TARGET
    assert len(fake_services.calls) == 1


# ---------------------------------------------------------------------------
# 3. Prefetch must not trigger from proximity to an intermediate waypoint
#    alone -- it must still be gated on distance to the final path goal.
# ---------------------------------------------------------------------------


def test_prefetch_uses_final_path_goal_distance_not_intermediate_waypoint_distance():
    agent = _make_agent(position=FIRST_WAYPOINT)
    agent.assign_path(
        target=FINAL_PATH_GOAL,
        waypoints=[FIRST_WAYPOINT, (6.0, 1.0), FINAL_PATH_GOAL],
        planner_reason="test route",
    )
    agent.last_prefetch_time = -1000.0  # cooldown clear

    observation = _make_observation(
        robot_xy=agent.position, goal_tolerance=0.25, grid_resolution=0.5, current_time=10.0,
    )
    assert agent.distance_to_active_target() <= 0.5
    assert agent.distance_to_active_path_goal() > 4.0

    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    behavior = ExplorationBehavior()
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind != "PREFETCH_NEXT_TARGET", (
        "must not prefetch based on proximity to an intermediate waypoint alone"
    )
    assert len(fake_services.calls) == 0


# ---------------------------------------------------------------------------
# 4. Prefetch DOES trigger once the robot is close to the final path goal.
# ---------------------------------------------------------------------------


def test_prefetch_triggers_near_final_path_goal():
    # 0.9 m from the final goal along y: outside goal_tolerance (so step 3
    # does not fire "reached") but inside the prefetch look-ahead distance.
    agent = _make_agent(position=(FINAL_PATH_GOAL[0], FINAL_PATH_GOAL[1] - 0.9))
    agent.assign_path(
        target=FINAL_PATH_GOAL,
        waypoints=[FINAL_PATH_GOAL],
        planner_reason="final leg",
    )
    agent.last_prefetch_time = -1000.0  # cooldown clear

    observation = _make_observation(
        robot_xy=agent.position, goal_tolerance=0.25, grid_resolution=0.5, current_time=10.0,
    )
    behavior = ExplorationBehavior()
    threshold = behavior.prefetch_distance(observation.grid_resolution, observation.goal_tolerance)

    assert agent.distance_to_active_path_goal() > observation.goal_tolerance, (
        "test setup: must not already count as 'reached'"
    )
    assert agent.distance_to_active_path_goal() <= threshold, (
        "test setup: must be within the prefetch look-ahead distance"
    )

    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "PREFETCH_NEXT_TARGET"
    assert decision.target == ALTERNATE_TARGET
