"""
Tests for this round's fixes to the navigation-debug diagnostics:
- the event log now records every tick (not just "relevant" events), so
  </> has real history to scrub through;
- planner/simplifier persist across ticks that did not just compute a
  fresh plan, instead of reading "unavailable" almost always.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.diagnostics.capture import PlanDebugCapture
from robotics_sim.diagnostics.event_log import NavigationDebugEventLog
from robotics_sim.diagnostics.navigation_snapshot import NavigationDebugEventKind
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.telemetry import TelemetryLogger


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]

    def active_waypoint(self):
        return self.waypoints[0] if getattr(self, "waypoints", None) else None


def _build_fake_engine() -> SimpleNamespace:
    position = (0.0, 0.0)
    # vision=... is required by _finalize_navigation_debug_snapshot()'s
    # sensor-polygon capture (see NavigationDebugSnapshot.sensor); config.
    # vision_model/obstacles feed the same call.
    robot = _FakeRobot(x=position[0], y=position[1], theta=0.0, v=0.0, vision=5.0)
    agent = RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        config=SimpleNamespace(
            planner_type="A*",
            path_simplifier="Direction changes",
            exploration_planner="FoV-aware directional frontier",
            goal_tolerance=0.25,
            grid_resolution=0.5,
            vision_model="LiDAR",
            obstacles=[],
        ),
        mapped_obstacle_points=[],
        current_exploration_target=None,
        route_result_count=0,
        last_goal_selection_reason="frontier selection reason",
        simulation_time=0.0,
        console_logs=[],
        planned_paths=[],
        exploration_targets=[],
        prefetch_workers={},
        prefetch_request_ids={0: 1},
        last_control=[0.1, 0.2],
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.collision_checker = CollisionChecker()
    fake.canvas = SimpleNamespace(
        set_planned_path=lambda path: fake.planned_paths.append(path),
        set_exploration_target=lambda target: fake.exploration_targets.append(target),
        set_status=lambda message: None,
    )
    fake.is_exploration_mode = lambda: True
    fake.safety_radius = lambda: 0.2
    fake.body_radius_for_robot = lambda robot=None: 0.15
    fake.safety_radius_for_robot = lambda robot=None: 0.2
    fake.planner_label = lambda: "A* / Direction changes + FoV-aware directional frontier"
    fake.clean_waypoints_for_current_start = lambda waypoints: [tuple(p) for p in waypoints]
    fake.final_goal_xy = lambda: (0.0, 0.0)
    fake.runtime_agent = lambda robot_index=None: fake.agent
    fake.active_target_xy = lambda: fake.agent.active_target()

    fake.navigation_debug_enabled = True
    fake.navigation_debug_log = NavigationDebugEventLog(max_size=10)
    fake._nav_debug_last_accepted_plan = None

    for name in (
        "apply_route_result",
        "_finalize_navigation_debug_snapshot",
        "log_route_assignment",
        "_navigation_debug_belief_frame",
        "_navigation_debug_hazard_frame",
        "_navigation_debug_hazard_belief_frame",
        "_navigation_debug_agent_state_frame",
        "_navigation_debug_metrics_frame",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


def test_every_finalize_call_is_recorded_not_only_relevant_ones():
    fake = _build_fake_engine()

    fake._finalize_navigation_debug_snapshot(
        agent=fake.agent,
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        event_kind=NavigationDebugEventKind.TICK,
        capture=None,
    )
    fake._finalize_navigation_debug_snapshot(
        agent=fake.agent,
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        event_kind=NavigationDebugEventKind.TICK,
        capture=None,
    )

    assert len(fake.navigation_debug_log) == 2, "TICK snapshots must now be recorded too, for </> to have real history"


def test_planner_and_simplifier_persist_after_the_accepting_tick():
    fake = _build_fake_engine()
    fake.mapped_obstacle_points = []  # clear route, will be accepted
    # Mirrors what compute_route() stashes on self before calling
    # apply_route_result() in the real engine -- see compute_route()'s
    # "_nav_debug_last_plan_capture" comment.
    fake._nav_debug_last_plan_capture = PlanDebugCapture(
        planner_name="A*", simplifier_name="Direction changes"
    )

    fake.apply_route_result(True, "path found with A*", [(1.0, 0.0)])
    accepted_event = fake.navigation_debug_log.latest()
    assert accepted_event.snapshot.path.planner_name.unavailable is False

    # A routine tick afterwards, with no fresh plan capture at all.
    fake._finalize_navigation_debug_snapshot(
        agent=fake.agent,
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        event_kind=NavigationDebugEventKind.TICK,
        capture=None,
    )
    routine_snapshot = fake.navigation_debug_log.latest().snapshot

    assert routine_snapshot.path.planner_name.unavailable is False
    assert routine_snapshot.path.planner_name.value == accepted_event.snapshot.path.planner_name.value
