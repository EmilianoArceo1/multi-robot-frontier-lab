"""Regression tests for per-robot Navigation Reasoning in Multiple mode."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.diagnostics.capture import NavigationDebugCapture, PlanDebugCapture
from robotics_sim.diagnostics.event_log import NavigationDebugEventLog
from robotics_sim.diagnostics.navigation_snapshot import NavigationDebugEventKind
from robotics_sim.simulation.engine import SimulationControllerMixin


class _Robot(SimpleNamespace):
    def active_waypoint(self):
        manager = self.waypoints
        if manager.current_index >= len(manager.waypoints):
            return None
        return manager.waypoints[manager.current_index]


class _Canvas:
    def __init__(self) -> None:
        self.snapshots = []
        self.events = []
        self.positions = []

    def set_navigation_debug_snapshot(self, snapshot) -> None:
        self.snapshots.append(snapshot)

    def set_navigation_debug_last_event(self, event) -> None:
        self.events.append(event)

    def set_navigation_debug_history_position(self, position, total) -> None:
        self.positions.append((position, total))


def _engine():
    robots = [
        _Robot(
            x=0.0,
            y=0.0,
            theta=0.0,
            v=0.1,
            vision=4.0,
            waypoints=SimpleNamespace(waypoints=[(1.0, 0.0)], current_index=0),
        ),
        _Robot(
            x=5.0,
            y=2.0,
            theta=1.0,
            v=0.2,
            vision=6.0,
            waypoints=SimpleNamespace(waypoints=[(5.0, 3.0)], current_index=0),
        ),
    ]
    agents = [
        RobotAgent(robot_id=i, position=(robot.x, robot.y), planner_mode="FoV-aware directional frontier")
        for i, robot in enumerate(robots)
    ]
    canvas = _Canvas()
    fake = SimpleNamespace(
        robot=robots[0],
        robots=robots,
        robot_agents=agents,
        selected_robot_index=0,
        config=SimpleNamespace(vision_model="LiDAR", obstacles=[]),
        mapped_obstacle_points=[],
        simulation_time=3.5,
        last_control=np.array([[9.0], [9.0]]),
        multi_last_controls=[np.zeros((2, 1)), np.zeros((2, 1))],
        multi_route_states=["ACTIVE", "WAITING_FOR_CORRIDOR"],
        navigation_debug_log=NavigationDebugEventLog(max_size=20),
        _nav_debug_seq=0,
        _nav_debug_last_accepted_plan=None,
        _nav_debug_last_accepted_plan_by_robot={
            0: PlanDebugCapture(planner_name="A*", simplifier_name="LOS"),
            1: PlanDebugCapture(planner_name="Dijkstra", simplifier_name="Direction changes"),
        },
        _nav_debug_live_snapshot=None,
        _nav_debug_live_snapshots_by_robot={},
        _nav_debug_last_event_by_robot={},
        _nav_debug_history_index=None,
        canvas=canvas,
    )
    fake.body_radius_for_robot = lambda robot: 0.15
    fake.safety_radius_for_robot = lambda robot: 0.25
    fake._navigation_debug_belief_frame = lambda: SimulationControllerMixin._navigation_debug_belief_frame(fake)
    fake._navigation_debug_hazard_frame = lambda: SimulationControllerMixin._navigation_debug_hazard_frame(fake)
    fake._navigation_debug_hazard_belief_frame = (
        lambda: SimulationControllerMixin._navigation_debug_hazard_belief_frame(fake)
    )
    fake._navigation_debug_agent_state_frame = (
        lambda agent: SimulationControllerMixin._navigation_debug_agent_state_frame(fake, agent)
    )
    fake._navigation_debug_metrics_frame = lambda: SimulationControllerMixin._navigation_debug_metrics_frame(fake)
    for name in (
        "_finalize_navigation_debug_snapshot",
        "navigation_debug_history_length",
        "select_navigation_debug_robot",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake, agents, canvas


def test_multiple_mode_records_distinct_snapshots_but_publishes_selected_robot_only():
    fake, agents, canvas = _engine()

    fake._finalize_navigation_debug_snapshot(
        agent=agents[0],
        robot=fake.robots[0],
        robot_index=0,
        decision_kind="FOLLOW_PATH",
        decision_reason="R1 active",
        event_kind=NavigationDebugEventKind.TICK,
        capture=NavigationDebugCapture(),
        control=np.array([[0.3], [0.4]]),
    )
    fake._finalize_navigation_debug_snapshot(
        agent=agents[1],
        robot=fake.robots[1],
        robot_index=1,
        decision_kind="FOLLOW_PATH",
        decision_reason="R2 waiting",
        event_kind=NavigationDebugEventKind.TICK,
        capture=NavigationDebugCapture(),
        control=np.array([[0.5], [0.6]]),
    )

    events = fake.navigation_debug_log.events()
    assert [event.snapshot.robot_id for event in events] == ["R1", "R2"]
    assert events[0].snapshot.path.active_segment == ((0.0, 0.0), (1.0, 0.0))
    assert events[1].snapshot.path.active_segment == ((5.0, 2.0), (5.0, 3.0))
    assert events[0].snapshot.path.planner_name.value == "A*"
    assert events[1].snapshot.path.planner_name.value == "Dijkstra"
    assert events[1].snapshot.navigation_state == "WAITING_FOR_CORRIDOR"
    assert events[1].snapshot.controller.omega == 0.6

    # R2 was recorded, but must not replace the selected R1 live overlay.
    assert [snapshot.robot_id for snapshot in canvas.snapshots] == ["R1"]
    assert fake._nav_debug_live_snapshot.robot_id == "R1"
    assert set(fake._nav_debug_live_snapshots_by_robot) == {0, 1}


def test_selecting_a_runtime_robot_swaps_in_its_frozen_live_reasoning_frame():
    fake, agents, canvas = _engine()
    for index in range(2):
        fake._finalize_navigation_debug_snapshot(
            agent=agents[index],
            robot=fake.robots[index],
            robot_index=index,
            decision_kind="FOLLOW_PATH",
            decision_reason=f"R{index + 1} active",
            event_kind=NavigationDebugEventKind.TICK,
            capture=NavigationDebugCapture(),
            control=np.zeros((2, 1)),
        )

    fake.selected_robot_index = 1
    fake.select_navigation_debug_robot(1)

    assert canvas.snapshots[-1].robot_id == "R2"
    assert fake._nav_debug_live_snapshot.robot_id == "R2"
    assert fake._nav_debug_history_index is None
    assert canvas.positions[-1] == (None, 1)


def test_sparse_relevant_event_is_cached_per_robot_not_shared_across_team():
    fake, agents, canvas = _engine()
    fake.selected_robot_index = 1

    fake._finalize_navigation_debug_snapshot(
        agent=agents[0],
        robot=fake.robots[0],
        robot_index=0,
        decision_kind="REPLAN_FOR_SAFETY",
        decision_reason="R1 segment blocked",
        event_kind=NavigationDebugEventKind.SAFETY_REPLAN,
        capture=NavigationDebugCapture(),
        control=np.zeros((2, 1)),
    )
    fake._finalize_navigation_debug_snapshot(
        agent=agents[1],
        robot=fake.robots[1],
        robot_index=1,
        decision_kind="REPLAN_FOR_SAFETY",
        decision_reason="R2 predicted collision",
        event_kind=NavigationDebugEventKind.PREDICTED_COLLISION,
        capture=NavigationDebugCapture(),
        control=np.zeros((2, 1)),
    )

    assert fake._nav_debug_last_event_by_robot[0].snapshot.robot_id == "R1"
    assert fake._nav_debug_last_event_by_robot[1].snapshot.robot_id == "R2"
    assert canvas.events[-1].snapshot.robot_id == "R2"
    assert canvas.events[-1].event_kind is NavigationDebugEventKind.PREDICTED_COLLISION


def test_routine_navigation_frames_are_rate_limited_per_robot() -> None:
    fake, _, _ = _engine()
    fake._nav_debug_last_tick_time_by_robot = {}
    fake.navigation_debug_tick_due = SimulationControllerMixin.navigation_debug_tick_due.__get__(fake)

    assert fake.navigation_debug_tick_due(0) is True
    assert fake.navigation_debug_tick_due(1) is True
    assert fake.navigation_debug_tick_due(0) is False
    assert fake.navigation_debug_tick_due(1) is False

    fake.simulation_time += 0.11
    assert fake.navigation_debug_tick_due(0) is True
    assert fake.navigation_debug_tick_due(1) is True
