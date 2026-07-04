"""
Runtime robot registry.

Robot is still the physical/dynamics object used by the simulator.
RobotAgent stores navigation state for each runtime robot.

This registry keeps both layers synchronized without forcing a full rewrite of
the existing Robot class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from robotics_sim.core.robot_agent import RobotAgent


def _robot_xy(robot) -> tuple[float, float]:
    return (float(robot.x), float(robot.y))


def _robot_heading(robot) -> float:
    return float(getattr(robot, "theta", 0.0))


@dataclass
class RuntimeRobotRegistry:
    agents: list[RobotAgent] = field(default_factory=list)

    def reset(self) -> None:
        self.agents = []

    def sync_from_robots(
        self,
        *,
        robots: Iterable,
        planner_mode: str,
        final_goal_xy: tuple[float, float] | None,
        radii: Iterable[float] | None = None,
    ) -> list[RobotAgent]:
        robot_list = list(robots)
        radius_list = list(radii) if radii is not None else [0.20 for _ in robot_list]

        while len(self.agents) < len(robot_list):
            idx = len(self.agents)
            robot = robot_list[idx]
            radius = float(radius_list[idx]) if idx < len(radius_list) else 0.20
            agent = RobotAgent(
                robot_id=idx,
                position=_robot_xy(robot),
                heading=_robot_heading(robot),
                radius=radius,
                planner_mode=str(planner_mode),
            )
            if final_goal_xy is not None:
                agent.set_final_goal(final_goal_xy)
            self.agents.append(agent)

        if len(self.agents) > len(robot_list):
            self.agents = self.agents[: len(robot_list)]

        for idx, robot in enumerate(robot_list):
            agent = self.agents[idx]
            agent.robot_id = idx
            agent.set_position(_robot_xy(robot))
            agent.set_heading(_robot_heading(robot))
            if idx < len(radius_list):
                agent.radius = float(radius_list[idx])
            agent.set_planner_mode(str(planner_mode))
            if final_goal_xy is not None:
                agent.set_final_goal(final_goal_xy)

        return self.agents

    def agent(self, index: int) -> RobotAgent | None:
        idx = int(index)
        if idx < 0 or idx >= len(self.agents):
            return None
        return self.agents[idx]

    def set_final_goal_for_all(self, final_goal_xy: tuple[float, float]) -> None:
        for agent in self.agents:
            agent.set_final_goal(final_goal_xy)

    def set_planner_mode_for_all(self, planner_mode: str) -> None:
        for agent in self.agents:
            agent.set_planner_mode(str(planner_mode))

    def exploration_targets(self) -> list[tuple[float, float] | None]:
        return [agent.exploration_target_xy for agent in self.agents]

    def sync_exploration_targets_from_legacy_list(
        self,
        targets: list[tuple[float, float] | None],
    ) -> None:
        for idx, target in enumerate(targets):
            agent = self.agent(idx)
            if agent is None:
                continue
            if target is None:
                agent.exploration_target_xy = None
            else:
                agent.exploration_target_xy = (float(target[0]), float(target[1]))

    def sync_legacy_list_from_agents(self) -> list[tuple[float, float] | None]:
        return self.exploration_targets()
