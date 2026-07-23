"""Mapping-architecture contract selected by the task-assignment algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robotics_sim.environment.belief_map import BeliefMap


class MappingArchitecture(str, Enum):
    CENTRALIZED = "centralized"
    DECENTRALIZED_SLAM = "decentralized_slam"


CENTRALIZED_ARCHITECTURE_LABEL = "Centralized architecture"
DECENTRALIZED_ARCHITECTURE_LABEL = "SLAM / decentralized architecture"

_DECENTRALIZED_TASK_ASSIGNERS = frozenset(
    {
        "Travel-time Voronoi + CQLite distributed Q-learning",
        # Backward compatibility with scenarios saved before the cited
        # coordinator was restored to the selector.
        "CQLite distributed Q-learning",
    }
)


def architecture_for_task_assignment(name: str) -> MappingArchitecture:
    # Import locally to keep the low-level map store independent during module
    # initialization while allowing UI/runtime taxonomy to be extended by
    # adding a paper profile.
    from robotics_sim.simulation.approach_profiles import (
        approach_profile_for_task_assignment,
    )

    profile = approach_profile_for_task_assignment(name)
    if profile is not None:
        return profile.mapping_architecture
    if str(name) in _DECENTRALIZED_TASK_ASSIGNERS:
        return MappingArchitecture.DECENTRALIZED_SLAM
    return MappingArchitecture.CENTRALIZED


def architecture_label(architecture: MappingArchitecture) -> str:
    if architecture is MappingArchitecture.DECENTRALIZED_SLAM:
        return DECENTRALIZED_ARCHITECTURE_LABEL
    return CENTRALIZED_ARCHITECTURE_LABEL


@dataclass
class BeliefMapArchitectureStore:
    """Own a compatibility team map and, when requested, one map per robot.

    The team map remains the rendering/metrics projection. Algorithms running
    under decentralized SLAM obtain maps only through ``map_for_robot``.
    """

    architecture: MappingArchitecture
    team_map: BeliefMap
    robot_maps: tuple[BeliefMap, ...]

    @classmethod
    def create(
        cls,
        *,
        architecture: MappingArchitecture,
        bounds: tuple[float, float, float, float],
        resolution: float,
        robot_count: int,
        initial_revision: int = 0,
    ) -> "BeliefMapArchitectureStore":
        count = max(1, int(robot_count))
        team = BeliefMap(
            bounds=bounds,
            resolution=resolution,
            robot_count=count,
            initial_revision=initial_revision,
        )
        if architecture is MappingArchitecture.DECENTRALIZED_SLAM:
            local = tuple(
                BeliefMap(
                    bounds=bounds,
                    resolution=resolution,
                    robot_count=1,
                    initial_revision=initial_revision,
                )
                for _ in range(count)
            )
        else:
            local = tuple(team for _ in range(count))
        return cls(architecture=architecture, team_map=team, robot_maps=local)

    @property
    def decentralized(self) -> bool:
        return self.architecture is MappingArchitecture.DECENTRALIZED_SLAM

    def map_for_robot(self, robot_index: int) -> BeliefMap:
        index = int(robot_index)
        if not 0 <= index < len(self.robot_maps):
            raise IndexError(f"robot map index {index} is out of range")
        return self.robot_maps[index]
