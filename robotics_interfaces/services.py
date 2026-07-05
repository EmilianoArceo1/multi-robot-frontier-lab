from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.proposals import ExplorationCandidate


@runtime_checkable
class FrontierProvider(Protocol):
    """Provides exploration candidates without exposing simulator internals."""

    def candidates_for_robot(
        self,
        robot: RobotCoordinationState,
        world: WorldSnapshot,
        blocked_targets: tuple[tuple[float, float], ...] = (),
    ) -> tuple[ExplorationCandidate, ...]:
        ...


@dataclass(frozen=True)
class CoordinationServices:
    """Optional service bundle injected by the simulator host.

    Plugins may use these services when explicit proposals are not enough.
    The service implementations live in the simulator side; the interfaces
    live here so algorithms remain independent from robotics_sim.
    """

    frontier_provider: FrontierProvider | None = None
