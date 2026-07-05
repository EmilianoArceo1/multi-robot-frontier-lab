from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, TYPE_CHECKING, runtime_checkable

from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.proposals import ExplorationCandidate

if TYPE_CHECKING:
    from robotics_interfaces.coordination import CoordinationRequest


@runtime_checkable
class FrontierProvider(Protocol):
    """Provides exploration candidates for one robot without exposing simulator internals.

    This remains useful as a fallback for simple algorithms. Team algorithms
    should prefer TeamFrontierProvider because frontier generation/allocation is
    usually a synchronized team decision.
    """

    def candidates_for_robot(
        self,
        robot: RobotCoordinationState,
        world: WorldSnapshot,
        blocked_targets: tuple[tuple[float, float], ...] = (),
    ) -> tuple[ExplorationCandidate, ...]:
        ...


@runtime_checkable
class TeamFrontierProvider(Protocol):
    """Provides exploration candidates for the whole team in one synchronized pass."""

    def candidates_for_team(
        self,
        request: "CoordinationRequest",
    ) -> Mapping[int, tuple[ExplorationCandidate, ...]]:
        ...


@dataclass(frozen=True)
class CoordinationServices:
    """Optional service bundle injected by the simulator host.

    Implementations live on the simulator side. External algorithms depend only
    on these protocols, not on robotics_sim, Qt, engine.py, or canvas objects.

    Prefer team_frontier_provider for multi-robot algorithms. frontier_provider
    is kept as a simple fallback and for single-robot compatibility.
    """

    frontier_provider: FrontierProvider | None = None
    team_frontier_provider: TeamFrontierProvider | None = None
