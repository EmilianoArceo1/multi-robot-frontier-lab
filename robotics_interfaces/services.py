from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, TYPE_CHECKING, runtime_checkable

from robotics_interfaces.frontiers import FrontierCluster
from robotics_interfaces.observations import Point2D, RobotCoordinationState, WorldSnapshot
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_interfaces.regions import CoveragePath, RegionTask
from robotics_interfaces.results import (
    CollisionCheckResult,
    MapQuerySnapshot,
    MetricsEvent,
    PathPlanningRequest,
    PathPlanningResponse,
)

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


@runtime_checkable
class PathPlanningService(Protocol):
    """Plans a path between two points without exposing the simulator's
    concrete planner (A*/Direct/Dijkstra/...) to the algorithm."""

    def plan_path(self, request: PathPlanningRequest) -> PathPlanningResponse:
        ...


@runtime_checkable
class CollisionCheckingService(Protocol):
    """Validates a candidate path against the environment and/or the team.

    path[0] is the start position; the remaining entries are waypoints, same
    convention as robotics_interfaces.commands.RobotCommand.path.
    """

    def is_path_safe(
        self,
        path: Sequence[Point2D],
        robot_id: int,
        safety_radius: float,
    ) -> CollisionCheckResult:
        ...


@runtime_checkable
class MapQueryService(Protocol):
    """Read-only access to the shared map for algorithms that only need a
    snapshot instead of building/updating WorldSnapshot themselves."""

    def map_snapshot(self) -> MapQuerySnapshot:
        ...


@runtime_checkable
class MetricsService(Protocol):
    """Lets an algorithm report a named event without knowing how/where the
    host records it (console log, file, dashboard, ...)."""

    def record_event(self, event: MetricsEvent) -> None:
        ...


@runtime_checkable
class FrontierInformationService(Protocol):
    """Exposes frontier clusters/viewpoints for FUEL/RACER-style algorithms,
    as an alternative to the plain FrontierProvider/TeamFrontierProvider
    candidate lists above."""

    def get_frontier_clusters(
        self,
        robot_id: int | None = None,
    ) -> tuple[FrontierCluster, ...]:
        ...


@runtime_checkable
class RegionDecompositionService(Protocol):
    """Exposes a region-level breakdown of unknown space, for algorithms that
    allocate by region/task instead of by single frontier target."""

    def get_region_tasks(self) -> tuple[RegionTask, ...]:
        ...


@runtime_checkable
class CoveragePathService(Protocol):
    """Plans a conceptual (non-trajectory) waypoint route covering the given
    regions for one robot."""

    def plan_coverage_path(
        self,
        robot_id: int,
        regions: tuple[RegionTask, ...],
    ) -> CoveragePath:
        ...


@dataclass(frozen=True)
class CoordinationServices:
    """Optional service bundle injected by the simulator host.

    Implementations live on the simulator side. External algorithms depend only
    on these protocols, not on robotics_sim, Qt, engine.py, or canvas objects.

    Prefer team_frontier_provider for multi-robot algorithms. frontier_provider
    is kept as a simple fallback and for single-robot compatibility. Every
    field here is optional and may be None if the host has not wired it up
    yet -- algorithms must treat every field here as optional.
    """

    frontier_provider: FrontierProvider | None = None
    team_frontier_provider: TeamFrontierProvider | None = None
    path_planning_service: PathPlanningService | None = None
    collision_checking_service: CollisionCheckingService | None = None
    map_query_service: MapQueryService | None = None
    metrics_service: MetricsService | None = None
    frontier_information_service: FrontierInformationService | None = None
    region_decomposition_service: RegionDecompositionService | None = None
    coverage_path_service: CoveragePathService | None = None
