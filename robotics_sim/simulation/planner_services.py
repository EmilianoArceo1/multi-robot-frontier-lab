"""
PlannerServices — thin façade over planning, selection, and future policies.

Injected into RobotAgent.step() so the agent can request plans and pick
exploration targets without importing engine internals, Qt, canvas, or concrete
robot physics.

Current responsibilities:
    plan_path()                 — synchronous A*/Dijkstra/Direct call.
    select_exploration_target() — frontier / informative target selection.

Forward-compatible responsibilities:
    The Protocols below define internal contracts for target selection, path
    planning, team coordination, control policies, and runtime parameter patch
    validation. They prepare the existing simulator for plugin-like algorithms
    without creating new top-level packages yet.

What does NOT live here:
    - Async worker management (PlannerWorker stays in engine.py for now).
    - BeliefMap construction or obstacle mapping.
    - Qt signals or canvas updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Callable, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_sim.environment.occupancy_grid import OccupancyGrid

# Lazy imports so the module can be loaded even when planning packages are
# absent (e.g., lightweight unit tests that only test the decision layer).
try:
    from robotics_sim.planning.planner_registry import compute_planned_waypoints as _cpw
except ImportError:
    _cpw = None  # type: ignore[assignment]

try:
    from robotics_sim.planning.exploration_planners import (
        detect_frontier_cells as _detect_frontier_cells,
        detect_frontier_cells_for_planner as _detect_frontier_cells_for_planner,
        exploration_planner_requires_clustering as _requires_clustering,
        select_exploration_goal as _seg,
    )
except ImportError:
    _seg = None  # type: ignore[assignment]
    _detect_frontier_cells = None  # type: ignore[assignment]
    _detect_frontier_cells_for_planner = None  # type: ignore[assignment]
    _requires_clustering = None  # type: ignore[assignment]

try:
    from robotics_sim.planning.frontier_clustering import (
        cluster_frontier_cells as _cluster_frontier_cells,
    )
except ImportError:
    _cluster_frontier_cells = None  # type: ignore[assignment]

try:
    from robotics_sim.planning.ryu_frontier_graph_bfs import RYU_FRONTIER_GRAPH_BFS
except ImportError:
    RYU_FRONTIER_GRAPH_BFS = "Ryu frontier-graph BFS exploration"


Point2D = tuple[float, float]


# ---------------------------------------------------------------------------
# Internal policy contracts. These are intentionally small and structural.
# Implementations may live in existing robotics_sim modules first; later they
# can be moved to external packages without changing RobotAgent.
# ---------------------------------------------------------------------------


class PathPlannerProtocol(Protocol):
    def plan_path(self, request: "PathPlanningRequest") -> "PathPlanningResponse":
        ...


class TargetSelectorProtocol(Protocol):
    def select_target(self, request: "TargetSelectionRequest") -> Any:
        ...


class CoordinationPolicyProtocol(Protocol):
    def assign(self, context: Any) -> Any:
        ...


class ControlPolicyProtocol(Protocol):
    def compute_control(self, context: Any) -> Any:
        ...


class ParameterPatchValidatorProtocol(Protocol):
    def validate_parameter_patch(self, patch: Any) -> "ParameterPatchValidation":
        ...


@dataclass(frozen=True)
class PathPlanningRequest:
    planner_type: str
    path_simplifier: str
    start_xy: Point2D
    goal_xy: Point2D
    planning_grid: "OccupancyGrid | None"
    robot_radius: float
    bounds: tuple[float, float, float, float]
    resolution: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PathPlanningResponse:
    success: bool
    reason: str
    waypoints: tuple[Point2D, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSelectionRequest:
    planner_name: str
    belief_map: Any
    robot_xy: Point2D
    robot_heading: float
    current_target: Point2D | None
    final_goal_xy: Point2D | None
    robot_radius: float
    sensor_range: float
    vision_model: str
    ipp_distance_penalty: float
    clustering_algorithm: str | None = None
    excluded_targets: tuple[Point2D, ...] = ()
    reserved_targets: tuple[Point2D, ...] = ()
    target_exclusion_radius: float = 0.0
    is_candidate_reachable: "Callable[[Point2D], bool] | None" = None
    planning_grid_provider: "Callable[[], OccupancyGrid] | None" = None
    known_hazards: tuple[Point2D, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParameterPatchValidation:
    success: bool
    reason: str
    normalized_patch: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Sentinel used when exploration planners are unavailable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FailedResult:
    reason: str = "exploration planner package not available"
    success: bool = False
    target: Point2D | None = None
    candidates: tuple = ()


# ---------------------------------------------------------------------------


class PlannerServices:
    """
    Provides planning services to RobotAgent.

    A single instance is created by the engine and passed into every
    agent.step() call, and is otherwise safe to share.

    is_candidate_reachable is the one piece of mutable state: an optional
    Callable[[Point2D], bool] the engine host may refresh each tick (see
    engine.ensure_planner_services()) so select_exploration_target() can
    reject exploration candidates the real navigation A* would immediately
    fail on, without exploration_planners.py importing engine internals.
    Left as None, behavior is unchanged from before this existed.

    planning_grid_provider follows the exact same pattern: an optional
    Callable[[], OccupancyGrid] the engine host may refresh each tick (see
    engine.ensure_planner_services()) so a planner that actually needs a
    real planning grid (today, only FoVAwareHazardFrontierPlanner) can
    build one lazily -- called at most once per select_goal() invocation,
    and never called at all by planners that never read it. This is
    deliberately a PROVIDER (a callable), not a pre-built grid: building an
    OccupancyGrid is not free, and most ticks never need one (target
    selection may not even run every tick, and non-FoV planners never
    consume it), so nothing here should build one until a planner that
    actually wants one asks for it. Left as None, behavior is unchanged
    from before this existed.
    """

    allowed_parameter_prefixes = (
        "robot.",
        "sensor.",
        "planner.",
        "mapping.",
        "coordination.",
        "communication.",
        "simulation.",
        "algorithm.",
    )

    is_candidate_reachable: "Callable[[Point2D], bool] | None" = None
    planning_grid_provider: "Callable[[], OccupancyGrid] | None" = None
    # Refreshed by the engine from RuntimeHazardService.discovered_sources().
    # These are discovered source centres, never omniscient ground truth.
    known_hazards: tuple[Point2D, ...] = ()
    # The engine always writes the configured selection here. None is kept
    # only for old direct unit callers that predate the explicit stage.
    clustering_algorithm: str | None = None

    # ------------------------------------------------------------------ path

    def plan_path(
        self,
        *,
        planner_type: str,
        path_simplifier: str,
        start_xy: Point2D,
        goal_xy: Point2D,
        planning_grid: "OccupancyGrid | None",
        robot_radius: float,
        bounds: tuple[float, float, float, float],
        resolution: float,
    ) -> tuple[bool, str, list[Point2D]]:
        """
        Synchronous A*/Dijkstra/Direct path planning.

        Returns (success, reason, waypoints).
        Waypoints are world-coordinate (x, y) tuples.
        """
        response = self.plan_path_request(
            PathPlanningRequest(
                planner_type=planner_type,
                path_simplifier=path_simplifier,
                start_xy=start_xy,
                goal_xy=goal_xy,
                planning_grid=planning_grid,
                robot_radius=robot_radius,
                bounds=bounds,
                resolution=resolution,
            )
        )
        return response.success, response.reason, list(response.waypoints)

    def plan_path_request(self, request: PathPlanningRequest) -> PathPlanningResponse:
        """Request-object variant used by future algorithm hosts."""
        if request.planner_type == "Direct":
            return PathPlanningResponse(
                success=True,
                reason="direct route",
                waypoints=(request.goal_xy,),
            )

        if _cpw is None:
            return PathPlanningResponse(False, "planner package not available")

        kwargs: dict[str, Any] = dict(
            planner_type=request.planner_type,
            start_xy=request.start_xy,
            goal_xy=request.goal_xy,
            obstacles=[],
            bounds=request.bounds,
            resolution=request.resolution,
            robot_radius=request.robot_radius,
            planning_grid=request.planning_grid,
            unknown_is_traversable=True,
            obstacle_points=[],
        )

        try:
            has_simplifier = "path_simplifier" in inspect.signature(_cpw).parameters
        except (TypeError, ValueError):
            has_simplifier = False

        try:
            if has_simplifier:
                ok, reason, waypoints = _cpw(
                    **kwargs,
                    path_simplifier=request.path_simplifier,
                )
            else:
                ok, reason, waypoints = _cpw(**kwargs)
        except Exception as exc:
            return PathPlanningResponse(False, f"planner error: {exc}")

        normalized = tuple((float(x), float(y)) for x, y in (waypoints or []))
        return PathPlanningResponse(bool(ok), str(reason), normalized)

    # ------------------------------------------------------------------ exploration

    def select_exploration_target(
        self,
        *,
        planner_name: str,
        belief_map,
        robot_xy: Point2D,
        robot_heading: float,
        current_target,
        final_goal_xy: Point2D | None,
        robot_radius: float,
        sensor_range: float,
        vision_model: str,
        ipp_distance_penalty: float,
        clustering_algorithm: str | None = None,
        excluded_targets: list[Point2D] | None = None,
        reserved_targets: list[Point2D] | tuple[Point2D, ...] | None = None,
        target_exclusion_radius: float | None = None,
        is_candidate_reachable: "Callable[[Point2D], bool] | None" = None,
        planning_grid_provider: "Callable[[], OccupancyGrid] | None" = None,
        known_hazards: list[Point2D] | tuple[Point2D, ...] | None = None,
    ):
        """
        Select the next exploration frontier target.

        Returns an ExplorationPlannerResult-like object with:
            .success  bool
            .target   tuple[float, float] | None
            .reason   str

        On failure returns a sentinel object with .success = False.

        is_candidate_reachable: optional per-call override. When omitted
        (the common case -- e.g. ExplorationBehavior._pick_next_target()
        never passes it), falls back to self.is_candidate_reachable, which
        the engine host may refresh every tick with a check backed by the
        real navigation planning grid.

        planning_grid_provider: optional per-call override, same fallback
        pattern as is_candidate_reachable -- when omitted, falls back to
        self.planning_grid_provider. Forwarded to select_exploration_goal()
        as-is; THIS METHOD NEVER CALLS IT. Only a planner that actually
        wants a real planning grid (today, only
        FoVAwareHazardFrontierPlanner) invokes it, at most once, deep
        inside select_goal().
        """
        excluded = tuple(excluded_targets or ())
        reserved = tuple(reserved_targets or ())
        exclusion_radius = (
            float(target_exclusion_radius)
            if target_exclusion_radius is not None
            else (max(float(robot_radius) * 2.0, 0.0) if excluded else 0.0)
        )
        reachability_check = (
            is_candidate_reachable if is_candidate_reachable is not None else self.is_candidate_reachable
        )
        grid_provider = (
            planning_grid_provider if planning_grid_provider is not None else self.planning_grid_provider
        )
        discovered_hazards = tuple(
            known_hazards if known_hazards is not None else self.known_hazards
        )

        return self.select_exploration_target_request(
            TargetSelectionRequest(
                planner_name=planner_name,
                belief_map=belief_map,
                robot_xy=robot_xy,
                robot_heading=robot_heading,
                current_target=current_target,
                final_goal_xy=final_goal_xy,
                robot_radius=robot_radius,
                sensor_range=sensor_range,
                vision_model=vision_model,
                ipp_distance_penalty=ipp_distance_penalty,
                clustering_algorithm=(
                    clustering_algorithm
                    if clustering_algorithm is not None
                    else self.clustering_algorithm
                ),
                excluded_targets=excluded,
                reserved_targets=reserved,
                target_exclusion_radius=exclusion_radius,
                is_candidate_reachable=reachability_check,
                planning_grid_provider=grid_provider,
                known_hazards=discovered_hazards,
            )
        )

    def select_exploration_target_request(self, request: TargetSelectionRequest):
        """Request-object variant used by future target-generation policies."""
        if _seg is None:
            return _FailedResult()

        frontier_clusters = None
        has_explicit_clustering_stage = request.clustering_algorithm is not None
        if (
            has_explicit_clustering_stage
            and callable(_requires_clustering)
            and _requires_clustering(request.planner_name)
        ):
            if not callable(_detect_frontier_cells) or not callable(_cluster_frontier_cells):
                return _FailedResult("frontier clustering service is not available")
            detector = (
                _detect_frontier_cells_for_planner
                if callable(_detect_frontier_cells_for_planner)
                else lambda _name, *, belief, robot_xy: _detect_frontier_cells(belief)
            )
            clustering = _cluster_frontier_cells(
                request.clustering_algorithm,
                belief_map=request.belief_map,
                frontier_cells=detector(
                    request.planner_name,
                    belief=request.belief_map,
                    robot_xy=request.robot_xy,
                ),
            )
            bfs_fallback = request.planner_name == RYU_FRONTIER_GRAPH_BFS
            if not clustering.success and not bfs_fallback:
                return _FailedResult(f"clustering stage rejected selection: {clustering.reason}")
            if clustering.success and clustering.clusters:
                frontier_clusters = clustering.clusters
            elif bfs_fallback:
                frontier_clusters = None
                clustering_fallback_reason = (
                    clustering.reason
                    if not clustering.success
                    else f"{request.clustering_algorithm} produced no frontier clusters"
                )

        kwargs = dict(
            belief_map=request.belief_map,
            robot_xy=request.robot_xy,
            robot_heading=request.robot_heading,
            current_target=request.current_target,
            final_goal_xy=request.final_goal_xy,
            robot_count=1,
            robot_radius=request.robot_radius,
            sensor_range=request.sensor_range,
            vision_model=request.vision_model,
            ipp_distance_penalty=request.ipp_distance_penalty,
            excluded_targets=list(request.excluded_targets),
            reserved_targets=list(request.reserved_targets),
            target_exclusion_radius=request.target_exclusion_radius,
            is_candidate_reachable=request.is_candidate_reachable,
            planning_grid_provider=request.planning_grid_provider,
            known_hazards=list(request.known_hazards),
        )
        if has_explicit_clustering_stage and frontier_clusters is not None:
            kwargs["clustering_algorithm"] = request.clustering_algorithm
            kwargs["frontier_clusters"] = frontier_clusters
        elif has_explicit_clustering_stage and request.planner_name == RYU_FRONTIER_GRAPH_BFS:
            kwargs["clustering_fallback_reason"] = clustering_fallback_reason

        return _seg(
            request.planner_name,
            **kwargs,
        )

    # ------------------------------------------------------------------ coordination / control hooks

    def assign_targets(self, *, coordinator: CoordinationPolicyProtocol, context: Any) -> Any:
        """Call a coordination policy without exposing RobotAgent to it."""
        return coordinator.assign(context)

    def compute_control(self, *, policy: ControlPolicyProtocol, context: Any) -> Any:
        """Call a control-level policy without exposing RobotAgent to it."""
        return policy.compute_control(context)

    # ------------------------------------------------------------------ parameter patches

    def validate_parameter_patch(self, patch: Any) -> ParameterPatchValidation:
        """Validate a future runtime parameter patch request.

        This does not apply the patch. The engine/runtime host remains the only
        component allowed to mutate simulator state.
        """
        normalized = self._normalize_parameter_patch(patch)
        if normalized is None:
            return ParameterPatchValidation(False, "invalid patch format")

        path = str(normalized.get("parameter_path", ""))
        if not path:
            return ParameterPatchValidation(False, "missing parameter_path")

        if not path.startswith(self.allowed_parameter_prefixes):
            return ParameterPatchValidation(
                False,
                f"parameter_path must start with one of {self.allowed_parameter_prefixes}",
            )

        if "value" not in normalized:
            return ParameterPatchValidation(False, "missing value")

        return ParameterPatchValidation(True, "valid parameter patch", normalized)

    @staticmethod
    def _normalize_parameter_patch(patch: Any) -> dict[str, Any] | None:
        if patch is None:
            return None

        if isinstance(patch, Mapping):
            data = dict(patch)
        else:
            data = {
                "scope": getattr(patch, "scope", None),
                "parameter_path": getattr(
                    patch,
                    "parameter_path",
                    getattr(patch, "path", None),
                ),
                "value": getattr(patch, "value", None),
                "robot_id": getattr(patch, "robot_id", None),
                "reason": getattr(patch, "reason", ""),
            }

        # Accept either fully-qualified parameter_path or scope + local path.
        path = data.get("parameter_path")
        scope = data.get("scope")
        if path and scope and not str(path).startswith(f"{scope}."):
            data["parameter_path"] = f"{scope}.{path}"

        return data
