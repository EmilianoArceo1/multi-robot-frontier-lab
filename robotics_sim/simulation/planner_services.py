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
from typing import Any, Mapping, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_sim.environment.occupancy_grid import OccupancyGrid

# Lazy imports so the module can be loaded even when planning packages are
# absent (e.g., lightweight unit tests that only test the decision layer).
try:
    from robotics_sim.planning.planner_registry import compute_planned_waypoints as _cpw
except ImportError:
    _cpw = None  # type: ignore[assignment]

try:
    from robotics_sim.planning.exploration_planners import select_exploration_goal as _seg
except ImportError:
    _seg = None  # type: ignore[assignment]


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
    excluded_targets: tuple[Point2D, ...] = ()
    target_exclusion_radius: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParameterPatchValidation:
    success: bool
    reason: str
    normalized_patch: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Sentinel used when exploration planners are unavailable.
# ---------------------------------------------------------------------------


class _FailedResult:
    success = False
    target = None
    reason = "exploration planner package not available"
    candidates: tuple = ()


# ---------------------------------------------------------------------------


class PlannerServices:
    """
    Provides planning services to RobotAgent.

    A single instance is created by the engine and passed into every
    agent.step() call. It is intentionally stateless and safe to share.
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
        excluded_targets: list[Point2D] | None = None,
        target_exclusion_radius: float | None = None,
    ):
        """
        Select the next exploration frontier target.

        Returns an ExplorationPlannerResult-like object with:
            .success  bool
            .target   tuple[float, float] | None
            .reason   str

        On failure returns a sentinel object with .success = False.
        """
        excluded = tuple(excluded_targets or ())
        exclusion_radius = (
            float(target_exclusion_radius)
            if target_exclusion_radius is not None
            else (max(float(robot_radius) * 2.0, 0.0) if excluded else 0.0)
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
                excluded_targets=excluded,
                target_exclusion_radius=exclusion_radius,
            )
        )

    def select_exploration_target_request(self, request: TargetSelectionRequest):
        """Request-object variant used by future target-generation policies."""
        if _seg is None:
            return _FailedResult()

        return _seg(
            request.planner_name,
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
            target_exclusion_radius=request.target_exclusion_radius,
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
