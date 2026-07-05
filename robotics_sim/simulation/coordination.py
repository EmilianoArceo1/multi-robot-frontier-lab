"""
Multi-robot coordination host for frontier target assignment.

This module is intentionally independent from Qt and from the visual canvas.
It keeps the public simulator-facing API used by engine.py, but delegates the
actual coordination strategy to dynamically discovered plugins under:

    algorithms/<plugin_name>/plugin.py

The simulator remains the host/bank of tests. Algorithms are interchangeable
plugins that receive plain snapshots and return plain coordination decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robotics_interfaces import (
    CoordinationRequest as PluginCoordinationRequest,
    CoordinationResult as PluginCoordinationResult,
    RobotCoordinationState as PluginRobotCoordinationState,
)
from robotics_sim.planning.coordinated_frontier_planner import assign_frontier_viewpoints
from robotics_sim.simulation.plugin_loader import (
    PluginLoadError,
    list_coordination_plugin_names,
    load_coordination_plugin,
)

NOIC_COORDINATOR = "NOIC information coordinator"
DEFAULT_COORDINATOR = NOIC_COORDINATOR


def _discover_coordinator_options() -> list[str]:
    """Return dynamically discoverable coordinator plugin names.

    Kept as a function so config/UI can still import COORDINATOR_OPTIONS while
    the actual source of truth is the plugin loader, not a hardcoded registry in
    this module.
    """

    try:
        names = list(list_coordination_plugin_names())
    except PluginLoadError:
        names = []

    if DEFAULT_COORDINATOR not in names:
        # Import-time fallback for partially checked out trees. If the plugin is
        # missing, MultiRobotCoordinator will still raise a clear error when it
        # tries to load it.
        names.insert(0, DEFAULT_COORDINATOR)

    return names


COORDINATOR_OPTIONS = _discover_coordinator_options()


@dataclass(frozen=True)
class RobotCoordinationState:
    """Plain robot state packet expected by the current simulator engine.

    This is the legacy simulator-facing type.  MultiRobotCoordinator converts it
    to robotics_interfaces.RobotCoordinationState before calling a plugin.
    """

    xy: tuple[float, float]
    safety_radius: float
    sensor_range: float
    vision_model: str
    theta: float = 0.0


@dataclass(frozen=True)
class CoordinationResult:
    """Result returned to engine.py after adapting a plugin decision."""

    targets: tuple[tuple[float, float] | None, ...]
    reasons: tuple[str, ...]
    strategy: str


def available_coordinator_options() -> list[str]:
    """Return coordinator options from the dynamic plugin loader."""

    return _discover_coordinator_options()


class MultiRobotCoordinator:
    """Simulator-facing coordination host.

    This class is not a coordination algorithm. It is an adapter/host that:
        1. receives legacy arguments from engine.py;
        2. converts them into robotics_interfaces.CoordinationRequest;
        3. calls the selected plugin;
        4. adapts the plugin result back to the legacy CoordinationResult shape.
    """

    def __init__(self, strategy: str = DEFAULT_COORDINATOR):
        available = available_coordinator_options()
        selected = strategy if strategy in available else DEFAULT_COORDINATOR

        try:
            self.plugin = load_coordination_plugin(selected)
        except PluginLoadError as exc:
            available_text = ", ".join(available) or "<none>"
            raise PluginLoadError(
                f"Could not load coordination plugin {selected!r}. Available: {available_text}"
            ) from exc

        self.strategy = self.plugin.metadata.name

    def assign_frontiers(
        self,
        *,
        planner_name: str,
        robot_states: list[RobotCoordinationState],
        existing_targets: list[tuple[float, float] | None],
        robots_to_assign: list[int],
        invalidated_targets_by_robot: list[list[tuple[float, float]]] | None,
        explored_points: list[tuple[float, float]],
        mapped_obstacle_points: list[tuple[float, float]],
        bounds: tuple[float, float, float, float],
        resolution: float,
        final_goal_xy: tuple[float, float] | None = None,
        ipp_distance_penalty: float = 0.5,
        target_exclusion_radius: float = 1.5,
        dynamic_obstacle_margin: float = 0.5,
        route_points_by_robot: list[list[tuple[float, float]]] | None = None,
        explored_points_by_robot: list[list[tuple[float, float]]] | None = None,
    ) -> CoordinationResult:
        request = self._build_plugin_request(
            planner_name=planner_name,
            robot_states=robot_states,
            existing_targets=existing_targets,
            robots_to_assign=robots_to_assign,
            invalidated_targets_by_robot=invalidated_targets_by_robot,
            explored_points=explored_points,
            mapped_obstacle_points=mapped_obstacle_points,
            bounds=bounds,
            resolution=resolution,
            final_goal_xy=final_goal_xy,
            ipp_distance_penalty=ipp_distance_penalty,
            target_exclusion_radius=target_exclusion_radius,
            dynamic_obstacle_margin=dynamic_obstacle_margin,
            route_points_by_robot=route_points_by_robot,
            explored_points_by_robot=explored_points_by_robot,
        )
        plugin_result = self.plugin.assign(request)
        return self._adapt_plugin_result(plugin_result, robot_count=len(robot_states))

    def _build_plugin_request(
        self,
        *,
        planner_name: str,
        robot_states: list[RobotCoordinationState],
        existing_targets: list[tuple[float, float] | None],
        robots_to_assign: list[int],
        invalidated_targets_by_robot: list[list[tuple[float, float]]] | None,
        explored_points: list[tuple[float, float]],
        mapped_obstacle_points: list[tuple[float, float]],
        bounds: tuple[float, float, float, float],
        resolution: float,
        final_goal_xy: tuple[float, float] | None,
        ipp_distance_penalty: float,
        target_exclusion_radius: float,
        dynamic_obstacle_margin: float,
        route_points_by_robot: list[list[tuple[float, float]]] | None,
        explored_points_by_robot: list[list[tuple[float, float]]] | None,
    ) -> PluginCoordinationRequest:
        plugin_robot_states = tuple(
            PluginRobotCoordinationState(
                robot_id=index,
                xy=(float(state.xy[0]), float(state.xy[1])),
                safety_radius=float(state.safety_radius),
                sensor_range=float(state.sensor_range),
                vision_model=str(state.vision_model),
                theta=float(state.theta),
                current_target=(
                    _normalize_target(existing_targets[index])
                    if index < len(existing_targets)
                    else None
                ),
                is_active=True,
            )
            for index, state in enumerate(robot_states)
        )

        blocked_targets_by_robot = {
            index: tuple(_normalize_target(target) for target in targets if _normalize_target(target) is not None)
            for index, targets in enumerate(invalidated_targets_by_robot or [])
        }

        route_points = tuple(
            tuple(_normalize_target(point) for point in route if _normalize_target(point) is not None)
            for route in (route_points_by_robot or [])
        )

        shared: dict[str, Any] = {
            "planner_name": planner_name,
            "explored_points": tuple(_normalize_target(point) for point in explored_points if _normalize_target(point) is not None),
            "mapped_obstacle_points": tuple(_normalize_target(point) for point in mapped_obstacle_points if _normalize_target(point) is not None),
            "bounds": tuple(float(value) for value in bounds),
            "resolution": float(resolution),
            "final_goal_xy": _normalize_target(final_goal_xy) or (0.0, 0.0),
            "ipp_distance_penalty": float(ipp_distance_penalty),
            "target_exclusion_radius": float(target_exclusion_radius),
            "dynamic_obstacle_margin": float(dynamic_obstacle_margin),
            "explored_points_by_robot": tuple(
                tuple(_normalize_target(point) for point in points if _normalize_target(point) is not None)
                for points in (explored_points_by_robot or [])
            ),
            # Legacy service injected by the simulator host. This keeps the
            # global_noic_legacy plugin outside robotics_sim while allowing it
            # to reuse the existing coordinated frontier planner during migration.
            "legacy_assign_frontier_viewpoints": assign_frontier_viewpoints,
        }

        return PluginCoordinationRequest(
            robot_states=plugin_robot_states,
            robots_to_assign=tuple(int(index) for index in robots_to_assign),
            existing_targets_by_robot={
                index: _normalize_target(target)
                for index, target in enumerate(existing_targets)
            },
            blocked_targets_by_robot=blocked_targets_by_robot,
            route_points_by_robot=route_points,
            shared=shared,
        )

    def _adapt_plugin_result(
        self,
        result: PluginCoordinationResult,
        *,
        robot_count: int,
    ) -> CoordinationResult:
        targets = list(result.targets[:robot_count])
        reasons = list(result.reasons[:robot_count])

        while len(targets) < robot_count:
            targets.append(None)
        while len(reasons) < robot_count:
            reasons.append(f"{result.strategy}: no decision returned")

        return CoordinationResult(
            targets=tuple(targets),
            reasons=tuple(reasons),
            strategy=result.strategy,
        )


def _normalize_target(target: object) -> tuple[float, float] | None:
    if target is None:
        return None
    try:
        x, y = target  # type: ignore[misc]
        return (float(x), float(y))
    except (TypeError, ValueError):
        return None


__all__ = [
    "COORDINATOR_OPTIONS",
    "DEFAULT_COORDINATOR",
    "NOIC_COORDINATOR",
    "CoordinationResult",
    "MultiRobotCoordinator",
    "RobotCoordinationState",
    "available_coordinator_options",
]
