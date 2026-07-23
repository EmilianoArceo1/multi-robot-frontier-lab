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

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from robotics_interfaces import (
    CoordinationAssignment,
    CoordinationRequest as PluginCoordinationRequest,
    CoordinationResult as PluginCoordinationResult,
    CoordinationServices,
    PluginRuntimeProfile,
    RobotCommand,
    RobotCoordinationState as PluginRobotCoordinationState,
    WorldSnapshot,
    build_runtime_profile,
)
from robotics_sim.planning.coordinated_frontier_planner import assign_frontier_viewpoints
from robotics_sim.simulation.coordination_services import (
    RuntimeFrontierProvider,
    RuntimeTeamFrontierProvider,
)
from robotics_sim.simulation.runtime_services import (
    RuntimeCollisionCheckingService,
    RuntimeFrontierInformationService,
    RuntimeMapQueryService,
    RuntimeMetricsService,
    RuntimePathPlanningService,
)
from robotics_sim.simulation.plugin_loader import (
    PluginLoadError,
    list_coordination_plugin_names,
    load_coordination_plugin,
)

NOIC_COORDINATOR = "NOIC information coordinator"
DEFAULT_COORDINATOR = NOIC_COORDINATOR

_LOGGER = logging.getLogger(__name__)


def _discover_coordinator_options() -> list[str]:
    """Return dynamically discoverable coordinator plugin names."""

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
    debug: dict[str, Any] = field(default_factory=dict)
    commands: tuple[RobotCommand, ...] = ()
    assignments: tuple[CoordinationAssignment, ...] = ()


def available_coordinator_options() -> list[str]:
    """Return coordinator options from the dynamic plugin loader."""

    return _discover_coordinator_options()


def runtime_profile_for_strategy(strategy: str) -> PluginRuntimeProfile:
    """Look up a plugin's runtime profile by name, without a full coordinator.

    The GUI uses this to compute compute_gui_control_policy() straight from
    the currently selected combo value, before any simulation frame/request is
    built. This intentionally does not build CoordinationServices, since it
    only needs metadata.capabilities.
    """

    available = available_coordinator_options()
    selected = strategy if strategy in available else DEFAULT_COORDINATOR
    plugin = load_coordination_plugin(selected)
    return build_runtime_profile(plugin.metadata)


class MultiRobotCoordinator:
    """Simulator-facing coordination host.

    This class is not a coordination algorithm. It is an adapter/host that:
        1. receives legacy arguments from engine.py;
        2. converts them into robotics_interfaces.CoordinationRequest;
        3. injects simulator-side services such as RuntimeFrontierProvider;
        4. calls the selected plugin;
        5. adapts the plugin result back to the legacy CoordinationResult shape.
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
        self.runtime_profile: PluginRuntimeProfile = build_runtime_profile(self.plugin.metadata)
        _LOGGER.info(
            "Algorithm runtime profile: plugin=%s owns_target_generation=%s "
            "owns_task_allocation=%s owns_path_planning=%s owns_control=%s "
            "uses_legacy_frontier_service=%s uses_external_path_planner=%s "
            "uses_external_motion_controller=%s",
            self.plugin.metadata.name,
            self.runtime_profile.owns_target_generation,
            self.runtime_profile.owns_task_allocation,
            self.runtime_profile.owns_path_planning,
            self.runtime_profile.owns_control,
            self.runtime_profile.uses_legacy_frontier_service,
            self.runtime_profile.uses_external_path_planner,
            self.runtime_profile.uses_external_motion_controller,
        )

    def selected_plugin_profile(self) -> PluginRuntimeProfile:
        """Return what the currently selected plugin actually controls."""
        return self.runtime_profile

    def plugin_owns_target_generation(self) -> bool:
        return self.runtime_profile.owns_target_generation

    def plugin_owns_path_planning(self) -> bool:
        return self.runtime_profile.owns_path_planning

    def plugin_owns_control(self) -> bool:
        return self.runtime_profile.owns_control

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
        goal_tolerance: float = 0.25,
        coordination_parameters: Mapping[str, Any] | None = None,
        mapping_architecture: str = "centralized",
        time_s: float = 0.0,
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
            goal_tolerance=goal_tolerance,
            coordination_parameters=coordination_parameters,
            mapping_architecture=mapping_architecture,
            time_s=time_s,
        )
        if self.runtime_profile.owns_target_generation:
            _LOGGER.debug(
                "Exploration source: %s; legacy frontier service (fallback only): %r",
                self.plugin.metadata.name,
                planner_name,
            )
        else:
            _LOGGER.debug("Exploration planner: %r", planner_name)
        plugin_result = self.plugin.assign(request)
        result = self._adapt_plugin_result(plugin_result, robot_count=len(robot_states))
        _LOGGER.debug(
            "coordination result: plugin=%s selected_targets=%s debug=%s",
            self.plugin.metadata.name,
            result.targets,
            result.debug,
        )
        _LOGGER.debug(
            "Plugin command runtime: commands_received=%d owns_path_planning=%s owns_control=%s",
            len(result.commands),
            self.runtime_profile.owns_path_planning,
            self.runtime_profile.owns_control,
        )
        return result

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
        goal_tolerance: float,
        coordination_parameters: Mapping[str, Any] | None = None,
        mapping_architecture: str = "centralized",
        time_s: float = 0.0,
    ) -> PluginCoordinationRequest:
        normalized_existing_targets = tuple(
            _normalize_target(target)
            for target in existing_targets
        )
        normalized_explored_points = tuple(
            point
            for value in explored_points
            if (point := _normalize_target(value)) is not None
        )
        normalized_obstacle_points = tuple(
            point
            for value in mapped_obstacle_points
            if (point := _normalize_target(value)) is not None
        )
        normalized_bounds = tuple(float(value) for value in bounds)
        normalized_goal = _normalize_target(final_goal_xy)
        normalized_route_points = tuple(
            tuple(
                point
                for value in route
                if (point := _normalize_target(value)) is not None
            )
            for route in (route_points_by_robot or [])
        )
        normalized_explored_by_robot = tuple(
            tuple(
                point
                for value in points
                if (point := _normalize_target(value)) is not None
            )
            for points in (explored_points_by_robot or [])
        )

        # A candidate target closer than this to the assigning robot produces a
        # near-zero-length route.  Two grid cells or two goal tolerances is the
        # smallest travel that still clears the goal-reached radius; this
        # mirrors engine.multi_frontier_exclusion_radius() so plugin-side
        # rejection and engine-side post-hoc validation agree on scale.
        min_frontier_travel_distance = max(
            2.0 * float(goal_tolerance),
            2.0 * float(resolution),
            0.75,
        )
        default_safety_radius = max(
            (float(state.safety_radius) for state in robot_states),
            default=0.35,
        )
        default_sensor_range = max(
            (float(state.sensor_range) for state in robot_states),
            default=2.5,
        )

        plugin_robot_states = tuple(
            PluginRobotCoordinationState(
                robot_id=index,
                xy=(float(state.xy[0]), float(state.xy[1])),
                safety_radius=float(state.safety_radius),
                sensor_range=float(state.sensor_range),
                vision_model=str(state.vision_model),
                theta=float(state.theta),
                current_target=(
                    normalized_existing_targets[index]
                    if index < len(normalized_existing_targets)
                    else None
                ),
                is_active=True,
            )
            for index, state in enumerate(robot_states)
        )

        blocked_targets_by_robot = {
            index: tuple(
                point
                for target in targets
                if (point := _normalize_target(target)) is not None
            )
            for index, targets in enumerate(invalidated_targets_by_robot or [])
        }

        world = WorldSnapshot(
            explored_points=normalized_explored_points,
            mapped_obstacle_points=normalized_obstacle_points,
            bounds=normalized_bounds,  # type: ignore[arg-type]
            resolution=float(resolution),
            final_goal_xy=normalized_goal,
            metadata={
                "planner_name": planner_name,
                "mapping_architecture": str(mapping_architecture),
            },
        )

        other_robot_disks_by_id = {
            state.robot_id: (float(state.xy[0]), float(state.xy[1]), float(state.safety_radius))
            for state in plugin_robot_states
        }
        other_routes_by_id = {
            index: normalized_route_points[index] if index < len(normalized_route_points) else ()
            for index in range(len(plugin_robot_states))
        }

        services = CoordinationServices(
            frontier_provider=RuntimeFrontierProvider(
                ipp_distance_penalty=float(ipp_distance_penalty),
                target_exclusion_radius=float(target_exclusion_radius),
                dynamic_obstacle_margin=float(dynamic_obstacle_margin),
                route_points_by_robot=normalized_route_points,
                explored_points_by_robot=normalized_explored_by_robot,
            ),
            team_frontier_provider=RuntimeTeamFrontierProvider(
                ipp_distance_penalty=float(ipp_distance_penalty),
                target_exclusion_radius=float(target_exclusion_radius),
                dynamic_obstacle_margin=float(dynamic_obstacle_margin),
            ),
            path_planning_service=RuntimePathPlanningService(),
            collision_checking_service=RuntimeCollisionCheckingService(
                other_robot_disks_by_id=other_robot_disks_by_id,
                other_routes_by_id=other_routes_by_id,
                margin=float(dynamic_obstacle_margin),
            ),
            map_query_service=RuntimeMapQueryService(
                explored_points=normalized_explored_points,
                mapped_obstacle_points=normalized_obstacle_points,
                bounds=normalized_bounds,
                resolution=float(resolution),
            ),
            metrics_service=RuntimeMetricsService(),
            frontier_information_service=RuntimeFrontierInformationService(
                explored_points=normalized_explored_points,
                mapped_obstacle_points=normalized_obstacle_points,
                bounds=normalized_bounds,
                resolution=float(resolution),
                robot_radius=default_safety_radius,
                sensor_range=default_sensor_range,
            ),
        )

        shared: dict[str, Any] = {
            "planner_name": planner_name,
            "explored_points": normalized_explored_points,
            "mapped_obstacle_points": normalized_obstacle_points,
            "bounds": normalized_bounds,
            "resolution": float(resolution),
            "final_goal_xy": normalized_goal or (0.0, 0.0),
            "ipp_distance_penalty": float(ipp_distance_penalty),
            "target_exclusion_radius": float(target_exclusion_radius),
            "dynamic_obstacle_margin": float(dynamic_obstacle_margin),
            "explored_points_by_robot": normalized_explored_by_robot,
            "mapping_architecture": str(mapping_architecture),
            # Legacy service injected by the simulator host. This keeps the
            # global_noic_legacy plugin outside robotics_sim while allowing it
            # to reuse the existing coordinated frontier planner during migration.
            "legacy_assign_frontier_viewpoints": assign_frontier_viewpoints,
        }

        algorithm_parameters = {
            str(key): value for key, value in dict(coordination_parameters or {}).items()
        }

        return PluginCoordinationRequest(
            robot_states=plugin_robot_states,
            robots_to_assign=tuple(int(index) for index in robots_to_assign),
            world=world,
            services=services,
            existing_targets_by_robot={
                index: target
                for index, target in enumerate(normalized_existing_targets)
            },
            blocked_targets_by_robot=blocked_targets_by_robot,
            route_points_by_robot=normalized_route_points,
            parameters={
                **algorithm_parameters,
                "planner_name": planner_name,
                "ipp_distance_penalty": float(ipp_distance_penalty),
                "target_exclusion_radius": float(target_exclusion_radius),
                "dynamic_obstacle_margin": float(dynamic_obstacle_margin),
                "reservation_resolution": max(float(resolution), 1e-6),
                "grid_resolution": float(resolution),
                "goal_tolerance": float(goal_tolerance),
                "safety_radius": default_safety_radius,
                "min_frontier_travel_distance": min_frontier_travel_distance,
            },
            shared=shared,
            time_s=float(time_s),
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
            debug=dict(result.debug),
            commands=tuple(result.commands),
            assignments=tuple(result.assignments),
        )


def _normalize_target(target: object) -> tuple[float, float] | None:
    if target is None:
        return None
    try:
        x, y = target  # type: ignore[misc]
        return (float(x), float(y))
    except (TypeError, ValueError):
        return None


def map_robot_commands_by_id(commands: Iterable[RobotCommand]) -> dict[int, RobotCommand]:
    """Index plugin-returned RobotCommand entries by robot_id.

    Not every robot is guaranteed a command (a plugin may only return
    commands for robots it just assigned); callers must use .get(robot_id)
    and keep a legacy fallback for missing entries.
    """

    return {int(command.robot_id): command for command in commands}


def select_runtime_path_source(
    profile: PluginRuntimeProfile,
    command: RobotCommand | None,
    legacy_path_provider: Callable[[], tuple[bool, str, list[tuple[float, float]]]],
) -> tuple[bool, str, list[tuple[float, float]]]:
    """Decide whether a robot's route comes from the plugin or the external planner.

    legacy_path_provider is only called when it is actually needed: the
    plugin does not own PATH_PLANNING, or it owns it but did not supply a
    usable command.path. This is what keeps a PATH_PLANNING-owning plugin
    from paying for (or being overridden by) an A*/Direct call it does not
    need.
    """

    if profile.owns_path_planning and command is not None and command.path:
        return True, "plugin path (PATH_PLANNING owned)", list(command.path)

    success, reason, waypoints = legacy_path_provider()
    if profile.owns_path_planning:
        reason = f"plugin owns PATH_PLANNING but command.path missing; using external planner fallback; {reason}"
    return success, reason, waypoints


def select_runtime_control_source(
    profile: PluginRuntimeProfile,
    command: RobotCommand | None,
    legacy_control: Any,
) -> tuple[Any, str]:
    """Decide the proposed control for one robot this frame.

    The caller must still run its own safety veto on whatever this returns --
    this helper only chooses the proposed control, it never bypasses safety.
    legacy_control is computed unconditionally by the caller (unlike path
    planning, running the nominal controller also advances the robot's state
    machine, so it cannot simply be skipped when a plugin owns CONTROL).
    """

    if not profile.owns_control:
        return legacy_control, "nominal control (plugin does not own CONTROL)"

    if command is not None and command.control_xy is not None:
        reason = "plugin control (CONTROL owned)"
        if command.reason:
            reason = f"{reason}; {command.reason}"
        return command.control_xy, reason

    return legacy_control, "plugin owns CONTROL but command.control_xy missing; using nominal control fallback"
