"""Static, JSON-scenario-backed adapters that satisfy the existing
robotics_interfaces coordination contracts, plus the scenario loader itself.

This module builds everything a coordination plugin's assign(request) needs
from a fixed, deterministic JSON scenario -- no BeliefMap, no A*, no Qt, no
engine.py, no runtime services. It never re-implements frontier detection,
clustering, or path planning: components/targets come straight from the
scenario file, and travel cost is a plain euclidean distance (see
euclidean_distance()).

Only StaticFrontierProvider is implemented as an adapter class, because
algorithms.independent_baseline.plugin.IndependentBaselinePlugin is the only
plugin this benchmark drives so far, and it only ever reads
request.services.frontier_provider (via the robotics_interfaces.services.
FrontierProvider protocol) -- never frontier_information_service or
path_planning_service. Adding StaticFrontierInformationService/
StaticPlanningService now would be an abstraction with no real consumer;
see this module's test coverage and experiments/run_experiment.py's
docstring for the same note.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from robotics_interfaces import (
    CoordinationRequest,
    CoordinationServices,
    ExplorationCandidate,
    RobotCoordinationState,
    WorldSnapshot,
)

DEFAULT_DUPLICATE_TOLERANCE = 1e-6
DEFAULT_WORLD_RESOLUTION = 0.5
DEFAULT_WORLD_MARGIN = 1.0

_TOP_LEVEL_REQUIRED_FIELDS = (
    "experiment_id",
    "scenario_id",
    "algorithm",
    "seed",
    "robots",
    "frontier_components",
    "observed_obstacles",
    "current_targets",
    "invalidated_targets_by_robot",
    "parameters",
)

# Beyond the illustrative format in the task brief: RobotCoordinationState
# has no default for safety_radius/sensor_range/vision_model, so this
# scenario format requires them explicitly per robot rather than inventing
# a silent default (see this module's tests for the exact error raised when
# one is missing).
_ROBOT_REQUIRED_FIELDS = ("robot_id", "position", "heading", "radius", "sensor_range", "vision_model")
_COMPONENT_REQUIRED_FIELDS = ("cluster_id", "cells", "centroid", "viewpoints", "information_gain", "valid")


class ScenarioConfigError(ValueError):
    """Raised for any invalid/malformed static allocation scenario."""


# ---------------------------------------------------------------------------
# Scenario data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioRobot:
    robot_id: int
    position: tuple[float, float]
    heading: float
    radius: float
    sensor_range: float
    vision_model: str


@dataclass(frozen=True)
class ScenarioFrontierComponent:
    cluster_id: str
    cells: tuple[tuple[float, float], ...]
    centroid: tuple[float, float] | None
    viewpoints: tuple[tuple[float, float], ...]
    information_gain: float
    valid: bool


@dataclass(frozen=True)
class StaticScenario:
    """Normalized, validated scenario -- the single input this benchmark
    ever needs. robots is sorted by robot_id and frontier_components is
    sorted by cluster_id (see _normalize_robots()/_normalize_components()),
    so two JSON files that differ only in array order or object key order
    produce an identical StaticScenario."""

    experiment_id: str
    scenario_id: str
    algorithm: str
    seed: int
    robots: tuple[ScenarioRobot, ...]
    frontier_components: tuple[ScenarioFrontierComponent, ...]
    observed_obstacles: tuple[tuple[float, float], ...]
    current_targets: Mapping[int, tuple[float, float]]
    invalidated_targets_by_robot: Mapping[int, tuple[tuple[float, float], ...]]
    parameters: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Field-level parsing/validation helpers
# ---------------------------------------------------------------------------


def _require_keys(data: Mapping[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ScenarioConfigError(f"{context}: missing required field(s) {missing}")


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioConfigError(f"{field_name} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ScenarioConfigError(f"{field_name} must be finite, got {value!r}")
    return result


def _point(value: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ScenarioConfigError(f"{field_name} must be a 2-element [x, y] array, got {value!r}")
    x, y = value
    return (
        _finite_float(x, field_name=f"{field_name}[0]"),
        _finite_float(y, field_name=f"{field_name}[1]"),
    )


def _optional_point(value: Any, *, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    return _point(value, field_name=field_name)


def _points_tuple(value: Any, *, field_name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list):
        raise ScenarioConfigError(f"{field_name} must be a list, got {type(value).__name__}")
    return tuple(_point(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value))


def _robot_id_value(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioConfigError(f"{field_name} must be an int, got {value!r}")
    return value


def _non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ScenarioConfigError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


def _bool_value(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ScenarioConfigError(f"{field_name} must be a bool, got {value!r}")
    return value


def _int_value(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScenarioConfigError(f"{field_name} must be an int, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# robots[] / frontier_components[] parsing + ID normalization
# ---------------------------------------------------------------------------


def _parse_robot(data: Mapping[str, Any], *, index: int) -> ScenarioRobot:
    context = f"robots[{index}]"
    if not isinstance(data, Mapping):
        raise ScenarioConfigError(f"{context} must be an object, got {type(data).__name__}")
    _require_keys(data, _ROBOT_REQUIRED_FIELDS, context=context)
    return ScenarioRobot(
        robot_id=_robot_id_value(data["robot_id"], field_name=f"{context}.robot_id"),
        position=_point(data["position"], field_name=f"{context}.position"),
        heading=_finite_float(data["heading"], field_name=f"{context}.heading"),
        radius=_finite_float(data["radius"], field_name=f"{context}.radius"),
        sensor_range=_finite_float(data["sensor_range"], field_name=f"{context}.sensor_range"),
        vision_model=_non_empty_string(data["vision_model"], field_name=f"{context}.vision_model"),
    )


def _parse_component(data: Mapping[str, Any], *, index: int) -> ScenarioFrontierComponent:
    context = f"frontier_components[{index}]"
    if not isinstance(data, Mapping):
        raise ScenarioConfigError(f"{context} must be an object, got {type(data).__name__}")
    _require_keys(data, _COMPONENT_REQUIRED_FIELDS, context=context)
    return ScenarioFrontierComponent(
        cluster_id=_non_empty_string(data["cluster_id"], field_name=f"{context}.cluster_id"),
        cells=_points_tuple(data["cells"], field_name=f"{context}.cells"),
        centroid=_optional_point(data["centroid"], field_name=f"{context}.centroid"),
        viewpoints=_points_tuple(data["viewpoints"], field_name=f"{context}.viewpoints"),
        information_gain=_finite_float(data["information_gain"], field_name=f"{context}.information_gain"),
        valid=_bool_value(data["valid"], field_name=f"{context}.valid"),
    )


def _normalize_robots(raw_robots: Any) -> tuple[ScenarioRobot, ...]:
    if not isinstance(raw_robots, list):
        raise ScenarioConfigError(f"robots must be a list, got {type(raw_robots).__name__}")
    by_id: dict[int, ScenarioRobot] = {}
    for index, raw in enumerate(raw_robots):
        robot = _parse_robot(raw, index=index)
        if robot.robot_id in by_id:
            raise ScenarioConfigError(f"duplicate robot_id in robots[]: {robot.robot_id!r}")
        by_id[robot.robot_id] = robot
    # Normalize by robot_id -- see StaticScenario's docstring for why this
    # makes input array order irrelevant to the final scenario.
    return tuple(by_id[robot_id] for robot_id in sorted(by_id))


def _normalize_components(raw_components: Any) -> tuple[ScenarioFrontierComponent, ...]:
    if not isinstance(raw_components, list):
        raise ScenarioConfigError(f"frontier_components must be a list, got {type(raw_components).__name__}")
    by_id: dict[str, ScenarioFrontierComponent] = {}
    for index, raw in enumerate(raw_components):
        component = _parse_component(raw, index=index)
        if component.cluster_id in by_id:
            raise ScenarioConfigError(f"duplicate cluster_id in frontier_components[]: {component.cluster_id!r}")
        by_id[component.cluster_id] = component
    return tuple(by_id[cluster_id] for cluster_id in sorted(by_id))


def _parse_robot_id_key(key: Any, *, context: str) -> int:
    # JSON object keys are always strings -- this is the one place a
    # string robot_id is expected and converted back to int, not a case of
    # silently stringifying an id (see this module's docstring/rule 2 in
    # the task brief).
    try:
        return int(key)
    except (TypeError, ValueError) as exc:
        raise ScenarioConfigError(f"{context}: key {key!r} is not a valid robot_id") from exc


def _normalize_current_targets(
    raw: Any, *, known_robot_ids: set[int]
) -> dict[int, tuple[float, float]]:
    if not isinstance(raw, Mapping):
        raise ScenarioConfigError(f"current_targets must be an object, got {type(raw).__name__}")
    result: dict[int, tuple[float, float]] = {}
    for key, value in raw.items():
        robot_id = _parse_robot_id_key(key, context="current_targets")
        if robot_id not in known_robot_ids:
            raise ScenarioConfigError(f"current_targets references unknown robot_id: {robot_id}")
        result[robot_id] = _point(value, field_name=f"current_targets[{key!r}]")
    return result


def _normalize_invalidated_targets(
    raw: Any, *, known_robot_ids: set[int]
) -> dict[int, tuple[tuple[float, float], ...]]:
    if not isinstance(raw, Mapping):
        raise ScenarioConfigError(f"invalidated_targets_by_robot must be an object, got {type(raw).__name__}")
    result: dict[int, tuple[tuple[float, float], ...]] = {}
    for key, value in raw.items():
        robot_id = _parse_robot_id_key(key, context="invalidated_targets_by_robot")
        if robot_id not in known_robot_ids:
            raise ScenarioConfigError(f"invalidated_targets_by_robot references unknown robot_id: {robot_id}")
        result[robot_id] = _points_tuple(value, field_name=f"invalidated_targets_by_robot[{key!r}]")
    return result


def scenario_from_dict(data: Any) -> StaticScenario:
    """Validate and normalize a raw JSON-decoded scenario dict.

    Deterministic and order-independent: robots/frontier_components arrays
    are normalized by id (see _normalize_robots()/_normalize_components()),
    and every other field is read by explicit key, never by dict iteration
    order -- so key order and array order in the source JSON never affect
    the result (rules 17-19 in the task brief).
    """
    if not isinstance(data, Mapping):
        raise ScenarioConfigError(f"scenario root must be a JSON object, got {type(data).__name__}")
    _require_keys(data, _TOP_LEVEL_REQUIRED_FIELDS, context="scenario")

    robots = _normalize_robots(data["robots"])
    components = _normalize_components(data["frontier_components"])
    known_robot_ids = {robot.robot_id for robot in robots}

    observed_obstacles = _points_tuple(data["observed_obstacles"], field_name="observed_obstacles")
    current_targets = _normalize_current_targets(data["current_targets"], known_robot_ids=known_robot_ids)
    invalidated_targets_by_robot = _normalize_invalidated_targets(
        data["invalidated_targets_by_robot"], known_robot_ids=known_robot_ids
    )

    parameters = data["parameters"]
    if not isinstance(parameters, Mapping):
        raise ScenarioConfigError(f"parameters must be an object, got {type(parameters).__name__}")

    return StaticScenario(
        experiment_id=_non_empty_string(data["experiment_id"], field_name="experiment_id"),
        scenario_id=_non_empty_string(data["scenario_id"], field_name="scenario_id"),
        algorithm=_non_empty_string(data["algorithm"], field_name="algorithm"),
        seed=_int_value(data["seed"], field_name="seed"),
        robots=robots,
        frontier_components=components,
        observed_obstacles=observed_obstacles,
        current_targets=current_targets,
        invalidated_targets_by_robot=invalidated_targets_by_robot,
        parameters=dict(parameters),
    )


# ---------------------------------------------------------------------------
# StaticFrontierProvider -- robotics_interfaces.services.FrontierProvider
# ---------------------------------------------------------------------------


def euclidean_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _select_component_target(component: ScenarioFrontierComponent) -> tuple[float, float] | None:
    """Target selection rule (fixed, documented in the task brief):
    1. first viewpoint, in scenario order, if any;
    2. else centroid, if present;
    3. else no candidate for this component."""
    if component.viewpoints:
        return component.viewpoints[0]
    if component.centroid is not None:
        return component.centroid
    return None


class StaticFrontierProvider:
    """FrontierProvider backed by a fixed set of ScenarioFrontierComponent.

    Implements only candidates_for_robot(), the one FrontierProvider method
    IndependentBaselinePlugin actually calls (see this module's docstring).
    Invalid components (valid=False) and components with no derivable
    target never produce a candidate. blocked_targets (sourced from the
    scenario's invalidated_targets_by_robot via CoordinationRequest.
    blocked_targets_by_robot) are excluded using the same euclidean
    duplicate_tolerance used for this benchmark's duplicate_target_count
    metric, so both use one documented notion of "same target".
    """

    def __init__(
        self,
        components: tuple[ScenarioFrontierComponent, ...],
        *,
        duplicate_tolerance: float = DEFAULT_DUPLICATE_TOLERANCE,
    ) -> None:
        self._components = tuple(components)
        self._duplicate_tolerance = max(0.0, float(duplicate_tolerance))

    def candidates_for_robot(
        self,
        robot: RobotCoordinationState,
        world: WorldSnapshot,
        blocked_targets: tuple[tuple[float, float], ...] = (),
    ) -> tuple[ExplorationCandidate, ...]:
        candidates: list[ExplorationCandidate] = []
        for component in self._components:
            if not component.valid:
                continue
            target = _select_component_target(component)
            if target is None:
                continue
            if self._is_blocked(target, blocked_targets):
                continue
            candidates.append(
                ExplorationCandidate(
                    target=target,
                    source="static_frontier_component",
                    information_gain=component.information_gain,
                    travel_cost=euclidean_distance(robot.xy, target),
                    metadata={
                        "cluster_id": component.cluster_id,
                        "size": len(component.cells),
                        "source": "static_frontier_component",
                    },
                )
            )
        return tuple(candidates)

    def _is_blocked(
        self, target: tuple[float, float], blocked_targets: tuple[tuple[float, float], ...]
    ) -> bool:
        return any(euclidean_distance(target, blocked) <= self._duplicate_tolerance for blocked in blocked_targets)


# ---------------------------------------------------------------------------
# CoordinationRequest construction
# ---------------------------------------------------------------------------


def build_robot_states(scenario: StaticScenario) -> tuple[RobotCoordinationState, ...]:
    """robots are already sorted by robot_id in StaticScenario, so this
    tuple's order is stable regardless of the source JSON's array order."""
    return tuple(
        RobotCoordinationState(
            robot_id=robot.robot_id,
            xy=robot.position,
            safety_radius=robot.radius,
            sensor_range=robot.sensor_range,
            vision_model=robot.vision_model,
            theta=robot.heading,
            current_target=scenario.current_targets.get(robot.robot_id),
            is_active=True,
            metadata={},
        )
        for robot in scenario.robots
    )


def _bounds_from_points(
    points: Iterable[tuple[float, float]], *, margin: float
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if not xs:
        return (-1.0, 1.0, -1.0, 1.0)
    return (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)


def build_world_snapshot(
    scenario: StaticScenario,
    *,
    resolution: float = DEFAULT_WORLD_RESOLUTION,
    margin: float = DEFAULT_WORLD_MARGIN,
) -> WorldSnapshot:
    """A WorldSnapshot is required by IndependentBaselinePlugin as a gate
    before it will call frontier_provider.candidates_for_robot() at all
    (see algorithms/independent_baseline/plugin.py's _candidate_pool()) --
    StaticFrontierProvider itself never reads from it. bounds is a loose
    bounding box over every point the scenario defines, purely so the
    snapshot is a faithful (not fabricated) description of the scenario."""
    points: list[tuple[float, float]] = [robot.position for robot in scenario.robots]
    for component in scenario.frontier_components:
        points.extend(component.cells)
        if component.centroid is not None:
            points.append(component.centroid)
        points.extend(component.viewpoints)
    points.extend(scenario.observed_obstacles)

    return WorldSnapshot(
        explored_points=(),
        mapped_obstacle_points=scenario.observed_obstacles,
        bounds=_bounds_from_points(points, margin=margin),
        resolution=resolution,
        final_goal_xy=None,
        metadata={},
    )


def build_coordination_request(
    scenario: StaticScenario, *, frontier_provider: StaticFrontierProvider
) -> CoordinationRequest:
    """Build a CoordinationRequest using only public robotics_interfaces
    contracts. robots_to_assign is left empty so IndependentBaselinePlugin's
    own default (every is_active robot) applies -- this benchmark is a
    one-shot full-team static allocation, not an incremental replan."""
    return CoordinationRequest(
        robot_states=build_robot_states(scenario),
        robots_to_assign=(),
        world=build_world_snapshot(scenario),
        proposals_by_robot={},
        existing_targets_by_robot=dict(scenario.current_targets),
        blocked_targets_by_robot=dict(scenario.invalidated_targets_by_robot),
        route_points_by_robot=(),
        services=CoordinationServices(frontier_provider=frontier_provider),
        parameters=dict(scenario.parameters),
        shared={},
        time_s=0.0,
    )
