"""
Shared simulation data, constants, geometry helpers, and scenario I/O.

This module intentionally contains no QWidget classes. It is the first file
to read when you want to understand the simulator state model, world bounds,
obstacle representation, sensor ray-casting helpers, and .sim serialization.
"""

from __future__ import annotations

import json
import math
import os
import copy
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtGui import QColor
from robotics_sim.control.wang_ames_barrier_certificate import (
    SAFETY_ALGORITHM_OPTIONS,
    WANG_AMES_BARRIER_CERTIFICATE,
)

try:
    from robotics_sim.planning.path_simplifier import (
        PATH_SIMPLIFIER_OPTIONS,
        DEFAULT_PATH_SIMPLIFIER,
    )
except ImportError:
    PATH_SIMPLIFIER_OPTIONS = [
        "Raw grid path",
        "Direction changes",
        "Direction changes + spacing",
        "RDP grid-safe",
        "Line of sight grid-safe",
    ]
    DEFAULT_PATH_SIMPLIFIER = "Direction changes"

try:
    from robotics_sim.planning.exploration_planners import (
        EXPLORATION_PLANNER_OPTIONS,
        DEFAULT_EXPLORATION_PLANNER,
        select_exploration_goal,
    )
except ImportError:
    EXPLORATION_PLANNER_OPTIONS = [
        "Goal seeking",
        "Nearest frontier",
        "Largest frontier",
        "Utility frontier",
        "Informative frontier / IPP-lite",
    ]
    DEFAULT_EXPLORATION_PLANNER = "Goal seeking"

    def select_exploration_goal(planner_name: str, **kwargs):
        class _FallbackResult:
            success = False
            target = kwargs.get("robot_xy", (0.0, 0.0))
            reason = "exploration planner package is not available; holding current position"
            candidates = ()

        return _FallbackResult()


try:
    from robotics_sim.planning.frontier_clustering import (
        CLUSTERING_ALGORITHM_OPTIONS,
        NO_CLUSTERING_ALGORITHM,
    )
except ImportError:
    CLUSTERING_ALGORITHM_OPTIONS = ()
    NO_CLUSTERING_ALGORITHM = "No clustering algorithm available"


try:
    from robotics_sim.simulation.coordination import (
        COORDINATOR_OPTIONS,
        DEFAULT_COORDINATOR,
    )
except ImportError:
    COORDINATOR_OPTIONS = [
        "Independent frontiers",
        "Reserved frontiers",
        "Synchronized greedy",
    ]
    DEFAULT_COORDINATOR = "Synchronized greedy"

# The configuration UI now exposes the two pipeline responsibilities by their
# actual roles instead of mixing frontier selection with task allocation.
# Keep the legacy implementations importable for paper-reproduction tests and
# archived experiment runners, but do not advertise them as selectable runtime
# options in the rebuilt pipeline.
REMOVED_FRONTIER_ALGORITHM_DETECTOR_OPTIONS = frozenset(
    {
        "Nav2D nearest-frontier wavefront",
        "Nearest frontier",
        "Largest frontier",
        "Utility frontier",
        "Informative frontier / IPP-lite",
    }
)
FRONTIER_ALGORITHM_DETECTOR_OPTIONS = tuple(
    option
    for option in EXPLORATION_PLANNER_OPTIONS
    if option not in REMOVED_FRONTIER_ALGORITHM_DETECTOR_OPTIONS
)

REMOVED_TASK_ASSIGN_ALGORITHM_OPTIONS = frozenset(
    {
        "CQLite distributed Q-learning",
        "FUEL frontier baseline coordinator",
        "Independent baseline coordinator",
        "MMPF explore coordinator",
        "NOIC information coordinator",
        "Nav2D multi-wavefront coordinator",
    }
)
TASK_ASSIGN_ALGORITHM_OPTIONS = tuple(
    option
    for option in COORDINATOR_OPTIONS
    if option in {
        "Frontier cluster Hungarian coordinator",
        "Travel-time Voronoi + CQLite distributed Q-learning",
        "MARVEL CTDE graph-attention policy",
        "MARVEL CTDE graph-attention policy (scaled environment)",
    }
)
from robotics_sim.simulation.algorithm_pipeline_profiles import (
    task_assignment_pipeline_profile,
)
NO_TASK_ASSIGN_ALGORITHM = "No task assign algorithm available"

# ============================================================

MAROON = "#500000"
MAROON_DARK = "#3A0000"
MAROON_SOFT = "#7A1E24"

BG = "#F4F5F7"
CARD = "#FFFFFF"
PANEL_CARD = "#FDFDFC"

TEXT = "#22252A"
TEXT_MUTED = "#777B84"
TEXT_FAINT = "#A5A9B2"

BORDER = "#DADFE7"
BORDER_SOFT = "#E9ECF1"

BLUE = "#236FCF"
BLUE_DARK = "#164491"
BLUE_LIGHT = "#DCEEFF"

GREEN = "#219653"
GREEN_DARK = "#48612A"
GREEN_LIGHT = "#E0F8E8"

RED = "#DC3434"
YELLOW = "#EFB229"
ORANGE = "#E17E26"

GRID = QColor(224, 228, 235)
GRID_AXIS = QColor(172, 181, 194)
OBSTACLE_FILL = QColor(211, 212, 216)
OBSTACLE_STROKE = QColor(88, 88, 92)

ROBOT_COLOR_HEXES = [
    "#236FCF",  # blue
    "#E17E26",  # orange
    "#219653",  # green
    "#9B51E0",  # purple
    "#DC3434",  # red
    "#00A3A3",  # teal
    "#B7791F",  # amber
    "#2D3748",  # slate
]


def robot_color(index: int) -> QColor:
    return QColor(ROBOT_COLOR_HEXES[int(index) % len(ROBOT_COLOR_HEXES)])


def camera_viewport_bounds(
    center_x: float, center_y: float, width: float, height: float
) -> tuple[float, float, float, float]:
    """Left, right, bottom, top world bounds of a camera_center_x/y +
    camera_width/height rectangle.

    The one canonical, render-independent formula for SimulationConfig's
    four camera_* fields -- the LOGICAL viewport / exploration-metric ROI,
    never the canvas's render-only aspect-ratio-fit viewport (see
    SimulationCanvas.render_view_bounds_world()). Kept here, not
    duplicated, so a consumer that only has a SimulationConfig (e.g.
    engine.py's exploration-coverage metric) does not need a
    SimulationCanvas instance to compute the same rectangle
    SimulationCanvas.camera_bounds_world() draws as the editable frame.
    """
    width = max(0.50, float(width))
    height = max(0.50, float(height))
    return (
        float(center_x) - width / 2.0,
        float(center_x) + width / 2.0,
        float(center_y) - height / 2.0,
        float(center_y) + height / 2.0,
    )


# ============================================================
# LAYOUT
# ============================================================

SIDE_PANEL_WIDTH = 520
WINDOW_TARGET_WIDTH = 1440
WINDOW_TARGET_HEIGHT = 900

WORLD_X_MIN = -10.0
WORLD_X_MAX = 10.0
WORLD_Y_MIN = -8.0
WORLD_Y_MAX = 8.0

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TAMU_IMAGE_CANDIDATES = [
    os.path.join(BASE_DIR, "robotics_sim", "assets", "tamu_building.jpg"),
    os.path.join(BASE_DIR, "robotics_sim", "assets", "tamu_building.png"),
    os.path.join(BASE_DIR, "assets", "tamu_building.jpg"),
    os.path.join(BASE_DIR, "assets", "tamu_building.png"),
    os.path.join(BASE_DIR, "tamu_building.jpg"),
    os.path.join(BASE_DIR, "tamu_building.png"),
]


# ============================================================
# DATA
# ============================================================

SIM_FILE_SCHEMA = "robotics_sim_lab.sim"
SIM_FILE_VERSION = 1

PLANNER_OPTIONS = [
    "Direct",
    "A*",
    "Dijkstra",
    "RRT (future)",
]

VISION_OPTIONS = [
    "LiDAR",
    "Camera / FoV",
    "Omnidirectional",
]

MAP_VISUALIZATION_OPTIONS = [
    "Current",
    "Monochrome Discovery",
    "Custom Discovery",
]
DEFAULT_MAP_VISUALIZATION = MAP_VISUALIZATION_OPTIONS[0]
DEFAULT_CUSTOM_UNEXPLORED_COLOR = "#000000"
DEFAULT_CUSTOM_EXPLORED_COLOR = "#F8F9FB"
DEFAULT_CUSTOM_OBSTACLE_COLOR = "#6A6D75"
DEFAULT_CUSTOM_EXPLORED_OPACITY = 1.0
DEFAULT_MAPPED_OBSTACLE_LINE_WIDTH = 0.5

ROBOT_ICON_OPTIONS = [
    "Circle",
    "Drone",
    "Wheeled Robot",
]
DEFAULT_ROBOT_ICON = ROBOT_ICON_OPTIONS[0]

DEFAULT_OBSTACLES: list[tuple[float, float, float, float]] = [
    (-7.0, 5.4, 1.0, 1.0),
    (-7.5, -6.4, 3.3, 1.2),
    (3.0, 4.6, 3.0, 1.0),
    (0.0, 2.6, 0.9, 0.9),
    (6.2, 1.4, 0.9, 0.9),
    (5.4, -4.0, 1.0, 1.0),
]

# A rectangle is considered fully discovered once most of its sampled boundary
# has been observed. This affects visualization only; planning still uses the
# actual mapped points.
OBSTACLE_COMPLETE_COVERAGE = 0.90

# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

TARGET_FRAME_MS = 16
SENSOR_UPDATE_PERIOD_SEC = 0.10
MIN_SENSOR_UPDATE_DISTANCE = 0.05
MIN_SENSOR_UPDATE_ROTATION = 0.10
# Navigation Reasoning stores compressed sensor/belief frames.  Ten samples
# per simulated second are enough for a readable replay and prevent Multiple
# mode from serializing one complete debug frame per robot at the GUI rate.
NAVIGATION_DEBUG_TICK_PERIOD_SEC = 0.10

# Ray counts intentionally differ by purpose. The current sensor footprint is
# visual, so it can use more rays. The explored-area cache and physics-loop
# sensor update are cheaper and throttled.
SENSOR_DRAW_RAYS_OMNI = 121
SENSOR_DRAW_RAYS_CAMERA = 61
EXPLORED_RAYS_OMNI = 72
EXPLORED_RAYS_CAMERA = 45

# Visual caches should not be rebuilt for every single new sensor point.
# Obstacle opacity is feedback only; it is not used by collision or planning.
OBSTACLE_VISUAL_REFRESH_POINT_STEP = 80

# Executed robot trajectories are retained for the complete simulation run.
# SimulationCanvas rasterizes only newly appended segments into persistent
# pixmaps, so render cost remains effectively constant as the history grows.
# Restarting the simulation replaces these lists and therefore clears the
# visual trail together with the rest of the runtime state.

# Keep only a short world-space history for explored polygons. The real explored
# area is already rasterized into a homogeneous pixmap cache.
EXPLORED_POLYGON_HISTORY_LIMIT = 40

# Cache the current blue sensor footprint for tiny pose changes. This reduces
# paintEvent ray-casting pressure while preserving the throttled mapping loop.
SENSOR_DRAW_RECOMPUTE_DISTANCE = 0.04
SENSOR_DRAW_RECOMPUTE_ROTATION = 0.06

# Spatial buckets are intentionally coarse. They are not used as a source of
# truth; they only reduce how many obstacles each ray-cast checks.
SPATIAL_BUCKET_SIZE = 2.5



@dataclass
class RobotStartConfig:
    """Editable per-robot configuration used in multi-robot mode.

    The final mission goal remains shared by the team. When Same Configuration
    is ON, these per-robot fields are overwritten from the global robot and
    dynamics controls except for x/y, so robots keep independent start poses.
    When Same Configuration is OFF, each robot can have its own pose, sensing,
    physical clearance, and dynamic limits.
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    vision: float = 2.5
    body_radius: float = 0.20
    safety_radius: float = 0.35
    max_speed: float = 1.2
    max_acceleration: float = 2.0
    max_angular_speed: float = 2.5
    goal_tolerance: float = 0.25
    acceleration_gain: float = 0.75


def default_robot_start_configs() -> list[RobotStartConfig]:
    """Stable default starts for previewing multi-robot mode."""
    return [
        RobotStartConfig(-1.0, -0.6, 0.0, 0.0),
        RobotStartConfig(0.0, 0.0, 0.0, 0.0),
        RobotStartConfig(-1.0, 0.6, 0.0, 0.0),
    ]


@dataclass
class SimulationConfig:
    """
    Application-level scenario configuration.

    This object is intentionally not a robot model. It describes the initial
    conditions, map, planner selection, and sensor selection needed to reproduce
    a simulation from a .sim file.
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    vision: float = 2.5
    camera_fov_degrees: float = 70.0

    # Physical robot body. This is the actual visual/physical size of the robot.
    body_radius: float = 0.20

    # Safety clearance radius r. It must never be smaller than body_radius.
    # Planning and collision checking use this radius to keep distance from obstacles.
    safety_radius: float = 0.35

    goal_x: float = 8.0
    goal_y: float = 6.0

    max_speed: float = 1.2
    max_acceleration: float = 2.0
    max_angular_speed: float = 2.5
    goal_tolerance: float = 0.25

    # Controller acceleration gain k_a. Higher values correct velocity faster;
    # lower values make acceleration/braking smoother.
    acceleration_gain: float = 0.75

    planner_type: str = "Direct"
    path_simplifier: str = DEFAULT_PATH_SIMPLIFIER

    # Exploration planner selects the next target. The path planner still
    # computes how to reach that target. Goal seeking preserves the old behavior.
    exploration_planner: str = DEFAULT_EXPLORATION_PLANNER

    # Explicit frontier-cell grouping stage. There is intentionally no
    # connected-component default: selectable implementations must be
    # registered with paper provenance in planning/frontier_clustering.py.
    clustering_algorithm: str = NO_CLUSTERING_ALGORITHM

    # Multi-robot frontier coordination strategy. This is only used in
    # Multiple Robot Mode when an exploration planner is active.
    coordinator_type: str = DEFAULT_COORDINATOR
    safety_algorithm: str = WANG_AMES_BARRIER_CERTIFICATE

    # Algorithm-specific coordinator settings.  These are intentionally a
    # plain JSON object so paper replicas can pin their published parameters
    # without adding one-off GUI controls for every external algorithm.
    coordination_parameters: dict = field(default_factory=dict)

    # Periodic full-team replanning interval, in simulated seconds. 0.0 (the
    # default) disables it entirely, preserving today's purely event-driven
    # coordination (missing/reached/invalidated targets only). See
    # robotics_sim.simulation.coordination_scheduler.
    coordination_replan_interval_s: float = 0.0

    # When True, coordination_service_audit.py raises a contract error
    # instead of only warning when a plugin's observed service usage
    # contradicts its declared CandidateInputMode. Intended for tests/
    # experiments, not default interactive use.
    coordination_strict_contracts: bool = False

    # Minimum simulated time between exploration-target replans.
    # Safety replans caused by newly discovered obstacles can still happen immediately.
    exploration_replan_cooldown: float = 1.00

    # IPP-lite distance penalty lambda. Higher values prefer closer frontiers;
    # lower values allow longer travel when expected information gain is high.
    ipp_distance_penalty: float = 0.20

    vision_model: str = "LiDAR"
    agent_mode: str = "Single Robot Mode"
    grid_resolution: float = 0.5

    # Rendering-only presentation choices. They never affect belief-map,
    # planning, coordination, or robot dynamics.
    map_visualization: str = DEFAULT_MAP_VISUALIZATION
    custom_unexplored_color: str = DEFAULT_CUSTOM_UNEXPLORED_COLOR
    custom_explored_color: str = DEFAULT_CUSTOM_EXPLORED_COLOR
    custom_obstacle_color: str = DEFAULT_CUSTOM_OBSTACLE_COLOR
    custom_explored_opacity: float = DEFAULT_CUSTOM_EXPLORED_OPACITY
    mapped_obstacle_line_width: float = DEFAULT_MAPPED_OBSTACLE_LINE_WIDTH
    robot_icon: str = DEFAULT_ROBOT_ICON

    # Dynamic fire/hazard layer. Occupancy remains UNKNOWN/FREE/OCCUPIED;
    # these parameters control a separate continuous thermal field.
    default_fire_intensity: float = 1.0
    default_fire_radius: float = 2.0
    fire_selection_radius: float = 0.6
    hazard_block_threshold: float = 0.55


    # Simulation camera/view rectangle -- the LOGICAL viewport (also the
    # exploration-metric ROI, see camera_viewport_bounds() below). In editor
    # mode this is shown as a red adjustable frame. In simulation mode this
    # rectangle is the configured world area of interest; SimulationCanvas
    # may render a larger area than exactly this rectangle when the canvas's
    # aspect ratio does not match width:height (see SimulationCanvas.
    # render_view_bounds_world()), so geometry never distorts -- that
    # render-only expansion never changes these four fields.
    camera_center_x: float = (WORLD_X_MIN + WORLD_X_MAX) / 2.0
    camera_center_y: float = (WORLD_Y_MIN + WORLD_Y_MAX) / 2.0
    camera_width: float = WORLD_X_MAX - WORLD_X_MIN
    camera_height: float = WORLD_Y_MAX - WORLD_Y_MIN

    obstacles: list[tuple[float, float, float, float]] = field(
        default_factory=lambda: list(DEFAULT_OBSTACLES)
    )

    show_goal_preview: bool = True
    # Remaining accepted route plus its goal/frontier endpoint.
    show_path: bool = True
    # Accumulated trajectory already executed by each robot.
    show_traveled_path: bool = False
    show_vision: bool = True

    # Accumulated visible area traced by the robot sensor. This is a world/map
    # layer, not a robot command layer, so it is independent from Robot Orders.
    show_explored_area: bool = True

    # Ground-truth obstacle visibility for the human viewer. This does not give
    # the planner access to the obstacles; the planner still uses mapped points.
    show_obstacles: bool = True

    # Robot Orders controls whether internal commands/debug layers are drawn:
    # safety radius r, planned route, waypoints, executed trajectory, and heading arrow.
    show_robot_orders: bool = False

    # Spacing used when the sensor converts real obstacle boundaries into sparse
    # mapped points. Smaller values reveal a denser map.
    mapping_point_spacing: float = 0.025

    # Multi-robot configuration. This first implementation focuses on stable
    # configuration/preview/dragging. Coordination and per-robot planning are
    # intentionally separate next steps.
    robot_count: int = 1
    selected_robot_index: int = 0
    same_robot_configuration: bool = True
    robots: list[RobotStartConfig] = field(default_factory=default_robot_start_configs)

    # Optional research-experiment metadata stored verbatim in the .sim file.
    # Keeping this separate from normal planner fields prevents a paper preset
    # from being confused with the simulator's frontier-exploration modes.
    experiment: dict = field(default_factory=dict)

    # Absolute path of the .sim file that produced this configuration.  This is
    # runtime-only (never serialized) and lets experiment assets be resolved
    # relative to a portable preset instead of the process working directory.
    source_path: str = field(default="", repr=False, compare=False)


# ============================================================
# HELPERS
# ============================================================


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _as_hex_color(value, fallback: str) -> str:
    """Return a normalized #RRGGBB color or a known-good fallback."""
    color = QColor(str(value))
    if not color.isValid():
        color = QColor(str(fallback))
    return color.name().upper()


def _as_obstacle_list(raw_obstacles) -> list[tuple[float, float, float, float]]:
    obstacles = []

    if not isinstance(raw_obstacles, list):
        return list(DEFAULT_OBSTACLES)

    for item in raw_obstacles:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            continue

        x, y, width, height = item
        obstacles.append(
            (
                _as_float(x, 0.0),
                _as_float(y, 0.0),
                max(0.0, _as_float(width, 0.0)),
                max(0.0, _as_float(height, 0.0)),
            )
        )

    return obstacles


def _as_robot_start_list(raw_robots) -> list[RobotStartConfig]:
    if not isinstance(raw_robots, list):
        return default_robot_start_configs()

    default = SimulationConfig()
    robots: list[RobotStartConfig] = []
    for item in raw_robots:
        if not isinstance(item, dict):
            continue

        body_radius = _as_float(item.get("body_radius", default.body_radius), default.body_radius)
        safety_radius = max(
            _as_float(item.get("safety_radius", default.safety_radius), default.safety_radius),
            body_radius,
        )
        robots.append(
            RobotStartConfig(
                x=_as_float(item.get("x", 0.0), 0.0),
                y=_as_float(item.get("y", 0.0), 0.0),
                theta=_as_float(item.get("theta", default.theta), default.theta),
                v=_as_float(item.get("v", default.v), default.v),
                vision=_as_float(item.get("vision", default.vision), default.vision),
                body_radius=body_radius,
                safety_radius=safety_radius,
                max_speed=_as_float(item.get("max_speed", default.max_speed), default.max_speed),
                max_acceleration=_as_float(item.get("max_acceleration", default.max_acceleration), default.max_acceleration),
                max_angular_speed=_as_float(item.get("max_angular_speed", default.max_angular_speed), default.max_angular_speed),
                goal_tolerance=_as_float(item.get("goal_tolerance", default.goal_tolerance), default.goal_tolerance),
                acceleration_gain=_as_float(item.get("acceleration_gain", default.acceleration_gain), default.acceleration_gain),
            )
        )

    return robots or default_robot_start_configs()


def normalized_robot_start_configs(config: SimulationConfig) -> list[RobotStartConfig]:
    count = max(1, min(8, int(config.robot_count)))
    robots = list(config.robots) if config.robots else default_robot_start_configs()
    defaults = default_robot_start_configs()

    while len(robots) < count:
        index = len(robots)
        if index < len(defaults):
            robots.append(defaults[index])
        else:
            row = index // 3
            col = index % 3
            robots.append(RobotStartConfig(-1.0 - 0.7 * row, -0.8 + 0.8 * col, 0.0, 0.0))

    robots = robots[:count]

    if config.same_robot_configuration:
        # Same configuration means global robot/dynamics/sensing parameters are
        # copied to every robot. Only x/y remain independent so the team does
        # not start stacked at one coordinate.
        robots = [
            RobotStartConfig(
                x=float(robot.x),
                y=float(robot.y),
                theta=float(config.theta),
                v=float(config.v),
                vision=float(config.vision),
                body_radius=float(config.body_radius),
                safety_radius=max(float(config.safety_radius), float(config.body_radius)),
                max_speed=float(config.max_speed),
                max_acceleration=float(config.max_acceleration),
                max_angular_speed=float(config.max_angular_speed),
                goal_tolerance=float(config.goal_tolerance),
                acceleration_gain=float(config.acceleration_gain),
            )
            for robot in robots
        ]
    else:
        normalized: list[RobotStartConfig] = []
        for robot in robots:
            body_radius = max(0.01, float(robot.body_radius))
            normalized.append(
                RobotStartConfig(
                    x=float(robot.x),
                    y=float(robot.y),
                    theta=float(robot.theta),
                    v=float(robot.v),
                    vision=max(0.01, float(robot.vision)),
                    body_radius=body_radius,
                    safety_radius=max(float(robot.safety_radius), body_radius),
                    max_speed=max(0.01, float(robot.max_speed)),
                    max_acceleration=max(0.01, float(robot.max_acceleration)),
                    max_angular_speed=max(0.01, float(robot.max_angular_speed)),
                    goal_tolerance=max(0.01, float(robot.goal_tolerance)),
                    acceleration_gain=max(0.01, float(robot.acceleration_gain)),
                )
            )
        robots = normalized

    return robots



def config_to_sim_payload(config: SimulationConfig) -> dict:
    """
    Convert a SimulationConfig into a stable .sim payload.

    A .sim file is JSON on purpose. It should be easy to inspect, edit by hand,
    version, and generate from tests.
    """
    payload = {
        "schema": SIM_FILE_SCHEMA,
        "version": SIM_FILE_VERSION,
        "world": {
            "x_min": WORLD_X_MIN,
            "x_max": WORLD_X_MAX,
            "y_min": WORLD_Y_MIN,
            "y_max": WORLD_Y_MAX,
        },
        "robot": {
            "x": config.x,
            "y": config.y,
            "theta": config.theta,
            "v": config.v,
            "body_radius": config.body_radius,
            "safety_radius": config.safety_radius,
            "max_speed": config.max_speed,
            "max_acceleration": config.max_acceleration,
            "max_angular_speed": config.max_angular_speed,
            "goal_tolerance": config.goal_tolerance,
            "acceleration_gain": config.acceleration_gain,
        },
        "goal": {
            "x": config.goal_x,
            "y": config.goal_y,
        },
        "map": {
            "obstacles": [list(obstacle) for obstacle in config.obstacles],
            "grid_resolution": config.grid_resolution,
        },
        "hazard": {
            "default_fire_intensity": config.default_fire_intensity,
            "default_fire_radius": config.default_fire_radius,
            "fire_selection_radius": config.fire_selection_radius,
            "block_threshold": config.hazard_block_threshold,
        },
        "camera": {
            "center_x": config.camera_center_x,
            "center_y": config.camera_center_y,
            "width": config.camera_width,
            "height": config.camera_height,
        },
        "planner": {
            "type": config.planner_type,
            "path_simplifier": config.path_simplifier,
        },
        "exploration": {
            "planner": config.exploration_planner,
            "clustering_algorithm": config.clustering_algorithm,
            "replan_cooldown": config.exploration_replan_cooldown,
            "ipp_distance_penalty": config.ipp_distance_penalty,
        },
        "coordination": {
            "strategy": config.coordinator_type,
            "safety_algorithm": config.safety_algorithm,
            "parameters": copy.deepcopy(config.coordination_parameters),
            "replan_interval_s": config.coordination_replan_interval_s,
            "strict_contracts": config.coordination_strict_contracts,
        },
        "sensor": {
            "type": config.vision_model,
            "range": config.vision,
            "camera_fov_degrees": config.camera_fov_degrees,
        },
        "multi_robot": {
            "robot_count": int(config.robot_count),
            "selected_robot_index": int(config.selected_robot_index),
            "same_robot_configuration": bool(config.same_robot_configuration),
            "robots": [
                {
                    "x": robot.x,
                    "y": robot.y,
                    "theta": robot.theta,
                    "v": robot.v,
                    "vision": robot.vision,
                    "body_radius": robot.body_radius,
                    "safety_radius": robot.safety_radius,
                    "max_speed": robot.max_speed,
                    "max_acceleration": robot.max_acceleration,
                    "max_angular_speed": robot.max_angular_speed,
                    "goal_tolerance": robot.goal_tolerance,
                    "acceleration_gain": robot.acceleration_gain,
                }
                for robot in normalized_robot_start_configs(config)
            ],
        },
        "simulation": {
            "agent_mode": config.agent_mode,
            "map_visualization": config.map_visualization,
            "custom_unexplored_color": config.custom_unexplored_color,
            "custom_explored_color": config.custom_explored_color,
            "custom_obstacle_color": config.custom_obstacle_color,
            "custom_explored_opacity": config.custom_explored_opacity,
            "mapped_obstacle_line_width": config.mapped_obstacle_line_width,
            "robot_icon": config.robot_icon,
            "show_goal_preview": config.show_goal_preview,
            "show_path": config.show_path,
            "show_traveled_path": config.show_traveled_path,
            "show_vision": config.show_vision,
            "show_explored_area": config.show_explored_area,
            "show_obstacles": config.show_obstacles,
            "show_robot_orders": config.show_robot_orders,
            "mapping_point_spacing": config.mapping_point_spacing,
        },
    }
    if config.experiment:
        payload["experiment"] = copy.deepcopy(config.experiment)
    return payload


def config_from_sim_payload(payload: dict) -> SimulationConfig:
    """
    Build a SimulationConfig from a .sim payload.

    The loader is intentionally tolerant: missing fields fall back to defaults.
    This keeps old .sim files usable as the project grows.
    """
    if not isinstance(payload, dict):
        raise ValueError("Invalid .sim file: expected a JSON object.")

    default = SimulationConfig()

    robot = payload.get("robot", {}) if isinstance(payload.get("robot", {}), dict) else {}
    goal = payload.get("goal", {}) if isinstance(payload.get("goal", {}), dict) else {}
    map_data = payload.get("map", {}) if isinstance(payload.get("map", {}), dict) else {}
    planner = payload.get("planner", {}) if isinstance(payload.get("planner", {}), dict) else {}
    exploration = payload.get("exploration", {}) if isinstance(payload.get("exploration", {}), dict) else {}
    coordination = payload.get("coordination", {}) if isinstance(payload.get("coordination", {}), dict) else {}
    coordination_parameters = (
        copy.deepcopy(coordination.get("parameters", {}))
        if isinstance(coordination.get("parameters", {}), dict)
        else {}
    )
    sensor = payload.get("sensor", {}) if isinstance(payload.get("sensor", {}), dict) else {}
    simulation = payload.get("simulation", {}) if isinstance(payload.get("simulation", {}), dict) else {}
    camera = payload.get("camera", {}) if isinstance(payload.get("camera", {}), dict) else {}
    multi_robot = payload.get("multi_robot", {}) if isinstance(payload.get("multi_robot", {}), dict) else {}
    hazard = payload.get("hazard", {}) if isinstance(payload.get("hazard", {}), dict) else {}
    experiment = payload.get("experiment", {}) if isinstance(payload.get("experiment", {}), dict) else {}

    planner_type = str(planner.get("type", default.planner_type))
    if planner_type not in PLANNER_OPTIONS:
        planner_type = default.planner_type

    path_simplifier = str(
        planner.get(
            "path_simplifier",
            planner.get("simplifier", default.path_simplifier),
        )
    )
    if path_simplifier not in PATH_SIMPLIFIER_OPTIONS:
        path_simplifier = default.path_simplifier

    exploration_planner = str(
        exploration.get(
            "planner",
            planner.get("exploration_planner", default.exploration_planner),
        )
    )
    if exploration_planner not in EXPLORATION_PLANNER_OPTIONS:
        exploration_planner = default.exploration_planner

    clustering_algorithm = str(
        exploration.get("clustering_algorithm", default.clustering_algorithm)
    )
    if clustering_algorithm not in CLUSTERING_ALGORITHM_OPTIONS:
        clustering_algorithm = NO_CLUSTERING_ALGORITHM

    coordinator_type = str(
        coordination.get(
            "strategy",
            coordination.get(
                "coordinator",
                exploration.get("coordinator", default.coordinator_type),
            ),
        )
    )
    if coordinator_type == "CQLite distributed Q-learning":
        coordinator_type = "Travel-time Voronoi + CQLite distributed Q-learning"
    if coordinator_type not in COORDINATOR_OPTIONS:
        coordinator_type = default.coordinator_type
    pipeline_profile = (
        task_assignment_pipeline_profile(coordinator_type)
        if "Multiple" in str(simulation.get("agent_mode", default.agent_mode))
        else None
    )
    if pipeline_profile is not None:
        clustering_algorithm = pipeline_profile.clustering_algorithm
        if pipeline_profile.frontier_detector is not None:
            exploration_planner = pipeline_profile.frontier_detector

    safety_algorithm = str(coordination.get("safety_algorithm", default.safety_algorithm))
    if safety_algorithm not in SAFETY_ALGORITHM_OPTIONS:
        safety_algorithm = WANG_AMES_BARRIER_CERTIFICATE

    vision_model = str(sensor.get("type", default.vision_model))
    if vision_model not in VISION_OPTIONS:
        vision_model = default.vision_model
    if (
        pipeline_profile is not None
        and pipeline_profile.lock_vision_model
        and pipeline_profile.default_vision_model is not None
    ):
        vision_model = pipeline_profile.default_vision_model

    agent_mode = str(simulation.get("agent_mode", default.agent_mode))
    if agent_mode not in ("Single Robot Mode", "Multiple Robot Mode"):
        agent_mode = default.agent_mode

    map_visualization = str(
        simulation.get("map_visualization", default.map_visualization)
    )
    if map_visualization not in MAP_VISUALIZATION_OPTIONS:
        map_visualization = default.map_visualization

    custom_unexplored_color = _as_hex_color(
        simulation.get("custom_unexplored_color", default.custom_unexplored_color),
        default.custom_unexplored_color,
    )
    custom_explored_color = _as_hex_color(
        simulation.get("custom_explored_color", default.custom_explored_color),
        default.custom_explored_color,
    )
    custom_obstacle_color = _as_hex_color(
        simulation.get("custom_obstacle_color", default.custom_obstacle_color),
        default.custom_obstacle_color,
    )
    custom_explored_opacity = min(
        1.0,
        max(
            0.0,
            _as_float(
                simulation.get("custom_explored_opacity", default.custom_explored_opacity),
                default.custom_explored_opacity,
            ),
        ),
    )
    mapped_obstacle_line_width = min(
        6.0,
        max(
            0.25,
            _as_float(
                simulation.get(
                    "mapped_obstacle_line_width",
                    default.mapped_obstacle_line_width,
                ),
                default.mapped_obstacle_line_width,
            ),
        ),
    )

    robot_icon = str(simulation.get("robot_icon", default.robot_icon))
    if robot_icon not in ROBOT_ICON_OPTIONS:
        robot_icon = default.robot_icon

    return SimulationConfig(
        x=_as_float(robot.get("x", default.x), default.x),
        y=_as_float(robot.get("y", default.y), default.y),
        theta=_as_float(robot.get("theta", default.theta), default.theta),
        v=_as_float(robot.get("v", default.v), default.v),
        vision=_as_float(sensor.get("range", default.vision), default.vision),
        camera_fov_degrees=_as_float(
            sensor.get("camera_fov_degrees", default.camera_fov_degrees),
            default.camera_fov_degrees,
        ),
        body_radius=_as_float(
            robot.get("body_radius", robot.get("robot_radius", default.body_radius)),
            default.body_radius,
        ),
        safety_radius=max(
            _as_float(
                robot.get("safety_radius", robot.get("robot_radius", default.safety_radius)),
                default.safety_radius,
            ),
            _as_float(
                robot.get("body_radius", robot.get("robot_radius", default.body_radius)),
                default.body_radius,
            ),
        ),
        goal_x=_as_float(goal.get("x", default.goal_x), default.goal_x),
        goal_y=_as_float(goal.get("y", default.goal_y), default.goal_y),
        max_speed=_as_float(robot.get("max_speed", default.max_speed), default.max_speed),
        max_acceleration=_as_float(
            robot.get("max_acceleration", default.max_acceleration),
            default.max_acceleration,
        ),
        max_angular_speed=_as_float(
            robot.get("max_angular_speed", default.max_angular_speed),
            default.max_angular_speed,
        ),
        goal_tolerance=_as_float(
            robot.get("goal_tolerance", default.goal_tolerance),
            default.goal_tolerance,
        ),
        acceleration_gain=_as_float(
            robot.get("acceleration_gain", default.acceleration_gain),
            default.acceleration_gain,
        ),
        planner_type=planner_type,
        path_simplifier=path_simplifier,
        exploration_planner=exploration_planner,
        clustering_algorithm=clustering_algorithm,
        coordinator_type=coordinator_type,
        safety_algorithm=safety_algorithm,
        coordination_parameters=coordination_parameters,
        coordination_replan_interval_s=_as_float(
            coordination.get("replan_interval_s", default.coordination_replan_interval_s),
            default.coordination_replan_interval_s,
        ),
        coordination_strict_contracts=bool(
            coordination.get("strict_contracts", default.coordination_strict_contracts)
        ),
        exploration_replan_cooldown=_as_float(
            exploration.get(
                "replan_cooldown",
                planner.get(
                    "exploration_replan_cooldown",
                    default.exploration_replan_cooldown,
                ),
            ),
            default.exploration_replan_cooldown,
        ),
        ipp_distance_penalty=_as_float(
            exploration.get(
                "ipp_distance_penalty",
                planner.get("ipp_distance_penalty", default.ipp_distance_penalty),
            ),
            default.ipp_distance_penalty,
        ),
        vision_model=vision_model,
        agent_mode=agent_mode,
        grid_resolution=_as_float(
            map_data.get("grid_resolution", default.grid_resolution),
            default.grid_resolution,
        ),
        map_visualization=map_visualization,
        custom_unexplored_color=custom_unexplored_color,
        custom_explored_color=custom_explored_color,
        custom_obstacle_color=custom_obstacle_color,
        custom_explored_opacity=custom_explored_opacity,
        mapped_obstacle_line_width=mapped_obstacle_line_width,
        robot_icon=robot_icon,
        default_fire_intensity=min(
            1.0,
            max(
                1e-6,
                _as_float(
                    hazard.get("default_fire_intensity", default.default_fire_intensity),
                    default.default_fire_intensity,
                ),
            ),
        ),
        default_fire_radius=max(
            1e-6,
            _as_float(
                hazard.get("default_fire_radius", default.default_fire_radius),
                default.default_fire_radius,
            ),
        ),
        fire_selection_radius=max(
            0.0,
            _as_float(
                hazard.get("fire_selection_radius", default.fire_selection_radius),
                default.fire_selection_radius,
            ),
        ),
        hazard_block_threshold=min(
            1.0,
            max(
                1e-6,
                _as_float(
                    hazard.get("block_threshold", default.hazard_block_threshold),
                    default.hazard_block_threshold,
                ),
            ),
        ),
        camera_center_x=_as_float(camera.get("center_x", default.camera_center_x), default.camera_center_x),
        camera_center_y=_as_float(camera.get("center_y", default.camera_center_y), default.camera_center_y),
        camera_width=max(1.0, _as_float(camera.get("width", default.camera_width), default.camera_width)),
        camera_height=max(1.0, _as_float(camera.get("height", default.camera_height), default.camera_height)),
        obstacles=_as_obstacle_list(map_data.get("obstacles", default.obstacles)),
        show_goal_preview=bool(
            simulation.get("show_goal_preview", default.show_goal_preview)
        ),
        show_path=bool(simulation.get("show_path", default.show_path)),
        show_traveled_path=bool(
            simulation.get("show_traveled_path", default.show_traveled_path)
        ),
        show_vision=bool(simulation.get("show_vision", default.show_vision)),
        show_explored_area=bool(
            simulation.get("show_explored_area", default.show_explored_area)
        ),
        show_obstacles=bool(simulation.get("show_obstacles", default.show_obstacles)),
        show_robot_orders=bool(
            simulation.get("show_robot_orders", default.show_robot_orders)
        ),
        mapping_point_spacing=_as_float(
            simulation.get("mapping_point_spacing", default.mapping_point_spacing),
            default.mapping_point_spacing,
        ),
        robot_count=max(1, min(8, int(_as_float(multi_robot.get("robot_count", default.robot_count), default.robot_count)))),
        selected_robot_index=max(0, int(_as_float(multi_robot.get("selected_robot_index", default.selected_robot_index), default.selected_robot_index))),
        same_robot_configuration=bool(multi_robot.get("same_robot_configuration", default.same_robot_configuration)),
        robots=_as_robot_start_list(multi_robot.get("robots", default.robots)),
        experiment=copy.deepcopy(experiment),
    )


def save_sim_file(path: str, config: SimulationConfig) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config_to_sim_payload(config), file, indent=2)


def load_sim_file(path: str) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    config = config_from_sim_payload(payload)
    config.source_path = os.path.abspath(path)
    return config


def find_tamu_image() -> str | None:
    for path in TAMU_IMAGE_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance_point_to_rect_world(
    point: tuple[float, float],
    obstacle: tuple[float, float, float, float],
) -> float:
    """
    Euclidean distance from a point to an axis-aligned rectangle.

    This is used as a cheap bounding-box rejection test before expensive
    ray/segment intersection.
    """
    px, py = point
    ox, oy, ow, oh = obstacle
    closest_x = clamp(px, ox, ox + ow)
    closest_y = clamp(py, oy, oy + oh)
    return math.hypot(px - closest_x, py - closest_y)


def filter_obstacles_by_sensor_range(
    origin: tuple[float, float],
    obstacles: list[tuple[float, float, float, float]],
    max_range: float,
    padding: float = 0.0,
) -> list[tuple[float, float, float, float]]:
    """
    Cheap pre-filter for ray-casting.

    Obstacles outside the sensor range cannot affect the first hit of a ray, so
    testing their four edges for every ray is wasted work.
    """
    limit = float(max_range) + float(padding)
    return [
        tuple(obstacle)
        for obstacle in obstacles
        if distance_point_to_rect_world(origin, tuple(obstacle)) <= limit
    ]


class SpatialObstacleIndex:
    """
    Coarse spatial hashing for static rectangular obstacles.

    This is intentionally simple. The goal is not to replace the collision
    checker; it only reduces the number of obstacles considered by sensor
    ray-casting and incremental mapping.
    """

    def __init__(self, cell_size: float = SPATIAL_BUCKET_SIZE):
        self.cell_size = max(float(cell_size), 0.25)
        self.buckets: dict[tuple[int, int], list[tuple[float, float, float, float]]] = {}
        self.obstacles: list[tuple[float, float, float, float]] = []

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(float(x) / self.cell_size), math.floor(float(y) / self.cell_size))

    def rebuild(self, obstacles: list[tuple[float, float, float, float]]) -> None:
        self.buckets.clear()
        self.obstacles = [tuple(obstacle) for obstacle in obstacles]

        for obstacle in self.obstacles:
            ox, oy, ow, oh = obstacle
            min_cell = self._cell(ox, oy)
            max_cell = self._cell(ox + ow, oy + oh)

            for cx in range(min_cell[0], max_cell[0] + 1):
                for cy in range(min_cell[1], max_cell[1] + 1):
                    self.buckets.setdefault((cx, cy), []).append(obstacle)

    def query_circle(
        self,
        origin: tuple[float, float],
        radius: float,
        padding: float = 0.0,
    ) -> list[tuple[float, float, float, float]]:
        if not self.buckets:
            return filter_obstacles_by_sensor_range(origin, self.obstacles, radius, padding)

        ox, oy = origin
        limit = float(radius) + float(padding)
        min_cell = self._cell(ox - limit, oy - limit)
        max_cell = self._cell(ox + limit, oy + limit)

        found: dict[tuple[float, float, float, float], None] = {}
        for cx in range(min_cell[0], max_cell[0] + 1):
            for cy in range(min_cell[1], max_cell[1] + 1):
                for obstacle in self.buckets.get((cx, cy), []):
                    if distance_point_to_rect_world(origin, obstacle) <= limit:
                        found[obstacle] = None

        return list(found.keys())


def rect_edges(rect: tuple[float, float, float, float]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Return the four boundary segments of an axis-aligned rectangular obstacle.

    The simulator stores obstacles as (x, y, width, height), where (x, y) is
    the lower-left world coordinate.
    """
    ox, oy, ow, oh = rect
    bottom_left = (float(ox), float(oy))
    bottom_right = (float(ox + ow), float(oy))
    top_right = (float(ox + ow), float(oy + oh))
    top_left = (float(ox), float(oy + oh))

    return [
        (bottom_left, bottom_right),
        (bottom_right, top_right),
        (top_right, top_left),
        (top_left, bottom_left),
    ]


def _cross_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def ray_segment_intersection_distance(
    origin: tuple[float, float],
    angle: float,
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> float | None:
    """
    Return distance from origin to the first intersection between a ray and a
    segment, or None if they do not intersect.

    Abstraction:
        This is the geometric primitive behind occlusion. A sensor ray should
        stop at the first obstacle boundary it hits; it must not see behind it.
    """
    ox, oy = origin
    ax, ay = segment_start
    bx, by = segment_end

    ray_dir = (math.cos(angle), math.sin(angle))
    segment_dir = (bx - ax, by - ay)
    denominator = _cross_2d(ray_dir, segment_dir)

    if abs(denominator) <= 1e-12:
        return None

    q_minus_p = (ax - ox, ay - oy)
    t = _cross_2d(q_minus_p, segment_dir) / denominator
    u = _cross_2d(q_minus_p, ray_dir) / denominator

    if t < 0.0:
        return None

    if u < -1e-9 or u > 1.0 + 1e-9:
        return None

    # ray_dir is unit length, so t is distance in meters.
    return float(t)


def first_ray_hit_distance(
    origin: tuple[float, float],
    angle: float,
    obstacles: list[tuple[float, float, float, float]],
    max_range: float,
) -> float:
    """
    Distance to the first obstacle hit along a sensor ray.

    If no obstacle is hit, max_range is returned. This makes it directly usable
    for drawing the visible sensor boundary.
    """
    nearest = float(max_range)
    origin_xy = (float(origin[0]), float(origin[1]))

    for obstacle in obstacles:
        obstacle = tuple(obstacle)
        if distance_point_to_rect_world(origin_xy, obstacle) > nearest:
            continue

        for start, end in rect_edges(obstacle):
            hit_distance = ray_segment_intersection_distance(origin, angle, start, end)
            if hit_distance is None:
                continue

            if 0.0 <= hit_distance < nearest:
                nearest = hit_distance

    return nearest


def sensor_visible_polygon_world(
    origin: tuple[float, float],
    theta: float,
    vision: float,
    vision_model: str,
    obstacles: list[tuple[float, float, float, float]],
    ray_count: int | None = None,
    camera_fov_degrees: float = 70.0,
) -> list[tuple[float, float]]:
    """
    Return the visible sensor area as a world-coordinate polygon.

    The polygon is generated by ray-casting against ground-truth obstacles, so
    it respects occlusion: rays stop at the first obstacle surface they hit.
    This function is used both for the current blue sensor footprint and for
    the accumulated explored-area trail.
    """
    vision = float(vision)
    if vision <= 0.0:
        return []

    x, y = float(origin[0]), float(origin[1])
    nearby_obstacles = filter_obstacles_by_sensor_range(
        origin=(x, y),
        obstacles=obstacles,
        max_range=vision,
        padding=0.05,
    )

    if "Camera" in vision_model:
        count = ray_count or 121
        count = max(3, int(count))
        camera_fov = math.radians(float(camera_fov_degrees))
        start_angle = float(theta) - camera_fov / 2.0
        end_angle = float(theta) + camera_fov / 2.0
        angles = [
            start_angle + (end_angle - start_angle) * i / (count - 1)
            for i in range(count)
        ]

        polygon = [(x, y)]
    else:
        count = ray_count or 241
        count = max(4, int(count))
        angles = [2.0 * math.pi * i / (count - 1) for i in range(count)]
        polygon = []

    for angle in angles:
        hit_distance = first_ray_hit_distance(
            origin=(x, y),
            angle=angle,
            obstacles=nearby_obstacles,
            max_range=vision,
        )
        polygon.append(
            (
                x + hit_distance * math.cos(angle),
                y + hit_distance * math.sin(angle),
            )
        )

    return polygon


def wrapped_angle_error(desired_angle: float, current_angle: float) -> float:
    """
    Smallest signed angular difference desired - current.
    """
    return (desired_angle - current_angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_is_inside_sensor_model(
    angle: float,
    robot_theta: float,
    vision_model: str,
    camera_fov: float = math.radians(70.0),
    camera_fov_degrees: float | None = None,
) -> bool:
    """
    Decide whether a direction belongs to the current sensor model.

    LiDAR and Omnidirectional are treated as 360-degree sensors in this 2D
    baseline. Camera / FoV uses a finite cone around robot_theta.
    """
    if "Camera" not in vision_model:
        return True

    if camera_fov_degrees is not None:
        camera_fov = math.radians(float(camera_fov_degrees))
    return abs(wrapped_angle_error(angle, robot_theta)) <= camera_fov / 2.0


def mode_name(robot) -> str:
    if robot is None:
        return "CONFIG"

    if hasattr(robot, "mode_name"):
        return robot.mode_name

    if hasattr(robot, "mode"):
        mode = robot.mode
        if hasattr(mode, "value"):
            return str(mode.value)
        return str(mode)

    return "RUNNING"


