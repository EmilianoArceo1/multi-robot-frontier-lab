"""
Regression tests for a sensor/planner field-of-view mismatch.

Symptom: robotics_sim/simulation/config.py's sensor_visible_polygon_world()
and angle_is_inside_sensor_model() -- the functions that actually decide what
the sensor discovers -- treat "Camera / FoV" as a 70-degree cone and treat
"LiDAR" (and any other current vision_model) as a 360-degree sensor. But
FoVAwareDirectionalFrontierPlanner.select_goal() in
robotics_sim/planning/exploration_planners.py estimated information gain
using kwargs.get("fov_angle", math.radians(120.0)) -- an invented 120-degree
default that matches neither real sensor whenever a caller does not pass
fov_angle explicitly (as engine.py's direct select_exploration_goal() call
does not). This silently biased information_gain and could change which
frontier gets selected.

Fix: a single source of truth, sensor_fov_angle_radians(vision_model), added
to robotics_sim/core/geometry.py (pure, no Qt/config/simulator imports).
    - config.py's sensor_visible_polygon_world() and
      angle_is_inside_sensor_model() now derive the Camera cone half-angle
      from it instead of a locally duplicated math.radians(70.0) literal.
    - exploration_planners.py's FoVAwareDirectionalFrontierPlanner gained
      _resolve_fov_angle(kwargs): an explicit, non-None kwargs["fov_angle"]
      still wins; otherwise the angle is derived from kwargs["vision_model"]
      via the same function; if vision_model is also absent, it falls back
      to the LiDAR/omnidirectional policy (2*pi) rather than 120 degrees.
    - planner_services.PlannerServices.select_exploration_target_request()
      now also passes fov_angle=sensor_fov_angle_radians(request.vision_model)
      explicitly, so the normal runtime path never relies on the planner's
      own fallback at all; vision_model is still passed too, for diagnostics
      and for direct callers like engine.py that bypass PlannerServices.

These tests exercise robotics_sim/core/geometry.py,
robotics_sim/simulation/config.py, robotics_sim/planning/exploration_planners.py
and robotics_sim/simulation/planner_services.py directly -- no Qt, no canvas,
no full engine/GUI instantiation.
"""
from __future__ import annotations

import math

import robotics_sim.planning.exploration_planners as exploration_planners
import robotics_sim.simulation.planner_services as planner_services_module
from robotics_sim.core.geometry import (
    CAMERA_FOV_ANGLE_RAD,
    OMNIDIRECTIONAL_FOV_ANGLE_RAD,
    sensor_fov_angle_radians,
)
from robotics_sim.environment.belief_map import UNKNOWN, BeliefMap
from robotics_sim.planning.exploration_planners import (
    _resolve_fov_angle,
    select_exploration_goal,
)
from robotics_sim.simulation.config import (
    angle_is_inside_sensor_model,
    sensor_visible_polygon_world,
)
from robotics_sim.simulation.planner_services import PlannerServices


def _angle_deg(point: tuple[float, float]) -> float:
    return math.degrees(math.atan2(point[1], point[0]))


# ---------------------------------------------------------------------------
# 1. Source of truth: sensor_fov_angle_radians() defines 70 vs 360 degrees.
# ---------------------------------------------------------------------------


def test_sensor_fov_angle_radians_camera_is_70_degrees():
    assert sensor_fov_angle_radians("Camera / FoV") == math.radians(70.0)
    assert sensor_fov_angle_radians("Camera / FoV") == CAMERA_FOV_ANGLE_RAD


def test_sensor_fov_angle_radians_lidar_is_360_degrees():
    assert sensor_fov_angle_radians("LiDAR") == 2.0 * math.pi
    assert sensor_fov_angle_radians("LiDAR") == OMNIDIRECTIONAL_FOV_ANGLE_RAD


def test_sensor_fov_angle_radians_defensive_str_conversion():
    # Any non-Camera label, including non-string input, must fall back to the
    # omnidirectional policy without raising.
    assert sensor_fov_angle_radians(None) == OMNIDIRECTIONAL_FOV_ANGLE_RAD
    assert sensor_fov_angle_radians("Omnidirectional") == OMNIDIRECTIONAL_FOV_ANGLE_RAD


# ---------------------------------------------------------------------------
# 2. Camera polygon: extreme rays land near +/-35 degrees around theta=0.
# ---------------------------------------------------------------------------


def test_camera_polygon_extreme_rays_are_near_plus_minus_35_degrees():
    polygon = sensor_visible_polygon_world(
        origin=(0.0, 0.0),
        theta=0.0,
        vision=2.0,
        vision_model="Camera / FoV",
        obstacles=[],
    )

    assert len(polygon) >= 3
    # polygon[0] is the origin; polygon[1] and polygon[-1] are the two
    # extreme rays of the cone (start_angle and end_angle).
    first_ray_deg = _angle_deg(polygon[1])
    last_ray_deg = _angle_deg(polygon[-1])

    assert math.isclose(first_ray_deg, -35.0, abs_tol=0.5)
    assert math.isclose(last_ray_deg, 35.0, abs_tol=0.5)


# ---------------------------------------------------------------------------
# 3. LiDAR polygon: a full sweep, not limited to 70 or 120 degrees.
# ---------------------------------------------------------------------------


def test_lidar_polygon_covers_a_full_sweep_not_70_or_120_degrees():
    polygon = sensor_visible_polygon_world(
        origin=(0.0, 0.0),
        theta=0.0,
        vision=2.0,
        vision_model="LiDAR",
        obstacles=[],
    )

    angles_deg = [_angle_deg(point) for point in polygon]

    # A 70 or 120 degree cone around theta=0 could never reach a point
    # directly behind the robot (~180 degrees). LiDAR must.
    assert any(abs(abs(a) - 180.0) < 5.0 for a in angles_deg)
    # And it must also cover a lateral direction outside even a 120-degree
    # cone's +/-60 degree half-angle.
    assert any(abs(a) > 100.0 for a in angles_deg)


# ---------------------------------------------------------------------------
# 4. Angular membership check: angle_is_inside_sensor_model().
# ---------------------------------------------------------------------------


def test_angle_is_inside_sensor_model_camera_cone_boundary():
    assert angle_is_inside_sensor_model(
        angle=math.radians(34.0), robot_theta=0.0, vision_model="Camera / FoV"
    )
    assert not angle_is_inside_sensor_model(
        angle=math.radians(36.0), robot_theta=0.0, vision_model="Camera / FoV"
    )


def test_angle_is_inside_sensor_model_lidar_is_omnidirectional():
    assert angle_is_inside_sensor_model(
        angle=math.radians(170.0), robot_theta=0.0, vision_model="LiDAR"
    )
    assert angle_is_inside_sensor_model(
        angle=math.radians(-170.0), robot_theta=0.0, vision_model="LiDAR"
    )


# ---------------------------------------------------------------------------
# 5. PlannerServices passes the numeric fov_angle explicitly at runtime.
# ---------------------------------------------------------------------------


def _select_via_planner_services(monkeypatch, vision_model: str) -> dict:
    captured: dict = {}

    def fake_seg(planner_name, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(planner_services_module, "_seg", fake_seg)

    services = PlannerServices()
    services.select_exploration_target(
        planner_name="FoV-aware directional frontier",
        belief_map=None,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_radius=0.2,
        sensor_range=2.5,
        vision_model=vision_model,
        ipp_distance_penalty=0.2,
    )

    return captured


def test_planner_services_passes_numeric_fov_angle_for_camera(monkeypatch):
    captured = _select_via_planner_services(monkeypatch, "Camera / FoV")
    assert captured["fov_angle"] == math.radians(70.0)
    assert captured["vision_model"] == "Camera / FoV"


def test_planner_services_passes_numeric_fov_angle_for_lidar(monkeypatch):
    captured = _select_via_planner_services(monkeypatch, "LiDAR")
    assert captured["fov_angle"] == 2.0 * math.pi
    assert captured["vision_model"] == "LiDAR"


# ---------------------------------------------------------------------------
# 6. FoVAwareDirectionalFrontierPlanner resolves fov_angle directly from
#    vision_model when called without fov_angle, as engine.py does.
# ---------------------------------------------------------------------------


def test_resolve_fov_angle_pure_function_camera_and_lidar():
    assert _resolve_fov_angle({"vision_model": "Camera / FoV"}) == math.radians(70.0)
    assert _resolve_fov_angle({"vision_model": "LiDAR"}) == 2.0 * math.pi


def test_resolve_fov_angle_missing_vision_model_uses_lidar_policy_not_120():
    # No fov_angle, no vision_model at all: must not silently invent 120
    # degrees, since no configured sensor uses that angle.
    resolved = _resolve_fov_angle({})
    assert resolved == 2.0 * math.pi
    assert not math.isclose(resolved, math.radians(120.0))


def _belief_map_open(resolution: float = 0.5) -> BeliefMap:
    return BeliefMap(bounds=(-8.0, 8.0, -8.0, 8.0), resolution=resolution, robot_count=1)


def test_planner_direct_call_derives_camera_and_lidar_fov_from_vision_model(monkeypatch):
    captured: list[float] = []
    original_score_candidate = exploration_planners._score_candidate

    def spy(**kwargs):
        captured.append(kwargs["fov_angle"])
        return original_score_candidate(**kwargs)

    monkeypatch.setattr(exploration_planners, "_score_candidate", spy)

    belief = _belief_map_open()
    result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_count=1,
        robot_radius=0.15,
        sensor_range=3.0,
        vision_model="Camera / FoV",
        ipp_distance_penalty=0.2,
    )

    assert result.success
    assert captured
    assert all(value == math.radians(70.0) for value in captured)


def test_planner_direct_call_derives_lidar_fov_from_vision_model(monkeypatch):
    captured: list[float] = []
    original_score_candidate = exploration_planners._score_candidate

    def spy(**kwargs):
        captured.append(kwargs["fov_angle"])
        return original_score_candidate(**kwargs)

    monkeypatch.setattr(exploration_planners, "_score_candidate", spy)

    belief = _belief_map_open()
    result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_count=1,
        robot_radius=0.15,
        sensor_range=3.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
    )

    assert result.success
    assert captured
    assert all(value == 2.0 * math.pi for value in captured)


# ---------------------------------------------------------------------------
# 7. An explicit fov_angle override always wins over vision_model.
# ---------------------------------------------------------------------------


def test_explicit_fov_angle_overrides_vision_model():
    override = math.radians(90.0)
    assert _resolve_fov_angle({"vision_model": "Camera / FoV", "fov_angle": override}) == override
    assert _resolve_fov_angle({"vision_model": "LiDAR", "fov_angle": override}) == override


def test_planner_direct_call_explicit_fov_angle_overrides_vision_model(monkeypatch):
    captured: list[float] = []
    original_score_candidate = exploration_planners._score_candidate

    def spy(**kwargs):
        captured.append(kwargs["fov_angle"])
        return original_score_candidate(**kwargs)

    monkeypatch.setattr(exploration_planners, "_score_candidate", spy)

    belief = _belief_map_open()
    override = math.radians(90.0)
    result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_count=1,
        robot_radius=0.15,
        sensor_range=3.0,
        vision_model="Camera / FoV",
        fov_angle=override,
        ipp_distance_penalty=0.2,
    )

    assert result.success
    assert captured
    assert all(value == override for value in captured)


# ---------------------------------------------------------------------------
# 8. Functional difference: same pose/range/map, Camera vs LiDAR must yield
#    different information_gain, since unknown area to the side/behind the
#    robot is only ever swept by an omnidirectional sensor.
# ---------------------------------------------------------------------------


def test_camera_and_lidar_produce_different_information_gain_same_map():
    # The whole grid starts UNKNOWN by default (see BeliefMap docstring):
    # unknown area exists directly ahead of the robot (heading=0, +x) as well
    # as to the sides and behind. With no obstacles, FoVAwareDirectionalFrontierPlanner's
    # _forward_candidate() always finds a target straight ahead, so both runs
    # score the *same* candidate/path -- isolating the FoV term.
    camera_belief = _belief_map_open()
    lidar_belief = _belief_map_open()

    common_kwargs = dict(
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_count=1,
        robot_radius=0.15,
        sensor_range=3.0,
        ipp_distance_penalty=0.2,
    )

    camera_result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=camera_belief,
        vision_model="Camera / FoV",
        **common_kwargs,
    )
    lidar_result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=lidar_belief,
        vision_model="LiDAR",
        **common_kwargs,
    )

    assert camera_result.success
    assert lidar_result.success
    assert len(camera_result.candidates) == 1
    assert len(lidar_result.candidates) == 1

    # Same map, same pose/range, same (only) candidate target -- but a
    # narrower FoV must see strictly less unknown area than an
    # omnidirectional one.
    assert camera_result.candidates[0].target == lidar_result.candidates[0].target
    assert lidar_result.candidates[0].information_gain > camera_result.candidates[0].information_gain


def test_fov_cells_camera_vs_lidar_directly_on_same_belief_map():
    # Lower-level confirmation of the same effect using the pure _fov_cells()
    # helper directly, independent of scoring/weights/A*.
    belief = _belief_map_open()
    belief.force_free_point((0.0, 0.0))

    camera_cells = exploration_planners._fov_cells(
        belief=belief,
        position=(0.0, 0.0),
        heading=0.0,
        sensor_range=3.0,
        fov_angle=sensor_fov_angle_radians("Camera / FoV"),
        use_occlusion=False,
    )
    lidar_cells = exploration_planners._fov_cells(
        belief=belief,
        position=(0.0, 0.0),
        heading=0.0,
        sensor_range=3.0,
        fov_angle=sensor_fov_angle_radians("LiDAR"),
        use_occlusion=False,
    )

    camera_unknown = sum(1 for cell in camera_cells if int(belief.grid[cell]) == UNKNOWN)
    lidar_unknown = sum(1 for cell in lidar_cells if int(belief.grid[cell]) == UNKNOWN)

    assert lidar_unknown > camera_unknown
