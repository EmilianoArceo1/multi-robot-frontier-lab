"""
Integration tests: FoVAwareDirectionalFrontierPlanner.select_goal()
(robotics_sim/planning/exploration_planners.py) no longer builds its own
grid via belief.to_planning_grid() when the caller supplies one -- it now
consumes kwargs["planning_grid"], the SAME PlanningCostmapBuilder-backed
OccupancyGrid engine.py's build_planning_grid_for_robot() already produces
for build_planner_kwargs()/build_planner_kwargs_for_goal()/
build_planner_kwargs_for_multi_robot()/make_exploration_reachability_
check(). engine.py's select_navigation_goal() -- the single-robot goal-
selection call site build_planner_kwargs() itself uses -- now builds that
grid via the SAME helper (build_planning_grid_for_robot(), reusing
_planning_costmap_inputs_for_robot()/_planning_grid_from_costmap_snapshot()
internally, never duplicated here) and passes it through as
planning_grid=.

When no planning_grid is supplied (any other/direct/test caller), the
planner falls back to its original belief.to_planning_grid() behavior --
unchanged, so existing callers that never pass this new kwarg keep working
exactly as before.

Multi-robot frontier ASSIGNMENT does not currently route through this
planner class at all: select_navigation_goal_for_multi_robot() ->
synchronize_multi_frontier_targets() -> MultiRobotCoordinator, a separate
plugin system explicitly out of scope for this change (see "No tocar
todavia: coordinated_frontier_planner.py ... algoritmos/plugins" in this
migration's own instructions). The dynamic-parity test below therefore
proves the planner's CONSUMPTION side is dynamic-point aware -- it uses
whatever grid it is given, including one built with other-robot dynamic
occupancy the exact same way build_planner_kwargs_for_multi_robot() builds
its own -- rather than claiming multi-robot target assignment is wired
through this class today (it is not).

Fakes bind the REAL SimulationControllerMixin methods under test (same
convention as test_planning_runtime_costmap_integration.py). A monkeypatch
spy wraps the REAL module-level _score_candidate()/PlanningCostmapBuilder.
build()/BeliefMap.to_planning_grid() where "what grid was actually used"
needs to be observed directly -- never a reimplementation of what any of
them does.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.environment.occupancy_grid import OCCUPIED as OG_OCCUPIED
import robotics_sim.planning.exploration_planners as exploration_planners_module
from robotics_sim.planning.exploration_planners import FoVAwareDirectionalFrontierPlanner
from robotics_sim.planning.planning_costmap_builder import PlanningCostmapBuilder
from robotics_sim.simulation.engine import SimulationControllerMixin

RESOLUTION = 1.0
ROBOT_RADIUS = 0.3


def _make_fake_engine(
    *,
    robot_positions: list[tuple[float, float]] | None = None,
    obstacles: list | None = None,
) -> SimpleNamespace:
    positions = robot_positions if robot_positions is not None else [(0.0, 0.0)]
    robots = [SimpleNamespace(x=x, y=y, theta=0.0, vision=3.0) for x, y in positions]

    config = SimpleNamespace(
        grid_resolution=RESOLUTION,
        mapping_point_spacing=0.5,
        body_radius=0.2,
        safety_radius=ROBOT_RADIUS,
        planner_type="A*",
        goal_tolerance=0.25,
        exploration_planner="Goal seeking",  # overridden per-test when the FoV branch is needed
        goal_x=9.0,
        goal_y=9.0,
        vision=3.0,
        ipp_distance_penalty=0.0,
        vision_model="LiDAR",
        default_fire_intensity=1.0,
        default_fire_radius=2.0,
        fire_selection_radius=0.6,
        hazard_block_threshold=0.55,
        obstacles=list(obstacles or []),
    )
    fake = SimpleNamespace(
        robot=robots[0],
        robots=robots,
        config=config,
        mapped_obstacle_points=[],
        multi_exploration_targets=[],
        current_exploration_target=None,
        last_goal_selection_reason="",
        simulation_time=0.0,
        canvas=SimpleNamespace(
            append_mapped_obstacle_points=lambda points: None,
            set_status=lambda message: None,
            set_exploration_target=lambda target: None,
            set_multi_exploration_targets=lambda targets: None,
        ),
        telemetry=SimpleNamespace(report_frontier_selection=lambda **kwargs: None),
    )
    # Not the subject under test here (agent/registry wiring is orthogonal
    # to the FoV costmap source) -- select_navigation_goal()'s exploration
    # branch tolerates None.
    fake.runtime_agent = lambda index=None: None

    for name in (
        "reset_belief_map",
        "ensure_belief_map",
        "sync_legacy_map_views_from_belief",
        "push_discovered_hazard_frame",
        "force_robot_pose_free_in_belief",
        "safety_radius_for_robot",
        "safety_radius",
        "body_radius_for_robot",
        "body_radius",
        "sanitize_planner_obstacle_points",
        "dynamic_robot_obstacle_points_for_robot",
        "build_planning_grid_for_robot",
        "_planning_costmap_inputs_for_robot",
        "_planning_grid_from_costmap_snapshot",
        "_dynamic_obstacle_points_for_robot_object",
        "final_goal_xy",
        "select_navigation_goal",
        "select_navigation_goal_for_multi_robot",
        "ensure_multi_exploration_target_slots",
        "publish_multi_exploration_targets",
        "is_exploration_mode",
        "exploration_planner_name",
        "build_planner_kwargs",
        "build_planner_kwargs_for_goal",
        "build_planner_kwargs_for_multi_robot",
        "observed_obstacle_snapshot",
        "make_exploration_reachability_check",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    fake.reset_belief_map()
    return fake


def _capture_calls(fake: SimpleNamespace, method_name: str) -> list[dict]:
    """Wrap the ALREADY-bound real method on fake, recording every call's
    arguments and return value before delegating to the real
    implementation."""
    real_method = getattr(fake, method_name)
    calls: list[dict] = []

    def _wrapper(robot, *, obstacle_points=None, robot_radius=None, dynamic_obstacle_points=()):
        result = real_method(
            robot,
            obstacle_points=obstacle_points,
            robot_radius=robot_radius,
            dynamic_obstacle_points=dynamic_obstacle_points,
        )
        calls.append(
            {
                "robot": robot,
                "obstacle_points": obstacle_points,
                "robot_radius": robot_radius,
                "dynamic_obstacle_points": tuple(dynamic_obstacle_points),
                "result": result,
            }
        )
        return result

    setattr(fake, method_name, _wrapper)
    return calls


def _spy_planning_costmap_builder(monkeypatch) -> list[dict]:
    """Wrap the REAL PlanningCostmapBuilder.build with a monkeypatched spy
    that records each call's kwargs before delegating to the real
    implementation -- proves the FoV path actually calls the builder,
    without copying or reimplementing anything it does."""
    calls: list[dict] = []
    real_build = PlanningCostmapBuilder.build

    def _spy(self, **kwargs):
        calls.append(kwargs)
        return real_build(self, **kwargs)

    monkeypatch.setattr(PlanningCostmapBuilder, "build", _spy)
    return calls


def _spy_belief_to_planning_grid(monkeypatch) -> list[dict]:
    """Wrap the REAL BeliefMap.to_planning_grid with a monkeypatched spy --
    proves the FoV path does NOT fall back to it when a planning_grid is
    supplied."""
    calls: list[dict] = []
    real_to_planning_grid = BeliefMap.to_planning_grid

    def _spy(self, **kwargs):
        calls.append(kwargs)
        return real_to_planning_grid(self, **kwargs)

    monkeypatch.setattr(BeliefMap, "to_planning_grid", _spy)
    return calls


def _capture_score_candidate_calls(monkeypatch) -> list:
    """Wrap the REAL module-level _score_candidate() (exploration_planners.py)
    with a monkeypatched spy, recording the planning_grid each call actually
    received -- proves what the FoV scorer used to score a candidate,
    without reimplementing scoring."""
    grids: list = []
    real_score_candidate = exploration_planners_module._score_candidate

    def _spy(**kwargs):
        grids.append(kwargs.get("planning_grid"))
        return real_score_candidate(**kwargs)

    monkeypatch.setattr(exploration_planners_module, "_score_candidate", _spy)
    return grids


# ---------------------------------------------------------------------------
# 1. Builder invocation.
# ---------------------------------------------------------------------------


def test_fov_path_invokes_planning_costmap_builder_not_belief_fallback(monkeypatch):
    fake = _make_fake_engine()
    fake.config.exploration_planner = "FoV-aware directional frontier"
    fake.mapped_obstacle_points = []
    builder_calls = _spy_planning_costmap_builder(monkeypatch)
    belief_fallback_calls = _spy_belief_to_planning_grid(monkeypatch)

    fake.select_navigation_goal((0.0, 0.0))

    assert len(builder_calls) >= 1, "the real FoV path must invoke PlanningCostmapBuilder"
    assert belief_fallback_calls == [], (
        "the FoV scorer must never fall back to belief.to_planning_grid() once engine.py "
        "supplies a planning_grid"
    )


# ---------------------------------------------------------------------------
# 2. Static observed obstacle.
# ---------------------------------------------------------------------------


def test_static_observed_obstacle_blocks_same_cell_in_fov_and_planning():
    fake = _make_fake_engine()
    fake.config.exploration_planner = "FoV-aware directional frontier"
    static_point = (5.0, 5.0)
    fake.mapped_obstacle_points = [static_point]
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    fake.select_navigation_goal((0.0, 0.0))
    fov_grid = calls[-1]["result"]

    radius = fake.safety_radius()
    dynamic_points = fake._dynamic_obstacle_points_for_robot_object(fake.robot)
    reference_grid = fake.build_planning_grid_for_robot(
        fake.robot, robot_radius=radius, dynamic_obstacle_points=dynamic_points,
    )

    cell = fov_grid.world_to_grid(*static_point)
    assert fov_grid.get_value(cell) == OG_OCCUPIED
    assert reference_grid.get_value(cell) == OG_OCCUPIED


# ---------------------------------------------------------------------------
# 3. Hazard parity.
# ---------------------------------------------------------------------------


def test_hazard_blocks_same_cell_in_fov_and_planning_never_via_obstacle_points():
    fake = _make_fake_engine()
    fake.config.exploration_planner = "FoV-aware directional frontier"
    fake.mapped_obstacle_points = []
    hazard_row, hazard_col = 2, 2
    fake.hazard_service.belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    fake.select_navigation_goal((0.0, 0.0))
    fov_grid = calls[-1]["result"]

    assert fov_grid.get_value(GridCell(hazard_row, hazard_col)) == OG_OCCUPIED

    radius = fake.safety_radius()
    reference_grid = fake.build_planning_grid_for_robot(fake.robot, robot_radius=radius)
    assert reference_grid.get_value(GridCell(hazard_row, hazard_col)) == OG_OCCUPIED

    # Hazard must never enter obstacle_points/mapped_obstacle_points.
    assert fake.mapped_obstacle_points == []
    assert fake.observed_obstacle_snapshot().points == ()


# ---------------------------------------------------------------------------
# 4. Dynamic parity.
# ---------------------------------------------------------------------------


def test_dynamic_other_robot_blocks_fov_consumption_and_multi_robot_planning(monkeypatch):
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    fake.mapped_obstacle_points = []
    other_robot_xy = (5.0, 5.0)

    dynamic_points = fake.dynamic_robot_obstacle_points_for_robot(0)
    assert dynamic_points, "sanity: a second robot exists, so there is something to sample"
    radius = fake.safety_radius_for_robot(fake.robots[0])
    fov_style_grid = fake.build_planning_grid_for_robot(
        fake.robots[0], robot_radius=radius, dynamic_obstacle_points=dynamic_points,
    )

    multi_kwargs, _reason = fake.build_planner_kwargs_for_multi_robot(0)
    multi_grid = multi_kwargs["planning_grid"]

    assert fov_style_grid.get_value(fov_style_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED
    assert multi_grid.get_value(multi_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED

    # Prove the FoV scorer's CONSUMPTION side genuinely uses this
    # dynamic-aware grid when supplied, via the REAL _score_candidate().
    score_grids = _capture_score_candidate_calls(monkeypatch)
    FoVAwareDirectionalFrontierPlanner().select_goal(
        belief_map=fake.belief_map,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        robot_radius=radius,
        planning_grid=fov_style_grid,
    )
    assert len(score_grids) >= 1
    used_grid = score_grids[0]
    assert used_grid.get_value(used_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED


# ---------------------------------------------------------------------------
# 5. Single-robot dynamic empty.
# ---------------------------------------------------------------------------


def test_single_robot_dynamic_obstacle_points_is_empty():
    fake = _make_fake_engine()
    assert fake._dynamic_obstacle_points_for_robot_object(fake.robot) == ()


# ---------------------------------------------------------------------------
# 6. Ground-truth exclusion.
# ---------------------------------------------------------------------------


def test_ground_truth_without_sensing_does_not_block_fov_grid():
    ground_truth_rect = (2.0, 2.0, 2.0, 2.0)  # x, y, width, height
    fake = _make_fake_engine(obstacles=[ground_truth_rect])
    fake.config.exploration_planner = "FoV-aware directional frontier"
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    fake.select_navigation_goal((0.0, 0.0))
    fov_grid = calls[-1]["result"]

    x, y, w, h = ground_truth_rect
    geometry = fake.belief_map.geometry
    steps = 5
    sampled_any = False
    for i in range(steps + 1):
        for j in range(steps + 1):
            cell = geometry.world_to_grid(x + w * i / steps, y + h * j / steps)
            if cell is None:
                continue
            sampled_any = True
            assert fov_grid.get_value(cell) != OG_OCCUPIED, (
                f"cell {cell!r} inside ground-truth rectangle {ground_truth_rect} must not be occupied"
            )
    assert sampled_any, "sanity: the rectangle actually maps to real grid cells"


# ---------------------------------------------------------------------------
# 7. Per-robot sanitization.
# ---------------------------------------------------------------------------


def test_per_robot_sanitization_produces_different_fov_grids():
    shared_points = [(0.0, 0.0), (5.0, 0.0)]  # one point sits exactly at each robot's own position

    fake_a = _make_fake_engine(robot_positions=[(0.0, 0.0)])
    fake_a.config.exploration_planner = "FoV-aware directional frontier"
    fake_a.mapped_obstacle_points = list(shared_points)
    calls_a = _capture_calls(fake_a, "build_planning_grid_for_robot")
    fake_a.select_navigation_goal((0.0, 0.0))
    grid_a = calls_a[-1]["result"]

    fake_b = _make_fake_engine(robot_positions=[(5.0, 0.0)])
    fake_b.config.exploration_planner = "FoV-aware directional frontier"
    fake_b.mapped_obstacle_points = list(shared_points)
    calls_b = _capture_calls(fake_b, "build_planning_grid_for_robot")
    fake_b.select_navigation_goal((5.0, 0.0))
    grid_b = calls_b[-1]["result"]

    assert not np.array_equal(grid_a.data == OG_OCCUPIED, grid_b.data == OG_OCCUPIED), (
        "the SAME static geometry must sanitize differently for two robots at different "
        "positions -- their FoV grids' occupied masks must differ"
    )


# ---------------------------------------------------------------------------
# 8. Same occupied mask.
# ---------------------------------------------------------------------------


def test_same_occupied_mask_between_fov_and_runtime_planning_grid():
    fake = _make_fake_engine()
    fake.config.exploration_planner = "FoV-aware directional frontier"
    static_point = (5.0, 5.0)
    fake.mapped_obstacle_points = [static_point]
    hazard_row, hazard_col = 2, 2
    fake.hazard_service.belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)

    calls = _capture_calls(fake, "build_planning_grid_for_robot")
    fake.select_navigation_goal((0.0, 0.0))
    fov_grid = calls[-1]["result"]

    radius = fake.safety_radius()
    dynamic_points = fake._dynamic_obstacle_points_for_robot_object(fake.robot)
    reference_grid = fake.build_planning_grid_for_robot(
        fake.robot, robot_radius=radius, dynamic_obstacle_points=dynamic_points,
    )

    assert np.array_equal(fov_grid.data == OG_OCCUPIED, reference_grid.data == OG_OCCUPIED)
    assert fov_grid is not reference_grid, "no object-identity requirement -- content parity only"


# ---------------------------------------------------------------------------
# 9. No legacy path.
# ---------------------------------------------------------------------------


def test_fov_path_never_passes_obstacle_points_explicitly():
    fake = _make_fake_engine()
    fake.config.exploration_planner = "FoV-aware directional frontier"
    fake.mapped_obstacle_points = [(3.0, 3.0)]
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    fake.select_navigation_goal((0.0, 0.0))

    assert len(calls) == 1
    assert calls[0]["obstacle_points"] is None, (
        "the FoV path must never pass obstacle_points explicitly -- doing so would select "
        "the LEGACY path instead of the new PlanningCostmapBuilder-backed one"
    )


# ---------------------------------------------------------------------------
# 10. Scoring preservation.
# ---------------------------------------------------------------------------


def test_scoring_result_unchanged_with_no_obstacles_or_hazards():
    """With no static/dynamic/hazard occupancy, the belief-only fallback
    (no planning_grid supplied -- the planner's original, pre-migration
    behavior) and the new builder-backed grid (via the real
    select_navigation_goal() path) must be equivalent, so the
    success/target ranking result existing tests already pin does not
    change."""
    fake = _make_fake_engine()
    fake.config.exploration_planner = "FoV-aware directional frontier"
    fake.mapped_obstacle_points = []

    belief = fake.belief_map
    for row, col in [(8, 8), (8, 9), (9, 8), (9, 9)]:
        belief.mark_free_cell((row, col))

    result_before = FoVAwareDirectionalFrontierPlanner().select_goal(
        belief_map=belief, robot_xy=(0.0, 0.0), robot_heading=0.0, robot_radius=ROBOT_RADIUS,
    )  # OLD behavior: no planning_grid supplied -> belief.to_planning_grid() fallback, unchanged

    calls = _capture_calls(fake, "build_planning_grid_for_robot")
    fake.select_navigation_goal((0.0, 0.0))
    grid = calls[-1]["result"]

    result_after = FoVAwareDirectionalFrontierPlanner().select_goal(
        belief_map=belief, robot_xy=(0.0, 0.0), robot_heading=0.0, robot_radius=ROBOT_RADIUS,
        planning_grid=grid,
    )

    assert result_before.success == result_after.success
    assert result_before.target == result_after.target
