"""
Integration tests: FoV/candidate scoring is wired to the SAME
PlanningCostmapBuilder-backed costmap the rest of the planning runtime
already uses (build_planner_kwargs()/build_planner_kwargs_for_goal()/
build_planner_kwargs_for_multi_robot()/make_exploration_reachability_
check()) -- via a LAZY provider, not an eagerly-built grid.

Design
------
PlannerServices gained planning_grid_provider: Callable[[], OccupancyGrid]
| None -- the exact same optional-callable, per-call-override-with-
instance-fallback pattern already used for is_candidate_reachable.
engine.ensure_planner_services() refreshes it every tick (see
_planning_grid_provider_for_robot()), closing over the current robot but
building NOTHING until actually called. FoVAwareDirectionalFrontierPlanner.
select_goal() is the ONLY planner that ever reads/calls it -- at most once
per real select_goal() invocation, and only when neither an already-built
kwargs["planning_grid"] nor a provider was actually reached in that
priority order. Every other exploration planner receives the same kwarg
via **kwargs and ignores it. engine.select_navigation_goal() -- the
single-robot goal-selection call site build_planner_kwargs() itself uses
-- passes the SAME lazy provider through select_exploration_goal(), never
building a grid eagerly for planners that will never consume it.

Two real call chains reach FoVAwareDirectionalFrontierPlanner.select_goal():
    A. PRIMARY runtime path (what actually runs every tick):
       agent.step() / ExplorationBehavior._pick_next_target()
       -> PlannerServices.select_exploration_target()
       -> select_exploration_goal() -> select_goal().
    B. Independent selector: engine.select_navigation_goal()
       -> select_exploration_goal() -> select_goal() (used by
       build_planner_kwargs()/build_planner_kwargs_for_route-style needs).
Both are exercised for real below -- a direct select_goal(planning_grid=...)
call proves only the CONSUMPTION logic, not that either real caller wires
the provider through, so it is used only for the legacy-fallback/scoring-
preservation tests that are explicitly about direct-call behavior.

Multi-robot frontier ASSIGNMENT does not currently route through this
planner class at all (select_navigation_goal_for_multi_robot() ->
synchronize_multi_frontier_targets() -> MultiRobotCoordinator, a separate
plugin system out of scope here) -- the dynamic-parity test below proves
the PRIMARY path's provider correctly includes another robot's dynamic
points for the target robot, compared against build_planner_kwargs_for_
multi_robot()'s own grid (both built via the same underlying adapter).

Fakes bind the REAL SimulationControllerMixin methods under test. Spies
wrap REAL callables (PlanningCostmapBuilder.build, BeliefMap.
to_planning_grid, the module-level _score_candidate(), and provider
closures built by the real _planning_grid_provider_for_robot()) to observe
what actually happened -- never a reimplementation of what any of them
does.

engine.ensure_planner_services(robot=None) now accepts an explicit target
robot (defaulting to self.robot for single-robot compat) so a caller that
loops over self.robots and calls agent.step() once per robot can refresh
the SAME shared PlannerServices instance's is_candidate_reachable/
planning_grid_provider for the EXACT robot that iteration's agent.step()
is about to run for -- see the "Multi-robot ensure_planner_services(robot)
wiring" tests below, which go through ensure_planner_services(robot) (not
a manually-assembled PlannerServices, and not a direct
_planning_grid_provider_for_robot(robot) call) so they exercise the real
engine wiring, not just its two ingredients in isolation.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.environment.occupancy_grid import OCCUPIED as OG_OCCUPIED
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
import robotics_sim.planning.exploration_planners as exploration_planners_module
from robotics_sim.planning.exploration_planners import FoVAwareDirectionalFrontierPlanner
from robotics_sim.planning.planning_costmap_builder import PlanningCostmapBuilder
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.planner_services import PlannerServices

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
        "_planning_grid_provider_for_robot",
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
        "ensure_planner_services",
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
    proves the FoV path does NOT fall back to it when a planning_grid/
    planning_grid_provider is supplied."""
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


def _run_primary_fov_path(fake: SimpleNamespace, monkeypatch, *, robot=None):
    """Drives the REAL PRIMARY runtime path -- ExplorationBehavior.
    _pick_next_target() -> PlannerServices.select_exploration_target() ->
    FoVAwareDirectionalFrontierPlanner.select_goal() -- using REAL
    RobotAgent/ExplorationBehavior/PlannerServices/RobotObservation
    instances, and returns the exact planning_grid _score_candidate() used.
    planner_services comes from the REAL engine.ensure_planner_services
    (target_robot), not a manually-assembled PlannerServices -- this
    exercises the actual per-robot refresh wiring, not just its two
    ingredients (is_candidate_reachable/planning_grid_provider) in
    isolation.
    """
    target_robot = robot if robot is not None else fake.robot
    score_grids = _capture_score_candidate_calls(monkeypatch)

    planner_services = fake.ensure_planner_services(target_robot)

    agent = RobotAgent(robot_id=0, position=(float(target_robot.x), float(target_robot.y)))
    observation = RobotObservation(
        robot_xy=(float(target_robot.x), float(target_robot.y)),
        robot_heading=float(target_robot.theta),
        robot_radius=fake.safety_radius_for_robot(target_robot),
        belief_map=fake.belief_map,
        planning_grid=None,
        mapped_obstacle_points=list(fake.mapped_obstacle_points),
        dynamic_obstacles=[],
        active_segment_blocked=False,
        predicted_collision=False,
        current_time=0.0,
        grid_resolution=RESOLUTION,
        goal_tolerance=0.25,
        sensor_range=3.0,
        final_goal_xy=None,
        vision_model="LiDAR",
        ipp_distance_penalty=0.0,
        excluded_targets=[],
        route_points_by_robot=[],
    )

    behavior = ExplorationBehavior()
    behavior._pick_next_target(agent, observation, planner_services)

    assert score_grids, "sanity: the FoV planner must have scored at least one candidate"
    return score_grids[-1]


# ---------------------------------------------------------------------------
# 1. Primary runtime path.
# ---------------------------------------------------------------------------


def test_primary_runtime_path_invokes_planning_costmap_builder(monkeypatch):
    """Not a direct select_goal(planning_grid=...) call -- that would only
    prove the consumption logic, not that the real primary path (agent.
    step()'s own ExplorationBehavior/PlannerServices chain) wires the
    builder through."""
    fake = _make_fake_engine()
    fake.mapped_obstacle_points = []
    builder_calls = _spy_planning_costmap_builder(monkeypatch)

    planner_services = fake.ensure_planner_services(fake.robot)

    agent = RobotAgent(robot_id=0, position=(0.0, 0.0))
    observation = RobotObservation(
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        robot_radius=ROBOT_RADIUS,
        belief_map=fake.belief_map,
        planning_grid=None,
        mapped_obstacle_points=[],
        dynamic_obstacles=[],
        active_segment_blocked=False,
        predicted_collision=False,
        current_time=0.0,
        grid_resolution=RESOLUTION,
        goal_tolerance=0.25,
        sensor_range=3.0,
        final_goal_xy=None,
        vision_model="LiDAR",
        ipp_distance_penalty=0.0,
        excluded_targets=[],
        route_points_by_robot=[],
    )

    behavior = ExplorationBehavior()
    behavior._pick_next_target(agent, observation, planner_services)

    assert len(builder_calls) >= 1, (
        "the primary runtime path (ExplorationBehavior -> PlannerServices -> FoV planner) "
        "must invoke PlanningCostmapBuilder"
    )


# ---------------------------------------------------------------------------
# 2. Provider lazy.
# ---------------------------------------------------------------------------


def test_provider_lazy_lifecycle():
    fake = _make_fake_engine()
    fake.mapped_obstacle_points = []
    belief = fake.belief_map
    belief.mark_free_cell((8, 8))  # gives non-FoV planners something to find too

    call_count = {"n": 0}
    real_provider = fake._planning_grid_provider_for_robot(fake.robot)

    def _counting_provider():
        call_count["n"] += 1
        return real_provider()

    # (a) creating the service does not invoke the provider.
    planner_services = PlannerServices()
    assert call_count["n"] == 0

    # (b) registering/refreshing it (what ensure_planner_services() does
    # every tick) only ASSIGNS the callable -- never calls it.
    planner_services.planning_grid_provider = _counting_provider
    assert call_count["n"] == 0

    # (c) a non-FoV planner receives the kwarg via **kwargs but never calls it.
    # final_goal_xy is a real point, not None -- FrontierExplorationPlanner's
    # own frontier_candidates() does `kwargs.get("final_goal_xy", robot_xy)`,
    # which only falls back to robot_xy when the key is OMITTED, not when
    # explicitly None; unrelated to the provider laziness under test here.
    planner_services.select_exploration_target(
        planner_name="Nearest frontier",
        belief_map=belief,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=(0.0, 0.0),
        robot_radius=ROBOT_RADIUS,
        sensor_range=3.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.0,
    )
    assert call_count["n"] == 0, "a non-FoV planner must never invoke planning_grid_provider"

    # (d) an FoV selection invokes it exactly once.
    planner_services.select_exploration_target(
        planner_name="FoV-aware directional frontier",
        belief_map=belief,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_radius=ROBOT_RADIUS,
        sensor_range=3.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.0,
    )
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# 3. Direct engine path.
# ---------------------------------------------------------------------------


def test_select_navigation_goal_fov_invokes_provider_once_and_uses_builder():
    fake = _make_fake_engine()
    fake.config.exploration_planner = "FoV-aware directional frontier"
    fake.mapped_obstacle_points = [(5.0, 5.0)]

    real_provider_factory = fake._planning_grid_provider_for_robot
    invocation_count = {"n": 0}

    def _spy_provider_factory(robot):
        real_provider = real_provider_factory(robot)
        if real_provider is None:
            return None

        def _counting_provider():
            invocation_count["n"] += 1
            return real_provider()

        return _counting_provider

    fake._planning_grid_provider_for_robot = _spy_provider_factory
    calls = _capture_calls(fake, "build_planning_grid_for_robot")

    fake.select_navigation_goal((0.0, 0.0))

    assert invocation_count["n"] == 1, "the provider must be invoked exactly once"
    assert len(calls) == 1, "the builder-backed path must have run exactly once"
    assert calls[0]["obstacle_points"] is None, (
        "select_navigation_goal() must never pass obstacle_points explicitly"
    )


# ---------------------------------------------------------------------------
# 4. Static parity.
# ---------------------------------------------------------------------------


def test_static_observed_obstacle_blocks_same_cell_primary_path_and_planning(monkeypatch):
    fake = _make_fake_engine()
    static_point = (5.0, 5.0)
    fake.mapped_obstacle_points = [static_point]

    fov_grid = _run_primary_fov_path(fake, monkeypatch)

    radius = fake.safety_radius()
    dynamic_points = fake._dynamic_obstacle_points_for_robot_object(fake.robot)
    reference_grid = fake.build_planning_grid_for_robot(
        fake.robot, robot_radius=radius, dynamic_obstacle_points=dynamic_points,
    )

    cell = fov_grid.world_to_grid(*static_point)
    assert fov_grid.get_value(cell) == OG_OCCUPIED
    assert reference_grid.get_value(cell) == OG_OCCUPIED


# ---------------------------------------------------------------------------
# 5. Hazard parity.
# ---------------------------------------------------------------------------


def test_hazard_blocks_same_cell_primary_path_and_planning_never_via_obstacle_points(monkeypatch):
    fake = _make_fake_engine()
    fake.mapped_obstacle_points = []
    hazard_row, hazard_col = 2, 2
    fake.hazard_service.belief.observe_cells([hazard_row], [hazard_col], [0.9], robot_index=0)

    fov_grid = _run_primary_fov_path(fake, monkeypatch)
    assert fov_grid.get_value(GridCell(hazard_row, hazard_col)) == OG_OCCUPIED

    radius = fake.safety_radius()
    reference_grid = fake.build_planning_grid_for_robot(fake.robot, robot_radius=radius)
    assert reference_grid.get_value(GridCell(hazard_row, hazard_col)) == OG_OCCUPIED

    assert fake.mapped_obstacle_points == []
    assert fake.observed_obstacle_snapshot().points == ()


# ---------------------------------------------------------------------------
# 6. Dynamic parity.
# ---------------------------------------------------------------------------


def test_dynamic_other_robot_blocks_primary_path_via_provider(monkeypatch):
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 5.0)])
    fake.mapped_obstacle_points = []
    other_robot_xy = (5.0, 5.0)

    # Goes through PlannerServices (via _run_primary_fov_path's
    # ExplorationBehavior._pick_next_target() call) -- never a
    # manually-built grid handed directly to the planner.
    fov_grid = _run_primary_fov_path(fake, monkeypatch, robot=fake.robots[0])
    assert fov_grid.get_value(fov_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED

    multi_kwargs, _reason = fake.build_planner_kwargs_for_multi_robot(0)
    multi_grid = multi_kwargs["planning_grid"]
    assert multi_grid.get_value(multi_grid.world_to_grid(*other_robot_xy)) == OG_OCCUPIED


# ---------------------------------------------------------------------------
# 7. Ground-truth exclusion.
# ---------------------------------------------------------------------------


def test_ground_truth_without_sensing_does_not_block_fov_costmap(monkeypatch):
    ground_truth_rect = (2.0, 2.0, 2.0, 2.0)  # x, y, width, height
    fake = _make_fake_engine(obstacles=[ground_truth_rect])

    fov_grid = _run_primary_fov_path(fake, monkeypatch)

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
# 8. Per-robot sanitization.
# ---------------------------------------------------------------------------


def test_per_robot_sanitization_produces_different_provider_grids(monkeypatch):
    shared_points = [(0.0, 0.0), (5.0, 0.0)]  # one point sits exactly at each robot's own position

    fake_a = _make_fake_engine(robot_positions=[(0.0, 0.0)])
    fake_a.mapped_obstacle_points = list(shared_points)
    grid_a = _run_primary_fov_path(fake_a, monkeypatch)

    fake_b = _make_fake_engine(robot_positions=[(5.0, 0.0)])
    fake_b.mapped_obstacle_points = list(shared_points)
    grid_b = _run_primary_fov_path(fake_b, monkeypatch)

    assert not np.array_equal(grid_a.data == OG_OCCUPIED, grid_b.data == OG_OCCUPIED), (
        "the SAME static geometry must sanitize differently for two robots at different "
        "positions -- their provider-built FoV grids' occupied masks must differ"
    )


# ---------------------------------------------------------------------------
# 9. Legacy fallback.
# ---------------------------------------------------------------------------


def test_direct_call_without_grid_or_provider_uses_legacy_fallback(monkeypatch):
    belief = BeliefMap(bounds=(-10.0, 10.0, -8.0, 8.0), resolution=RESOLUTION, robot_count=1)
    belief.mark_free_cell((8, 8))
    fallback_calls = _spy_belief_to_planning_grid(monkeypatch)

    FoVAwareDirectionalFrontierPlanner().select_goal(
        belief_map=belief, robot_xy=(0.0, 0.0), robot_heading=0.0, robot_radius=ROBOT_RADIUS,
    )

    assert len(fallback_calls) == 1, (
        "a direct call with neither planning_grid nor planning_grid_provider must still use "
        "belief.to_planning_grid() -- unchanged legacy fallback"
    )


# ---------------------------------------------------------------------------
# 10. Scoring preservation.
# ---------------------------------------------------------------------------


def test_scoring_result_unchanged_between_provider_and_legacy_fallback_when_empty():
    """In an empty scene (no obstacles/hazards), the unified provider and
    the legacy belief-only fallback must produce the same target/result."""
    fake = _make_fake_engine()
    fake.mapped_obstacle_points = []
    belief = fake.belief_map
    for row, col in [(8, 8), (8, 9), (9, 8), (9, 9)]:
        belief.mark_free_cell((row, col))

    result_legacy = FoVAwareDirectionalFrontierPlanner().select_goal(
        belief_map=belief, robot_xy=(0.0, 0.0), robot_heading=0.0, robot_radius=ROBOT_RADIUS,
    )  # neither planning_grid nor planning_grid_provider supplied -> legacy fallback

    provider = fake._planning_grid_provider_for_robot(fake.robot)
    result_provider = FoVAwareDirectionalFrontierPlanner().select_goal(
        belief_map=belief, robot_xy=(0.0, 0.0), robot_heading=0.0, robot_radius=ROBOT_RADIUS,
        planning_grid_provider=provider,
    )

    assert result_legacy.success == result_provider.success
    assert result_legacy.target == result_provider.target


# ---------------------------------------------------------------------------
# Multi-robot ensure_planner_services(robot) wiring.
#
# self.robot is only ever ONE of possibly several robots -- it does not
# track "the robot the current loop iteration is for". A caller that loops
# over self.robots and calls agent.step() once per robot must pass that
# robot explicitly to ensure_planner_services(robot) right before each
# agent.step() call, or every robot's agent.step() silently receives a
# provider/reachability check built for the same (wrong) robot. These
# tests go through the real ensure_planner_services(robot) -- never a
# direct _planning_grid_provider_for_robot(robot) call -- because that
# would only prove the closure factory works, not that the engine's own
# per-robot refresh call site is correct.
# ---------------------------------------------------------------------------


# 11. Single-robot compatibility.


def test_ensure_planner_services_no_arg_uses_self_robot_and_stays_lazy():
    fake = _make_fake_engine()
    fake.mapped_obstacle_points = [(5.0, 5.0)]

    received_robots = []
    real_factory = fake._planning_grid_provider_for_robot

    def _spy_factory(robot):
        received_robots.append(robot)
        return real_factory(robot)

    fake._planning_grid_provider_for_robot = _spy_factory
    build_calls = _capture_calls(fake, "build_planning_grid_for_robot")

    services = fake.ensure_planner_services()

    assert received_robots == [fake.robot], (
        "ensure_planner_services() with no explicit robot must resolve target_robot from self.robot"
    )
    assert services.planning_grid_provider is not None
    assert build_calls == [], (
        "creating/refreshing services must never build a grid -- the provider stays uninvoked"
    )


# 12. Provider per robot.


def test_ensure_planner_services_per_robot_provider_differs():
    shared_points = [(0.0, 0.0), (5.0, 0.0)]  # one point sits exactly at each robot's own position
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 0.0)])
    fake.mapped_obstacle_points = list(shared_points)
    r1, r2 = fake.robots

    # Capture each closure BEFORE the next ensure_planner_services() call
    # overwrites the shared instance's planning_grid_provider attribute.
    services = fake.ensure_planner_services(r1)
    provider_r1 = services.planning_grid_provider
    assert provider_r1 is not None

    services_again = fake.ensure_planner_services(r2)
    provider_r2 = services_again.planning_grid_provider
    assert provider_r2 is not None
    assert provider_r2 is not provider_r1

    assert services_again is services, (
        "the SAME shared PlannerServices instance must be reused across robots, "
        "not a new instance per robot"
    )

    grid_r1 = provider_r1()
    grid_r2 = provider_r2()
    assert not np.array_equal(grid_r1.data == OG_OCCUPIED, grid_r2.data == OG_OCCUPIED), (
        "the SAME static geometry sanitized for two different target robots must differ"
    )


# 13. Dynamic obstacles resolve to the correct robot.


def test_ensure_planner_services_per_robot_provider_blocks_the_other_robot():
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 0.0)])
    fake.mapped_obstacle_points = []
    r1, r2 = fake.robots

    provider_r1 = fake.ensure_planner_services(r1).planning_grid_provider
    provider_r2 = fake.ensure_planner_services(r2).planning_grid_provider

    grid_r1 = provider_r1()
    grid_r2 = provider_r2()

    assert grid_r1.get_value(grid_r1.world_to_grid(5.0, 0.0)) == OG_OCCUPIED, (
        "R1's provider-built grid must block R2's position as a dynamic obstacle"
    )
    assert grid_r2.get_value(grid_r2.world_to_grid(0.0, 0.0)) == OG_OCCUPIED, (
        "R2's provider-built grid must block R1's position as a dynamic obstacle"
    )


# 14. Real multi-robot loop wiring (not a direct factory call).


def test_multi_robot_loop_slice_refreshes_provider_per_robot_before_each_agent_step(monkeypatch):
    """Minimal real slice of a multi-robot loop: for each robot, refresh
    PlannerServices via the real ensure_planner_services(robot) and then
    call the REAL agent.step() -- proving both that ensure_planner_services
    is called once per robot in iteration order, and that each agent.step()
    actually scores candidates against the grid built for ITS OWN robot,
    not a stale one left over from a previous robot's iteration."""
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 0.0)])
    fake.mapped_obstacle_points = [(0.0, 0.0), (5.0, 0.0)]
    r1, r2 = fake.robots

    ensure_calls = []
    real_ensure = fake.ensure_planner_services

    def _spy_ensure(robot=None):
        ensure_calls.append(robot)
        return real_ensure(robot)

    fake.ensure_planner_services = _spy_ensure

    score_grids = _capture_score_candidate_calls(monkeypatch)
    grids_by_robot = []

    for robot in (r1, r2):
        planner_services = fake.ensure_planner_services(robot)
        agent = RobotAgent(robot_id=id(robot), position=(float(robot.x), float(robot.y)))
        observation = RobotObservation(
            robot_xy=(float(robot.x), float(robot.y)),
            robot_heading=float(robot.theta),
            robot_radius=fake.safety_radius_for_robot(robot),
            belief_map=fake.belief_map,
            planning_grid=None,
            mapped_obstacle_points=list(fake.mapped_obstacle_points),
            dynamic_obstacles=[],
            active_segment_blocked=False,
            predicted_collision=False,
            current_time=0.0,
            grid_resolution=RESOLUTION,
            goal_tolerance=0.25,
            sensor_range=3.0,
            # A real point, not None: an obstacle sits at each robot's own
            # position here (see mapped_obstacle_points above), so the
            # primary FoV search can come back empty and fall through to
            # ExplorationBehavior's map-wide fallback planner, whose
            # frontier_candidates() does kwargs.get("final_goal_xy",
            # robot_xy) -- that only falls back to robot_xy when the key is
            # OMITTED, not when explicitly None. Unrelated to the per-robot
            # provider wiring under test here.
            final_goal_xy=(0.0, 0.0),
            vision_model="LiDAR",
            ipp_distance_penalty=0.0,
            excluded_targets=[],
            route_points_by_robot=[],
        )
        before = len(score_grids)
        agent.step(observation, planner_services, dt=0.1)
        assert len(score_grids) > before, (
            "agent.step() must have scored at least one candidate via the FoV planner"
        )
        grids_by_robot.append(score_grids[-1])

    assert ensure_calls == [r1, r2], (
        "ensure_planner_services() must be called once per robot, in loop order R1 then R2"
    )

    grid_for_r1, grid_for_r2 = grids_by_robot
    assert grid_for_r1.get_value(grid_for_r1.world_to_grid(5.0, 0.0)) == OG_OCCUPIED, (
        "R1's agent.step() must have scored against a grid where R2 is a dynamic obstacle"
    )
    assert grid_for_r2.get_value(grid_for_r2.world_to_grid(0.0, 0.0)) == OG_OCCUPIED, (
        "R2's agent.step() must have scored against a grid where R1 is a dynamic obstacle"
    )


# 15. Reachability/provider share the same target robot.


def test_ensure_planner_services_reachability_and_provider_share_target_robot():
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 0.0)])
    r1, r2 = fake.robots

    reachability_robots = []
    provider_robots = []
    real_reachability_factory = fake.make_exploration_reachability_check
    real_provider_factory = fake._planning_grid_provider_for_robot

    def _spy_reachability(robot):
        reachability_robots.append(robot)
        return real_reachability_factory(robot)

    def _spy_provider(robot):
        provider_robots.append(robot)
        return real_provider_factory(robot)

    fake.make_exploration_reachability_check = _spy_reachability
    fake._planning_grid_provider_for_robot = _spy_provider

    fake.ensure_planner_services(r1)
    fake.ensure_planner_services(r2)

    assert reachability_robots == [r1, r2], (
        "make_exploration_reachability_check() must be called with the same target_robot, in order"
    )
    assert provider_robots == [r1, r2], (
        "_planning_grid_provider_for_robot() must be called with the same target_robot, in order"
    )


# 16. Refreshing services never builds a costmap -- only invoking the provider does.


def test_ensure_planner_services_refresh_never_builds_costmap(monkeypatch):
    fake = _make_fake_engine(robot_positions=[(0.0, 0.0), (5.0, 0.0)])
    r1, r2 = fake.robots
    builder_calls = _spy_planning_costmap_builder(monkeypatch)

    services = fake.ensure_planner_services(r1)
    provider_r1 = services.planning_grid_provider
    services = fake.ensure_planner_services(r2)
    provider_r2 = services.planning_grid_provider

    assert builder_calls == [], (
        "refreshing PlannerServices for either robot must never call PlanningCostmapBuilder.build()"
    )

    provider_r1()
    assert len(builder_calls) == 1, "invoking the provider must be the only thing that builds a costmap"
    provider_r2()
    assert len(builder_calls) == 2
