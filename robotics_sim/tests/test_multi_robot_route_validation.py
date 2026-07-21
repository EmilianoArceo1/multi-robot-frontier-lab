from __future__ import annotations

from types import SimpleNamespace

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.plugins import build_runtime_profile
from robotics_sim.planning.coordinated_frontier_planner import validate_multi_robot_corridor
from robotics_sim.simulation import engine as engine_module
from robotics_sim.simulation.coordination import select_runtime_path_source
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.plugin_loader import load_coordination_plugin
from robotics_sim.simulation.telemetry import TelemetryLogger
from robotics_sim.environment.collision_checker import CollisionChecker


def test_direct_route_crossing_robot_safety_is_rejected_before_active():
    """A Direct (single-segment) corridor is still a full corridor to validate."""
    result = validate_multi_robot_corridor(
        start=(0.0, 0.0),
        waypoints=[(4.0, 0.0)],
        ego_safety_radius=0.35,
        other_robot_disks=[(2.0, 0.0, 0.35)],
        margin=0.25,
    )

    assert result.is_valid is False
    assert result.reason_code == "route_conflict_with_robot_safety_zone"


def test_non_conflicting_direct_routes_are_accepted():
    result = validate_multi_robot_corridor(
        start=(0.0, 0.0),
        waypoints=[(4.0, 0.0)],
        ego_safety_radius=0.35,
        other_robot_disks=[(2.0, 5.0, 0.35)],
        margin=0.25,
    )

    assert result.is_valid is True
    assert result.reason_code == ""


def test_corridor_starting_close_to_teammate_is_not_a_false_positive():
    """Two robots spawning side by side is a formation, not a corridor crossing."""
    result = validate_multi_robot_corridor(
        start=(0.0, 0.0),
        waypoints=[(4.0, 0.0)],
        ego_safety_radius=0.35,
        other_robot_disks=[(0.2, 0.0, 0.35)],
        margin=0.25,
    )

    assert result.is_valid is True


def test_route_crossing_active_teammate_route_is_rejected():
    result = validate_multi_robot_corridor(
        start=(0.0, 0.0),
        waypoints=[(0.0, 4.0)],
        ego_safety_radius=0.35,
        other_robot_disks=[(10.0, 10.0, 0.35)],
        other_routes=[[(-2.0, 2.0), (2.0, 2.0)]],
        margin=0.25,
    )

    assert result.is_valid is False
    assert result.reason_code == "route_conflict_with_active_route"


def test_reserved_corridor_conflict_is_detected():
    result = validate_multi_robot_corridor(
        start=(0.0, 0.0),
        waypoints=[(0.0, 4.0)],
        ego_safety_radius=0.35,
        reserved_corridors=[[(-2.0, 2.0), (2.0, 2.0)]],
        margin=0.25,
    )

    assert result.is_valid is False
    assert result.reason_code == "corridor_reservation_conflict"


def test_rejected_route_target_is_blacklisted_for_replan_round():
    """invalidate_current_multi_frontier() is the blacklist mechanism the
    engine calls when a corridor is rejected -- it must clear the target and
    remember it for this replanning round."""
    fake_self = SimpleNamespace(
        robots=[object()],
        multi_exploration_targets=[(4.0, 0.0)],
        multi_invalidated_exploration_targets=[[]],
    )
    fake_self.publish_multi_exploration_targets = SimulationControllerMixin.publish_multi_exploration_targets.__get__(
        fake_self
    )
    fake_self.ensure_multi_exploration_target_slots = (
        SimulationControllerMixin.ensure_multi_exploration_target_slots.__get__(fake_self)
    )

    SimulationControllerMixin.invalidate_current_multi_frontier(
        fake_self, 0, "route_conflict_with_robot_safety_zone"
    )

    assert fake_self.multi_exploration_targets[0] is None
    assert (4.0, 0.0) in fake_self.multi_invalidated_exploration_targets[0]


def test_mmpf_targets_still_use_external_path_planner_when_no_path_planning_capability():
    """Corridor validation is a runtime/engine concern; it must not change
    which plugin owns PATH_PLANNING. MMPF still does not own it."""
    profile = build_runtime_profile(load_coordination_plugin(MMPF_COORDINATOR).metadata)
    assert profile.owns_path_planning is False

    command = RobotCommand(
        robot_id=0,
        status="ASSIGNED",
        target=(3.0, 0.0),
        path=((0.0, 0.0), (3.0, 0.0)),
    )
    legacy_calls: list[bool] = []

    def legacy_provider():
        legacy_calls.append(True)
        return True, "legacy A* route", [(3.0, 0.0)]

    success, reason, waypoints = select_runtime_path_source(profile, command, legacy_provider)

    assert legacy_calls == [True]
    assert waypoints == [(3.0, 0.0)]


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = waypoints


def _build_fake_engine(*, planner_type: str = "A*", route_plan_responses):
    """A minimal SimulationControllerMixin instance for exercising the real
    _assign_route_to_multi_robot_with_corridor_validation()/hold_multi_robot_position()
    control flow without a full engine (no Qt, no collision checker).

    Only the "heavy" collaborators (route computation, corridor validation,
    the A* fallback's inner planner call) are faked; everything else that is
    cheap (state bookkeeping, blacklisting, logging) runs for real.
    """
    robot = _FakeRobot(x=0.0, y=0.0)
    fake = SimpleNamespace(
        robots=[robot],
        robot=robot,
        config=SimpleNamespace(
            planner_type=planner_type,
            path_simplifier="Direction changes",
            exploration_planner="test exploration planner",
        ),
        mapped_obstacle_points=[],
        multi_exploration_targets=[(4.0, 0.0)],
        multi_invalidated_exploration_targets=[[]],
        multi_planned_path_points=[[]],
        route_request_count=0,
        route_result_count=0,
        last_goal_selection_reason="",
        console_logs=[],
        collision_checker=CollisionChecker(),
    )

    fake.is_exploration_mode = lambda: True
    fake.dynamic_robot_obstacle_points_for_robot = lambda index: []
    fake.dynamic_robot_obstacles_for_target_selection = lambda index: []
    fake.multi_active_route_points_by_robot = lambda: [[]]
    fake.multi_dynamic_target_margin = lambda: 0.25
    fake.safety_radius_for_robot = lambda robot: 0.35
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    # log_route_assignment() (bound for real below) reads self.telemetry;
    # SimpleNamespace does not resolve @property descriptors from
    # SimulationControllerMixin, so it is provided directly as a plain
    # instance attribute wired to the same fake console sink.
    fake.telemetry = TelemetryLogger(sink=fake.log_console_message)
    # None of these tests exercise a PATH_PLANNING-owning plugin; the Direct
    # ->A* fallback must still be reachable, so owns_path_planning is False.
    fake.coordinator_runtime_profile = lambda: SimpleNamespace(owns_path_planning=False)

    responses = list(route_plan_responses)

    def fake_compute_route_for_multi_robot(index, force_new_exploration_target=False):
        if len(responses) > 1:
            return responses.pop(0)
        return responses[0]

    fake.compute_route_for_multi_robot = fake_compute_route_for_multi_robot
    fake.route_plan_call_count = lambda: len(route_plan_responses) - len(responses) + (
        1 if len(responses) == len(route_plan_responses) else 0
    )

    # build_planner_kwargs_for_multi_robot / call_compute_planned_waypoints are
    # only reached by the real compute_grid_safe_fallback_route_for_multi_robot
    # when planner_type == "Direct"; tests that do not exercise the fallback
    # never call these.
    fake.build_planner_kwargs_for_multi_robot = lambda index, force_new_exploration_target=False: (
        {"goal_xy": (4.0, 0.0)},
        "goal_reason",
    )
    fake.call_compute_planned_waypoints = lambda kwargs, path_simplifier=None: (
        True,
        "A* fallback route",
        [(0.0, 2.0), (4.0, 0.0)],
    )

    for name in (
        "ROUTE_STATE_ACTIVE",
        "ROUTE_STATE_HOLD_NO_FRONTIER",
        "ROUTE_STATE_STUCK_SAFETY",
        "ROUTE_STATE_ESCAPE_LOCAL",
        "ROUTE_STATE_HOLD_ROUTE_BLOCKED",
        "ROUTE_STATE_WAITING_FOR_CORRIDOR",
        "MAX_ROUTE_RECOVERY_ATTEMPTS",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name))

    for name in (
        "hold_multi_robot_position",
        "ensure_multi_exploration_target_slots",
        "publish_multi_exploration_targets",
        "set_multi_route_state",
        "ensure_multi_route_state_slots",
        "invalidate_current_multi_frontier",
        "assign_route_to_multi_robot",
        "_assign_route_to_multi_robot_with_corridor_validation",
        "_activate_multi_robot_route",
        "compute_grid_safe_fallback_route_for_multi_robot",
        "set_robot_goal_or_waypoints",
        "clean_waypoints_for_robot",
        "route_points_intersect_new_map_information",
        "log_route_assignment",
        "_xy_text",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    return fake


def test_hold_no_frontier_only_when_no_candidates_exist(monkeypatch):
    """No plugin candidate at all -> HOLD_NO_FRONTIER is correct here."""
    fake = _build_fake_engine(route_plan_responses=[(False, "no target assigned yet", [])])

    result = fake.assign_route_to_multi_robot(0)

    assert result is True  # hold_multi_robot_position always returns True
    assert fake.multi_route_states[0] == SimulationControllerMixin.ROUTE_STATE_HOLD_NO_FRONTIER


def test_route_conflict_does_not_report_no_frontier(monkeypatch):
    """A real frontier can have no currently safe departure corridor.

    After bounded alternative-target retries this is HOLD_ROUTE_BLOCKED, not
    the semantically false HOLD_NO_FRONTIER. CQLite separately guarantees that
    clearing this failed target is not learned as successful exploration.
    """
    fake = _build_fake_engine(
        planner_type="A*",
        route_plan_responses=[(True, "target found", [(4.0, 0.0)])],
    )
    monkeypatch.setattr(
        engine_module,
        "validate_multi_robot_corridor",
        lambda **kwargs: validate_multi_robot_corridor(
            start=kwargs["start"],
            waypoints=kwargs["waypoints"],
            ego_safety_radius=kwargs["ego_safety_radius"],
            other_robot_disks=[(2.0, 0.0, 0.35)],
            margin=kwargs.get("margin", 0.25),
        ),
    )

    result = fake.assign_route_to_multi_robot(0)

    assert result is True
    assert fake.multi_route_states[0] == SimulationControllerMixin.ROUTE_STATE_HOLD_ROUTE_BLOCKED
    assert fake.multi_route_states[0] != SimulationControllerMixin.ROUTE_STATE_HOLD_NO_FRONTIER
    assert fake.multi_exploration_targets[0] is None
    assert fake.multi_invalidated_exploration_targets[0] == [(4.0, 0.0)]
    assert any("HOLD_ROUTE_BLOCKED" in message for message in fake.console_logs)


def test_route_blocked_waits_instead_of_false_no_frontier(monkeypatch):
    """Every candidate crosses a teammate's active route specifically -> this
    is transient (the teammate is moving), so it should wait rather than
    report a permanent no-frontier hold."""
    fake = _build_fake_engine(
        planner_type="A*",
        route_plan_responses=[(True, "target found", [(0.0, 4.0)])],
    )
    monkeypatch.setattr(
        engine_module,
        "validate_multi_robot_corridor",
        lambda **kwargs: validate_multi_robot_corridor(
            start=kwargs["start"],
            waypoints=kwargs["waypoints"],
            ego_safety_radius=kwargs["ego_safety_radius"],
            other_routes=[[(-2.0, 2.0), (2.0, 2.0)]],
            margin=kwargs.get("margin", 0.25),
        ),
    )

    result = fake.assign_route_to_multi_robot(0)

    assert result is True
    assert fake.multi_route_states[0] == SimulationControllerMixin.ROUTE_STATE_WAITING_FOR_CORRIDOR
    assert any("waiting for corridor" in message for message in fake.console_logs)


def test_recovery_tries_multiple_candidate_targets_before_hold(monkeypatch):
    """The first two candidates are blocked; the third is clear. The runtime
    must try all three before giving up (MAX_ROUTE_RECOVERY_ATTEMPTS=3), and
    must activate the third candidate instead of holding."""
    fake = _build_fake_engine(
        planner_type="A*",
        route_plan_responses=[
            (True, "candidate 1", [(4.0, 0.0)]),
            (True, "candidate 2", [(0.0, 4.0)]),
            (True, "candidate 3 (clear)", [(9.0, 9.0)]),
        ],
    )

    def fake_validate(**kwargs):
        # candidate 1/2 conflict with a non-transient reserved corridor;
        # candidate 3 is clear. This isolates the bounded alternative-target
        # recovery path from the current-disk case covered by the prior test.
        blocked = {(4.0, 0.0), (0.0, 4.0)}
        target = tuple(kwargs["waypoints"][-1])
        if target in blocked:
            reservation = (
                [(2.0, -1.0), (2.0, 1.0)]
                if target == (4.0, 0.0)
                else [(-1.0, 2.0), (1.0, 2.0)]
            )
            return validate_multi_robot_corridor(
                start=kwargs["start"],
                waypoints=kwargs["waypoints"],
                ego_safety_radius=kwargs["ego_safety_radius"],
                reserved_corridors=[reservation],
                margin=kwargs.get("margin", 0.25),
            )
        return validate_multi_robot_corridor(
            start=kwargs["start"],
            waypoints=kwargs["waypoints"],
            ego_safety_radius=kwargs["ego_safety_radius"],
            margin=kwargs.get("margin", 0.25),
        )

    monkeypatch.setattr(engine_module, "validate_multi_robot_corridor", fake_validate)

    result = fake.assign_route_to_multi_robot(0)

    assert result is True
    assert fake.multi_route_states[0] == SimulationControllerMixin.ROUTE_STATE_ACTIVE
    assert fake.multi_planned_path_points[0][-1] == (9.0, 9.0)
    assert any("trying alternative target 2/3" in message for message in fake.console_logs)
    assert any("trying alternative target 3/3" in message for message in fake.console_logs)


def test_current_robot_disks_only_gate_the_immediately_executable_segment(monkeypatch):
    """An untimed current pose must not reserve every future route segment.

    The second segment passes through where a teammate is now, but the first
    segment is clear. Runtime safety will re-check the second segment later.
    """
    fake = _build_fake_engine(
        planner_type="A*",
        route_plan_responses=[(True, "detour", [(0.0, 2.0), (4.0, 0.0)])],
    )
    fake._preserve_next_route_waypoints = True
    seen_waypoints = []

    def capture_validation(**kwargs):
        seen_waypoints.append(list(kwargs["waypoints"]))
        return validate_multi_robot_corridor(
            start=kwargs["start"],
            waypoints=kwargs["waypoints"],
            ego_safety_radius=kwargs["ego_safety_radius"],
            other_robot_disks=[(2.0, 0.0, 0.35)],
            margin=kwargs.get("margin", 0.25),
        )

    monkeypatch.setattr(engine_module, "validate_multi_robot_corridor", capture_validation)

    result = fake.assign_route_to_multi_robot(0)

    assert result is True
    assert seen_waypoints == [[(0.0, 2.0)]]
    assert fake.multi_route_states[0] == SimulationControllerMixin.ROUTE_STATE_ACTIVE
    assert fake.multi_planned_path_points[0] == [(0.0, 0.0), (0.0, 2.0), (4.0, 0.0)]


def test_astar_retries_static_topology_when_teammates_disconnect_dynamic_grid():
    """Dynamic robot rings may close a doorway globally, but safety is handled
    later by exact corridor/disk checks, so A* retains the static route for a
    transient wait instead of reporting that no route exists."""
    robot = _FakeRobot(x=0.0, y=0.0)
    dynamic_grid = object()
    static_grid = object()
    planner_calls = []
    static_grid_calls = []
    fake = SimpleNamespace(
        robots=[robot],
        config=SimpleNamespace(planner_type="A*", path_simplifier="Direction changes"),
        multi_robot_commands_by_id={},
    )
    fake.build_planner_kwargs_for_multi_robot = lambda index, force_new_exploration_target=False: (
        {
            "planner_type": "A*",
            "start_xy": (0.0, 0.0),
            "goal_xy": (4.0, 0.0),
            "robot_radius": 0.35,
            "planning_grid": dynamic_grid,
        },
        "frontier assigned",
    )
    fake.dynamic_robot_obstacle_points_for_robot = lambda index: [(1.0, 0.0)]

    def build_grid(robot_arg, *, robot_radius, dynamic_obstacle_points):
        static_grid_calls.append((robot_arg, robot_radius, dynamic_obstacle_points))
        return static_grid

    def compute_waypoints(kwargs, *, path_simplifier, debug_capture):
        planner_calls.append(kwargs["planning_grid"])
        if kwargs["planning_grid"] is dynamic_grid:
            return False, "no path in dynamic grid", []
        return True, "path found", [(2.0, 1.0), (4.0, 0.0)]

    fake.build_planning_grid_for_robot = build_grid
    fake.call_compute_planned_waypoints = compute_waypoints
    fake.coordinator_runtime_profile = lambda: SimpleNamespace(owns_path_planning=False)
    fake.log_console_message = lambda message: None
    fake.compute_route_for_multi_robot = SimulationControllerMixin.compute_route_for_multi_robot.__get__(fake)

    success, reason, waypoints = fake.compute_route_for_multi_robot(0)

    assert success is True
    assert planner_calls == [dynamic_grid, static_grid]
    assert static_grid_calls == [(robot, 0.35, ())]
    assert waypoints == [(2.0, 1.0), (4.0, 0.0)]
    assert "dynamic occupancy disconnected planning grid" in reason


def test_astar_refines_static_grid_when_coarse_rasterization_closes_passage():
    robot = _FakeRobot(x=0.0, y=0.0)
    dynamic_grid = object()
    coarse_static_grid = object()
    refined_static_grid = SimpleNamespace(resolution=0.125)
    planner_calls = []
    fake = SimpleNamespace(
        robots=[robot],
        config=SimpleNamespace(planner_type="A*", path_simplifier="Direction changes"),
        multi_robot_commands_by_id={},
    )
    fake.build_planner_kwargs_for_multi_robot = lambda index, force_new_exploration_target=False: (
        {
            "planner_type": "A*",
            "start_xy": (0.0, 0.0),
            "goal_xy": (4.0, 0.0),
            "robot_radius": 0.22,
            "planning_grid": dynamic_grid,
            "resolution": 0.25,
        },
        "frontier assigned",
    )
    fake.dynamic_robot_obstacle_points_for_robot = lambda index: [(1.0, 0.0)]
    fake.build_planning_grid_for_robot = lambda *args, **kwargs: coarse_static_grid
    fake.build_refined_static_planning_grid_for_robot = lambda *args, **kwargs: refined_static_grid

    def compute_waypoints(kwargs, *, path_simplifier, debug_capture):
        planner_calls.append(kwargs["planning_grid"])
        if kwargs["planning_grid"] is refined_static_grid:
            assert kwargs["resolution"] == 0.125
            return True, "path found through geometric gap", [(2.0, -1.0), (4.0, 0.0)]
        return False, "no path found", []

    fake.call_compute_planned_waypoints = compute_waypoints
    fake.coordinator_runtime_profile = lambda: SimpleNamespace(owns_path_planning=False)
    fake.log_console_message = lambda message: None
    fake.compute_route_for_multi_robot = SimulationControllerMixin.compute_route_for_multi_robot.__get__(fake)

    success, reason, waypoints = fake.compute_route_for_multi_robot(0)

    assert success is True
    assert planner_calls == [dynamic_grid, coarse_static_grid, refined_static_grid]
    assert waypoints == [(2.0, -1.0), (4.0, 0.0)]
    assert "refined static-topology fallback (0.125 m)" in reason


def test_direct_route_rejected_can_try_astar_fallback(monkeypatch):
    """Direct's straight-line corridor is blocked, but the grid-safe A*
    fallback route is clear -- the runtime must use the fallback instead of
    immediately invalidating the target and asking for a new one."""
    fake = _build_fake_engine(
        planner_type="Direct",
        route_plan_responses=[(True, "direct route", [(4.0, 0.0)])],
    )

    # Keep the synthetic detour vertices: this test's mocked validator, rather
    # than a mapped point cloud, is the source of the straight-line obstacle.
    fake._preserve_next_route_waypoints = True

    def fake_validate(**kwargs):
        if tuple(kwargs["waypoints"]) == ((4.0, 0.0),):
            return validate_multi_robot_corridor(
                start=kwargs["start"],
                waypoints=kwargs["waypoints"],
                ego_safety_radius=kwargs["ego_safety_radius"],
                other_robot_disks=[(2.0, 0.0, 0.35)],
                margin=kwargs.get("margin", 0.25),
            )
        # The A* fallback waypoints from _build_fake_engine's
        # call_compute_planned_waypoints stub: [(0.0, 2.0), (4.0, 0.0)].
        return validate_multi_robot_corridor(
            start=kwargs["start"],
            waypoints=kwargs["waypoints"],
            ego_safety_radius=kwargs["ego_safety_radius"],
            margin=kwargs.get("margin", 0.25),
        )

    monkeypatch.setattr(engine_module, "validate_multi_robot_corridor", fake_validate)

    result = fake.assign_route_to_multi_robot(0)

    assert result is True
    assert fake.multi_route_states[0] == SimulationControllerMixin.ROUTE_STATE_ACTIVE
    assert fake.multi_planned_path_points[0][-1] == (4.0, 0.0)
    assert fake.multi_planned_path_points[0] == [(0.0, 0.0), (0.0, 2.0), (4.0, 0.0)]
    assert any("Direct route rejected, trying A* fallback" in message for message in fake.console_logs)
    # The fallback succeeded on the first try, so the target must not have
    # been blacklisted/retried.
    assert not any("target_blacklisted_after_route_rejection" in message for message in fake.console_logs)


def test_direct_route_blocked_by_known_static_obstacle_uses_astar_fallback():
    """Known static geometry must trigger the same local fallback as a
    teammate-corridor conflict instead of ACTIVE/STUCK_SAFETY oscillation."""
    fake = _build_fake_engine(
        planner_type="Direct",
        route_plan_responses=[(True, "direct route", [(4.0, 0.0)])],
    )
    fake.mapped_obstacle_points = [(2.0, 0.0)]

    result = fake.assign_route_to_multi_robot(0)

    assert result is True
    assert fake.multi_route_states[0] == SimulationControllerMixin.ROUTE_STATE_ACTIVE
    assert fake.multi_planned_path_points[0] == [
        (0.0, 0.0),
        (0.0, 2.0),
        (4.0, 0.0),
    ]
    assert any("crosses known obstacle, trying A* fallback" in msg for msg in fake.console_logs)


def test_engine_does_not_hard_lock_untimed_teammate_future_routes(monkeypatch):
    """Crossing future polylines have no arrival times and therefore cannot
    be permanent corridor obstacles.  Current teammate disks remain in the
    validator call and enforce the real instantaneous safety constraint."""
    fake = _build_fake_engine(
        planner_type="A*",
        route_plan_responses=[(True, "crossing route", [(0.0, 4.0)])],
    )
    fake.multi_active_route_points_by_robot = lambda: [
        [(0.0, 0.0), (0.0, 4.0)],
        [(-2.0, 2.0), (2.0, 2.0)],
    ]
    seen_other_routes = []

    def capture_validation(**kwargs):
        seen_other_routes.append(list(kwargs.get("other_routes", [])))
        return validate_multi_robot_corridor(**kwargs)

    monkeypatch.setattr(engine_module, "validate_multi_robot_corridor", capture_validation)

    result = fake.assign_route_to_multi_robot(0)

    assert result is True
    assert seen_other_routes == [[]]
    assert fake.multi_route_states[0] == SimulationControllerMixin.ROUTE_STATE_ACTIVE
