"""Characterization tests for the current (pre-refactor) coordination pipeline.

These tests pin down real, observed behavior of
SimulationControllerMixin.synchronize_multi_frontier_targets() and the
plugin/runtime ownership plumbing it depends on, *before* the architectural
refactor described in the exploration-pipeline-architecture design. They are
not a statement of desired UX -- some of what they assert (e.g. "changing the
legacy exploration_planner does not affect a plugin that ignores it") is
existing, intentional plugin behavior that the refactor must not silently
change.

Style note: this module builds a minimal SimpleNamespace "fake engine" bound
to the real SimulationControllerMixin methods it needs, the same pattern
test_plugin_runtime_ownership.py uses for
test_theta_real_reaches_coordination_request. Spinning up a full Qt
SimulationEngine is not done anywhere else in this test suite.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from robotics_interfaces.coordination import CoordinationRequest, CoordinationResult
from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.plugins import PluginCapability, PluginMetadata
from robotics_sim.simulation import engine as engine_module
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.plugin_loader import load_coordination_plugin
from robotics_sim.simulation.runtime_robot_registry import RuntimeRobotRegistry


def _fake_robot(x: float, y: float, theta: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        x=x,
        y=y,
        theta=theta,
        _sim_body_radius=0.2,
        _sim_safety_radius=0.35,
        vision=2.5,
    )


def _fake_engine(robots, *, coordinator_type: str = MMPF_COORDINATOR, exploration_planner: str = "Nearest frontier"):
    fake_self = SimpleNamespace(
        robots=robots,
        config=SimpleNamespace(
            vision_model="Camera / FoV",
            vision=2.5,
            grid_resolution=0.5,
            exploration_planner=exploration_planner,
            coordinator_type=coordinator_type,
            ipp_distance_penalty=0.5,
            goal_tolerance=0.25,
            coordination_parameters={},
            goal_x=10.0,
            goal_y=10.0,
        ),
        multi_exploration_targets=[None] * len(robots),
        multi_invalidated_exploration_targets=[[] for _ in robots],
        belief_map=SimpleNamespace(robot_explored_points=lambda index: []),
        explored_free_points=[(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)],
        mapped_obstacle_points=[],
        current_route_points_for_robot=lambda robot: [(robot.x, robot.y)],
        simulation_time=0.0,
        selected_robot_index=0,
        last_goal_selection_reason="",
        runtime_robot_registry=RuntimeRobotRegistry(),
    )
    fake_self.body_radius_for_robot = SimulationControllerMixin.body_radius_for_robot.__get__(fake_self)
    fake_self.safety_radius_for_robot = SimulationControllerMixin.safety_radius_for_robot.__get__(fake_self)
    fake_self.is_goal_seeking_mode = SimulationControllerMixin.is_goal_seeking_mode.__get__(fake_self)
    fake_self.exploration_planner_name = SimulationControllerMixin.exploration_planner_name.__get__(fake_self)
    fake_self.ensure_multi_exploration_target_slots = (
        SimulationControllerMixin.ensure_multi_exploration_target_slots.__get__(fake_self)
    )
    fake_self.multi_active_route_points_by_robot = (
        SimulationControllerMixin.multi_active_route_points_by_robot.__get__(fake_self)
    )
    fake_self.multi_robot_coordination_states = (
        SimulationControllerMixin.multi_robot_coordination_states.__get__(fake_self)
    )
    fake_self.multi_frontier_exclusion_radius = (
        SimulationControllerMixin.multi_frontier_exclusion_radius.__get__(fake_self)
    )
    fake_self.multi_dynamic_target_margin = (
        SimulationControllerMixin.multi_dynamic_target_margin.__get__(fake_self)
    )
    fake_self.final_goal_xy = SimulationControllerMixin.final_goal_xy.__get__(fake_self)
    fake_self.ensure_runtime_robot_registry = (
        SimulationControllerMixin.ensure_runtime_robot_registry.__get__(fake_self)
    )
    fake_self.coordinator_runtime_profile = (
        SimulationControllerMixin.coordinator_runtime_profile.__get__(fake_self)
    )
    fake_self.publish_multi_exploration_targets = (
        SimulationControllerMixin.publish_multi_exploration_targets.__get__(fake_self)
    )
    fake_self.synchronize_multi_frontier_targets = (
        SimulationControllerMixin.synchronize_multi_frontier_targets.__get__(fake_self)
    )
    return fake_self


class _FixedPlugin:
    """Test double: returns a caller-supplied CoordinationResult unmodified.

    Lets a test dictate exactly which targets/commands the "plugin" hands
    back, so the surrounding engine merge/preserve logic can be characterized
    in isolation from any real algorithm's scoring.
    """

    metadata = PluginMetadata(
        name="fixed result test double",
        version="0.0.0",
        description="returns a fixed CoordinationResult",
        capabilities=(PluginCapability.COORDINATION, PluginCapability.TASK_ALLOCATION),
    )

    def __init__(self, result: CoordinationResult):
        self._result = result

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        return self._result


def _patch_coordinator(monkeypatch, plugin) -> None:
    class _FakeCoordinatorFactory:
        def __new__(cls, strategy):
            from robotics_sim.simulation.coordination import MultiRobotCoordinator as RealCoordinator

            coordinator = RealCoordinator.__new__(RealCoordinator)
            coordinator.plugin = plugin
            coordinator.strategy = plugin.metadata.name
            from robotics_interfaces.plugins import build_runtime_profile

            coordinator.runtime_profile = build_runtime_profile(plugin.metadata)
            return coordinator

    monkeypatch.setattr(engine_module, "MultiRobotCoordinator", _FakeCoordinatorFactory)


def test_changing_exploration_planner_does_not_change_mmpf_assignment():
    """MMPF does not read planner_name; changing the legacy combo value must
    not change its selected target. This documents existing behavior, not
    desired UX -- Goal seeking mode is excluded (see next test), since Goal
    seeking bypasses coordination entirely rather than reading planner_name."""
    robots = [_fake_robot(0.0, 0.0)]

    engine_a = _fake_engine(robots, exploration_planner="Nearest frontier")
    engine_a.synchronize_multi_frontier_targets(requesting_robot_index=0)
    target_a = engine_a.multi_exploration_targets[0]

    robots_b = [_fake_robot(0.0, 0.0)]
    engine_b = _fake_engine(robots_b, exploration_planner="FoV-aware directional frontier")
    engine_b.synchronize_multi_frontier_targets(requesting_robot_index=0)
    target_b = engine_b.multi_exploration_targets[0]

    assert target_a == target_b
    assert target_a is not None


def test_goal_seeking_mode_skips_coordination_entirely():
    """Goal seeking is the only mode where coordination must not run at all."""
    robots = [_fake_robot(0.0, 0.0)]
    engine = _fake_engine(robots, exploration_planner="Goal seeking")

    assert engine.is_goal_seeking_mode() is True

    engine.synchronize_multi_frontier_targets(requesting_robot_index=0)

    # No assignment happened: the target slot is untouched (still None), and
    # no coordinator was ever constructed.
    assert engine.multi_exploration_targets == [None]
    assert not hasattr(engine, "_multi_robot_coordinator")


def test_goal_seeking_guard_runs_before_any_coordinator_work():
    """Structural guarantee: is_goal_seeking_mode() is checked before the
    method does any coordinator/plugin work, so a future refactor cannot
    accidentally move real work ahead of the bypass."""
    source = inspect.getsource(SimulationControllerMixin.synchronize_multi_frontier_targets)
    guard_index = source.index("is_goal_seeking_mode()")
    coordinator_index = source.index("MultiRobotCoordinator(")
    assert guard_index < coordinator_index


def test_robots_not_mentioned_in_result_keep_their_previous_target(monkeypatch):
    """A plugin only returns entries for the robots it was asked to
    (re)assign; robots outside that batch must keep whatever target they had
    before this call -- this is the "preserve unmentioned robots" contract
    that robotics_sim/simulation/coordination_result_applier.py must later
    formalize (see engine.py:2665-2684 preference-order comment)."""
    robots = [_fake_robot(0.0, 0.0), _fake_robot(5.0, 5.0)]
    engine = _fake_engine(robots)
    engine.multi_exploration_targets = [(1.0, 1.0), None]

    plugin = _FixedPlugin(
        CoordinationResult(
            targets=(None, (9.0, 9.0)),
            reasons=("no decision returned", "assigned"),
            strategy="fixed",
        )
    )
    _patch_coordinator(monkeypatch, plugin)

    engine.synchronize_multi_frontier_targets(requesting_robot_index=1)

    # Robot 0 was not part of this decision (result.targets[0] is None) and
    # must keep its previous target untouched.
    assert engine.multi_exploration_targets[0] == (1.0, 1.0)
    assert engine.multi_exploration_targets[1] == (9.0, 9.0)


def test_command_target_takes_priority_over_result_targets_entry(monkeypatch):
    """Priority order today: command.target > result.targets[index] > the
    robot's previous target. This is exactly the implicit priority problem H
    in the refactor brief documents; pin it down before formalizing it."""
    robots = [_fake_robot(0.0, 0.0)]
    engine = _fake_engine(robots)

    command = RobotCommand(robot_id=0, status="ASSIGNED", target=(3.0, 3.0))
    plugin = _FixedPlugin(
        CoordinationResult(
            targets=((1.0, 1.0),),
            reasons=("legacy target field",),
            strategy="fixed",
            commands=(command,),
        )
    )
    _patch_coordinator(monkeypatch, plugin)

    engine.synchronize_multi_frontier_targets(requesting_robot_index=0)

    assert engine.multi_exploration_targets[0] == (3.0, 3.0)


def test_stale_multi_robot_commands_are_overwritten_by_a_newer_decision(monkeypatch):
    """multi_robot_commands_by_id.update(...) is a dict.update(): a robot_id
    present in an older decision but absent from a newer one keeps its old
    command entry. This test documents that today's merge does NOT clear
    stale commands for robots outside the newest result -- the applier this
    refactor introduces (Phase 6) must decide, explicitly, whether that is
    still acceptable once assignments are authoritative per decision."""
    robots = [_fake_robot(0.0, 0.0), _fake_robot(5.0, 5.0)]
    engine = _fake_engine(robots)
    engine.multi_exploration_targets = [None, None]

    first_round = _FixedPlugin(
        CoordinationResult(
            targets=((1.0, 1.0), (2.0, 2.0)),
            reasons=("assigned", "assigned"),
            strategy="fixed",
            commands=(
                RobotCommand(robot_id=0, status="ASSIGNED", target=(1.0, 1.0)),
                RobotCommand(robot_id=1, status="ASSIGNED", target=(2.0, 2.0)),
            ),
        )
    )
    _patch_coordinator(monkeypatch, first_round)
    engine.synchronize_multi_frontier_targets(requesting_robot_index=0)
    assert engine.multi_robot_commands_by_id[0].target == (1.0, 1.0)
    assert engine.multi_robot_commands_by_id[1].target == (2.0, 2.0)

    # A newer decision only re-assigns robot 0; robot 1 is untouched.
    engine.multi_exploration_targets[0] = None
    second_round = _FixedPlugin(
        CoordinationResult(
            targets=((7.0, 7.0),),
            reasons=("re-assigned",),
            strategy="fixed",
            commands=(RobotCommand(robot_id=0, status="ASSIGNED", target=(7.0, 7.0)),),
        )
    )
    _patch_coordinator(monkeypatch, second_round)
    engine.synchronize_multi_frontier_targets(requesting_robot_index=0)

    assert engine.multi_robot_commands_by_id[0].target == (7.0, 7.0)
    # Documented current behavior: robot 1's stale command from the FIRST
    # round is still present even though it was not part of the SECOND
    # decision at all.
    assert engine.multi_robot_commands_by_id[1].target == (2.0, 2.0)


def test_plugin_can_return_explicit_decisions_for_the_whole_team(monkeypatch):
    """A plugin is free to return targets/commands for every requested robot
    in one call -- this is the existing mechanism Phase 7's periodic
    FULL_TEAM replanning will reuse, not a new capability."""
    robots = [_fake_robot(0.0, 0.0), _fake_robot(5.0, 5.0), _fake_robot(2.0, 2.0)]
    engine = _fake_engine(robots)
    engine.multi_exploration_targets = [None, None, None]

    plugin = _FixedPlugin(
        CoordinationResult(
            targets=((1.0, 0.0), (5.0, 1.0), (2.0, 1.0)),
            reasons=("assigned", "assigned", "assigned"),
            strategy="fixed",
            commands=tuple(
                RobotCommand(robot_id=i, status="ASSIGNED", target=target)
                for i, target in enumerate(((1.0, 0.0), (5.0, 1.0), (2.0, 1.0)))
            ),
        )
    )
    _patch_coordinator(monkeypatch, plugin)

    engine.synchronize_multi_frontier_targets(requesting_robot_index=0)

    assert engine.multi_exploration_targets == [(1.0, 0.0), (5.0, 1.0), (2.0, 1.0)]


def test_nav2d_wavefront_assigns_targets_without_any_host_frontier_service():
    """nav2d_wavefront does its own grid/frontier detection internally
    (algorithms/nav2d_wavefront/plugin.py); it must produce targets even when
    request.services is None, i.e. with no frontier_provider,
    team_frontier_provider, or frontier_information_service available."""
    from algorithms.nav2d_wavefront.plugin import NAV2D_WAVEFRONT_COORDINATOR
    from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot

    plugin = load_coordination_plugin(NAV2D_WAVEFRONT_COORDINATOR)

    explored = tuple(
        (x * 0.5, y * 0.5)
        for x in range(-6, 7)
        for y in range(-6, 7)
    )
    request = CoordinationRequest(
        robot_states=(
            RobotCoordinationState(
                robot_id=0,
                xy=(0.0, 0.0),
                safety_radius=0.35,
                sensor_range=2.5,
                vision_model="Camera / FoV",
            ),
        ),
        robots_to_assign=(0,),
        world=WorldSnapshot(
            explored_points=explored,
            mapped_obstacle_points=(),
            bounds=(-5.0, 5.0, -5.0, 5.0),
            resolution=0.5,
        ),
        services=None,
    )

    result = plugin.assign(request)

    assert len(result.targets) == 1


def test_independent_mmpf_cqlite_assign_from_host_candidates_without_claiming_detection():
    """independent_baseline/mmpf_explore/cqlite all consume host-provided
    candidates (team_frontier_provider/frontier_provider/frontier_information_
    service) rather than detecting frontiers themselves. This is the exact
    ownership mismatch problem A in the refactor brief documents: they
    currently still declare PluginCapability.TARGET_GENERATION even though
    they do not detect anything -- pinning this down here so Phase 5's
    metadata migration has a concrete before/after."""
    from algorithms.independent_baseline.plugin import INDEPENDENT_BASELINE_COORDINATOR
    from algorithms.cqlite.plugin import CQLITE_COORDINATOR

    for name in (INDEPENDENT_BASELINE_COORDINATOR, MMPF_COORDINATOR, CQLITE_COORDINATOR):
        plugin = load_coordination_plugin(name)
        assert PluginCapability.TARGET_GENERATION in plugin.metadata.capabilities
