from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from robotics_interfaces.coordination import CoordinationRequest, CoordinationResult
from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.plugins import (
    PluginCapability,
    PluginMetadata,
    PluginRuntimeProfile,
    build_runtime_profile,
    validate_coordination_plugin,
)
from robotics_interfaces.proposals import CandidateProposal
from robotics_sim.simulation.coordination import (
    MultiRobotCoordinator,
    map_robot_commands_by_id,
    select_runtime_control_source,
    select_runtime_path_source,
)
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.plugin_loader import load_coordination_plugin


def _robot(robot_id: int, x: float, y: float) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
    )


class FakePathPlanningPlugin:
    """Test double: a plugin that declares PATH_PLANNING and always returns
    a fixed RobotCommand.path. Not a real algorithm -- it exists only to
    prove the runtime honors PATH_PLANNING ownership end to end, from
    metadata.capabilities through to the RobotCommand the runtime consumes.
    """

    metadata = PluginMetadata(
        name="fake path planning plugin",
        version="0.0.0",
        description="test double for PATH_PLANNING ownership",
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TARGET_GENERATION,
            PluginCapability.TASK_ALLOCATION,
            PluginCapability.PATH_PLANNING,
        ),
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        command = RobotCommand(
            robot_id=0,
            status="ASSIGNED",
            target=(3.0, 0.0),
            path=((0.0, 0.0), (1.0, 0.0), (3.0, 0.0)),
        )
        return CoordinationResult(
            targets=(command.target,),
            reasons=("fake path planning plugin: assigned",),
            strategy=self.metadata.name,
            commands=(command,),
        )


class FakeControlPlugin:
    """Test double: a plugin that declares CONTROL and always returns a
    fixed RobotCommand.control_xy. Not a real controller -- it exists only to
    prove the runtime honors CONTROL ownership end to end.
    """

    metadata = PluginMetadata(
        name="fake control plugin",
        version="0.0.0",
        description="test double for CONTROL ownership",
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TARGET_GENERATION,
            PluginCapability.TASK_ALLOCATION,
            PluginCapability.CONTROL,
        ),
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        command = RobotCommand(robot_id=0, status="ASSIGNED", control_xy=(0.5, 0.0))
        return CoordinationResult(
            targets=(None,),
            reasons=("fake control plugin: assigned",),
            strategy=self.metadata.name,
            commands=(command,),
        )


def test_mmpf_runtime_profile():
    plugin = load_coordination_plugin(MMPF_COORDINATOR)

    assert PluginCapability.TARGET_GENERATION in plugin.metadata.capabilities
    assert PluginCapability.TASK_ALLOCATION in plugin.metadata.capabilities
    assert PluginCapability.PATH_PLANNING not in plugin.metadata.capabilities
    assert PluginCapability.CONTROL not in plugin.metadata.capabilities

    profile = build_runtime_profile(plugin.metadata)

    assert profile.owns_target_generation is True
    assert profile.owns_task_allocation is True
    assert profile.owns_path_planning is False
    assert profile.owns_control is False
    assert profile.uses_legacy_frontier_service is False
    assert profile.uses_external_path_planner is True
    assert profile.uses_external_motion_controller is True


def test_coordination_result_can_return_robot_commands():
    command = RobotCommand(robot_id=0, status="ASSIGNED", target=(1.0, 2.0))
    result = CoordinationResult(
        targets=((1.0, 2.0),),
        reasons=("ok",),
        strategy="test",
        commands=(command,),
    )

    assert result.targets == ((1.0, 2.0),)
    assert result.reasons == ("ok",)
    assert result.debug == {}
    assert result.assignments == ()
    assert result.commands == (command,)
    assert result.commands[0].target == (1.0, 2.0)


def test_runtime_detects_plugin_ownership():
    coordinator = MultiRobotCoordinator(strategy=MMPF_COORDINATOR)

    assert coordinator.plugin_owns_target_generation() is True
    assert coordinator.plugin_owns_path_planning() is False
    assert coordinator.plugin_owns_control() is False

    profile = coordinator.selected_plugin_profile()
    assert profile.owns_target_generation is True
    assert profile.owns_path_planning is False
    assert profile.owns_control is False


def test_mmpf_returns_robot_commands_for_assigned_targets():
    plugin = load_coordination_plugin(MMPF_COORDINATOR)
    request = CoordinationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        robots_to_assign=(0,),
        proposals_by_robot={
            0: (CandidateProposal(robot_id=0, target=(2.0, 0.0), score=5.0),),
        },
    )

    result = plugin.assign(request)

    assert len(result.commands) == 1
    command = result.commands[0]
    assert command.robot_id == 0
    assert command.status == "ASSIGNED"
    assert command.target == (2.0, 0.0)


def _mmpf_profile() -> PluginRuntimeProfile:
    return build_runtime_profile(load_coordination_plugin(MMPF_COORDINATOR).metadata)


def test_theta_real_reaches_coordination_request():
    """engine.multi_robot_coordination_states() must forward robot.theta.

    FoV-aware plugins need real orientation, not the theta=0.0 default that
    RobotCoordinationState would otherwise silently fall back to.
    """
    fake_robot = SimpleNamespace(
        x=1.0,
        y=2.0,
        theta=0.75,
        _sim_body_radius=0.2,
        _sim_safety_radius=0.35,
        vision=2.5,
    )
    fake_self = SimpleNamespace(
        robots=[fake_robot],
        config=SimpleNamespace(vision_model="LiDAR", vision=2.5),
    )
    fake_self.body_radius_for_robot = SimulationControllerMixin.body_radius_for_robot.__get__(fake_self)
    fake_self.safety_radius_for_robot = SimulationControllerMixin.safety_radius_for_robot.__get__(fake_self)

    states = SimulationControllerMixin.multi_robot_coordination_states(fake_self)

    assert len(states) == 1
    assert states[0].theta == 0.75
    assert states[0].xy == (1.0, 2.0)


def test_runtime_stores_robot_commands_by_robot_id():
    commands = (
        RobotCommand(robot_id=0, status="ASSIGNED", target=(2.0, 1.0)),
        RobotCommand(robot_id=2, status="HOLD"),
    )

    mapped = map_robot_commands_by_id(commands)

    assert mapped[0].target == (2.0, 1.0)
    assert mapped[0].status == "ASSIGNED"
    assert mapped[2].status == "HOLD"
    assert 1 not in mapped


def test_path_planning_ownership_uses_command_path():
    plugin = FakePathPlanningPlugin()
    validate_coordination_plugin(plugin)
    profile = build_runtime_profile(plugin.metadata)
    assert profile.owns_path_planning is True

    request = CoordinationRequest(robot_states=(_robot(0, 0.0, 0.0),), robots_to_assign=(0,))
    result = plugin.assign(request)
    command = map_robot_commands_by_id(result.commands)[0]
    assert command.path == ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0))

    legacy_calls: list[bool] = []

    def legacy_provider():
        legacy_calls.append(True)
        return True, "external planner", [(9.9, 9.9)]

    success, reason, waypoints = select_runtime_path_source(profile, command, legacy_provider)

    assert success is True
    assert waypoints == [(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)]
    assert legacy_calls == []


def test_no_path_planning_keeps_legacy_path_planner():
    """MMPF does not declare PATH_PLANNING, so the external planner always wins,
    even if a command.path happens to be present."""
    profile = _mmpf_profile()
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

    assert waypoints == [(3.0, 0.0)]
    assert legacy_calls == [True]
    assert "legacy A* route" in reason


def test_control_ownership_uses_command_control_xy():
    plugin = FakeControlPlugin()
    validate_coordination_plugin(plugin)
    profile = build_runtime_profile(plugin.metadata)
    assert profile.owns_control is True

    request = CoordinationRequest(robot_states=(_robot(0, 0.0, 0.0),), robots_to_assign=(0,))
    result = plugin.assign(request)
    command = map_robot_commands_by_id(result.commands)[0]
    assert command.control_xy == (0.5, 0.0)

    legacy_control = np.array([[0.0], [0.0]])

    control, reason = select_runtime_control_source(profile, command, legacy_control)

    assert control == (0.5, 0.0)
    assert "plugin control" in reason


def test_no_control_keeps_nominal_control():
    profile = _mmpf_profile()
    legacy_control = np.array([[1.0], [0.0]])
    command = RobotCommand(robot_id=0, status="ASSIGNED", control_xy=(9.0, 9.0))

    control, reason = select_runtime_control_source(profile, command, legacy_control)

    assert control is legacy_control
    assert "nominal control" in reason


def test_plugin_control_does_not_bypass_safety_veto_if_testable():
    """A CONTROL-owning plugin only proposes a control; it must not be able
    to skip the safety veto that runs before the control is actually applied.

    Full physics is out of scope here (see engine.py's per-robot movement
    loop), so this test proves the guarantee at the two points that actually
    matter:
      1. select_runtime_control_source() is a pure decision point -- it does
         not mutate its inputs or touch a robot itself.
      2. engine.py's source keeps the call order plugin proposal ->
         predicted_motion_report() (the veto) -> robot.update() (the
         actuation), so a future edit that reordered them would fail this
         test instead of silently letting a plugin bypass safety.
    """
    plugin = FakeControlPlugin()
    profile = build_runtime_profile(plugin.metadata)
    assert profile.owns_control is True

    request = CoordinationRequest(robot_states=(_robot(0, 0.0, 0.0),), robots_to_assign=(0,))
    command = map_robot_commands_by_id(plugin.assign(request).commands)[0]

    legacy_control = np.array([[0.0], [0.0]])
    control, _ = select_runtime_control_source(profile, command, legacy_control)

    # Pure decision point: the plugin's proposal is returned as-is, and the
    # legacy control passed in is untouched.
    assert control == (0.5, 0.0)
    assert legacy_control.tolist() == [[0.0], [0.0]]

    # Structural guarantee: in the movement loop, whatever select_runtime_
    # control_source() returns must reach predicted_motion_report() (the
    # safety veto) strictly before robot.update() (the actuation).
    source = inspect.getsource(SimulationControllerMixin)
    call_site = source.index("select_runtime_control_source(")
    veto_call = source.index("predicted_motion_report(", call_site)
    update_call = source.index("robot.update(control, dt)", call_site)
    assert call_site < veto_call < update_call
