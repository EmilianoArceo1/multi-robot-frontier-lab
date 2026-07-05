from __future__ import annotations

from types import SimpleNamespace

from algorithms.mmpf_explore.plugin import MMPF_COORDINATOR
from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.plugins import build_runtime_profile
from robotics_sim.planning.coordinated_frontier_planner import validate_multi_robot_corridor
from robotics_sim.simulation.coordination import select_runtime_path_source
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.plugin_loader import load_coordination_plugin


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
