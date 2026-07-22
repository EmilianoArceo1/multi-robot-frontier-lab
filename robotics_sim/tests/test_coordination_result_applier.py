"""Tests for robotics_sim/simulation/coordination_result_applier.py.

This is the formal, testable version of the merge logic that used to live
inline in SimulationControllerMixin.synchronize_multi_frontier_targets() --
see test_exploration_pipeline_characterization.py for the characterization
of the old inline behavior this replaces/fixes.
"""

from __future__ import annotations

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import CoordinationAssignment, CoordinationResult
from robotics_sim.simulation.coordination_result_applier import apply_coordination_result


def test_robots_not_mentioned_are_preserved_untouched():
    result = CoordinationResult(targets=(None, (9.0, 9.0)), reasons=("", "assigned"), strategy="test")

    applied = apply_coordination_result(
        result,
        known_robot_ids=(0, 1),
        previous_targets_by_robot={0: (1.0, 1.0), 1: None},
    )

    assert applied.targets_by_robot[0] == (1.0, 1.0)
    assert applied.targets_by_robot[1] == (9.0, 9.0)
    assert applied.report.preserved_robot_ids == (0,)
    assert applied.report.updated_robot_ids == (1,)


def test_command_target_outranks_assignment_target_and_legacy_target():
    result = CoordinationResult(
        targets=((1.0, 1.0),),
        reasons=("legacy",),
        strategy="test",
        assignments=(CoordinationAssignment(robot_id=0, status="ASSIGNED", target=(2.0, 2.0)),),
        commands=(RobotCommand(robot_id=0, status="ASSIGNED", target=(3.0, 3.0)),),
    )

    applied = apply_coordination_result(result, known_robot_ids=(0,), previous_targets_by_robot={0: None})

    assert applied.targets_by_robot[0] == (3.0, 3.0)
    assert applied.report.target_source_by_robot[0] == "command.target"


def test_assignment_target_outranks_legacy_target_when_no_command():
    result = CoordinationResult(
        targets=((1.0, 1.0),),
        reasons=("legacy",),
        strategy="test",
        assignments=(CoordinationAssignment(robot_id=0, status="ASSIGNED", target=(2.0, 2.0)),),
    )

    applied = apply_coordination_result(result, known_robot_ids=(0,), previous_targets_by_robot={0: None})

    assert applied.targets_by_robot[0] == (2.0, 2.0)
    assert applied.report.target_source_by_robot[0] == "assignment.target"


def test_legacy_target_applies_when_no_assignment_or_command():
    result = CoordinationResult(targets=((5.0, 5.0),), reasons=("legacy only",), strategy="test")

    applied = apply_coordination_result(result, known_robot_ids=(0,), previous_targets_by_robot={0: None})

    assert applied.targets_by_robot[0] == (5.0, 5.0)
    assert applied.report.target_source_by_robot[0] == "result.targets"


def test_explicit_clear_command_removes_the_target():
    result = CoordinationResult(
        targets=(None,),
        reasons=("cleared",),
        strategy="test",
        commands=(RobotCommand(robot_id=0, status="CLEAR"),),
    )

    applied = apply_coordination_result(result, known_robot_ids=(0,), previous_targets_by_robot={0: (1.0, 1.0)})

    assert applied.targets_by_robot[0] is None
    assert applied.report.cleared_robot_ids == (0,)
    assert applied.report.updated_robot_ids == ()


def test_explicit_clear_assignment_removes_the_target():
    result = CoordinationResult(
        targets=(None,),
        reasons=("cleared",),
        strategy="test",
        assignments=(CoordinationAssignment(robot_id=0, status="CLEAR", target=None),),
    )

    applied = apply_coordination_result(result, known_robot_ids=(0,), previous_targets_by_robot={0: (1.0, 1.0)})

    assert applied.targets_by_robot[0] is None
    assert applied.report.cleared_robot_ids == (0,)


def test_hold_without_target_is_not_confused_with_clear():
    """A HOLD assignment (no target, not CLEAR) means "no new decision" --
    the robot's previous target must survive, not be wiped."""
    result = CoordinationResult(
        targets=(None,),
        reasons=("no candidates available",),
        strategy="test",
        assignments=(CoordinationAssignment(robot_id=0, status="HOLD", target=None),),
    )

    applied = apply_coordination_result(result, known_robot_ids=(0,), previous_targets_by_robot={0: (1.0, 1.0)})

    assert applied.targets_by_robot[0] == (1.0, 1.0)
    assert applied.report.cleared_robot_ids == ()
    assert 0 in applied.report.preserved_robot_ids


def test_unknown_robot_id_in_command_is_rejected_and_does_not_crash():
    result = CoordinationResult(
        targets=(),
        reasons=(),
        strategy="test",
        commands=(RobotCommand(robot_id=99, status="ASSIGNED", target=(1.0, 1.0)),),
    )

    applied = apply_coordination_result(result, known_robot_ids=(0,), previous_targets_by_robot={0: None})

    assert applied.report.rejected_robot_ids == (99,)
    assert applied.targets_by_robot[0] is None


def test_duplicate_robot_id_in_commands_is_rejected_and_preserves_previous_target():
    result = CoordinationResult(
        targets=(),
        reasons=(),
        strategy="test",
        commands=(
            RobotCommand(robot_id=0, status="ASSIGNED", target=(1.0, 1.0)),
            RobotCommand(robot_id=0, status="ASSIGNED", target=(2.0, 2.0)),
        ),
    )

    applied = apply_coordination_result(result, known_robot_ids=(0,), previous_targets_by_robot={0: (5.0, 5.0)})

    assert 0 in applied.report.rejected_robot_ids
    assert applied.targets_by_robot[0] == (5.0, 5.0)


def test_stale_command_for_a_reassigned_robot_is_dropped_not_carried_forward():
    """This is the real bug problem I in the refactor brief describes: a
    robot IS re-decided this round (fresh legacy target), but the plugin's
    new result.commands does not include a fresh command for it. The OLD
    command object (with its now-mismatched .path) from a previous decision
    must not survive attached to this robot_id."""
    stale_command = RobotCommand(
        robot_id=0, status="ASSIGNED", target=(1.0, 1.0), path=((0.0, 0.0), (1.0, 1.0))
    )
    result = CoordinationResult(targets=((9.0, 9.0),), reasons=("re-assigned",), strategy="test")

    applied = apply_coordination_result(
        result,
        known_robot_ids=(0,),
        previous_targets_by_robot={0: (1.0, 1.0)},
        previous_commands_by_robot={0: stale_command},
    )

    assert applied.targets_by_robot[0] == (9.0, 9.0)
    assert 0 not in applied.commands_by_robot


def test_command_for_an_unmentioned_robot_survives_untouched():
    result = CoordinationResult(targets=(None, (9.0, 9.0)), reasons=("", "assigned"), strategy="test")
    untouched_command = RobotCommand(robot_id=0, status="ASSIGNED", target=(1.0, 1.0))

    applied = apply_coordination_result(
        result,
        known_robot_ids=(0, 1),
        previous_targets_by_robot={0: (1.0, 1.0), 1: None},
        previous_commands_by_robot={0: untouched_command},
    )

    assert applied.commands_by_robot[0] is untouched_command
    assert 0 in applied.report.preserved_robot_ids


def test_whole_team_explicit_decision_updates_every_robot_in_one_call():
    result = CoordinationResult(
        targets=((1.0, 0.0), (2.0, 0.0), (3.0, 0.0)),
        reasons=("a", "b", "c"),
        strategy="test",
    )

    applied = apply_coordination_result(
        result,
        known_robot_ids=(0, 1, 2),
        previous_targets_by_robot={0: None, 1: None, 2: None},
    )

    assert applied.targets_by_robot == {0: (1.0, 0.0), 1: (2.0, 0.0), 2: (3.0, 0.0)}
    assert applied.report.updated_robot_ids == (0, 1, 2)
