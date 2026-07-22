"""Applies a CoordinationResult to per-robot target/command state explicitly.

This formalizes the merge logic that used to live inline in
SimulationControllerMixin.synchronize_multi_frontier_targets() (see
robotics_sim/tests/test_exploration_pipeline_characterization.py for the
characterization of that inline behavior): a plugin only returns entries for
the robots it was asked to (re)assign, so applying its CoordinationResult
must not disturb any robot outside that decision, while still resolving a
clear priority order for robots that ARE part of it.

Priority per mentioned robot, highest first:
    1. command.status == "CLEAR"    -> explicit clear (target -> None)
    2. command.target is not None   -> plugin-authoritative richer command
    3. assignment.status == "CLEAR" -> explicit clear (target -> None)
    4. assignment.target is not None
    5. result.targets[index] (legacy positional field)
    6. mentioned but no target and not CLEAR (e.g. HOLD/FAILED) -> preserved

Robots never mentioned by assignments, commands, or a non-None legacy
target entry are preserved untouched -- this is bucket 6's sibling case,
just without ever appearing in this decision at all.

engine.py's multi_robot_commands_by_id.update(...) never removed a stale
command for a robot outside the newest partial decision (see
test_stale_multi_robot_commands_are_overwritten_by_a_newer_decision for the
documented old behavior). This applier fixes the part of that which was
actually a bug: for every robot this decision DOES mention, any previous
RobotCommand for that same robot_id is dropped before a fresh one (if any)
is installed, so a stale .path/.control_xy from several decisions ago can
never survive under a robot_id that is being actively re-decided. Robots
outside this decision keep their old command entirely untouched, same as
their target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import CoordinationResult
from robotics_interfaces.observations import Point2D

_CLEAR_STATUSES = {"CLEAR"}


@dataclass(frozen=True)
class ApplyReport:
    """Structured record of what apply_coordination_result() actually did."""

    updated_robot_ids: tuple[int, ...] = ()
    preserved_robot_ids: tuple[int, ...] = ()
    cleared_robot_ids: tuple[int, ...] = ()
    rejected_robot_ids: tuple[int, ...] = ()
    target_source_by_robot: Mapping[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ApplyResult:
    targets_by_robot: Mapping[int, Point2D | None]
    commands_by_robot: Mapping[int, RobotCommand]
    report: ApplyReport


def apply_coordination_result(
    result: CoordinationResult,
    *,
    known_robot_ids: Sequence[int],
    previous_targets_by_robot: Mapping[int, Point2D | None],
    previous_commands_by_robot: Mapping[int, RobotCommand] | None = None,
) -> ApplyResult:
    """Merge one CoordinationResult into per-robot target/command state.

    known_robot_ids is the full active-robot roster this decision could
    possibly affect; previous_targets_by_robot/previous_commands_by_robot are
    the state from before this decision. Every known robot_id ends up with an
    entry in the returned targets_by_robot (either the new value or the
    preserved previous one); commands_by_robot only contains entries for
    robots that have one (preserved-untouched or freshly applied).
    """

    known = {int(robot_id) for robot_id in known_robot_ids}
    previous_commands_by_robot = previous_commands_by_robot or {}

    assignments_by_id, duplicate_assignment_ids = _index_by_robot_id(
        result.assignments, key=lambda item: item.robot_id
    )
    commands_by_id, duplicate_command_ids = _index_by_robot_id(
        result.commands, key=lambda item: item.robot_id
    )

    rejected: set[int] = set(duplicate_assignment_ids | duplicate_command_ids)
    for robot_id in (*assignments_by_id, *commands_by_id):
        if robot_id not in known:
            rejected.add(robot_id)

    legacy_targets_by_id: dict[int, Point2D] = {}
    for index, target in enumerate(result.targets):
        if target is None:
            continue
        if index not in known:
            rejected.add(index)
            continue
        if index in rejected:
            continue
        legacy_targets_by_id[index] = target

    mentioned_ids = {
        robot_id
        for robot_id in (*assignments_by_id, *commands_by_id, *legacy_targets_by_id)
        if robot_id in known and robot_id not in rejected
    }

    targets_by_robot: dict[int, Point2D | None] = dict(previous_targets_by_robot)
    commands_by_robot: dict[int, RobotCommand] = dict(previous_commands_by_robot)

    updated_ids: list[int] = []
    cleared_ids: list[int] = []
    preserved_ids: list[int] = []
    source_by_robot: dict[int, str] = {}

    for robot_id in sorted(mentioned_ids):
        # Any stale command from an earlier decision must not survive under
        # a robot_id this decision is actively re-deciding.
        commands_by_robot.pop(robot_id, None)

        assignment = assignments_by_id.get(robot_id)
        command = commands_by_id.get(robot_id)

        if command is not None and str(command.status) in _CLEAR_STATUSES:
            targets_by_robot[robot_id] = None
            cleared_ids.append(robot_id)
            source_by_robot[robot_id] = "command.CLEAR"
            continue

        if command is not None and command.target is not None:
            targets_by_robot[robot_id] = command.target
            commands_by_robot[robot_id] = command
            updated_ids.append(robot_id)
            source_by_robot[robot_id] = "command.target"
            continue

        if assignment is not None and str(assignment.status) in _CLEAR_STATUSES:
            targets_by_robot[robot_id] = None
            cleared_ids.append(robot_id)
            source_by_robot[robot_id] = "assignment.CLEAR"
            continue

        if assignment is not None and assignment.target is not None:
            targets_by_robot[robot_id] = assignment.target
            if command is not None:
                commands_by_robot[robot_id] = command
            updated_ids.append(robot_id)
            source_by_robot[robot_id] = "assignment.target"
            continue

        if robot_id in legacy_targets_by_id:
            targets_by_robot[robot_id] = legacy_targets_by_id[robot_id]
            if command is not None:
                commands_by_robot[robot_id] = command
            updated_ids.append(robot_id)
            source_by_robot[robot_id] = "result.targets"
            continue

        # Mentioned (had an assignment/command) but it carried no target and
        # was not an explicit CLEAR -- e.g. HOLD/FAILED. This is "no new
        # decision", not "clear the target": preserve whatever the robot had.
        preserved_ids.append(robot_id)
        source_by_robot[robot_id] = "preserved (HOLD/no target)"
        if command is not None:
            commands_by_robot[robot_id] = command

    for robot_id in known:
        if robot_id not in mentioned_ids and robot_id not in rejected:
            preserved_ids.append(robot_id)
            source_by_robot.setdefault(robot_id, "preserved (not mentioned)")

    report = ApplyReport(
        updated_robot_ids=tuple(sorted(set(updated_ids))),
        preserved_robot_ids=tuple(sorted(set(preserved_ids))),
        cleared_robot_ids=tuple(sorted(set(cleared_ids))),
        rejected_robot_ids=tuple(sorted(rejected)),
        target_source_by_robot=dict(source_by_robot),
    )

    return ApplyResult(
        targets_by_robot=targets_by_robot,
        commands_by_robot=commands_by_robot,
        report=report,
    )


def _index_by_robot_id(items: Iterable[object], *, key) -> tuple[dict[int, object], set[int]]:
    indexed: dict[int, object] = {}
    duplicates: set[int] = set()
    for item in items:
        robot_id = int(key(item))
        if robot_id in indexed:
            duplicates.add(robot_id)
            continue
        indexed[robot_id] = item
    return indexed, duplicates
