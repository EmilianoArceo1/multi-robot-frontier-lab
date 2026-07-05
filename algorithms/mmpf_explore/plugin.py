from __future__ import annotations

from robotics_interfaces import (
    CandidateProposal,
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
    PluginCapability,
    PluginMetadata,
)


MMPF_EXPLORE_COORDINATOR = "MMPF explore coordinator"


class MmpfExplorePlugin:
    """Minimal multi-robot potential-field style target allocator.

    This first version is intentionally small: it consumes candidate proposals
    that the host already prepared, ranks them by a local utility, and reserves
    selected targets so two robots do not receive the same candidate.
    """

    metadata = PluginMetadata(
        name=MMPF_EXPLORE_COORDINATOR,
        version="0.1.0",
        description=(
            "Minimal proposal-based MMPF exploration coordinator with target reservations."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TASK_ALLOCATION,
        ),
        source="multi-robot-frontier-lab phase 3 plugin scaffold",
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        robot_states = tuple(request.robot_states)
        count = len(robot_states)
        selected_robot_ids = _selected_robot_ids(request)
        reservation_radius = float(request.shared.get("target_reservation_radius", 1e-9))

        targets: list[tuple[float, float] | None] = [
            _existing_target(request, index, robot.robot_id)
            for index, robot in enumerate(robot_states)
        ]
        reasons: list[str] = [
            "kept existing target" if target is not None else "no target assigned yet"
            for target in targets
        ]

        reserved_targets: list[tuple[float, float]] = [
            target
            for index, target in enumerate(targets)
            if target is not None and robot_states[index].robot_id not in selected_robot_ids
        ]

        assignments: list[CoordinationAssignment | None] = [None] * count

        for index, robot in enumerate(robot_states):
            robot_id = robot.robot_id

            if robot_id not in selected_robot_ids:
                assignments[index] = CoordinationAssignment(
                    robot_id=robot_id,
                    status="ASSIGNED" if targets[index] is not None else "HOLD",
                    target=targets[index],
                    reason=reasons[index],
                )
                continue

            if not robot.is_active:
                targets[index] = None
                reasons[index] = f"{self.metadata.name}: HOLD inactive robot"
                assignments[index] = CoordinationAssignment(
                    robot_id=robot_id,
                    status="HOLD",
                    target=None,
                    reason=reasons[index],
                )
                continue

            proposals = _proposals_for_robot(request, index, robot_id)
            blocked_targets = tuple(_blocked_targets_for_robot(request, index, robot_id))
            selected_proposal = _choose_best_available_proposal(
                proposals=proposals,
                blocked_targets=blocked_targets,
                reserved_targets=tuple(reserved_targets),
                reservation_radius=reservation_radius,
            )

            if selected_proposal is None:
                targets[index] = None
                reasons[index] = f"{self.metadata.name}: HOLD no feasible proposal"
                assignments[index] = CoordinationAssignment(
                    robot_id=robot_id,
                    status="HOLD",
                    target=None,
                    reason=reasons[index],
                )
                continue

            target = _as_point(selected_proposal.target)
            targets[index] = target
            reserved_targets.append(target)
            utility = _proposal_utility(selected_proposal)
            reasons[index] = (
                f"{self.metadata.name}: assigned proposal "
                f"utility={utility:.3f}; score={selected_proposal.score:.3f}; "
                f"info_gain={selected_proposal.information_gain:.3f}"
            )
            assignments[index] = CoordinationAssignment(
                robot_id=robot_id,
                status="ASSIGNED",
                target=target,
                reason=reasons[index],
                proposal=selected_proposal,
            )

        completed_assignments = tuple(
            assignment
            if assignment is not None
            else CoordinationAssignment(
                robot_id=robot_states[index].robot_id,
                status="HOLD",
                target=targets[index],
                reason=reasons[index],
            )
            for index, assignment in enumerate(assignments)
        )

        return CoordinationResult(
            targets=tuple(targets),
            reasons=tuple(reasons),
            strategy=self.metadata.name,
            assignments=completed_assignments,
            debug={
                "algorithm": self.metadata.name,
                "selected_robot_ids": tuple(sorted(selected_robot_ids)),
                "reserved_targets": tuple(reserved_targets),
            },
        )


def _selected_robot_ids(request: CoordinationRequest) -> set[int]:
    robot_states = tuple(request.robot_states)
    robot_ids = {robot.robot_id for robot in robot_states}

    if not request.robots_to_assign:
        return {robot.robot_id for robot in robot_states if robot.is_active}

    selected: set[int] = set()
    for raw_value in request.robots_to_assign:
        value = int(raw_value)
        if value in robot_ids:
            selected.add(value)
        elif 0 <= value < len(robot_states):
            selected.add(robot_states[value].robot_id)

    return selected


def _existing_target(
    request: CoordinationRequest,
    index: int,
    robot_id: int,
) -> tuple[float, float] | None:
    target = request.existing_targets_by_robot.get(robot_id)
    if target is None:
        target = request.existing_targets_by_robot.get(index)
    return None if target is None else _as_point(target)


def _proposals_for_robot(
    request: CoordinationRequest,
    index: int,
    robot_id: int,
) -> tuple[CandidateProposal, ...]:
    proposals = request.proposals_by_robot.get(robot_id)
    if proposals is None:
        proposals = request.proposals_by_robot.get(index, ())
    return tuple(proposals)


def _blocked_targets_for_robot(
    request: CoordinationRequest,
    index: int,
    robot_id: int,
) -> tuple[tuple[float, float], ...]:
    blocked = request.blocked_targets_by_robot.get(robot_id)
    if blocked is None:
        blocked = request.blocked_targets_by_robot.get(index, ())
    return tuple(_as_point(target) for target in blocked)


def _choose_best_available_proposal(
    proposals: tuple[CandidateProposal, ...],
    blocked_targets: tuple[tuple[float, float], ...],
    reserved_targets: tuple[tuple[float, float], ...],
    reservation_radius: float,
) -> CandidateProposal | None:
    ordered = sorted(
        proposals,
        key=lambda proposal: (
            -_proposal_utility(proposal),
            proposal.travel_cost,
            proposal.target[0],
            proposal.target[1],
        ),
    )

    for proposal in ordered:
        target = _as_point(proposal.target)
        if _matches_any_target(target, blocked_targets, reservation_radius):
            continue
        if _matches_any_target(target, reserved_targets, reservation_radius):
            continue
        return proposal

    return None


def _proposal_utility(proposal: CandidateProposal) -> float:
    return (
        proposal.score
        + proposal.information_gain
        - proposal.travel_cost
        - proposal.overlap_cost
        - proposal.safety_cost
        - proposal.heading_cost
    )


def _matches_any_target(
    target: tuple[float, float],
    candidates: tuple[tuple[float, float], ...],
    radius: float,
) -> bool:
    radius_sq = radius * radius
    return any(_distance_sq(target, candidate) <= radius_sq for candidate in candidates)


def _distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _as_point(point: tuple[float, float]) -> tuple[float, float]:
    return (float(point[0]), float(point[1]))


def create_plugin() -> MmpfExplorePlugin:
    return MmpfExplorePlugin()
