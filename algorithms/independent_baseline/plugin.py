from __future__ import annotations

from typing import Mapping

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.observations import Point2D, RobotCoordinationState
from robotics_interfaces.plugins import (
    CandidateInputMode,
    CoordinationPlugin,
    PluginCapability,
    PluginMetadata,
)
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate

INDEPENDENT_BASELINE_COORDINATOR = "Independent baseline coordinator"


class IndependentBaselinePlugin:
    """Deterministic, minimal reference plugin.

    This is a template for future independent algorithms (MMPF, MARVEL,
    auction-based, frontier variants, ...): it proves the full contract end
    to end -- explicit proposals or services -> WorldSnapshot ->
    ExplorationCandidate -> RobotCommand -- using the simplest possible
    selection rule on purpose. It is not meant to be competitive; it exists to
    be copied and replaced by a real algorithm.

    Rule: highest information_gain wins; travel_cost breaks ties. That's it.
    """

    metadata = PluginMetadata(
        name=INDEPENDENT_BASELINE_COORDINATOR,
        version="0.1.0",
        description=(
            "Deterministic greedy baseline (highest information gain, "
            "travel_cost breaks ties). Template for new independent plugins."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TARGET_GENERATION,
            PluginCapability.TASK_ALLOCATION,
        ),
        # TARGET_GENERATION above is the deprecated capability, kept for
        # backward compatibility (PluginRuntimeProfile.owns_target_generation,
        # existing tests). This plugin does not detect frontiers or reduce
        # candidates into new tasks -- it only ranks candidates already
        # provided by team_frontier_provider/frontier_provider/explicit
        # proposals, hence HOST_CANDIDATES rather than PLUGIN_INTERNAL.
        candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        robots_by_id = {robot.robot_id: robot for robot in request.robot_states}
        robots_to_assign = self._robots_to_assign(request, robots_by_id)
        team_candidates = self._team_candidate_pool(request)

        # Robots not being reassigned this call keep whatever target they
        # already have -- both as the semantic result (see targets below) and
        # as a reservation so a newly-assigning robot does not duplicate it.
        reserved_targets: set[tuple[int, int]] = {
            self._target_key(robot.current_target, request)
            for robot in request.robot_states
            if robot.robot_id not in robots_to_assign and robot.current_target is not None
        }

        assignments: list[CoordinationAssignment] = []
        commands: list[RobotCommand] = []
        targets_by_robot: dict[int, Point2D | None] = {}
        reasons_by_robot: dict[int, str] = {}

        for robot_id in robots_to_assign:
            robot = robots_by_id.get(robot_id)
            if robot is None:
                reason = "robot id not present in request.robot_states"
                assignments.append(
                    CoordinationAssignment(robot_id=robot_id, target=None, status="FAILED", reason=reason)
                )
                commands.append(RobotCommand(robot_id=robot_id, status="FAILED", reason=reason))
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                continue

            candidates = self._candidate_pool(request, robot, team_candidates)
            chosen = self._choose_candidate(candidates, reserved_targets, request)

            if chosen is None:
                reason = "no candidates available from proposals or frontier services"
                assignments.append(
                    CoordinationAssignment(robot_id=robot_id, target=None, status="HOLD", reason=reason)
                )
                commands.append(RobotCommand(robot_id=robot_id, status="HOLD", reason=reason))
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                continue

            target = chosen.target
            reserved_targets.add(self._target_key(target, request))
            reason = f"selected by {self.metadata.name}: highest information gain wins"
            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id, target=target, status="ASSIGNED", proposal=chosen, reason=reason
                )
            )
            commands.append(
                RobotCommand(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=target,
                    heading_rad=chosen.heading_rad,
                    reason=reason,
                )
            )
            targets_by_robot[robot_id] = target
            reasons_by_robot[robot_id] = reason

        targets = tuple(
            targets_by_robot.get(robot.robot_id, robot.current_target) for robot in request.robot_states
        )
        reasons = tuple(
            reasons_by_robot.get(
                robot.robot_id,
                "kept existing target" if robot.current_target is not None else "not requested",
            )
            for robot in request.robot_states
        )

        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            debug={
                "plugin": self.metadata.name,
                "capabilities": tuple(capability.value for capability in self.metadata.capabilities),
                "robots_to_assign": tuple(robots_to_assign),
            },
            commands=tuple(commands),
        )

    def _robots_to_assign(
        self,
        request: CoordinationRequest,
        robots_by_id: dict[int, RobotCoordinationState],
    ) -> tuple[int, ...]:
        if request.robots_to_assign:
            return tuple(request.robots_to_assign)
        return tuple(robot_id for robot_id, robot in robots_by_id.items() if robot.is_active)

    def _team_candidate_pool(
        self,
        request: CoordinationRequest,
    ) -> Mapping[int, tuple[ExplorationCandidate, ...]]:
        if request.services is None:
            return {}
        team_provider = getattr(request.services, "team_frontier_provider", None)
        if team_provider is None:
            return {}
        return {
            int(robot_id): tuple(self._as_candidate(candidate) for candidate in candidates)
            for robot_id, candidates in team_provider.candidates_for_team(request).items()
        }

    def _candidate_pool(
        self,
        request: CoordinationRequest,
        robot: RobotCoordinationState,
        team_candidates: Mapping[int, tuple[ExplorationCandidate, ...]],
    ) -> tuple[ExplorationCandidate, ...]:
        explicit = request.proposals_by_robot.get(robot.robot_id, ())
        if explicit:
            return tuple(self._as_candidate(item) for item in explicit)

        if robot.robot_id in team_candidates:
            return team_candidates[robot.robot_id]

        frontier_provider = None
        if request.services is not None:
            frontier_provider = request.services.frontier_provider

        if frontier_provider is None or request.world is None:
            return ()

        blocked = request.blocked_targets_by_robot.get(robot.robot_id, ())
        return tuple(
            frontier_provider.candidates_for_robot(robot=robot, world=request.world, blocked_targets=blocked)
        )

    def _choose_candidate(
        self,
        candidates: tuple[ExplorationCandidate, ...],
        reserved_targets: set[tuple[int, int]],
        request: CoordinationRequest,
    ) -> ExplorationCandidate | None:
        ranked = sorted(
            candidates,
            key=lambda candidate: (candidate.information_gain, -candidate.travel_cost),
            reverse=True,
        )
        for candidate in ranked:
            if self._target_key(candidate.target, request) not in reserved_targets:
                return candidate
        return None

    def _as_candidate(
        self,
        value: ExplorationCandidate | CandidateProposal,
    ) -> ExplorationCandidate:
        if isinstance(value, ExplorationCandidate):
            return value
        if isinstance(value, CandidateProposal):
            return value.as_candidate(source="explicit_proposal")
        target = getattr(value, "target")
        return ExplorationCandidate(
            target=target,
            source="duck_typed_proposal",
            information_gain=float(getattr(value, "information_gain", 0.0)),
            travel_cost=float(getattr(value, "travel_cost", 0.0)),
        )

    def _target_key(self, target: Point2D, request: CoordinationRequest) -> tuple[int, int]:
        resolution = float(request.parameters.get("reservation_resolution", 0.01))
        resolution = max(resolution, 1e-6)
        return (round(target[0] / resolution), round(target[1] / resolution))


def create_plugin() -> CoordinationPlugin:
    return IndependentBaselinePlugin()
