from __future__ import annotations

import math
from typing import Iterable

from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.observations import Point2D, RobotCoordinationState
from robotics_interfaces.plugins import (
    CoordinationPlugin,
    PluginCapability,
    PluginMetadata,
)
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate

MMPF_COORDINATOR = "MMPF explore coordinator"


class MmpfExplorePlugin:
    """Minimal MMPF-style coordination plugin.

    This is intentionally not a full potential-field implementation yet.
    Its job in Phase 3 is to prove that an external algorithm can obtain the
    data it needs from contracts instead of importing simulator internals.
    """

    metadata = PluginMetadata(
        name=MMPF_COORDINATOR,
        version="0.2.0",
        description=(
            "External MMPF-style exploration coordinator using explicit "
            "candidates or an injected frontier provider."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TASK_ALLOCATION,
            PluginCapability.TARGET_GENERATION,
        ),
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        robots_by_id = {robot.robot_id: robot for robot in request.robot_states}
        robots_to_assign = self._robots_to_assign(request, robots_by_id)

        reserved_targets: set[tuple[int, int]] = set()
        assignments: list[CoordinationAssignment] = []
        targets_by_robot: dict[int, Point2D | None] = {}
        reasons_by_robot: dict[int, str] = {}

        for robot_id in robots_to_assign:
            robot = robots_by_id.get(robot_id)
            if robot is None:
                assignments.append(
                    CoordinationAssignment(
                        robot_id=robot_id,
                        target=None,
                        status="FAILED",
                        reason="robot id not present in request.robot_states",
                    )
                )
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = "robot id not present in request.robot_states"
                continue

            candidates = self._candidate_pool(request, robot)
            chosen = self._choose_candidate(candidates, reserved_targets)

            if chosen is None:
                reason = "no candidates available from proposals or frontier provider"
                assignments.append(
                    CoordinationAssignment(
                        robot_id=robot_id,
                        target=None,
                        status="HOLD",
                        reason=reason,
                    )
                )
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                continue

            target = chosen.target
            reserved_targets.add(self._target_key(target, request))
            reason = f"selected by {self.metadata.name}"
            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id,
                    target=target,
                    status="ASSIGNED",
                    proposal=chosen,
                    reason=reason,
                )
            )
            targets_by_robot[robot_id] = target
            reasons_by_robot[robot_id] = reason

        targets = tuple(targets_by_robot.get(robot.robot_id) for robot in request.robot_states)
        reasons = tuple(reasons_by_robot.get(robot.robot_id, "not requested") for robot in request.robot_states)

        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            debug={
                "plugin": self.metadata.name,
                "robots_to_assign": tuple(robots_to_assign),
                "source": "proposals_or_frontier_provider",
            },
        )

    def _robots_to_assign(
        self,
        request: CoordinationRequest,
        robots_by_id: dict[int, RobotCoordinationState],
    ) -> tuple[int, ...]:
        if request.robots_to_assign:
            return tuple(request.robots_to_assign)
        return tuple(robot_id for robot_id, robot in robots_by_id.items() if robot.is_active)

    def _candidate_pool(
        self,
        request: CoordinationRequest,
        robot: RobotCoordinationState,
    ) -> tuple[ExplorationCandidate, ...]:
        explicit = request.proposals_by_robot.get(robot.robot_id, ())
        if explicit:
            return tuple(self._as_candidate(item) for item in explicit)

        frontier_provider = None
        if request.services is not None:
            frontier_provider = request.services.frontier_provider

        if frontier_provider is None or request.world is None:
            return ()

        blocked = request.blocked_targets_by_robot.get(robot.robot_id, ())
        return tuple(
            frontier_provider.candidates_for_robot(
                robot=robot,
                world=request.world,
                blocked_targets=blocked,
            )
        )

    def _choose_candidate(
        self,
        candidates: Iterable[ExplorationCandidate],
        reserved_targets: set[tuple[int, int]],
    ) -> ExplorationCandidate | None:
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                self._score(candidate),
                -candidate.target[0],
                -candidate.target[1],
            ),
            reverse=True,
        )

        for candidate in ranked:
            if self._target_key(candidate.target, None) not in reserved_targets:
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
            safety_cost=float(getattr(value, "safety_cost", 0.0)),
            overlap_cost=float(getattr(value, "overlap_cost", 0.0)),
            heading_cost=float(getattr(value, "heading_cost", 0.0)),
            metadata={"score": float(getattr(value, "score", 0.0))},
        )

    def _score(self, candidate: ExplorationCandidate) -> float:
        score = candidate.metadata.get("score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            return float(score)
        return candidate.utility

    def _target_key(
        self,
        target: Point2D,
        request: CoordinationRequest | None,
    ) -> tuple[int, int]:
        resolution = 0.01
        if request is not None:
            resolution = float(request.parameters.get("reservation_resolution", resolution))
        return (round(target[0] / resolution), round(target[1] / resolution))


def create_plugin() -> CoordinationPlugin:
    return MmpfExplorePlugin()
