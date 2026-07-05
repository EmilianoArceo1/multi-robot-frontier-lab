from __future__ import annotations

import logging
import math
from typing import Iterable, Mapping

from robotics_interfaces.commands import RobotCommand
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

_LOGGER = logging.getLogger(__name__)


class MmpfExplorePlugin:
    """Minimal MMPF-style coordination plugin.

    Phase 3 goal: prove that an external algorithm can obtain team-level
    frontier candidates from contracts/services instead of importing simulator
    internals. This is still a small baseline, not a full potential-field paper
    implementation.
    """

    metadata = PluginMetadata(
        name=MMPF_COORDINATOR,
        version="0.3.0",
        description=(
            "External MMPF-style exploration coordinator using explicit "
            "candidates, a team frontier provider, or a single-robot fallback."
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
        team_candidates = self._team_candidate_pool(request)

        # Robots not being reassigned this call still occupy their current
        # target. Seeding reserved_targets with them stops MMPF from handing a
        # newly-assigning robot a near-duplicate of an already-active F_j,
        # which was previously only caught later by the engine's own
        # validation and caused replan loops.
        reserved_targets: set[tuple[int, int]] = {
            self._target_key(robot.current_target, request)
            for robot in request.robot_states
            if robot.robot_id not in robots_to_assign and robot.current_target is not None
        }

        assignments: list[CoordinationAssignment] = []
        commands: list[RobotCommand] = []
        targets_by_robot: dict[int, Point2D | None] = {}
        reasons_by_robot: dict[int, str] = {}
        rejections: list[tuple[int, str]] = []
        candidates_received: dict[int, int] = {}

        for robot_id in robots_to_assign:
            robot = robots_by_id.get(robot_id)
            if robot is None:
                reason = "robot id not present in request.robot_states"
                assignments.append(
                    CoordinationAssignment(
                        robot_id=robot_id,
                        target=None,
                        status="FAILED",
                        reason=reason,
                    )
                )
                commands.append(
                    RobotCommand(robot_id=robot_id, status="FAILED", reason=reason)
                )
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                continue

            raw_candidates = self._candidate_pool(request, robot, team_candidates)
            candidates_received[robot_id] = len(raw_candidates)
            candidates = self._filter_candidates(
                request, robot, raw_candidates, reserved_targets, rejections
            )
            chosen = self._choose_candidate(candidates, reserved_targets, request)

            if chosen is None:
                reason = "no candidates available from proposals or frontier services"
                if raw_candidates:
                    reason = "all candidates rejected (too close, blocked, or conflicting)"
                assignments.append(
                    CoordinationAssignment(
                        robot_id=robot_id,
                        target=None,
                        status="HOLD",
                        reason=reason,
                    )
                )
                commands.append(
                    RobotCommand(robot_id=robot_id, status="HOLD", reason=reason)
                )
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                continue

            target = chosen.target
            reserved_targets.add(self._target_key(target, request))
            reason = f"selected by {self.metadata.name}"
            provider_reason = chosen.metadata.get("reason")
            if isinstance(provider_reason, str) and provider_reason:
                reason = f"{reason}; {provider_reason}"

            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id,
                    target=target,
                    status="ASSIGNED",
                    proposal=chosen,
                    reason=reason,
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

        targets = tuple(targets_by_robot.get(robot.robot_id) for robot in request.robot_states)
        reasons = tuple(reasons_by_robot.get(robot.robot_id, "not requested") for robot in request.robot_states)

        debug = {
            "plugin": self.metadata.name,
            "capabilities": tuple(capability.value for capability in self.metadata.capabilities),
            "robots_to_assign": tuple(robots_to_assign),
            "source": self._debug_source(request, team_candidates),
            "candidates_received": dict(candidates_received),
            "candidates_rejected": tuple(
                f"R{robot_id}:{reason}" for robot_id, reason in rejections
            ),
            "min_frontier_travel_distance": self._min_frontier_travel_distance(request),
        }
        _LOGGER.debug("mmpf assign: %s", debug)

        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            debug=debug,
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
            frontier_provider.candidates_for_robot(
                robot=robot,
                world=request.world,
                blocked_targets=blocked,
            )
        )

    def _filter_candidates(
        self,
        request: CoordinationRequest,
        robot: RobotCoordinationState,
        candidates: tuple[ExplorationCandidate, ...],
        reserved_targets: set[tuple[int, int]],
        rejections: list[tuple[int, str]],
    ) -> tuple[ExplorationCandidate, ...]:
        """Drop candidates that are evidently bad before ranking/selection.

        This is deliberately conservative: it removes near-self targets
        (near-zero-length routes), targets already blocked/reserved for this
        robot or a teammate, and targets sitting on a teammate's active route.
        It does not try to be a full collision planner.
        """

        min_distance = self._min_frontier_travel_distance(request)
        blocked = {
            self._target_key(point, request)
            for point in request.blocked_targets_by_robot.get(robot.robot_id, ())
        }

        accepted: list[ExplorationCandidate] = []
        for candidate in candidates:
            target = candidate.target
            key = self._target_key(target, request)

            distance_to_robot = math.hypot(target[0] - robot.xy[0], target[1] - robot.xy[1])
            if distance_to_robot < min_distance:
                rejections.append((robot.robot_id, "too_close_to_robot"))
                continue

            if key in blocked:
                rejections.append((robot.robot_id, "blocked_target"))
                continue

            if key in reserved_targets:
                rejections.append((robot.robot_id, "reservation_conflict"))
                continue

            if not self._is_far_from_other_routes(request, robot, target, min_distance):
                rejections.append((robot.robot_id, "route_conflict"))
                continue

            accepted.append(candidate)

        return tuple(accepted)

    def _is_far_from_other_routes(
        self,
        request: CoordinationRequest,
        robot: RobotCoordinationState,
        target: Point2D,
        min_distance: float,
    ) -> bool:
        routes = request.route_points_by_robot
        if len(routes) != len(request.robot_states):
            # Routes are only meaningful when aligned 1:1 with robot_states
            # (the runtime host guarantees this; ad-hoc requests may not).
            return True

        for index, other in enumerate(request.robot_states):
            if other.robot_id == robot.robot_id:
                continue
            for point in routes[index]:
                if math.hypot(target[0] - point[0], target[1] - point[1]) < min_distance:
                    return False
        return True

    def _min_frontier_travel_distance(self, request: CoordinationRequest) -> float:
        explicit = request.parameters.get(
            "min_frontier_travel_distance",
            request.shared.get("min_frontier_travel_distance"),
        )
        if explicit is not None:
            try:
                return max(float(explicit), 0.0)
            except (TypeError, ValueError):
                pass

        goal_tolerance = self._numeric_parameter(request, "goal_tolerance", 0.25)
        grid_resolution = self._numeric_parameter(request, "grid_resolution", 0.5)
        return max(2.0 * goal_tolerance, 2.0 * grid_resolution, 0.75)

    def _numeric_parameter(
        self,
        request: CoordinationRequest,
        key: str,
        default: float,
    ) -> float:
        value = request.parameters.get(key, request.shared.get(key, default))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _choose_candidate(
        self,
        candidates: Iterable[ExplorationCandidate],
        reserved_targets: set[tuple[int, int]],
        request: CoordinationRequest,
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
        request: CoordinationRequest,
    ) -> tuple[int, int]:
        resolution = float(request.parameters.get("reservation_resolution", 0.01))
        resolution = max(resolution, 1e-6)
        return (round(target[0] / resolution), round(target[1] / resolution))

    def _debug_source(
        self,
        request: CoordinationRequest,
        team_candidates: Mapping[int, tuple[ExplorationCandidate, ...]],
    ) -> str:
        if request.proposals_by_robot:
            return "explicit_proposals"
        if team_candidates:
            return "team_frontier_provider"
        if request.services is not None and request.services.frontier_provider is not None:
            return "single_robot_frontier_provider"
        return "none"


def create_plugin() -> CoordinationPlugin:
    return MmpfExplorePlugin()
