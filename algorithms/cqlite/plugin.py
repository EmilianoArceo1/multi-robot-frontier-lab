"""Simulator-native implementation of Latif and Parasuraman's CQLite.

The paper is the algorithm specification.  The authors' public ROS repository
does not contain the distributed Q update shown in Algorithm 1, so this module
implements equations (1), (5)--(7), the travel-time Voronoi partition in (9),
and the lite neighbor exchange directly against ``robotics_interfaces``.

This is deliberately a coordinator, not a SLAM implementation.  The host owns
frontier detection, occupancy mapping, path planning, and motion control.  A
persistent plugin instance owns only the per-robot Q tables and the compact
messages that would be exchanged by distributed robots.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
from robotics_interfaces.results import MetricsEvent, PathPlanningRequest


CQLITE_COORDINATOR = "CQLite distributed Q-learning"

# Wire-size proxy used by the native experiments.  One lite message contains
# robot/state id, x/y frontier coordinates, one Q value, and a small header.
# It intentionally excludes transport/ROS serialization overhead.
CQLITE_Q_UPDATE_PAYLOAD_BYTES = 40
CQLITE_EXPLORED_STATE_PAYLOAD_BYTES = 24

StateKey = tuple[int, int]


@dataclass
class _RobotLearner:
    q_values: dict[StateKey, float] = field(default_factory=dict)
    explored: set[StateKey] = field(default_factory=set)
    state_points: dict[StateKey, Point2D] = field(default_factory=dict)
    last_assigned_key: StateKey | None = None
    last_assigned_point: Point2D | None = None
    q_updates: int = 0
    cumulative_reward: float = 0.0


@dataclass(frozen=True)
class _ScoredCandidate:
    candidate: ExplorationCandidate
    key: StateKey
    path_length: float
    travel_time: float
    overlap_probability: float
    reward: float
    q_before: float
    q_after: float
    priority: float
    in_voronoi_region: bool


class CQLitePlugin:
    """Persistent, deterministic CQLite task-allocation adapter.

    Defaults marked ``paper`` below are reported explicitly in the article.
    ``rho`` and ``sigma`` are host assumptions because the article defines but
    does not publish numeric values for them.
    """

    metadata = PluginMetadata(
        name=CQLITE_COORDINATOR,
        version="1.0.0",
        description=(
            "Coverage-biased distributed Q-learning with per-robot Q tables, "
            "travel-time Voronoi allocation, overlap avoidance, and neighbor-"
            "only lite Q/state exchange."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TASK_ALLOCATION,
            PluginCapability.TARGET_GENERATION,
        ),
    )

    def __init__(self) -> None:
        self._learners: dict[int, _RobotLearner] = {}
        self._last_time_s: float | None = None
        self._decision_index = 0
        self._communication_bytes = 0
        self._message_count = 0
        self._map_merge_requests = 0

    def reset(self) -> None:
        """Clear all learned state at an experiment/run boundary."""
        self._learners.clear()
        self._last_time_s = None
        self._decision_index = 0
        self._communication_bytes = 0
        self._message_count = 0
        self._map_merge_requests = 0

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        self._reset_on_time_rewind(request)
        self._decision_index += 1

        robots_by_id = {int(robot.robot_id): robot for robot in request.robot_states}
        requested_ids = self._robots_to_assign(request, robots_by_id)
        for robot_id in robots_by_id:
            self._learners.setdefault(robot_id, _RobotLearner())

        alpha = self._bounded_parameter(request, "cqlite_alpha", 0.60, 0.0, 1.0)
        gamma = self._bounded_parameter(request, "cqlite_gamma", 0.95, 0.0, 1.0)
        step_cost = self._positive_parameter(request, "cqlite_step_cost", 2.0)
        overlap_radius = self._positive_parameter(request, "cqlite_overlap_radius", 1.0)
        communication_range = self._positive_parameter(request, "cqlite_communication_range", 50.0)
        rho = self._numeric_parameter(request, "cqlite_rho", 2.0)
        sigma = self._numeric_parameter(request, "cqlite_sigma", 0.01)
        speed = max(self._positive_parameter(request, "cqlite_nominal_speed", 0.5), 1e-6)
        information_weight = self._numeric_parameter(request, "cqlite_information_weight", 0.0)
        voronoi_fallback = self._bool_parameter(request, "cqlite_voronoi_fallback", True)
        min_travel = max(
            self._positive_parameter(request, "min_frontier_travel_distance", 0.75),
            0.0,
        )
        reservation_radius = max(
            self._positive_parameter(
                request,
                "target_exclusion_radius",
                self._positive_parameter(request, "reservation_resolution", 0.5),
            ),
            1e-6,
        )

        neighbors = self._neighbor_graph(robots_by_id, communication_range)
        call_messages = 0
        call_bytes = 0

        # Reassignment after a target disappears means the previous frontier
        # was reached, unless the host explicitly blacklisted that target.
        completed_by_robot: dict[int, tuple[StateKey, Point2D]] = {}
        for robot_id in requested_ids:
            learner = self._learners.get(robot_id)
            robot = robots_by_id.get(robot_id)
            if learner is None or robot is None or learner.last_assigned_key is None:
                continue
            point = learner.last_assigned_point
            if robot.current_target is not None or point is None:
                continue
            if self._point_near_any(
                point,
                request.blocked_targets_by_robot.get(robot_id, ()),
                reservation_radius,
            ):
                continue
            learner.explored.add(learner.last_assigned_key)
            learner.state_points[learner.last_assigned_key] = point
            completed_by_robot[robot_id] = (learner.last_assigned_key, point)

        # Algorithm 1 lines 22--24: share only the newly explored state and
        # the current Q update with graph neighbors, never the complete table.
        for sender, (key, point) in sorted(completed_by_robot.items()):
            q_value = self._learners[sender].q_values.get(key, 0.0)
            for receiver in neighbors.get(sender, ()):
                peer = self._learners[receiver]
                peer.explored.add(key)
                peer.state_points[key] = point
                peer.q_values[key] = max(peer.q_values.get(key, -math.inf), q_value)
                call_messages += 1
                call_bytes += CQLITE_EXPLORED_STATE_PAYLOAD_BYTES

        candidates_by_robot, source = self._candidate_pools(request, robots_by_id, requested_ids)

        # Existing targets remain hard reservations even though those robots
        # are not part of this incremental reassignment call.
        reserved_points: list[Point2D] = [
            robot.current_target
            for robot in request.robot_states
            if robot.current_target is not None and robot.robot_id not in requested_ids
        ]
        chosen_by_robot: dict[int, _ScoredCandidate] = {}
        scored_by_robot: dict[int, list[_ScoredCandidate]] = {}
        per_robot_debug: dict[str, object] = {}

        # Update Q values before allocation.  Robots with the fewest valid
        # actions are allocated first; this avoids starving one robot merely
        # because its id sorts later while preserving deterministic behavior.
        for robot_id in requested_ids:
            robot = robots_by_id.get(robot_id)
            if robot is None:
                continue
            raw = tuple(candidates_by_robot.get(robot_id, ()))
            scored, rejected = self._score_candidates(
                request=request,
                robot=robot,
                robots_by_id=robots_by_id,
                candidates=raw,
                alpha=alpha,
                gamma=gamma,
                step_cost=step_cost,
                overlap_radius=overlap_radius,
                communication_range=communication_range,
                rho=rho,
                sigma=sigma,
                speed=speed,
                information_weight=information_weight,
                min_travel=min_travel,
                reservation_radius=reservation_radius,
            )
            scored_by_robot[robot_id] = scored
            per_robot_debug[str(robot_id)] = {
                "raw_candidates": len(raw),
                "scored_candidates": len(scored),
                "rejected": rejected,
            }

        allocation_order = sorted(requested_ids, key=lambda rid: (len(scored_by_robot.get(rid, ())), rid))
        for robot_id in allocation_order:
            scored = scored_by_robot.get(robot_id, [])
            eligible = [item for item in scored if item.in_voronoi_region]
            used_voronoi_fallback = False
            if not eligible and voronoi_fallback:
                eligible = list(scored)
                used_voronoi_fallback = bool(eligible)
            eligible.sort(
                key=lambda item: (
                    item.priority,
                    item.q_after,
                    -item.travel_time,
                    -item.candidate.target[0],
                    -item.candidate.target[1],
                ),
                reverse=True,
            )
            selected = next(
                (
                    item
                    for item in eligible
                    if not self._point_near_any(item.candidate.target, reserved_points, reservation_radius)
                ),
                None,
            )
            if selected is not None:
                chosen_by_robot[robot_id] = selected
                reserved_points.append(selected.candidate.target)
                learner = self._learners[robot_id]
                learner.last_assigned_key = selected.key
                learner.last_assigned_point = selected.candidate.target
                debug = per_robot_debug[str(robot_id)]
                if isinstance(debug, dict):
                    debug.update(
                        {
                            "selected_state": list(selected.key),
                            "selected_target": list(selected.candidate.target),
                            "q_before": round(selected.q_before, 6),
                            "q_after": round(selected.q_after, 6),
                            "reward": round(selected.reward, 6),
                            "overlap_probability": round(selected.overlap_probability, 6),
                            "path_length": round(selected.path_length, 6),
                            "travel_time": round(selected.travel_time, 6),
                            "priority": round(selected.priority, 6),
                            "voronoi_fallback": used_voronoi_fallback,
                        }
                    )

                # Share one updated (state, Q) packet with immediate neighbors.
                for receiver in neighbors.get(robot_id, ()):
                    peer = self._learners[receiver]
                    peer.q_values[selected.key] = max(
                        peer.q_values.get(selected.key, -math.inf), selected.q_after
                    )
                    peer.state_points[selected.key] = selected.candidate.target
                    call_messages += 1
                    call_bytes += CQLITE_Q_UPDATE_PAYLOAD_BYTES

        assignments: list[CoordinationAssignment] = []
        commands: list[RobotCommand] = []
        targets_by_robot: dict[int, Point2D | None] = {}
        reasons_by_robot: dict[int, str] = {}

        for robot_id in requested_ids:
            robot = robots_by_id.get(robot_id)
            if robot is None:
                reason = "robot id not present in request.robot_states"
                assignments.append(CoordinationAssignment(robot_id, "FAILED", None, reason))
                commands.append(RobotCommand(robot_id=robot_id, status="FAILED", reason=reason))
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                continue

            selected = chosen_by_robot.get(robot_id)
            if selected is None:
                self._map_merge_requests += 1
                reason = (
                    "CQLite ad-hoc map merge requested; no new feasible frontier "
                    "remained after explored/Voronoi/reservation filters"
                )
                assignments.append(
                    CoordinationAssignment(robot_id=robot_id, status="HOLD", target=None, reason=reason)
                )
                commands.append(RobotCommand(robot_id=robot_id, status="HOLD", reason=reason))
                targets_by_robot[robot_id] = None
                reasons_by_robot[robot_id] = reason
                continue

            metadata = {
                **dict(selected.candidate.metadata),
                "cqlite_state_key": selected.key,
                "cqlite_q_before": selected.q_before,
                "cqlite_q_after": selected.q_after,
                "cqlite_reward": selected.reward,
                "cqlite_overlap_probability": selected.overlap_probability,
                "cqlite_path_length": selected.path_length,
                "cqlite_travel_time": selected.travel_time,
                "cqlite_priority": selected.priority,
                "cqlite_in_voronoi_region": selected.in_voronoi_region,
            }
            proposal = replace(
                selected.candidate,
                travel_cost=selected.path_length,
                overlap_cost=selected.overlap_probability,
                metadata=metadata,
            )
            reason = (
                f"CQLite q={selected.q_after:.3f}, reward={selected.reward:.3f}, "
                f"travel={selected.travel_time:.2f}s, overlap={selected.overlap_probability:.2f}"
            )
            assignments.append(
                CoordinationAssignment(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=selected.candidate.target,
                    reason=reason,
                    proposal=proposal,
                )
            )
            commands.append(
                RobotCommand(
                    robot_id=robot_id,
                    status="ASSIGNED",
                    target=selected.candidate.target,
                    heading_rad=selected.candidate.heading_rad,
                    reason=reason,
                    metadata=metadata,
                )
            )
            targets_by_robot[robot_id] = selected.candidate.target
            reasons_by_robot[robot_id] = reason

        self._message_count += call_messages
        self._communication_bytes += call_bytes
        self._last_time_s = float(request.time_s)

        targets = tuple(
            targets_by_robot.get(robot.robot_id, robot.current_target)
            for robot in request.robot_states
        )
        reasons = tuple(
            reasons_by_robot.get(
                robot.robot_id,
                "kept existing target" if robot.current_target is not None else "not requested",
            )
            for robot in request.robot_states
        )
        edge_count = sum(len(value) for value in neighbors.values()) // 2
        debug = {
            "plugin": self.metadata.name,
            "decision_index": self._decision_index,
            "candidate_source": source,
            "robots_to_assign": list(requested_ids),
            "parameters": {
                "alpha": alpha,
                "gamma": gamma,
                "step_cost": step_cost,
                "overlap_radius": overlap_radius,
                "communication_range": communication_range,
                "rho": rho,
                "sigma": sigma,
                "nominal_speed": speed,
                "information_weight": information_weight,
                "voronoi_fallback": voronoi_fallback,
            },
            "network": {
                "undirected_edge_count": edge_count,
                "neighbors_by_robot": {str(key): list(value) for key, value in neighbors.items()},
            },
            "communication": {
                "messages_this_decision": call_messages,
                "payload_bytes_this_decision": call_bytes,
                "messages_cumulative": self._message_count,
                "payload_bytes_cumulative": self._communication_bytes,
                "map_merge_requests_cumulative": self._map_merge_requests,
                "payload_model": "compact fields only; excludes middleware/transport overhead",
            },
            "q_table_sizes": {
                str(robot_id): len(learner.q_values)
                for robot_id, learner in sorted(self._learners.items())
            },
            "q_updates": {
                str(robot_id): learner.q_updates
                for robot_id, learner in sorted(self._learners.items())
            },
            "per_robot": per_robot_debug,
        }
        self._record_metrics(request, debug)

        return CoordinationResult(
            targets=targets,
            reasons=reasons,
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            debug=debug,
            commands=tuple(commands),
        )

    def _score_candidates(
        self,
        *,
        request: CoordinationRequest,
        robot: RobotCoordinationState,
        robots_by_id: Mapping[int, RobotCoordinationState],
        candidates: tuple[ExplorationCandidate, ...],
        alpha: float,
        gamma: float,
        step_cost: float,
        overlap_radius: float,
        communication_range: float,
        rho: float,
        sigma: float,
        speed: float,
        information_weight: float,
        min_travel: float,
        reservation_radius: float,
    ) -> tuple[list[_ScoredCandidate], dict[str, int]]:
        learner = self._learners[robot.robot_id]
        rejected = {
            "invalid": 0,
            "too_close": 0,
            "blocked": 0,
            "already_explored": 0,
            "unreachable": 0,
        }
        normalized: list[ExplorationCandidate] = []
        seen: set[StateKey] = set()
        for candidate in candidates:
            point = self._normalize_point(candidate.target)
            if point is None:
                rejected["invalid"] += 1
                continue
            key = self._state_key(point, request)
            if key in seen:
                continue
            seen.add(key)
            candidate = replace(candidate, target=point)
            if key in learner.explored:
                # Equation (7) still updates the loop-back action negatively,
                # but it is not offered to the allocator afterward.  Check
                # this before the near-robot gate because a just-completed
                # frontier is normally exactly at the robot's current pose.
                old = learner.q_values.get(key, 0.0)
                learner.q_values[key] = (1.0 - alpha) * old + alpha * (-step_cost)
                learner.q_updates += 1
                learner.cumulative_reward -= step_cost
                rejected["already_explored"] += 1
                continue
            distance = self._distance(robot.xy, point)
            if distance < min_travel:
                rejected["too_close"] += 1
                continue
            if self._point_near_any(
                point,
                request.blocked_targets_by_robot.get(robot.robot_id, ()),
                reservation_radius,
            ):
                rejected["blocked"] += 1
                continue
            normalized.append(candidate)

        if not normalized:
            return [], rejected

        prior_next_max = max(
            (learner.q_values.get(self._state_key(item.target, request), 0.0) for item in normalized),
            default=0.0,
        )
        max_information = max((max(float(item.information_gain), 0.0) for item in normalized), default=1.0)
        if max_information <= 0.0:
            max_information = 1.0

        scored: list[_ScoredCandidate] = []
        for candidate in normalized:
            point = candidate.target
            key = self._state_key(point, request)
            learner.state_points[key] = point
            path_length = self._path_length(request, robot, point)
            if not math.isfinite(path_length):
                rejected["unreachable"] += 1
                continue

            travel_time = path_length / speed
            overlap = self._overlap_probability(robot.robot_id, point, overlap_radius)
            old_q = learner.q_values.get(key, 0.0)
            reward = step_cost - old_q + rho * (1.0 - overlap) + sigma * communication_range
            new_q = (1.0 - alpha) * old_q + alpha * (reward + gamma * prior_next_max)
            learner.q_values[key] = new_q
            learner.q_updates += 1
            learner.cumulative_reward += reward

            information_bonus = information_weight * max(float(candidate.information_gain), 0.0) / max_information
            priority = new_q - step_cost * travel_time + information_bonus
            own_time = self._distance(robot.xy, point) / speed
            best_time = min(
                (self._distance(other.xy, point) / speed for other in robots_by_id.values() if other.is_active),
                default=own_time,
            )
            in_voronoi = own_time <= best_time + 1e-9
            scored.append(
                _ScoredCandidate(
                    candidate=candidate,
                    key=key,
                    path_length=path_length,
                    travel_time=travel_time,
                    overlap_probability=overlap,
                    reward=reward,
                    q_before=old_q,
                    q_after=new_q,
                    priority=priority,
                    in_voronoi_region=in_voronoi,
                )
            )
        return scored, rejected

    def _candidate_pools(
        self,
        request: CoordinationRequest,
        robots_by_id: Mapping[int, RobotCoordinationState],
        requested_ids: tuple[int, ...],
    ) -> tuple[dict[int, tuple[ExplorationCandidate, ...]], str]:
        if request.proposals_by_robot:
            return (
                {
                    robot_id: tuple(self._as_candidate(value) for value in request.proposals_by_robot.get(robot_id, ()))
                    for robot_id in requested_ids
                },
                "explicit_proposals",
            )

        services = request.services
        team_provider = getattr(services, "team_frontier_provider", None) if services is not None else None
        if team_provider is not None:
            pools = {
                int(robot_id): tuple(self._as_candidate(value) for value in values)
                for robot_id, values in team_provider.candidates_for_team(request).items()
            }
            if any(pools.values()):
                return pools, "team_frontier_provider"

        frontier_information = (
            getattr(services, "frontier_information_service", None) if services is not None else None
        )
        if frontier_information is not None:
            clusters = tuple(frontier_information.get_frontier_clusters())
            candidates: list[ExplorationCandidate] = []
            for cluster in clusters:
                if not bool(getattr(cluster, "valid", True)):
                    continue
                if getattr(cluster, "viewpoints", ()):
                    for viewpoint in cluster.viewpoints:
                        candidate = viewpoint.as_exploration_candidate(
                            source="frontier_information_service"
                        )
                        candidates.append(
                            replace(
                                candidate,
                                metadata={
                                    **dict(candidate.metadata),
                                    "cluster_id": str(getattr(cluster, "cluster_id", "")),
                                    "frontier_cell_count": len(getattr(cluster, "cells", ())),
                                },
                            )
                        )
                elif getattr(cluster, "centroid", None) is not None:
                    candidates.append(
                        ExplorationCandidate(
                            target=cluster.centroid,
                            source="frontier_information_service",
                            information_gain=float(getattr(cluster, "information_gain", 0.0)),
                            metadata={"cluster_id": str(getattr(cluster, "cluster_id", ""))},
                        )
                    )
            if candidates:
                shared = tuple(candidates)
                return {robot_id: shared for robot_id in requested_ids}, "frontier_information_service"

        frontier_provider = getattr(services, "frontier_provider", None) if services is not None else None
        if frontier_provider is not None and request.world is not None:
            pools = {}
            for robot_id in requested_ids:
                robot = robots_by_id.get(robot_id)
                if robot is None:
                    continue
                pools[robot_id] = tuple(
                    self._as_candidate(value)
                    for value in frontier_provider.candidates_for_robot(
                        robot,
                        request.world,
                        request.blocked_targets_by_robot.get(robot_id, ()),
                    )
                )
            return pools, "frontier_provider"
        return {}, "none"

    def _path_length(
        self,
        request: CoordinationRequest,
        robot: RobotCoordinationState,
        target: Point2D,
    ) -> float:
        use_service = self._bool_parameter(request, "cqlite_use_path_service", False)
        service = getattr(request.services, "path_planning_service", None) if request.services else None
        if use_service and service is not None and request.world is not None and request.world.bounds is not None:
            response = service.plan_path(
                PathPlanningRequest(
                    start=robot.xy,
                    goal=target,
                    robot_radius=robot.safety_radius,
                    bounds=request.world.bounds,
                    resolution=request.world.resolution,
                    obstacle_points=request.world.mapped_obstacle_points,
                    planner_type=str(request.parameters.get("cqlite_path_planner", "A*")),
                    robot_id=robot.robot_id,
                    metadata={"purpose": "cqlite_travel_time"},
                )
            )
            if not response.success:
                return math.inf
            points = (robot.xy,) + tuple(response.waypoints)
            return sum(self._distance(a, b) for a, b in zip(points, points[1:]))
        return self._distance(robot.xy, target)

    def _overlap_probability(self, robot_id: int, point: Point2D, radius: float) -> float:
        known: dict[StateKey, Point2D] = {}
        learner = self._learners[robot_id]
        for key in learner.explored:
            known[key] = learner.state_points.get(key, point)
        if not known:
            return 0.0
        overlaps = sum(1 for value in known.values() if self._distance(point, value) <= radius)
        return overlaps / len(known)

    def _neighbor_graph(
        self,
        robots_by_id: Mapping[int, RobotCoordinationState],
        communication_range: float,
    ) -> dict[int, tuple[int, ...]]:
        graph: dict[int, list[int]] = {robot_id: [] for robot_id in robots_by_id}
        ids = sorted(robots_by_id)
        for index, left_id in enumerate(ids):
            for right_id in ids[index + 1 :]:
                if self._distance(robots_by_id[left_id].xy, robots_by_id[right_id].xy) <= communication_range:
                    graph[left_id].append(right_id)
                    graph[right_id].append(left_id)
        return {key: tuple(value) for key, value in graph.items()}

    def _reset_on_time_rewind(self, request: CoordinationRequest) -> None:
        now = float(request.time_s)
        if self._last_time_s is not None and now + 1e-9 < self._last_time_s:
            self.reset()

    def _robots_to_assign(
        self,
        request: CoordinationRequest,
        robots_by_id: Mapping[int, RobotCoordinationState],
    ) -> tuple[int, ...]:
        if request.robots_to_assign:
            return tuple(sorted({int(robot_id) for robot_id in request.robots_to_assign}))
        return tuple(sorted(robot_id for robot_id, robot in robots_by_id.items() if robot.is_active))

    def _state_key(self, point: Point2D, request: CoordinationRequest) -> StateKey:
        resolution = max(
            self._positive_parameter(
                request,
                "cqlite_state_resolution",
                self._positive_parameter(request, "grid_resolution", 0.5),
            ),
            1e-6,
        )
        return (round(point[0] / resolution), round(point[1] / resolution))

    def _as_candidate(self, value: ExplorationCandidate | CandidateProposal) -> ExplorationCandidate:
        if isinstance(value, ExplorationCandidate):
            return value
        if isinstance(value, CandidateProposal):
            return value.as_candidate(source="explicit_proposal")
        return ExplorationCandidate(
            target=getattr(value, "target"),
            source="duck_typed_proposal",
            information_gain=float(getattr(value, "information_gain", 0.0)),
            travel_cost=float(getattr(value, "travel_cost", 0.0)),
            metadata=dict(getattr(value, "metadata", {}) or {}),
        )

    def _record_metrics(self, request: CoordinationRequest, debug: Mapping[str, object]) -> None:
        service = getattr(request.services, "metrics_service", None) if request.services else None
        if service is None:
            return
        try:
            service.record_event(MetricsEvent(name="cqlite_decision", data=dict(debug)))
        except Exception:
            # Metrics must never change an allocation decision.
            return

    def _numeric_parameter(self, request: CoordinationRequest, key: str, default: float) -> float:
        value = request.parameters.get(key, request.shared.get(key, default))
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if math.isfinite(result) else float(default)

    def _positive_parameter(self, request: CoordinationRequest, key: str, default: float) -> float:
        return max(self._numeric_parameter(request, key, default), 0.0)

    def _bounded_parameter(
        self,
        request: CoordinationRequest,
        key: str,
        default: float,
        low: float,
        high: float,
    ) -> float:
        return min(max(self._numeric_parameter(request, key, default), low), high)

    def _bool_parameter(self, request: CoordinationRequest, key: str, default: bool) -> bool:
        value = request.parameters.get(key, request.shared.get(key, default))
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)

    @staticmethod
    def _normalize_point(value: object) -> Point2D | None:
        try:
            x, y = value  # type: ignore[misc]
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return (x, y)

    @staticmethod
    def _distance(left: Point2D, right: Point2D) -> float:
        return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))

    @classmethod
    def _point_near_any(cls, point: Point2D, values: Iterable[Point2D], radius: float) -> bool:
        return any(cls._distance(point, value) <= radius for value in values)


def create_plugin() -> CoordinationPlugin:
    return CQLitePlugin()
