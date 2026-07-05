from __future__ import annotations

from robotics_interfaces import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
    PluginCapability,
    PluginMetadata,
)


NOIC_COORDINATOR = "NOIC information coordinator"


class GlobalNoicLegacyPlugin:
    """Adapter for the current NOIC/coordinated-frontier behavior.

    This plugin intentionally preserves the existing behavior while moving the
    algorithm boundary outside robotics_sim.simulation.coordination.

    During migration, the simulator host injects the existing coordinated
    frontier planner as request.shared["legacy_assign_frontier_viewpoints"].
    That inversion keeps this plugin import-independent from robotics_sim.
    """

    metadata = PluginMetadata(
        name=NOIC_COORDINATOR,
        version="0.1.0",
        description="Legacy NOIC coordinator wrapped as a dynamic plugin.",
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TASK_ALLOCATION,
        ),
        source="legacy simulator NOIC / coordinated frontier planner",
    )

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        count = len(request.robot_states)
        targets: list[tuple[float, float] | None] = [
            request.existing_targets_by_robot.get(index)
            for index in range(count)
        ]
        reasons: list[str] = [
            "kept existing target" if targets[index] is not None else "no target assigned yet"
            for index in range(count)
        ]

        assign_set = {
            int(index)
            for index in request.robots_to_assign
            if 0 <= int(index) < count
        }

        if not assign_set:
            return CoordinationResult(
                targets=tuple(targets),
                reasons=tuple(reasons),
                strategy=self.metadata.name,
                assignments=tuple(
                    CoordinationAssignment(
                        robot_id=index,
                        status="ASSIGNED" if targets[index] is not None else "HOLD",
                        target=targets[index],
                        reason=reasons[index],
                    )
                    for index in range(count)
                ),
                debug={"algorithm": self.metadata.name, "assigned_count": 0},
            )

        legacy_assign = request.shared.get("legacy_assign_frontier_viewpoints")
        if not callable(legacy_assign):
            raise ValueError(
                "global_noic_legacy requires request.shared['legacy_assign_frontier_viewpoints']"
            )

        planner_res = legacy_assign(
            robot_states=request.robot_states,
            existing_targets=targets,
            robots_to_assign=sorted(assign_set),
            invalidated_targets_by_robot=_blocked_targets_as_sequence(request, count),
            explored_points=request.shared.get("explored_points", ()),
            mapped_obstacle_points=request.shared.get("mapped_obstacle_points", ()),
            bounds=request.shared.get("bounds", (0.0, 0.0, 0.0, 0.0)),
            resolution=request.shared.get("resolution", 1.0),
            final_goal_xy=request.shared.get("final_goal_xy", (0.0, 0.0)),
            ipp_distance_penalty=request.shared.get("ipp_distance_penalty", 0.5),
            target_exclusion_radius=request.shared.get("target_exclusion_radius", 1.5),
            dynamic_obstacle_margin=request.shared.get("dynamic_obstacle_margin", 0.5),
            route_points_by_robot=request.route_points_by_robot,
            explored_points_by_robot=request.shared.get("explored_points_by_robot", ()),
        )

        assigned_indices: set[int] = set()
        assignments: list[CoordinationAssignment] = []

        for index in range(count):
            planner_assignment = (
                planner_res.assignments[index]
                if index < len(planner_res.assignments)
                else None
            )

            if index in assign_set and planner_assignment is not None:
                target = tuple(planner_assignment.target)
                targets[index] = target
                reasons[index] = (
                    f"{self.metadata.name}: asignación coordinada; "
                    f"{planner_assignment.reason}; "
                    f"info_gain={planner_assignment.information_gain:.1f}; "
                    f"dist={planner_assignment.distance:.1f}; "
                    f"map_overlap={planner_assignment.other_map_ratio:.2f}; "
                    f"route_overlap={planner_assignment.route_overlap_ratio:.2f}"
                )
                assigned_indices.add(index)
                assignments.append(
                    CoordinationAssignment(
                        robot_id=index,
                        status="ASSIGNED",
                        target=target,
                        reason=reasons[index],
                    )
                )
            else:
                assignments.append(
                    CoordinationAssignment(
                        robot_id=index,
                        status="HOLD" if targets[index] is None else "ASSIGNED",
                        target=targets[index],
                        reason=reasons[index],
                    )
                )

        for index in sorted(assign_set - assigned_indices):
            targets[index] = None
            planner_reason = (
                planner_res.reasons[index]
                if index < len(planner_res.reasons)
                and planner_res.reasons[index] != "no target assigned yet"
                else ""
            )
            suffix = f"; {planner_reason}" if planner_reason else ""
            reasons[index] = (
                f"{self.metadata.name}: En espera (HOLD) para evitar solapamiento "
                f"de visión o rutas cruzadas{suffix}"
            )
            assignments[index] = CoordinationAssignment(
                robot_id=index,
                status="HOLD",
                target=None,
                reason=reasons[index],
            )

        return CoordinationResult(
            targets=tuple(targets),
            reasons=tuple(reasons),
            strategy=self.metadata.name,
            assignments=tuple(assignments),
            debug={
                "algorithm": self.metadata.name,
                "assigned_indices": tuple(sorted(assigned_indices)),
                "requested_indices": tuple(sorted(assign_set)),
            },
        )


def _blocked_targets_as_sequence(
    request: CoordinationRequest,
    count: int,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        tuple(request.blocked_targets_by_robot.get(index, ()))
        for index in range(count)
    )


def create_plugin() -> GlobalNoicLegacyPlugin:
    return GlobalNoicLegacyPlugin()
    