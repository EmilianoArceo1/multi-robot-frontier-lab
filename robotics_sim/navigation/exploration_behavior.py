"""
ExplorationBehavior — continuous frontier-to-frontier exploration policy.

Owns the decision logic for smooth, non-stop exploration:
  • when to prefetch the next target (before reaching the current one),
  • when to accept a prefetched path (smooth turn) or brake (sharp turn),
  • when to HOLD because no frontier is available,
  • when to trigger a safety replan.

What does NOT live here:
  • Qt, canvas, widgets — zero rendering.
  • Robot physics, dynamics, WaypointManager internals.
  • Path planning math — delegated to PlannerServices.
  • Multi-robot frontier coordination — still handled by the engine's
    MultiRobotCoordinator while the migration is in progress.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robotics_sim.navigation.navigation_decision import (
    NavigationDecision,
    accept_pending_path,
    brake,
    follow,
    hold,
    prefetch_next_target,
    replan_for_safety,
    request_plan,
)
from robotics_sim.simulation.navigation_modes import is_goal_seeking_planner

if TYPE_CHECKING:
    from robotics_sim.core.robot_agent import RobotAgent
    from robotics_sim.simulation.observation import RobotObservation
    from robotics_sim.simulation.planner_services import PlannerServices


class ExplorationBehavior:
    """
    Exploration navigation policy for one robot agent.

    Create one instance per RobotAgent and call update() each simulation step.

    Parameters
    ----------
    prefetch_distance_factor:
        Multiplier applied to grid_resolution to compute the distance at which
        prefetch starts.  Default 2.5 gives about 2–3 cells of look-ahead.
    max_smooth_turn_angle_rad:
        If the heading change required to start the pending path is below this
        threshold the robot does NOT brake when switching paths.  Default π/3
        (60°) allows moderate turns; set lower for tighter constraints.
    """

    def __init__(
        self,
        *,
        prefetch_distance_factor: float = 2.5,
        max_smooth_turn_angle_rad: float = math.pi / 3,
    ) -> None:
        self.prefetch_distance_factor = float(prefetch_distance_factor)
        self.max_smooth_turn_angle_rad = float(max_smooth_turn_angle_rad)

    # ------------------------------------------------------------------ thresholds

    def prefetch_distance(self, grid_resolution: float, goal_tolerance: float) -> float:
        """Distance from the current frontier at which prefetch should start."""
        return max(
            0.75,
            self.prefetch_distance_factor * float(grid_resolution),
            2.0 * float(goal_tolerance),
        )

    # ------------------------------------------------------------------ helpers

    _PREFETCH_COOLDOWN: float = 0.75  # seconds between consecutive prefetch attempts

    def should_prefetch_next_target(
        self,
        agent: "RobotAgent",
        observation: "RobotObservation",
    ) -> bool:
        """True when the agent is close enough to start computing the next target."""
        if agent.pending_path is not None:
            return False  # path already delivered, waiting for ACCEPT
        if agent.pending_target_xy is not None:
            return False  # worker in-flight, result not yet arrived
        if agent.active_target() is None:
            return False
        if observation.current_time - agent.last_prefetch_time < self._PREFETCH_COOLDOWN:
            return False
        # Measure distance to the END of the current route, not just the next
        # waypoint, so we only prefetch once the frontier is nearly reached.
        dist = agent.distance_to_active_path_goal()
        threshold = self.prefetch_distance(observation.grid_resolution, observation.goal_tolerance)
        return dist <= threshold

    def _turn_angle(
        self,
        agent: "RobotAgent",
        path: list,
    ) -> float:
        """Absolute angle between the agent heading and the first segment of *path*."""
        if not path:
            return 0.0
        tx, ty = float(path[0][0]), float(path[0][1])
        dx = tx - agent.position[0]
        dy = ty - agent.position[1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        angle_to = math.atan2(dy, dx)
        diff = angle_to - agent.heading
        # Normalize to (-π, π]
        while diff > math.pi:
            diff -= 2.0 * math.pi
        while diff < -math.pi:
            diff += 2.0 * math.pi
        return abs(diff)

    def should_brake_for_turn(
        self,
        agent: "RobotAgent",
        next_path: list,
    ) -> bool:
        """True when the first segment of *next_path* requires a sharp turn."""
        return self._turn_angle(agent, next_path) > self.max_smooth_turn_angle_rad

    # ------------------------------------------------------------------ main decision

    def update(
        self,
        agent: "RobotAgent",
        observation: "RobotObservation",
        planner_services: "PlannerServices",
    ) -> NavigationDecision:
        """
        Compute one navigation decision for this frame.

        Called by RobotAgent.step() when the planner mode is an exploration mode
        (i.e. not "Goal seeking").

        Decision priority (highest first):
          1. Safety: blocked or predicted collision → REPLAN_FOR_SAFETY
          2. Accept pending path when close to current frontier (smooth turn)
             or BRAKE (sharp turn)
          3. Current frontier reached → pick next, REQUEST_PLAN or HOLD
          4. Approaching frontier → PREFETCH_NEXT_TARGET (background)
          5. Following current path → FOLLOW_PATH
          6. No path at all → REQUEST_PLAN (first frontier) or HOLD
        """
        # ── 1. Safety ────────────────────────────────────────────────────────
        if observation.active_segment_blocked or observation.predicted_collision:
            reason = (
                "predicted collision"
                if observation.predicted_collision
                else "active segment blocked"
            )
            agent.safety_replan_count += 1
            return replan_for_safety(agent.desired_target_from_mode(), reason=reason)

        # ── 2. Pending path ready — should we switch? ─────────────────────
        if agent.pending_path is not None:
            same_target_radius = max(
                observation.grid_resolution, 2.0 * observation.goal_tolerance
            )
            # Discard silently if the prefetch landed on the same frontier we're
            # already following (FoV-aware hysteresis can return the same target).
            if (
                agent.pending_target_xy is not None
                and agent.active_path_goal_xy is not None
                and math.hypot(
                    agent.pending_target_xy[0] - agent.active_path_goal_xy[0],
                    agent.pending_target_xy[1] - agent.active_path_goal_xy[1],
                ) <= same_target_radius
            ):
                agent.reject_pending_path("pending target equals current target; discarding")
                # fall through — no brake, no route reset, robot keeps moving
            else:
                dist = agent.distance_to_active_target()
                threshold = self.prefetch_distance(
                    observation.grid_resolution, observation.goal_tolerance
                )
                if dist <= threshold:
                    sharp = self.should_brake_for_turn(agent, agent.pending_path)
                    reason = (
                        "accepting prefetched path; sharp turn handled by robot controller"
                        if sharp
                        else "accepting prefetched path; turn is smooth"
                    )
                    return accept_pending_path(reason=reason)

        # ── 3. Frontier reached — need a new target ───────────────────────
        target = agent.active_target()
        if target is not None and agent.distance_to_active_target() <= observation.goal_tolerance:
            agent.target_switch_count += 1
            # Clear the reached frontier so that neither _pick_next_target()
            # nor the engine's select_navigation_goal() can return it again
            # via hysteresis (both use agent.exploration_target_xy as their
            # current_target hint).
            agent.exploration_target_xy = None
            next_target = self._pick_next_target(agent, observation, planner_services)
            if next_target is None:
                agent.stop_count_exploration += 1
                return hold(reason="frontier reached; no valid next frontier available")
            return request_plan(
                next_target,
                reason="frontier reached; requesting next frontier",
                force_new_target=True,
            )

        # ── 4. Approaching — prefetch next target ────────────────────────
        if self.should_prefetch_next_target(agent, observation):
            next_target = self._pick_next_target(agent, observation, planner_services)
            if next_target is not None:
                # Skip if the planner re-proposes the same frontier (hysteresis).
                same_target_radius = max(
                    observation.grid_resolution, 2.0 * observation.goal_tolerance
                )
                if (
                    agent.active_path_goal_xy is not None
                    and math.hypot(
                        next_target[0] - agent.active_path_goal_xy[0],
                        next_target[1] - agent.active_path_goal_xy[1],
                    ) <= same_target_radius
                ):
                    pass  # same target — fall through to FOLLOW
                else:
                    # Mark in-flight immediately so should_prefetch_next_target
                    # returns False next frame without waiting for the worker.
                    agent.pending_target_xy = next_target
                    agent.last_prefetch_time = observation.current_time
                    return prefetch_next_target(
                        next_target, reason="approaching frontier; prefetching next"
                    )
            # No valid next target, or same as current: keep current route.
            agent.prefetch_fail_count += 1

        # ── 5. Follow current path ───────────────────────────────────────
        if target is not None:
            return follow(target, reason="following active path to frontier")

        # ── 6. No path — need first plan ─────────────────────────────────
        desired = agent.desired_target_from_mode()
        if desired is None:
            agent.stop_count_exploration += 1
            return hold(reason="no target available in current exploration mode")

        return request_plan(desired, reason="no active path; requesting initial frontier plan")

    # ------------------------------------------------------------------ private

    def _pick_next_target(
        self,
        agent: "RobotAgent",
        observation: "RobotObservation",
        planner_services: "PlannerServices",
    ) -> tuple[float, float] | None:
        """
        Ask PlannerServices for the next exploration target.

        Returns None when no valid frontier is available (map fully explored,
        planner unavailable, etc.).  Never falls back to final_goal_xy in
        exploration mode.
        """
        if is_goal_seeking_planner(agent.planner_mode):
            # Should not be called in goal-seeking, but guard defensively.
            return agent.final_goal_xy

        result = planner_services.select_exploration_target(
            planner_name=agent.planner_mode,
            belief_map=observation.belief_map,
            robot_xy=observation.robot_xy,
            robot_heading=observation.robot_heading,
            current_target=agent.exploration_target_xy,
            final_goal_xy=observation.final_goal_xy,
            robot_radius=observation.robot_radius,
            sensor_range=observation.sensor_range,
            vision_model=observation.vision_model,
            ipp_distance_penalty=observation.ipp_distance_penalty,
            excluded_targets=list(observation.excluded_targets),
        )

        if not result.success or result.target is None:
            return None

        return (float(result.target[0]), float(result.target[1]))
