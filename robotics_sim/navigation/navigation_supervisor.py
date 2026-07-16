"""
NavigationSupervisor — centralized single-robot navigation invariants.

Context
-------
Recent bug fixes (route-less REPLAN_FOR_SAFETY, routes accepted without
reaching their goal, targets re-planned after already being reached, engine
re-deriving a target ExplorationBehavior already chose) each got fixed in a
different place: RobotAgent.step(), ExplorationBehavior.update(), and three
separate spots inside engine.py's apply_navigation_decision()/
apply_route_result()/on_prefetch_route_ready(). Each fix was individually
correct, but the underlying invariants they enforce were never written down
in one place, so every new bug in this area meant re-discovering where the
relevant check should live.

This module does not replace target scoring (still ExplorationBehavior /
PlannerServices), A* (still planner_registry / grid_planners), or the robot
controller (still Robot / engine physics). It is a small, stateless set of
guards that the engine and RobotAgent can call before acting on a
NavigationDecision or a planned route, so these invariants exist in exactly
one testable place instead of being reimplemented ad hoc at each call site
that happens to need them.

Invariants centralized here
----------------------------
- No active route means REPLAN_FOR_SAFETY is meaningless -> normalize to HOLD.
- No planning to a target already within goal_tolerance of the robot.
- No route accepted unless its final waypoint actually reaches the goal it
  was planned for (or a prefetched route reaches the pending target).

Everything here is pure: no Qt, no canvas, no robot physics, no mutation of
the arguments passed in. Callers (engine.py) are still the ones that apply
the resulting decision to robot/canvas/telemetry state.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from robotics_sim.navigation.navigation_decision import NavigationDecision, hold

if TYPE_CHECKING:
    from robotics_sim.core.robot_agent import RobotAgent
    from robotics_sim.simulation.observation import RobotObservation

Point2D = tuple[float, float]


class NavigationSupervisor:
    """Stateless guards for single-robot exploration navigation decisions."""

    @staticmethod
    def normalize_decision(
        agent: "RobotAgent | None",
        observation: "RobotObservation | None",
        decision: NavigationDecision,
        goal_tolerance: float,
        map_signature: int = 0,
    ) -> NavigationDecision:
        """Return *decision*, or a HOLD replacing it when it violates an invariant.

        Two cases are normalized here today:

        1. REPLAN_FOR_SAFETY with no active route. RobotAgent.step() emits
           this unconditionally whenever active_segment_blocked/
           predicted_collision is set, even with nothing to replan (e.g.
           right after exploration_exhausted() put the agent into a stable
           HOLD). A route-less safety replan means nothing to the planner
           and should never reach the engine's REPLAN_FOR_SAFETY handling.
        2. REQUEST_PLAN whose target is already within goal_tolerance of the
           robot. Requesting a route here produces a near-zero-length route
           that gets "reached" again within a tick or two -- the
           [ROUTE ok] ... length=0.02 loop.

        Any other decision kind is returned unchanged.
        """
        if decision.kind == "REPLAN_FOR_SAFETY":
            if not NavigationSupervisor._has_active_route(agent):
                reason = decision.reason or "no active route"
                if agent is not None and agent.exploration_exhausted(map_signature=map_signature):
                    # Prefer the exhaustion reason so the console/logs stay
                    # coherent: "exhausted, holding" beats "no active route,
                    # holding" when both are simultaneously true.
                    reason = "exploration exhausted: no reachable frontier candidates"
                return hold(reason=reason)
            return decision

        if decision.kind == "REQUEST_PLAN":
            # target=None is a legitimate "let the caller derive its own
            # target" signal (e.g. request_route_async() falling back to
            # select_navigation_goal()) -- not a reachability question, so
            # it must pass through unchanged rather than being treated as
            # "nothing to request".
            robot_xy = getattr(observation, "robot_xy", None) if observation is not None else None
            if (
                decision.target is not None
                and robot_xy is not None
                and not NavigationSupervisor.should_request_route(
                    robot_xy,
                    decision.target,
                    goal_tolerance,
                    active_path_goal=getattr(agent, "active_path_goal_xy", None) if agent is not None else None,
                    pending_target=getattr(agent, "pending_target_xy", None) if agent is not None else None,
                )
            ):
                return hold(reason="target already within goal_tolerance of robot position")
            return decision

        return decision

    @staticmethod
    def should_request_route(
        robot_xy: Point2D | None,
        target: Point2D | None,
        goal_tolerance: float,
        *,
        active_path_goal: Point2D | None = None,
        pending_target: Point2D | None = None,
    ) -> bool:
        """False when no new route request should be launched for *target*.

        active_path_goal/pending_target are accepted for API symmetry with
        the rest of this phase's invariants (a future check could reject a
        duplicate request for a target already active or already
        prefetched); the current guard only enforces the "already reached"
        invariant that caused the observed bug.
        """
        if robot_xy is None or target is None:
            return False
        return not NavigationSupervisor._within_tolerance(robot_xy, target, goal_tolerance)

    @staticmethod
    def validate_route_endpoint(
        waypoints: list[Point2D],
        goal: Point2D | None,
        tolerance: float,
    ) -> bool:
        """True when the route's final waypoint is within *tolerance* of *goal*.

        A "successful" route (compute_planned_waypoints returning
        success=True, e.g. after relocating an occupied goal cell to the
        nearest traversable one) does not always actually terminate at the
        goal it was asked to reach. Accepting such a route anyway means the
        robot follows it faithfully to a different endpoint and then gets
        stuck there: whatever tracks the route's intended destination
        (active_path_goal_xy / pending_target_xy) keeps pointing at the
        original goal, so distance-to-goal never drops below goal_tolerance
        and "frontier reached" never fires.
        """
        if not waypoints or goal is None:
            return False
        last = waypoints[-1]
        return NavigationSupervisor._within_tolerance(last, goal, tolerance)

    # ------------------------------------------------------------------ private

    @staticmethod
    def _has_active_route(agent: "RobotAgent | None") -> bool:
        return (
            agent is not None
            and agent.active_path_goal_xy is not None
            and agent.active_target() is not None
        )

    @staticmethod
    def _within_tolerance(a: Point2D, b: Point2D, tolerance: float) -> bool:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])) <= float(tolerance)
