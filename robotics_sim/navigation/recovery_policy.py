"""
RecoveryPolicy — a small, deterministic fallback layer tried before
declaring single-robot exploration exhausted.

Context
-------
Manual Office.sim runs showed exploration declaring itself exhausted too
early (around 20% explored), even though the console state was already
coherent (no route-less REPLAN_FOR_SAFETY, no already-reached-target loops,
no route endpoint mismatches -- all fixed in prior rounds via
NavigationSupervisor). The remaining problem is that "normal frontier
selection found nothing this cycle" was treated as equivalent to "nothing
useful remains anywhere" -- there was no attempt to fall back to a nearby,
previously-known-reachable point before giving up.

This module does not replace frontier scoring (still PlannerServices /
exploration_planners), A* (still planner_registry / grid_planners), or the
robot controller. It only proposes a fallback target -- from data
RobotAgent already tracks -- when normal frontier selection has failed for
this cycle. ExplorationBehavior decides when to ask it and what to do with
the answer; RecoveryPolicy itself is a pure, stateless, deterministic
lookup with no randomness and no search.
"""
from __future__ import annotations

import math
from typing import Iterable

Point2D = tuple[float, float]


class RecoveryPolicy:
    """Stateless, deterministic recovery-target proposal.

    Does not own frontier scoring, A*, GUI, robot physics, or multi-robot
    coordination -- it only proposes a fallback target when normal
    frontier selection has already failed.
    """

    @staticmethod
    def propose_recovery_target(
        robot_xy: Point2D,
        goal_tolerance: float,
        recent_safe_positions: Iterable[Point2D],
        recently_failed_targets: Iterable[Point2D] = (),
        recent_recovery_targets: Iterable[Point2D] = (),
        *,
        active_path_goal: Point2D | None = None,
        pending_target: Point2D | None = None,
        last_completed_path_goal: Point2D | None = None,
    ) -> Point2D | None:
        """Return a deterministic fallback target, or None if none qualifies.

        Candidates are RobotAgent.recent_safe_positions -- a small bounded
        history of positions recorded whenever a route was successfully
        assigned or a prefetched route was promoted to active, i.e. points
        from which normal exploration was known to be working. They are
        considered most-recent-first: backtracking to where exploration
        was last known to work is the most explainable and most likely to
        still be reachable choice. The first candidate that clears every
        exclusion below is returned -- no scoring, no randomness, no search.

        A candidate is rejected when it is within goal_tolerance of:
            - the robot's current position (nothing to gain by "recovering"
              to where the robot already is),
            - a recently-failed exploration target (planning there was
              already tried and failed),
            - active_path_goal_xy or pending_target_xy (already the
              active/in-flight route goal),
            - a target already used for recovery during the current
              recovery episode (recent_recovery_targets) -- without this,
              two safe positions can ping-pong forever: reaching A makes B
              the "most recent not-at-robot" candidate and vice versa,
              since each successful move adds a fresh entry to
              recent_safe_positions. Excluding targets already tried for
              recovery this episode guarantees the candidate pool shrinks
              every attempt instead of cycling. The caller (RobotAgent)
              decides when an episode ends -- this function has no opinion
              on that, it only excludes whatever list it is given.
            - last_completed_path_goal -- the point the robot's active
              route most recently ended at. Distance-based exclusion
              against robot_xy alone only protects against a candidate
              close to the robot's CURRENT position; it says nothing about
              "this exact point was just a route destination" once the
              robot has since drifted, or once recovery is evaluated a few
              ticks after that route finished. recent_safe_positions can
              still contain that same point (recorded by a later, unrelated
              successful route assignment made from that same resting
              position -- see RobotAgent.assign_path()), so it needs its
              own explicit exclusion independent of current distance.
        """
        for candidate in reversed(list(recent_safe_positions)):
            if RecoveryPolicy._is_valid_candidate(
                candidate,
                robot_xy,
                goal_tolerance,
                recently_failed_targets,
                recent_recovery_targets,
                active_path_goal,
                pending_target,
                last_completed_path_goal,
            ):
                return candidate
        return None

    @staticmethod
    def _is_valid_candidate(
        candidate: Point2D,
        robot_xy: Point2D,
        goal_tolerance: float,
        recently_failed_targets: Iterable[Point2D],
        recent_recovery_targets: Iterable[Point2D],
        active_path_goal: Point2D | None,
        pending_target: Point2D | None,
        last_completed_path_goal: Point2D | None = None,
    ) -> bool:
        if RecoveryPolicy._within(candidate, robot_xy, goal_tolerance):
            return False
        for failed in recently_failed_targets:
            if RecoveryPolicy._within(candidate, failed, goal_tolerance):
                return False
        for attempted in recent_recovery_targets:
            if RecoveryPolicy._within(candidate, attempted, goal_tolerance):
                return False
        if active_path_goal is not None and RecoveryPolicy._within(candidate, active_path_goal, goal_tolerance):
            return False
        if pending_target is not None and RecoveryPolicy._within(candidate, pending_target, goal_tolerance):
            return False
        if last_completed_path_goal is not None and RecoveryPolicy._within(
            candidate, last_completed_path_goal, goal_tolerance
        ):
            return False
        return True

    @staticmethod
    def _within(a: Point2D, b: Point2D, tolerance: float) -> bool:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])) <= float(tolerance)
