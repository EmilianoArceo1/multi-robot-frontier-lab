"""Structured provenance contracts for coordination decisions.

These types answer "why did the host run this coordination decision, and
what does it actually know about the current route/visit state" without
relying on free-form strings. They are additive: nothing here changes
CoordinationRequest/CoordinationResult defaults, and no existing caller is
required to populate them yet.

CoordinationDecisionContext is the trigger/scope contract (see
CoordinationTrigger/CoordinationScope). RobotRouteSnapshot and
VisitCountSnapshot are read-model snapshots a host *may* attach so plugins
and reasoning panels can describe route/coverage state precisely instead of
inferring it -- both are entirely optional and every field that the host
cannot honestly derive from state it already has must stay None/unavailable
rather than being fabricated.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from robotics_interfaces.observations import Point2D


class CoordinationTrigger(str, Enum):
    """Why the host ran a coordination decision right now.

    This is the structured replacement for inferring "why" from log strings
    or plugin names. See robotics_sim/simulation/coordination_scheduler.py
    for the component that decides which trigger applies.
    """

    INITIAL_ASSIGNMENT = "initial_assignment"
    MISSING_TARGET = "missing_target"
    TARGET_REACHED = "target_reached"
    TARGET_INVALIDATED = "target_invalidated"
    PERIODIC_TEAM_REPLAN = "periodic_team_replan"
    FORCED_TEAM_REPLAN = "forced_team_replan"


class CoordinationScope(str, Enum):
    """Which robots a decision is allowed to (re)assign.

    REQUESTED_ROBOTS is today's behavior: only the robots explicitly listed
    in robots_to_assign. FULL_TEAM is reserved for triggers that must
    consider the whole team at once (PERIODIC_TEAM_REPLAN/
    FORCED_TEAM_REPLAN) -- see coordination_scheduler.py.
    """

    REQUESTED_ROBOTS = "requested_robots"
    FULL_TEAM = "full_team"


def _new_decision_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class CoordinationDecisionContext:
    """Structured "why/what/when" for one coordination decision.

    requesting_robot_id is kept only as a convenience for callers that have
    exactly one requesting robot (the common case today); it is not the
    source of truth -- requesting_robot_ids is. When both are given,
    requesting_robot_id should be a member of requesting_robot_ids, but this
    is not enforced here to keep construction cheap on the hot path.
    """

    trigger: CoordinationTrigger
    scope: CoordinationScope
    requesting_robot_ids: tuple[int, ...] = ()
    requesting_robot_id: int | None = None
    decision_id: str = field(default_factory=_new_decision_id)
    time_s: float = 0.0
    reason_detail: str | None = None


@dataclass(frozen=True)
class RobotRouteSnapshot:
    """What the host actually knows about one robot's route right now.

    Every field defaults to None/"unavailable" rather than a fabricated
    value. remaining_length is the one derived field: when
    remaining_waypoints is known, it is the exact polyline length of those
    waypoints, not an estimate.
    """

    robot_id: int
    remaining_waypoints: tuple[Point2D, ...] | None = None
    target: Point2D | None = None
    status: str | None = None
    source: str | None = None
    remaining_length: float | None = None
    updated_at_s: float | None = None

    # route_points_by_robot (CoordinationRequest) remains the legacy
    # representation during migration -- see robotics_interfaces.coordination.
    # RobotRouteSnapshot is the richer, per-robot replacement new code should
    # prefer once a host actually builds one per robot.


def build_robot_route_snapshot(
    robot_id: int,
    *,
    remaining_waypoints: tuple[Point2D, ...] | None = None,
    target: Point2D | None = None,
    status: str | None = None,
    source: str | None = None,
    updated_at_s: float | None = None,
) -> RobotRouteSnapshot:
    """Build a RobotRouteSnapshot from state a host already has in hand.

    remaining_length is computed from remaining_waypoints (exact polyline
    length) when waypoints are given; otherwise it stays None. This never
    invents waypoints/targets/status -- callers only pass what they already
    know.
    """

    remaining_length = None
    if remaining_waypoints:
        remaining_length = sum(
            _distance(remaining_waypoints[index], remaining_waypoints[index + 1])
            for index in range(len(remaining_waypoints) - 1)
        )

    return RobotRouteSnapshot(
        robot_id=robot_id,
        remaining_waypoints=remaining_waypoints,
        target=target,
        status=status,
        source=source,
        remaining_length=remaining_length,
        updated_at_s=updated_at_s,
    )


def _distance(a: Point2D, b: Point2D) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


@dataclass(frozen=True)
class VisitCountSnapshot:
    """Sparse, immutable per-cell (or per-point) visit counts.

    This exists so a future coverage/edge-weighting algorithm (e.g. the
    Baghyari paper) has a stable contract to depend on. No runtime component
    builds a real one yet -- every current integration point should return
    VisitCountSnapshot() (available=False) rather than reconstructing counts
    expensively per frame or fabricating numbers. counts is a tuple of
    (point, count) pairs instead of a dict so the dataclass stays genuinely
    immutable; entries with count 0 should simply be omitted (sparse).
    """

    counts: tuple[tuple[Point2D, int], ...] = ()
    resolution: float | None = None
    available: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def count_at(self, point: Point2D) -> int:
        for candidate, count in self.counts:
            if candidate == point:
                return count
        return 0
