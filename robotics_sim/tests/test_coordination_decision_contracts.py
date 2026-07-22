"""Tests for the new structured coordination decision contracts.

See robotics_interfaces/decision_context.py: CoordinationTrigger,
CoordinationScope, CoordinationDecisionContext, RobotRouteSnapshot,
VisitCountSnapshot. These are additive contracts introduced ahead of the
scheduler/applier/audit runtime components (later phases); this module only
tests the contracts themselves and their backward-compatible wiring into
CoordinationRequest.
"""

from __future__ import annotations

from robotics_interfaces import CoordinationRequest
from robotics_interfaces.decision_context import (
    CoordinationDecisionContext,
    CoordinationScope,
    CoordinationTrigger,
    VisitCountSnapshot,
    build_robot_route_snapshot,
)


def test_coordination_request_defaults_decision_context_to_none():
    """Every existing construction site omits decision_context; it must
    default to None so no caller needs to change."""
    request = CoordinationRequest(robot_states=())
    assert request.decision_context is None


def test_decision_context_carries_trigger_and_scope():
    context = CoordinationDecisionContext(
        trigger=CoordinationTrigger.TARGET_REACHED,
        scope=CoordinationScope.REQUESTED_ROBOTS,
        requesting_robot_ids=(2,),
        requesting_robot_id=2,
        time_s=12.5,
    )

    assert context.trigger is CoordinationTrigger.TARGET_REACHED
    assert context.scope is CoordinationScope.REQUESTED_ROBOTS
    assert context.requesting_robot_ids == (2,)
    assert context.requesting_robot_id == 2
    assert context.decision_id
    assert context.reason_detail is None


def test_decision_context_generates_a_distinct_decision_id_by_default():
    first = CoordinationDecisionContext(
        trigger=CoordinationTrigger.MISSING_TARGET,
        scope=CoordinationScope.REQUESTED_ROBOTS,
    )
    second = CoordinationDecisionContext(
        trigger=CoordinationTrigger.MISSING_TARGET,
        scope=CoordinationScope.REQUESTED_ROBOTS,
    )

    assert first.decision_id != second.decision_id


def test_periodic_and_forced_team_replan_triggers_use_full_team_scope_by_convention():
    """This is only a naming/contract check (the enums exist and are
    distinct) -- the scheduler in a later phase is what actually enforces
    "PERIODIC_TEAM_REPLAN implies FULL_TEAM"."""
    periodic = CoordinationDecisionContext(
        trigger=CoordinationTrigger.PERIODIC_TEAM_REPLAN,
        scope=CoordinationScope.FULL_TEAM,
    )
    forced = CoordinationDecisionContext(
        trigger=CoordinationTrigger.FORCED_TEAM_REPLAN,
        scope=CoordinationScope.FULL_TEAM,
    )
    assert periodic.trigger != forced.trigger
    assert periodic.scope is forced.scope is CoordinationScope.FULL_TEAM


def test_robot_route_snapshot_computes_exact_remaining_length_from_waypoints():
    snapshot = build_robot_route_snapshot(
        3,
        remaining_waypoints=((0.0, 0.0), (3.0, 0.0), (3.0, 4.0)),
        target=(3.0, 4.0),
        status="ACTIVE",
        source="plugin path (PATH_PLANNING owned)",
        updated_at_s=7.0,
    )

    assert snapshot.robot_id == 3
    assert snapshot.remaining_length == 7.0  # 3 + 4, exact polyline length
    assert snapshot.status == "ACTIVE"
    assert snapshot.updated_at_s == 7.0


def test_robot_route_snapshot_leaves_unavailable_fields_as_none_not_fabricated():
    snapshot = build_robot_route_snapshot(1)

    assert snapshot.remaining_waypoints is None
    assert snapshot.target is None
    assert snapshot.status is None
    assert snapshot.source is None
    assert snapshot.remaining_length is None
    assert snapshot.updated_at_s is None


def test_visit_count_snapshot_defaults_to_unavailable_and_sparse():
    snapshot = VisitCountSnapshot()

    assert snapshot.available is False
    assert snapshot.counts == ()
    assert snapshot.count_at((1.0, 1.0)) == 0


def test_visit_count_snapshot_looks_up_sparse_entries():
    snapshot = VisitCountSnapshot(
        counts=(((0.0, 0.0), 3), ((1.5, 0.0), 1)),
        resolution=0.5,
        available=True,
    )

    assert snapshot.count_at((0.0, 0.0)) == 3
    assert snapshot.count_at((1.5, 0.0)) == 1
    assert snapshot.count_at((9.0, 9.0)) == 0
