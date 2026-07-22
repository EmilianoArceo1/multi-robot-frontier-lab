"""Tests for robotics_sim/simulation/coordination_scheduler.py.

Covers both halves: event_driven_decision() (today's existing triggers,
just named) and maybe_periodic_team_replan_decision() (the new,
interval-gated, disabled-by-default capability). forced_team_replan_decision()
is covered too since it shares the same FULL_TEAM/context shape.
"""

from __future__ import annotations

from robotics_interfaces.decision_context import CoordinationScope, CoordinationTrigger
from robotics_interfaces.plugins import CandidateInputMode, PluginRuntimeProfile
from robotics_sim.simulation.coordination_scheduler import (
    CoordinationSchedulerConfig,
    event_driven_decision,
    forced_team_replan_decision,
    maybe_periodic_team_replan_decision,
)


def _profile(supports_periodic_replan: bool = True) -> PluginRuntimeProfile:
    return PluginRuntimeProfile(
        detects_frontiers=False,
        generates_tasks=False,
        allocates_tasks=True,
        plans_paths=False,
        controls_motion=False,
        candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
        supports_periodic_replan=supports_periodic_replan,
    )


def test_event_driven_decision_defaults_to_missing_target():
    context = event_driven_decision(requesting_robot_ids=(1,), time_s=3.0)

    assert context.trigger is CoordinationTrigger.MISSING_TARGET
    assert context.scope is CoordinationScope.REQUESTED_ROBOTS
    assert context.requesting_robot_ids == (1,)
    assert context.requesting_robot_id == 1


def test_event_driven_decision_names_initial_assignment():
    context = event_driven_decision(requesting_robot_ids=(0, 1), time_s=0.0, initial_assignment=True)
    assert context.trigger is CoordinationTrigger.INITIAL_ASSIGNMENT


def test_event_driven_decision_names_target_reached():
    context = event_driven_decision(requesting_robot_ids=(2,), time_s=5.0, reached=True)
    assert context.trigger is CoordinationTrigger.TARGET_REACHED


def test_event_driven_decision_names_target_invalidated():
    context = event_driven_decision(requesting_robot_ids=(2,), time_s=5.0, invalidated=True)
    assert context.trigger is CoordinationTrigger.TARGET_INVALIDATED


def test_forced_team_replan_decision_uses_full_team_scope():
    context = forced_team_replan_decision(active_robot_ids=(0, 1, 2), time_s=9.0, reason_detail="operator request")

    assert context.trigger is CoordinationTrigger.FORCED_TEAM_REPLAN
    assert context.scope is CoordinationScope.FULL_TEAM
    assert context.requesting_robot_ids == (0, 1, 2)
    assert context.reason_detail == "operator request"


def test_periodic_replan_disabled_by_default_interval_zero():
    config = CoordinationSchedulerConfig()  # replan_interval_s=0.0
    decision = maybe_periodic_team_replan_decision(
        config=config,
        active_robot_ids=(0, 1),
        time_s=100.0,
        last_periodic_replan_time_s=None,
        is_goal_seeking_mode=False,
    )
    assert decision is None


def test_periodic_replan_does_not_fire_before_the_interval_elapses():
    config = CoordinationSchedulerConfig(replan_interval_s=5.0)
    decision = maybe_periodic_team_replan_decision(
        config=config,
        active_robot_ids=(0, 1),
        time_s=3.0,
        last_periodic_replan_time_s=0.0,
        is_goal_seeking_mode=False,
    )
    assert decision is None


def test_periodic_replan_fires_once_the_interval_has_elapsed():
    config = CoordinationSchedulerConfig(replan_interval_s=5.0)
    decision = maybe_periodic_team_replan_decision(
        config=config,
        active_robot_ids=(0, 1, 2),
        time_s=5.0,
        last_periodic_replan_time_s=0.0,
        is_goal_seeking_mode=False,
    )

    assert decision is not None
    assert decision.trigger is CoordinationTrigger.PERIODIC_TEAM_REPLAN
    assert decision.scope is CoordinationScope.FULL_TEAM
    assert decision.requesting_robot_ids == (0, 1, 2)


def test_periodic_replan_fires_on_the_very_first_call_with_no_prior_time():
    config = CoordinationSchedulerConfig(replan_interval_s=5.0)
    decision = maybe_periodic_team_replan_decision(
        config=config,
        active_robot_ids=(0,),
        time_s=0.1,
        last_periodic_replan_time_s=None,
        is_goal_seeking_mode=False,
    )
    assert decision is not None


def test_periodic_replan_never_fires_in_goal_seeking_mode():
    config = CoordinationSchedulerConfig(replan_interval_s=5.0)
    decision = maybe_periodic_team_replan_decision(
        config=config,
        active_robot_ids=(0,),
        time_s=100.0,
        last_periodic_replan_time_s=0.0,
        is_goal_seeking_mode=True,
    )
    assert decision is None


def test_periodic_replan_respects_supports_periodic_replan_false():
    config = CoordinationSchedulerConfig(replan_interval_s=5.0)
    decision = maybe_periodic_team_replan_decision(
        config=config,
        active_robot_ids=(0,),
        time_s=100.0,
        last_periodic_replan_time_s=0.0,
        is_goal_seeking_mode=False,
        profile=_profile(supports_periodic_replan=False),
    )
    assert decision is None


def test_periodic_replan_does_not_fire_with_no_active_robots():
    config = CoordinationSchedulerConfig(replan_interval_s=5.0)
    decision = maybe_periodic_team_replan_decision(
        config=config,
        active_robot_ids=(),
        time_s=100.0,
        last_periodic_replan_time_s=0.0,
        is_goal_seeking_mode=False,
    )
    assert decision is None
