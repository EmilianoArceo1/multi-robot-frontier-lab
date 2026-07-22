"""Decides WHEN and with what trigger/scope a coordination decision should
run. This module performs no frontier detection, no candidate generation,
and no allocation -- it only classifies/gates decisions; the actual work
still happens in the selected plugin via MultiRobotCoordinator.

Two kinds of decision this module produces:

  * event_driven_decision() -- names today's existing triggers (initial
    assignment, missing target, target reached, target invalidated) so a
    CoordinationRequest can carry a real CoordinationDecisionContext instead
    of the caller inferring "why" from nothing. This does not change when
    coordination runs; it only labels the existing event.

  * maybe_periodic_team_replan_decision() -- the new capability: an
    interval-gated, whole-team replan. Disabled by default
    (CoordinationSchedulerConfig.replan_interval_s == 0.0), which reproduces
    today's purely event-driven behavior exactly. When enabled, it fires at
    most once per interval, always with CoordinationScope.FULL_TEAM and
    CoordinationTrigger.PERIODIC_TEAM_REPLAN, and never in Goal seeking mode
    or for a plugin whose profile.supports_periodic_replan is False.

  * forced_team_replan_decision() -- a manual/administrative "replan the
    whole team right now" request, distinct from the timer-driven periodic
    case (same FULL_TEAM scope, different trigger for provenance).
"""

from __future__ import annotations

from dataclasses import dataclass

from robotics_interfaces.decision_context import (
    CoordinationDecisionContext,
    CoordinationScope,
    CoordinationTrigger,
)
from robotics_interfaces.plugins import PluginRuntimeProfile


@dataclass(frozen=True)
class CoordinationSchedulerConfig:
    """Backward-compatible scheduling configuration.

    replan_interval_s=0.0 (the default) disables periodic team replanning
    entirely, reproducing today's purely event-driven behavior. This mirrors
    robotics_sim.simulation.config.SimulationConfig.coordination_replan_
    interval_s/coordination_strict_contracts -- the engine reads its config
    into this dataclass rather than passing config.py directly, so this
    module has no dependency on the simulator's config module.
    """

    replan_interval_s: float = 0.0
    strict_contracts: bool = False


def event_driven_decision(
    *,
    requesting_robot_ids: tuple[int, ...],
    time_s: float,
    initial_assignment: bool = False,
    reached: bool = False,
    invalidated: bool = False,
    reason_detail: str | None = None,
) -> CoordinationDecisionContext:
    """Classify one of today's existing event-driven coordination calls.

    Exactly one of initial_assignment/reached/invalidated should be True for
    the caller's actual reason; when none are set this is a plain
    MISSING_TARGET decision (a target slot is None with no more specific
    event attached). This never changes whether/when coordination runs --
    engine.py's existing gating (is_goal_seeking_mode(), "any target is
    None", force_new_target, cooldowns) stays exactly as it is; this only
    names the result for provenance.
    """

    if initial_assignment:
        trigger = CoordinationTrigger.INITIAL_ASSIGNMENT
    elif reached:
        trigger = CoordinationTrigger.TARGET_REACHED
    elif invalidated:
        trigger = CoordinationTrigger.TARGET_INVALIDATED
    else:
        trigger = CoordinationTrigger.MISSING_TARGET

    return CoordinationDecisionContext(
        trigger=trigger,
        scope=CoordinationScope.REQUESTED_ROBOTS,
        requesting_robot_ids=tuple(requesting_robot_ids),
        requesting_robot_id=requesting_robot_ids[0] if requesting_robot_ids else None,
        time_s=float(time_s),
        reason_detail=reason_detail,
    )


def forced_team_replan_decision(
    *,
    active_robot_ids: tuple[int, ...],
    time_s: float,
    reason_detail: str | None = None,
) -> CoordinationDecisionContext:
    """A manual/administrative "replan the whole team now" request.

    Distinct from maybe_periodic_team_replan_decision(): this is not
    timer-gated -- a caller (e.g. an experiment harness, a GUI action) has
    already decided a full-team replan should happen, and only wants the
    correct FULL_TEAM/FORCED_TEAM_REPLAN provenance attached to it.
    """

    return CoordinationDecisionContext(
        trigger=CoordinationTrigger.FORCED_TEAM_REPLAN,
        scope=CoordinationScope.FULL_TEAM,
        requesting_robot_ids=tuple(active_robot_ids),
        time_s=float(time_s),
        reason_detail=reason_detail,
    )


def maybe_periodic_team_replan_decision(
    *,
    config: CoordinationSchedulerConfig,
    active_robot_ids: tuple[int, ...],
    time_s: float,
    last_periodic_replan_time_s: float | None,
    is_goal_seeking_mode: bool,
    profile: PluginRuntimeProfile | None = None,
) -> CoordinationDecisionContext | None:
    """Return a PERIODIC_TEAM_REPLAN decision if one is due right now, else
    None.

    Never fires when:
      - config.replan_interval_s <= 0 (disabled -- the default);
      - is_goal_seeking_mode is True (Goal seeking never runs coordination);
      - profile is given and profile.supports_periodic_replan is False;
      - there are no active robots;
      - less than replan_interval_s simulated seconds have elapsed since
        last_periodic_replan_time_s (so it fires at most once per interval).

    This is a pure timing decision -- it does no detection, allocation, or
    IO, so calling it every simulation tick is cheap even when it returns
    None.
    """

    if config.replan_interval_s <= 0.0:
        return None
    if is_goal_seeking_mode:
        return None
    if profile is not None and not profile.supports_periodic_replan:
        return None
    if not active_robot_ids:
        return None

    if last_periodic_replan_time_s is not None:
        elapsed = float(time_s) - float(last_periodic_replan_time_s)
        if elapsed < config.replan_interval_s:
            return None

    return CoordinationDecisionContext(
        trigger=CoordinationTrigger.PERIODIC_TEAM_REPLAN,
        scope=CoordinationScope.FULL_TEAM,
        requesting_robot_ids=tuple(active_robot_ids),
        time_s=float(time_s),
    )
