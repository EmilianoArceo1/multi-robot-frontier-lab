"""
Regression tests for a premature repeated_safety_replan failure.

Observed bug (manual Office.sim run):

    NAV kind=REPLAN_FOR_SAFETY reason="predicted collision"
    [ROUTE ok]                                    <- safety replan succeeds
    Holding: repeated safety replan for the same target (predicted
        collision); marking target as failed and re-selecting.
    ... HOLD, recovery cooldown, fresh frontier, repeat ...

Root cause:
    RobotAgent.safety_replan_allowed() throttles identical (reason, target)
    signatures within a cooldown window, treating any recurrence as "the
    same blocked situation as before -- give up." But a genuinely new route
    accepted for that exact target (agent.assign_path() just ran) looks
    IDENTICAL to a still-stuck route by this signature alone: same reason
    ("predicted collision"), same target, recurring well within cooldown.
    engine.apply_navigation_decision()'s REPLAN_FOR_SAFETY branch always
    brakes for the tick a replan is requested/accepted, so the newly
    accepted route never gets a single tick to actually execute before the
    next observation's predicted_collision flag (recomputed fresh every
    tick) can trip the SAME throttle and immediately mark the target
    failed -- even though the route itself may well have been fine.

Fix:
    RobotAgent.route_generation is bumped by assign_path() every time a
    route is actually accepted. safety_replan_allowed() now accepts an
    optional route_generation and, when it has advanced since the current
    (reason, target) signature was last recorded, allows exactly ONE more
    recurrence through (a one-time grace pass per signature, tracked in
    _safety_replan_grace_used_for) instead of immediately denying it. Once
    that grace pass has been used for a given signature, further
    recurrences are denied exactly as before, regardless of
    route_generation -- so a target where every retried route keeps
    predicting collision again still trips repeated_safety_replan; a
    stream of technically-new-but-still-unsafe routes cannot bypass the
    guard forever.

These tests exercise RobotAgent directly -- no Qt, no canvas, no engine/GUI
instantiation, matching test_single_robot_safety_replan.py's approach.
"""
from __future__ import annotations

from robotics_sim.core.robot_agent import RobotAgent

TARGET_A = (7.75, -4.75)
TARGET_B = (2.0, 3.0)
REASON = "predicted collision"


def _make_agent(position=(0.0, 0.0)) -> RobotAgent:
    return RobotAgent(
        robot_id=0,
        position=position,
        planner_mode="FoV-aware directional frontier",
    )


# ---------------------------------------------------------------------------
# 1. A successful safety replan for the same target must not be immediately
#    treated as a repeat and must not mark the target failed right away.
# ---------------------------------------------------------------------------


def test_successful_safety_replan_does_not_immediately_mark_target_failed():
    agent = _make_agent()
    agent.set_exploration_target(TARGET_A, reason="frontier reached; requesting next frontier")
    cooldown = 0.5

    # Tick 1: first predicted-collision safety replan for this target.
    assert agent.safety_replan_allowed(
        reason=REASON,
        target=TARGET_A,
        current_time=0.0,
        cooldown=cooldown,
        route_generation=agent.route_generation,
    ), "the first safety replan request for a target must be allowed"

    # The safety replan succeeds: a fresh route is accepted for the SAME
    # target (mirrors engine.apply_route_result() -> agent.assign_path()).
    agent.assign_path(target=TARGET_A, waypoints=[TARGET_A], planner_reason="safety replan: predicted collision")
    assert agent.route_generation == 1, "assign_path() must bump route_generation on every accepted route"

    # Tick 2, immediately after (well within cooldown): predicted_collision
    # fires again for the exact same (reason, target) -- but a NEW route
    # was just accepted for it, so this must not be treated as a repeat.
    allowed_again = agent.safety_replan_allowed(
        reason=REASON,
        target=TARGET_A,
        current_time=0.02,
        cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    assert allowed_again, (
        "a route accepted moments ago must get at least one grace tick "
        "before an identical safety-replan signature is treated as repeated"
    )

    # Since the throttle was not denied, engine.py's REPLAN_FOR_SAFETY
    # branch never reaches the "mark target failed" elif -- the target and
    # route remain exactly as assign_path() left them.
    assert agent.exploration_target_xy == TARGET_A
    assert agent.active_path_goal_xy == TARGET_A
    assert agent.waypoints.has_path()


# ---------------------------------------------------------------------------
# 2. If the SAME route (no new acceptance in between) keeps predicting
#    collision after the grace tick, repeated_safety_replan must still fire.
# ---------------------------------------------------------------------------


def test_repeated_safety_replan_still_fails_after_unchanged_route_has_been_tried():
    agent = _make_agent()
    agent.set_exploration_target(TARGET_A, reason="frontier reached; requesting next frontier")
    cooldown = 0.5

    # Tick 1: first request, allowed.
    assert agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.0, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    agent.assign_path(target=TARGET_A, waypoints=[TARGET_A], planner_reason="safety replan")

    # Tick 2: the one-time grace pass for this signature.
    assert agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.05, cooldown=cooldown,
        route_generation=agent.route_generation,
    ), "the grace pass must be available on the first recurrence"

    # No new route was accepted since the grace pass (route_generation is
    # unchanged) -- the SAME just-accepted route is still predicting
    # collision. This must now be denied: the grace budget for this
    # signature has already been used.
    allowed_third_time = agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.1, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    assert not allowed_third_time, (
        "a route that keeps predicting collision after already using its "
        "grace pass must still trip repeated_safety_replan -- safety must "
        "not be bypassed indefinitely"
    )

    # Mirrors engine.py's REPLAN_FOR_SAFETY elif branch: throttle denied ->
    # hard-stop (robot.force_stop(), engine-side) + mark the target failed.
    agent.invalidate_failed_exploration_route(
        reason=f"repeated safety replan: {REASON}",
        current_time=0.1,
    )
    assert agent.exploration_target_xy is None, "the repeatedly-unsafe target must be cleared"
    assert TARGET_A in agent.recently_failed_exploration_targets(current_time=0.1, cooldown=5.0)


# ---------------------------------------------------------------------------
# 3. A new path_goal must not be blocked by stale safety-replan state left
#    over from a previous, different target.
# ---------------------------------------------------------------------------


def test_safety_replan_guard_resets_when_path_goal_changes():
    agent = _make_agent()
    cooldown = 0.5

    # Build up repeated-safety-replan state for target A: request, accept a
    # route, use the grace pass, then get denied (as in test 2).
    agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.0, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    agent.assign_path(target=TARGET_A, waypoints=[TARGET_A], planner_reason="safety replan")
    agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.05, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    denied = agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.1, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    assert not denied

    # A completely different target (e.g. after recovery/re-selection) must
    # not inherit target A's throttle/grace state.
    allowed_for_new_target = agent.safety_replan_allowed(
        reason=REASON, target=TARGET_B, current_time=0.11, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    assert allowed_for_new_target, (
        "a different target is a different signature and must not be "
        "blocked by a previous target's exhausted grace/throttle state"
    )

    # And it gets its own full grace pass too, exactly like a fresh target.
    agent.assign_path(target=TARGET_B, waypoints=[TARGET_B], planner_reason="safety replan")
    assert agent.safety_replan_allowed(
        reason=REASON, target=TARGET_B, current_time=0.12, cooldown=cooldown,
        route_generation=agent.route_generation,
    ), "the new target must get its own one-time grace pass"


# ---------------------------------------------------------------------------
# 4. After a target is marked failed (repeated_safety_replan -> HOLD ->
#    recovery), stale safety-replan memory must not poison a later,
#    unrelated target.
# ---------------------------------------------------------------------------


def test_safety_replan_guard_resets_after_route_failure_hold():
    agent = _make_agent()
    cooldown = 0.5

    # Drive target A to a repeated_safety_replan failure (as in test 2).
    agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.0, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    agent.assign_path(target=TARGET_A, waypoints=[TARGET_A], planner_reason="safety replan")
    agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.05, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    agent.safety_replan_allowed(
        reason=REASON, target=TARGET_A, current_time=0.1, cooldown=cooldown,
        route_generation=agent.route_generation,
    )
    agent.invalidate_failed_exploration_route(
        reason=f"repeated safety replan: {REASON}",
        current_time=0.1,
    )
    assert agent.exploration_target_xy is None

    # Recovery selects a fresh, unrelated target (B) and a route is
    # assigned to it, well after the HOLD/recovery cooldown.
    later_time = 10.0
    agent.set_exploration_target(TARGET_B, reason="recovery selected fresh target")
    agent.assign_path(target=TARGET_B, waypoints=[TARGET_B], planner_reason="fresh route")

    # A safety replan for B must behave exactly like a brand new target --
    # allowed on first request, with its own fresh grace pass -- unaffected
    # by A's exhausted state.
    assert agent.safety_replan_allowed(
        reason=REASON, target=TARGET_B, current_time=later_time, cooldown=cooldown,
        route_generation=agent.route_generation,
    ), "target B's first safety replan must not be poisoned by target A's failure history"
    agent.assign_path(target=TARGET_B, waypoints=[TARGET_B], planner_reason="safety replan")
    assert agent.safety_replan_allowed(
        reason=REASON, target=TARGET_B, current_time=later_time + 0.02, cooldown=cooldown,
        route_generation=agent.route_generation,
    ), "target B must get its own grace pass, not a pre-exhausted one from target A"
