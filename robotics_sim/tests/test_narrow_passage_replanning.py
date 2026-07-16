"""
Regression tests for narrow-passage instability: repeated route_affected
replans, FPS collapse, and a stale/unsafe route slipping through prefetch.

Manual Office.sim telemetry (robot progressing past the previous stuck
points, ~24.5% explored, near narrow passages):

    New obstacle affects current route. Replanning...
    [ROUTE ok]
    New obstacle affects current route. Replanning...
    [ROUTE ok]
    ... (repeats every sensor-update tick)
    [NAV] REPLAN_FOR_SAFETY reason="predicted collision"
    Planner failed ... rejected: first segment blocked on arrival.
    [ROUTE fail] reason=first_segment_blocked

Root cause (Part B, round 1): simulation_step()'s route_affected branch
called replan_after_new_information() completely unthrottled -- ANY
newly-mapped obstacle sample intersecting the current route triggered a
full replan. Near a narrow passage, routine sensor updates add boundary
samples on nearly every tick, so this became a background full-replan
storm for the SAME path_goal, competing with rendering for CPU (the
reported FPS collapse) and repeatedly discarding/reassigning routes
instead of letting one execute.

Root cause (round 2 -- the throttle added in round 1 did not actually
throttle anything): the first version of route_affected_replan_allowed()
let an active_segment_unsafe=True flag bypass the cooldown entirely,
computed via route_first_segment_blocked() against the current active
target using the SAME newly-mapped obstacle points that had just
triggered route_affected in the first place. Near a narrow passage that
coarse check reads "blocked" on nearly every call -- exactly the
situation the throttle exists for -- so the bypass fired almost
unconditionally and defeated the throttle completely (confirmed via
manual Office.sim telemetry: repeated route_affected for the same
path_goal with zero observed throttling).

Fix:
    - RobotAgent.route_affected_replan_allowed() throttles repeated
      route_affected repairs for the SAME path_goal within a cooldown
      (engine.route_affected_replan_cooldown_seconds()), deliberately
      separate from safety_replan_allowed() -- REPLAN_FOR_SAFETY never
      calls this method, so predicted-collision handling is unaffected.
      There is NO active_segment_unsafe bypass anymore: a genuinely
      urgent, imminent collision is REPLAN_FOR_SAFETY's own job, driven
      independently and throttled separately.
    - New route_repair_in_progress_for_goal guard: once a repair for a
      goal is launched, a repeated route_affected for the SAME goal is
      denied immediately (not just cooldown-throttled) until the route
      result actually arrives (assign_path()/invalidate_route() clear
      it) -- so a repeated event while an async worker is still computing
      cannot enqueue a second planner job on top of the first.
    - A throttled (denied) route_affected -- whether by the in-progress
      guard or the cooldown -- arms
      RobotAgent.narrow_passage_slowdown_until_time (see
      trigger_narrow_passage_slowdown()/is_narrow_passage_slowdown_active()).
      engine.sync_narrow_passage_speed_cap() temporarily caps
      robot.max_speed (an existing runtime hook) while that window is
      active, restoring config.max_speed once it expires. config.max_speed
      itself and robot dynamics are never touched.
    - engine.format_narrow_passage_diagnostic() logs a throttled
      [NARROW_DIAG] line (DEBUG-level, naturally rate-limited to at most
      once per cooldown per path_goal by the throttle itself) reporting
      route_affected_recent/first_segment_blocked/predicted_collision
      counts and an approximate min_clearance.
    - Part D gap closed: engine.on_prefetch_route_ready() previously only
      validated a prefetched route's ENDPOINT
      (route_reaches_goal()) before storing it in agent.pending_path, not
      its first segment. It now also rejects a prefetch whose first
      segment (from the robot's CURRENT position at callback time) is
      blocked, using the exact same route_first_segment_blocked() rule
      apply_route_result() already uses for the main route-acceptance
      path -- so a route accepted moments before new obstacle samples
      appeared cannot be silently promoted into a now-unsafe segment via
      ACCEPT_PENDING_PATH.

CBF note (Part E): no CBF code is added in this round -- see the
docstring TODO on apply_navigation_decision()'s REPLAN_FOR_SAFETY case in
engine.py.

These tests exercise RobotAgent directly (pure, no engine/Qt) for the
throttle/slowdown contract, and the same lightweight duck-typed engine
fake test_recovery_rejects_reached_targets.py /
test_pending_path_invalidated_by_replan.py use for the engine-boundary
prefetch-rejection behavior.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.simulation.engine import SimulationControllerMixin, format_narrow_passage_diagnostic


PATH_GOAL = (5.0, 5.0)
OTHER_PATH_GOAL = (2.0, 2.0)


def _make_agent(position=(0.0, 0.0)) -> RobotAgent:
    return RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")


class _FakeRobot(SimpleNamespace):
    pass


# ---------------------------------------------------------------------------
# 1. Repeated route_affected for the same path_goal within cooldown is
#    throttled.
# ---------------------------------------------------------------------------


def test_route_affected_replan_is_throttled_for_same_path_goal():
    agent = _make_agent()
    cooldown = 1.0

    assert agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=0.0, cooldown=cooldown
    ), "the first route_affected repair for a target must be allowed"

    assert not agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=0.1, cooldown=cooldown
    ), "an identical route_affected within cooldown must be throttled"
    assert not agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=0.5, cooldown=cooldown
    )


# ---------------------------------------------------------------------------
# 2. After the cooldown elapses, route_affected can request a replan again.
# ---------------------------------------------------------------------------


def test_route_affected_replan_allowed_after_cooldown_same_goal():
    agent = _make_agent()
    cooldown = 1.0

    agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.0, cooldown=cooldown)
    # Simulate the repair's route result arriving (success) -- clears the
    # in-progress guard, but must NOT reset the cooldown clock itself (see
    # test_route_affected_replan_throttled_after_successful_repair_same_goal).
    agent.assign_path(target=PATH_GOAL, waypoints=[PATH_GOAL], planner_reason="repair")

    assert not agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=0.5, cooldown=cooldown
    ), "still within cooldown even though the previous repair already completed"

    assert agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=1.5, cooldown=cooldown
    ), "once the cooldown has elapsed, the same path_goal must be allowed to repair again"


def test_route_affected_replan_allowed_for_different_path_goal():
    """Changing path_goal resets the guard -- a different target is never
    throttled by a prior, unrelated target's state."""
    agent = _make_agent()
    cooldown = 1.0

    agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.0, cooldown=cooldown)
    assert not agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.1, cooldown=cooldown)

    assert agent.route_affected_replan_allowed(
        path_goal=OTHER_PATH_GOAL, current_time=0.11, cooldown=cooldown
    ), "a different path_goal must not be throttled by another target's in-progress/cooldown state"


# ---------------------------------------------------------------------------
# 3. REPLAN_FOR_SAFETY's own throttle is completely independent -- a
#    route_affected throttle for the same target never blocks it.
# ---------------------------------------------------------------------------


def test_safety_replan_still_allowed_while_route_affected_throttled():
    agent = _make_agent()
    cooldown = 1.0

    agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.0, cooldown=cooldown)
    assert not agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.1, cooldown=cooldown)

    assert agent.safety_replan_allowed(
        reason="predicted collision",
        target=PATH_GOAL,
        current_time=0.1,
        cooldown=cooldown,
        route_generation=agent.route_generation,
    ), "REPLAN_FOR_SAFETY must never be throttled by the route_affected guard"


def test_route_affected_active_segment_unsafe_does_not_bypass_into_replan_storm():
    """There is no active_segment_unsafe bypass anymore: repeated
    route_affected for the same path_goal stays throttled no matter what
    (a genuinely urgent segment is REPLAN_FOR_SAFETY's separate job, not
    this guard's) -- never multiple route_affected repair requests in a
    tight loop."""
    agent = _make_agent()
    cooldown = 1.0

    assert agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.0, cooldown=cooldown)
    for t in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8):
        assert not agent.route_affected_replan_allowed(
            path_goal=PATH_GOAL, current_time=t, cooldown=cooldown
        ), f"route_affected must stay throttled at t={t}, not spam repair requests"

    # The safety path remains available throughout, independent of this.
    assert agent.safety_replan_allowed(
        reason="predicted collision",
        target=PATH_GOAL,
        current_time=0.8,
        cooldown=cooldown,
        route_generation=agent.route_generation,
    )


# ---------------------------------------------------------------------------
# "route repair in progress" guard: a repeated route_affected for the SAME
# goal while a repair is still in flight must not launch a second one, and
# must remain suppressed even after the repair succeeds (until cooldown).
# ---------------------------------------------------------------------------


def test_route_affected_replan_not_relaunched_while_repair_in_progress():
    agent = _make_agent()
    cooldown = 1.0

    assert agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=0.0, cooldown=cooldown
    ), "the first route_affected repair must launch"
    assert agent.route_repair_in_progress_for_goal == PATH_GOAL

    # Before the route result returns, another route_affected for the
    # SAME goal arrives (e.g. another sensor-update tick).
    assert not agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=0.02, cooldown=cooldown
    ), "no second replan request while a repair for this goal is already in flight"
    assert not agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=0.04, cooldown=cooldown
    )


def test_route_affected_replan_throttled_after_successful_repair_same_goal():
    agent = _make_agent()
    cooldown = 1.0

    agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.0, cooldown=cooldown)
    # The repair succeeds -- mirrors apply_route_result() -> assign_path().
    agent.assign_path(target=PATH_GOAL, waypoints=[PATH_GOAL], planner_reason="repair")
    assert agent.route_repair_in_progress_for_goal is None, "a successful repair must clear the in-progress guard"

    # Immediately another route_affected for the SAME goal occurs.
    assert not agent.route_affected_replan_allowed(
        path_goal=PATH_GOAL, current_time=0.01, cooldown=cooldown
    ), "a successful repair must not immediately allow another repair for the same goal -- cooldown still applies"


# ---------------------------------------------------------------------------
# 4. Repeated (throttled) route_affected events arm the narrow-passage
#    speed-cap window; the engine applies/lifts it via robot.max_speed.
# ---------------------------------------------------------------------------


def test_narrow_passage_slowdown_flag_set_after_repeated_route_affected():
    agent = _make_agent()
    cooldown = 1.0

    assert not agent.is_narrow_passage_slowdown_active(0.0)

    agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.0, cooldown=cooldown)
    assert not agent.is_narrow_passage_slowdown_active(0.05), (
        "the first (allowed) route_affected must not arm the slowdown by itself"
    )

    agent.route_affected_replan_allowed(path_goal=PATH_GOAL, current_time=0.1, cooldown=cooldown)
    assert agent.is_narrow_passage_slowdown_active(0.1), (
        "a throttled (repeated) route_affected must arm the slowdown window"
    )
    window_end = 0.1 + RobotAgent._NARROW_PASSAGE_SLOWDOWN_WINDOW_S
    assert agent.is_narrow_passage_slowdown_active(window_end - 0.01)
    assert not agent.is_narrow_passage_slowdown_active(window_end + 0.01)


def test_engine_applies_and_lifts_narrow_passage_speed_cap():
    agent = _make_agent()
    agent.trigger_narrow_passage_slowdown(current_time=0.0)

    robot = _FakeRobot(max_speed=1.20)
    fake = SimpleNamespace(
        robot=robot,
        simulation_time=0.5,
        config=SimpleNamespace(max_speed=1.20),
    )

    SimulationControllerMixin.sync_narrow_passage_speed_cap(fake, agent)
    assert robot.max_speed == RobotAgent._NARROW_PASSAGE_SLOWDOWN_SPEED_CAP
    assert robot.max_speed < 1.20

    # Once the window expires, the configured speed is restored.
    fake.simulation_time = 0.5 + RobotAgent._NARROW_PASSAGE_SLOWDOWN_WINDOW_S + 0.1
    SimulationControllerMixin.sync_narrow_passage_speed_cap(fake, agent)
    assert robot.max_speed == 1.20


# ---------------------------------------------------------------------------
# 5. A route whose first segment is blocked is rejected before it can
#    become the active/pending path -- exercised via the prefetch path
#    (engine.on_prefetch_route_ready()), the gap this round closes.
# ---------------------------------------------------------------------------


def test_first_segment_blocked_route_is_rejected_before_acceptance():
    agent = _make_agent(position=(0.0, 0.0))
    agent.mark_pending_path_requested((5.0, 0.0))

    console_logs: list[str] = []
    fake = SimpleNamespace(
        robot=_FakeRobot(x=0.0, y=0.0),
        prefetch_workers={0: object()},
        prefetch_request_ids={0: 1},
        mapped_obstacle_points=[(1.0, 0.0)],  # sits directly on the first segment
        config=SimpleNamespace(goal_tolerance=0.25),
        simulation_time=1.0,
        collision_checker=CollisionChecker(),
    )
    fake.runtime_agent = lambda robot_index=None: agent
    fake.safety_radius = lambda: 0.3
    fake.log_console_message = lambda msg: console_logs.append(msg)

    SimulationControllerMixin.on_prefetch_route_ready(
        fake,
        request_id=1,
        robot_index=0,
        success=True,
        reason="ok",
        waypoints=[(2.0, 0.0), (5.0, 0.0)],
    )

    assert agent.pending_path is None, (
        "a route whose first segment is blocked must not become the pending/active path"
    )
    assert agent.first_segment_blocked_count == 1
    assert any("first segment blocked" in line for line in console_logs)


def test_first_segment_clear_route_is_accepted_as_pending():
    """Sanity check: the new gate must not reject a genuinely clear route."""
    agent = _make_agent(position=(0.0, 0.0))
    agent.mark_pending_path_requested((5.0, 0.0))

    fake = SimpleNamespace(
        robot=_FakeRobot(x=0.0, y=0.0),
        prefetch_workers={0: object()},
        prefetch_request_ids={0: 1},
        mapped_obstacle_points=[],  # no obstacles at all
        config=SimpleNamespace(goal_tolerance=0.25),
        simulation_time=1.0,
        collision_checker=CollisionChecker(),
    )
    fake.runtime_agent = lambda robot_index=None: agent
    fake.safety_radius = lambda: 0.3
    fake.log_console_message = lambda msg: None

    SimulationControllerMixin.on_prefetch_route_ready(
        fake,
        request_id=1,
        robot_index=0,
        success=True,
        reason="ok",
        waypoints=[(2.0, 0.0), (5.0, 0.0)],
    )

    assert agent.pending_path == [(2.0, 0.0), (5.0, 0.0)]
    assert agent.first_segment_blocked_count == 0


# ---------------------------------------------------------------------------
# Formatter (Part A diagnostic line) is a pure, independently-testable
# function.
# ---------------------------------------------------------------------------


def test_narrow_passage_diagnostic_formatter_includes_required_fields():
    line = format_narrow_passage_diagnostic(
        path_goal=PATH_GOAL,
        route_affected_recent=5,
        first_segment_blocked=1,
        predicted_collision=1,
        min_clearance=0.38,
        action="slowdown",
    )

    assert "[NARROW_DIAG]" in line
    assert "path_goal=" in line
    assert "route_affected_recent=5" in line
    assert "first_segment_blocked=1" in line
    assert "predicted_collision=1" in line
    assert "min_clearance=0.38" in line
    assert "action=slowdown" in line
