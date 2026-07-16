"""
Regression tests for the Codex-review finding: "prefetch workers obsoletos
sobreviven despues de descartar pending paths."

Root cause: engine.py tracked one prefetch's worker (prefetch_workers[idx])
and request id (prefetch_request_ids[idx]) per robot slot, but nothing
discarding pending state on the RobotAgent (invalidate_route(),
invalidate_pending_path(), invalidate_failed_exploration_route(),
reject_pending_path(), or a direct pending_path/pending_target_xy=None
assignment during snapshot restore) ever touched those engine-level dicts.
A still-running worker for an abandoned request therefore:

    1. blocked request_prefetch_route_async()'s "already running" guard
       from launching a replacement for the same robot slot, and
    2. once it finally completed, on_prefetch_route_ready() validated its
       route against agent.pending_target_xy -- LIVE agent state that, by
       the time the stale callback landed, could belong to an unrelated,
       newer request -- risking blacklisting a target that request never
       even attempted.

Fix: engine._invalidate_prefetch_request(robot_id, reason) is now called
at every site that discards pending state (see engine.py's call sites),
and on_prefetch_route_ready() validates against prefetch_targets[idx] (the
target THIS request captured at launch), never agent.pending_target_xy.

This file exercises the full A/B lifecycle end to end via a minimal
duck-typed engine fake, matching the pattern in test_route_endpoint_
validation.py / test_narrow_passage_replanning.py.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.simulation.engine import SimulationControllerMixin

TARGET_A = (2.0, 2.0)
ROUTE_A = [(1.0, 1.0), TARGET_A]
TARGET_B = (5.0, 5.0)
ROUTE_B = [(4.0, 4.0), TARGET_B]


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine() -> SimpleNamespace:
    agent = RobotAgent(robot_id=0, position=(0.0, 0.0), planner_mode="FoV-aware directional frontier")
    fake = SimpleNamespace(
        robot=_FakeRobot(x=0.0, y=0.0),
        config=SimpleNamespace(goal_tolerance=0.25),
        mapped_obstacle_points=[],
        simulation_time=10.0,
        console_logs=[],
        prefetch_workers={},
        prefetch_request_ids={},
        prefetch_targets={},
    )
    fake.collision_checker = CollisionChecker()  # no obstacle points -> never blocks
    fake.safety_radius = lambda: 0.2
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.runtime_agent = lambda robot_index=None: agent
    for name in ("on_prefetch_route_ready", "_invalidate_prefetch_request", "_invalidate_all_prefetch_requests"):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake, agent


# ---------------------------------------------------------------------------
# Full A/B narrative: 1-launch A, 2-invalidate A, 3-launch B before A
# finishes, 4-A's callback lands late, 5-A is ignored, 6-B is not
# blacklisted, 7-B's callback can accept its own route, 8-reset/restore
# invalidates whatever is left.
# ---------------------------------------------------------------------------


def test_stale_prefetch_callback_is_ignored_and_does_not_disturb_the_newer_request():
    fake, agent = _build_fake_engine()

    # 1. Launch A.
    agent.mark_pending_path_requested(TARGET_A)
    fake.prefetch_workers[0] = object()
    fake.prefetch_request_ids[0] = 1
    fake.prefetch_targets[0] = TARGET_A
    assert fake.prefetch_request_ids[0] == 1
    assert fake.prefetch_targets[0] == TARGET_A

    # 2. Invalidate A (e.g. a route-affected repair or safety replan would
    # call agent.invalidate_pending_path() alongside this).
    agent.invalidate_pending_path(reason="route-affected repair")
    fake._invalidate_prefetch_request(0, reason="route-affected repair")
    assert 0 not in fake.prefetch_workers
    assert 0 not in fake.prefetch_request_ids
    assert 0 not in fake.prefetch_targets
    assert agent.pending_target_xy is None

    # 3. Launch B before A's worker has actually reported back -- the slot
    # is free immediately (this is exactly what request_prefetch_route_
    # async()'s "already running" guard needed: prefetch_workers no longer
    # contains a leftover entry for A).
    agent.mark_pending_path_requested(TARGET_B)
    fake.prefetch_workers[0] = object()
    fake.prefetch_request_ids[0] = 2
    fake.prefetch_targets[0] = TARGET_B

    # 4. A's callback finally lands late (request_id=1).
    fake.on_prefetch_route_ready(1, 0, True, "ok", list(ROUTE_A))

    # 5. A is ignored: B's request-tracking state is completely untouched.
    assert agent.pending_path is None, "A's late callback must not install A's route"
    assert agent.pending_target_xy == TARGET_B, "A's callback must not touch B's live pending target"
    assert fake.prefetch_request_ids[0] == 2, "A's callback must not pop B's request id"
    assert fake.prefetch_targets[0] == TARGET_B, "A's callback must not pop B's captured target"

    # 6. B is not blacklisted by A's rejection-adjacent bookkeeping.
    failed = agent.recently_failed_exploration_targets(current_time=fake.simulation_time, cooldown=999.0)
    assert TARGET_B not in failed
    assert TARGET_A not in failed  # A was invalidated, not rejected -- never blacklisted either

    # 7. B's own (matching) callback is accepted normally.
    fake.on_prefetch_route_ready(2, 0, True, "ok", list(ROUTE_B))
    assert agent.pending_path == ROUTE_B
    accepted = agent.accept_pending_path()
    assert accepted == ROUTE_B
    assert agent.active_path_goal_xy == TARGET_B
    assert 0 not in fake.prefetch_request_ids, "B's own completion must retire its request id too"


def test_reset_or_restore_invalidates_every_in_flight_prefetch_request():
    fake, agent = _build_fake_engine()

    agent.mark_pending_path_requested(TARGET_A)
    fake.prefetch_workers[0] = object()
    fake.prefetch_request_ids[0] = 1
    fake.prefetch_targets[0] = TARGET_A
    # A second robot slot, e.g. multi-robot mode.
    fake.prefetch_workers[1] = object()
    fake.prefetch_request_ids[1] = 7
    fake.prefetch_targets[1] = (9.0, 9.0)

    fake._invalidate_all_prefetch_requests(reason="simulation reset/restore")

    assert fake.prefetch_workers == {}
    assert fake.prefetch_request_ids == {}
    assert fake.prefetch_targets == {}

    # A late callback for either retired request is now silently ignored.
    fake.on_prefetch_route_ready(1, 0, True, "ok", list(ROUTE_A))
    assert agent.pending_path is None


# ---------------------------------------------------------------------------
# The helper itself: a pure no-op when nothing is in flight for that robot.
# ---------------------------------------------------------------------------


def test_invalidate_prefetch_request_is_a_no_op_with_nothing_in_flight():
    fake, _agent = _build_fake_engine()

    fake._invalidate_prefetch_request(0, reason="nothing to invalidate")

    assert fake.prefetch_workers == {}
    assert fake.prefetch_request_ids == {}
    assert fake.prefetch_targets == {}
    assert fake.console_logs == [], "a no-op invalidation must not log a spurious [PREFETCH] message"


def test_invalidate_prefetch_request_only_touches_the_given_robot_slot():
    fake, _agent = _build_fake_engine()
    fake.prefetch_workers = {0: object(), 1: object()}
    fake.prefetch_request_ids = {0: 1, 1: 2}
    fake.prefetch_targets = {0: TARGET_A, 1: TARGET_B}

    fake._invalidate_prefetch_request(0, reason="robot 0 only")

    assert 0 not in fake.prefetch_workers
    assert 0 not in fake.prefetch_request_ids
    assert 0 not in fake.prefetch_targets
    assert fake.prefetch_workers[1] is not None
    assert fake.prefetch_request_ids[1] == 2
    assert fake.prefetch_targets[1] == TARGET_B
