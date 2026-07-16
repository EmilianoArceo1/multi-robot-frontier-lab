"""
Regression tests for SimulationControllerMixin._should_skip_for_exhausted_hold().

Manual Office.sim PERF evidence showed sim_step_ms/render_ms continuing to
grow (and FPS collapsing to ~10) even AFTER exploration exhaustion, when
the robot has nothing left to route to. Root cause: route_affected checks,
belief-snapshot rebuilds, and forced canvas repaints all kept running at
full per-tick rate regardless of whether the agent was latched in an
exploration-exhausted HOLD.

Fix: _should_skip_for_exhausted_hold() reads RobotAgent.
exploration_exhausted_map_signature (a persistent flag already set/cleared
entirely by ExplorationBehavior/RobotAgent's own existing exhaustion
logic -- this reads it, never calls exploration_exhausted() itself or any
navigation/decision method) and throttles the gated work
(route_affected_check, maybe_snapshot_belief, canvas.set_runtime_state) to
at most once every ~1 simulated second while exhausted, instead of every
tick, while leaving normal (non-exhausted) behavior completely untouched.

These tests exercise the method directly via a lightweight duck-typed
engine fake, the same pattern used throughout this test suite (see
test_pending_path_invalidated_by_replan.py) -- no Qt/sensor/planner stack
needed since the method only reads agent.exploration_exhausted_map_signature
and self.simulation_time.
"""
from __future__ import annotations

import time as _time
from types import SimpleNamespace

from robotics_sim.simulation.engine import SimulationControllerMixin


def _make_fake_engine(
    *, exhausted: bool, sim_time: float = 0.0, active_path_goal_xy=None
) -> SimpleNamespace:
    agent = SimpleNamespace(
        exploration_exhausted_map_signature=(5 if exhausted else None),
        active_path_goal_xy=active_path_goal_xy,
    )
    fake = SimpleNamespace(simulation_time=sim_time)
    fake.runtime_agent = lambda robot_index=None: agent
    fake._should_skip_for_exhausted_hold = (
        SimulationControllerMixin._should_skip_for_exhausted_hold.__get__(fake)
    )
    return fake


# ---------------------------------------------------------------------------
# A. While exhausted, high-frequency (sub-1Hz) calls are skipped after the
#    first one.
# ---------------------------------------------------------------------------


def test_exhausted_hold_skips_high_frequency_runtime_updates():
    fake = _make_fake_engine(exhausted=True, sim_time=10.0)

    assert fake._should_skip_for_exhausted_hold() is False, "the first check in a hold episode is always due"

    fake.simulation_time = 10.2
    assert fake._should_skip_for_exhausted_hold() is True, "within the ~1Hz window -- must skip"

    fake.simulation_time = 10.5
    assert fake._should_skip_for_exhausted_hold() is True

    fake.simulation_time = 10.9
    assert fake._should_skip_for_exhausted_hold() is True


def test_not_exhausted_never_skips_regardless_of_call_rate():
    fake = _make_fake_engine(exhausted=False, sim_time=0.0)

    assert fake._should_skip_for_exhausted_hold() is False
    fake.simulation_time = 0.01
    assert fake._should_skip_for_exhausted_hold() is False, "normal (non-exhausted) behavior is never throttled"
    fake.simulation_time = 0.02
    assert fake._should_skip_for_exhausted_hold() is False


# ---------------------------------------------------------------------------
# B. Once the ~1Hz interval elapses, a low-rate update is allowed again --
#    the UI/artifacts never go completely silent while exhausted.
# ---------------------------------------------------------------------------


def test_exhausted_hold_allows_low_rate_status_update():
    fake = _make_fake_engine(exhausted=True, sim_time=0.0)

    assert fake._should_skip_for_exhausted_hold() is False
    fake.simulation_time = 0.5
    assert fake._should_skip_for_exhausted_hold() is True
    fake.simulation_time = 1.1
    assert fake._should_skip_for_exhausted_hold() is False, "due again once ~1 simulated second has elapsed"
    fake.simulation_time = 1.3
    assert fake._should_skip_for_exhausted_hold() is True
    fake.simulation_time = 2.2
    assert fake._should_skip_for_exhausted_hold() is False


# ---------------------------------------------------------------------------
# Regression: a STALE exploration_exhausted_map_signature (left over from an
# earlier genuine exhaustion episode) must not suppress route_affected
# checks/belief snapshots/canvas updates once the agent has an active route
# again (safety-replan-loop/recovering) -- this was the exact root cause of
# route_check_ms staying 0.0 despite route_affected=yes events in the app
# log.
# ---------------------------------------------------------------------------


def test_stale_exhausted_flag_does_not_skip_when_route_is_active():
    fake = _make_fake_engine(exhausted=True, sim_time=5.0, active_path_goal_xy=(7.0, 3.0))

    assert fake._should_skip_for_exhausted_hold() is False, (
        "an active route means the agent is not actually idle, regardless of a stale "
        "exploration_exhausted_map_signature -- must never skip"
    )
    fake.simulation_time = 5.1
    assert fake._should_skip_for_exhausted_hold() is False
    fake.simulation_time = 5.2
    assert fake._should_skip_for_exhausted_hold() is False


# ---------------------------------------------------------------------------
# Exhausted-idle fast path: _exhausted_idle_fast_path_ready() broadens
# _should_skip_for_exhausted_hold()'s own narrower gate (which only
# throttled route_affected/canvas/belief-snapshot work) to cover the ENTIRE
# per-tick pipeline -- sensor update, agent decision, motion integration,
# telemetry -- while the agent is latched exhausted, stopped, and has no
# planner/path work in flight. Real Office.sim evidence: avg_sim_step_ms
# stayed around 20-21ms (unaccounted_ms ~14-16ms) even after nav=exhausted,
# with planner_jobs_started/completed and trace_queue all at 0 -- i.e. the
# engine was doing full per-tick work for an agent that provably had
# nothing left to do.
#
# These tests exercise _exhausted_idle_fast_path_ready() directly via the
# same duck-typed fake pattern as _should_skip_for_exhausted_hold() above,
# plus (for the two "actually skips work" tests) drive the real
# simulation_step() end-to-end for the specific tick where it returns
# early -- proving the early return genuinely prevents
# should_run_sensor_update()/update_sensed_obstacles()/
# canvas.set_runtime_state() from being called, not just that the pure
# gate function returns True in isolation. Building a full working fake
# for the REST of simulation_step() (agent.step(), robot.update(), etc. --
# reached only on non-idle or heartbeat ticks) would mean re-implementing
# most of the engine and is already covered by the rest of this test
# suite; it is intentionally out of scope here.
# ---------------------------------------------------------------------------


def _make_agent(
    *,
    exhausted: bool = True,
    active_path_goal_xy=None,
    pending_path=None,
    pending_target_xy=None,
) -> SimpleNamespace:
    return SimpleNamespace(
        exploration_exhausted_map_signature=(5 if exhausted else None),
        active_path_goal_xy=active_path_goal_xy,
        pending_path=pending_path,
        pending_target_xy=pending_target_xy,
    )


def _make_idle_fake_engine(
    *,
    agent,
    robot_v: float = 0.0,
    stop_speed_tolerance: float = 0.01,
    planning_in_progress: bool = False,
    obstacles=None,
    sim_time: float = 0.0,
) -> SimpleNamespace:
    fake = SimpleNamespace(
        simulation_time=sim_time,
        robot=SimpleNamespace(v=robot_v, stop_speed_tolerance=stop_speed_tolerance),
        planning_in_progress=planning_in_progress,
        config=SimpleNamespace(
            obstacles=list(obstacles) if obstacles is not None else [(0.0, 0.0, 1.0, 1.0)]
        ),
    )
    fake.runtime_agent = lambda robot_index=None: agent
    for name in (
        "_compute_nav_state",
        "_exhausted_idle_fast_path_ready",
        "_should_skip_for_exhausted_hold",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


def test_exhausted_idle_fast_path_ready_when_fully_idle():
    agent = _make_agent(exhausted=True)
    fake = _make_idle_fake_engine(agent=agent)
    assert fake._exhausted_idle_fast_path_ready(agent) is True


def test_exhausted_idle_fast_path_not_ready_with_active_route():
    """Nothing left to route to is required, not just the latched flag --
    same reasoning as _should_skip_for_exhausted_hold()'s own
    active_path_goal_xy check above."""
    agent = _make_agent(exhausted=True, active_path_goal_xy=(3.0, 4.0))
    fake = _make_idle_fake_engine(agent=agent)
    assert fake._exhausted_idle_fast_path_ready(agent) is False


def test_exhausted_idle_fast_path_not_ready_when_not_exhausted():
    agent = _make_agent(exhausted=False)
    fake = _make_idle_fake_engine(agent=agent)
    assert fake._exhausted_idle_fast_path_ready(agent) is False


def test_exhausted_idle_fast_path_does_not_run_with_pending_planner_job():
    # Engine-level planner job in flight.
    agent = _make_agent(exhausted=True)
    fake = _make_idle_fake_engine(agent=agent, planning_in_progress=True)
    assert fake._exhausted_idle_fast_path_ready(agent) is False

    # Agent-level prefetch result awaiting acceptance -- a distinct
    # in-flight-work signal from planning_in_progress (prefetch never
    # touches that flag), and just as disqualifying.
    agent_with_pending_path = _make_agent(exhausted=True, pending_path=[(1.0, 1.0)])
    fake_pending_path = _make_idle_fake_engine(agent=agent_with_pending_path)
    assert fake_pending_path._exhausted_idle_fast_path_ready(agent_with_pending_path) is False

    agent_with_pending_target = _make_agent(exhausted=True, pending_target_xy=(2.0, 2.0))
    fake_pending_target = _make_idle_fake_engine(agent=agent_with_pending_target)
    assert fake_pending_target._exhausted_idle_fast_path_ready(agent_with_pending_target) is False


def test_exhausted_idle_fast_path_does_not_run_while_robot_moving():
    agent = _make_agent(exhausted=True)
    fake_moving = _make_idle_fake_engine(agent=agent, robot_v=0.5, stop_speed_tolerance=0.01)
    assert fake_moving._exhausted_idle_fast_path_ready(agent) is False

    # At/below the stop tolerance -- provably not moving -- is ready.
    fake_stopped = _make_idle_fake_engine(agent=agent, robot_v=0.005, stop_speed_tolerance=0.01)
    assert fake_stopped._exhausted_idle_fast_path_ready(agent) is True


def test_exhausted_idle_fast_path_exits_when_map_changes():
    agent = _make_agent(exhausted=True)
    fake = _make_idle_fake_engine(agent=agent, obstacles=[(0.0, 0.0, 1.0, 1.0)])

    assert fake._exhausted_idle_fast_path_ready(agent) is True

    # New ground-truth obstacles appear (e.g. edited into the running
    # scenario) -- must exit immediately, not keep treating the situation
    # as unchanged just because the robot/agent state itself is the same.
    fake.config.obstacles.append((5.0, 5.0, 1.0, 1.0))
    assert fake._exhausted_idle_fast_path_ready(agent) is False

    # Once the new baseline is established, a further unchanged tick is
    # ready again.
    assert fake._exhausted_idle_fast_path_ready(agent) is True


class _RaisingStub:
    """Callable that fails the test loudly if invoked -- used to prove a
    method was never reached, rather than merely asserting a call count
    after the fact."""

    def __init__(self, message: str):
        self._message = message

    def __call__(self, *args, **kwargs):
        raise AssertionError(self._message)


def _make_full_fake_engine_for_fast_path_tick(
    *, agent, sim_time: float, last_exhausted_low_rate_time: float
) -> SimpleNamespace:
    """A fake engine complete enough to drive the REAL simulation_step()
    through the exhausted-idle fast path's early return -- but no
    further; should_run_sensor_update/update_sensed_obstacles/
    canvas.set_runtime_state all raise if reached, proving the early
    return actually happens rather than merely asserting call counts."""
    fake = _make_idle_fake_engine(agent=agent, sim_time=sim_time)
    fake.last_time = _time.perf_counter() - 1.0
    fake.running = True
    fake.robots = []
    fake.paused = False
    fake.robot = SimpleNamespace(v=0.0, stop_speed_tolerance=0.01, x=1.0, y=2.0)
    fake.simulation_speed = 1.0
    fake.collision_checker = object()
    fake._last_exhausted_low_rate_time = last_exhausted_low_rate_time
    fake.exhausted_idle_fast_path_hits = 0
    fake.exhausted_idle_full_updates = 0
    fake.exhausted_idle_skipped_canvas_updates = 0
    fake.exhausted_idle_skipped_sensor_updates = 0
    fake.should_run_sensor_update = _RaisingStub(
        "should_run_sensor_update must not be called while the exhausted-idle fast path is skipping this tick"
    )
    fake.update_sensed_obstacles = _RaisingStub(
        "update_sensed_obstacles must not be called while the exhausted-idle fast path is skipping this tick"
    )
    fake.canvas = SimpleNamespace(
        set_runtime_state=_RaisingStub(
            "canvas.set_runtime_state must not be called while the exhausted-idle fast path is skipping this tick"
        )
    )
    fake.simulation_step = SimulationControllerMixin.simulation_step.__get__(fake)
    return fake


def test_exhausted_idle_fast_path_skips_obstacle_extraction():
    agent = _make_agent(exhausted=True)
    # 0.5s ahead of the throttle baseline -- well within the ~1Hz window,
    # so this tick is not "due" and must take the fast path.
    fake = _make_full_fake_engine_for_fast_path_tick(
        agent=agent, sim_time=0.5, last_exhausted_low_rate_time=0.0
    )

    fake.simulation_step()  # would raise via should_run_sensor_update's stub if it reached the sensor block

    assert fake.exhausted_idle_fast_path_hits == 1
    assert fake.exhausted_idle_skipped_sensor_updates == 1


def test_exhausted_idle_fast_path_skips_canvas_until_heartbeat():
    agent = _make_agent(exhausted=True)
    fake = _make_full_fake_engine_for_fast_path_tick(
        agent=agent, sim_time=0.5, last_exhausted_low_rate_time=0.0
    )

    fake.simulation_step()  # would raise via canvas.set_runtime_state's stub if reached

    assert fake.exhausted_idle_skipped_canvas_updates == 1
    assert fake.exhausted_idle_fast_path_hits == 1

    # Once the ~1 simulated second interval elapses, the heartbeat must
    # become due (skip_for_exhausted_hold=False) even though the agent is
    # still otherwise fully idle-eligible -- confirms the skip is not
    # permanent, just low-rate.
    fake.simulation_time = 1.6
    assert fake._exhausted_idle_fast_path_ready(agent) is True
    assert fake._should_skip_for_exhausted_hold() is False, (
        "once due, the heartbeat must run -- the fast path must not suppress it forever"
    )
