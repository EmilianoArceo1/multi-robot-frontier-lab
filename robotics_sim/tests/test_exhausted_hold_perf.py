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
