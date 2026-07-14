"""
Regression tests for the diagnosis-only instrumentation added around
SimulationControllerMixin.ensure_planner_services()/
make_exploration_reachability_check().

Hypothesis under test (see engine.py's docstrings for the full reasoning):
ensure_planner_services() sat between the "controller" and "nav_decision"
timers in simulation_step() with no timer of its own, and its own docstring
says it refreshes is_candidate_reachable on EVERY call -- rebuilding a
sanitized obstacle-point list and a planning grid from
mapped_obstacle_points regardless of whether that tick actually selects a
new exploration target. This round only ADDS timers/counters around that
existing call chain; it must not change how often ensure_planner_services()
refreshes the callback, what it builds, or any navigation/frontier
selection behavior.

Exercises ensure_planner_services()/make_exploration_reachability_check()
directly via a lightweight duck-typed engine fake (the same pattern used
throughout this test suite, see test_exhausted_hold_perf.py) -- sanitize_
planner_obstacle_points()/build_planning_grid_for_robot() are stubbed to
cheap fakes so this does not need a real belief map/planning grid.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.perf_monitor import PerfMonitor


def _make_fake_engine(*, monitor: PerfMonitor | None = None) -> SimpleNamespace:
    robot = SimpleNamespace(x=1.0, y=2.0, theta=0.0, vision=3.0)
    config = SimpleNamespace(grid_resolution=0.5, planner_type="A*", goal_tolerance=0.25)
    fake = SimpleNamespace(
        robot=robot,
        config=config,
        mapped_obstacle_points=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        _planner_services=None,
    )
    fake.safety_radius_for_robot = lambda robot: 0.3
    # Cheap stand-ins for the real (belief-map-backed) implementations --
    # this test is about the TIMING/COUNTING wiring, not the reachability
    # geometry itself, which belongs to planning/* (out of scope here).
    fake.sanitize_planner_obstacle_points = lambda points, **kwargs: (list(points), 0)
    fake.build_planning_grid_for_robot = lambda robot, **kwargs: object()
    if monitor is not None:
        fake.ensure_perf_monitor = lambda: monitor
    for name in ("ensure_planner_services", "make_exploration_reachability_check"):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


# ---------------------------------------------------------------------------
# A. ensure_planner_services() must still refresh is_candidate_reachable
#    (build exactly one fresh reachability context) on EVERY call -- this
#    round only adds timers/counters, never a new gate or cache.
# ---------------------------------------------------------------------------


def test_ensure_planner_services_still_refreshes_every_call():
    fake = _make_fake_engine()

    first_callback = fake.ensure_planner_services().is_candidate_reachable
    assert fake.reachability_context_builds == 1

    second_callback = fake.ensure_planner_services().is_candidate_reachable
    assert fake.reachability_context_builds == 2
    assert second_callback is not first_callback, (
        "is_candidate_reachable must still be rebuilt fresh every call, exactly as before"
    )

    fake.ensure_planner_services()
    assert fake.reachability_context_builds == 3


def test_make_exploration_reachability_check_returns_none_without_a_robot():
    """Unchanged early-return behavior: no robot means no callback built,
    and the build counter must not increment for a call that built nothing."""
    fake = _make_fake_engine()

    result = fake.make_exploration_reachability_check(None)

    assert result is None
    assert getattr(fake, "reachability_context_builds", 0) == 0


# ---------------------------------------------------------------------------
# B. The new top-level/nested timings are actually recorded, with the
#    correct top-level vs. nested relationship, when a real PerfMonitor is
#    wired in.
# ---------------------------------------------------------------------------


def test_ensure_planner_services_records_top_level_and_nested_timings():
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    fake = _make_fake_engine(monitor=monitor)

    fake.ensure_planner_services()

    assert monitor._section_count.get("planner_services_refresh", 0) == 1
    assert monitor._section_count.get("reachability_context_build", 0) == 1
    assert monitor._section_count.get("reachability_obstacle_prepare", 0) == 1
    assert monitor._section_count.get("reachability_grid_build", 0) == 1

    # The top-level section must take at least as long as its own nested
    # work (it wraps make_exploration_reachability_check() entirely).
    assert monitor._section_sum["planner_services_refresh"] >= monitor._section_sum["reachability_context_build"]


def test_reachability_context_builds_counter_matches_call_count():
    fake = _make_fake_engine()

    for _ in range(5):
        fake.ensure_planner_services()

    assert fake.reachability_context_builds == 5
