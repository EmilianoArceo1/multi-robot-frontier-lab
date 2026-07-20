"""
Regression tests for the diagnosis-only instrumentation, and the lazy
reachability-context optimization, around
SimulationControllerMixin.ensure_planner_services()/
make_exploration_reachability_check().

Background: ensure_planner_services() refreshes is_candidate_reachable on
EVERY tick regardless of whether that tick's exploration target selection
ever actually calls it. The instrumentation round confirmed this cost scales
with mapped_obstacle_points and sits, unmeasured, between the "controller"
and "nav_decision" timers. This round makes the expensive part (sanitizing
obstacles + building a planning grid) LAZY: creating the
is_candidate_reachable(xy) callback is now cheap (just a pose/config
snapshot); the real work happens only on the callback's first actual
invocation, and is cached for any further invocations of that SAME callback
object. A fresh callback is built every tick (unchanged), so this needs no
cross-tick invalidation policy -- the next tick's callback simply hasn't
built its context yet, and will do so fresh (with that tick's current pose/
map) the first time it is actually invoked.

Exercises ensure_planner_services()/make_exploration_reachability_check()
directly via a lightweight duck-typed engine fake (the same pattern used
throughout this test suite, see test_exhausted_hold_perf.py). Sections A-D
use cheap stand-ins for sanitize_planner_obstacle_points()/
build_planning_grid_for_robot() (this suite is about the LAZY WIRING, not
the reachability geometry itself, which belongs to planning/* -- out of
scope here); section E additionally exercises the REAL BeliefMap-backed
implementations to confirm the lazy callback produces the exact same
result the old eager code would have, for a representative set of
candidates.
"""
from __future__ import annotations

from types import SimpleNamespace

import robotics_sim.simulation.engine as engine_module
from robotics_sim.simulation.config import WORLD_X_MAX, WORLD_X_MIN, WORLD_Y_MAX, WORLD_Y_MIN
from robotics_sim.simulation.engine import SimulationControllerMixin, candidate_reachable_on_planning_grid
from robotics_sim.simulation.perf_monitor import PerfMonitor


def _stub_out_candidate_reachable_on_planning_grid(monkeypatch) -> None:
    """Sections A-D are about WHEN the reachability context is built, not
    WHAT candidate_reachable_on_planning_grid() computes from it (that is
    covered realistically in section E) -- so the fake planning-grid stub
    (a plain object(), not a real PlanningGrid) never actually needs to
    reach compute_planned_waypoints()."""
    monkeypatch.setattr(engine_module, "candidate_reachable_on_planning_grid", lambda *a, **kw: True)


def _make_fake_engine(*, monitor: PerfMonitor | None = None) -> SimpleNamespace:
    """Fake with cheap stand-ins for the expensive reachability internals --
    for tests about WHEN those internals run (lazy wiring), not what they
    compute."""
    robot = SimpleNamespace(x=1.0, y=2.0, theta=0.0, vision=3.0)
    config = SimpleNamespace(grid_resolution=0.5, planner_type="A*", goal_tolerance=0.25)
    fake = SimpleNamespace(
        robot=robot,
        config=config,
        mapped_obstacle_points=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        _planner_services=None,
    )
    fake.safety_radius_for_robot = lambda robot: 0.3
    fake.sanitize_planner_obstacle_points_calls = []
    fake.build_planning_grid_for_robot_calls = []

    def _fake_sanitize(points, **kwargs):
        fake.sanitize_planner_obstacle_points_calls.append((list(points), kwargs))
        return list(points), 0

    def _fake_build_grid(robot, **kwargs):
        # The real build_planning_grid_for_robot() now sanitizes internally
        # (see engine.py's _planning_costmap_inputs_for_robot()) instead of
        # the caller sanitizing first and passing obstacle_points= in --
        # this stub replicates that one observable call so the existing
        # sanitize_planner_obstacle_points_calls-based assertions below
        # (content and count) still pin the same "when does the expensive
        # work happen" behavior this file is about.
        fake.sanitize_planner_obstacle_points(
            list(fake.mapped_obstacle_points),
            start_xy=(float(robot.x), float(robot.y)),
            robot_radius=kwargs.get("robot_radius"),
            resolution=float(fake.config.grid_resolution),
        )
        fake.build_planning_grid_for_robot_calls.append((robot, kwargs))
        return object()

    fake.sanitize_planner_obstacle_points = _fake_sanitize
    fake.build_planning_grid_for_robot = _fake_build_grid
    if monitor is not None:
        fake.ensure_perf_monitor = lambda: monitor
    for name in (
        "ensure_planner_services",
        "make_exploration_reachability_check",
        # Real, not stubbed: it is a cheap self.robots lookup (no belief/
        # sanitize/grid work of its own -- see its own docstring), and this
        # fake has no "robots" attribute at all, so it always resolves to
        # the single-robot case (empty dynamic points) without needing any
        # extra fixture state.
        "_dynamic_obstacle_points_for_robot_object",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


# ---------------------------------------------------------------------------
# A. Creating the callback (either via ensure_planner_services() or
#    make_exploration_reachability_check() directly) must NOT sanitize
#    obstacles or build a planning grid -- that work is deferred.
# ---------------------------------------------------------------------------


def test_creating_callback_does_not_sanitize_obstacles_or_build_grid():
    fake = _make_fake_engine()

    callback = fake.make_exploration_reachability_check(fake.robot)

    assert callback is not None
    assert fake.sanitize_planner_obstacle_points_calls == []
    assert fake.build_planning_grid_for_robot_calls == []
    assert getattr(fake, "reachability_context_builds", 0) == 0


def test_ensure_planner_services_does_not_build_grid_either():
    fake = _make_fake_engine()

    fake.ensure_planner_services()

    assert fake.sanitize_planner_obstacle_points_calls == []
    assert fake.build_planning_grid_for_robot_calls == []
    assert getattr(fake, "reachability_context_builds", 0) == 0


def test_make_exploration_reachability_check_returns_none_without_a_robot():
    """Unchanged early-return behavior: no robot means no callback built."""
    fake = _make_fake_engine()

    result = fake.make_exploration_reachability_check(None)

    assert result is None
    assert getattr(fake, "reachability_context_builds", 0) == 0


# ---------------------------------------------------------------------------
# B. The FIRST real invocation builds exactly one context; further
#    invocations of the SAME callback reuse it.
# ---------------------------------------------------------------------------


def test_first_invocation_builds_exactly_one_context(monkeypatch):
    _stub_out_candidate_reachable_on_planning_grid(monkeypatch)
    fake = _make_fake_engine()
    callback = fake.make_exploration_reachability_check(fake.robot)

    callback((5.0, 5.0))

    assert len(fake.sanitize_planner_obstacle_points_calls) == 1
    assert len(fake.build_planning_grid_for_robot_calls) == 1
    assert fake.reachability_context_builds == 1


def test_further_invocations_of_same_callback_reuse_context(monkeypatch):
    _stub_out_candidate_reachable_on_planning_grid(monkeypatch)
    fake = _make_fake_engine()
    callback = fake.make_exploration_reachability_check(fake.robot)

    callback((5.0, 5.0))
    callback((6.0, 6.0))
    callback((7.0, 7.0))

    assert len(fake.sanitize_planner_obstacle_points_calls) == 1, (
        "a second/third invocation of the SAME callback must reuse the cached context"
    )
    assert len(fake.build_planning_grid_for_robot_calls) == 1
    assert fake.reachability_context_builds == 1


def test_reachability_context_builds_stays_zero_without_any_invocation():
    """If exploration target selection never calls is_candidate_reachable()
    at all this tick, no grid may be built -- this is the whole point of
    going lazy."""
    fake = _make_fake_engine()

    fake.ensure_planner_services()  # builds/assigns the callback, never calls it

    assert getattr(fake, "reachability_context_builds", 0) == 0
    assert fake.sanitize_planner_obstacle_points_calls == []
    assert fake.build_planning_grid_for_robot_calls == []


# ---------------------------------------------------------------------------
# C. A callback created on a LATER tick builds using THAT tick's current
#    state -- no cross-tick caching, no invalidation policy needed.
# ---------------------------------------------------------------------------


def test_later_tick_callback_builds_with_updated_state(monkeypatch):
    _stub_out_candidate_reachable_on_planning_grid(monkeypatch)
    fake = _make_fake_engine()

    first_tick_callback = fake.make_exploration_reachability_check(fake.robot)
    first_tick_callback((5.0, 5.0))
    assert fake.reachability_context_builds == 1
    assert fake.sanitize_planner_obstacle_points_calls[0][0] == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]

    # Simulate the map growing and the robot moving between ticks.
    fake.mapped_obstacle_points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    fake.robot = SimpleNamespace(x=9.0, y=9.0, theta=0.0, vision=3.0)

    second_tick_callback = fake.make_exploration_reachability_check(fake.robot)
    assert second_tick_callback is not first_tick_callback
    # Not yet invoked -- still must not have built anything for tick 2.
    assert fake.reachability_context_builds == 1

    second_tick_callback((6.0, 6.0))
    assert fake.reachability_context_builds == 2
    assert fake.sanitize_planner_obstacle_points_calls[1][0] == [
        (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)
    ]
    assert fake.sanitize_planner_obstacle_points_calls[1][1]["start_xy"] == (9.0, 9.0)

    # The FIRST tick's callback, invoked again, must still be pinned to that
    # tick's own (already-built, now stale-looking but correct-for-then)
    # context -- it never silently "catches up" to the new state either.
    first_tick_callback((5.5, 5.5))
    assert fake.reachability_context_builds == 2, (
        "re-invoking an already-built callback must reuse ITS OWN cached "
        "context, not rebuild against the newer state"
    )


# ---------------------------------------------------------------------------
# D. Nested timing/counter accounting: no double counting.
# ---------------------------------------------------------------------------


def test_ensure_planner_services_records_only_the_cheap_snapshot():
    """planner_services_refresh_ms must now measure only the cheap
    snapshot/closure-creation work -- NOT reachability_context_build_ms
    (which only runs later, on first invocation)."""
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    fake = _make_fake_engine(monitor=monitor)

    fake.ensure_planner_services()

    assert monitor._section_count.get("planner_services_refresh", 0) == 1
    assert monitor._section_count.get("reachability_context_build", 0) == 0
    assert monitor._section_count.get("reachability_obstacle_prepare", 0) == 0
    assert monitor._section_count.get("reachability_grid_build", 0) == 0


def test_invoking_callback_records_nested_timings_without_double_counting(monkeypatch):
    _stub_out_candidate_reachable_on_planning_grid(monkeypatch)
    monitor = PerfMonitor(env={"SIM_PERF_LOG": "1"})
    fake = _make_fake_engine(monitor=monitor)

    callback = fake.ensure_planner_services().is_candidate_reachable
    callback((5.0, 5.0))
    callback((6.0, 6.0))  # reuses the cached context -- must not add another sample

    assert monitor._section_count.get("reachability_context_build", 0) == 1
    assert monitor._section_count.get("reachability_obstacle_prepare", 0) == 1
    assert monitor._section_count.get("reachability_grid_build", 0) == 1
    # The top-level section must take at least as long as its own nested work.
    assert monitor._section_sum["reachability_context_build"] >= monitor._section_sum["reachability_obstacle_prepare"]


def test_reachability_context_builds_counter_matches_actual_builds_not_calls(monkeypatch):
    _stub_out_candidate_reachable_on_planning_grid(monkeypatch)
    fake = _make_fake_engine()

    for i in range(5):
        callback = fake.ensure_planner_services().is_candidate_reachable
        # Only invoke (and thus actually build) on odd-numbered ticks.
        if i % 2 == 1:
            callback((5.0, 5.0))

    assert fake.reachability_context_builds == 2, (
        "must count actual builds (2, on ticks 1 and 3), not callback creations (5)"
    )


# ---------------------------------------------------------------------------
# E. Functional equivalence with the (former) eager implementation: for a
#    representative set of candidates, the lazy callback must return the
#    exact same result as manually building the context eagerly via the
#    same public methods (sanitize_planner_obstacle_points()/
#    build_planning_grid_for_robot()) and calling
#    candidate_reachable_on_planning_grid() directly -- using the REAL
#    BeliefMap-backed implementations, not stand-ins.
# ---------------------------------------------------------------------------


def _make_real_fake_engine(*, obstacles: list[tuple[float, float, float, float]]) -> SimpleNamespace:
    config = SimpleNamespace(
        grid_resolution=0.5,
        planner_type="A*",
        goal_tolerance=0.25,
        obstacles=list(obstacles),
    )
    robot = SimpleNamespace(x=0.0, y=0.0, theta=0.0, vision=3.0)
    fake = SimpleNamespace(
        robot=robot,
        config=config,
        robots=[],
        mapped_obstacle_points=[
            (float(ox) + 0.05 * i, float(oy))
            for i, (ox, oy, ow, oh) in enumerate(obstacles)
        ],
    )
    fake.safety_radius_for_robot = lambda robot: 0.2
    for name in (
        "ensure_planner_services",
        "make_exploration_reachability_check",
        "reset_belief_map",
        "ensure_belief_map",
        "sanitize_planner_obstacle_points",
        "build_planning_grid_for_robot",
        "_dynamic_obstacle_points_for_robot_object",
        "_planning_costmap_inputs_for_robot",
        "_planning_grid_from_costmap_snapshot",
        "observed_obstacle_snapshot",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    return fake


def _eager_reference_result(fake: SimpleNamespace, candidate_xy: tuple[float, float]) -> bool:
    """Rebuilds exactly what the OLD eager make_exploration_reachability_check()
    did, using the same public methods the lazy version still uses --
    the reference this test compares the lazy callback's result against."""
    robot = fake.robot
    robot_radius = fake.safety_radius_for_robot(robot)
    resolution = float(fake.config.grid_resolution)
    start_xy = (float(robot.x), float(robot.y))
    obstacle_points, _ = fake.sanitize_planner_obstacle_points(
        list(fake.mapped_obstacle_points), start_xy=start_xy, robot_radius=robot_radius, resolution=resolution,
    )
    planning_grid = fake.build_planning_grid_for_robot(robot, obstacle_points=obstacle_points, robot_radius=robot_radius)
    planner_type = str(fake.config.planner_type)
    bounds = (WORLD_X_MIN, WORLD_X_MAX, WORLD_Y_MIN, WORLD_Y_MAX)
    goal_tolerance = float(fake.config.goal_tolerance)
    return candidate_reachable_on_planning_grid(
        planning_grid, planner_type, start_xy, candidate_xy,
        bounds=bounds, resolution=resolution, robot_radius=robot_radius, goal_tolerance=goal_tolerance,
    )


def test_lazy_result_matches_eager_reference_for_reachable_candidate():
    fake = _make_real_fake_engine(obstacles=[(5.0, 5.0, 1.0, 1.0)])
    candidate = (2.0, 0.0)  # open space, no obstacle in the way

    expected = _eager_reference_result(fake, candidate)
    callback = fake.make_exploration_reachability_check(fake.robot)
    actual = callback(candidate)

    assert actual == expected


def test_lazy_result_matches_eager_reference_for_blocked_candidate():
    # A wall of obstacles directly between the robot and the candidate.
    fake = _make_real_fake_engine(
        obstacles=[(1.0 + 0.4 * i, -3.0, 0.3, 6.0) for i in range(10)]
    )
    candidate = (8.0, 0.0)

    expected = _eager_reference_result(fake, candidate)
    callback = fake.make_exploration_reachability_check(fake.robot)
    actual = callback(candidate)

    assert actual == expected


def test_lazy_result_matches_eager_reference_for_out_of_bounds_candidate():
    fake = _make_real_fake_engine(obstacles=[])
    candidate = (WORLD_X_MAX + 50.0, WORLD_Y_MAX + 50.0)

    expected = _eager_reference_result(fake, candidate)
    callback = fake.make_exploration_reachability_check(fake.robot)
    actual = callback(candidate)

    assert actual == expected


def test_lazy_result_matches_eager_reference_for_empty_map():
    fake = _make_real_fake_engine(obstacles=[])
    fake.mapped_obstacle_points = []
    candidate = (3.0, 3.0)

    expected = _eager_reference_result(fake, candidate)
    callback = fake.make_exploration_reachability_check(fake.robot)
    actual = callback(candidate)

    assert actual == expected
