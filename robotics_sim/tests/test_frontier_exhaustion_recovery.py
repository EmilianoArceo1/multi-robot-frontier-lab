"""
Regression tests for premature exploration exhaustion at ~20.9% explored.

Manual Office.sim telemetry, after the repeated_safety_replan grace-period
fix (test_safety_replan_loop.py) stopped killing valid routes prematurely:

    [NAV] R1 kind=HOLD reason="frontier reached; no valid next frontier available"
    [NAV] R1 kind=REQUEST_PLAN reason="recovery: trying recent safe target before exhaustion"
    ...
    [NAV] R1 kind=HOLD reason="exploration exhausted: no reachable frontier candidates after recovery"
    [NAV] R1 kind=HOLD reason="exploration exhausted: no reachable frontier candidates"

The robot was moving and many routes succeeded ([ROUTE ok]) -- this is not
the safety-replan bug. The problem is in frontier candidate generation.

Root cause: ExplorationBehavior._pick_next_target() always asks
PlannerServices for agent.planner_mode's candidates -- by default "FoV-aware
directional frontier", which restricts candidates to the robot's current
field of view / heading (see FoVAwareDirectionalFrontierPlanner in
exploration_planners.py). When that local/directional window is
momentarily empty, the ONLY fallback tried before declaring exhaustion was
RecoveryPolicy -- which only ever proposes points from
agent.recent_safe_positions, i.e. places the robot has already BEEN. That
pool is bounded (RobotAgent._RECENT_SAFE_POSITION_LIMIT = 20) and gets
permanently excluded entry-by-entry within a recovery episode
(agent.recent_recovery_targets, never map_signature-scoped -- see
test_exploration_recovery_policy.py). So a local FoV window running dry,
combined with RecoveryPolicy eventually exhausting its finite backtrack
pool, together declared exploration exhausted even while genuinely
unexplored, reachable map remained elsewhere on the belief map.

Fix: ExplorationBehavior._pick_map_wide_fallback_target() retries frontier
selection with "Nearest frontier" -- an existing, already-registered
exploration planner (FrontierExplorationPlanner) that scores candidates
directly from the belief map with no FoV/heading restriction -- using the
exact same PlannerServices.select_exploration_target() call every other
target-selection path already goes through. Tried in both places
_pick_next_target() can return None (step 3 "frontier reached", step 6 "no
path"), before RecoveryPolicy's backtrack fallback and before exhaustion
bookkeeping. No new planner code, no A* changes, no change to endpoint
validation, failed-target memory, or the exhaustion budget itself -- only
one additional existing-planner-name attempt before giving up.

A small, DEBUG-level-only diagnostic ([EXHAUSTION_DIAG]) is logged by
engine.apply_navigation_decision() at the exact moment
exploration_exhausted() actually fires, using data already computed for
that decision (never a new per-tick pass) -- see test 4 below.

These tests exercise RobotAgent and ExplorationBehavior directly (no Qt, no
canvas, no real BeliefMap/A*), matching the existing pattern in
test_exploration_recovery_policy.py / test_exploration_target_recovery.py.
Test 4 additionally exercises engine.apply_navigation_decision() via the
same minimal duck-typed engine fake test_recovery_rejects_reached_targets.py
uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.telemetry import DEBUG, TelemetryLogger


LOCAL_PLANNER = "FoV-aware directional frontier"
MAP_WIDE_PLANNER = "Nearest frontier"
FAILED_TARGET = (7.75, -4.75)
ALTERNATE_TARGET = (2.0, 3.0)
MAP_WIDE_TARGET = (-6.0, 4.5)


@dataclass
class _FakePlannerServices:
    """Stand-in for PlannerServices.select_exploration_target() -- same
    canned result regardless of planner_name (matches the fakes used
    throughout the sibling exploration-behavior test files)."""

    target: tuple[float, float] | None
    calls: list[dict] = field(default_factory=list)

    def select_exploration_target(self, **kwargs) -> ExplorationPlannerResult:
        self.calls.append(kwargs)
        if self.target is None:
            return ExplorationPlannerResult(False, None, "no valid frontier candidates found")
        return ExplorationPlannerResult(True, self.target, "fake planner: selected target")


@dataclass
class _PlannerNameAwareFakePlannerServices:
    """Returns a different canned result per planner_name, so tests can
    simulate "the configured local/FoV planner finds nothing, but the
    map-wide fallback planner does" (or vice versa)."""

    targets_by_planner: dict[str, tuple[float, float] | None]
    calls: list[dict] = field(default_factory=list)

    def select_exploration_target(self, **kwargs) -> ExplorationPlannerResult:
        self.calls.append(kwargs)
        planner_name = kwargs.get("planner_name")
        target = self.targets_by_planner.get(planner_name)
        candidates = () if target is None else (object(),)
        if target is None:
            return ExplorationPlannerResult(
                False, None, f"{planner_name}: no reachable frontier candidates", candidates
            )
        return ExplorationPlannerResult(True, target, f"{planner_name}: selected target", candidates)


def _make_agent(position=(0.0, 0.0)) -> RobotAgent:
    return RobotAgent(robot_id=0, position=position, planner_mode=LOCAL_PLANNER)


def _make_observation(**overrides) -> RobotObservation:
    defaults = dict(
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        robot_radius=0.2,
        belief_map=None,
        planning_grid=None,
        mapped_obstacle_points=[],
        dynamic_obstacles=[],
        active_segment_blocked=False,
        predicted_collision=False,
        current_time=0.0,
        grid_resolution=0.5,
        goal_tolerance=0.25,
        sensor_range=2.5,
        final_goal_xy=None,
    )
    defaults.update(overrides)
    return RobotObservation(**defaults)


# ---------------------------------------------------------------------------
# 1. Local/FoV candidates exhausted, but a map-wide frontier candidate
#    exists -- exhaustion must not be declared; the fallback is used.
# ---------------------------------------------------------------------------


def test_exhaustion_not_declared_when_map_frontier_fallback_exists():
    agent = _make_agent()
    behavior = ExplorationBehavior()
    fake_services = _PlannerNameAwareFakePlannerServices(
        targets_by_planner={
            LOCAL_PLANNER: None,
            MAP_WIDE_PLANNER: MAP_WIDE_TARGET,
        }
    )
    observation = _make_observation(robot_xy=agent.position, current_time=1.0)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == MAP_WIDE_TARGET
    assert "exhausted" not in decision.reason
    assert agent.consecutive_exploration_failures == 0, (
        "finding a target via the map-wide fallback must not be counted as a failure"
    )
    # Both planners were actually consulted (local first, then fallback).
    planner_names_tried = [call["planner_name"] for call in fake_services.calls]
    assert planner_names_tried == [LOCAL_PLANNER, MAP_WIDE_PLANNER]


def test_map_wide_fallback_is_skipped_when_it_is_already_the_configured_planner():
    """No pointless duplicate call when agent.planner_mode already IS the
    fallback planner."""
    agent = RobotAgent(robot_id=0, position=(0.0, 0.0), planner_mode=MAP_WIDE_PLANNER)
    behavior = ExplorationBehavior()
    fake_services = _FakePlannerServices(target=None)
    observation = _make_observation(robot_xy=agent.position, current_time=1.0)

    behavior.update(agent, observation, fake_services)

    assert len(fake_services.calls) == 1, (
        "the fallback must not re-run the same planner a second time this cycle"
    )


# ---------------------------------------------------------------------------
# 2. A target outside the failed-target exclusion window must not stay
#    permanently blacklisted -- exhaustion must not be declared solely
#    because of stale failed_recent entries.
# ---------------------------------------------------------------------------


def test_failed_recent_filter_does_not_permanently_blacklist_all_frontiers():
    agent = _make_agent()
    behavior = ExplorationBehavior()

    agent.mark_exploration_target_failed(FAILED_TARGET, current_time=0.0)
    assert FAILED_TARGET in agent.recently_failed_exploration_targets(
        current_time=0.5, cooldown=behavior._FAILED_TARGET_EXCLUSION_WINDOW
    )

    # Time passes beyond the exclusion window.
    later = behavior._FAILED_TARGET_EXCLUSION_WINDOW + 1.0
    assert FAILED_TARGET not in agent.recently_failed_exploration_targets(
        current_time=later, cooldown=behavior._FAILED_TARGET_EXCLUSION_WINDOW
    ), "a target outside the exclusion window must no longer read as recently-failed"

    fake_services = _FakePlannerServices(target=ALTERNATE_TARGET)
    observation = _make_observation(robot_xy=agent.position, current_time=later)
    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == ALTERNATE_TARGET
    assert "exhausted" not in decision.reason
    excluded = fake_services.calls[0]["excluded_targets"]
    assert FAILED_TARGET not in excluded, (
        "a target outside the failed-target exclusion window must not still be "
        "passed as excluded, or it could never be reconsidered even after real "
        "map/time progress"
    )


# ---------------------------------------------------------------------------
# 3. When recovery has nothing useful to offer (its only candidate is the
#    robot's own current position), the map-wide fallback is tried before
#    exhaustion -- recovery never gets a chance to loop without progress.
# ---------------------------------------------------------------------------


def test_recovery_does_not_loop_between_recent_safe_targets_without_progress():
    agent = _make_agent()
    behavior = ExplorationBehavior()
    # The only "recent safe position" is where the robot already is --
    # RecoveryPolicy will reject it (too close to robot_xy), so recovery
    # alone has nothing useful to offer this cycle.
    agent.recent_safe_positions.append(agent.position)

    fake_services = _PlannerNameAwareFakePlannerServices(
        targets_by_planner={
            LOCAL_PLANNER: None,
            MAP_WIDE_PLANNER: MAP_WIDE_TARGET,
        }
    )
    observation = _make_observation(robot_xy=agent.position, current_time=1.0)

    decision = behavior.update(agent, observation, fake_services)

    assert decision.kind == "REQUEST_PLAN"
    assert decision.target == MAP_WIDE_TARGET, (
        "with recovery unable to offer anything useful, the map-wide fallback "
        "must be tried before declaring exhaustion"
    )
    assert agent.recovery_targets() == [], (
        "a target found via the map-wide fallback is real exploration progress -- "
        "it must end the recovery episode exactly like a normal frontier target, "
        "and RecoveryPolicy must never even have been consulted"
    )


# ---------------------------------------------------------------------------
# 4. When exploration genuinely IS exhausted (map-wide fallback also finds
#    nothing), the engine logs a diagnostic explaining why -- DEBUG-level
#    only, not a routine per-tick line.
# ---------------------------------------------------------------------------


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def test_exhaustion_reports_diagnostic_counts():
    agent = _make_agent()
    agent.last_map_wide_fallback_attempted = True
    agent.last_frontier_candidate_count = 3
    agent.last_frontier_selection_reason = "Nearest frontier: no reachable frontier candidates"
    agent.recent_recovery_targets.append((1.0, 1.0))

    robot = _FakeRobot(x=0.0, y=0.0)
    console_logs: list[str] = []
    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        belief_map=None,
        mapped_obstacle_points=[],
        simulation_time=5.0,
        config=SimpleNamespace(goal_tolerance=0.25),
    )
    fake.telemetry = TelemetryLogger(level=DEBUG, sink=console_logs.append)
    fake.canvas = SimpleNamespace(set_exploration_target=lambda target: None)
    fake.is_exploration_mode = lambda: True
    fake.runtime_agent = lambda robot_index=None: agent
    fake.set_robot_goal_or_waypoints = lambda robot_obj, waypoints: robot_obj.set_waypoints(
        waypoints or [(robot_obj.x, robot_obj.y)]
    )
    fake.apply_navigation_decision = SimulationControllerMixin.apply_navigation_decision.__get__(fake)

    decision = SimpleNamespace(
        kind="HOLD",
        reason="exploration exhausted: no reachable frontier candidates after recovery",
        target=None,
        brake=False,
        force_new_target=False,
    )

    SimulationControllerMixin.apply_navigation_decision(fake, robot, agent, decision)

    diag_lines = [line for line in console_logs if "[EXHAUSTION_DIAG]" in line]
    assert len(diag_lines) == 1, "exactly one diagnostic line, not a routine per-tick spam"
    line = diag_lines[0]
    assert "unknown_cells=" in line
    assert "recovery_candidates=1" in line
    assert "map_wide_fallback_tried=True" in line
    assert "last_frontier_candidates=3" in line
    assert "exploration exhausted: no reachable frontier candidates after recovery" in line


def test_exhaustion_diagnostic_not_logged_for_non_exhaustion_holds():
    """The diagnostic is scoped to genuine exhaustion holds -- an ordinary
    HOLD (e.g. retry cooldown) must not produce an [EXHAUSTION_DIAG] line."""
    agent = _make_agent()
    robot = _FakeRobot(x=0.0, y=0.0)
    console_logs: list[str] = []
    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        belief_map=None,
        mapped_obstacle_points=[],
        simulation_time=5.0,
        config=SimpleNamespace(goal_tolerance=0.25),
    )
    fake.telemetry = TelemetryLogger(level=DEBUG, sink=console_logs.append)
    fake.canvas = SimpleNamespace(set_exploration_target=lambda target: None)
    fake.is_exploration_mode = lambda: True
    fake.runtime_agent = lambda robot_index=None: agent
    fake.set_robot_goal_or_waypoints = lambda robot_obj, waypoints: robot_obj.set_waypoints(
        waypoints or [(robot_obj.x, robot_obj.y)]
    )
    fake.apply_navigation_decision = SimulationControllerMixin.apply_navigation_decision.__get__(fake)

    decision = SimpleNamespace(
        kind="HOLD",
        reason="recovering after planner failure; retry cooldown active",
        target=None,
        brake=False,
        force_new_target=False,
    )

    SimulationControllerMixin.apply_navigation_decision(fake, robot, agent, decision)

    assert not any("[EXHAUSTION_DIAG]" in line for line in console_logs)
