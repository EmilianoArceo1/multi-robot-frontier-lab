"""
Regression tests for prefetch endpoint-mismatch rejections not being
remembered, letting the same unreachable target be retried immediately.

Manual Office.sim telemetry:

    [PREFETCH] requested target=(0.25, -3.75)
    [PREFETCH] rejected: final waypoint does not reach target;
    path found with A*; goal adjusted to nearest traversable cell

    ... (later, the same target requested again)

    Planner failed in exploration mode:
    path found with A*; goal adjusted to nearest traversable cell;
    rejected: final waypoint does not reach path goal

Root cause: engine.on_prefetch_route_ready()'s endpoint-mismatch branch
called agent.reject_pending_path(reason), which only clears
pending_path/pending_target_xy and bumps prefetch_fail_count -- it never
adds the rejected target to agent.failed_exploration_targets, the same
memory _pick_next_target() and engine.select_navigation_goal() already
consult to avoid re-proposing a target that just failed to plan. The
*non*-prefetch REQUEST_PLAN failure path (apply_route_result() ->
agent.invalidate_failed_exploration_route() -> mark_exploration_target_failed())
already did this correctly; the prefetch path was the one gap.

Fix: on_prefetch_route_ready()'s rejection branch now also calls
agent.mark_exploration_target_failed(rejected_target, current_time=...)
before discarding it, so the same exclusion window
(_pick_next_target()/select_navigation_goal() via
recently_failed_exploration_targets()) applies to prefetch-rejected targets
exactly as it already does to REQUEST_PLAN-rejected ones.

Test 1 exercises engine.on_prefetch_route_ready() via a minimal duck-typed
engine fake, matching test_route_endpoint_validation.py's pattern. Test 2
exercises ExplorationBehavior._pick_next_target() directly with a
_FakePlannerServices stub, matching test_frontier_reached_target_rejection.py's
pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import ExplorationPlannerResult
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.telemetry import TelemetryLogger


REJECTED_TARGET = (0.25, -3.75)
ROUTE_ENDPOINT_TOO_FAR = (0.75, -3.25)  # planner's "adjusted to nearest traversable cell"


# ---------------------------------------------------------------------------
# 1. on_prefetch_route_ready()'s endpoint-mismatch rejection must mark the
#    pending target as a failed exploration target.
# ---------------------------------------------------------------------------


class _FakeRobot(SimpleNamespace):
    def set_waypoints(self, waypoints):
        self.waypoints = [tuple(p) for p in waypoints]


def _build_fake_engine(*, position=(0.0, -3.5), goal_tolerance=0.25) -> SimpleNamespace:
    robot = _FakeRobot(x=position[0], y=position[1])
    agent = RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")

    fake = SimpleNamespace(
        robot=robot,
        agent=agent,
        config=SimpleNamespace(goal_tolerance=goal_tolerance),
        simulation_time=12.0,
        console_logs=[],
        prefetch_workers={},
        prefetch_request_ids={0: 1},
        # The target request_id=1 was launched for -- on_prefetch_route_
        # ready() now validates against this captured target, not against
        # agent.pending_target_xy directly (see _invalidate_prefetch_
        # request()'s docstring).
        prefetch_targets={0: REJECTED_TARGET},
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.runtime_agent = lambda robot_index=None: fake.agent

    fake.on_prefetch_route_ready = SimulationControllerMixin.on_prefetch_route_ready.__get__(fake)
    fake._invalidate_prefetch_request = SimulationControllerMixin._invalidate_prefetch_request.__get__(fake)
    return fake


def test_prefetch_endpoint_rejection_marks_pending_target_unreachable_or_failed():
    fake = _build_fake_engine()
    fake.agent.pending_target_xy = REJECTED_TARGET

    fake.on_prefetch_route_ready(1, 0, True, "path found with A*; goal adjusted to nearest traversable cell",
                                  [ROUTE_ENDPOINT_TOO_FAR])

    assert fake.agent.pending_path is None, "an endpoint-mismatched route must not be accepted as a pending path"
    assert fake.agent.pending_target_xy is None

    still_failed = fake.agent.recently_failed_exploration_targets(
        current_time=fake.simulation_time, cooldown=999.0
    )
    assert any(
        abs(p[0] - REJECTED_TARGET[0]) < 1e-6 and abs(p[1] - REJECTED_TARGET[1]) < 1e-6 for p in still_failed
    ), "the endpoint-mismatched pending target must be recorded in the same failed-target memory used elsewhere"


# ---------------------------------------------------------------------------
# 2. A target rejected this way must not be immediately re-proposed by
#    normal target selection while still within the exclusion window.
# ---------------------------------------------------------------------------


@dataclass
class _FakePlannerServices:
    """Stand-in for PlannerServices.select_exploration_target()."""

    target: tuple[float, float] | None
    exclusion_radius: float = 0.3
    calls: list[dict] = field(default_factory=list)

    def select_exploration_target(self, **kwargs) -> ExplorationPlannerResult:
        self.calls.append(kwargs)
        if self.target is None:
            return ExplorationPlannerResult(False, None, "no valid frontier candidates found")
        excluded = kwargs.get("excluded_targets") or []
        for point in excluded:
            if abs(self.target[0] - point[0]) <= self.exclusion_radius and abs(self.target[1] - point[1]) <= self.exclusion_radius:
                return ExplorationPlannerResult(False, None, "no reachable frontier candidates: target excluded")
        return ExplorationPlannerResult(True, self.target, "fake planner: selected target")


def _make_agent(position=(0.0, 0.0)) -> RobotAgent:
    return RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")


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


def test_rejected_prefetch_target_is_not_retried_until_new_information_or_cooldown():
    agent = _make_agent(position=(0.0, -3.5))
    # Simulate what on_prefetch_route_ready()'s rejection now does: mark the
    # target failed at t=12.0.
    agent.mark_exploration_target_failed(REJECTED_TARGET, current_time=12.0)

    behavior = ExplorationBehavior()
    # A tick shortly after, well within ExplorationBehavior._FAILED_TARGET_EXCLUSION_WINDOW.
    observation = _make_observation(robot_xy=agent.position, current_time=12.5)
    # The fake planner would (mis)propose the exact rejected target again
    # unless the caller properly excludes it.
    fake_services = _FakePlannerServices(target=REJECTED_TARGET, exclusion_radius=0.3)

    result = behavior._pick_next_target(agent, observation, fake_services)

    assert result is None, (
        "a target rejected moments ago by prefetch endpoint validation must not be "
        "immediately re-proposed -- no repeated [PREFETCH] requested/rejected loop"
    )
    excluded = fake_services.calls[0]["excluded_targets"]
    assert any(
        abs(REJECTED_TARGET[0] - p[0]) < 1e-6 and abs(REJECTED_TARGET[1] - p[1]) < 1e-6 for p in excluded
    ), "the rejected target must be passed as an excluded target to the planner"
