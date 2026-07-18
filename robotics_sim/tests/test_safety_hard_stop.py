"""
Regression tests for residual robot velocity surviving a safety HOLD /
repeated safety replan, letting the robot coast into a collision after
navigation state has already decided to stop.

Manual Office.sim telemetry:

    [NAV] R1 kind=REPLAN_FOR_SAFETY reason="predicted collision"
    safety replan: predicted collision ...
    [ROUTE ok] ... goal=(4.25,-3.75)

    safety replan: predicted collision ...
    [ROUTE ok] ... goal=(4.25,-3.75)

    Holding: repeated safety replan for the same target (predicted collision);
    marking target as failed and re-selecting.
    [ROUTE fail] ... reason=repeated_safety_replan

    [NAV] HOLD reason="recovering after planner failure; retry cooldown active"
    [NAV] HOLD reason="predicted collision"

    [STATE] ... v=0.47 state=IDLE hold_pos=(5.60,-4.13) target=None path_goal=None
    COLLISION: robot entered an obstacle safety region after update.

Root cause: navigation state (waypoints/active_path_goal_xy/exploration
target) was correctly invalidated at every safety HOLD point, but nothing
zeroed the robot's actual velocity. Robot.brake_control() only decelerates
gradually (acceleration = -v, clamped to max_acceleration), and
DynamicUnicycle2D.step() advances POSITION using the velocity FROM BEFORE
this tick's acceleration is applied:

    x_{k+1} = x_k + v_k * cos(theta_k) * dt   (v_k is the OLD velocity)
    v_{k+1} = v_k + a_k * dt

So even a textbook brake control still lets the robot travel v_k*dt
further on the very tick braking starts, and takes multiple ticks to fully
stop -- during which the robot keeps moving with decreasing but nonzero
velocity. If the robot is already close to an obstacle when a safety HOLD
fires, this residual coasting alone can carry it into the unsafe region,
even though navigation logic had already correctly decided to stop.

Fix: Robot.force_stop() (robot.py -- where velocity actually lives, not
RobotAgent, which tracks no dynamics) sets state.v = 0.0 directly,
bypassing the gradual deceleration model entirely, and resets the
controller's state machine so the next control computation starts from a
clean mode rather than a stale TRACK/ROTATE one. engine.apply_navigation_decision()
calls it at the two single-robot call sites that can leave a safety-invalidated
route in place: the generic HOLD handler, and the repeated-safety-replan
branch inside REPLAN_FOR_SAFETY handling.

Test 4 from the spec (distinguishing "normal HOLD" from "safety HOLD") is
intentionally not added: NavigationDecision has no such distinction --
every HOLD (kind="HOLD") is handled by the exact same generic branch in
apply_navigation_decision(), regardless of why ExplorationBehavior/
RobotAgent produced it ("frontier reached", "exploration exhausted",
"predicted collision" normalized by NavigationSupervisor, etc.). Per the
task's own guidance, this file documents that ALL HOLD after route
invalidation is a hard stop, rather than adding a test for a distinction
that does not exist in the architecture.

These tests exercise engine.apply_navigation_decision() via a minimal
duck-typed engine fake wrapping a REAL Robot instance (robot.py) -- a real
instance is used (not a further fake) because tests 3 and 5 need the
actual velocity/dynamics-integration behavior, not just a recorded call.
"""
from __future__ import annotations

from types import SimpleNamespace

from robot import Robot
from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.telemetry import TelemetryLogger


ACTIVE_TARGET = (4.25, -3.75)


def _build_fake_engine(*, position=(5.60, -4.13), v=0.47, goal_tolerance=0.25) -> SimpleNamespace:
    robot = Robot(x=position[0], y=position[1], v=v, max_speed=1.2, max_acceleration=2.0)
    agent = RobotAgent(robot_id=0, position=position, planner_mode="FoV-aware directional frontier")

    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        agent=agent,
        config=SimpleNamespace(
            planner_type="A*",
            path_simplifier="Direction changes",
            exploration_planner="FoV-aware directional frontier",
            goal_tolerance=goal_tolerance,
            grid_resolution=0.5,
            exploration_replan_cooldown=1.0,
        ),
        mapped_obstacle_points=[],
        current_exploration_target=None,
        simulation_time=30.0,
        console_logs=[],
        exploration_targets=[],
        replan_calls=[],
    )
    fake.telemetry = TelemetryLogger(sink=fake.console_logs.append)
    fake.log_console_message = lambda message, **kwargs: fake.console_logs.append(message)
    fake.canvas = SimpleNamespace(
        set_planned_path=lambda path: None,
        set_exploration_target=lambda target: fake.exploration_targets.append(target),
        set_status=lambda message: None,
        set_last_control=lambda control: None,
    )
    fake.is_exploration_mode = lambda: True
    fake.runtime_agent = lambda robot_index=None: fake.agent
    fake.set_robot_goal_or_waypoints = SimulationControllerMixin.set_robot_goal_or_waypoints.__get__(fake)
    fake.safety_replan_cooldown_seconds = lambda: 0.5
    fake.replan_after_new_information = lambda reason: fake.replan_calls.append(reason) or False

    fake.apply_navigation_decision = SimulationControllerMixin.apply_navigation_decision.__get__(fake)
    fake._invalidate_prefetch_request = SimulationControllerMixin._invalidate_prefetch_request.__get__(fake)
    return fake


def _give_agent_active_route(fake, *, target=ACTIVE_TARGET):
    fake.agent.assign_path(target=target, waypoints=[(5.0, -4.0), target], planner_reason="initial route")


def _replan_for_safety_decision(target, reason="predicted collision"):
    return SimpleNamespace(kind="REPLAN_FOR_SAFETY", reason=reason, target=target, brake=True)


# ---------------------------------------------------------------------------
# 1. Repeated safety replan (cooldown-blocked) must force velocity to zero.
# ---------------------------------------------------------------------------


def test_repeated_safety_replan_forces_robot_velocity_to_zero():
    fake = _build_fake_engine(v=0.47)
    _give_agent_active_route(fake)

    # First REPLAN_FOR_SAFETY: allowed, consumes the cooldown window.
    fake.apply_navigation_decision(fake.robot, fake.agent, _replan_for_safety_decision(ACTIVE_TARGET))
    assert fake.replan_calls == ["safety replan: predicted collision"]

    # Second REPLAN_FOR_SAFETY, same tick's target/reason, within cooldown:
    # blocked -> the repeated-safety-replan branch fires.
    fake.apply_navigation_decision(fake.robot, fake.agent, _replan_for_safety_decision(ACTIVE_TARGET))

    assert fake.robot.v == 0.0, "residual velocity must not survive a repeated-safety-replan hold"
    assert fake.agent.active_path_goal_xy is None
    assert fake.agent.active_target() is None


# ---------------------------------------------------------------------------
# 2. predicted_collision with no active route normalizes to HOLD, which
#    must also hard-stop the robot.
# ---------------------------------------------------------------------------


def test_predicted_collision_hold_without_active_route_hard_stops_robot():
    fake = _build_fake_engine(v=0.6)
    assert fake.agent.active_path_goal_xy is None
    assert fake.agent.active_target() is None

    decision = _replan_for_safety_decision(None)
    should_brake = fake.apply_navigation_decision(fake.robot, fake.agent, decision)

    assert should_brake is False  # normalized to HOLD, not a real safety replan
    assert fake.robot.v == 0.0, "a route-less predicted-collision HOLD must still zero velocity"


# ---------------------------------------------------------------------------
# 3. The hard stop happens synchronously inside apply_navigation_decision(),
#    before any subsequent robot.update() call -- not deferred.
# ---------------------------------------------------------------------------


def test_safety_route_failure_hard_stops_before_next_simulation_step():
    fake = _build_fake_engine(v=0.47)
    _give_agent_active_route(fake)
    fake.apply_navigation_decision(fake.robot, fake.agent, _replan_for_safety_decision(ACTIVE_TARGET))

    fake.apply_navigation_decision(fake.robot, fake.agent, _replan_for_safety_decision(ACTIVE_TARGET))

    # Velocity must already be zero right here -- before simulation_step()
    # would go on to call robot.update(control, dt) for this same tick.
    assert fake.robot.v == 0.0


# ---------------------------------------------------------------------------
# 5. With velocity hard-stopped, a subsequent physics update must not move
#    the robot further -- no coasting into a collision purely from
#    residual velocity.
# ---------------------------------------------------------------------------


def test_collision_warning_does_not_advance_robot_after_safety_hold():
    fake = _build_fake_engine(position=(5.60, -4.13), v=0.47)
    _give_agent_active_route(fake)
    fake.apply_navigation_decision(fake.robot, fake.agent, _replan_for_safety_decision(ACTIVE_TARGET))

    fake.apply_navigation_decision(fake.robot, fake.agent, _replan_for_safety_decision(ACTIVE_TARGET))
    assert fake.robot.v == 0.0

    x_before, y_before = fake.robot.x, fake.robot.y
    # Simulate the next physics integration tick with whatever control the
    # (now-idle) controller would produce -- even a nonzero-looking control
    # cannot move the robot this tick since v starts at 0 and dt is small.
    fake.robot.update(fake.robot.brake_control(), dt=0.1)

    assert fake.robot.x == x_before
    assert fake.robot.y == y_before
    assert fake.robot.v == 0.0
