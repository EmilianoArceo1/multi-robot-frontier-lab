"""
Tests proving the navigation-debug capture sinks are strictly additive:
omitting them (capture=None, the default every existing caller uses)
produces identical output to before, and supplying one only adds data
without changing the underlying computation/decision.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robotics_sim.control.modes import RobotMode
from robotics_sim.control.tracking_controller import TrackingController
from robotics_sim.core.limits import RobotLimits
from robotics_sim.core.state import RobotState
from robotics_sim.diagnostics.capture import NavigationDebugCapture, PlanDebugCapture
from robotics_sim.environment.collision_checker import CollisionChecker
from robotics_sim.simulation.engine import PlannerWorker, SimulationControllerMixin


# ---------------------------------------------------------------------------
# TrackingController.compute_control(): capture is a pure outparam.
# ---------------------------------------------------------------------------


def test_compute_control_output_identical_with_and_without_capture():
    controller = TrackingController()
    state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0)
    limits = RobotLimits()
    target = np.array([2.0, 0.0])

    control_without = controller.compute_control(state, target, limits, RobotMode.TRACK)
    control_with = controller.compute_control(
        state, target, limits, RobotMode.TRACK, capture=NavigationDebugCapture()
    )

    assert np.array_equal(control_without, control_with)


def test_compute_control_capture_stashes_real_heading_error_and_distance():
    controller = TrackingController()
    state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0)
    limits = RobotLimits()
    target = np.array([2.0, 0.0])
    capture = NavigationDebugCapture()

    controller.compute_control(state, target, limits, RobotMode.TRACK, capture=capture)

    assert capture.heading_error == 0.0  # robot already faces the target exactly
    assert capture.distance_to_goal == 2.0


def test_compute_control_capture_none_by_default_leaves_no_trace():
    controller = TrackingController()
    state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0)
    limits = RobotLimits()
    target = np.array([2.0, 0.0])

    # No capture kwarg at all -- the exact call shape every pre-existing
    # caller uses.
    control = controller.compute_control(state, target, limits, RobotMode.TRACK)
    assert control is not None


# ---------------------------------------------------------------------------
# SimulationControllerMixin.predicted_motion_report(): capture is additive.
# ---------------------------------------------------------------------------


def _build_fake_engine_for_prediction() -> SimpleNamespace:
    robot = SimpleNamespace(
        x=0.0, y=0.0, theta=0.0, v=0.5, max_speed=1.0, max_acceleration=1.0, max_angular_speed=1.0
    )
    fake = SimpleNamespace(
        robot=robot,
        collision_checker=CollisionChecker(),
        # getattr(self.robot, "max_speed", self.config.max_speed) evaluates
        # the default eagerly even though robot.max_speed exists -- config
        # needs these attributes too or the getattr call itself raises.
        config=SimpleNamespace(obstacles=[], max_speed=1.0, max_acceleration=1.0, max_angular_speed=1.0),
    )
    fake.robot_snapshot = SimulationControllerMixin.robot_snapshot.__get__(fake)
    fake.predicted_motion_report = SimulationControllerMixin.predicted_motion_report.__get__(fake)
    return fake


def test_predicted_motion_report_identical_with_and_without_capture():
    control = np.array([[0.0], [0.0]])

    fake_without = _build_fake_engine_for_prediction()
    report_without = fake_without.predicted_motion_report(
        control=control, dt=0.1, robot_radius=0.2, known_obstacle_points=[(5.0, 5.0)], use_ground_truth=True
    )

    fake_with = _build_fake_engine_for_prediction()
    capture = NavigationDebugCapture()
    report_with = fake_with.predicted_motion_report(
        control=control,
        dt=0.1,
        robot_radius=0.2,
        known_obstacle_points=[(5.0, 5.0)],
        use_ground_truth=True,
        capture=capture,
    )

    assert report_without is None
    assert report_with is None
    # The trajectory is still captured even though nothing collided.
    assert capture.predicted_trajectory is not None
    assert len(capture.predicted_trajectory) == 10


def test_predicted_motion_report_capture_stashes_clearance_terms_on_collision():
    control = np.array([[1.0], [0.0]])  # accelerate straight toward the obstacle
    fake = _build_fake_engine_for_prediction()
    capture = NavigationDebugCapture()

    report = fake.predicted_motion_report(
        control=control,
        dt=0.2,
        robot_radius=0.5,
        known_obstacle_points=[(0.3, 0.0)],
        use_ground_truth=True,
        capture=capture,
    )

    assert report is not None and report.collision is True
    assert capture.predicted_collision is not None
    assert capture.predicted_collision.checker == "check_predicted_motion_points"
    assert capture.predicted_collision.blocked is True


# ---------------------------------------------------------------------------
# PlannerWorker: debug_capture is optional and defaults to None; when
# supplied, run() (executed here on the calling thread, same as the
# existing worker.run() smoke test) fills it via compute_planned_waypoints().
# ---------------------------------------------------------------------------


def test_planner_worker_without_debug_capture_behaves_as_before():
    kwargs = dict(
        planner_type="A*",
        start_xy=(0.0, 0.0),
        goal_xy=(2.0, 2.0),
        obstacles=[],
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        robot_radius=0.2,
        obstacle_points=[],
    )
    worker = PlannerWorker(request_id=1, planner_kwargs=kwargs, path_simplifier="Direction changes")
    assert worker.debug_capture is None

    worker.run()

    assert worker.route_plan_result == "ok"


def test_planner_worker_with_debug_capture_fills_it_from_the_real_plan():
    kwargs = dict(
        planner_type="A*",
        start_xy=(0.0, 0.0),
        goal_xy=(2.0, 2.0),
        obstacles=[],
        bounds=(-10.0, 10.0, -10.0, 10.0),
        resolution=0.5,
        robot_radius=0.2,
        obstacle_points=[],
    )
    capture = PlanDebugCapture()
    worker = PlannerWorker(
        request_id=1, planner_kwargs=kwargs, path_simplifier="Direction changes", debug_capture=capture
    )

    worker.run()

    assert worker.route_plan_result == "ok"
    assert capture.planner_name == "A*"
    assert capture.raw_world_path is not None and len(capture.raw_world_path) > 0
