from math import pi
from typing import TYPE_CHECKING

import numpy as np

from robotics_sim.control.modes import RobotMode
from robotics_sim.core.geometry import goal_metrics

if TYPE_CHECKING:
    from robotics_sim.diagnostics.capture import NavigationDebugCapture


class TrackingController:
    """
    Nominal controller for tracking a local target.

    Abstraction:
        Converts the intent:

            "move toward this waypoint"

        into control:

            u = [a, omega]^T

    It is nominal because it does not guarantee obstacle safety.
    Later, a safety module or CBF can correct this nominal control.
    """

    def __init__(
        self,
        acceleration_gain: float = 1.0,
        angular_gain: float = 2.0,
        speed_gain: float = 0.5,
    ):
        """
        Design gains.

        acceleration_gain:
            How aggressively the robot tries to reach the desired speed.

        angular_gain:
            How strongly the robot corrects angular error.

        speed_gain:
            Converts remaining distance into desired speed.
        """
        self.acceleration_gain = float(acceleration_gain)
        self.angular_gain = float(angular_gain)
        self.speed_gain = float(speed_gain)

    def brake_control(self, state) -> np.ndarray:
        """
        Control used to brake.

        Braking is not the same as sending zero control. If v > 0 and a = 0,
        the robot keeps moving.
        """
        acceleration = self.acceleration_gain * (0.0 - state.v)
        angular_velocity = 0.0

        return np.array(
            [[acceleration], [angular_velocity]],
            dtype=float,
        )

    def rotate_control(self, state, error_theta: float) -> np.ndarray:
        """
        Control used to align with the waypoint without moving forward.
        """
        acceleration = self.acceleration_gain * (0.0 - state.v)
        angular_velocity = self.angular_gain * error_theta

        return np.array(
            [[acceleration], [angular_velocity]],
            dtype=float,
        )

    def track_control(
        self,
        state,
        limits,
        distance_to_goal: float,
        error_theta: float,
    ) -> np.ndarray:
        """
        Control used to move toward the active waypoint.

        Desired speed is reduced when:
            - the robot is close to the target
            - angular error is large
            - max_speed would be exceeded
        """
        angular_velocity = self.angular_gain * error_theta

        effective_distance = max(
            distance_to_goal - limits.goal_tolerance,
            0.0,
        )

        if abs(error_theta) > pi / 2:
            desired_speed = 0.0
        else:
            proposed_speed = (
                self.speed_gain
                * effective_distance
                * np.cos(error_theta)
            )

            desired_speed = min(
                proposed_speed,
                limits.max_speed,
            )

        acceleration = self.acceleration_gain * (
            desired_speed - state.v
        )

        return np.array(
            [[acceleration], [angular_velocity]],
            dtype=float,
        )

    def clip_control(self, control, limits) -> np.ndarray:
        """
        Saturate control according to physical limits.
        """
        control = np.asarray(control, dtype=float).reshape(-1)

        if control.size != 2:
            raise ValueError("Control must have two components: [a, omega].")

        acceleration = float(
            np.clip(
                control[0],
                -limits.max_acceleration,
                limits.max_acceleration,
            )
        )

        angular_velocity = float(
            np.clip(
                control[1],
                -limits.max_angular_speed,
                limits.max_angular_speed,
            )
        )

        return np.array(
            [[acceleration], [angular_velocity]],
            dtype=float,
        )

    def compute_control(
        self,
        state,
        target,
        limits,
        mode: RobotMode,
        capture: "NavigationDebugCapture | None" = None,
    ) -> np.ndarray:
        """
        Compute nominal control according to the current mode.

        Contract:
            - no target: brake
            - ROTATE: rotate and brake
            - TRACK: move toward target
            - STOP/BLOCKED/FAILED: brake

        capture: optional diagnostic sink. When provided and a target
        exists, stashes the heading_error/distance_to_goal this method
        already computes via goal_metrics() before mode dispatch discards
        the rest of the dict. None (the default) costs nothing extra.
        """
        metrics = goal_metrics(state, target)

        if metrics is None:
            return self.clip_control(
                self.brake_control(state),
                limits,
            )

        distance_to_goal = metrics["distance_to_goal"]
        error_theta = metrics["error_theta"]
        desired_angle = metrics["desired_angle"]

        if capture is not None:
            capture.heading_error = float(error_theta)
            capture.distance_to_goal = float(distance_to_goal)
            capture.desired_heading = float(desired_angle)

        if mode == RobotMode.IDLE:
            control = self.brake_control(state)

        elif mode == RobotMode.ROTATE:
            control = self.rotate_control(state, error_theta)

        elif mode == RobotMode.TRACK:
            control = self.track_control(
                state,
                limits,
                distance_to_goal,
                error_theta,
            )

        elif mode == RobotMode.STOP:
            control = self.brake_control(state)

        elif mode == RobotMode.BLOCKED:
            control = self.brake_control(state)

        elif mode == RobotMode.FAILED:
            control = self.brake_control(state)

        else:
            raise RuntimeError(f"Unknown robot mode: {mode}")

        applied = self.clip_control(control, limits)
        if capture is not None:
            capture.nominal_control = (float(control[0, 0]), float(control[1, 0]))
            capture.applied_control = (float(applied[0, 0]), float(applied[1, 0]))
        return applied