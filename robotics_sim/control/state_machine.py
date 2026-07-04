import numpy as np

from robotics_sim.control.modes import RobotMode


class TrackingStateMachine:
    """
    State machine for local waypoint tracking.

    Abstraction:
        Decides the logical mode of the robot based on:
            - existence of a local target
            - distance to target
            - angular error
            - external blocked signal
            - external failed signal

    It does not compute controls.
    It does not update physics.
    It does not modify waypoints.
    """

    def __init__(self):
        self.mode = RobotMode.IDLE
        self.previous_mode = self.mode

        # Angular hysteresis:
        # Two thresholds prevent rapid ROTATE/TRACK switching when the angular
        # error is close to the boundary.
        self.rotate_to_track_threshold = np.deg2rad(10.0)
        self.track_to_rotate_threshold = np.deg2rad(20.0)

    def set_mode(self, new_mode: RobotMode) -> None:
        """
        Change mode while preserving the previous mode.
        """
        if new_mode != self.mode:
            self.previous_mode = self.mode
            self.mode = new_mode

    def reset(self) -> None:
        """
        Reset the FSM to IDLE.
        """
        self.set_mode(RobotMode.IDLE)

    def update(
        self,
        has_target: bool,
        distance_to_target: float | None,
        error_theta: float | None,
        goal_tolerance: float,
        blocked: bool = False,
        failed: bool = False,
    ) -> RobotMode:
        """
        Decide the logical mode of the robot.

        Priority:
            1. FAILED
            2. BLOCKED
            3. IDLE if there is no target
            4. STOP if the target has been reached
            5. ROTATE/TRACK depending on angular alignment
        """
        if failed:
            self.set_mode(RobotMode.FAILED)
            return self.mode

        if blocked:
            self.set_mode(RobotMode.BLOCKED)
            return self.mode

        if self.mode == RobotMode.BLOCKED and not blocked:
            self.set_mode(RobotMode.IDLE)

        if self.mode == RobotMode.FAILED and not failed:
            self.set_mode(RobotMode.IDLE)

        if not has_target:
            self.set_mode(RobotMode.IDLE)
            return self.mode

        if distance_to_target is None or error_theta is None:
            self.set_mode(RobotMode.IDLE)
            return self.mode

        abs_error = abs(error_theta)

        if distance_to_target <= goal_tolerance:
            self.set_mode(RobotMode.STOP)
            return self.mode

        if self.mode == RobotMode.STOP:
            self.set_mode(RobotMode.IDLE)

        if self.mode == RobotMode.IDLE:
            if abs_error <= self.rotate_to_track_threshold:
                self.set_mode(RobotMode.TRACK)
            else:
                self.set_mode(RobotMode.ROTATE)
            return self.mode

        if self.mode == RobotMode.ROTATE:
            if abs_error <= self.rotate_to_track_threshold:
                self.set_mode(RobotMode.TRACK)
            return self.mode

        if self.mode == RobotMode.TRACK:
            if abs_error >= self.track_to_rotate_threshold:
                self.set_mode(RobotMode.ROTATE)
            return self.mode

        self.set_mode(RobotMode.IDLE)
        return self.mode