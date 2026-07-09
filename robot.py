import numpy as np

from robotics_sim.control.modes import RobotMode
from robotics_sim.control.state_machine import TrackingStateMachine
from robotics_sim.control.tracking_controller import TrackingController
from robotics_sim.core.geometry import goal_metrics, parse_point, wrap_angle
from robotics_sim.core.limits import RobotLimits
from robotics_sim.core.state import RobotState
from robotics_sim.models.dynamic_unicycle import DynamicUnicycle2D
from robotics_sim.planning.waypoint_manager import WaypointManager


class Robot:
    """
    Compatibility facade for the current simulator.

    Abstraction:
        Robot coordinates smaller modules:

            RobotState              physical state
            RobotLimits             limits and tolerances
            DynamicUnicycle2D       dynamic evolution
            TrackingStateMachine    logical mode
            TrackingController      nominal control
            WaypointManager         local route

    Design note:
        The global planner, obstacles, maps, and A* should NOT live here.
        This class connects the existing GUI to the modular architecture.
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        v: float = 0.0,
        vision: float = 50.0,
        goal=None,
        max_speed: float = 1.0,
        max_acceleration: float = 1.0,
        max_angular_speed: float = 1.0,
        goal_tolerance: float = 0.05,
        stop_speed_tolerance: float = 0.01,
        robot_radius: float = 0.20,
    ):
        self.state = RobotState(
            x=float(x),
            y=float(y),
            theta=float(theta),
            v=float(v),
        )

        self.limits = RobotLimits(
            max_speed=float(max_speed),
            max_acceleration=float(max_acceleration),
            max_angular_speed=float(max_angular_speed),
            goal_tolerance=float(goal_tolerance),
            stop_speed_tolerance=float(stop_speed_tolerance),
            robot_radius=float(robot_radius),
        )

        self.vision = float(vision)

        self.dynamics = DynamicUnicycle2D()
        self.state_machine = TrackingStateMachine()
        self.controller = TrackingController()
        self.waypoints = WaypointManager()

        if goal is not None:
            self.set_goal(goal)

    @property
    def vector(self) -> np.ndarray:
        """
        State as a column vector for compatibility.
        """
        return self.state.as_column_vector()

    @vector.setter
    def vector(self, new_vector) -> None:
        self.state.set_from_column_vector(new_vector)

    @staticmethod
    def parse_goal(goal) -> np.ndarray:
        """
        Interpret a goal as a 2D point.
        """
        return parse_point(goal)

    @property
    def goal(self):
        """
        GUI-visible goal.

        Internal control uses active_waypoint().
        """
        return self.waypoints.display_goal()

    @goal.setter
    def goal(self, new_goal) -> None:
        if new_goal is None:
            self.clear_goal()
        else:
            self.set_goal(new_goal)

    def set_goal(self, goal) -> None:
        """
        Assign a single goal as a one-waypoint route.
        """
        if self.waypoints.same_goal_as(goal):
            return

        self.waypoints.set_goal(goal)
        self.state_machine.reset()

    def set_waypoints(self, waypoints) -> None:
        """
        Assign a route with multiple waypoints.
        """
        self.waypoints.set_waypoints(waypoints)
        self.state_machine.reset()

    def clear_goal(self) -> None:
        """
        Remove the active goal or route.
        """
        self.waypoints.clear()
        self.state_machine.reset()

    def active_waypoint(self):
        """
        Waypoint currently tracked by the controller.
        """
        return self.waypoints.active_waypoint()

    def advance_waypoint_if_needed(self) -> bool:
        """
        Advance to the next waypoint if the active one has been reached.
        """
        advanced = self.waypoints.advance_if_reached(
            position=(self.x, self.y),
            tolerance=self.limits.goal_tolerance,
        )

        if advanced and self.active_waypoint() is not None:
            self.state_machine.reset()

        return advanced

    def displacement(self, gx: float, gy: float) -> np.ndarray:
        """
        Vector from the robot to a given position.
        """
        return np.array(
            [
                [gx - self.x],
                [gy - self.y],
            ],
            dtype=float,
        )

    def goal_metrics(self):
        """
        Geometric metrics relative to the active waypoint.
        """
        return goal_metrics(
            self.state,
            self.active_waypoint(),
        )

    @property
    def mode(self) -> RobotMode:
        return self.state_machine.mode

    @mode.setter
    def mode(self, new_mode: RobotMode) -> None:
        self.state_machine.set_mode(new_mode)

    @property
    def previous_mode(self) -> RobotMode:
        return self.state_machine.previous_mode

    def set_mode(self, new_mode: RobotMode) -> None:
        self.state_machine.set_mode(new_mode)

    def update_state_machine(
        self,
        blocked: bool = False,
        failed: bool = False,
    ) -> None:
        """
        Update the robot logical mode.

        blocked and failed are external signals. Obstacle detection and planning
        do not live inside Robot.
        """
        self.advance_waypoint_if_needed()

        target = self.active_waypoint()

        if target is None:
            if self.waypoints.is_finished():
                self.state_machine.set_mode(RobotMode.STOP)
            else:
                self.state_machine.set_mode(RobotMode.IDLE)
            return

        metrics = self.goal_metrics()

        self.state_machine.update(
            has_target=True,
            distance_to_target=metrics["distance_to_goal"],
            error_theta=metrics["error_theta"],
            goal_tolerance=self.limits.goal_tolerance,
            blocked=blocked,
            failed=failed,
        )

    def brake_control(self) -> np.ndarray:
        return self.controller.brake_control(self.state)

    def rotate_control(self, error_theta: float) -> np.ndarray:
        return self.controller.rotate_control(
            self.state,
            error_theta,
        )

    def track_control(
        self,
        distance_to_goal: float,
        error_theta: float,
    ) -> np.ndarray:
        return self.controller.track_control(
            self.state,
            self.limits,
            distance_to_goal,
            error_theta,
        )

    def clip_control(self, control) -> np.ndarray:
        return self.controller.clip_control(
            control,
            self.limits,
        )

    def nominal_control(
        self,
        goal=None,
        blocked: bool = False,
        failed: bool = False,
    ) -> np.ndarray:
        """
        Compute nominal control.

        Flow:
            1. update goal if a new one was received
            2. update logical mode
            3. get active waypoint
            4. compute nominal control
        """
        if goal is not None:
            self.set_goal(goal)

        self.update_state_machine(
            blocked=blocked,
            failed=failed,
        )

        target = self.active_waypoint()

        control = self.controller.compute_control(
            state=self.state,
            target=target,
            limits=self.limits,
            mode=self.mode,
        )

        return self.clip_control(control)

    def update(self, control, dt: float) -> None:
        """
        Apply one dynamic step.

        This function does not decide the control; it only applies an already
        computed control.
        """
        self.dynamics.step(
            state=self.state,
            control=control,
            limits=self.limits,
            dt=dt,
        )

        self.advance_waypoint_if_needed()

    def force_stop(self, reason: str = "") -> None:
        """
        Immediately zero velocity, bypassing the gradual deceleration model.

        brake_control() only decelerates gradually: acceleration = -v
        (clamped to max_acceleration), and DynamicUnicycle2D.step() advances
        POSITION using the velocity from BEFORE this tick's acceleration is
        applied (x_{k+1} = x_k + v_k*cos(theta_k)*dt, THEN v_{k+1} = v_k +
        a_k*dt). So even a textbook brake control still lets the robot
        travel v_k*dt further on the very tick braking starts, and can take
        multiple ticks to fully stop -- exactly how a robot already
        committed to a safety HOLD can still coast into a collision.

        This exists for safety-critical situations where residual motion
        itself is the hazard (predicted collision, repeated safety replan,
        route invalidated for safety): it sets velocity to zero directly,
        then resets the controller's state machine so the next control
        computation starts from a clean STOP/IDLE mode rather than a stale
        TRACK/ROTATE one. reason is kept for caller-side logging/debugging;
        this method does not log anything itself.
        """
        self.state.v = 0.0
        self.state_machine.reset()

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return wrap_angle(angle)

    @property
    def mode_name(self) -> str:
        return self.mode.value

    @property
    def x(self) -> float:
        return self.state.x

    @x.setter
    def x(self, new_value: float) -> None:
        self.state.x = float(new_value)

    @property
    def y(self) -> float:
        return self.state.y

    @y.setter
    def y(self, new_value: float) -> None:
        self.state.y = float(new_value)

    @property
    def theta(self) -> float:
        return self.state.theta

    @theta.setter
    def theta(self, new_value: float) -> None:
        self.state.theta = float(new_value)

    @property
    def v(self) -> float:
        return self.state.v

    @v.setter
    def v(self, new_value: float) -> None:
        self.state.v = float(new_value)

    @property
    def max_speed(self) -> float:
        return self.limits.max_speed

    @max_speed.setter
    def max_speed(self, new_value: float) -> None:
        self.limits.max_speed = float(new_value)

    @property
    def max_acceleration(self) -> float:
        return self.limits.max_acceleration

    @max_acceleration.setter
    def max_acceleration(self, new_value: float) -> None:
        self.limits.max_acceleration = float(new_value)

    @property
    def max_angular_speed(self) -> float:
        return self.limits.max_angular_speed

    @max_angular_speed.setter
    def max_angular_speed(self, new_value: float) -> None:
        self.limits.max_angular_speed = float(new_value)

    @property
    def goal_tolerance(self) -> float:
        return self.limits.goal_tolerance

    @goal_tolerance.setter
    def goal_tolerance(self, new_value: float) -> None:
        self.limits.goal_tolerance = float(new_value)

    @property
    def stop_speed_tolerance(self) -> float:
        return self.limits.stop_speed_tolerance

    @stop_speed_tolerance.setter
    def stop_speed_tolerance(self, new_value: float) -> None:
        self.limits.stop_speed_tolerance = float(new_value)