import numpy as np

from robotics_sim.core.geometry import wrap_angle


class DynamicUnicycle2D:
    """
    Dynamic unicycle model with linear velocity as part of the state.

    State:
        X = [x, y, theta, v]^T

    Control:
        u = [a, omega]^T

    Abstraction:
        This model only answers:

            "If I apply this control for dt seconds,
             what is the new state?"

    It does not know about goals, obstacles, maps, waypoints, or planners.
    """

    def step(self, state, control, limits, dt: float) -> None:
        """
        Advance the state using explicit Euler integration.

        Model:
            x_{k+1}     = x_k + v_k cos(theta_k) dt
            y_{k+1}     = y_k + v_k sin(theta_k) dt
            theta_{k+1} = theta_k + omega_k dt
            v_{k+1}     = v_k + a_k dt

        Current restriction:
            Reverse motion is not allowed. Velocity is saturated in [0, max_speed].
        """
        if dt <= 0:
            raise ValueError("dt must be greater than zero.")

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

        x_k = state.x
        y_k = state.y
        theta_k = state.theta
        v_k = state.v

        state.x = x_k + v_k * np.cos(theta_k) * dt
        state.y = y_k + v_k * np.sin(theta_k) * dt
        state.theta = wrap_angle(theta_k + angular_velocity * dt)

        state.v = float(
            np.clip(
                v_k + acceleration * dt,
                0.0,
                limits.max_speed,
            )
        )