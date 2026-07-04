from dataclasses import dataclass


@dataclass
class RobotLimits:
    """
    Physical limits and tolerances for the robot.

    Abstraction:
        These parameters are not behavior. They are constraints that other
        modules must respect.

    Includes:
        - linear speed, acceleration, and angular speed limits
        - tolerance for considering a waypoint reached
        - physical or safety radius of the robot
    """

    max_speed: float = 1.0
    max_acceleration: float = 1.0
    max_angular_speed: float = 1.0

    goal_tolerance: float = 0.05
    stop_speed_tolerance: float = 0.01

    robot_radius: float = 0.20

    def __post_init__(self) -> None:
        """
        Validate that the parameters are physically meaningful.
        """
        if self.max_speed <= 0:
            raise ValueError("max_speed must be greater than zero.")

        if self.max_acceleration <= 0:
            raise ValueError("max_acceleration must be greater than zero.")

        if self.max_angular_speed <= 0:
            raise ValueError("max_angular_speed must be greater than zero.")

        if self.goal_tolerance <= 0:
            raise ValueError("goal_tolerance must be greater than zero.")

        if self.stop_speed_tolerance < 0:
            raise ValueError("stop_speed_tolerance cannot be negative.")

        if self.robot_radius < 0:
            raise ValueError("robot_radius cannot be negative.")