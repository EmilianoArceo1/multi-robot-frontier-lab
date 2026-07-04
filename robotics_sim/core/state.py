from dataclasses import dataclass

import numpy as np


@dataclass
class RobotState:
    """
    Minimal physical state of the robot.

    Abstraction:
        This class represents only the instantaneous dynamic configuration
        of the robot.

            X = [x, y, theta, v]^T

        where:
            x, y   : position in the 2D world
            theta  : robot heading in radians
            v      : current linear velocity

    Responsibility:
        Store physical state. It does not decide goals, does not control,
        does not plan, and does not know about obstacles.
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0

    def as_column_vector(self) -> np.ndarray:
        """
        Return the state as a column vector.

        This preserves compatibility with code that expects:

            [x, y, theta, v]^T
        """
        return np.array(
            [[self.x], [self.y], [self.theta], [self.v]],
            dtype=float,
        )

    def set_from_column_vector(self, vector) -> None:
        """
        Update the state from a vector representation.

        Contract:
            The vector must contain exactly four values:
            x, y, theta, and v.
        """
        vector = np.asarray(vector, dtype=float).reshape(-1)

        if vector.size != 4:
            raise ValueError("RobotState requires 4 values: x, y, theta, v.")

        self.x = float(vector[0])
        self.y = float(vector[1])
        self.theta = float(vector[2])
        self.v = float(vector[3])

    @property
    def position(self) -> tuple[float, float]:
        """
        Cartesian robot position as a 2D point.
        """
        return self.x, self.y