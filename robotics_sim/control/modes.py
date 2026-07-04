from enum import Enum


class RobotMode(Enum):
    """
    Logical states of the robot during local tracking.

    These modes describe operational intent, not physical state.

    IDLE:
        No active task.

    ROTATE:
        The robot should align itself before moving forward.

    TRACK:
        The robot can move toward the active waypoint.

    STOP:
        The robot should brake.

    BLOCKED:
        The local path is blocked.

    FAILED:
        A planning or execution failure occurred.
    """

    IDLE = "IDLE"
    ROTATE = "ROTATE"
    TRACK = "TRACK"
    STOP = "STOP"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"