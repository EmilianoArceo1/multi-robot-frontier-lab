import numpy as np

# Single source of truth for the numeric field-of-view angle behind each
# vision_model string. Sensor ray-casting (robotics_sim/simulation/config.py)
# and exploration information-gain estimation
# (robotics_sim/planning/exploration_planners.py) must both derive their FoV
# from this function so they never silently diverge.
CAMERA_FOV_ANGLE_RAD = float(np.radians(70.0))
OMNIDIRECTIONAL_FOV_ANGLE_RAD = float(2.0 * np.pi)


def sensor_fov_angle_radians(vision_model: str) -> float:
    """
    Translate a vision_model label into its numeric field-of-view angle.

    Contract:
        "Camera / FoV" (or any label containing "Camera") -> radians(70).
        Any other current model, including "LiDAR" and "Omnidirectional"
        -> 2*pi (treated as a 360-degree sensor in this 2D baseline).
    """
    if "Camera" in str(vision_model):
        return CAMERA_FOV_ANGLE_RAD

    return OMNIDIRECTIONAL_FOV_ANGLE_RAD


def wrap_angle(angle: float) -> float:
    """
    Normalize an angle to the interval [-pi, pi).

    Abstraction:
        Orientation is circular. Without normalization, small angular errors can
        look numerically huge near pi and -pi.
    """
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def distance(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """
    Euclidean distance between two 2D points.

    This is the basic spatial metric used for waypoint arrival,
    obstacle proximity, and candidate comparison in planning.
    """
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def displacement(
    position: tuple[float, float],
    target: tuple[float, float],
) -> np.ndarray:
    """
    Displacement vector from a position to a target.

    This is not an absolute position. It represents both direction
    and magnitude of the required spatial change.
    """
    return np.array(
        [
            [target[0] - position[0]],
            [target[1] - position[1]],
        ],
        dtype=float,
    )


def desired_angle_to_point(
    position: tuple[float, float],
    target: tuple[float, float],
    fallback_angle: float = 0.0,
) -> float:
    """
    Desired heading angle from a position to a target.

    If position and target coincide, there is no unique geometric direction.
    In that degenerate case, fallback_angle is preserved.
    """
    dx = target[0] - position[0]
    dy = target[1] - position[1]

    if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
        return float(fallback_angle)

    return float(np.arctan2(dy, dx))


def angle_error(
    current_angle: float,
    desired_angle: float,
) -> float:
    """
    Normalized angular error.

    Answers:
        How much the robot should rotate, and in which direction,
        to align with the desired heading.
    """
    return wrap_angle(desired_angle - current_angle)


def parse_point(point) -> np.ndarray:
    """
    Interpret external input as a 2D point.

    Contract:
        Accepts lists, tuples, or arrays with at least two numeric values.
        Returns only the first two components.
    """
    point_array = np.asarray(point, dtype=float).reshape(-1)

    if point_array.size < 2:
        raise ValueError("The point must contain at least x and y.")

    return point_array[:2]


def goal_metrics(state, target):
    """
    Translate a spatial target into geometric signals useful for the FSM
    and the controller.

    Returns:
        - target position
        - displacement vector
        - distance to target
        - desired heading angle
        - angular error

    This function does not decide whether to move, rotate, or brake.
    """
    if target is None:
        return None

    target_position = parse_point(target)

    position = (state.x, state.y)
    target_tuple = (float(target_position[0]), float(target_position[1]))

    displacement_vector = displacement(position, target_tuple)
    distance_to_goal = float(np.linalg.norm(displacement_vector))

    desired_angle = desired_angle_to_point(
        position,
        target_tuple,
        fallback_angle=state.theta,
    )

    error_theta = angle_error(
        current_angle=state.theta,
        desired_angle=desired_angle,
    )

    return {
        "goal_position": target_position,
        "displacement_vector": displacement_vector,
        "distance_to_goal": distance_to_goal,
        "desired_angle": desired_angle,
        "error_theta": error_theta,
    }