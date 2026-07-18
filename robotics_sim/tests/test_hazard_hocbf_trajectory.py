"""Dynamic validation of the Observed Hazard OGM-HOCBF safety filter.

Simulates ``DynamicUnicycle2D`` directly, step by step, over many ticks --
no Qt, no full ``SimulationControllerMixin``/engine. Each tick:

    1. ``TrackingController.compute_control()`` (the real nominal
       controller, unmodified) produces a nominal control toward a fixed
       target.
    2. ``HazardSafetyRuntime.filter_control()`` (the real runtime adapter,
       unmodified) filters it against a real ``HazardBelief`` snapshot.
    3. ``DynamicUnicycle2D.step()`` (the real, unmodified dynamics) advances
       the state with the filtered control.

These tests assert INVARIANTS and METRICS (control stays within limits,
minimum_h stays above a discretized tolerance, feasibility, omega/accel
usage, cache reuse counts) -- never an exact trajectory, which would be
brittle and would not actually prove anything about the safety filter.
"""
from __future__ import annotations

import numpy as np

from robotics_sim.control.hazard_hocbf_filter import HazardHOCBFResult
from robotics_sim.control.modes import RobotMode
from robotics_sim.control.tracking_controller import TrackingController
from robotics_sim.core.limits import RobotLimits
from robotics_sim.core.state import RobotState
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.models.dynamic_unicycle import DynamicUnicycle2D
from robotics_sim.simulation.hazard_safety_runtime import HazardSafetyRuntime

_BOUNDS = (0.0, 12.0, 0.0, 12.0)
_RESOLUTION = 0.5
_THRESHOLD = 0.55
_DT = 0.05
_SAFETY_RADIUS = 0.35


def _geometry() -> GridGeometry:
    return GridGeometry(_BOUNDS, _RESOLUTION)


def _mark_blob(belief: HazardBelief, geometry: GridGeometry, center, *, half_extent: float = 0.75, value: float = 0.9) -> None:
    """Observe every cell in a square footprint around `center` as hazardous."""
    center_x, center_y = center
    rows: list[int] = []
    cols: list[int] = []
    for dy in np.arange(-half_extent, half_extent + 1e-9, geometry.resolution):
        for dx in np.arange(-half_extent, half_extent + 1e-9, geometry.resolution):
            cell = geometry.world_to_grid(center_x + dx, center_y + dy)
            if cell is not None:
                rows.append(cell.row)
                cols.append(cell.col)
    belief.observe_cells(rows=rows, cols=cols, values=[value] * len(rows), robot_index=0)


def _runtime(**overrides) -> HazardSafetyRuntime:
    kwargs = dict(
        block_threshold=_THRESHOLD,
        margin=0.20,
        activation_distance=3.0,
        k1=2.0,
        k2=2.0,
        pyramid_levels=1,
        smoothing_sigma_cells=0.75,
        acceleration_weight=1.0,
        angular_weight=0.35,
    )
    kwargs.update(overrides)
    return HazardSafetyRuntime(**kwargs)


def _limits(**overrides) -> RobotLimits:
    kwargs = dict(max_speed=2.0, max_acceleration=2.0, max_angular_speed=2.5, robot_radius=0.20)
    kwargs.update(overrides)
    return RobotLimits(**kwargs)


def _assert_within_limits(result: HazardHOCBFResult, limits: RobotLimits) -> None:
    a = float(result.control[0, 0])
    omega = float(result.control[1, 0])
    tolerance = 1e-9
    assert -limits.max_acceleration - tolerance <= a <= limits.max_acceleration + tolerance
    assert -limits.max_angular_speed - tolerance <= omega <= limits.max_angular_speed + tolerance


def _run_trajectory(
    *,
    runtime: HazardSafetyRuntime,
    belief_frame,
    geometry: GridGeometry,
    state: RobotState,
    limits: RobotLimits,
    target: tuple[float, float],
    steps: int,
    dt: float = _DT,
    safety_radius: float = _SAFETY_RADIUS,
):
    controller = TrackingController()
    dynamics = DynamicUnicycle2D()
    results: list[HazardHOCBFResult] = []
    positions: list[tuple[float, float]] = []

    for _ in range(steps):
        nominal = controller.compute_control(state, target, limits, mode=RobotMode.TRACK)
        result = runtime.filter_control(
            belief_frame=belief_frame,
            geometry=geometry,
            state=state,
            limits=limits,
            nominal_control=nominal,
            safety_radius=safety_radius,
        )
        positions.append((state.x, state.y))
        dynamics.step(state, result.control, limits, dt)
        results.append(result)
        _assert_within_limits(result, limits)

    return results, positions


# ---------------------------------------------------------------------------
# 1. Unobserved hazard: robot advances without intervention; filtered
#    control is identical to nominal at every step.
# ---------------------------------------------------------------------------


def test_unobserved_hazard_never_intervenes():
    geometry = _geometry()
    belief = HazardBelief(geometry, robot_count=1)  # nothing ever observed

    runtime = _runtime()
    limits = _limits()
    state = RobotState(x=1.0, y=6.0, theta=0.0, v=0.0)
    target = (11.0, 6.0)

    results, positions = _run_trajectory(
        runtime=runtime, belief_frame=belief.snapshot(), geometry=geometry,
        state=state, limits=limits, target=target, steps=80,
    )

    assert all(not result.active for result in results)
    assert all(result.feasible for result in results)
    # Sanity: the robot actually moved toward the target under its nominal control.
    assert positions[-1][0] > positions[0][0] + 1.0


# ---------------------------------------------------------------------------
# 2. Hazard discovered with enough lead time: filter activates, the robot
#    never enters the unsafe set, minimum_h never drops below a discretized
#    tolerance, and the control stays within limits throughout.
#
# 4. Same trajectory doubles as the frontal-approach measurement: does the
#    robot brake, turn, or both, when the hazard is dead ahead?
# ---------------------------------------------------------------------------


def test_hazard_discovered_with_lead_time_avoids_unsafe_set_and_frontal_approach_brakes():
    geometry = _geometry()
    belief = HazardBelief(geometry, robot_count=1)
    _mark_blob(belief, geometry, center=(6.0, 6.0))

    runtime = _runtime()
    limits = _limits()
    state = RobotState(x=1.0, y=6.0, theta=0.0, v=0.0)  # heading straight at the blob
    target = (11.0, 6.0)

    results, positions = _run_trajectory(
        runtime=runtime, belief_frame=belief.snapshot(), geometry=geometry,
        state=state, limits=limits, target=target, steps=200,
    )

    assert any(result.active for result in results), "filter must engage as the robot nears the hazard"
    assert all(result.feasible for result in results)

    frame = runtime.field_frame
    assert frame is not None and frame.has_hazards
    discretized_tolerance = 0.5 * geometry.resolution
    min_phi_visited = min(frame.sample(x, y)[0].value for x, y in positions)
    assert min_phi_visited >= -discretized_tolerance, (
        "robot must never enter the unsafe set (phi < 0) with lead-time discovery",
        min_phi_visited,
    )

    finite_minimum_hs = [result.minimum_h for result in results if result.minimum_h is not None]
    assert finite_minimum_hs
    assert min(finite_minimum_hs) >= -discretized_tolerance

    # Frontal approach (hazard dead ahead, no lateral offset): the correction
    # mechanism used is deceleration, not turning.
    accelerations = [float(r.control[0, 0]) for r in results]
    omegas = [float(r.control[1, 0]) for r in results]
    assert min(accelerations) < -0.5, "expected the filter to brake for a head-on approach"
    assert max(abs(o) for o in omegas) < 1e-6, "a purely frontal approach should not need to turn"


# ---------------------------------------------------------------------------
# 3. Hazard discovered too late: initial_condition_valid=False, no
#    forward-invariance guarantee is asserted, recovery/braking is observed,
#    and the minimum distance reached is reported.
# ---------------------------------------------------------------------------


def test_hazard_discovered_too_late_reports_invalid_initial_condition_and_recovery():
    geometry = _geometry()
    belief = HazardBelief(geometry, robot_count=1)
    _mark_blob(belief, geometry, center=(6.0, 6.0))

    runtime = _runtime()
    limits = _limits(max_speed=3.0)
    # Starts already inside the hazard's footprint at high speed -- the
    # hazard is "discovered" (belief observed) only at this very late point.
    state = RobotState(x=5.3, y=6.0, theta=0.0, v=3.0)
    target = (11.0, 6.0)

    results, positions = _run_trajectory(
        runtime=runtime, belief_frame=belief.snapshot(), geometry=geometry,
        state=state, limits=limits, target=target, steps=60,
    )

    first_result = results[0]
    assert first_result.initial_condition_valid is False
    assert first_result.minimum_h is not None and first_result.minimum_h < 0.0

    # No forward-invariance guarantee is claimed: the QP is allowed to be
    # infeasible here, and the test does not require phi to stay >= 0.
    frame = runtime.field_frame
    assert frame is not None
    minimum_phi_reached = min(frame.sample(x, y)[0].value for x, y in positions)
    # Reported, not asserted to any particular bound -- this is exactly the
    # "no guarantee" case; only that it is a finite, sane number.
    assert np.isfinite(minimum_phi_reached)

    # Recovery is attempted: the filter must still try to brake hard rather
    # than doing nothing (e.g. passing the unsafe nominal control through).
    first_accelerations = [float(r.control[0, 0]) for r in results[:3]]
    assert min(first_accelerations) < -1.0, "expected the filter to attempt hard braking immediately"


# ---------------------------------------------------------------------------
# 5. Diagonal approach: omega must contribute meaningfully when useful.
# ---------------------------------------------------------------------------


def test_diagonal_approach_uses_omega():
    geometry = _geometry()
    belief = HazardBelief(geometry, robot_count=1)
    _mark_blob(belief, geometry, center=(7.0, 7.0))

    runtime = _runtime()
    limits = _limits()
    state = RobotState(x=4.5, y=5.0, theta=np.radians(35.0), v=0.0)
    target = (11.0, 8.0)

    results, _positions = _run_trajectory(
        runtime=runtime, belief_frame=belief.snapshot(), geometry=geometry,
        state=state, limits=limits, target=target, steps=200,
    )

    assert any(result.active for result in results)
    omegas = [float(r.control[1, 0]) for r in results]
    assert max(abs(o) for o in omegas) > 0.5, "a diagonal approach should let omega meaningfully participate"


# ---------------------------------------------------------------------------
# 6. Corridor between two hazardous regions: the QP must remain feasible,
#    and the robot must not oscillate or deadlock.
# ---------------------------------------------------------------------------


def test_corridor_between_two_hazards_stays_feasible_without_oscillation_or_deadlock():
    geometry = _geometry()
    belief = HazardBelief(geometry, robot_count=1)
    _mark_blob(belief, geometry, center=(6.0, 4.0))
    _mark_blob(belief, geometry, center=(6.0, 8.0))

    runtime = _runtime()
    limits = _limits()
    state = RobotState(x=1.0, y=6.0, theta=0.0, v=0.0)  # straight down the middle
    target = (11.0, 6.0)

    results, positions = _run_trajectory(
        runtime=runtime, belief_frame=belief.snapshot(), geometry=geometry,
        state=state, limits=limits, target=target, steps=250,
    )

    assert all(result.feasible for result in results), "a comfortably wide corridor must not force infeasibility"

    omegas = [float(r.control[1, 0]) for r in results]
    sign_changes = sum(
        1 for previous, current in zip(omegas, omegas[1:]) if previous * current < -1e-6
    )
    assert sign_changes <= 2, f"omega oscillated {sign_changes} times crossing the corridor"

    # No deadlock: the robot must have made real forward progress, not
    # stalled at v=0 in front of the corridor entrance.
    assert positions[-1][0] > positions[0][0] + 5.0
    assert abs(positions[-1][0] - target[0]) < 1.0


# ---------------------------------------------------------------------------
# 7. Four robots sharing the same team belief: the SDF is reused, and each
#    robot gets its own, independently filtered control.
# ---------------------------------------------------------------------------


def test_four_robots_share_belief_reuse_sdf_and_get_independent_controls():
    geometry = _geometry()
    belief = HazardBelief(geometry, robot_count=1)
    _mark_blob(belief, geometry, center=(6.0, 6.0))
    belief_frame = belief.snapshot()

    runtime = _runtime()
    limits = _limits()

    robot_states = [
        RobotState(x=4.5, y=6.0, theta=0.0, v=1.5),           # approaching head-on
        RobotState(x=6.0, y=4.5, theta=np.pi / 2.0, v=1.5),   # approaching from below
        RobotState(x=1.0, y=1.0, theta=0.0, v=1.0),           # far away, uninvolved
        RobotState(x=8.0, y=8.0, theta=np.pi, v=1.0),         # heading away
    ]
    nominal = np.array([[0.0], [0.0]])

    results = [
        runtime.filter_control(
            belief_frame=belief_frame,
            geometry=geometry,
            state=robot_state,
            limits=limits,
            nominal_control=nominal,
            safety_radius=_SAFETY_RADIUS,
        )
        for robot_state in robot_states
    ]

    # One shared build for the whole team belief, reused for the other three robots.
    assert runtime.field_rebuild_count == 1
    assert runtime.field_reuse_count == 3

    # Each robot's own state/position produced its own outcome -- not one
    # shared/aliased result object, and not all identical.
    controls = [tuple(result.control.ravel()) for result in results]
    assert len(set(controls)) > 1, "different robot states must not collapse to one shared control"
    assert results[2].active is False, "the far-away, uninvolved robot must not be affected"
