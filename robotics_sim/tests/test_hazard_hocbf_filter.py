"""Unit tests for HazardHOCBFSafetyFilter (robotics_sim.control.hazard_hocbf_filter).

Exercises the filter directly against real HazardDistanceFieldFrame objects
(built by HazardDistanceFieldBuilder from a real HazardBelief) plus real
RobotState/RobotLimits -- no engine, no Qt, no QP solver dependency.
"""
from __future__ import annotations

import numpy as np

from robotics_sim.control.hazard_hocbf_filter import (
    HazardHOCBFSafetyFilter,
    constraint_margin,
    evaluate_hazard_constraint,
)
from robotics_sim.core.limits import RobotLimits
from robotics_sim.core.state import RobotState
from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.environment.hazard_distance_field import HazardDistanceFieldBuilder
from robotics_sim.simulation.config import (
    SimulationConfig,
    config_from_sim_payload,
    config_to_sim_payload,
)

_BOUNDS = (0.0, 10.0, 0.0, 10.0)
_RESOLUTION = 1.0
_THRESHOLD = 0.55

# Hazard cell (row=5, col=8) -> world center (8.5, 5.5).
_HAZARD_ROW = 5
_HAZARD_COL = 8


def _build_field_frame(*, pyramid_levels: int = 1):
    geometry = GridGeometry(_BOUNDS, _RESOLUTION)
    belief = HazardBelief(geometry, robot_count=1)
    belief.observe_cells(rows=[_HAZARD_ROW], cols=[_HAZARD_COL], values=[0.90], robot_index=0)

    return HazardDistanceFieldBuilder().build(
        belief_frame=belief.snapshot(),
        geometry=geometry,
        block_threshold=_THRESHOLD,
        pyramid_levels=pyramid_levels,
        smoothing_sigma_cells=0.75,
    )


def _filter(**kwargs) -> HazardHOCBFSafetyFilter:
    return HazardHOCBFSafetyFilter(**kwargs)


def _limits(max_acceleration=2.0, max_angular_speed=2.5) -> RobotLimits:
    return RobotLimits(max_acceleration=max_acceleration, max_angular_speed=max_angular_speed)


# ---------------------------------------------------------------------------
# 11. No active field -> control passes through unchanged.
# ---------------------------------------------------------------------------


def test_no_active_field_returns_identical_control():
    nominal = np.array([[0.3], [0.1]])
    result = _filter().filter(
        distance_field_frame=None,
        state=RobotState(x=5.5, y=5.5, theta=0.0, v=1.0),
        limits=_limits(),
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=1.5,
    )

    assert np.array_equal(result.control, nominal)
    assert result.active is False
    assert result.feasible is True


def test_no_hazards_in_belief_returns_identical_control():
    geometry = GridGeometry(_BOUNDS, _RESOLUTION)
    empty_frame = HazardDistanceFieldBuilder().build(
        belief_frame=HazardBelief(geometry).snapshot(),
        geometry=geometry,
        block_threshold=_THRESHOLD,
    )
    nominal = np.array([[0.3], [0.1]])

    result = _filter().filter(
        distance_field_frame=empty_frame,
        state=RobotState(x=5.5, y=5.5, theta=0.0, v=1.0),
        limits=_limits(),
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=1.5,
    )

    assert np.array_equal(result.control, nominal)
    assert result.active is False


# ---------------------------------------------------------------------------
# 12. Hazard outside activation distance -> control unchanged.
# ---------------------------------------------------------------------------


def test_hazard_outside_activation_distance_returns_identical_control():
    frame = _build_field_frame()
    nominal = np.array([[0.5], [0.0]])

    # Robot far from the hazard cell (8.5, 5.5); activation_distance is tiny.
    result = _filter().filter(
        distance_field_frame=frame,
        state=RobotState(x=0.5, y=0.5, theta=0.0, v=0.5),
        limits=_limits(),
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=0.5,
    )

    assert np.array_equal(result.control, nominal)
    assert result.active is False
    assert result.constraint_count == 0


# ---------------------------------------------------------------------------
# 13. Nominal already safe -> no intervention.
# ---------------------------------------------------------------------------


def test_nominal_already_safe_has_zero_intervention():
    frame = _build_field_frame()
    # Heading away from the hazard (theta=pi -> e=(-1,0)); braking nominal.
    nominal = np.array([[-0.2], [0.0]])

    result = _filter().filter(
        distance_field_frame=frame,
        state=RobotState(x=7.5, y=5.5, theta=np.pi, v=0.5),
        limits=_limits(),
        nominal_control=nominal,
        safety_radius=0.20,
        margin=0.10,
        activation_distance=2.0,
    )

    assert result.nominal_feasible is True
    assert result.intervention_norm == 0.0
    assert np.array_equal(result.control, nominal)


# ---------------------------------------------------------------------------
# 14. Robot heading toward the hazard -> intervention satisfies all constraints.
# ---------------------------------------------------------------------------


def test_approaching_hazard_creates_satisfied_constraints():
    frame = _build_field_frame()
    state = RobotState(x=5.5, y=5.5, theta=0.0, v=2.0)  # heading straight at (8.5, 5.5)
    limits = _limits()
    nominal = np.array([[0.0], [0.0]])  # coasting straight at the hazard

    result = _filter().filter(
        distance_field_frame=frame,
        state=state,
        limits=limits,
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=3.0,
    )

    assert result.constraint_count >= 1
    assert result.active is True

    samples = frame.sample(state.x, state.y)
    constraints = [
        evaluate_hazard_constraint(
            sample=sample, v=state.v, theta=state.theta, safety_radius=0.35, margin=0.20, k1=2.0, k2=2.0
        )
        for sample in samples
    ]
    for constraint in constraints:
        assert constraint_margin(constraint, result.control) >= -1e-6
    assert result.minimum_psi_2 is not None and result.minimum_psi_2 >= -1e-6

    assert result.control.shape == (2, 1)
    assert -limits.max_acceleration - 1e-9 <= float(result.control[0, 0]) <= limits.max_acceleration + 1e-9
    assert -limits.max_angular_speed - 1e-9 <= float(result.control[1, 0]) <= limits.max_angular_speed + 1e-9


# ---------------------------------------------------------------------------
# 15. Robot moving away -> no unnecessary intervention.
# ---------------------------------------------------------------------------


def test_moving_away_from_hazard_has_no_intervention():
    frame = _build_field_frame()
    nominal = np.array([[0.1], [0.0]])

    result = _filter().filter(
        distance_field_frame=frame,
        state=RobotState(x=5.5, y=5.5, theta=np.pi, v=2.0),  # heading away, at (8.5, 5.5)
        limits=_limits(),
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=3.0,
    )

    assert result.intervention_norm == 0.0
    assert np.array_equal(result.control, nominal)


# ---------------------------------------------------------------------------
# 16. Lateral approach -> omega may participate in the correction.
# ---------------------------------------------------------------------------


def test_lateral_approach_allows_omega_correction():
    frame = _build_field_frame()
    # Robot below-left of the hazard cell, heading at a 45-degree angle --
    # neither straight at it (pure-`a` correction, see test 14) nor purely
    # tangential at a safe distance (no correction at all, see test 15).
    # This off-axis approach gives the constraint's `A` a meaningful
    # component along BOTH e (a) and e_perp (omega), so the cheaper-to-turn
    # (angular_weight < acceleration_weight) minimum-intervention solution
    # uses omega, not just deceleration.
    state = RobotState(x=6.5, y=4.5, theta=np.pi / 4.0, v=2.0)
    nominal = np.array([[0.0], [0.0]])

    result = _filter().filter(
        distance_field_frame=frame,
        state=state,
        limits=_limits(),
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=3.0,
    )

    assert result.constraint_count >= 1
    assert result.nominal_feasible is False
    assert result.feasible is True
    assert float(result.control[1, 0]) != 0.0

    samples = frame.sample(state.x, state.y)
    constraints = [
        evaluate_hazard_constraint(
            sample=sample, v=state.v, theta=state.theta, safety_radius=0.35, margin=0.20, k1=2.0, k2=2.0
        )
        for sample in samples
    ]
    for constraint in constraints:
        assert constraint_margin(constraint, result.control) >= -1e-6


# ---------------------------------------------------------------------------
# 17. h < 0 -> invalid initial condition.
# ---------------------------------------------------------------------------


def test_h_negative_marks_invalid_initial_condition():
    frame = _build_field_frame()
    limits = _limits()

    result = _filter().filter(
        distance_field_frame=frame,
        state=RobotState(x=8.5, y=5.5, theta=0.0, v=0.5),  # inside the hazard cell itself
        limits=limits,
        nominal_control=np.array([[0.0], [0.0]]),
        safety_radius=0.35,
        margin=0.20,
        activation_distance=3.0,
    )

    assert result.minimum_h is not None and result.minimum_h < 0.0
    assert result.initial_condition_valid is False
    assert -limits.max_acceleration - 1e-9 <= float(result.control[0, 0]) <= limits.max_acceleration + 1e-9
    assert -limits.max_angular_speed - 1e-9 <= float(result.control[1, 0]) <= limits.max_angular_speed + 1e-9


# ---------------------------------------------------------------------------
# 18. psi_1 < 0 (h still >= 0) -> invalid initial condition.
# ---------------------------------------------------------------------------


def test_psi_1_negative_marks_invalid_initial_condition():
    frame = _build_field_frame()
    state = RobotState(x=7.5, y=5.5, theta=0.0, v=1.0)
    safety_radius, margin, k1, k2 = 0.20, 0.10, 2.0, 2.0

    sample = frame.sample(state.x, state.y)[0]
    probe = evaluate_hazard_constraint(
        sample=sample, v=1.0, theta=0.0, safety_radius=safety_radius, margin=margin, k1=k1, k2=k2
    )
    assert probe.h >= 0.0, "test setup must start from a non-negative h to isolate psi_1 < 0"

    # Pick a velocity large enough that h_dot = v*(g.e) alone drives
    # psi_1 = h_dot + k1*h below zero, using the SAME probe's g.e (no
    # equation duplicated -- only the sign/magnitude already computed above
    # is used to size v for this test).
    g_dot_e = probe.h_dot / 1.0  # v=1.0 was used above, so h_dot == g_dot_e
    assert g_dot_e < 0.0, "heading must be toward the hazard for this scenario"
    required_v = (abs(k1 * probe.h) + 1.0) / abs(g_dot_e)
    state = RobotState(x=state.x, y=state.y, theta=0.0, v=required_v)

    result = _filter().filter(
        distance_field_frame=frame,
        state=state,
        limits=_limits(),
        nominal_control=np.array([[0.0], [0.0]]),
        safety_radius=safety_radius,
        margin=margin,
        activation_distance=3.0,
    )

    assert result.minimum_h is not None and result.minimum_h >= 0.0
    assert result.minimum_psi_1 is not None and result.minimum_psi_1 < 0.0
    assert result.initial_condition_valid is False


# ---------------------------------------------------------------------------
# 19. Multiple pyramid levels -> every feasible constraint is satisfied.
# ---------------------------------------------------------------------------


def test_multiple_levels_all_constraints_satisfied():
    frame = _build_field_frame(pyramid_levels=2)
    assert len(frame.levels) == 2

    state = RobotState(x=5.5, y=5.5, theta=0.0, v=2.0)
    limits = _limits()

    result = _filter().filter(
        distance_field_frame=frame,
        state=state,
        limits=limits,
        nominal_control=np.array([[0.0], [0.0]]),
        safety_radius=0.35,
        margin=0.20,
        activation_distance=4.0,
    )

    assert result.constraint_count == 2

    samples = frame.sample(state.x, state.y)
    constraints = [
        evaluate_hazard_constraint(
            sample=sample, v=state.v, theta=state.theta, safety_radius=0.35, margin=0.20, k1=2.0, k2=2.0
        )
        for sample in samples
    ]
    for constraint in constraints:
        assert constraint_margin(constraint, result.control) >= -1e-6 or result.feasible is False


# ---------------------------------------------------------------------------
# 20. Infeasible QP -> bounded backup control, no exception.
# ---------------------------------------------------------------------------


class _FakeSample:
    def __init__(self, value, gradient, hessian, level):
        self.value = value
        self.gradient = gradient
        self.hessian = hessian
        self.level = level


class _FakeInfeasibleFrame:
    has_hazards = True

    def sample(self, x, y):
        return (
            _FakeSample(value=0.05, gradient=np.array([1.0, 0.0]), hessian=np.zeros((2, 2)), level=0),
            _FakeSample(value=0.05, gradient=np.array([-1.0, 0.0]), hessian=np.zeros((2, 2)), level=1),
        )


def test_infeasible_qp_returns_bounded_backup_without_raising():
    limits = _limits(max_acceleration=2.0, max_angular_speed=2.5)

    result = _filter().filter(
        distance_field_frame=_FakeInfeasibleFrame(),
        state=RobotState(x=0.0, y=0.0, theta=0.0, v=0.0),
        limits=limits,
        nominal_control=np.array([[0.0], [0.0]]),
        # Deliberately huge R -> both contradictory-gradient constraints
        # demand |a| far beyond max_acceleration, with no omega leverage
        # (both A vectors have a zero second component).
        safety_radius=50.0,
        margin=50.0,
        activation_distance=1000.0,
    )

    assert result.feasible is False
    assert -limits.max_acceleration - 1e-9 <= float(result.control[0, 0]) <= limits.max_acceleration + 1e-9
    assert -limits.max_angular_speed - 1e-9 <= float(result.control[1, 0]) <= limits.max_angular_speed + 1e-9


# ---------------------------------------------------------------------------
# 21. Determinism.
# ---------------------------------------------------------------------------


def test_filter_is_deterministic():
    frame = _build_field_frame()
    state = RobotState(x=5.5, y=5.5, theta=0.0, v=2.0)
    limits = _limits()
    nominal = np.array([[0.0], [0.0]])
    kwargs = dict(
        distance_field_frame=frame,
        state=state,
        limits=limits,
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=3.0,
    )

    result1 = _filter().filter(**kwargs)
    result2 = _filter().filter(**kwargs)

    assert np.array_equal(result1.control, result2.control)
    assert result1.feasible == result2.feasible
    assert result1.reason == result2.reason
    assert result1.intervention_norm == result2.intervention_norm


# ---------------------------------------------------------------------------
# 22. Control always has shape (2, 1).
# ---------------------------------------------------------------------------


def test_control_shape_is_always_two_by_one():
    frame = _build_field_frame()

    for nominal in (np.array([[0.0], [0.0]]), np.array([0.0, 0.0])):
        result = _filter().filter(
            distance_field_frame=frame,
            state=RobotState(x=5.5, y=5.5, theta=0.0, v=2.0),
            limits=_limits(),
            nominal_control=nominal,
            safety_radius=0.35,
            margin=0.20,
            activation_distance=3.0,
        )
        assert result.control.shape == (2, 1)


# ---------------------------------------------------------------------------
# Numeric audit: analytic h_dot/h_ddot vs finite differences along an
# independently RK4-integrated trajectory under a FIXED control, sampling
# the SAME SDF -- confirms the algebraic derivation (module docstring)
# against the actual, possibly-rough discretized field, not just symbolically.
#
# h_dot is checked via a direct central difference of h(t): the SDF value
# itself is smooth enough (Lipschitz) for this to converge as dt shrinks.
#
# h_ddot is checked via a central difference of h_dot(t) -- NOT a second
# difference of h(t). h(t) is only piecewise-BILINEARLY interpolated (C^0,
# with kinks in the gradient at cell boundaries, so effectively not C^2);
# a literal second finite difference of h(t) was verified NOT to converge
# as dt -> 0 (it diverges, dominated by interpolation kinks -- see the
# audit report). h_dot(t) = v(t) * (gradient(t) . e(t)) is instead built
# from the SEPARATELY precomputed (once-differentiated, smoother) gradient
# field, and differencing THAT converges cleanly -- which is exactly what
# the analytic h_ddot formula claims: it predicts the rate of change of
# h_dot using the stored Hessian sample, not a re-differentiation of the
# interpolated value.
# ---------------------------------------------------------------------------


def _rk4_integrate(state0, control, duration, substeps=200):
    a, omega = control

    def derivative(state):
        x, y, theta, v = state
        return np.array([v * np.cos(theta), v * np.sin(theta), omega, a])

    dt = duration / substeps
    state = np.array(state0, dtype=float)
    for _ in range(substeps):
        k1 = derivative(state)
        k2 = derivative(state + dt / 2.0 * k1)
        k3 = derivative(state + dt / 2.0 * k2)
        k4 = derivative(state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def test_analytic_derivatives_match_finite_differences_along_trajectory():
    frame = _build_field_frame()
    safety_radius, margin, k1, k2 = 0.35, 0.20, 2.0, 2.0
    total_r = safety_radius + margin

    # A state comfortably away from the hazard's own boundary discontinuity
    # and off exact grid-cell seams ("estados alejados de discontinuidades
    # del grid"), with a nonzero fixed control so both a and omega exercise
    # the h_ddot formula.
    x0, y0, theta0, v0 = 5.5, 5.75, 0.15, 1.3
    a0, omega0 = 0.30, 0.15
    state0 = (x0, y0, theta0, v0)

    sample0 = frame.sample(x0, y0)[0]
    constraint0 = evaluate_hazard_constraint(
        sample=sample0, v=v0, theta=theta0, safety_radius=safety_radius, margin=margin, k1=k1, k2=k2
    )
    e0 = np.array([np.cos(theta0), np.sin(theta0)])
    h_ddot_analytic = float(constraint0.A @ np.array([a0, omega0])) + (v0**2) * float(
        e0 @ sample0.hessian @ e0
    )

    def h_at(x, y):
        return frame.sample(x, y)[0].value - total_r

    def h_dot_at(x, y, theta, v):
        gradient = frame.sample(x, y)[0].gradient
        e = np.array([np.cos(theta), np.sin(theta)])
        return v * float(gradient @ e)

    dt = 0.05
    state_plus = _rk4_integrate(state0, (a0, omega0), dt)
    state_minus = _rk4_integrate(state0, (a0, omega0), -dt)

    h_dot_fd = (h_at(state_plus[0], state_plus[1]) - h_at(state_minus[0], state_minus[1])) / (2.0 * dt)
    h_ddot_fd = (h_dot_at(*state_plus) - h_dot_at(*state_minus)) / (2.0 * dt)

    assert abs(h_dot_fd - constraint0.h_dot) < 0.05, (h_dot_fd, constraint0.h_dot)
    assert abs(h_ddot_fd - h_ddot_analytic) < 0.05, (h_ddot_fd, h_ddot_analytic)


# ---------------------------------------------------------------------------
# Multiscale audit finding -- KEPT AS EVIDENCE for a decision already made,
# not as an open question: imposing every pyramid level as an independent
# hard constraint can make the QP infeasible for a state that was perfectly
# feasible using level 0 (the finest level) alone. This is exactly why
# SimulationConfig.hazard_cbf_pyramid_levels now defaults to 1 (production
# uses level 0 only; see config.py) -- multi-level is accepted as an
# experimental/research-only configuration, not removed, so this test must
# keep passing to document the reason it is not the default.
#
# This is a concrete, deterministic reproduction of that finding (found via
# a fixed-seed random search over 2000 states near the blob), not a claim
# that it happens often -- a random audit over the same blob found it in
# ~1.4% of activated states, with the coarse level never flipping the fine
# level's gradient DIRECTION (0/430 sampled) but sometimes disagreeing
# enough in magnitude (from Gaussian-blur boundary smearing before 2x
# downsampling) to add a contradictory-in-practice constraint.
# ---------------------------------------------------------------------------


def _build_blob_field(*, pyramid_levels: int, resolution: float = 0.5):
    geometry = GridGeometry(_BOUNDS, resolution)
    belief = HazardBelief(geometry, robot_count=1)
    center_x, center_y = 8.25, 5.75
    rows: list[int] = []
    cols: list[int] = []
    for dy in np.arange(-0.75, 0.76, resolution):
        for dx in np.arange(-0.75, 0.76, resolution):
            cell = geometry.world_to_grid(center_x + dx, center_y + dy)
            if cell is not None:
                rows.append(cell.row)
                cols.append(cell.col)
    belief.observe_cells(rows=rows, cols=cols, values=[0.9] * len(rows), robot_index=0)
    return HazardDistanceFieldBuilder().build(
        belief_frame=belief.snapshot(),
        geometry=geometry,
        block_threshold=_THRESHOLD,
        pyramid_levels=pyramid_levels,
        smoothing_sigma_cells=0.75,
    )


def test_multiscale_pyramid_can_introduce_spurious_infeasibility_audit_finding():
    frame_level0_only = _build_blob_field(pyramid_levels=1)
    frame_both_levels = _build_blob_field(pyramid_levels=2)
    limits = _limits()

    # Found by a fixed-seed (np.random.seed(1)) search over 2000 uniformly
    # sampled states near this same blob -- this is the first one found
    # where level 0 alone is feasible but level 0 + level 1 combined is not.
    state = RobotState(x=8.40409512771545, y=8.325438666456687, theta=-1.1722904624648016, v=2.0769678470079422)
    nominal = np.array([[0.0], [0.0]])

    result_level0_only = _filter().filter(
        distance_field_frame=frame_level0_only,
        state=state,
        limits=limits,
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=3.0,
    )
    result_both_levels = _filter().filter(
        distance_field_frame=frame_both_levels,
        state=state,
        limits=limits,
        nominal_control=nominal,
        safety_radius=0.35,
        margin=0.20,
        activation_distance=3.0,
    )

    assert result_level0_only.feasible is True
    assert result_both_levels.feasible is False, (
        "audit finding: combining pyramid levels as simultaneous hard constraints "
        "can make a level-0-feasible state infeasible -- see the audit report "
        "before changing the multiscale policy"
    )


# ---------------------------------------------------------------------------
# Production posture (post-audit decision): SimulationConfig must default to
# a single pyramid level, and that value must survive a .sim
# serialize/deserialize round trip. Values in [2, 4] must remain LOADABLE
# for research use (the multiscale finding above is exactly why they are not
# the default) -- config_from_sim_payload's clamp range is unchanged.
# ---------------------------------------------------------------------------


def test_simulation_config_defaults_to_single_pyramid_level():
    config = SimulationConfig()
    assert config.hazard_cbf_pyramid_levels == 1


def test_pyramid_level_default_round_trips_through_sim_payload():
    config = SimulationConfig()
    payload = config_to_sim_payload(config)

    assert payload["hazard"]["cbf_pyramid_levels"] == 1

    restored = config_from_sim_payload(payload)
    assert restored.hazard_cbf_pyramid_levels == 1


def test_research_pyramid_levels_remain_loadable_but_are_not_the_default():
    config = SimulationConfig()
    payload = config_to_sim_payload(config)

    for research_value in (2, 3, 4):
        payload["hazard"]["cbf_pyramid_levels"] = research_value
        restored = config_from_sim_payload(payload)
        assert restored.hazard_cbf_pyramid_levels == research_value

    # Out-of-range values still clamp to the existing [1, 4] bounds.
    payload["hazard"]["cbf_pyramid_levels"] = 9
    assert config_from_sim_payload(payload).hazard_cbf_pyramid_levels == 4
    payload["hazard"]["cbf_pyramid_levels"] = 0
    assert config_from_sim_payload(payload).hazard_cbf_pyramid_levels == 1

    # A freshly constructed config (no payload involved) still defaults to 1.
    assert SimulationConfig().hazard_cbf_pyramid_levels == 1
