"""Observed-Hazard OGM-HOCBF safety filter.

Grounded in three papers:

  1. Raja et al., "OGM-CBF: Occupancy Grid Map-based Control Barrier
     Function for Safe Mobile Robot Control with Memory of out of View
     Obstacles" -- the idea of a minimally-invasive safety filter built on
     top of a continuous geometry derived from a persistent, discovered-only
     occupancy/hazard grid (see ``hazard_distance_field.py``), so a hazard
     that is currently out of the sensor field of view still constrains the
     robot as long as it was observed before.
  2. Xiao & Belta, "Control Barrier Functions for Systems with High
     Relative Degree" -- the position barrier here has relative degree 2
     with respect to acceleration (this simulator's dynamic unicycle is
     controlled by ``u = [a, omega]``, not ``[v, omega]``), so a High-Order
     CBF (HOCBF) is used instead of a first-order CBF.
  3. Ames et al., "Control Barrier Function Based Quadratic Programs for
     Safety Critical Systems" -- the minimum-intervention QP formulation:
     find the control closest (in a weighted norm) to the nominal control
     that still satisfies every barrier constraint.

Explicit deviation from OGM-CBF: OGM-CBF's own unicycle is kinematic and
controlled directly by ``[v, omega]``, giving its position barrier relative
degree 1. This simulator's ``DynamicUnicycle2D`` (robotics_sim/models/
dynamic_unicycle.py) is controlled by ``u = [a, omega]`` with velocity as a
state, giving the SAME position barrier relative degree 2 -- hence "OGM-
HOCBF" rather than a claim of reproducing OGM-CBF exactly.

Derivation (verified algebraically before implementation; kept here rather
than only in code so the two can be checked against each other)
------------------------------------------------------------------

State subset used here: p = (x, y), theta, v. Control: u = [a, omega].
Dynamics (unchanged, see dynamic_unicycle.py): pdot = v * e, thetadot =
omega, vdot = a, where e = [cos(theta), sin(theta)] and
e_perp = [-sin(theta), cos(theta)] (so de/dt = omega * e_perp).

Let phi(p) be the Signed Distance Field value at p (positive outside the
observed-unsafe set, negative inside -- see hazard_distance_field.py),
g = gradient(phi) (2-vector), H = hessian(phi) (2x2), R = safety_radius +
hazard_cbf_margin (both constants w.r.t. time), and

    h(p) = phi(p) - R

Relative degree 2 in a (h depends on p, not on v or a directly):

    h_dot   = grad(phi) . pdot = v * (g . e)

    d/dt(g . e) = (dg/dt) . e + g . (de/dt)
                = (H @ pdot) . e + g . (omega * e_perp)
                = v * (e^T H e) + omega * (g . e_perp)

    h_ddot = d/dt(v * (g.e)) = vdot*(g.e) + v * d/dt(g.e)
           = a*(g.e) + v^2*(e^T H e) + v*omega*(g.e_perp)

which matches the assignment's stated h_ddot exactly. Linear class-K HOCBF
(Xiao & Belta):

    psi_1 = h_dot + k1*h
    psi_2 = psi_1_dot + k2*psi_1
          = h_ddot + k1*h_dot + k2*(h_dot + k1*h)
          = h_ddot + (k1+k2)*h_dot + k1*k2*h

Requiring psi_2 >= 0 and collecting the terms that are linear in u = [a,
omega] (using h_dot = v*(g.e) to substitute):

    psi_2 = a*(g.e) + v*omega*(g.e_perp)                     <- linear in u
          + v^2*(e^T H e) + (k1+k2)*v*(g.e) + k1*k2*h         <- "drift"

So the constraint "psi_2 >= 0" is exactly the linear inequality

    A @ u >= b,   A = [g.e, v*(g.e_perp)],   b = -drift

with drift = v^2*(e^T H e) + (k1+k2)*v*(g.e) + k1*k2*h. This is the exact
form implemented below (``evaluate_hazard_constraint``) and used by both the
filter and the tests via the same pure helper -- the equations are not
duplicated anywhere else.

Limitations (not hidden):
  - phi/g/H are numerical approximations of a discretized field (see
    hazard_distance_field.py's own docstring); this filter inherits that
    approximation and adds none of its own beyond linearizing psi_2 in u.
  - Everything here is evaluated once per discrete simulation tick and the
    resulting control is integrated by the existing explicit-Euler
    DynamicUnicycle2D.step() -- a continuous-time forward-invariance
    argument does not automatically transfer to the discrete-time
    integrator.
  - A newly discovered hazard can appear with h < 0 or psi_1 < 0 already
    (the robot was already inside the nominal safety margin, or already
    approaching too fast, before this cell was ever observed). This filter
    reports that explicitly via ``initial_condition_valid=False`` instead of
    silently claiming a forward-invariance guarantee it cannot back up; it
    still attempts a minimum-intervention correction biased toward braking
    (braking candidates are always in the candidate pool -- see
    ``_solve_minimum_intervention_qp``), and existing hard-stop/replanning
    safety nets (predicted_motion_report, hard stop) remain in place above
    this filter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class HazardHOCBFConstraint:
    """One pyramid level's linearized HOCBF constraint ``A @ u >= b``."""

    level: int
    h: float
    h_dot: float
    psi_1: float
    A: np.ndarray
    b: float


def evaluate_hazard_constraint(
    *,
    sample,
    v: float,
    theta: float,
    safety_radius: float,
    margin: float,
    k1: float,
    k2: float,
) -> HazardHOCBFConstraint:
    """Pure evaluation of h/h_dot/psi_1/A/b for one HazardDistanceSample.

    This is the single source of truth for the HOCBF algebra -- both
    ``HazardHOCBFSafetyFilter`` and the test suite call this same function,
    so filter behavior and test expectations cannot silently drift apart.
    """
    phi = float(sample.value)
    gradient = np.asarray(sample.gradient, dtype=float).reshape(2)
    hessian = np.asarray(sample.hessian, dtype=float).reshape(2, 2)

    safety_radius_total = float(safety_radius) + float(margin)
    h = phi - safety_radius_total

    e = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    e_perp = np.array([-math.sin(theta), math.cos(theta)], dtype=float)

    g_dot_e = float(gradient @ e)
    g_dot_e_perp = float(gradient @ e_perp)
    e_h_e = float(e @ hessian @ e)

    h_dot = float(v) * g_dot_e
    psi_1 = h_dot + float(k1) * h

    drift = (float(v) ** 2) * e_h_e + (float(k1) + float(k2)) * float(v) * g_dot_e + float(k1) * float(k2) * h
    A = np.array([g_dot_e, float(v) * g_dot_e_perp], dtype=float)
    b = -drift

    return HazardHOCBFConstraint(level=int(sample.level), h=h, h_dot=h_dot, psi_1=psi_1, A=A, b=float(b))


def constraint_margin(constraint: HazardHOCBFConstraint, control) -> float:
    """psi_2 achieved by ``control`` under this constraint -- the same
    ``A @ u - b`` expression enforced as ``A @ u >= b`` (i.e. psi_2 >= 0)."""
    u = np.asarray(control, dtype=float).reshape(2)
    return float(constraint.A @ u) - constraint.b


@dataclass(frozen=True)
class HazardHOCBFResult:
    """Outcome of one ``HazardHOCBFSafetyFilter.filter()`` call."""

    control: np.ndarray
    active: bool
    feasible: bool
    nominal_feasible: bool
    initial_condition_valid: bool
    constraint_count: int
    minimum_h: float | None
    minimum_psi_1: float | None
    minimum_psi_2: float | None
    intervention_norm: float
    reason: str


class HazardHOCBFSafetyFilter:
    """Minimum-intervention HOCBF safety filter over a cached distance field.

    Never touches ground truth, never mutates its inputs, never raises for
    ordinary infeasibility -- see the module docstring for the full
    derivation and documented limitations.
    """

    def __init__(
        self,
        *,
        k1: float = 2.0,
        k2: float = 2.0,
        acceleration_weight: float = 1.0,
        angular_weight: float = 0.35,
        tolerance: float = _FEASIBILITY_TOLERANCE,
    ) -> None:
        if k1 <= 0:
            raise ValueError(f"k1 must be > 0, got {k1}.")
        if k2 <= 0:
            raise ValueError(f"k2 must be > 0, got {k2}.")
        if acceleration_weight <= 0:
            raise ValueError(f"acceleration_weight must be > 0, got {acceleration_weight}.")
        if angular_weight <= 0:
            raise ValueError(f"angular_weight must be > 0, got {angular_weight}.")

        self.k1 = float(k1)
        self.k2 = float(k2)
        self.acceleration_weight = float(acceleration_weight)
        self.angular_weight = float(angular_weight)
        self.tolerance = float(tolerance)

    def filter(
        self,
        *,
        distance_field_frame,
        state,
        limits,
        nominal_control,
        safety_radius: float,
        margin: float,
        activation_distance: float,
    ) -> HazardHOCBFResult:
        # .copy() guarantees the caller's array is never aliased/mutated,
        # regardless of the input shape ((2,1) column vector or (2,)).
        nominal = np.asarray(nominal_control, dtype=float).reshape(2).copy()

        if distance_field_frame is None or not getattr(distance_field_frame, "has_hazards", False):
            return _passthrough_result(nominal, reason="no hazard distance field active")

        samples = distance_field_frame.sample(float(state.x), float(state.y))
        activation_distance = float(activation_distance)

        constraints = [
            evaluate_hazard_constraint(
                sample=sample,
                v=float(state.v),
                theta=float(state.theta),
                safety_radius=safety_radius,
                margin=margin,
                k1=self.k1,
                k2=self.k2,
            )
            for sample in samples
            if float(sample.value) <= activation_distance
        ]

        if not constraints:
            return _passthrough_result(nominal, reason="no observed hazard within activation distance")

        minimum_h = min(c.h for c in constraints)
        minimum_psi_1 = min(c.psi_1 for c in constraints)
        initial_condition_valid = minimum_h >= 0.0 and minimum_psi_1 >= 0.0

        nominal_feasible = all(
            constraint_margin(c, nominal) >= -self.tolerance for c in constraints
        )

        max_acceleration = float(limits.max_acceleration)
        max_angular_speed = float(limits.max_angular_speed)

        if nominal_feasible:
            control = nominal
            feasible = True
            reason = "nominal control already satisfies all HOCBF constraints"
        else:
            control, feasible = _solve_minimum_intervention_qp(
                constraints=constraints,
                nominal=nominal,
                acceleration_weight=self.acceleration_weight,
                angular_weight=self.angular_weight,
                max_acceleration=max_acceleration,
                max_angular_speed=max_angular_speed,
                tolerance=self.tolerance,
            )
            reason = (
                "minimum-intervention HOCBF correction"
                if feasible
                else "no feasible HOCBF correction found; falling back to a bounded braking control"
            )

        minimum_psi_2 = min(constraint_margin(c, control) for c in constraints)
        intervention_norm = float(np.linalg.norm(control - nominal))

        return HazardHOCBFResult(
            control=control.reshape(2, 1),
            active=True,
            feasible=feasible,
            nominal_feasible=nominal_feasible,
            initial_condition_valid=initial_condition_valid,
            constraint_count=len(constraints),
            minimum_h=minimum_h,
            minimum_psi_1=minimum_psi_1,
            minimum_psi_2=minimum_psi_2,
            intervention_norm=intervention_norm,
            reason=reason,
        )


def _passthrough_result(nominal: np.ndarray, *, reason: str) -> HazardHOCBFResult:
    return HazardHOCBFResult(
        control=nominal.reshape(2, 1),
        active=False,
        feasible=True,
        nominal_feasible=True,
        initial_condition_valid=True,
        constraint_count=0,
        minimum_h=None,
        minimum_psi_1=None,
        minimum_psi_2=None,
        intervention_norm=0.0,
        reason=reason,
    )


# ============================================================
# Minimum-intervention QP: dimension-2 enumeration (Ames et al. formulation,
# no external solver -- see module/task docstring for why enumeration is
# exact for a 2D box-constrained linearly-constrained convex QP).
# ============================================================


def _solve_minimum_intervention_qp(
    *,
    constraints: list[HazardHOCBFConstraint],
    nominal: np.ndarray,
    acceleration_weight: float,
    angular_weight: float,
    max_acceleration: float,
    max_angular_speed: float,
    tolerance: float,
) -> tuple[np.ndarray, bool]:
    weights = np.array([acceleration_weight, angular_weight], dtype=float)
    weights_inv = 1.0 / weights

    candidates: list[np.ndarray] = [nominal]

    for constraint in constraints:
        projected = _weighted_projection(nominal, constraint.A, constraint.b, weights_inv)
        if projected is not None:
            candidates.append(projected)

    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            point = _line_intersection(
                constraints[i].A, constraints[i].b, constraints[j].A, constraints[j].b
            )
            if point is not None:
                candidates.append(point)

    box_a_edges = (-max_acceleration, max_acceleration)
    box_omega_edges = (-max_angular_speed, max_angular_speed)

    for constraint in constraints:
        for a_fixed in box_a_edges:
            point = _solve_for_omega(constraint.A, constraint.b, a_fixed)
            if point is not None:
                candidates.append(point)
        for omega_fixed in box_omega_edges:
            point = _solve_for_acceleration(constraint.A, constraint.b, omega_fixed)
            if point is not None:
                candidates.append(point)

    for a_fixed in box_a_edges:
        for omega_fixed in box_omega_edges:
            candidates.append(np.array([a_fixed, omega_fixed], dtype=float))

    clipped_nominal_omega = float(np.clip(nominal[1], -max_angular_speed, max_angular_speed))
    candidates.append(np.array([-max_acceleration, 0.0], dtype=float))
    candidates.append(np.array([-max_acceleration, clipped_nominal_omega], dtype=float))

    scored: list[tuple[float, float, float, np.ndarray]] = []
    for candidate in candidates:
        clipped = np.array(
            [
                float(np.clip(candidate[0], -max_acceleration, max_acceleration)),
                float(np.clip(candidate[1], -max_angular_speed, max_angular_speed)),
            ],
            dtype=float,
        )
        if _within_all_constraints(clipped, constraints, tolerance):
            cost = float(0.5 * np.sum(weights * (clipped - nominal) ** 2))
            scored.append((cost, clipped[0], clipped[1], clipped))

    if scored:
        scored.sort(key=lambda item: (item[0], item[1], item[2]))
        return scored[0][3], True

    backup = np.array([-max_acceleration, 0.0], dtype=float)
    return backup, False


def _weighted_projection(u0: np.ndarray, A: np.ndarray, b: float, weights_inv: np.ndarray) -> np.ndarray | None:
    denom = float(A @ (weights_inv * A))
    if abs(denom) < 1e-12:
        return None
    lam = (b - float(A @ u0)) / denom
    return u0 + weights_inv * A * lam


def _line_intersection(A_i: np.ndarray, b_i: float, A_j: np.ndarray, b_j: float) -> np.ndarray | None:
    matrix = np.array([A_i, A_j], dtype=float)
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(det) < 1e-12:
        return None
    rhs = np.array([b_i, b_j], dtype=float)
    return np.linalg.solve(matrix, rhs)


def _solve_for_omega(A: np.ndarray, b: float, a_fixed: float) -> np.ndarray | None:
    if abs(A[1]) < 1e-12:
        return None
    omega = (b - A[0] * a_fixed) / A[1]
    return np.array([a_fixed, omega], dtype=float)


def _solve_for_acceleration(A: np.ndarray, b: float, omega_fixed: float) -> np.ndarray | None:
    if abs(A[0]) < 1e-12:
        return None
    acceleration = (b - A[1] * omega_fixed) / A[0]
    return np.array([acceleration, omega_fixed], dtype=float)


def _within_all_constraints(u: np.ndarray, constraints: list[HazardHOCBFConstraint], tolerance: float) -> bool:
    for constraint in constraints:
        if constraint_margin(constraint, u) < -tolerance:
            return False
    return True
