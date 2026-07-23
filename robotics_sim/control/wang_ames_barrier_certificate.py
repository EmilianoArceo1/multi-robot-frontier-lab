"""Decentralized safety barrier certificate from Wang, Ames & Egerstedt.

Implements the nominal decentralized QP in Eq. (12) and the infeasible-QP
braking branch in Eq. (17) of "Safety Barrier Certificates for
Collision-Free Multirobot Systems", IEEE TRO 33(3), 2017.

The paper models planar double-integrator inputs.  For this simulator's
dynamic unicycle, the experimental diffeomorphism described in Sec. VIII is
realized through ``u_xy = a*e + v*omega*e_perp``.  Consequently every paper
half-space remains linear in the native control ``[a, omega]``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

WANG_AMES_BARRIER_CERTIFICATE = "Wang-Ames-Egerstedt decentralized barrier certificate"
SAFETY_ALGORITHM_OPTIONS = (WANG_AMES_BARRIER_CERTIFICATE,)


@dataclass(frozen=True)
class BarrierCertificateResult:
    control: np.ndarray
    active: bool
    feasible: bool
    constraint_count: int
    minimum_h: float | None
    reason: str


def _native_constraint(
    *, ego, other, safety_distance: float, gamma: float
) -> tuple[np.ndarray, float, float] | None:
    pi = np.array([float(ego.x), float(ego.y)])
    pj = np.array([float(other.x), float(other.y)])
    vi = float(ego.v) * np.array([math.cos(float(ego.theta)), math.sin(float(ego.theta))])
    vj = float(other.v) * np.array([math.cos(float(other.theta)), math.sin(float(other.theta))])
    dp = pi - pj
    dv = vi - vj
    distance = float(np.linalg.norm(dp))
    if distance <= 1e-12:
        return None

    alpha_i = float(ego.limits.max_acceleration)
    alpha_j = float(other.limits.max_acceleration)
    alpha_sum = alpha_i + alpha_j
    clearance = distance - float(safety_distance)
    radial_velocity = float(dp @ dv) / distance
    h = math.sqrt(max(0.0, 2.0 * alpha_sum * clearance)) + radial_velocity

    # Eq. (7). Outside the theorem's initially-safe domain, force the hybrid
    # braking branch rather than claiming a certificate.
    if clearance <= 0.0:
        return np.zeros(2), -math.inf, h
    denominator = math.sqrt(2.0 * alpha_sum * clearance)
    bij = (
        float(gamma) * h**3 * distance
        - float(dp @ dv) ** 2 / distance**2
        + float(dv @ dv)
        + alpha_sum * float(dv @ dp) / denominator
    )
    bound = (alpha_i / alpha_sum) * bij

    e = np.array([math.cos(float(ego.theta)), math.sin(float(ego.theta))])
    e_perp = np.array([-e[1], e[0]])
    # -dp^T u_i <= alpha_i/(alpha_i+alpha_j) b_ij, Eq. (12).
    native_a = np.array([-float(dp @ e), -float(ego.v) * float(dp @ e_perp)])
    return native_a, float(bound), h


def filter_control(
    *,
    ego,
    others: Sequence[object],
    nominal_control,
    safety_distance: float | Callable[[object], float],
    gamma: float = 1.0,
) -> BarrierCertificateResult:
    nominal = np.asarray(nominal_control, dtype=float).reshape(2)
    constraints = [
        value
        for other in others
        if other is not ego
        for value in [
            _native_constraint(
                ego=ego,
                other=other,
                safety_distance=(
                    float(safety_distance(other))
                    if callable(safety_distance)
                    else float(safety_distance)
                ),
                gamma=gamma,
            )
        ]
        if value is not None
    ]
    if not constraints:
        return BarrierCertificateResult(nominal.reshape(2, 1), False, True, 0, None, "no neighboring agents")

    if any(not math.isfinite(bound) for _a, bound, _h in constraints):
        braking = np.array([-float(ego.limits.max_acceleration), 0.0])
        return BarrierCertificateResult(braking.reshape(2, 1), True, False, len(constraints), min(c[2] for c in constraints), "Eq. (17) hybrid braking")

    max_a = float(ego.limits.max_acceleration)
    max_w = float(ego.limits.max_angular_speed)
    candidates = [nominal.copy()]
    for a_vec, bound, _h in constraints:
        denom = float(a_vec @ a_vec)
        if denom > 1e-12:
            candidates.append(nominal - a_vec * max(0.0, (float(a_vec @ nominal) - bound) / denom))
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            matrix = np.array([constraints[i][0], constraints[j][0]])
            if abs(float(np.linalg.det(matrix))) > 1e-12:
                candidates.append(np.linalg.solve(matrix, np.array([constraints[i][1], constraints[j][1]])))
    for a in (-max_a, max_a):
        for w in (-max_w, max_w):
            candidates.append(np.array([a, w]))

    feasible: list[tuple[float, float, float, np.ndarray]] = []
    for candidate in candidates:
        u = np.array([np.clip(candidate[0], -max_a, max_a), np.clip(candidate[1], -max_w, max_w)])
        if all(float(a_vec @ u) <= bound + 1e-8 for a_vec, bound, _h in constraints):
            feasible.append((float(np.sum((u - nominal) ** 2)), float(u[0]), float(u[1]), u))
    if not feasible:
        braking = np.array([-max_a, 0.0])
        return BarrierCertificateResult(braking.reshape(2, 1), True, False, len(constraints), min(c[2] for c in constraints), "Eq. (17) hybrid braking")

    feasible.sort(key=lambda item: item[:3])
    control = feasible[0][3]
    changed = not np.allclose(control, nominal, atol=1e-9)
    return BarrierCertificateResult(
        control.reshape(2, 1), changed, True, len(constraints), min(c[2] for c in constraints),
        "Eq. (12) minimum-intervention QP" if changed else "nominal control is certified",
    )
