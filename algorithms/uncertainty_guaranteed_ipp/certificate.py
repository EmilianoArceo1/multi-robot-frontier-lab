"""Model-conditional uncertainty certificates for selected sensing points."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from algorithms.uncertainty_guaranteed_ipp.coverage import build_binary_coverage_matrix
from algorithms.uncertainty_guaranteed_ipp.gp import (
    ArrayLike,
    KernelProtocol,
    _points,
    _readonly,
    posterior_variance,
)


@dataclass(frozen=True)
class UncertaintyCertificate:
    """Finite-set certificate under the supplied GP model.

    ``certified`` uses the exact joint GP posterior variance.  The separate
    ``theorem_coverage_certified`` flag says whether the conservative union of
    Theorem-1 single-candidate coverage rows is also complete.  Joint GP
    effects can make the former true while the latter remains false.

    This is a model-conditional, finite-evaluation-set statement.  It is not a
    guarantee of physical MSE under kernel misspecification, localization
    error, or between evaluation points.
    """

    certified: bool
    theorem_coverage_certified: bool
    selected_indices: tuple[int, ...]
    target_variance: float
    max_posterior_variance: float
    posterior_variance: np.ndarray
    violating_indices: tuple[int, ...]
    theorem_uncovered_indices: tuple[int, ...]
    model_scope: str = "finite evaluation set under the supplied GP kernel/noise model"


def certify_plan(
    candidate_points: ArrayLike,
    evaluation_points: ArrayLike,
    selected_indices: Iterable[int],
    *,
    kernel: KernelProtocol,
    target_variance: float,
    noise_variance: float = 0.0,
    tolerance: float = 1e-9,
    jitter: float = 1e-10,
) -> UncertaintyCertificate:
    """Certify a selected sensing set by exact posterior and Theorem-1 union."""

    candidates = _points(candidate_points, name="candidate_points")
    evaluations = _points(
        evaluation_points,
        name="evaluation_points",
        expected_dimension=candidates.shape[1],
    )
    selected = tuple(int(index) for index in selected_indices)
    if len(set(selected)) != len(selected):
        raise ValueError("selected_indices must not contain duplicates")
    if any(index < 0 or index >= candidates.shape[0] for index in selected):
        raise ValueError("selected_indices contains an out-of-range candidate")

    target = float(target_variance)
    noise = float(noise_variance)
    comparison_tolerance = float(tolerance)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("target_variance must be finite and nonnegative")
    if not math.isfinite(noise) or noise < 0.0:
        raise ValueError("noise_variance must be finite and nonnegative")
    if not math.isfinite(comparison_tolerance) or comparison_tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")

    selected_points = (
        candidates[np.asarray(selected, dtype=int)]
        if selected
        else np.empty((0, candidates.shape[1]), dtype=float)
    )
    variances = posterior_variance(
        selected_points,
        evaluations,
        kernel=kernel,
        noise_variance=noise,
        jitter=jitter,
    )
    violating = tuple(
        int(index)
        for index in np.flatnonzero(variances > target + comparison_tolerance)
    )
    max_variance = float(np.max(variances)) if variances.size else 0.0

    coverage = build_binary_coverage_matrix(
        candidates,
        evaluations,
        kernel=kernel,
        target_variance=target,
        noise_variance=noise,
        comparison_tolerance=comparison_tolerance,
    )
    theorem_covered = coverage.initially_satisfied.copy()
    if selected:
        theorem_covered |= np.any(
            coverage.matrix[np.asarray(selected, dtype=int)],
            axis=0,
        )
    theorem_uncovered = tuple(int(index) for index in np.flatnonzero(~theorem_covered))

    return UncertaintyCertificate(
        certified=not violating,
        theorem_coverage_certified=not theorem_uncovered,
        selected_indices=selected,
        target_variance=target,
        max_posterior_variance=max_variance,
        posterior_variance=_readonly(np.asarray(variances, dtype=float)),
        violating_indices=violating,
        theorem_uncovered_indices=theorem_uncovered,
    )
