"""Theorem-1 binary coverage maps and deterministic GreedyCover."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from algorithms.uncertainty_guaranteed_ipp.gp import ArrayLike, KernelProtocol, _points, _readonly


@dataclass(frozen=True)
class BinaryCoverageMap:
    """Conservative per-candidate coverage of finite evaluation points.

    ``matrix[i, j]`` is true exactly when Theorem 1's single-observation
    condition certifies evaluation point ``j`` from candidate ``i``.
    ``initially_satisfied`` marks evaluation points whose prior variance is
    already below the target and therefore require no selected candidate.
    """

    matrix: np.ndarray
    covariance_thresholds: np.ndarray
    candidate_prior_variance: np.ndarray
    evaluation_prior_variance: np.ndarray
    initially_satisfied: np.ndarray
    target_variance: float
    noise_variance: float

    @property
    def candidate_count(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def evaluation_count(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def uncoverable_evaluation_indices(self) -> tuple[int, ...]:
        if self.evaluation_count == 0:
            return ()
        covered_by_any = self.initially_satisfied.copy()
        if self.candidate_count:
            covered_by_any |= np.any(self.matrix, axis=0)
        return tuple(int(i) for i in np.flatnonzero(~covered_by_any))


@dataclass(frozen=True)
class GreedyCoverResult:
    selected_indices: tuple[int, ...]
    covered_mask: np.ndarray
    marginal_gains: tuple[int, ...]
    complete: bool
    uncovered_indices: tuple[int, ...]

    @property
    def covered_count(self) -> int:
        return int(np.count_nonzero(self.covered_mask))


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _coverage_matrix(value: np.ndarray | Sequence[Sequence[bool]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=bool)
    if matrix.ndim != 2:
        raise ValueError("coverage_matrix must be a 2D candidate-by-evaluation matrix")
    return np.ascontiguousarray(matrix, dtype=bool)


def _initial_mask(value: Iterable[bool] | np.ndarray | None, size: int) -> np.ndarray:
    if value is None:
        return np.zeros(size, dtype=bool)
    mask = np.asarray(value, dtype=bool).reshape(-1)
    if mask.shape != (size,):
        raise ValueError(f"initially_covered must have shape ({size},)")
    return mask.copy()


def build_binary_coverage_matrix(
    candidate_points: ArrayLike,
    evaluation_points: ArrayLike,
    *,
    kernel: KernelProtocol,
    target_variance: float,
    noise_variance: float = 0.0,
    comparison_tolerance: float = 1e-12,
) -> BinaryCoverageMap:
    """Construct the paper's binary map from Theorem 1.

    For candidate ``c`` and evaluation point ``v``, a single noisy observation
    certifies the target exactly when

    ``abs(k(c,v)) >= sqrt((k(v,v)-target) * (k(c,c)+noise))``.

    The radicand's first factor is clamped at zero because an evaluation point
    whose prior variance already satisfies the target needs no measurement.
    Such points are also exposed through ``initially_satisfied`` so GreedyCover
    does not select a gratuitous first candidate when every point is already
    below threshold.
    """

    candidates = _points(candidate_points, name="candidate_points")
    evaluations = _points(
        evaluation_points,
        name="evaluation_points",
        expected_dimension=candidates.shape[1],
    )
    target = _finite_nonnegative(target_variance, name="target_variance")
    noise = _finite_nonnegative(noise_variance, name="noise_variance")
    tolerance = _finite_nonnegative(comparison_tolerance, name="comparison_tolerance")

    candidate_variance = np.asarray(kernel.diagonal(candidates), dtype=float).reshape(-1)
    evaluation_variance = np.asarray(kernel.diagonal(evaluations), dtype=float).reshape(-1)
    if candidate_variance.shape != (candidates.shape[0],):
        raise ValueError("kernel.diagonal(candidate_points) returned an invalid shape")
    if evaluation_variance.shape != (evaluations.shape[0],):
        raise ValueError("kernel.diagonal(evaluation_points) returned an invalid shape")
    if not np.isfinite(candidate_variance).all() or not np.isfinite(evaluation_variance).all():
        raise ValueError("kernel diagonal contains non-finite values")
    if np.any(candidate_variance < -tolerance) or np.any(evaluation_variance < -tolerance):
        raise ValueError("kernel diagonal contains a negative prior variance")
    candidate_variance = np.maximum(candidate_variance, 0.0)
    evaluation_variance = np.maximum(evaluation_variance, 0.0)

    covariance = np.asarray(kernel(candidates, evaluations), dtype=float)
    if covariance.shape != (candidates.shape[0], evaluations.shape[0]):
        raise ValueError("kernel(candidate_points, evaluation_points) returned an invalid shape")
    if not np.isfinite(covariance).all():
        raise ValueError("candidate/evaluation covariance contains non-finite values")

    required_reduction = np.maximum(evaluation_variance - target, 0.0)
    thresholds = np.sqrt(
        (candidate_variance[:, np.newaxis] + noise)
        * required_reduction[np.newaxis, :]
    )
    matrix = np.abs(covariance) + tolerance >= thresholds
    initially_satisfied = evaluation_variance <= target + tolerance

    return BinaryCoverageMap(
        matrix=_readonly(matrix.astype(bool, copy=False)),
        covariance_thresholds=_readonly(thresholds),
        candidate_prior_variance=_readonly(candidate_variance),
        evaluation_prior_variance=_readonly(evaluation_variance),
        initially_satisfied=_readonly(initially_satisfied.astype(bool, copy=False)),
        target_variance=target,
        noise_variance=noise,
    )


def greedy_cover(
    coverage_matrix: np.ndarray | Sequence[Sequence[bool]],
    *,
    initially_covered: Iterable[bool] | np.ndarray | None = None,
) -> GreedyCoverResult:
    """Greedy maximum-marginal-coverage selection with stable tie-breaking.

    Ties are resolved by the smallest candidate index.  The procedure stops
    once every evaluation point is covered or no unselected candidate has
    positive marginal gain; it never pretends an uncoverable set is complete.
    """

    matrix = _coverage_matrix(coverage_matrix)
    candidate_count, evaluation_count = matrix.shape
    covered = _initial_mask(initially_covered, evaluation_count)
    selected: list[int] = []
    gains: list[int] = []
    available = np.ones(candidate_count, dtype=bool)

    while evaluation_count and not bool(np.all(covered)):
        uncovered = ~covered
        best_index = -1
        best_gain = 0
        for index in range(candidate_count):
            if not available[index]:
                continue
            gain = int(np.count_nonzero(matrix[index] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_index = index

        if best_index < 0 or best_gain <= 0:
            break

        available[best_index] = False
        selected.append(best_index)
        gains.append(best_gain)
        covered |= matrix[best_index]

    complete = bool(np.all(covered))
    uncovered_indices = tuple(int(i) for i in np.flatnonzero(~covered))
    return GreedyCoverResult(
        selected_indices=tuple(selected),
        covered_mask=_readonly(covered.astype(bool, copy=False)),
        marginal_gains=tuple(gains),
        complete=complete,
        uncovered_indices=uncovered_indices,
    )
