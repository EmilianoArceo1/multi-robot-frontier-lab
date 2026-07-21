"""Small exact Gaussian-process regression primitives using only NumPy.

The RBF model in this module is an integration/smoke-test model.  The paper's
experimental model is a learned non-stationary Attentive kernel implemented
through SGP-Tools.  Callers can provide that future model through
``KernelProtocol`` while keeping coverage and routing independent of a GP
framework.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


ArrayLike = Sequence[float] | Sequence[Sequence[float]] | np.ndarray


def _points(value: ArrayLike, *, name: str, expected_dimension: int | None = None) -> np.ndarray:
    """Normalize points to a finite ``(count, dimension)`` float array.

    A one-dimensional non-empty input is interpreted as scalar input points,
    i.e. ``[0, 1]`` means two points in one dimension.  Multi-dimensional
    callers should pass ``[[x, y], ...]`` explicitly.
    """

    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        if array.size == 0:
            dimension = 1 if expected_dimension is None else int(expected_dimension)
            array = array.reshape(0, dimension)
        else:
            array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D scalar-point list or a 2D point matrix")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must have at least one coordinate per point")
    if expected_dimension is not None and array.shape[1] != int(expected_dimension):
        raise ValueError(
            f"{name} dimension {array.shape[1]} != expected {int(expected_dimension)}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return np.ascontiguousarray(array, dtype=float)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@runtime_checkable
class KernelProtocol(Protocol):
    """Covariance interface consumed by the paper core.

    ``__call__(x, y)`` returns the full ``len(x) x len(y)`` covariance matrix;
    ``diagonal(x)`` returns prior variances at ``x``.  This intentionally
    accommodates non-stationary kernels whose diagonal need not be constant.
    """

    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        ...

    def diagonal(self, x: ArrayLike) -> np.ndarray:
        ...


@dataclass(frozen=True)
class RBFKernel:
    """Stationary squared-exponential kernel for a NumPy-only MVP.

    ``variance`` is signal variance (not standard deviation). ``length_scale``
    may be scalar or one positive value per input dimension.
    """

    variance: float = 1.0
    length_scale: float | tuple[float, ...] = 1.0

    def __post_init__(self) -> None:
        variance = float(self.variance)
        if not math.isfinite(variance) or variance <= 0.0:
            raise ValueError("variance must be finite and greater than zero")

        raw_scale = np.asarray(self.length_scale, dtype=float)
        if raw_scale.ndim > 1 or raw_scale.size == 0:
            raise ValueError("length_scale must be a positive scalar or 1D sequence")
        flat_scale = raw_scale.reshape(-1)
        if not np.isfinite(flat_scale).all() or np.any(flat_scale <= 0.0):
            raise ValueError("length_scale values must be finite and greater than zero")

        object.__setattr__(self, "variance", variance)
        if flat_scale.size == 1:
            object.__setattr__(self, "length_scale", float(flat_scale[0]))
        else:
            object.__setattr__(self, "length_scale", tuple(float(v) for v in flat_scale))

    def _scale_for_dimension(self, dimension: int) -> np.ndarray:
        scale = np.asarray(self.length_scale, dtype=float).reshape(-1)
        if scale.size == 1:
            return np.full(dimension, float(scale[0]), dtype=float)
        if scale.size != dimension:
            raise ValueError(
                f"anisotropic length_scale has {scale.size} values for {dimension}D points"
            )
        return scale

    def __call__(self, x: ArrayLike, y: ArrayLike) -> np.ndarray:
        x_points = _points(x, name="x")
        y_points = _points(y, name="y", expected_dimension=x_points.shape[1])
        scale = self._scale_for_dimension(x_points.shape[1])
        delta = (x_points[:, np.newaxis, :] - y_points[np.newaxis, :, :]) / scale
        squared_distance = np.einsum("ijk,ijk->ij", delta, delta, optimize=True)
        return self.variance * np.exp(-0.5 * squared_distance)

    def diagonal(self, x: ArrayLike) -> np.ndarray:
        points = _points(x, name="x")
        self._scale_for_dimension(points.shape[1])
        return np.full(points.shape[0], self.variance, dtype=float)


@dataclass(frozen=True)
class GaussianProcessPosterior:
    """Posterior prediction at a fixed query set."""

    mean: np.ndarray
    covariance: np.ndarray
    variance: np.ndarray
    applied_jitter: float


def _nonnegative_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _stable_cholesky(
    matrix: np.ndarray,
    *,
    jitter: float,
    max_attempts: int = 7,
) -> tuple[np.ndarray, float]:
    """Cholesky factorization with deterministic diagonal-jitter escalation."""

    if matrix.shape[0] == 0:
        return matrix.copy(), 0.0

    base_jitter = _nonnegative_finite(jitter, name="jitter")
    current = base_jitter
    identity = np.eye(matrix.shape[0], dtype=float)
    last_error: np.linalg.LinAlgError | None = None

    for attempt in range(max(1, int(max_attempts))):
        try:
            return np.linalg.cholesky(matrix + current * identity), current
        except np.linalg.LinAlgError as exc:
            last_error = exc
            current = (1e-12 if current == 0.0 else current * 10.0)

    raise np.linalg.LinAlgError(
        f"kernel matrix is not numerically positive definite after jitter={current:g}"
    ) from last_error


def posterior_variance(
    training_points: ArrayLike,
    query_points: ArrayLike,
    *,
    kernel: KernelProtocol,
    noise_variance: float = 0.0,
    jitter: float = 1e-10,
) -> np.ndarray:
    """Return latent GP posterior variance at ``query_points``.

    The result conditions on noisy observations at ``training_points`` but is
    independent of their realized values, which is precisely why the paper can
    plan from covariance alone.
    """

    query = _points(query_points, name="query_points")
    training = _points(
        training_points,
        name="training_points",
        expected_dimension=query.shape[1],
    )
    noise = _nonnegative_finite(noise_variance, name="noise_variance")
    prior = np.asarray(kernel.diagonal(query), dtype=float).reshape(-1)
    if prior.shape != (query.shape[0],) or not np.isfinite(prior).all():
        raise ValueError("kernel.diagonal(query_points) returned an invalid shape or value")
    if np.any(prior < -1e-10):
        raise ValueError("kernel returned a negative prior variance")

    if training.shape[0] == 0:
        return _readonly(np.maximum(prior, 0.0))

    train_cov = np.asarray(kernel(training, training), dtype=float)
    expected = (training.shape[0], training.shape[0])
    if train_cov.shape != expected or not np.isfinite(train_cov).all():
        raise ValueError("kernel(training, training) returned an invalid matrix")
    train_cov = 0.5 * (train_cov + train_cov.T)
    train_cov += noise * np.eye(training.shape[0], dtype=float)
    chol, _ = _stable_cholesky(train_cov, jitter=jitter)

    train_query_cov = np.asarray(kernel(training, query), dtype=float)
    if train_query_cov.shape != (training.shape[0], query.shape[0]):
        raise ValueError("kernel(training, query) returned an invalid matrix shape")
    projected = np.linalg.solve(chol, train_query_cov)
    variance = prior - np.einsum("ij,ij->j", projected, projected, optimize=True)

    # Roundoff can produce tiny negatives at training locations.  A materially
    # negative value indicates a broken kernel and is not silently hidden.
    if np.any(variance < -1e-7):
        raise ValueError("posterior variance became materially negative; check the kernel")
    return _readonly(np.maximum(variance, 0.0))


def gp_posterior(
    training_points: ArrayLike,
    training_values: Sequence[float] | np.ndarray,
    query_points: ArrayLike,
    *,
    kernel: KernelProtocol,
    noise_variance: float = 0.0,
    mean: float = 0.0,
    jitter: float = 1e-10,
) -> GaussianProcessPosterior:
    """Compute an exact latent GP posterior using Cholesky solves."""

    query = _points(query_points, name="query_points")
    training = _points(
        training_points,
        name="training_points",
        expected_dimension=query.shape[1],
    )
    values = np.asarray(training_values, dtype=float).reshape(-1)
    if values.shape != (training.shape[0],):
        raise ValueError("training_values must contain one value per training point")
    if not np.isfinite(values).all():
        raise ValueError("training_values must be finite")
    prior_mean = float(mean)
    if not math.isfinite(prior_mean):
        raise ValueError("mean must be finite")
    noise = _nonnegative_finite(noise_variance, name="noise_variance")

    query_cov = np.asarray(kernel(query, query), dtype=float)
    if query_cov.shape != (query.shape[0], query.shape[0]):
        raise ValueError("kernel(query, query) returned an invalid matrix shape")
    query_cov = 0.5 * (query_cov + query_cov.T)

    if training.shape[0] == 0:
        variance = np.maximum(np.diag(query_cov), 0.0)
        return GaussianProcessPosterior(
            mean=_readonly(np.full(query.shape[0], prior_mean, dtype=float)),
            covariance=_readonly(query_cov),
            variance=_readonly(variance),
            applied_jitter=0.0,
        )

    train_cov = np.asarray(kernel(training, training), dtype=float)
    if train_cov.shape != (training.shape[0], training.shape[0]):
        raise ValueError("kernel(training, training) returned an invalid matrix shape")
    train_cov = 0.5 * (train_cov + train_cov.T)
    train_cov += noise * np.eye(training.shape[0], dtype=float)
    chol, applied_jitter = _stable_cholesky(train_cov, jitter=jitter)

    train_query_cov = np.asarray(kernel(training, query), dtype=float)
    if train_query_cov.shape != (training.shape[0], query.shape[0]):
        raise ValueError("kernel(training, query) returned an invalid matrix shape")
    centered_values = values - prior_mean
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, centered_values))
    posterior_mean = prior_mean + train_query_cov.T @ alpha
    projected = np.linalg.solve(chol, train_query_cov)
    posterior_covariance = query_cov - projected.T @ projected
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    diagonal = np.diag(posterior_covariance).copy()
    if np.any(diagonal < -1e-7):
        raise ValueError("posterior covariance has a materially negative diagonal")
    diagonal = np.maximum(diagonal, 0.0)
    if diagonal.size:
        posterior_covariance[np.diag_indices_from(posterior_covariance)] = diagonal

    return GaussianProcessPosterior(
        mean=_readonly(posterior_mean),
        covariance=_readonly(posterior_covariance),
        variance=_readonly(diagonal),
        applied_jitter=float(applied_jitter),
    )
