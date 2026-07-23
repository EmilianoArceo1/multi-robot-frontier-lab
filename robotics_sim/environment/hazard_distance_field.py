"""Signed Distance Field built from the team's DISCOVERED hazard belief.

Grounded in Raja et al., "OGM-CBF: Occupancy Grid Map-based Control Barrier
Function for Safe Mobile Robot Control with Memory of out of View
Obstacles" -- specifically its ideas of a persistent grid, memory of
hazards that are currently out of the sensor field of view, converting the
occupancy grid into a Signed Distance Field to get continuous geometry out
of a discrete map, and using multiple scales (a pyramid) of that distance
field.

This module ONLY builds continuous geometry out of an already-observed
``HazardBelief`` snapshot (a ``HazardBeliefFrame``). It never reads ground
truth (``HazardField``/``FireSource``) and performs no CBF/QP math -- see
the active cited safety controller for that. A hazard the team has
never observed can never appear in the unsafe mask; a hazard the team
observed earlier and is now outside every robot's field of view stays in
the mask until the belief itself says otherwise, because both are properties
of ``HazardBeliefFrame`` (see hazard_belief.py), not of this module.

Unsafe mask (the ONLY rule used to decide what is geometry here):

    unsafe = belief_frame.observed AND belief_frame.values >= block_threshold

Approximation notice (documented, not hidden): the base-level field is a
brute-force, cell-center Euclidean-ish signed distance transform -- positive
outside the unsafe set, negative inside, in meters. Coarser pyramid levels
are a Gaussian blur of the finer level followed by a 2x spatial downsample
(an image-pyramid style approximation, not a re-derived exact distance
transform at coarser resolution). Gradients/Hessians are finite-difference
estimates (central differences in the interior, one-sided at borders, via
``numpy.gradient``) of that discretized field. None of this is an exact
continuous geometric distance function, so it carries no exact continuous-
time mathematical guarantee by itself -- consistent with OGM-CBF's own
grid-derived SDF, which is likewise a numerical approximation of the true
environment geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from robotics_sim.environment.grid_geometry import GridGeometry


@dataclass(frozen=True)
class HazardDistanceSample:
    """One interpolated sample of the distance field at one pyramid level."""

    value: float
    gradient: np.ndarray
    hessian: np.ndarray
    level: int


@dataclass(frozen=True)
class _HazardDistanceLevel:
    """Precomputed per-level arrays -- internal storage for one pyramid level."""

    level: int
    resolution: float
    x_centers: np.ndarray
    y_centers: np.ndarray
    value: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    hessian_xx: np.ndarray
    hessian_xy: np.ndarray
    hessian_yy: np.ndarray


@dataclass(frozen=True)
class HazardDistanceFieldFrame:
    """Immutable, cacheable Signed Distance Field pyramid.

    ``revision`` is the ``HazardBeliefFrame.revision`` at which this frame's
    arrays were actually (re)computed -- NOT necessarily the revision of the
    belief frame passed to the builder on a given call, since a reused frame
    (identical blocked mask) keeps the revision of when it was last actually
    rebuilt (see ``HazardDistanceFieldBuilder.build()``).
    """

    revision: int
    blocked_signature: object
    levels: tuple[_HazardDistanceLevel, ...]
    bounds: tuple[float, float, float, float]
    base_resolution: float
    has_hazards: bool

    def sample(self, x: float, y: float) -> tuple[HazardDistanceSample, ...]:
        """Interpolate value/gradient/Hessian at world point (x, y) for every
        pyramid level. Returns an empty tuple when there are no observed
        hazards at or above the block threshold -- callers must treat that
        as "no constraints", not as "zero distance everywhere"."""
        if not self.has_hazards:
            return ()
        return tuple(_sample_level(level, float(x), float(y)) for level in self.levels)


class HazardDistanceFieldBuilder:
    """Builds (and lets callers reuse) a ``HazardDistanceFieldFrame`` from a
    ``HazardBeliefFrame`` snapshot -- never from ``HazardBelief`` directly,
    so the caller controls exactly which snapshot's revision this reflects.
    """

    def build(
        self,
        *,
        belief_frame,
        geometry: GridGeometry,
        block_threshold: float,
        pyramid_levels: int = 2,
        smoothing_sigma_cells: float = 0.75,
        previous_frame: HazardDistanceFieldFrame | None = None,
    ) -> HazardDistanceFieldFrame:
        unsafe_mask = belief_frame.observed & (belief_frame.values >= float(block_threshold))
        signature = unsafe_mask.tobytes()
        has_hazards = bool(unsafe_mask.any())

        bounds = geometry.bounds
        resolution = float(geometry.resolution)

        if (
            previous_frame is not None
            and previous_frame.blocked_signature == signature
            and previous_frame.bounds == bounds
            and abs(previous_frame.base_resolution - resolution) < 1e-12
        ):
            # Same unsafe geometry -- reuse exactly, including its own
            # revision, per requirement 7/9: a belief revision bump that
            # does not change the blocked mask (e.g. re-attribution by a
            # different robot) must not trigger a rebuild.
            return previous_frame

        if not has_hazards:
            return HazardDistanceFieldFrame(
                revision=int(belief_frame.revision),
                blocked_signature=signature,
                levels=(),
                bounds=bounds,
                base_resolution=resolution,
                has_hazards=False,
            )

        levels = _build_pyramid(
            unsafe_mask,
            geometry,
            pyramid_levels=max(1, min(4, int(pyramid_levels))),
            smoothing_sigma_cells=max(0.0, float(smoothing_sigma_cells)),
        )

        return HazardDistanceFieldFrame(
            revision=int(belief_frame.revision),
            blocked_signature=signature,
            levels=levels,
            bounds=bounds,
            base_resolution=resolution,
            has_hazards=True,
        )


# ============================================================
# Pyramid construction (module-private)
# ============================================================


def _build_pyramid(
    unsafe_mask: np.ndarray,
    geometry: GridGeometry,
    *,
    pyramid_levels: int,
    smoothing_sigma_cells: float,
) -> tuple[_HazardDistanceLevel, ...]:
    resolution = float(geometry.resolution)
    x_centers = geometry.x_min + (np.arange(geometry.width, dtype=np.float64) + 0.5) * resolution
    y_centers = geometry.y_min + (np.arange(geometry.height, dtype=np.float64) + 0.5) * resolution

    phi = _signed_distance(unsafe_mask, x_centers, y_centers)
    levels = [_level_from_field(0, phi, resolution, x_centers, y_centers)]

    for level_index in range(1, pyramid_levels):
        if phi.shape[0] < 2 or phi.shape[1] < 2:
            break
        blurred = _gaussian_blur_separable(phi, smoothing_sigma_cells)
        phi, x_centers, y_centers = _downsample2(blurred, x_centers, y_centers)
        resolution *= 2.0
        levels.append(_level_from_field(level_index, phi, resolution, x_centers, y_centers))

    return tuple(levels)


def _signed_distance(unsafe_mask: np.ndarray, x_centers: np.ndarray, y_centers: np.ndarray) -> np.ndarray:
    height, width = unsafe_mask.shape
    xx, yy = np.meshgrid(x_centers, y_centers)
    flat_x = xx.reshape(-1)
    flat_y = yy.reshape(-1)

    unsafe_rows, unsafe_cols = np.nonzero(unsafe_mask)
    free_rows, free_cols = np.nonzero(~unsafe_mask)

    # Finite fallback distance for the degenerate case where one side of the
    # mask has no cells at all (e.g. the whole grid is unsafe) -- guarantees
    # every sample stays finite (requirement 16) instead of producing inf.
    fallback = float(math.hypot(x_centers[-1] - x_centers[0], y_centers[-1] - y_centers[0])) + 1.0

    dist_to_unsafe = _nearest_seed_distance(
        flat_x, flat_y, x_centers[unsafe_cols], y_centers[unsafe_rows], fallback
    ).reshape(height, width)
    dist_to_free = _nearest_seed_distance(
        flat_x, flat_y, x_centers[free_cols], y_centers[free_rows], fallback
    ).reshape(height, width)

    sdf = np.where(unsafe_mask, -dist_to_free, dist_to_unsafe)
    return sdf.astype(np.float64, copy=False)


def _nearest_seed_distance(
    query_x: np.ndarray,
    query_y: np.ndarray,
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    fallback: float,
    batch_size: int = 256,
) -> np.ndarray:
    best = np.full(query_x.shape[0], fallback, dtype=np.float64)
    n_seeds = seed_x.shape[0]
    if n_seeds == 0:
        return best

    for start in range(0, n_seeds, batch_size):
        sx = seed_x[start : start + batch_size]
        sy = seed_y[start : start + batch_size]
        dx = query_x[:, None] - sx[None, :]
        dy = query_y[:, None] - sy[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.minimum(best, dist.min(axis=1), out=best)

    return best


def _gaussian_kernel_1d(sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(3.0 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel /= kernel.sum()
    return kernel


def _convolve1d_edge(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1D convolution along ``axis`` with edge-replicated padding."""
    radius = kernel.shape[0] // 2
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(array, pad_width, mode="edge")

    moved = np.moveaxis(padded, axis, 0)
    out_len = moved.shape[0] - 2 * radius
    result = np.zeros((out_len,) + moved.shape[1:], dtype=np.float64)
    for offset, weight in enumerate(kernel):
        result += weight * moved[offset : offset + out_len]
    return np.moveaxis(result, 0, axis)


def _gaussian_blur_separable(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0.0:
        return field
    kernel = _gaussian_kernel_1d(sigma_cells)
    blurred = _convolve1d_edge(field, kernel, axis=0)
    blurred = _convolve1d_edge(blurred, kernel, axis=1)
    return blurred


def _downsample2(
    field: np.ndarray, x_centers: np.ndarray, y_centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = field.shape
    even_h = height - (height % 2)
    even_w = width - (width % 2)

    cropped = field[:even_h, :even_w]
    pooled = cropped.reshape(even_h // 2, 2, even_w // 2, 2).mean(axis=(1, 3))

    new_x_centers = 0.5 * (x_centers[:even_w:2] + x_centers[1:even_w:2])
    new_y_centers = 0.5 * (y_centers[:even_h:2] + y_centers[1:even_h:2])

    return pooled, new_x_centers, new_y_centers


def _safe_gradient2d(field: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """``np.gradient`` with degenerate (size-1) axes padded first -- np.gradient
    itself requires at least 2 samples per axis. Central differences in the
    interior, one-sided at borders, exactly as ``np.gradient`` already does
    (requirements 13/14)."""
    height, width = field.shape
    pad_h = 1 if height < 2 else 0
    pad_w = 1 if width < 2 else 0

    padded = field
    if pad_h or pad_w:
        padded = np.pad(field, ((0, pad_h), (0, pad_w)), mode="edge")

    grad_y, grad_x = np.gradient(padded, resolution, resolution)

    if pad_h or pad_w:
        grad_y = grad_y[:height, :width]
        grad_x = grad_x[:height, :width]

    return grad_y, grad_x


def _level_from_field(
    level_index: int,
    phi: np.ndarray,
    resolution: float,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> _HazardDistanceLevel:
    phi = np.asarray(phi, dtype=np.float64)
    grad_y, grad_x = _safe_gradient2d(phi, resolution)
    dgx_dy, hxx = _safe_gradient2d(grad_x, resolution)
    hyy, dgy_dx = _safe_gradient2d(grad_y, resolution)
    hxy = 0.5 * (dgx_dy + dgy_dx)

    return _HazardDistanceLevel(
        level=int(level_index),
        resolution=float(resolution),
        x_centers=np.asarray(x_centers, dtype=np.float64),
        y_centers=np.asarray(y_centers, dtype=np.float64),
        value=phi,
        gradient_x=grad_x,
        gradient_y=grad_y,
        hessian_xx=hxx,
        hessian_xy=hxy,
        hessian_yy=hyy,
    )


def _clamp_index_and_frac(centers: np.ndarray, value: float) -> tuple[int, int, float]:
    n = centers.shape[0]
    if n == 1:
        return 0, 0, 0.0
    if value <= centers[0]:
        return 0, 1, 0.0
    if value >= centers[-1]:
        return n - 2, n - 1, 1.0

    idx = int(np.searchsorted(centers, value, side="right") - 1)
    idx = max(0, min(idx, n - 2))
    span = centers[idx + 1] - centers[idx]
    frac = 0.0 if span <= 0 else (value - centers[idx]) / span
    return idx, idx + 1, float(np.clip(frac, 0.0, 1.0))


def _bilinear(array2d: np.ndarray, x_centers: np.ndarray, y_centers: np.ndarray, x: float, y: float) -> float:
    xi0, xi1, fx = _clamp_index_and_frac(x_centers, x)
    yi0, yi1, fy = _clamp_index_and_frac(y_centers, y)

    v00 = array2d[yi0, xi0]
    v01 = array2d[yi0, xi1]
    v10 = array2d[yi1, xi0]
    v11 = array2d[yi1, xi1]

    top = v00 * (1.0 - fx) + v01 * fx
    bottom = v10 * (1.0 - fx) + v11 * fx
    return float(top * (1.0 - fy) + bottom * fy)


def _sample_level(level: _HazardDistanceLevel, x: float, y: float) -> HazardDistanceSample:
    value = _bilinear(level.value, level.x_centers, level.y_centers, x, y)
    gx = _bilinear(level.gradient_x, level.x_centers, level.y_centers, x, y)
    gy = _bilinear(level.gradient_y, level.x_centers, level.y_centers, x, y)
    hxx = _bilinear(level.hessian_xx, level.x_centers, level.y_centers, x, y)
    hxy = _bilinear(level.hessian_xy, level.x_centers, level.y_centers, x, y)
    hyy = _bilinear(level.hessian_yy, level.x_centers, level.y_centers, x, y)

    gradient = np.array([gx, gy], dtype=np.float64)
    hessian = np.array([[hxx, hxy], [hxy, hyy]], dtype=np.float64)

    return HazardDistanceSample(value=value, gradient=gradient, hessian=hessian, level=level.level)
