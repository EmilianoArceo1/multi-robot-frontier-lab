"""Pure loader for RSS26 informative-path-planning visualization bundles.

The format deliberately keeps large numeric assets outside ``.sim`` files and
outside the Qt/runtime layers.  A bundle directory contains a JSON manifest
and one relative NPZ archive::

    {
      "schema": "robotics_sim.rss26_ipp_bundle",
      "version": 1,
      "npz": "assets/N47W124.npz",
      "dataset_bounds": [0.0, 400.0, 0.0, 300.0],
      "raster_origin": "lower",
      "arrays": {
        "field": "field",
        "prior_variance": "prior_variance",
        "posterior_variance": "posterior_variance",
        "mask": "mask",
        "pilot_path": "pilot_path",
        "solution_path": "solution_path",
        "sensing_points": "sensing_points",
        "fovs": "fovs"
      },
      "metrics": {"method": "GreedyCover", "distance_m": 238.0}
    }

``arrays`` is optional when the NPZ uses the canonical names shown above.
Raster arrays are 2-D and share one shape.  Coordinate arrays use dataset
coordinates: paths and sensing points have shape ``(N, 2)``; FoV polygons have
shape ``(K, V, 2)`` with a fixed vertex count per polygon.  Loaded coordinate
arrays are exposed in aspect-fitted simulator-world coordinates.  The
``AspectFitTransform`` retained on the result converts in both directions.

This module intentionally imports neither Qt nor simulation config/engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


IPP_BUNDLE_SCHEMA = "robotics_sim.rss26_ipp_bundle"
IPP_BUNDLE_VERSION = 1
DEFAULT_IPP_WORLD_BOUNDS = (-10.0, 10.0, -8.0, 8.0)

_ARRAY_ROLES = (
    "field",
    "prior_variance",
    "posterior_variance",
    "mask",
    "pilot_path",
    "solution_path",
    "sensing_points",
    "fovs",
)


class IppBundleError(ValueError):
    """Raised when an IPP visualization bundle violates the contract."""


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _bounds(value: Any, *, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise IppBundleError(f"{label} must be [x_min, x_max, y_min, y_max].")
    try:
        x_min, x_max, y_min, y_max = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise IppBundleError(f"{label} must contain four real numbers.") from exc
    if not all(math.isfinite(item) for item in (x_min, x_max, y_min, y_max)):
        raise IppBundleError(f"{label} must contain only finite values.")
    if x_max <= x_min or y_max <= y_min:
        raise IppBundleError(f"{label} must have strictly increasing axes.")
    return x_min, x_max, y_min, y_max


def _points(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise IppBundleError(f"{label} must have shape (N, 2); got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise IppBundleError(f"{label} must contain real numeric coordinates.")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise IppBundleError(f"{label} must contain only finite coordinates.")
    return result


def _fov_polygons(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2:] != (2,):
        raise IppBundleError(f"fovs must have shape (K, V, 2); got {array.shape}.")
    if array.shape[0] > 0 and array.shape[1] < 3:
        raise IppBundleError("each FoV polygon must have at least three vertices.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise IppBundleError("fovs must contain real numeric coordinates.")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise IppBundleError("fovs must contain only finite coordinates.")
    return result


def _real_grid(value: Any, *, label: str, nonnegative: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.size == 0:
        raise IppBundleError(f"{label} must be a non-empty 2-D array; got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise IppBundleError(f"{label} must contain real numeric values.")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise IppBundleError(f"{label} must contain only finite values.")
    if nonnegative and np.any(result < 0.0):
        raise IppBundleError(f"{label} cannot contain negative variance.")
    return result


def _mask(value: Any, *, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise IppBundleError(
            f"mask must match raster shape {expected_shape}; got {array.shape}."
        )
    if np.issubdtype(array.dtype, np.bool_):
        result = np.asarray(array, dtype=np.bool_)
    else:
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise IppBundleError("mask must be boolean or contain only 0/1 values.")
        if not np.all(np.isfinite(array)) or not np.all((array == 0) | (array == 1)):
            raise IppBundleError("mask must be finite and contain only 0/1 values.")
        result = np.asarray(array, dtype=np.bool_)
    if not np.any(result):
        raise IppBundleError("mask must contain at least one valid dataset cell.")
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _validate_json_value(value: Any, *, label: str) -> Any:
    """Validate and detach metrics while keeping ordinary JSON containers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IppBundleError(f"{label} contains a non-finite number.")
        return value
    if isinstance(value, list):
        return tuple(
            _validate_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise IppBundleError(f"{label} object keys must be strings.")
        return MappingProxyType(
            {
                key: _validate_json_value(item, label=f"{label}.{key}")
                for key, item in value.items()
            }
        )
    raise IppBundleError(f"{label} must contain JSON-compatible values.")


def _safe_relative_npz(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise IppBundleError("manifest field 'npz' must be a non-empty relative path.")
    relative = Path(value)
    if relative.is_absolute():
        raise IppBundleError("manifest field 'npz' must be relative to the manifest.")
    base = base.resolve(strict=True)
    try:
        candidate = (base / relative).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IppBundleError(f"NPZ asset does not exist or cannot be resolved: {value!r}.") from exc
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise IppBundleError("NPZ asset escapes the manifest directory.") from exc
    if candidate.suffix.lower() != ".npz" or not candidate.is_file():
        raise IppBundleError("manifest field 'npz' must reference an existing .npz file.")
    return candidate


@dataclass(frozen=True)
class AspectFitTransform:
    """Uniform, centered dataset/world coordinate transform."""

    dataset_bounds: tuple[float, float, float, float]
    world_bounds: tuple[float, float, float, float]
    scale: float
    data_world_bounds: tuple[float, float, float, float]

    @classmethod
    def create(
        cls,
        dataset_bounds: tuple[float, float, float, float],
        world_bounds: tuple[float, float, float, float] = DEFAULT_IPP_WORLD_BOUNDS,
    ) -> "AspectFitTransform":
        dataset_bounds = _bounds(dataset_bounds, label="dataset_bounds")
        world_bounds = _bounds(world_bounds, label="world_bounds")
        dx0, dx1, dy0, dy1 = dataset_bounds
        wx0, wx1, wy0, wy1 = world_bounds
        scale = min((wx1 - wx0) / (dx1 - dx0), (wy1 - wy0) / (dy1 - dy0))
        wcx, wcy = (wx0 + wx1) / 2.0, (wy0 + wy1) / 2.0
        fitted_width = (dx1 - dx0) * scale
        fitted_height = (dy1 - dy0) * scale
        data_world_bounds = (
            wcx - fitted_width / 2.0,
            wcx + fitted_width / 2.0,
            wcy - fitted_height / 2.0,
            wcy + fitted_height / 2.0,
        )
        return cls(dataset_bounds, world_bounds, float(scale), data_world_bounds)

    def dataset_to_world(self, coordinates: Any) -> np.ndarray:
        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise IppBundleError("coordinates must have shape (..., 2).")
        if not np.all(np.isfinite(points)):
            raise IppBundleError("coordinates must contain only finite values.")
        dx0, dx1, dy0, dy1 = self.dataset_bounds
        wx0, wx1, wy0, wy1 = self.world_bounds
        result = np.array(points, copy=True)
        result[..., 0] = (points[..., 0] - (dx0 + dx1) / 2.0) * self.scale + (wx0 + wx1) / 2.0
        result[..., 1] = (points[..., 1] - (dy0 + dy1) / 2.0) * self.scale + (wy0 + wy1) / 2.0
        return result

    def world_to_dataset(self, coordinates: Any) -> np.ndarray:
        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim < 1 or points.shape[-1] != 2:
            raise IppBundleError("coordinates must have shape (..., 2).")
        if not np.all(np.isfinite(points)):
            raise IppBundleError("coordinates must contain only finite values.")
        dx0, dx1, dy0, dy1 = self.dataset_bounds
        wx0, wx1, wy0, wy1 = self.world_bounds
        result = np.array(points, copy=True)
        result[..., 0] = (points[..., 0] - (wx0 + wx1) / 2.0) / self.scale + (dx0 + dx1) / 2.0
        result[..., 1] = (points[..., 1] - (wy0 + wy1) / 2.0) / self.scale + (dy0 + dy1) / 2.0
        return result


@dataclass(frozen=True)
class IppVisualizationBundle:
    """Validated, immutable visualization data in simulator coordinates."""

    manifest_path: Path
    npz_path: Path
    transform: AspectFitTransform
    raster_origin: str
    field: np.ndarray
    prior_variance: np.ndarray
    posterior_variance: np.ndarray
    mask: np.ndarray
    pilot_path: np.ndarray
    solution_path: np.ndarray
    sensing_points: np.ndarray
    fovs: np.ndarray
    metrics: Mapping[str, Any]

    @property
    def dataset_bounds(self) -> tuple[float, float, float, float]:
        return self.transform.dataset_bounds

    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        return self.transform.world_bounds

    @property
    def data_world_bounds(self) -> tuple[float, float, float, float]:
        return self.transform.data_world_bounds

    def dataset_to_world(self, coordinates: Any) -> np.ndarray:
        return self.transform.dataset_to_world(coordinates)

    def world_to_dataset(self, coordinates: Any) -> np.ndarray:
        return self.transform.world_to_dataset(coordinates)


def load_ipp_bundle(
    manifest_path: str | Path,
    *,
    world_bounds: tuple[float, float, float, float] = DEFAULT_IPP_WORLD_BOUNDS,
) -> IppVisualizationBundle:
    """Load and validate an RSS26 IPP visualization bundle.

    Only the explicitly referenced relative NPZ is opened, with NumPy pickle
    loading disabled.  The resolved path must remain beneath the manifest's
    directory, including after resolving symlinks.
    """
    try:
        manifest_path = Path(manifest_path).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IppBundleError("manifest does not exist or cannot be resolved.") from exc
    if not manifest_path.is_file():
        raise IppBundleError("manifest_path must reference a JSON file.")

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise IppBundleError(f"invalid IPP bundle manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise IppBundleError("IPP bundle manifest must be a JSON object.")
    if manifest.get("schema") != IPP_BUNDLE_SCHEMA:
        raise IppBundleError(f"unsupported IPP bundle schema: {manifest.get('schema')!r}.")
    version = manifest.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version != IPP_BUNDLE_VERSION:
        raise IppBundleError(f"unsupported IPP bundle version: {version!r}.")

    dataset_bounds = _bounds(manifest.get("dataset_bounds"), label="dataset_bounds")
    transform = AspectFitTransform.create(dataset_bounds, _bounds(world_bounds, label="world_bounds"))
    raster_origin = manifest.get("raster_origin", "lower")
    if raster_origin not in {"lower", "upper"}:
        raise IppBundleError("raster_origin must be either 'lower' or 'upper'.")

    array_names = manifest.get("arrays", {})
    if not isinstance(array_names, dict):
        raise IppBundleError("manifest field 'arrays' must be a JSON object.")
    unknown_roles = set(array_names) - set(_ARRAY_ROLES)
    if unknown_roles:
        raise IppBundleError(f"unknown array roles: {sorted(unknown_roles)!r}.")
    resolved_names: dict[str, str] = {}
    for role in _ARRAY_ROLES:
        name = array_names.get(role, role)
        if not isinstance(name, str) or not name:
            raise IppBundleError(f"array name for {role!r} must be a non-empty string.")
        resolved_names[role] = name

    npz_path = _safe_relative_npz(manifest_path.parent, manifest.get("npz"))
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = [name for name in resolved_names.values() if name not in archive.files]
            if missing:
                raise IppBundleError(f"NPZ is missing required arrays: {sorted(set(missing))!r}.")
            raw = {role: np.array(archive[name], copy=True) for role, name in resolved_names.items()}
    except IppBundleError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise IppBundleError(f"could not read NPZ asset safely: {exc}") from exc

    field = _real_grid(raw["field"], label="field")
    raster_shape = tuple(int(size) for size in field.shape)
    prior_variance = _real_grid(raw["prior_variance"], label="prior_variance", nonnegative=True)
    posterior_variance = _real_grid(
        raw["posterior_variance"], label="posterior_variance", nonnegative=True
    )
    for label, array in (
        ("prior_variance", prior_variance),
        ("posterior_variance", posterior_variance),
    ):
        if array.shape != raster_shape:
            raise IppBundleError(
                f"{label} must match field shape {raster_shape}; got {array.shape}."
            )
    mask = _mask(raw["mask"], expected_shape=raster_shape)

    pilot_path_dataset = _points(raw["pilot_path"], label="pilot_path")
    solution_path_dataset = _points(raw["solution_path"], label="solution_path")
    sensing_points_dataset = _points(raw["sensing_points"], label="sensing_points")
    fovs_dataset = _fov_polygons(raw["fovs"])

    metrics_raw = manifest.get("metrics", {})
    if not isinstance(metrics_raw, dict):
        raise IppBundleError("manifest field 'metrics' must be a JSON object.")
    metrics = _validate_json_value(metrics_raw, label="metrics")

    return IppVisualizationBundle(
        manifest_path=manifest_path,
        npz_path=npz_path,
        transform=transform,
        raster_origin=str(raster_origin),
        field=_readonly(field),
        prior_variance=_readonly(prior_variance),
        posterior_variance=_readonly(posterior_variance),
        mask=_readonly(mask),
        pilot_path=_readonly(transform.dataset_to_world(pilot_path_dataset)),
        solution_path=_readonly(transform.dataset_to_world(solution_path_dataset)),
        sensing_points=_readonly(transform.dataset_to_world(sensing_points_dataset)),
        fovs=_readonly(transform.dataset_to_world(fovs_dataset)),
        metrics=metrics,
    )
