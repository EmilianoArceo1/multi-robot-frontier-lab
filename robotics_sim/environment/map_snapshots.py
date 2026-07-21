"""Immutable environment map-layer snapshot contracts.

These are simulator-HOST-internal contracts, not robotics_interfaces
contracts. External coordination plugins must never receive a live
BeliefMap, an internal OccupancyGrid, raw mapped_obstacle_points, or
engine.py -- they receive host-computed results instead (e.g. future
FrontierObservation/RouteEvaluation/RouteReservation/HazardBeliefQuery/
CoordinationResult contracts in robotics_interfaces). These snapshots exist
so different simulator-internal modules (belief map, sensor mapping,
planning) can agree on one validated, read-only shape for the same data,
without leaking a mutable BeliefMap/OccupancyGrid reference around.

    - ``ObservedObstacleSnapshot``   partial obstacle geometry a sensor has
                                      actually observed (points, not ground
                                      truth).
    - ``ExplorationMapSnapshot``     the team's discrete UNKNOWN/FREE/OCCUPIED
                                      knowledge of the world (an exploration
                                      belief grid), not exact collision
                                      geometry.

Nothing here is wired to a producer or a consumer yet -- these are contracts
only. No simulator runtime behavior changes when this module is added.

Boundary rule: this module may import numpy,
robotics_interfaces.observations (for the shared Point2D/WorldBounds
aliases), and robotics_sim.environment.grid_geometry (the project's single
source of truth for world<->grid conversion), and may be imported by other
robotics_sim modules. It must never import Qt/PySide/PyQt, MainWindow,
canvas, or engine.py.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from robotics_interfaces.observations import Point2D, WorldBounds
from robotics_sim.environment.grid_geometry import GridGeometry

# Mirrors robotics_sim.environment.belief_map/occupancy_grid's cell-state
# convention.
UNKNOWN = -1
FREE = 0
OCCUPIED = 1

_ALLOWED_GRID_VALUES = (UNKNOWN, FREE, OCCUPIED)


# ---------------------------------------------------------------------------
# Validation helpers (module-private)
# ---------------------------------------------------------------------------


def _validate_bounds(bounds) -> WorldBounds:
    try:
        x_min, x_max, y_min, y_max = bounds
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"bounds must be a 4-tuple (x_min, x_max, y_min, y_max), got {bounds!r}."
        ) from exc

    try:
        x_min, x_max, y_min, y_max = (float(x_min), float(x_max), float(y_min), float(y_max))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bounds values must be numeric, got {bounds!r}.") from exc

    if not all(math.isfinite(v) for v in (x_min, x_max, y_min, y_max)):
        raise ValueError(f"bounds values must be finite, got {bounds!r}.")

    if not x_max > x_min:
        raise ValueError(f"bounds x_max must be greater than x_min, got {bounds!r}.")

    if not y_max > y_min:
        raise ValueError(f"bounds y_max must be greater than y_min, got {bounds!r}.")

    return (x_min, x_max, y_min, y_max)


def _validate_resolution(resolution) -> float:
    try:
        value = float(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"resolution must be numeric, got {resolution!r}.") from exc

    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"resolution must be finite and > 0, got {resolution!r}.")

    return value


def _validate_revision(revision) -> int:
    """Accept only real, non-boolean integers -- never silently truncate a
    float (1.8 -> 1) and never accept bool (a subclass of int in Python, so
    it would otherwise pass a bare `isinstance(x, int)`/`int(x)` check).

    numpy integer scalars (e.g. numpy.int64) are accepted and normalized to
    plain int, since numbers.Integral already covers them.
    """
    if isinstance(revision, bool) or not isinstance(revision, numbers.Integral):
        raise ValueError(
            f"revision must be a non-boolean integer, got {revision!r} ({type(revision).__name__})."
        )

    value = int(revision)
    if value < 0:
        raise ValueError(f"revision must be >= 0, got {revision!r}.")

    return value


def _validate_points(points) -> tuple[Point2D, ...]:
    """Validate and tuple-ize a point sequence without altering coordinates.

    Converts each coordinate to float (a lossless type normalization, not a
    geometric one) and rejects non-finite values. Never rounds, clamps, or
    snaps a point to any grid -- that is not this contract's job.
    """
    try:
        raw_points = list(points)
    except TypeError as exc:
        raise ValueError(f"points must be an iterable of (x, y) pairs, got {points!r}.") from exc

    validated: list[Point2D] = []
    for point in raw_points:
        try:
            x, y = point
        except (TypeError, ValueError) as exc:
            raise ValueError(f"each point must be an (x, y) pair, got {point!r}.") from exc

        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"point coordinates must be numeric, got {point!r}.") from exc

        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"point coordinates must be finite, got {point!r}.")

        validated.append((x, y))

    return tuple(validated)


def _validate_source(source) -> str:
    """Accept only an already-string, non-blank source label.

    Deliberately does not coerce (None -> "None", 123 -> "123") -- a caller
    passing a non-string source almost certainly has a bug, and silently
    stringifying it would hide that instead of failing loudly.
    """
    if not isinstance(source, str):
        raise ValueError(f"source must be a string, got {source!r} ({type(source).__name__}).")

    stripped = source.strip()
    if not stripped:
        raise ValueError(f"source must not be empty or whitespace-only, got {source!r}.")

    return stripped


def _freeze_grid(grid) -> np.ndarray:
    """Copy, validate, and permanently freeze an UNKNOWN/FREE/OCCUPIED grid.

    Values are validated BEFORE the dtype cast to int8, so an input that
    cannot exactly represent -1/0/1 (e.g. a stray 0.5) is rejected instead of
    silently truncated. The returned array is always a fresh, independent
    copy -- mutating the caller's original array after construction never
    affects it.
    """
    try:
        copied = np.array(grid, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grid must be array-like, got {grid!r}.") from exc

    if copied.ndim != 2:
        raise ValueError(f"grid must be a 2D array, got shape {copied.shape!r}.")

    if not np.isin(copied, _ALLOWED_GRID_VALUES).all():
        bad_values = sorted(set(np.unique(copied).tolist()) - set(_ALLOWED_GRID_VALUES))
        raise ValueError(f"grid contains values outside {{-1, 0, 1}}: {bad_values!r}.")

    frozen = copied.astype(np.int8, copy=False)
    frozen.setflags(write=False)
    return frozen


def _validate_grid_matches_geometry(grid: np.ndarray, bounds: WorldBounds, resolution: float) -> None:
    """Ensure grid.shape agrees with the (height, width) GridGeometry(bounds,
    resolution) implies -- GridGeometry is the project's single source of
    truth for world<->grid conversion (see its own module docstring), so
    this reuses it instead of re-deriving ceil/width/height locally.
    """
    geometry = GridGeometry(bounds, resolution)
    expected_shape = (geometry.height, geometry.width)

    if grid.shape != expected_shape:
        raise ValueError(
            f"grid shape {grid.shape!r} does not match the shape implied by "
            f"bounds={bounds!r} and resolution={resolution!r}: expected "
            f"shape (height, width)={expected_shape!r}."
        )


# ---------------------------------------------------------------------------
# Contract: Observed obstacle geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservedObstacleSnapshot:
    """Partial obstacle geometry a sensor has actually observed.

    points is sparse, sensor-derived boundary geometry -- never ground
    truth. This snapshot deliberately knows nothing about robot radius or
    inflation (that is a planning/safety concern, applied by consumers), and
    carries no hazard or dynamic-robot points -- those are separate
    contracts.

    revision identifies the producer's version of this data. It is supplied
    by the caller, not derived here: `revision = len(points)` is NOT a valid
    strategy (it breaks after a reset/restore, and two different maps can
    happen to have the same point count). Deciding how revision increments
    is an adapter's responsibility, not this contract's.
    """

    points: tuple[Point2D, ...]
    bounds: WorldBounds
    resolution: float
    revision: int
    source: str = "observed_obstacles"

    def __post_init__(self) -> None:
        validated_points = _validate_points(self.points)
        validated_bounds = _validate_bounds(self.bounds)
        validated_resolution = _validate_resolution(self.resolution)
        validated_revision = _validate_revision(self.revision)
        validated_source = _validate_source(self.source)

        object.__setattr__(self, "points", validated_points)
        object.__setattr__(self, "bounds", validated_bounds)
        object.__setattr__(self, "resolution", validated_resolution)
        object.__setattr__(self, "revision", validated_revision)
        object.__setattr__(self, "source", validated_source)


# ---------------------------------------------------------------------------
# Contract: Exploration belief grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplorationMapSnapshot:
    """The team's discrete UNKNOWN/FREE/OCCUPIED exploration knowledge.

    Cell states:
        UNKNOWN  = -1
        FREE     = 0
        OCCUPIED = 1

    This represents *observed knowledge*, not exact continuous collision
    geometry -- a cell being OCCUPIED means the team has evidence something
    is there, not a precise boundary. Frontier detection, coverage, and
    rasterization are deliberately NOT methods here; this is a data
    contract, not an algorithm host.
    """

    grid: np.ndarray
    bounds: WorldBounds
    resolution: float
    revision: int

    def __post_init__(self) -> None:
        frozen_grid = _freeze_grid(self.grid)
        validated_bounds = _validate_bounds(self.bounds)
        validated_resolution = _validate_resolution(self.resolution)
        validated_revision = _validate_revision(self.revision)
        _validate_grid_matches_geometry(frozen_grid, validated_bounds, validated_resolution)

        object.__setattr__(self, "grid", frozen_grid)
        object.__setattr__(self, "bounds", validated_bounds)
        object.__setattr__(self, "resolution", validated_resolution)
        object.__setattr__(self, "revision", validated_revision)

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)
