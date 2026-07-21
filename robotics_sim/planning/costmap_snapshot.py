"""Immutable planning costmap snapshot contract.

This is a simulator-HOST-internal contract, not a robotics_interfaces
contract. External coordination plugins must never receive this snapshot,
the internal OccupancyGrid it is derived from, or engine.py -- the simulator
keeps sole responsibility for building the costmap, applying inflation, and
running A*/Dijkstra. Plugins consume host-computed results instead (future
FrontierObservation/RouteEvaluation/RouteReservation/HazardBeliefQuery/
CoordinationResult contracts in robotics_interfaces).

Nothing here is wired to a producer or a consumer yet -- this is a contract
only. No simulator runtime behavior changes when this module is added.

Boundary rule: this module may import numpy,
robotics_interfaces.observations (for the shared WorldBounds alias), and
robotics_sim.environment.grid_geometry (the project's single source of truth
for world<->grid conversion), and may be imported by other robotics_sim
modules. It must never import Qt/PySide/PyQt, MainWindow, canvas, or
engine.py.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np

from robotics_interfaces.observations import WorldBounds
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


def _validate_bool(value, field_name: str) -> bool:
    """Accept only an actual bool (or numpy.bool_) -- never bool(value),
    which would silently accept 0/1/"True"/"False"/None/[]/() by truthiness
    instead of rejecting them as the wrong type.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    raise ValueError(
        f"{field_name} must be a bool (or numpy.bool_), got {value!r} ({type(value).__name__})."
    )


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


def _normalize_source_revisions(source_revisions) -> tuple[tuple[str, int], ...]:
    """Validate and sort (name, revision) pairs into a deterministic tuple.

    Stored as a tuple of tuples -- never a dict/Mapping/MappingProxyType --
    so the result is deeply immutable by construction, not by convention.
    """
    try:
        raw_items = list(source_revisions)
    except TypeError as exc:
        raise ValueError(
            f"source_revisions must be an iterable of (name, revision) pairs, got {source_revisions!r}."
        ) from exc

    seen_names: set[str] = set()
    normalized: list[tuple[str, int]] = []

    for item in raw_items:
        try:
            name, revision = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"each source_revisions entry must be a (name, revision) pair, got {item!r}."
            ) from exc

        if not isinstance(name, str):
            raise ValueError(
                f"source_revisions entry name must be a string, got {name!r} ({type(name).__name__})."
            )
        name = name.strip()
        if not name:
            raise ValueError("source_revisions entry name must not be empty.")
        if name in seen_names:
            raise ValueError(f"duplicate source_revisions name: {name!r}.")
        seen_names.add(name)

        normalized.append((name, _validate_revision(revision)))

    normalized.sort(key=lambda pair: pair[0])
    return tuple(normalized)


# ---------------------------------------------------------------------------
# Contract: Planning costmap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanningCostmapSnapshot:
    """A derived, planner-ready discrete grid with named source revisions.

    Cell states use the same UNKNOWN/FREE/OCCUPIED convention as
    ExplorationMapSnapshot, compatible with the simulator's current
    OccupancyGrid. source_revisions names which inputs (e.g. "belief",
    "observed_obstacles", "hazard") this grid was built from and at which
    revision, so a consumer can detect staleness -- it is intentionally a
    sorted tuple of (name, revision) pairs, not a dict, so it stays
    deeply immutable and order-independent to construct.

    This is the minimal snapshot only. built_at/timestamp/planner name/
    robot radius/hazard threshold/builder/cache belong to a future
    builder/policy, not here.
    """

    grid: np.ndarray
    bounds: WorldBounds
    resolution: float
    unknown_is_traversable: bool
    source_revisions: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        frozen_grid = _freeze_grid(self.grid)
        validated_bounds = _validate_bounds(self.bounds)
        validated_resolution = _validate_resolution(self.resolution)
        _validate_grid_matches_geometry(frozen_grid, validated_bounds, validated_resolution)
        validated_unknown_is_traversable = _validate_bool(
            self.unknown_is_traversable, "unknown_is_traversable"
        )
        normalized_revisions = _normalize_source_revisions(self.source_revisions)

        object.__setattr__(self, "grid", frozen_grid)
        object.__setattr__(self, "bounds", validated_bounds)
        object.__setattr__(self, "resolution", validated_resolution)
        object.__setattr__(self, "unknown_is_traversable", validated_unknown_is_traversable)
        object.__setattr__(self, "source_revisions", normalized_revisions)

    @property
    def height(self) -> int:
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        return int(self.grid.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)
