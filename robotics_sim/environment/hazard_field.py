"""Dynamic continuous hazard field for runtime fire events.

Occupancy and hazard are deliberately separate layers:

- ``BeliefMap.grid`` keeps UNKNOWN/FREE/OCCUPIED semantics.
- ``HazardField.values`` stores continuous thermal risk in ``[0, 1]``.
- ``FireSource`` objects are the authoritative dynamic entities.

The field is rebuilt from the registered sources whenever a source is added or
removed. This makes removal deterministic even when several fires overlap and
prevents the occupancy belief from being corrupted by temporary hazards.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from robotics_sim.environment.grid_geometry import GridCell, GridGeometry


@dataclass(frozen=True)
class FireSource:
    """One world-space source contributing to the thermal hazard field."""

    fire_id: int
    position: tuple[float, float]
    intensity: float
    radius: float


class HazardField:
    """Rasterized continuous hazard layer aligned with the occupancy grid."""

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        resolution: float,
    ) -> None:
        self.geometry = GridGeometry(bounds, resolution)
        self._values = np.zeros(
            (self.geometry.height, self.geometry.width),
            dtype=np.float32,
        )
        self._sources: dict[int, FireSource] = {}
        self._next_fire_id = 1
        self._version = 0

        self._x_centers = (
            self.geometry.x_min
            + (np.arange(self.geometry.width, dtype=np.float32) + 0.5)
            * self.geometry.resolution
        )
        self._y_centers = (
            self.geometry.y_min
            + (np.arange(self.geometry.height, dtype=np.float32) + 0.5)
            * self.geometry.resolution
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.geometry.bounds

    @property
    def resolution(self) -> float:
        return float(self.geometry.resolution)

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    @property
    def version(self) -> int:
        return int(self._version)

    @property
    def next_fire_id(self) -> int:
        return int(self._next_fire_id)

    def in_bounds_world(self, position: tuple[float, float]) -> bool:
        return self.geometry.in_bounds_world(float(position[0]), float(position[1]))

    def sources(self) -> tuple[FireSource, ...]:
        return tuple(self._sources[key] for key in sorted(self._sources))

    def values(self, *, copy: bool = True) -> np.ndarray:
        """Return the thermal field.

        A copy is returned by default so GUI/debug consumers cannot mutate the
        authoritative runtime field by accident.
        """

        if copy:
            return self._values.copy()
        view = self._values.view()
        view.setflags(write=False)
        return view

    def add_fire(
        self,
        position: tuple[float, float],
        *,
        intensity: float = 1.0,
        radius: float = 2.0,
    ) -> FireSource:
        x, y = float(position[0]), float(position[1])
        intensity = float(intensity)
        radius = float(radius)

        if not self.in_bounds_world((x, y)):
            raise ValueError(f"Fire position {(x, y)} is outside hazard bounds {self.bounds}.")
        if not (0.0 < intensity <= 1.0):
            raise ValueError("Fire intensity must be in the interval (0, 1].")
        if radius <= 0.0:
            raise ValueError("Fire radius must be greater than zero.")

        source = FireSource(
            fire_id=self._next_fire_id,
            position=(x, y),
            intensity=intensity,
            radius=radius,
        )
        self._next_fire_id += 1
        self._sources[source.fire_id] = source
        self._rebuild()
        return source

    def remove_fire(self, fire_id: int) -> FireSource | None:
        source = self._sources.pop(int(fire_id), None)
        if source is not None:
            self._rebuild()
        return source

    def nearest_fire(
        self,
        position: tuple[float, float],
        *,
        max_distance: float,
    ) -> FireSource | None:
        max_distance = max(0.0, float(max_distance))
        x, y = float(position[0]), float(position[1])
        best: FireSource | None = None
        best_distance = max_distance
        for source in self.sources():
            distance = math.hypot(source.position[0] - x, source.position[1] - y)
            if distance <= best_distance:
                best = source
                best_distance = distance
        return best

    def remove_nearest_fire(
        self,
        position: tuple[float, float],
        *,
        max_distance: float,
    ) -> FireSource | None:
        source = self.nearest_fire(position, max_distance=max_distance)
        if source is None:
            return None
        return self.remove_fire(source.fire_id)

    def restore_sources(self, sources: Iterable[FireSource], *, next_fire_id: int) -> None:
        """Replace every source with an exact prior set and rebuild the
        continuous field from them -- used by navigation-debug snapshot
        restore (see engine.restore_navigation_debug_snapshot()) to roll
        hazards back to what they were at a past tick.

        Bounds/resolution are unchanged (a run never resizes them mid-way);
        only which fires exist and their footprint changes. `next_fire_id`
        is restored too so a fire added immediately after gets the id it
        would have received at that point in time, not one collided with /
        skipped ahead by fires that existed only in the discarded future.
        Bumps `version` via `_rebuild()`, same as add_fire()/remove_fire()
        -- callers relying on version-keyed caches (e.g. the canvas's hazard
        heatmap pixmap) invalidate automatically.
        """
        self._sources = {int(source.fire_id): source for source in sources}
        self._next_fire_id = max(1, int(next_fire_id))
        self._rebuild()

    def clear(self) -> bool:
        if not self._sources and not np.any(self._values):
            return False
        self._sources.clear()
        self._values.fill(0.0)
        self._version += 1
        return True

    def blocked_cells(self, threshold: float) -> tuple[GridCell, ...]:
        threshold = float(threshold)
        if not (0.0 < threshold <= 1.0):
            raise ValueError("Hazard threshold must be in the interval (0, 1].")
        rows, cols = np.where(self._values >= threshold)
        return tuple(GridCell(row=int(row), col=int(col)) for row, col in zip(rows, cols))

    def blocked_world_points(self, threshold: float) -> tuple[tuple[float, float], ...]:
        return tuple(self.geometry.grid_to_world(cell) for cell in self.blocked_cells(threshold))

    def _rebuild(self) -> None:
        self._values.fill(0.0)
        if self._sources:
            xx = self._x_centers[np.newaxis, :]
            yy = self._y_centers[:, np.newaxis]
            for source in self.sources():
                distance = np.hypot(xx - source.position[0], yy - source.position[1])
                contribution = np.where(
                    distance < source.radius,
                    source.intensity * (1.0 - distance / source.radius),
                    0.0,
                ).astype(np.float32, copy=False)
                np.maximum(self._values, contribution, out=self._values)
        np.clip(self._values, 0.0, 1.0, out=self._values)
        self._version += 1
