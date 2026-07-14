"""Runtime service that owns temporary fire sources and their hazard field."""

from __future__ import annotations

from dataclasses import dataclass

from robotics_sim.environment.hazard_field import FireSource, HazardField


@dataclass(frozen=True)
class HazardChange:
    action: str
    source: FireSource | None
    version: int

    @property
    def changed(self) -> bool:
        return self.action in {"added", "removed", "cleared"}


class RuntimeHazardService:
    """Host-side owner of dynamic hazards; contains no Qt or planner logic."""

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        resolution: float,
        default_intensity: float = 1.0,
        default_radius: float = 2.0,
        selection_radius: float = 0.6,
        block_threshold: float = 0.55,
    ) -> None:
        self.field = HazardField(bounds=bounds, resolution=resolution)
        self.default_intensity = float(default_intensity)
        self.default_radius = float(default_radius)
        self.selection_radius = max(0.0, float(selection_radius))
        self.block_threshold = float(block_threshold)

    def add_fire(self, position: tuple[float, float]) -> HazardChange:
        source = self.field.add_fire(
            position,
            intensity=self.default_intensity,
            radius=self.default_radius,
        )
        return HazardChange("added", source, self.field.version)

    def remove_fire_near(self, position: tuple[float, float]) -> HazardChange:
        source = self.field.remove_nearest_fire(
            position,
            max_distance=self.selection_radius,
        )
        if source is None:
            return HazardChange("no_change", None, self.field.version)
        return HazardChange("removed", source, self.field.version)

    def toggle_fire_at(self, position: tuple[float, float]) -> HazardChange:
        removed = self.remove_fire_near(position)
        if removed.changed:
            return removed
        return self.add_fire(position)

    def clear(self) -> HazardChange:
        changed = self.field.clear()
        return HazardChange(
            "cleared" if changed else "no_change",
            None,
            self.field.version,
        )

    def sources(self) -> tuple[FireSource, ...]:
        return self.field.sources()

    def blocked_world_points(self) -> tuple[tuple[float, float], ...]:
        return self.field.blocked_world_points(self.block_threshold)

    def snapshot(self) -> dict:
        return {
            "version": self.field.version,
            "bounds": self.field.bounds,
            "resolution": self.field.resolution,
            "grid": self.field.values(copy=True),
            "sources": tuple(self.field.sources()),
            "block_threshold": self.block_threshold,
        }
