"""Runtime service that owns temporary fire sources, their ground-truth
hazard field, and the team's discovered hazard belief."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reuses BeliefMap's own point-in-polygon test rather than duplicating the
# ray-casting algorithm here -- belief_map.py itself is out of scope for
# this phase (occupancy/explored rasterization stays exactly as it is).
from robotics_sim.environment.belief_map import _point_inside_polygon
from robotics_sim.environment.grid_geometry import GridCell
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.environment.hazard_field import FireSource, HazardField


@dataclass(frozen=True)
class HazardChange:
    action: str
    source: FireSource | None
    version: int

    @property
    def changed(self) -> bool:
        return self.action in {"added", "removed", "cleared"}


@dataclass(frozen=True)
class HazardObservationResult:
    """Result of one observe_visible_polygon() call.

    affected_bounds is (row_min, row_max, col_min, col_max) over the cells
    the polygon actually rasterized to -- None when the polygon covered no
    cell at all (degenerate, out of bounds, or fewer than 3 points).
    """

    changed: bool
    newly_observed_cells: int
    changed_value_cells: int
    newly_attributed_cells: int
    affected_bounds: tuple[int, int, int, int] | None = None


class RuntimeHazardService:
    """Host-side owner of dynamic hazards; contains no Qt or planner logic.

    Owns two independent layers sharing one GridGeometry:
        - ``field``    GroundTruth HazardField, rebuilt from FireSource
                        objects whenever one is added or removed (see
                        add_fire()/remove_fire_near()/toggle_fire_at()).
        - ``belief``   Team HazardBelief -- only what robots have actually
                        observed via observe_visible_polygon(). Creating or
                        removing a FireSource never touches it; a cell's
                        belief only changes the next time it re-enters a
                        robot's real, occlusion-aware sensor FoV.
    """

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        resolution: float,
        robot_count: int = 1,
        default_intensity: float = 1.0,
        default_radius: float = 2.0,
        selection_radius: float = 0.6,
        block_threshold: float = 0.55,
    ) -> None:
        self.field = HazardField(bounds=bounds, resolution=resolution)
        self._belief = HazardBelief(self.field.geometry, robot_count=max(1, int(robot_count)))
        self.default_intensity = float(default_intensity)
        self.default_radius = float(default_radius)
        self.selection_radius = max(0.0, float(selection_radius))
        self.block_threshold = float(block_threshold)

    @property
    def belief(self) -> HazardBelief:
        """Team HazardBelief -- read-only access. Mutate only through
        observe_visible_polygon() (or HazardBelief.restore() for snapshot
        restore, a later phase), never by reassigning this property."""
        return self._belief

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
        field_changed = self.field.clear()
        # Keep both layers coherent: a cleared ground truth with a stale
        # belief still attached would let old observations linger forever
        # with nothing left to ever re-observe them against.
        belief_revision_before = self._belief.revision
        self._belief.clear()
        belief_changed = self._belief.revision != belief_revision_before
        # Either layer actually changing counts -- a field that was already
        # empty must not report "no_change" when the belief still had
        # observations to discard (and vice versa).
        changed = field_changed or belief_changed
        return HazardChange(
            "cleared" if changed else "no_change",
            None,
            self.field.version,
        )

    def sources(self) -> tuple[FireSource, ...]:
        return self.field.sources()

    def blocked_world_points(self) -> tuple[tuple[float, float], ...]:
        return self.field.blocked_world_points(self.block_threshold)

    def observe_visible_polygon(
        self,
        polygon: list[tuple[float, float]],
        robot_index: int,
    ) -> HazardObservationResult:
        """Fuse ground-truth hazard into the team belief for exactly the
        cells covered by *polygon* -- the real, occlusion-aware sensor FoV
        already computed by the runtime, never a geometric approximation.

        Mirrors BeliefMap.mark_visible_polygon()'s bounding-box + point-in-
        polygon rasterization so occupancy and hazard observation agree on
        exactly which cells are "visible" for the same polygon this tick.

        1. read HazardField.values at each visible cell (ground truth);
        2. write those values into HazardBelief;
        3. mark observed=True;
        4. mark observed_by_robot[robot_index]=True.
        """
        empty = HazardObservationResult(False, 0, 0, 0, None)
        if len(polygon) < 3:
            return empty

        geometry = self.field.geometry
        xs = [float(p[0]) for p in polygon]
        ys = [float(p[1]) for p in polygon]

        min_x = max(geometry.x_min, min(xs))
        max_x = min(geometry.x_max, max(xs))
        min_y = max(geometry.y_min, min(ys))
        max_y = min(geometry.y_max, max(ys))

        start_cell = geometry.world_to_grid(min_x, min_y, clamp=True)
        end_cell = geometry.world_to_grid(max_x, max_y, clamp=True)
        if start_cell is None or end_cell is None:
            return empty

        r0 = max(0, min(start_cell.row, end_cell.row))
        r1 = min(geometry.height - 1, max(start_cell.row, end_cell.row))
        c0 = max(0, min(start_cell.col, end_cell.col))
        c1 = min(geometry.width - 1, max(start_cell.col, end_cell.col))

        rows: list[int] = []
        cols: list[int] = []
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                world = geometry.grid_to_world(GridCell(row=row, col=col))
                if _point_inside_polygon(world, polygon):
                    rows.append(row)
                    cols.append(col)

        if not rows:
            return empty

        rows_arr = np.asarray(rows, dtype=np.int64)
        cols_arr = np.asarray(cols, dtype=np.int64)
        # Read-only view -- observe_cells() only reads these values (fancy
        # indexing below returns a fresh array), never mutates ground truth.
        ground_truth = self.field.values(copy=False)
        values = ground_truth[rows_arr, cols_arr]

        update = self._belief.observe_cells(rows_arr, cols_arr, values, robot_index=int(robot_index))

        return HazardObservationResult(
            changed=update.changed,
            newly_observed_cells=update.newly_observed_cells,
            changed_value_cells=update.changed_value_cells,
            newly_attributed_cells=update.newly_attributed_cells,
            affected_bounds=(
                int(rows_arr.min()), int(rows_arr.max()), int(cols_arr.min()), int(cols_arr.max())
            ),
        )

    def snapshot(self) -> dict:
        return {
            "version": self.field.version,
            "bounds": self.field.bounds,
            "resolution": self.field.resolution,
            "grid": self.field.values(copy=True),
            "sources": tuple(self.field.sources()),
            "block_threshold": self.block_threshold,
        }
