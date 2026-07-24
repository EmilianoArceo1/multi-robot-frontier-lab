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

    newly_blocked_cells/newly_unblocked_cells/blocked_state_changed compare
    each affected cell's blocked state (observed AND value >= block_
    threshold) before vs. after this call -- independent of `changed`,
    which is also True for e.g. a new robot merely attributing an
    already-known, already-blocked cell. Route replanning should key off
    newly_blocked_cells, not `changed`: creating/repeating/re-attributing an
    observation must never look like "new hazard discovered" just because
    something in the belief changed.
    """

    changed: bool
    newly_observed_cells: int
    changed_value_cells: int
    newly_attributed_cells: int
    affected_bounds: tuple[int, int, int, int] | None = None
    newly_blocked_cells: int = 0
    newly_unblocked_cells: int = 0
    blocked_state_changed: bool = False
    # FireSource entities whose physical footprint became visible for the
    # first time during this exact observation.  A source is detected when at
    # least one positive-contribution cell from its radius is inside the real,
    # occlusion-aware FoV polygon; its centre cell does not need to be visible.
    newly_discovered_sources: tuple[FireSource, ...] = ()


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
        # Source identity is mission state, not a geometric re-test on every
        # planner call.  IDs enter this set only through a real sensor
        # observation (or an explicit snapshot-state rebuild) and are pruned
        # when the corresponding source disappears.
        self._discovered_fire_ids: set[int] = set()

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
        self._discovered_fire_ids.discard(int(source.fire_id))
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
        self._discovered_fire_ids.clear()
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

    def discovered_sources(self) -> tuple[FireSource, ...]:
        """Return sources detected by at least one real FoV observation.

        Detection is footprint-based, not centre-cell-based.  The set is
        updated only by :meth:`observe_visible_polygon`, which receives the
        runtime's already occlusion-resolved sensor polygon.  Therefore a fire
        remains hidden until some positive part of its physical radius is
        actually visible, while seeing the edge is sufficient to expose the
        source centre as a planning target.
        """
        sources = self.field.sources()
        live_ids = {int(source.fire_id) for source in sources}
        self._discovered_fire_ids.intersection_update(live_ids)
        return tuple(
            source
            for source in sources
            if int(source.fire_id) in self._discovered_fire_ids
        )

    def refresh_discovered_sources_from_belief(self) -> tuple[FireSource, ...]:
        """Rebuild discovery identity after a snapshot restore.

        Normal runtime discovery must go through observe_visible_polygon().
        Snapshot restore is the one exception: both FireSource state and the
        previously observed HazardBelief are restored independently, so source
        identity is reconstructed from cells that are observed, positive, and
        physically inside each source's radius.
        """
        discovered_ids: set[int] = set()
        for source in self.field.sources():
            rows, cols = self._source_footprint_indices(source)
            if rows.size == 0:
                continue
            values, observed = self._belief.read_cells(rows, cols)
            if bool(np.any(observed & (values > 0.0))):
                discovered_ids.add(int(source.fire_id))
        self._discovered_fire_ids = discovered_ids
        return self.discovered_sources()

    def _source_footprint_indices(
        self,
        source: FireSource,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grid cells whose centres receive positive contribution from source."""
        geometry = self.field.geometry
        radius = max(float(source.radius), 0.0)
        if radius <= 0.0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        start = geometry.world_to_grid(
            float(source.position[0]) - radius,
            float(source.position[1]) - radius,
            clamp=True,
        )
        end = geometry.world_to_grid(
            float(source.position[0]) + radius,
            float(source.position[1]) + radius,
            clamp=True,
        )
        if start is None or end is None:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        row_values = np.arange(
            max(0, min(start.row, end.row)),
            min(geometry.height - 1, max(start.row, end.row)) + 1,
            dtype=np.int64,
        )
        col_values = np.arange(
            max(0, min(start.col, end.col)),
            min(geometry.width - 1, max(start.col, end.col)) + 1,
            dtype=np.int64,
        )
        if row_values.size == 0 or col_values.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        rows, cols = np.meshgrid(row_values, col_values, indexing="ij")
        xs = geometry.x_min + (cols.astype(np.float64) + 0.5) * geometry.resolution
        ys = geometry.y_min + (rows.astype(np.float64) + 0.5) * geometry.resolution
        inside = np.hypot(
            xs - float(source.position[0]),
            ys - float(source.position[1]),
        ) < radius
        return rows[inside].astype(np.int64), cols[inside].astype(np.int64)

    def blocked_world_points(self) -> tuple[tuple[float, float], ...]:
        """Ground-truth blocked points. Kept for its own legacy contract/
        tests -- route validation must use observed_blocked_world_points()
        instead; this is omniscient and must not drive planning/replanning."""
        return self.field.blocked_world_points(self.block_threshold)

    def observed_blocked_world_points(
        self,
        block_threshold: float | None = None,
    ) -> tuple[tuple[float, float], ...]:
        """Discovered-hazard counterpart to blocked_world_points(): only
        cells the team has actually observed (observed=True) with a value
        at or above threshold. Never reads FireSource/HazardField --
        deterministic given the current belief state alone."""
        threshold = self.block_threshold if block_threshold is None else float(block_threshold)
        rows, cols = self._belief.blocked_cells(threshold)
        geometry = self._belief.geometry
        return tuple(
            geometry.grid_to_world(GridCell(row=int(row), col=int(col)))
            for row, col in zip(rows, cols)
        )

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

        # Detect source entities from ANY visible positive part of their
        # footprint.  This uses only cells inside the real polygon above; the
        # source centre may remain outside the FoV.  Identity is recorded once
        # so repeated observations do not create a replan storm.
        xs_visible = (
            geometry.x_min
            + (cols_arr.astype(np.float64) + 0.5) * geometry.resolution
        )
        ys_visible = (
            geometry.y_min
            + (rows_arr.astype(np.float64) + 0.5) * geometry.resolution
        )
        newly_discovered_sources: list[FireSource] = []
        for source in self.field.sources():
            fire_id = int(source.fire_id)
            if fire_id in self._discovered_fire_ids:
                continue
            distance = np.hypot(
                xs_visible - float(source.position[0]),
                ys_visible - float(source.position[1]),
            )
            if bool(np.any(distance < float(source.radius))):
                self._discovered_fire_ids.add(fire_id)
                newly_discovered_sources.append(source)

        # Blocked state BEFORE this observation, for exactly the affected
        # cells -- captured before observe_cells() mutates the belief, so
        # replanning can be gated on an actual threshold CROSSING rather
        # than on `changed` (which is also True for e.g. a new robot merely
        # attributing an already-known, already-blocked cell). read_cells()
        # is O(len(rows_arr)), not a full-grid O(H*W) snapshot() copy -- this
        # runs once per robot per sensor update, so that difference matters.
        previous_values, previous_observed = self._belief.read_cells(rows_arr, cols_arr)
        before_blocked = previous_observed & (previous_values >= self.block_threshold)

        update = self._belief.observe_cells(rows_arr, cols_arr, values, robot_index=int(robot_index))

        # After this call, every affected cell is observed=True with
        # exactly `values` (already clamped to [0, 1] by HazardField) --
        # no second snapshot needed to know the "after" blocked state.
        after_blocked = values >= self.block_threshold
        newly_blocked = after_blocked & ~before_blocked
        newly_unblocked = before_blocked & ~after_blocked
        newly_blocked_cells = int(np.count_nonzero(newly_blocked))
        newly_unblocked_cells = int(np.count_nonzero(newly_unblocked))

        return HazardObservationResult(
            changed=update.changed,
            newly_observed_cells=update.newly_observed_cells,
            changed_value_cells=update.changed_value_cells,
            newly_attributed_cells=update.newly_attributed_cells,
            affected_bounds=(
                int(rows_arr.min()), int(rows_arr.max()), int(cols_arr.min()), int(cols_arr.max())
            ),
            newly_blocked_cells=newly_blocked_cells,
            newly_unblocked_cells=newly_unblocked_cells,
            blocked_state_changed=bool(newly_blocked_cells or newly_unblocked_cells),
            newly_discovered_sources=tuple(newly_discovered_sources),
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