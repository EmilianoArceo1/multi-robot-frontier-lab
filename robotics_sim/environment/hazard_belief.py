"""Team hazard belief: discovered-only thermal risk layer.

This is deliberately a third, independent layer alongside:

    - ``BeliefMap.grid``      logical UNKNOWN/FREE/OCCUPIED occupancy
    - ``HazardField.values``  continuous ground-truth thermal risk

``HazardBelief`` stores only what robots have actually *observed* about the
hazard field -- never the omniscient ground truth. A cell with
``observed=False`` means the team has no information about that cell's
hazard state; ``observed=True`` with ``values=0.0`` means the team has
confirmed the cell is safe; ``observed=True`` with ``values>0.0`` means the
team has confirmed a hazard there. Cells are only ever written by
``observe_cells()`` -- there is no coupling to ``FireSource``/``HazardField``
in this module, so ground-truth changes (a fire being created or removed)
never silently leak into the belief until a robot actually re-observes those
cells.

FoV rasterization (turning a sensor polygon into row/col/value arrays) is a
Phase 2 concern owned by ``RuntimeHazardService`` -- this module only stores
and fuses whatever cells it is given.

No Qt, engine, planner, HazardField, or FireSource imports here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics_sim.environment.grid_geometry import GridGeometry


@dataclass(frozen=True)
class HazardBeliefFrame:
    """Immutable snapshot of one HazardBelief's state.

    Arrays are copies with ``writeable=False`` -- see
    ``HazardBelief.snapshot()``. Safe to hold onto indefinitely; later writes
    to the originating ``HazardBelief`` never affect an already-taken frame.
    """

    values: np.ndarray
    observed: np.ndarray
    observed_by_robot: np.ndarray
    revision: int


@dataclass(frozen=True)
class HazardBeliefUpdate:
    """Result of one ``observe_cells()`` call.

    The three counters are independent, not cumulative: a cell observed for
    the first time with a nonzero value contributes to both
    ``newly_observed_cells`` and ``changed_value_cells`` (and, since it is
    also the first time this robot attributed it, to
    ``newly_attributed_cells``); a cell re-observed by a *different* robot
    with the same value contributes only to ``newly_attributed_cells``.
    """

    changed: bool
    newly_observed_cells: int
    changed_value_cells: int
    newly_attributed_cells: int


class HazardBelief:
    """Owns the team's shared discovered-hazard state.

    ``values``/``observed`` are shared across the whole team (there is one
    team belief, not one per robot); ``observed_by_robot`` is the only
    per-robot layer, kept purely for attribution.
    """

    def __init__(self, geometry: GridGeometry, robot_count: int = 1) -> None:
        robot_count = int(robot_count)
        if robot_count < 1:
            raise ValueError(f"robot_count must be >= 1, got {robot_count}.")

        self.geometry = geometry
        self.robot_count = robot_count
        self.height = int(geometry.height)
        self.width = int(geometry.width)

        self._values = np.zeros((self.height, self.width), dtype=np.float32)
        self._observed = np.zeros((self.height, self.width), dtype=bool)
        self._observed_by_robot = np.zeros(
            (self.robot_count, self.height, self.width), dtype=bool
        )
        self._revision = 0

    @property
    def revision(self) -> int:
        return int(self._revision)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def _validate_robot_index(self, robot_index: int) -> int:
        robot_index = int(robot_index)
        if not (0 <= robot_index < self.robot_count):
            raise ValueError(
                f"robot_index {robot_index} out of range [0, {self.robot_count})."
            )
        return robot_index

    def observe_cells(
        self,
        rows,
        cols,
        values,
        robot_index: int,
    ) -> HazardBeliefUpdate:
        """Fuse a batch of (row, col, value) observations from one robot.

        ``values`` are clipped to ``[0, 1]`` before being stored. Duplicate
        (row, col) pairs within one call resolve to the last entry for that
        cell, matching plain NumPy fancy-assignment semantics.
        """
        robot_index = self._validate_robot_index(robot_index)

        rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols_arr = np.asarray(cols, dtype=np.int64).reshape(-1)
        values_arr = np.asarray(values, dtype=np.float32).reshape(-1)

        if not (rows_arr.shape == cols_arr.shape == values_arr.shape):
            raise ValueError(
                "rows, cols, and values must have matching shapes: "
                f"{np.asarray(rows).shape}, {np.asarray(cols).shape}, {np.asarray(values).shape}."
            )

        if rows_arr.size == 0:
            return HazardBeliefUpdate(
                changed=False,
                newly_observed_cells=0,
                changed_value_cells=0,
                newly_attributed_cells=0,
            )

        if (
            rows_arr.min() < 0
            or rows_arr.max() >= self.height
            or cols_arr.min() < 0
            or cols_arr.max() >= self.width
        ):
            raise ValueError(
                f"Cell indices out of bounds for HazardBelief shape {self.shape}."
            )

        clipped_values = np.clip(values_arr, 0.0, 1.0).astype(np.float32, copy=False)

        # Deduplicate (row, col) pairs, keeping the last occurrence in input
        # order -- identical to what plain `arr[rows, cols] = values` would
        # leave behind, so counts below match the actual final state.
        flat_index = rows_arr * self.width + cols_arr
        reversed_index = flat_index[::-1]
        reversed_values = clipped_values[::-1]
        unique_flat, first_pos_in_reversed = np.unique(reversed_index, return_index=True)
        unique_rows = (unique_flat // self.width).astype(np.int64)
        unique_cols = (unique_flat % self.width).astype(np.int64)
        unique_values = reversed_values[first_pos_in_reversed]

        prev_observed = self._observed[unique_rows, unique_cols]
        prev_values = self._values[unique_rows, unique_cols]
        prev_attributed = self._observed_by_robot[robot_index, unique_rows, unique_cols]

        newly_observed_mask = ~prev_observed
        value_changed_mask = prev_values != unique_values
        newly_attributed_mask = ~prev_attributed

        newly_observed_cells = int(np.count_nonzero(newly_observed_mask))
        changed_value_cells = int(np.count_nonzero(value_changed_mask))
        newly_attributed_cells = int(np.count_nonzero(newly_attributed_mask))

        changed = bool(newly_observed_cells or changed_value_cells or newly_attributed_cells)

        if changed:
            self._values[unique_rows, unique_cols] = unique_values
            self._observed[unique_rows, unique_cols] = True
            self._observed_by_robot[robot_index, unique_rows, unique_cols] = True
            self._revision += 1

        return HazardBeliefUpdate(
            changed=changed,
            newly_observed_cells=newly_observed_cells,
            changed_value_cells=changed_value_cells,
            newly_attributed_cells=newly_attributed_cells,
        )

    def read_cells(self, rows, cols) -> tuple[np.ndarray, np.ndarray]:
        """Read values/observed at specific cells -- O(len(rows)) work, not
        O(height*width) like snapshot(). Never calls snapshot() and never
        returns a view into internal state (fancy indexing already
        allocates a new array; .copy() below makes that independence
        explicit rather than relying on it as an implementation detail).

        Use this instead of snapshot() on a hot path that only needs a
        handful of cells -- e.g. checking a batch of cells' prior state
        before observe_cells() writes new values into them.
        """
        rows_arr = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols_arr = np.asarray(cols, dtype=np.int64).reshape(-1)

        if rows_arr.shape != cols_arr.shape:
            raise ValueError(
                "rows and cols must have matching shapes: "
                f"{np.asarray(rows).shape}, {np.asarray(cols).shape}."
            )

        if rows_arr.size and (
            rows_arr.min() < 0
            or rows_arr.max() >= self.height
            or cols_arr.min() < 0
            or cols_arr.max() >= self.width
        ):
            raise ValueError(
                f"Cell indices out of bounds for HazardBelief shape {self.shape}."
            )

        values = self._values[rows_arr, cols_arr].copy()
        observed = self._observed[rows_arr, cols_arr].copy()
        return values, observed

    def blocked_cells(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (rows, cols) of cells with observed=True and values >=
        threshold -- deterministic, never calls snapshot(), never returns a
        view into internal state (np.where() always allocates fresh arrays).
        """
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {threshold}.")

        blocked_mask = self._observed & (self._values >= threshold)
        rows, cols = np.where(blocked_mask)
        return rows, cols

    def snapshot(self) -> HazardBeliefFrame:
        """Return an immutable, independently-owned copy of the current state."""
        values = self._values.copy()
        values.setflags(write=False)
        observed = self._observed.copy()
        observed.setflags(write=False)
        observed_by_robot = self._observed_by_robot.copy()
        observed_by_robot.setflags(write=False)

        return HazardBeliefFrame(
            values=values,
            observed=observed,
            observed_by_robot=observed_by_robot,
            revision=self.revision,
        )

    def restore(self, frame: HazardBeliefFrame) -> None:
        """Replace the current state with a previously captured frame.

        Validates shape, dtype, and robot_count against this instance's own
        geometry before mutating anything -- a rejected restore leaves the
        current state untouched.
        """
        expected_2d = (self.height, self.width)
        expected_3d = (self.robot_count, self.height, self.width)

        values = frame.values
        observed = frame.observed
        observed_by_robot = frame.observed_by_robot

        if values.shape != expected_2d:
            raise ValueError(
                f"HazardBeliefFrame.values shape {values.shape} != expected {expected_2d}."
            )
        if observed.shape != expected_2d:
            raise ValueError(
                f"HazardBeliefFrame.observed shape {observed.shape} != expected {expected_2d}."
            )
        if observed_by_robot.shape != expected_3d:
            raise ValueError(
                "HazardBeliefFrame.observed_by_robot shape "
                f"{observed_by_robot.shape} != expected {expected_3d}."
            )
        if values.dtype != np.float32:
            raise ValueError(f"HazardBeliefFrame.values dtype must be float32, got {values.dtype}.")
        if observed.dtype != np.bool_:
            raise ValueError(f"HazardBeliefFrame.observed dtype must be bool, got {observed.dtype}.")
        if observed_by_robot.dtype != np.bool_:
            raise ValueError(
                f"HazardBeliefFrame.observed_by_robot dtype must be bool, got {observed_by_robot.dtype}."
            )

        self._values = values.astype(np.float32, copy=True)
        self._observed = observed.astype(bool, copy=True)
        self._observed_by_robot = observed_by_robot.astype(bool, copy=True)
        self._revision = int(frame.revision)

    def clear(self) -> None:
        """Reset every cell to unobserved. A no-op (no revision bump) if
        there was nothing observed to clear."""
        had_state = bool(np.any(self._observed)) or bool(np.any(self._observed_by_robot))
        if not had_state:
            return

        self._values.fill(0.0)
        self._observed.fill(False)
        self._observed_by_robot.fill(False)
        self._revision += 1
