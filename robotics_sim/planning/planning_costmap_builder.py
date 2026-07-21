"""Internal builder that unifies planning-costmap construction.

Not connected to runtime yet. This is a composition point meant to
eventually replace the several independent ways the simulator currently
derives a discrete planning grid (see
robotics_sim/tests/test_planning_map_characterization.py for the
characterized divergences). It reproduces today's runtime composition
order and inflation policy, with ONE deliberate, documented divergence:
legacy exploration-belief OCCUPIED cells are no longer treated as physical
obstacle occupancy (see "Legacy belief occupancy vs. observed obstacle
geometry" below). Every other aspect -- hazard projection, observed-
obstacle-point inflation, composition order, canonical geometry -- still
matches build_planning_grid_for_robot() exactly.

Reference contract this reproduces (with one intentional divergence)
----------------------------------------------------------------------
``SimulationControllerMixin.build_planning_grid_for_robot()``
(robotics_sim/simulation/engine.py) currently does, in order:

    1. belief = self.ensure_belief_map()
    2. planning_grid = belief.to_planning_grid(unknown_is_traversable=True,
           inflate_radius=radius)
       -- BeliefMap.to_planning_grid() builds a fresh OccupancyGrid,
       explicitly marks every belief-FREE cell FREE, then either rasterizes
       belief-OCCUPIED cell centers through add_obstacle_points(padding=
       inflate_radius) when inflate_radius > 0, or marks them OCCUPIED
       directly (no inflation) when inflate_radius <= 0.
       ** This builder does NOT reproduce that OCCUPIED-cell handling. **
       See "Legacy belief occupancy vs. observed obstacle geometry" below.
    3. if obstacle_points: planning_grid.add_obstacle_points(obstacle_points,
           padding=radius)
       -- obstacle_points is whatever the CALLER already decided to project
       (already sanitized upstream by sanitize_planner_obstacle_points() --
       this builder does not sanitize; see "Sanitization" below). This step
       IS reproduced exactly.
    4. if hazard_service: apply_hazard_belief_to_planning_grid(planning_grid,
           hazard_service.belief, block_threshold=..., inflate_radius=radius)
       -- discovered-only hazard belief, never ground-truth HazardField.
       This step IS reproduced exactly.

The SAME ``radius`` value (robot's effective safety radius,
safety_radius_for_robot()) is used for steps 3-4 above. This builder
mirrors that: PlanningCostmapPolicy.obstacle_padding is that one shared
value, applied identically to observed-obstacle-point inflation and hazard
inflation. It is no longer applied to exploration-grid projection at all --
there is nothing left there to inflate (see below). Nothing arrives
"pre-inflated" into this builder -- every physical-occupancy source is
inflated inside build(), exactly once, by the same policy value.

Legacy belief occupancy vs. observed obstacle geometry
----------------------------------------------------------
ExplorationMapSnapshot.grid's OCCUPIED=1 cells record that the team's
belief map *once* marked a cell occupied, from whatever heuristic
BeliefMap itself currently uses to set that state -- which is not
necessarily confirmed obstacle geometry. Conflating that belief state with
physical occupancy is exactly the coupling this migration is separating
out. So this builder treats BOTH FREE=0 and OCCUPIED=1 exploration cells as
"observed and traversable" when projecting the base grid -- neither one
places a physical obstacle by itself. Only two things ever mark a cell as
physically blocked in this builder's output:

    - ObservedObstacleSnapshot.points, inflated by obstacle_padding (via
      OccupancyGrid.add_obstacle_points()) -- unchanged from before.
    - Observed HazardBeliefFrame cells at/above hazard_block_threshold,
      inflated by obstacle_padding (via
      apply_hazard_belief_to_planning_grid()) -- unchanged from before.

UNKNOWN=-1 cells are unaffected by this change: they still resolve to
traversable or blocked purely from policy.unknown_is_traversable, exactly
as before -- this divergence only changes what happens to OCCUPIED=1.

This is an intentional behavior change from build_planning_grid_for_robot()
today, not a bug: a robot's live BeliefMap can carry OCCUPIED cells that
this builder now routes through as traversable unless
ObservedObstacleSnapshot (or hazard) independently confirms occupancy
there. See
test_builder_intentionally_ignores_legacy_belief_occupancy_without_observed_geometry
in the test file for the exact contract. PlanningCostmapBuilder is still
not connected to runtime (see top of this docstring), so this divergence
has no effect on simulator behavior yet.

Inputs are the Phase 1 snapshot contracts
------------------------------------------
    - ExplorationMapSnapshot   (robotics_sim/environment/map_snapshots.py)
      replaces a live BeliefMap as the "belief" source.
    - ObservedObstacleSnapshot (robotics_sim/environment/map_snapshots.py)
      replaces the raw obstacle_points list.
    - HazardBeliefFrame        (robotics_sim/environment/hazard_belief.py)
      replaces hazard_service.belief; optional, discovered-only.

Composition order: exploration knowledge -> observed static obstacles ->
dynamic obstacle points -> observed hazards. See "Dynamic obstacle points"
below for the third layer.

Dynamic obstacle points
------------------------
``dynamic_obstacle_points: tuple[Point2D, ...] = ()`` is a THIRD, separate
physical-occupancy source, alongside ObservedObstacleSnapshot.points and
observed hazard cells (see "Legacy belief occupancy vs. observed obstacle
geometry" above for why legacy BeliefMap.OCCUPIED is never one of these
three). It exists to let a caller project runtime-only, non-static
occupancy -- today, other runtime robots
(SimulationControllerMixin.dynamic_robot_obstacle_points_for_robot()) --
without inventing a second static-geometry contract for it.

It is deliberately NOT an ObservedObstacleSnapshot and there is no
DynamicObstacleSnapshot type: unlike static observed geometry, these points
have no meaningful revision of their own to track (a robot's position
changes every tick; a "revision" counter for that would either be
meaningless or would have to be the tick count in disguise) and are not
persisted/cached anywhere -- they are supplied fresh, per build() call, by
the caller, and never appear in source_revisions. Two calls to build() with
the same dynamic_obstacle_points content are expected to be made with a
FRESH tuple each time (built by the caller from live state), not reused
from a cached snapshot. It is inflated by the exact same
policy.obstacle_padding as ObservedObstacleSnapshot.points (see rule 3
below), and validated the same way (see _validate_dynamic_obstacle_points()
below) -- normalized to a fresh tuple of finite float pairs, never mutating
whatever the caller passed in.

Hazard geometry
----------------
HazardBeliefFrame carries no bounds/resolution of its own (see hazard_belief.py
-- it is plain arrays + revision). The live runtime always builds
hazard_service with the SAME bounds/resolution as the belief map, but this
builder must not silently assume that here -- so build() takes an explicit
``hazard_geometry: GridGeometry | None`` alongside ``hazard_belief``. Both
must be given together or both omitted; when both are given, hazard_geometry
must match exploration's own bounds/resolution, and hazard_belief's array
shapes must match exploration.grid.shape (see _validate_hazard_inputs()).

Sanitization
------------
build_planning_grid_for_robot() itself does not sanitize either --
sanitize_planner_obstacle_points() already runs upstream, before the
caller builds obstacle_points. This builder does not decide which points
are near-robot sensor artifacts; ObservedObstacleSnapshot.points is
whatever the caller already decided to project. That stays a separate,
earlier, explicit phase until the producer itself migrates.

Boundary rule: pure with respect to simulator state. Never imports
engine.py, Qt/PySide/PyQt, MainWindow, canvas, or robotics_sim.simulation.
config. Never mutates BeliefMap, HazardBelief, or any input snapshot. Builds
a brand-new OccupancyGrid every call and returns an immutable
PlanningCostmapSnapshot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from robotics_interfaces.observations import Point2D
from robotics_sim.environment.grid_geometry import GridCell, GridGeometry
from robotics_sim.environment.hazard_belief import HazardBeliefFrame
from robotics_sim.environment.map_snapshots import FREE as SNAPSHOT_FREE
from robotics_sim.environment.map_snapshots import OCCUPIED as SNAPSHOT_OCCUPIED
from robotics_sim.environment.map_snapshots import (
    ExplorationMapSnapshot,
    ObservedObstacleSnapshot,
)
from robotics_sim.environment.occupancy_grid import (
    FREE as OG_FREE,
    UNKNOWN as OG_UNKNOWN,
    OccupancyGrid,
)
from robotics_sim.planning.costmap_snapshot import PlanningCostmapSnapshot
from robotics_sim.planning.planning_costmap import apply_hazard_belief_to_planning_grid


def _validate_strict_bool(value, field_name: str) -> bool:
    """Accept only an actual bool (or numpy.bool_) -- never bool(value),
    which would silently accept 0/1/"True"/None/[]/() by truthiness instead
    of rejecting them as the wrong type.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    raise ValueError(
        f"{field_name} must be a bool (or numpy.bool_), got {value!r} ({type(value).__name__})."
    )


def _validate_real(
    value,
    *,
    field_name: str,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    """Accept only a real, non-boolean number -- int/float/numpy.integer/
    numpy.floating -- and normalize it to a plain Python float.

    Rejects bool/numpy.bool_ (bool is a subclass of int in Python, so a
    bare isinstance(x, (int, float)) check would otherwise accept it),
    strings, None, lists/tuples, and non-finite values (NaN/inf/-inf).
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a real number, got bool {value!r}.")

    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field_name} must be a real number, got {value!r} ({type(value).__name__}).")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}.")

    if minimum is not None:
        if minimum_inclusive and not result >= minimum:
            raise ValueError(f"{field_name} must be >= {minimum}, got {result!r}.")
        if not minimum_inclusive and not result > minimum:
            raise ValueError(f"{field_name} must be > {minimum}, got {result!r}.")

    if maximum is not None:
        if maximum_inclusive and not result <= maximum:
            raise ValueError(f"{field_name} must be <= {maximum}, got {result!r}.")
        if not maximum_inclusive and not result < maximum:
            raise ValueError(f"{field_name} must be < {maximum}, got {result!r}.")

    return result


def _validate_dynamic_obstacle_points(points) -> tuple[Point2D, ...]:
    """Validate and tuple-ize dynamic_obstacle_points -- an iterable of
    (x, y) pairs, normalized the same way ObservedObstacleSnapshot's own
    points are (map_snapshots.py's _validate_points()): float-normalized,
    finite-only, never reordered. Duplicated here rather than imported
    (matching this file's existing validators and map_snapshots.py's own
    "duplicate small validators per module" convention) -- kept small and
    self-contained, not a new validation framework.

    Unlike ObservedObstacleSnapshot.points, this ALSO rejects bool
    coordinates explicitly (bool is an int subclass in Python, so a bare
    float(x) would otherwise silently accept True/False as 1.0/0.0) --
    dynamic points are fresh, caller-constructed live-state samples, not
    data that already passed through a validated snapshot contract.
    """
    try:
        raw_points = list(points)
    except TypeError as exc:
        raise ValueError(
            f"dynamic_obstacle_points must be an iterable of (x, y) pairs, got {points!r}."
        ) from exc

    validated: list[Point2D] = []
    for point in raw_points:
        try:
            x, y = point
        except (TypeError, ValueError) as exc:
            raise ValueError(f"each dynamic obstacle point must be an (x, y) pair, got {point!r}.") from exc

        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError(f"dynamic obstacle point coordinates must not be bool, got {point!r}.")

        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dynamic obstacle point coordinates must be numeric, got {point!r}.") from exc

        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"dynamic obstacle point coordinates must be finite, got {point!r}.")

        validated.append((x, y))

    return tuple(validated)


@dataclass(frozen=True)
class PlanningCostmapPolicy:
    """Explicit traversability/inflation policy for one build() call.

    obstacle_padding is the single shared inflation radius applied to
    observed obstacle points AND hazard cells -- matching
    build_planning_grid_for_robot()'s current single `radius` value (see
    module docstring). It is NOT applied to legacy exploration-belief
    OCCUPIED cells, which this builder never treats as physical occupancy
    (see module docstring's "Legacy belief occupancy vs. observed obstacle
    geometry"). It corresponds to whatever the caller's robot/safety radius
    currently resolves to
    (SimulationControllerMixin.safety_radius_for_robot()) -- this policy
    object does not read that itself, the caller supplies the number.

    All numeric/boolean fields are strictly validated and normalized in
    __post_init__ (never a silent coercion): unknown_is_traversable is
    stored as a real bool, obstacle_padding and hazard_block_threshold
    (when not None) are stored as real floats.
    """

    unknown_is_traversable: bool
    obstacle_padding: float
    hazard_block_threshold: float | None = None

    def __post_init__(self) -> None:
        unknown_is_traversable = _validate_strict_bool(
            self.unknown_is_traversable, "unknown_is_traversable"
        )
        obstacle_padding = _validate_real(
            self.obstacle_padding,
            field_name="obstacle_padding",
            minimum=0.0,
            minimum_inclusive=True,
        )

        hazard_block_threshold: float | None
        if self.hazard_block_threshold is None:
            hazard_block_threshold = None
        else:
            # (0, 1]: a threshold of exactly 0 would block every observed
            # cell regardless of value, which is never a meaningful policy.
            hazard_block_threshold = _validate_real(
                self.hazard_block_threshold,
                field_name="hazard_block_threshold",
                minimum=0.0,
                minimum_inclusive=False,
                maximum=1.0,
                maximum_inclusive=True,
            )

        object.__setattr__(self, "unknown_is_traversable", unknown_is_traversable)
        object.__setattr__(self, "obstacle_padding", obstacle_padding)
        object.__setattr__(self, "hazard_block_threshold", hazard_block_threshold)


@dataclass(frozen=True)
class _HazardBeliefFrameAdapter:
    """Presents a HazardBeliefFrame through the narrow read-only surface
    apply_hazard_belief_to_planning_grid() expects from a HazardBelief
    (.shape, .geometry, .blocked_cells()) -- without needing the mutable
    HazardBelief class itself, and without modifying planning_costmap.py.

    .blocked_cells() replicates HazardBelief.blocked_cells()'s own one-line
    formula (observed AND value >= threshold) against the frame's arrays --
    the frame carries the same underlying values, just already copied out.
    """

    frame: HazardBeliefFrame
    geometry: GridGeometry

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame.values.shape

    def blocked_cells(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """threshold is expected to already satisfy PlanningCostmapPolicy's
        own (0, 1] validation before this is ever called from build() --
        re-checked defensively here too, since this adapter's contract does
        not otherwise guarantee that precondition on its own.
        """
        threshold = float(threshold)
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold!r}.")
        blocked_mask = self.frame.observed & (self.frame.values >= threshold)
        return np.where(blocked_mask)


def _validate_matching_geometry(
    exploration: ExplorationMapSnapshot,
    observed_obstacles: ObservedObstacleSnapshot,
) -> None:
    if observed_obstacles.bounds != exploration.bounds:
        raise ValueError(
            f"observed_obstacles.bounds {observed_obstacles.bounds!r} does not match "
            f"exploration.bounds {exploration.bounds!r}."
        )
    if abs(observed_obstacles.resolution - exploration.resolution) > 1e-9:
        raise ValueError(
            f"observed_obstacles.resolution {observed_obstacles.resolution!r} does not match "
            f"exploration.resolution {exploration.resolution!r}."
        )


def _validate_hazard_inputs(
    hazard_belief: HazardBeliefFrame | None,
    hazard_geometry: GridGeometry | None,
    exploration: ExplorationMapSnapshot,
) -> None:
    """hazard_belief and hazard_geometry must be provided together or both
    omitted -- HazardBeliefFrame carries no geometry of its own (see module
    docstring's "Hazard geometry" section), so this builder never invents
    one from exploration.
    """
    if (hazard_belief is None) != (hazard_geometry is None):
        raise ValueError(
            "hazard_belief and hazard_geometry must both be provided together or both "
            f"omitted (hazard_belief is None: {hazard_belief is None}, "
            f"hazard_geometry is None: {hazard_geometry is None})."
        )

    if hazard_belief is None:
        return

    if hazard_geometry.bounds != exploration.bounds:
        raise ValueError(
            f"hazard_geometry.bounds {hazard_geometry.bounds!r} does not match "
            f"exploration.bounds {exploration.bounds!r}."
        )
    if abs(hazard_geometry.resolution - exploration.resolution) > 1e-9:
        raise ValueError(
            f"hazard_geometry.resolution {hazard_geometry.resolution!r} does not match "
            f"exploration.resolution {exploration.resolution!r}."
        )
    if hazard_belief.values.shape != exploration.grid.shape:
        raise ValueError(
            f"hazard_belief.values.shape {hazard_belief.values.shape!r} does not match "
            f"exploration.grid.shape {exploration.grid.shape!r}."
        )
    if hazard_belief.observed.shape != exploration.grid.shape:
        raise ValueError(
            f"hazard_belief.observed.shape {hazard_belief.observed.shape!r} does not match "
            f"exploration.grid.shape {exploration.grid.shape!r}."
        )


def _project_exploration_grid(
    exploration: ExplorationMapSnapshot,
    *,
    unknown_is_traversable: bool,
) -> OccupancyGrid:
    """Projects ExplorationMapSnapshot.grid into a base OccupancyGrid.

    Both FREE and legacy OCCUPIED belief cells are treated as observed and
    traversable here -- deliberately NOT as physical occupancy (see module
    docstring's "Legacy belief occupancy vs. observed obstacle geometry"
    section). UNKNOWN cells are left at whatever initial_value
    unknown_is_traversable already resolved to. Physical occupancy is
    projected afterwards, in build(), from ObservedObstacleSnapshot.points
    and observed hazard cells only -- never from this function.
    """
    initial_value = OG_FREE if unknown_is_traversable else OG_UNKNOWN

    grid = OccupancyGrid.from_bounds(
        x_min=exploration.bounds[0],
        x_max=exploration.bounds[1],
        y_min=exploration.bounds[2],
        y_max=exploration.bounds[3],
        resolution=exploration.resolution,
        initial_value=initial_value,
        unknown_is_traversable=unknown_is_traversable,
    )

    observed_rows, observed_cols = np.where(
        (exploration.grid == SNAPSHOT_FREE) | (exploration.grid == SNAPSHOT_OCCUPIED)
    )
    for row, col in zip(observed_rows, observed_cols):
        grid.set_value(GridCell(int(row), int(col)), OG_FREE)

    return grid


class PlanningCostmapBuilder:
    """Builds a PlanningCostmapSnapshot from environment/exploration
    snapshots -- see module docstring for the exact runtime contract this
    reproduces and what it deliberately does not do (no A*, no
    reachability, no frontier detection, no BeliefMap mutation).
    """

    def build(
        self,
        *,
        exploration: ExplorationMapSnapshot,
        observed_obstacles: ObservedObstacleSnapshot,
        policy: PlanningCostmapPolicy,
        dynamic_obstacle_points: tuple[Point2D, ...] = (),
        hazard_belief: HazardBeliefFrame | None = None,
        hazard_geometry: GridGeometry | None = None,
    ) -> PlanningCostmapSnapshot:
        _validate_matching_geometry(exploration, observed_obstacles)
        _validate_hazard_inputs(hazard_belief, hazard_geometry, exploration)
        validated_dynamic_points = _validate_dynamic_obstacle_points(dynamic_obstacle_points)

        # 1. exploration belief -> base grid. Legacy belief-OCCUPIED cells
        #    become FREE here -- observed-but-not-independently-confirmed
        #    occupancy is not physical occupancy (see module docstring's
        #    "Legacy belief occupancy vs. observed obstacle geometry").
        grid = _project_exploration_grid(
            exploration,
            unknown_is_traversable=policy.unknown_is_traversable,
        )

        padding = max(0.0, float(policy.obstacle_padding))

        # 2. observed STATIC obstacle projection/inflation.
        if observed_obstacles.points:
            grid.add_obstacle_points(observed_obstacles.points, padding=padding)

        # 3. DYNAMIC obstacle point projection/inflation -- the SAME
        #    padding as static observed points (see module docstring's
        #    "Dynamic obstacle points"). Ephemeral, per-call input: never
        #    part of ObservedObstacleSnapshot, never tracked in
        #    source_revisions. A point already present in
        #    observed_obstacles.points is simply re-marked OCCUPIED here --
        #    OccupancyGrid has no notion of cumulative "more occupied", so
        #    this is a harmless no-op, not a double-inflation bug.
        if validated_dynamic_points:
            grid.add_obstacle_points(validated_dynamic_points, padding=padding)

        source_revisions: list[tuple[str, int]] = [
            ("exploration", exploration.revision),
            ("observed_obstacles", observed_obstacles.revision),
        ]

        # 4. observed hazard projection -- only ever observed cells, never
        #    ground truth. Applied last, matching build_planning_grid_for_
        #    robot()'s current composition order.
        if hazard_belief is not None:
            if policy.hazard_block_threshold is None:
                raise ValueError(
                    "policy.hazard_block_threshold is required when hazard_belief is provided."
                )

            adapter = _HazardBeliefFrameAdapter(frame=hazard_belief, geometry=hazard_geometry)
            apply_hazard_belief_to_planning_grid(
                grid,
                adapter,
                block_threshold=policy.hazard_block_threshold,
                inflate_radius=padding,
            )
            source_revisions.append(("hazard", hazard_belief.revision))

        # exploration.bounds/exploration.resolution -- not grid.bounds/
        # grid.resolution -- are the canonical geometry here. OccupancyGrid.
        # from_bounds() stores back an internally *expanded* max bound
        # (x_min + width*resolution, y_min + height*resolution) so the grid
        # covers a whole number of cells; for some decimal bounds/resolution
        # combinations that expanded bound is not exactly representable in
        # float64, and re-deriving width/height from it a second time (as
        # PlanningCostmapSnapshot's own geometry check does) can then land
        # on a different ceil() result than the original bounds did --
        # even though grid.data itself is unaffected either way. exploration
        # was already validated as internally coherent (ExplorationMapSnapshot's
        # own __post_init__), so it is the geometric source of truth this
        # builder was given -- not a value OccupancyGrid derived and rounded
        # internally for its own bookkeeping.
        return PlanningCostmapSnapshot(
            grid=grid.data,
            bounds=exploration.bounds,
            resolution=exploration.resolution,
            unknown_is_traversable=policy.unknown_is_traversable,
            source_revisions=tuple(source_revisions),
        )
