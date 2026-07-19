"""Host-side adapter wiring HazardBelief -> HazardDistanceField -> HOCBF filter.

This is the ONLY object the simulation engine talks to for Observed Hazard
OGM-HOCBF safety filtering (see robotics_sim/control/hazard_hocbf_filter.py
and robotics_sim/environment/hazard_distance_field.py for the actual math).
``HazardSafetyRuntime`` itself contains no CBF/QP equations and no SDF
construction code -- it only:

  - keeps the ``HazardDistanceFieldFrame`` cached across ticks and rebuilds
    it only when the observed-hazard blocked mask actually changes (the
    builder itself decides reuse-vs-rebuild; this class just counts which
    happened for the metrics below);
  - gathers belief frame / geometry / state / limits / safety radius and
    calls the filter;
  - accumulates small runtime metrics for diagnostics.

It never selects routes or targets -- that stays the planner's job.
"""

from __future__ import annotations

import time

from robotics_sim.control.hazard_hocbf_filter import HazardHOCBFResult, HazardHOCBFSafetyFilter
from robotics_sim.environment.hazard_distance_field import (
    HazardDistanceFieldBuilder,
    HazardDistanceFieldFrame,
)


class HazardSafetyRuntime:
    """Per-simulation-run cache + wiring for the Observed Hazard OGM-HOCBF filter."""

    def __init__(
        self,
        *,
        block_threshold: float,
        margin: float,
        activation_distance: float,
        k1: float,
        k2: float,
        pyramid_levels: int,
        smoothing_sigma_cells: float,
        acceleration_weight: float,
        angular_weight: float,
    ) -> None:
        self.block_threshold = float(block_threshold)
        self.margin = float(margin)
        self.activation_distance = float(activation_distance)
        self.pyramid_levels = int(pyramid_levels)
        self.smoothing_sigma_cells = float(smoothing_sigma_cells)

        self._builder = HazardDistanceFieldBuilder()
        self._filter = HazardHOCBFSafetyFilter(
            k1=k1,
            k2=k2,
            acceleration_weight=acceleration_weight,
            angular_weight=angular_weight,
        )
        self._field_frame: HazardDistanceFieldFrame | None = None

        self.activation_count = 0
        self.intervention_count = 0
        self.infeasible_count = 0
        self.invalid_initial_condition_count = 0
        self.field_rebuild_count = 0
        self.field_reuse_count = 0
        self.maximum_intervention_norm = 0.0
        self.minimum_h_seen: float | None = None

        # Wall-clock diagnostics (time.perf_counter() only -- no printing,
        # no GUI, no threads). field_*_build_ms measure ONLY real rebuilds
        # (see filter_control()): a cache reuse's cheap mask-recompute is
        # deliberately excluded so these numbers reflect actual SDF/pyramid
        # construction cost, not the reuse fast path.
        self.field_last_build_ms = 0.0
        self.field_total_build_ms = 0.0
        self.field_max_build_ms = 0.0
        self.filter_last_ms = 0.0
        self.filter_total_ms = 0.0
        self.filter_max_ms = 0.0

    @property
    def field_frame(self) -> HazardDistanceFieldFrame | None:
        """The currently cached distance field frame, if one has been built yet."""
        return self._field_frame

    def filter_control(
        self,
        *,
        belief_frame,
        geometry,
        state,
        limits,
        nominal_control,
        safety_radius: float,
    ) -> HazardHOCBFResult:
        """Filter one tick's nominal control for one robot.

        ``belief_frame`` must be a ``HazardBeliefFrame`` snapshot (or
        equivalent) -- never ``HazardBelief`` itself, never any ground-truth
        object. ``geometry`` is the belief's ``GridGeometry``. ``state`` and
        ``limits`` are the target robot's ``RobotState``/``RobotLimits``.
        """
        previous_frame = self._field_frame
        build_start = time.perf_counter()
        frame = self._builder.build(
            belief_frame=belief_frame,
            geometry=geometry,
            block_threshold=self.block_threshold,
            pyramid_levels=self.pyramid_levels,
            smoothing_sigma_cells=self.smoothing_sigma_cells,
            previous_frame=previous_frame,
        )
        build_elapsed_ms = (time.perf_counter() - build_start) * 1000.0

        if previous_frame is not None and frame is previous_frame:
            self.field_reuse_count += 1
        else:
            self.field_rebuild_count += 1
            self.field_last_build_ms = build_elapsed_ms
            self.field_total_build_ms += build_elapsed_ms
            if build_elapsed_ms > self.field_max_build_ms:
                self.field_max_build_ms = build_elapsed_ms
        self._field_frame = frame

        filter_start = time.perf_counter()
        result = self._filter.filter(
            distance_field_frame=frame,
            state=state,
            limits=limits,
            nominal_control=nominal_control,
            safety_radius=safety_radius,
            margin=self.margin,
            activation_distance=self.activation_distance,
        )
        filter_elapsed_ms = (time.perf_counter() - filter_start) * 1000.0
        self.filter_last_ms = filter_elapsed_ms
        self.filter_total_ms += filter_elapsed_ms
        if filter_elapsed_ms > self.filter_max_ms:
            self.filter_max_ms = filter_elapsed_ms

        self._record_metrics(result)
        return result

    def _record_metrics(self, result: HazardHOCBFResult) -> None:
        if result.active:
            self.activation_count += 1
        if result.intervention_norm > 0.0:
            self.intervention_count += 1
        if not result.feasible:
            self.infeasible_count += 1
        if not result.initial_condition_valid:
            self.invalid_initial_condition_count += 1
        if result.intervention_norm > self.maximum_intervention_norm:
            self.maximum_intervention_norm = float(result.intervention_norm)
        if result.minimum_h is not None:
            if self.minimum_h_seen is None or result.minimum_h < self.minimum_h_seen:
                self.minimum_h_seen = float(result.minimum_h)
