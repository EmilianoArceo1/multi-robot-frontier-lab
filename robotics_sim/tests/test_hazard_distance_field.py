"""Unit tests for HazardDistanceFieldBuilder/HazardDistanceFieldFrame
(robotics_sim.environment.hazard_distance_field) -- pure geometry built from
an already-observed HazardBeliefFrame. No engine, no CBF/QP math, no ground
truth (HazardField/FireSource) anywhere in this file.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robotics_sim.environment.grid_geometry import GridGeometry
from robotics_sim.environment.hazard_belief import HazardBelief
from robotics_sim.environment.hazard_distance_field import (
    HazardDistanceFieldBuilder,
    HazardDistanceFieldFrame,
)

_BOUNDS = (0.0, 10.0, 0.0, 10.0)
_RESOLUTION = 1.0
_THRESHOLD = 0.55


def _geometry() -> GridGeometry:
    return GridGeometry(_BOUNDS, _RESOLUTION)


def _belief() -> HazardBelief:
    return HazardBelief(_geometry(), robot_count=2)


def _build(belief_frame, geometry=None, *, previous_frame=None, pyramid_levels=2, smoothing_sigma_cells=0.75):
    return HazardDistanceFieldBuilder().build(
        belief_frame=belief_frame,
        geometry=geometry or _geometry(),
        block_threshold=_THRESHOLD,
        pyramid_levels=pyramid_levels,
        smoothing_sigma_cells=smoothing_sigma_cells,
        previous_frame=previous_frame,
    )


# ---------------------------------------------------------------------------
# 1. Belief with no hazards at all.
# ---------------------------------------------------------------------------


def test_belief_without_hazards_has_no_constraints():
    belief = _belief()
    frame = _build(belief.snapshot())

    assert frame.has_hazards is False
    assert frame.levels == ()
    assert frame.sample(5.0, 5.0) == ()


# ---------------------------------------------------------------------------
# 2. A ground-truth-hot cell that was never observed must not leak in.
# ---------------------------------------------------------------------------


def test_unobserved_hot_cell_never_appears_in_unsafe_mask():
    """A hand-built belief frame simulates "ground truth is hot here" by
    giving a cell a high value while observed=False for it -- the builder
    must only ever look at observed AND value>=threshold, never value alone.
    """
    values = np.zeros((10, 10), dtype=np.float32)
    values[5, 5] = 0.99  # hot in "ground truth", but never observed
    observed = np.zeros((10, 10), dtype=bool)  # observed=False everywhere

    fake_frame = SimpleNamespace(values=values, observed=observed, revision=1)
    frame = _build(fake_frame)

    assert frame.has_hazards is False


# ---------------------------------------------------------------------------
# 3. Observed but below threshold is not unsafe.
# ---------------------------------------------------------------------------


def test_observed_cell_below_threshold_is_not_unsafe():
    belief = _belief()
    belief.observe_cells(rows=[5], cols=[5], values=[0.30], robot_index=0)

    frame = _build(belief.snapshot())

    assert frame.has_hazards is False


# ---------------------------------------------------------------------------
# 4. Observed at/above threshold: SDF negative inside, positive outside.
# ---------------------------------------------------------------------------


def test_observed_hazard_above_threshold_has_signed_distance():
    belief = _belief()
    belief.observe_cells(rows=[5], cols=[5], values=[0.90], robot_index=0)

    frame = _build(belief.snapshot())
    assert frame.has_hazards is True

    geometry = _geometry()
    hazard_world = geometry.grid_to_world(geometry.world_to_grid(5.5, 5.5))

    inside_samples = frame.sample(*hazard_world)
    assert inside_samples[0].value < 0.0

    far_samples = frame.sample(0.5, 0.5)
    assert far_samples[0].value > 0.0


# ---------------------------------------------------------------------------
# 5. Gradient points toward greater separation from the unsafe set.
# ---------------------------------------------------------------------------


def test_gradient_points_away_from_hazard():
    belief = _belief()
    belief.observe_cells(rows=[5], cols=[5], values=[0.90], robot_index=0)
    frame = _build(belief.snapshot())

    # Query point is to the right of (and outside) the hazard cell -- the
    # direction of increasing distance from the hazard, at this point, is
    # predominantly +x.
    sample = frame.sample(8.5, 5.5)[0]
    assert sample.gradient[0] > 0.0


# ---------------------------------------------------------------------------
# 6. Distances are expressed in meters, not grid cells.
# ---------------------------------------------------------------------------


def test_distance_values_are_in_meters():
    belief = _belief()
    belief.observe_cells(rows=[5], cols=[5], values=[0.90], robot_index=0)
    frame = _build(belief.snapshot())

    # Query point 3 cells (3 meters, resolution=1.0) away from the hazard
    # cell center -- the discretized value should land near 3.0 meters, not
    # near 3 "pixels" scaled by some other unit.
    sample = frame.sample(8.5, 5.5)[0]
    assert 2.0 <= sample.value <= 4.0


# ---------------------------------------------------------------------------
# 7. Every value/gradient/Hessian produced is finite.
# ---------------------------------------------------------------------------


def test_all_samples_are_finite():
    belief = _belief()
    belief.observe_cells(rows=[5], cols=[5], values=[0.90], robot_index=0)
    frame = _build(belief.snapshot())

    for level in frame.levels:
        assert np.isfinite(level.value).all()
        assert np.isfinite(level.gradient_x).all()
        assert np.isfinite(level.gradient_y).all()
        assert np.isfinite(level.hessian_xx).all()
        assert np.isfinite(level.hessian_xy).all()
        assert np.isfinite(level.hessian_yy).all()

    for x, y in [(0.5, 0.5), (5.5, 5.5), (9.5, 9.5)]:
        for sample in frame.sample(x, y):
            assert np.isfinite(sample.value)
            assert np.isfinite(sample.gradient).all()
            assert np.isfinite(sample.hessian).all()


# ---------------------------------------------------------------------------
# 8. Pyramid levels keep correct (shared, world) bounds.
# ---------------------------------------------------------------------------


def test_pyramid_levels_keep_correct_bounds():
    belief = _belief()
    belief.observe_cells(rows=[5], cols=[5], values=[0.90], robot_index=0)
    frame = _build(belief.snapshot(), pyramid_levels=2)

    assert len(frame.levels) == 2
    assert frame.bounds == _BOUNDS
    assert frame.levels[1].resolution == frame.levels[0].resolution * 2.0


# ---------------------------------------------------------------------------
# 9. Attribution-only change (no blocked-mask change) reuses the field.
# ---------------------------------------------------------------------------


def test_attribution_only_change_reuses_field():
    belief = _belief()
    belief.observe_cells(rows=[5], cols=[5], values=[0.90], robot_index=0)
    frame1 = _build(belief.snapshot())
    revision_before = belief.revision

    # A different robot re-observes the SAME cell with the SAME value --
    # this bumps HazardBelief.revision (newly_attributed) but must not
    # change the unsafe mask at all.
    belief.observe_cells(rows=[5], cols=[5], values=[0.90], robot_index=1)
    assert belief.revision != revision_before

    frame2 = _build(belief.snapshot(), previous_frame=frame1)

    assert frame2 is frame1


# ---------------------------------------------------------------------------
# 10. A real blocked-mask change forces a rebuild.
# ---------------------------------------------------------------------------


def test_blocked_mask_change_forces_rebuild():
    belief = _belief()
    belief.observe_cells(rows=[5], cols=[5], values=[0.90], robot_index=0)
    frame1 = _build(belief.snapshot())

    belief.observe_cells(rows=[2], cols=[2], values=[0.95], robot_index=0)
    frame2 = _build(belief.snapshot(), previous_frame=frame1)

    assert frame2 is not frame1
    assert frame2.has_hazards is True


# ---------------------------------------------------------------------------
# Audit: Hessian stability across representative regions.
#
# Regions covered: a flat area far from any hazard, a straight blob edge, a
# blob corner, the blob's interior/center, an exact cell-boundary sample
# (grid_geometry's cell centers land on half-integers at resolution=0.5, so
# 0.5-spaced query points sit exactly on the interpolation seam), and a
# sample close to the world bounds (map border, where _safe_gradient2d's
# edge-padding is exercised). All of these must stay finite and within a
# generous, resolution-aware bound -- not exact values, since the SDF/
# gradient/Hessian are numerical approximations (see module docstring).
# ---------------------------------------------------------------------------


def _blob_belief_frame(resolution: float, pyramid_levels: int = 1):
    geometry = GridGeometry(_BOUNDS, resolution)
    belief = HazardBelief(geometry, robot_count=1)
    center_x, center_y = 8.25, 5.75
    rows: list[int] = []
    cols: list[int] = []
    for dy in np.arange(-0.75, 0.76, resolution):
        for dx in np.arange(-0.75, 0.76, resolution):
            cell = geometry.world_to_grid(center_x + dx, center_y + dy)
            if cell is not None:
                rows.append(cell.row)
                cols.append(cell.col)
    belief.observe_cells(rows=rows, cols=cols, values=[0.9] * len(rows), robot_index=0)
    frame = HazardDistanceFieldBuilder().build(
        belief_frame=belief.snapshot(),
        geometry=geometry,
        block_threshold=_THRESHOLD,
        pyramid_levels=pyramid_levels,
        smoothing_sigma_cells=0.75,
    )
    return frame, geometry


def test_hessian_is_finite_and_bounded_across_representative_regions():
    resolution = 0.5
    frame, geometry = _blob_belief_frame(resolution)

    # A generous, resolution-aware bound: Hessian entries are second finite
    # differences with spacing ~resolution, so their magnitude legitimately
    # grows as resolution shrinks (see the audit report); this only catches
    # true blow-ups (NaN/Inf or values orders of magnitude beyond what a
    # bounded, twice-differentiated distance-like field should produce).
    magnitude_cap = 20.0 / (resolution ** 2)

    regions = {
        "flat_far": (1.0, 1.0),
        "straight_edge": (7.4, 5.75),
        "corner": (7.6, 4.6),
        "hazard_center": (8.25, 5.75),
        "cell_transition": (7.5, 5.5),  # exactly on a cell-center seam
        "map_border": (geometry.x_min + 1e-3, geometry.y_min + 1e-3),
        "map_border_far_corner": (geometry.x_max - 1e-3, geometry.y_max - 1e-3),
    }

    for name, (x, y) in regions.items():
        sample = frame.sample(x, y)[0]
        assert np.isfinite(sample.value), name
        assert np.isfinite(sample.gradient).all(), name
        assert np.isfinite(sample.hessian).all(), name
        assert np.all(np.abs(sample.hessian) <= magnitude_cap), (
            name,
            sample.hessian,
            magnitude_cap,
        )


def test_hessian_magnitude_grows_as_resolution_refines():
    """Documents (does not silently hide) a real limitation: the same
    absolute curvature produces a larger discretized Hessian estimate at
    finer resolutions, since Hessian entries are second finite differences
    with spacing ~resolution -- noise/curvature-estimate magnitude scales
    roughly like 1/resolution^2. hazard_cbf_sdf_smoothing_sigma_cells is
    specified in CELL units, so it does not compensate for this in absolute
    (meter) terms. See the audit report for the practical implication on
    QP conditioning at fine resolutions."""
    magnitudes = []
    for resolution in (1.0, 0.5, 0.25):
        frame, _ = _blob_belief_frame(resolution)
        sample = frame.sample(8.25, 5.75)[0]  # hazard center, every resolution
        magnitudes.append(float(np.max(np.abs(sample.hessian))))
        assert np.isfinite(sample.hessian).all()

    assert magnitudes[0] <= magnitudes[1] <= magnitudes[2]
