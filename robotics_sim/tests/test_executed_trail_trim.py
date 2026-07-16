"""
Regression tests for SimulationControllerMixin._append_executed_path_point().

Real Office.sim evidence: once executed_trail_points reached 1200, render
started failing repeatedly -- executed_trail_cache_hit=False on nearly every
frame, executed_trail_build_ms climbing to 5-10ms+, executed_trail_segments_
painted back up near the full trail length, and route_path_ms losing its
stability (spikes as high as route_path_ms=197.2, total_ms=283.4).

Root cause: the trail trim previously fired as soon as len(path_points)
exceeded EXECUTED_TRAIL_MAX_POINTS (1200) -- but since exactly one point is
appended every tick, once the trail reaches the cap this condition is true
on EVERY subsequent tick, so `self.path_points = self.path_points[-1200:]`
replaced path_points with a brand-new list object every single tick,
forever, once the cap was first reached. SimulationCanvas's executed-trail
pixmap cache (draw_executed_path() in simulation_canvas.py) uses object
IDENTITY to distinguish "grew in place" (cheap incremental append) from
"replaced/truncated" (full pixmap rebuild) -- so this permanently defeated
the cache the moment the trail hit 1200 points, forcing a full rebuild of
the whole trail every frame forever after.

Fix: trim only once EXECUTED_TRAIL_TRIM_MARGIN extra points accumulate past
the cap (not on the very next tick), so the identity change -- and the
rebuild it forces -- happens once every EXECUTED_TRAIL_TRIM_MARGIN ticks
instead of every tick.

Exercises _append_executed_path_point() directly via a lightweight
duck-typed engine fake, the same pattern used throughout this test suite
(see test_exhausted_hold_perf.py) -- no Qt/canvas/planner stack needed.
"""
from __future__ import annotations

from types import SimpleNamespace

from robotics_sim.simulation.engine import (
    EXECUTED_TRAIL_MAX_POINTS,
    EXECUTED_TRAIL_TRIM_MARGIN,
    SimulationControllerMixin,
)


def _make_fake_engine() -> SimpleNamespace:
    fake = SimpleNamespace(path_points=[], total_distance_traveled=0.0)
    fake._append_executed_path_point = (
        SimulationControllerMixin._append_executed_path_point.__get__(fake)
    )
    return fake


# ---------------------------------------------------------------------------
# A. Below the cap, every append grows the SAME list object (identity
#    preserved) -- this is the case the pixmap cache appends to
#    incrementally, and must never be disturbed.
# ---------------------------------------------------------------------------


def test_append_below_cap_never_replaces_list_object():
    fake = _make_fake_engine()
    original = fake.path_points

    for i in range(EXECUTED_TRAIL_MAX_POINTS - 1):
        fake._append_executed_path_point((float(i), 0.0))

    assert fake.path_points is original
    assert len(fake.path_points) == EXECUTED_TRAIL_MAX_POINTS - 1


# ---------------------------------------------------------------------------
# B. Crossing the cap alone does NOT trim -- only crossing cap + margin
#    does. This is the actual bug fix: the old code trimmed (and thus
#    replaced the list object) the instant len() exceeded the cap.
# ---------------------------------------------------------------------------


def test_crossing_cap_alone_does_not_trim():
    fake = _make_fake_engine()

    for i in range(EXECUTED_TRAIL_MAX_POINTS + 1):
        fake._append_executed_path_point((float(i), 0.0))

    assert len(fake.path_points) == EXECUTED_TRAIL_MAX_POINTS + 1, (
        "must be allowed to grow past the cap, up to the margin, without trimming yet"
    )


def test_object_identity_preserved_for_margin_ticks_past_the_cap():
    """This is the exact regression: once the trail hits the cap, identity
    must survive EXECUTED_TRAIL_TRIM_MARGIN more ticks, not just one --
    otherwise the pixmap cache rebuilds every single frame forever."""
    fake = _make_fake_engine()

    for i in range(EXECUTED_TRAIL_MAX_POINTS):
        fake._append_executed_path_point((float(i), 0.0))
    original = fake.path_points

    for i in range(EXECUTED_TRAIL_TRIM_MARGIN):
        fake._append_executed_path_point((float(EXECUTED_TRAIL_MAX_POINTS + i), 0.0))
        assert fake.path_points is original, (
            f"tick {i} past the cap replaced the list object -- the pixmap cache "
            "would incorrectly rebuild every frame instead of every "
            "EXECUTED_TRAIL_TRIM_MARGIN ticks"
        )


# ---------------------------------------------------------------------------
# C. Once cap + margin is exceeded, trim fires (new list object), and the
#    trail is trimmed back down to exactly the cap, not the cap + margin.
# ---------------------------------------------------------------------------


def test_trim_fires_once_margin_is_exceeded_and_resets_to_cap():
    fake = _make_fake_engine()

    total_ticks = EXECUTED_TRAIL_MAX_POINTS + EXECUTED_TRAIL_TRIM_MARGIN + 1
    for i in range(total_ticks):
        fake._append_executed_path_point((float(i), 0.0))

    assert len(fake.path_points) == EXECUTED_TRAIL_MAX_POINTS, (
        "trim must reset the trail back down to the cap, not leave it at cap + margin"
    )
    # The most recent points must be preserved (a sliding window, not an
    # arbitrary reset) -- the last point appended must still be the last
    # point in the trimmed trail.
    assert fake.path_points[-1] == (float(total_ticks - 1), 0.0)


def test_trim_cycle_repeats_indefinitely():
    """Confirms the trim/grow cycle keeps working correctly across MULTIPLE
    trim events, not just the first one -- a long-running simulation
    crosses this boundary repeatedly. After the first trim (at cap +
    margin), each further trim happens every EXECUTED_TRAIL_TRIM_MARGIN
    ticks -- definitely NOT every tick, which is the exact bug this round
    fixes (that would mean one identity change per tick, i.e.
    identity_changes == total_ticks - EXECUTED_TRAIL_MAX_POINTS)."""
    fake = _make_fake_engine()
    identity_changes = 0
    previous = fake.path_points

    total_ticks = 3 * (EXECUTED_TRAIL_MAX_POINTS + EXECUTED_TRAIL_TRIM_MARGIN)
    for i in range(total_ticks):
        fake._append_executed_path_point((float(i), 0.0))
        if fake.path_points is not previous:
            identity_changes += 1
            previous = fake.path_points
        assert len(fake.path_points) <= EXECUTED_TRAIL_MAX_POINTS + EXECUTED_TRAIL_TRIM_MARGIN

    # One identity change roughly every EXECUTED_TRAIL_TRIM_MARGIN ticks
    # after the trail first fills up -- a small, bounded number of trims,
    # never one per tick.
    expected = (total_ticks - EXECUTED_TRAIL_MAX_POINTS) // EXECUTED_TRAIL_TRIM_MARGIN
    assert abs(identity_changes - expected) <= 1
    assert identity_changes < total_ticks / 10, (
        "identity must not change on nearly every tick -- that is the exact "
        "bug this round fixes (a full pixmap rebuild every frame)"
    )
