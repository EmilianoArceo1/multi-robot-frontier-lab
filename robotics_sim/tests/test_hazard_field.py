"""
Pure unit tests for HazardField/FireSource/RuntimeHazardService
(robotics_sim.environment.hazard_field / robotics_sim.simulation.hazard_
service) -- no engine, no BeliefMap coupling. See test_fire_hazards.py for
the engine-level add_fire()/remove_fire_near() entry points and their
(lack of) effect on occupancy, and test_planning_costmap_hazard.py for how
this field projects into a planning grid.
"""
from __future__ import annotations

from robotics_sim.environment.hazard_field import FireSource, HazardField
from robotics_sim.simulation.hazard_service import RuntimeHazardService

_BOUNDS = (-10.0, 10.0, -10.0, 10.0)
_RESOLUTION = 0.5


def _make_field() -> HazardField:
    return HazardField(bounds=_BOUNDS, resolution=_RESOLUTION)


# ---------------------------------------------------------------------------
# 5. A fire appears in HazardField.sources().
# ---------------------------------------------------------------------------


def test_added_fire_appears_in_sources():
    field = _make_field()

    source = field.add_fire((1.0, 1.0), intensity=0.8, radius=1.5)

    sources = field.sources()
    assert len(sources) == 1
    assert sources[0] == source
    assert sources[0].position == (1.0, 1.0)
    assert sources[0].intensity == 0.8
    assert sources[0].radius == 1.5


def test_removed_fire_no_longer_appears_in_sources():
    field = _make_field()
    source = field.add_fire((1.0, 1.0))

    field.remove_fire(source.fire_id)

    assert field.sources() == ()


# ---------------------------------------------------------------------------
# 6. Heat appears in HazardField.values() near the source, and only near it.
# ---------------------------------------------------------------------------


def test_heat_appears_in_values_near_the_source():
    field = _make_field()
    # Placed exactly on a cell center (resolution=0.5 -> centers at ..., -0.25,
    # 0.25, ...) so that cell's distance-to-source is exactly 0.
    field.add_fire((0.25, 0.25), intensity=1.0, radius=2.0)

    values = field.values(copy=False)
    assert values.max() > 0.0

    center_cell = field.geometry.world_to_grid(0.25, 0.25)
    assert values[center_cell.row, center_cell.col] == 1.0  # distance 0 -> full intensity

    far_cell = field.geometry.world_to_grid(9.0, 9.0)
    assert values[far_cell.row, far_cell.col] == 0.0  # outside the fire's radius


def test_values_are_all_zero_with_no_sources():
    field = _make_field()
    assert not field.values(copy=False).any()


def test_removing_the_only_fire_zeroes_the_heat():
    field = _make_field()
    source = field.add_fire((0.0, 0.0), intensity=1.0, radius=2.0)

    field.remove_fire(source.fire_id)

    assert not field.values(copy=False).any()


# ---------------------------------------------------------------------------
# 9. Two fires are independently removable (pure HazardField level -- see
# test_fire_hazards.py for the engine-entry-point equivalent).
# ---------------------------------------------------------------------------


def test_two_fires_are_independently_removable():
    field = _make_field()
    first = field.add_fire((1.0, 1.0))
    second = field.add_fire((5.0, 5.0))

    removed = field.remove_fire(first.fire_id)

    assert removed == first
    assert field.sources() == (second,)
    assert field.values(copy=False).max() > 0.0  # second fire's heat remains

    field.remove_fire(second.fire_id)
    assert field.sources() == ()
    assert not field.values(copy=False).any()


def test_remove_nearest_fire_only_removes_the_closest_one():
    field = _make_field()
    near = field.add_fire((1.0, 1.0))
    field.add_fire((8.0, 8.0))

    removed = field.remove_nearest_fire((1.1, 1.1), max_distance=1.0)

    assert removed == near
    assert len(field.sources()) == 1
    assert field.sources()[0].position == (8.0, 8.0)


# ---------------------------------------------------------------------------
# restore_sources() -- used by navigation-debug snapshot restore (see
# test_navigation_snapshot_restore.py for the full engine-level round trip).
# ---------------------------------------------------------------------------


def test_restore_sources_replaces_the_source_set_and_rebuilds_heat():
    field = _make_field()
    field.add_fire((1.0, 1.0))
    stale_id = field.sources()[0].fire_id

    restored = (FireSource(fire_id=7, position=(3.25, 3.25), intensity=0.5, radius=1.0),)
    version_before = field.version

    field.restore_sources(restored, next_fire_id=8)

    assert field.sources() == restored
    assert field.next_fire_id == 8
    assert field.version > version_before
    cell = field.geometry.world_to_grid(3.25, 3.25)
    assert field.values(copy=False)[cell.row, cell.col] > 0.0
    assert stale_id not in {s.fire_id for s in field.sources()}


def test_restore_sources_to_empty_clears_all_heat():
    field = _make_field()
    field.add_fire((1.0, 1.0))

    field.restore_sources((), next_fire_id=1)

    assert field.sources() == ()
    assert not field.values(copy=False).any()


def test_restore_sources_next_fire_id_is_used_by_the_next_add():
    field = _make_field()

    field.restore_sources((), next_fire_id=41)
    new_source = field.add_fire((0.0, 0.0))

    assert new_source.fire_id == 41


# ---------------------------------------------------------------------------
# RuntimeHazardService thin wrapper: same guarantees through the service
# layer the engine actually calls (add_fire()/remove_fire_near()/
# toggle_fire_at()).
# ---------------------------------------------------------------------------


def test_service_add_and_remove_round_trip():
    service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)

    added = service.add_fire((2.0, 2.0))
    assert added.action == "added"
    assert len(service.sources()) == 1

    removed = service.remove_fire_near((2.0, 2.0))
    assert removed.action == "removed"
    assert service.sources() == ()


def test_service_blocked_world_points_respects_threshold():
    service = RuntimeHazardService(
        bounds=_BOUNDS, resolution=_RESOLUTION, default_intensity=1.0, default_radius=2.0, block_threshold=0.9
    )
    # Exactly on a cell center (see test_heat_appears_in_values_near_the_
    # source) so its contribution is the full intensity, comfortably over
    # the 0.9 threshold.
    service.add_fire((0.25, 0.25))

    blocked = service.blocked_world_points()

    assert (0.25, 0.25) in blocked
    far_blocked = [p for p in blocked if abs(p[0]) > 5.0 or abs(p[1]) > 5.0]
    assert far_blocked == []
