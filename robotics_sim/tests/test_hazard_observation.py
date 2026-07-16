"""
Pure unit tests for RuntimeHazardService.observe_visible_polygon()
(robotics_sim.simulation.hazard_service) -- the Phase 2 bridge between the
ground-truth HazardField and the team HazardBelief.

Creating/removing a FireSource only ever touches ground truth (see
test_fire_hazards.py / test_hazard_field.py); a cell's belief only changes
the next time it is actually re-observed through a real, occlusion-aware
sensor polygon -- never a geometric radius/circle approximation.

Tests 7-9 and 13-14 exercise engine.update_explored_free_points_from_
polygon() through a minimal duck-typed engine fake (real BeliefMap + real
RuntimeHazardService, same lightweight-fake pattern used throughout this
test suite) because that is where the visual-canvas robot_index=None ->
belief robot 0 convention is actually resolved, and where occupancy/
explored_by_robot non-interference has to be proven against the real call
site, not re-implemented here.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from robotics_sim.environment.belief_map import BeliefMap, FREE, OCCUPIED, UNKNOWN
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.hazard_service import HazardObservationResult, RuntimeHazardService

_BOUNDS = (0.0, 10.0, 0.0, 10.0)
_RESOLUTION = 1.0  # -> 10x10 grid, cell centers at 0.5, 1.5, ..., 9.5


def _make_service(robot_count: int = 1) -> RuntimeHazardService:
    return RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=robot_count)


def _square_polygon(x0: float, y0: float, x1: float, y1: float) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _l_shaped_polygon() -> list[tuple[float, float]]:
    """Covers x in [0,6) for y in [0,3), and x in [0,3) for y in [3,6) --
    excludes the [3,6) x [3,6) quadrant entirely, standing in for a wall
    occluding that region. Used to prove rasterization is a real point-in-
    polygon test, not a bounding-box/circle approximation."""
    return [(0.0, 0.0), (6.0, 0.0), (6.0, 3.0), (3.0, 3.0), (3.0, 6.0), (0.0, 6.0)]


def _build_fake_engine(*, robot_count: int = 1) -> SimpleNamespace:
    belief_map = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=robot_count)
    hazard_service = _make_service(robot_count=robot_count)
    fake = SimpleNamespace(
        belief_map=belief_map,
        hazard_service=hazard_service,
        simulation_time=3.0,
        explored_free_points=set(),
    )
    fake.ensure_belief_map = lambda: fake.belief_map
    fake.update_explored_free_points_from_polygon = (
        SimulationControllerMixin.update_explored_free_points_from_polygon.__get__(fake)
    )
    return fake


# ---------------------------------------------------------------------------
# 1. Service starts with a fully unobserved HazardBelief.
# ---------------------------------------------------------------------------


def test_service_starts_with_fully_unobserved_belief():
    service = _make_service()

    frame = service.belief.snapshot()

    assert not frame.observed.any()
    assert (frame.values == 0.0).all()
    assert frame.revision == 0


# ---------------------------------------------------------------------------
# 2. Creating a fire outside any observed FoV never touches the belief.
# ---------------------------------------------------------------------------


def test_creating_fire_outside_fov_does_not_modify_belief():
    service = _make_service()

    service.add_fire((8.5, 8.5))

    frame = service.belief.snapshot()
    assert not frame.observed.any()
    assert frame.revision == 0


# ---------------------------------------------------------------------------
# 3-4. A visible safe cell is observed with value 0; a visible hot cell
# copies the ground-truth value exactly.
# ---------------------------------------------------------------------------


def test_visible_safe_cell_is_observed_with_zero_value():
    service = _make_service()
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)  # covers cells (row,col) 4,5,6

    result = service.observe_visible_polygon(polygon, robot_index=0)

    cell = service.field.geometry.world_to_grid(5.5, 5.5)
    frame = service.belief.snapshot()
    assert frame.observed[cell.row, cell.col] == True  # noqa: E712
    assert frame.values[cell.row, cell.col] == 0.0
    assert result.changed is True
    assert result.newly_observed_cells == 9  # 3x3 block


def test_visible_hot_cell_copies_ground_truth_value():
    service = _make_service()
    # service.add_fire() only takes a position (uses default_intensity=1.0);
    # use the field API directly for a non-trivial exact intensity value.
    service.field.add_fire((5.5, 5.5), intensity=0.6, radius=2.0)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    service.observe_visible_polygon(polygon, robot_index=0)

    cell = service.field.geometry.world_to_grid(5.5, 5.5)
    ground_truth_value = float(service.field.values(copy=False)[cell.row, cell.col])
    assert ground_truth_value == pytest.approx(0.6, abs=1e-6)

    frame = service.belief.snapshot()
    assert frame.observed[cell.row, cell.col] == True  # noqa: E712
    assert float(frame.values[cell.row, cell.col]) == pytest.approx(ground_truth_value, abs=1e-9)


# ---------------------------------------------------------------------------
# 5. Cells outside the polygon remain intact.
# ---------------------------------------------------------------------------


def test_cells_outside_polygon_remain_intact():
    service = _make_service()
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    service.observe_visible_polygon(polygon, robot_index=0)

    far_cell = service.field.geometry.world_to_grid(0.5, 0.5)
    frame = service.belief.snapshot()
    assert frame.observed[far_cell.row, far_cell.col] == False  # noqa: E712
    assert int(frame.observed.sum()) == 9  # exactly the 3x3 block, nothing more


# ---------------------------------------------------------------------------
# 6. The polygon rasterization respects real occlusion shapes -- an L-shaped
# polygon (already carved by the sensor's occlusion resolution) must not
# observe cells in its notch, even though they sit inside its bounding box.
# ---------------------------------------------------------------------------


def test_polygon_rasterization_respects_occluded_notch():
    service = _make_service()
    polygon = _l_shaped_polygon()

    service.observe_visible_polygon(polygon, robot_index=0)

    frame = service.belief.snapshot()
    geometry = service.field.geometry

    inside_bottom_strip = geometry.world_to_grid(4.5, 1.5)  # inside the L
    inside_left_strip = geometry.world_to_grid(1.5, 4.5)  # inside the L
    inside_notch = geometry.world_to_grid(4.5, 4.5)  # excluded quadrant (occluded)

    assert frame.observed[inside_bottom_strip.row, inside_bottom_strip.col] == True  # noqa: E712
    assert frame.observed[inside_left_strip.row, inside_left_strip.col] == True  # noqa: E712
    assert frame.observed[inside_notch.row, inside_notch.col] == False, (  # noqa: E712
        "a cell inside the polygon's bounding box but outside its actual "
        "(occluded) shape must not be observed -- rasterization must not "
        "degrade to a bounding-box/circle approximation"
    )


# ---------------------------------------------------------------------------
# 7. Single robot: the canvas's robot_index=None convention maps to belief
# robot 0 (never passed through as None).
# ---------------------------------------------------------------------------


def test_single_robot_none_robot_index_maps_to_belief_robot_zero():
    fake = _build_fake_engine(robot_count=1)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    fake.update_explored_free_points_from_polygon(polygon, robot_index=None)

    frame = fake.hazard_service.belief.snapshot()
    assert frame.observed.any()
    assert frame.observed_by_robot[0].any()


# ---------------------------------------------------------------------------
# 8-9. Per-robot attribution is preserved, and the team belief fuses
# observations from multiple robots.
# ---------------------------------------------------------------------------


def test_robot_attribution_is_preserved_through_the_service():
    service = _make_service(robot_count=2)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    service.observe_visible_polygon(polygon, robot_index=0)

    frame = service.belief.snapshot()
    cell = service.field.geometry.world_to_grid(5.5, 5.5)
    assert frame.observed_by_robot[0, cell.row, cell.col] == True  # noqa: E712
    assert frame.observed_by_robot[1, cell.row, cell.col] == False  # noqa: E712


def test_two_robots_fuse_observations_into_the_same_team_belief():
    fake = _build_fake_engine(robot_count=2)
    polygon_a = _square_polygon(0.0, 0.0, 3.0, 3.0)
    polygon_b = _square_polygon(6.0, 6.0, 9.0, 9.0)

    fake.update_explored_free_points_from_polygon(polygon_a, robot_index=0)
    fake.update_explored_free_points_from_polygon(polygon_b, robot_index=1)

    frame = fake.hazard_service.belief.snapshot()
    geometry = fake.hazard_service.field.geometry
    cell_a = geometry.world_to_grid(1.5, 1.5)
    cell_b = geometry.world_to_grid(7.5, 7.5)

    assert frame.observed[cell_a.row, cell_a.col]
    assert frame.observed[cell_b.row, cell_b.col]
    assert frame.observed_by_robot[0, cell_a.row, cell_a.col]
    assert not frame.observed_by_robot[1, cell_a.row, cell_a.col]
    assert frame.observed_by_robot[1, cell_b.row, cell_b.col]
    assert not frame.observed_by_robot[0, cell_b.row, cell_b.col]


# ---------------------------------------------------------------------------
# 10. Repeating the identical observation does not bump the belief revision.
# ---------------------------------------------------------------------------


def test_repeating_the_identical_observation_does_not_increment_revision():
    service = _make_service()
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    service.observe_visible_polygon(polygon, robot_index=0)
    revision_after_first = service.belief.revision

    result = service.observe_visible_polygon(polygon, robot_index=0)

    assert service.belief.revision == revision_after_first
    assert result.changed is False


# ---------------------------------------------------------------------------
# 11-12. Fire removal semantics: the belief keeps the last observation until
# the cell is actually re-observed, at which point it updates to whatever
# ground truth is now (including back to 0.0).
# ---------------------------------------------------------------------------


def test_removing_a_fire_outside_the_fov_preserves_the_last_observation():
    service = _make_service()
    service.field.add_fire((5.5, 5.5), intensity=1.0, radius=2.0)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    service.observe_visible_polygon(polygon, robot_index=0)
    cell = service.field.geometry.world_to_grid(5.5, 5.5)
    observed_value_before = float(service.belief.snapshot().values[cell.row, cell.col])
    assert observed_value_before > 0.0

    # Removed while the robot is no longer looking at it -- no re-observation.
    service.field.remove_fire(service.field.sources()[0].fire_id)

    frame_after_removal = service.belief.snapshot()
    assert float(frame_after_removal.values[cell.row, cell.col]) == observed_value_before
    assert frame_after_removal.observed[cell.row, cell.col] == True  # noqa: E712


def test_reobserving_after_fire_removed_updates_belief_to_zero():
    service = _make_service()
    service.field.add_fire((5.5, 5.5), intensity=1.0, radius=2.0)
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    service.observe_visible_polygon(polygon, robot_index=0)
    service.field.remove_fire(service.field.sources()[0].fire_id)

    result = service.observe_visible_polygon(polygon, robot_index=0)

    cell = service.field.geometry.world_to_grid(5.5, 5.5)
    frame = service.belief.snapshot()
    assert frame.observed[cell.row, cell.col] == True  # noqa: E712
    assert float(frame.values[cell.row, cell.col]) == 0.0
    assert result.changed is True


# ---------------------------------------------------------------------------
# 13-14. Regressions: hazard observation never touches occupancy, and
# creating/removing a fire never touches explored_by_robot.
# ---------------------------------------------------------------------------


def test_hazard_observation_never_modifies_occupancy_grid():
    """RuntimeHazardService.observe_visible_polygon() has no reference to
    BeliefMap at all -- calling it directly must leave an independently
    constructed belief_map completely untouched. (The combined engine call
    update_explored_free_points_from_polygon() legitimately DOES mark newly
    visible cells FREE via BeliefMap.mark_visible_polygon() -- that is its
    normal geometric job, not something hazard observation causes; see
    test_two_robots_fuse_observations_into_the_same_team_belief() etc. for
    that combined path exercised elsewhere in this file.)"""
    belief_map = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1)
    belief_map.grid[3, 3] = OCCUPIED
    belief_map.grid[1, 1] = FREE
    grid_before = belief_map.grid.copy()
    service = _make_service()
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    service.observe_visible_polygon(polygon, robot_index=0)

    assert (belief_map.grid == grid_before).all()


def test_creating_and_removing_fire_never_modifies_explored_by_robot():
    fake = _build_fake_engine(robot_count=1)
    explored_before = fake.belief_map.explored_by_robot.copy()

    source_change = fake.hazard_service.add_fire((5.5, 5.5))
    assert (fake.belief_map.explored_by_robot == explored_before).all()

    fake.hazard_service.remove_fire_near((5.5, 5.5))
    assert (fake.belief_map.explored_by_robot == explored_before).all()
    assert source_change.changed is True


# ---------------------------------------------------------------------------
# 15. Reset recreates an empty HazardBelief with the correct geometry.
# ---------------------------------------------------------------------------


def test_reset_recreates_an_empty_hazard_belief_with_correct_geometry():
    """service.clear() (the real in-place reset -- see reset_belief_map()'s
    docstring on why the engine also just recreates the whole service on a
    full restart) must wipe BOTH layers, not just leave a freshly
    constructed instance looking empty."""
    service = _make_service(robot_count=2)
    service.add_fire((5.5, 5.5))
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    service.observe_visible_polygon(polygon, robot_index=0)

    # 4. Sanity check: both layers actually hold state before clearing.
    assert service.sources() != ()
    assert service.field.values(copy=False).any()
    belief_frame_before = service.belief.snapshot()
    assert belief_frame_before.observed.any()
    assert belief_frame_before.values.any()

    geometry_before = service.field.geometry
    grid_shape_before = service.field.shape
    belief_shape_before = belief_frame_before.observed.shape
    robot_count_before = service.belief.robot_count

    change = service.clear()

    assert change.changed is True
    # No FireSource survives, and ground truth is entirely zero.
    assert service.sources() == ()
    assert not service.field.values(copy=False).any()
    # HazardBelief is entirely wiped too -- values, observed, and
    # observed_by_robot all back to their zero-state defaults.
    frame_after = service.belief.snapshot()
    assert not frame_after.values.any()
    assert not frame_after.observed.any()
    assert not frame_after.observed_by_robot.any()
    # Geometry/shapes/robot_count are preserved -- clear() resets state, it
    # does not rebuild the service with different geometry.
    assert service.field.geometry is geometry_before
    assert service.field.shape == grid_shape_before
    assert frame_after.observed.shape == belief_shape_before
    assert service.belief.robot_count == robot_count_before


def test_clear_reports_change_when_only_hazard_belief_was_nonempty():
    """A field that is already empty must not make clear() report
    "no_change" when the belief still had observations to discard."""
    service = _make_service()
    service.add_fire((5.5, 5.5))
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)
    service.observe_visible_polygon(polygon, robot_index=0)
    # Fire removed (ground truth now empty) without re-observing -- the
    # belief keeps its last observation (see the removal-semantics tests
    # above), so the field is empty but the belief is not.
    service.field.remove_fire(service.field.sources()[0].fire_id)
    assert service.sources() == ()
    assert not service.field.values(copy=False).any()
    assert service.belief.snapshot().observed.any()

    change = service.clear()

    assert change.changed is True
    assert not service.belief.snapshot().observed.any()


# ---------------------------------------------------------------------------
# Contract checks not covered by the numbered list above.
# ---------------------------------------------------------------------------


def test_observe_visible_polygon_returns_hazard_observation_result():
    service = _make_service()
    polygon = _square_polygon(4.0, 4.0, 7.0, 7.0)

    result = service.observe_visible_polygon(polygon, robot_index=0)

    assert isinstance(result, HazardObservationResult)
    assert result.affected_bounds is not None
    row_min, row_max, col_min, col_max = result.affected_bounds
    assert row_min <= row_max
    assert col_min <= col_max


def test_degenerate_polygon_is_a_safe_no_op():
    service = _make_service()

    result = service.observe_visible_polygon([(1.0, 1.0), (2.0, 2.0)], robot_index=0)

    assert result.changed is False
    assert result.affected_bounds is None
    assert not service.belief.snapshot().observed.any()


def test_belief_property_returns_the_same_object_not_a_copy_each_time():
    service = _make_service()

    assert service.belief is service.belief
