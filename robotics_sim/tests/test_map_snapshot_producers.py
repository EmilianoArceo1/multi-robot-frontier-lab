"""
Tests for the internal snapshot producers added in this phase:

    - BeliefMap.revision / BeliefMap.snapshot() (robotics_sim/environment/
      belief_map.py) -- ExplorationMapSnapshot producer.
    - SimulationControllerMixin.observed_obstacle_snapshot() (robotics_sim/
      simulation/engine.py) -- ObservedObstacleSnapshot producer, backed by
      a new self.mapped_obstacle_revision counter.

Neither producer is wired into build_planning_grid_for_robot() or
PlanningCostmapBuilder yet -- see the characterization test at the bottom,
which pins that the runtime path still does not call them.

Fakes below are lightweight duck-typed SimpleNamespace objects binding the
REAL SimulationControllerMixin methods under test -- the same pattern
already used by test_planning_map_characterization.py and test_discovered_
hazard_planning.py -- not mocks.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.simulation.engine import SimulationControllerMixin

BOUNDS = (0.0, 10.0, 0.0, 10.0)
RESOLUTION = 1.0
ROBOT_RADIUS = 0.3


# ---------------------------------------------------------------------------
# BeliefMap.revision / BeliefMap.snapshot()
# ---------------------------------------------------------------------------


def test_belief_map_revision_starts_at_zero():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    assert belief.revision == 0


def test_belief_map_initial_snapshot_has_revision_zero():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    snapshot = belief.snapshot()
    assert snapshot.revision == 0
    assert snapshot.bounds == belief.bounds
    assert snapshot.resolution == belief.resolution


def test_marking_free_changes_revision():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    before = belief.revision
    belief.mark_free_cell((1, 1))
    assert belief.revision != before
    assert belief.revision > before  # monotonic, not just "different"


def test_marking_occupied_changes_revision():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    before = belief.revision
    belief.mark_occupied_cell((2, 2))
    assert belief.revision > before


def test_previous_snapshot_is_unaffected_by_later_mutation():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    belief.mark_free_cell((1, 1))
    earlier = belief.snapshot()

    belief.mark_occupied_cell((3, 3))

    assert earlier.grid[3, 3] != 1  # still UNKNOWN(-1) in the earlier snapshot
    assert earlier.revision == 1


def test_new_snapshot_reflects_updated_grid_and_revision():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    belief.mark_free_cell((1, 1))
    earlier = belief.snapshot()

    belief.mark_occupied_cell((3, 3))
    later = belief.snapshot()

    assert later.revision > earlier.revision
    assert later.grid[3, 3] == 1  # OCCUPIED
    assert later.grid[1, 1] == 0  # still FREE, unaffected


def test_reset_increments_revision():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    belief.mark_free_cell((1, 1))
    before = belief.revision

    belief.reset()

    assert belief.revision > before


def test_two_belief_maps_with_equal_known_cell_counts_do_not_necessarily_share_revision():
    # Independent BeliefMap instances have independent counters -- revision
    # is a per-instance change counter, not something derived from (and
    # therefore not something that can be inferred from) how many cells are
    # known. Different mutation histories reaching the SAME known-cell
    # count end up at different revisions.
    a = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    a.mark_free_cell((1, 1))  # one mutation -> revision 1, 1 known cell (FREE)

    b = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    b.mark_free_cell((2, 2))  # revision 0 -> 1
    b.mark_occupied_cell((2, 2))  # same cell overwritten FREE -> OCCUPIED: revision 1 -> 2,
    # known-cell COUNT unchanged (still exactly 1 known cell, just OCCUPIED instead of FREE)

    a_known = len(a.occupied_points()) + len(a.explored_points())
    b_known = len(b.occupied_points()) + len(b.explored_points())
    assert a_known == b_known == 1

    assert a.revision != b.revision


def test_revision_is_read_only():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    with pytest.raises(AttributeError):
        belief.revision = 999


def test_idempotent_mark_free_cell_does_not_bump_revision_again():
    # Documents the actual implemented policy (conditional on the `changed`
    # flag inside mark_free_cell()), not an assumption imposed by this test:
    # re-marking an already-FREE cell (same robot_index=None, so no new
    # explored_by_robot attribution either) is a no-op write and must not
    # inflate the counter.
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    belief.mark_free_cell((1, 1))
    after_first = belief.revision

    belief.mark_free_cell((1, 1))

    assert belief.revision == after_first


def test_restore_grid_state_bumps_revision_once():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    belief.mark_free_cell((1, 1))
    before = belief.revision

    belief.restore_grid_state(
        grid=belief.grid.copy(),
        explored_by_robot=belief.explored_by_robot.copy(),
        visit_count=belief.visit_count.copy(),
        last_seen=belief.last_seen.copy(),
    )

    assert belief.revision == before + 1


# ---------------------------------------------------------------------------
# BeliefMap(initial_revision=...): seeding revision across instance
# replacement, so a host that swaps in a new BeliefMap (see reset_belief_
# map() below) never makes the observable revision sequence go backwards.
# ---------------------------------------------------------------------------


def test_belief_map_default_construction_still_starts_at_zero():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1)
    assert belief.revision == 0


def test_belief_map_initial_revision_seeds_starting_value():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1, initial_revision=7)
    assert belief.revision == 7


@pytest.mark.parametrize("bad_value", [1.5, True, False, "3", -1, None])
def test_belief_map_rejects_invalid_initial_revision(bad_value):
    with pytest.raises(ValueError):
        BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1, initial_revision=bad_value)


def test_mutations_continue_from_the_seeded_revision():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1, initial_revision=7)
    belief.mark_free_cell((1, 1))
    assert belief.revision == 8
    belief.mark_occupied_cell((2, 2))
    assert belief.revision == 9


def test_seeded_belief_map_initial_snapshot_carries_the_seeded_revision():
    belief = BeliefMap(bounds=BOUNDS, resolution=RESOLUTION, robot_count=1, initial_revision=42)
    assert belief.snapshot().revision == 42


# ---------------------------------------------------------------------------
# Observed obstacles: mapped_obstacle_revision / observed_obstacle_snapshot()
# ---------------------------------------------------------------------------


def _make_fake_engine() -> SimpleNamespace:
    config = SimpleNamespace(
        grid_resolution=RESOLUTION,
        mapping_point_spacing=0.5,
        default_fire_intensity=1.0,
        default_fire_radius=2.0,
        fire_selection_radius=0.6,
        hazard_block_threshold=0.55,
        obstacles=[],
    )
    robot = SimpleNamespace(x=0.0, y=0.0, theta=0.0, vision=3.0)
    fake = SimpleNamespace(
        robot=robot,
        robots=[],
        config=config,
        canvas=SimpleNamespace(
            append_mapped_obstacle_points=lambda points: None,
            set_status=lambda message: None,
        ),
    )
    # Sensor-geometry collaborators of update_sensed_obstacles() are
    # deliberately simple, controllable stubs -- this file is about the
    # revision counter, not re-testing boundary sampling/visibility, which
    # have their own coverage elsewhere.
    fake.visible_candidate_obstacles = lambda: [(5.0, 5.0, 1.0, 1.0)]
    fake.sample_obstacle_boundary_points = lambda obstacle, spacing: [(5.5, 5.0), (6.0, 5.0)]
    fake.point_visible_from_robot = lambda point, candidate_obstacles=None: True
    fake.quantize_map_point = lambda point, resolution: (round(float(point[0]), 3), round(float(point[1]), 3))
    fake.force_all_robot_poses_free_in_belief = lambda: 0

    for name in (
        "reset_belief_map",
        "ensure_belief_map",
        "ensure_hazard_service",
        "push_discovered_hazard_frame",
        "sync_legacy_map_views_from_belief",
        "update_sensed_obstacles",
        "observed_obstacle_snapshot",
        "_truncate_mapped_obstacle_points",
        "build_planning_grid_for_robot",
        "safety_radius_for_robot",
        "safety_radius",
        "body_radius_for_robot",
        "body_radius",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))

    fake.reset_belief_map()
    return fake


def test_mapped_obstacle_revision_starts_correctly():
    fake = _make_fake_engine()
    assert fake.mapped_obstacle_revision == 0
    assert fake.observed_obstacle_snapshot().revision == 0


def test_adding_new_points_increments_revision():
    fake = _make_fake_engine()
    before = fake.mapped_obstacle_revision

    newly_mapped = fake.update_sensed_obstacles(force_status=False)

    assert newly_mapped  # sanity: the stubbed sensor did find new points
    assert fake.mapped_obstacle_revision > before


def test_repeat_observation_with_no_new_points_does_not_increment_revision():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    after_first = fake.mapped_obstacle_revision

    visibility_calls = 0

    def _count_visibility(point, candidate_obstacles=None):
        nonlocal visibility_calls
        visibility_calls += 1
        return True

    fake.point_visible_from_robot = _count_visibility

    newly_mapped_again = fake.update_sensed_obstacles(force_status=False)

    assert newly_mapped_again == []  # same stubbed points, already known
    assert fake.mapped_obstacle_revision == after_first
    assert visibility_calls == 0  # known samples are rejected before ray casting


def test_snapshot_points_are_an_independent_tuple():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)

    snapshot = fake.observed_obstacle_snapshot()

    assert isinstance(snapshot.points, tuple)
    fake.mapped_obstacle_points.append((9.0, 9.0))
    assert (9.0, 9.0) not in snapshot.points


def test_mutating_the_list_later_does_not_change_an_earlier_snapshot():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    earlier = fake.observed_obstacle_snapshot()

    fake.mapped_obstacle_points.append((7.0, 7.0))
    fake.mapped_obstacle_revision += 1

    assert (7.0, 7.0) not in earlier.points
    later = fake.observed_obstacle_snapshot()
    assert (7.0, 7.0) in later.points
    assert later.revision > earlier.revision


def test_reset_increments_revision_when_content_changes_but_not_when_already_empty():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    assert fake.mapped_obstacle_points  # sanity: there is content to clear

    before = fake.mapped_obstacle_revision
    fake.reset_belief_map()
    assert fake.mapped_obstacle_points == []
    assert fake.mapped_obstacle_revision > before

    # Resetting an already-empty list is not a content change.
    still = fake.mapped_obstacle_revision
    fake.reset_belief_map()
    assert fake.mapped_obstacle_revision == still


# ---------------------------------------------------------------------------
# BeliefMap.revision across reset_belief_map()'s object replacement.
# ---------------------------------------------------------------------------


def test_reset_belief_map_after_mutation_produces_a_strictly_higher_revision():
    fake = _make_fake_engine()
    fake.belief_map.mark_free_cell((1, 1))
    fake.belief_map.mark_occupied_cell((2, 2))
    old_snapshot = fake.belief_map.snapshot()

    fake.reset_belief_map()

    assert fake.belief_map is not old_snapshot  # sanity: a genuinely new object
    assert fake.belief_map.revision > old_snapshot.revision


def test_two_consecutive_resets_do_not_regress_revision():
    fake = _make_fake_engine()
    first = fake.belief_map.revision

    fake.reset_belief_map()
    second = fake.belief_map.revision
    fake.reset_belief_map()
    third = fake.belief_map.revision

    assert first <= second < third


def test_new_belief_maps_initial_snapshot_carries_the_new_seeded_revision():
    fake = _make_fake_engine()
    fake.belief_map.mark_free_cell((1, 1))
    old_revision = fake.belief_map.revision

    fake.reset_belief_map()

    assert fake.belief_map.snapshot().revision == fake.belief_map.revision
    assert fake.belief_map.snapshot().revision > old_revision


# ---------------------------------------------------------------------------
# _truncate_mapped_obstacle_points(): the REAL production helper, not a
# test-local copy of its policy.
# ---------------------------------------------------------------------------


def test_truncate_from_two_to_one_increments_revision():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    assert len(fake.mapped_obstacle_points) == 2  # sanity, matches the stub
    before = fake.mapped_obstacle_revision

    changed = fake._truncate_mapped_obstacle_points(1)

    assert changed is True
    assert len(fake.mapped_obstacle_points) == 1
    assert fake.mapped_obstacle_revision > before


def test_truncate_to_the_same_size_does_not_increment_revision():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    before = fake.mapped_obstacle_revision

    changed = fake._truncate_mapped_obstacle_points(len(fake.mapped_obstacle_points))

    assert changed is False
    assert fake.mapped_obstacle_revision == before


def test_truncate_to_zero_increments_revision_when_there_were_points():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    assert fake.mapped_obstacle_points
    before = fake.mapped_obstacle_revision

    changed = fake._truncate_mapped_obstacle_points(0)

    assert changed is True
    assert fake.mapped_obstacle_points == []
    assert fake.mapped_obstacle_revision > before


def test_truncate_count_larger_than_length_does_not_change_content():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    points_before = list(fake.mapped_obstacle_points)
    before = fake.mapped_obstacle_revision

    changed = fake._truncate_mapped_obstacle_points(len(points_before) + 10)

    assert changed is False
    assert fake.mapped_obstacle_points == points_before
    assert fake.mapped_obstacle_revision == before


@pytest.mark.parametrize("bad_count", [1.5, True, False, "1", None])
def test_truncate_rejects_invalid_count_type(bad_count):
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)

    with pytest.raises(TypeError):
        fake._truncate_mapped_obstacle_points(bad_count)


def test_truncate_negative_count_is_clamped_to_zero_and_still_reports_change():
    # Documents the contract's chosen policy (see _truncate_mapped_obstacle_
    # points()'s own docstring): a negative count is clamped, not rejected,
    # since it is expected to originate from a possibly-stale persisted
    # snapshot field rather than a programmer error.
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)
    before = fake.mapped_obstacle_revision

    changed = fake._truncate_mapped_obstacle_points(-5)

    assert changed is True
    assert fake.mapped_obstacle_points == []
    assert fake.mapped_obstacle_revision > before


def test_two_different_point_lists_with_equal_length_have_different_revisions_after_their_own_mutations():
    # revision must never be derivable from len(points) alone: two
    # independently-mutated engines can reach the same point COUNT while
    # having taken different numbers of mutating steps to get there.
    fake_a = _make_fake_engine()
    fake_a.update_sensed_obstacles(force_status=False)  # -> 2 points, 1 mutation

    fake_b = _make_fake_engine()
    fake_b.sample_obstacle_boundary_points = lambda obstacle, spacing: [(5.5, 5.0)]
    fake_b.update_sensed_obstacles(force_status=False)  # -> 1 point, 1 mutation
    fake_b.sample_obstacle_boundary_points = lambda obstacle, spacing: [(5.5, 5.0), (8.0, 8.0)]
    fake_b.update_sensed_obstacles(force_status=False)  # -> 2 points, 2 mutations total

    assert len(fake_a.mapped_obstacle_points) == len(fake_b.mapped_obstacle_points) == 2
    assert fake_a.mapped_obstacle_revision != fake_b.mapped_obstacle_revision


def test_observed_obstacle_snapshot_excludes_hazard_dynamic_and_ground_truth():
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)

    snapshot = fake.observed_obstacle_snapshot()

    field_names = {f.name for f in dataclasses.fields(snapshot)}
    assert field_names == {"points", "bounds", "resolution", "revision", "source"}
    assert snapshot.source == "mapped_obstacle_points"
    # points contains exactly (and only) the sensor-mapped boundary samples
    # the stub produced -- no hazard points, no dynamic-robot disks, no
    # config.obstacles ground-truth rectangles mixed in.
    assert set(snapshot.points) == {(5.5, 5.0), (6.0, 5.0)}


# ---------------------------------------------------------------------------
# Characterization: the runtime path does not use these producers yet.
# ---------------------------------------------------------------------------


def test_build_planning_grid_for_robot_does_not_use_new_snapshot_producers(monkeypatch):
    fake = _make_fake_engine()
    fake.update_sensed_obstacles(force_status=False)

    def _raise_belief_snapshot(*args, **kwargs):
        raise AssertionError(
            "BeliefMap.snapshot() must not be called by build_planning_grid_for_robot() in this phase"
        )

    def _raise_observed_snapshot(*args, **kwargs):
        raise AssertionError(
            "observed_obstacle_snapshot() must not be called by build_planning_grid_for_robot() in this phase"
        )

    monkeypatch.setattr(BeliefMap, "snapshot", _raise_belief_snapshot)
    monkeypatch.setattr(SimulationControllerMixin, "observed_obstacle_snapshot", _raise_observed_snapshot)

    # The real method must still work normally -- neither monkeypatch should
    # be reachable from it.
    planning_grid = fake.build_planning_grid_for_robot(
        fake.robot, obstacle_points=list(fake.mapped_obstacle_points), robot_radius=ROBOT_RADIUS,
    )

    assert planning_grid is not None
    assert planning_grid.data.shape == (fake.belief_map.height, fake.belief_map.width)
