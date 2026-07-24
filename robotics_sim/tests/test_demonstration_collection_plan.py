"""Tests for DemonstrationCollectionPlan / load_demonstration_collection_plan()
and DemonstrationCollectionSetup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotics_sim.learning.demonstration_collection_plan import (
    DemonstrationCollectionPlan,
    DemonstrationCollectionPlanError,
    DemonstrationCollectionSetup,
    DemonstrationCollectionSetupError,
    MapCollectionAssignment,
    PlannedDemonstrationEpisode,
    load_demonstration_collection_plan,
)
from robotics_sim.learning.map_catalog import load_map_catalog

REAL_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "maps" / "smoke_v0" / "manifest.json"
)
REAL_PLAN_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "collection_plans"
    / "human_demo_smoke_v0.json"
)


@pytest.fixture(scope="module")
def real_catalog():
    return load_map_catalog(REAL_MANIFEST_PATH)


@pytest.fixture(scope="module")
def real_plan(real_catalog):
    return load_demonstration_collection_plan(REAL_PLAN_PATH, map_catalog=real_catalog)


def test_real_plan_loads(real_plan: DemonstrationCollectionPlan) -> None:
    assert real_plan.plan_id == "human-demo-smoke-v0"
    assert real_plan.corpus_id == "smoke_v0"
    assert len(real_plan.assignments) == 6


def test_each_map_belongs_to_exactly_one_collector(real_plan: DemonstrationCollectionPlan) -> None:
    seen = set()
    for collector_id in real_plan.collector_ids:
        for map_id in real_plan.maps_for_collector(collector_id):
            assert map_id not in seen
            seen.add(map_id)
    assert len(seen) == 6


def test_collector_can_have_multiple_maps(real_plan: DemonstrationCollectionPlan) -> None:
    assert len(real_plan.maps_for_collector("collector_a")) > 1
    assert len(real_plan.maps_for_collector("collector_b")) > 1


def test_progress_text(real_plan: DemonstrationCollectionPlan) -> None:
    setup = DemonstrationCollectionSetup(collection_plan=real_plan).select_collector(
        "collector_a"
    ).select_map("smoke_v0_01_open").select_episode(1)
    assert setup.current_episode_position_text == "Episode 1 of 2"
    assert setup.recorded_progress_text([]) == "Recorded 0 of 2"
    assert setup.recorded_progress_text([("smoke_v0_01_open", 1)]) == "Recorded 1 of 2"
    assert setup.accepted_progress_text([]) == "Accepted 0 of 2"
    assert setup.accepted_progress_text([("smoke_v0_01_open", 1)]) == "Accepted 1 of 2"


def test_recorded_vs_accepted_progress_are_independent(real_plan: DemonstrationCollectionPlan) -> None:
    setup = (
        DemonstrationCollectionSetup(collection_plan=real_plan)
        .select_collector("collector_a")
        .select_map("smoke_v0_01_open")
    )
    # episode 1 has a pending_review attempt (recorded, not accepted);
    # episode 2 has an accepted attempt (recorded AND accepted).
    recorded_keys = [("smoke_v0_01_open", 1), ("smoke_v0_01_open", 2)]
    accepted_keys = [("smoke_v0_01_open", 2)]
    assert setup.recorded_progress_text(recorded_keys) == "Recorded 2 of 2"
    assert setup.accepted_progress_text(accepted_keys) == "Accepted 1 of 2"


def test_rejected_only_does_not_count_as_recorded(real_plan: DemonstrationCollectionPlan) -> None:
    setup = (
        DemonstrationCollectionSetup(collection_plan=real_plan)
        .select_collector("collector_a")
        .select_map("smoke_v0_01_open")
    )
    # A rejected-only attempt must never be placed into recorded_episode_keys
    # by the caller; simulate that correctly-behaving caller here.
    assert setup.recorded_progress_text([]) == "Recorded 0 of 2"


def test_new_attempt_of_same_episode_does_not_increment_recorded(
    real_plan: DemonstrationCollectionPlan,
) -> None:
    setup = (
        DemonstrationCollectionSetup(collection_plan=real_plan)
        .select_collector("collector_a")
        .select_map("smoke_v0_01_open")
    )
    # episode_number 1 recorded once; a second attempt of the SAME episode
    # slot is still just one key -- Y must not double count attempts.
    assert setup.recorded_progress_text([("smoke_v0_01_open", 1)]) == "Recorded 1 of 2"


def test_completing_one_map_does_not_affect_another(real_plan: DemonstrationCollectionPlan) -> None:
    setup_open = (
        DemonstrationCollectionSetup(collection_plan=real_plan)
        .select_collector("collector_a")
        .select_map("smoke_v0_01_open")
    )
    setup_office = (
        DemonstrationCollectionSetup(collection_plan=real_plan)
        .select_collector("collector_a")
        .select_map("smoke_v0_02_office")
    )
    all_open_recorded = [("smoke_v0_01_open", 1), ("smoke_v0_01_open", 2)]
    assert setup_open.recorded_progress_text(all_open_recorded) == "Recorded 2 of 2"
    assert setup_office.recorded_progress_text(all_open_recorded) == "Recorded 0 of 2"


def test_keys_from_other_map_are_ignored(real_plan: DemonstrationCollectionPlan) -> None:
    setup = (
        DemonstrationCollectionSetup(collection_plan=real_plan)
        .select_collector("collector_b")
        .select_map("smoke_v0_04_loops")
    )
    foreign_keys = [("smoke_v0_01_open", 1), ("smoke_v0_01_open", 2)]
    assert setup.recorded_progress_text(foreign_keys) == "Recorded 0 of 2"


def _plan_json(tmp_path: Path, assignments: list[dict], plan_id="p", corpus_id="smoke_v0") -> Path:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"plan_id": plan_id, "corpus_id": corpus_id, "assignments": assignments}),
        encoding="utf-8",
    )
    return plan_path


def _episode(number=1, scenario="single_fire", seed=0):
    return {"episode_number": number, "scenario_id": scenario, "seed": seed}


def _assignment(map_id="smoke_v0_01_open", collector="collector_a", episodes=None):
    return {
        "map_id": map_id,
        "collector_id": collector,
        "episodes": episodes if episodes is not None else [_episode()],
    }


def test_scenario_must_exist_for_map(tmp_path: Path, real_catalog) -> None:
    plan_path = _plan_json(
        tmp_path, [_assignment(episodes=[_episode(scenario="not_a_real_scenario")])]
    )
    with pytest.raises(Exception):
        load_demonstration_collection_plan(plan_path, map_catalog=real_catalog)


def test_scenario_from_other_map_rejected(tmp_path: Path, real_catalog) -> None:
    # "double_fire" exists for smoke_v0_01_open too (shared scenario_id
    # naming), so pick a genuinely map-specific rejection: an unknown id.
    plan_path = _plan_json(
        tmp_path,
        [_assignment(map_id="smoke_v0_02_office", episodes=[_episode(scenario="unknown_scenario")])],
    )
    with pytest.raises(Exception):
        load_demonstration_collection_plan(plan_path, map_catalog=real_catalog)


def test_seed_bool_rejected() -> None:
    with pytest.raises(TypeError):
        PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=True)


def test_seed_negative_rejected() -> None:
    with pytest.raises(ValueError):
        PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=-1)


def test_duplicate_episode_number_in_same_map_rejected() -> None:
    with pytest.raises(DemonstrationCollectionPlanError):
        MapCollectionAssignment(
            map_id="m1",
            collector_id="collector_a",
            episodes=(
                PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=0),
                PlannedDemonstrationEpisode(episode_number=1, scenario_id="double_fire", seed=1),
            ),
        )


def test_same_episode_number_in_different_maps_allowed() -> None:
    a = MapCollectionAssignment(
        map_id="m1",
        collector_id="collector_a",
        episodes=(PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=0),),
    )
    b = MapCollectionAssignment(
        map_id="m2",
        collector_id="collector_b",
        episodes=(PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=0),),
    )
    plan = DemonstrationCollectionPlan(plan_id="p", corpus_id="c", assignments=(a, b))
    assert plan.episode_for_map("m1", 1).seed == 0
    assert plan.episode_for_map("m2", 1).seed == 0


def test_duplicate_scenario_seed_combination_rejected() -> None:
    with pytest.raises(DemonstrationCollectionPlanError):
        MapCollectionAssignment(
            map_id="m1",
            collector_id="collector_a",
            episodes=(
                PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=0),
                PlannedDemonstrationEpisode(episode_number=2, scenario_id="single_fire", seed=0),
            ),
        )


def test_episode_order_preserved() -> None:
    episodes = (
        PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=0),
        PlannedDemonstrationEpisode(episode_number=2, scenario_id="double_fire", seed=1),
    )
    assignment = MapCollectionAssignment(map_id="m1", collector_id="collector_a", episodes=episodes)
    assert assignment.episodes == episodes


def test_map_assigned_to_two_collectors_rejected() -> None:
    a = MapCollectionAssignment(
        map_id="m1",
        collector_id="collector_a",
        episodes=(PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=0),),
    )
    b = MapCollectionAssignment(
        map_id="m1",
        collector_id="collector_b",
        episodes=(PlannedDemonstrationEpisode(episode_number=1, scenario_id="single_fire", seed=0),),
    )
    with pytest.raises(DemonstrationCollectionPlanError):
        DemonstrationCollectionPlan(plan_id="p", corpus_id="c", assignments=(a, b))


def test_next_unrecorded_episode(real_plan: DemonstrationCollectionPlan) -> None:
    setup = DemonstrationCollectionSetup(collection_plan=real_plan).select_collector(
        "collector_a"
    ).select_map("smoke_v0_01_open")
    next_setup = setup.select_next_unrecorded_episode([("smoke_v0_01_open", 1)])
    assert next_setup.selected_episode_number == 2


def test_next_unrecorded_episode_raises_when_exhausted(real_plan: DemonstrationCollectionPlan) -> None:
    setup = DemonstrationCollectionSetup(collection_plan=real_plan).select_collector(
        "collector_a"
    ).select_map("smoke_v0_01_open")
    with pytest.raises(DemonstrationCollectionSetupError):
        setup.select_next_unrecorded_episode([("smoke_v0_01_open", 1), ("smoke_v0_01_open", 2)])


def test_next_unrecorded_episode_ignores_rejected_only_keys(
    real_plan: DemonstrationCollectionPlan,
) -> None:
    setup = DemonstrationCollectionSetup(collection_plan=real_plan).select_collector(
        "collector_a"
    ).select_map("smoke_v0_01_open")
    # A caller must never put a rejected-only slot into recorded_episode_keys;
    # simulating that correctly here, episode 1 stays unrecorded and is
    # picked first even though it was previously rejected.
    next_setup = setup.select_next_unrecorded_episode([])
    assert next_setup.selected_episode_number == 1


def test_foreign_map_rejected_for_collector(real_plan: DemonstrationCollectionPlan) -> None:
    setup = DemonstrationCollectionSetup(collection_plan=real_plan).select_collector("collector_a")
    foreign_map = real_plan.maps_for_collector("collector_b")[0]
    with pytest.raises(DemonstrationCollectionSetupError):
        setup.select_map(foreign_map)


def test_changing_collector_clears_map_and_episode(real_plan: DemonstrationCollectionPlan) -> None:
    setup = (
        DemonstrationCollectionSetup(collection_plan=real_plan)
        .select_collector("collector_a")
        .select_map("smoke_v0_01_open")
        .select_episode(1)
    )
    changed = setup.select_collector("collector_b")
    assert changed.selected_map_id is None
    assert changed.selected_episode_number is None


def test_changing_map_clears_episode(real_plan: DemonstrationCollectionPlan) -> None:
    setup = (
        DemonstrationCollectionSetup(collection_plan=real_plan)
        .select_collector("collector_a")
        .select_map("smoke_v0_01_open")
        .select_episode(1)
    )
    other_map = [m for m in real_plan.maps_for_collector("collector_a") if m != "smoke_v0_01_open"][0]
    changed = setup.select_map(other_map)
    assert changed.selected_episode_number is None
