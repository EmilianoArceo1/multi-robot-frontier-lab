"""Tests for EpisodeListEntry / EpisodeNavigationModel."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from robotics_sim.learning.demonstration_episode import (
    DemonstrationEpisodeIdentity,
    DemonstrationEpisodeStorageState,
)
from robotics_sim.learning.episode_navigation import (
    EpisodeListEntry,
    EpisodeNavigationError,
    EpisodeNavigationModel,
)

PENDING = DemonstrationEpisodeStorageState.PENDING_REVIEW
ACCEPTED = DemonstrationEpisodeStorageState.ACCEPTED
REJECTED = DemonstrationEpisodeStorageState.REJECTED


def make_identity(
    episode_id, map_id="smoke_v0_01_open", episode_number=1, attempt_number=1, collector_id="collector_a"
):
    return DemonstrationEpisodeIdentity(
        episode_id=episode_id,
        plan_id="human-demo-smoke-v0",
        episode_number=episode_number,
        attempt_number=attempt_number,
        collector_id=collector_id,
        corpus_id="smoke_v0",
        map_id=map_id,
        scenario_id="single_fire",
        seed=0,
        created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def make_entry(
    episode_id,
    map_id="smoke_v0_01_open",
    episode_number=1,
    attempt_number=1,
    collector_id="collector_a",
    status=PENDING,
):
    identity = make_identity(episode_id, map_id, episode_number, attempt_number, collector_id)
    return EpisodeListEntry(
        identity=identity,
        status=status,
        episode_directory=Path("/tmp") / identity.folder_name,
        decision_count=3,
        valid_integrity_report=True,
    )


def _uuid(n: int) -> str:
    return f"{n:08x}-0000-0000-0000-000000000000"


def test_episode_1_of_n() -> None:
    entries = [make_entry(_uuid(1), episode_number=1), make_entry(_uuid(2), episode_number=2)]
    model = EpisodeNavigationModel(entries)
    assert model.current_position_text == "Episode 1 of 2"


def test_map_progress_text() -> None:
    entries = [
        make_entry(_uuid(1), episode_number=1, status=ACCEPTED),
        make_entry(_uuid(2), episode_number=2, status=PENDING),
        make_entry(_uuid(3), episode_number=3, status=ACCEPTED),
    ]
    model = EpisodeNavigationModel(entries)
    assert model.current_map_progress_text == "Map smoke_v0_01_open: completed 2 of 3"


def test_previous_next() -> None:
    entries = [make_entry(_uuid(i), episode_number=i) for i in (1, 2, 3)]
    model = EpisodeNavigationModel(entries)
    assert model.current_entry.identity.episode_number == 1
    model.next()
    assert model.current_entry.identity.episode_number == 2
    model.next()
    assert model.current_entry.identity.episode_number == 3
    model.next()  # clamp at end
    assert model.current_entry.identity.episode_number == 3
    model.previous()
    assert model.current_entry.identity.episode_number == 2
    model.previous()
    model.previous()  # clamp at start
    assert model.current_entry.identity.episode_number == 1


def test_filter_by_collector() -> None:
    entries = [
        make_entry(_uuid(1), collector_id="collector_a"),
        make_entry(_uuid(2), collector_id="collector_b", map_id="smoke_v0_04_loops"),
    ]
    model = EpisodeNavigationModel(entries).filter_by_collector("collector_a")
    assert len(model.entries) == 1
    assert model.entries[0].identity.collector_id == "collector_a"


def test_filter_by_map() -> None:
    entries = [
        make_entry(_uuid(1), map_id="smoke_v0_01_open"),
        make_entry(_uuid(2), map_id="smoke_v0_02_office"),
    ]
    model = EpisodeNavigationModel(entries).filter_by_map("smoke_v0_02_office")
    assert len(model.entries) == 1
    assert model.entries[0].identity.map_id == "smoke_v0_02_office"


def test_filter_by_status() -> None:
    entries = [
        make_entry(_uuid(1), status=ACCEPTED),
        make_entry(_uuid(2), episode_number=2, status=REJECTED),
    ]
    model = EpisodeNavigationModel(entries).filter_by_status(ACCEPTED)
    assert len(model.entries) == 1
    assert model.entries[0].status is ACCEPTED


def test_go_to_episode_id() -> None:
    entries = [make_entry(_uuid(i), episode_number=i) for i in (1, 2, 3)]
    model = EpisodeNavigationModel(entries)
    entry = model.go_to_episode_id(_uuid(3))
    assert entry.identity.episode_number == 3
    assert model.current_entry.identity.episode_number == 3
    with pytest.raises(EpisodeNavigationError):
        model.go_to_episode_id("does-not-exist")


def test_go_to_planned_episode() -> None:
    entries = [
        make_entry(_uuid(1), episode_number=5, attempt_number=1),
        make_entry(_uuid(2), episode_number=5, attempt_number=2),
    ]
    model = EpisodeNavigationModel(entries)
    entry = model.go_to_planned_episode("smoke_v0_01_open", 5, attempt_number=1)
    assert entry.identity.episode_id == _uuid(1)
    latest = model.go_to_planned_episode("smoke_v0_01_open", 5)
    assert latest.identity.episode_id == _uuid(2)
    with pytest.raises(EpisodeNavigationError):
        model.go_to_planned_episode("smoke_v0_01_open", 99)


def test_same_episode_number_different_maps_allowed() -> None:
    entries = [
        make_entry(_uuid(1), map_id="smoke_v0_01_open", episode_number=1),
        make_entry(_uuid(2), map_id="smoke_v0_02_office", episode_number=1),
    ]
    model = EpisodeNavigationModel(entries)
    assert len(model.entries) == 2


def test_duplicate_episode_id_rejected() -> None:
    entries = [make_entry(_uuid(1)), make_entry(_uuid(1), episode_number=2)]
    with pytest.raises(EpisodeNavigationError):
        EpisodeNavigationModel(entries)


def test_duplicate_map_episode_attempt_rejected() -> None:
    entries = [
        make_entry(_uuid(1), episode_number=1, attempt_number=1),
        make_entry(_uuid(2), episode_number=1, attempt_number=1),
    ]
    with pytest.raises(EpisodeNavigationError):
        EpisodeNavigationModel(entries)


def test_rejected_episode_can_have_attempt_two() -> None:
    entries = [
        make_entry(_uuid(1), episode_number=1, attempt_number=1, status=REJECTED),
        make_entry(_uuid(2), episode_number=1, attempt_number=2, status=PENDING),
    ]
    model = EpisodeNavigationModel(entries)
    assert len(model.entries) == 2


def test_order_stable() -> None:
    entries = [
        make_entry(_uuid(3), map_id="smoke_v0_02_office", episode_number=1),
        make_entry(_uuid(1), map_id="smoke_v0_01_open", episode_number=2),
        make_entry(_uuid(2), map_id="smoke_v0_01_open", episode_number=1),
    ]
    model = EpisodeNavigationModel(entries)
    ordered_ids = [e.identity.episode_id for e in model.entries]
    assert ordered_ids == [_uuid(2), _uuid(1), _uuid(3)]


def test_original_list_not_mutated() -> None:
    entries = [make_entry(_uuid(2), episode_number=2), make_entry(_uuid(1), episode_number=1)]
    original_order = list(entries)
    EpisodeNavigationModel(entries)
    assert entries == original_order


def test_multiple_attempts_of_same_episode_do_not_inflate_total() -> None:
    entries = [
        make_entry(_uuid(1), episode_number=1, attempt_number=1, status=REJECTED),
        make_entry(_uuid(2), episode_number=1, attempt_number=2, status=PENDING),
        make_entry(_uuid(3), episode_number=1, attempt_number=3, status=ACCEPTED),
        make_entry(_uuid(4), episode_number=2, attempt_number=1, status=PENDING),
    ]
    model = EpisodeNavigationModel(entries)
    model.go_to_planned_episode("smoke_v0_01_open", 1, attempt_number=3)
    assert model.current_map_progress_text == "Map smoke_v0_01_open: completed 1 of 2"


def test_three_attempts_of_same_episode_coexist_without_error() -> None:
    entries = [
        make_entry(_uuid(1), episode_number=1, attempt_number=1, status=REJECTED),
        make_entry(_uuid(2), episode_number=1, attempt_number=2, status=REJECTED),
        make_entry(_uuid(3), episode_number=1, attempt_number=3, status=PENDING),
    ]
    model = EpisodeNavigationModel(entries)
    assert len(model.entries) == 3
    assert model.current_map_progress_text == "Map smoke_v0_01_open: completed 0 of 1"


def test_empty_list_valid() -> None:
    model = EpisodeNavigationModel([])
    assert model.entries == ()
    assert model.current_entry is None
    assert model.current_position_text == "Episode 0 of 0"
    assert model.previous() is None
    assert model.next() is None
