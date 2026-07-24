"""Tests for DemonstrationEpisodeIdentity, DemonstrationEpisodeLayout,
DemonstrationDecisionRecord and DemonstrationEpisodeRecord."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from robotics_interfaces.learning import CandidateKind
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.learning.capture_inputs import CandidateCaptureInput
from robotics_sim.learning.demonstration_episode import (
    DemonstrationDecisionRecord,
    DemonstrationEpisodeIdentity,
    DemonstrationEpisodeLayout,
    DemonstrationEpisodeRecord,
    DemonstrationEpisodeStorageState,
)

FIXED_UUID = "91f3ab20-1111-2222-3333-444455556666"


def make_identity(**overrides) -> DemonstrationEpisodeIdentity:
    kwargs = dict(
        episode_id=FIXED_UUID,
        plan_id="human-demo-smoke-v0",
        episode_number=7,
        attempt_number=1,
        collector_id="collector_a",
        corpus_id="smoke_v0",
        map_id="smoke_v0_03_corridors",
        scenario_id="double_fire",
        seed=12,
        created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    kwargs.update(overrides)
    return DemonstrationEpisodeIdentity(**kwargs)


def make_candidate_capture(target=(1.0, 2.0), enabled=True) -> CandidateCaptureInput:
    candidate = ExplorationCandidate(target=target, source="test", information_gain=1.0)
    return CandidateCaptureInput(
        candidate=candidate, kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=enabled, reachable=True
    )


def make_decision(**overrides) -> DemonstrationDecisionRecord:
    pool = (make_candidate_capture(target=(0.0, 0.0)), make_candidate_capture(target=(1.0, 2.0)))
    kwargs = dict(
        episode_id=FIXED_UUID,
        decision_step=0,
        robot_id=0,
        candidate_pool=pool,
        selected_candidate_index=1,
        selected_candidate_id="robot-0/step-0/candidate-1",
        target_xy=(1.0, 2.0),
        candidate_pool_hash="deadbeef",
        simulation_time_s=1.5,
        human_response_time_s=0.4,
    )
    kwargs.update(overrides)
    return DemonstrationDecisionRecord(**kwargs)


# --- identity: exact folder name ---------------------------------------

def test_folder_name_exact() -> None:
    identity = DemonstrationEpisodeIdentity(
        episode_id="91f3ab20-0000-0000-0000-000000000000",
        plan_id="human-demo-smoke-v0",
        episode_number=7,
        attempt_number=1,
        collector_id="collector_a",
        corpus_id="smoke_v0",
        map_id="smoke_v0_03_corridors",
        scenario_id="double_fire",
        seed=12,
        created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert identity.folder_name == (
        "hdemo__plan-human-demo-smoke-v0__map-smoke-v0-03-corridors__"
        "scenario-double-fire__seed-0012__collector-collector-a__"
        "ep-0007__attempt-01__id-91f3ab20"
    )


def test_folder_name_is_single_line_and_windows_safe() -> None:
    identity = make_identity()
    name = identity.folder_name
    assert "\n" not in name
    for forbidden in (" ", ":", "\\", "/", "*", "?", '"', "<", ">", "|"):
        assert forbidden not in name


def test_map_scenario_collector_episode_attempt_present_in_folder_name() -> None:
    identity = make_identity()
    name = identity.folder_name
    assert "map-smoke-v0-03-corridors" in name
    assert "scenario-double-fire" in name
    assert "collector-collector-a" in name
    assert "ep-0007" in name
    assert "attempt-01" in name
    assert "seed-0012" in name


def test_episode_id_short_is_first_8_chars() -> None:
    identity = make_identity()
    assert identity.episode_id_short == "91f3ab20"
    assert identity.folder_name.endswith("id-91f3ab20")


def test_two_maps_episode_one_produce_distinct_names() -> None:
    a = make_identity(map_id="smoke_v0_01_open")
    b = make_identity(map_id="smoke_v0_02_office")
    assert a.folder_name != b.folder_name


def test_two_attempts_produce_distinct_names() -> None:
    a = make_identity(attempt_number=1, episode_id="11111111-0000-0000-0000-000000000000")
    b = make_identity(attempt_number=2, episode_id="22222222-0000-0000-0000-000000000000")
    assert a.folder_name != b.folder_name


def test_distinct_uuids_produce_distinct_names() -> None:
    a = make_identity(episode_id="11111111-0000-0000-0000-000000000000")
    b = make_identity(episode_id="22222222-0000-0000-0000-000000000000")
    assert a.folder_name != b.folder_name


def test_created_at_must_be_utc() -> None:
    with pytest.raises(ValueError):
        make_identity(created_at_utc=datetime(2026, 1, 1))  # naive
    with pytest.raises(ValueError):
        make_identity(
            created_at_utc=datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=-5)))
        )


def test_episode_id_must_be_a_real_uuid() -> None:
    with pytest.raises(ValueError):
        make_identity(episode_id="not-a-uuid")


def test_episode_id_never_derived_only_from_number_is_injectable() -> None:
    injected = str(uuid.uuid4())
    identity = make_identity(episode_id=injected)
    assert identity.episode_id == injected


def test_attempt_number_must_be_positive() -> None:
    with pytest.raises(ValueError):
        make_identity(attempt_number=0)


def test_seed_padding_in_folder_name() -> None:
    identity = make_identity(seed=5)
    assert "seed-0005" in identity.folder_name


# --- layout --------------------------------------------------------------

def test_layout_paths(tmp_path: Path) -> None:
    identity = make_identity()
    layout = DemonstrationEpisodeLayout(
        output_root=tmp_path,
        storage_state=DemonstrationEpisodeStorageState.PENDING_REVIEW,
        folder_name=identity.folder_name,
    )
    assert layout.episode_directory == tmp_path / "pending_review" / identity.folder_name
    assert layout.metadata_path.name == "metadata.json"
    assert layout.decisions_path.name == "decisions.jsonl"
    assert layout.metrics_path.name == "metrics.json"
    assert layout.integrity_report_path.name == "integrity_report.json"


def test_layout_with_storage_state(tmp_path: Path) -> None:
    identity = make_identity()
    layout = DemonstrationEpisodeLayout(
        output_root=tmp_path,
        storage_state=DemonstrationEpisodeStorageState.PENDING_REVIEW,
        folder_name=identity.folder_name,
    )
    accepted = layout.with_storage_state(DemonstrationEpisodeStorageState.ACCEPTED)
    assert accepted.episode_directory == tmp_path / "accepted" / identity.folder_name
    assert layout.episode_directory != accepted.episode_directory


# --- DemonstrationDecisionRecord -----------------------------------------

def test_decision_record_valid() -> None:
    decision = make_decision()
    assert decision.target_xy == (1.0, 2.0)


def test_decision_selected_index_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        make_decision(selected_candidate_index=5)


def test_decision_disabled_candidate_rejected() -> None:
    pool = (make_candidate_capture(target=(0.0, 0.0), enabled=False),)
    with pytest.raises(ValueError):
        make_decision(candidate_pool=pool, selected_candidate_index=0, target_xy=(0.0, 0.0))


def test_decision_target_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        make_decision(target_xy=(99.0, 99.0))


def test_decision_negative_time_rejected() -> None:
    with pytest.raises(ValueError):
        make_decision(simulation_time_s=-1.0)


def test_decision_non_finite_time_rejected() -> None:
    with pytest.raises(ValueError):
        make_decision(simulation_time_s=float("nan"))
    with pytest.raises(ValueError):
        make_decision(human_response_time_s=float("inf"))


# --- DemonstrationEpisodeRecord ------------------------------------------

def make_record(**overrides) -> DemonstrationEpisodeRecord:
    identity = overrides.pop("identity", make_identity())
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    finished = started + timedelta(seconds=30)
    kwargs = dict(
        identity=identity,
        started_at_utc=started,
        finished_at_utc=finished,
        termination_reason="goal_reached",
        completed=True,
        decisions=(make_decision(episode_id=identity.episode_id),),
        final_metrics={"coverage": 0.9},
        fire_detection_threshold=0.5,
        schema_version=1,
    )
    kwargs.update(overrides)
    return DemonstrationEpisodeRecord(**kwargs)


def test_record_valid() -> None:
    record = make_record()
    assert record.completed is True
    assert len(record.decisions) == 1


def test_record_rejects_decision_from_other_episode() -> None:
    with pytest.raises(ValueError):
        make_record(decisions=(make_decision(episode_id=str(uuid.uuid4())),))


def test_record_rejects_duplicate_decision_step() -> None:
    identity = make_identity()
    d0 = make_decision(episode_id=identity.episode_id, decision_step=0)
    d1 = make_decision(episode_id=identity.episode_id, decision_step=0)
    with pytest.raises(ValueError):
        make_record(identity=identity, decisions=(d0, d1))


def test_record_rejects_completed_with_zero_decisions() -> None:
    with pytest.raises(ValueError):
        make_record(completed=True, decisions=())


def test_record_allows_aborted_with_zero_decisions() -> None:
    record = make_record(completed=False, decisions=(), termination_reason="aborted")
    assert record.decisions == ()


def test_record_rejects_nan_metric() -> None:
    with pytest.raises(ValueError):
        make_record(final_metrics={"coverage": float("nan")})


def test_record_finished_before_started_rejected() -> None:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        make_record(started_at_utc=started, finished_at_utc=started - timedelta(seconds=1))


def test_record_decisions_preserved_as_tuple_in_order() -> None:
    identity = make_identity()
    d0 = make_decision(episode_id=identity.episode_id, decision_step=1)
    d1 = make_decision(episode_id=identity.episode_id, decision_step=0)
    record = make_record(identity=identity, decisions=(d0, d1))
    assert record.decisions == (d0, d1)
