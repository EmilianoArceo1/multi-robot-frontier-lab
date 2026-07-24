"""Tests for DemonstrationEpisodeWriter.write_pending_episode()."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from robotics_interfaces.learning import CandidateKind
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.learning.capture_inputs import CandidateCaptureInput
from robotics_sim.learning.demonstration_episode import (
    DemonstrationDecisionRecord,
    DemonstrationEpisodeIdentity,
    DemonstrationEpisodeRecord,
    DemonstrationEpisodeStorageState,
)
from robotics_sim.learning.demonstration_episode_writer import (
    DemonstrationEpisodeAlreadyExistsError,
    DemonstrationEpisodeWriter,
    DemonstrationIntegrityReport,
)


def make_identity(**overrides) -> DemonstrationEpisodeIdentity:
    kwargs = dict(
        episode_id=overrides.pop("episode_id", "91f3ab20-1111-2222-3333-444455556666"),
        plan_id="human-demo-smoke-v0",
        episode_number=1,
        attempt_number=1,
        collector_id="collector_a",
        corpus_id="smoke_v0",
        map_id="smoke_v0_01_open",
        scenario_id="single_fire",
        seed=0,
        created_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    kwargs.update(overrides)
    return DemonstrationEpisodeIdentity(**kwargs)


def make_candidate_capture(target=(1.0, 1.0)) -> CandidateCaptureInput:
    candidate = ExplorationCandidate(target=target, source="frontier", information_gain=1.0)
    return CandidateCaptureInput(
        candidate=candidate, kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=True, reachable=True
    )


def make_record(identity=None, decisions=None) -> DemonstrationEpisodeRecord:
    identity = identity or make_identity()
    pool = (make_candidate_capture((0.0, 0.0)), make_candidate_capture((1.0, 1.0)))
    if decisions is None:
        decisions = (
            DemonstrationDecisionRecord(
                episode_id=identity.episode_id,
                decision_step=0,
                robot_id=0,
                candidate_pool=pool,
                selected_candidate_index=1,
                selected_candidate_id="robot-0/step-0/candidate-1",
                target_xy=(1.0, 1.0),
                candidate_pool_hash="abc123",
                simulation_time_s=1.0,
                human_response_time_s=0.5,
            ),
        )
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return DemonstrationEpisodeRecord(
        identity=identity,
        started_at_utc=started,
        finished_at_utc=started + timedelta(seconds=10),
        termination_reason="goal_reached",
        completed=bool(decisions),
        decisions=decisions,
        final_metrics={"coverage": 0.75},
        fire_detection_threshold=0.5,
        schema_version=1,
    )


def make_integrity_report(valid=True) -> DemonstrationIntegrityReport:
    return DemonstrationIntegrityReport(
        valid=valid, errors=(), warnings=(), validator_version="v0"
    )


def test_writes_one_folder_with_four_files(tmp_path: Path) -> None:
    record = make_record()
    writer = DemonstrationEpisodeWriter()
    layout = writer.write_pending_episode(
        record, output_root=tmp_path, integrity_report=make_integrity_report()
    )
    assert layout.episode_directory.is_dir()
    files = sorted(p.name for p in layout.episode_directory.iterdir())
    assert files == ["decisions.jsonl", "integrity_report.json", "metadata.json", "metrics.json"]


def test_files_contain_valid_json(tmp_path: Path) -> None:
    record = make_record()
    writer = DemonstrationEpisodeWriter()
    layout = writer.write_pending_episode(
        record, output_root=tmp_path, integrity_report=make_integrity_report()
    )
    metadata = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    assert metadata["episode_id"] == record.identity.episode_id
    metrics = json.loads(layout.metrics_path.read_text(encoding="utf-8"))
    assert metrics["coverage"] == 0.75
    integrity = json.loads(layout.integrity_report_path.read_text(encoding="utf-8"))
    assert integrity["valid"] is True


def test_decisions_jsonl_one_line_per_decision(tmp_path: Path) -> None:
    record = make_record()
    writer = DemonstrationEpisodeWriter()
    layout = writer.write_pending_episode(
        record, output_root=tmp_path, integrity_report=make_integrity_report()
    )
    lines = layout.decisions_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(record.decisions)
    parsed = json.loads(lines[0])
    assert parsed["robot_id"] == 0
    assert len(parsed["candidate_pool"]) == 2


def test_no_global_dataset_or_index_file(tmp_path: Path) -> None:
    writer = DemonstrationEpisodeWriter()
    writer.write_pending_episode(make_record(), output_root=tmp_path, integrity_report=make_integrity_report())
    writer.write_pending_episode(
        make_record(identity=make_identity(episode_id="aaaaaaaa-0000-0000-0000-000000000000")),
        output_root=tmp_path,
        integrity_report=make_integrity_report(),
    )
    top_level = {p.name for p in tmp_path.iterdir()}
    assert top_level == {"pending_review"}
    for forbidden in ("dataset.npz", "transitions.npz", "runs.jsonl", "index.json"):
        assert not (tmp_path / forbidden).exists()
        assert not (tmp_path / "pending_review" / forbidden).exists()


def test_existing_folder_rejected(tmp_path: Path) -> None:
    writer = DemonstrationEpisodeWriter()
    record = make_record()
    writer.write_pending_episode(record, output_root=tmp_path, integrity_report=make_integrity_report())
    with pytest.raises(DemonstrationEpisodeAlreadyExistsError):
        writer.write_pending_episode(record, output_root=tmp_path, integrity_report=make_integrity_report())


def test_failure_cleans_up_temp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import robotics_sim.learning.demonstration_episode_writer as writer_module

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(writer_module, "_metadata_dict", _boom)
    writer = DemonstrationEpisodeWriter()
    record = make_record()
    with pytest.raises(RuntimeError):
        writer.write_pending_episode(record, output_root=tmp_path, integrity_report=make_integrity_report())

    pending_dir = tmp_path / "pending_review"
    leftovers = list(pending_dir.iterdir()) if pending_dir.exists() else []
    assert leftovers == []


def test_no_overwrite_and_no_silent_suffix(tmp_path: Path) -> None:
    writer = DemonstrationEpisodeWriter()
    record = make_record()
    layout = writer.write_pending_episode(record, output_root=tmp_path, integrity_report=make_integrity_report())
    before = set(layout.episode_directory.parent.iterdir())
    with pytest.raises(DemonstrationEpisodeAlreadyExistsError):
        writer.write_pending_episode(record, output_root=tmp_path, integrity_report=make_integrity_report())
    after = set(layout.episode_directory.parent.iterdir())
    assert before == after


def test_two_different_maps_do_not_collide(tmp_path: Path) -> None:
    writer = DemonstrationEpisodeWriter()
    record_a = make_record(identity=make_identity(map_id="smoke_v0_01_open"))
    record_b = make_record(
        identity=make_identity(
            map_id="smoke_v0_02_office", episode_id="bbbbbbbb-0000-0000-0000-000000000000"
        )
    )
    layout_a = writer.write_pending_episode(record_a, output_root=tmp_path, integrity_report=make_integrity_report())
    layout_b = writer.write_pending_episode(record_b, output_root=tmp_path, integrity_report=make_integrity_report())
    assert layout_a.episode_directory != layout_b.episode_directory
    assert layout_a.episode_directory.is_dir()
    assert layout_b.episode_directory.is_dir()


def test_two_attempts_do_not_collide(tmp_path: Path) -> None:
    writer = DemonstrationEpisodeWriter()
    record_1 = make_record(identity=make_identity(attempt_number=1))
    record_2 = make_record(
        identity=make_identity(
            attempt_number=2, episode_id="cccccccc-0000-0000-0000-000000000000"
        )
    )
    layout_1 = writer.write_pending_episode(record_1, output_root=tmp_path, integrity_report=make_integrity_report())
    layout_2 = writer.write_pending_episode(record_2, output_root=tmp_path, integrity_report=make_integrity_report())
    assert layout_1.episode_directory != layout_2.episode_directory


def test_invalid_record_type_not_written(tmp_path: Path) -> None:
    writer = DemonstrationEpisodeWriter()
    with pytest.raises(TypeError):
        writer.write_pending_episode(
            object(), output_root=tmp_path, integrity_report=make_integrity_report()
        )
    assert not any(tmp_path.iterdir())
