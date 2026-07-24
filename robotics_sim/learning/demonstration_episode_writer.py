"""Atomic, per-episode writer: one DemonstrationEpisodeRecord -> exactly one
self-contained folder under ``output_root/pending_review/``.

No dataset-wide file is ever touched: no dataset.npz, no transitions blob,
no runs.jsonl, no global counter, no index required to later read an
episode back. Every episode folder is written in isolation, atomically
(build in a temp directory, validate, then a single os.rename into place),
and never overwritten.

Allowed dependency direction: robotics_sim.learning -> stdlib
(json/shutil/uuid/dataclasses/pathlib) + robotics_interfaces.proposals
(ExplorationCandidate, read-only) + robotics_sim.learning.capture_inputs
(CandidateCaptureInput) + robotics_sim.learning.demonstration_episode. No
Qt, robotics_sim.app, robotics_sim.simulation or engine imports. No
pickle.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from robotics_sim.learning.capture_inputs import CandidateCaptureInput
from robotics_sim.learning.demonstration_episode import (
    DemonstrationDecisionRecord,
    DemonstrationEpisodeLayout,
    DemonstrationEpisodeRecord,
    DemonstrationEpisodeStorageState,
)


class DemonstrationEpisodeWriterError(RuntimeError):
    """Base class for DemonstrationEpisodeWriter errors."""


class DemonstrationEpisodeAlreadyExistsError(DemonstrationEpisodeWriterError):
    """The target episode folder already exists; overwriting is never
    allowed, and no silent alternate name is ever chosen."""


@dataclass(frozen=True)
class DemonstrationIntegrityReport:
    """Result of validating one episode record before/while writing it."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    validator_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise TypeError(f"valid must be a bool, got {type(self.valid).__name__}")
        object.__setattr__(self, "errors", tuple(str(e) for e in self.errors))
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))
        if not isinstance(self.validator_version, str) or not self.validator_version.strip():
            raise ValueError(
                f"validator_version must be a non-empty string, got {self.validator_version!r}"
            )

    def to_json_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "validator_version": self.validator_version,
        }


def _candidate_capture_to_dict(candidate_capture: CandidateCaptureInput) -> dict:
    candidate = candidate_capture.candidate
    return {
        "target": list(candidate.target),
        "source": candidate.source,
        "information_gain": candidate.information_gain,
        "travel_cost": candidate.travel_cost,
        "safety_cost": candidate.safety_cost,
        "overlap_cost": candidate.overlap_cost,
        "heading_cost": candidate.heading_cost,
        "heading_rad": candidate.heading_rad,
        "kind": candidate_capture.kind.value,
        "enabled": candidate_capture.enabled,
        "reachable": candidate_capture.reachable,
        "rejection_reasons": list(candidate_capture.rejection_reasons),
    }


def _decision_to_json_line(decision: DemonstrationDecisionRecord) -> str:
    payload = {
        "episode_id": decision.episode_id,
        "decision_step": decision.decision_step,
        "robot_id": decision.robot_id,
        "candidate_pool": [_candidate_capture_to_dict(c) for c in decision.candidate_pool],
        "selected_candidate_index": decision.selected_candidate_index,
        "selected_candidate_id": decision.selected_candidate_id,
        "target_xy": list(decision.target_xy),
        "candidate_pool_hash": decision.candidate_pool_hash,
        "simulation_time_s": decision.simulation_time_s,
        "human_response_time_s": decision.human_response_time_s,
    }
    return json.dumps(payload, sort_keys=True)


def _metadata_dict(record: DemonstrationEpisodeRecord) -> dict:
    identity = record.identity
    return {
        "episode_id": identity.episode_id,
        "plan_id": identity.plan_id,
        "episode_number": identity.episode_number,
        "attempt_number": identity.attempt_number,
        "collector_id": identity.collector_id,
        "created_at_utc": identity.created_at_utc.isoformat(),
        "map_id": identity.map_id,
        "scenario_id": identity.scenario_id,
        "seed": identity.seed,
        "corpus_id": identity.corpus_id,
        "schema_version": record.schema_version,
        "contract_bundle_hash": identity.contract_bundle_hash,
        "fire_detection_threshold": record.fire_detection_threshold,
        "termination_reason": record.termination_reason,
        "started_at_utc": record.started_at_utc.isoformat(),
        "finished_at_utc": record.finished_at_utc.isoformat(),
        "completed": record.completed,
    }


class DemonstrationEpisodeWriter:
    """Writes exactly one episode folder per call, atomically, under
    ``output_root/pending_review/``."""

    def write_pending_episode(
        self,
        record: DemonstrationEpisodeRecord,
        *,
        output_root: Path,
        integrity_report: DemonstrationIntegrityReport,
    ) -> DemonstrationEpisodeLayout:
        if not isinstance(record, DemonstrationEpisodeRecord):
            raise TypeError(
                f"record must be a DemonstrationEpisodeRecord, got {type(record).__name__}"
            )
        if not isinstance(integrity_report, DemonstrationIntegrityReport):
            raise TypeError(
                f"integrity_report must be a DemonstrationIntegrityReport, got "
                f"{type(integrity_report).__name__}"
            )

        layout = DemonstrationEpisodeLayout(
            output_root=Path(output_root),
            storage_state=DemonstrationEpisodeStorageState.PENDING_REVIEW,
            folder_name=record.identity.folder_name,
        )
        final_dir = layout.episode_directory
        if final_dir.exists():
            raise DemonstrationEpisodeAlreadyExistsError(f"{final_dir} already exists")

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = final_dir.parent / f"{layout.folder_name}.tmp-{uuid.uuid4().hex}"

        try:
            tmp_dir.mkdir(parents=False, exist_ok=False)

            metadata_path = tmp_dir / "metadata.json"
            decisions_path = tmp_dir / "decisions.jsonl"
            metrics_path = tmp_dir / "metrics.json"
            integrity_path = tmp_dir / "integrity_report.json"

            metadata_path.write_text(
                json.dumps(_metadata_dict(record), sort_keys=True, indent=2), encoding="utf-8"
            )
            decisions_path.write_text(
                "".join(_decision_to_json_line(d) + "\n" for d in record.decisions),
                encoding="utf-8",
            )
            metrics_path.write_text(
                json.dumps(dict(record.final_metrics), sort_keys=True, indent=2), encoding="utf-8"
            )
            integrity_path.write_text(
                json.dumps(integrity_report.to_json_dict(), sort_keys=True, indent=2),
                encoding="utf-8",
            )

            for path in (metadata_path, decisions_path, metrics_path, integrity_path):
                if not path.is_file():
                    raise DemonstrationEpisodeWriterError(f"expected file {path} was not written")
            with metadata_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            with metrics_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            with integrity_path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            with decisions_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        json.loads(line)

            tmp_dir.rename(final_dir)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        return layout
