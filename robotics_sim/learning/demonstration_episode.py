"""Identity, on-disk layout, and the full completed-episode record for one
human demonstration episode.

Every episode is a fully independent artifact: DemonstrationEpisodeIdentity
names it, DemonstrationEpisodeLayout says where its four files live (never
touching the filesystem itself -- that is
robotics_sim.learning.demonstration_episode_writer's job), and
DemonstrationEpisodeRecord/DemonstrationDecisionRecord hold everything that
went into it. Nothing here reads or writes a file, computes a route, or
carries an engine/Qt/GUI/plugin object.

Allowed dependency direction: robotics_sim.learning -> stdlib
(uuid/re/datetime/dataclasses/pathlib) + robotics_interfaces.observations
(Point2D) + robotics_sim.learning.capture_inputs (CandidateCaptureInput,
the existing, real candidate-pool wrapper -- no second candidate type is
defined here). No Qt, robotics_sim.app, robotics_sim.simulation or engine
imports.
"""

from __future__ import annotations

import enum
import math
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

from robotics_interfaces.observations import Point2D
from robotics_sim.learning.capture_inputs import CandidateCaptureInput


def _require_nonempty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def _require_utc(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware, got a naive datetime")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must be UTC (utcoffset=0), got offset {value.utcoffset()!r}")
    return value


_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"{value!r} has no slug-able characters")
    return slug


class DemonstrationEpisodeStorageState(enum.Enum):
    PENDING_REVIEW = "pending_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


_STORAGE_STATE_DIRNAME: Mapping[DemonstrationEpisodeStorageState, str] = {
    DemonstrationEpisodeStorageState.PENDING_REVIEW: "pending_review",
    DemonstrationEpisodeStorageState.ACCEPTED: "accepted",
    DemonstrationEpisodeStorageState.REJECTED: "rejected",
}


@dataclass(frozen=True)
class DemonstrationEpisodeIdentity:
    """The globally unique identity of one human demonstration episode
    (attempt), plus everything needed to derive a stable, Windows-safe,
    human-readable folder name for it.

    ``episode_id`` is the real identity: a full UUID, always supplied
    explicitly (injectable in tests) -- never derived only from
    episode_number, and never generated implicitly here. ``attempt_number``
    lets a rejected episode be retried without ever overwriting the
    rejected attempt.
    """

    episode_id: str
    plan_id: str
    episode_number: int
    attempt_number: int
    collector_id: str
    corpus_id: str
    map_id: str
    scenario_id: str
    seed: int
    created_at_utc: datetime
    contract_bundle_hash: str | None = None

    def __post_init__(self) -> None:
        episode_id = _require_nonempty_str("episode_id", self.episode_id)
        try:
            uuid.UUID(episode_id)
        except ValueError as exc:
            raise ValueError(f"episode_id must be a valid UUID string, got {episode_id!r}") from exc

        _require_nonempty_str("plan_id", self.plan_id)
        _require_nonempty_str("collector_id", self.collector_id)
        _require_nonempty_str("corpus_id", self.corpus_id)
        _require_nonempty_str("map_id", self.map_id)
        _require_nonempty_str("scenario_id", self.scenario_id)
        _require_int("episode_number", self.episode_number, minimum=1)
        _require_int("attempt_number", self.attempt_number, minimum=1)
        _require_int("seed", self.seed, minimum=0)
        _require_utc("created_at_utc", self.created_at_utc)

        if self.contract_bundle_hash is not None:
            _require_nonempty_str("contract_bundle_hash", self.contract_bundle_hash)

    @property
    def episode_id_short(self) -> str:
        return self.episode_id.split("-")[0]

    @property
    def folder_name(self) -> str:
        return (
            f"hdemo__plan-{_slugify(self.plan_id)}__map-{_slugify(self.map_id)}__"
            f"scenario-{_slugify(self.scenario_id)}__seed-{self.seed:04d}__"
            f"collector-{_slugify(self.collector_id)}__ep-{self.episode_number:04d}__"
            f"attempt-{self.attempt_number:02d}__id-{self.episode_id_short}"
        )


@dataclass(frozen=True)
class DemonstrationEpisodeLayout:
    """Where one episode's four files live, for exactly one storage state.

    Pure path computation only -- never touches the filesystem. Writing,
    creating directories, and moving between storage states belongs to
    demonstration_episode_writer.DemonstrationEpisodeWriter.
    """

    output_root: Path
    storage_state: DemonstrationEpisodeStorageState
    folder_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        if not isinstance(self.storage_state, DemonstrationEpisodeStorageState):
            raise TypeError(
                f"storage_state must be a DemonstrationEpisodeStorageState, got "
                f"{type(self.storage_state).__name__}"
            )
        folder_name = _require_nonempty_str("folder_name", self.folder_name)
        for forbidden in ("/", "\\", ":"):
            if forbidden in folder_name:
                raise ValueError(f"folder_name must not contain {forbidden!r}, got {folder_name!r}")
        if " " in folder_name:
            raise ValueError(f"folder_name must not contain spaces, got {folder_name!r}")

    @property
    def episode_directory(self) -> Path:
        return self.output_root / _STORAGE_STATE_DIRNAME[self.storage_state] / self.folder_name

    @property
    def metadata_path(self) -> Path:
        return self.episode_directory / "metadata.json"

    @property
    def decisions_path(self) -> Path:
        return self.episode_directory / "decisions.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.episode_directory / "metrics.json"

    @property
    def integrity_report_path(self) -> Path:
        return self.episode_directory / "integrity_report.json"

    def with_storage_state(
        self, storage_state: DemonstrationEpisodeStorageState
    ) -> "DemonstrationEpisodeLayout":
        return DemonstrationEpisodeLayout(
            output_root=self.output_root, storage_state=storage_state, folder_name=self.folder_name
        )


@dataclass(frozen=True)
class DemonstrationDecisionRecord:
    """One human decision within one episode: the full candidate pool shown
    to the robot's turn, which candidate was chosen, and timing.

    Reuses the real CandidateCaptureInput/ExplorationCandidate contracts --
    no second candidate type is defined here. ``decision_step`` is local to
    the episode (assigned by whoever is sequencing decisions -- e.g.
    ManualDemonstrationSelectionSession) and must be unique within one
    DemonstrationEpisodeRecord.
    """

    episode_id: str
    decision_step: int
    robot_id: int
    candidate_pool: tuple[CandidateCaptureInput, ...]
    selected_candidate_index: int
    selected_candidate_id: str
    target_xy: Point2D
    candidate_pool_hash: str
    simulation_time_s: float
    human_response_time_s: float | None

    def __post_init__(self) -> None:
        _require_nonempty_str("episode_id", self.episode_id)
        _require_int("decision_step", self.decision_step, minimum=0)
        _require_int("robot_id", self.robot_id, minimum=0)

        candidate_pool = tuple(self.candidate_pool)
        for i, candidate_capture in enumerate(candidate_pool):
            if not isinstance(candidate_capture, CandidateCaptureInput):
                raise TypeError(
                    f"candidate_pool[{i}] must be a CandidateCaptureInput, got "
                    f"{type(candidate_capture).__name__}"
                )
        object.__setattr__(self, "candidate_pool", candidate_pool)

        index = self.selected_candidate_index
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"selected_candidate_index must be an int, got {type(index).__name__}")
        if not (0 <= index < len(candidate_pool)):
            raise ValueError(
                f"selected_candidate_index={index} out of range [0, {len(candidate_pool)}) -- "
                f"the selected candidate must belong to the shown pool"
            )

        selected = candidate_pool[index]
        if not selected.enabled:
            raise ValueError(
                f"candidate at selected_candidate_index={index} is not enabled"
            )

        _require_nonempty_str("selected_candidate_id", self.selected_candidate_id)
        _require_nonempty_str("candidate_pool_hash", self.candidate_pool_hash)

        target_xy = tuple(self.target_xy)
        if len(target_xy) != 2:
            raise ValueError(f"target_xy must be an (x, y) pair, got {target_xy!r}")
        if tuple(selected.candidate.target) != target_xy:
            raise ValueError(
                f"target_xy={target_xy!r} does not match the selected candidate's target "
                f"{tuple(selected.candidate.target)!r}"
            )
        object.__setattr__(self, "target_xy", target_xy)

        _require_finite("simulation_time_s", self.simulation_time_s, minimum=0.0)
        if self.human_response_time_s is not None:
            _require_finite("human_response_time_s", self.human_response_time_s, minimum=0.0)


@dataclass(frozen=True)
class DemonstrationEpisodeRecord:
    """The complete record of one finished (or aborted) demonstration
    episode: identity, timing, every decision, final metrics.

    ``decisions`` is preserved as a plain, ordered tuple -- never sorted,
    never deduplicated here. No engine, Qt, GUI, planner or plugin object
    can appear anywhere in this record (nothing in this module even has an
    import path to one).
    """

    identity: DemonstrationEpisodeIdentity
    started_at_utc: datetime
    finished_at_utc: datetime
    termination_reason: str
    completed: bool
    decisions: tuple[DemonstrationDecisionRecord, ...]
    final_metrics: Mapping[str, float]
    fire_detection_threshold: float
    schema_version: int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DemonstrationEpisodeIdentity):
            raise TypeError(
                f"identity must be a DemonstrationEpisodeIdentity, got "
                f"{type(self.identity).__name__}"
            )
        _require_utc("started_at_utc", self.started_at_utc)
        _require_utc("finished_at_utc", self.finished_at_utc)
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError("finished_at_utc must not be before started_at_utc")

        _require_nonempty_str("termination_reason", self.termination_reason)
        if not isinstance(self.completed, bool):
            raise TypeError(f"completed must be a bool, got {type(self.completed).__name__}")

        decisions = tuple(self.decisions)
        seen_steps: dict[int, int] = {}
        for i, decision in enumerate(decisions):
            if not isinstance(decision, DemonstrationDecisionRecord):
                raise TypeError(
                    f"decisions[{i}] must be a DemonstrationDecisionRecord, got "
                    f"{type(decision).__name__}"
                )
            if decision.episode_id != self.identity.episode_id:
                raise ValueError(
                    f"decisions[{i}].episode_id={decision.episode_id!r} does not match "
                    f"identity.episode_id={self.identity.episode_id!r}"
                )
            if decision.decision_step in seen_steps:
                raise ValueError(
                    f"decisions contains duplicate decision_step {decision.decision_step} "
                    f"(indices {seen_steps[decision.decision_step]} and {i})"
                )
            seen_steps[decision.decision_step] = i
        object.__setattr__(self, "decisions", decisions)

        if self.completed and not decisions:
            raise ValueError("a completed episode must have at least one decision")

        final_metrics: dict[str, float] = {}
        for name, value in self.final_metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"final_metrics keys must be non-empty strings, got {name!r}")
            final_metrics[name] = _require_finite(f"final_metrics[{name!r}]", value)
        object.__setattr__(self, "final_metrics", final_metrics)

        _require_finite("fire_detection_threshold", self.fire_detection_threshold, minimum=0.0)
        _require_int("schema_version", self.schema_version, minimum=1)
