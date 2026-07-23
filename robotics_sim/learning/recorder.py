"""In-memory episode recorder for learning transitions.

Memory only: no filesystem, no NPZ/Parquet/JSON, no numpy/pandas.  Actual
export is a future, separate concern configured by TrajectoryExportSpec.

Ground truth is stored *next to* -- never inside -- the transitions, in a
separate per-step block, preserving the privileged-information boundary.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping

from robotics_interfaces.learning.export import EpisodeFireMetrics, EpisodeMetadata
from robotics_interfaces.learning.observations import GroundTruthSnapshot
from robotics_interfaces.learning.transitions import LearningTransition
from robotics_interfaces.learning.versioning import (
    build_contract_manifest,
    compute_contract_bundle_hash,
)


class RecorderError(RuntimeError):
    """Base class for recorder errors."""


class RecorderStateError(RecorderError):
    """Operation not valid in the recorder's current state."""


class EpisodeIdMismatchError(RecorderError):
    """A transition's episode_id does not match the active episode."""


class NonMonotonicDecisionStepError(RecorderError):
    """decision_step is not strictly increasing within the episode."""


class ContractBundleHashMismatchError(RecorderError):
    """metadata.contract_bundle_hash does not match the current contracts."""


def _assert_no_ground_truth_inside(transition: LearningTransition) -> None:
    """Defensive check via the public dataclass API: no field of the
    transition (nor any mapping value) may be a GroundTruthSnapshot."""

    for field in dataclasses.fields(transition):
        value = getattr(transition, field.name)
        if isinstance(value, GroundTruthSnapshot):
            raise RecorderStateError(
                f"transition field {field.name!r} is a GroundTruthSnapshot; ground truth "
                f"must be passed separately to append()"
            )
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, GroundTruthSnapshot):
                    raise RecorderStateError(
                        f"transition field {field.name!r}[{key!r}] is a "
                        f"GroundTruthSnapshot; ground truth must be passed separately"
                    )
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                if isinstance(item, GroundTruthSnapshot):
                    raise RecorderStateError(
                        f"transition field {field.name!r}[{i}] is a GroundTruthSnapshot; "
                        f"ground truth must be passed separately"
                    )


@dataclass(frozen=True)
class EpisodeRecord:
    """Immutable result of one recorded episode.

    ``ground_truth_by_step`` is a separate block keyed by decision_step; it
    is never embedded in the transitions.
    """

    metadata: EpisodeMetadata
    transitions: tuple[LearningTransition, ...]
    ground_truth_by_step: tuple[tuple[int, GroundTruthSnapshot], ...]
    fire_metrics: EpisodeFireMetrics | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(
            self, "ground_truth_by_step", tuple(tuple(item) for item in self.ground_truth_by_step)
        )


class InMemoryTrajectoryRecorder:
    """Records exactly one episode at a time, entirely in memory.

    The recorder never mutates the frozen objects it receives and exposes
    no filesystem paths anywhere in its API.
    """

    def __init__(self) -> None:
        self._metadata: EpisodeMetadata | None = None
        self._transitions: list[LearningTransition] = []
        self._ground_truth_by_step: dict[int, GroundTruthSnapshot] = {}
        self._fire_metrics: EpisodeFireMetrics | None = None

    @property
    def is_recording(self) -> bool:
        return self._metadata is not None

    def start_episode(self, metadata: EpisodeMetadata) -> None:
        if self._metadata is not None:
            raise RecorderStateError(
                f"episode {self._metadata.episode_id!r} is already active; finish it before "
                f"starting a new one"
            )
        if not isinstance(metadata, EpisodeMetadata):
            raise TypeError(f"metadata must be an EpisodeMetadata, got {type(metadata).__name__}")
        expected_hash = compute_contract_bundle_hash(build_contract_manifest())
        if metadata.contract_bundle_hash != expected_hash:
            raise ContractBundleHashMismatchError(
                f"metadata.contract_bundle_hash {metadata.contract_bundle_hash!r} does not "
                f"match the current contract bundle hash {expected_hash!r}; the metadata was "
                f"built against different contract versions"
            )
        self._metadata = metadata

    def append(
        self,
        transition: LearningTransition,
        ground_truth: GroundTruthSnapshot | None = None,
    ) -> None:
        if self._metadata is None:
            raise RecorderStateError("append() called with no active episode")
        if not isinstance(transition, LearningTransition):
            raise TypeError(
                f"transition must be a LearningTransition, got {type(transition).__name__}"
            )
        if transition.episode_id != self._metadata.episode_id:
            raise EpisodeIdMismatchError(
                f"transition.episode_id {transition.episode_id!r} does not match active "
                f"episode {self._metadata.episode_id!r}"
            )
        if self._transitions:
            last_step = self._transitions[-1].decision_step
            if transition.decision_step <= last_step:
                raise NonMonotonicDecisionStepError(
                    f"decision_step must be strictly increasing: got "
                    f"{transition.decision_step} after {last_step}"
                )
        _assert_no_ground_truth_inside(transition)
        if ground_truth is not None:
            if not isinstance(ground_truth, GroundTruthSnapshot):
                raise TypeError(
                    f"ground_truth must be a GroundTruthSnapshot, got "
                    f"{type(ground_truth).__name__}"
                )
            # A duplicate step is impossible here: decision_step is strictly
            # increasing, so each append targets a fresh step.
            self._ground_truth_by_step[transition.decision_step] = ground_truth
        self._transitions.append(transition)

    def set_fire_metrics(self, metrics: EpisodeFireMetrics) -> None:
        if self._metadata is None:
            raise RecorderStateError("set_fire_metrics() called with no active episode")
        if not isinstance(metrics, EpisodeFireMetrics):
            raise TypeError(
                f"metrics must be an EpisodeFireMetrics, got {type(metrics).__name__}"
            )
        self._fire_metrics = metrics

    def abort_episode(self) -> None:
        """Discard the active episode's recorded state without producing an
        EpisodeRecord.  Unlike finish_episode(), nothing is returned; the
        transitions, ground truth and fire metrics collected so far are
        simply dropped."""

        if self._metadata is None:
            raise RecorderStateError("abort_episode() called with no active episode")
        self._metadata = None
        self._transitions = []
        self._ground_truth_by_step = {}
        self._fire_metrics = None

    def finish_episode(self) -> EpisodeRecord:
        if self._metadata is None:
            raise RecorderStateError("finish_episode() called with no active episode")
        record = EpisodeRecord(
            metadata=self._metadata,
            transitions=tuple(self._transitions),
            ground_truth_by_step=tuple(sorted(self._ground_truth_by_step.items())),
            fire_metrics=self._fire_metrics,
        )
        self._metadata = None
        self._transitions = []
        self._ground_truth_by_step = {}
        self._fire_metrics = None
        return record
