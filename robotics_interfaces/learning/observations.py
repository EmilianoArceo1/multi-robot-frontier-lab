"""Actor / critic / ground-truth observation contracts for learning.

Three deliberately independent frozen types:

- :class:`ActorObservation`: only what a deployed policy may see.
- :class:`CriticState`: global training-time information for a centralized
  critic (CTDE), never available to the actor.
- :class:`GroundTruthSnapshot`: privileged simulator truth for training and
  evaluation only.

They do not inherit from each other and never reference each other, which
makes it structurally hard for privileged information to leak into the
actor input.  ActorObservation intentionally has no ``metadata`` field: an
open mapping would be a back door for arbitrary (possibly privileged)
content.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

Point2D = tuple[float, float]
Pose2D = tuple[float, float, float]


def _as_tuple(value: Sequence[Any]) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else tuple(value)


def _check_finite(name: str, values: Sequence[float]) -> None:
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{name}[{i}] must be a real number, got {type(v).__name__}")
        if not math.isfinite(v):
            raise ValueError(f"{name}[{i}] must be finite, got {v!r}")


def _check_feature_matrix(
    name: str, rows: Sequence[Sequence[float]], width: int, width_source: str
) -> tuple[tuple[float, ...], ...]:
    normalized: list[tuple[float, ...]] = []
    for i, row in enumerate(rows):
        row_t = _as_tuple(row)
        if len(row_t) != width:
            raise ValueError(
                f"{name}[{i}] has width {len(row_t)}, expected {width} (from {width_source})"
            )
        _check_finite(f"{name}[{i}]", row_t)
        normalized.append(row_t)
    return tuple(normalized)


@dataclass(frozen=True)
class ActorObservation:
    """Per-robot, per-decision-step observation available to the policy.

    Contains only locally observable, non-privileged data.  All sequences
    are tuples; dimensions are validated in ``__post_init__`` and NaN or
    infinite feature values are rejected.
    """

    schema_version: str
    robot_id: int
    decision_step: int
    time_s: float
    robot_feature_names: tuple[str, ...]
    robot_features: tuple[float, ...]
    candidate_feature_names: tuple[str, ...]
    candidate_features: tuple[tuple[float, ...], ...]
    candidate_ids: tuple[str, ...]
    action_mask: tuple[bool, ...]
    graph_edges: tuple[tuple[int, int], ...]
    visible_teammate_feature_names: tuple[str, ...]
    visible_teammate_features: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "robot_feature_names", _as_tuple(self.robot_feature_names))
        object.__setattr__(self, "robot_features", _as_tuple(self.robot_features))
        object.__setattr__(self, "candidate_feature_names", _as_tuple(self.candidate_feature_names))
        object.__setattr__(self, "candidate_ids", _as_tuple(self.candidate_ids))
        object.__setattr__(self, "action_mask", _as_tuple(self.action_mask))
        object.__setattr__(
            self,
            "visible_teammate_feature_names",
            _as_tuple(self.visible_teammate_feature_names),
        )

        if self.robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {self.robot_id}")
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")
        if not math.isfinite(self.time_s):
            raise ValueError(f"time_s must be finite, got {self.time_s!r}")

        if len(self.robot_features) != len(self.robot_feature_names):
            raise ValueError(
                f"robot_features has {len(self.robot_features)} values but "
                f"robot_feature_names has {len(self.robot_feature_names)} names"
            )
        _check_finite("robot_features", self.robot_features)

        object.__setattr__(
            self,
            "candidate_features",
            _check_feature_matrix(
                "candidate_features",
                self.candidate_features,
                len(self.candidate_feature_names),
                "candidate_feature_names",
            ),
        )

        n_candidates = len(self.candidate_ids)
        if len(self.candidate_features) != n_candidates:
            raise ValueError(
                f"candidate_features has {len(self.candidate_features)} rows but "
                f"candidate_ids has {n_candidates} entries"
            )
        if len(self.action_mask) != n_candidates:
            raise ValueError(
                f"action_mask has {len(self.action_mask)} entries but "
                f"candidate_ids has {n_candidates} entries"
            )
        for i, flag in enumerate(self.action_mask):
            if not isinstance(flag, bool):
                raise TypeError(f"action_mask[{i}] must be bool, got {type(flag).__name__}")

        edges: list[tuple[int, int]] = []
        for i, edge in enumerate(self.graph_edges):
            edge_t = _as_tuple(edge)
            if len(edge_t) != 2:
                raise ValueError(f"graph_edges[{i}] must be a (src, dst) pair, got {edge_t!r}")
            src, dst = edge_t
            for endpoint in (src, dst):
                if isinstance(endpoint, bool) or not isinstance(endpoint, int):
                    raise TypeError(
                        f"graph_edges[{i}] endpoints must be int, got {type(endpoint).__name__}"
                    )
                if not 0 <= endpoint < n_candidates:
                    raise ValueError(
                        f"graph_edges[{i}] references index {endpoint}, valid range is "
                        f"[0, {n_candidates})"
                    )
            edges.append((src, dst))
        object.__setattr__(self, "graph_edges", tuple(edges))

        object.__setattr__(
            self,
            "visible_teammate_features",
            _check_feature_matrix(
                "visible_teammate_features",
                self.visible_teammate_features,
                len(self.visible_teammate_feature_names),
                "visible_teammate_feature_names",
            ),
        )


# Field names that must never appear on ActorObservation.  Kept public so
# tests (and future builders) can assert the boundary explicitly.
FORBIDDEN_ACTOR_FIELDS: tuple[str, ...] = (
    "ground_truth",
    "true_fire",
    "true_occupancy",
    "critic_state",
    "privileged",
    "metadata",
)

_actual_actor_fields = {f.name for f in fields(ActorObservation)}
for _forbidden in FORBIDDEN_ACTOR_FIELDS:
    if _forbidden in _actual_actor_fields:
        raise AssertionError(f"ActorObservation must not define field {_forbidden!r}")


@dataclass(frozen=True)
class CriticState:
    """Global, training-time state for a centralized critic.

    May aggregate team-wide information that individual actors cannot see,
    but must not embed a GroundTruthSnapshot directly -- ground truth stays
    in its own block so leaks require a deliberate, visible step.
    """

    schema_version: str
    decision_step: int
    time_s: float
    global_feature_names: tuple[str, ...]
    global_features: tuple[float, ...]
    per_robot_feature_names: tuple[str, ...]
    per_robot_features: Mapping[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_feature_names", _as_tuple(self.global_feature_names))
        object.__setattr__(self, "global_features", _as_tuple(self.global_features))
        object.__setattr__(
            self, "per_robot_feature_names", _as_tuple(self.per_robot_feature_names)
        )
        if len(self.global_features) != len(self.global_feature_names):
            raise ValueError(
                f"global_features has {len(self.global_features)} values but "
                f"global_feature_names has {len(self.global_feature_names)} names"
            )
        _check_finite("global_features", self.global_features)
        normalized: dict[int, tuple[float, ...]] = {}
        for robot_id, row in self.per_robot_features.items():
            if isinstance(row, GroundTruthSnapshot):
                raise TypeError("CriticState must not contain a GroundTruthSnapshot")
            row_t = _as_tuple(row)
            if len(row_t) != len(self.per_robot_feature_names):
                raise ValueError(
                    f"per_robot_features[{robot_id}] has {len(row_t)} values but "
                    f"per_robot_feature_names has {len(self.per_robot_feature_names)} names"
                )
            _check_finite(f"per_robot_features[{robot_id}]", row_t)
            normalized[robot_id] = row_t
        object.__setattr__(self, "per_robot_features", normalized)


@dataclass(frozen=True)
class GroundTruthSnapshot:
    """Privileged simulator truth, for training/evaluation only.

    Never part of an ActorObservation and never embedded in a CriticState
    or a LearningTransition; it is exported in a separate block.
    """

    schema_version: str
    decision_step: int
    time_s: float
    true_robot_poses: Mapping[int, Pose2D]
    true_occupancy: tuple[tuple[int, ...], ...]
    true_fire_locations: tuple[Point2D, ...]
    global_coverage_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "true_occupancy",
            tuple(_as_tuple(row) for row in self.true_occupancy),
        )
        object.__setattr__(
            self,
            "true_fire_locations",
            tuple(_as_tuple(p) for p in self.true_fire_locations),
        )
        if not 0.0 <= self.global_coverage_fraction <= 1.0:
            raise ValueError(
                f"global_coverage_fraction must be in [0, 1], got {self.global_coverage_fraction}"
            )
