"""Trajectory export contracts (declarative configuration only).

Nothing here writes files.  These types describe how a future exporter
will lay out episodes: tensors, events, metadata, and a *separate*
privileged ground-truth block.

v0 task semantics: fire is traversable and produces no navigation cost or
damage (``fire_traversable=True``, ``fire_damage_model="none"``).  The
fire-crossing metrics exist to measure how much the v0 policy learns to
fly through fire, so a future thermal-damage version can quantify how much
behavior must be unlearned.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

TASK_VERSION_V0 = "fire_search_v0"
FIRE_DAMAGE_MODEL_V0 = "none"


@dataclass(frozen=True)
class EpisodeFireMetrics:
    """Future-facing per-episode fire-interaction metrics.

    - ``fire_crossing_time_s``: total time robots spend inside fire cells.
    - ``fire_overflight_distance``: total distance flown over fire cells.

    In v0 fire is traversable and free, so these only *measure* behavior;
    they let a later thermal-damage model quantify how much fire-crossing
    behavior the policy must unlearn.
    """

    fire_crossing_time_s: float = 0.0
    fire_overflight_distance: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fire_crossing_time_s", "fire_overflight_distance"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


@dataclass(frozen=True)
class EpisodeMetadata:
    """Reproducibility metadata for one exported episode.

    v0 invariants are enforced at construction: task_version is
    "fire_search_v0", fire is traversable, and the fire damage model is
    "none".
    """

    episode_id: str
    seed: int
    map_id: str
    robot_count: int
    fire_count: int
    sensor_range: float
    field_of_view_deg: float
    communication_range: float
    max_steps: int
    simulator_commit: str
    contract_versions: Mapping[str, str]
    contract_bundle_hash: str
    task_version: str = TASK_VERSION_V0
    fire_traversable: bool = True
    fire_damage_model: str = FIRE_DAMAGE_MODEL_V0
    fire_metrics: EpisodeFireMetrics = field(default_factory=EpisodeFireMetrics)

    def __post_init__(self) -> None:
        if self.task_version != TASK_VERSION_V0:
            raise ValueError(
                f"task_version must be {TASK_VERSION_V0!r} in v0, got {self.task_version!r}"
            )
        if self.fire_traversable is not True:
            raise ValueError("fire_traversable must be True in v0 (fire has no navigation cost)")
        if self.fire_damage_model != FIRE_DAMAGE_MODEL_V0:
            raise ValueError(
                f"fire_damage_model must be {FIRE_DAMAGE_MODEL_V0!r} in v0, got "
                f"{self.fire_damage_model!r}"
            )
        if not self.contract_bundle_hash.strip():
            raise ValueError("contract_bundle_hash must not be empty")
        if self.robot_count <= 0:
            raise ValueError(f"robot_count must be positive, got {self.robot_count}")
        if self.fire_count < 0:
            raise ValueError(f"fire_count must be non-negative, got {self.fire_count}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        for name in ("sensor_range", "field_of_view_deg", "communication_range"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        for contract_name, version in self.contract_versions.items():
            if not isinstance(contract_name, str) or not isinstance(version, str):
                raise TypeError("contract_versions must map str contract names to str versions")


@dataclass(frozen=True)
class TrajectoryExportSpec:
    """Declarative export configuration.  No files are written yet."""

    schema_version: str
    tensor_format: str = "npz"
    event_format: str = "parquet"
    metadata_format: str = "json"
    include_critic_state: bool = True
    include_ground_truth_separately: bool = True

    def __post_init__(self) -> None:
        for name in ("tensor_format", "event_format", "metadata_format"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
