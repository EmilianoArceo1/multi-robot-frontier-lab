"""Typed, JSON-safe result records for the static allocation benchmark.

This module owns the OUTPUT contract of experiments/run_experiment.py: what
one benchmark run produces, how it serializes to JSON, and how its
deterministic fingerprint is computed. It has no knowledge of
CoordinationRequest/CoordinationPlugin -- experiments/static_services.py and
experiments/run_experiment.py build these records from a CoordinationResult,
not the other way around.

Determinism contract (see run_experiment.py for where this is invoked):
  1. Build an ExperimentRecord with deterministic_fingerprint="" and only
     deterministic diagnostics.
  2. Convert it to a JSON-safe dict via record_to_json_dict().
  3. Compute compute_fingerprint() on that dict -- this EXCLUDES the
     "deterministic_fingerprint" key itself and "diagnostics.wall_clock_ms"
     (if present) before hashing, so neither the field nor a wall-clock
     timing value can ever affect the hash.
  4. Replace deterministic_fingerprint with the computed value and write
     the final dict.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"

# Diagnostics keys that are allowed to vary between otherwise-identical runs
# (wall-clock timing, in particular) and must therefore never participate in
# the deterministic fingerprint. Keep this list in sync with anything
# run_experiment.py ever writes into ExperimentRecord.diagnostics.
_NON_DETERMINISTIC_DIAGNOSTICS_KEYS = ("wall_clock_ms",)


@dataclass(frozen=True)
class AssignmentRecord:
    """One robot that received a target this run (decision == "ASSIGNED").

    distance is always the euclidean distance from the robot's start
    position to `target`, computed independently by the benchmark -- never
    copied from the plugin's own travel_cost, which a different algorithm
    could define differently (see run_experiment.py's metrics computation).
    """

    robot_id: int
    target: tuple[float, float]
    cluster_id: str | None
    decision: str
    reason: str
    distance: float
    information_gain: float


@dataclass(frozen=True)
class HoldRecord:
    """One robot that did NOT receive a target this run (HOLD or FAILED)."""

    robot_id: int
    decision: str
    reason: str


@dataclass(frozen=True)
class AllocationMetrics:
    """Static allocation-quality metrics computed over one CoordinationResult.

    Intermediate/compute-time structure only -- its fields are flattened
    directly onto ExperimentRecord's top level, not nested under a
    "metrics" key, to match the experiment record's flat JSON schema.
    """

    duplicate_target_count: int
    assigned_robot_count: int
    unassigned_robot_count: int
    total_assignment_distance: float
    mean_assignment_distance: float
    blocked_assignment_count: int


@dataclass(frozen=True)
class ExperimentRecord:
    """Immutable, JSON-serializable record of one benchmark run."""

    schema_version: str
    experiment_id: str
    scenario_id: str
    algorithm: str
    seed: int
    robot_count: int
    raw_frontier_components: int
    valid_frontier_components: int
    assignments: tuple[AssignmentRecord, ...]
    holds: tuple[HoldRecord, ...]
    duplicate_target_count: int
    assigned_robot_count: int
    unassigned_robot_count: int
    total_assignment_distance: float
    mean_assignment_distance: float
    blocked_assignment_count: int
    success: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    deterministic_fingerprint: str = ""


def _finite_float(value: Any, *, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return result


def _point_to_list(point: tuple[float, float], *, field_name: str) -> list[float]:
    x, y = point
    return [
        _finite_float(x, field_name=f"{field_name}[0]"),
        _finite_float(y, field_name=f"{field_name}[1]"),
    ]


def _json_safe(value: Any) -> Any:
    """Recursively convert plain Python data into JSON-safe primitives.

    Accepts only what this module's own records ever actually produce:
    None/bool/int/str, finite float, list/tuple (-> list), and Mapping
    (-> dict with string keys, sorted at dump time via sort_keys=True).
    Anything else (Path, bytes, Enum, arbitrary object, dataclass) is
    rejected loudly instead of silently stringified -- diagnostics must
    stay plain data, per this module's docstring.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float is not JSON-safe: {value!r}")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = key if isinstance(key, str) else str(key)
            safe[safe_key] = _json_safe(item)
        return safe
    raise TypeError(
        f"value of type {type(value).__name__!r} is not JSON-safe diagnostics/record data: {value!r}"
    )


def assignment_record_to_dict(record: AssignmentRecord) -> dict[str, Any]:
    return {
        "robot_id": record.robot_id,
        "target": _point_to_list(record.target, field_name="assignment.target"),
        "cluster_id": record.cluster_id,
        "decision": record.decision,
        "reason": record.reason,
        "distance": _finite_float(record.distance, field_name="assignment.distance"),
        "information_gain": _finite_float(record.information_gain, field_name="assignment.information_gain"),
    }


def hold_record_to_dict(record: HoldRecord) -> dict[str, Any]:
    return {
        "robot_id": record.robot_id,
        "decision": record.decision,
        "reason": record.reason,
    }


def record_to_json_dict(record: ExperimentRecord) -> dict[str, Any]:
    """Full JSON-safe dict representation of an ExperimentRecord.

    assignments/holds are sorted by robot_id (see module docstring's
    determinism contract) so two ExperimentRecords describing the same
    outcome always serialize identically regardless of construction order.
    """
    assignments = sorted(record.assignments, key=lambda item: item.robot_id)
    holds = sorted(record.holds, key=lambda item: item.robot_id)

    return {
        "schema_version": record.schema_version,
        "experiment_id": record.experiment_id,
        "scenario_id": record.scenario_id,
        "algorithm": record.algorithm,
        "seed": record.seed,
        "robot_count": record.robot_count,
        "raw_frontier_components": record.raw_frontier_components,
        "valid_frontier_components": record.valid_frontier_components,
        "assignments": [assignment_record_to_dict(item) for item in assignments],
        "holds": [hold_record_to_dict(item) for item in holds],
        "duplicate_target_count": record.duplicate_target_count,
        "assigned_robot_count": record.assigned_robot_count,
        "unassigned_robot_count": record.unassigned_robot_count,
        "total_assignment_distance": _finite_float(
            record.total_assignment_distance, field_name="total_assignment_distance"
        ),
        "mean_assignment_distance": _finite_float(
            record.mean_assignment_distance, field_name="mean_assignment_distance"
        ),
        "blocked_assignment_count": record.blocked_assignment_count,
        "success": record.success,
        "diagnostics": _json_safe(record.diagnostics),
        "deterministic_fingerprint": record.deterministic_fingerprint,
    }


def _fingerprint_payload(json_dict: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of json_dict that participates in the deterministic
    fingerprint: everything except the fingerprint field itself and any
    explicitly non-deterministic diagnostics key."""
    payload = {key: value for key, value in json_dict.items() if key != "deterministic_fingerprint"}

    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        filtered_diagnostics = {
            key: value for key, value in diagnostics.items() if key not in _NON_DETERMINISTIC_DIAGNOSTICS_KEYS
        }
        payload["diagnostics"] = filtered_diagnostics

    return payload


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def compute_fingerprint(json_dict: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of the canonical JSON of the deterministic subset
    of json_dict (see _fingerprint_payload). json_dict is expected to be
    the output of record_to_json_dict() -- already JSON-safe."""
    payload = _fingerprint_payload(json_dict)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def finalize_record_json(record: ExperimentRecord) -> dict[str, Any]:
    """Build the final JSON-safe dict for `record`, with
    deterministic_fingerprint computed and filled in.

    `record.deterministic_fingerprint` is ignored as input -- always
    recomputed from the rest of the record, so a caller cannot accidentally
    write a stale/mismatched fingerprint.
    """
    json_dict = record_to_json_dict(record)
    fingerprint = compute_fingerprint(json_dict)
    json_dict["deterministic_fingerprint"] = fingerprint
    return json_dict
