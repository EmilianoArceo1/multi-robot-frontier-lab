"""Contract tests for experiments/records.py: the ExperimentRecord JSON
schema, its canonical serialization, and its deterministic fingerprint.

These tests never touch experiments/static_services.py or
experiments/run_experiment.py -- they exercise the record/serialization
layer in isolation, with hand-built ExperimentRecord instances.
"""
from __future__ import annotations

import json
import math

import pytest

from experiments.records import (
    SCHEMA_VERSION,
    AssignmentRecord,
    ExperimentRecord,
    HoldRecord,
    canonical_json,
    compute_fingerprint,
    finalize_record_json,
    record_to_json_dict,
)


def _sample_record(**overrides) -> ExperimentRecord:
    defaults = dict(
        schema_version=SCHEMA_VERSION,
        experiment_id="exp-1",
        scenario_id="scenario-1",
        algorithm="Independent baseline coordinator",
        seed=17,
        robot_count=3,
        raw_frontier_components=5,
        valid_frontier_components=4,
        assignments=(
            AssignmentRecord(
                robot_id=0,
                target=(1.2, 4.2),
                cluster_id="f1",
                decision="ASSIGNED",
                reason="selected: highest information gain wins",
                distance=4.368065933568311,
                information_gain=3.5,
            ),
            AssignmentRecord(
                robot_id=1,
                target=(4.25, 2.0),
                cluster_id="f0",
                decision="ASSIGNED",
                reason="selected: highest information gain wins",
                distance=2.1360009363293826,
                information_gain=2.0,
            ),
        ),
        holds=(HoldRecord(robot_id=2, decision="HOLD", reason="no candidates available"),),
        duplicate_target_count=0,
        assigned_robot_count=2,
        unassigned_robot_count=1,
        total_assignment_distance=6.504066869897694,
        mean_assignment_distance=3.252033434948847,
        blocked_assignment_count=0,
        success=True,
        diagnostics={},
        deterministic_fingerprint="",
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


# ---------------------------------------------------------------------------
# 1. Round-trip: ExperimentRecord -> dict -> JSON -> dict preserves every
#    deterministic field.
# ---------------------------------------------------------------------------


def test_round_trip_preserves_all_deterministic_fields():
    record = _sample_record()
    json_dict = finalize_record_json(record)

    round_tripped = json.loads(json.dumps(json_dict))

    assert round_tripped == json_dict
    assert round_tripped["experiment_id"] == "exp-1"
    assert round_tripped["scenario_id"] == "scenario-1"
    assert round_tripped["algorithm"] == "Independent baseline coordinator"
    assert round_tripped["seed"] == 17
    assert round_tripped["robot_count"] == 3
    assert round_tripped["schema_version"] == SCHEMA_VERSION
    assert len(round_tripped["assignments"]) == 2
    assert len(round_tripped["holds"]) == 1


# ---------------------------------------------------------------------------
# 2. Serialization stable: two logically-equivalent records (assignments/
#    holds built in a different tuple order) produce the exact same
#    canonical JSON.
# ---------------------------------------------------------------------------


def test_canonical_json_is_stable_across_equivalent_construction_order():
    a0 = AssignmentRecord(
        robot_id=0, target=(1.0, 2.0), cluster_id="f0", decision="ASSIGNED",
        reason="r0", distance=1.0, information_gain=1.0,
    )
    a1 = AssignmentRecord(
        robot_id=1, target=(3.0, 4.0), cluster_id="f1", decision="ASSIGNED",
        reason="r1", distance=2.0, information_gain=2.0,
    )
    record_a = _sample_record(assignments=(a0, a1), holds=())
    record_b = _sample_record(assignments=(a1, a0), holds=())

    json_a = canonical_json(record_to_json_dict(record_a))
    json_b = canonical_json(record_to_json_dict(record_b))

    assert json_a == json_b


# ---------------------------------------------------------------------------
# 3. Fingerprint stable: the same deterministic payload produces the same
#    SHA-256 digest.
# ---------------------------------------------------------------------------


def test_fingerprint_is_stable_for_the_same_payload():
    record = _sample_record()

    fingerprint_1 = compute_fingerprint(record_to_json_dict(record))
    fingerprint_2 = compute_fingerprint(record_to_json_dict(record))

    assert fingerprint_1 == fingerprint_2
    assert len(fingerprint_1) == 64  # sha256 hex digest length
    int(fingerprint_1, 16)  # valid hex


# ---------------------------------------------------------------------------
# 4. Fingerprint ignores diagnostics.wall_clock_ms.
# ---------------------------------------------------------------------------


def test_fingerprint_ignores_wall_clock_ms():
    record_fast = _sample_record(diagnostics={"wall_clock_ms": 1.0})
    record_slow = _sample_record(diagnostics={"wall_clock_ms": 987654.0})

    fingerprint_fast = compute_fingerprint(record_to_json_dict(record_fast))
    fingerprint_slow = compute_fingerprint(record_to_json_dict(record_slow))

    assert fingerprint_fast == fingerprint_slow


# ---------------------------------------------------------------------------
# 5. The fingerprint is never computed over a payload that already contains
#    itself.
# ---------------------------------------------------------------------------


def test_fingerprint_does_not_include_itself_in_the_hashed_payload():
    record_empty_placeholder = _sample_record(deterministic_fingerprint="")
    record_stale_placeholder = _sample_record(deterministic_fingerprint="stale-value-from-a-previous-run")

    fingerprint_1 = compute_fingerprint(record_to_json_dict(record_empty_placeholder))
    fingerprint_2 = compute_fingerprint(record_to_json_dict(record_stale_placeholder))

    assert fingerprint_1 == fingerprint_2

    # And the finalized JSON actually carries the recomputed value, not the
    # stale placeholder that was passed in.
    finalized = finalize_record_json(record_stale_placeholder)
    assert finalized["deterministic_fingerprint"] == fingerprint_1
    assert finalized["deterministic_fingerprint"] != "stale-value-from-a-previous-run"


# ---------------------------------------------------------------------------
# 6. Coordinates serialize as JSON lists, not tuples/repr strings.
# ---------------------------------------------------------------------------


def test_coordinates_serialize_as_json_lists():
    record = _sample_record()
    json_dict = record_to_json_dict(record)

    target = json_dict["assignments"][0]["target"]
    assert isinstance(target, list)
    assert target == [1.2, 4.2]

    raw = canonical_json(json_dict)
    assert "(" not in raw  # no python tuple repr leaked into the JSON text


# ---------------------------------------------------------------------------
# 7 & 8. No NaN/Infinity in the final JSON; allow_nan=False rejects
#    non-finite values instead of silently emitting NaN/Infinity tokens.
# ---------------------------------------------------------------------------


def test_finite_record_serializes_without_nan_or_infinity():
    record = _sample_record()
    json_dict = record_to_json_dict(record)

    raw = json.dumps(json_dict, allow_nan=False, sort_keys=True)
    assert "NaN" not in raw
    assert "Infinity" not in raw


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_total_distance_is_rejected(bad_value):
    record = _sample_record(total_assignment_distance=bad_value)

    with pytest.raises(ValueError):
        record_to_json_dict(record)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_assignment_distance_is_rejected(bad_value):
    bad_assignment = AssignmentRecord(
        robot_id=0, target=(0.0, 0.0), cluster_id="f0", decision="ASSIGNED",
        reason="r", distance=bad_value, information_gain=1.0,
    )
    record = _sample_record(assignments=(bad_assignment,), holds=())

    with pytest.raises(ValueError):
        record_to_json_dict(record)


def test_non_finite_target_coordinate_is_rejected():
    bad_assignment = AssignmentRecord(
        robot_id=0, target=(math.nan, 0.0), cluster_id="f0", decision="ASSIGNED",
        reason="r", distance=1.0, information_gain=1.0,
    )
    record = _sample_record(assignments=(bad_assignment,), holds=())

    with pytest.raises(ValueError):
        record_to_json_dict(record)


# ---------------------------------------------------------------------------
# 9 & 10. assignments/holds are sorted by robot_id regardless of
#    construction order.
# ---------------------------------------------------------------------------


def test_assignments_are_sorted_by_robot_id():
    a2 = AssignmentRecord(
        robot_id=2, target=(0.0, 0.0), cluster_id="f2", decision="ASSIGNED",
        reason="r2", distance=1.0, information_gain=1.0,
    )
    a0 = AssignmentRecord(
        robot_id=0, target=(1.0, 1.0), cluster_id="f0", decision="ASSIGNED",
        reason="r0", distance=1.0, information_gain=1.0,
    )
    a1 = AssignmentRecord(
        robot_id=1, target=(2.0, 2.0), cluster_id="f1", decision="ASSIGNED",
        reason="r1", distance=1.0, information_gain=1.0,
    )
    record = _sample_record(assignments=(a2, a0, a1), holds=())

    json_dict = record_to_json_dict(record)

    assert [item["robot_id"] for item in json_dict["assignments"]] == [0, 1, 2]


def test_holds_are_sorted_by_robot_id():
    h2 = HoldRecord(robot_id=2, decision="HOLD", reason="r2")
    h0 = HoldRecord(robot_id=0, decision="HOLD", reason="r0")
    h1 = HoldRecord(robot_id=1, decision="HOLD", reason="r1")
    record = _sample_record(assignments=(), holds=(h2, h0, h1))

    json_dict = record_to_json_dict(record)

    assert [item["robot_id"] for item in json_dict["holds"]] == [0, 1, 2]
