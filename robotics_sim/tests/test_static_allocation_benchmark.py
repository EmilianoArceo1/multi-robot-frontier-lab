"""End-to-end contract tests for the deterministic static allocation
benchmark ("Experiment 0"): experiments/static_services.py's scenario
loader/validation and experiments/run_experiment.py's CLI runner.

Every test here either runs entirely in-process against
run_static_allocation_benchmark()/scenario_from_dict(), or -- for the tests
that must prove the package is genuinely headless/Qt-free (23-25) and the
CLI's own exit-code contract (26-29) -- launches a fresh subprocess, since
by the time this file runs other test modules in the same pytest session
may already have imported PySide6, which would make an in-process
sys.modules check meaningless.
"""
from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.records import record_to_json_dict
from experiments.run_experiment import (
    _build_assignment_and_hold_records,
    _compute_metrics,
    load_scenario,
    run_static_allocation_benchmark,
    write_experiment_record,
)
from experiments.static_services import ScenarioConfigError, StaticScenario, scenario_from_dict
from robotics_interfaces.coordination import CoordinationAssignment, CoordinationResult
from robotics_interfaces.proposals import ExplorationCandidate

REPO_ROOT = Path(__file__).resolve().parents[2]
RAPID_ALLOCATION_CONFIG = REPO_ROOT / "experiments" / "configs" / "rapid_allocation.json"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
EXPERIMENTS_SOURCE_FILES = [
    REPO_ROOT / "experiments" / "records.py",
    REPO_ROOT / "experiments" / "static_services.py",
    REPO_ROOT / "experiments" / "run_experiment.py",
]


# ---------------------------------------------------------------------------
# Scenario fixtures
# ---------------------------------------------------------------------------


def _base_scenario_dict() -> dict:
    """Self-contained copy of the shipped rapid_allocation.json scenario:
    3 robots, 5 frontier components (one invalid=false), one blocked target
    for robot 2, fixed seed. Independent of the committed file's future
    contents on purpose -- see test_shipped_config_matches_minimum_test_fixture
    for the one test that reads the real file."""
    return {
        "experiment_id": "rapid-allocation-v1",
        "scenario_id": "three-robots-five-frontiers",
        "algorithm": "Independent baseline coordinator",
        "seed": 17,
        "robots": [
            {"robot_id": 0, "position": [0.0, 0.0], "heading": 0.0, "radius": 0.2,
             "sensor_range": 2.5, "vision_model": "Camera / FoV"},
            {"robot_id": 1, "position": [5.0, 0.0], "heading": 0.0, "radius": 0.2,
             "sensor_range": 2.5, "vision_model": "Camera / FoV"},
            {"robot_id": 2, "position": [0.0, 5.0], "heading": 0.0, "radius": 0.2,
             "sensor_range": 2.5, "vision_model": "Camera / FoV"},
        ],
        "frontier_components": [
            {"cluster_id": "f0", "cells": [[4.0, 2.0], [4.5, 2.0]], "centroid": [4.25, 2.0],
             "viewpoints": [], "information_gain": 2.0, "valid": True},
            {"cluster_id": "f1", "cells": [[1.0, 4.0], [1.5, 4.0], [1.0, 4.5]], "centroid": [1.17, 4.17],
             "viewpoints": [[1.2, 4.2]], "information_gain": 3.5, "valid": True},
            {"cluster_id": "f2", "cells": [[6.0, 6.0]], "centroid": [6.0, 6.0],
             "viewpoints": [], "information_gain": 1.0, "valid": True},
            {"cluster_id": "f3", "cells": [[-3.0, -3.0], [-3.5, -3.0]], "centroid": [-3.25, -3.0],
             "viewpoints": [], "information_gain": 4.0, "valid": False},
            {"cluster_id": "f4", "cells": [[2.0, -4.0]], "centroid": [2.0, -4.0],
             "viewpoints": [], "information_gain": 0.5, "valid": True},
        ],
        "observed_obstacles": [[3.0, 1.0], [3.0, 1.5]],
        "current_targets": {"1": [5.0, 0.0]},
        "invalidated_targets_by_robot": {"2": [[6.0, 6.0]]},
        "parameters": {"assignment_duplicate_tolerance": 1e-6},
    }


def _reorder_keys_recursive(value):
    """Return a structurally-equal copy with every dict's key insertion
    order reversed, recursively. Lists are left in place (array-order
    tests are separate, see test 19)."""
    if isinstance(value, dict):
        return {key: _reorder_keys_recursive(value[key]) for key in reversed(list(value.keys()))}
    if isinstance(value, list):
        return [_reorder_keys_recursive(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# 11 & 12. Determinism + the fixed 3-robot/5-component/seed-17 scenario.
# ---------------------------------------------------------------------------


def test_shipped_config_has_three_robots_five_components_fixed_seed():
    scenario = load_scenario(str(RAPID_ALLOCATION_CONFIG))

    assert len(scenario.robots) == 3
    assert len(scenario.frontier_components) == 5
    assert scenario.seed == 17


def test_same_scenario_run_twice_produces_identical_record():
    scenario = scenario_from_dict(_base_scenario_dict())

    record_1 = run_static_allocation_benchmark(scenario)
    record_2 = run_static_allocation_benchmark(scenario)

    json_1 = record_to_json_dict(record_1)
    json_2 = record_to_json_dict(record_2)

    assert json_1["assignments"] == json_2["assignments"]
    assert json_1["holds"] == json_2["holds"]
    assert json_1["duplicate_target_count"] == json_2["duplicate_target_count"]
    assert json_1["total_assignment_distance"] == json_2["total_assignment_distance"]
    assert json_1["mean_assignment_distance"] == json_2["mean_assignment_distance"]
    assert json_1["deterministic_fingerprint"] == json_2["deterministic_fingerprint"]


# ---------------------------------------------------------------------------
# 13. No duplicate targets when the baseline/contract already avoid them.
# ---------------------------------------------------------------------------


def test_no_duplicate_targets_in_the_shipped_scenario():
    scenario = scenario_from_dict(_base_scenario_dict())
    record = run_static_allocation_benchmark(scenario)

    assert record.duplicate_target_count == 0
    targets = [item.target for item in record.assignments]
    assert len(targets) == len(set(targets))


# ---------------------------------------------------------------------------
# 14. More robots than components -> partial assignments, HOLD for the
#    rest, no crash.
# ---------------------------------------------------------------------------


def test_more_robots_than_components_produces_partial_assignment_and_holds():
    data = _base_scenario_dict()
    data["robots"] = [
        {"robot_id": rid, "position": [float(rid), 0.0], "heading": 0.0, "radius": 0.2,
         "sensor_range": 2.5, "vision_model": "Camera / FoV"}
        for rid in range(6)
    ]
    data["current_targets"] = {}
    data["invalidated_targets_by_robot"] = {}

    scenario = scenario_from_dict(data)
    record = run_static_allocation_benchmark(scenario)

    assert record.robot_count == 6
    assert record.assigned_robot_count == 4  # 4 valid components (f3 is invalid)
    assert record.unassigned_robot_count == 2
    assert record.assigned_robot_count + record.unassigned_robot_count == record.robot_count
    assert record.success is True


# ---------------------------------------------------------------------------
# 15. Zero components -> all HOLD, success, zero distance.
# ---------------------------------------------------------------------------


def test_zero_components_produces_all_holds_and_zero_distance():
    data = _base_scenario_dict()
    data["frontier_components"] = []
    data["invalidated_targets_by_robot"] = {}

    scenario = scenario_from_dict(data)
    record = run_static_allocation_benchmark(scenario)

    assert record.assigned_robot_count == 0
    assert record.unassigned_robot_count == record.robot_count
    assert len(record.assignments) == 0
    assert len(record.holds) == record.robot_count
    assert record.total_assignment_distance == 0.0
    assert record.mean_assignment_distance == 0.0
    assert record.success is True


# ---------------------------------------------------------------------------
# 16. valid=false components never become assignable candidates.
# ---------------------------------------------------------------------------


def test_invalid_component_never_produces_an_assignable_candidate():
    data = _base_scenario_dict()
    data["robots"] = [
        {"robot_id": 0, "position": [-3.25, -3.0], "heading": 0.0, "radius": 0.2,
         "sensor_range": 2.5, "vision_model": "Camera / FoV"},
    ]
    data["frontier_components"] = [
        {"cluster_id": "f3", "cells": [[-3.0, -3.0], [-3.5, -3.0]], "centroid": [-3.25, -3.0],
         "viewpoints": [], "information_gain": 4.0, "valid": False},
    ]
    data["current_targets"] = {}
    data["invalidated_targets_by_robot"] = {}

    scenario = scenario_from_dict(data)
    record = run_static_allocation_benchmark(scenario)

    assert record.assigned_robot_count == 0
    assert record.unassigned_robot_count == 1
    assert record.valid_frontier_components == 0
    assert record.raw_frontier_components == 1


# ---------------------------------------------------------------------------
# 17. A target present in invalidated_targets_by_robot is never assigned to
#    that robot.
# ---------------------------------------------------------------------------


def test_invalidated_target_is_not_assigned_to_that_robot():
    data = _base_scenario_dict()
    data["robots"] = [
        {"robot_id": 0, "position": [6.0, 6.0], "heading": 0.0, "radius": 0.2,
         "sensor_range": 2.5, "vision_model": "Camera / FoV"},
    ]
    data["frontier_components"] = [
        {"cluster_id": "f2", "cells": [[6.0, 6.0]], "centroid": [6.0, 6.0],
         "viewpoints": [], "information_gain": 1.0, "valid": True},
    ]
    data["current_targets"] = {}
    data["invalidated_targets_by_robot"] = {"0": [[6.0, 6.0]]}

    scenario = scenario_from_dict(data)
    record = run_static_allocation_benchmark(scenario)

    # The only candidate this robot could have received is blocked for it,
    # so it must HOLD rather than receive that target.
    assert record.assigned_robot_count == 0
    assert record.unassigned_robot_count == 1
    assert record.holds[0].robot_id == 0


# ---------------------------------------------------------------------------
# 18. JSON object key order does not change the canonical result.
# ---------------------------------------------------------------------------


def test_object_key_order_does_not_change_the_result():
    data = _base_scenario_dict()
    reordered = _reorder_keys_recursive(data)
    assert list(reordered.keys()) != list(data.keys())  # sanity: reordering actually happened

    scenario_a = scenario_from_dict(data)
    scenario_b = scenario_from_dict(reordered)

    record_a = run_static_allocation_benchmark(scenario_a)
    record_b = run_static_allocation_benchmark(scenario_b)

    json_a = record_to_json_dict(record_a)
    json_b = record_to_json_dict(record_b)
    assert json_a["assignments"] == json_b["assignments"]
    assert json_a["holds"] == json_b["holds"]


# ---------------------------------------------------------------------------
# 19. robots[]/frontier_components[] array order does not change the
#    result, because the loader normalizes by id.
# ---------------------------------------------------------------------------


def test_robots_and_components_array_order_does_not_change_the_result():
    data = _base_scenario_dict()
    reordered = copy.deepcopy(data)
    reordered["robots"] = list(reversed(reordered["robots"]))
    reordered["frontier_components"] = list(reversed(reordered["frontier_components"]))

    scenario_a = scenario_from_dict(data)
    scenario_b = scenario_from_dict(reordered)

    assert scenario_a.robots == scenario_b.robots
    assert scenario_a.frontier_components == scenario_b.frontier_components

    record_a = run_static_allocation_benchmark(scenario_a)
    record_b = run_static_allocation_benchmark(scenario_b)
    assert record_to_json_dict(record_a) == record_to_json_dict(record_b)


# ---------------------------------------------------------------------------
# 20. cells order within a component is preserved (never sorted/reordered).
# ---------------------------------------------------------------------------


def test_cell_order_within_a_component_is_preserved():
    data = _base_scenario_dict()
    data["frontier_components"][1]["cells"] = [[9.0, 9.0], [1.0, 1.0], [5.0, 5.0]]

    scenario = scenario_from_dict(data)
    f1 = next(c for c in scenario.frontier_components if c.cluster_id == "f1")

    assert f1.cells == ((9.0, 9.0), (1.0, 1.0), (5.0, 5.0))


# ---------------------------------------------------------------------------
# 21. Duplicate IDs produce a clear configuration error.
# ---------------------------------------------------------------------------


def test_duplicate_robot_id_raises_scenario_config_error():
    data = _base_scenario_dict()
    data["robots"][1]["robot_id"] = data["robots"][0]["robot_id"]

    with pytest.raises(ScenarioConfigError, match="duplicate robot_id"):
        scenario_from_dict(data)


def test_duplicate_cluster_id_raises_scenario_config_error():
    data = _base_scenario_dict()
    data["frontier_components"][1]["cluster_id"] = data["frontier_components"][0]["cluster_id"]

    with pytest.raises(ScenarioConfigError, match="duplicate cluster_id"):
        scenario_from_dict(data)


# ---------------------------------------------------------------------------
# 22. Invalid coordinates produce a clear configuration error.
# ---------------------------------------------------------------------------


def test_wrong_length_position_raises_scenario_config_error():
    data = _base_scenario_dict()
    data["robots"][0]["position"] = [0.0, 0.0, 0.0]

    with pytest.raises(ScenarioConfigError):
        scenario_from_dict(data)


def test_nan_coordinate_raises_scenario_config_error():
    data = _base_scenario_dict()
    data["robots"][0]["position"] = [math.nan, 0.0]

    with pytest.raises(ScenarioConfigError, match="finite"):
        scenario_from_dict(data)


def test_infinite_coordinate_raises_scenario_config_error():
    data = _base_scenario_dict()
    data["frontier_components"][0]["centroid"] = [math.inf, 0.0]

    with pytest.raises(ScenarioConfigError, match="finite"):
        scenario_from_dict(data)


def test_literal_json_infinity_token_is_rejected_end_to_end(tmp_path):
    raw = json.dumps(_base_scenario_dict())
    raw = raw.replace('"position": [0.0, 0.0]', '"position": [Infinity, 0.0]', 1)
    config_path = tmp_path / "bad_infinity.json"
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ScenarioConfigError, match="finite"):
        load_scenario(str(config_path))


def test_current_targets_referencing_unknown_robot_raises_scenario_config_error():
    data = _base_scenario_dict()
    data["current_targets"] = {"999": [0.0, 0.0]}

    with pytest.raises(ScenarioConfigError, match="unknown robot_id"):
        scenario_from_dict(data)


def test_invalidated_targets_referencing_unknown_robot_raises_scenario_config_error():
    data = _base_scenario_dict()
    data["invalidated_targets_by_robot"] = {"999": [[0.0, 0.0]]}

    with pytest.raises(ScenarioConfigError, match="unknown robot_id"):
        scenario_from_dict(data)


def test_missing_required_field_raises_scenario_config_error():
    data = _base_scenario_dict()
    del data["seed"]

    with pytest.raises(ScenarioConfigError, match="seed"):
        scenario_from_dict(data)


# ---------------------------------------------------------------------------
# 23-25. The package is genuinely headless: no Qt, no engine, no A*, no
#    wall-clock-as-simulation-time. Verified two ways: a static source
#    check (defense in depth, always correct regardless of test order) and
#    a fresh-subprocess import check (the only reliable way to check
#    sys.modules, since other test files in this same pytest session may
#    already have imported PySide6 before this file runs).
# ---------------------------------------------------------------------------


# Matched against actual CODE lines only (see _non_docstring_code_lines()),
# so module docstrings are free to explain what this package deliberately
# does NOT import/call (see e.g. run_experiment.py's own docstring) without
# tripping this check. Patterns are import/usage shapes, not bare words,
# so a comment mentioning "engine.py" in prose still cannot false-positive.
_FORBIDDEN_CODE_PATTERNS = (
    "import PySide6",
    "from PySide6",
    "QApplication(",
    "QTimer(",
    "MainWindow(",
    "SimulationCanvas(",
    ".simulation_step(",
    "AStarPlanner(",
    "perf_counter(",
    "from robotics_sim.simulation.engine",
    "from robotics_sim.app.main_window",
    "from robotics_sim.app.simulation_canvas",
)


def _non_docstring_code_lines(text: str) -> list[str]:
    """Best-effort strip of module/function docstrings (``\"\"\"...\"\"\"``
    blocks) so this check scans only real code lines, not prose that
    legitimately names the things this package avoids importing."""
    without_docstrings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    return without_docstrings.splitlines()


def test_experiments_source_never_mentions_forbidden_runtime_symbols():
    for path in EXPERIMENTS_SOURCE_FILES:
        code_lines = _non_docstring_code_lines(path.read_text(encoding="utf-8"))
        code_text = "\n".join(code_lines)
        for forbidden in _FORBIDDEN_CODE_PATTERNS:
            assert forbidden not in code_text, f"{path.name} must not contain {forbidden!r}"


def test_importing_experiments_run_experiment_in_a_fresh_process_imports_no_qt_or_engine():
    probe = (
        "import sys\n"
        "import experiments.run_experiment\n"
        "forbidden = ['PySide6', 'PySide6.QtWidgets', 'robotics_sim.simulation.engine',\n"
        "             'robotics_sim.app.main_window', 'robotics_sim.app.simulation_canvas']\n"
        "leaked = [name for name in forbidden if name in sys.modules]\n"
        "assert not leaked, f'unexpected modules imported: {leaked}'\n"
        "print('CLEAN')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "CLEAN" in result.stdout


# ---------------------------------------------------------------------------
# 26-29. CLI exit-code contract.
# ---------------------------------------------------------------------------


def _run_cli(*, config: str, output: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "experiments.run_experiment", "--config", config, "--output", output],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_with_valid_config_creates_parseable_json_and_exits_zero(tmp_path):
    output_path = RESULTS_DIR / "_pytest_cli_valid_result.json"
    if output_path.exists():
        output_path.unlink()

    try:
        result = _run_cli(config=str(RAPID_ALLOCATION_CONFIG), output=str(output_path))

        assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
        assert output_path.exists()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "1.0"
        assert "deterministic_fingerprint" in payload
    finally:
        if output_path.exists():
            output_path.unlink()


def test_cli_with_invalid_config_exits_nonzero_and_writes_no_output(tmp_path):
    bad_config = tmp_path / "bad_config.json"
    bad_config.write_text(json.dumps({"experiment_id": "only-one-field"}), encoding="utf-8")

    output_path = RESULTS_DIR / "_pytest_cli_invalid_result.json"
    if output_path.exists():
        output_path.unlink()

    try:
        result = _run_cli(config=str(bad_config), output=str(output_path))

        assert result.returncode != 0
        assert result.stderr.strip() != ""
        assert not output_path.exists()
    finally:
        if output_path.exists():
            output_path.unlink()


def test_cli_output_must_resolve_inside_experiments_results():
    outside_output = REPO_ROOT / "experiments" / "_pytest_outside_results.json"
    if outside_output.exists():
        outside_output.unlink()

    try:
        result = _run_cli(config=str(RAPID_ALLOCATION_CONFIG), output=str(outside_output))

        assert result.returncode != 0
        assert not outside_output.exists()
    finally:
        if outside_output.exists():
            outside_output.unlink()


# ---------------------------------------------------------------------------
# 30-32. Final JSON schema surface.
# ---------------------------------------------------------------------------


def _assert_only_json_safe_types(value) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for item in value:
            _assert_only_json_safe_types(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_only_json_safe_types(item)
        return
    raise AssertionError(f"non-JSON-safe value leaked into the record: {value!r} ({type(value).__name__})")


def test_final_json_contains_schema_version_and_fingerprint_and_only_json_safe_types():
    scenario = scenario_from_dict(_base_scenario_dict())
    record = run_static_allocation_benchmark(scenario)
    output_path = RESULTS_DIR / "_pytest_schema_check_result.json"
    if output_path.exists():
        output_path.unlink()

    try:
        json_dict = write_experiment_record(record, str(output_path))

        assert json_dict["schema_version"] == "1.0"
        assert isinstance(json_dict["deterministic_fingerprint"], str)
        assert len(json_dict["deterministic_fingerprint"]) == 64

        _assert_only_json_safe_types(json_dict)

        # And the file on disk round-trips to the exact same structure.
        on_disk = json.loads(output_path.read_text(encoding="utf-8"))
        assert on_disk == json_dict
    finally:
        if output_path.exists():
            output_path.unlink()


# ---------------------------------------------------------------------------
# 33-34. run_static_allocation_benchmark() itself returns a real, non-empty
#    fingerprint that reacts to the seed (see experiments/records.py's
#    finalize_experiment_record()).
# ---------------------------------------------------------------------------


def test_run_static_allocation_benchmark_returns_a_real_fingerprint():
    scenario = scenario_from_dict(_base_scenario_dict())

    record = run_static_allocation_benchmark(scenario)

    assert len(record.deterministic_fingerprint) == 64
    int(record.deterministic_fingerprint, 16)  # valid hex


def test_run_static_allocation_benchmark_fingerprint_changes_with_seed():
    data_a = _base_scenario_dict()
    data_b = copy.deepcopy(data_a)
    data_b["seed"] = data_a["seed"] + 1

    record_a = run_static_allocation_benchmark(scenario_from_dict(data_a))
    record_b = run_static_allocation_benchmark(scenario_from_dict(data_b))

    assert record_a.deterministic_fingerprint != record_b.deterministic_fingerprint


# ---------------------------------------------------------------------------
# 35-42. Provenance validation of ASSIGNED results in
#    _build_assignment_and_hold_records(), exercised with a small, explicit
#    fake CoordinationResult built from the real robotics_interfaces
#    dataclasses -- no new plugin needed for these (see this file's module
#    docstring).
# ---------------------------------------------------------------------------


def _provenance_test_scenario() -> StaticScenario:
    return scenario_from_dict(_base_scenario_dict())


def _record_for(records, robot_id):
    return next(item for item in records if item.robot_id == robot_id)


def test_assigned_with_no_proposal_is_rejected():
    scenario = _provenance_test_scenario()
    result = CoordinationResult(
        assignments=(
            CoordinationAssignment(robot_id=0, status="ASSIGNED", target=(1.2, 4.2), reason="r", proposal=None),
            CoordinationAssignment(robot_id=1, status="HOLD", target=None, reason="r"),
            CoordinationAssignment(robot_id=2, status="HOLD", target=None, reason="r"),
        )
    )

    assignments, holds, problems = _build_assignment_and_hold_records(result, scenario, tolerance=1e-6)

    assert assignments == []
    assert problems != []
    assert _record_for(holds, 0).decision == "FAILED"
    assert len(assignments) + len(holds) == len(scenario.robots)


def test_assigned_with_proposal_missing_cluster_id_is_rejected():
    scenario = _provenance_test_scenario()
    proposal = ExplorationCandidate(target=(1.2, 4.2), metadata={})
    result = CoordinationResult(
        assignments=(
            CoordinationAssignment(
                robot_id=0, status="ASSIGNED", target=(1.2, 4.2), reason="r", proposal=proposal
            ),
            CoordinationAssignment(robot_id=1, status="HOLD", target=None, reason="r"),
            CoordinationAssignment(robot_id=2, status="HOLD", target=None, reason="r"),
        )
    )

    assignments, holds, problems = _build_assignment_and_hold_records(result, scenario, tolerance=1e-6)

    assert assignments == []
    assert problems != []
    assert _record_for(holds, 0).decision == "FAILED"


def test_assigned_with_unknown_cluster_id_is_rejected():
    scenario = _provenance_test_scenario()
    proposal = ExplorationCandidate(target=(1.2, 4.2), metadata={"cluster_id": "does-not-exist"})
    result = CoordinationResult(
        assignments=(
            CoordinationAssignment(
                robot_id=0, status="ASSIGNED", target=(1.2, 4.2), reason="r", proposal=proposal
            ),
            CoordinationAssignment(robot_id=1, status="HOLD", target=None, reason="r"),
            CoordinationAssignment(robot_id=2, status="HOLD", target=None, reason="r"),
        )
    )

    assignments, holds, problems = _build_assignment_and_hold_records(result, scenario, tolerance=1e-6)

    assert assignments == []
    assert problems != []
    assert _record_for(holds, 0).decision == "FAILED"


def test_assigned_with_invalid_component_cluster_id_is_rejected():
    """f3 in the shipped scenario has valid=False -- its target must never
    be accepted even though the component itself exists."""
    scenario = _provenance_test_scenario()
    proposal = ExplorationCandidate(target=(-3.25, -3.0), metadata={"cluster_id": "f3"})
    result = CoordinationResult(
        assignments=(
            CoordinationAssignment(
                robot_id=0, status="ASSIGNED", target=(-3.25, -3.0), reason="r", proposal=proposal
            ),
            CoordinationAssignment(robot_id=1, status="HOLD", target=None, reason="r"),
            CoordinationAssignment(robot_id=2, status="HOLD", target=None, reason="r"),
        )
    )

    assignments, holds, problems = _build_assignment_and_hold_records(result, scenario, tolerance=1e-6)

    assert assignments == []
    assert problems != []
    assert _record_for(holds, 0).decision == "FAILED"


def test_assigned_with_valid_cluster_id_but_mismatched_target_is_rejected():
    scenario = _provenance_test_scenario()
    proposal = ExplorationCandidate(target=(1.2, 4.2), information_gain=3.5, metadata={"cluster_id": "f1"})
    result = CoordinationResult(
        assignments=(
            CoordinationAssignment(
                robot_id=0, status="ASSIGNED", target=(99.0, 99.0), reason="r", proposal=proposal
            ),
            CoordinationAssignment(robot_id=1, status="HOLD", target=None, reason="r"),
            CoordinationAssignment(robot_id=2, status="HOLD", target=None, reason="r"),
        )
    )

    assignments, holds, problems = _build_assignment_and_hold_records(result, scenario, tolerance=1e-6)

    assert assignments == []
    assert problems != []
    assert _record_for(holds, 0).decision == "FAILED"


def test_assigned_with_valid_cluster_id_and_matching_target_is_accepted():
    scenario = _provenance_test_scenario()
    proposal = ExplorationCandidate(target=(1.2, 4.2), information_gain=3.5, metadata={"cluster_id": "f1"})
    result = CoordinationResult(
        assignments=(
            CoordinationAssignment(
                robot_id=0, status="ASSIGNED", target=(1.2, 4.2), reason="r", proposal=proposal
            ),
            CoordinationAssignment(robot_id=1, status="HOLD", target=None, reason="r"),
            CoordinationAssignment(robot_id=2, status="HOLD", target=None, reason="r"),
        )
    )

    assignments, holds, problems = _build_assignment_and_hold_records(result, scenario, tolerance=1e-6)

    assert problems == []
    assert len(assignments) == 1
    assert assignments[0].cluster_id == "f1"
    assert assignments[0].robot_id == 0


def test_malformed_assignment_preserves_robot_count_invariant():
    scenario = _provenance_test_scenario()
    good_proposal = ExplorationCandidate(target=(1.2, 4.2), information_gain=3.5, metadata={"cluster_id": "f1"})
    result = CoordinationResult(
        assignments=(
            CoordinationAssignment(
                robot_id=0, status="ASSIGNED", target=(1.2, 4.2), reason="r", proposal=good_proposal
            ),
            CoordinationAssignment(
                robot_id=1, status="ASSIGNED", target=(4.25, 2.0), reason="r", proposal=None
            ),
            CoordinationAssignment(robot_id=2, status="HOLD", target=None, reason="no candidates"),
        )
    )

    assignments, holds, problems = _build_assignment_and_hold_records(result, scenario, tolerance=1e-6)
    metrics = _compute_metrics(assignments, holds, tolerance=1e-6)

    assert problems != []
    assert metrics.assigned_robot_count == 1
    assert metrics.unassigned_robot_count == 2
    assert metrics.assigned_robot_count + metrics.unassigned_robot_count == len(scenario.robots)
    assert _record_for(holds, 1).decision == "FAILED"


def test_problems_appear_in_deterministic_robot_id_order():
    scenario = _provenance_test_scenario()
    result = CoordinationResult(
        assignments=(
            CoordinationAssignment(robot_id=2, status="ASSIGNED", target=(0.0, 0.0), reason="r", proposal=None),
            CoordinationAssignment(robot_id=0, status="ASSIGNED", target=(0.0, 0.0), reason="r", proposal=None),
            CoordinationAssignment(robot_id=1, status="ASSIGNED", target=(0.0, 0.0), reason="r", proposal=None),
        )
    )

    _, _, problems = _build_assignment_and_hold_records(result, scenario, tolerance=1e-6)

    assert len(problems) == 3
    assert problems[0].startswith("robot 0:")
    assert problems[1].startswith("robot 1:")
    assert problems[2].startswith("robot 2:")
