"""Headless, deterministic static-allocation benchmark runner ("Experiment 0").

    python -m experiments.run_experiment \
        --config experiments/configs/rapid_allocation.json \
        --output experiments/results/rapid_allocation_result.json

This is NOT a physical simulation. It never imports PySide6, MainWindow,
SimulationCanvas, QTimer, engine.py, A*/AStarPlanner, or any runtime
controller -- it loads a static JSON scenario (fixed robot positions, fixed
frontier components, fixed observed obstacles, fixed seed), builds a
CoordinationRequest from the existing robotics_interfaces contracts (via
experiments/static_services.py), invokes one existing coordination plugin's
assign(request) through the existing plugin loader
(robotics_sim.simulation.plugin_loader.load_coordination_plugin), and writes
a deterministic JSON record (experiments/records.py) describing the
resulting allocation.

The plugin driven by this first commit is whatever
experiments/configs/rapid_allocation.json names in its "algorithm" field --
today that is IndependentBaselinePlugin's published metadata.name,
"Independent baseline coordinator" (see algorithms/independent_baseline/
plugin.py's INDEPENDENT_BASELINE_COORDINATOR). Nothing in this file hardcodes
that string; it is only ever read from the scenario and handed to
load_coordination_plugin(). A later commit can point a new scenario at a
"frontier_cluster_hungarian" plugin without changing this runner, the
scenario schema, or the result schema.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

from experiments.records import (
    SCHEMA_VERSION,
    AllocationMetrics,
    AssignmentRecord,
    ExperimentRecord,
    HoldRecord,
    finalize_experiment_record,
    finalize_record_json,
)
from experiments.static_services import (
    DEFAULT_DUPLICATE_TOLERANCE,
    ScenarioConfigError,
    StaticFrontierProvider,
    StaticScenario,
    build_coordination_request,
    component_assignment_target,
    euclidean_distance,
    scenario_from_dict,
)
from robotics_sim.simulation.plugin_loader import PluginLoadError, load_coordination_plugin

_RESULTS_DIR = (Path(__file__).resolve().parent / "results").resolve()


# ---------------------------------------------------------------------------
# 1. load_scenario
# ---------------------------------------------------------------------------


def load_scenario(config_path: str) -> StaticScenario:
    path = Path(config_path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioConfigError(f"could not read config file {config_path!r}: {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ScenarioConfigError(f"config file {config_path!r} is not valid JSON: {exc}") from exc

    return scenario_from_dict(data)


# ---------------------------------------------------------------------------
# 2. run_static_allocation_benchmark
# ---------------------------------------------------------------------------


def _robots_by_id(scenario: StaticScenario) -> dict[int, Any]:
    return {robot.robot_id: robot for robot in scenario.robots}


def _valid_component_target_by_id(scenario: StaticScenario) -> dict[str, tuple[float, float]]:
    """cluster_id -> derived target, for every valid=True component that has
    one (see static_services.component_assignment_target() -- the single
    place this selection rule is implemented). A cluster_id absent from this
    dict is not assignable: either it names an invalid component or one with
    no derivable target, which this benchmark treats identically."""
    result: dict[str, tuple[float, float]] = {}
    for component in scenario.frontier_components:
        if not component.valid:
            continue
        target = component_assignment_target(component)
        if target is None:
            continue
        result[component.cluster_id] = target
    return result


def _validate_assignment_provenance(
    assignment: Any,
    *,
    valid_targets_by_cluster_id: Mapping[str, tuple[float, float]],
    tolerance: float,
) -> tuple[str | None, str | None]:
    """Verify that an ASSIGNED CoordinationAssignment can be traced back to
    a valid=True scenario frontier component's derived target.

    Returns (cluster_id, None) on success, or (None, problem) when the
    assignment's provenance cannot be verified. Never raises -- a malformed
    assignment is a soft benchmark outcome (drives ExperimentRecord.success),
    not a runner bug.
    """
    if assignment.target is None:
        return None, f"robot {assignment.robot_id}: status ASSIGNED but target is None"

    proposal = assignment.proposal
    if proposal is None:
        return None, (
            f"robot {assignment.robot_id}: status ASSIGNED but proposal is None -- "
            "target has no verifiable provenance"
        )

    metadata = getattr(proposal, "metadata", None) or {}
    cluster_id = metadata.get("cluster_id")
    if not isinstance(cluster_id, str) or not cluster_id:
        return None, f"robot {assignment.robot_id}: proposal.metadata is missing a non-empty 'cluster_id'"

    expected_target = valid_targets_by_cluster_id.get(cluster_id)
    if expected_target is None:
        return None, (
            f"robot {assignment.robot_id}: cluster_id {cluster_id!r} is not a currently valid "
            "frontier component with a derivable target"
        )

    if euclidean_distance(assignment.target, expected_target) > tolerance:
        got = (float(assignment.target[0]), float(assignment.target[1]))
        return None, (
            f"robot {assignment.robot_id}: assigned target {got!r} does not match the target "
            f"{expected_target!r} derived from cluster_id {cluster_id!r}"
        )

    return cluster_id, None


def _build_assignment_and_hold_records(
    result: Any, scenario: StaticScenario, *, tolerance: float
) -> tuple[list[AssignmentRecord], list[HoldRecord], list[str]]:
    """Normalize a CoordinationResult into (assignments, holds, problems).

    problems is non-empty exactly when an ASSIGNED result's target
    provenance cannot be verified against scenario.frontier_components (see
    _validate_assignment_provenance()) or a robot is missing from the
    result -- these drive ExperimentRecord.success, they never raise (a
    raised exception here would be a plugin/runner bug, not a soft
    benchmark outcome).

    Every robot_id the CoordinationResult reports ends up in exactly one of
    assignments/holds -- a malformed ASSIGNED entry becomes a HoldRecord
    with decision="FAILED" rather than silently vanishing, so
    assigned_robot_count + unassigned_robot_count always accounts for every
    robot this function has seen.
    """
    robots_by_id = _robots_by_id(scenario)
    valid_targets_by_cluster_id = _valid_component_target_by_id(scenario)

    assignments: list[AssignmentRecord] = []
    holds: list[HoldRecord] = []
    problems: list[str] = []
    seen_robot_ids: set[int] = set()

    ordered_assignments = sorted(result.assignments, key=lambda item: item.robot_id)
    for assignment in ordered_assignments:
        seen_robot_ids.add(assignment.robot_id)

        if assignment.status != "ASSIGNED":
            holds.append(
                HoldRecord(robot_id=assignment.robot_id, decision=assignment.status, reason=assignment.reason)
            )
            continue

        robot = robots_by_id.get(assignment.robot_id)
        if robot is None:
            problem = f"robot {assignment.robot_id}: ASSIGNED but not present in scenario robots"
            problems.append(problem)
            holds.append(HoldRecord(robot_id=assignment.robot_id, decision="FAILED", reason=problem))
            continue

        cluster_id, problem = _validate_assignment_provenance(
            assignment, valid_targets_by_cluster_id=valid_targets_by_cluster_id, tolerance=tolerance
        )
        if problem is not None:
            problems.append(problem)
            holds.append(HoldRecord(robot_id=assignment.robot_id, decision="FAILED", reason=problem))
            continue

        information_gain = float(getattr(assignment.proposal, "information_gain", 0.0))
        assignments.append(
            AssignmentRecord(
                robot_id=assignment.robot_id,
                target=(float(assignment.target[0]), float(assignment.target[1])),
                cluster_id=cluster_id,
                decision=assignment.status,
                reason=assignment.reason,
                distance=euclidean_distance(robot.position, assignment.target),
                information_gain=information_gain,
            )
        )

    requested_ids = {robot.robot_id for robot in scenario.robots}
    missing_ids = sorted(requested_ids - seen_robot_ids)
    if missing_ids:
        problems.append(f"CoordinationResult is missing assignments for robot_id(s): {missing_ids}")

    return assignments, holds, problems


def _duplicate_target_count(assignments: list[AssignmentRecord], *, tolerance: float) -> int:
    """Convention: process assignments ordered by robot_id; a robot counts
    as a duplicate iff its target is within `tolerance` of some EARLIER
    (lower robot_id) assignment's target already seen in that order. This
    counts additional robots sharing an already-claimed target, not every
    pairwise combination -- e.g. 3 robots on the exact same target counts
    as 2 duplicates, not 3 (one pair each way)."""
    ordered = sorted(assignments, key=lambda item: item.robot_id)
    seen_targets: list[tuple[float, float]] = []
    duplicate_count = 0
    for item in ordered:
        if any(euclidean_distance(item.target, seen) <= tolerance for seen in seen_targets):
            duplicate_count += 1
        seen_targets.append(item.target)
    return duplicate_count


def _compute_metrics(
    assignments: list[AssignmentRecord], holds: list[HoldRecord], *, tolerance: float
) -> AllocationMetrics:
    assigned_count = len(assignments)
    unassigned_count = len(holds)
    total_distance = sum(item.distance for item in assignments)
    mean_distance = total_distance / assigned_count if assigned_count else 0.0
    return AllocationMetrics(
        duplicate_target_count=_duplicate_target_count(assignments, tolerance=tolerance),
        assigned_robot_count=assigned_count,
        unassigned_robot_count=unassigned_count,
        total_assignment_distance=total_distance,
        mean_assignment_distance=mean_distance,
        # No block/obstacle-factor/line-of-sight/collision modeling in this
        # first benchmark (see experiments/records.py's AllocationMetrics
        # docstring) -- always 0 until a later commit adds that information.
        blocked_assignment_count=0,
    )


def run_static_allocation_benchmark(scenario: StaticScenario) -> ExperimentRecord:
    random.seed(scenario.seed)
    try:
        import numpy  # already a project dependency; not added by this change.

        numpy.random.seed(scenario.seed)
    except ImportError:
        pass

    plugin = load_coordination_plugin(scenario.algorithm)

    duplicate_tolerance = float(
        scenario.parameters.get("assignment_duplicate_tolerance", DEFAULT_DUPLICATE_TOLERANCE)
    )
    frontier_provider = StaticFrontierProvider(
        scenario.frontier_components, duplicate_tolerance=duplicate_tolerance
    )
    request = build_coordination_request(scenario, frontier_provider=frontier_provider)

    result = plugin.assign(request)

    assignments, holds, problems = _build_assignment_and_hold_records(
        result, scenario, tolerance=duplicate_tolerance
    )
    metrics = _compute_metrics(assignments, holds, tolerance=duplicate_tolerance)

    diagnostics: dict[str, Any] = {}
    if problems:
        diagnostics["problems"] = list(problems)
    plugin_debug = getattr(result, "debug", None)
    if plugin_debug:
        diagnostics["plugin_debug"] = dict(plugin_debug)

    record = ExperimentRecord(
        schema_version=SCHEMA_VERSION,
        experiment_id=scenario.experiment_id,
        scenario_id=scenario.scenario_id,
        algorithm=scenario.algorithm,
        seed=scenario.seed,
        robot_count=len(scenario.robots),
        raw_frontier_components=len(scenario.frontier_components),
        valid_frontier_components=sum(1 for c in scenario.frontier_components if c.valid),
        assignments=tuple(assignments),
        holds=tuple(holds),
        duplicate_target_count=metrics.duplicate_target_count,
        assigned_robot_count=metrics.assigned_robot_count,
        unassigned_robot_count=metrics.unassigned_robot_count,
        total_assignment_distance=metrics.total_assignment_distance,
        mean_assignment_distance=metrics.mean_assignment_distance,
        blocked_assignment_count=metrics.blocked_assignment_count,
        success=not problems,
        diagnostics=diagnostics,
    )
    return finalize_experiment_record(record)


# ---------------------------------------------------------------------------
# 3. write_experiment_record
# ---------------------------------------------------------------------------


def write_experiment_record(record: ExperimentRecord, output_path: str) -> dict[str, Any]:
    """Write `record` as canonical-ish (indented, but still JSON) to
    output_path, which MUST resolve inside experiments/results/ -- this
    runner never creates directories or writes files anywhere else."""
    json_dict = finalize_record_json(record)

    resolved_output = Path(output_path).resolve()
    try:
        resolved_output.relative_to(_RESULTS_DIR)
    except ValueError as exc:
        raise ScenarioConfigError(f"--output must be inside {_RESULTS_DIR} (got {output_path!r})") from exc

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output.open("w", encoding="utf-8") as handle:
        json.dump(json_dict, handle, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")

    return json_dict


# ---------------------------------------------------------------------------
# 4. main / CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run_experiment",
        description="Deterministic, headless static allocation benchmark runner.",
    )
    parser.add_argument("--config", required=True, help="Path to a scenario JSON file.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the ExperimentRecord JSON. Must resolve inside experiments/results/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        scenario = load_scenario(args.config)
    except ScenarioConfigError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    try:
        record = run_static_allocation_benchmark(scenario)
    except ScenarioConfigError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2
    except PluginLoadError as exc:
        print(f"error: could not load coordination plugin: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:  # the plugin itself raised, or result normalization crashed hard
        print(f"error: coordination plugin failed: {exc!r}", file=sys.stderr)
        return 4

    try:
        json_dict = write_experiment_record(record, args.output)
    except ScenarioConfigError as exc:
        print(f"error: invalid --output path: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not write output file: {exc}", file=sys.stderr)
        return 5

    print(f"experiment_id: {json_dict['experiment_id']}")
    print(f"scenario_id: {json_dict['scenario_id']}")
    print(f"algorithm: {json_dict['algorithm']}")
    print(f"assigned_robot_count: {json_dict['assigned_robot_count']}")
    print(f"unassigned_robot_count: {json_dict['unassigned_robot_count']}")
    print(f"duplicate_target_count: {json_dict['duplicate_target_count']}")
    print(f"deterministic_fingerprint: {json_dict['deterministic_fingerprint']}")
    print(f"output: {Path(args.output).resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
