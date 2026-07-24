"""Tests for ManualDemonstrationSelectionSession."""

from __future__ import annotations

import dataclasses
import inspect
from datetime import datetime, timezone

import pytest

from robotics_interfaces.coordination import CoordinationResult
from robotics_interfaces.learning import CandidateKind
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.learning.capture_inputs import CandidateCaptureInput
from robotics_sim.learning.demonstration_episode import DemonstrationEpisodeIdentity
from robotics_sim.learning.manual_demonstration import (
    FrozenCandidateSlot,
    ManualDemonstrationSelectionError,
    ManualDemonstrationSelectionSession,
    ManualDemonstrationSessionState,
    ManualDemonstrationStateError,
)
from robotics_sim.learning.observation_batch import build_candidate_id


def make_identity(**overrides) -> DemonstrationEpisodeIdentity:
    kwargs = dict(
        episode_id="91f3ab20-1111-2222-3333-444455556666",
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


def make_candidate_capture(target, enabled=True, kind=CandidateKind.FRONTIER_VIEWPOINT):
    candidate = ExplorationCandidate(target=target, source="frontier", information_gain=2.0)
    return CandidateCaptureInput(candidate=candidate, kind=kind, enabled=enabled, reachable=True)


def make_pool(two_robots=True):
    pool = {
        0: (make_candidate_capture((0.0, 0.0)), make_candidate_capture((1.0, 1.0))),
    }
    if two_robots:
        pool[1] = (make_candidate_capture((2.0, 2.0)), make_candidate_capture((3.0, 3.0), enabled=False))
    return pool


def make_session(pool=None, robot_ids=(0, 1), steps=None):
    pool = pool if pool is not None else make_pool()
    steps = steps if steps is not None else {rid: i for i, rid in enumerate(robot_ids)}
    return ManualDemonstrationSelectionSession(
        identity=make_identity(),
        simulation_time_s=10.0,
        candidate_pool=pool,
        robot_ids_pending=robot_ids,
        decision_steps_by_robot=steps,
    )


def test_two_robot_pool_selection_and_apply() -> None:
    session = make_session()
    session.select_candidate(
        robot_id=0, candidate_index=1, candidate_id=build_candidate_id(0, 0, 1)
    )
    assert session.state is ManualDemonstrationSessionState.WAITING_FOR_SELECTION
    session.select_candidate(
        robot_id=1, candidate_index=0, candidate_id=build_candidate_id(1, 1, 0)
    )
    assert session.ready_to_apply
    result = session.build_manual_coordination_result()
    assert isinstance(result, CoordinationResult)
    assert session.state is ManualDemonstrationSessionState.APPLIED


def test_valid_candidate_selection_records_decision() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    session.select_candidate(robot_id=0, candidate_index=1, candidate_id=build_candidate_id(0, 0, 1))
    decisions = session.decisions()
    assert len(decisions) == 1
    assert decisions[0].robot_id == 0
    assert decisions[0].target_xy == (1.0, 1.0)


def test_wrong_candidate_id_rejected() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    with pytest.raises(ManualDemonstrationSelectionError):
        session.select_candidate(robot_id=0, candidate_index=1, candidate_id="bogus-id")


def test_out_of_range_index_rejected() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    with pytest.raises(ManualDemonstrationSelectionError):
        session.select_candidate(robot_id=0, candidate_index=99, candidate_id="whatever")


def test_disabled_candidate_rejected() -> None:
    pool = {1: make_pool()[1]}
    session = make_session(robot_ids=(1,), pool=pool, steps={1: 0})
    with pytest.raises(ManualDemonstrationSelectionError):
        session.select_candidate(
            robot_id=1, candidate_index=1, candidate_id=build_candidate_id(1, 0, 1)
        )


def test_unknown_robot_rejected() -> None:
    session = make_session()
    with pytest.raises(ManualDemonstrationSelectionError):
        session.select_candidate(robot_id=999, candidate_index=0, candidate_id="x")
    with pytest.raises(ManualDemonstrationSelectionError):
        session.select_robot(999)


def test_selection_can_change_before_apply() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    session.select_candidate(robot_id=0, candidate_index=0, candidate_id=build_candidate_id(0, 0, 0))
    assert session.decisions()[0].selected_candidate_index == 0
    session.select_candidate(robot_id=0, candidate_index=1, candidate_id=build_candidate_id(0, 0, 1))
    assert session.decisions()[0].selected_candidate_index == 1


def test_ready_to_apply_incomplete_then_complete() -> None:
    session = make_session()
    assert not session.ready_to_apply
    session.select_candidate(robot_id=0, candidate_index=0, candidate_id=build_candidate_id(0, 0, 0))
    assert not session.ready_to_apply
    session.select_candidate(robot_id=1, candidate_index=0, candidate_id=build_candidate_id(1, 1, 0))
    assert session.ready_to_apply


def test_coordination_result_exact() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    session.select_candidate(robot_id=0, candidate_index=1, candidate_id=build_candidate_id(0, 0, 1))
    result = session.build_manual_coordination_result()
    assert len(result.assignments) == 1
    assignment = result.assignments[0]
    assert assignment.robot_id == 0
    assert assignment.status == "ASSIGNED"
    assert assignment.target == (1.0, 1.0)
    assert result.commands[0].target == (1.0, 1.0)
    assert result.strategy == "manual_demonstration"


def test_pool_is_deeply_frozen_from_caller_mutation() -> None:
    mutable_metadata = {"note": "original"}
    candidate = ExplorationCandidate(target=(5.0, 5.0), metadata=mutable_metadata)
    capture = CandidateCaptureInput(
        candidate=candidate, kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=True, reachable=True
    )
    pool = {0: (capture,)}
    session = make_session(robot_ids=(0,), pool=pool, steps={0: 0})

    mutable_metadata["note"] = "mutated after construction"

    session.select_candidate(robot_id=0, candidate_index=0, candidate_id=build_candidate_id(0, 0, 0))
    stored_candidate = session.candidates_for_robot(0)[0].candidate
    assert stored_candidate.metadata["note"] == "original"
    assert stored_candidate is not candidate


def test_apply_only_once() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    session.select_candidate(robot_id=0, candidate_index=0, candidate_id=build_candidate_id(0, 0, 0))
    session.build_manual_coordination_result()
    with pytest.raises(ManualDemonstrationStateError):
        session.build_manual_coordination_result()


def test_abort_prevents_apply() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    session.select_candidate(robot_id=0, candidate_index=0, candidate_id=build_candidate_id(0, 0, 0))
    session.abort()
    assert session.state is ManualDemonstrationSessionState.ABORTED
    with pytest.raises(ManualDemonstrationStateError):
        session.build_manual_coordination_result()
    with pytest.raises(ManualDemonstrationStateError):
        session.select_candidate(robot_id=0, candidate_index=1, candidate_id="x")


def test_identity_and_collector_preserved_in_decision() -> None:
    identity = make_identity(collector_id="collector_b", map_id="smoke_v0_02_office")
    session = ManualDemonstrationSelectionSession(
        identity=identity,
        simulation_time_s=3.0,
        candidate_pool={0: make_pool()[0]},
        robot_ids_pending=(0,),
        decision_steps_by_robot={0: 0},
    )
    session.select_candidate(robot_id=0, candidate_index=0, candidate_id=build_candidate_id(0, 0, 0))
    decision = session.decisions()[0]
    assert decision.episode_id == identity.episode_id


# --- 1. Deep pool immutability -------------------------------------------

def test_outer_candidate_pool_mapping_is_immutable() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    with pytest.raises(TypeError):
        session.candidate_pool[0] = ()
    with pytest.raises(TypeError):
        session.candidate_pool[1] = ()  # even a brand-new key is rejected


def test_per_robot_candidate_sequence_is_immutable() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    sequence = session.candidates_for_robot(0)
    assert isinstance(sequence, tuple)
    with pytest.raises(TypeError):
        sequence[0] = sequence[1]
    with pytest.raises(AttributeError):
        sequence.append(sequence[0])


def test_candidate_and_enabled_state_are_immutable() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    slot = session.candidates_for_robot(0)[0]
    assert isinstance(slot, FrozenCandidateSlot)

    with pytest.raises(dataclasses.FrozenInstanceError):
        slot.candidate_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        slot.candidate_capture.enabled = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        slot.candidate.target = (9.0, 9.0)
    with pytest.raises(TypeError):
        slot.candidate.metadata["injected"] = 1  # MappingProxyType rejects writes


def test_original_pool_mutation_after_construction_does_not_reach_session() -> None:
    pool = {0: make_pool()[0]}
    original_length = len(pool[0])
    session = make_session(robot_ids=(0,), pool=pool, steps={0: 0})

    pool[0] = ()  # mutate the caller's own mapping/tuple after construction
    pool.clear()

    assert len(session.candidates_for_robot(0)) == original_length
    assert session.candidate_pool != {}


# --- 2. candidate_id frozen exactly once, at open ------------------------

def test_select_candidate_never_recomputes_or_reconsults_build_candidate_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import robotics_sim.learning.manual_demonstration as module

    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    shown_id_0 = session.candidates_for_robot(0)[0].candidate_id
    shown_id_1 = session.candidates_for_robot(0)[1].candidate_id

    def _spy(*args, **kwargs):
        raise AssertionError("build_candidate_id must not be called after construction")

    monkeypatch.setattr(module, "build_candidate_id", _spy)

    # Two clicks, including changing the selection -- neither may touch
    # build_candidate_id again; both read the id already frozen in the pool.
    session.select_candidate(robot_id=0, candidate_index=0, candidate_id=shown_id_0)
    session.select_candidate(robot_id=0, candidate_index=1, candidate_id=shown_id_1)


def test_candidate_id_identical_across_shown_selected_record_and_result() -> None:
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps={0: 0})
    shown_id = session.candidates_for_robot(0)[1].candidate_id

    session.select_candidate(robot_id=0, candidate_index=1, candidate_id=shown_id)
    recorded_id = session.decisions()[0].selected_candidate_id
    assert recorded_id == shown_id

    result = session.build_manual_coordination_result()
    assert result.commands[0].metadata["candidate_id"] == shown_id
    assert result.debug["candidate_id_by_robot"][0] == shown_id


def test_decision_steps_by_robot_only_used_to_freeze_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """decision_steps_by_robot is read once, at construction; select_candidate
    must not depend on the original mapping object staying alive or correct."""

    steps = {0: 5}
    session = make_session(robot_ids=(0,), pool={0: make_pool()[0]}, steps=steps)
    steps[0] = 999  # mutate the caller's mapping after construction
    steps.clear()

    shown_id = session.candidates_for_robot(0)[0].candidate_id
    session.select_candidate(robot_id=0, candidate_index=0, candidate_id=shown_id)
    decision = session.decisions()[0]
    assert decision.decision_step == 5  # unaffected by the post-construction mutation


# --- 3. Exact public result type ------------------------------------------

def test_build_manual_coordination_result_matches_expected_public_type() -> None:
    # robotics_sim/simulation/coordination.py (read-only reference, not
    # imported here): CoordinationResultBuilder.assign_frontiers()'s
    # request_executor hook requires
    # isinstance(plugin_result, robotics_interfaces.coordination.
    # CoordinationResult) -- exactly the type asserted below.
    session = make_session(robot_ids=(1, 0), pool={1: make_pool()[1], 0: make_pool()[0]}, steps={1: 0, 0: 1})
    session.select_candidate(robot_id=1, candidate_index=0, candidate_id=build_candidate_id(1, 0, 0))
    session.select_candidate(robot_id=0, candidate_index=1, candidate_id=build_candidate_id(0, 1, 1))

    result = session.build_manual_coordination_result()

    assert type(result) is CoordinationResult
    assert [a.robot_id for a in result.assignments] == [1, 0]
    assert [c.robot_id for c in result.commands] == [1, 0]
    assert all(a.status == "ASSIGNED" for a in result.assignments)
    assert result.targets == (result.assignments[0].target, result.assignments[1].target)


def test_no_planner_or_plugin_assign_is_ever_invoked() -> None:
    import ast

    import robotics_sim.learning.manual_demonstration as module

    tree = ast.parse(inspect.getsource(module))
    assign_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "assign"
    ]
    assert assign_calls == [], "no plugin.assign()/planner call may appear in this module"

    constructor_params = set(inspect.signature(ManualDemonstrationSelectionSession.__init__).parameters)
    assert "plugin" not in constructor_params
    assert "planner" not in constructor_params


def test_no_qt_app_engine_imports_in_module() -> None:
    import ast

    import robotics_sim.learning.manual_demonstration as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.append(node.module)

    forbidden_prefixes = (
        "robotics_sim.app",
        "robotics_sim.simulation",
        "robotics_sim.planning",
        "PyQt",
        "PySide",
        "engine",
    )
    for name in imported_names:
        assert not any(name.startswith(prefix) for prefix in forbidden_prefixes), name
