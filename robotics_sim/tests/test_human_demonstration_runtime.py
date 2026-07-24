"""Tests for HumanDemonstrationRuntime -- no Qt, no real engine.

The host is a plain FakeHost object; wait_for_human_resume() is a
per-test-configurable closure that plays the role of the GUI's blocking
event loop, letting each test drive selection + resume() synchronously.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from robotics_interfaces.coordination import CoordinationRequest, CoordinationResult
from robotics_interfaces.observations import RobotCoordinationState
from robotics_interfaces.proposals import ExplorationCandidate
from robotics_sim.learning.demonstration_collection_plan import (
    DemonstrationCollectionSetupError,
    load_demonstration_collection_plan,
)
from robotics_sim.learning.map_catalog import load_map_catalog
from robotics_sim.simulation.human_demonstration_runtime import (
    HumanDemonstrationHostBindings,
    HumanDemonstrationRuntime,
    HumanDemonstrationRuntimeState,
    HumanDemonstrationRuntimeStateError,
)


class FakeHost:
    def __init__(self) -> None:
        self.loaded_sim_paths: list[Path] = []
        self.fires: list[tuple[float, float]] = []
        self.clear_calls = 0
        self.hazard_reset_calls = 0
        self.pause_calls = 0
        self.wait_calls = 0
        self.simulation_time_s = 0.0
        self.final_metrics: dict[str, float] = {"coverage": 0.5}
        self.on_wait = None  # set per-test to drive selection + resume()

    def load_sim_file(self, path: Path) -> None:
        self.loaded_sim_paths.append(Path(path))

    def clear_fires(self) -> None:
        self.fires = []
        self.clear_calls += 1

    def add_fire(self, x: float, y: float) -> None:
        self.fires.append((x, y))

    def reset_hazard_belief(self) -> None:
        self.hazard_reset_calls += 1

    def request_pause(self) -> None:
        self.pause_calls += 1

    def wait_for_human_resume(self) -> None:
        self.wait_calls += 1
        if self.on_wait is not None:
            self.on_wait()

    def get_simulation_time_s(self) -> float:
        return self.simulation_time_s

    def get_final_metrics(self):
        return dict(self.final_metrics)


def make_bindings(fake: FakeHost) -> HumanDemonstrationHostBindings:
    return HumanDemonstrationHostBindings(
        load_sim_file=fake.load_sim_file,
        clear_fires=fake.clear_fires,
        add_fire=fake.add_fire,
        reset_hazard_belief=fake.reset_hazard_belief,
        request_pause=fake.request_pause,
        wait_for_human_resume=fake.wait_for_human_resume,
        get_simulation_time_s=fake.get_simulation_time_s,
        get_final_metrics=fake.get_final_metrics,
    )


@pytest.fixture
def catalog_plan_dir(tmp_path: Path):
    manifest_dir = tmp_path / "maps"
    manifest_dir.mkdir()
    (manifest_dir / "01_open.sim").write_text("{}")
    (manifest_dir / "02_office.sim").write_text("{}")
    manifest = {
        "corpus_id": "test_corpus",
        "schema_version": 1,
        "maps": [
            {
                "map_id": "map_open",
                "filename": "01_open.sim",
                "family": "open",
                "difficulty": "smoke",
                "fire_scenarios": [
                    {"scenario_id": "single_fire", "fires": [{"x": 1.0, "y": 2.0}]},
                    {
                        "scenario_id": "double_fire",
                        "fires": [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
                    },
                ],
            },
            {
                "map_id": "map_office",
                "filename": "02_office.sim",
                "family": "office",
                "difficulty": "smoke",
                "fire_scenarios": [{"scenario_id": "single_fire", "fires": [{"x": 5.0, "y": 5.0}]}],
            },
        ],
    }
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    map_catalog = load_map_catalog(manifest_path)

    plan = {
        "plan_id": "test-plan",
        "corpus_id": "test_corpus",
        "assignments": [
            {
                "map_id": "map_open",
                "collector_id": "collector_a",
                "episodes": [
                    {"episode_number": 1, "scenario_id": "single_fire", "seed": 0},
                    {"episode_number": 2, "scenario_id": "double_fire", "seed": 1},
                ],
            },
            {
                "map_id": "map_office",
                "collector_id": "collector_b",
                "episodes": [{"episode_number": 1, "scenario_id": "single_fire", "seed": 0}],
            },
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    collection_plan = load_demonstration_collection_plan(plan_path, map_catalog=map_catalog)

    return map_catalog, collection_plan, manifest_dir


def make_runtime(tmp_path: Path, catalog_plan_dir, fake: FakeHost | None = None):
    map_catalog, collection_plan, manifest_dir = catalog_plan_dir
    fake = fake or FakeHost()
    output_root = tmp_path / "output"
    runtime = HumanDemonstrationRuntime(
        map_catalog=map_catalog,
        collection_plan=collection_plan,
        sim_directory=manifest_dir,
        output_root=output_root,
        host=make_bindings(fake),
    )
    return runtime, fake, output_root


def make_candidate(target=(1.0, 1.0)) -> ExplorationCandidate:
    return ExplorationCandidate(target=target, source="test", information_gain=1.0)


def make_request(robot_ids: tuple[int, ...], candidates_by_robot: dict) -> CoordinationRequest:
    robot_states = tuple(
        RobotCoordinationState(
            robot_id=rid, xy=(0.0, 0.0), safety_radius=0.3, sensor_range=3.0, vision_model="cone"
        )
        for rid in robot_ids
    )
    return CoordinationRequest(
        robot_states=robot_states,
        robots_to_assign=tuple(robot_ids),
        proposals_by_robot={rid: candidates_by_robot[rid] for rid in robot_ids},
    )


# --- collector/map/episode selection --------------------------------------


def test_collector_only_sees_own_maps(tmp_path, catalog_plan_dir):
    runtime, _, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    assert runtime.setup.available_map_ids == ("map_open",)


def test_foreign_map_rejected(tmp_path, catalog_plan_dir):
    runtime, _, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    with pytest.raises(Exception):
        runtime.select_map("map_office")


def test_load_episode_resolves_scenario_and_seed(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(2)
    identity = runtime.load_episode()
    assert identity.scenario_id == "double_fire"
    assert identity.seed == 1
    assert identity.episode_number == 2
    assert identity.map_id == "map_open"
    assert identity.collector_id == "collector_a"


# --- fire injection ---------------------------------------------------------


def test_fire_injected_via_add_fire(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(2)
    runtime.load_episode()
    assert fake.fires == [(1.0, 2.0), (3.0, 4.0)]
    assert runtime.fires_loaded_count == 2


def test_fire_not_added_to_obstacles(tmp_path, catalog_plan_dir):
    # FakeHost only exposes add_fire/clear_fires -- there is no obstacles
    # parameter anywhere in HumanDemonstrationHostBindings, so the runtime
    # has no way to reach an obstacles collection at all.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(HumanDemonstrationHostBindings)}
    assert not any("obstacle" in name for name in field_names)

    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(HumanDemonstrationRuntime.load_episode)))
    attribute_names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not any("obstacle" in name.lower() for name in attribute_names)


def test_load_episode_previous_fires_cleared_first(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    runtime.abort_episode()
    runtime.select_episode(2)
    runtime.load_episode()
    assert fake.clear_calls == 2
    assert fake.fires == [(1.0, 2.0), (3.0, 4.0)]


def test_ground_truth_fire_not_exposed_by_runtime_state(tmp_path, catalog_plan_dir):
    """The runtime's own public surface must never leak raw fire
    coordinates -- only a count. Ground-truth visibility is controlled by
    the (separately tested) canvas rendering toggles, which this runtime
    never touches."""

    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(2)
    runtime.load_episode()

    public_attrs = [name for name in dir(runtime) if not name.startswith("_")]
    for name in public_attrs:
        assert "fire" not in name.lower() or name == "fires_loaded_count"


# --- request_executor / manual round -----------------------------------


def test_request_executor_never_calls_plugin_assign():
    source = inspect.getsource(
        __import__(
            "robotics_sim.simulation.human_demonstration_runtime", fromlist=["_"]
        )
    )
    tree = ast.parse(source)
    assign_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "assign"
    ]
    assert assign_calls == []
    constructor_params = set(inspect.signature(HumanDemonstrationRuntime.__init__).parameters)
    assert "plugin" not in constructor_params


def test_request_executor_creates_exactly_one_session_and_pauses(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    seen_sessions = []

    def on_wait():
        seen_sessions.append(runtime.active_session)
        runtime.select_robot(0)
        runtime.select_candidate(
            robot_id=0,
            candidate_index=0,
            candidate_id=runtime.candidates_for_robot(0)[0].candidate_id,
        )
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)),)})
    result = runtime.request_executor(request)

    assert isinstance(result, CoordinationResult)
    assert len(seen_sessions) == 1
    assert fake.pause_calls == 1
    assert runtime.state is HumanDemonstrationRuntimeState.RECORDING


def test_candidate_pool_frozen_and_matches_request(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    captured_ids = []

    def on_wait():
        slots = runtime.candidates_for_robot(0)
        captured_ids.append([s.candidate_id for s in slots])
        runtime.select_candidate(robot_id=0, candidate_index=0, candidate_id=slots[0].candidate_id)
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)), make_candidate((5.0, 5.0)))})
    runtime.request_executor(request)

    assert len(captured_ids[0]) == 2
    with pytest.raises(Exception):
        runtime.candidate_pool  # not a public attribute of the runtime itself


def test_select_robot_invalid_rejected(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    errors = []

    def on_wait():
        try:
            runtime.select_robot(999)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        runtime.select_candidate(
            robot_id=0, candidate_index=0, candidate_id=runtime.candidates_for_robot(0)[0].candidate_id
        )
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)),)})
    runtime.request_executor(request)
    assert len(errors) == 1


def test_select_candidate_wrong_id_rejected(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    errors = []

    def on_wait():
        try:
            runtime.select_candidate(robot_id=0, candidate_index=0, candidate_id="bogus")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        runtime.select_candidate(
            robot_id=0, candidate_index=0, candidate_id=runtime.candidates_for_robot(0)[0].candidate_id
        )
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)),)})
    runtime.request_executor(request)
    assert len(errors) == 1


def test_ready_only_with_all_selections(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    ready_before = []

    def on_wait():
        ready_before.append(runtime.active_session.ready_to_apply)
        runtime.select_candidate(
            robot_id=0, candidate_index=0, candidate_id=runtime.candidates_for_robot(0)[0].candidate_id
        )
        ready_before.append(runtime.active_session.ready_to_apply)
        runtime.select_candidate(
            robot_id=1, candidate_index=0, candidate_id=runtime.candidates_for_robot(1)[0].candidate_id
        )
        assert runtime.active_session.ready_to_apply
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request(
        (0, 1), {0: (make_candidate((2.0, 3.0)),), 1: (make_candidate((5.0, 5.0)),)}
    )
    runtime.request_executor(request)
    assert ready_before == [False, False]


def test_resume_incomplete_rejected(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    def on_wait():
        with pytest.raises(HumanDemonstrationRuntimeStateError):
            runtime.resume()
        runtime.select_candidate(
            robot_id=0, candidate_index=0, candidate_id=runtime.candidates_for_robot(0)[0].candidate_id
        )
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)),)})
    runtime.request_executor(request)


def test_coordination_result_exact(tmp_path, catalog_plan_dir):
    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    def on_wait():
        runtime.select_candidate(
            robot_id=0, candidate_index=0, candidate_id=runtime.candidates_for_robot(0)[0].candidate_id
        )
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)),)})
    result = runtime.request_executor(request)

    assert type(result) is CoordinationResult
    assert result.assignments[0].robot_id == 0
    assert result.assignments[0].status == "ASSIGNED"
    assert result.assignments[0].target == (2.0, 3.0)


def test_planner_still_computes_route_after_apply(tmp_path, catalog_plan_dir):
    """The manual result never sets .path -- confirming the host's real
    path planner (not this runtime) is what computes the route."""

    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    def on_wait():
        runtime.select_candidate(
            robot_id=0, candidate_index=0, candidate_id=runtime.candidates_for_robot(0)[0].candidate_id
        )
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)),)})
    result = runtime.request_executor(request)
    assert result.commands[0].path == ()


def test_empty_robots_to_assign_does_not_pause_or_create_session(tmp_path, catalog_plan_dir):
    """Mirrors the real engine: synchronize_multi_frontier_targets() never
    even calls assign_frontiers() when robots_to_assign is empty, so a
    safety replan (which keeps the existing target and never re-enters
    coordination) can never register a second human decision."""

    runtime, fake, _ = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    request = make_request((), {})
    result = runtime.request_executor(request)
    assert fake.pause_calls == 0
    assert fake.wait_calls == 0
    assert runtime.decision_count == 0
    assert result.assignments == ()


# --- Finish / Abort -------------------------------------------------------


def _run_one_full_round(runtime, fake, robot_ids=(0,), targets=None):
    targets = targets or {rid: (float(rid), float(rid)) for rid in robot_ids}

    def on_wait():
        for rid in robot_ids:
            slot = runtime.candidates_for_robot(rid)[0]
            runtime.select_candidate(robot_id=rid, candidate_index=0, candidate_id=slot.candidate_id)
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request(robot_ids, {rid: (make_candidate(targets[rid]),) for rid in robot_ids})
    return runtime.request_executor(request)


def test_finish_generates_one_folder_with_four_files(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)

    layout = runtime.finish_episode()
    assert layout.episode_directory.is_dir()
    files = sorted(p.name for p in layout.episode_directory.iterdir())
    assert files == ["decisions.jsonl", "integrity_report.json", "metadata.json", "metrics.json"]


def test_decisions_jsonl_one_line_per_decision(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    _run_one_full_round(runtime, fake, robot_ids=(0, 1))

    layout = runtime.finish_episode()
    lines = layout.decisions_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_metadata_contains_collector_map_scenario_seed_episode(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(2)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)

    layout = runtime.finish_episode()
    metadata = json.loads(layout.metadata_path.read_text(encoding="utf-8"))
    assert metadata["collector_id"] == "collector_a"
    assert metadata["map_id"] == "map_open"
    assert metadata["scenario_id"] == "double_fire"
    assert metadata["seed"] == 1
    assert metadata["episode_number"] == 2


def test_finish_with_incomplete_round_rejected(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    def on_wait():
        with pytest.raises(HumanDemonstrationRuntimeStateError):
            runtime.finish_episode()
        runtime.select_candidate(
            robot_id=0, candidate_index=0, candidate_id=runtime.candidates_for_robot(0)[0].candidate_id
        )
        runtime.resume()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)),)})
    runtime.request_executor(request)
    runtime.finish_episode()


def test_finish_with_zero_decisions_rejected(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    with pytest.raises(HumanDemonstrationRuntimeStateError):
        runtime.finish_episode()


def test_abort_generates_no_folder(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)  # one decision recorded but never finished
    runtime.abort_episode()

    assert not (output_root / "pending_review").exists()
    assert not runtime.has_active_episode


def test_abort_mid_round_returns_hold_and_unblocks(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()

    def on_wait():
        runtime.abort_episode()

    fake.on_wait = on_wait
    request = make_request((0,), {0: (make_candidate((2.0, 3.0)),)})
    result = runtime.request_executor(request)
    assert result.assignments[0].status == "HOLD"
    assert not runtime.has_active_episode


def test_restart_does_not_mix_episodes(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)
    runtime.abort_episode()  # simulates "Restart" forcing an explicit abort

    runtime.select_episode(1)
    runtime.load_episode()
    assert runtime.decision_count == 0


def test_two_episodes_generate_two_independent_folders(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)
    layout_1 = runtime.finish_episode()

    runtime.select_episode(2)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)
    layout_2 = runtime.finish_episode()

    assert layout_1.episode_directory != layout_2.episode_directory
    assert layout_1.episode_directory.is_dir()
    assert layout_2.episode_directory.is_dir()


def test_finish_selects_next_unrecorded_episode_without_starting_it(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)
    runtime.finish_episode()

    assert runtime.setup.selected_episode_number == 2
    assert not runtime.has_active_episode


def test_map_complete_text(tmp_path, catalog_plan_dir):
    runtime, fake, output_root = make_runtime(tmp_path, catalog_plan_dir)
    runtime.select_collector("collector_a")
    runtime.select_map("map_open")
    runtime.select_episode(1)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)
    runtime.finish_episode()
    assert runtime.map_complete_text() is None

    runtime.select_episode(2)
    runtime.load_episode()
    _run_one_full_round(runtime, fake)
    runtime.finish_episode()
    assert runtime.map_complete_text() == "Map complete: Recorded 2 of 2"
