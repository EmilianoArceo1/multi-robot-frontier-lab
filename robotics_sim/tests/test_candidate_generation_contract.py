"""Tests for the new candidate-generation contract and its host adapter.

See robotics_interfaces/candidate_generation.py (CandidateGenerationRequest/
Result/CandidateGenerator) and robotics_sim/simulation/coordination_services.
py (HostFrontierCandidateGenerator, the adapter over the existing
detect_global_frontier_candidates/detect_connected_frontier_components
pipeline). Also covers the new PluginCapability.FRONTIER_DETECTION/
TASK_GENERATION members and PluginMetadata.candidate_input_mode.
"""

from __future__ import annotations

from robotics_interfaces.candidate_generation import (
    CandidateGenerationRequest,
    CandidateGenerationResult,
    CandidateGenerator,
)
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.plugins import CandidateInputMode, PluginCapability, PluginMetadata
from robotics_sim.simulation import coordination_services as services_module
from robotics_sim.simulation.coordination_services import HostFrontierCandidateGenerator


def _robot(robot_id: int, x: float, y: float) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
    )


def _two_island_world() -> WorldSnapshot:
    island_a = [(x * 0.5, y * 0.5) for x in range(-4, 1) for y in range(-2, 3)]
    island_b = [(x * 0.5 + 10.0, y * 0.5) for x in range(-4, 1) for y in range(-2, 3)]
    return WorldSnapshot(
        explored_points=tuple(island_a + island_b),
        mapped_obstacle_points=(),
        bounds=(-6.0, 16.0, -6.0, 6.0),
        resolution=0.5,
    )


def test_new_stage_capabilities_exist_and_are_distinct():
    assert PluginCapability.FRONTIER_DETECTION != PluginCapability.TASK_GENERATION
    assert PluginCapability.FRONTIER_DETECTION != PluginCapability.TARGET_GENERATION
    assert PluginCapability.TASK_ALLOCATION.value == "task_allocation"


def test_candidate_input_mode_has_the_five_expected_values():
    assert {mode.value for mode in CandidateInputMode} == {
        "host_candidates",
        "host_frontier_clusters",
        "plugin_internal",
        "hybrid",
        "legacy_integrated",
    }


def test_plugin_metadata_defaults_candidate_input_mode_to_none():
    metadata = PluginMetadata(
        name="unmigrated plugin",
        version="0.0.0",
        description="",
        capabilities=(PluginCapability.COORDINATION,),
    )
    assert metadata.candidate_input_mode is None


def test_plugin_metadata_can_declare_candidate_input_mode():
    metadata = PluginMetadata(
        name="migrated plugin",
        version="0.0.0",
        description="",
        capabilities=(PluginCapability.COORDINATION,),
        candidate_input_mode=CandidateInputMode.PLUGIN_INTERNAL,
    )
    assert metadata.candidate_input_mode is CandidateInputMode.PLUGIN_INTERNAL


def test_candidate_generator_protocol_is_satisfied_by_the_host_adapter():
    generator = HostFrontierCandidateGenerator()
    assert isinstance(generator, CandidateGenerator)


def test_host_adapter_returns_empty_result_without_a_world():
    generator = HostFrontierCandidateGenerator()
    result = generator.generate(CandidateGenerationRequest(robot_states=(_robot(0, 0.0, 0.0),)))

    assert isinstance(result, CandidateGenerationResult)
    assert result.candidates_by_robot == {}
    assert result.frontier_clusters is None
    assert result.source_name == "host_frontier_candidate_pipeline"


def test_host_adapter_delegates_to_the_existing_detectors_without_reimplementing_them(monkeypatch):
    """The adapter must call the same module-level detectors
    coordination_services.py already uses -- not a second, parallel frontier
    algorithm."""
    calls = {"global": 0, "clusters": 0}

    def fake_global(**kwargs):
        calls["global"] += 1
        return ()

    def fake_clusters(**kwargs):
        calls["clusters"] += 1
        return ()

    monkeypatch.setattr(services_module, "detect_global_frontier_candidates", fake_global)
    monkeypatch.setattr(services_module, "detect_connected_frontier_components", fake_clusters)
    services_module._cached_global_frontier_candidates.cache_clear()

    generator = HostFrontierCandidateGenerator()
    request = CandidateGenerationRequest(
        robot_states=(_robot(0, 0.0, 0.0),),
        world=_two_island_world(),
    )
    generator.generate(request)

    assert calls["global"] == 1
    assert calls["clusters"] == 1
    services_module._cached_global_frontier_candidates.cache_clear()


def test_host_adapter_produces_candidates_and_clusters_from_a_real_two_island_map():
    generator = HostFrontierCandidateGenerator()
    request = CandidateGenerationRequest(
        robot_states=(_robot(0, 0.0, 0.0), _robot(1, 10.0, 0.0)),
        robot_ids=(0, 1),
        world=_two_island_world(),
    )

    result = generator.generate(request)

    assert set(result.candidates_by_robot) == {0, 1}
    assert result.diagnostics["raw_candidate_count"] >= 1
    # Two disconnected explored islands should produce at least two clusters.
    assert result.frontier_clusters is not None
    assert len(result.frontier_clusters) >= 2


def test_host_adapter_excludes_blocked_targets_per_robot():
    generator = HostFrontierCandidateGenerator()
    world = _two_island_world()
    baseline = generator.generate(
        CandidateGenerationRequest(robot_states=(_robot(0, 0.0, 0.0),), robot_ids=(0,), world=world)
    )
    assert baseline.candidates_by_robot[0], "expected at least one candidate to block for this test to be meaningful"
    blocked_target = baseline.candidates_by_robot[0][0].target

    filtered = generator.generate(
        CandidateGenerationRequest(
            robot_states=(_robot(0, 0.0, 0.0),),
            robot_ids=(0,),
            world=world,
            blocked_targets_by_robot={0: (blocked_target,)},
        )
    )

    assert blocked_target not in [c.target for c in filtered.candidates_by_robot[0]]
