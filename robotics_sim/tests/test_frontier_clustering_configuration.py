"""The runtime exposes only explicitly registered, cited clustering."""

from types import SimpleNamespace

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.planning.exploration_planners import select_exploration_goal
from robotics_sim.planning.frontier_clustering import (
    CLUSTERING_ALGORITHM_OPTIONS,
    GRIT_DBSCAN_TWO_STAGE,
    NO_CLUSTERING_ALGORITHM,
    FrontierClusteringRegistry,
    cluster_frontier_cells,
)
from robotics_sim.simulation.config import (
    SimulationConfig,
    config_from_sim_payload,
    config_to_sim_payload,
)
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.planner_services import PlannerServices


CQLITE_TASK_ASSIGNMENT = "Travel-time Voronoi + CQLite distributed Q-learning"
HUNGARIAN_TASK_ASSIGNMENT = "Frontier cluster Hungarian coordinator"


def _belief() -> BeliefMap:
    belief = BeliefMap(bounds=(-2.0, 2.0, -2.0, 2.0), resolution=0.5)
    belief.force_free_point((0.0, 0.0))
    return belief


def test_global_registry_advertises_the_cited_grit_dbscan_implementation():
    assert CLUSTERING_ALGORITHM_OPTIONS == (GRIT_DBSCAN_TWO_STAGE,)
    result = cluster_frontier_cells(
        NO_CLUSTERING_ALGORITHM,
        belief_map=_belief(),
        frontier_cells={(4, 4)},
    )
    assert result.success is False
    assert result.clusters == ()
    assert "no clustering algorithm" in result.reason


def test_registered_two_stage_clustering_merges_near_components_and_keeps_far_one():
    result = cluster_frontier_cells(
        GRIT_DBSCAN_TWO_STAGE,
        belief_map=_belief(),
        frontier_cells={(0, 0), (1, 0), (5, 0), (20, 0)},
    )

    assert result.success is True
    assert result.citation
    assert result.clusters == (((0, 0), (1, 0), (5, 0)), ((20, 0),))


def test_future_registry_entries_require_a_paper_citation():
    registry = FrontierClusteringRegistry()

    try:
        registry.register(
            name="Uncited connected components",
            citation="",
            clusterer=lambda **_: (),
        )
    except ValueError as exc:
        assert "citation" in str(exc)
    else:  # pragma: no cover - documents the non-negotiable contract
        raise AssertionError("uncited clustering algorithm was accepted")


def test_sim_round_trip_records_explicit_absence_without_legacy_fallback():
    config = SimulationConfig(clustering_algorithm=NO_CLUSTERING_ALGORITHM)
    payload = config_to_sim_payload(config)

    assert payload["exploration"]["clustering_algorithm"] == (
        NO_CLUSTERING_ALGORITHM
    )
    assert config_from_sim_payload(payload).clustering_algorithm == (
        NO_CLUSTERING_ALGORITHM
    )

    payload["exploration"]["clustering_algorithm"] = "Default"
    assert config_from_sim_payload(payload).clustering_algorithm == (
        NO_CLUSTERING_ALGORITHM
    )


def test_multi_robot_task_assignment_normalizes_paper_pipeline_dependencies():
    cqlite_payload = config_to_sim_payload(
        SimulationConfig(
            agent_mode="Multiple Robot Mode",
            coordinator_type=CQLITE_TASK_ASSIGNMENT,
            clustering_algorithm=GRIT_DBSCAN_TWO_STAGE,
        )
    )
    cqlite = config_from_sim_payload(cqlite_payload)
    assert cqlite.clustering_algorithm == NO_CLUSTERING_ALGORITHM
    assert cqlite.exploration_planner == "Keidar-Kaminka WFD-INC frontier detector"

    hungarian_payload = config_to_sim_payload(
        SimulationConfig(
            agent_mode="Multiple Robot Mode",
            coordinator_type=HUNGARIAN_TASK_ASSIGNMENT,
            clustering_algorithm=NO_CLUSTERING_ALGORITHM,
        )
    )
    hungarian = config_from_sim_payload(hungarian_payload)
    assert hungarian.clustering_algorithm == GRIT_DBSCAN_TWO_STAGE


def test_fov_start_is_blocked_but_goal_seeking_does_not_need_clustering():
    fake = SimpleNamespace()
    fov_config = SimulationConfig(
        exploration_planner="FoV-aware directional frontier",
        clustering_algorithm=NO_CLUSTERING_ALGORITHM,
    )
    goal_config = SimulationConfig(
        exploration_planner="Goal seeking",
        clustering_algorithm=NO_CLUSTERING_ALGORITHM,
    )

    error = SimulationControllerMixin.clustering_configuration_error(fake, fov_config)
    assert error is not None
    assert "requires a Clustering Algorithm" in error
    assert "implicit connected-component clustering has been removed" in error
    assert (
        SimulationControllerMixin.clustering_configuration_error(fake, goal_config)
        is None
    )


def test_start_stops_before_creating_runtime_state_when_clustering_is_missing():
    messages: list[str] = []
    statuses: list[str] = []
    trace_starts: list[bool] = []
    config = SimulationConfig(
        exploration_planner="FoV-aware directional frontier",
        clustering_algorithm=NO_CLUSTERING_ALGORITHM,
    )
    fake = SimpleNamespace(
        read_config=lambda: config,
        clustering_configuration_error=lambda value: (
            SimulationControllerMixin.clustering_configuration_error(None, value)
        ),
        log_console_message=messages.append,
        canvas=SimpleNamespace(set_status=statuses.append),
        start_belief_trace_run=lambda: trace_starts.append(True),
    )

    SimulationControllerMixin.start_simulation(fake)

    assert messages == statuses
    assert len(messages) == 1
    assert trace_starts == []


def test_agent_planner_services_cannot_use_legacy_clustering_when_configured():
    services = PlannerServices()
    services.clustering_algorithm = NO_CLUSTERING_ALGORITHM

    result = services.select_exploration_target(
        planner_name="FoV-aware directional frontier",
        belief_map=_belief(),
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=None,
        robot_radius=0.2,
        sensor_range=2.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
    )

    assert result.success is False
    assert "clustering stage rejected selection" in result.reason


def test_explicit_runtime_call_cannot_fall_back_to_fov_four_connectivity():
    result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=_belief(),
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        final_goal_xy=(1.0, 1.0),
        clustering_algorithm=NO_CLUSTERING_ALGORITHM,
        frontier_clusters=None,
    )

    assert result.success is False
    assert result.target is None
    assert "Clustering Algorithm" in result.reason
