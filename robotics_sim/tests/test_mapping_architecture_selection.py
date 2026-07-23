"""Architecture switching driven by multi-robot task assignment."""

from robotics_sim.simulation.mapping_architecture import (
    BeliefMapArchitectureStore,
    MappingArchitecture,
    architecture_for_task_assignment,
    architecture_label,
)
from algorithms.cqlite.plugin import CQLITE_CITATION


CQLITE = "Travel-time Voronoi + CQLite distributed Q-learning"
HUNGARIAN = "Frontier cluster Hungarian coordinator"


def test_task_assigner_declares_centralized_or_decentralized_mapping():
    assert architecture_for_task_assignment(HUNGARIAN) is MappingArchitecture.CENTRALIZED
    assert architecture_for_task_assignment(CQLITE) is MappingArchitecture.DECENTRALIZED_SLAM
    assert architecture_label(MappingArchitecture.CENTRALIZED) == "Centralized architecture"
    assert (
        architecture_label(MappingArchitecture.DECENTRALIZED_SLAM)
        == "SLAM / decentralized architecture"
    )
    assert "10.1109/LRA.2024.3358095" in CQLITE_CITATION


def test_decentralized_store_owns_one_independent_belief_map_per_robot():
    store = BeliefMapArchitectureStore.create(
        architecture=MappingArchitecture.DECENTRALIZED_SLAM,
        bounds=(-2.0, 2.0, -2.0, 2.0),
        resolution=0.5,
        robot_count=3,
    )

    assert store.decentralized is True
    assert len({id(item) for item in store.robot_maps}) == 3
    assert all(item is not store.team_map for item in store.robot_maps)

    store.map_for_robot(0).mark_free_cell((1, 1))
    assert store.map_for_robot(0).grid[1, 1] != store.map_for_robot(1).grid[1, 1]


def test_centralized_store_routes_every_robot_to_the_team_map():
    store = BeliefMapArchitectureStore.create(
        architecture=MappingArchitecture.CENTRALIZED,
        bounds=(-2.0, 2.0, -2.0, 2.0),
        resolution=0.5,
        robot_count=3,
    )

    assert store.decentralized is False
    assert all(item is store.team_map for item in store.robot_maps)
