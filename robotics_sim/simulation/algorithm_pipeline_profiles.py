"""Paper-derived dependencies between selectable exploration pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from robotics_sim.planning.frontier_clustering import (
    GRIT_DBSCAN_TWO_STAGE,
    NO_CLUSTERING_ALGORITHM,
)
from robotics_sim.planning.keidar_kaminka_efd import KEIDAR_KAMINKA_WFD_INC


CQLITE_TASK_ASSIGNMENT = (
    "Travel-time Voronoi + CQLite distributed Q-learning"
)
HUNGARIAN_TASK_ASSIGNMENT = "Frontier cluster Hungarian coordinator"


@dataclass(frozen=True)
class TaskAssignmentPipelineProfile:
    clustering_algorithm: str
    lock_clustering: bool
    frontier_detector: str | None
    lock_frontier_detector: bool
    explanation: str


_PROFILES = {
    CQLITE_TASK_ASSIGNMENT: TaskAssignmentPipelineProfile(
        clustering_algorithm=NO_CLUSTERING_ALGORITHM,
        lock_clustering=True,
        frontier_detector=KEIDAR_KAMINKA_WFD_INC,
        lock_frontier_detector=True,
        explanation=(
            "CQLite uses Keidar-Kaminka efficient frontier detection and does "
            "not use the separate DBSCAN clustering stage."
        ),
    ),
    HUNGARIAN_TASK_ASSIGNMENT: TaskAssignmentPipelineProfile(
        clustering_algorithm=GRIT_DBSCAN_TWO_STAGE,
        lock_clustering=True,
        frontier_detector=None,
        lock_frontier_detector=False,
        explanation=(
            "The Hungarian task allocator consumes the cited two-stage "
            "GriT-DBSCAN frontier clusters."
        ),
    ),
}


def task_assignment_pipeline_profile(
    task_assignment: str,
) -> TaskAssignmentPipelineProfile | None:
    return _PROFILES.get(str(task_assignment))
