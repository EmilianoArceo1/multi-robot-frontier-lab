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
MARVEL_TASK_ASSIGNMENT = "MARVEL CTDE graph-attention policy"
MARVEL_SCALED_TASK_ASSIGNMENT = (
    "MARVEL CTDE graph-attention policy (scaled environment)"
)


@dataclass(frozen=True)
class TaskAssignmentPipelineProfile:
    clustering_algorithm: str
    lock_clustering: bool
    frontier_detector: str | None
    lock_frontier_detector: bool
    explanation: str
    default_vision_model: str | None = None
    default_sensor_range: float | None = None
    default_camera_fov_degrees: float | None = None
    lock_vision_model: bool = False
    sensing_note: str = ""
    bootstrap_panorama: bool = False


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
    MARVEL_TASK_ASSIGNMENT: TaskAssignmentPipelineProfile(
        clustering_algorithm=NO_CLUSTERING_ALGORITHM,
        lock_clustering=True,
        frontier_detector=None,
        lock_frontier_detector=True,
        explanation=(
            "MARVEL generates a sampled viewpoint graph and encodes frontier "
            "distributions internally; it does not use a separate clustering "
            "or host frontier-detector stage."
        ),
        default_vision_model="Camera / FoV",
        default_sensor_range=10.0,
        default_camera_fov_degrees=120.0,
        lock_vision_model=True,
        bootstrap_panorama=True,
        sensing_note=(
            "Original paper scale: 4.0 m graph nodes, 0.8 m frontier voxels "
            "and a 60 m local map. Reducing only sensor range is outside the "
            "published configuration and can leave the graph disconnected."
        ),
    ),
    MARVEL_SCALED_TASK_ASSIGNMENT: TaskAssignmentPipelineProfile(
        clustering_algorithm=NO_CLUSTERING_ALGORITHM,
        lock_clustering=True,
        frontier_detector=None,
        lock_frontier_detector=True,
        explanation=(
            "MARVEL scaled uses the same PolicyNet checkpoint and internal "
            "frontier graph, but scales all physical graph lengths together "
            "for compact simulator environments."
        ),
        default_vision_model="Camera / FoV",
        default_sensor_range=3.0,
        default_camera_fov_degrees=120.0,
        lock_vision_model=True,
        bootstrap_panorama=True,
        sensing_note=(
            "Scale-normalized adapter: at the 3.0 m default it uses 1.2 m "
            "graph nodes, a 2.7 m utility range, approximately 0.24 m "
            "frontier voxels and an 18 m local map."
        ),
    ),
}


def task_assignment_pipeline_profile(
    task_assignment: str,
) -> TaskAssignmentPipelineProfile | None:
    return _PROFILES.get(str(task_assignment))
