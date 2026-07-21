"""Second-stage, deterministic reduction over host-provided FrontierCluster
components.

The host (robotics_sim.planning.coordinated_frontier_planner via
FrontierInformationService) already performs the first-stage clustering:
raw frontier cells grouped into 8-connected FrontierCluster components. This
module treats those components as fixed input and applies ONE more
deterministic reduction pass on top: nearby components are merged into a
single ReducedFrontierTask using a coarse secondary grid plus a centroid
merge-radius check.

This is a deterministic grid-density APPROXIMATION loosely inspired by the
grid-based density clustering idea in Gao et al., "A Novel Frontier-Based
Multi-Robot Cooperative Exploration Method" -- it is explicitly NOT an
implementation of GriT-DBSCAN. There is no density/eps/minPts model, no
core-point/border-point distinction, and no claim of matching the paper's
algorithm beyond the general idea of "bucket by a coarse grid, then merge
what's close." Every rule this module applies is documented in the
functions below.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from robotics_interfaces.frontiers import FrontierCluster
from robotics_interfaces.observations import Point2D
from robotics_interfaces.proposals import ExplorationCandidate

CLUSTERING_METHOD = "deterministic_grid_density_approximation"

# 8-connected secondary-grid neighborhood, including (0, 0) so two clusters
# that land in the exact same secondary cell are always compared too.
_NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
)


@dataclass(frozen=True)
class ReducedFrontierTask:
    """One assignable task after second-stage reduction: either a single
    FrontierCluster passed through unchanged, or several nearby clusters
    merged into one. Internal to this algorithm -- not part of
    robotics_interfaces, since no other consumer needs it."""

    task_id: str
    source_cluster_ids: tuple[str, ...]
    representative_cluster_id: str
    cells: tuple[Point2D, ...]
    centroid: Point2D
    target: Point2D
    heading_rad: float | None
    information_gain: float
    candidate: ExplorationCandidate
    metadata: Mapping[str, Any]


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _grid_cell(point: Point2D, grid_size: float) -> tuple[int, int]:
    return (math.floor(point[0] / grid_size), math.floor(point[1] / grid_size))


def _validate_positive_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite, positive number, got {value!r}")
    return result


def _validate_non_negative_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value!r}")
    return result


def _group_clusters(
    ordered: Sequence[FrontierCluster], *, grid_size: float, merge_radius: float
) -> list[list[str]]:
    """8-connected-secondary-grid + merge_radius BFS grouping.

    Rules (see module docstring): each cluster's centroid is bucketed into
    a secondary grid cell (floor(x/grid_size), floor(y/grid_size)); two
    clusters can share a group only when their secondary cells are
    8-connected AND their centroids are within merge_radius of each other.
    Seeds are taken from `ordered` (already sorted by cluster_id) in that
    stable order -- never set.pop() -- and each group's members end up
    sorted by cluster_id. Groups themselves are returned sorted by their
    own smallest cluster_id.
    """
    cluster_by_id = {cluster.cluster_id: cluster for cluster in ordered}
    grid_cell_by_id = {
        cluster.cluster_id: _grid_cell(cluster.centroid, grid_size) for cluster in ordered
    }
    by_grid_cell: dict[tuple[int, int], list[str]] = {}
    for cluster in ordered:
        by_grid_cell.setdefault(grid_cell_by_id[cluster.cluster_id], []).append(cluster.cluster_id)

    def neighbor_ids(cluster_id: str) -> list[str]:
        gx, gy = grid_cell_by_id[cluster_id]
        centroid = cluster_by_id[cluster_id].centroid
        found: list[str] = []
        for dx, dy in _NEIGHBOR_OFFSETS:
            for other_id in by_grid_cell.get((gx + dx, gy + dy), ()):
                if other_id == cluster_id:
                    continue
                if _distance(centroid, cluster_by_id[other_id].centroid) <= merge_radius:
                    found.append(other_id)
        return sorted(found)

    visited: set[str] = set()
    groups: list[list[str]] = []
    for cluster in ordered:
        seed_id = cluster.cluster_id
        if seed_id in visited:
            continue
        visited.add(seed_id)
        group = [seed_id]
        queue: deque[str] = deque([seed_id])
        while queue:
            current_id = queue.popleft()
            for neighbor_id in neighbor_ids(current_id):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                group.append(neighbor_id)
                queue.append(neighbor_id)
        groups.append(sorted(group))

    groups.sort(key=lambda group: group[0])
    return groups


def _select_representative(clusters: Sequence[FrontierCluster]) -> FrontierCluster:
    """Rule: highest information_gain wins; tie -> most cells; tie ->
    lexicographically smallest cluster_id."""
    return sorted(
        clusters,
        key=lambda cluster: (-cluster.information_gain, -len(cluster.cells), cluster.cluster_id),
    )[0]


def _build_reduced_task(
    group_ids: Sequence[str],
    cluster_by_id: Mapping[str, FrontierCluster],
    *,
    task_index: int,
) -> ReducedFrontierTask | None:
    group_clusters = [cluster_by_id[cluster_id] for cluster_id in group_ids]

    cell_set: set[Point2D] = set()
    for cluster in group_clusters:
        cell_set.update(cluster.cells)
    cells = tuple(sorted(cell_set))

    total_weight = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for cluster in group_clusters:
        weight = float(len(cluster.cells)) if cluster.cells else 1.0
        sum_x += cluster.centroid[0] * weight
        sum_y += cluster.centroid[1] * weight
        total_weight += weight
    centroid = (sum_x / total_weight, sum_y / total_weight)

    information_gain = sum(max(cluster.information_gain, 0.0) for cluster in group_clusters)

    representative = _select_representative(group_clusters)
    candidate = representative.as_exploration_candidate()
    if candidate is None:
        # Precondition violation: reduce_frontier_clusters() only accepts
        # already-validated clusters (see this module's docstring) --
        # returning None here just means this malformed group produces no
        # task, rather than raising for what should be an upstream bug.
        return None

    task_id = f"reduced-task-{task_index:04d}"
    source_cluster_ids = tuple(group_ids)  # already sorted by the caller

    metadata = {
        **dict(candidate.metadata),
        "task_id": task_id,
        "source_cluster_ids": source_cluster_ids,
        "representative_cluster_id": representative.cluster_id,
        "reduced_cluster_count": len(group_clusters),
        "reduced_frontier_cell_count": len(cells),
        "clustering_method": CLUSTERING_METHOD,
    }
    stamped_candidate = replace(candidate, metadata=metadata)

    return ReducedFrontierTask(
        task_id=task_id,
        source_cluster_ids=source_cluster_ids,
        representative_cluster_id=representative.cluster_id,
        cells=cells,
        centroid=centroid,
        target=stamped_candidate.target,
        heading_rad=stamped_candidate.heading_rad,
        information_gain=information_gain,
        candidate=stamped_candidate,
        metadata=metadata,
    )


def _pick_duplicate_winner(
    a: ReducedFrontierTask, b: ReducedFrontierTask
) -> tuple[ReducedFrontierTask, ReducedFrontierTask]:
    """Rule: higher information_gain wins; tie -> lexicographically
    smaller task_id wins. Returns (winner, loser)."""
    if a.information_gain != b.information_gain:
        return (a, b) if a.information_gain > b.information_gain else (b, a)
    return (a, b) if a.task_id < b.task_id else (b, a)


def _deduplicate_by_target(
    tasks: Sequence[ReducedFrontierTask], *, tolerance: float
) -> tuple[tuple[ReducedFrontierTask, ...], tuple[str, ...]]:
    """When two tasks end up with equivalent targets within `tolerance`,
    keep only the winner (see _pick_duplicate_winner). Processes tasks in
    task_id order for determinism; returns (kept, removed_task_ids), both
    sorted by task_id."""
    ordered = sorted(tasks, key=lambda task: task.task_id)
    kept: list[ReducedFrontierTask] = []
    removed_ids: list[str] = []
    for task in ordered:
        match_index = None
        for index, existing in enumerate(kept):
            if _distance(task.target, existing.target) <= tolerance:
                match_index = index
                break
        if match_index is None:
            kept.append(task)
            continue
        winner, loser = _pick_duplicate_winner(kept[match_index], task)
        kept[match_index] = winner
        removed_ids.append(loser.task_id)

    kept.sort(key=lambda task: task.task_id)
    removed_ids.sort()
    return tuple(kept), tuple(removed_ids)


def reduce_frontier_clusters_with_diagnostics(
    clusters: Sequence[FrontierCluster],
    *,
    grid_size: float,
    merge_radius: float,
    duplicate_tolerance: float,
) -> tuple[tuple[ReducedFrontierTask, ...], tuple[str, ...]]:
    """Full second-stage pipeline: group -> build tasks -> deduplicate by
    target. Returns (surviving_tasks, removed_duplicate_task_ids), both
    sorted by task_id -- callers that need to report which task_ids were
    dropped by deduplication (e.g. for plugin diagnostics) should call this
    instead of reduce_frontier_clusters().

    `clusters` must already be validated by the caller (valid=True, has a
    centroid, has viewpoints, has a finite assignable candidate) -- this
    function does not re-validate.
    """
    grid_size = _validate_positive_finite(grid_size, name="grid_size")
    merge_radius = _validate_non_negative_finite(merge_radius, name="merge_radius")
    duplicate_tolerance = _validate_non_negative_finite(duplicate_tolerance, name="duplicate_tolerance")

    ordered = sorted(clusters, key=lambda cluster: cluster.cluster_id)
    if not ordered:
        return (), ()

    groups = _group_clusters(ordered, grid_size=grid_size, merge_radius=merge_radius)
    cluster_by_id = {cluster.cluster_id: cluster for cluster in ordered}

    raw_tasks: list[ReducedFrontierTask] = []
    for index, group in enumerate(groups):
        task = _build_reduced_task(group, cluster_by_id, task_index=index)
        if task is not None:
            raw_tasks.append(task)

    return _deduplicate_by_target(raw_tasks, tolerance=duplicate_tolerance)


def reduce_frontier_clusters(
    clusters: Sequence[FrontierCluster],
    *,
    grid_size: float,
    merge_radius: float,
    duplicate_tolerance: float,
) -> tuple[ReducedFrontierTask, ...]:
    """Deterministic grid-density approximation second-stage reduction over
    already-validated FrontierCluster objects (see this module's
    docstring). Returns only the tasks that survive target deduplication;
    use reduce_frontier_clusters_with_diagnostics() to also learn which
    task_ids were removed as duplicates.
    """
    tasks, _removed = reduce_frontier_clusters_with_diagnostics(
        clusters,
        grid_size=grid_size,
        merge_radius=merge_radius,
        duplicate_tolerance=duplicate_tolerance,
    )
    return tasks
