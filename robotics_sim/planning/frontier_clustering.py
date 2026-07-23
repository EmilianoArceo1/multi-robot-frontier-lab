"""Explicit, cited frontier-clustering algorithm registry.

Frontier detection and frontier clustering are different pipeline stages:

    belief map -> frontier cells -> cited clustering algorithm -> candidates

The registered implementation follows Gao et al.'s two-stage pipeline:
continuity clustering followed by exact DBSCAN over the first-stage
centroids.  Radius queries use the equal grid partition described by
GriT-DBSCAN; all frontier cells are retained by the conservative MinPts=1
runtime profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_sim.environment.belief_map import BeliefMap


FrontierCell = tuple[int, int]
FrontierCluster = tuple[FrontierCell, ...]
FrontierClusterer = Callable[..., Iterable[Iterable[FrontierCell]]]

NO_CLUSTERING_ALGORITHM = "No clustering algorithm available"
GRIT_DBSCAN_TWO_STAGE = "GriT-DBSCAN two-stage frontier clustering"
GRIT_DBSCAN_CITATION = (
    "Gao et al., A Novel Frontier-Based Multi-Robot Cooperative Exploration Method; "
    "Huang et al., GriT-DBSCAN: A Spatial Clustering Algorithm for Very Large Databases, "
    "Pattern Recognition 142 (2023) 109658."
)

# The UI currently selects an algorithm, not its hyperparameters.  Six map
# cells is deliberately conservative for the second-stage radius and MinPts=1
# guarantees that isolated but valid frontiers are never discarded.
DEFAULT_GRIT_EPS_CELLS = 6.0
DEFAULT_GRIT_MIN_POINTS = 1


def _continuity_components(frontier_cells: frozenset[FrontierCell]) -> list[list[FrontierCell]]:
    """First stage from Gao et al.: deterministic 8-connected components."""
    remaining = set(frontier_cells)
    components: list[list[FrontierCell]] = []
    offsets = tuple(
        (dx, dy)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx != 0 or dy != 0
    )
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = [seed]
        queue: deque[FrontierCell] = deque([seed])
        while queue:
            x, y = queue.popleft()
            for dx, dy in offsets:
                neighbour = (x + dx, y + dy)
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.append(neighbour)
                    queue.append(neighbour)
        components.append(sorted(component))
    return components


def _two_stage_grit_dbscan(*, belief_map, frontier_cells: Iterable[FrontierCell]):
    """Continuity clustering plus grid-indexed exact DBSCAN on centroids."""
    del belief_map  # Coordinates are grid indices; resolution cancels out.
    detected = frozenset((int(x), int(y)) for x, y in frontier_cells)
    components = _continuity_components(detected)
    if not components:
        return ()

    centroids = tuple(
        (
            sum(cell[0] for cell in component) / len(component),
            sum(cell[1] for cell in component) / len(component),
        )
        for component in components
    )
    eps = DEFAULT_GRIT_EPS_CELLS
    cell_side = eps / math.sqrt(2.0)
    spatial_index: dict[tuple[int, int], list[int]] = {}
    grid_keys: list[tuple[int, int]] = []
    for index, (x, y) in enumerate(centroids):
        key = (math.floor(x / cell_side), math.floor(y / cell_side))
        grid_keys.append(key)
        spatial_index.setdefault(key, []).append(index)

    grid_reach = math.ceil(eps / cell_side)
    neighbours: list[tuple[int, ...]] = []
    for index, (x, y) in enumerate(centroids):
        gx, gy = grid_keys[index]
        found: list[int] = []
        for dx in range(-grid_reach, grid_reach + 1):
            for dy in range(-grid_reach, grid_reach + 1):
                for other in spatial_index.get((gx + dx, gy + dy), ()):
                    ox, oy = centroids[other]
                    if math.hypot(x - ox, y - oy) <= eps:
                        found.append(other)
        neighbours.append(tuple(sorted(set(found))))

    core = {
        index for index, nearby in enumerate(neighbours)
        if len(nearby) >= DEFAULT_GRIT_MIN_POINTS
    }
    visited: set[int] = set()
    groups: list[list[int]] = []
    for seed in range(len(components)):
        if seed not in core or seed in visited:
            continue
        visited.add(seed)
        group = [seed]
        queue: deque[int] = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbour in neighbours[current]:
                if neighbour in core and neighbour not in visited:
                    visited.add(neighbour)
                    group.append(neighbour)
                    queue.append(neighbour)
        groups.append(sorted(group))

    core_group = {member: group_index for group_index, group in enumerate(groups) for member in group}
    for index in range(len(components)):
        if index in core:
            continue
        adjacent = sorted({core_group[n] for n in neighbours[index] if n in core})
        if adjacent:
            groups[adjacent[0]].append(index)

    return tuple(
        tuple(sorted(cell for component_index in group for cell in components[component_index]))
        for group in groups
    )


@dataclass(frozen=True)
class FrontierClusteringAlgorithm:
    """One selectable implementation and the source it reproduces."""

    name: str
    citation: str
    clusterer: FrontierClusterer


@dataclass(frozen=True)
class FrontierClusteringResult:
    success: bool
    algorithm: str
    citation: str
    clusters: tuple[FrontierCluster, ...]
    reason: str


class FrontierClusteringRegistry:
    """Registry with no fallback and mandatory provenance metadata."""

    def __init__(self) -> None:
        self._algorithms: dict[str, FrontierClusteringAlgorithm] = {}

    def register(
        self,
        *,
        name: str,
        citation: str,
        clusterer: FrontierClusterer,
    ) -> None:
        normalized_name = str(name).strip()
        normalized_citation = str(citation).strip()
        if not normalized_name:
            raise ValueError("A clustering algorithm requires a non-empty name.")
        if not normalized_citation:
            raise ValueError(
                f"Clustering algorithm {normalized_name!r} requires a paper citation."
            )
        if not callable(clusterer):
            raise TypeError("clusterer must be callable")
        if normalized_name in self._algorithms:
            raise ValueError(f"Clustering algorithm {normalized_name!r} is already registered.")
        self._algorithms[normalized_name] = FrontierClusteringAlgorithm(
            name=normalized_name,
            citation=normalized_citation,
            clusterer=clusterer,
        )

    def names(self) -> tuple[str, ...]:
        return tuple(self._algorithms)

    def run(
        self,
        algorithm_name: str,
        *,
        belief_map: "BeliefMap",
        frontier_cells: Iterable[FrontierCell],
    ) -> FrontierClusteringResult:
        requested = str(algorithm_name).strip()
        algorithm = self._algorithms.get(requested)
        if algorithm is None:
            return FrontierClusteringResult(
                success=False,
                algorithm=requested or NO_CLUSTERING_ALGORITHM,
                citation="",
                clusters=(),
                reason=(
                    "no clustering algorithm was selected"
                    if not requested or requested == NO_CLUSTERING_ALGORITHM
                    else f"clustering algorithm {requested!r} is not registered"
                ),
            )

        detected = frozenset((int(cell[0]), int(cell[1])) for cell in frontier_cells)
        try:
            raw_clusters = algorithm.clusterer(
                belief_map=belief_map,
                frontier_cells=detected,
            )
            normalized: list[FrontierCluster] = []
            for raw_cluster in raw_clusters:
                cluster = tuple(
                    dict.fromkeys(
                        (int(cell[0]), int(cell[1])) for cell in raw_cluster
                    )
                )
                if not cluster:
                    continue
                outside = set(cluster).difference(detected)
                if outside:
                    raise ValueError(
                        "returned cells that were not produced by the frontier detector: "
                        f"{sorted(outside)!r}"
                    )
                normalized.append(cluster)
        except Exception as exc:
            return FrontierClusteringResult(
                success=False,
                algorithm=algorithm.name,
                citation=algorithm.citation,
                clusters=(),
                reason=f"clustering algorithm failed: {exc}",
            )

        return FrontierClusteringResult(
            success=True,
            algorithm=algorithm.name,
            citation=algorithm.citation,
            clusters=tuple(normalized),
            reason=(
                f"{algorithm.name} produced {len(normalized)} cluster(s) from "
                f"{len(detected)} frontier cell(s)"
            ),
        )


FRONTIER_CLUSTERING_REGISTRY = FrontierClusteringRegistry()

FRONTIER_CLUSTERING_REGISTRY.register(
    name=GRIT_DBSCAN_TWO_STAGE,
    citation=GRIT_DBSCAN_CITATION,
    clusterer=_two_stage_grit_dbscan,
)

CLUSTERING_ALGORITHM_OPTIONS = FRONTIER_CLUSTERING_REGISTRY.names()


def cluster_frontier_cells(
    algorithm_name: str,
    *,
    belief_map: "BeliefMap",
    frontier_cells: Iterable[FrontierCell],
) -> FrontierClusteringResult:
    return FRONTIER_CLUSTERING_REGISTRY.run(
        algorithm_name,
        belief_map=belief_map,
        frontier_cells=frontier_cells,
    )
