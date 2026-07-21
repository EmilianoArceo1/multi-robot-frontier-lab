"""Frontier cluster and viewpoint contracts.

These exist to let a plugin work in FUEL/RACER-style terms (frontier
clusters, sampled viewpoints with heading) without requiring the full
pipeline (ESDF, B-splines, trajectory optimization) those papers use. A
viewpoint/cluster can always be converted down to a plain
robotics_interfaces.proposals.ExplorationCandidate, so existing runtime code
(MMPF, NOIC legacy, independent_baseline) does not need to know this module
exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from robotics_interfaces.observations import Point2D
from robotics_interfaces.proposals import ExplorationCandidate


@dataclass(frozen=True)
class ViewpointCandidate:
    """A sampled sensing pose: where to stand and which way to face.

    This is the FUEL/RACER-style analogue of ExplorationCandidate. It exists
    separately because a viewpoint's value depends on orientation
    (heading_rad) and expected coverage, not just position and information
    gain -- fields ExplorationCandidate does not need for simpler algorithms.
    """

    xy: Point2D
    heading_rad: float | None = None
    information_gain: float = 0.0
    coverage_fraction: float = 0.0
    visible_cell_count: int = 0
    travel_cost: float = 0.0
    safety_cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def utility(self) -> float:
        return self.information_gain - self.travel_cost - self.safety_cost

    def as_exploration_candidate(self, source: str = "viewpoint") -> ExplorationCandidate:
        return ExplorationCandidate(
            target=self.xy,
            source=source,
            information_gain=self.information_gain,
            travel_cost=self.travel_cost,
            safety_cost=self.safety_cost,
            heading_rad=self.heading_rad,
            metadata={
                **dict(self.metadata),
                "coverage_fraction": self.coverage_fraction,
                "visible_cell_count": self.visible_cell_count,
            },
        )


@dataclass(frozen=True)
class FrontierCluster:
    """A connected group of frontier cells and the viewpoints sampled to
    observe it. cells/centroid describe the frontier itself; viewpoints are
    candidate sensing poses for covering it."""

    cluster_id: str
    cells: tuple[Point2D, ...] = ()
    centroid: Point2D | None = None
    viewpoints: tuple[ViewpointCandidate, ...] = ()
    information_gain: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    valid: bool = True

    @property
    def best_viewpoint(self) -> ViewpointCandidate | None:
        """Highest-utility sampled viewpoint for this cluster, if any."""
        if not self.viewpoints:
            return None
        return max(self.viewpoints, key=lambda viewpoint: viewpoint.utility)

    def as_exploration_candidate(self, source: str = "frontier_cluster") -> ExplorationCandidate | None:
        """Convert this cluster's best viewpoint into a plain
        ExplorationCandidate, so any existing target-generation plugin can
        consume a frontier cluster without knowing this module exists.

        Always stamps cluster_id/frontier_cell_count/cluster_valid into the
        result's metadata -- this is the provenance a benchmark/plugin needs
        to trace an assignment back to this cluster -- without touching any
        other key the viewpoint's own metadata already set.
        """
        best = self.best_viewpoint
        if best is None:
            return None
        candidate = best.as_exploration_candidate(source=source)
        metadata = {
            **dict(candidate.metadata),
            "cluster_id": self.cluster_id,
            "frontier_cell_count": len(self.cells),
            "cluster_valid": self.valid,
        }
        return replace(candidate, metadata=metadata)
