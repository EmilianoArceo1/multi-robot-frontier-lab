from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from robotics_interfaces.observations import Point2D


@dataclass(frozen=True)
class ExplorationCandidate:
    """A simulator-independent candidate exploration target.

    The candidate can be created by the simulator host, by an injected
    service, or by the plugin itself.  The fields are intentionally generic
    enough for simple greedy baselines, frontier-based planners, and later
    potential-field variants.
    """

    target: Point2D
    source: str = "unknown"
    information_gain: float = 0.0
    travel_cost: float = 0.0
    safety_cost: float = 0.0
    overlap_cost: float = 0.0
    heading_cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def utility(self) -> float:
        return (
            self.information_gain
            - self.travel_cost
            - self.safety_cost
            - self.overlap_cost
            - self.heading_cost
        )


@dataclass(frozen=True)
class CandidateProposal:
    """Backward-compatible target proposal used by older tests/plugins.

    New code should prefer ExplorationCandidate unless it needs a proposal
    tied to a specific robot id.
    """

    robot_id: int
    target: Point2D
    score: float
    information_gain: float = 0.0
    travel_cost: float = 0.0
    overlap_cost: float = 0.0
    safety_cost: float = 0.0
    heading_cost: float = 0.0
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_candidate(self, source: str = "proposal") -> ExplorationCandidate:
        return ExplorationCandidate(
            target=self.target,
            source=source,
            information_gain=self.information_gain,
            travel_cost=self.travel_cost,
            safety_cost=self.safety_cost,
            overlap_cost=self.overlap_cost,
            heading_cost=self.heading_cost,
            metadata={
                **dict(self.metadata),
                "robot_id": self.robot_id,
                "score": self.score,
                "reason": self.reason,
            },
        )
