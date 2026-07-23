"""Paper-grounded UI taxonomy for interchangeable coordination approaches."""

from __future__ import annotations

from dataclasses import dataclass

from robotics_sim.simulation.mapping_architecture import MappingArchitecture


MARVEL_COORDINATOR = "MARVEL CTDE graph-attention policy"
MARVEL_SCALED_COORDINATOR = (
    "MARVEL CTDE graph-attention policy (scaled environment)"
)
CQLITE_COORDINATOR = "Travel-time Voronoi + CQLite distributed Q-learning"
HUNGARIAN_COORDINATOR = "Frontier cluster Hungarian coordinator"

# Three independent binary axes used by the approach badges. Architecture is
# displayed separately because CTDE can combine centralized training/shared
# information with decentralized policy execution.
APPROACH_CATEGORY_OPTIONS = {
    "Paradigm": ("Conventional", "Learning-based"),
    "Decision": ("Goal-level", "Action-level"),
    "Communication": ("Unconstrained", "Constrained"),
}


@dataclass(frozen=True)
class ApproachBadge:
    category: str
    label: str
    color: str


@dataclass(frozen=True)
class CoordinationApproachProfile:
    architecture_label: str
    architecture_color: str
    mapping_architecture: MappingArchitecture
    badges: tuple[ApproachBadge, ApproachBadge, ApproachBadge]


_PROFILES = {
    HUNGARIAN_COORDINATOR: CoordinationApproachProfile(
        architecture_label="Centralized architecture",
        architecture_color="#2563A6",
        mapping_architecture=MappingArchitecture.CENTRALIZED,
        badges=(
            ApproachBadge("Paradigm", "Conventional", "#B45309"),
            ApproachBadge("Decision", "Goal-level", "#0F766E"),
            ApproachBadge("Communication", "Unconstrained", "#475569"),
        ),
    ),
    CQLITE_COORDINATOR: CoordinationApproachProfile(
        architecture_label="SLAM / decentralized architecture",
        architecture_color="#6D3AA8",
        mapping_architecture=MappingArchitecture.DECENTRALIZED_SLAM,
        badges=(
            ApproachBadge("Paradigm", "Learning-based", "#7C3AED"),
            ApproachBadge("Decision", "Goal-level", "#0F766E"),
            ApproachBadge("Communication", "Constrained", "#C2410C"),
        ),
    ),
    MARVEL_COORDINATOR: CoordinationApproachProfile(
        architecture_label="Decentralized execution (CTDE)",
        architecture_color="#4338CA",
        # MARVEL assumes perfect communication and maintains one shared map,
        # even though each actor executes its learned policy independently.
        mapping_architecture=MappingArchitecture.CENTRALIZED,
        badges=(
            ApproachBadge("Paradigm", "Learning-based", "#7C3AED"),
            ApproachBadge("Decision", "Goal-level", "#0F766E"),
            ApproachBadge("Communication", "Unconstrained", "#D97706"),
        ),
    ),
    MARVEL_SCALED_COORDINATOR: CoordinationApproachProfile(
        architecture_label="Decentralized execution (CTDE) - scaled",
        architecture_color="#0E7490",
        mapping_architecture=MappingArchitecture.CENTRALIZED,
        badges=(
            ApproachBadge("Paradigm", "Learning-based", "#7C3AED"),
            ApproachBadge("Decision", "Goal-level", "#0F766E"),
            ApproachBadge("Communication", "Unconstrained", "#D97706"),
        ),
    ),
}


def approach_profile_for_task_assignment(
    task_assignment: str,
) -> CoordinationApproachProfile:
    return _PROFILES.get(str(task_assignment), _PROFILES[HUNGARIAN_COORDINATOR])
