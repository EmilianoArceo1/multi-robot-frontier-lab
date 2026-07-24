"""Neutral, versioned data contracts for the future learning pipeline.

First iteration: import explicitly from ``robotics_interfaces.learning``.
The top-level ``robotics_interfaces`` package deliberately does not
re-export these yet.

Interfaces and validation only -- no runtime integration, no dataset
files, no training code.  No robotics_sim, Qt, numpy, torch or pandas.
"""

from __future__ import annotations

from robotics_interfaces.learning.primitives import (
    Primitive,
    UnsupportedPrimitiveError,
    to_primitive,
)
from robotics_interfaces.learning.observations import (
    FORBIDDEN_ACTOR_FIELDS,
    ActorObservation,
    CriticState,
    GroundTruthSnapshot,
)
from robotics_interfaces.learning.candidates import (
    CandidateKind,
    CandidateObservation,
    CandidateSetSpec,
    HoldPolicy,
    HoldReason,
    validate_action_mask,
)
from robotics_interfaces.learning.actions import LearningAction
from robotics_interfaces.learning.rewards import (
    LinearWeightWarmup,
    RewardPhase,
    RewardSpec,
    RewardTerm,
    RewardTermSpec,
)
from robotics_interfaces.learning.transitions import LearningTransition, RewardComponent
from robotics_interfaces.learning.termination import TerminationReason, TerminationSpec
from robotics_interfaces.learning.reservations import (
    KNOWN_BIAS_ROBOT_ID_TIE_BREAK,
    ReservationSpec,
    ReservationTieBreaker,
    RouteReservation,
)
from robotics_interfaces.learning.export import (
    EpisodeFireMetrics,
    EpisodeMetadata,
    TrajectoryExportSpec,
)
from robotics_interfaces.learning.versioning import (
    CONTRACT_VERSIONS,
    build_contract_manifest,
    compute_contract_bundle_hash,
)

__all__ = [
    "ActorObservation",
    "CandidateKind",
    "CandidateObservation",
    "CandidateSetSpec",
    "CONTRACT_VERSIONS",
    "CriticState",
    "EpisodeFireMetrics",
    "EpisodeMetadata",
    "FORBIDDEN_ACTOR_FIELDS",
    "GroundTruthSnapshot",
    "HoldPolicy",
    "HoldReason",
    "KNOWN_BIAS_ROBOT_ID_TIE_BREAK",
    "LearningAction",
    "LearningTransition",
    "LinearWeightWarmup",
    "Primitive",
    "ReservationSpec",
    "ReservationTieBreaker",
    "RewardComponent",
    "RewardPhase",
    "RewardSpec",
    "RewardTerm",
    "RewardTermSpec",
    "RouteReservation",
    "TerminationReason",
    "TerminationSpec",
    "TrajectoryExportSpec",
    "UnsupportedPrimitiveError",
    "build_contract_manifest",
    "compute_contract_bundle_hash",
    "to_primitive",
    "validate_action_mask",
]
