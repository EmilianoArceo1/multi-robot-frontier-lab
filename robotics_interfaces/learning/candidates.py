"""Candidate observation contracts for the learning pipeline.

These types describe candidate *data* as seen by the learning stack.  They
are deliberately distinct from the runtime CandidateGenerator contract in
``robotics_interfaces.candidate_generation`` (which describes the host-side
generator) and must not duplicate it.

HOLD is not a free policy action in v0: :class:`HoldPolicy` encodes that
restriction contractually, and :func:`validate_action_mask` enforces that
HOLD can only be enabled when no non-HOLD action is valid.  A reward
penalty alone would invite reward hacking; the constraint lives here.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Sequence

Point2D = tuple[float, float]


class CandidateKind(enum.Enum):
    FRONTIER_VIEWPOINT = "frontier_viewpoint"
    FIRE_INFORMATION_VIEWPOINT = "fire_information_viewpoint"
    RECOVERY_VIEWPOINT = "recovery_viewpoint"
    HOLD = "hold"


class HoldReason(enum.Enum):
    NO_VALID_CANDIDATE = "no_valid_candidate"
    WAITING_FOR_RESERVATION = "waiting_for_reservation"
    RECOVERY_COOLDOWN = "recovery_cooldown"


@dataclass(frozen=True)
class HoldPolicy:
    """Explicit v0 decision: HOLD is a host-side fallback, never a policy
    choice, and never available while a non-HOLD action is valid.

    The invariants are enforced at construction so the restriction cannot
    be silently configured away.
    """

    policy_selectable: bool = False
    allow_when_non_hold_available: bool = False
    host_fallback_only: bool = True

    def __post_init__(self) -> None:
        if self.policy_selectable:
            raise ValueError("HoldPolicy v0: HOLD must not be policy_selectable")
        if self.allow_when_non_hold_available:
            raise ValueError(
                "HoldPolicy v0: HOLD must not be allowed while a non-HOLD action is available"
            )
        if not self.host_fallback_only:
            raise ValueError("HoldPolicy v0: HOLD must be host_fallback_only")


@dataclass(frozen=True)
class CandidateObservation:
    """One candidate viewpoint as exposed to the learning stack."""

    candidate_id: str
    kind: CandidateKind
    xy: Point2D
    heading_candidates: tuple[float, ...]
    source: str
    reachable: bool
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CandidateKind):
            raise TypeError(f"kind must be a CandidateKind, got {type(self.kind).__name__}")
        xy = tuple(self.xy)
        if len(xy) != 2:
            raise ValueError(f"xy must be an (x, y) pair, got {xy!r}")
        for v in xy:
            if not math.isfinite(v):
                raise ValueError(f"xy must be finite, got {xy!r}")
        object.__setattr__(self, "xy", xy)
        headings = tuple(self.heading_candidates)
        for h in headings:
            if not math.isfinite(h):
                raise ValueError(f"heading_candidates must be finite, got {h!r}")
        object.__setattr__(self, "heading_candidates", headings)
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))


@dataclass(frozen=True)
class CandidateSetSpec:
    """Shape rules for a candidate set.

    Only a total maximum and form rules -- no per-kind quotas or fixed
    source proportions in v0.
    """

    schema_version: str
    max_candidates: int
    max_headings_per_candidate: int
    deterministic_ordering: bool
    deduplication_distance: float
    hold_policy: HoldPolicy

    def __post_init__(self) -> None:
        if self.max_candidates <= 0:
            raise ValueError(f"max_candidates must be positive, got {self.max_candidates}")
        if self.max_headings_per_candidate <= 0:
            raise ValueError(
                f"max_headings_per_candidate must be positive, got "
                f"{self.max_headings_per_candidate}"
            )
        if not (math.isfinite(self.deduplication_distance) and self.deduplication_distance >= 0.0):
            raise ValueError(
                f"deduplication_distance must be finite and >= 0, got "
                f"{self.deduplication_distance!r}"
            )
        if not isinstance(self.hold_policy, HoldPolicy):
            raise TypeError(
                f"hold_policy must be a HoldPolicy, got {type(self.hold_policy).__name__}"
            )


def validate_action_mask(
    candidates: Sequence[CandidateObservation],
    action_mask: Sequence[bool],
) -> None:
    """Enforce the contractual HOLD restriction on an action mask.

    Raises ``ValueError`` if HOLD is enabled while at least one non-HOLD
    action is valid.  HOLD may only appear enabled as a fallback when no
    non-HOLD action is available.
    """

    if len(candidates) != len(action_mask):
        raise ValueError(
            f"action_mask has {len(action_mask)} entries but there are "
            f"{len(candidates)} candidates"
        )
    non_hold_available = any(
        flag and candidate.kind is not CandidateKind.HOLD
        for candidate, flag in zip(candidates, action_mask)
    )
    if non_hold_available:
        for candidate, flag in zip(candidates, action_mask):
            if flag and candidate.kind is CandidateKind.HOLD:
                raise ValueError(
                    f"HOLD candidate {candidate.candidate_id!r} must not be enabled in "
                    f"action_mask while a non-HOLD action is available"
                )
