"""Reward contract with training phases and per-term warm-up.

Weights here are configuration/traceability values, not final scientific
coefficients.  The phase + warm-up design lets a term be introduced
gradually when moving from one curriculum phase to the next.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass


class RewardPhase(enum.Enum):
    COVERAGE = "coverage"
    FIRE_SEARCH = "fire_search"
    MULTI_ROBOT = "multi_robot"


class RewardTerm(enum.Enum):
    NEW_COVERAGE = "new_coverage"
    PATH_COST = "path_cost"
    COMPLETION = "completion"
    FIRE_INFORMATION_GAIN = "fire_information_gain"
    NEW_FIRE_DETECTED = "new_fire_detected"
    SENSING_OVERLAP = "sensing_overlap"
    DUPLICATE_TARGET = "duplicate_target"
    ROBOT_PROXIMITY = "robot_proximity"
    ROUTE_CONFLICT = "route_conflict"
    COLLISION = "collision"


@dataclass(frozen=True)
class LinearWeightWarmup:
    """Linear ramp for a reward term's weight.

    weight(step) is 0 before ``start_step``, interpolates linearly on
    [start_step, end_step], and equals ``target_weight`` afterwards.
    """

    start_step: int
    end_step: int
    target_weight: float

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}")
        if self.end_step <= self.start_step:
            raise ValueError(
                f"end_step ({self.end_step}) must be greater than start_step ({self.start_step})"
            )
        if not math.isfinite(self.target_weight):
            raise ValueError(f"target_weight must be finite, got {self.target_weight!r}")

    def weight_at(self, step: int) -> float:
        if step < self.start_step:
            return 0.0
        if step >= self.end_step:
            return self.target_weight
        fraction = (step - self.start_step) / (self.end_step - self.start_step)
        return self.target_weight * fraction


@dataclass(frozen=True)
class RewardTermSpec:
    """Configuration of one reward term inside a RewardSpec."""

    term: RewardTerm
    phase_introduced: RewardPhase
    warmup: LinearWeightWarmup | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.term, RewardTerm):
            raise TypeError(f"term must be a RewardTerm, got {type(self.term).__name__}")
        if not isinstance(self.phase_introduced, RewardPhase):
            raise TypeError(
                f"phase_introduced must be a RewardPhase, got "
                f"{type(self.phase_introduced).__name__}"
            )
        if self.warmup is not None and not isinstance(self.warmup, LinearWeightWarmup):
            raise TypeError(
                f"warmup must be a LinearWeightWarmup or None, got {type(self.warmup).__name__}"
            )


@dataclass(frozen=True)
class RewardSpec:
    """Versioned collection of reward term configurations."""

    schema_version: str
    terms: tuple[RewardTermSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        seen: set[RewardTerm] = set()
        for spec in self.terms:
            if not isinstance(spec, RewardTermSpec):
                raise TypeError(
                    f"terms entries must be RewardTermSpec, got {type(spec).__name__}"
                )
            if spec.term in seen:
                raise ValueError(f"duplicate reward term {spec.term.name}")
            seen.add(spec.term)
