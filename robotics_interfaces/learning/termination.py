"""Episode termination contract.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class TerminationReason(enum.Enum):
    RUNNING = "running"
    COVERAGE_COMPLETE = "coverage_complete"
    ALL_FIRE_FOUND = "all_fire_found"
    COVERAGE_AND_FIRE_COMPLETE = "coverage_and_fire_complete"
    MAX_STEPS = "max_steps"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_VALID_ACTION = "no_valid_action"
    COLLISION = "collision"
    EXTERNAL_STOP = "external_stop"
    ERROR = "error"


@dataclass(frozen=True)
class TerminationSpec:
    """Versioned episode termination configuration."""

    schema_version: str
    max_steps: int
    require_coverage: bool
    require_all_fire_found: bool
    coverage_threshold: float

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if not 0.0 <= self.coverage_threshold <= 1.0:
            raise ValueError(
                f"coverage_threshold must be in [0, 1], got {self.coverage_threshold}"
            )
