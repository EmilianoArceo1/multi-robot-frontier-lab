"""Action contract for the learning pipeline.

An action is the selection of a candidate viewpoint plus a heading -- no
velocities and no motor commands.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningAction:
    """Selected viewpoint + heading for one robot at one decision step."""

    robot_id: int
    candidate_id: str
    candidate_index: int
    heading_index: int
    action_index: int
    issued_at_step: int

    def __post_init__(self) -> None:
        for name in ("robot_id", "candidate_index", "heading_index", "action_index", "issued_at_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
