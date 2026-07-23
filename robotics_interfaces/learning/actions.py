"""Action contract for the learning pipeline.

An action is the selection of a candidate viewpoint plus an optional
heading -- no velocities and no motor commands.

v0 candidate/heading semantics: one ExplorationCandidate is one selectable
action, and a candidate carries at most one heading.  ``heading_index`` is
not a second action dimension -- it only records whether the selected
candidate had an explicit heading (``0``) or not (``None``).  Representing
the same viewpoint with several headings requires the candidate generator
to emit separate candidates, never a single candidate with several
headings.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LearningAction:
    """Selected viewpoint + optional heading for one robot at one decision
    step.

    ``heading_index`` is ``None`` when the selected candidate carried no
    explicit heading; otherwise it is a non-negative int.  It is never a
    second tensor dimension: v0's action space is one-dimensional per
    candidate.
    """

    robot_id: int
    candidate_id: str
    candidate_index: int
    heading_index: int | None
    action_index: int
    issued_at_step: int

    def __post_init__(self) -> None:
        for name in ("robot_id", "candidate_index", "action_index", "issued_at_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        if self.heading_index is not None:
            if isinstance(self.heading_index, bool) or not isinstance(self.heading_index, int):
                raise TypeError(
                    f"heading_index must be an int or None, got "
                    f"{type(self.heading_index).__name__}"
                )
            if self.heading_index < 0:
                raise ValueError(f"heading_index must be non-negative, got {self.heading_index}")
