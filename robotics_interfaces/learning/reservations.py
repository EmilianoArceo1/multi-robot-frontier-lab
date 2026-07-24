"""Route reservation contract (data only -- no protocol implementation).

This module defines only the shape of reservations for a future
coordination protocol.  It does not implement reservation, runtime
arbitration, reservation-based action masks, or route modification.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

Point2D = tuple[float, float]

KNOWN_BIAS_ROBOT_ID_TIE_BREAK = (
    "Tie-breaking by lowest robot_id after age is deterministic but biased: "
    "low-id robots systematically win contested reservations, which can cause "
    "structural starvation of high-id robots. This bias is accepted for v0 "
    "and must be re-evaluated (e.g. rotating or lottery tie-breaks) in a "
    "future phase."
)


class ReservationTieBreaker(enum.Enum):
    OLDEST_THEN_LOWEST_ROBOT_ID = "oldest_then_lowest_robot_id"


@dataclass(frozen=True)
class RouteReservation:
    """One robot's claim over a route polyline for a time window."""

    robot_id: int
    route_id: str
    polyline: tuple[Point2D, ...]
    start_time: float
    estimated_end_time: float
    safety_radius: float
    priority: int
    created_at: float
    expires_at: float

    def __post_init__(self) -> None:
        if self.robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {self.robot_id}")
        object.__setattr__(self, "polyline", tuple(tuple(p) for p in self.polyline))
        for i, point in enumerate(self.polyline):
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError(f"polyline[{i}] must be a finite (x, y) pair, got {point!r}")
        if self.safety_radius < 0:
            raise ValueError(f"safety_radius must be non-negative, got {self.safety_radius}")
        if self.estimated_end_time < self.start_time:
            raise ValueError(
                f"estimated_end_time ({self.estimated_end_time}) must not precede "
                f"start_time ({self.start_time})"
            )
        if self.expires_at < self.created_at:
            raise ValueError(
                f"expires_at ({self.expires_at}) must not precede created_at "
                f"({self.created_at})"
            )


@dataclass(frozen=True)
class ReservationSpec:
    """Versioned reservation-protocol configuration (contract only).

    ``known_bias`` documents explicitly that using robot_id as tie-break can
    cause structural starvation and must be re-evaluated in a future phase.
    """

    schema_version: str
    tie_breaker: ReservationTieBreaker
    ttl_s: float
    known_bias: str = KNOWN_BIAS_ROBOT_ID_TIE_BREAK

    def __post_init__(self) -> None:
        if not isinstance(self.tie_breaker, ReservationTieBreaker):
            raise TypeError(
                f"tie_breaker must be a ReservationTieBreaker, got "
                f"{type(self.tie_breaker).__name__}"
            )
        if not (math.isfinite(self.ttl_s) and self.ttl_s > 0):
            raise ValueError(f"ttl_s must be finite and positive, got {self.ttl_s!r}")
        if not self.known_bias.strip():
            raise ValueError("known_bias must document the tie-break bias explicitly")
