"""
Bounded in-memory history of navigation-debug snapshots, one per simulation
tick while the layer is enabled -- this is the "cache" the </> step buttons
scrub through. Bounded (not one per render frame -- ticks, not paint
frames) and explicitly cleared whenever a simulation starts/resets (see
engine.py's reset_simulation_state() call sites), so it never touches disk
and never survives a restart or window close.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from robotics_sim.diagnostics.navigation_snapshot import (
    NavigationDebugEventKind,
    NavigationDebugSnapshot,
)

# ~20k ticks. At the simulator's ~60Hz tick rate that is roughly 5-6 minutes
# of real time -- generous for a debugging session without growing without
# bound. Each entry is a handful of small dataclasses/tuples, not an image.
DEFAULT_MAX_EVENTS = 20000


@dataclass(frozen=True)
class NavigationDebugEvent:
    event_kind: NavigationDebugEventKind
    snapshot: NavigationDebugSnapshot


class NavigationDebugEventLog:
    """Explicit-bound ring buffer. Records one entry per tick (see engine.
    _finalize_navigation_debug_snapshot()); event_at(index) is what the
    </> history-step buttons scrub through while paused."""

    def __init__(self, max_size: int = DEFAULT_MAX_EVENTS) -> None:
        self._events: deque[NavigationDebugEvent] = deque(maxlen=max(1, int(max_size)))

    def record(self, event_kind: NavigationDebugEventKind, snapshot: NavigationDebugSnapshot) -> None:
        self._events.append(NavigationDebugEvent(event_kind=event_kind, snapshot=snapshot))

    def __len__(self) -> int:
        return len(self._events)

    def events(self) -> tuple[NavigationDebugEvent, ...]:
        return tuple(self._events)

    def latest(self) -> NavigationDebugEvent | None:
        return self._events[-1] if self._events else None

    def event_at(self, index: int) -> NavigationDebugEvent | None:
        if 0 <= index < len(self._events):
            return self._events[index]
        return None
