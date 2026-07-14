"""
Bounded ring buffer of relevant navigation-debug events.

Entries are pushed only for actual decision/route transitions (see the
tagging call sites in robotics_sim.simulation.engine), never one per render
frame -- so a paused/idle simulation neither grows nor overwrites this log.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from robotics_sim.diagnostics.navigation_snapshot import (
    NavigationDebugEventKind,
    NavigationDebugSnapshot,
)

DEFAULT_MAX_EVENTS = 50


@dataclass(frozen=True)
class NavigationDebugEvent:
    event_kind: NavigationDebugEventKind
    snapshot: NavigationDebugSnapshot


class NavigationDebugEventLog:
    """Explicit-bound ring buffer. Indexed access (event_at) is provided now
    so a future prev/next history UI has a contract to call into, even
    though this MVP round only wires `latest()` to the canvas."""

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
