"""
async_trace_writer.py

Non-blocking background file writer for belief-trace artifacts.

Wraps a BeliefTraceWriter (the actual file I/O in belief_trace_writer.py)
behind a bounded, priority-aware queue serviced by one daemon worker
thread, so simulation_step()/paintEvent() never block waiting on disk I/O.

Priority model, implemented as two separate collections.deque instances
(NOT a single queue.Queue) so every operation -- enqueue, evict-to-make-
room, dequeue -- is O(1). A single queue.Queue with an O(n) scan-for-
something-to-evict was tried first and rejected: once the queue fills
with all-high-priority items (the common case under load, since low-
priority map/obstacle events are already throttled at the source), that
scan degenerates into a repeated O(n) no-op on every subsequent enqueue
call -- exactly the blocking-ish behavior this module exists to avoid.

    HIGH priority (route, decision, frontier, route_affected events, and
    summary flushes): always enqueued if there is room. If both deques are
    full, the OLDEST low-priority item is evicted (deque.popleft(), O(1))
    to make room; only when there is no low-priority item left to evict
    does a high-priority enqueue actually get dropped (counted
    separately).

    LOW priority (periodic map/obstacle-section/belief-snapshot events):
    dropped outright when the queue is full -- these are periodic/
    redundant by nature (trace_map()'s own throttle already limits how
    often they occur), so losing one under backpressure is inconsequential
    for debugging, and low-priority items never evict anything.

The worker thread drains all pending high-priority items before any
low-priority one -- a deliberate, simple priority policy (not a strict
global FIFO across both classes), appropriate for best-effort diagnostic
output where losing/reordering a periodic map snapshot relative to a
route event is immaterial.

Same public method names as BeliefTraceWriter (record_event,
record_route_event, record_frontier_event, record_decision_event,
record_obstacle_section, record_route_affected_event, write_belief_snapshot,
flush_summary) so RobotTrace's existing trace_*() call sites need no
changes at all -- only what RobotTrace.start_run() constructs as self.writer
changes (see robot_trace.py).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

DEFAULT_QUEUE_MAXSIZE = 500
DEFAULT_FLUSH_TIMEOUT_S = 1.0

# event_type values (record_event's first positional arg) that are
# low-priority/droppable; everything else defaults to high-priority.
_LOW_PRIORITY_EVENT_TYPES = frozenset({"map", "obstacles"})


@dataclass
class _QueueItem:
    method: str
    kwargs: dict


class AsyncTraceWriter:
    """Non-blocking facade in front of a BeliefTraceWriter.

    start_worker=False (tests only) constructs the queue/drop-accounting
    without ever starting the background thread, so enqueue()/drop
    behavior can be asserted deterministically without a race against a
    concurrently-draining worker.
    """

    def __init__(
        self,
        writer,
        *,
        maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        warn: Callable[[str], None] | None = None,
        start_worker: bool = True,
    ):
        self._writer = writer
        self._maxsize = int(maxsize)
        self._high: deque = deque()
        self._low: deque = deque()
        self._not_empty = threading.Condition()
        self._warn = warn or (lambda message: None)
        self.enabled = True
        self.dropped_low_priority = 0
        self.dropped_high_priority = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if start_worker:
            self._thread = threading.Thread(
                target=self._run, name="belief-trace-writer", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------------ enqueue (called from the simulation thread)

    def _enqueue(self, method: str, priority: str, kwargs: dict) -> bool:
        """O(1), never blocks. Returns True if the item was queued, False
        if dropped (queue full)."""
        if not self.enabled:
            return False
        item = _QueueItem(method=method, kwargs=kwargs)
        with self._not_empty:
            total = len(self._high) + len(self._low)
            if priority == "low":
                if total >= self._maxsize:
                    self.dropped_low_priority += 1
                    return False
                self._low.append(item)
                self._not_empty.notify()
                return True

            # High priority.
            if total >= self._maxsize:
                if self._low:
                    self._low.popleft()  # O(1) eviction, oldest low-priority item
                    self.dropped_low_priority += 1
                else:
                    self.dropped_high_priority += 1
                    return False
            self._high.append(item)
            self._not_empty.notify()
            return True

    # ------------------------------------------------------------------ BeliefTraceWriter-compatible facade

    def record_event(self, event_type: str, **kwargs: Any) -> bool:
        priority = "low" if event_type in _LOW_PRIORITY_EVENT_TYPES else "high"
        return self._enqueue("record_event", priority, dict(event_type=event_type, **kwargs))

    def record_route_event(self, **kwargs: Any) -> bool:
        return self._enqueue("record_route_event", "high", kwargs)

    def record_frontier_event(self, **kwargs: Any) -> bool:
        return self._enqueue("record_frontier_event", "high", kwargs)

    def record_decision_event(self, **kwargs: Any) -> bool:
        return self._enqueue("record_decision_event", "high", kwargs)

    def record_route_affected_event(self, **kwargs: Any) -> bool:
        return self._enqueue("record_route_affected_event", "high", kwargs)

    def record_obstacle_section(self, **kwargs: Any) -> bool:
        return self._enqueue("record_obstacle_section", "low", kwargs)

    def write_belief_snapshot(self, snapshot: dict) -> bool:
        return self._enqueue("write_belief_snapshot", "low", dict(snapshot=snapshot))

    def flush_summary(self) -> bool:
        return self._enqueue("flush_summary", "high", {})

    # ------------------------------------------------------------------ worker thread

    def _run(self) -> None:
        while True:
            item = self._dequeue(timeout=0.2)
            if item is None:
                if self._stop_event.is_set() and self.queue_size == 0:
                    return
                continue
            self._process(item)

    def _dequeue(self, timeout: float) -> "_QueueItem | None":
        """High-priority items always drain before any low-priority one."""
        with self._not_empty:
            if not self._high and not self._low:
                self._not_empty.wait(timeout=timeout)
            if self._high:
                return self._high.popleft()
            if self._low:
                return self._low.popleft()
            return None

    def _process(self, item: _QueueItem) -> None:
        try:
            getattr(self._writer, item.method)(**item.kwargs)
        except Exception as exc:  # best-effort: never let a background error surface
            self.enabled = False
            self._warn(f"[BELIEF TRACE] background writer disabled after error: {exc}")

    # ------------------------------------------------------------------ lifecycle

    @property
    def queue_size(self) -> int:
        with self._not_empty:
            return len(self._high) + len(self._low)

    @property
    def dropped_total(self) -> int:
        return self.dropped_low_priority + self.dropped_high_priority

    def flush(self, timeout: float = DEFAULT_FLUSH_TIMEOUT_S) -> None:
        """Best-effort: wait up to *timeout* seconds for the queue to
        drain. Never raises; if the deadline passes with items still
        queued, they are simply left for the worker to keep draining (or
        lost on process exit, since the thread is a daemon)."""
        deadline = time.monotonic() + float(timeout)
        while self.queue_size > 0 and time.monotonic() < deadline:
            time.sleep(0.01)

    def close(self, timeout: float = DEFAULT_FLUSH_TIMEOUT_S) -> None:
        """Best-effort shutdown: flush what we can, then stop the worker.
        Safe to call even if start_worker=False (no-op thread join)."""
        self.flush(timeout=timeout)
        self._stop_event.set()
        with self._not_empty:
            self._not_empty.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
