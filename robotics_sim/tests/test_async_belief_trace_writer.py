"""
Tests for async_trace_writer.py: the bounded-queue, daemon-thread facade
that keeps belief-trace file writes off the simulation thread.

start_worker=False is used for the enqueue/drop-priority tests (A, B, C)
so queue state can be asserted deterministically without racing a
concurrently-draining background thread. start_worker=True (the real,
default behavior) is used for the lifecycle tests (D, E) that need the
background thread to actually process items.
"""
from __future__ import annotations

import time

from robotics_sim.simulation.async_trace_writer import AsyncTraceWriter


class _RecordingWriter:
    """BeliefTraceWriter double: records every call it receives."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def record_event(self, **kwargs):
        self.calls.append(("record_event", kwargs))

    def record_route_event(self, **kwargs):
        self.calls.append(("record_route_event", kwargs))

    def record_frontier_event(self, **kwargs):
        self.calls.append(("record_frontier_event", kwargs))

    def record_decision_event(self, **kwargs):
        self.calls.append(("record_decision_event", kwargs))

    def record_route_affected_event(self, **kwargs):
        self.calls.append(("record_route_affected_event", kwargs))

    def record_obstacle_section(self, **kwargs):
        self.calls.append(("record_obstacle_section", kwargs))

    def write_belief_snapshot(self, snapshot):
        self.calls.append(("write_belief_snapshot", {"snapshot": snapshot}))

    def flush_summary(self):
        self.calls.append(("flush_summary", {}))


class _RaisingWriter:
    """BeliefTraceWriter double whose record_route_event always raises."""

    def record_route_event(self, **kwargs):
        raise OSError("disk full (simulated)")


# ---------------------------------------------------------------------------
# A. enqueue() never blocks, even when the queue is completely full.
# ---------------------------------------------------------------------------


def test_async_trace_writer_enqueue_does_not_block_when_queue_full():
    writer = AsyncTraceWriter(_RecordingWriter(), maxsize=2, start_worker=False)

    assert writer.record_event(event_type="map", simulation_time=1.0) is True
    assert writer.record_event(event_type="map", simulation_time=2.0) is True

    start = time.perf_counter()
    accepted = writer.record_event(event_type="map", simulation_time=3.0)
    elapsed = time.perf_counter() - start

    assert accepted is False
    assert elapsed < 0.5, "enqueue must never block waiting for queue space"
    assert writer.dropped_low_priority == 1


# ---------------------------------------------------------------------------
# B. Low-priority (map/obstacles) events are dropped first to make room
#    for a high-priority event when the queue is full.
# ---------------------------------------------------------------------------


def test_async_trace_writer_drops_low_priority_map_events_first():
    writer = AsyncTraceWriter(_RecordingWriter(), maxsize=2, start_worker=False)
    writer.record_event(event_type="map", simulation_time=1.0)
    writer.record_event(event_type="map", simulation_time=2.0)
    assert writer.queue_size == 2

    accepted = writer.record_route_event(simulation_time=3.0, robot_id="R1", result="ok")

    assert accepted is True, "a high-priority event must evict a low-priority one instead of being dropped"
    assert writer.dropped_low_priority == 1
    assert writer.queue_size == 2  # one map event evicted, the route event took its place


# ---------------------------------------------------------------------------
# C. A route-fail (high priority) event survives queue pressure as long as
#    any low-priority item can be evicted to make room.
# ---------------------------------------------------------------------------


def test_async_trace_writer_preserves_route_fail_events():
    writer = AsyncTraceWriter(_RecordingWriter(), maxsize=2, start_worker=False)
    writer.record_event(event_type="map", simulation_time=1.0)
    writer.record_decision_event(simulation_time=2.0, robot_id="R1", kind="HOLD")
    assert writer.queue_size == 2

    accepted = writer.record_route_event(
        simulation_time=3.0, robot_id="R1", result="fail", reason="no_path"
    )

    assert accepted is True
    assert writer.dropped_high_priority == 0
    assert writer.dropped_low_priority == 1


# ---------------------------------------------------------------------------
# D. close() flushes best-effort: everything enqueued before close() is
#    eventually processed by the background thread.
# ---------------------------------------------------------------------------


def test_async_trace_writer_flushes_on_close_best_effort():
    underlying = _RecordingWriter()
    writer = AsyncTraceWriter(underlying, maxsize=50, start_worker=True)

    for i in range(10):
        writer.record_route_event(simulation_time=float(i), robot_id="R1", result="ok")
    writer.record_decision_event(simulation_time=10.0, robot_id="R1", kind="HOLD")

    writer.close(timeout=2.0)

    assert len(underlying.calls) == 11
    assert writer._thread is not None
    assert not writer._thread.is_alive()


# ---------------------------------------------------------------------------
# E. A write error in the background thread disables the sink and warns
#    exactly once -- never raises into the caller/simulation thread.
# ---------------------------------------------------------------------------


def test_async_trace_writer_write_error_disables_sink_without_crashing():
    warnings: list[str] = []
    writer = AsyncTraceWriter(_RaisingWriter(), maxsize=50, warn=warnings.append, start_worker=True)

    writer.record_route_event(simulation_time=1.0, robot_id="R1", result="ok")
    writer.close(timeout=2.0)

    assert writer.enabled is False
    assert len(warnings) == 1

    # Further enqueues after disable are silent no-ops, never raise.
    accepted = writer.record_route_event(simulation_time=2.0, robot_id="R1", result="ok")
    assert accepted is False
