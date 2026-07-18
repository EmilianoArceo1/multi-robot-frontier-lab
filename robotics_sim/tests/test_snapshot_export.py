"""Tests for the navigation-snapshot export selection/exporter
(robotics_sim.diagnostics.snapshot_export).

Two kinds of fixtures are used on purpose:

  - Lightweight, duck-typed fake events (SimpleNamespace) for the PURE
    selection algorithm (select_navigation_snapshot_events()) -- it only
    ever reads event_kind and a handful of snapshot fields (robot_id,
    simulation_time, navigation_state, tracking_mode, decision_kind,
    agent_state.active_path_mode/.route_generation), so building full
    NavigationDebugSnapshot objects for thousands of synthetic events would
    be pure overhead with no extra coverage.
  - A minimal-but-complete NavigationDebugSnapshot builder (mirroring
    test_navigation_debug_event_log.py's own fixture) for the handful of
    tests that exercise snapshot_rows()/export_navigation_snapshots_xlsx()
    end to end, since those DO read the full contract.
"""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest

from robotics_sim.diagnostics.event_log import NavigationDebugEvent
from robotics_sim.diagnostics.navigation_snapshot import (
    ControllerDebug,
    FrontierDebug,
    Maybe,
    NavigationDebugEventKind,
    NavigationDebugSnapshot,
    PathDebug,
    PlanningGridDebug,
    Pose,
    PredictedMotionDebug,
    RouteValidationDebug,
    SafetyDebug,
)
from robotics_sim.diagnostics.snapshot_export import (
    DEFAULT_AUTO_TARGET_ROWS,
    SnapshotExportError,
    export_navigation_snapshots_xlsx,
    select_navigation_snapshot_events,
    snapshot_rows,
)

_MANDATORY_KINDS = (
    NavigationDebugEventKind.PLAN_ACCEPTED,
    NavigationDebugEventKind.PATH_SIMPLIFIED,
    NavigationDebugEventKind.ROUTE_REJECTED,
    NavigationDebugEventKind.SAFETY_REPLAN,
)


# ---------------------------------------------------------------------------
# Lightweight fakes for the pure selection algorithm.
# ---------------------------------------------------------------------------


def _fake_event(
    event_kind: NavigationDebugEventKind,
    robot_id: str,
    simulation_time: float,
    *,
    navigation_state: str = "moving",
    tracking_mode: str = "TRACK",
    decision_kind: str = "FOLLOW_PATH",
    active_path_mode: str = "A*",
    route_generation: int = 0,
):
    snapshot = SimpleNamespace(
        robot_id=robot_id,
        simulation_time=simulation_time,
        navigation_state=navigation_state,
        tracking_mode=tracking_mode,
        decision_kind=decision_kind,
        agent_state=Maybe.of(
            SimpleNamespace(active_path_mode=active_path_mode, route_generation=route_generation)
        ),
    )
    return SimpleNamespace(event_kind=event_kind, snapshot=snapshot)


def _routine_events(robot_id: str, count: int, *, start_time: float = 0.0) -> list:
    return [
        _fake_event(NavigationDebugEventKind.TICK, robot_id, start_time + float(i))
        for i in range(count)
    ]


def _interleaved_events(robot_ids: tuple[str, ...], ticks: int) -> list:
    events = []
    for tick in range(ticks):
        for robot_id in robot_ids:
            events.append(_fake_event(NavigationDebugEventKind.TICK, robot_id, float(tick)))
    return events


# ---------------------------------------------------------------------------
# 1. Raw mode.
# ---------------------------------------------------------------------------


def test_raw_mode_preserves_all_events_order_and_one_based_indices():
    events = _routine_events("R1", 250)

    selection = select_navigation_snapshot_events(events, mode="raw")

    assert selection.mode == "raw"
    assert selection.routine_stride == 1
    assert selection.target_rows is None
    assert selection.exported_count == 250
    assert selection.source_count == 250
    assert selection.events == tuple(events)
    assert selection.source_indices == tuple(range(1, 251))


# ---------------------------------------------------------------------------
# 2. Custom stride, single robot.
# ---------------------------------------------------------------------------


def test_custom_stride_single_robot_preserves_first_last_and_samples_periodically():
    events = _routine_events("R1", 300)

    selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=10)

    assert selection.source_indices[0] == 1
    assert selection.source_indices[-1] == 300
    # Order must be preserved (ascending source_indices).
    assert list(selection.source_indices) == sorted(selection.source_indices)
    # Roughly one row every 10 routine events (300/10 = 30, +/- the
    # explicit first/last bookends).
    assert 25 <= selection.exported_count <= 35


# ---------------------------------------------------------------------------
# 3. Custom stride, four interleaved robots.
# ---------------------------------------------------------------------------


def test_custom_stride_four_interleaved_robots_samples_each_robot_evenly():
    robot_ids = ("R1", "R2", "R3", "R4")
    events = _interleaved_events(robot_ids, ticks=100)

    selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=5)

    counts = Counter(event.snapshot.robot_id for event in selection.events)
    assert set(counts) == set(robot_ids)

    # No robot is favored just because it appears first each tick -- every
    # robot's sample count must be within 1 of every other's.
    values = list(counts.values())
    assert max(values) - min(values) <= 1

    for robot_id in robot_ids:
        robot_events = [event for event in events if event.snapshot.robot_id == robot_id]
        first_time = robot_events[0].snapshot.simulation_time
        last_time = robot_events[-1].snapshot.simulation_time
        selected_times = [
            event.snapshot.simulation_time for event in selection.events if event.snapshot.robot_id == robot_id
        ]
        assert first_time in selected_times
        assert last_time in selected_times

    assert list(selection.source_indices) == sorted(selection.source_indices)


# ---------------------------------------------------------------------------
# 4. Discrete/mandatory events survive regardless of stride phase.
# ---------------------------------------------------------------------------


def test_mandatory_event_kinds_are_always_preserved():
    events = _routine_events("R1", 40)
    # Plant mandatory events at positions that do NOT line up with a large
    # stride's periodic samples.
    mandatory_positions = {5: NavigationDebugEventKind.PLAN_ACCEPTED, 13: NavigationDebugEventKind.PATH_SIMPLIFIED,
                           22: NavigationDebugEventKind.ROUTE_REJECTED, 31: NavigationDebugEventKind.SAFETY_REPLAN}
    for position, kind in mandatory_positions.items():
        events[position] = _fake_event(kind, "R1", float(position))

    selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=25)

    kept_source_indices = set(selection.source_indices)
    for position in mandatory_positions:
        assert (position + 1) in kept_source_indices  # 1-based
    assert selection.semantic_events_preserved == len(mandatory_positions)


# ---------------------------------------------------------------------------
# 5. Repeated streaks collapse instead of producing one row per tick.
# ---------------------------------------------------------------------------


def test_long_identical_streak_does_not_produce_one_row_per_event():
    events = [
        _fake_event(NavigationDebugEventKind.HOLD, "R1", float(i), navigation_state="holding")
        for i in range(500)
    ]

    selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=10)

    assert selection.exported_count < 100, "500 identical HOLD events must not survive nearly whole"
    assert selection.source_indices[0] == 1  # streak start
    assert selection.source_indices[-1] == 500  # last event of the robot
    # Periodic samples should still show up roughly every 10 positions.
    assert selection.exported_count >= 500 // 10


# ---------------------------------------------------------------------------
# 6. A semantic transition is preserved even off the stride phase.
# ---------------------------------------------------------------------------


def test_navigation_state_transition_is_preserved_off_stride():
    events = _routine_events("R1", 40, start_time=0.0)
    # Position 17 (1-based 18) does not land on a multiple of stride=20.
    events[17] = _fake_event(
        NavigationDebugEventKind.TICK, "R1", 17.0, navigation_state="blocked"
    )

    selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=20)

    assert 18 in selection.source_indices


def test_decision_kind_transition_is_preserved_off_stride():
    events = _routine_events("R1", 40, start_time=0.0)
    events[23] = _fake_event(
        NavigationDebugEventKind.TICK, "R1", 23.0, decision_kind="REPLAN_FOR_SAFETY"
    )

    selection = select_navigation_snapshot_events(events, mode="custom_stride", routine_stride=20)

    assert 24 in selection.source_indices


# ---------------------------------------------------------------------------
# 7. Automatic mode on a realistic 4,500-event history.
# ---------------------------------------------------------------------------


def test_automatic_mode_on_4500_events_lands_near_target_with_small_stride():
    events = _routine_events("R1", 4500)

    selection = select_navigation_snapshot_events(events, mode="automatic_filtered")

    assert selection.mode == "automatic_filtered"
    assert selection.target_rows == DEFAULT_AUTO_TARGET_ROWS
    assert 2 <= selection.routine_stride <= 4, "4500 routine events at a 1500 target should need a stride near 3"
    assert selection.exported_count <= DEFAULT_AUTO_TARGET_ROWS
    assert selection.source_indices[0] == 1
    assert selection.source_indices[-1] == 4500


# ---------------------------------------------------------------------------
# 8. Automatic mode with too many mandatory events: never drops them.
# ---------------------------------------------------------------------------


def test_automatic_mode_never_drops_mandatory_events_to_hit_target():
    events = [
        _fake_event(NavigationDebugEventKind.PLAN_ACCEPTED, "R1", float(i))
        for i in range(2000)
    ]

    selection = select_navigation_snapshot_events(events, mode="automatic_filtered", target_rows=1500)

    assert selection.exported_count == 2000, "mandatory events must never be thinned out"
    assert selection.semantic_events_preserved == 2000
    assert selection.exported_count > selection.target_rows


# ---------------------------------------------------------------------------
# 9/11. Original indices reach snapshot_rows() unrenumbered; compatibility
# with the pre-existing unfiltered call shape.
# ---------------------------------------------------------------------------


def _make_full_snapshot(
    snapshot_id: int,
    *,
    robot_id: str = "R1",
    simulation_time: float | None = None,
) -> NavigationDebugSnapshot:
    """Minimal-but-COMPLETE NavigationDebugSnapshot -- mirrors the fixture
    already used by test_navigation_debug_event_log.py, needed only for the
    handful of tests that exercise snapshot_rows()/export_navigation_
    snapshots_xlsx() (which read the full contract), not the pure selection
    algorithm above."""
    return NavigationDebugSnapshot(
        snapshot_id=snapshot_id,
        simulation_time=float(snapshot_id if simulation_time is None else simulation_time),
        robot_id=robot_id,
        navigation_state="moving",
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        robot_pose=Pose(x=0.0, y=0.0, theta=0.0, v=0.0),
        path=PathDebug(
            raw_path=Maybe.missing(),
            simplified_path=Maybe.missing(),
            active_path=(),
            pending_path=(),
            active_segment=None,
            active_waypoint_index=None,
            planner_name=Maybe.missing(),
            simplifier_name=Maybe.missing(),
        ),
        route=RouteValidationDebug(first_segment=Maybe.missing(), endpoint_reaches_goal=None),
        predicted_motion=PredictedMotionDebug(trajectory=Maybe.missing(), collision=Maybe.missing()),
        safety=SafetyDebug(robot_radius=0.2, safety_radius=0.3, active_segment=Maybe.missing()),
        planning_grid=PlanningGridDebug(
            start_cell=Maybe.missing(),
            start_cell_world=Maybe.missing(),
            first_waypoint_cell=Maybe.missing(),
            first_waypoint_world=Maybe.missing(),
            unknown_is_traversable=Maybe.missing(),
            start_cell_cleared=Maybe.missing(),
        ),
        controller=ControllerDebug(
            v=0.0, omega=0.0, acceleration=0.0, heading_error=Maybe.missing(), distance_to_goal=Maybe.missing()
        ),
        frontier=FrontierDebug(
            candidate_count=Maybe.missing(),
            selected_target=Maybe.missing(),
            selected_score=Maybe.missing(),
            reason=Maybe.missing(),
        ),
    )


def _full_event(snapshot_id: int, *, kind=NavigationDebugEventKind.TICK, **kwargs) -> NavigationDebugEvent:
    return NavigationDebugEvent(event_kind=kind, snapshot=_make_full_snapshot(snapshot_id, **kwargs))


def test_snapshot_rows_uses_original_source_indices_without_renumbering():
    full_events = [_full_event(i) for i in range(5)]
    # Simulate a filtered selection: keep positions 1, 4, 7 (1-based, with
    # gaps) out of a larger original history -- snapshot_rows() must use
    # these exact numbers, not 1..len(events).
    kept_events = (full_events[0], full_events[2], full_events[4])
    original_indices = (1, 4, 7)

    headers, rows = snapshot_rows(kept_events, event_indices=original_indices)

    event_index_col = headers.index("event_index")
    assert [row[event_index_col] for row in rows] == [1, 4, 7]


def test_snapshot_rows_rejects_mismatched_length_event_indices():
    full_events = [_full_event(i) for i in range(3)]
    with pytest.raises(ValueError):
        snapshot_rows(full_events, event_indices=(1, 2))


def test_snapshot_rows_rejects_non_positive_event_indices():
    full_events = [_full_event(0)]
    with pytest.raises(ValueError):
        snapshot_rows(full_events, event_indices=(0,))


def test_export_xlsx_without_new_arguments_keeps_prior_raw_behavior(tmp_path):
    full_events = tuple(_full_event(i) for i in range(6))
    output = tmp_path / "raw_compat.xlsx"

    count = export_navigation_snapshots_xlsx(full_events, output)

    assert count == 6
    assert output.exists()


def test_export_xlsx_raises_snapshot_export_error_for_empty_events(tmp_path):
    with pytest.raises(SnapshotExportError):
        export_navigation_snapshots_xlsx((), tmp_path / "empty.xlsx")


# ---------------------------------------------------------------------------
# 10. Metadata sheet contents for a filtered export.
# ---------------------------------------------------------------------------


def _read_metadata_sheet(xlsx_path) -> dict[str, object]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(xlsx_path) as archive:
        xml_bytes = archive.read("xl/worksheets/sheet2.xml")
    root = ElementTree.fromstring(xml_bytes)
    fields: dict[str, object] = {}
    rows = root.findall(".//m:sheetData/m:row", ns)
    for row in rows[1:]:  # row 1 is the "field"/"value" header
        cells = row.findall("m:c", ns)
        if len(cells) < 2:
            continue
        field_cell, value_cell = cells[0], cells[1]

        def _cell_text(cell):
            inline = cell.find("m:is/m:t", ns)
            if inline is not None:
                return inline.text or ""
            numeric = cell.find("m:v", ns)
            if numeric is not None:
                return numeric.text
            return None

        field_name = _cell_text(field_cell)
        fields[field_name] = _cell_text(value_cell)
    return fields


def test_metadata_sheet_reflects_a_filtered_export(tmp_path):
    full_events = tuple(_full_event(i, robot_id="R1") for i in range(20))
    kept = full_events[::4]  # 5 events: indices 1, 5, 9, 13, 17 (1-based)
    kept_indices = tuple(range(1, 21, 4))
    output = tmp_path / "filtered.xlsx"

    count = export_navigation_snapshots_xlsx(
        kept,
        output,
        source_indices=kept_indices,
        source_count=20,
        export_mode="custom_stride",
        routine_stride=4,
        target_rows=None,
        semantic_events_preserved=0,
    )

    assert count == 5
    fields = _read_metadata_sheet(output)
    assert fields["source_snapshot_count"] == "20"
    assert fields["exported_snapshot_count"] == "5"
    assert fields["snapshot_count"] == "5"
    assert fields["export_mode"] == "custom_stride"
    assert fields["routine_stride"] == "4"
    assert (
        fields["event_index_note"]
        == "Original 1-based source-history position; gaps indicate export filtering."
    )
