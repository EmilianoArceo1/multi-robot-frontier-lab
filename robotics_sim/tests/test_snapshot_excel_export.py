"""
Tests for Team HazardBelief support in the Excel snapshot exporter
(robotics_sim.diagnostics.snapshot_export): the new hazard_belief_* summary
columns on the `Snapshots` sheet and the new `Hazard Belief Cells` detail
sheet.

Ground-truth hazard_* columns (FireSource) are asserted unchanged -- see
test_ground_truth_hazard_columns_unchanged() and test_unobserved_fire_
source_does_not_appear_as_belief_observed(), which prove the two concepts
stay independent.

Workbook content is read back with a small stdlib-only xlsx reader
(zipfile + xml.etree.ElementTree) mirroring exactly the cell encoding
_xml_cell() writes -- no openpyxl/pandas dependency for tests either, same
as the exporter itself.
"""
from __future__ import annotations

import inspect
import json
import re
import zlib
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest

from robotics_sim.diagnostics import snapshot_export
from robotics_sim.diagnostics.event_log import NavigationDebugEvent
from robotics_sim.diagnostics.navigation_snapshot import (
    ControllerDebug,
    FrontierDebug,
    HazardBeliefDebug,
    HazardDebug,
    HazardSourceDebug,
    Maybe,
    NavigationDebugEventKind,
    NavigationDebugSnapshot,
    PathDebug,
    PlanningGridDebug,
    Pose,
    PredictedMotionDebug,
    RouteValidationDebug,
    SafetyDebug,
    SensorDebug,
)
from robotics_sim.diagnostics.snapshot_export import (
    export_navigation_snapshots_xlsx,
    hazard_belief_cell_rows,
    snapshot_rows,
)

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_REL_NS = {"rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
_R_ID_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


# ---------------------------------------------------------------------------
# Minimal stdlib-only xlsx reader -- mirrors _xml_cell()'s own encoding.
# ---------------------------------------------------------------------------


def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group()
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch) - 64)
    return index


def _cell_value(cell_elem):
    cell_type = cell_elem.get("t")
    if cell_type == "inlineStr":
        text_elem = cell_elem.find("m:is/m:t", _NS)
        return text_elem.text if text_elem is not None and text_elem.text is not None else ""
    value_elem = cell_elem.find("m:v", _NS)
    if value_elem is None:
        return None
    if cell_type == "b":
        return value_elem.text == "1"
    text = value_elem.text
    if text is None:
        return None
    try:
        if "." not in text and "e" not in text.lower():
            return int(text)
        return float(text)
    except ValueError:
        return text


def _read_sheet(xlsx_path, sheet_name: str) -> list[dict[str, object]]:
    """Read one sheet back into a list of {header: value} dicts (header row
    is row 1) -- no external xlsx library, just the same XML shape
    _write_sheet() produces."""
    with ZipFile(xlsx_path) as zf:
        workbook_xml = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_xml = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {
            rel.get("Id"): rel.get("Target") for rel in rels_xml.findall("rel:Relationship", _REL_NS)
        }
        sheet_rid = None
        for sheet_elem in workbook_xml.find("m:sheets", _NS):
            if sheet_elem.get("name") == sheet_name:
                sheet_rid = sheet_elem.get(_R_ID_ATTR)
                break
        if sheet_rid is None:
            raise KeyError(f"no sheet named {sheet_name!r}")
        sheet_xml = ET.fromstring(zf.read(f"xl/{rid_to_target[sheet_rid]}"))

        raw_rows: list[list[object]] = []
        sheet_data = sheet_xml.find("m:sheetData", _NS)
        for row_elem in sheet_data:
            row_cells: dict[int, object] = {}
            for cell_elem in row_elem.findall("m:c", _NS):
                row_cells[_col_index(cell_elem.get("r"))] = _cell_value(cell_elem)
            max_col = max(row_cells) if row_cells else 0
            raw_rows.append([row_cells.get(i) for i in range(1, max_col + 1)])

    if not raw_rows:
        return []
    headers = raw_rows[0]
    return [dict(zip(headers, row)) for row in raw_rows[1:]]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _make_hazard_belief_frame(
    values: np.ndarray, observed: np.ndarray, observed_by_robot: np.ndarray, *, revision: int
) -> HazardBeliefDebug:
    values = np.ascontiguousarray(values, dtype=np.float32)
    observed = np.ascontiguousarray(observed, dtype=bool)
    observed_by_robot = np.ascontiguousarray(observed_by_robot, dtype=bool)
    packed_observed = np.packbits(observed.reshape(-1), bitorder="little")
    packed_observed_by_robot = np.packbits(observed_by_robot.reshape(-1), bitorder="little")
    return HazardBeliefDebug(
        shape=(int(values.shape[0]), int(values.shape[1])),
        robot_count=int(observed_by_robot.shape[0]),
        revision=revision,
        values_zlib=zlib.compress(values.tobytes(order="C"), level=1),
        observed_packbits_zlib=zlib.compress(packed_observed.tobytes(), level=1),
        observed_by_robot_packbits_zlib=zlib.compress(packed_observed_by_robot.tobytes(), level=1),
    )


def _make_snapshot(
    *,
    snapshot_id: int,
    simulation_time: float = 1.0,
    hazard_belief_frame: HazardBeliefDebug | None = None,
    hazard_frame: HazardDebug | None = None,
) -> NavigationDebugSnapshot:
    return NavigationDebugSnapshot(
        snapshot_id=snapshot_id,
        simulation_time=simulation_time,
        robot_id="R1",
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
        sensor=SensorDebug(),
        hazard=Maybe.of(hazard_frame) if hazard_frame is not None else Maybe.missing(),
        hazard_belief=Maybe.of(hazard_belief_frame) if hazard_belief_frame is not None else Maybe.missing(),
    )


def _make_event(snapshot: NavigationDebugSnapshot) -> NavigationDebugEvent:
    return NavigationDebugEvent(event_kind=NavigationDebugEventKind.TICK, snapshot=snapshot)


def _row_dict(headers: list[str], row: list[object]) -> dict[str, object]:
    return dict(zip(headers, row))


# ---------------------------------------------------------------------------
# 1-10: hazard_belief_* summary columns on the Snapshots sheet.
# ---------------------------------------------------------------------------


def test_snapshot_without_hazard_belief_exports_available_false():
    snapshot = _make_snapshot(snapshot_id=1)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_belief_available"] is False
    assert row["hazard_belief_revision"] is None
    assert row["hazard_belief_height"] is None
    assert row["hazard_belief_width"] is None
    assert row["hazard_belief_robot_count"] is None
    assert row["hazard_observed_cell_count"] == 0
    assert row["hazard_observed_fraction"] == 0.0
    assert row["hazard_nonzero_observed_cell_count"] == 0
    assert row["hazard_max_observed_value"] == 0.0
    assert row["hazard_mean_observed_value"] == 0.0
    assert row["hazard_belief_decode_error"] == ""


def test_valid_snapshot_exports_revision():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=7)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_belief_available"] is True
    assert row["hazard_belief_revision"] == 7


def test_valid_snapshot_exports_height_and_width():
    shape = (3, 5)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_belief_height"] == 3
    assert row["hazard_belief_width"] == 5


def test_valid_snapshot_exports_robot_count():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    frame = _make_hazard_belief_frame(values, observed, np.zeros((3,) + shape, dtype=bool), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_belief_robot_count"] == 3


def test_observed_cell_count_correct():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    observed[0, 0] = True
    observed[1, 1] = True
    observed[2, 2] = True
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_observed_cell_count"] == 3


def test_observed_fraction_correct():
    shape = (4, 4)  # 16 cells total
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    observed[0, :] = True  # 4 of 16
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_observed_fraction"] == pytest.approx(0.25)


def test_nonzero_count_only_uses_observed_cells():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    values[0, 0] = 0.5  # observed, nonzero
    values[1, 1] = 0.0  # observed, zero
    values[2, 2] = 0.9  # NOT observed -- must not count
    observed = np.zeros(shape, dtype=bool)
    observed[0, 0] = True
    observed[1, 1] = True
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_nonzero_observed_cell_count"] == 1


def test_max_only_uses_observed_cells():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    values[0, 0] = 0.3
    values[1, 1] = 0.6
    values[2, 2] = 0.99  # NOT observed -- must not affect max
    observed = np.zeros(shape, dtype=bool)
    observed[0, 0] = True
    observed[1, 1] = True
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_max_observed_value"] == pytest.approx(0.6)


def test_mean_only_uses_observed_cells():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    values[0, 0] = 0.2
    values[1, 1] = 0.4
    values[2, 2] = 1.0  # NOT observed -- must not affect mean
    observed = np.zeros(shape, dtype=bool)
    observed[0, 0] = True
    observed[1, 1] = True
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_mean_observed_value"] == pytest.approx(0.3)


def test_empty_observed_produces_zero_max_and_mean():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)  # nothing observed at all
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_observed_cell_count"] == 0
    assert row["hazard_max_observed_value"] == 0.0
    assert row["hazard_mean_observed_value"] == 0.0


# ---------------------------------------------------------------------------
# 11-12: corrupt payload never aborts the export.
# ---------------------------------------------------------------------------


def _make_corrupt_frame(*, shape=(4, 4), robot_count: int = 1, revision: int = 1) -> HazardBeliefDebug:
    return HazardBeliefDebug(
        shape=shape,
        robot_count=robot_count,
        revision=revision,
        values_zlib=b"not-valid-zlib-data",
        observed_packbits_zlib=b"not-valid-zlib-data",
        observed_by_robot_packbits_zlib=b"not-valid-zlib-data",
    )


def test_corrupt_payload_does_not_abort_the_export(tmp_path):
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=_make_corrupt_frame())
    output = tmp_path / "export.xlsx"

    count = export_navigation_snapshots_xlsx([_make_event(snapshot)], output)

    assert count == 1
    assert output.exists()


def test_corrupt_payload_fills_decode_error():
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=_make_corrupt_frame())

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_belief_available"] is True
    assert row["hazard_belief_decode_error"] != ""
    assert "values_zlib" in row["hazard_belief_decode_error"]
    assert row["hazard_observed_cell_count"] == 0
    assert row["hazard_max_observed_value"] == 0.0
    assert row["hazard_mean_observed_value"] == 0.0


# ---------------------------------------------------------------------------
# 13-14: ground truth (FireSource) stays independent of the belief.
# ---------------------------------------------------------------------------


def test_ground_truth_hazard_columns_unchanged():
    source = HazardSourceDebug(fire_id=1, position=(0.5, 0.5), intensity=1.0, radius=2.0)
    hazard_frame = HazardDebug(version=3, next_fire_id=2, sources=(source,))
    snapshot = _make_snapshot(snapshot_id=1, hazard_frame=hazard_frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_available"] is True
    assert row["hazard_version"] == 3
    assert row["hazard_next_fire_id"] == 2
    assert row["hazard_source_count"] == 1
    assert json.loads(row["hazard_sources_json"]) == [
        {"fire_id": 1, "x": 0.5, "y": 0.5, "intensity": 1.0, "radius": 2.0}
    ]


def test_unobserved_fire_source_does_not_appear_as_belief_observed():
    source = HazardSourceDebug(fire_id=1, position=(0.5, 0.5), intensity=1.0, radius=2.0)
    hazard_frame = HazardDebug(version=1, next_fire_id=2, sources=(source,))
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)  # the fire was never actually observed
    belief_frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_frame=hazard_frame, hazard_belief_frame=belief_frame)

    headers, rows = snapshot_rows([_make_event(snapshot)])
    detail_headers, detail_rows = hazard_belief_cell_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["hazard_available"] is True  # ground truth IS there
    assert row["hazard_observed_cell_count"] == 0  # but the team never saw it
    assert detail_rows == []


# ---------------------------------------------------------------------------
# 15-19: the "Hazard Belief Cells" detail sheet.
# ---------------------------------------------------------------------------


def test_hazard_belief_cells_sheet_contains_only_observed_true(tmp_path):
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    values[1, 2] = 0.7
    observed = np.zeros(shape, dtype=bool)
    observed[1, 2] = True
    frame = _make_hazard_belief_frame(values, observed, observed.reshape((1,) + shape), revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)
    output = tmp_path / "export.xlsx"

    export_navigation_snapshots_xlsx([_make_event(snapshot)], output)

    detail_rows = _read_sheet(output, "Hazard Belief Cells")
    assert len(detail_rows) == 1
    assert detail_rows[0]["row"] == 1
    assert detail_rows[0]["col"] == 2
    assert detail_rows[0]["value"] == pytest.approx(0.7)


def test_detail_rows_are_ordered_by_snapshot_then_row_then_col():
    shape = (4, 4)
    values_a = np.zeros(shape, dtype=np.float32)
    observed_a = np.zeros(shape, dtype=bool)
    observed_a[2, 1] = True
    observed_a[0, 3] = True
    observed_a[2, 0] = True
    frame_a = _make_hazard_belief_frame(values_a, observed_a, observed_a.reshape((1,) + shape), revision=1)
    snapshot_a = _make_snapshot(snapshot_id=10, simulation_time=1.0, hazard_belief_frame=frame_a)

    values_b = np.zeros(shape, dtype=np.float32)
    observed_b = np.zeros(shape, dtype=bool)
    observed_b[0, 0] = True
    frame_b = _make_hazard_belief_frame(values_b, observed_b, observed_b.reshape((1,) + shape), revision=2)
    snapshot_b = _make_snapshot(snapshot_id=11, simulation_time=2.0, hazard_belief_frame=frame_b)

    headers, rows = hazard_belief_cell_rows([_make_event(snapshot_a), _make_event(snapshot_b)])

    triples = [
        (_row_dict(headers, r)["snapshot_id"], _row_dict(headers, r)["row"], _row_dict(headers, r)["col"])
        for r in rows
    ]
    assert triples == [(10, 0, 3), (10, 2, 0), (10, 2, 1), (11, 0, 0)]


def test_observed_by_robots_exported_sorted():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    observed[1, 1] = True
    observed_by_robot = np.zeros((3,) + shape, dtype=bool)
    observed_by_robot[2, 1, 1] = True  # attributed out of order: 2 then 0
    observed_by_robot[0, 1, 1] = True
    frame = _make_hazard_belief_frame(values, observed, observed_by_robot, revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = hazard_belief_cell_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert row["observed_by_robots"] == "[0,2]"


def test_two_robots_attributed_to_the_same_cell_are_both_preserved():
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=bool)
    observed[1, 1] = True
    observed_by_robot = np.zeros((2,) + shape, dtype=bool)
    observed_by_robot[0, 1, 1] = True
    observed_by_robot[1, 1, 1] = True
    frame = _make_hazard_belief_frame(values, observed, observed_by_robot, revision=1)
    snapshot = _make_snapshot(snapshot_id=1, hazard_belief_frame=frame)

    headers, rows = hazard_belief_cell_rows([_make_event(snapshot)])

    row = _row_dict(headers, rows[0])
    assert json.loads(row["observed_by_robots"]) == [0, 1]


def test_old_snapshot_without_hazard_belief_creates_no_detail_rows():
    snapshot = _make_snapshot(snapshot_id=1)  # no hazard_belief_frame at all

    headers, rows = hazard_belief_cell_rows([_make_event(snapshot)])

    assert rows == []


# ---------------------------------------------------------------------------
# 20: determinism across two exports of the same events.
# ---------------------------------------------------------------------------


def test_exporting_twice_produces_equivalent_content(tmp_path):
    shape = (4, 4)
    values = np.zeros(shape, dtype=np.float32)
    values[1, 1] = 0.5
    observed = np.zeros(shape, dtype=bool)
    observed[1, 1] = True
    observed[2, 2] = True
    observed_by_robot = np.zeros((2,) + shape, dtype=bool)
    observed_by_robot[0, 1, 1] = True
    observed_by_robot[1, 2, 2] = True
    frame = _make_hazard_belief_frame(values, observed, observed_by_robot, revision=1)
    events = [
        _make_event(_make_snapshot(snapshot_id=1, simulation_time=1.0, hazard_belief_frame=frame)),
        _make_event(_make_snapshot(snapshot_id=2, simulation_time=2.0)),
    ]

    output_a = tmp_path / "a.xlsx"
    output_b = tmp_path / "b.xlsx"
    export_navigation_snapshots_xlsx(events, output_a)
    export_navigation_snapshots_xlsx(events, output_b)

    # Snapshots/Hazard Belief Cells are pure functions of `events` -- unlike
    # Metadata (exported_at_utc), neither has any wall-clock content.
    assert _read_sheet(output_a, "Snapshots") == _read_sheet(output_b, "Snapshots")
    assert _read_sheet(output_a, "Hazard Belief Cells") == _read_sheet(output_b, "Hazard Belief Cells")


# ---------------------------------------------------------------------------
# 21-22: architectural boundaries (Qt-free, no live-state access).
# ---------------------------------------------------------------------------


def test_exporter_module_never_imports_qt():
    source = inspect.getsource(snapshot_export)
    for forbidden in ("PySide6", "PyQt", "QtCore", "QtWidgets", "QtGui"):
        assert forbidden not in source


def test_hazard_belief_decode_never_references_engine_or_live_service():
    combined = (
        inspect.getsource(snapshot_export._decode_hazard_belief)
        + inspect.getsource(snapshot_export._hazard_belief_summary)
        + inspect.getsource(snapshot_export.hazard_belief_cell_rows)
    )
    # Checked as actual code patterns (constructor/attribute access), not
    # bare substrings -- these functions' own docstrings mention some of
    # these names in prose (explaining what must NOT happen) without
    # violating the rule.
    for forbidden in ("RuntimeHazardService(", "HazardField(", "hazard_service.", "self.canvas", "FireSource("):
        assert forbidden not in combined, (
            f"hazard-belief export code must never contain {forbidden!r} -- the exported "
            "NavigationDebugSnapshot must be the only source of truth"
        )
