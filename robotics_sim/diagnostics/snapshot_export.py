"""Excel export for the in-memory navigation snapshot history.

This module is deliberately Qt-free and dependency-free.  It writes a small
OOXML workbook directly with the Python standard library, so exporting does not
require pandas/openpyxl/xlsxwriter on the user's machine.

The `Snapshots` sheet contains exactly one row per NavigationDebugEvent.  Large
or nested values (paths and hazard sources) are represented as compact JSON so
the workbook stays flat and easy to analyze with filters, formulas, Python, R,
or another ChatGPT session.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import zlib
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from robotics_sim.diagnostics.event_log import NavigationDebugEvent
from robotics_sim.environment.belief_map import FREE, OCCUPIED, UNKNOWN

SCHEMA_VERSION = "1.0"
_EXCEL_TEXT_LIMIT = 32767


class SnapshotExportError(RuntimeError):
    """Raised when a snapshot workbook cannot be produced."""


def _enum_value(value):
    return getattr(value, "value", value)


def _maybe_value(maybe):
    if maybe is None or bool(getattr(maybe, "unavailable", False)):
        return None
    return getattr(maybe, "value", None)


def _point_xy(point) -> tuple[float | None, float | None]:
    if point is None:
        return None, None
    try:
        return float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None, None


def _grid_cell_rc(cell) -> tuple[int | None, int | None]:
    if cell is None:
        return None, None
    row = getattr(cell, "row", None)
    col = getattr(cell, "col", None)
    if row is None or col is None:
        try:
            row, col = cell
        except (TypeError, ValueError):
            return None, None
    return int(row), int(col)


def _json_value(value) -> str:
    if value is None:
        return ""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _path_json(path) -> str:
    return _json_value([[float(point[0]), float(point[1])] for point in (path or ())])


def _clean_text(value: object) -> str:
    text = str(value)
    # XML 1.0 forbids most ASCII control characters.
    text = "".join(ch for ch in text if ch in "\t\n\r" or ord(ch) >= 32)
    if len(text) > _EXCEL_TEXT_LIMIT:
        text = text[: _EXCEL_TEXT_LIMIT - 14] + "…[truncated]"
    return text


def _clearance_values(clearance) -> dict[str, object]:
    if clearance is None:
        return {
            "available": False,
            "checker": None,
            "distance": None,
            "required_clearance": None,
            "blocked": None,
            "blocking_x": None,
            "blocking_y": None,
            "reason": None,
        }
    blocking_x, blocking_y = _point_xy(getattr(clearance, "blocking_point", None))
    return {
        "available": True,
        "checker": str(getattr(clearance, "checker", "")),
        "distance": _maybe_value(getattr(clearance, "distance", None)),
        "required_clearance": float(getattr(clearance, "required_clearance", 0.0)),
        "blocked": bool(getattr(clearance, "blocked", False)),
        "blocking_x": blocking_x,
        "blocking_y": blocking_y,
        "reason": str(getattr(clearance, "reason", "")),
    }


def _belief_values(frame, cache: dict[int, dict[str, object]]) -> dict[str, object]:
    if frame is None:
        return {
            "available": False,
            "revision": None,
            "resolution": None,
            "bounds_min_x": None,
            "bounds_max_x": None,
            "bounds_min_y": None,
            "bounds_max_y": None,
            "rows": None,
            "cols": None,
            "unknown_cells": None,
            "free_cells": None,
            "occupied_cells": None,
            "known_cells": None,
            "explored_cells": None,
            "explored_coverage_pct": None,
            "explored_by_robot_json": "",
            "grid_sha256": "",
            "explored_sha256": "",
        }

    key = id(frame)
    cached = cache.get(key)
    if cached is not None:
        return cached

    grid_bytes = zlib.decompress(frame.grid_zlib)
    grid = np.frombuffer(grid_bytes, dtype=np.int8).reshape(frame.grid_shape)
    packed_explored = zlib.decompress(frame.explored_packbits_zlib)
    explored = np.unpackbits(
        np.frombuffer(packed_explored, dtype=np.uint8),
        bitorder="little",
        count=int(np.prod(frame.explored_shape)),
    ).reshape(frame.explored_shape).astype(bool)

    unknown_cells = int(np.count_nonzero(grid == UNKNOWN))
    free_cells = int(np.count_nonzero(grid == FREE))
    occupied_cells = int(np.count_nonzero(grid == OCCUPIED))
    known_cells = int(grid.size - unknown_cells)
    explored_union = explored.any(axis=0) if explored.ndim == 3 else explored.astype(bool)
    explored_cells = int(np.count_nonzero(explored_union))
    robot_counts = [int(np.count_nonzero(explored[index])) for index in range(explored.shape[0])]
    min_x, max_x, min_y, max_y = (float(v) for v in frame.bounds)

    cached = {
        "available": True,
        "revision": int(frame.revision),
        "resolution": float(frame.resolution),
        "bounds_min_x": min_x,
        "bounds_max_x": max_x,
        "bounds_min_y": min_y,
        "bounds_max_y": max_y,
        "rows": int(frame.grid_shape[0]),
        "cols": int(frame.grid_shape[1]),
        "unknown_cells": unknown_cells,
        "free_cells": free_cells,
        "occupied_cells": occupied_cells,
        "known_cells": known_cells,
        "explored_cells": explored_cells,
        "explored_coverage_pct": (100.0 * explored_cells / float(grid.size)) if grid.size else 0.0,
        "explored_by_robot_json": _json_value(robot_counts),
        "grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
        "explored_sha256": hashlib.sha256(packed_explored).hexdigest(),
    }
    cache[key] = cached
    return cached


def snapshot_row(event: NavigationDebugEvent, event_index: int, *, belief_cache=None) -> dict[str, object]:
    """Flatten one event into a stable, analysis-friendly row."""
    belief_cache = belief_cache if belief_cache is not None else {}
    snapshot = event.snapshot
    path = snapshot.path
    controller = snapshot.controller

    heading_error = _maybe_value(controller.heading_error)
    desired_heading = _maybe_value(controller.desired_heading)
    nominal_control = _maybe_value(controller.nominal_control)
    applied_control = _maybe_value(controller.applied_control)
    raw_path = _maybe_value(path.raw_path)
    simplified_path = _maybe_value(path.simplified_path)

    active_segment_start = active_segment_end = None
    if path.active_segment is not None:
        active_segment_start, active_segment_end = path.active_segment
    seg_start_x, seg_start_y = _point_xy(active_segment_start)
    seg_end_x, seg_end_y = _point_xy(active_segment_end)

    route_clearance = _clearance_values(_maybe_value(snapshot.route.first_segment))
    predicted_clearance = _clearance_values(_maybe_value(snapshot.predicted_motion.collision))
    safety_clearance = _clearance_values(_maybe_value(snapshot.safety.active_segment))

    planning = snapshot.planning_grid
    start_cell_row, start_cell_col = _grid_cell_rc(_maybe_value(planning.start_cell))
    first_cell_row, first_cell_col = _grid_cell_rc(_maybe_value(planning.first_waypoint_cell))
    start_world_x, start_world_y = _point_xy(_maybe_value(planning.start_cell_world))
    first_world_x, first_world_y = _point_xy(_maybe_value(planning.first_waypoint_world))

    frontier = snapshot.frontier
    frontier_target_x, frontier_target_y = _point_xy(_maybe_value(frontier.selected_target))

    belief_frame = _maybe_value(snapshot.belief_map)
    belief = _belief_values(belief_frame, belief_cache)

    hazard = _maybe_value(snapshot.hazard)
    hazard_sources = []
    if hazard is not None:
        hazard_sources = [
            {
                "fire_id": int(source.fire_id),
                "x": float(source.position[0]),
                "y": float(source.position[1]),
                "intensity": float(source.intensity),
                "radius": float(source.radius),
            }
            for source in hazard.sources
        ]

    agent = _maybe_value(snapshot.agent_state)
    metrics = _maybe_value(snapshot.metrics)
    final_goal_x, final_goal_y = _point_xy(getattr(agent, "final_goal_xy", None))
    exploration_target_x, exploration_target_y = _point_xy(getattr(agent, "exploration_target_xy", None))
    active_goal_x, active_goal_y = _point_xy(getattr(agent, "active_path_goal_xy", None))

    pose = snapshot.robot_pose
    sensor = snapshot.sensor
    predicted_trajectory = _maybe_value(snapshot.predicted_motion.trajectory)

    row = {
        "event_index": int(event_index),
        "snapshot_id": int(snapshot.snapshot_id),
        "event_kind": str(_enum_value(event.event_kind)),
        "simulation_time_s": float(snapshot.simulation_time),
        "robot_id": str(snapshot.robot_id),
        "navigation_state": str(snapshot.navigation_state),
        "tracking_mode": str(snapshot.tracking_mode),
        "decision_kind": str(snapshot.decision_kind),
        "decision_reason": str(snapshot.decision_reason),
        "explanation": str(snapshot.explanation),
        "rotate_threshold_rad": _maybe_value(snapshot.rotate_threshold),
        "rotate_threshold_deg": math.degrees(_maybe_value(snapshot.rotate_threshold)) if _maybe_value(snapshot.rotate_threshold) is not None else None,
        "pose_x": float(pose.x),
        "pose_y": float(pose.y),
        "pose_theta_rad": float(pose.theta),
        "pose_theta_deg": math.degrees(float(pose.theta)),
        "pose_velocity": float(pose.v),
        "controller_v": float(controller.v),
        "controller_omega": float(controller.omega),
        "controller_acceleration": float(controller.acceleration),
        "heading_error_rad": heading_error,
        "heading_error_deg": math.degrees(heading_error) if heading_error is not None else None,
        "distance_to_goal": _maybe_value(controller.distance_to_goal),
        "desired_heading_rad": desired_heading,
        "desired_heading_deg": math.degrees(desired_heading) if desired_heading is not None else None,
        "nominal_acceleration": nominal_control[0] if nominal_control else None,
        "nominal_omega": nominal_control[1] if nominal_control else None,
        "applied_acceleration": applied_control[0] if applied_control else None,
        "applied_omega": applied_control[1] if applied_control else None,
        "planner_name": _maybe_value(path.planner_name),
        "simplifier_name": _maybe_value(path.simplifier_name),
        "active_waypoint_index": path.active_waypoint_index,
        "raw_path_count": len(raw_path or ()),
        "simplified_path_count": len(simplified_path or ()),
        "active_path_count": len(path.active_path),
        "pending_path_count": len(path.pending_path),
        "active_segment_start_x": seg_start_x,
        "active_segment_start_y": seg_start_y,
        "active_segment_end_x": seg_end_x,
        "active_segment_end_y": seg_end_y,
        "raw_path_json": _path_json(raw_path),
        "simplified_path_json": _path_json(simplified_path),
        "active_path_json": _path_json(path.active_path),
        "pending_path_json": _path_json(path.pending_path),
        "route_first_segment_available": route_clearance["available"],
        "route_first_segment_checker": route_clearance["checker"],
        "route_first_segment_distance": route_clearance["distance"],
        "route_first_segment_required_clearance": route_clearance["required_clearance"],
        "route_first_segment_blocked": route_clearance["blocked"],
        "route_first_segment_blocking_x": route_clearance["blocking_x"],
        "route_first_segment_blocking_y": route_clearance["blocking_y"],
        "route_first_segment_reason": route_clearance["reason"],
        "route_endpoint_reaches_goal": snapshot.route.endpoint_reaches_goal,
        "predicted_trajectory_count": len(predicted_trajectory or ()),
        "predicted_trajectory_json": _path_json(predicted_trajectory),
        "predicted_collision_available": predicted_clearance["available"],
        "predicted_collision_checker": predicted_clearance["checker"],
        "predicted_collision_distance": predicted_clearance["distance"],
        "predicted_collision_required_clearance": predicted_clearance["required_clearance"],
        "predicted_collision_blocked": predicted_clearance["blocked"],
        "predicted_collision_blocking_x": predicted_clearance["blocking_x"],
        "predicted_collision_blocking_y": predicted_clearance["blocking_y"],
        "predicted_collision_reason": predicted_clearance["reason"],
        "robot_radius": float(snapshot.safety.robot_radius),
        "safety_radius": float(snapshot.safety.safety_radius),
        "safety_active_segment_available": safety_clearance["available"],
        "safety_active_segment_checker": safety_clearance["checker"],
        "safety_active_segment_distance": safety_clearance["distance"],
        "safety_active_segment_required_clearance": safety_clearance["required_clearance"],
        "safety_active_segment_blocked": safety_clearance["blocked"],
        "safety_active_segment_blocking_x": safety_clearance["blocking_x"],
        "safety_active_segment_blocking_y": safety_clearance["blocking_y"],
        "safety_active_segment_reason": safety_clearance["reason"],
        "planning_start_cell_row": start_cell_row,
        "planning_start_cell_col": start_cell_col,
        "planning_start_world_x": start_world_x,
        "planning_start_world_y": start_world_y,
        "planning_first_waypoint_cell_row": first_cell_row,
        "planning_first_waypoint_cell_col": first_cell_col,
        "planning_first_waypoint_world_x": first_world_x,
        "planning_first_waypoint_world_y": first_world_y,
        "planning_unknown_is_traversable": _maybe_value(planning.unknown_is_traversable),
        "planning_start_cell_cleared": _maybe_value(planning.start_cell_cleared),
        "frontier_candidate_count": _maybe_value(frontier.candidate_count),
        "frontier_selected_target_x": frontier_target_x,
        "frontier_selected_target_y": frontier_target_y,
        "frontier_selected_score": _maybe_value(frontier.selected_score),
        "frontier_reason": _maybe_value(frontier.reason),
        "mapped_obstacle_points_count": int(snapshot.mapped_obstacle_points_count),
        "vision_range": float(sensor.vision_range),
        "visible_polygon_count": int(sensor.visible_polygon_count),
        "belief_available": belief["available"],
        "belief_revision": belief["revision"],
        "belief_resolution": belief["resolution"],
        "belief_bounds_min_x": belief["bounds_min_x"],
        "belief_bounds_max_x": belief["bounds_max_x"],
        "belief_bounds_min_y": belief["bounds_min_y"],
        "belief_bounds_max_y": belief["bounds_max_y"],
        "belief_rows": belief["rows"],
        "belief_cols": belief["cols"],
        "belief_unknown_cells": belief["unknown_cells"],
        "belief_free_cells": belief["free_cells"],
        "belief_occupied_cells": belief["occupied_cells"],
        "belief_known_cells": belief["known_cells"],
        "belief_explored_cells": belief["explored_cells"],
        "belief_explored_coverage_pct": belief["explored_coverage_pct"],
        "belief_explored_by_robot_json": belief["explored_by_robot_json"],
        "belief_grid_sha256": belief["grid_sha256"],
        "belief_explored_sha256": belief["explored_sha256"],
        "hazard_available": hazard is not None,
        "hazard_version": int(hazard.version) if hazard is not None else None,
        "hazard_next_fire_id": int(hazard.next_fire_id) if hazard is not None else None,
        "hazard_source_count": len(hazard_sources),
        "hazard_sources_json": _json_value(hazard_sources),
        "agent_state_available": agent is not None,
        "final_goal_x": final_goal_x,
        "final_goal_y": final_goal_y,
        "exploration_target_x": exploration_target_x,
        "exploration_target_y": exploration_target_y,
        "active_path_goal_x": active_goal_x,
        "active_path_goal_y": active_goal_y,
        "active_path_mode": getattr(agent, "active_path_mode", None),
        "route_generation": getattr(agent, "route_generation", None),
        "route_affected_replan_count": getattr(agent, "route_affected_replan_count", None),
        "first_segment_blocked_count": getattr(agent, "first_segment_blocked_count", None),
        "last_frontier_candidate_count": getattr(agent, "last_frontier_candidate_count", None),
        "prefetch_success_count": getattr(agent, "prefetch_success_count", None),
        "prefetch_fail_count": getattr(agent, "prefetch_fail_count", None),
        "agent_safety_replan_count": getattr(agent, "safety_replan_count", None),
        "target_switch_count": getattr(agent, "target_switch_count", None),
        "metrics_available": metrics is not None,
        "total_distance_traveled": getattr(metrics, "total_distance_traveled", None),
        "route_request_count": getattr(metrics, "route_request_count", None),
        "route_result_count": getattr(metrics, "route_result_count", None),
        "route_failure_count": getattr(metrics, "route_failure_count", None),
        "sensor_update_count": getattr(metrics, "sensor_update_count", None),
        "mapping_update_count": getattr(metrics, "mapping_update_count", None),
        "metrics_safety_replan_count": getattr(metrics, "safety_replan_count", None),
        "exploration_replan_count": getattr(metrics, "exploration_replan_count", None),
        "planner_jobs_started": getattr(metrics, "planner_jobs_started", None),
        "planner_jobs_completed": getattr(metrics, "planner_jobs_completed", None),
    }
    return row


def snapshot_rows(events) -> tuple[list[str], list[list[object]]]:
    """Return stable headers and one flat row per event."""
    events = tuple(events)
    belief_cache: dict[int, dict[str, object]] = {}
    dict_rows = [
        snapshot_row(event, index, belief_cache=belief_cache)
        for index, event in enumerate(events, start=1)
    ]
    if not dict_rows:
        # Keep a deterministic schema even for an empty history.
        return list(snapshot_row.__annotations__.keys())[:0], []
    headers = list(dict_rows[0].keys())
    return headers, [[row.get(header) for header in headers] for row in dict_rows]


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _xml_cell(ref: str, value: object, *, style: int = 0) -> str:
    style_attr = f' s="{style}"' if style else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style_attr}/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}" t="b"{style_attr}><v>{1 if bool(value) else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            if isinstance(value, (int, np.integer)):
                text = str(int(value))
            else:
                text = repr(numeric)
            return f'<c r="{ref}"{style_attr}><v>{text}</v></c>'
    text = escape(_clean_text(value))
    preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t{preserve}>{text}</t></is></c>'


def _column_width(header: str) -> float:
    lowered = header.lower()
    if lowered.endswith("_json"):
        return 46.0
    if any(token in lowered for token in ("reason", "explanation", "sha256")):
        return 34.0
    if any(token in lowered for token in ("time", "count", "index", "available", "blocked")):
        return 14.0
    return min(24.0, max(12.0, len(header) + 2.0))


def _write_sheet(zip_file: ZipFile, path: str, headers: list[str], rows: list[list[object]]) -> None:
    last_col = _column_name(max(1, len(headers)))
    last_row = max(1, len(rows) + 1)
    with zip_file.open(path, "w") as handle:
        def write(text: str) -> None:
            handle.write(text.encode("utf-8"))

        write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
        write('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">')
        write('<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>')
        write('<sheetFormatPr defaultRowHeight="15"/>')
        if headers:
            write('<cols>')
            for idx, header in enumerate(headers, start=1):
                write(f'<col min="{idx}" max="{idx}" width="{_column_width(header):.1f}" customWidth="1"/>')
            write('</cols>')
        write('<sheetData>')
        if headers:
            write('<row r="1" ht="24" customHeight="1">')
            for col_index, header in enumerate(headers, start=1):
                write(_xml_cell(f'{_column_name(col_index)}1', header, style=1))
            write('</row>')
        for row_index, row in enumerate(rows, start=2):
            write(f'<row r="{row_index}">')
            for col_index, value in enumerate(row, start=1):
                write(_xml_cell(f'{_column_name(col_index)}{row_index}', value))
            write('</row>')
        write('</sheetData>')
        if headers:
            write(f'<autoFilter ref="A1:{last_col}{last_row}"/>')
        write('</worksheet>')


def export_navigation_snapshots_xlsx(events, output_path: str | os.PathLike[str]) -> int:
    """Write one row per event and return the number of exported snapshots."""
    events = tuple(events)
    if not events:
        raise SnapshotExportError("There are no navigation snapshots to export.")

    output = Path(output_path)
    if output.suffix.lower() != ".xlsx":
        output = output.with_suffix(".xlsx")
    output.parent.mkdir(parents=True, exist_ok=True)

    headers, rows = snapshot_rows(events)
    metadata_headers = ["field", "value"]
    metadata_rows = [
        ["schema_version", SCHEMA_VERSION],
        ["exported_at_utc", datetime.now(timezone.utc).isoformat()],
        ["snapshot_count", len(events)],
        ["first_simulation_time_s", float(events[0].snapshot.simulation_time)],
        ["last_simulation_time_s", float(events[-1].snapshot.simulation_time)],
        ["note", "Each row in Snapshots is one immutable NavigationDebugEvent/NavigationDebugSnapshot."],
    ]

    try:
        with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as workbook:
            workbook.writestr(
                "[Content_Types].xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
                '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                '</Types>',
            )
            workbook.writestr(
                "_rels/.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                '</Relationships>',
            )
            workbook.writestr(
                "xl/workbook.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                '<sheets><sheet name="Snapshots" sheetId="1" r:id="rId1"/>'
                '<sheet name="Metadata" sheetId="2" r:id="rId2"/></sheets></workbook>',
            )
            workbook.writestr(
                "xl/_rels/workbook.xml.rels",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
                '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
                '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                '</Relationships>',
            )
            workbook.writestr(
                "xl/styles.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                '<fonts count="2"><font><sz val="10"/><name val="Aptos"/></font>'
                '<font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font></fonts>'
                '<fills count="3"><fill><patternFill patternType="none"/></fill>'
                '<fill><patternFill patternType="gray125"/></fill>'
                '<fill><patternFill patternType="solid"><fgColor rgb="FF500000"/><bgColor indexed="64"/></patternFill></fill></fills>'
                '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
                '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf></cellXfs>'
                '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
                '</styleSheet>',
            )
            created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            workbook.writestr(
                "docProps/core.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                '<dc:creator>Robotics Simulation Lab</dc:creator><dc:title>Navigation snapshots</dc:title>'
                f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created></cp:coreProperties>',
            )
            workbook.writestr(
                "docProps/app.xml",
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
                'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
                '<Application>Robotics Simulation Lab</Application></Properties>',
            )
            _write_sheet(workbook, "xl/worksheets/sheet1.xml", headers, rows)
            _write_sheet(workbook, "xl/worksheets/sheet2.xml", metadata_headers, metadata_rows)
    except (OSError, ValueError, TypeError, zlib.error) as exc:
        raise SnapshotExportError(str(exc)) from exc

    return len(events)
