"""
Standalone "Navigation Reasoning" window.

Shows the same full field breakdown that used to be drawn as a fixed card
on top of the simulation canvas -- moved into its own OS window so it can
never sit on top of / obstruct the map, robot, or title bar. Pure consumer:
every value is read straight off the pushed NavigationDebugSnapshot;
nothing here is computed.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget


def _maybe_text(maybe_value, formatter=str) -> str:
    if maybe_value.unavailable:
        return "unavailable"
    return formatter(maybe_value.value)


def _clearance_text(maybe_terms) -> str:
    if maybe_terms.unavailable or maybe_terms.value is None:
        return "unavailable"
    terms = maybe_terms.value
    distance_text = "n/a" if terms.distance.unavailable else f"{terms.distance.value:.2f}"
    status = "BLOCKED" if terms.blocked else "clear"
    return f"{status} (d={distance_text}m, req={terms.required_clearance:.2f}m, {terms.checker})"


class NavigationReasoningWindow(QWidget):
    """Floating, independent top-level window -- not a child docked inside
    the canvas -- so it never overlaps the simulation view. Closing it (the
    window X button) just hides it; the eye icon on the canvas is what
    re-opens it."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Navigation Reasoning")
        self.resize(460, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        self._label = QLabel()
        self._label.setFont(QFont("Consolas", 10))
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._label.setWordWrap(True)
        self._label.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(self._label)

        self.set_no_snapshot()

    def set_no_snapshot(self) -> None:
        self._label.setText("No navigation decisions captured yet.")

    def update_snapshot(self, snapshot, last_event, history_position) -> None:
        if snapshot is None:
            self.set_no_snapshot()
            return

        position, total = history_position
        view_text = f"HISTORY {position}/{total}" if position is not None else ("LIVE" if total else "no history yet")

        c = snapshot.controller
        theta_t = "n/a" if c.desired_heading.unavailable else f"{math.degrees(c.desired_heading.value):.1f}°"
        e_theta = "n/a" if c.heading_error.unavailable else f"{math.degrees(c.heading_error.value):.1f}°"
        rotate_thr = (
            "n/a" if snapshot.rotate_threshold.unavailable else f"{math.degrees(snapshot.rotate_threshold.value):.1f}°"
        )
        nominal = "n/a / n/a" if c.nominal_control.unavailable else f"{c.nominal_control.value[0]:.2f} / {c.nominal_control.value[1]:.2f}"
        applied = "n/a / n/a" if c.applied_control.unavailable else f"{c.applied_control.value[0]:.2f} / {c.applied_control.value[1]:.2f}"

        last_event_text = (
            f"{last_event.event_kind.value} @ t={last_event.snapshot.simulation_time:.2f}s -- {last_event.snapshot.decision_reason or '-'}"
            if last_event is not None
            else "(none yet)"
        )

        lines = [
            "<b>NAVIGATION REASONING</b>",
            f"Robot: {snapshot.robot_id}&nbsp;&nbsp;&nbsp;{view_text}",
            f"Time: {snapshot.simulation_time:.2f}s&nbsp;&nbsp;&nbsp;snapshot #{snapshot.snapshot_id}",
            f"State: {snapshot.tracking_mode or '-'}&nbsp;&nbsp;&nbsp;Decision: {snapshot.decision_kind}",
            f"Reason: {snapshot.decision_reason or '-'}",
            f"θ={math.degrees(snapshot.robot_pose.theta):.1f}°&nbsp;&nbsp;&nbsp;θt={theta_t}",
            f"eθ={e_theta}&nbsp;&nbsp;&nbsp;rotate_threshold={rotate_thr}",
            f"waypoint#={snapshot.path.active_waypoint_index if snapshot.path.active_waypoint_index is not None else '-'}"
            f"&nbsp;&nbsp;&nbsp;d_goal={_maybe_text(c.distance_to_goal, lambda v: f'{v:.2f}m')}",
            f"v={c.v:.2f}&nbsp;&nbsp;&nbsp;a_nom / ω_nom = {nominal}",
            f"a_cmd / ω_cmd = {applied}",
            f"planner={_maybe_text(snapshot.path.planner_name)}&nbsp;&nbsp;&nbsp;simplifier={_maybe_text(snapshot.path.simplifier_name)}",
            f"route validation: {_clearance_text(snapshot.route.first_segment)}",
            f"safety (active segment): {_clearance_text(snapshot.safety.active_segment)}",
            f"predicted collision: {_clearance_text(snapshot.predicted_motion.collision)}",
            f"radius body={snapshot.safety.robot_radius:.2f}m&nbsp;&nbsp;&nbsp;safety={snapshot.safety.safety_radius:.2f}m",
            f"last event: {last_event_text}",
            f"<span style='color:#236FCF'>{snapshot.explanation or '-'}</span>",
        ]
        self._label.setText("<br>".join(lines))
