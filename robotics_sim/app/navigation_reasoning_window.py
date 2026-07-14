"""Embedded Navigation Reasoning panel.

The panel is a pure consumer of immutable NavigationDebugSnapshot objects. It
never computes planner/controller values and never mutates simulation state.

The widget deliberately uses native labels and small information cards instead
of one HTML blob. This keeps the content readable inside the narrow side bar,
allows long reasons to wrap, and avoids inheriting a dark viewport background
from the application-wide stylesheet.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from robotics_sim.simulation.config import (
    BLUE,
    BLUE_LIGHT,
    BORDER,
    BORDER_SOFT,
    CARD,
    GREEN,
    GREEN_LIGHT,
    MAROON,
    ORANGE,
    PANEL_CARD,
    TEXT,
    TEXT_MUTED,
)


def _maybe_text(maybe_value, formatter=str) -> str:
    if maybe_value.unavailable:
        return "Unavailable"
    return formatter(maybe_value.value)


def _clearance_text(maybe_terms) -> str:
    if maybe_terms.unavailable or maybe_terms.value is None:
        return "Unavailable"
    terms = maybe_terms.value
    distance_text = "n/a" if terms.distance.unavailable else f"{terms.distance.value:.2f} m"
    status = "Blocked" if terms.blocked else "Clear"
    reason = f" — {terms.reason}" if getattr(terms, "reason", "") else ""
    return (
        f"{status} · d={distance_text} · required={terms.required_clearance:.2f} m"
        f" · {terms.checker}{reason}"
    )


class _InfoSection(QFrame):
    """Small two-column information card used by the reasoning panel."""

    def __init__(self, title: str, fields: tuple[tuple[str, str], ...], parent=None):
        super().__init__(parent)
        self.setObjectName("reasoningSection")
        self.values: dict[str, QLabel] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 11)
        root.setSpacing(8)

        heading = QLabel(title)
        heading.setObjectName("reasoningSectionTitle")
        root.addWidget(heading)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)

        for row, (key, caption) in enumerate(fields):
            label = QLabel(caption)
            label.setObjectName("reasoningFieldName")
            label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

            value = QLabel("—")
            value.setObjectName("reasoningFieldValue")
            value.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            grid.addWidget(label, row, 0)
            grid.addWidget(value, row, 1)
            self.values[key] = value

        root.addLayout(grid)

    def set_value(self, key: str, value: object) -> None:
        label = self.values.get(key)
        if label is not None:
            label.setText("—" if value is None or value == "" else str(value))


class NavigationReasoningWindow(QFrame):
    """Dockable, full-height Navigation Reasoning panel.

    The historical class name is kept to avoid breaking imports. The widget is
    embedded as a tab in the right-side panel deck; it is not a top-level OS
    window and it no longer competes vertically with the configuration panel.
    """

    closeRequested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("navigationReasoningPanel")
        self.setMinimumWidth(320)
        self.setStyleSheet(
            f"""
            QFrame#navigationReasoningPanel {{
                background: {CARD};
                border: none;
            }}
            QLabel {{
                color: {TEXT};
                background: transparent;
            }}
            QLabel#reasoningTitle {{
                color: {TEXT};
                font-size: 15px;
                font-weight: 900;
            }}
            QLabel#reasoningSubtitle {{
                color: {TEXT_MUTED};
                font-size: 10px;
                font-weight: 600;
            }}
            QLabel#reasoningViewBadge {{
                color: {BLUE};
                background: {BLUE_LIGHT};
                border: 1px solid #B9D5FA;
                border-radius: 9px;
                padding: 3px 8px;
                font-size: 9px;
                font-weight: 900;
            }}
            QFrame#reasoningSummaryCard {{
                background: #F8FAFD;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QLabel#reasoningDecisionBadge {{
                color: {GREEN};
                background: {GREEN_LIGHT};
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: 900;
            }}
            QLabel#reasoningExplanation {{
                color: {TEXT};
                font-size: 11px;
                font-weight: 800;
            }}
            QLabel#reasoningMeta {{
                color: {TEXT_MUTED};
                font-size: 9px;
                font-weight: 700;
            }}
            QFrame#reasoningSection {{
                background: {PANEL_CARD};
                border: 1px solid {BORDER_SOFT};
                border-radius: 9px;
            }}
            QLabel#reasoningSectionTitle {{
                color: {MAROON};
                font-size: 10px;
                font-weight: 900;
            }}
            QLabel#reasoningFieldName {{
                color: {TEXT_MUTED};
                font-size: 9px;
                font-weight: 800;
                min-width: 82px;
            }}
            QLabel#reasoningFieldValue {{
                color: {TEXT};
                font-family: Consolas, "Courier New", monospace;
                font-size: 9px;
                font-weight: 650;
            }}
            QFrame#reasoningPlaceholder {{
                background: #F8FAFD;
                border: 1px dashed {BORDER};
                border-radius: 11px;
            }}
            QLabel#reasoningPlaceholderTitle {{
                color: {TEXT};
                font-size: 12px;
                font-weight: 900;
            }}
            QLabel#reasoningPlaceholderBody {{
                color: {TEXT_MUTED};
                font-size: 10px;
                font-weight: 600;
            }}
            QFrame#reasoningFooter {{
                background: #F8F9FB;
                border-top: 1px solid {BORDER_SOFT};
            }}
            QLabel#reasoningHistoryLabel {{
                color: {TEXT_MUTED};
                font-size: 9px;
                font-weight: 800;
            }}
            QPushButton#historyStepButton {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: 6px;
                color: {TEXT};
                font-size: 13px;
                font-weight: 900;
            }}
            QPushButton#historyStepButton:hover:enabled {{
                border-color: {BLUE};
                background: {BLUE_LIGHT};
                color: {BLUE};
            }}
            QPushButton#historyStepButton:disabled {{
                color: rgba(90,100,110,0.30);
                background: #F2F3F5;
            }}
            QPushButton#panelCloseButton {{
                border: none;
                background: transparent;
                color: {TEXT_MUTED};
                font-size: 18px;
                font-weight: 700;
            }}
            QPushButton#panelCloseButton:hover {{
                color: {TEXT};
                background: rgba(20,25,35,0.06);
                border-radius: 5px;
            }}
            QScrollArea#navigationReasoningScroll {{
                background: #F4F6F9;
                border: none;
            }}
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header_widget = QWidget(self)
        header_widget.setObjectName("reasoningHeader")
        header = QHBoxLayout(header_widget)
        header.setContentsMargins(14, 12, 10, 10)
        header.setSpacing(8)

        title_stack = QVBoxLayout()
        title_stack.setContentsMargins(0, 0, 0, 0)
        title_stack.setSpacing(1)
        self.title_label = QLabel("Navigation Reasoning")
        self.title_label.setObjectName("reasoningTitle")
        subtitle = QLabel("Planner, controller and safety decisions")
        subtitle.setObjectName("reasoningSubtitle")
        title_stack.addWidget(self.title_label)
        title_stack.addWidget(subtitle)
        header.addLayout(title_stack, 1)

        self._view_badge = QLabel("WAITING")
        self._view_badge.setObjectName("reasoningViewBadge")
        self._view_badge.setAlignment(Qt.AlignCenter)
        header.addWidget(self._view_badge)

        self.close_button = QPushButton("×")
        self.close_button.setObjectName("panelCloseButton")
        self.close_button.setFixedSize(28, 26)
        self.close_button.setToolTip("Close Navigation Reasoning panel")
        self.close_button.clicked.connect(self.closeRequested.emit)
        header.addWidget(self.close_button)
        root.addWidget(header_widget)

        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("navigationReasoningScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.viewport().setStyleSheet("background: #F4F6F9;")
        root.addWidget(self.scroll, 1)

        self.content = QWidget()
        self.content.setObjectName("navigationReasoningContent")
        self.content.setStyleSheet("QWidget#navigationReasoningContent { background: #F4F6F9; }")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(10, 10, 10, 12)
        content_layout.setSpacing(9)

        self.placeholder = QFrame()
        self.placeholder.setObjectName("reasoningPlaceholder")
        placeholder_layout = QVBoxLayout(self.placeholder)
        placeholder_layout.setContentsMargins(18, 26, 18, 26)
        placeholder_layout.setSpacing(6)
        placeholder_layout.addStretch(1)
        placeholder_title = QLabel("No navigation snapshot yet")
        placeholder_title.setObjectName("reasoningPlaceholderTitle")
        placeholder_title.setAlignment(Qt.AlignCenter)
        self._placeholder_body = QLabel(
            "Start the simulation and enable Navigation Debug.\n"
            "The latest accepted plan and control decision will appear here."
        )
        self._placeholder_body.setObjectName("reasoningPlaceholderBody")
        self._placeholder_body.setAlignment(Qt.AlignCenter)
        self._placeholder_body.setWordWrap(True)
        placeholder_layout.addWidget(placeholder_title)
        placeholder_layout.addWidget(self._placeholder_body)
        placeholder_layout.addStretch(1)
        content_layout.addWidget(self.placeholder, 1)

        self.details = QWidget()
        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(9)

        self.summary_card = QFrame()
        self.summary_card.setObjectName("reasoningSummaryCard")
        summary_layout = QVBoxLayout(self.summary_card)
        summary_layout.setContentsMargins(12, 11, 12, 11)
        summary_layout.setSpacing(7)
        summary_top = QHBoxLayout()
        summary_top.setContentsMargins(0, 0, 0, 0)
        self._decision_badge = QLabel("—")
        self._decision_badge.setObjectName("reasoningDecisionBadge")
        self._decision_badge.setAlignment(Qt.AlignCenter)
        summary_top.addWidget(self._decision_badge)
        summary_top.addStretch(1)
        self._meta_label = QLabel("—")
        self._meta_label.setObjectName("reasoningMeta")
        self._meta_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        summary_top.addWidget(self._meta_label)
        summary_layout.addLayout(summary_top)
        self._explanation_label = QLabel("—")
        self._explanation_label.setObjectName("reasoningExplanation")
        self._explanation_label.setWordWrap(True)
        self._explanation_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addWidget(self._explanation_label)
        self._reason_label = QLabel("—")
        self._reason_label.setObjectName("reasoningMeta")
        self._reason_label.setWordWrap(True)
        self._reason_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        summary_layout.addWidget(self._reason_label)
        details_layout.addWidget(self.summary_card)

        self.runtime_section = _InfoSection(
            "ROBOT STATE",
            (
                ("navigation_state", "Navigation"),
                ("tracking_state", "Tracking"),
                ("position", "Position"),
                ("heading", "Heading"),
                ("velocity", "Velocity"),
                ("acceleration", "Acceleration"),
                ("angular_velocity", "Angular ω"),
                ("distance", "Distance"),
                ("mapped_points", "Mapped points"),
            ),
        )
        details_layout.addWidget(self.runtime_section)

        self.motion_section = _InfoSection(
            "TRACKING GEOMETRY",
            (
                ("target_heading", "Target θ"),
                ("heading_error", "Error θ"),
                ("rotate_threshold", "Rotate limit"),
                ("waypoint", "Waypoint"),
            ),
        )
        details_layout.addWidget(self.motion_section)

        self.control_section = _InfoSection(
            "CONTROL",
            (
                ("speed", "Velocity"),
                ("nominal", "Nominal a / ω"),
                ("applied", "Applied a / ω"),
            ),
        )
        details_layout.addWidget(self.control_section)

        self.planning_section = _InfoSection(
            "PLANNING",
            (
                ("planner", "Planner"),
                ("simplifier", "Simplifier"),
                ("route", "Route check"),
            ),
        )
        details_layout.addWidget(self.planning_section)

        self.safety_section = _InfoSection(
            "SAFETY",
            (
                ("active_segment", "Active segment"),
                ("predicted", "Prediction"),
                ("radii", "Radii"),
            ),
        )
        details_layout.addWidget(self.safety_section)

        self.event_section = _InfoSection(
            "LATEST EVENT",
            (("event", "Event"),),
        )
        details_layout.addWidget(self.event_section)
        details_layout.addStretch(1)

        content_layout.addWidget(self.details)
        self.scroll.setWidget(self.content)

        footer = QFrame(self)
        footer.setObjectName("reasoningFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 8, 10, 8)
        footer_layout.setSpacing(7)
        self._history_label = QLabel("No history")
        self._history_label.setObjectName("reasoningHistoryLabel")
        footer_layout.addWidget(self._history_label, 1)

        self.step_back_button = QPushButton("‹")
        self.step_forward_button = QPushButton("›")
        for button in (self.step_back_button, self.step_forward_button):
            button.setObjectName("historyStepButton")
            button.setFixedSize(32, 28)
            button.setVisible(False)
            button.setEnabled(False)
            # Repetition is owned by MainWindow so speed can ramp to 3x.
            button.setAutoRepeat(False)
        self.step_back_button.setToolTip("Previous snapshot (hold to accelerate)")
        self.step_forward_button.setToolTip("Next snapshot (hold to accelerate)")
        footer_layout.addWidget(self.step_back_button)
        footer_layout.addWidget(self.step_forward_button)
        root.addWidget(footer)

        # Compatibility hook used by a few existing tests and external debug
        # scripts. It points to the human-readable explanation field rather
        # than an invisible HTML document.
        self._label = self._explanation_label

        self.set_no_snapshot()

    def set_no_snapshot(self) -> None:
        self.placeholder.setVisible(True)
        self.details.setVisible(False)
        self._view_badge.setText("WAITING")
        self._history_label.setText("No history")
        self._decision_badge.setText("—")
        self._explanation_label.setText("No navigation decisions captured yet.")

    def set_history_controls(
        self,
        *,
        visible: bool,
        back_enabled: bool,
        forward_enabled: bool,
    ) -> None:
        """Reflect engine-owned history state without owning navigation logic."""
        visible = bool(visible)
        self.step_back_button.setVisible(visible)
        self.step_forward_button.setVisible(visible)
        self.step_back_button.setEnabled(visible and bool(back_enabled))
        self.step_forward_button.setEnabled(visible and bool(forward_enabled))

    def _set_decision_accent(self, tracking_mode: str, decision_kind: str) -> None:
        text = f"{tracking_mode} {decision_kind}".upper()
        if any(token in text for token in ("STOP", "BLOCK", "COLLISION", "FAILED")):
            foreground, background = "#B42318", "#FDE8E7"
        elif any(token in text for token in ("ROTATE", "REPLAN", "HOLD")):
            foreground, background = ORANGE, "#FFF1E5"
        elif "TRACK" in text or "FOLLOW" in text:
            foreground, background = GREEN, GREEN_LIGHT
        else:
            foreground, background = BLUE, BLUE_LIGHT
        self._decision_badge.setStyleSheet(
            f"color: {foreground}; background: {background}; border-radius: 8px; "
            "padding: 4px 8px; font-size: 10px; font-weight: 900;"
        )

    def update_snapshot(self, snapshot, last_event, history_position) -> None:
        if snapshot is None:
            self.set_no_snapshot()
            return

        self.placeholder.setVisible(False)
        self.details.setVisible(True)

        position, total = history_position
        if position is not None:
            view_text = f"HISTORY {position}/{total}"
            self._history_label.setText(f"Historical snapshot {position} of {total}")
        elif total:
            view_text = "LIVE"
            self._history_label.setText(f"Live · {total} saved snapshots")
        else:
            view_text = "LIVE"
            self._history_label.setText("Live · no saved snapshots")
        self._view_badge.setText(view_text)

        c = snapshot.controller
        tracking_mode = snapshot.tracking_mode or "—"
        decision_kind = snapshot.decision_kind or "—"
        self._decision_badge.setText(f"{tracking_mode} · {decision_kind}")
        self._set_decision_accent(tracking_mode, decision_kind)
        self._meta_label.setText(
            f"Robot {snapshot.robot_id}  ·  t={snapshot.simulation_time:.2f}s  ·  #{snapshot.snapshot_id}"
        )
        self._explanation_label.setText(snapshot.explanation or "No explanation was recorded.")
        self._reason_label.setText(f"Reason: {snapshot.decision_reason or '—'}")

        theta_t = _maybe_text(
            c.desired_heading,
            lambda value: f"{math.degrees(value):.1f}°",
        )
        heading_error = _maybe_text(
            c.heading_error,
            lambda value: f"{math.degrees(value):.1f}°",
        )
        rotate_threshold = _maybe_text(
            snapshot.rotate_threshold,
            lambda value: f"{math.degrees(value):.1f}°",
        )
        nominal = _maybe_text(
            c.nominal_control,
            lambda value: f"{value[0]:.2f} / {value[1]:.2f}",
        )
        applied = _maybe_text(
            c.applied_control,
            lambda value: f"{value[0]:.2f} / {value[1]:.2f}",
        )

        self.runtime_section.set_value("navigation_state", snapshot.navigation_state or "—")
        self.runtime_section.set_value("tracking_state", tracking_mode)
        self.runtime_section.set_value(
            "position",
            f"x={snapshot.robot_pose.x:.2f} m · y={snapshot.robot_pose.y:.2f} m",
        )
        self.runtime_section.set_value(
            "heading",
            f"{snapshot.robot_pose.theta:.3f} rad · {math.degrees(snapshot.robot_pose.theta):.1f}°",
        )
        self.runtime_section.set_value("velocity", f"{c.v:.3f} m/s")
        self.runtime_section.set_value("acceleration", f"{c.acceleration:.3f} m/s²")
        self.runtime_section.set_value("angular_velocity", f"{c.omega:.3f} rad/s")
        self.runtime_section.set_value(
            "distance",
            _maybe_text(c.distance_to_goal, lambda value: f"{value:.3f} m"),
        )
        self.runtime_section.set_value(
            "mapped_points",
            str(int(getattr(snapshot, "mapped_obstacle_points_count", 0))),
        )

        self.motion_section.set_value("target_heading", theta_t)
        self.motion_section.set_value("heading_error", heading_error)
        self.motion_section.set_value("rotate_threshold", rotate_threshold)
        waypoint_index = snapshot.path.active_waypoint_index
        self.motion_section.set_value("waypoint", "—" if waypoint_index is None else f"#{waypoint_index}")

        self.control_section.set_value("speed", f"{c.v:.2f} m/s")
        self.control_section.set_value("nominal", nominal)
        self.control_section.set_value("applied", applied)

        self.planning_section.set_value("planner", _maybe_text(snapshot.path.planner_name))
        self.planning_section.set_value("simplifier", _maybe_text(snapshot.path.simplifier_name))
        self.planning_section.set_value("route", _clearance_text(snapshot.route.first_segment))

        self.safety_section.set_value("active_segment", _clearance_text(snapshot.safety.active_segment))
        self.safety_section.set_value("predicted", _clearance_text(snapshot.predicted_motion.collision))
        self.safety_section.set_value(
            "radii",
            f"body={snapshot.safety.robot_radius:.2f} m · safety={snapshot.safety.safety_radius:.2f} m",
        )

        if last_event is None:
            event_text = "No event recorded yet."
        else:
            event_text = (
                f"{last_event.event_kind.value} @ {last_event.snapshot.simulation_time:.2f}s"
                f" — {last_event.snapshot.decision_reason or '—'}"
            )
        self.event_section.set_value("event", event_text)
