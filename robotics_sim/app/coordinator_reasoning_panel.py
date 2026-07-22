"""Read-only audit panel for multi-robot coordination decisions."""
from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from robotics_sim.app.theme import ThemeMode, theme_colors


_NUMBER = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")


def _json(value) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def _mapping(value) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


class CoordinatorReasoningPanel(QFrame):
    """Explain team allocation separately from frontier and route planning."""

    closeRequested = Signal()
    robotSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("coordinatorReasoningPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._last_update = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.setContentsMargins(14, 12, 10, 10)
        title = QLabel("Coordinator Reasoning")
        title.setObjectName("coordinatorReasoningTitle")
        header.addWidget(title, 1)
        self.robot_selector = QComboBox()
        self.robot_selector.setObjectName("coordinatorRobotSelector")
        self.robot_selector.addItem("R1")
        self.robot_selector.currentIndexChanged.connect(self._robot_changed)
        header.addWidget(self.robot_selector)
        close = QPushButton("×")
        close.setObjectName("coordinatorReasoningClose")
        close.setFixedSize(28, 26)
        close.clicked.connect(self.closeRequested.emit)
        header.addWidget(close)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setObjectName("coordinatorReasoningScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        body.setObjectName("coordinatorReasoningContent")
        body.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 8, 10, 12)
        layout.setSpacing(9)
        self.summary = self._label("Waiting for a coordination decision", "coordinatorSummary")
        layout.addWidget(self.summary)
        self.ownership = self._card(layout, "ALGORITHM OWNERSHIP")
        self.formula = self._card(layout, "COORDINATION / ASSIGNMENT FORMULA", rich=True)
        self.variables = self._card(layout, "REAL INPUTS FOR THE SELECTED ROBOT", rich=True)
        self.computation = self._card(layout, "STEP-BY-STEP COORDINATION", rich=True)
        self.assignments = self._card(layout, "TEAM ASSIGNMENTS")
        self.matrix = self._card(layout, "UTILITY / FEASIBILITY MATRIX")
        self.debug = self._card(layout, "EXPORTED COORDINATOR DEBUG")
        layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        self.set_theme_mode(ThemeMode.LIGHT)

    @staticmethod
    def _label(text: str, name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(name)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def _card(self, layout, title: str, *, rich: bool = False) -> QLabel:
        frame = QFrame()
        frame.setObjectName("coordinatorReasoningCard")
        box = QVBoxLayout(frame)
        heading = self._label(title, "coordinatorCardTitle")
        value = self._label("—", "coordinatorCardValue")
        value.setTextFormat(Qt.RichText if rich else Qt.PlainText)
        box.addWidget(heading)
        box.addWidget(value)
        layout.addWidget(frame)
        return value

    def set_theme_mode(self, mode: ThemeMode | str) -> None:
        c = theme_colors(ThemeMode(mode))
        self.setStyleSheet(f"""
            QFrame#coordinatorReasoningPanel {{ background:{c.card_background}; border:none; }}
            QScrollArea#coordinatorReasoningScroll {{ background:{c.app_background}; border:none; }}
            QWidget#coordinatorReasoningContent {{ background:{c.app_background}; border:none; }}
            QLabel {{ color:{c.text_primary}; background:transparent; }}
            QLabel#coordinatorReasoningTitle {{ font-size:15px; font-weight:900; }}
            QLabel#coordinatorSummary {{ color:{c.accent}; font-weight:800; padding:8px; }}
            QFrame#coordinatorReasoningCard {{ background:{c.panel_background}; border:1px solid {c.border}; border-radius:9px; }}
            QLabel#coordinatorCardTitle {{ color:{c.accent}; font-size:10px; font-weight:900; }}
            QLabel#coordinatorCardValue {{ font-family:Consolas,monospace; font-size:9px; }}
            QPushButton#coordinatorReasoningClose {{ border:none; background:transparent; color:{c.text_secondary}; font-size:18px; }}
            QComboBox#coordinatorRobotSelector {{ min-width:64px; padding:4px; border:1px solid {c.border}; border-radius:6px; }}
        """)

    def set_robot_selector(self, index: int, count: int) -> None:
        self.robot_selector.blockSignals(True)
        self.robot_selector.clear()
        self.robot_selector.addItems([f"R{i + 1}" for i in range(max(1, int(count)))])
        self.robot_selector.setCurrentIndex(max(0, min(int(index), self.robot_selector.count() - 1)))
        self.robot_selector.setVisible(int(count) > 1)
        self.robot_selector.blockSignals(False)
        self._render()

    def _robot_changed(self, index: int) -> None:
        if index >= 0:
            self.robotSelected.emit(index)
            self._render()

    def clear(self) -> None:
        self._last_update = None
        self.summary.setText("Waiting for a coordination decision")
        for label in (
            self.ownership,
            self.formula,
            self.variables,
            self.computation,
            self.assignments,
            self.matrix,
            self.debug,
        ):
            label.setText("—")

    def update_coordination(
        self,
        *,
        planner: str,
        coordinator: str,
        result,
        time_s: float,
        runtime_profile=None,
    ) -> None:
        self._last_update = (planner, coordinator, result, float(time_s), runtime_profile)
        count = max(
            len(getattr(result, "targets", ()) or ()),
            len(getattr(result, "reasons", ()) or ()),
            1,
        )
        self.set_robot_selector(max(0, self.robot_selector.currentIndex()), count)

    @staticmethod
    def _assignment_for_robot(result, robot_index: int):
        for assignment in tuple(getattr(result, "assignments", ()) or ()):
            if int(getattr(assignment, "robot_id", -1)) == int(robot_index):
                return assignment
        return None

    @staticmethod
    def _per_robot_debug(debug: dict, robot_index: int) -> dict:
        per_robot = _mapping(debug.get("per_robot", {}))
        return _mapping(per_robot.get(str(robot_index), per_robot.get(robot_index, {})))

    def _render(self) -> None:
        if self._last_update is None:
            return
        planner, coordinator, result, time_s, profile = self._last_update
        idx = max(0, self.robot_selector.currentIndex())
        targets = list(getattr(result, "targets", ()) or ())
        reasons = list(getattr(result, "reasons", ()) or ())
        target = targets[idx] if idx < len(targets) else None
        reason = reasons[idx] if idx < len(reasons) else "no per-robot reason exported"
        debug = _mapping(getattr(result, "debug", {}))
        robot_debug = self._per_robot_debug(debug, idx)
        assignment = self._assignment_for_robot(result, idx)
        status = str(getattr(assignment, "status", "UNKNOWN"))
        proposal = getattr(assignment, "proposal", None)
        proposal_inputs = {}
        if proposal is not None:
            proposal_inputs = {
                "source": getattr(proposal, "source", None),
                "information_gain": getattr(proposal, "information_gain", None),
                "travel_cost": getattr(proposal, "travel_cost", None),
                "safety_cost": getattr(proposal, "safety_cost", None),
                "overlap_cost": getattr(proposal, "overlap_cost", None),
                "heading_cost": getattr(proposal, "heading_cost", None),
                "metadata": _mapping(getattr(proposal, "metadata", {})),
            }

        self.summary.setText(
            f"t={time_s:.2f}s · {coordinator}\n"
            f"Inspecting R{idx + 1} · status={status} · target={target}\n{reason}"
        )
        if profile is None:
            self.ownership.setText(f"Exploration planner: {planner}\nCoordinator: {coordinator}")
        else:
            self.ownership.setText(
                f"owns target generation: {bool(getattr(profile, 'owns_target_generation', False))}\n"
                f"owns task allocation: {bool(getattr(profile, 'owns_task_allocation', False))}\n"
                f"owns path planning: {bool(getattr(profile, 'owns_path_planning', False))}\n"
                f"owns control: {bool(getattr(profile, 'owns_control', False))}"
            )

        scalar_terms = {key: float(value) for key, value in _NUMBER.findall(str(reason))}
        if "utility_matrix" in debug:
            weights = _mapping(debug.get("weight_configuration", {}))
            self.formula.setText(
                "<b>U(i,j)=w<sub>I</sub>I(j) − w<sub>d</sub>d(i,j) − "
                "w<sub>o</sub>O(i,j)</b><br>"
                "Hungarian selects π that maximizes <b>Σ<sub>i</sub> U(i,π(i))</b> "
                "with at most one robot per task."
            )
            selected_by_robot = _mapping(debug.get("selected_task_by_robot", {}))
            self.variables.setText(
                f"<b>R{idx + 1}</b>; target={html.escape(str(target))}<br>"
                f"weights={html.escape(str(weights))}<br>"
                f"selected task={html.escape(str(selected_by_robot.get(str(idx), 'unavailable')))}"
                + ("<pre>" + html.escape(_json(proposal_inputs)) + "</pre>" if proposal_inputs else "")
            )
            self.matrix.setText(_json({
                "robot row order": debug.get("robots_to_assign", "unavailable"),
                "task columns": debug.get("task_ids", "unavailable"),
                "utility": debug.get("utility_matrix"),
                "feasible": debug.get("feasible_matrix"),
            }))
        elif "CQLite" in str(coordinator) or "q_table_sizes" in debug:
            self.formula.setText(
                "<b>Q(s,a) ← Q(s,a) + α[r + γ max Q(s′,a′) − Q(s,a)]</b><br>"
                "The eligible frontier with the highest learned priority is reserved for R<sub>i</sub>."
            )
            self.variables.setText(
                "<pre>" + html.escape(_json({
                    "plugin per-robot debug": robot_debug or scalar_terms,
                    "selected proposal": proposal_inputs,
                })) + "</pre>"
            )
            self.matrix.setText("CQLite performs distributed per-robot ranking; it does not export a Hungarian matrix.")
        elif "FUEL" in str(coordinator):
            self.formula.setText(
                "<b>S=w<sub>I</sub>log(1+I) − w<sub>d</sub>C<sub>travel</sub> − "
                "w<sub>h</sub>C<sub>heading</sub> − w<sub>s</sub>C<sub>safety</sub> − "
                "w<sub>o</sub>C<sub>overlap</sub></b>"
            )
            self.variables.setText("<pre>" + html.escape(_json({
                "plugin per-robot debug": robot_debug or scalar_terms,
                "selected proposal": proposal_inputs,
            })) + "</pre>")
            self.matrix.setText("FUEL ranks clustered viewpoints per robot; no global utility matrix was exported.")
        else:
            self.formula.setText(
                "<b>Coordinator-native allocation:</b> choose a feasible, non-conflicting target "
                "for every requested robot according to the active plugin."
            )
            inputs = {
                "reason terms": scalar_terms,
                "plugin per-robot debug": robot_debug,
            }
            if proposal_inputs:
                inputs["selected proposal"] = proposal_inputs
            self.variables.setText("<pre>" + html.escape(_json(inputs)) + "</pre>")
            self.matrix.setText("The active coordinator did not export a utility matrix.")

        requested = debug.get("robots_to_assign", debug.get("requested_indices", "unavailable"))
        self.computation.setText(
            "<ol>"
            f"<li>Receive the team state and requested robots: {html.escape(str(requested))}.</li>"
            f"<li>Obtain candidates from {html.escape(str(planner))} or from coordinator-owned generation.</li>"
            "<li>Apply reservations, overlap, feasibility, and plugin-specific constraints.</li>"
            f"<li>Return R{idx + 1} → {html.escape(str(target))} with status {html.escape(status)}.</li>"
            "</ol>"
        )
        self.assignments.setText("\n".join(
            f"{'▶' if i == idx else ' '} R{i + 1}: {targets[i] if i < len(targets) else None} · "
            f"{reasons[i] if i < len(reasons) else ''}"
            for i in range(max(len(targets), len(reasons)))
        ) or "No assignments")
        self.debug.setText(_json(debug) if debug else "No debug payload exported by the coordinator.")
