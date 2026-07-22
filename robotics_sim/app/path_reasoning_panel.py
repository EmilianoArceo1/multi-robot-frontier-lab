"""Live explanation of the path-planning computation."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QToolTip, QVBoxLayout, QWidget

from robotics_sim.app.theme import ThemeMode, theme_colors


_HELP = {
    "g": "<b>g(n) — coste acumulado</b><br>Suma el coste de cada movimiento desde la celda inicial hasta n. Un movimiento horizontal/vertical cuesta la resolución del grid; uno diagonal cuesta sqrt(2) veces la resolución.",
    "h": "<b>h(n) — heurística</b><br>A* usa distancia octile hasta la meta: (sqrt(2)·min(Δfila,Δcol) + max(Δfila,Δcol)−min(...))·resolución. Dijkstra usa h(n)=0.",
    "f": "<b>f(n) — prioridad</b><br>A* extrae de OPEN la celda con menor f(n)=g(n)+h(n). Dijkstra hace lo mismo con f(n)=g(n).",
    "open": "<b>OPEN</b><br>Cola de prioridad de celdas descubiertas pero todavía no expandidas. La siguiente expansión es la de menor f.",
    "closed": "<b>CLOSED</b><br>Celdas ya expandidas. No se vuelven a procesar.",
    "raw": "<b>Raw path</b><br>Cadena de celdas reconstruida desde la meta siguiendo came_from hasta el inicio, antes de simplificar.",
    "simplified": "<b>Simplified path</b><br>Ruta después del simplificador configurado. Direction changes conserva giros; line-of-sight elimina puntos solo si el segmento completo es seguro en el mismo grid.",
    "cost": "<b>Coste total</b><br>g(goal): suma exacta de los costes de los movimientos de la ruta encontrada por el buscador, expresada en metros del grid.",
    "unknown": "<b>Unknown policy</b><br>Determina si las celdas −1 pueden atravesarse. No cambia el mapa; cambia qué vecinos admite la búsqueda.",
}


def _link(key: str, text: str) -> str:
    return f"<a href='pathvar://{key}' style='color:inherit;text-decoration:underline'>{text}</a>"


def _table(rows) -> str:
    return "<table cellspacing='4' cellpadding='2'>" + "".join(
        f"<tr><td><b>{a}</b></td><td>=</td><td>{b}</td></tr>" for a, b in rows
    ) + "</table>"


class PathReasoningPanel(QFrame):
    closeRequested = Signal()
    robotSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pathReasoningPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.setContentsMargins(14, 12, 10, 10)
        title = QLabel("Path Reasoning")
        title.setObjectName("pathReasoningTitle")
        header.addWidget(title, 1)
        self.robot_selector = QComboBox()
        self.robot_selector.setObjectName("pathRobotSelector")
        self.robot_selector.addItem("R1")
        self.robot_selector.currentIndexChanged.connect(self._robot_changed)
        header.addWidget(self.robot_selector)
        close = QPushButton("×")
        close.setObjectName("pathReasoningClose")
        close.setFixedSize(28, 26)
        close.clicked.connect(self.closeRequested.emit)
        header.addWidget(close)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 8, 10, 12)
        layout.setSpacing(9)
        self.summary = self._label("Waiting for a route calculation", "pathSummary")
        layout.addWidget(self.summary)
        self.live_state = self._card(layout, "LIVE ROBOT POSITION", rich=True)
        self.formula = self._card(layout, "SEARCH FORMULA", rich=True)
        self.variables = self._card(layout, "VARIABLES AND HOW THEY ARE CALCULATED", rich=True)
        self.variables.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.variables.linkHovered.connect(self._show_help)
        self.procedure = self._card(layout, "STEP-BY-STEP ROUTE COMPUTATION", rich=True)
        self.result = self._card(layout, "COMPLETE CALCULATION WITH REAL VALUES", rich=True)
        self.path = self._card(layout, "RAW → SIMPLIFIED → EXECUTABLE PATH")
        layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        self.set_theme_mode(ThemeMode.LIGHT)
        self._plan_start_xy = None
        self._plan_goal_xy = None
        self._executable_waypoints = ()
        self._routes_by_robot = {}
        self._poses_by_robot = {}

    @staticmethod
    def _label(text: str, name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(name)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def _card(self, layout, title: str, *, rich: bool = False) -> QLabel:
        card = QFrame()
        card.setObjectName("pathReasoningCard")
        card_layout = QVBoxLayout(card)
        heading = self._label(title, "pathCardTitle")
        value = self._label("—", "pathCardValue")
        value.setTextFormat(Qt.RichText if rich else Qt.PlainText)
        card_layout.addWidget(heading)
        card_layout.addWidget(value)
        layout.addWidget(card)
        return value

    def _show_help(self, href: str) -> None:
        text = _HELP.get(str(href).removeprefix("pathvar://"))
        if text:
            QToolTip.showText(QCursor.pos(), text, self.variables)
        else:
            QToolTip.hideText()

    def set_theme_mode(self, mode: ThemeMode | str) -> None:
        c = theme_colors(ThemeMode(mode))
        self.setStyleSheet(f"""
            QFrame#pathReasoningPanel {{ background:{c.card_background}; border:none; }}
            QScrollArea, QWidget {{ background:{c.app_background}; border:none; }}
            QLabel {{ color:{c.text_primary}; background:transparent; }}
            QLabel#pathReasoningTitle {{ font-size:15px; font-weight:900; }}
            QLabel#pathSummary {{ color:{c.accent}; font-weight:800; padding:8px; }}
            QFrame#pathReasoningCard {{ background:{c.panel_background}; border:1px solid {c.border}; border-radius:9px; }}
            QLabel#pathCardTitle {{ color:{c.accent}; font-size:10px; font-weight:900; }}
            QLabel#pathCardValue {{ font-family:Consolas, monospace; font-size:9px; }}
            QPushButton#pathReasoningClose {{ border:none; background:transparent; color:{c.text_secondary}; font-size:18px; }}
            QComboBox#pathRobotSelector {{ min-width:64px; padding:4px; border:1px solid {c.border}; border-radius:6px; }}
        """)

    def set_robot_selector(self, index: int, count: int) -> None:
        self.robot_selector.blockSignals(True)
        self.robot_selector.clear()
        self.robot_selector.addItems([f"R{i + 1}" for i in range(max(1, int(count)))])
        self.robot_selector.setCurrentIndex(max(0, min(int(index), self.robot_selector.count() - 1)))
        self.robot_selector.setVisible(int(count) > 1)
        self.robot_selector.blockSignals(False)
        self._render_selected_robot()

    def _robot_changed(self, index: int) -> None:
        if index >= 0:
            self.robotSelected.emit(index)
            self._render_selected_robot()

    def _render_selected_robot(self) -> None:
        index = max(0, self.robot_selector.currentIndex())
        payload = self._routes_by_robot.get(index)
        if payload is not None:
            self.update_route(**payload, robot_index=index, _from_cache=True)
        pose = self._poses_by_robot.get(index)
        if pose is not None:
            self.update_live_pose(pose, robot_label=f"R{index + 1}", robot_index=index)

    def clear(self) -> None:
        self._routes_by_robot.clear()
        self._poses_by_robot.clear()
        self.summary.setText("Waiting for a route calculation")
        self._plan_start_xy = None
        self._plan_goal_xy = None
        self._executable_waypoints = ()
        for label in (self.live_state, self.formula, self.variables, self.procedure, self.result, self.path):
            label.setText("—")

    def update_live_pose(self, robot_xy, *, robot_label: str = "R1", robot_index: int = 0) -> None:
        """Refresh motion state without rewriting the historical plan inputs."""
        if robot_xy is not None:
            self._poses_by_robot[int(robot_index)] = tuple(robot_xy)
        if int(robot_index) != max(0, self.robot_selector.currentIndex()):
            return
        if robot_xy is None:
            self.live_state.setText("—")
            return
        current = (float(robot_xy[0]), float(robot_xy[1]))
        goal = self._plan_goal_xy
        next_waypoint = self._executable_waypoints[0] if self._executable_waypoints else None
        goal_distance = math.dist(current, goal) if goal is not None else None
        waypoint_distance = math.dist(current, next_waypoint) if next_waypoint is not None else None
        self.live_state.setText(
            f"<b>{robot_label} current R(t)</b> = ({current[0]:.6f}, {current[1]:.6f})<br>"
            f"R at planning time R<sub>plan</sub> = {self._plan_start_xy}<br>"
            f"distance to next waypoint = {'unavailable' if waypoint_distance is None else f'{waypoint_distance:.6f} m'}<br>"
            f"distance to planned goal = {'unavailable' if goal_distance is None else f'{goal_distance:.6f} m'}"
        )

    def update_route(self, *, planner: str, simplifier: str, success: bool, reason: str,
                     capture=None, waypoints=(), start_xy=None, goal_xy=None, time_s: float = 0.0,
                     robot_index: int = 0, _from_cache: bool = False) -> None:
        if not _from_cache:
            self._routes_by_robot[int(robot_index)] = dict(
                planner=planner, simplifier=simplifier, success=success, reason=reason,
                capture=capture, waypoints=tuple(waypoints or ()), start_xy=start_xy,
                goal_xy=goal_xy, time_s=time_s,
            )
        if int(robot_index) != max(0, self.robot_selector.currentIndex()):
            return
        raw = list(getattr(capture, "raw_world_path", ()) or ())
        simplified = list(getattr(capture, "simplified_world_path", ()) or ())
        total_cost = getattr(capture, "total_cost", None)
        expanded = getattr(capture, "expanded_nodes", None)
        unknown = getattr(capture, "unknown_is_traversable", None)
        start_cell = getattr(capture, "start_cell", None)
        goal_cell = getattr(capture, "goal_cell", None)
        resolution = getattr(capture, "grid_resolution", None)
        if start_xy is None and raw:
            start_xy = raw[0]
        if goal_xy is None and raw:
            goal_xy = raw[-1]
        direct = math.dist(start_xy, goal_xy) if start_xy is not None and goal_xy is not None else None
        self._plan_start_xy = start_xy
        self._plan_goal_xy = goal_xy
        self._executable_waypoints = tuple(waypoints or ())
        index = int(robot_index)
        # R_plan is historical.  Never let it overwrite a newer live pose
        # when the user switches back to a cached robot route.
        if index not in self._poses_by_robot and start_xy is not None:
            self._poses_by_robot[index] = tuple(start_xy)
        self.update_live_pose(
            self._poses_by_robot.get(index),
            robot_label=f"R{index + 1}",
            robot_index=index,
        )

        if str(planner).lower() == "direct":
            formula = "<b>P</b> = [R, G] if the direct segment is accepted"
        elif str(planner).lower() == "dijkstra":
            formula = "<b>f(n) = g(n)</b>; expand arg min<sub>n∈OPEN</sub> f(n)"
        else:
            formula = "<b>f(n) = g(n) + h<sub>octile</sub>(n,G)</b>; expand arg min<sub>n∈OPEN</sub> f(n)"
        self.formula.setText(formula)
        self.variables.setText(_table([
            (_link("g", "g(n)"), "accumulated movement cost from start to n"),
            (_link("h", "h(n)"), "estimated remaining grid cost to goal"),
            (_link("f", "f(n)"), "priority used to choose the next cell"),
            (_link("open", "OPEN"), "discovered cells pending expansion"),
            (_link("closed", "CLOSED"), "already-expanded cells"),
            ("expanded nodes", "unavailable" if expanded is None else str(expanded)),
            (_link("unknown", "unknown policy"), str(unknown)),
            (_link("cost", "g(goal)"), "unavailable" if total_cost is None else f"{float(total_cost):.6f} m"),
            (_link("raw", "raw path"), f"{len(raw)} points"),
            (_link("simplified", "simplified path"), f"{len(simplified)} points using {simplifier}"),
        ]))
        steps = [
            f"Convert R_plan={start_xy} and G={goal_xy} to grid cells.",
            f"Apply traversability policy: unknown_is_traversable={unknown}.",
            "Insert the start in OPEN and repeatedly extract the smallest priority.",
            "Relax every traversable neighbor: tentative_g = g(current) + movement_cost.",
            f"Reconstruct came_from at the goal: {len(raw)} raw points.",
            f"Apply '{simplifier}': {len(raw)} → {len(simplified)} points.",
            f"Drop the start cell and publish {len(tuple(waypoints or ()))} executable waypoints.",
        ]
        if start_cell is not None and goal_cell is not None and resolution is not None:
            dr = abs(int(start_cell.row) - int(goal_cell.row))
            dc = abs(int(start_cell.col) - int(goal_cell.col))
            diagonal = min(dr, dc)
            straight = max(dr, dc) - diagonal
            heuristic = 0.0 if str(planner).lower() == "dijkstra" else (
                math.sqrt(2.0) * diagonal + straight
            ) * float(resolution)
            steps.insert(2, (
                f"Initial heuristic: Δrow={dr}, Δcol={dc}; h(R)=({diagonal}·√2 + "
                f"{straight})·{float(resolution):.6f} = {heuristic:.6f}."
            ))
        running_cost = 0.0
        for index, (a, b) in enumerate(zip(raw[:-1], raw[1:]), start=1):
            segment_cost = math.dist(a, b)
            running_cost += segment_cost
            steps.append(
                f"Raw segment {index}: {tuple(round(v, 3) for v in a)} → "
                f"{tuple(round(v, 3) for v in b)}; Δg={segment_cost:.6f}; g={running_cost:.6f}."
            )
        self.procedure.setText("<ol>" + "".join(f"<li>{line}</li>" for line in steps) + "</ol>")
        cost_text = "unavailable" if total_cost is None else f"{float(total_cost):.6f}"
        direct_text = "unavailable" if direct is None else f"{direct:.6f}"
        self.result.setText(
            f"<b>R_plan={start_xy}, G={goal_xy}</b><br>direct distance at planning time = {direct_text} m<br>"
            f"g(G) = Σ movement costs = <b>{cost_text} m</b><br>"
            f"expanded nodes = <b>{'unavailable' if expanded is None else expanded}</b><br>"
            f"raw points {len(raw)} → simplified points {len(simplified)} → executable points {len(tuple(waypoints or ()))}"
        )
        def fmt(points):
            return " → ".join(f"({float(x):.2f},{float(y):.2f})" for x, y in points) or "unavailable"
        self.path.setText(f"RAW: {fmt(raw)}\n\nSIMPLIFIED: {fmt(simplified)}\n\nEXECUTABLE: {fmt(waypoints)}")
        self.summary.setText(f"R{int(robot_index) + 1} · t={time_s:.2f}s · {planner} / {simplifier}\n{'ACCEPTED' if success else 'REJECTED'} · {reason}")
