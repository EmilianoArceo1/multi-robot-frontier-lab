"""Live, read-only explanation panel for frontier-selection decisions."""
from __future__ import annotations

import html
import json
import math
import re
from types import SimpleNamespace

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QToolTip, QVBoxLayout, QWidget,
)

from robotics_interfaces.plugins import CandidateInputMode
from robotics_sim.app.theme import ThemeMode, theme_colors


_TERM_RE = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")

# Per-CandidateInputMode description of where a coordination decision's
# candidate actually came from. Never says "the coordinator generated the
# frontier" for a plugin that only consumed a host-provided pool/cluster --
# see problem J in the exploration-pipeline-architecture refactor brief.
_CANDIDATE_SOURCE_DESCRIPTIONS = {
    CandidateInputMode.HOST_CANDIDATES: (
        "The host frontier candidate pipeline generated the candidate pool; "
        "{coordinator} selected/allocated one for this robot from it."
    ),
    CandidateInputMode.HOST_FRONTIER_CLUSTERS: (
        "The host detected frontier clusters; {coordinator} reduced/allocated "
        "a task from them for this robot."
    ),
    CandidateInputMode.PLUGIN_INTERNAL: (
        "Frontier/task generation is internal to {coordinator}; it does not "
        "consume host-provided candidates for this decision."
    ),
    CandidateInputMode.HYBRID: (
        "{coordinator} may use host-provided candidates or its own fallback "
        "generation; see \"selected proposal source\" below for the source "
        "actually used in this decision."
    ),
    CandidateInputMode.LEGACY_INTEGRATED: (
        "{coordinator} is a legacy integrated pipeline: frontier detection "
        "and task allocation are not actually separated inside it."
    ),
}
_DEFAULT_CANDIDATE_SOURCE_DESCRIPTION = (
    "{coordinator} produced this target; the actual candidate source for "
    "this decision was not declared (candidate_input_mode unavailable)."
)

_VARIABLE_HELP = {
    "R": "<b>R — robot</b><br>La posición actual del robot: (x<sub>R</sub>, y<sub>R</sub>). Es el origen para calcular distancia, dirección y ruta.",
    "F": "<b>F — frontier candidato</b><br>Una frontera es un grupo de celdas libres conocidas que toca celdas desconocidas. F es el punto representativo elegido dentro de ese grupo.",
    "G": "<b>G — goal final</b><br>La meta global configurada para la misión. No es necesariamente el próximo destino durante exploración.",
    "cluster_size": "<b>|C<sub>F</sub>| — tamaño de frontera</b><br>Número de celdas conectadas que forman el cluster de F. Un cluster grande suele representar una abertura mayor hacia espacio desconocido.",
    "robot_distance": "<b>d(R,F) — distancia robot–frontier</b><br>Se calcula como sqrt((x<sub>F</sub>−x<sub>R</sub>)² + (y<sub>F</sub>−y<sub>R</sub>)²). En planners con A*, la longitud de ruta se calcula aparte.",
    "goal_distance": "<b>d(F,G) — distancia frontier–goal</b><br>Distancia euclidiana sqrt((x<sub>G</sub>−x<sub>F</sub>)² + (y<sub>G</sub>−y<sub>F</sub>)²). Favorece frontiers que también avanzan hacia la meta final.",
    "lambda": "<b>λ — peso de desplazamiento</b><br>Multiplicador que convierte distancia recorrida en penalización. Un λ mayor prefiere destinos cercanos aunque descubran menos.",
    "information_gain": "<b>I(F) — information gain</b><br>Cantidad estimada de celdas desconocidas que el sensor podría revelar desde F. Se proyecta el FoV sobre el mapa de creencias y se cuentan las celdas UNKNOWN visibles.",
    "info_utility": "<b>info_utility — utilidad normalizada de descubrimiento</b><br>Combina tres señales: 0.40·novelty durante la ruta + 0.40·terminal_novelty en F + 0.20·gain_norm. novelty es la fracción UNKNOWN del FoV barrido; terminal_novelty es esa fracción desde F; gain_norm limita a 1 la ganancia absoluta. Rango aproximado: 0–1; más alto es mejor.",
    "frontier_norm": "<b>frontier_norm — tamaño normalizado</b><br>log(1 + tamaño del cluster) / log(1 + tamaño del cluster más grande). Rango 0–1. Usa logaritmo para que una frontera enorme no domine por sí sola.",
    "align": "<b>align — alineación inicial</b><br>cos(ángulo del primer tramo de la ruta − orientación actual del robot). 1 significa seguir de frente, 0 un giro de 90°, −1 ir en dirección contraria.",
    "hazard": "<b>hazard — atracción por incendios</b><br>Suma kernels gaussianos exp(−d²/(2σ²)) desde el candidato hasta cada fuente descubierta. Vale 1 en el centro de una fuente aislada y decae suavemente con la distancia; σ=4 m por defecto.",
    "length_norm": "<b>length_norm — longitud de ruta normalizada</b><br>Coste total de la ruta A* dividido entre la diagonal completa del grid. Hace comparable la distancia entre mapas de tamaños distintos; menor es mejor.",
    "repetition": "<b>repetition — repetición combinada</b><br>Promedio de fov_repeat y path_repeat. Resume en un solo término R(F) cuánto repite el candidato visión y recorrido ya observados.",
    "fov_repeat": "<b>fov_repeat — repetición del FoV</b><br>Promedio de cuántas veces ya fueron observadas las celdas cubiertas por el FoV a lo largo de la ruta, limitado por seen_saturation. 0 significa visión nueva; valores altos indican volver a mirar lo conocido.",
    "path_repeat": "<b>path_repeat — repetición de ruta</b><br>La misma penalización de observación, pero evaluada sobre las celdas por donde pasa la ruta A*. Penaliza recorrer zonas ya transitadas.",
    "turn": "<b>turn — coste de giro</b><br>Suma el giro desde la orientación actual al primer segmento y los cambios de dirección de toda la ruta; después divide entre π y limita el resultado a 1.",
    "detour": "<b>detour — desvío</b><br>1 − distancia_directa(R,F) / coste_ruta_A*. Vale 0 para una ruta prácticamente recta y aumenta cuando los obstáculos obligan a rodear.",
    "backtrack": "<b>backtrack — retroceso</b><br>max(0, −cos(dirección hacia F − orientación actual)). Solo penaliza candidatos situados detrás del robot.",
    "switch": "<b>switch — cambio de objetivo</b><br>Vale 0 si F coincide con el objetivo actual dentro de una celda; vale 1 si obliga a cambiar de frontier. Reduce oscilaciones entre decisiones.",
    "multi": "<b>multi — interferencia multi-robot M'</b><br>Suma penalizaciones gaussianas exp(−d²/σ²) respecto a objetivos reservados por otros robots. Los cuerpos dinámicos siguen en el costmap de A*, pero no contaminan este término de asignación.",
}


def _variable_link(key: str, label: str) -> str:
    return f"<a href='var://{key}' style='color:inherit; text-decoration:underline'>{label}</a>"


def _terms(reason: str) -> dict[str, float]:
    return {key: float(value) for key, value in _TERM_RE.findall(str(reason))}


def _table(rows: list[tuple[str, str]]) -> str:
    return "<table cellspacing='4' cellpadding='2'>" + "".join(
        f"<tr><td><b>{name}</b></td><td>=</td><td>{value}</td></tr>" for name, value in rows
    ) + "</table>"


def _steps(lines: list[str]) -> str:
    return "<ol style='margin-left:14px'>" + "".join(f"<li>{line}</li>" for line in lines) + "</ol>"


def _result(calculated: float | None, reported: float) -> str:
    if calculated is None:
        return f"<b>Calculated:</b> unavailable<br><b>Reported:</b> {reported:.6f}"
    delta = calculated - reported
    status = "CONSISTENT" if abs(delta) <= 1e-4 else "CHECK MISMATCH"
    color = "#238636" if abs(delta) <= 1e-4 else "#C62828"
    return (
        f"<b>Calculated:</b> {calculated:.6f}<br>"
        f"<b>Reported:</b> {reported:.6f}<br>"
        f"<b>Delta:</b> {delta:+.6f}<br>"
        f"<span style='color:{color}; font-weight:800'>{status}</span>"
    )


def _raw_frontier_formula_explanation(planner: str, candidate: dict | None) -> tuple[str, str, str, str]:
    """Return symbolic formula, substitution, computation steps and audit."""
    candidate = candidate or {}
    values = _terms(candidate.get("reason", ""))
    size = int(candidate.get("size", 0))
    distance = float(candidate.get("distance", 0.0))
    info = float(candidate.get("information_gain", 0.0))
    score = float(candidate.get("score", 0.0))

    if planner == "Nav2D nearest-frontier wavefront":
        wave = values.get("wavefront_distance", distance)
        return (
            "<b>F<sup>*</sup></b> = first valid frontier reached by 4-connected BFS",
            _table([("d<sub>BFS</sub>", f"{wave:.2f} m")]),
            _steps(["Initialize the selected robot cell with level 0.",
                    "Expand only through known FREE cells using four neighbors.",
                    f"Accept the first valid frontier reached at {wave:.2f} m."]),
            f"<b>Selected BFS distance:</b> {wave:.2f} m",
        )
    if planner == "Nearest frontier":
        return (
            "<b>F<sup>*</sup></b> = arg min d(R,F)<br><small>tie → max cluster size</small>",
            _table([("d(R,F)", f"{distance:.2f} m"), ("|cluster|", str(size))]),
            _steps([f"Compute Euclidean distance: d(R,F) = {distance:.2f} m.",
                    "Compare this distance with every reachable candidate.",
                    f"If tied, prefer the larger cluster (current size {size})."]),
            f"<b>Ranking key:</b> ({distance:.6f}, {-size})",
        )
    if planner == "Largest frontier":
        return (
            "<b>F<sup>*</sup></b> = arg max |cluster(F)|<br><small>tie → min distance</small>",
            _table([("|cluster|", str(size)), ("d(R,F)", f"{distance:.2f} m")]),
            _steps([f"Count frontier cells in the cluster: {size}.",
                    "Compare cluster sizes across reachable candidates.",
                    f"If tied, prefer the shortest distance ({distance:.2f} m)."]),
            f"<b>Ranking key:</b> ({size}, {-distance:.6f})",
        )
    if planner == "Utility frontier":
        goal = values.get("goal_distance", 0.0)
        calculated = size - 0.75 * distance - 0.15 * goal
        return (
            "<b>U(F)</b> = |C<sub>F</sub>| − 0.75 d(R,F) − 0.15 d(F,G)",
            f"<b>U(F)</b> = {size} − 0.75({distance:.2f}) − 0.15({goal:.2f})",
            _steps([f"Cluster contribution: +{size:.6f}",
                    f"Robot-distance penalty: −0.75 × {distance:.6f} = −{0.75 * distance:.6f}",
                    f"Goal-distance penalty: −0.15 × {goal:.6f} = −{0.15 * goal:.6f}",
                    f"Sum: {size:.6f} − {0.75 * distance:.6f} − {0.15 * goal:.6f} = {calculated:.6f}"]),
            _result(calculated, score),
        )
    if planner == "Informative frontier / IPP-lite":
        penalty = (info - score) / distance if distance > 1e-9 else 0.0
        calculated = info - penalty * distance
        return (
            "<b>U(F)</b> = I(F) − λ d(R,F)",
            f"<b>U(F)</b> = {info:.1f} − {penalty:.6f}({distance:.2f})",
            _steps([f"Information contribution: +{info:.6f}",
                    f"Travel penalty: −{penalty:.6f} × {distance:.6f} = −{penalty * distance:.6f}",
                    f"Sum: {info:.6f} − {penalty * distance:.6f} = {calculated:.6f}"]),
            _result(calculated, score),
        )
    if planner == "FoV-aware directional frontier":
        ordered = (
            "info_utility", "frontier_norm", "align", "hazard", "length_norm",
            "repetition", "turn", "multi",
        )
        breakdown = "\n".join(
            f"{key} = {values[key]:.3f}" for key in ordered if key in values
        ) or str(candidate.get("reason", "No terms captured"))
        required = (
            "info_utility", "frontier_norm", "align", "hazard", "length_norm",
            "repetition", "turn", "multi",
        )
        calculated = None
        if all(key in values for key in required):
            calculated = (
                3.0 * values["info_utility"] + 0.7 * values["frontier_norm"]
                + 1.2 * values["align"] + 4.0 * values["hazard"]
                - values["length_norm"] - 2.2 * values["repetition"]
                - values["turn"]
                - 1.2 * values["multi"]
            )
        coefficients = {
            "info_utility": 3.0, "frontier_norm": 0.7, "align": 1.2,
            "hazard": 4.0, "length_norm": -1.0, "repetition": -2.2,
            "turn": -1.0, "multi": -1.2,
        }
        step_lines = [
            f"{key}: {coefficient:+.2f} × {values[key]:.6f} = {coefficient * values[key]:+.6f}"
            for key, coefficient in coefficients.items() if key in values
        ]
        if calculated is not None:
            step_lines.append("Sum all signed contributions = " f"{calculated:.6f}")
        return (
            "<b>S(F)</b> = 3I + 0.7F + 1.2A + 4H − L − 2.2R − T − 1.2M'",
            _table([(key, f"{values[key]:.6f}") for key in ordered if key in values]),
            _steps(step_lines),
            _result(calculated, score),
        )
    if planner == "Goal seeking":
        return ("<b>target</b> = G", "Final mission goal", _steps(["Read configured goal G.", "Return G without frontier ranking."]), "No score")
    return (
        "Formula unavailable for this coordinator/planner",
        str(candidate.get("reason", "No candidate details captured")),
        _steps(["The active algorithm did not export computational terms."]),
        f"<b>Reported score:</b> {score:.6f}",
    )


def frontier_formula_explanation(planner: str, candidate: dict | None) -> tuple[str, str, str, str]:
    """Explain symbols, derive variables, trace arithmetic, then substitute fully."""
    candidate = candidate or {}
    formula, variables, steps, final = _raw_frontier_formula_explanation(planner, candidate)
    values = _terms(candidate.get("reason", ""))
    size = int(candidate.get("size", 0))
    distance = float(candidate.get("distance", 0.0))
    info = float(candidate.get("information_gain", 0.0))
    score = float(candidate.get("score", 0.0))

    if planner in {"Nearest frontier", "Nav2D nearest-frontier wavefront"}:
        variables = _table([
            (_variable_link("R", "R"), "current robot position"),
            (_variable_link("F", "F"), "candidate frontier position"),
            (_variable_link("robot_distance", "d(R,F)"), f"distance used for ranking = {distance:.6f} m"),
            (_variable_link("cluster_size", "|C<sub>F</sub>|"), f"tie-break cluster size = {size}"),
        ])
    elif planner == "Largest frontier":
        variables = _table([
            (_variable_link("F", "F"), "candidate frontier position"),
            (_variable_link("cluster_size", "|C<sub>F</sub>|"), f"connected frontier cells = {size}"),
            (_variable_link("robot_distance", "d(R,F)"), f"tie-break distance = {distance:.6f} m"),
        ])
    elif planner == "Utility frontier":
        goal = values.get("goal_distance", 0.0)
        calculated = size - 0.75 * distance - 0.15 * goal
        variables = _table([
            (_variable_link("R", "R"), "current robot position"),
            (_variable_link("F", "F"), "candidate frontier position"),
            (_variable_link("G", "G"), "final mission-goal position"),
            (_variable_link("cluster_size", "|C<sub>F</sub>|") , f"cells in F's cluster = {size}"),
            (_variable_link("robot_distance", "d(R,F)"), f"Euclidean distance from R to F = {distance:.6f} m"),
            (_variable_link("goal_distance", "d(F,G)"), f"Euclidean distance from F to G = {goal:.6f} m"),
        ])
        final = (
            f"<b>U(F)</b> = {size} &minus; 0.75({distance:.6f}) &minus; "
            f"0.15({goal:.6f}) = <b>{calculated:.6f}</b><hr>{_result(calculated, score)}"
        )
    elif planner == "Informative frontier / IPP-lite":
        penalty = (info - score) / distance if distance > 1e-9 else 0.0
        calculated = info - penalty * distance
        variables = _table([
            (_variable_link("R", "R"), "current robot position"),
            (_variable_link("F", "F"), "candidate frontier position"),
            (_variable_link("information_gain", "I(F)"), f"estimated newly observable information at F = {info:.6f}"),
            (_variable_link("robot_distance", "d(R,F)"), f"Euclidean distance from R to F = {distance:.6f} m"),
            (_variable_link("lambda", "&lambda;"), f"travel-cost weight = {penalty:.6f}"),
        ])
        final = (
            f"<b>U(F)</b> = {info:.6f} &minus; {penalty:.6f}({distance:.6f}) = "
            f"<b>{calculated:.6f}</b><hr>{_result(calculated, score)}"
        )
    elif planner == "FoV-aware directional frontier":
        descriptions = {
            "info_utility": "normalized discovery utility I",
            "frontier_norm": "normalized frontier size F",
            "align": "heading/frontier alignment A",
            "hazard": "Gaussian attraction to discovered fires H",
            "length_norm": "normalized route length L",
            "repetition": "combined FoV/path repetition penalty R",
            "turn": "turning penalty T",
            "multi": "reserved-target interference penalty M'",
        }
        variables = _table([
            (_variable_link(key, key), f"{descriptions[key]} = {values[key]:.6f}")
            for key in descriptions if key in values
        ])
        coefficients = {
            "info_utility": 3.0, "frontier_norm": 0.7, "align": 1.2,
            "hazard": 4.0, "length_norm": -1.0, "repetition": -2.2,
            "turn": -1.0, "multi": -1.2,
        }
        present = [(key, coefficients[key], values[key]) for key in coefficients if key in values]
        calculated = sum(coef * value for _key, coef, value in present) if present else None
        final = "<b>S(F)</b> = " + " ".join(
            f"{coef:+.2f}({value:.6f})" for _key, coef, value in present
        )
        if calculated is not None:
            final += f" = <b>{calculated:.6f}</b>"
        final += "<hr>" + _result(calculated, score)
    return formula, variables, steps, final


class FrontierReasoningPanel(QFrame):
    closeRequested = Signal()
    robotSelected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("frontierReasoningPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._theme_mode = ThemeMode.LIGHT
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        header.setContentsMargins(14, 12, 10, 10)
        title = QLabel("Frontier Reasoning")
        title.setObjectName("frontierReasoningTitle")
        header.addWidget(title, 1)
        self.robot_selector = QComboBox()
        self.robot_selector.setObjectName("frontierRobotSelector")
        self.robot_selector.addItem("R1")
        self.robot_selector.currentIndexChanged.connect(self._coordination_robot_changed)
        header.addWidget(self.robot_selector)
        close = QPushButton("×")
        close.setObjectName("frontierReasoningClose")
        close.setFixedSize(28, 26)
        close.clicked.connect(self.closeRequested.emit)
        header.addWidget(close)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setObjectName("frontierReasoningScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("frontierReasoningContent")
        body.setAttribute(Qt.WA_StyledBackground, True)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 8, 10, 12)
        layout.setSpacing(9)

        self.summary = self._label("Waiting for a frontier decision", "frontierSummary")
        self.formula = self._card(layout, "SYMBOLIC FORMULA", "formula", rich=True)
        self.formula.setObjectName("frontierFormulaValue")
        self.substitution = self._card(layout, "VARIABLES AND HOW THEY ARE CALCULATED", "substitution", rich=True)
        self.substitution.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.substitution.setOpenExternalLinks(False)
        self.substitution.linkHovered.connect(self._show_variable_tooltip)
        self.procedure = self._card(layout, "STEP-BY-STEP COMPUTATION", "procedure", rich=True)
        self.result = self._card(layout, "COMPLETE FORMULA WITH REAL VALUES", "result", rich=True)
        # Compatibility with callers that used the former real-values card.
        self.breakdown = self.substitution
        candidate_card = QFrame()
        candidate_card.setObjectName("frontierReasoningCard")
        candidate_card.setAttribute(Qt.WA_StyledBackground, True)
        candidate_layout = QVBoxLayout(candidate_card)
        candidate_header = QHBoxLayout()
        candidate_header.addWidget(self._label("CANDIDATE RANKING", "frontierCardTitle"), 1)
        self.candidate_map_view = QComboBox()
        self.candidate_map_view.setObjectName("frontierCandidateMapView")
        self.candidate_map_view.addItems(["Frontiers", "Clusters"])
        self.candidate_map_view.setToolTip(
            "Frontiers: show individual frontier cells. Clusters: color each connected component."
        )
        self.candidate_previous = QPushButton("<")
        self.candidate_previous.setObjectName("frontierCandidatePrevious")
        self.candidate_position = QLabel("0 / 0")
        self.candidate_position.setObjectName("frontierCandidatePosition")
        self.candidate_position.setAlignment(Qt.AlignCenter)
        self.candidate_next = QPushButton(">")
        self.candidate_next.setObjectName("frontierCandidateNext")
        candidate_header.addWidget(self.candidate_map_view)
        candidate_header.addWidget(self.candidate_previous)
        candidate_header.addWidget(self.candidate_position)
        candidate_header.addWidget(self.candidate_next)
        candidate_layout.addLayout(candidate_header)
        self.candidates = self._label("No candidates", "frontierCardValue")
        candidate_layout.addWidget(self.candidates)
        layout.addWidget(candidate_card)
        self._ranked_candidates = []
        self._cluster_source_candidates = []
        self._selected_target = None
        self._candidate_index = -1
        self.candidate_previous.clicked.connect(lambda: self._step_candidate(-1))
        self.candidate_next.clicked.connect(lambda: self._step_candidate(1))
        self.candidate_map_view.currentTextChanged.connect(self._candidate_map_view_changed)
        self._refresh_candidate_inspection()
        self._coordination_update = None
        layout.insertWidget(0, self.summary)
        layout.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)
        self.set_theme_mode(self._theme_mode)

    @staticmethod
    def _label(text: str, name: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName(name)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return label

    def _card(self, layout: QVBoxLayout, title: str, attr: str, *, rich: bool = False) -> QLabel:
        card = QFrame()
        card.setObjectName("frontierReasoningCard")
        card.setAttribute(Qt.WA_StyledBackground, True)
        card_layout = QVBoxLayout(card)
        heading = self._label(title, "frontierCardTitle")
        value = self._label("—", "frontierCardValue")
        value.setTextFormat(Qt.RichText if rich else Qt.PlainText)
        card_layout.addWidget(heading)
        card_layout.addWidget(value)
        layout.addWidget(card)
        setattr(self, attr, value)
        return value

    def set_theme_mode(self, mode: ThemeMode | str) -> None:
        self._theme_mode = ThemeMode(mode)
        c = theme_colors(self._theme_mode)
        self.setStyleSheet(f"""
            QFrame#frontierReasoningPanel {{ background: {c.card_background}; border: none; }}
            QScrollArea#frontierReasoningScroll {{ background: {c.app_background}; border: none; }}
            QWidget#frontierReasoningContent {{ background: {c.app_background}; }}
            QLabel {{ color: {c.text_primary}; background: transparent; }}
            QLabel#frontierReasoningTitle {{ font-size: 15px; font-weight: 900; }}
            QLabel#frontierSummary {{ color: {c.accent}; font-weight: 800; padding: 8px; }}
            QFrame#frontierReasoningCard {{ background: {c.panel_background}; border: 1px solid {c.border}; border-radius: 9px; }}
            QLabel#frontierCardTitle {{ color: {c.accent}; font-size: 10px; font-weight: 900; }}
            QLabel#frontierCardValue {{ font-family: Consolas, monospace; font-size: 9px; }}
            QLabel#frontierFormulaValue {{ font-family: "Segoe UI"; font-size: 13px; font-weight: 700; padding: 6px; }}
            QPushButton#frontierReasoningClose {{ border: none; background: transparent; color: {c.text_secondary}; font-size: 18px; }}
            QPushButton#frontierCandidatePrevious, QPushButton#frontierCandidateNext {{
                min-width: 28px; min-height: 24px; border: 1px solid {c.border}; border-radius: 6px;
                background: {c.card_background}; color: {c.text_primary}; font-weight: 900;
            }}
            QPushButton#frontierCandidatePrevious:hover, QPushButton#frontierCandidateNext:hover {{
                border-color: {c.accent}; color: {c.accent};
            }}
            QLabel#frontierCandidatePosition {{ color: {c.text_secondary}; min-width: 44px; font-weight: 800; }}
            QComboBox#frontierCandidateMapView {{ min-height: 24px; border: 1px solid {c.border};
                border-radius: 6px; background: {c.card_background}; color: {c.text_primary}; padding: 0 6px; }}
            QComboBox#frontierRobotSelector {{ min-width: 64px; min-height: 24px; border: 1px solid {c.border};
                border-radius: 6px; background: {c.card_background}; color: {c.text_primary}; padding: 0 6px; }}
        """)

    def set_robot_selector(self, index: int, count: int) -> None:
        self.robot_selector.blockSignals(True)
        self.robot_selector.clear()
        self.robot_selector.addItems([f"R{i + 1}" for i in range(max(1, int(count)))])
        self.robot_selector.setCurrentIndex(max(0, min(int(index), self.robot_selector.count() - 1)))
        self.robot_selector.setVisible(int(count) > 1)
        self.robot_selector.blockSignals(False)
        if self._coordination_update is not None:
            self._render_coordination(self.robot_selector.currentIndex())

    def _coordination_robot_changed(self, index: int) -> None:
        if index >= 0:
            self.robotSelected.emit(index)
            if self._coordination_update is not None:
                self._render_coordination(index)

    def _candidate_map_view_changed(self, text: str) -> None:
        canvas = getattr(self.window(), "canvas", None)
        if canvas is not None and hasattr(canvas, "set_frontier_reasoning_cluster_view_enabled"):
            canvas.set_frontier_reasoning_cluster_view_enabled(str(text) == "Clusters")

    def _step_candidate(self, delta: int) -> None:
        if not self._ranked_candidates:
            return
        self._candidate_index = (self._candidate_index + int(delta)) % len(self._ranked_candidates)
        self._refresh_candidate_inspection()

    def _refresh_candidate_inspection(self) -> None:
        count = len(self._ranked_candidates)
        valid = count > 0 and 0 <= self._candidate_index < count
        self.candidate_position.setText(f"{self._candidate_index + 1 if valid else 0} / {count}")
        self.candidate_previous.setEnabled(count > 1)
        self.candidate_next.setEnabled(count > 1)
        inspected = self._ranked_candidates[self._candidate_index] if valid else None
        lines = []
        for index, item in enumerate(self._ranked_candidates):
            selected_mark = "✓" if self._selected_target is not None and tuple(item.target) == self._selected_target else " "
            inspected_mark = "▶" if index == self._candidate_index else " "
            reason = str(item.reason)
            reachability = (
                reason.split("reachability_reason=", 1)[-1]
                if "reachability_reason=" in reason else "reachability reason not reported"
            )
            lines.append(
                f"{inspected_mark}{selected_mark} {index + 1}. {tuple(round(v, 2) for v in item.target)}  "
                f"score={item.score:.3f}  d={item.distance_from_robot:.2f}  size={item.size}  info={item.information_gain:.1f}\n"
                f"    {reachability}"
            )
        self.candidates.setText("\n".join(lines) or "No candidates")
        # QTabWidget reparents pages into its internal QStackedWidget, so
        # parent() is not the MainWindow once the panel is mounted.
        canvas = getattr(self.window(), "canvas", None)
        if canvas is not None and hasattr(canvas, "set_frontier_reasoning_inspection"):
            canvas.set_frontier_reasoning_inspection(None if inspected is None else {
                "frontier": tuple(inspected.target), "index": self._candidate_index + 1, "count": count,
            })
        if canvas is not None and hasattr(canvas, "set_frontier_reasoning_clusters"):
            unique_clusters = {}
            for item in self._cluster_source_candidates:
                points = tuple(getattr(item, "cluster_points", ()) or ())
                if not points:
                    continue
                key = tuple(sorted((round(float(x), 6), round(float(y), 6)) for x, y in points))
                unique_clusters.setdefault(key, {
                    "points": points,
                    "resolution": float(getattr(item, "cluster_resolution", 0.0) or 0.0),
                })
            canvas.set_frontier_reasoning_clusters(tuple(unique_clusters.values()))

    def _show_variable_tooltip(self, href: str) -> None:
        key = str(href).removeprefix("var://")
        explanation = _VARIABLE_HELP.get(key)
        if explanation:
            QToolTip.showText(QCursor.pos(), explanation, self.substitution)
        else:
            QToolTip.hideText()

    def update_decision(
        self, *, planner: str, result, robot_label: str, time_s: float,
        robot_xy: tuple[float, float] | None = None,
        configured_planner: str | None = None,
        attempt_role: str | None = None,
    ) -> None:
        self._coordination_update = None
        selected = None
        candidates = list(getattr(result, "candidates", ()) or ())
        target = getattr(result, "target", None)
        if target is not None:
            selected = min(
                candidates,
                key=lambda item: (item.target[0] - target[0]) ** 2 + (item.target[1] - target[1]) ** 2,
                default=None,
            )
        selected_dict = None if selected is None else {
            "target": tuple(selected.target), "score": float(selected.score),
            "size": int(selected.size), "distance": float(selected.distance_from_robot),
            "information_gain": float(selected.information_gain), "reason": str(selected.reason),
        }
        if selected_dict is None:
            formula = "<b>No frontier was selected.</b>"
            substitution = "No selected candidate; zero-valued substitutions would be misleading."
            procedure = html.escape(str(getattr(result, "reason", "No decision reason reported")))
            result_audit = "<b>Result:</b> no executable frontier"
        else:
            formula, substitution, procedure, result_audit = frontier_formula_explanation(planner, selected_dict)
        configured = str(configured_planner or planner)
        role = str(attempt_role or ("configured planner" if configured == planner else "map-wide fallback"))
        self.summary.setText(
            f"{robot_label} · t={time_s:.2f}s\nConfigured planner: {configured}\n"
            f"Attempt shown: {planner} ({role})\nSelected: {target}\n{getattr(result, 'reason', '')}"
        )
        self.formula.setText(formula)
        self.substitution.setText(substitution)
        self.procedure.setText(procedure)
        self.result.setText(result_audit)
        owner = self.window()
        canvas = getattr(owner, "canvas", None)
        if canvas is not None and hasattr(canvas, "set_frontier_reasoning_decision"):
            terms = _terms(selected_dict.get("reason", "")) if selected_dict else {}
            concise_terms = [
                f"{name}={value:.3f}" for name, value in terms.items()
                if name not in {"score", "distance", "goal_distance"}
            ]
            canvas.set_frontier_reasoning_decision(None if selected_dict is None else {
                "robot": robot_xy,
                "frontier": selected_dict["target"],
                "distance": selected_dict["distance"],
                "planner": planner,
                "terms": concise_terms,
            })
        if planner in {"Nav2D nearest-frontier wavefront", "Nearest frontier"}:
            ranked = sorted(candidates, key=lambda item: (float(item.distance_from_robot), -int(item.size)))
        elif planner == "Largest frontier":
            ranked = sorted(candidates, key=lambda item: (-int(item.size), float(item.distance_from_robot)))
        else:
            ranked = sorted(
                candidates,
                key=lambda item: (-float(item.score), -float(item.information_gain), -int(item.size), float(item.distance_from_robot)),
            )
        self._cluster_source_candidates = ranked
        self._ranked_candidates = ranked[:12]
        self._selected_target = tuple(target) if target is not None else None
        self._candidate_index = next(
            (index for index, item in enumerate(self._ranked_candidates)
             if tuple(item.target) == self._selected_target),
            0 if self._ranked_candidates else -1,
        )
        self._refresh_candidate_inspection()

    def clear(self) -> None:
        """Remove decisions from the previous run, including canvas focus."""
        self._coordination_update = None
        self.summary.setText("Waiting for a frontier decision")
        for label in (self.formula, self.substitution, self.procedure, self.result):
            label.setText("—")
        self._ranked_candidates = []
        self._cluster_source_candidates = []
        self._selected_target = None
        self._candidate_index = -1
        self._refresh_candidate_inspection()
        canvas = getattr(self.window(), "canvas", None)
        if canvas is not None and hasattr(canvas, "set_frontier_reasoning_decision"):
            canvas.set_frontier_reasoning_decision(None)

    def restore_from_snapshot(self, *, snapshot, configured_planner: str, robot_label: str = "R1") -> None:
        """Clear discarded-future content and identify the restored decision."""
        frontier = getattr(snapshot, "frontier", None)
        configured_field = getattr(frontier, "configured_planner", None)
        configured_value = (
            configured_field.value
            if configured_field is not None and not getattr(configured_field, "unavailable", True)
            else configured_planner
        )
        effective = getattr(frontier, "effective_planner", None)
        effective_value = (
            effective.value if effective is not None and not getattr(effective, "unavailable", True)
            else configured_value
        )
        role_field = getattr(frontier, "attempt_role", None)
        role = (
            role_field.value if role_field is not None and not getattr(role_field, "unavailable", True)
            else ("configured planner" if effective_value == configured_value else "map-wide fallback")
        )
        reason_field = getattr(frontier, "reason", None)
        reason = (
            reason_field.value if reason_field is not None and not getattr(reason_field, "unavailable", True)
            else "No frontier decision captured in this snapshot."
        )
        self.summary.setText(
            f"{robot_label} · t={float(snapshot.simulation_time):.2f}s · RESTORED SNAPSHOT\n"
            f"Configured planner: {configured_value}\nAttempt shown: {effective_value} ({role})\n{reason}"
        )
        self.formula.setText("Restored snapshot: waiting for the next live frontier computation.")
        self.substitution.setText("No candidate values are reconstructed after rollback.")
        self.procedure.setText("The discarded future decision was cleared.")
        self.result.setText("<b>Snapshot synchronized.</b>")
        self._ranked_candidates = []
        self._cluster_source_candidates = []
        self._selected_target = None
        self._candidate_index = -1
        self._refresh_candidate_inspection()
        self.candidates.setText("Candidate list was not retained by this snapshot schema.")

    def update_coordination(
        self, *, planner: str, coordinator: str, result, robot_index: int, time_s: float,
        runtime_profile=None, robot_positions=(),
    ) -> None:
        """Cache the complete team decision and render the inspected robot."""
        self._coordination_update = (
            planner, coordinator, result, float(time_s), runtime_profile,
            tuple(robot_positions or ()),
        )
        count = max(len(getattr(result, "targets", ()) or ()), 1)
        self.set_robot_selector(robot_index, count)

    def _render_coordination(self, robot_index: int) -> None:
        planner, coordinator, result, time_s, runtime_profile, robot_positions = self._coordination_update
        targets = list(getattr(result, "targets", ()) or ())
        reasons = list(getattr(result, "reasons", ()) or ())
        selected = targets[robot_index] if 0 <= robot_index < len(targets) else None
        reason = reasons[robot_index] if 0 <= robot_index < len(reasons) else "No per-robot reason exported"
        assignment = next(
            (
                item for item in tuple(getattr(result, "assignments", ()) or ())
                if int(getattr(item, "robot_id", -1)) == int(robot_index)
            ),
            None,
        )
        proposal = getattr(assignment, "proposal", None)
        metadata = dict(getattr(proposal, "metadata", {}) or {}) if proposal is not None else {}
        robot_xy = (
            tuple(robot_positions[robot_index])
            if 0 <= robot_index < len(robot_positions) else None
        )
        parsed = _terms(f"{reason}; {metadata.get('reason', '')}")
        distance = (
            math.dist(robot_xy, selected)
            if robot_xy is not None and selected is not None
            else float(parsed.get("dist", parsed.get("distance", metadata.get("distance", 0.0))))
        )
        information_gain = float(
            getattr(proposal, "information_gain", parsed.get("info_gain", 0.0))
            if proposal is not None else parsed.get("info_gain", 0.0)
        )
        size = int(metadata.get(
            "frontier_size",
            metadata.get("reduced_frontier_cell_count", parsed.get("size", 1 if selected is not None else 0)),
        ))
        score = float(metadata.get(
            "score",
            metadata.get("raw_score", metadata.get("assignment_utility", parsed.get("score", 0.0))),
        ))
        candidate_reason = f"{reason}; {metadata.get('reason', '')}".strip("; ")
        selected_dict = None if selected is None else {
            "target": tuple(selected),
            "score": score,
            "size": size,
            "distance": distance,
            "information_gain": information_gain,
            "reason": candidate_reason,
        }
        # Provenance is read from candidate_input_mode, not the deprecated
        # owns_target_generation. The owns_target_generation fallback below
        # only exists for callers/test doubles that still pass a bare
        # profile stand-in without the new field.
        candidate_input_mode = getattr(runtime_profile, "candidate_input_mode", None)
        if candidate_input_mode is not None:
            coordinator_owns_generation = candidate_input_mode != CandidateInputMode.LEGACY_INTEGRATED
        else:
            coordinator_owns_generation = bool(getattr(runtime_profile, "owns_target_generation", False))
        self.summary.setText(
            f"R{robot_index + 1} · t={time_s:.2f}s · {planner}\n"
            f"Coordinator: {coordinator}\nSelected: {selected}\n{reason}"
        )
        if selected_dict is None:
            self.formula.setText("<b>No frontier was assigned to this robot.</b>")
            self.substitution.setText("No selected frontier; zero-valued substitutions would be misleading.")
            self.procedure.setText(html.escape(str(reason)))
            self.result.setText(f"<b>R{robot_index + 1}: HOLD / no frontier</b>")
        elif not coordinator_owns_generation:
            formula, substitution, procedure, result_audit = frontier_formula_explanation(
                planner, selected_dict
            )
            self.formula.setText(formula)
            self.substitution.setText(substitution)
            self.procedure.setText(procedure)
            self.result.setText(result_audit)
        else:
            source_description_template = _CANDIDATE_SOURCE_DESCRIPTIONS.get(
                candidate_input_mode, _DEFAULT_CANDIDATE_SOURCE_DESCRIPTION
            )
            source_description = source_description_template.format(coordinator=html.escape(str(coordinator)))
            self.formula.setText(
                f"<b>{source_description}</b><br>"
                "The configured single-robot exploration formula is not substituted because it did not own this target."
            )
            rows = [
                ("Rᵢ", f"R{robot_index + 1} at {robot_xy}"),
                ("Fᵢ", str(selected)),
                ("I(Fᵢ)", f"{information_gain:.6f}"),
                ("d(Rᵢ,Fᵢ)", f"{distance:.6f} m"),
                (
                    "candidate_input_mode",
                    str(getattr(candidate_input_mode, "value", candidate_input_mode or "unavailable")),
                ),
                ("selected proposal source", str(getattr(proposal, "source", "not exported"))),
            ]
            self.substitution.setText(_table(rows) + "<pre>" + html.escape(
                json.dumps(metadata, indent=2, default=str, ensure_ascii=False)
            ) + "</pre>")
            self.procedure.setText(
                f"1. {source_description}<br>"
                f"2. {html.escape(str(coordinator))} selected the proposal for R{robot_index + 1}.<br>"
                "3. The coordinator then applied team reservations/assignment constraints.<br>"
                "4. See Coordinator Reasoning for the plugin-native score and team allocation."
            )
            self.result.setText(
                f"<b>R{robot_index + 1} frontier = {html.escape(str(selected))}</b><br>"
                f"status={html.escape(str(getattr(assignment, 'status', 'unknown')))}<br>"
                f"{html.escape(str(reason))}"
            )

        self._cluster_source_candidates = []
        self._ranked_candidates = []
        if selected_dict is not None:
            self._ranked_candidates = [SimpleNamespace(
                target=selected_dict["target"],
                score=selected_dict["score"],
                size=selected_dict["size"],
                distance_from_robot=selected_dict["distance"],
                information_gain=selected_dict["information_gain"],
                reason=f"{selected_dict['reason']}; reachability_reason=selected coordinator proposal",
            )]
        self._selected_target = tuple(selected) if selected is not None else None
        self._candidate_index = 0 if self._ranked_candidates else -1
        self._refresh_candidate_inspection()
        if self._ranked_candidates:
            self.candidates.setText(
                self.candidates.text()
                + "\n\nOnly the selected proposal was exported by this coordinator decision; "
                  "a complete per-robot ranking is not available."
            )
        canvas = getattr(self.window(), "canvas", None)
        if canvas is not None and hasattr(canvas, "set_frontier_reasoning_decision"):
            canvas.set_frontier_reasoning_decision(None if selected is None else {
                "robot": robot_xy,
                "frontier": tuple(selected),
                "distance": distance,
                "planner": planner,
                "terms": [f"R{robot_index + 1}", f"coordinator={coordinator}"],
            })
