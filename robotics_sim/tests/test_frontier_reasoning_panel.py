from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTabWidget, QWidget

from robotics_sim.app.frontier_reasoning_panel import (
    FrontierReasoningPanel,
    frontier_formula_explanation,
)
from robotics_sim.app.theme import ThemeMode, theme_colors
from robotics_sim.diagnostics.navigation_snapshot import FrontierDebug, Maybe


_app = QApplication.instance() or QApplication([])


def _candidate(**overrides):
    values = dict(
        target=(2.0, 3.0), score=4.25, size=8,
        distance_from_robot=2.0, information_gain=6.0,
        reason="frontier size=8, info_gain=6.0, distance=2.00, goal_distance=1.00, score=4.25",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_utility_formula_substitutes_the_real_candidate_values():
    formula, substitution, procedure, result = frontier_formula_explanation(
        "Utility frontier",
        {
            "score": 4.25, "size": 8, "distance": 2.0,
            "information_gain": 6.0,
            "reason": "goal_distance=1.00",
        },
    )

    assert "0.75" in formula
    assert "8" in substitution
    assert "2.00" in substitution
    assert "1.00" in substitution
    assert "current robot position" in substitution
    assert "candidate frontier position" in substitution
    assert "Euclidean distance" in substitution
    assert "6.350000" in procedure
    assert "Calculated:" in result
    assert "6.350000" in result
    assert "Reported:" in result
    assert "4.250000" in result
    assert "Delta:" in result
    assert "+2.100000" in result
    assert "CHECK MISMATCH" in result
    assert "0.75(2.000000)" in result
    assert "0.15(1.000000)" in result


def test_informative_formula_explains_symbols_and_finishes_with_full_substitution():
    formula, variables, steps, final = frontier_formula_explanation(
        "Informative frontier / IPP-lite",
        {"score": 4.0, "size": 5, "distance": 2.0, "information_gain": 6.0, "reason": ""},
    )

    assert "I(F)" in formula
    assert "current robot position" in variables
    assert "estimated newly observable information" in variables
    assert "Euclidean distance from R to F" in variables
    assert "Travel penalty" in steps
    assert "6.000000" in final
    assert "1.000000(2.000000)" in final
    assert "4.000000" in final


def test_panel_displays_selected_formula_values_and_candidate_ranking():
    panel = FrontierReasoningPanel()
    candidate = _candidate()
    result = SimpleNamespace(
        target=candidate.target,
        candidates=(candidate,),
        reason="Utility frontier selected candidate",
    )

    panel.update_decision(
        planner="Utility frontier", result=result, robot_label="R1", time_s=3.5,
    )

    assert "Utility frontier" in panel.summary.text()
    assert "8" in panel.substitution.text()
    assert "6.350000" in panel.procedure.text()
    assert "Reported:" in panel.result.text()
    assert "score=4.250" in panel.candidates.text()


def test_candidate_buttons_browse_ranking_and_highlight_frontier_on_canvas():
    class CanvasSpy:
        def __init__(self):
            self.inspected = []

        def set_frontier_reasoning_decision(self, _decision):
            pass

        def set_frontier_reasoning_inspection(self, candidate):
            self.inspected.append(candidate)

    owner = QWidget()
    owner.canvas = CanvasSpy()
    panel = FrontierReasoningPanel(owner)
    best = _candidate(target=(1.0, 1.0), score=5.0)
    second = _candidate(target=(4.0, 2.0), score=3.0)
    panel.update_decision(
        planner="Utility frontier",
        result=SimpleNamespace(target=best.target, candidates=(second, best), reason="selected"),
        robot_label="R1", time_s=1.0,
    )

    assert panel.candidate_position.text() == "1 / 2"
    assert owner.canvas.inspected[-1]["frontier"] == (1.0, 1.0)
    panel.candidate_next.click()
    assert panel.candidate_position.text() == "2 / 2"
    assert owner.canvas.inspected[-1]["frontier"] == (4.0, 2.0)
    assert "▶" in panel.candidates.text()
    panel.candidate_previous.click()
    assert owner.canvas.inspected[-1]["frontier"] == (1.0, 1.0)


def test_candidate_highlight_reaches_canvas_after_tab_widget_reparents_panel():
    class CanvasSpy:
        def __init__(self):
            self.inspected = None

        def set_frontier_reasoning_decision(self, _decision):
            pass

        def set_frontier_reasoning_inspection(self, candidate):
            self.inspected = candidate

    window = QWidget()
    window.canvas = CanvasSpy()
    tabs = QTabWidget(window)
    panel = FrontierReasoningPanel(window)
    tabs.addTab(panel, "Frontiers")  # reparents panel into QTabWidget internals
    candidate = _candidate(target=(7.0, -2.0))

    panel.update_decision(
        planner="Utility frontier",
        result=SimpleNamespace(target=candidate.target, candidates=(candidate,), reason="selected"),
        robot_label="R1", time_s=1.0,
    )

    assert panel.parent() is not window
    assert window.canvas.inspected["frontier"] == (7.0, -2.0)


def test_cluster_view_exports_unique_real_clusters_to_canvas():
    class CanvasSpy:
        def __init__(self):
            self.clusters = ()
            self.cluster_view = False

        def set_frontier_reasoning_decision(self, _decision): pass
        def set_frontier_reasoning_inspection(self, _candidate): pass
        def set_frontier_reasoning_clusters(self, clusters): self.clusters = clusters
        def set_frontier_reasoning_cluster_view_enabled(self, enabled): self.cluster_view = enabled

    window = QWidget()
    window.canvas = CanvasSpy()
    panel = FrontierReasoningPanel(window)
    points = ((1.0, 1.0), (1.25, 1.0))
    first = _candidate(target=points[0], score=2.0, reason="kind=frontier", cluster_points=points, cluster_resolution=0.25)
    second = _candidate(target=points[1], score=1.0, reason="kind=frontier", cluster_points=points, cluster_resolution=0.25)
    panel.update_decision(
        planner="FoV-aware directional frontier",
        result=SimpleNamespace(target=first.target, candidates=(first, second), reason="selected"),
        robot_label="R1", time_s=1.0,
    )

    assert len(window.canvas.clusters) == 1
    assert window.canvas.clusters[0]["points"] == points
    panel.candidate_map_view.setCurrentText("Clusters")
    assert window.canvas.cluster_view is True


def test_fallback_is_identified_without_fake_zero_substitution():
    panel = FrontierReasoningPanel()
    rejected = _candidate(
        reason="frontier size=8, reachability=rejected, reachability_reason=path planner failed: no path found"
    )
    result = SimpleNamespace(target=None, candidates=(rejected,), reason="all candidates rejected")
    panel.update_decision(
        planner="Nearest frontier", configured_planner="FoV-aware directional frontier",
        attempt_role="map-wide fallback", result=result, robot_label="R1", time_s=8.0,
    )

    assert "Configured planner: FoV-aware directional frontier" in panel.summary.text()
    assert "Attempt shown: Nearest frontier (map-wide fallback)" in panel.summary.text()
    assert "No frontier was selected" in panel.formula.text()
    assert "zero-valued substitutions would be misleading" in panel.substitution.text()
    assert "path planner failed: no path found" in panel.candidates.text()


def test_snapshot_restore_replaces_discarded_future_decision():
    panel = FrontierReasoningPanel()
    panel.summary.setText("R1 · t=88.74s · discarded future")
    snapshot = SimpleNamespace(
        simulation_time=85.70,
        frontier=FrontierDebug(
            candidate_count=Maybe.of(13), selected_target=Maybe.missing(),
            selected_score=Maybe.missing(), reason=Maybe.of("all 13 candidates rejected"),
            configured_planner=Maybe.of("FoV-aware directional frontier"),
            effective_planner=Maybe.of("Nearest frontier"),
            attempt_role=Maybe.of("map-wide fallback"),
        ),
    )
    panel.restore_from_snapshot(
        snapshot=snapshot, configured_planner="FoV-aware directional frontier"
    )

    assert "t=85.70s · RESTORED SNAPSHOT" in panel.summary.text()
    assert "t=88.74" not in panel.summary.text()
    assert "Attempt shown: Nearest frontier (map-wide fallback)" in panel.summary.text()
    assert "discarded future decision was cleared" in panel.procedure.text()


def test_fov_variables_are_hoverable_and_explain_the_actual_calculation(monkeypatch):
    panel = FrontierReasoningPanel()
    shown = []
    monkeypatch.setattr(
        "robotics_sim.app.frontier_reasoning_panel.QToolTip.showText",
        lambda *args: shown.append(args[1]),
    )
    formula, variables, _steps, _final = frontier_formula_explanation(
        "FoV-aware directional frontier",
        {
            "score": 1.0, "size": 5, "distance": 2.0, "information_gain": 4.0,
            "reason": "info_utility=0.8, frontier_norm=1, align=0.5, length_norm=0.1",
        },
    )
    panel.substitution.setText(variables)
    panel._show_variable_tooltip("var://info_utility")

    assert "var://info_utility" in panel.substitution.text()
    assert shown
    assert "0.40" in shown[0]
    assert "terminal_novelty" in shown[0]
    assert "UNKNOWN" in shown[0]


def test_fov_hazard_formula_audits_the_replacement_score_exactly():
    formula, variables, steps, final = frontier_formula_explanation(
        "FoV-aware directional frontier",
        {
            "score": 5.86,
            "size": 5,
            "distance": 2.0,
            "information_gain": 4.0,
            "reason": (
                "info_utility=0.8, frontier_norm=1, align=0.5, hazard=0.9, "
                "length_norm=0.1, repetition=0.2, turn=0.3, multi=0.5, "
                "score=5.86"
            ),
        },
    )

    assert "+ 4H" in formula
    assert "2.2R" in formula
    assert "detour" not in formula
    assert "backtrack" not in formula
    assert "target-switch" not in variables
    assert "Gaussian attraction" in variables
    assert "hazard: +4.00" in steps
    assert "CONSISTENT" in final


def test_light_theme_styles_panel_scroll_and_content_explicitly():
    panel = FrontierReasoningPanel()
    panel.set_theme_mode(ThemeMode.LIGHT)
    colors = theme_colors(ThemeMode.LIGHT)
    stylesheet = panel.styleSheet()

    assert f"background: {colors.app_background}" in stylesheet
    assert panel.testAttribute(Qt.WA_StyledBackground)
    content = panel.findChild(QWidget, "frontierReasoningContent")
    assert content is not None
    assert content.testAttribute(Qt.WA_StyledBackground)


def test_multi_robot_frontier_reasoning_switches_robot_and_canvas_target():
    class CanvasSpy:
        def __init__(self):
            self.decision = None
            self.inspection = None

        def set_frontier_reasoning_decision(self, decision): self.decision = decision
        def set_frontier_reasoning_inspection(self, inspection): self.inspection = inspection
        def set_frontier_reasoning_clusters(self, _clusters): pass

    owner = QWidget()
    owner.canvas = CanvasSpy()
    panel = FrontierReasoningPanel(owner)
    proposals = (
        SimpleNamespace(
            source="team_provider", information_gain=8.0,
            metadata={"frontier_size": 5, "score": 5.0, "reason": "goal_distance=2.0"},
        ),
        SimpleNamespace(
            source="team_provider", information_gain=14.0,
            metadata={"frontier_size": 9, "score": 9.0, "reason": "goal_distance=1.0"},
        ),
    )
    assignments = tuple(
        SimpleNamespace(robot_id=index, status="ASSIGNED", proposal=proposals[index])
        for index in range(2)
    )
    result = SimpleNamespace(
        targets=((1.0, 0.0), (8.0, 3.0)),
        reasons=("score=5.0", "score=9.0"),
        assignments=assignments,
        debug={},
    )
    panel.update_coordination(
        planner="Utility frontier", coordinator="NOIC information coordinator",
        result=result, robot_index=0, time_s=4.0,
        runtime_profile=SimpleNamespace(owns_target_generation=False),
        robot_positions=((0.0, 0.0), (6.0, 3.0)),
    )

    assert "R1" in panel.summary.text()
    assert owner.canvas.decision["frontier"] == (1.0, 0.0)
    assert owner.canvas.decision["robot"] == (0.0, 0.0)

    panel.robot_selector.setCurrentIndex(1)
    assert "R2" in panel.summary.text()
    assert "(8.0, 3.0)" in panel.summary.text()
    assert owner.canvas.decision["frontier"] == (8.0, 3.0)
    assert owner.canvas.decision["robot"] == (6.0, 3.0)
    assert "Only the selected proposal" in panel.candidates.text()
