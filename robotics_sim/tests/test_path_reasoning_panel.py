from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from robotics_sim.app.path_reasoning_panel import PathReasoningPanel


_app = QApplication.instance() or QApplication([])


def test_astar_path_reasoning_shows_exact_cost_and_path_stages(monkeypatch):
    panel = PathReasoningPanel()
    capture = SimpleNamespace(
        raw_world_path=((0.0, 0.0), (1.0, 0.0), (2.0, 1.0)),
        simplified_world_path=((0.0, 0.0), (2.0, 1.0)),
        total_cost=2.414214,
        unknown_is_traversable=True,
    )
    panel.update_route(
        planner="A*", simplifier="Line of sight grid-safe", success=True,
        reason="path found", capture=capture, waypoints=((2.0, 1.0),),
        start_xy=(0.0, 0.0), goal_xy=(2.0, 1.0), time_s=3.0,
    )

    assert "g(n)" in panel.formula.text()
    assert "h<sub>octile</sub>" in panel.formula.text()
    assert "2.414214" in panel.result.text()
    assert "raw points 3 → simplified points 2 → executable points 1" in panel.result.text()
    assert "RAW:" in panel.path.text()
    assert "SIMPLIFIED:" in panel.path.text()

    panel.update_live_pose((0.75, 0.25))
    assert "R1 current R(t)" in panel.live_state.text()
    assert "(0.750000, 0.250000)" in panel.live_state.text()
    assert "R<sub>plan</sub> = (0.0, 0.0)" in panel.live_state.text()
    assert "R_plan=(0.0, 0.0)" in panel.result.text()


def test_path_variable_hover_explains_astar_priority(monkeypatch):
    panel = PathReasoningPanel()
    shown = []
    monkeypatch.setattr(
        "robotics_sim.app.path_reasoning_panel.QToolTip.showText",
        lambda *args: shown.append(args[1]),
    )
    panel._show_help("pathvar://h")
    assert shown
    assert "octile" in shown[0]
    assert "Dijkstra" in shown[0]


def test_multi_robot_routes_and_live_poses_are_kept_separately():
    panel = PathReasoningPanel()
    panel.set_robot_selector(0, 2)
    panel.update_route(
        planner="A*", simplifier="Direction changes", success=True,
        reason="R1 route", waypoints=((1.0, 0.0),),
        start_xy=(0.0, 0.0), goal_xy=(1.0, 0.0), time_s=1.0, robot_index=0,
    )
    panel.update_route(
        planner="Dijkstra", simplifier="Line of sight grid-safe", success=False,
        reason="R2 no path", waypoints=(),
        start_xy=(5.0, 5.0), goal_xy=(8.0, 5.0), time_s=2.0, robot_index=1,
    )
    panel.update_live_pose((0.25, 0.0), robot_label="R1", robot_index=0)
    panel.update_live_pose((5.5, 5.0), robot_label="R2", robot_index=1)

    assert "R1 route" in panel.summary.text()
    assert "(0.250000, 0.000000)" in panel.live_state.text()
    panel.robot_selector.setCurrentIndex(1)
    assert "R2 no path" in panel.summary.text()
    assert "REJECTED" in panel.summary.text()
    assert "(5.500000, 5.000000)" in panel.live_state.text()

    panel.clear()
    assert panel._routes_by_robot == {}
    assert panel._poses_by_robot == {}


def test_clear_removes_previous_route_reasoning():
    panel = PathReasoningPanel()
    panel.summary.setText("old route")
    panel.result.setText("old result")
    panel.clear()
    assert "Waiting" in panel.summary.text()
    assert panel.result.text() == "—"
