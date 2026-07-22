from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from robotics_sim.app.coordinator_reasoning_panel import CoordinatorReasoningPanel


_app = QApplication.instance() or QApplication([])


def _result():
    assignments = (
        SimpleNamespace(robot_id=0, status="ASSIGNED", target=(1.0, 2.0), proposal=None),
        SimpleNamespace(robot_id=1, status="ASSIGNED", target=(7.0, -1.0), proposal=None),
    )
    return SimpleNamespace(
        targets=((1.0, 2.0), (7.0, -1.0)),
        reasons=("task=t0", "task=t1"),
        assignments=assignments,
        debug={
            "robots_to_assign": [0, 1],
            "task_ids": ["t0", "t1"],
            "weight_configuration": {"information": 1.0, "distance": 0.4, "obstacle": 2.0},
            "utility_matrix": [[4.0, 2.0], [1.0, 5.0]],
            "feasible_matrix": [[True, True], [True, True]],
            "selected_task_by_robot": {"0": "t0", "1": "t1"},
        },
    )


def test_hungarian_coordination_is_auditable_per_robot():
    panel = CoordinatorReasoningPanel()
    profile = SimpleNamespace(
        owns_target_generation=True,
        owns_task_allocation=True,
        owns_path_planning=False,
        owns_control=False,
    )
    panel.update_coordination(
        planner="FoV-aware directional frontier",
        coordinator="Frontier Cluster Hungarian",
        result=_result(),
        time_s=12.5,
        runtime_profile=profile,
    )

    assert "Inspecting R1" in panel.summary.text()
    assert "Hungarian" in panel.formula.text()
    assert "task columns" in panel.matrix.text()
    assert "R1:" in panel.assignments.text()
    assert "R2:" in panel.assignments.text()

    selected = []
    panel.robotSelected.connect(selected.append)
    panel.robot_selector.setCurrentIndex(1)
    assert selected == [1]
    assert "Inspecting R2" in panel.summary.text()
    assert "t1" in panel.variables.text()


def test_clear_removes_previous_team_decision():
    panel = CoordinatorReasoningPanel()
    panel.update_coordination(
        planner="Utility frontier",
        coordinator="Independent baseline",
        result=_result(),
        time_s=1.0,
    )
    panel.clear()
    assert "Waiting" in panel.summary.text()
    assert panel.assignments.text() == "—"
