"""Paper-level contracts for Keidar and Kaminka's WFD-INC detector."""

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.planning.exploration_planners import (
    EXPLORATION_PLANNER_OPTIONS,
    detect_frontier_cells_for_planner,
)
from robotics_sim.planning.keidar_kaminka_efd import (
    KEIDAR_KAMINKA_EFD_CITATION,
    KEIDAR_KAMINKA_WFD_INC,
    WFDIncrementalDetector,
)
from robotics_sim.simulation.config import FRONTIER_ALGORITHM_DETECTOR_OPTIONS


def _belief() -> tuple[BeliefMap, tuple[float, float]]:
    belief = BeliefMap(bounds=(-4.0, 4.0, -4.0, 4.0), resolution=1.0)
    for cell in ((4, 2), (4, 3), (4, 4), (4, 5), (3, 4), (2, 4)):
        belief.mark_free_cell(cell)
    return belief, belief.cell_to_world((4, 2))


def test_wfd_inc_is_cited_and_visible_in_frontier_detector_selector():
    assert KEIDAR_KAMINKA_WFD_INC in EXPLORATION_PLANNER_OPTIONS
    assert KEIDAR_KAMINKA_WFD_INC in FRONTIER_ALGORITHM_DETECTOR_OPTIONS
    assert "10.1177/0278364913494911" in KEIDAR_KAMINKA_EFD_CITATION


def test_first_wfd_inc_call_scans_known_space_and_returns_frontiers():
    belief, robot_xy = _belief()
    result = WFDIncrementalDetector().detect(belief, robot_xy)

    assert result.full_scan is True
    assert result.scanned_cells > 0
    assert result.frontier_cells


def test_dispatch_uses_wfd_inc_for_the_selected_detector():
    belief, robot_xy = _belief()
    detected = detect_frontier_cells_for_planner(
        KEIDAR_KAMINKA_WFD_INC,
        belief=belief,
        robot_xy=robot_xy,
    )

    assert detected
    assert all(belief.grid[row, col] == 0 for row, col in detected)
