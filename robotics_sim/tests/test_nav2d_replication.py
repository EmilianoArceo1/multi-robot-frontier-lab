from __future__ import annotations

import json
from pathlib import Path

from algorithms.nav2d_wavefront.plugin import (
    NAV2D_WAVEFRONT_COORDINATOR,
    create_plugin,
)
from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_sim.environment.belief_map import BeliefMap, FREE, OCCUPIED, UNKNOWN
from robotics_sim.planning.exploration_planners import (
    NAV2D_NEAREST_FRONTIER_PLANNER,
    select_exploration_goal,
)


ROOT = Path(__file__).resolve().parents[2]


def _cell_center(row: int, col: int) -> tuple[float, float]:
    return (col + 0.5, row + 0.5)


def test_nav2d_single_wavefront_uses_reachable_grid_distance() -> None:
    belief = BeliefMap(bounds=(0.0, 9.0, 0.0, 9.0), resolution=1.0)
    belief.grid.fill(FREE)
    belief.grid[0, :] = OCCUPIED
    belief.grid[-1, :] = OCCUPIED
    belief.grid[:, 0] = OCCUPIED
    belief.grid[:, -1] = OCCUPIED

    # A close-looking frontier lies behind a wall whose only gap is at the
    # bottom.  A second frontier is two free wavefront steps straight up.
    belief.grid[1:8, 3] = OCCUPIED
    belief.grid[4, 5] = UNKNOWN
    belief.grid[1, 1] = UNKNOWN

    result = select_exploration_goal(
        NAV2D_NEAREST_FRONTIER_PLANNER,
        belief_map=belief,
        robot_xy=_cell_center(4, 1),
        excluded_targets=[],
        target_exclusion_radius=0.0,
    )

    assert result.success
    assert result.target == _cell_center(2, 1)
    assert "four-connected wavefront" in result.reason


def test_nav2d_multi_wavefront_assigns_distinct_voronoi_frontiers() -> None:
    explored = tuple(
        _cell_center(row, col)
        for row in (1, 2)
        for col in range(1, 9)
    )
    request = CoordinationRequest(
        robot_states=(
            RobotCoordinationState(
                robot_id=0,
                xy=_cell_center(1, 1),
                safety_radius=0.0,
                sensor_range=3.0,
                vision_model="LiDAR",
            ),
            RobotCoordinationState(
                robot_id=1,
                xy=_cell_center(1, 8),
                safety_radius=0.0,
                sensor_range=3.0,
                vision_model="LiDAR",
            ),
        ),
        robots_to_assign=(0, 1),
        world=WorldSnapshot(
            explored_points=explored,
            mapped_obstacle_points=(),
            bounds=(0.0, 10.0, 0.0, 4.0),
            resolution=1.0,
        ),
        parameters={
            "min_frontier_travel_distance": 0.9,
            "target_exclusion_radius": 1.0,
        },
    )

    result = create_plugin().assign(request)

    assert result.strategy == NAV2D_WAVEFRONT_COORDINATOR
    assert len(result.assignments) == 2
    assert all(item.status == "ASSIGNED" for item in result.assignments)
    targets = [item.target for item in result.assignments]
    assert targets[0] is not None and targets[1] is not None
    assert targets[0] != targets[1]
    assert targets[0][0] < targets[1][0]


def test_nav2d_scenarios_pin_source_and_scaled_parameters() -> None:
    scenarios = [
        ROOT / "examples" / "nav2d_tutorial3_single.sim",
        ROOT / "examples" / "nav2d_tutorial4_multi.sim",
    ]

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in scenarios]
    for payload in payloads:
        assert payload["experiment"]["source_commit"] == (
            "3c27da9b0f5699559d9048c13ef4885815193981"
        )
        assert payload["experiment"]["uniform_scale"] == 0.5
        assert payload["map"]["grid_resolution"] == 0.05
        assert len(payload["map"]["obstacles"]) == 54
        assert payload["planner"] == {
            "type": "Dijkstra",
            "path_simplifier": "Raw grid path",
        }
        assert payload["exploration"]["planner"] == NAV2D_NEAREST_FRONTIER_PLANNER
        assert payload["sensor"] == {"type": "LiDAR", "range": 5.0}

    assert payloads[0]["simulation"]["agent_mode"] == "Single Robot Mode"
    assert payloads[1]["simulation"]["agent_mode"] == "Multiple Robot Mode"
    assert payloads[1]["coordination"]["strategy"] == NAV2D_WAVEFRONT_COORDINATOR
    assert payloads[1]["multi_robot"]["robot_count"] == 2

