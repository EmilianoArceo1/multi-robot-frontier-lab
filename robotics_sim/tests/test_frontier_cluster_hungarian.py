"""Contract tests for the two-stage frontier Hungarian coordinator
(algorithms/frontier_cluster_hungarian/): plugin discovery, the hard
cluster filter, the second-stage reduction, obstacle-factor behavior,
feasibility/reservations, and the overall assign() orchestration.

No engine, no Qt, no MainWindow -- CoordinationRequest/CoordinationServices
are built by hand with a small counting FrontierInformationService double.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.frontiers import FrontierCluster, ViewpointCandidate
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_interfaces.plugins import PluginCapability
from robotics_interfaces.services import CoordinationServices

from algorithms.frontier_cluster_hungarian.plugin import (
    FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR,
    create_plugin,
)
from algorithms.frontier_cluster_hungarian.utility import normalize_weights
from algorithms.frontier_cluster_hungarian.clustering import reduce_frontier_clusters
from robotics_sim.simulation.plugin_loader import load_coordination_plugin

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_DIR = REPO_ROOT / "algorithms" / "frontier_cluster_hungarian"
ALGORITHM_SOURCE_FILES = sorted(ALGORITHM_DIR.glob("*.py"))


# ---------------------------------------------------------------------------
# Fixtures / test doubles
# ---------------------------------------------------------------------------


def _robot(
    robot_id: int,
    x: float,
    y: float,
    *,
    safety_radius: float = 0.3,
    sensor_range: float = 2.5,
    current_target=None,
    is_active: bool = True,
) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=(x, y),
        safety_radius=safety_radius,
        sensor_range=sensor_range,
        vision_model="Camera / FoV",
        current_target=current_target,
        is_active=is_active,
    )


def _cluster(
    cluster_id: str,
    *,
    cells=(),
    centroid=None,
    viewpoints=(),
    information_gain: float = 0.0,
    valid: bool = True,
) -> FrontierCluster:
    viewpoint_objects = tuple(
        ViewpointCandidate(xy=xy, information_gain=information_gain) for xy in viewpoints
    )
    return FrontierCluster(
        cluster_id=cluster_id,
        cells=cells,
        centroid=centroid,
        viewpoints=viewpoint_objects,
        information_gain=information_gain,
        valid=valid,
    )


class CountingFrontierInformationService:
    """Counts get_frontier_clusters() calls and records the robot_id each
    call was made with -- the plugin must call this exactly once per
    assign(), always with robot_id=None."""

    def __init__(self, clusters):
        self._clusters = tuple(clusters)
        self.call_count = 0
        self.call_robot_ids: list[int | None] = []

    def get_frontier_clusters(self, robot_id=None):
        self.call_count += 1
        self.call_robot_ids.append(robot_id)
        return self._clusters


class _PoisonService:
    """Raises if ANY method is called -- proves the plugin never reaches
    for frontier_provider/team_frontier_provider/path_planning_service/
    collision_checking_service/map_query_service."""

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"plugin must not use {name!r} on this service")

        return _raise


def _build_request(
    *,
    robots,
    robots_to_assign=(),
    clusters=(),
    parameters=None,
    blocked_targets_by_robot=None,
    existing_targets_by_robot=None,
    mapped_obstacle_points=(),
    bounds=(-50.0, 50.0, -50.0, 50.0),
    resolution=0.5,
    service=None,
    no_service: bool = False,
):
    if no_service:
        coordination_services = CoordinationServices(
            frontier_provider=_PoisonService(),
            team_frontier_provider=_PoisonService(),
            path_planning_service=_PoisonService(),
            collision_checking_service=_PoisonService(),
            map_query_service=_PoisonService(),
            frontier_information_service=None,
        )
        service = None
    else:
        service = CountingFrontierInformationService(clusters) if service is None else service
        coordination_services = CoordinationServices(
            frontier_provider=_PoisonService(),
            team_frontier_provider=_PoisonService(),
            path_planning_service=_PoisonService(),
            collision_checking_service=_PoisonService(),
            map_query_service=_PoisonService(),
            frontier_information_service=service,
        )

    request = CoordinationRequest(
        robot_states=tuple(robots),
        robots_to_assign=tuple(robots_to_assign),
        world=WorldSnapshot(
            explored_points=(),
            mapped_obstacle_points=tuple(mapped_obstacle_points),
            bounds=bounds,
            resolution=resolution,
        ),
        blocked_targets_by_robot=dict(blocked_targets_by_robot or {}),
        existing_targets_by_robot=dict(existing_targets_by_robot or {}),
        services=coordination_services,
        parameters=dict(parameters or {}),
    )
    return request, service


def _assignment_for(result, robot_id):
    return next(item for item in result.assignments if item.robot_id == robot_id)


# ---------------------------------------------------------------------------
# 1-4. Discovery, metadata, architecture boundaries.
# ---------------------------------------------------------------------------


def test_plugin_is_discovered_by_metadata_name():
    plugin = load_coordination_plugin(FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR)
    assert plugin.metadata.name == FRONTIER_CLUSTER_HUNGARIAN_COORDINATOR


def test_metadata_declares_coordination_and_task_allocation():
    plugin = create_plugin()
    assert PluginCapability.COORDINATION in plugin.metadata.capabilities
    assert PluginCapability.TASK_ALLOCATION in plugin.metadata.capabilities


def test_metadata_does_not_declare_generation_planning_control_or_full_stack():
    plugin = create_plugin()
    for capability in (
        PluginCapability.TARGET_GENERATION,
        PluginCapability.PATH_PLANNING,
        PluginCapability.CONTROL,
        PluginCapability.FULL_STACK,
    ):
        assert capability not in plugin.metadata.capabilities


_FORBIDDEN_IMPORT_PATTERNS = (
    "import robotics_sim",
    "from robotics_sim",
    "import experiments",
    "from experiments",
    "import PySide6",
    "from PySide6",
    "QApplication",
    "MainWindow",
    "SimulationCanvas",
    "scipy.optimize",
    "import scipy",
    "from scipy",
    "import sklearn",
    "from sklearn",
)


def _non_docstring_code(text: str) -> str:
    return re.sub(r'""".*?"""', "", text, flags=re.DOTALL)


def test_algorithm_package_source_never_mentions_forbidden_symbols():
    for path in ALGORITHM_SOURCE_FILES:
        code = _non_docstring_code(path.read_text(encoding="utf-8"))
        for forbidden in _FORBIDDEN_IMPORT_PATTERNS:
            assert forbidden not in code, f"{path.name} must not contain {forbidden!r}"


def test_importing_plugin_in_a_fresh_process_never_imports_robotics_sim_or_qt():
    probe = (
        "import sys\n"
        "import algorithms.frontier_cluster_hungarian.plugin\n"
        "forbidden = ['PySide6', 'PySide6.QtWidgets', 'robotics_sim', 'experiments']\n"
        "leaked = [name for name in forbidden if name in sys.modules]\n"
        "assert not leaked, f'unexpected modules imported: {leaked}'\n"
        "print('CLEAN')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "CLEAN" in result.stdout


# ---------------------------------------------------------------------------
# 5-6. Exactly one service call, without robot_id.
# ---------------------------------------------------------------------------


def test_frontier_information_service_is_called_exactly_once_without_robot_id():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0),
    )
    request, service = _build_request(
        robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0,), clusters=clusters
    )

    plugin = create_plugin()
    plugin.assign(request)

    assert service.call_count == 1
    assert service.call_robot_ids == [None]


# ---------------------------------------------------------------------------
# 7-8. Second-stage reduction: nearby merges, far ones stay separate.
# ---------------------------------------------------------------------------


def test_two_nearby_components_reduce_to_one_task():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0), (0.5, 0.0)), centroid=(0.25, 0.0), viewpoints=((0.25, 0.0),), information_gain=3.0),
        _cluster("f1", cells=((0.6, 0.0),), centroid=(0.6, 0.0), viewpoints=((0.6, 0.0),), information_gain=2.0),
    )
    request, service = _build_request(
        robots=[_robot(0, -1.0, 0.0)],
        robots_to_assign=(0,),
        clusters=clusters,
        parameters={"secondary_cluster_grid_size": 2.0, "secondary_cluster_merge_radius": 3.0},
    )

    plugin = create_plugin()
    result = plugin.assign(request)

    assert result.debug["reduced_task_count"] == 1
    assert result.debug["task_source_cluster_ids"]["reduced-task-0000"] == ["f0", "f1"]


def test_two_far_components_remain_as_two_tasks():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0),
        _cluster("f1", cells=((30.0, 30.0),), centroid=(30.0, 30.0), viewpoints=((30.0, 30.0),), information_gain=2.0),
    )
    request, service = _build_request(
        robots=[_robot(0, -1.0, 0.0), _robot(1, 29.0, 30.0)],
        robots_to_assign=(0, 1),
        clusters=clusters,
        parameters={"secondary_cluster_grid_size": 2.0, "secondary_cluster_merge_radius": 3.0},
    )

    plugin = create_plugin()
    result = plugin.assign(request)

    assert result.debug["reduced_task_count"] == 2


# ---------------------------------------------------------------------------
# 9. Reordering the service's raw clusters never changes the outcome.
# ---------------------------------------------------------------------------


def test_reordering_input_clusters_does_not_change_the_result():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0),
        _cluster("f1", cells=((30.0, 30.0),), centroid=(30.0, 30.0), viewpoints=((30.0, 30.0),), information_gain=2.0),
        _cluster("f2", cells=(), centroid=None, viewpoints=(), information_gain=1.0, valid=False),
    )
    robots = [_robot(0, -1.0, 0.0), _robot(1, 29.0, 30.0)]

    request_a, _ = _build_request(robots=robots, robots_to_assign=(0, 1), clusters=clusters)
    request_b, _ = _build_request(robots=robots, robots_to_assign=(0, 1), clusters=tuple(reversed(clusters)))

    plugin = create_plugin()
    result_a = plugin.assign(request_a)
    result_b = plugin.assign(request_b)

    assert result_a.targets == result_b.targets
    assert result_a.reasons == result_b.reasons
    assert [(a.robot_id, a.status, a.target, a.reason) for a in result_a.assignments] == [
        (b.robot_id, b.status, b.target, b.reason) for b in result_b.assignments
    ]
    assert result_a.debug["task_ids"] == result_b.debug["task_ids"]
    assert result_a.debug["rejected_clusters"] == result_b.debug["rejected_clusters"]
    assert result_a.debug["utility_matrix"] == result_b.debug["utility_matrix"]
    assert result_a.debug["selected_task_by_robot"] == result_b.debug["selected_task_by_robot"]


# ---------------------------------------------------------------------------
# 10-14. Hard filter: rejection reasons, and rejected clusters never reach
#    the matrix or an assignment.
# ---------------------------------------------------------------------------


def test_invalid_cluster_is_rejected_before_secondary_clustering():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0, valid=False),
    )
    request, _ = _build_request(robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0,), clusters=clusters)

    result = create_plugin().assign(request)

    assert result.debug["rejected_clusters"] == [{"cluster_id": "f0", "reason": "cluster marked invalid"}]
    assert result.debug["reduced_task_count"] == 0
    assert _assignment_for(result, 0).status == "HOLD"


def test_cluster_without_centroid_is_rejected():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=None, viewpoints=((0.0, 0.0),), information_gain=3.0),
    )
    request, _ = _build_request(robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0,), clusters=clusters)

    result = create_plugin().assign(request)

    assert result.debug["rejected_clusters"] == [{"cluster_id": "f0", "reason": "missing centroid"}]


def test_cluster_without_viewpoints_is_rejected():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=(), information_gain=3.0),
    )
    request, _ = _build_request(robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0,), clusters=clusters)

    result = create_plugin().assign(request)

    assert result.debug["rejected_clusters"] == [{"cluster_id": "f0", "reason": "no viewpoints"}]


def test_rejected_clusters_never_enter_the_matrix_or_get_assigned():
    valid_cluster = _cluster(
        "f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0
    )
    invalid_cluster = _cluster(
        "f1", cells=((0.1, 0.1),), centroid=(0.1, 0.1), viewpoints=((0.1, 0.1),), information_gain=9.0, valid=False
    )
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0,), clusters=(valid_cluster, invalid_cluster)
    )

    result = create_plugin().assign(request)

    assert result.debug["task_ids"] == ["reduced-task-0000"]
    assert result.debug["reduced_task_count"] == 1
    proposal = _assignment_for(result, 0).proposal
    assert proposal.metadata["cluster_id"] == "f0"
    assert proposal.metadata["representative_cluster_id"] == "f0"
    for source_cluster_ids in result.debug["task_source_cluster_ids"].values():
        assert "f1" not in source_cluster_ids


# ---------------------------------------------------------------------------
# 15-16. Source cluster ordering and representative selection rule.
# ---------------------------------------------------------------------------


def test_source_cluster_ids_are_ordered():
    clusters = (
        _cluster("fz", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=1.0),
        _cluster("fa", cells=((0.1, 0.0),), centroid=(0.1, 0.0), viewpoints=((0.1, 0.0),), information_gain=1.0),
        _cluster("fm", cells=((0.2, 0.0),), centroid=(0.2, 0.0), viewpoints=((0.2, 0.0),), information_gain=1.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0)],
        robots_to_assign=(0,),
        clusters=clusters,
        parameters={"secondary_cluster_grid_size": 5.0, "secondary_cluster_merge_radius": 5.0},
    )

    result = create_plugin().assign(request)

    assert result.debug["task_source_cluster_ids"]["reduced-task-0000"] == ["fa", "fm", "fz"]


def test_representative_selection_prefers_gain_then_cells_then_id():
    close_a = _cluster("fa", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=5.0)
    close_b = _cluster(
        "fb", cells=((0.1, 0.0), (0.1, 0.1), (0.1, 0.2)), centroid=(0.1, 0.1), viewpoints=((0.1, 0.1),), information_gain=5.0
    )
    close_c = _cluster(
        "fc", cells=tuple((0.2, y) for y in range(10)), centroid=(0.2, 4.5), viewpoints=((0.2, 4.5),), information_gain=2.0
    )
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0)],
        robots_to_assign=(0,),
        clusters=(close_a, close_b, close_c),
        parameters={"secondary_cluster_grid_size": 5.0, "secondary_cluster_merge_radius": 5.0},
    )

    result = create_plugin().assign(request)

    assert result.debug["reduced_task_count"] == 1
    proposal = _assignment_for(result, 0).proposal
    # fb ties fa on the highest information_gain (5.0) but has more cells.
    assert proposal.metadata["representative_cluster_id"] == "fb"


def test_representative_selection_breaks_full_tie_by_cluster_id():
    tied_z = _cluster("fz", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=4.0)
    tied_a = _cluster("fa", cells=((0.1, 0.0),), centroid=(0.1, 0.0), viewpoints=((0.1, 0.0),), information_gain=4.0)
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0)],
        robots_to_assign=(0,),
        clusters=(tied_z, tied_a),
        parameters={"secondary_cluster_grid_size": 5.0, "secondary_cluster_merge_radius": 5.0},
    )

    result = create_plugin().assign(request)

    proposal = _assignment_for(result, 0).proposal
    assert proposal.metadata["representative_cluster_id"] == "fa"


# ---------------------------------------------------------------------------
# 17-19. Assigned proposal provenance and duplicate-target avoidance.
# ---------------------------------------------------------------------------


def test_assigned_proposal_keeps_cluster_id_and_task_metadata():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0),
    )
    request, _ = _build_request(robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0,), clusters=clusters)

    result = create_plugin().assign(request)

    proposal = _assignment_for(result, 0).proposal
    assert proposal is not None
    assert proposal.metadata["cluster_id"] == "f0"
    assert proposal.metadata["task_id"] == "reduced-task-0000"
    assert proposal.metadata["source_cluster_ids"] == ("f0",)


def test_no_duplicate_targets_among_assigned_robots():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0),
        _cluster("f1", cells=((10.0, 0.0),), centroid=(10.0, 0.0), viewpoints=((10.0, 0.0),), information_gain=3.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0), _robot(1, 9.0, 0.0)],
        robots_to_assign=(0, 1),
        clusters=clusters,
        parameters={"secondary_cluster_grid_size": 1.0, "secondary_cluster_merge_radius": 1.0},
    )

    result = create_plugin().assign(request)

    assigned_targets = [a.target for a in result.assignments if a.status == "ASSIGNED"]
    assert len(assigned_targets) == len(set(assigned_targets))
    assert len(assigned_targets) == 2


# ---------------------------------------------------------------------------
# 20-23. Cardinality edge cases.
# ---------------------------------------------------------------------------


def test_more_robots_than_tasks_holds_the_rest():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0), _robot(1, -2.0, 0.0), _robot(2, -3.0, 0.0)],
        robots_to_assign=(0, 1, 2),
        clusters=clusters,
    )

    result = create_plugin().assign(request)

    statuses = [a.status for a in result.assignments]
    assert statuses.count("ASSIGNED") == 1
    assert statuses.count("HOLD") == 2


def test_more_tasks_than_robots_does_not_crash():
    clusters = tuple(
        _cluster(f"f{i}", cells=((float(i) * 10.0, 0.0),), centroid=(float(i) * 10.0, 0.0), viewpoints=((float(i) * 10.0, 0.0),), information_gain=1.0)
        for i in range(5)
    )
    request, _ = _build_request(
        robots=[_robot(0, 0.0, 0.0), _robot(1, 5.0, 0.0)],
        robots_to_assign=(0, 1),
        clusters=clusters,
        parameters={"secondary_cluster_grid_size": 1.0, "secondary_cluster_merge_radius": 1.0},
    )

    result = create_plugin().assign(request)

    assert len(result.assignments) == 2
    assert all(a.status == "ASSIGNED" for a in result.assignments)


def test_zero_clusters_produces_all_hold():
    request, _ = _build_request(robots=[_robot(0, 0.0, 0.0), _robot(1, 1.0, 0.0)], robots_to_assign=(0, 1), clusters=())

    result = create_plugin().assign(request)

    assert all(a.status == "HOLD" for a in result.assignments)
    assert all(a.reason == "no valid frontier clusters" for a in result.assignments)


def test_all_invalid_clusters_produces_all_hold():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0, valid=False),
        _cluster("f1", cells=((1.0, 0.0),), centroid=(1.0, 0.0), viewpoints=((1.0, 0.0),), information_gain=2.0, valid=False),
    )
    request, _ = _build_request(robots=[_robot(0, 0.0, 0.0)], robots_to_assign=(0,), clusters=clusters)

    result = create_plugin().assign(request)

    assert all(a.status == "HOLD" for a in result.assignments)
    assert all(a.reason == "no valid frontier clusters" for a in result.assignments)


# ---------------------------------------------------------------------------
# 24-28. Blocking, reservations, unknown ids.
# ---------------------------------------------------------------------------


def test_blocked_target_excludes_that_task_for_that_robot():
    target = (5.0, 0.0)
    clusters = (
        _cluster("f0", cells=((5.0, 0.0),), centroid=target, viewpoints=(target,), information_gain=3.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, 0.0, 0.0)],
        robots_to_assign=(0,),
        clusters=clusters,
        blocked_targets_by_robot={0: (target,)},
    )

    result = create_plugin().assign(request)

    assert _assignment_for(result, 0).status == "HOLD"
    assert _assignment_for(result, 0).reason == "no feasible frontier task"


def test_target_blocked_for_one_robot_can_still_be_feasible_for_another():
    target = (5.0, 0.0)
    clusters = (
        _cluster("f0", cells=((5.0, 0.0),), centroid=target, viewpoints=(target,), information_gain=3.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, 0.0, 0.0), _robot(1, 4.0, 0.0)],
        robots_to_assign=(0, 1),
        clusters=clusters,
        blocked_targets_by_robot={0: (target,)},
    )

    result = create_plugin().assign(request)

    assert _assignment_for(result, 0).status == "HOLD"
    assert _assignment_for(result, 1).status == "ASSIGNED"
    assert _assignment_for(result, 1).target == target


def test_non_reassigned_robot_target_is_reserved_globally():
    target = (5.0, 0.0)
    clusters = (
        _cluster("f0", cells=((5.0, 0.0),), centroid=target, viewpoints=(target,), information_gain=3.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, 4.9, 0.0, current_target=target), _robot(1, 4.0, 0.0)],
        robots_to_assign=(1,),
        clusters=clusters,
    )

    result = create_plugin().assign(request)

    assert _assignment_for(result, 1).status == "HOLD"
    assert _assignment_for(result, 1).reason == "no feasible frontier task"


def test_non_reassigned_robots_keep_target_and_reason():
    target = (5.0, 0.0)
    clusters = (
        _cluster("f0", cells=((5.0, 0.0),), centroid=target, viewpoints=(target,), information_gain=3.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, 4.9, 0.0, current_target=target), _robot(1, 0.0, 0.0)],
        robots_to_assign=(1,),
        clusters=clusters,
    )

    result = create_plugin().assign(request)

    assert result.targets[0] == target
    assert result.reasons[0] == "kept existing target"
    assert not any(a.robot_id == 0 for a in result.assignments)


def test_unknown_requested_robot_id_produces_failed():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=3.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0, 99), clusters=clusters
    )

    result = create_plugin().assign(request)

    failed = _assignment_for(result, 99)
    assert failed.status == "FAILED"


# ---------------------------------------------------------------------------
# 29. Obstacle factor changes assignment on an information/distance tie.
# ---------------------------------------------------------------------------


def test_obstacle_factor_changes_assignment_on_tied_information_and_distance():
    target_a = (5.0, 0.0)  # will be blocked
    target_b = (0.0, 5.0)  # stays clear
    clusters = (
        _cluster("fa", cells=((5.0, 0.0),), centroid=target_a, viewpoints=(target_a,), information_gain=4.0),
        _cluster("fb", cells=((0.0, 5.0),), centroid=target_b, viewpoints=(target_b,), information_gain=4.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, 0.0, 0.0, safety_radius=0.3)],
        robots_to_assign=(0,),
        clusters=clusters,
        mapped_obstacle_points=[(2.5, 0.0)],
        parameters={
            "secondary_cluster_grid_size": 1.0,
            "secondary_cluster_merge_radius": 1.0,
            "obstacle_point_tolerance": 0.5,
        },
    )

    result = create_plugin().assign(request)

    assignment = _assignment_for(result, 0)
    assert assignment.status == "ASSIGNED"
    assert assignment.target == target_b
    assert assignment.proposal.metadata["blocked_line_fraction"] == 0.0


# ---------------------------------------------------------------------------
# 30-34. Obstacle factor geometry itself.
# ---------------------------------------------------------------------------


def test_no_obstacles_means_fully_clear():
    from algorithms.frontier_cluster_hungarian.obstacle_factor import (
        five_line_blocked_fraction,
        five_line_clearance_score,
    )

    fraction = five_line_blocked_fraction(
        robot_xy=(0.0, 0.0), target_xy=(5.0, 0.0), observed_obstacle_points=[], safety_radius=0.3, point_tolerance=0.2
    )

    assert fraction == 0.0
    assert five_line_clearance_score(fraction) == 1.0


def test_exactly_five_lines_are_evaluated():
    from algorithms.frontier_cluster_hungarian.obstacle_factor import LINE_COUNT

    assert LINE_COUNT == 5


def test_central_blocked_line_alone_yields_point_two():
    from algorithms.frontier_cluster_hungarian.obstacle_factor import five_line_blocked_fraction

    fraction = five_line_blocked_fraction(
        robot_xy=(0.0, 0.0),
        target_xy=(5.0, 0.0),
        observed_obstacle_points=[(2.5, 0.0)],
        safety_radius=0.3,
        point_tolerance=0.05,
    )

    assert fraction == pytest.approx(0.2)


def test_all_five_lines_blocked_yields_one():
    from algorithms.frontier_cluster_hungarian.obstacle_factor import five_line_blocked_fraction

    fraction = five_line_blocked_fraction(
        robot_xy=(0.0, 0.0),
        target_xy=(5.0, 0.0),
        observed_obstacle_points=[(2.5, 0.0)],
        safety_radius=0.3,
        point_tolerance=0.5,
    )

    assert fraction == pytest.approx(1.0)


def test_zero_length_segment_does_not_raise():
    from algorithms.frontier_cluster_hungarian.obstacle_factor import five_line_blocked_fraction

    fraction = five_line_blocked_fraction(
        robot_xy=(1.0, 1.0),
        target_xy=(1.0, 1.0),
        observed_obstacle_points=[(1.0, 1.0)],
        safety_radius=0.3,
        point_tolerance=0.05,
    )

    assert 0.0 <= fraction <= 1.0


# ---------------------------------------------------------------------------
# 35-36. Invalid parameters raise clear ValueErrors.
# ---------------------------------------------------------------------------


def test_invalid_weights_raise_value_error():
    with pytest.raises(ValueError):
        normalize_weights(information_weight=-1.0, distance_weight=0.5, obstacle_weight=0.5)

    with pytest.raises(ValueError):
        normalize_weights(information_weight=0.0, distance_weight=0.0, obstacle_weight=0.0)

    with pytest.raises(ValueError):
        normalize_weights(information_weight=float("nan"), distance_weight=0.5, obstacle_weight=0.5)


def test_invalid_geometric_parameters_raise_value_error():
    cluster = _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=1.0)

    with pytest.raises(ValueError):
        reduce_frontier_clusters([cluster], grid_size=0.0, merge_radius=1.0, duplicate_tolerance=1e-6)

    with pytest.raises(ValueError):
        reduce_frontier_clusters([cluster], grid_size=1.0, merge_radius=-1.0, duplicate_tolerance=1e-6)


def test_plugin_propagates_invalid_weight_parameters_as_value_error():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=1.0),
    )
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0)],
        robots_to_assign=(0,),
        clusters=clusters,
        parameters={"hungarian_information_weight": -1.0},
    )

    with pytest.raises(ValueError):
        create_plugin().assign(request)


# ---------------------------------------------------------------------------
# 37-38. Result shape.
# ---------------------------------------------------------------------------


def test_targets_and_reasons_match_robot_states_length():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=1.0),
    )
    robots = [_robot(0, -1.0, 0.0), _robot(1, 5.0, 5.0), _robot(2, 9.0, 9.0)]
    request, _ = _build_request(robots=robots, robots_to_assign=(0,), clusters=clusters)

    result = create_plugin().assign(request)

    assert len(result.targets) == len(robots)
    assert len(result.reasons) == len(robots)


def test_each_reassigned_robot_has_exactly_one_assignment():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=1.0),
        _cluster("f1", cells=((10.0, 0.0),), centroid=(10.0, 0.0), viewpoints=((10.0, 0.0),), information_gain=1.0),
    )
    robots = [_robot(0, -1.0, 0.0), _robot(1, 9.0, 0.0)]
    request, _ = _build_request(
        robots=robots,
        robots_to_assign=(0, 1),
        clusters=clusters,
        parameters={"secondary_cluster_grid_size": 1.0, "secondary_cluster_merge_radius": 1.0},
    )

    result = create_plugin().assign(request)

    for robot_id in (0, 1):
        matches = [a for a in result.assignments if a.robot_id == robot_id]
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# 39. Other services are never touched (also implicitly proven by every
#    other test above via _build_request()'s _PoisonService wiring).
# ---------------------------------------------------------------------------


def test_other_services_are_never_invoked():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=1.0),
    )
    request, _ = _build_request(robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0,), clusters=clusters)

    # Must not raise: _PoisonService would raise AssertionError the moment
    # any attribute on frontier_provider/team_frontier_provider/
    # path_planning_service/collision_checking_service/map_query_service is
    # touched.
    create_plugin().assign(request)


def test_no_frontier_information_service_holds_without_touching_other_services():
    request, _ = _build_request(
        robots=[_robot(0, -1.0, 0.0), _robot(1, 99, 0.0)],
        robots_to_assign=(0, 99),
        no_service=True,
    )

    result = create_plugin().assign(request)

    assert _assignment_for(result, 0).status == "HOLD"
    assert _assignment_for(result, 0).reason == "no frontier information service"
    assert _assignment_for(result, 99).status == "FAILED"
    assert result.debug["frontier_information_service_calls"] == 0


# ---------------------------------------------------------------------------
# 40. Debug contents.
# ---------------------------------------------------------------------------


def test_debug_contains_the_required_fields():
    clusters = (
        _cluster("f0", cells=((0.0, 0.0),), centroid=(0.0, 0.0), viewpoints=((0.0, 0.0),), information_gain=1.0),
    )
    request, _ = _build_request(robots=[_robot(0, -1.0, 0.0)], robots_to_assign=(0,), clusters=clusters)

    result = create_plugin().assign(request)

    for key in (
        "raw_cluster_count",
        "valid_cluster_count",
        "rejected_cluster_count",
        "reduced_task_count",
        "frontier_information_service_calls",
        "selected_task_by_robot",
    ):
        assert key in result.debug
    assert result.debug["frontier_information_service_calls"] == 1
