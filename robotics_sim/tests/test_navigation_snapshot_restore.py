"""
Tests for restoring the live simulation from a navigation-debug snapshot
(engine.restore_navigation_debug_snapshot() / can_restore_navigation_debug_
snapshot()) and for NavigationDebugEventLog.truncate_after(), the history
primitive the restore uses to drop the discarded future.

Same lightweight duck-typed engine fake pattern as test_navigation_debug_
history_navigation.py, but restore touches enough real collaborators (Robot,
RobotAgent, BeliefMap, RuntimeHazardService) that those are constructed for
real rather than stubbed -- only the canvas and the agent lookup are faked,
exactly as elsewhere in this test suite. The one exception is the final
end-to-end test at the bottom of this file, which drives a real MainWindow
through start_simulation() and simulation_step() directly -- no fakes at
all -- to prove the restore also works against the real engine, not just
against a hand-built fixture.
"""
from __future__ import annotations

import os
import zlib
from types import SimpleNamespace

import numpy as np

# Never write belief-trace artifacts to disk from a test run (see engine.
# start_belief_trace_run()'s docstring) -- must be set before MainWindow()
# is constructed anywhere in this module.
os.environ.setdefault("BELIEF_TRACE_ARTIFACTS", "0")

from PySide6.QtWidgets import QApplication

from robot import Robot
from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.diagnostics.event_log import NavigationDebugEventKind, NavigationDebugEventLog
from robotics_sim.diagnostics.navigation_snapshot import (
    AgentStateDebug,
    BeliefMapDebug,
    ControllerDebug,
    FrontierDebug,
    HazardDebug,
    HazardSourceDebug,
    Maybe,
    NavigationDebugSnapshot,
    PathDebug,
    PlanningGridDebug,
    Pose,
    PredictedMotionDebug,
    RouteValidationDebug,
    RuntimeMetricsDebug,
    SafetyDebug,
    SensorDebug,
)
from robotics_sim.app.main_window import MainWindow
from robotics_sim.environment.belief_map import BeliefMap, FREE, OCCUPIED, UNKNOWN
from robotics_sim.environment.hazard_field import FireSource
from robotics_sim.simulation.config import SimulationConfig
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.hazard_service import RuntimeHazardService

# toggle_pause()/canvas construction paths import Qt -- same requirement as
# the other test_navigation_debug_*.py files.
_app = QApplication.instance() or QApplication([])


def _make_belief_frame(grid: np.ndarray, explored: np.ndarray, *, revision: int, resolution: float, bounds) -> BeliefMapDebug:
    grid = np.ascontiguousarray(grid, dtype=np.int8)
    explored_u8 = np.ascontiguousarray(explored, dtype=np.uint8)
    packed = np.packbits(explored_u8.reshape(-1), bitorder="little")
    return BeliefMapDebug(
        revision=revision,
        resolution=resolution,
        bounds=tuple(float(v) for v in bounds),
        grid_shape=(int(grid.shape[0]), int(grid.shape[1])),
        grid_zlib=zlib.compress(grid.tobytes(order="C"), level=1),
        explored_shape=(int(explored_u8.shape[0]), int(explored_u8.shape[1]), int(explored_u8.shape[2])),
        explored_packbits_zlib=zlib.compress(packed.tobytes(), level=1),
    )


def _make_hazard_frame(sources: tuple[HazardSourceDebug, ...] = (), *, next_fire_id: int = 1, version: int = 0) -> HazardDebug:
    return HazardDebug(version=version, next_fire_id=next_fire_id, sources=sources)


def _make_agent_state(
    *,
    final_goal_xy=None,
    exploration_target_xy=None,
    active_path_goal_xy=None,
    active_path_mode: str | None = "FoV-aware directional frontier",
    route_generation: int = 0,
    route_affected_replan_count: int = 0,
    first_segment_blocked_count: int = 0,
    last_frontier_candidate_count: int = 0,
    prefetch_success_count: int = 0,
    prefetch_fail_count: int = 0,
    safety_replan_count: int = 0,
    target_switch_count: int = 0,
) -> AgentStateDebug:
    return AgentStateDebug(
        final_goal_xy=final_goal_xy,
        exploration_target_xy=exploration_target_xy,
        active_path_goal_xy=active_path_goal_xy,
        active_path_mode=active_path_mode,
        route_generation=route_generation,
        route_affected_replan_count=route_affected_replan_count,
        first_segment_blocked_count=first_segment_blocked_count,
        last_frontier_candidate_count=last_frontier_candidate_count,
        prefetch_success_count=prefetch_success_count,
        prefetch_fail_count=prefetch_fail_count,
        safety_replan_count=safety_replan_count,
        target_switch_count=target_switch_count,
    )


def _make_metrics(
    *,
    total_distance_traveled: float = 0.0,
    route_request_count: int = 0,
    route_result_count: int = 0,
    route_failure_count: int = 0,
    sensor_update_count: int = 0,
    mapping_update_count: int = 0,
    safety_replan_count: int = 0,
    exploration_replan_count: int = 0,
    planner_jobs_started: int = 0,
    planner_jobs_completed: int = 0,
) -> RuntimeMetricsDebug:
    return RuntimeMetricsDebug(
        total_distance_traveled=total_distance_traveled,
        route_request_count=route_request_count,
        route_result_count=route_result_count,
        route_failure_count=route_failure_count,
        sensor_update_count=sensor_update_count,
        mapping_update_count=mapping_update_count,
        safety_replan_count=safety_replan_count,
        exploration_replan_count=exploration_replan_count,
        planner_jobs_started=planner_jobs_started,
        planner_jobs_completed=planner_jobs_completed,
    )


def _make_snapshot(
    *,
    snapshot_id: int,
    simulation_time: float,
    pose: tuple[float, float, float, float],
    active_path: tuple[tuple[float, float], ...] = (),
    active_waypoint_index: int | None = None,
    navigation_state: str = "moving",
    mapped_obstacle_points_count: int = 0,
    belief_frame: BeliefMapDebug | None = None,
    hazard_frame: HazardDebug | None = None,
    agent_state: AgentStateDebug | None = None,
    metrics: RuntimeMetricsDebug | None = None,
) -> NavigationDebugSnapshot:
    x, y, theta, v = pose
    return NavigationDebugSnapshot(
        snapshot_id=snapshot_id,
        simulation_time=simulation_time,
        robot_id="R1",
        navigation_state=navigation_state,
        decision_kind="FOLLOW_PATH",
        decision_reason="",
        robot_pose=Pose(x=x, y=y, theta=theta, v=v),
        path=PathDebug(
            raw_path=Maybe.missing(),
            simplified_path=Maybe.missing(),
            active_path=active_path,
            pending_path=(),
            active_segment=(( x, y), active_path[-1]) if active_path else None,
            active_waypoint_index=active_waypoint_index,
            planner_name=Maybe.missing(),
            simplifier_name=Maybe.missing(),
        ),
        route=RouteValidationDebug(first_segment=Maybe.missing(), endpoint_reaches_goal=None),
        predicted_motion=PredictedMotionDebug(trajectory=Maybe.missing(), collision=Maybe.missing()),
        safety=SafetyDebug(robot_radius=0.2, safety_radius=0.3, active_segment=Maybe.missing()),
        planning_grid=PlanningGridDebug(
            start_cell=Maybe.missing(),
            start_cell_world=Maybe.missing(),
            first_waypoint_cell=Maybe.missing(),
            first_waypoint_world=Maybe.missing(),
            unknown_is_traversable=Maybe.missing(),
            start_cell_cleared=Maybe.missing(),
        ),
        controller=ControllerDebug(
            v=v, omega=0.0, acceleration=0.0, heading_error=Maybe.missing(), distance_to_goal=Maybe.missing()
        ),
        frontier=FrontierDebug(
            candidate_count=Maybe.missing(),
            selected_target=Maybe.missing(),
            selected_score=Maybe.missing(),
            reason=Maybe.missing(),
        ),
        mapped_obstacle_points_count=mapped_obstacle_points_count,
        sensor=SensorDebug(),
        belief_map=Maybe.of(belief_frame) if belief_frame is not None else Maybe.missing(),
        hazard=Maybe.of(hazard_frame) if hazard_frame is not None else Maybe.missing(),
        agent_state=Maybe.of(agent_state) if agent_state is not None else Maybe.missing(),
        metrics=Maybe.of(metrics) if metrics is not None else Maybe.missing(),
    )


class _FakeCanvas:
    def __init__(self):
        self.status_messages: list[str] = []
        self.pushed_snapshots: list[int] = []
        self.history_positions: list[tuple[int | None, int]] = []
        self.explored_area_polygons: list = None
        self.mapped_obstacle_points: list = None
        self.robot = None
        self.path = None
        self.planned_path = None
        self.hazard_snapshots: list = []
        self.exploration_target = "unset"
        self.explored_area_seed = None

    def set_status(self, message):
        self.status_messages.append(message)

    def set_navigation_debug_snapshot(self, snapshot):
        self.pushed_snapshots.append(snapshot.snapshot_id if snapshot is not None else None)

    def set_navigation_debug_last_event(self, event):
        pass

    def set_navigation_debug_history_position(self, position, total):
        self.history_positions.append((position, total))

    def set_explored_area_polygons(self, polygons):
        self.explored_area_polygons = polygons

    def set_explored_area_seed(self, mask, resolution, bounds):
        self.explored_area_seed = (mask, resolution, bounds)

    def set_mapped_obstacle_points(self, points):
        self.mapped_obstacle_points = list(points)

    def set_robot(self, robot):
        self.robot = robot

    def set_path(self, path):
        self.path = path

    def set_planned_path(self, path):
        self.planned_path = path

    def set_simulation_metrics(self, *_a, **_k):
        pass

    def set_hazard_snapshot(self, snapshot):
        self.hazard_snapshots.append(snapshot)

    def set_exploration_target(self, target):
        self.exploration_target = target


_BOUNDS = (-2.0, 2.0, -2.0, 2.0)
_RESOLUTION = 1.0


def _build_fake_engine(
    *,
    snapshots: list[NavigationDebugSnapshot],
    history_index: int | None,
    agent_mode: str = "Single Robot Mode",
    live_fire_at: tuple[float, float] | None = None,
):
    log = NavigationDebugEventLog(max_size=50)
    for snap in snapshots:
        log.record(NavigationDebugEventKind.TICK, snap)

    robot = Robot(x=0.0, y=0.0, theta=0.0, v=0.0)
    agent = RobotAgent(robot_id=0, position=(0.0, 0.0), planner_mode="FoV-aware directional frontier")

    belief = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1)
    hazard_service = RuntimeHazardService(bounds=_BOUNDS, resolution=_RESOLUTION)
    if live_fire_at is not None:
        # Represents a fire that only exists in the "future" relative to the
        # snapshot being restored to -- set up before restore_navigation_
        # debug_snapshot() runs, exactly like a fire the user added after
        # the selected snapshot was captured.
        hazard_service.add_fire(live_fire_at)

    fake = SimpleNamespace(
        navigation_debug_enabled=True,
        navigation_debug_log=log,
        paused=True,
        running=True,
        robot=robot,
        robots=[],
        config=SimulationConfig(agent_mode=agent_mode),
        canvas=_FakeCanvas(),
        belief_map=belief,
        hazard_service=hazard_service,
        mapped_obstacle_points=[(float(i), float(i)) for i in range(6)],
        mapped_obstacle_point_keys=None,
        explored_area_polygons=[["stale-polygon-from-the-future"]],
        path_points=[(9.0, 9.0)],
        simulation_time=999.0,
        last_time=0.0,
        simulation_speed=1.0,
        total_distance_traveled=42.0,
        route_request_count=99,
        route_result_count=99,
        route_failure_count=99,
        sensor_update_count=99,
        mapping_update_count=99,
        safety_replan_count=99,
        exploration_replan_count=99,
        planner_jobs_started=99,
        planner_jobs_completed=99,
        current_exploration_target=(77.0, 77.0),
        _nav_debug_history_index=history_index,
        _nav_debug_seq=999,
        _nav_debug_belief_frame_key=("stale",),
        _nav_debug_belief_frame_cache=object(),
        _nav_debug_pending_plan_capture_by_robot={0: object()},
        _nav_debug_last_plan_capture=object(),
        _nav_debug_last_accepted_plan=object(),
        _nav_debug_live_snapshot=None,
        planning_in_progress=True,
        route_request_id=5,
        active_planner_workers={7: object()},
        prefetch_workers={0: object()},
        prefetch_request_ids={0: 5},
        runtime_agent=lambda robot_index=None: agent,
        is_exploration_mode=lambda: False,
        start_button=SimpleNamespace(setText=lambda *_a: None, setIcon=lambda *_a: None),
    )
    fake.mapped_obstacle_point_keys = {
        (round(p[0], 3), round(p[1], 3)) for p in fake.mapped_obstacle_points
    }
    for name in (
        "can_restore_navigation_debug_snapshot",
        "restore_navigation_debug_snapshot",
        "ensure_belief_map",
        "ensure_hazard_service",
        "push_hazard_snapshot",
        "sync_legacy_map_views_from_belief",
        "update_navigation_debug_step_buttons",
        "update_start_pause_button",
        "navigation_debug_history_length",
    ):
        setattr(fake, name, getattr(SimulationControllerMixin, name).__get__(fake))
    fake.agent = agent
    return fake


def _grid_with_two_occupied_cells() -> np.ndarray:
    grid = np.zeros(BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1).grid.shape, dtype=np.int8)
    grid[:] = UNKNOWN
    grid[0, 0] = OCCUPIED
    grid[1, 1] = FREE
    return grid


def _explored_all_true(shape) -> np.ndarray:
    return np.ones(shape, dtype=bool)


def _make_ten_snapshots() -> list[NavigationDebugSnapshot]:
    belief_shape = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1).grid.shape
    explored_shape = (1,) + belief_shape
    snapshots = []
    for i in range(10):
        grid = np.full(belief_shape, UNKNOWN, dtype=np.int8)
        grid[0, 0] = OCCUPIED
        # Each snapshot's explored mask grows by one more free cell so the
        # decoded arrays differ snapshot-to-snapshot (distinguishable).
        explored = np.zeros(explored_shape, dtype=bool)
        explored[0, : i + 1, 0] = True
        frame = _make_belief_frame(grid, explored, revision=i, resolution=_RESOLUTION, bounds=_BOUNDS)
        snapshots.append(
            _make_snapshot(
                snapshot_id=100 + i,
                simulation_time=float(i) * 0.5,
                pose=(float(i), float(i) * 2.0, float(i) * 0.1, 0.3),
                active_path=((float(i), float(i)), (float(i) + 1.0, float(i))),
                active_waypoint_index=0,
                navigation_state="moving",
                mapped_obstacle_points_count=min(i + 1, 6),
                belief_frame=frame,
                hazard_frame=_make_hazard_frame(),  # no fire in any of these 10
                # active_path_goal_xy deliberately does NOT equal
                # active_path[-1] == (i+1, i) -- proves restore uses this
                # explicit field rather than inferring it from the path.
                agent_state=_make_agent_state(
                    final_goal_xy=(float(i) + 70.0, float(i) + 70.0),
                    exploration_target_xy=(float(i) + 60.0, float(i) + 60.0),
                    active_path_goal_xy=(float(i) + 50.0, float(i) + 50.0),
                    active_path_mode="FoV-aware directional frontier",
                    route_generation=i,
                    route_affected_replan_count=i,
                    first_segment_blocked_count=i,
                    last_frontier_candidate_count=i,
                    prefetch_success_count=i,
                    prefetch_fail_count=i,
                    safety_replan_count=i,
                    target_switch_count=i,
                ),
                metrics=_make_metrics(
                    total_distance_traveled=float(i) * 1.5,
                    route_request_count=i,
                    route_result_count=i,
                    route_failure_count=i,
                    sensor_update_count=i,
                    mapping_update_count=i,
                    safety_replan_count=i,
                    exploration_replan_count=i,
                    planner_jobs_started=i,
                    planner_jobs_completed=i,
                ),
            )
        )
    return snapshots


# ---------------------------------------------------------------------------
# NavigationDebugEventLog.truncate_after()
# ---------------------------------------------------------------------------


def test_truncate_after_keeps_prefix_and_drops_rest():
    log = NavigationDebugEventLog(max_size=10)
    for i in range(5):
        log.record(NavigationDebugEventKind.TICK, _make_snapshot(snapshot_id=i, simulation_time=float(i), pose=(0, 0, 0, 0)))

    log.truncate_after(2)

    assert len(log) == 3
    assert [e.snapshot.snapshot_id for e in log.events()] == [0, 1, 2]


def test_truncate_after_negative_index_clears_everything():
    log = NavigationDebugEventLog(max_size=10)
    log.record(NavigationDebugEventKind.TICK, _make_snapshot(snapshot_id=1, simulation_time=1.0, pose=(0, 0, 0, 0)))

    log.truncate_after(-1)

    assert len(log) == 0


def test_truncate_after_preserves_the_bound():
    log = NavigationDebugEventLog(max_size=3)
    for i in range(3):
        log.record(NavigationDebugEventKind.TICK, _make_snapshot(snapshot_id=i, simulation_time=float(i), pose=(0, 0, 0, 0)))

    log.truncate_after(2)
    for i in range(3, 6):
        log.record(NavigationDebugEventKind.TICK, _make_snapshot(snapshot_id=i, simulation_time=float(i), pose=(0, 0, 0, 0)))

    assert len(log) == 3  # bound still respected after truncation + refill


# ---------------------------------------------------------------------------
# can_restore_navigation_debug_snapshot()
# ---------------------------------------------------------------------------


def test_can_restore_false_in_live():
    fake = _build_fake_engine(snapshots=_make_ten_snapshots(), history_index=None)
    can_restore, reason = fake.can_restore_navigation_debug_snapshot()
    assert can_restore is False
    assert "historical snapshot" in reason.lower()


def test_can_restore_true_in_history():
    fake = _build_fake_engine(snapshots=_make_ten_snapshots(), history_index=2)
    can_restore, reason = fake.can_restore_navigation_debug_snapshot()
    assert can_restore is True
    assert reason == ""


def test_can_restore_false_when_navigation_disabled():
    fake = _build_fake_engine(snapshots=_make_ten_snapshots(), history_index=2)
    fake.navigation_debug_enabled = False
    can_restore, reason = fake.can_restore_navigation_debug_snapshot()
    assert can_restore is False
    assert "enable navigation" in reason.lower()


def test_can_restore_false_for_multi_robot_mode():
    fake = _build_fake_engine(snapshots=_make_ten_snapshots(), history_index=2, agent_mode="Multiple Robot Mode")
    can_restore, reason = fake.can_restore_navigation_debug_snapshot()
    assert can_restore is False
    assert "single-robot" in reason.lower()


# ---------------------------------------------------------------------------
# restore_navigation_debug_snapshot() -- the minimal scenario from the task:
# 10 snapshots, jump to #3 (index 2), Resume, verify everything, verify the
# simulation is left in a state ready to continue.
# ---------------------------------------------------------------------------


def test_restore_returns_false_when_not_actionable():
    fake = _build_fake_engine(snapshots=_make_ten_snapshots(), history_index=None)
    result = fake.restore_navigation_debug_snapshot()
    assert result is False
    assert fake.simulation_time == 999.0  # untouched


def test_restore_rewinds_time_and_pose():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)

    result = fake.restore_navigation_debug_snapshot()

    assert result is True
    target = snapshots[2]
    assert fake.simulation_time == target.simulation_time
    assert fake.robot.x == target.robot_pose.x
    assert fake.robot.y == target.robot_pose.y
    assert fake.robot.theta == target.robot_pose.theta
    assert fake.robot.v == target.robot_pose.v


def test_restore_defaults_to_the_currently_viewed_history_index():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=4)

    fake.restore_navigation_debug_snapshot()  # no explicit index passed

    assert fake.simulation_time == snapshots[4].simulation_time


def test_restore_truncates_future_snapshots():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)
    assert len(fake.navigation_debug_log) == 10

    fake.restore_navigation_debug_snapshot()

    assert len(fake.navigation_debug_log) == 3  # indices 0, 1, 2 survive
    assert fake.navigation_debug_log.event_at(3) is None
    assert fake._nav_debug_seq == snapshots[2].snapshot_id


def test_restore_returns_view_to_live():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)

    fake.restore_navigation_debug_snapshot()

    assert fake._nav_debug_history_index is None
    assert fake.canvas.history_positions[-1] == (None, 3)


def test_restore_pauses_the_simulation():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)
    fake.paused = False  # arrange an unusual pre-state to prove step 1 forces it

    fake.restore_navigation_debug_snapshot()

    assert fake.paused is True


def test_restore_clears_stale_async_planner_state():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)
    assert fake.route_request_id == 5

    fake.restore_navigation_debug_snapshot()

    assert fake.planning_in_progress is False
    assert fake.active_planner_workers == {}
    assert fake.prefetch_workers == {}
    assert fake.prefetch_request_ids == {}
    assert fake.route_request_id == 6  # bumped, invalidating any in-flight worker
    assert fake._nav_debug_pending_plan_capture_by_robot == {}
    assert fake._nav_debug_last_plan_capture is None
    assert fake._nav_debug_last_accepted_plan is None


def test_restore_restores_belief_grid_and_explored_area():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)
    frame = snapshots[2].belief_map.value
    expected_grid = np.frombuffer(zlib.decompress(frame.grid_zlib), dtype=np.int8).reshape(frame.grid_shape)
    expected_explored = np.unpackbits(
        np.frombuffer(zlib.decompress(frame.explored_packbits_zlib), dtype=np.uint8),
        bitorder="little",
        count=int(np.prod(frame.explored_shape)),
    ).reshape(frame.explored_shape).astype(bool)
    revision_before = fake.belief_map.revision

    fake.restore_navigation_debug_snapshot()

    assert np.array_equal(fake.belief_map.grid, expected_grid)
    assert np.array_equal(fake.belief_map.explored_by_robot, expected_explored)
    assert fake.belief_map.revision > revision_before
    # The bounded sensor-sweep polygon list is cleared (see the
    # NavigationDebugSnapshot docstring) -- belief_map.explored_by_robot
    # above is the authoritative state and IS restored exactly.
    assert fake.explored_area_polygons == []
    assert fake.canvas.explored_area_polygons == []
    # But the visible explored-area *coverage* must not regress just
    # because that polygon list is gone -- the canvas is reseeded directly
    # from the same restored mask instead.
    assert fake.canvas.explored_area_seed is not None
    seeded_mask, seeded_resolution, seeded_bounds = fake.canvas.explored_area_seed
    assert np.array_equal(seeded_mask, expected_explored)
    assert seeded_resolution == frame.resolution
    assert tuple(seeded_bounds) == fake.belief_map.bounds


def test_restore_truncates_mapped_obstacle_points_using_the_append_only_invariant():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)
    assert len(fake.mapped_obstacle_points) == 6  # the "current/future" live list

    fake.restore_navigation_debug_snapshot()

    expected_count = snapshots[2].mapped_obstacle_points_count
    assert len(fake.mapped_obstacle_points) == expected_count
    assert len(fake.mapped_obstacle_point_keys) == expected_count


def test_restore_restores_route_and_active_waypoint_index():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)

    fake.restore_navigation_debug_snapshot()

    target = snapshots[2]
    restored_waypoints = [(float(p[0]), float(p[1])) for p in fake.robot.waypoints.waypoints]
    assert restored_waypoints == list(target.path.active_path)
    assert fake.robot.waypoints.current_index == target.path.active_waypoint_index
    assert fake.agent.waypoints.current_index == target.path.active_waypoint_index
    assert fake.agent.status == target.navigation_state

    # active_path_goal_xy comes from the explicit AgentStateDebug field, NOT
    # active_path[-1] -- the fixture deliberately makes them differ (52, 52)
    # vs (3, 2) to prove this is never inferred.
    state = target.agent_state.value
    assert state.active_path_goal_xy != tuple(target.path.active_path[-1])
    assert fake.agent.active_path_goal_xy == state.active_path_goal_xy
    assert fake.agent.final_goal_xy == state.final_goal_xy
    assert fake.agent.exploration_target_xy == state.exploration_target_xy
    assert fake.agent.active_path_mode == state.active_path_mode
    assert fake.agent.route_generation == state.route_generation
    assert fake.current_exploration_target == state.exploration_target_xy
    assert fake.canvas.exploration_target == state.exploration_target_xy


def test_restore_clears_route_when_snapshot_had_no_active_path():
    snapshots = _make_ten_snapshots()
    idle_snapshot = _make_snapshot(
        snapshot_id=500,
        simulation_time=3.0,
        pose=(1.0, 1.0, 0.0, 0.0),
        active_path=(),
        active_waypoint_index=None,
        navigation_state="idle",
        mapped_obstacle_points_count=2,
        belief_frame=snapshots[0].belief_map.value,
    )
    fake = _build_fake_engine(snapshots=[snapshots[0], idle_snapshot], history_index=1)
    # Give the agent a stale route the restore must clear.
    fake.agent.assign_path(target=(9.0, 9.0), waypoints=[(9.0, 9.0)], planner_reason="stale")

    fake.restore_navigation_debug_snapshot()

    assert fake.robot.waypoints.has_path() is False
    assert fake.agent.active_path_goal_xy is None
    assert fake.agent.status == "idle"


def test_restore_is_a_noop_when_the_snapshot_has_no_belief_map():
    snapshot_without_belief = _make_snapshot(
        snapshot_id=1, simulation_time=1.0, pose=(0.0, 0.0, 0.0, 0.0), belief_frame=None
    )
    fake = _build_fake_engine(snapshots=[snapshot_without_belief], history_index=0)

    result = fake.restore_navigation_debug_snapshot()

    assert result is False
    assert fake.simulation_time == 999.0


# ---------------------------------------------------------------------------
# Hazards. Occupancy (belief_map.grid) is asserted untouched in every case --
# hazards and occupancy are separate layers (see HazardField's module
# docstring), and restore must not blur that line.
# ---------------------------------------------------------------------------


def test_restore_removes_a_fire_that_was_only_added_in_the_future():
    """snapshot 3 (index 2) has no fire; a fire is then added live (the
    "future" relative to that snapshot); restoring must leave no fire and
    no heat."""
    snapshots = _make_ten_snapshots()  # every snapshot's hazard_frame is empty
    fake = _build_fake_engine(snapshots=snapshots, history_index=2, live_fire_at=(0.5, 0.5))
    assert len(fake.hazard_service.field.sources()) == 1  # the future fire exists before restore
    frame = snapshots[2].belief_map.value
    expected_grid = np.frombuffer(zlib.decompress(frame.grid_zlib), dtype=np.int8).reshape(frame.grid_shape)

    fake.restore_navigation_debug_snapshot()

    assert fake.hazard_service.field.sources() == ()
    assert not np.any(fake.hazard_service.field.values(copy=False)), "no heat must remain either"
    # Occupancy reflects only the belief-map restore (snapshot 2's own
    # OCCUPIED cell) -- hazard restore must not additionally touch it.
    assert np.array_equal(fake.belief_map.grid, expected_grid), "hazard restore must not touch occupancy"
    assert fake.canvas.hazard_snapshots, "push_hazard_snapshot() must run so the render cache invalidates"


def test_restore_brings_back_a_fire_that_was_removed_in_the_future():
    """snapshot 3 (index 2) has a fire; it is then removed live (the
    "future"); restoring must make it reappear with its exact original
    position/intensity/radius."""
    belief_shape = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1).grid.shape
    grid = np.full(belief_shape, UNKNOWN, dtype=np.int8)
    explored = np.zeros((1,) + belief_shape, dtype=bool)
    frame = _make_belief_frame(grid, explored, revision=1, resolution=_RESOLUTION, bounds=_BOUNDS)
    fire_source = HazardSourceDebug(fire_id=1, position=(0.5, 0.5), intensity=1.0, radius=2.0)
    snapshot_with_fire = _make_snapshot(
        snapshot_id=1,
        simulation_time=1.0,
        pose=(0.0, 0.0, 0.0, 0.0),
        belief_frame=frame,
        hazard_frame=_make_hazard_frame((fire_source,), next_fire_id=2),
    )
    fake = _build_fake_engine(
        snapshots=[snapshot_with_fire], history_index=0, live_fire_at=(0.5, 0.5)
    )
    # Simulate the fire being removed after the snapshot was captured.
    live_fire = fake.hazard_service.field.sources()[0]
    fake.hazard_service.field.remove_fire(live_fire.fire_id)
    assert fake.hazard_service.field.sources() == ()

    fake.restore_navigation_debug_snapshot()

    restored = fake.hazard_service.field.sources()
    assert len(restored) == 1
    assert restored[0].position == fire_source.position
    assert restored[0].intensity == fire_source.intensity
    assert restored[0].radius == fire_source.radius
    assert np.any(fake.hazard_service.field.values(copy=False)), "heat must reappear too"


def test_restore_sets_next_fire_id_from_the_snapshot():
    belief_shape = BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1).grid.shape
    grid = np.full(belief_shape, UNKNOWN, dtype=np.int8)
    explored = np.zeros((1,) + belief_shape, dtype=bool)
    frame = _make_belief_frame(grid, explored, revision=1, resolution=_RESOLUTION, bounds=_BOUNDS)
    snapshot = _make_snapshot(
        snapshot_id=1,
        simulation_time=1.0,
        pose=(0.0, 0.0, 0.0, 0.0),
        belief_frame=frame,
        hazard_frame=_make_hazard_frame((), next_fire_id=41),
    )
    fake = _build_fake_engine(snapshots=[snapshot], history_index=0)

    fake.restore_navigation_debug_snapshot()

    new_fire = fake.hazard_service.add_fire((1.0, 1.0))
    assert new_fire.source.fire_id == 41


# ---------------------------------------------------------------------------
# Cumulative engine-level metrics (RuntimeMetricsDebug).
# ---------------------------------------------------------------------------


def test_restore_restores_cumulative_metrics_instead_of_leaving_live_totals():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)
    # The fixture seeds the live engine with metrics far ahead (as if the
    # run continued ~42s past the snapshot at t=1.0s) -- restore must not
    # leave simulation_time behind while these keep reading that later total.
    assert fake.total_distance_traveled == 42.0

    fake.restore_navigation_debug_snapshot()

    target_metrics = snapshots[2].metrics.value
    assert fake.total_distance_traveled == target_metrics.total_distance_traveled
    assert fake.route_request_count == target_metrics.route_request_count
    assert fake.route_result_count == target_metrics.route_result_count
    assert fake.route_failure_count == target_metrics.route_failure_count
    assert fake.sensor_update_count == target_metrics.sensor_update_count
    assert fake.mapping_update_count == target_metrics.mapping_update_count
    assert fake.safety_replan_count == target_metrics.safety_replan_count
    assert fake.exploration_replan_count == target_metrics.exploration_replan_count
    assert fake.planner_jobs_started == target_metrics.planner_jobs_started
    assert fake.planner_jobs_completed == target_metrics.planner_jobs_completed
    assert fake.simulation_time == snapshots[2].simulation_time
    # The literal failure mode called out in the task: simulation_time must
    # never end up behind cumulative counters from a later run.
    assert fake.simulation_time < 2.0
    assert fake.total_distance_traveled < 42.0


def test_restore_leaves_metrics_untouched_when_snapshot_lacks_them():
    snapshot_without_metrics = _make_snapshot(
        snapshot_id=1,
        simulation_time=1.0,
        pose=(0.0, 0.0, 0.0, 0.0),
        belief_frame=_make_belief_frame(
            np.full(BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1).grid.shape, UNKNOWN, dtype=np.int8),
            np.zeros((1,) + BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=1).grid.shape, dtype=bool),
            revision=1,
            resolution=_RESOLUTION,
            bounds=_BOUNDS,
        ),
    )
    fake = _build_fake_engine(snapshots=[snapshot_without_metrics], history_index=0)

    fake.restore_navigation_debug_snapshot()

    assert fake.total_distance_traveled == 42.0  # unchanged, no crash


# ---------------------------------------------------------------------------
# Pending path / prefetch bookkeeping is always cleared, regardless of
# whether an active path exists after restore.
# ---------------------------------------------------------------------------


def test_restore_clears_pending_path_and_prefetch_bookkeeping():
    snapshots = _make_ten_snapshots()
    fake = _build_fake_engine(snapshots=snapshots, history_index=2)
    fake.agent.pending_path = [(9.0, 9.0)]
    fake.agent.pending_target_xy = (9.0, 9.0)
    fake.agent.pending_path_route_generation = 999
    fake.agent.pending_path_created_for_active_goal = (9.0, 9.0)
    fake.agent.route_repair_in_progress_for_goal = (9.0, 9.0)

    fake.restore_navigation_debug_snapshot()

    assert fake.agent.pending_path is None
    assert fake.agent.pending_target_xy is None
    assert fake.agent.pending_path_route_generation is None
    assert fake.agent.pending_path_created_for_active_goal is None
    assert fake.agent.route_repair_in_progress_for_goal is None


# ---------------------------------------------------------------------------
# End-to-end: a real MainWindow, a real Direct-planner single-robot run,
# real simulation_step() ticks -- no fakes anywhere in this test. The other
# tests in this file prove each piece of restore logic in isolation; this
# one proves the real engine's collaborators (Robot, RobotAgent,
# RuntimeRobotRegistry, BeliefMap, RuntimeHazardService, PlannerWorker
# bookkeeping) actually accept being rewound and keep working afterward.
# ---------------------------------------------------------------------------


def test_end_to_end_restore_then_continue_on_a_real_running_simulation():
    window = MainWindow()
    window.on_navigation_debug_toggled(True)
    window.start_simulation()
    assert window.running is True

    for _ in range(15):
        window.simulation_step()

    history_length_before_restore = window.navigation_debug_history_length()
    assert history_length_before_restore >= 10, "need enough history to pick an earlier snapshot"

    restore_index = 5
    target_event = window.navigation_debug_log.event_at(restore_index)
    target_snapshot = target_event.snapshot
    old_route_request_id = window.route_request_id

    # Simulate an async planner callback that was already in flight before
    # the user restored -- it must be silently ignored, not applied on top
    # of the restored state (on_async_route_ready()'s own request_id guard
    # is what does this; route_request_id is bumped by restore below).
    stale_waypoints = [(-99.0, -99.0)]

    # Enter HISTORY at restore_index first -- Resume is only actionable once
    # a historical snapshot is actually selected (can_restore_navigation_
    # debug_snapshot() requires it), same as clicking `<` in the real UI.
    window.paused = True
    window._push_navigation_debug_history_view(restore_index)

    ok = window.restore_navigation_debug_snapshot()

    assert ok is True
    # -- time advances from the restored point, not from before restore --
    assert window.simulation_time == target_snapshot.simulation_time
    assert window.simulation_time < 15 * 0.02  # well before the pre-restore run's t

    # -- history truncated at the restore point --
    assert window.navigation_debug_history_length() == restore_index + 1
    assert window.navigation_debug_log.event_at(restore_index + 1) is None

    # -- robot pose restored --
    assert window.robot.x == target_snapshot.robot_pose.x
    assert window.robot.y == target_snapshot.robot_pose.y
    assert window.robot.theta == target_snapshot.robot_pose.theta

    # -- a callback from the discarded future is a no-op, not applied --
    window.on_async_route_ready(old_route_request_id, True, "stale future callback", stale_waypoints)
    for point in window.robot.waypoints.waypoints:
        assert (float(point[0]), float(point[1])) != (-99.0, -99.0)

    # -- robot and RobotAgent are synchronized right after restore --
    agent = window.runtime_agent(None)
    assert agent is not None
    assert agent.status == target_snapshot.navigation_state
    assert agent.pending_path is None, "no leftover prefetch from the discarded future"

    # -- reanudar: resume ticking from the restored point --
    window.paused = False
    time_after_restore = window.simulation_time
    for _ in range(3):
        window.simulation_step()

    # -- time keeps advancing forward from the restored point, no exception --
    assert window.simulation_time > time_after_restore

    # -- a new snapshot was appended after the truncated point --
    assert window.navigation_debug_history_length() == restore_index + 1 + 3

    # -- robot/agent stay coherent: pose is finite, a target exists (Direct
    # planner + a goal-seeking robot away from goal keeps tracking one),
    # and the route/target the engine reports is internally consistent --
    assert window.robot.x == window.robot.x  # not NaN
    assert window.robot.y == window.robot.y  # not NaN
    active_target = window.active_target_xy()
    if active_target is not None:
        assert active_target == active_target  # not NaN
    agent_after = window.runtime_agent(None)
    assert agent_after is not None
    assert agent_after.status in {"idle", "moving", "finished", "blocked", "responding_event"}


def test_end_to_end_capture_runs_before_navigation_is_ever_enabled():
    """Capture is unconditional (see engine._finalize_navigation_debug_
    snapshot()'s docstring): ticks are recorded regardless of navigation_
    debug_enabled, so turning the switch on mid-run finds history already
    there instead of an empty log the user has to wait out."""
    window = MainWindow()
    assert window.navigation_debug_enabled is False
    window.start_simulation()
    assert window.running is True

    for _ in range(10):
        window.simulation_step()

    # -- history already exists even though Navigation was never turned on --
    assert window.navigation_debug_history_length() >= 10

    # -- turning it on now (real switch, real signal) finds that history
    # immediately, with no gap and no need to wait for new ticks --
    history_length_at_toggle = window.navigation_debug_history_length()
    window.navigation_snapshot_switch.setChecked(True)
    assert window.navigation_debug_enabled is True
    assert window.navigation_debug_history_length() == history_length_at_toggle

    # -- pausing now makes that pre-existing history immediately browsable --
    window.paused = True
    window.step_navigation_debug_history(-1)
    assert window._nav_debug_history_index is not None
    selected = window.navigation_debug_log.event_at(window._nav_debug_history_index)
    assert selected is not None
    assert selected.snapshot.simulation_time <= window.simulation_time

    # -- Resume from snapshot works immediately too, no warm-up tick needed --
    can_restore, _reason = window.can_restore_navigation_debug_snapshot()
    assert can_restore is True
