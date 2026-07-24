"""Tests for MultiRobotCoordinator's optional request_executor seam.

assign_frontiers() can either call self.plugin.assign(request) (the default,
legacy path) or delegate to a caller-supplied request_executor callback. The
guarantee under test is that exactly one of those two ever runs per call, and
that both paths converge on the same _adapt_plugin_result() adaptation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from robotics_interfaces import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
    RobotCommand,
)
from robotics_interfaces.plugins import CandidateInputMode, PluginCapability, PluginMetadata
from robotics_sim.simulation import coordination as sim_coord


def _robot_state(x: float, y: float) -> sim_coord.RobotCoordinationState:
    return sim_coord.RobotCoordinationState(
        xy=(x, y),
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="Camera / FoV",
    )


def _result_for(count: int, *, strategy: str = "fake plugin result") -> CoordinationResult:
    targets = tuple((float(i), float(i)) for i in range(count))
    reasons = tuple(f"reason-{i}" for i in range(count))
    commands = tuple(
        RobotCommand(robot_id=i, status="ASSIGNED", target=targets[i], reason=reasons[i])
        for i in range(count)
    )
    assignments = tuple(
        CoordinationAssignment(robot_id=i, status="ASSIGNED", target=targets[i], reason=reasons[i])
        for i in range(count)
    )
    return CoordinationResult(
        targets=targets,
        reasons=reasons,
        strategy=strategy,
        assignments=assignments,
        debug={"fake": True},
        commands=commands,
    )


class _CountingPlugin:
    """Deterministic test double: counts assign() calls and remembers requests."""

    metadata = PluginMetadata(
        name="counting test plugin",
        version="0.0.0",
        description="counts assign() calls; must stay uncalled when an executor runs",
        capabilities=(PluginCapability.COORDINATION, PluginCapability.TASK_ALLOCATION),
        candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
    )

    def __init__(self, result: CoordinationResult):
        self.calls = 0
        self.received_requests: list[CoordinationRequest] = []
        self._result = result

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        self.calls += 1
        self.received_requests.append(request)
        return self._result


class _CountingExecutor:
    """Deterministic request_executor double: counts calls and remembers requests."""

    def __init__(self, result: CoordinationResult):
        self.calls = 0
        self.received_requests: list[CoordinationRequest] = []
        self._result = result

    def __call__(self, request: CoordinationRequest) -> CoordinationResult:
        self.calls += 1
        self.received_requests.append(request)
        return self._result


def _make_coordinator(plugin) -> sim_coord.MultiRobotCoordinator:
    """Build a coordinator around a test-double plugin, bypassing the dynamic
    plugin loader entirely (mirrors the pattern used in
    test_exploration_pipeline_characterization.py's _patch_coordinator)."""

    from robotics_interfaces.plugins import build_runtime_profile

    coordinator = sim_coord.MultiRobotCoordinator.__new__(sim_coord.MultiRobotCoordinator)
    coordinator.plugin = plugin
    coordinator.strategy = plugin.metadata.name
    coordinator.runtime_profile = build_runtime_profile(plugin.metadata)
    return coordinator


def _call_kwargs(count: int) -> dict:
    robot_states = [_robot_state(float(i), 0.0) for i in range(count)]
    return dict(
        planner_name="test planner",
        robot_states=robot_states,
        existing_targets=[None] * count,
        robots_to_assign=list(range(count)),
        invalidated_targets_by_robot=[[] for _ in range(count)],
        explored_points=[(0.0, 0.0)],
        mapped_obstacle_points=[],
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
        route_points_by_robot=[[] for _ in range(count)],
        explored_points_by_robot=[[(0.0, 0.0)] for _ in range(count)],
    )


# --- Default path (no executor): legacy behavior must stay identical. ------


class TestDefaultPathUnchanged:
    def test_plugin_assign_called_exactly_once(self):
        plugin = _CountingPlugin(_result_for(2))
        coordinator = _make_coordinator(plugin)

        coordinator.assign_frontiers(**_call_kwargs(2))

        assert plugin.calls == 1

    def test_result_is_adapted(self):
        plugin = _CountingPlugin(_result_for(2, strategy="counting test plugin"))
        coordinator = _make_coordinator(plugin)

        result = coordinator.assign_frontiers(**_call_kwargs(2))

        assert result.targets == ((0.0, 0.0), (1.0, 1.0))
        assert result.reasons == ("reason-0", "reason-1")
        assert result.strategy == "counting test plugin"
        assert result.debug == {"fake": True}
        assert len(result.commands) == 2
        assert len(result.assignments) == 2

    def test_request_contains_expected_robots_and_services(self):
        plugin = _CountingPlugin(_result_for(2))
        coordinator = _make_coordinator(plugin)

        coordinator.assign_frontiers(**_call_kwargs(2))

        request = plugin.received_requests[0]
        assert request.robots_to_assign == (0, 1)
        assert len(request.robot_states) == 2
        assert request.services is not None
        assert request.world is not None
        assert request.parameters["planner_name"] == "test planner"


# --- request_executor path ---------------------------------------------------


class TestRequestExecutorSeam:
    def test_executor_called_once_and_plugin_not_called(self):
        plugin = _CountingPlugin(_result_for(2))
        coordinator = _make_coordinator(plugin)
        executor = _CountingExecutor(_result_for(2))

        coordinator.assign_frontiers(**_call_kwargs(2), request_executor=executor)

        assert executor.calls == 1
        assert plugin.calls == 0

    def test_executor_receives_the_single_built_request(self, monkeypatch):
        plugin = _CountingPlugin(_result_for(2))
        coordinator = _make_coordinator(plugin)
        executor = _CountingExecutor(_result_for(2))

        original_build = coordinator._build_plugin_request
        built_requests: list[CoordinationRequest] = []

        def spy_build(**kwargs):
            built = original_build(**kwargs)
            built_requests.append(built)
            return built

        monkeypatch.setattr(coordinator, "_build_plugin_request", spy_build)

        coordinator.assign_frontiers(**_call_kwargs(2), request_executor=executor)

        assert len(built_requests) == 1
        assert executor.received_requests[0] is built_requests[0]

    def test_executor_request_preserves_expected_fields(self):
        plugin = _CountingPlugin(_result_for(2))
        coordinator = _make_coordinator(plugin)
        executor = _CountingExecutor(_result_for(2))
        kwargs = _call_kwargs(2)

        coordinator.assign_frontiers(**kwargs, request_executor=executor)

        request = executor.received_requests[0]
        assert request.robots_to_assign == (0, 1)
        assert [state.xy for state in request.robot_states] == [(0.0, 0.0), (1.0, 0.0)]
        assert request.world is not None
        assert request.world.bounds == kwargs["bounds"]
        assert request.services is not None
        assert request.services.frontier_provider is not None
        assert request.parameters["planner_name"] == "test planner"
        assert request.blocked_targets_by_robot == {0: (), 1: ()}

    def test_executor_result_is_adapted(self):
        plugin = _CountingPlugin(_result_for(2))
        coordinator = _make_coordinator(plugin)
        executor = _CountingExecutor(_result_for(2, strategy="executor-provided strategy"))

        result = coordinator.assign_frontiers(**_call_kwargs(2), request_executor=executor)

        assert result.strategy == "executor-provided strategy"
        assert result.targets == ((0.0, 0.0), (1.0, 1.0))
        assert result.reasons == ("reason-0", "reason-1")
        assert len(result.commands) == 2

    def test_non_callable_executor_rejected(self):
        plugin = _CountingPlugin(_result_for(1))
        coordinator = _make_coordinator(plugin)

        with pytest.raises(TypeError):
            coordinator.assign_frontiers(**_call_kwargs(1), request_executor="not callable")

        assert plugin.calls == 0

    def test_invalid_executor_result_rejected(self):
        plugin = _CountingPlugin(_result_for(1))
        coordinator = _make_coordinator(plugin)

        def bad_executor(request):
            return {"not": "a CoordinationResult"}

        with pytest.raises(TypeError):
            coordinator.assign_frontiers(**_call_kwargs(1), request_executor=bad_executor)

        assert plugin.calls == 0

    def test_executor_exception_propagates_unwrapped(self):
        plugin = _CountingPlugin(_result_for(1))
        coordinator = _make_coordinator(plugin)

        class _Boom(Exception):
            pass

        def raising_executor(request):
            raise _Boom("executor failed")

        with pytest.raises(_Boom):
            coordinator.assign_frontiers(**_call_kwargs(1), request_executor=raising_executor)

    def test_successive_calls_do_not_share_executor_or_state(self):
        plugin = _CountingPlugin(_result_for(1))
        coordinator = _make_coordinator(plugin)
        executor_a = _CountingExecutor(_result_for(1))

        coordinator.assign_frontiers(**_call_kwargs(1), request_executor=executor_a)
        assert executor_a.calls == 1
        assert plugin.calls == 0

        # No executor on the second call: must fall back to the plugin, and
        # must not silently keep reusing executor_a.
        coordinator.assign_frontiers(**_call_kwargs(1))
        assert executor_a.calls == 1
        assert plugin.calls == 1

        executor_b = _CountingExecutor(_result_for(1))
        coordinator.assign_frontiers(**_call_kwargs(1), request_executor=executor_b)
        assert executor_b.calls == 1
        assert executor_a.calls == 1
        assert plugin.calls == 1

    def test_no_double_execution_matches_spec(self):
        """Core criterion: with an executor the plugin never runs, and
        without one the executor is never consulted."""

        plugin_with_executor = _CountingPlugin(_result_for(1))
        coordinator_with_executor = _make_coordinator(plugin_with_executor)
        executor = _CountingExecutor(_result_for(1))

        coordinator_with_executor.assign_frontiers(**_call_kwargs(1), request_executor=executor)
        assert plugin_with_executor.calls == 0
        assert executor.calls == 1

        plugin_without_executor = _CountingPlugin(_result_for(1))
        coordinator_without_executor = _make_coordinator(plugin_without_executor)
        unused_executor = _CountingExecutor(_result_for(1))

        coordinator_without_executor.assign_frontiers(**_call_kwargs(1))
        assert plugin_without_executor.calls == 1
        assert unused_executor.calls == 0


# --- Architectural boundary: coordination.py stays learning-agnostic. ------


_FORBIDDEN_MODULE_COMPONENTS = {"learning", "engine", "app", "pandas", "torch"}
_FORBIDDEN_NAMES = {"RuntimeLearningCaptureService", "engine", "app"}


def test_coordination_module_has_no_learning_or_forbidden_imports():
    source_path = Path(__file__).resolve().parents[1] / "simulation" / "coordination.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                components = set(alias.name.split("."))
                if components & _FORBIDDEN_MODULE_COMPONENTS or "qt" in alias.name.lower():
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            components = set(module.split("."))
            if components & _FORBIDDEN_MODULE_COMPONENTS or "qt" in module.lower():
                violations.append(module)
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if (
                    alias.name in _FORBIDDEN_NAMES
                    or bound_name in _FORBIDDEN_NAMES
                    or "qt" in alias.name.lower()
                ):
                    violations.append(f"{module}.{alias.name}")

    assert not violations, f"coordination.py has forbidden imports: {violations}"
