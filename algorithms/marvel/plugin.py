"""MARVEL CTDE coordination-plugin boundary.

Paper: Zhang et al., "MARVEL: Multi-Agent Reinforcement Learning for
constrained field-of-view multi-robot exploration", arXiv:2502.20217 (2025).

The plugin is intentionally discoverable before the large checkpoint is
installed. It reports an explicit HOLD instead of silently substituting a
heuristic policy when the official weights are absent.
"""

from __future__ import annotations

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import (
    CoordinationAssignment,
    CoordinationRequest,
    CoordinationResult,
)
from robotics_interfaces.plugins import (
    CandidateInputMode,
    CoordinationPlugin,
    PluginCapability,
    PluginMetadata,
)

from algorithms.marvel.runtime import MarvelRuntimeConfiguration


MARVEL_COORDINATOR = "MARVEL CTDE graph-attention policy"
MARVEL_SOURCE = "https://arxiv.org/abs/2502.20217"


class MarvelPlugin:
    metadata = PluginMetadata(
        name=MARVEL_COORDINATOR,
        version="0.1.0",
        description=(
            "MARVEL graph-attention waypoint-heading policy using centralized "
            "training, decentralized execution, perfect communication, and a "
            "shared occupancy map."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TASK_GENERATION,
            PluginCapability.TASK_ALLOCATION,
        ),
        source=MARVEL_SOURCE,
        candidate_input_mode=CandidateInputMode.PLUGIN_INTERNAL,
    )

    def __init__(self) -> None:
        self.runtime = MarvelRuntimeConfiguration.from_environment()
        self._policy = None

    def assign(self, request: CoordinationRequest) -> CoordinationResult:
        readiness_error = self.runtime.readiness_error()
        if readiness_error is not None:
            return self._hold(request, readiness_error)

        # Loading validates that the user supplied the authors' expected
        # PolicyNet checkpoint. The simulator-to-tensor bridge is a separate,
        # swappable architecture boundary and must be registered explicitly;
        # never replace the paper policy with an uncited heuristic.
        if self._policy is None:
            try:
                self._policy = self.runtime.load_policy()
            except Exception as exc:
                return self._hold(
                    request,
                    "MARVEL runtime could not load the official PolicyNet "
                    f"checkpoint: {exc}",
                )

        backend = request.shared.get("marvel_observation_backend")
        if backend is None:
            return self._hold(
                request,
                "MARVEL weights loaded; simulator observation backend is not "
                "registered yet.",
            )
        return backend.assign(request, self._policy)

    def _hold(self, request: CoordinationRequest, reason: str) -> CoordinationResult:
        requested = set(request.robots_to_assign)
        if not requested:
            requested = {
                robot.robot_id for robot in request.robot_states if robot.is_active
            }
        assignments = tuple(
            CoordinationAssignment(robot.robot_id, "HOLD", None, reason)
            for robot in request.robot_states
            if robot.robot_id in requested
        )
        commands = tuple(
            RobotCommand(robot_id=item.robot_id, status="HOLD", reason=reason)
            for item in assignments
        )
        return CoordinationResult(
            targets=tuple(robot.current_target for robot in request.robot_states),
            reasons=tuple(reason for _ in request.robot_states),
            strategy=self.metadata.name,
            assignments=assignments,
            commands=commands,
            debug={
                "checkpoint": str(self.runtime.checkpoint_path),
                "ready": False,
                "paper_source": MARVEL_SOURCE,
            },
        )


def create_plugin() -> CoordinationPlugin:
    return MarvelPlugin()
