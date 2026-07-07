# robotics_sim/tests/test_team_frontier_provider_candidate_pool.py

from __future__ import annotations

import pytest

from robotics_interfaces.coordination import CoordinationRequest
from robotics_interfaces.observations import RobotCoordinationState, WorldSnapshot
from robotics_sim.simulation.coordination_services import RuntimeTeamFrontierProvider


def _robot(robot_id: int, xy: tuple[float, float]) -> RobotCoordinationState:
    return RobotCoordinationState(
        robot_id=robot_id,
        xy=xy,
        safety_radius=0.35,
        sensor_range=2.5,
        vision_model="LiDAR",
    )


def _world_with_multiple_frontier_sources() -> WorldSnapshot:
    # Two separated explored islands are enough to create multiple global
    # frontier candidates. The provider should expose those candidates as a
    # pool to every active robot; it must not pre-allocate one target per robot.
    return WorldSnapshot(
        explored_points=(
            (-2.0, 0.0),
            (-1.5, 0.0),
            (1.5, 0.0),
            (2.0, 0.0),
        ),
        mapped_obstacle_points=(),
        bounds=(-5.0, 5.0, -5.0, 5.0),
        resolution=0.5,
        final_goal_xy=(5.0, 5.0),
    )


def _debug_candidate_pools(pools: dict[int, tuple] | object) -> str:
    if not isinstance(pools, dict):
        return f"candidate_pools is not a dict: {type(pools).__name__} -> {pools!r}"

    lines: list[str] = []
    for robot_id, candidates in sorted(pools.items()):
        lines.append(f"robot {robot_id}: {len(candidates)} candidate(s)")
        for index, candidate in enumerate(candidates[:8]):
            metadata = dict(getattr(candidate, "metadata", {}) or {})
            lines.append(
                "  "
                f"[{index}] target={getattr(candidate, 'target', None)} "
                f"source={getattr(candidate, 'source', None)} "
                f"utility={getattr(candidate, 'utility', None)} "
                f"metadata={metadata}"
            )
    return "\n".join(lines) if lines else "candidate_pools is empty"


def test_team_frontier_provider_returns_candidate_pools_without_task_assignment(monkeypatch):
    """RuntimeTeamFrontierProvider must generate candidate pools, not allocate.

    This is intentionally a red test for the current legacy behavior.

    The current provider calls assign_frontier_viewpoints(), which already
    performs team assignment and returns one target per robot. That couples the
    provider to allocation and prevents plugins from comparing the same
    candidate pool with their own policy.
    """
    from robotics_sim.simulation import coordination_services as services_module

    def forbidden_assignment_call(**kwargs):
        pytest.fail(
            "RuntimeTeamFrontierProvider must not call assign_frontier_viewpoints().\n"
            "That function performs team assignment; a TeamFrontierProvider "
            "should only expose raw candidate pools.\n"
            f"robots_to_assign={kwargs.get('robots_to_assign')!r}\n"
            f"existing_targets={kwargs.get('existing_targets')!r}\n"
            f"invalidated_targets_by_robot={kwargs.get('invalidated_targets_by_robot')!r}"
        )

    monkeypatch.setattr(
        services_module,
        "assign_frontier_viewpoints",
        forbidden_assignment_call,
    )

    request = CoordinationRequest(
        robot_states=(
            _robot(0, (-2.0, -1.0)),
            _robot(1, (2.0, -1.0)),
        ),
        robots_to_assign=(0, 1),
        world=_world_with_multiple_frontier_sources(),
        existing_targets_by_robot={0: None, 1: None},
        blocked_targets_by_robot={0: (), 1: ()},
        route_points_by_robot=((), ()),
        shared={
            "explored_points_by_robot": (
                ((-2.0, 0.0),),
                ((2.0, 0.0),),
            )
        },
    )

    provider = RuntimeTeamFrontierProvider(ipp_distance_penalty=0.25)
    candidate_pools = dict(provider.candidates_for_team(request))
    debug = _debug_candidate_pools(candidate_pools)

    assert set(candidate_pools) == {0, 1}, debug

    for robot_id in (0, 1):
        candidates = candidate_pools[robot_id]

        assert len(candidates) >= 2, (
            "Expected a candidate pool with at least two frontier options per robot.\n"
            "A single candidate usually means the provider still pre-assigned a target.\n"
            f"Debug candidate pools:\n{debug}"
        )

        unique_targets = {candidate.target for candidate in candidates}
        assert len(unique_targets) >= 2, (
            "Expected multiple distinct frontier targets in the provider output.\n"
            f"Debug candidate pools:\n{debug}"
        )

        for candidate in candidates:
            assert candidate.source == "runtime_team_frontier_provider", debug
            assert candidate.metadata.get("provider") == "RuntimeTeamFrontierProvider", debug
            assert candidate.metadata.get("team_synchronized") is True, debug