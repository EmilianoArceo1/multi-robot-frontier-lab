"""
Tests answering one question: does the simulator ever assign an exploration
target INSIDE a completely known region, or are the markers users observe
"over blue" (already-explored area) actually valid frontiers?

Project definition of frontier (see exploration_planners.is_frontier_cell()):
    a FREE/observed cell with at least one UNKNOWN 4-neighbor.
A frontier target legitimately sits INSIDE the explored area's visual
footprint -- that alone is never a bug (test 1 documents this explicitly).

Two real, reproduced defects this file guards against, both in the shared
FoV-aware planner (robotics_sim/planning/exploration_planners.py) and the
per-tick behavior that follows an already-assigned route
(robotics_sim/navigation/exploration_behavior.py):

1. Stale current-target reuse (test 3): FoVAwareDirectionalFrontierPlanner.
   select_goal() calls _current_candidate(belief, current_target), which
   used to accept ANY current_target cell that was merely not OCCUPIED --
   never checking whether it was still adjacent to UNKNOWN. Once every
   neighboring UNKNOWN cell around a previously-selected target had since
   been observed (by this robot or a teammate), the target still got
   injected into the candidate pool, and once it was the only candidate
   left (e.g. the whole map was by then fully explored), select_goal()
   picked it anyway, labeled "selected best FoV-aware target" -- a reason
   that reads like a fresh, informed choice, not a zero-information repeat
   of a dead cell. Fix: _current_candidate() now also requires
   is_frontier_cell(belief, cell).

2. Frontier-status churn while following (test 7): a target often stops
   satisfying the strict frontier predicate as a direct consequence of the
   approaching robot observing its last UNKNOWN neighbour. Invalidating on
   that single map update caused rapid target switches. Step 5 still performs
   its O(1) local check, but now gives remote targets a bounded grace period
   and commits locally sensed targets until normal arrival. A remote target
   that remains stale is still reselected after the grace period, and safety
   replans retain higher priority.

These tests exercise the real BeliefMap, the real FoV-aware planner (via
select_exploration_goal()), the real ExplorationBehavior/RobotAgent, and (for
the single-robot engine-level selector, select_navigation_goal()) a
duck-typed SimulationControllerMixin fake -- the same pattern already used
by test_navigation_snapshot_restore.py. No Qt, no canvas, no theme: validity
here depends only on the BeliefMap (see test 8).

select_navigation_goal_for_multi_robot() was inspected but is NOT exercised
directly here: it delegates to MultiRobotCoordinator / coordinated_frontier_
planner.py for actual frontier assignment, which are explicitly out of scope
for this change (see the task's coordinator/coordinated_frontier_planner.py
exclusions). Test 6 instead exercises select_exploration_goal() twice
against one shared BeliefMap with two robot poses/exclusions -- the same
underlying per-robot selection primitive both the single-robot path and
(indirectly, through the coordinator) the multi-robot path are built from,
and the one place "is this target still a real frontier" is actually
decided.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

from robotics_sim.core.robot_agent import RobotAgent
from robotics_sim.environment.belief_map import BeliefMap, FREE, UNKNOWN
from robotics_sim.navigation.exploration_behavior import ExplorationBehavior
from robotics_sim.planning.exploration_planners import (
    _frontier_cells,
    _neighbors4,
    is_frontier_cell,
    select_exploration_goal,
)
from robotics_sim.simulation.config import SimulationConfig
from robotics_sim.simulation.engine import SimulationControllerMixin
from robotics_sim.simulation.observation import RobotObservation
from robotics_sim.simulation.planner_services import PlannerServices

_BOUNDS = (-10.0, 10.0, -10.0, 10.0)
_RESOLUTION = 1.0


# ---------------------------------------------------------------------------
# Caracterizacion minima: per-target classification helper (TEST-only).
# ---------------------------------------------------------------------------


@dataclass
class TargetClassification:
    target_world: tuple[float, float]
    target_cell: tuple[int, int] | None
    belief_value: int | None
    explored_by_any_robot: bool
    unknown_neighbor_count: int
    is_current_frontier: bool
    source: str
    reason: str


def _classify_target(
    belief: BeliefMap,
    target_world: tuple[float, float],
    *,
    source: str,
    reason: str = "",
) -> TargetClassification:
    """Compute the fields needed to judge one exploration target, reusing
    the project's real geometry/neighbor definitions (_neighbors4,
    is_frontier_cell) -- never an alternate/duplicated frontier definition.
    """
    cell = belief.world_to_cell(target_world, clamp=True)
    if cell is None:
        return TargetClassification(
            target_world=target_world,
            target_cell=None,
            belief_value=None,
            explored_by_any_robot=False,
            unknown_neighbor_count=0,
            is_current_frontier=False,
            source=source,
            reason=reason,
        )

    row, col = cell
    belief_value = int(belief.grid[row, col])
    explored_by_any = bool(belief.explored_by_robot[:, row, col].any())
    unknown_neighbor_count = sum(
        1
        for (nr, nc) in _neighbors4(cell)
        if 0 <= nr < belief.height and 0 <= nc < belief.width and int(belief.grid[nr, nc]) == UNKNOWN
    )

    return TargetClassification(
        target_world=target_world,
        target_cell=cell,
        belief_value=belief_value,
        explored_by_any_robot=explored_by_any,
        unknown_neighbor_count=unknown_neighbor_count,
        is_current_frontier=is_frontier_cell(belief, cell),
        source=source,
        reason=reason,
    )


def _kind_from_reason(reason: str) -> str:
    """Pull the FrontierCandidate.reason's `kind=...` tag out of a reason
    string the real planner already produced -- pure string parsing, not a
    second definition of frontier/candidate semantics."""
    marker = "kind="
    idx = reason.find(marker)
    if idx == -1:
        return "unknown"
    tail = reason[idx + len(marker):]
    return tail.split(",", 1)[0].strip()


# ---------------------------------------------------------------------------
# Shared belief/engine builders
# ---------------------------------------------------------------------------


def _empty_belief(robot_count: int = 1) -> BeliefMap:
    return BeliefMap(bounds=_BOUNDS, resolution=_RESOLUTION, robot_count=robot_count)


def _fill_free(belief: BeliefMap) -> None:
    for row in range(belief.height):
        for col in range(belief.width):
            belief.mark_free_cell((row, col))


def _build_fake_single_robot_engine(
    belief: BeliefMap,
    robot_xy: tuple[float, float],
    *,
    robot_heading: float = 0.0,
    current_target: tuple[float, float] | None = None,
    exploration_planner: str = "FoV-aware directional frontier",
):
    """Minimal duck-typed SimulationControllerMixin host for
    select_navigation_goal() -- same pattern as test_navigation_snapshot_
    restore.py's _build_fake_engine(): a SimpleNamespace holding just the
    state that method reads/writes, with its collaborator methods stubbed
    directly (no Qt, no canvas widget, no coordinator)."""
    agent = RobotAgent(robot_id=0, position=robot_xy, heading=robot_heading, planner_mode=exploration_planner)
    agent.exploration_target_xy = current_target

    config = SimulationConfig(
        exploration_planner=exploration_planner,
        vision=3.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
        goal_tolerance=0.25,
        grid_resolution=belief.resolution,
    )

    fake = SimpleNamespace(
        config=config,
        belief_map=belief,
        robot=SimpleNamespace(x=robot_xy[0], y=robot_xy[1], theta=robot_heading),
        robots=[],
        canvas=SimpleNamespace(set_exploration_target=lambda *_a, **_k: None),
        telemetry=SimpleNamespace(report_frontier_selection=lambda **_k: None),
        simulation_time=0.0,
        current_exploration_target=current_target,
        last_goal_selection_reason="",
        runtime_agent=lambda robot_index=None: agent,
        ensure_belief_map=lambda: belief,
        final_goal_xy=lambda: (float(config.goal_x), float(config.goal_y)),
        safety_radius=lambda: 0.2,
        _planning_grid_provider_for_robot=lambda robot: None,
    )
    return fake, agent


def _make_observation(belief: BeliefMap, robot_xy: tuple[float, float], **overrides) -> RobotObservation:
    defaults = dict(
        robot_xy=robot_xy,
        robot_heading=0.0,
        robot_radius=0.2,
        belief_map=belief,
        planning_grid=None,
        mapped_obstacle_points=[],
        dynamic_obstacles=[],
        active_segment_blocked=False,
        predicted_collision=False,
        current_time=10.0,
        grid_resolution=belief.resolution,
        goal_tolerance=0.25,
        sensor_range=3.0,
        final_goal_xy=None,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
    )
    defaults.update(overrides)
    return RobotObservation(**defaults)


# ---------------------------------------------------------------------------
# 1. Frontier semantics: a FREE/explored cell adjacent to UNKNOWN is a valid
#    target. "Target sits inside the explored/blue area" is not, on its own,
#    a bug.
# ---------------------------------------------------------------------------


def test_frontier_semantics_free_explored_cell_can_be_a_valid_target():
    belief = _empty_belief(robot_count=1)
    cell = belief.world_to_cell((0.0, 0.0), clamp=True)
    belief.mark_free_cell(cell, robot_index=0, time_s=0.0)
    target_world = belief.cell_to_world(cell)

    classification = _classify_target(belief, target_world, source="frontier", reason="test setup: isolated FREE cell")

    assert classification.belief_value == FREE
    assert classification.explored_by_any_robot is True, "the target lies inside the explored area -- expected, not a bug"
    assert classification.unknown_neighbor_count > 0
    assert classification.is_current_frontier is True


# ---------------------------------------------------------------------------
# 2. A cell deep inside a fully-known region must never appear as a frontier
#    candidate.
# ---------------------------------------------------------------------------


def test_deep_explored_cell_never_appears_as_a_frontier_candidate():
    belief = _empty_belief(robot_count=1)
    for x in range(-5, 5):
        for y in range(-5, 5):
            cell = belief.world_to_cell((float(x), float(y)))
            if cell is not None:
                belief.mark_free_cell(cell)
    belief.force_free_point((0.0, 0.0))

    deep_cell = belief.world_to_cell((0.0, 0.0), clamp=True)
    deep_world = belief.cell_to_world(deep_cell)

    classification = _classify_target(belief, deep_world, source="probe", reason="")
    assert classification.unknown_neighbor_count == 0
    assert classification.is_current_frontier is False
    assert deep_cell not in _frontier_cells(belief)

    result = select_exploration_goal(
        "Nearest frontier",
        belief_map=belief, robot_xy=(0.0, 0.0), robot_heading=0.0,
        current_target=None, final_goal_xy=(0.0, 0.0), robot_count=1,
        robot_radius=0.2, sensor_range=3.0, vision_model="LiDAR", ipp_distance_penalty=0.2,
    )
    assert result.success, "the region's own edge should still produce real frontier candidates"
    candidate_cells = {belief.world_to_cell(c.target, clamp=True) for c in result.candidates}
    assert deep_cell not in candidate_cells, (
        "a fully-explored interior cell must never be proposed as a frontier candidate"
    )
    assert result.target != deep_world


# ---------------------------------------------------------------------------
# 3. Stale current target: once every UNKNOWN neighbor around a previously
#    selected target has been observed, selection must not reuse it as if
#    it were still valid.
# ---------------------------------------------------------------------------


def test_stale_current_target_is_not_reused_by_selection():
    belief = _empty_belief(robot_count=1)
    _fill_free(belief)
    # A small UNKNOWN patch, far from the robot, is the only remaining
    # frontier source. Block "forward" immediately so only genuine
    # frontier/current candidates compete (isolates the mechanism under
    # test from the unrelated forward-fallback).
    patch_cells = [(1, belief.width - 2), (1, belief.width - 1), (2, belief.width - 2), (2, belief.width - 1)]
    for r, c in patch_cells:
        belief.grid[r, c] = UNKNOWN
        belief.explored_by_robot[0, r, c] = False
    robot_xy = (0.0, 0.0)
    belief.force_free_point(robot_xy)
    belief.mark_occupied_cell(belief.world_to_cell((1.0, 0.0)))

    first = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief, robot_xy=robot_xy, robot_heading=0.0,
        current_target=None, final_goal_xy=None, robot_count=1,
        robot_radius=0.2, sensor_range=3.0, vision_model="LiDAR", ipp_distance_penalty=0.2,
    )
    assert first.success
    assert _kind_from_reason(first.reason) == "frontier"
    T1 = first.target
    classification_before = _classify_target(belief, T1, source="frontier", reason=first.reason)
    assert classification_before.is_current_frontier is True

    # The robot (or a teammate) has since observed the entire remaining
    # patch -- T1 is no longer adjacent to any UNKNOWN cell, and no
    # frontier exists anywhere in the map.
    for r, c in patch_cells:
        belief.mark_free_cell((r, c))
    assert not _frontier_cells(belief)

    second = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief, robot_xy=robot_xy, robot_heading=0.0,
        current_target=T1, final_goal_xy=None, robot_count=1,
        robot_radius=0.2, sensor_range=3.0, vision_model="LiDAR", ipp_distance_penalty=0.2,
    )

    # The actual bug this guards against: `second` used to succeed with
    # target == T1 (a fully-explored, no-longer-frontier cell), reason
    # "selected best FoV-aware target" -- indistinguishable from a genuine
    # fresh pick.
    assert not (second.success and second.target == T1), (
        "a stale current_target (no longer a frontier, and nothing else exists) "
        "must be invalidated, not reused as-is"
    )
    if second.success:
        classification_after = _classify_target(belief, second.target, source="reselected", reason=second.reason)
        assert classification_after.is_current_frontier is True, (
            "if selection still succeeds, the returned target must itself be a genuine frontier"
        )


# ---------------------------------------------------------------------------
# 4. Independent selector parity: ExplorationBehavior._pick_next_target()
#    and engine.select_navigation_goal() must never disagree about whether
#    a source=frontier target is actually a current frontier.
# ---------------------------------------------------------------------------


def _belief_with_two_frontier_regions() -> BeliefMap:
    belief = _empty_belief(robot_count=1)
    for x in range(1, 6):
        for y in range(-2, 3):
            cell = belief.world_to_cell((float(x), float(y)))
            if cell is not None:
                belief.mark_free_cell(cell)
    small_cell = belief.world_to_cell((-8.0, 8.0))
    if small_cell is not None:
        belief.mark_free_cell(small_cell)
    belief.force_free_point((0.0, 0.0))
    return belief


def test_independent_selectors_never_disagree_on_frontier_validity():
    belief = _belief_with_two_frontier_regions()
    robot_xy = (0.0, 0.0)

    agent = RobotAgent(robot_id=0, position=robot_xy, planner_mode="FoV-aware directional frontier")
    behavior = ExplorationBehavior()
    services = PlannerServices()
    observation = _make_observation(belief, robot_xy, sensor_range=6.0)

    target_a = behavior._pick_next_target(agent, observation, services)
    assert target_a is not None
    class_a = _classify_target(
        belief, target_a, source="ExplorationBehavior._pick_next_target",
        reason=agent.last_frontier_selection_reason,
    )

    fake, _agent_b = _build_fake_single_robot_engine(belief, robot_xy)
    target_b, reason_b = SimulationControllerMixin.select_navigation_goal(fake, robot_xy)
    assert target_b is not None
    class_b = _classify_target(belief, target_b, source="engine.select_navigation_goal", reason=reason_b)

    # Not required to pick the identical target (ranking criteria may
    # differ) -- but any target either selector labels source=frontier
    # must be a real, current frontier cell.
    for classification in (class_a, class_b):
        if _kind_from_reason(classification.reason) == "frontier":
            assert classification.is_current_frontier is True, classification


# ---------------------------------------------------------------------------
# 5. Bootstrap/fallback classification: a non-frontier target must be
#    labeled as such, never asserted to be a frontier.
# ---------------------------------------------------------------------------


def test_bootstrap_fallback_target_is_not_misclassified_as_frontier():
    belief = _empty_belief(robot_count=1)
    _fill_free(belief)  # no UNKNOWN anywhere -- zero frontier candidates
    robot_xy = (0.0, 0.0)
    belief.force_free_point(robot_xy)

    result = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief, robot_xy=robot_xy, robot_heading=0.0,
        current_target=None, final_goal_xy=None, robot_count=1,
        robot_radius=0.2, sensor_range=3.0, vision_model="LiDAR", ipp_distance_penalty=0.2,
    )
    assert not _frontier_cells(belief)
    assert result.success, "the forward/bootstrap fallback should still produce a target on a fully-explored map"

    kind = _kind_from_reason(result.reason)
    assert kind != "frontier", f"expected a non-frontier bootstrap/fallback kind, got kind={kind}"

    classification = _classify_target(belief, result.target, source=kind, reason=result.reason)
    assert classification.is_current_frontier is False, (
        "a bootstrap/fallback target is not required to be a frontier cell -- "
        "but it must never be reported/treated as one (policy unchanged here)"
    )


# ---------------------------------------------------------------------------
# 5b. Regression: a bootstrap/fallback target (never required to satisfy
# is_frontier_cell(), see test 5 above) must not make ExplorationBehavior.
# update() loop REQUEST_PLAN forever. Found via manual smoke testing after
# this file's original fix landed: on a mostly-explored map with only a
# forward/bootstrap candidate left, step 5's staleness check
# (_active_target_is_frontier()) failed for it on EVERY tick (by design --
# it was never a frontier to begin with), forcing continuous re-selection;
# since the robot's heading/position had not changed, re-selection kept
# proposing the exact same point, so the agent never reached ordinary
# FOLLOW_PATH long enough to actually move -- an infinite REQUEST_PLAN loop
# with the robot frozen in place.
#
# Two layered fixes now guard this:
#   1. Step 5 restores and follows an unchanged re-selected target instead
#      of looping (this file's earlier commit).
#   2. _pick_next_target()/_pick_map_wide_fallback_target() now reject any
#      candidate that _excluded_targets() already flagged -- including the
#      robot's own active_path_goal_xy -- even when the candidate SOURCE
#      (e.g. _forward_candidate(), a pure function of robot pose/heading)
#      cannot itself honor exclusions (fix/exploration-candidate-exclusion-
#      enforcement). In this test's scenario the bootstrap target IS the
#      active path goal, so re-selection now correctly returns None (not
#      the same point), which yields one HOLD tick before step 5's re-check
#      guard (agent.exploration_target_xy is not None) goes false and
#      subsequent ticks fall through to plain FOLLOW_PATH -- still no
#      REQUEST_PLAN loop, just via a different, more defensive path.
# ---------------------------------------------------------------------------


def test_unchanged_bootstrap_fallback_target_does_not_loop_request_plan():
    belief = _empty_belief(robot_count=1)
    _fill_free(belief)  # no UNKNOWN anywhere -- zero frontier candidates
    robot_xy = (0.0, 0.0)
    belief.force_free_point(robot_xy)

    agent = RobotAgent(robot_id=0, position=robot_xy, planner_mode="FoV-aware directional frontier")
    behavior = ExplorationBehavior()
    services = PlannerServices()
    observation = _make_observation(belief, robot_xy)

    first_target = behavior._pick_next_target(agent, observation, services)
    assert first_target is not None
    assert _kind_from_reason(agent.last_frontier_selection_reason) == "forward", (
        "test setup must produce a non-frontier bootstrap/fallback target"
    )

    agent.set_exploration_target(first_target, reason="initial")
    agent.assign_path(target=first_target, waypoints=[first_target], planner_reason="initial")

    # Several consecutive ticks with the robot never actually moving
    # (matching the real bug: the loop repeated so fast within one
    # wall-clock second that the robot never got a physics tick's worth of
    # forward progress) -- the decision must never REQUEST_PLAN the exact
    # same stale point twice in a row, and must settle into a stable
    # FOLLOW_PATH rather than oscillating forever.
    kinds = []
    for _ in range(5):
        decision = behavior.update(agent, observation, services)
        kinds.append(decision.kind)
        assert decision.kind != "REQUEST_PLAN" or decision.target != first_target, (
            "must never re-request a plan for the exact same stale target "
            f"tick after tick; got {decision.kind}: {decision.reason}"
        )

    assert kinds[-1] == "FOLLOW_PATH", f"must settle into FOLLOW_PATH, not oscillate forever; got {kinds}"
    assert kinds.count("HOLD") <= 1, f"must not repeatedly HOLD on the same unresolved target; got {kinds}"


# ---------------------------------------------------------------------------
# 5c. Regression: a target already marked failed (e.g. by engine.py's
# repeated_safety_replan handling, which calls RobotAgent.invalidate_failed_
# exploration_route() -> mark_exploration_target_failed()) must not come
# straight back from _pick_next_target()/_pick_map_wide_fallback_target()
# while the robot's pose is unchanged.
#
# Root cause: _forward_candidate() (exploration_planners.py) is a pure
# function of belief/robot_xy/robot_heading -- it takes no excluded_targets
# parameter at all, unlike _frontier_candidates(), so it cannot itself skip
# a point _excluded_targets() already flagged. On a map with no genuine
# frontier nearby, the forward ray-cast is the ONLY candidate available, so
# a target invalidated this tick came right back unchanged next tick,
# defeating the exclusion and reproducing the exact live freeze this test
# guards against: REPLAN_FOR_SAFETY -> repeated_safety_replan -> invalidate
# -> "recovered after planner failure; requesting fresh frontier" -> the
# identical point -> REPLAN_FOR_SAFETY again, forever, robot frozen in
# place. Fix: _pick_next_target()/_pick_map_wide_fallback_target() now
# reject any candidate matching an excluded target as a defensive floor,
# regardless of whether the candidate's own source could honor exclusions.
# ---------------------------------------------------------------------------


def test_forward_candidate_is_not_reselected_after_being_marked_failed():
    belief = _empty_belief(robot_count=1)
    _fill_free(belief)  # no UNKNOWN anywhere -- zero genuine frontier candidates
    robot_xy = (0.0, 0.0)
    belief.force_free_point(robot_xy)

    agent = RobotAgent(robot_id=0, position=robot_xy, planner_mode="FoV-aware directional frontier")
    behavior = ExplorationBehavior()
    services = PlannerServices()
    observation = _make_observation(belief, robot_xy)

    first_target = behavior._pick_next_target(agent, observation, services)
    assert first_target is not None
    assert _kind_from_reason(agent.last_frontier_selection_reason) == "forward", (
        "test setup must produce a non-frontier bootstrap/fallback target"
    )

    # Simulate what engine.py's repeated-safety-replan handling does when a
    # route to this exact target keeps failing: invalidate_failed_
    # exploration_route() clears the target and records it as failed.
    agent.exploration_target_xy = first_target
    agent.invalidate_failed_exploration_route(
        reason="repeated safety replan: predicted collision",
        current_time=observation.current_time,
        map_signature=len(observation.mapped_obstacle_points),
    )
    assert first_target in agent.recently_failed_exploration_targets(
        current_time=observation.current_time, cooldown=999.0
    )

    # Robot pose is unchanged (matches the real bug: the robot never got a
    # physics tick's worth of forward progress), so a naive re-selection
    # would recompute to the exact same point.
    reselected = behavior._pick_next_target(agent, observation, services)
    assert reselected != first_target, (
        "a target just marked failed must not be re-proposed by "
        "_pick_next_target() while the robot's pose is unchanged"
    )

    fallback_reselected = behavior._pick_map_wide_fallback_target(agent, observation, services)
    assert fallback_reselected != first_target, (
        "a target just marked failed must not be re-proposed by "
        "_pick_map_wide_fallback_target() either"
    )


# ---------------------------------------------------------------------------
# 6. Multi-robot: each source=frontier target must belong to the CURRENT
#    frontier-cell set of the shared belief; R1 and R2 may differ; neither
#    may be a known cell with no UNKNOWN neighbor.
# ---------------------------------------------------------------------------


def _shared_two_frontier_belief_for_multi_robot() -> BeliefMap:
    belief = _empty_belief(robot_count=2)
    for x in range(-2, 3):
        for y in range(-2, 3):
            cell = belief.world_to_cell((float(x), float(y)))
            if cell is not None:
                belief.mark_free_cell(cell)
    for x in range(3, 7):
        cell = belief.world_to_cell((float(x), 0.0))
        if cell is not None:
            belief.mark_free_cell(cell)
    for y in range(3, 7):
        cell = belief.world_to_cell((0.0, float(y)))
        if cell is not None:
            belief.mark_free_cell(cell)
    belief.force_free_point((0.0, 0.0))
    return belief


def test_multi_robot_targets_are_each_valid_and_may_differ():
    belief = _shared_two_frontier_belief_for_multi_robot()
    robot_xy = (0.0, 0.0)

    result_r1 = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief, robot_xy=robot_xy, robot_heading=0.0,
        current_target=None, final_goal_xy=None, robot_count=2,
        robot_radius=0.2, sensor_range=3.0, vision_model="LiDAR", ipp_distance_penalty=0.2,
        excluded_targets=[],
    )
    assert result_r1.success

    result_r2 = select_exploration_goal(
        "FoV-aware directional frontier",
        belief_map=belief, robot_xy=robot_xy, robot_heading=math.pi / 2.0,
        current_target=None, final_goal_xy=None, robot_count=2,
        robot_radius=0.2, sensor_range=3.0, vision_model="LiDAR", ipp_distance_penalty=0.2,
        excluded_targets=[result_r1.target],
    )
    assert result_r2.success

    class_r1 = _classify_target(belief, result_r1.target, source="R1", reason=result_r1.reason)
    class_r2 = _classify_target(belief, result_r2.target, source="R2", reason=result_r2.reason)

    frontier_cells_now = _frontier_cells(belief)
    for classification in (class_r1, class_r2):
        if _kind_from_reason(classification.reason) == "frontier":
            assert classification.target_cell in frontier_cells_now
            assert classification.is_current_frontier is True
            assert classification.unknown_neighbor_count > 0

    assert result_r1.target != result_r2.target


# ---------------------------------------------------------------------------
# 7. Frontier-loss hysteresis: one observation must not cancel a healthy
#    assigned route.  A remote target that remains stale is still replaced
#    after the bounded grace period.
# ---------------------------------------------------------------------------


def test_remote_target_becomes_stale_only_reselects_after_grace():
    belief = _empty_belief(robot_count=1)
    _fill_free(belief)
    patch_cells = [(1, belief.width - 2), (1, belief.width - 1), (2, belief.width - 2), (2, belief.width - 1)]
    for r, c in patch_cells:
        belief.grid[r, c] = UNKNOWN
        belief.explored_by_robot[0, r, c] = False
    robot_xy = (0.0, 0.0)
    belief.force_free_point(robot_xy)

    frontier_cell = (0, belief.width - 2)
    assert frontier_cell in _frontier_cells(belief)
    target_world = belief.cell_to_world(frontier_cell)

    agent = RobotAgent(robot_id=0, position=robot_xy, planner_mode="FoV-aware directional frontier")
    agent.set_exploration_target(target_world, reason="initial frontier")
    # Route assigned, robot kept far from its final destination (only an
    # intermediate waypoint is nearby) -- exactly the "conserva el robot
    # lejos del objetivo" setup.
    agent.assign_path(target=target_world, waypoints=[(5.0, 0.0), target_world], planner_reason="initial")
    assert agent.active_target() == (5.0, 0.0)

    # The belief changes so the frontier disappears entirely before the
    # robot arrives.
    for r, c in patch_cells:
        belief.mark_free_cell((r, c))
    assert not _frontier_cells(belief)

    behavior = ExplorationBehavior()
    services = PlannerServices()
    observation = _make_observation(belief, robot_xy)

    first = behavior.update(agent, observation, services)
    assert first.kind == "FOLLOW_PATH"
    assert agent.exploration_target_xy == target_world

    after_grace = _make_observation(
        belief,
        robot_xy,
        current_time=observation.current_time + behavior._STALE_TARGET_GRACE + 0.01,
    )
    decision = behavior.update(agent, after_grace, services)

    assert decision.kind != "FOLLOW_PATH", (
        "a remote target that stays stale beyond the grace period must still "
        "enter the normal reselection/exhaustion path"
    )
    assert agent.exploration_target_xy != target_world


def test_locally_observed_frontier_loss_keeps_committed_route():
    belief = _empty_belief(robot_count=1)
    _fill_free(belief)
    robot_xy = (0.0, 0.0)
    target_world = (2.0, 0.0)  # inside the configured 3 m sensor range

    agent = RobotAgent(robot_id=0, position=robot_xy, planner_mode="FoV-aware directional frontier")
    agent.set_exploration_target(target_world, reason="frontier observed while approaching")
    agent.assign_path(target=target_world, waypoints=[(1.0, 0.0), target_world], planner_reason="initial")

    behavior = ExplorationBehavior()
    services = PlannerServices()
    first = behavior.update(agent, _make_observation(belief, robot_xy, current_time=10.0), services)
    much_later = behavior.update(agent, _make_observation(belief, robot_xy, current_time=30.0), services)

    assert first.kind == "FOLLOW_PATH"
    assert much_later.kind == "FOLLOW_PATH"
    assert agent.exploration_target_xy == target_world
    assert agent.target_switch_count == 0


def test_stale_target_hysteresis_never_overrides_safety_replan():
    belief = _empty_belief(robot_count=1)
    _fill_free(belief)
    robot_xy = (0.0, 0.0)
    target_world = (2.0, 0.0)

    agent = RobotAgent(robot_id=0, position=robot_xy, planner_mode="FoV-aware directional frontier")
    agent.set_exploration_target(target_world, reason="initial")
    agent.assign_path(target=target_world, waypoints=[target_world], planner_reason="initial")

    decision = ExplorationBehavior().update(
        agent,
        _make_observation(belief, robot_xy, predicted_collision=True),
        PlannerServices(),
    )

    assert decision.kind == "REPLAN_FOR_SAFETY"


# ---------------------------------------------------------------------------
# 8. No visual coupling: validity here depends only on the BeliefMap.
# ---------------------------------------------------------------------------


def test_no_visual_coupling_in_this_test_module():
    with open(__file__, encoding="utf-8") as handle:
        source = handle.read()

    forbidden_substrings = (
        "explored_area_polygons",
        "simulation_canvas",
        "ThemeMode",
        "theme_colors",
        "_explored_area_cache",
        "canvas cache",
    )
    for token in forbidden_substrings:
        # Each token appears only inside this very assertion's own tuple
        # literal above (as a string to check FOR) -- guard against that by
        # counting occurrences instead of a naive substring search.
        occurrences = source.count(token)
        expected_occurrences = 1  # the tuple entry itself, defined above
        assert occurrences <= expected_occurrences, (
            token, occurrences, "must not be used anywhere else in this test module"
        )
