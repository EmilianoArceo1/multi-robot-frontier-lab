"""
Regression tests for FrontierExplorationPlanner.select_goal() ignoring
is_candidate_reachable.

Symptom: real Office.sim runs periodically show

    Planner failed in exploration mode: <reason>. Holding current position;
    not falling back to G.

followed, after _EXPLORATION_FAILURE_BUDGET (3) consecutive failures, by a
permanent "exploration exhausted" state -- even when reachable, unexplored
area still exists elsewhere on the map.

Root cause: PlannerServices.select_exploration_target() always threads an
is_candidate_reachable(xy) -> bool callback through to
select_exploration_goal() (falling back to self.is_candidate_reachable,
refreshed every tick by engine.ensure_planner_services() from the exact
planning grid the real single-robot A* uses -- see
engine.make_exploration_reachability_check()). Of the six registered
planners, only FoVAwareDirectionalFrontierPlanner.select_goal() actually
read that kwarg (see test_exploration_candidate_reachability.py).
FrontierExplorationPlanner.select_goal() -- the shared base for
NearestFrontierPlanner, LargestFrontierPlanner, UtilityFrontierPlanner and
InformativeFrontierPlanner -- never read it at all, so it could rank and
return a candidate the real navigation grid was already known to reject.

This is not a manual-selection-only edge case:
ExplorationBehavior._pick_map_wide_fallback_target() retries with
_MAP_WIDE_FALLBACK_PLANNER = "Nearest frontier" whenever the configured
(commonly FoV-aware) planner finds nothing this cycle -- see
test_frontier_exhaustion_recovery.py -- so the unfiltered base-class path
is exercised on ordinary default-configuration runs too, not only when a
user explicitly picks one of these four planners in the GUI.

Fix: FrontierExplorationPlanner.select_goal() now applies the same
reachability gate FoVAwareDirectionalFrontierPlanner already used --
dropping candidates is_candidate_reachable(xy) rejects before calling
choose_candidate() -- so each subclass's own ranking policy (nearest,
largest, utility score, IPP-lite score) still applies, just restricted to
the reachable subset. When every candidate is rejected, select_goal()
returns success=False, target=None, and a reason naming
"no reachable frontier candidates" instead of fabricating a target.

These tests exercise exploration_planners.py directly -- no Qt, no canvas,
no full engine/GUI instantiation.
"""
from __future__ import annotations

from robotics_sim.environment.belief_map import BeliefMap
from robotics_sim.planning.exploration_planners import select_exploration_goal


def _belief_with_near_small_and_far_large_frontiers() -> BeliefMap:
    """Two well-separated, single-cluster frontier regions:

    - a small (1-cell) region close to the robot, and
    - a large (12-cell) region far from the robot,

    with an unknown gap between them (and between the small region and the
    robot) so they never merge into one cluster. This makes
    NearestFrontierPlanner and LargestFrontierPlanner disagree on the
    default (unfiltered) choice, so filtering one out has an observable,
    unambiguous effect on the other planner's selection.
    """
    belief = BeliefMap(bounds=(-15.0, 15.0, -15.0, 15.0), resolution=1.0, robot_count=1)

    # Small, close region: a single isolated free cell.
    near_cell = belief.world_to_cell((2.0, 0.0))
    assert near_cell is not None
    belief.mark_free_cell(near_cell)

    # Large, far region: a solid 4x4 block, well clear of the small region.
    for x in range(10, 14):
        for y in range(-2, 2):
            cell = belief.world_to_cell((float(x), float(y)))
            if cell is not None:
                belief.mark_free_cell(cell)

    return belief


def _select(planner_name: str, belief: BeliefMap, *, is_candidate_reachable=None):
    return select_exploration_goal(
        planner_name,
        belief_map=belief,
        robot_xy=(0.0, 0.0),
        robot_heading=0.0,
        current_target=None,
        final_goal_xy=(0.0, 0.0),
        robot_count=1,
        robot_radius=0.2,
        sensor_range=6.0,
        vision_model="LiDAR",
        ipp_distance_penalty=0.2,
        is_candidate_reachable=is_candidate_reachable,
    )


# ---------------------------------------------------------------------------
# 1. Nearest frontier: the nearest candidate is rejected by reachability ->
#    the next-nearest reachable candidate is selected instead of failing or
#    returning the rejected one.
# ---------------------------------------------------------------------------


def test_nearest_frontier_rejects_unreachable_nearest_picks_next_reachable():
    belief = _belief_with_near_small_and_far_large_frontiers()

    baseline = _select("Nearest frontier", belief)
    assert baseline.success
    near_target = baseline.target
    assert near_target == (2.5, 0.5), "sanity: default choice is the close small region"

    def is_candidate_reachable(xy) -> bool:
        return (round(xy[0], 3), round(xy[1], 3)) != (round(near_target[0], 3), round(near_target[1], 3))

    filtered = _select("Nearest frontier", belief, is_candidate_reachable=is_candidate_reachable)

    assert filtered.success
    assert filtered.target != near_target, "the rejected nearest candidate must not be selected"


# ---------------------------------------------------------------------------
# 2. Largest frontier: the largest candidate is rejected by reachability ->
#    the next-largest reachable candidate is selected.
# ---------------------------------------------------------------------------


def test_largest_frontier_rejects_unreachable_largest_picks_next_reachable():
    belief = _belief_with_near_small_and_far_large_frontiers()

    baseline = _select("Largest frontier", belief)
    assert baseline.success
    large_target = baseline.target
    assert large_target == (12.5, -1.5), "sanity: default choice is the far large region"

    def is_candidate_reachable(xy) -> bool:
        return (round(xy[0], 3), round(xy[1], 3)) != (round(large_target[0], 3), round(large_target[1], 3))

    filtered = _select("Largest frontier", belief, is_candidate_reachable=is_candidate_reachable)

    assert filtered.success
    assert filtered.target != large_target, "the rejected largest candidate must not be selected"
    assert filtered.target == (2.5, 0.5), "the only remaining reachable candidate must be selected"


# ---------------------------------------------------------------------------
# 3. When every candidate is rejected, selection must fail explicitly --
#    never fabricate a target, never silently return the best-scored one.
# ---------------------------------------------------------------------------


def test_frontier_planner_reports_no_reachable_candidates_when_all_rejected():
    belief = _belief_with_near_small_and_far_large_frontiers()

    for planner_name in ("Nearest frontier", "Largest frontier", "Utility frontier", "Informative frontier / IPP-lite"):
        baseline = _select(planner_name, belief)
        assert baseline.success, planner_name

        result = _select(planner_name, belief, is_candidate_reachable=lambda xy: False)

        assert not result.success, planner_name
        assert result.target is None, planner_name
        assert "no reachable frontier candidates" in result.reason, planner_name


# ---------------------------------------------------------------------------
# 4. A candidate rejected by the reachability check must never be handed
#    back as `.target` from the same select_goal() call -- there is no
#    later point in this call where a rejected candidate could sneak back
#    in, but this pins that invariant explicitly against regression.
# ---------------------------------------------------------------------------


def test_rejected_candidate_never_returned_as_target_same_call():
    belief = _belief_with_near_small_and_far_large_frontiers()

    baseline = _select("Nearest frontier", belief)
    all_targets = {c.target for c in baseline.candidates}
    assert len(all_targets) >= 2

    rejected = {baseline.target}

    def is_candidate_reachable(xy) -> bool:
        key = (round(xy[0], 3), round(xy[1], 3))
        return key not in {(round(r[0], 3), round(r[1], 3)) for r in rejected}

    result = _select("Nearest frontier", belief, is_candidate_reachable=is_candidate_reachable)
    assert result.success
    assert result.target not in rejected


# ---------------------------------------------------------------------------
# 5 & 6. Utility / informative planners keep their own scoring and ranking
#    policy among the reachable subset -- filtering must not flatten their
#    distinct choose_candidate() criteria back to plain nearest/largest.
# ---------------------------------------------------------------------------


def test_utility_frontier_keeps_its_ranking_criteria_among_reachable_candidates():
    belief = _belief_with_near_small_and_far_large_frontiers()

    baseline = _select("Utility frontier", belief)
    assert baseline.success
    # Utility frontier's own score_candidate() (size - 0.75*distance -
    # 0.15*distance_to_goal) drives the baseline pick, not plain nearest or
    # largest ordering. Whatever it picks by that formula, filtering it out
    # must fall back to the formula's next-best reachable candidate -- not
    # fail, and not silently keep returning the rejected target.
    top_choice = baseline.target

    def is_candidate_reachable(xy) -> bool:
        return (round(xy[0], 3), round(xy[1], 3)) != (round(top_choice[0], 3), round(top_choice[1], 3))

    filtered = _select("Utility frontier", belief, is_candidate_reachable=is_candidate_reachable)
    assert filtered.success
    assert filtered.target != top_choice


def test_informative_frontier_keeps_its_ranking_criteria_among_reachable_candidates():
    belief = _belief_with_near_small_and_far_large_frontiers()

    baseline = _select("Informative frontier / IPP-lite", belief)
    assert baseline.success

    def is_candidate_reachable(xy) -> bool:
        key = (round(xy[0], 3), round(xy[1], 3))
        return key != (round(baseline.target[0], 3), round(baseline.target[1], 3))

    filtered = _select("Informative frontier / IPP-lite", belief, is_candidate_reachable=is_candidate_reachable)
    assert filtered.success
    assert filtered.target != baseline.target


# ---------------------------------------------------------------------------
# 7. A broken reachability callback must not take exploration down -- same
#    documented "assume reachable" policy FoVAwareDirectionalFrontierPlanner
#    already uses.
# ---------------------------------------------------------------------------


def test_broken_reachability_callback_does_not_crash_and_assumes_reachable():
    belief = _belief_with_near_small_and_far_large_frontiers()

    def broken(xy):
        raise RuntimeError("boom")

    result = _select("Nearest frontier", belief, is_candidate_reachable=broken)

    assert result.success
    assert result.target == (2.5, 0.5)


# ---------------------------------------------------------------------------
# 8. Legacy behavior: when is_candidate_reachable is absent (the default),
#    selection is unaffected by the new filtering code path.
# ---------------------------------------------------------------------------


def test_reachability_filtering_is_a_noop_when_callback_absent():
    belief = _belief_with_near_small_and_far_large_frontiers()

    result = _select("Nearest frontier", belief, is_candidate_reachable=None)

    assert result.success
    assert result.target == (2.5, 0.5)
