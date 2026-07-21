"""Contract tests for the pure-Python Hungarian solver
(algorithms/frontier_cluster_hungarian/assignment.py) in isolation -- no
robots, no clusters, no plugin, just plain utility/feasibility matrices.
"""
from __future__ import annotations

import re
from pathlib import Path

from algorithms.frontier_cluster_hungarian.assignment import solve_max_utility_assignment

MODULE_PATH = Path(__file__).resolve().parents[2] / "algorithms" / "frontier_cluster_hungarian" / "assignment.py"


def _non_docstring_code_lines(text: str) -> str:
    """Strip module/function docstrings so a source scan checks only real
    code lines, not prose that legitimately explains what this module does
    NOT depend on (see this file's SciPy-import check below)."""
    return re.sub(r'""".*?"""', "", text, flags=re.DOTALL)


def _all_feasible(n_rows: int, n_cols: int) -> list[list[bool]]:
    return [[True] * n_cols for _ in range(n_rows)]


# ---------------------------------------------------------------------------
# 1. Square matrix, known optimal solution.
# ---------------------------------------------------------------------------


def test_square_matrix_known_optimal_solution():
    # Obvious optimum: robot i should take task i (diagonal dominance).
    utility = [
        [0.9, 0.1, 0.1],
        [0.1, 0.9, 0.1],
        [0.1, 0.1, 0.9],
    ]
    feasible = _all_feasible(3, 3)

    result = solve_max_utility_assignment(utility, feasible)

    assert result == (0, 1, 2)


# ---------------------------------------------------------------------------
# 2. More robots than tasks -> HOLD for the extras.
# ---------------------------------------------------------------------------


def test_more_robots_than_tasks_produces_hold():
    utility = [
        [0.9, 0.1],
        [0.1, 0.9],
        [0.5, 0.5],
    ]
    feasible = _all_feasible(3, 2)

    result = solve_max_utility_assignment(utility, feasible)

    assert len(result) == 3
    assigned = [value for value in result if value is not None]
    assert sorted(assigned) == [0, 1]
    assert result.count(None) == 1


# ---------------------------------------------------------------------------
# 3. More tasks than robots -> each robot gets a distinct task.
# ---------------------------------------------------------------------------


def test_more_tasks_than_robots_assigns_distinct_tasks():
    utility = [
        [0.9, 0.2, 0.1, 0.05],
        [0.1, 0.2, 0.9, 0.05],
    ]
    feasible = _all_feasible(2, 4)

    result = solve_max_utility_assignment(utility, feasible)

    assert len(result) == 2
    assert None not in result
    assert len(set(result)) == 2  # no two robots share a task


# ---------------------------------------------------------------------------
# 4 & 5. A row with no feasible tasks holds; HOLD beats "unavailable".
# ---------------------------------------------------------------------------


def test_row_with_no_feasible_tasks_holds():
    utility = [
        [0.9, 0.8],
        [0.9, 0.8],
    ]
    feasible = [
        [True, True],
        [False, False],
    ]

    result = solve_max_utility_assignment(utility, feasible)

    assert result[0] is not None
    assert result[1] is None


# ---------------------------------------------------------------------------
# 6. Any feasible task beats HOLD.
# ---------------------------------------------------------------------------


def test_any_feasible_task_beats_hold():
    # Even a low-utility feasible task must be chosen over HOLD.
    utility = [[0.01]]
    feasible = [[True]]

    result = solve_max_utility_assignment(utility, feasible)

    assert result == (0,)


# ---------------------------------------------------------------------------
# 7 & 8. Ties resolve deterministically, and repeated runs agree.
# ---------------------------------------------------------------------------


def test_ties_resolve_deterministically_and_repeatably():
    utility = [[0.5, 0.5]]
    feasible = [[True, True]]

    results = {solve_max_utility_assignment(utility, feasible) for _ in range(10)}

    assert len(results) == 1
    (only_result,) = results
    assert only_result[0] in (0, 1)


def test_running_ten_times_produces_the_same_result():
    utility = [
        [0.7, 0.3, 0.9],
        [0.2, 0.8, 0.4],
        [0.6, 0.6, 0.1],
    ]
    feasible = _all_feasible(3, 3)

    results = [solve_max_utility_assignment(utility, feasible) for _ in range(10)]

    assert all(result == results[0] for result in results)


# ---------------------------------------------------------------------------
# 9 & 10. Empty matrix; zero tasks with robots.
# ---------------------------------------------------------------------------


def test_empty_matrix_produces_empty_tuple():
    result = solve_max_utility_assignment([], [])
    assert result == ()


def test_zero_tasks_with_robots_produces_all_none():
    utility = [[], []]
    feasible = [[], []]

    result = solve_max_utility_assignment(utility, feasible)

    assert result == (None, None)


# ---------------------------------------------------------------------------
# 11. No SciPy.
# ---------------------------------------------------------------------------


def test_module_does_not_import_scipy():
    code_only = _non_docstring_code_lines(MODULE_PATH.read_text(encoding="utf-8"))
    assert "scipy" not in code_only.lower()
    assert "numpy" not in code_only.lower()


# ---------------------------------------------------------------------------
# 12. Not greedy: a classic counter-example where the locally-best cell is
#    not part of the globally-optimal assignment.
# ---------------------------------------------------------------------------


def test_solver_finds_global_optimum_not_greedy_local_choice():
    # Greedy picks the single highest cell first (robot0/task0 = 0.9),
    # leaving robot1 stuck with task1 (0.1): total 1.0.
    # The optimal assignment is robot0/task1 (0.8) + robot1/task0 (0.85):
    # total 1.65, strictly better, and it does NOT include the greedy
    # first pick.
    utility = [
        [0.9, 0.8],
        [0.85, 0.1],
    ]
    feasible = _all_feasible(2, 2)

    result = solve_max_utility_assignment(utility, feasible)

    assert result == (1, 0)


# ---------------------------------------------------------------------------
# 13. Never returns an index outside the real task range.
# ---------------------------------------------------------------------------


def test_result_never_indexes_beyond_real_tasks():
    utility = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ]
    feasible = _all_feasible(4, 2)

    result = solve_max_utility_assignment(utility, feasible)

    assert len(result) == 4
    for value in result:
        assert value is None or 0 <= value < 2
