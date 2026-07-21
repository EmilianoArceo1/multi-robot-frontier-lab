"""Pure-Python Hungarian (Kuhn-Munkres) solver for the robot-task
assignment, O(n^3), no SciPy/NumPy dependency.

The public entry point, solve_max_utility_assignment(), takes a utility
matrix (higher is better) plus a feasibility mask and returns, for each
robot row, the column index of the assigned task or None for HOLD. This
module knows nothing about robots, clusters, FrontierCluster, or the
simulator -- it only ever sees plain floats and bools.
"""
from __future__ import annotations

from typing import Sequence

# Costs (this is a MINIMIZATION solver; utility is converted to cost as
# 1.0 - utility before calling _square_hungarian_min_cost):
#   feasible real task : 1.0 - utility, i.e. in [0.0, 1.0] since utility is
#                         always in [0.0, 1.0] for a feasible cell.
#   HOLD dummy column   : 2.0 -- strictly worse than ANY feasible real task,
#                         strictly better than any infeasible one.
#   infeasible real task: effectively unavailable.
HOLD_COST = 2.0
UNAVAILABLE_COST = 1_000_000_000.0


def solve_max_utility_assignment(
    utility_matrix: Sequence[Sequence[float]],
    feasible_matrix: Sequence[Sequence[bool]],
) -> tuple[int | None, ...]:
    """Maximum-utility robot-task assignment via the Hungarian algorithm.

    utility_matrix / feasible_matrix are n_robots x n_tasks (rows = robots
    in a caller-chosen stable order, e.g. ascending robot_id; columns =
    tasks in a caller-chosen stable order, e.g. ascending task_id -- see
    utility.build_utility_matrix()). Enough HOLD dummy columns (one per
    robot) are added internally so every robot can always HOLD even when
    there are more tasks than robots; this keeps n_rows <= n_cols, which
    the rectangular solver requires.

    Returns one entry per robot row: the assigned task's column index into
    utility_matrix (always < n_tasks, i.e. never a dummy/padding index), or
    None for HOLD.
    """
    n_robots = len(utility_matrix)
    if n_robots == 0:
        return ()

    n_tasks = len(utility_matrix[0])
    for row in utility_matrix:
        if len(row) != n_tasks:
            raise ValueError("utility_matrix rows must all have the same length")
    if len(feasible_matrix) != n_robots or any(len(row) != n_tasks for row in feasible_matrix):
        raise ValueError("feasible_matrix must have the same shape as utility_matrix")

    n_dummy = n_robots  # enough HOLD capacity for every robot simultaneously
    n_columns = n_tasks + n_dummy

    cost: list[list[float]] = [[0.0] * n_columns for _ in range(n_robots)]
    for row_index in range(n_robots):
        for task_index in range(n_tasks):
            if feasible_matrix[row_index][task_index]:
                utility = float(utility_matrix[row_index][task_index])
                cost[row_index][task_index] = 1.0 - utility
            else:
                cost[row_index][task_index] = UNAVAILABLE_COST
        for dummy_index in range(n_dummy):
            cost[row_index][n_tasks + dummy_index] = HOLD_COST

    col_for_row = _solve_rectangular_min_cost(cost, n_rows=n_robots, n_cols=n_columns)

    result: list[int | None] = []
    for row_index in range(n_robots):
        column = col_for_row[row_index]
        result.append(column if column < n_tasks else None)
    return tuple(result)


def _solve_rectangular_min_cost(
    cost: list[list[float]], *, n_rows: int, n_cols: int
) -> list[int]:
    """n_rows <= n_cols required. Pads with (n_cols - n_rows) zero-cost
    dummy rows so the matrix is square, runs the square Hungarian
    algorithm, and returns the column index assigned to each of the first
    n_rows real rows (the dummy rows' assignments are discarded)."""
    if n_rows > n_cols:
        raise ValueError("n_rows must be <= n_cols for this rectangular solver")

    padded = [list(row) for row in cost]
    for _ in range(n_cols - n_rows):
        padded.append([0.0] * n_cols)

    full_assignment = _square_hungarian_min_cost(padded)
    return full_assignment[:n_rows]


def _square_hungarian_min_cost(cost: list[list[float]]) -> list[int]:
    """Classic O(n^3) Hungarian algorithm (Kuhn-Munkres with potentials,
    the standard "shortest augmenting path" formulation) for an n x n
    minimization cost matrix. Returns col_for_row: for each row i, the
    assigned column index.

    Every loop iterates indices in ascending order, so ties are always
    resolved toward the lowest column index encountered first -- given
    columns are ordered [real tasks ascending task_id, then dummy HOLD
    columns] and rows are ordered [robots ascending robot_id], this makes
    the result fully deterministic and reproducible run to run. No
    randomness, no hash(), no epsilon noise.
    """
    n = len(cost)
    if n == 0:
        return []

    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)  # p[j] = 1-indexed row currently assigned to column j (0 = none)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    col_for_row = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            col_for_row[p[j] - 1] = j - 1
    return col_for_row
