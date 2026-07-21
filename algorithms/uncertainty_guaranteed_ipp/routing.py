"""Nearest-insertion routing and budgeted GCBCover.

The paper refines selected points with a TSP solver and uses nearest insertion
to cheaply estimate marginal routing cost during GCBCover.  This NumPy-only
MVP uses nearest insertion for both the estimate and the refined route; it is
deterministic and dependency-free, but it is not a replacement for the
paper-faithful SGP-Tools/Attentive/TSP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from algorithms.uncertainty_guaranteed_ipp.coverage import (
    _coverage_matrix,
    _initial_mask,
    greedy_cover,
)
from algorithms.uncertainty_guaranteed_ipp.gp import ArrayLike, _points, _readonly


@dataclass(frozen=True)
class InsertionChoice:
    candidate_index: int
    insertion_position: int
    cost_increment: float
    resulting_cost: float


@dataclass(frozen=True)
class GCBCoverResult:
    """Best budget-feasible solution among GCB and paper-style baselines."""

    selected_indices: tuple[int, ...]
    route_indices: tuple[int, ...]
    selection_order: tuple[int, ...]
    route_cost: float
    covered_mask: np.ndarray
    complete: bool
    uncovered_indices: tuple[int, ...]
    budget: float | None
    strategy: str

    @property
    def covered_count(self) -> int:
        return int(np.count_nonzero(self.covered_mask))


def _point(value: Sequence[float] | np.ndarray | None, *, name: str, dimension: int) -> np.ndarray | None:
    if value is None:
        return None
    point = np.asarray(value, dtype=float).reshape(-1)
    if point.shape != (dimension,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain exactly {dimension} finite coordinates")
    return point


def _indices(value: Iterable[int], *, point_count: int, name: str) -> tuple[int, ...]:
    result = tuple(int(index) for index in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate candidate indices")
    if any(index < 0 or index >= point_count for index in result):
        raise ValueError(f"{name} contains an out-of-range candidate index")
    return result


def _polyline_cost(
    route_indices: Sequence[int],
    points: np.ndarray,
    *,
    start_point: np.ndarray | None,
    return_to_start: bool,
) -> float:
    if not route_indices:
        return 0.0
    route = points[np.asarray(route_indices, dtype=int)]
    total = 0.0
    if start_point is not None:
        total += float(np.linalg.norm(route[0] - start_point))
    if route.shape[0] > 1:
        total += float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    if return_to_start:
        if start_point is not None:
            total += float(np.linalg.norm(route[-1] - start_point))
        elif route.shape[0] > 1:
            total += float(np.linalg.norm(route[-1] - route[0]))
    return total


def route_cost(
    route_indices: Iterable[int],
    candidate_points: ArrayLike,
    *,
    start_point: Sequence[float] | np.ndarray | None = None,
    return_to_start: bool = False,
) -> float:
    """Cost of an ordered sensing route under an explicit depot convention."""

    points = _points(candidate_points, name="candidate_points")
    route = _indices(route_indices, point_count=points.shape[0], name="route_indices")
    start = _point(start_point, name="start_point", dimension=points.shape[1])
    return _polyline_cost(route, points, start_point=start, return_to_start=return_to_start)


def nearest_insertion_increment(
    route_indices: Iterable[int],
    candidate_index: int,
    candidate_points: ArrayLike,
    *,
    start_point: Sequence[float] | np.ndarray | None = None,
    return_to_start: bool = False,
) -> InsertionChoice:
    """Cheapest insertion of one candidate into the current ordered route."""

    points = _points(candidate_points, name="candidate_points")
    route = _indices(route_indices, point_count=points.shape[0], name="route_indices")
    candidate = int(candidate_index)
    if candidate < 0 or candidate >= points.shape[0]:
        raise ValueError("candidate_index is out of range")
    if candidate in route:
        raise ValueError("candidate_index is already present in route_indices")
    start = _point(start_point, name="start_point", dimension=points.shape[1])
    current_cost = _polyline_cost(
        route,
        points,
        start_point=start,
        return_to_start=return_to_start,
    )

    best_position = 0
    best_cost = math.inf
    for position in range(len(route) + 1):
        trial = route[:position] + (candidate,) + route[position:]
        trial_cost = _polyline_cost(
            trial,
            points,
            start_point=start,
            return_to_start=return_to_start,
        )
        if trial_cost < best_cost - 1e-12:
            best_position = position
            best_cost = trial_cost

    increment = max(0.0, best_cost - current_cost)
    return InsertionChoice(
        candidate_index=candidate,
        insertion_position=best_position,
        cost_increment=float(increment),
        resulting_cost=float(best_cost),
    )


def nearest_insertion_route(
    selected_indices: Iterable[int],
    candidate_points: ArrayLike,
    *,
    start_point: Sequence[float] | np.ndarray | None = None,
    return_to_start: bool = False,
) -> tuple[int, ...]:
    """Insert candidates in selection order at their cheapest route position."""

    points = _points(candidate_points, name="candidate_points")
    selected = _indices(
        selected_indices,
        point_count=points.shape[0],
        name="selected_indices",
    )
    route: tuple[int, ...] = ()
    for candidate in selected:
        choice = nearest_insertion_increment(
            route,
            candidate,
            points,
            start_point=start_point,
            return_to_start=return_to_start,
        )
        route = (
            route[: choice.insertion_position]
            + (candidate,)
            + route[choice.insertion_position :]
        )
    return route


@dataclass(frozen=True)
class _RouteOption:
    route: tuple[int, ...]
    selection_order: tuple[int, ...]
    cost: float
    covered: np.ndarray
    strategy: str


def _covered_for_route(matrix: np.ndarray, route: Sequence[int], initial: np.ndarray) -> np.ndarray:
    covered = initial.copy()
    if route:
        covered |= np.any(matrix[np.asarray(route, dtype=int)], axis=0)
    return covered


def _truncate_route(
    route: Sequence[int],
    points: np.ndarray,
    *,
    start_point: Sequence[float] | np.ndarray | None,
    return_to_start: bool,
    budget: float,
) -> tuple[int, ...]:
    accepted: list[int] = []
    for candidate in route:
        trial = tuple(accepted) + (int(candidate),)
        cost = route_cost(
            trial,
            points,
            start_point=start_point,
            return_to_start=return_to_start,
        )
        if cost > budget + 1e-10:
            break
        accepted.append(int(candidate))
    return tuple(accepted)


def _best_option(options: Sequence[_RouteOption]) -> _RouteOption:
    # Coverage is the paper objective.  Equal-coverage ties prefer shorter
    # routes, then the GCB solution, then lexicographically smaller routes.
    strategy_priority = {"gcb": 2, "truncated_greedy": 1, "best_singleton": 0}
    return max(
        options,
        key=lambda option: (
            int(np.count_nonzero(option.covered)),
            -float(option.cost),
            strategy_priority.get(option.strategy, -1),
            tuple(-index for index in option.route),
        ),
    )


def gcb_cover(
    coverage_matrix: np.ndarray | Sequence[Sequence[bool]],
    candidate_points: ArrayLike,
    *,
    budget: float | None,
    start_point: Sequence[float] | np.ndarray | None = None,
    return_to_start: bool = False,
    initially_covered: Iterable[bool] | np.ndarray | None = None,
) -> GCBCoverResult:
    """Budgeted generalized cost-benefit coverage with nearest insertion.

    Remaining candidates are ranked by marginal coverage divided by their
    cheapest insertion increment.  The top candidate is inserted and checked
    against the budget; an infeasible top candidate is removed from future
    consideration, matching the paper's GCB loop.  The returned solution is
    compared with a budget-truncated GreedyCover route and a feasible best
    singleton guard for deterministic knapsack-style behavior.
    """

    matrix = _coverage_matrix(coverage_matrix)
    points = _points(candidate_points, name="candidate_points")
    if points.shape[0] != matrix.shape[0]:
        raise ValueError("candidate_points count must match coverage_matrix rows")
    initial = _initial_mask(initially_covered, matrix.shape[1])

    if budget is None:
        numeric_budget = math.inf
        budget_value: float | None = None
    else:
        numeric_budget = float(budget)
        if not math.isfinite(numeric_budget) or numeric_budget < 0.0:
            raise ValueError("budget must be None or a finite nonnegative distance")
        budget_value = numeric_budget

    # Ratio-greedy GCB branch.
    route: tuple[int, ...] = ()
    selection_order: list[int] = []
    covered = initial.copy()
    remaining = set(range(matrix.shape[0]))

    while remaining and not bool(np.all(covered)):
        ranked: list[tuple[float, int, float, int, InsertionChoice]] = []
        uncovered = ~covered
        for candidate in sorted(remaining):
            gain = int(np.count_nonzero(matrix[candidate] & uncovered))
            if gain <= 0:
                continue
            choice = nearest_insertion_increment(
                route,
                candidate,
                points,
                start_point=start_point,
                return_to_start=return_to_start,
            )
            ratio = math.inf if choice.cost_increment <= 1e-12 else gain / choice.cost_increment
            ranked.append((ratio, gain, -choice.cost_increment, -candidate, choice))

        if not ranked:
            break

        _, _, _, _, choice = max(ranked, key=lambda item: item[:4])
        candidate = choice.candidate_index
        remaining.remove(candidate)
        if choice.resulting_cost > numeric_budget + 1e-10:
            continue

        route = (
            route[: choice.insertion_position]
            + (candidate,)
            + route[choice.insertion_position :]
        )
        selection_order.append(candidate)
        covered |= matrix[candidate]

    gcb_cost = route_cost(
        route,
        points,
        start_point=start_point,
        return_to_start=return_to_start,
    )
    options: list[_RouteOption] = [
        _RouteOption(route, tuple(selection_order), gcb_cost, covered.copy(), "gcb")
    ]

    # Paper comparison branch: GreedyCover, route it, then retain the
    # budget-feasible prefix in execution order.
    greedy = greedy_cover(matrix, initially_covered=initial)
    greedy_route = nearest_insertion_route(
        greedy.selected_indices,
        points,
        start_point=start_point,
        return_to_start=return_to_start,
    )
    truncated_route = _truncate_route(
        greedy_route,
        points,
        start_point=start_point,
        return_to_start=return_to_start,
        budget=numeric_budget,
    )
    truncated_cost = route_cost(
        truncated_route,
        points,
        start_point=start_point,
        return_to_start=return_to_start,
    )
    options.append(
        _RouteOption(
            truncated_route,
            truncated_route,
            truncated_cost,
            _covered_for_route(matrix, truncated_route, initial),
            "truncated_greedy",
        )
    )

    # Classical GCB implementations compare a singleton as a guard against a
    # ratio-greedy miss.  It is harmless when GCB/greedy already dominate.
    singleton_options: list[_RouteOption] = []
    for candidate in range(matrix.shape[0]):
        singleton = (candidate,)
        cost = route_cost(
            singleton,
            points,
            start_point=start_point,
            return_to_start=return_to_start,
        )
        if cost <= numeric_budget + 1e-10:
            singleton_options.append(
                _RouteOption(
                    singleton,
                    singleton,
                    cost,
                    _covered_for_route(matrix, singleton, initial),
                    "best_singleton",
                )
            )
    if singleton_options:
        options.append(_best_option(singleton_options))

    best = _best_option(options)
    complete = bool(np.all(best.covered))
    uncovered = tuple(int(index) for index in np.flatnonzero(~best.covered))
    selected = tuple(sorted(best.route))
    return GCBCoverResult(
        selected_indices=selected,
        route_indices=best.route,
        selection_order=best.selection_order,
        route_cost=float(best.cost),
        covered_mask=_readonly(best.covered.astype(bool, copy=False)),
        complete=complete,
        uncovered_indices=uncovered,
        budget=budget_value,
        strategy=best.strategy,
    )
