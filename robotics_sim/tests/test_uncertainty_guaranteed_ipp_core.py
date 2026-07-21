"""Mathematical and contract tests for the NumPy-only paper core."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from algorithms.uncertainty_guaranteed_ipp import (
    RBFKernel,
    build_binary_coverage_matrix,
    certify_plan,
    gcb_cover,
    gp_posterior,
    greedy_cover,
    nearest_insertion_increment,
    nearest_insertion_route,
    posterior_variance,
    route_cost,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_rbf_posterior_variance_matches_single_observation_closed_form():
    kernel = RBFKernel(variance=2.0, length_scale=1.25)
    noise = 0.3
    candidate = np.array([[0.0]])
    evaluation = np.array([[0.8]])

    actual = posterior_variance(
        candidate,
        evaluation,
        kernel=kernel,
        noise_variance=noise,
        jitter=0.0,
    )[0]
    k_cc = kernel(candidate, candidate)[0, 0]
    k_vv = kernel(evaluation, evaluation)[0, 0]
    k_cv = kernel(candidate, evaluation)[0, 0]
    expected = k_vv - (k_cv * k_cv) / (k_cc + noise)

    assert actual == pytest.approx(expected, abs=1e-12)


def test_gp_posterior_returns_value_dependent_mean_but_same_planning_variance():
    kernel = RBFKernel(variance=1.0, length_scale=0.75)
    train = np.array([[0.0], [1.0]])
    query = np.array([[0.5]])

    positive = gp_posterior(train, [1.0, 1.0], query, kernel=kernel, noise_variance=0.05)
    negative = gp_posterior(train, [-1.0, -1.0], query, kernel=kernel, noise_variance=0.05)

    assert positive.mean[0] > 0.0
    assert negative.mean[0] < 0.0
    np.testing.assert_allclose(positive.variance, negative.variance, atol=1e-12)
    assert not positive.mean.flags.writeable
    assert not positive.covariance.flags.writeable


def test_theorem1_binary_matrix_is_exactly_covariance_squared_condition():
    kernel = RBFKernel(variance=1.4, length_scale=0.9)
    candidates = np.array([[0.0], [1.5], [3.0]])
    evaluations = np.array([[0.25], [1.0], [2.75]])
    target = 0.55
    noise = 0.08

    result = build_binary_coverage_matrix(
        candidates,
        evaluations,
        kernel=kernel,
        target_variance=target,
        noise_variance=noise,
        comparison_tolerance=0.0,
    )
    covariance = kernel(candidates, evaluations)
    c_var = kernel.diagonal(candidates)[:, None]
    v_var = kernel.diagonal(evaluations)[None, :]
    expected = covariance**2 >= (v_var - target) * (c_var + noise)

    np.testing.assert_array_equal(result.matrix, expected)
    np.testing.assert_allclose(
        result.covariance_thresholds**2,
        (v_var - target) * (c_var + noise),
        atol=1e-12,
    )


def test_prior_satisfied_points_need_no_gratuitous_greedy_selection():
    kernel = RBFKernel(variance=1.0, length_scale=1.0)
    coverage = build_binary_coverage_matrix(
        [[0.0], [2.0]],
        [[0.5], [1.5]],
        kernel=kernel,
        target_variance=1.1,
        noise_variance=0.1,
    )
    result = greedy_cover(coverage.matrix, initially_covered=coverage.initially_satisfied)

    assert coverage.initially_satisfied.tolist() == [True, True]
    assert result.complete
    assert result.selected_indices == ()


def test_greedy_cover_is_deterministic_on_ties_and_reports_marginal_gains():
    matrix = np.array(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ],
        dtype=bool,
    )

    result = greedy_cover(matrix)

    assert result.selected_indices == (0, 2)
    assert result.marginal_gains == (2, 1)
    assert result.complete
    assert result.uncovered_indices == ()


def test_greedy_cover_never_claims_an_uncoverable_evaluation_point():
    result = greedy_cover([[True, False], [True, False]])

    assert result.selected_indices == (0,)
    assert not result.complete
    assert result.uncovered_indices == (1,)


def test_nearest_insertion_finds_zero_cost_collinear_insertion():
    points = np.array([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
    choice = nearest_insertion_increment(
        (0, 1),
        2,
        points,
        start_point=(0.0, 0.0),
    )

    assert choice.insertion_position == 1
    assert choice.cost_increment == pytest.approx(0.0)
    assert choice.resulting_cost == pytest.approx(3.0)


def test_nearest_insertion_route_and_route_cost_share_one_cost_contract():
    points = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    route = nearest_insertion_route((0, 1, 2), points, start_point=(0.0, 0.0))

    assert set(route) == {0, 1, 2}
    assert route_cost(route, points, start_point=(0.0, 0.0)) <= route_cost(
        (0, 1, 2), points, start_point=(0.0, 0.0)
    )


def test_gcb_cover_obeys_strict_budget_and_reports_uncovered_points():
    # Candidate 2 has excellent coverage but is much too far away.  The
    # budget permits only the two nearby candidates.
    matrix = np.array(
        [
            [True, False, False],
            [False, True, False],
            [True, True, True],
        ],
        dtype=bool,
    )
    points = np.array([[1.0, 0.0], [2.0, 0.0], [20.0, 0.0]])

    result = gcb_cover(matrix, points, budget=2.0, start_point=(0.0, 0.0))

    assert result.route_cost <= 2.0 + 1e-12
    assert set(result.route_indices).issubset({0, 1})
    assert result.covered_count == 2
    assert not result.complete
    assert result.uncovered_indices == (2,)


def test_gcb_zero_increment_insertion_can_complete_coverage_without_budget_growth():
    # Candidate 0 wins the first ratio tie by larger gain and establishes a
    # two-metre route. Candidate 1 then lies exactly on that route, so its
    # remaining coverage can be inserted with zero additional travel.
    matrix = np.array(
        [[True, True, False], [False, False, True]],
        dtype=bool,
    )
    points = np.array([[2.0, 0.0], [1.0, 0.0]])

    result = gcb_cover(matrix, points, budget=2.0, start_point=(0.0, 0.0))

    assert result.complete
    assert result.route_cost == pytest.approx(2.0)
    assert set(result.selected_indices) == {0, 1}


def test_theorem_complete_selection_implies_exact_joint_posterior_certificate():
    kernel = RBFKernel(variance=1.0, length_scale=0.35)
    candidates = np.array([[0.0], [1.0], [2.0]])
    evaluations = candidates.copy()
    target = 0.12
    noise = 0.02
    coverage = build_binary_coverage_matrix(
        candidates,
        evaluations,
        kernel=kernel,
        target_variance=target,
        noise_variance=noise,
    )
    selection = greedy_cover(
        coverage.matrix,
        initially_covered=coverage.initially_satisfied,
    )
    certificate = certify_plan(
        candidates,
        evaluations,
        selection.selected_indices,
        kernel=kernel,
        target_variance=target,
        noise_variance=noise,
    )

    assert selection.complete
    assert certificate.theorem_coverage_certified
    assert certificate.certified
    assert certificate.max_posterior_variance <= target + 1e-9
    assert not certificate.posterior_variance.flags.writeable


def test_joint_posterior_can_certify_when_conservative_binary_union_does_not():
    # Each point alone leaves Var[f(0)] around 0.665, but the pair reduces it
    # below 0.5.  This is the conservative joint-effect gap discussed in the
    # paper appendix.
    kernel = RBFKernel(variance=1.0, length_scale=1.0)
    candidates = np.array([[-1.0], [1.0]])
    evaluations = np.array([[0.0]])
    target = 0.5
    noise = 0.1

    coverage = build_binary_coverage_matrix(
        candidates,
        evaluations,
        kernel=kernel,
        target_variance=target,
        noise_variance=noise,
    )
    certificate = certify_plan(
        candidates,
        evaluations,
        (0, 1),
        kernel=kernel,
        target_variance=target,
        noise_variance=noise,
    )

    assert not coverage.matrix.any()
    assert not certificate.theorem_coverage_certified
    assert certificate.certified
    assert certificate.max_posterior_variance < target


def test_certificate_rejects_missing_required_sensing_locations():
    kernel = RBFKernel(variance=1.0, length_scale=0.2)
    certificate = certify_plan(
        [[0.0], [2.0]],
        [[0.0], [2.0]],
        (0,),
        kernel=kernel,
        target_variance=0.1,
        noise_variance=0.01,
    )

    assert not certificate.certified
    assert certificate.violating_indices == (1,)
    assert certificate.theorem_uncovered_indices == (1,)


def test_package_import_is_numpy_only_and_does_not_import_simulator_or_gp_frameworks():
    script = """
import json, sys
import algorithms.uncertainty_guaranteed_ipp
blocked = [
    name for name in sys.modules
    if name == 'robotics_sim' or name.startswith('robotics_sim.')
    or name == 'scipy' or name.startswith('scipy.')
    or name == 'sklearn' or name.startswith('sklearn.')
    or name == 'tensorflow' or name.startswith('tensorflow.')
    or name == 'gpflow' or name.startswith('gpflow.')
    or name == 'sgptools' or name.startswith('sgptools.')
]
print(json.dumps(blocked))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "[]"


def test_rbf_is_explicitly_documented_as_mvp_not_attentive_reproduction():
    import algorithms.uncertainty_guaranteed_ipp as package

    documentation = (package.__doc__ or "").lower()
    assert "attentive" in documentation
    assert "sgp-tools" in documentation or "sgptools" in documentation
    assert "mvp" in documentation
    assert "must not be described as a faithful reproduction" in documentation
