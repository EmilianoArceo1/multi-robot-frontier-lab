"""Generate the small, dependency-free RSS26 integration smoke bundle.

This is deliberately *not* a substitute for the paper benchmark.  It uses a
stationary RBF kernel so the theorem, bundle, canvas, and waypoint plumbing can
be exercised without TensorFlow/SGP-Tools.  Use ``run_official_benchmark.py``
for the pinned Attentive-kernel reproduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from algorithms.uncertainty_guaranteed_ipp import (
    RBFKernel,
    build_binary_coverage_matrix,
    certify_plan,
    greedy_cover,
    nearest_insertion_route,
    posterior_variance,
    route_cost,
)
from robotics_sim.experiments import IPP_BUNDLE_SCHEMA, IPP_BUNDLE_VERSION


SEED = 1234
DATASET_BOUNDS = (0.0, 20.0, 0.0, 16.0)


def _deduplicate_consecutive(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-10
    return points[keep]


def generate(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    x_axis = np.linspace(DATASET_BOUNDS[0], DATASET_BOUNDS[1], 41)
    y_axis = np.linspace(DATASET_BOUNDS[2], DATASET_BOUNDS[3], 33)
    xx, yy = np.meshgrid(x_axis, y_axis)
    mask = ((xx - 10.0) / 9.5) ** 2 + ((yy - 8.0) / 7.5) ** 2 <= 1.0
    # A small inlet makes the smoke domain non-convex without introducing
    # occupancy geometry (the paper's domain is known, not frontier-mapped).
    mask &= ~((xx > 14.0) & (yy > 6.0) & (yy < 10.0))
    evaluation_points = np.column_stack((xx[mask], yy[mask]))

    candidate_x = np.arange(1.0, 20.0, 2.0)
    candidate_y = np.arange(1.0, 16.0, 2.0)
    cxx, cyy = np.meshgrid(candidate_x, candidate_y)
    candidate_points = np.column_stack((cxx.ravel(), cyy.ravel()))
    candidate_mask = (
        ((candidate_points[:, 0] - 10.0) / 9.5) ** 2
        + ((candidate_points[:, 1] - 8.0) / 7.5) ** 2
        <= 1.0
    )
    candidate_mask &= ~(
        (candidate_points[:, 0] > 14.0)
        & (candidate_points[:, 1] > 6.0)
        & (candidate_points[:, 1] < 10.0)
    )
    candidate_points = candidate_points[candidate_mask]

    kernel = RBFKernel(variance=1.0, length_scale=3.0)
    target_variance = 0.50
    noise_variance = 0.02
    coverage = build_binary_coverage_matrix(
        candidate_points,
        evaluation_points,
        kernel=kernel,
        target_variance=target_variance,
        noise_variance=noise_variance,
    )
    selected = greedy_cover(
        coverage.matrix,
        initially_covered=coverage.initially_satisfied,
    )
    if not selected.complete or not selected.selected_indices:
        raise RuntimeError("RBF smoke grid did not produce a complete cover")

    start_point = candidate_points[selected.selected_indices[0]]
    route_indices = nearest_insertion_route(
        selected.selected_indices,
        candidate_points,
        start_point=start_point,
        return_to_start=False,
    )
    solution_path = _deduplicate_consecutive(
        np.vstack((start_point, candidate_points[np.asarray(route_indices, dtype=int)]))
    )
    certificate = certify_plan(
        candidate_points,
        evaluation_points,
        selected.selected_indices,
        kernel=kernel,
        target_variance=target_variance,
        noise_variance=noise_variance,
    )

    prior_grid = np.zeros_like(xx, dtype=np.float64)
    prior_grid[mask] = kernel.variance
    posterior_grid = np.zeros_like(xx, dtype=np.float64)
    posterior_grid[mask] = posterior_variance(
        candidate_points[np.asarray(selected.selected_indices, dtype=int)],
        evaluation_points,
        kernel=kernel,
        noise_variance=noise_variance,
    )
    field = (
        1.4 * np.exp(-((xx - 5.0) ** 2 + (yy - 11.0) ** 2) / 18.0)
        - 0.9 * np.exp(-((xx - 14.0) ** 2 + (yy - 4.0) ** 2) / 12.0)
        + 0.15 * np.sin(xx / 1.8) * np.cos(yy / 2.2)
    )
    field[~mask] = 0.0
    pilot_path = np.array(
        [
            [2.0, 3.0],
            [5.0, 5.0],
            [8.0, 3.5],
            [11.0, 5.5],
            [13.0, 8.0],
            [11.0, 11.0],
            [8.0, 12.5],
            [5.0, 11.0],
            [3.0, 8.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    fovs = np.empty((0, 4, 2), dtype=np.float64)

    npz_path = output_dir / "data.npz"
    np.savez_compressed(
        npz_path,
        field=field,
        prior_variance=prior_grid,
        posterior_variance=posterior_grid,
        mask=mask,
        pilot_path=pilot_path,
        solution_path=solution_path,
        sensing_points=candidate_points[np.asarray(selected.selected_indices, dtype=int)],
        fovs=fovs,
    )

    distance = route_cost(
        route_indices,
        candidate_points,
        start_point=start_point,
        return_to_start=False,
    )
    manifest = {
        "schema": IPP_BUNDLE_SCHEMA,
        "version": IPP_BUNDLE_VERSION,
        "npz": npz_path.name,
        "dataset_bounds": list(DATASET_BOUNDS),
        "raster_origin": "lower",
        "metrics": {
            "method": "GreedyCover (RBF smoke test)",
            "fidelity": "integration_smoke_not_paper_benchmark",
            "seed": SEED,
            "kernel": "stationary RBF",
            "target_variance": target_variance,
            "noise_variance": noise_variance,
            "sensing_point_count": len(selected.selected_indices),
            "path_length_dataset_units": distance,
            "certificate_passed": certificate.certified,
            "theorem_coverage_passed": certificate.theorem_coverage_certified,
            "max_posterior_variance": certificate.max_posterior_variance,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path, npz_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/rss26_ipp_rbf_smoke"),
    )
    args = parser.parse_args()
    manifest, archive = generate(args.output)
    print(f"Wrote {manifest}")
    print(f"Wrote {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
