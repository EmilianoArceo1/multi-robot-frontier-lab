"""Pinned runner/normalizer for the authors' RSS26 benchmark.

The reference repository is never modified.  After verifying its exact commit
and dataset hash, this runner makes a temporary copy of ``benchmark.py`` and
adds one observation-only NPZ export at the point where the official code has
already computed each solution.  Planner construction, optimization, metrics,
and figures remain the authors' code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Sequence

import numpy as np

from robotics_sim.experiments import IPP_BUNDLE_SCHEMA, IPP_BUNDLE_VERSION, load_ipp_bundle


REFERENCE_URL = "https://github.com/itskalvik/uncertainty-guaranteed-ipp.git"
REFERENCE_COMMIT = "f387c57bcaa61bf218d26e63212bf789ef42a534"
DATASET_SHA256 = {
    "N02E021.npy": "34c168eea266280011edc46c6f94670239216996fc81587385c51f6c0ceb0ee1",
    "N17E073.npy": "a83da6ecf20d48346cbff8dc031c396b155e6b41cc81cb1d50bc142ac72756b4",
    "N45W123.npy": "214bd67ffe43f40c41c5e273259bce2b3f5e2e9c0dd257f8d42a6477b26dc009",
    "N47W124.npy": "81f19ee4ecddaae1e86e229aefc420cc4dc04853d1bd9f4e58962509eac567e5",
}
DEFAULT_VARIANCE_RATIOS = (0.9, 0.8, 0.7, 0.6, 0.5)
DEFAULT_METHODS = (
    "HexCover",
    "GreedyCover",
    "GCBCover",
    "GCBCover-Dist",
    "ContinuousSGP",
)


class ReproductionError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: Sequence[str], *, cwd: Path | None = None, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd) if cwd else None,
        env=env,
        check=True,
        text=True,
    )


def verify_reference(reference: Path, dataset_name: str, *, git: str = "git") -> Path:
    reference = reference.resolve()
    benchmark = reference / "benchmark.py"
    dataset = reference / "datasets" / dataset_name
    if not benchmark.is_file():
        raise ReproductionError(f"Missing official benchmark.py under {reference}")
    if dataset_name not in DATASET_SHA256 or not dataset.is_file():
        raise ReproductionError(f"Unsupported or missing official dataset: {dataset_name}")
    try:
        completed = subprocess.run(
            [git, "-c", f"safe.directory={reference.as_posix()}", "-C", str(reference), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReproductionError(f"Could not verify reference git commit: {exc}") from exc
    actual_commit = completed.stdout.strip().lower()
    if actual_commit != REFERENCE_COMMIT:
        raise ReproductionError(
            f"Reference commit mismatch: expected {REFERENCE_COMMIT}, got {actual_commit or '<empty>'}"
        )
    actual_hash = sha256_file(dataset)
    if actual_hash != DATASET_SHA256[dataset_name]:
        raise ReproductionError(
            f"Dataset hash mismatch for {dataset_name}: expected {DATASET_SHA256[dataset_name]}, got {actual_hash}"
        )
    source = benchmark.read_text(encoding="utf-8")
    if "np.random.seed(1234)" not in source or "tf.random.set_seed(1234)" not in source:
        raise ReproductionError("Official benchmark no longer contains the published seed 1234 contract.")
    return dataset


def instrument_benchmark(source: str) -> str:
    """Add numeric exports only; fail if the pinned source shape changed."""
    before_result = "            run_result = {\n"
    append_result = '            results["runs"].append(run_result)\n'
    if source.count(before_result) != 1 or source.count(append_result) != 1:
        raise ReproductionError("Pinned benchmark export insertion points were not found exactly once.")
    export = '''            # robotics_sim observation-only export (planner state is already final)
            _ratio_tag = str(target_var_ratio).replace(".", "p")
            _bundle_stem = f"{method}_ratio{_ratio_tag}"
            _bundle_dir = os.path.join(output_dir, "bundles", _bundle_stem)
            os.makedirs(_bundle_dir, exist_ok=True)
            _fov_arrays = [np.asarray(item.exterior.coords, dtype=float) for item in fovs]
            if _fov_arrays and len({item.shape for item in _fov_arrays}) == 1:
                _fovs_export = np.stack(_fov_arrays, axis=0)
            else:
                _fovs_export = np.empty((0, 3, 2), dtype=float)
            _bundle_npz_path = os.path.join(_bundle_dir, "data.npz")
            np.savez_compressed(
                _bundle_npz_path,
                field=np.asarray(y_grid).reshape(x_dim, y_dim).T,
                prior_variance=np.asarray(prior_var.numpy()).reshape(x_dim, y_dim).T,
                posterior_variance=np.asarray(var_np).reshape(x_dim, y_dim).T,
                mask=np.ones((y_dim, x_dim), dtype=bool),
                pilot_path=np.asarray(X_init, dtype=float),
                solution_path=np.asarray(X_sol, dtype=float),
                sensing_points=np.asarray(X_sol, dtype=float),
                fovs=_fovs_export,
                dataset_bounds=np.asarray(extent, dtype=float),
            )
'''
    source = source.replace(before_result, export + before_result)
    source = source.replace(
        append_result,
        '            run_result["bundle_npz"] = os.path.relpath(_bundle_npz_path, output_dir)\n'
        + append_result,
    )
    return source


def build_official_command(
    python: str,
    script: Path,
    dataset: Path,
    output: Path,
    *,
    kernel: str,
    variance_ratios: Sequence[float],
    methods: Sequence[str],
) -> list[str]:
    return [
        python,
        str(script),
        str(dataset),
        "--kernel",
        kernel,
        "--variance-ratios",
        *(str(float(value)) for value in variance_ratios),
        "--methods",
        *methods,
        "--num-initial",
        "350",
        "--num-train",
        "5000",
        "--num-inducing",
        "15",
        "--grid-size",
        "100",
        "100",
        "--output-dir",
        str(output),
    ]


def validate_benchmark_selection(
    variance_ratios: Sequence[float], methods: Sequence[str]
) -> None:
    if not variance_ratios or any(not 0.0 < float(value) <= 1.0 for value in variance_ratios):
        raise ReproductionError("Every variance ratio must lie in (0, 1].")
    unknown = [method for method in methods if method not in DEFAULT_METHODS]
    if not methods or unknown:
        raise ReproductionError(f"Unsupported benchmark methods: {unknown or '<empty>'}")
    greedy_index = methods.index("GreedyCover") if "GreedyCover" in methods else None
    for dependent in ("GCBCover-Dist", "ContinuousSGP"):
        if dependent in methods and (
            greedy_index is None or greedy_index > methods.index(dependent)
        ):
            raise ReproductionError(
                f"{dependent} requires GreedyCover earlier in --methods because the official "
                "script derives its distance/site reference from that run."
            )


def normalize_results(results_path: Path) -> Path:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    output_dir = results_path.parent.resolve()
    manifests = []
    scenarios = []
    for run in data.get("runs", []):
        relative_npz = run.get("bundle_npz")
        if not isinstance(relative_npz, str):
            raise ReproductionError("Instrumented result is missing bundle_npz.")
        npz_path = (output_dir / relative_npz).resolve()
        try:
            npz_path.relative_to(output_dir)
        except ValueError as exc:
            raise ReproductionError("Instrumented bundle escaped the result directory.") from exc
        with np.load(npz_path, allow_pickle=False) as archive:
            bounds = [float(value) for value in np.asarray(archive["dataset_bounds"]).reshape(-1)]
        if len(bounds) != 4:
            raise ReproductionError("Instrumented bundle has invalid dataset bounds.")
        metrics = {
            **{key: value for key, value in run.items() if key != "bundle_npz"},
            "fidelity": "official_pinned_benchmark_with_observation_only_export",
            "paper_seed": 1234,
            "source_commit": REFERENCE_COMMIT,
            "kernel": data.get("kernel"),
            "dataset": Path(str(data.get("dataset", ""))).name,
        }
        manifest = {
            "schema": IPP_BUNDLE_SCHEMA,
            "version": IPP_BUNDLE_VERSION,
            "npz": npz_path.name,
            "dataset_bounds": bounds,
            "raster_origin": "lower",
            "metrics": metrics,
        }
        manifest_path = npz_path.parent / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        bundle = load_ipp_bundle(manifest_path)
        if len(bundle.solution_path) < 2:
            raise ReproductionError("Official solution must contain at least two route points.")
        start = bundle.solution_path[0]
        second = bundle.solution_path[1]
        goal = bundle.solution_path[-1]
        theta = math.atan2(float(second[1] - start[1]), float(second[0] - start[0]))
        experiment_id = f"rss26-{metrics['dataset']}-{run.get('method')}-{run.get('variance_ratio')}"
        robot = {
            "x": float(start[0]), "y": float(start[1]), "theta": theta, "v": 0.0,
            "body_radius": 0.16, "safety_radius": 0.25,
            "max_speed": 1.0, "max_acceleration": 1.5,
            "max_angular_speed": 2.5, "goal_tolerance": 0.15,
            "acceleration_gain": 0.75,
        }
        scenario = {
            "schema": "robotics_sim_lab.sim",
            "version": 1,
            "experiment": {
                "id": experiment_id,
                "kind": "uncertainty_guaranteed_ipp_rss26",
                "paper": "Informative Path Planning with Guaranteed Estimation Uncertainty",
                "arxiv": "2602.05198v3",
                "source_repository": REFERENCE_URL.removesuffix(".git"),
                "source_commit": REFERENCE_COMMIT,
                "fidelity": metrics["fidelity"],
                "model": f"{data.get('kernel')} kernel through official sgptools benchmark",
                "bundle": "manifest.json",
                "description": "Pinned official RSS26 benchmark result executed as a single-robot sensing tour.",
            },
            "world": {"x_min": -10.0, "x_max": 10.0, "y_min": -8.0, "y_max": 8.0},
            "robot": robot,
            "goal": {"x": float(goal[0]), "y": float(goal[1])},
            "map": {"grid_resolution": 0.25, "obstacles": []},
            "camera": {"center_x": 0.0, "center_y": 0.0, "width": 20.0, "height": 16.0},
            "planner": {"type": "Direct", "path_simplifier": "Raw grid path"},
            "exploration": {"planner": "Goal seeking", "replan_cooldown": 1.0, "ipp_distance_penalty": 0.0},
            "coordination": {"strategy": "Independent frontiers"},
            "sensor": {"type": "Camera / FoV", "range": 0.8},
            "multi_robot": {
                "robot_count": 1,
                "selected_robot_index": 0,
                "same_robot_configuration": True,
                "robots": [{**robot, "vision": 0.8}],
            },
            "simulation": {
                "agent_mode": "Single Robot Mode",
                "map_visualization": "Current",
                "robot_icon": "Wheeled Robot",
                "show_goal_preview": False,
                "show_path": True,
                "show_vision": False,
                "show_explored_area": False,
                "show_obstacles": False,
                "show_robot_orders": True,
                "mapping_point_spacing": 0.025,
            },
        }
        scenario_path = manifest_path.parent / "scenario.sim"
        scenario_path.write_text(
            json.dumps(scenario, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifests.append(str(manifest_path.relative_to(output_dir)))
        scenarios.append(str(scenario_path.relative_to(output_dir)))

    normalized = {
        "schema": "robotics_sim.rss26_ipp_results",
        "version": 1,
        "source_repository": REFERENCE_URL,
        "source_commit": REFERENCE_COMMIT,
        "paper_seed": 1234,
        "official_results": results_path.name,
        "manifests": manifests,
        "scenarios": scenarios,
        "benchmark": data,
    }
    normalized_path = output_dir / "normalized_results.json"
    normalized_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return normalized_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_SHA256), default="N47W124.npy")
    parser.add_argument("--output", type=Path, default=Path("runs/rss26_ipp"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--git", default="git")
    parser.add_argument("--kernel", choices=("Attentive", "RBF"), default="Attentive")
    parser.add_argument("--variance-ratios", nargs="+", type=float, default=list(DEFAULT_VARIANCE_RATIOS))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_benchmark_selection(args.variance_ratios, args.methods)
    dataset = verify_reference(args.reference, args.dataset, git=args.git)
    print(f"Verified {REFERENCE_COMMIT} and {args.dataset} ({DATASET_SHA256[args.dataset]}).")
    if args.verify_only:
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (args.reference.resolve() / "benchmark.py").read_text(encoding="utf-8")
    instrumented = instrument_benchmark(source)
    with tempfile.TemporaryDirectory(prefix="rss26_ipp_", dir=output) as temporary:
        script = Path(temporary) / "benchmark_instrumented.py"
        script.write_text(instrumented, encoding="utf-8")
        command = build_official_command(
            args.python,
            script,
            dataset,
            output,
            kernel=args.kernel,
            variance_ratios=args.variance_ratios,
            methods=args.methods,
        )
        environment = dict(os.environ)
        environment.setdefault("MPLBACKEND", "Agg")
        _run(command, cwd=args.reference.resolve(), env=environment)

    results = output / dataset.stem / args.kernel / "results.json"
    if not results.is_file():
        raise ReproductionError(f"Official benchmark did not produce {results}")
    normalized = normalize_results(results)
    print(f"Normalized results: {normalized}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReproductionError as exc:
        print(f"RSS26 reproduction failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
