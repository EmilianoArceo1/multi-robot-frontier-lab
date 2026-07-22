from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.rss26_ipp.runner import (
    REFERENCE_COMMIT,
    build_official_command,
    instrument_benchmark,
    normalize_results,
    ReproductionError,
    validate_benchmark_selection,
)
from robotics_sim.experiments import load_ipp_bundle


def test_official_command_pins_paper_scale_parameters() -> None:
    command = build_official_command(
        "paper-python",
        Path("instrumented.py"),
        Path("N47W124.npy"),
        Path("out"),
        kernel="Attentive",
        variance_ratios=(0.9, 0.7, 0.5),
        methods=("GreedyCover", "GCBCover"),
    )
    joined = " ".join(command)
    assert "--num-initial 350" in joined
    assert "--num-train 5000" in joined
    assert "--num-inducing 15" in joined
    assert "--grid-size 100 100" in joined
    assert "--kernel Attentive" in joined


def test_method_order_preserves_official_reference_dependencies() -> None:
    validate_benchmark_selection(
        (0.9, 0.5),
        ("GreedyCover", "GCBCover-Dist", "ContinuousSGP"),
    )
    with pytest.raises(ReproductionError, match="requires GreedyCover earlier"):
        validate_benchmark_selection((0.7,), ("ContinuousSGP", "GreedyCover"))
    with pytest.raises(ReproductionError, match=r"\(0, 1\]"):
        validate_benchmark_selection((0.0,), ("GreedyCover",))


def test_instrumentation_is_observation_only_and_requires_pinned_markers() -> None:
    source = '''def main():
    for method in methods:
            run_result = {
                "method": method,
            }
            results["runs"].append(run_result)
'''
    instrumented = instrument_benchmark(source)
    assert "np.savez_compressed" in instrumented
    assert 'run_result["bundle_npz"]' in instrumented
    assert "np.savez_compressed" not in source
    compile(instrumented, "instrumented_benchmark.py", "exec")


def test_normalizer_emits_loadable_bundle_and_pinned_provenance(tmp_path) -> None:
    bundle_dir = tmp_path / "bundles" / "GreedyCover_ratio0p7"
    bundle_dir.mkdir(parents=True)
    shape = (2, 3)
    np.savez_compressed(
        bundle_dir / "data.npz",
        field=np.arange(6, dtype=float).reshape(shape),
        prior_variance=np.ones(shape),
        posterior_variance=np.full(shape, 0.4),
        mask=np.ones(shape, dtype=bool),
        pilot_path=np.array([[0.0, 0.0], [2.0, 1.0]]),
        solution_path=np.array([[0.0, 0.0], [1.0, 1.0]]),
        sensing_points=np.array([[1.0, 1.0]]),
        fovs=np.empty((0, 3, 2)),
        dataset_bounds=np.array([0.0, 2.0, 0.0, 1.0]),
    )
    results = {
        "dataset": "N47W124.npy",
        "kernel": "Attentive",
        "runs": [
            {
                "method": "GreedyCover",
                "variance_ratio": 0.7,
                "max_posterior_var": 0.4,
                "bundle_npz": "bundles/GreedyCover_ratio0p7/data.npz",
            }
        ],
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    normalized_path = normalize_results(results_path)
    normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    assert normalized["source_commit"] == REFERENCE_COMMIT
    manifest_path = tmp_path / normalized["manifests"][0]
    scenario_path = tmp_path / normalized["scenarios"][0]
    bundle = load_ipp_bundle(manifest_path)
    assert bundle.metrics["fidelity"] == (
        "official_pinned_benchmark_with_observation_only_export"
    )
    assert bundle.metrics["kernel"] == "Attentive"
    assert bundle.solution_path.shape == (2, 2)
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    assert scenario["experiment"]["bundle"] == "manifest.json"
    assert scenario["simulation"]["agent_mode"] == "Single Robot Mode"
    assert scenario["robot"]["x"] == bundle.solution_path[0, 0]
