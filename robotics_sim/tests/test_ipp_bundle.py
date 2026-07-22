"""Contract tests for the pure RSS26 IPP visualization-bundle loader."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robotics_sim.experiments.ipp_bundle import (
    IPP_BUNDLE_SCHEMA,
    IppBundleError,
    load_ipp_bundle,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "field": np.arange(6, dtype=float).reshape(2, 3),
        "prior_variance": np.full((2, 3), 2.0),
        "posterior_variance": np.full((2, 3), 0.5),
        "mask": np.array([[1, 1, 0], [1, 1, 1]], dtype=np.uint8),
        "pilot_path": np.array([[0.0, 0.0], [4.0, 2.0]]),
        "solution_path": np.array([[0.0, 2.0], [4.0, 0.0]]),
        "sensing_points": np.array([[2.0, 1.0]]),
        "fovs": np.array([[[1.5, 0.5], [2.5, 0.5], [2.5, 1.5], [1.5, 1.5]]]),
    }


def _write_bundle(tmp_path, *, arrays=None, manifest_updates=None):
    assets = tmp_path / "assets"
    assets.mkdir()
    np.savez(assets / "data.npz", **(_arrays() if arrays is None else arrays))
    manifest = {
        "schema": IPP_BUNDLE_SCHEMA,
        "version": 1,
        "npz": "assets/data.npz",
        "dataset_bounds": [0.0, 4.0, 0.0, 2.0],
        "raster_origin": "lower",
        "metrics": {"method": "GreedyCover", "distance_m": 238.0},
    }
    manifest.update(manifest_updates or {})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_loads_valid_bundle_and_aspect_fits_coordinates(tmp_path):
    bundle = load_ipp_bundle(_write_bundle(tmp_path))

    assert bundle.field.shape == (2, 3)
    assert bundle.mask.dtype == np.bool_
    assert bundle.metrics["method"] == "GreedyCover"
    assert bundle.metrics["distance_m"] == 238.0
    # 4:2 dataset fits the 20:16 world by width: x fills [-10, 10] and the
    # data occupies a centered 10-unit-high band y=[-5, 5].
    assert bundle.data_world_bounds == pytest.approx((-10.0, 10.0, -5.0, 5.0))
    np.testing.assert_allclose(bundle.pilot_path, [[-10.0, -5.0], [10.0, 5.0]])
    np.testing.assert_allclose(bundle.sensing_points, [[0.0, 0.0]])
    np.testing.assert_allclose(
        bundle.world_to_dataset(bundle.solution_path), [[0.0, 2.0], [4.0, 0.0]]
    )


def test_loaded_arrays_are_read_only(tmp_path):
    bundle = load_ipp_bundle(_write_bundle(tmp_path))
    for array in (
        bundle.field,
        bundle.prior_variance,
        bundle.posterior_variance,
        bundle.mask,
        bundle.pilot_path,
        bundle.solution_path,
        bundle.sensing_points,
        bundle.fovs,
    ):
        assert not array.flags.writeable


def test_custom_npz_array_names_are_supported(tmp_path):
    arrays = {f"rss_{name}": value for name, value in _arrays().items()}
    manifest = _write_bundle(
        tmp_path,
        arrays=arrays,
        manifest_updates={"arrays": {name: f"rss_{name}" for name in _arrays()}},
    )
    bundle = load_ipp_bundle(manifest)
    np.testing.assert_allclose(bundle.posterior_variance, np.full((2, 3), 0.5))


def test_npz_path_cannot_escape_manifest_directory(tmp_path):
    outside = tmp_path.parent / "outside.npz"
    np.savez(outside, **_arrays())
    manifest = _write_bundle(tmp_path, manifest_updates={"npz": "../outside.npz"})
    with pytest.raises(IppBundleError, match="escapes"):
        load_ipp_bundle(manifest)


@pytest.mark.parametrize(
    ("role", "bad_value", "message"),
    [
        ("prior_variance", np.ones((3, 2)), "match field shape"),
        ("posterior_variance", np.array([[0.0, np.nan, 0.0], [0.0, 0.0, 0.0]]), "finite"),
        ("posterior_variance", -np.ones((2, 3)), "negative variance"),
        ("mask", np.array([[0, 1, 2], [0, 1, 0]]), "only 0/1"),
        ("mask", np.zeros((2, 3)), "at least one"),
        ("pilot_path", np.ones((2, 3)), r"shape \(N, 2\)"),
        ("fovs", np.ones((1, 2, 2)), "at least three"),
    ],
)
def test_rejects_invalid_numeric_assets(tmp_path, role, bad_value, message):
    arrays = _arrays()
    arrays[role] = bad_value
    with pytest.raises(IppBundleError, match=message):
        load_ipp_bundle(_write_bundle(tmp_path, arrays=arrays))


def test_transform_rejects_nonfinite_or_malformed_coordinates(tmp_path):
    bundle = load_ipp_bundle(_write_bundle(tmp_path))
    with pytest.raises(IppBundleError, match="shape"):
        bundle.dataset_to_world([1.0, 2.0, 3.0])
    with pytest.raises(IppBundleError, match="finite"):
        bundle.world_to_dataset([[0.0, np.inf]])


def test_rejects_nonfinite_metrics_in_manifest(tmp_path):
    path = _write_bundle(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["metrics"] = {"distance_m": float("nan")}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(IppBundleError, match="non-finite"):
        load_ipp_bundle(path)
