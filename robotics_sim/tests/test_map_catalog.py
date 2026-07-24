"""Tests for MapCatalog / load_map_catalog()."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from robotics_sim.learning.map_catalog import (
    MapCatalog,
    MapCatalogEntry,
    MapCatalogError,
    MapScenarioEntry,
    load_map_catalog,
)

REAL_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "maps" / "smoke_v0" / "manifest.json"
)


@pytest.fixture(scope="module")
def real_catalog() -> MapCatalog:
    return load_map_catalog(REAL_MANIFEST_PATH)


def test_loads_six_maps(real_catalog: MapCatalog) -> None:
    assert len(real_catalog.map_ids) == 6


def test_map_order_is_stable(real_catalog: MapCatalog) -> None:
    assert real_catalog.map_ids == (
        "smoke_v0_01_open",
        "smoke_v0_02_office",
        "smoke_v0_03_corridors",
        "smoke_v0_04_loops",
        "smoke_v0_05_bottleneck",
        "smoke_v0_06_mixed",
    )


def test_scenarios_per_map(real_catalog: MapCatalog) -> None:
    scenarios = real_catalog.scenarios_for_map("smoke_v0_03_corridors")
    assert [s.scenario_id for s in scenarios] == ["single_fire", "double_fire"]
    single = real_catalog.get_scenario("smoke_v0_03_corridors", "single_fire")
    assert single.fire_positions == ((-3.0, 5.0),)
    double = real_catalog.get_scenario("smoke_v0_03_corridors", "double_fire")
    assert double.fire_positions == ((-3.0, 4.0), (-3.0, -7.0))


def test_get_map_unknown_raises(real_catalog: MapCatalog) -> None:
    with pytest.raises(MapCatalogError):
        real_catalog.get_map("does_not_exist")


def test_get_scenario_unknown_raises(real_catalog: MapCatalog) -> None:
    with pytest.raises(MapCatalogError):
        real_catalog.get_scenario("smoke_v0_01_open", "no_such_scenario")


def test_structures_are_frozen(real_catalog: MapCatalog) -> None:
    entry = real_catalog.get_map("smoke_v0_01_open")
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.map_id = "other"
    scenario = entry.scenarios[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        scenario.scenario_id = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        real_catalog.corpus_id = "other"


def _write_manifest(tmp_path: Path, maps: list[dict], corpus_id="c", schema_version=1) -> Path:
    manifest = {"corpus_id": corpus_id, "schema_version": schema_version, "maps": maps}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _map_block(map_id="m1", filename="m1.sim", family="open", difficulty="smoke", scenarios=None):
    return {
        "map_id": map_id,
        "filename": filename,
        "family": family,
        "difficulty": difficulty,
        "fire_scenarios": scenarios
        if scenarios is not None
        else [{"scenario_id": "single_fire", "fires": [{"x": 1.0, "y": 2.0}]}],
    }


def test_missing_sim_file_raises(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [_map_block()])
    with pytest.raises(MapCatalogError):
        load_map_catalog(manifest_path)


def test_duplicate_map_id_rejected(tmp_path: Path) -> None:
    (tmp_path / "m1.sim").write_text("{}")
    (tmp_path / "m2.sim").write_text("{}")
    manifest_path = _write_manifest(
        tmp_path,
        [_map_block(map_id="dup", filename="m1.sim"), _map_block(map_id="dup", filename="m2.sim")],
    )
    with pytest.raises(MapCatalogError):
        load_map_catalog(manifest_path)


def test_duplicate_filename_rejected(tmp_path: Path) -> None:
    (tmp_path / "shared.sim").write_text("{}")
    manifest_path = _write_manifest(
        tmp_path,
        [
            _map_block(map_id="m1", filename="shared.sim"),
            _map_block(map_id="m2", filename="shared.sim"),
        ],
    )
    with pytest.raises(MapCatalogError):
        load_map_catalog(manifest_path)


def test_duplicate_scenario_id_within_map_rejected(tmp_path: Path) -> None:
    (tmp_path / "m1.sim").write_text("{}")
    manifest_path = _write_manifest(
        tmp_path,
        [
            _map_block(
                scenarios=[
                    {"scenario_id": "dup", "fires": [{"x": 0.0, "y": 0.0}]},
                    {"scenario_id": "dup", "fires": [{"x": 1.0, "y": 1.0}]},
                ]
            )
        ],
    )
    with pytest.raises(MapCatalogError):
        load_map_catalog(manifest_path)


def test_manifest_file_not_mutated(tmp_path: Path) -> None:
    (tmp_path / "m1.sim").write_text("{}")
    manifest_path = _write_manifest(tmp_path, [_map_block()])
    before = manifest_path.read_text(encoding="utf-8")
    load_map_catalog(manifest_path)
    after = manifest_path.read_text(encoding="utf-8")
    assert before == after


def test_load_map_catalog_writes_no_files(tmp_path: Path) -> None:
    (tmp_path / "m1.sim").write_text("{}")
    manifest_path = _write_manifest(tmp_path, [_map_block()])
    before = set(tmp_path.iterdir())
    load_map_catalog(manifest_path)
    after = set(tmp_path.iterdir())
    assert before == after
