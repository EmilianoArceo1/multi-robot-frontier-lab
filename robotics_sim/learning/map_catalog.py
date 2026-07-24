"""Read-only view over one map-corpus manifest (e.g. smoke_v0).

MapCatalog is a pure data lookup: it parses a manifest.json produced by the
map corpus tooling, defensively copies every structure, and confirms that
every referenced .sim file actually exists next to the manifest -- but it
never opens, parses, or reads a .sim file's contents, never writes
anything, and never invents a map, scenario, or fire position that is not
already in the manifest.

Allowed dependency direction: stdlib only (json, pathlib, dataclasses). No
Qt, robotics_sim.app, robotics_sim.simulation, robotics_interfaces or
engine imports -- this module has no notion of a running simulation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


class MapCatalogError(ValueError):
    """The manifest is structurally invalid or internally inconsistent."""


@dataclass(frozen=True)
class MapScenarioEntry:
    """One named fire-placement scenario for one map."""

    scenario_id: str
    fire_positions: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError(f"scenario_id must be a non-empty string, got {self.scenario_id!r}")

        fires: list[tuple[float, float]] = []
        for i, point in enumerate(self.fire_positions):
            values = tuple(point)
            if len(values) != 2:
                raise ValueError(f"fire_positions[{i}] must be an (x, y) pair, got {values!r}")
            x, y = values
            for component in (x, y):
                if isinstance(component, bool) or not isinstance(component, (int, float)):
                    raise TypeError(
                        f"fire_positions[{i}] components must be real numbers, got "
                        f"{type(component).__name__}"
                    )
                if not math.isfinite(component):
                    raise ValueError(f"fire_positions[{i}] must be finite, got {values!r}")
            fires.append((float(x), float(y)))
        object.__setattr__(self, "fire_positions", tuple(fires))


@dataclass(frozen=True)
class MapCatalogEntry:
    """One map's identity plus its ordered scenario list."""

    map_id: str
    filename: str
    family: str
    difficulty: str
    scenarios: tuple[MapScenarioEntry, ...]

    def __post_init__(self) -> None:
        for name in ("map_id", "filename", "family", "difficulty"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string, got {value!r}")

        scenarios = tuple(self.scenarios)
        seen_scenario_ids: set[str] = set()
        for i, scenario in enumerate(scenarios):
            if not isinstance(scenario, MapScenarioEntry):
                raise TypeError(
                    f"scenarios[{i}] must be a MapScenarioEntry, got {type(scenario).__name__}"
                )
            if scenario.scenario_id in seen_scenario_ids:
                raise MapCatalogError(
                    f"map {self.map_id!r} has duplicate scenario_id {scenario.scenario_id!r}"
                )
            seen_scenario_ids.add(scenario.scenario_id)
        object.__setattr__(self, "scenarios", scenarios)


@dataclass(frozen=True)
class MapCatalog:
    """An ordered, duplicate-free collection of MapCatalogEntry.

    ``_index`` is a private, non-init field built once in __post_init__ so
    get_map()/scenarios_for_map()/get_scenario() do not re-scan ``entries``
    on every call; it is never part of equality or repr and is never
    exposed publicly.
    """

    corpus_id: str
    schema_version: int
    entries: tuple[MapCatalogEntry, ...]
    _index: Mapping[str, MapCatalogEntry] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.corpus_id, str) or not self.corpus_id.strip():
            raise ValueError(f"corpus_id must be a non-empty string, got {self.corpus_id!r}")
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise TypeError(
                f"schema_version must be an int, got {type(self.schema_version).__name__}"
            )
        if self.schema_version <= 0:
            raise ValueError(f"schema_version must be positive, got {self.schema_version}")

        entries = tuple(self.entries)
        index: dict[str, MapCatalogEntry] = {}
        seen_filenames: set[str] = set()
        for i, entry in enumerate(entries):
            if not isinstance(entry, MapCatalogEntry):
                raise TypeError(f"entries[{i}] must be a MapCatalogEntry, got {type(entry).__name__}")
            if entry.map_id in index:
                raise MapCatalogError(f"duplicate map_id {entry.map_id!r}")
            if entry.filename in seen_filenames:
                raise MapCatalogError(f"duplicate filename {entry.filename!r}")
            index[entry.map_id] = entry
            seen_filenames.add(entry.filename)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_index", index)

    @property
    def map_ids(self) -> tuple[str, ...]:
        return tuple(entry.map_id for entry in self.entries)

    def get_map(self, map_id: str) -> MapCatalogEntry:
        try:
            return self._index[map_id]
        except KeyError:
            raise MapCatalogError(f"unknown map_id {map_id!r}") from None

    def scenarios_for_map(self, map_id: str) -> tuple[MapScenarioEntry, ...]:
        return self.get_map(map_id).scenarios

    def get_scenario(self, map_id: str, scenario_id: str) -> MapScenarioEntry:
        for scenario in self.scenarios_for_map(map_id):
            if scenario.scenario_id == scenario_id:
                return scenario
        raise MapCatalogError(f"map {map_id!r} has no scenario_id {scenario_id!r}")


def _build_scenario(raw: Mapping[str, object]) -> MapScenarioEntry:
    fires_raw = raw.get("fires", [])
    fire_positions = tuple((float(p["x"]), float(p["y"])) for p in fires_raw)
    return MapScenarioEntry(scenario_id=str(raw["scenario_id"]), fire_positions=fire_positions)


def _build_map_entry(raw: Mapping[str, object], *, manifest_dir: Path) -> MapCatalogEntry:
    filename = str(raw["filename"])
    sim_path = manifest_dir / filename
    if not sim_path.is_file():
        raise MapCatalogError(
            f"map {raw.get('map_id')!r} references filename {filename!r}, but "
            f"{sim_path} does not exist"
        )
    scenarios = tuple(_build_scenario(dict(s)) for s in raw.get("fire_scenarios", []))
    return MapCatalogEntry(
        map_id=str(raw["map_id"]),
        filename=filename,
        family=str(raw["family"]),
        difficulty=str(raw["difficulty"]),
        scenarios=scenarios,
    )


def load_map_catalog(manifest_path: Path) -> MapCatalog:
    """Load one manifest.json into a MapCatalog.

    Reads the manifest, preserves the declared order of maps and scenarios,
    defensively copies every structure into frozen dataclasses, and
    confirms that every referenced .sim file exists next to the manifest.
    Never writes anything and never mutates the input file.
    """

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    entries = tuple(
        _build_map_entry(dict(m), manifest_dir=manifest_path.parent) for m in raw.get("maps", [])
    )

    return MapCatalog(
        corpus_id=str(raw["corpus_id"]),
        schema_version=int(raw["schema_version"]),
        entries=entries,
    )
