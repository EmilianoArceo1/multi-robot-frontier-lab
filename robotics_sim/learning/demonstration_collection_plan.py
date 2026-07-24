"""Per-map human-demonstration collection plan, and a pure collector/map/
episode selection cursor over it.

A DemonstrationCollectionPlan only organizes work: which map each
collector must produce episodes for, and how many, with which scenario and
seed. It never contains a decision (no route, no target, no candidate) --
those belong to a human, captured elsewhere. It never touches the
filesystem beyond the one load_demonstration_collection_plan() read, and it
never mutates the MapCatalog it was validated against.

Allowed dependency direction: robotics_sim.learning -> stdlib
(json/pathlib/dataclasses) + robotics_sim.learning.map_catalog only. No Qt,
robotics_sim.app, robotics_sim.simulation, robotics_interfaces or engine
imports.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from robotics_sim.learning.map_catalog import MapCatalog


class DemonstrationCollectionPlanError(ValueError):
    """The plan file is structurally invalid or inconsistent with the
    MapCatalog it was loaded against."""


class DemonstrationCollectionSetupError(ValueError):
    """An invalid collector/map/episode selection was attempted."""


@dataclass(frozen=True)
class PlannedDemonstrationEpisode:
    """One planned episode slot within one map: which scenario, which seed,
    and its local (per-map) episode_number."""

    episode_number: int
    scenario_id: str
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.episode_number, bool) or not isinstance(self.episode_number, int):
            raise TypeError(
                f"episode_number must be an int, got {type(self.episode_number).__name__}"
            )
        if self.episode_number < 1:
            raise ValueError(f"episode_number must start at 1, got {self.episode_number}")
        if not isinstance(self.scenario_id, str) or not self.scenario_id.strip():
            raise ValueError(f"scenario_id must be a non-empty string, got {self.scenario_id!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(f"seed must be an int, got {type(self.seed).__name__}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


@dataclass(frozen=True)
class MapCollectionAssignment:
    """One map's collector plus its ordered list of planned episodes.

    episode_number values must be exactly {1, ..., len(episodes)} -- unique
    and gapless -- so "Episode X of N" progress reporting is well-defined.
    Declaration order is preserved in ``episodes``; it is never sorted.
    """

    map_id: str
    collector_id: str
    episodes: tuple[PlannedDemonstrationEpisode, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.map_id, str) or not self.map_id.strip():
            raise ValueError(f"map_id must be a non-empty string, got {self.map_id!r}")
        if not isinstance(self.collector_id, str) or not self.collector_id.strip():
            raise ValueError(f"collector_id must be a non-empty string, got {self.collector_id!r}")

        episodes = tuple(self.episodes)
        for i, episode in enumerate(episodes):
            if not isinstance(episode, PlannedDemonstrationEpisode):
                raise TypeError(
                    f"episodes[{i}] must be a PlannedDemonstrationEpisode, got "
                    f"{type(episode).__name__}"
                )

        numbers = [e.episode_number for e in episodes]
        if len(set(numbers)) != len(numbers):
            raise DemonstrationCollectionPlanError(
                f"map {self.map_id!r} has duplicate episode_number values: {numbers}"
            )
        if set(numbers) != set(range(1, len(episodes) + 1)):
            raise DemonstrationCollectionPlanError(
                f"map {self.map_id!r} episode_number values must be exactly "
                f"1..{len(episodes)}, got {sorted(numbers)}"
            )

        combos = [(e.scenario_id, e.seed) for e in episodes]
        if len(set(combos)) != len(combos):
            raise DemonstrationCollectionPlanError(
                f"map {self.map_id!r} has a repeated (scenario_id, seed) combination: {combos}"
            )

        object.__setattr__(self, "episodes", episodes)


@dataclass(frozen=True)
class DemonstrationCollectionPlan:
    """A whole collection plan: one MapCollectionAssignment per map, at
    most, keyed implicitly by map_id (no map_id appears twice)."""

    plan_id: str
    corpus_id: str
    assignments: tuple[MapCollectionAssignment, ...]
    _by_map: Mapping[str, MapCollectionAssignment] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _collector_ids: tuple[str, ...] = field(
        default_factory=tuple, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError(f"plan_id must be a non-empty string, got {self.plan_id!r}")
        if not isinstance(self.corpus_id, str) or not self.corpus_id.strip():
            raise ValueError(f"corpus_id must be a non-empty string, got {self.corpus_id!r}")

        assignments = tuple(self.assignments)
        by_map: dict[str, MapCollectionAssignment] = {}
        collector_ids: list[str] = []
        for i, assignment in enumerate(assignments):
            if not isinstance(assignment, MapCollectionAssignment):
                raise TypeError(
                    f"assignments[{i}] must be a MapCollectionAssignment, got "
                    f"{type(assignment).__name__}"
                )
            if assignment.map_id in by_map:
                raise DemonstrationCollectionPlanError(
                    f"map_id {assignment.map_id!r} is assigned more than once"
                )
            by_map[assignment.map_id] = assignment
            if assignment.collector_id not in collector_ids:
                collector_ids.append(assignment.collector_id)

        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "_by_map", by_map)
        object.__setattr__(self, "_collector_ids", tuple(collector_ids))

    @property
    def collector_ids(self) -> tuple[str, ...]:
        return self._collector_ids

    def maps_for_collector(self, collector_id: str) -> tuple[str, ...]:
        return tuple(a.map_id for a in self.assignments if a.collector_id == collector_id)

    def assignment_for_map(self, map_id: str) -> MapCollectionAssignment:
        try:
            return self._by_map[map_id]
        except KeyError:
            raise DemonstrationCollectionPlanError(f"unknown map_id {map_id!r}") from None

    def episodes_for_map(self, map_id: str) -> tuple[PlannedDemonstrationEpisode, ...]:
        return self.assignment_for_map(map_id).episodes

    def episode_for_map(self, map_id: str, episode_number: int) -> PlannedDemonstrationEpisode:
        for episode in self.episodes_for_map(map_id):
            if episode.episode_number == episode_number:
                return episode
        raise DemonstrationCollectionPlanError(
            f"map {map_id!r} has no episode_number {episode_number}"
        )

    def total_episodes_for_map(self, map_id: str) -> int:
        return len(self.episodes_for_map(map_id))


def _build_episode(raw: Mapping[str, object]) -> PlannedDemonstrationEpisode:
    return PlannedDemonstrationEpisode(
        episode_number=int(raw["episode_number"]),
        scenario_id=str(raw["scenario_id"]),
        seed=int(raw["seed"]),
    )


def _build_assignment(raw: Mapping[str, object], *, map_catalog: MapCatalog) -> MapCollectionAssignment:
    map_id = str(raw["map_id"])
    if map_id not in map_catalog.map_ids:
        raise DemonstrationCollectionPlanError(
            f"plan references unknown map_id {map_id!r} (not in the map catalog)"
        )
    episodes = tuple(_build_episode(dict(e)) for e in raw.get("episodes", []))
    for episode in episodes:
        map_catalog.get_scenario(map_id, episode.scenario_id)  # raises if missing
    return MapCollectionAssignment(
        map_id=map_id, collector_id=str(raw["collector_id"]), episodes=episodes
    )


def load_demonstration_collection_plan(
    plan_path: Path,
    *,
    map_catalog: MapCatalog,
) -> DemonstrationCollectionPlan:
    """Load one collection-plan JSON file, cross-validated against
    ``map_catalog``. Never writes anything."""

    plan_path = Path(plan_path)
    with plan_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    assignments = tuple(
        _build_assignment(dict(a), map_catalog=map_catalog) for a in raw.get("assignments", [])
    )

    return DemonstrationCollectionPlan(
        plan_id=str(raw["plan_id"]), corpus_id=str(raw["corpus_id"]), assignments=assignments
    )


EpisodeKey = tuple[str, int]
"""A (map_id, episode_number) pair identifying one planned episode slot,
independent of attempt_number -- an episode slot counts as "completed" as
soon as any attempt for it has been accepted."""


@dataclass(frozen=True)
class DemonstrationCollectionSetup:
    """Pure collector/map/episode selection cursor over one
    DemonstrationCollectionPlan.

    Every ``select_*`` method returns a *new* DemonstrationCollectionSetup;
    the plan itself and any prior setup instance are never mutated. Never
    touches the filesystem: ``completed_episode_keys`` is always supplied
    explicitly by the caller, never inferred from disk.
    """

    collection_plan: DemonstrationCollectionPlan
    collector_id: str | None = None
    selected_map_id: str | None = None
    selected_episode_number: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.collection_plan, DemonstrationCollectionPlan):
            raise TypeError(
                f"collection_plan must be a DemonstrationCollectionPlan, got "
                f"{type(self.collection_plan).__name__}"
            )

        if self.collector_id is not None:
            if self.collector_id not in self.collection_plan.collector_ids:
                raise DemonstrationCollectionSetupError(
                    f"collector_id {self.collector_id!r} is not in this plan"
                )
        elif self.selected_map_id is not None or self.selected_episode_number is not None:
            raise DemonstrationCollectionSetupError(
                "selected_map_id/selected_episode_number require a collector_id"
            )

        if self.selected_map_id is not None:
            allowed_maps = self.collection_plan.maps_for_collector(self.collector_id)
            if self.selected_map_id not in allowed_maps:
                raise DemonstrationCollectionSetupError(
                    f"map_id {self.selected_map_id!r} is not assigned to collector "
                    f"{self.collector_id!r}"
                )
        elif self.selected_episode_number is not None:
            raise DemonstrationCollectionSetupError(
                "selected_episode_number requires a selected_map_id"
            )

        if self.selected_episode_number is not None:
            # Raises DemonstrationCollectionPlanError if the episode does not
            # exist for this map -- never guessed, never silently clamped.
            self.collection_plan.episode_for_map(self.selected_map_id, self.selected_episode_number)

    @property
    def available_map_ids(self) -> tuple[str, ...]:
        if self.collector_id is None:
            return ()
        return self.collection_plan.maps_for_collector(self.collector_id)

    def select_collector(self, collector_id: str) -> "DemonstrationCollectionSetup":
        # Changing collector always clears map/episode: a map assigned to
        # the previous collector may not even be visible to the new one.
        return dataclasses.replace(
            self, collector_id=collector_id, selected_map_id=None, selected_episode_number=None
        )

    def select_map(self, map_id: str) -> "DemonstrationCollectionSetup":
        if self.collector_id is None:
            raise DemonstrationCollectionSetupError("select_collector() must be called first")
        return dataclasses.replace(self, selected_map_id=map_id, selected_episode_number=None)

    def select_episode(self, episode_number: int) -> "DemonstrationCollectionSetup":
        if self.selected_map_id is None:
            raise DemonstrationCollectionSetupError("select_map() must be called first")
        return dataclasses.replace(self, selected_episode_number=episode_number)

    def select_next_unrecorded_episode(
        self, recorded_episode_keys: Iterable[EpisodeKey]
    ) -> "DemonstrationCollectionSetup":
        """Select the first planned episode of the current map that has no
        attempt yet in pending_review or accepted.

        ``recorded_episode_keys`` is the caller-supplied set of (map_id,
        episode_number) pairs with at least one recorded (pending_review or
        accepted) attempt -- rejected-only episodes must not appear in it.
        Keys for other maps are simply ignored (episode_number stays local
        to each map).
        """

        if self.selected_map_id is None:
            raise DemonstrationCollectionSetupError("select_map() must be called first")
        recorded = frozenset(recorded_episode_keys)
        for episode in self.collection_plan.episodes_for_map(self.selected_map_id):
            key = (self.selected_map_id, episode.episode_number)
            if key not in recorded:
                return self.select_episode(episode.episode_number)
        raise DemonstrationCollectionSetupError(
            f"map {self.selected_map_id!r} has no unrecorded episode left"
        )

    @property
    def current_episode_position_text(self) -> str:
        if self.selected_map_id is None or self.selected_episode_number is None:
            raise DemonstrationCollectionSetupError("no episode is currently selected")
        total = self.collection_plan.total_episodes_for_map(self.selected_map_id)
        return f"Episode {self.selected_episode_number} of {total}"

    def _progress_count(self, episode_keys: Iterable[EpisodeKey]) -> tuple[int, int]:
        total = self.collection_plan.total_episodes_for_map(self.selected_map_id)
        keys = frozenset(episode_keys)
        done = sum(
            1
            for episode in self.collection_plan.episodes_for_map(self.selected_map_id)
            if (self.selected_map_id, episode.episode_number) in keys
        )
        return done, total

    def recorded_progress_text(self, recorded_episode_keys: Iterable[EpisodeKey]) -> str:
        """"Recorded Y of N": Y is how many of this map's planned episode
        slots have at least one attempt saved in pending_review or accepted
        (a rejected-only slot, or a second attempt of an already-recorded
        slot, never changes Y). Keys belonging to other maps are ignored."""

        if self.selected_map_id is None:
            raise DemonstrationCollectionSetupError("no map is currently selected")
        done, total = self._progress_count(recorded_episode_keys)
        return f"Recorded {done} of {total}"

    def accepted_progress_text(self, accepted_episode_keys: Iterable[EpisodeKey]) -> str:
        """"Accepted Y of N": Y is how many of this map's planned episode
        slots have at least one accepted attempt. Keys belonging to other
        maps are ignored."""

        if self.selected_map_id is None:
            raise DemonstrationCollectionSetupError("no map is currently selected")
        done, total = self._progress_count(accepted_episode_keys)
        return f"Accepted {done} of {total}"
