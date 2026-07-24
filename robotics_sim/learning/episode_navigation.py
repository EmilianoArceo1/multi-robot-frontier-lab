"""Pure, in-memory navigation over a list of independent demonstration
episodes.

Every episode is a self-contained folder (see demonstration_episode.py); this
module never reads the filesystem itself -- it only organizes
EpisodeListEntry values a caller already collected (e.g. by scanning
pending_review/accepted/rejected once, elsewhere). No global index file is
required to make sense of this list.

Allowed dependency direction: robotics_sim.learning -> stdlib
(dataclasses/pathlib) + robotics_sim.learning.demonstration_episode
(DemonstrationEpisodeIdentity, DemonstrationEpisodeStorageState). No Qt,
robotics_sim.app, robotics_sim.simulation, robotics_interfaces or engine
imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from robotics_sim.learning.demonstration_episode import (
    DemonstrationEpisodeIdentity,
    DemonstrationEpisodeStorageState,
)


class EpisodeNavigationError(ValueError):
    """An invalid entry list or navigation target was supplied."""


@dataclass(frozen=True)
class EpisodeListEntry:
    """One independently-stored episode, as seen by the navigator: its
    identity, current storage state, and a few cheap-to-carry summary
    fields. Never carries decisions, metrics, or any live object."""

    identity: DemonstrationEpisodeIdentity
    status: DemonstrationEpisodeStorageState
    episode_directory: Path
    decision_count: int
    valid_integrity_report: bool

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DemonstrationEpisodeIdentity):
            raise TypeError(
                f"identity must be a DemonstrationEpisodeIdentity, got {type(self.identity).__name__}"
            )
        if not isinstance(self.status, DemonstrationEpisodeStorageState):
            raise TypeError(
                f"status must be a DemonstrationEpisodeStorageState, got {type(self.status).__name__}"
            )
        object.__setattr__(self, "episode_directory", Path(self.episode_directory))
        if isinstance(self.decision_count, bool) or not isinstance(self.decision_count, int):
            raise TypeError(
                f"decision_count must be an int, got {type(self.decision_count).__name__}"
            )
        if self.decision_count < 0:
            raise ValueError(f"decision_count must be non-negative, got {self.decision_count}")
        if not isinstance(self.valid_integrity_report, bool):
            raise TypeError(
                f"valid_integrity_report must be a bool, got "
                f"{type(self.valid_integrity_report).__name__}"
            )


def _sort_key(entry: EpisodeListEntry):
    identity = entry.identity
    return (identity.map_id, identity.episode_number, identity.attempt_number, identity.created_at_utc)


class EpisodeNavigationModel:
    """A read-only, ordered view over a list of EpisodeListEntry, with a
    movable cursor.

    Ordering is always (map_id, episode_number, attempt_number,
    created_at_utc); filtering never mutates ``self`` -- it returns a new
    EpisodeNavigationModel over a subset of the original entries. The list
    passed to the constructor is never mutated, and an empty list is a
    valid, fully-functional model with no current_entry.
    """

    def __init__(self, entries: Iterable[EpisodeListEntry]) -> None:
        entries = tuple(entries)
        seen_episode_ids: set[str] = set()
        seen_slots: set[tuple[str, int, int]] = set()
        for i, entry in enumerate(entries):
            if not isinstance(entry, EpisodeListEntry):
                raise TypeError(f"entries[{i}] must be an EpisodeListEntry, got {type(entry).__name__}")
            episode_id = entry.identity.episode_id
            if episode_id in seen_episode_ids:
                raise EpisodeNavigationError(f"duplicate episode_id {episode_id!r}")
            seen_episode_ids.add(episode_id)

            slot = (entry.identity.map_id, entry.identity.episode_number, entry.identity.attempt_number)
            if slot in seen_slots:
                raise EpisodeNavigationError(
                    f"duplicate (map_id, episode_number, attempt_number) {slot!r}"
                )
            seen_slots.add(slot)

        self._entries: tuple[EpisodeListEntry, ...] = tuple(sorted(entries, key=_sort_key))
        self._position: int | None = 0 if self._entries else None

    @property
    def entries(self) -> tuple[EpisodeListEntry, ...]:
        return self._entries

    def filter_by_collector(self, collector_id: str) -> "EpisodeNavigationModel":
        return EpisodeNavigationModel(
            e for e in self._entries if e.identity.collector_id == collector_id
        )

    def filter_by_map(self, map_id: str) -> "EpisodeNavigationModel":
        return EpisodeNavigationModel(e for e in self._entries if e.identity.map_id == map_id)

    def filter_by_status(self, status: DemonstrationEpisodeStorageState) -> "EpisodeNavigationModel":
        return EpisodeNavigationModel(e for e in self._entries if e.status is status)

    @property
    def current_entry(self) -> EpisodeListEntry | None:
        if self._position is None:
            return None
        return self._entries[self._position]

    def previous(self) -> EpisodeListEntry | None:
        if self._position is None:
            return None
        self._position = max(0, self._position - 1)
        return self.current_entry

    def next(self) -> EpisodeListEntry | None:
        if self._position is None:
            return None
        self._position = min(len(self._entries) - 1, self._position + 1)
        return self.current_entry

    def go_to_episode_id(self, episode_id: str) -> EpisodeListEntry:
        for i, entry in enumerate(self._entries):
            if entry.identity.episode_id == episode_id:
                self._position = i
                return entry
        raise EpisodeNavigationError(f"no entry with episode_id {episode_id!r}")

    def go_to_planned_episode(
        self, map_id: str, episode_number: int, attempt_number: int | None = None
    ) -> EpisodeListEntry:
        matches = [
            (i, entry)
            for i, entry in enumerate(self._entries)
            if entry.identity.map_id == map_id
            and entry.identity.episode_number == episode_number
            and (attempt_number is None or entry.identity.attempt_number == attempt_number)
        ]
        if not matches:
            raise EpisodeNavigationError(
                f"no entry for map_id={map_id!r}, episode_number={episode_number}, "
                f"attempt_number={attempt_number!r}"
            )
        # attempt_number=None resolves to the latest attempt, deterministically.
        i, entry = max(matches, key=lambda pair: pair[1].identity.attempt_number)
        self._position = i
        return entry

    @property
    def current_position_text(self) -> str:
        if not self._entries:
            return "Episode 0 of 0"
        return f"Episode {self._position + 1} of {len(self._entries)}"

    def _map_progress(self, map_id: str) -> tuple[int, int]:
        """(completed, total) for one map, counted over distinct
        episode_number values -- never over entries/attempts.

        Two or more EpisodeListEntry with the same episode_number but
        different attempt_number (a rejected attempt followed by a retry,
        for instance) always collapse into the same set element here, so
        adding another attempt of an already-seen episode_number can never
        inflate ``total``, and an already-accepted episode_number can never
        be double-counted in ``completed`` either.
        """

        entries_for_map = [e for e in self._entries if e.identity.map_id == map_id]
        total_numbers = {e.identity.episode_number for e in entries_for_map}
        completed_numbers = {
            e.identity.episode_number
            for e in entries_for_map
            if e.status is DemonstrationEpisodeStorageState.ACCEPTED
        }
        return len(completed_numbers), len(total_numbers)

    @property
    def current_map_progress_text(self) -> str:
        entry = self.current_entry
        if entry is None:
            raise EpisodeNavigationError("no current entry")
        completed, total = self._map_progress(entry.identity.map_id)
        return f"Map {entry.identity.map_id}: completed {completed} of {total}"
