"""Host-side orchestration for one human-demonstration recording session.

This module is the *only* place where "Human Demonstration" domain logic
lives: MainWindow and engine.py only wire small hooks/callbacks into it (see
their own docstrings/comments at the call sites). It never imports Qt,
robotics_sim.app, or any GUI toolkit -- every host action (loading a .sim
file, adding fire, pausing, reading the simulation clock/metrics, blocking
until the human resumes) is received as a plain injected callable via
:class:`HumanDemonstrationHostBindings`.

Ownership boundaries (nothing here is duplicated):
- MapCatalog / DemonstrationCollectionPlan / DemonstrationCollectionSetup
  (robotics_sim.learning) already own map/plan/selection-cursor logic; this
  module only sequences calls into them.
- ManualDemonstrationSelectionSession (robotics_sim.learning) already owns
  one coordination round's candidate freezing, robot/candidate selection,
  and CoordinationResult construction; this module only creates one session
  per round and reads its outputs.
- EpisodeDecisionStepAllocator (robotics_sim.learning) already owns the
  one-episode-global decision_step counter; this module never counts steps
  itself.
- DemonstrationEpisodeWriter (robotics_sim.learning) already owns the
  atomic, per-episode filesystem write; this module never writes files
  itself beyond calling it once per Finish.

The request_executor seam
==========================

``robotics_sim.simulation.coordination.MultiRobotCoordinator.assign_frontiers()``
already accepts an optional
``request_executor: Callable[[CoordinationRequest], CoordinationResult]``
that, when supplied, is called *instead of* ``plugin.assign(request)``. This
module's :meth:`HumanDemonstrationRuntime.request_executor` is exactly that
callable. It is invoked synchronously, inside the same call stack as the
simulator's per-tick coordination call
(``engine.py``'s ``synchronize_multi_frontier_targets``) -- so making the
human's asynchronous, click-driven selection fit that synchronous contract
requires a genuine blocking wait: ``request_executor`` asks the host to
pause (``request_pause()``) and then blocks on ``wait_for_human_resume()``
until the host reports that the human has finished (typically implemented
host-side with a nested Qt event loop -- see main_window.py). While
blocked, ordinary Qt event processing continues, so mouse clicks routed
into :meth:`select_robot`/:meth:`select_candidate` still update the open
session. This module does not know or care *how* the host blocks; it only
needs the call to return once the human is done.

No second coordination pipeline is created: after resume, this module
returns the *real* ``robotics_interfaces.coordination.CoordinationResult``
produced by ``ManualDemonstrationSelectionSession.build_manual_coordination_
result()`` straight back to ``assign_frontiers()``, which continues through
its existing adaptation/target-write-back path exactly as it does for any
plugin. Routes are still computed by whichever path planner the host
already uses for the underlying coordinator strategy -- this module never
computes one.
"""

from __future__ import annotations

import enum
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from robotics_interfaces.commands import RobotCommand
from robotics_interfaces.coordination import CoordinationAssignment, CoordinationRequest, CoordinationResult
from robotics_interfaces.learning import CandidateKind
from robotics_interfaces.proposals import CandidateProposal, ExplorationCandidate
from robotics_sim.learning.capture_inputs import CandidateCaptureInput
from robotics_sim.learning.demonstration_collection_plan import (
    DemonstrationCollectionPlan,
    DemonstrationCollectionSetup,
    DemonstrationCollectionSetupError,
    EpisodeKey,
    load_demonstration_collection_plan,
)
from robotics_sim.learning.demonstration_episode import (
    DemonstrationDecisionRecord,
    DemonstrationEpisodeIdentity,
    DemonstrationEpisodeRecord,
)
from robotics_sim.learning.demonstration_episode_writer import (
    DemonstrationEpisodeLayout,
    DemonstrationEpisodeWriter,
    DemonstrationIntegrityReport,
)
from robotics_sim.learning.decision_steps import EpisodeDecisionStepAllocator
from robotics_sim.learning.manual_demonstration import (
    ManualDemonstrationSelectionSession,
    ManualDemonstrationSessionState,
)
from robotics_sim.learning.map_catalog import MapCatalog, load_map_catalog

DEFAULT_MANIFEST_PATH = Path("experiments/maps/smoke_v0/manifest.json")
DEFAULT_PLAN_PATH = Path("experiments/collection_plans/human_demo_smoke_v0.json")
DEFAULT_OUTPUT_ROOT = Path("experiments/datasets/human_demonstrations_v0")

# The coordinator-selector sentinel label (see config_panel.py/main_window.py
# wiring). Deliberately NOT a real plugin name: selecting it must never be
# passed to MultiRobotCoordinator/load_coordination_plugin (which would try
# to import algorithms/<name>/plugin.py and fail). It only flips a host-side
# mode flag; SimulationConfig.coordinator_type keeps pointing at whatever
# real HOST_CANDIDATES-compatible plugin is generating candidates.
HUMAN_DEMONSTRATION_COORDINATOR_LABEL = "Human Demonstration (manual)"

_DEFAULT_FIRE_DETECTION_THRESHOLD = 0.5


class HumanDemonstrationRuntimeError(RuntimeError):
    """Base class for HumanDemonstrationRuntime errors."""


class HumanDemonstrationRuntimeStateError(HumanDemonstrationRuntimeError):
    """Operation not valid in the runtime's current lifecycle state."""


class HumanDemonstrationRuntimeState(enum.Enum):
    SETUP = "setup"
    RECORDING = "recording"
    WAITING_FOR_SELECTION = "waiting_for_selection"


@dataclass(frozen=True)
class HumanDemonstrationHostBindings:
    """Every host (engine/GUI) action this runtime needs, as plain
    callables -- the only way this module ever touches the running
    simulator, and the reason it never needs to import Qt.

    ``load_sim_file``: load a .sim file via the real simulator API (the
    same path ``MainWindow.load_simulation_config`` uses).
    ``clear_fires`` / ``add_fire``: the real ground-truth hazard field API
    (never the ``obstacles`` collection).
    ``reset_hazard_belief``: reset discovered-hazard belief state via the
    real path so a new episode starts undiscovered.
    ``request_pause``: ask the host to pause (sets the real ``paused``
    flag through its own toggle machinery).
    ``wait_for_human_resume``: blocks (host-implemented) until the human
    has finished this round -- see module docstring.
    ``get_simulation_time_s``: read the live, real simulation clock.
    ``get_final_metrics``: read real, already-contractual engine metrics.
    """

    load_sim_file: Callable[[Path], None]
    clear_fires: Callable[[], None]
    add_fire: Callable[[float, float], None]
    reset_hazard_belief: Callable[[], None]
    request_pause: Callable[[], None]
    wait_for_human_resume: Callable[[], None]
    get_simulation_time_s: Callable[[], float]
    get_final_metrics: Callable[[], Mapping[str, float]]


def _normalize_candidate(value: object) -> ExplorationCandidate:
    if isinstance(value, ExplorationCandidate):
        return value
    if isinstance(value, CandidateProposal):
        return value.as_candidate(source="explicit_proposal")
    raise HumanDemonstrationRuntimeError(
        f"proposals_by_robot entry is neither ExplorationCandidate nor CandidateProposal, "
        f"got {type(value).__name__}"
    )


def capture_candidate_pool(
    request: CoordinationRequest, robot_ids: tuple[int, ...]
) -> dict[int, tuple[CandidateCaptureInput, ...]]:
    """Obtain the exact candidate pool for one manual decision round.

    Mirrors the same three-tier priority
    ``robotics_sim.learning.coordination_decision_source.
    LearningCoordinationDecisionSource._obtain_pool()`` already implements
    for plugin-driven decisions (explicit ``proposals_by_robot`` -> team
    frontier provider -> per-robot frontier provider -> empty pool). It is
    duplicated here, not imported, only because that method is private and
    inseparably tied to wrapping a plugin's ``assign()`` call -- which this
    module must never invoke. No second candidate type is introduced: every
    candidate stays a real ``ExplorationCandidate``, wrapped in the real
    ``CandidateCaptureInput``.
    """

    if robot_ids and all(request.proposals_by_robot.get(rid) for rid in robot_ids):
        raw_by_robot = {
            rid: tuple(_normalize_candidate(item) for item in request.proposals_by_robot[rid])
            for rid in robot_ids
        }
    else:
        services = request.services
        if services is not None and services.team_frontier_provider is not None:
            raw = services.team_frontier_provider.candidates_for_team(request)
            raw_by_robot = {rid: tuple(raw.get(rid, ())) for rid in robot_ids}
        elif services is not None and services.frontier_provider is not None and request.world is not None:
            robots_by_id = {robot.robot_id: robot for robot in request.robot_states}
            raw_by_robot = {}
            for rid in robot_ids:
                robot = robots_by_id.get(rid)
                if robot is None:
                    raw_by_robot[rid] = ()
                    continue
                blocked = tuple(request.blocked_targets_by_robot.get(rid, ()))
                raw_by_robot[rid] = tuple(
                    services.frontier_provider.candidates_for_robot(
                        robot=robot, world=request.world, blocked_targets=blocked
                    )
                )
        else:
            raw_by_robot = {rid: () for rid in robot_ids}

    pool: dict[int, tuple[CandidateCaptureInput, ...]] = {}
    for rid in robot_ids:
        pool[rid] = tuple(
            CandidateCaptureInput(
                candidate=candidate, kind=CandidateKind.FRONTIER_VIEWPOINT, enabled=True, reachable=True
            )
            for candidate in raw_by_robot.get(rid, ())
        )
    return pool


def _scan_metadata_field(directory: Path, field: str) -> list[tuple[Path, dict]]:
    results: list[tuple[Path, dict]] = []
    if not directory.is_dir():
        return results
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        metadata_path = child / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if field in metadata:
            results.append((child, metadata))
    return results


class HumanDemonstrationRuntime:
    """Owns exactly one active human-demonstration episode at a time.

    Holds no filesystem-global state: every ``recorded``/``accepted`` query
    re-scans ``output_root`` explicitly (no cached, potentially-stale
    index), and every written episode is its own independent folder via
    :class:`DemonstrationEpisodeWriter` -- no dataset.npz, no runs.jsonl, no
    global counter.
    """

    def __init__(
        self,
        *,
        map_catalog: MapCatalog,
        collection_plan: DemonstrationCollectionPlan,
        sim_directory: Path,
        output_root: Path,
        host: HumanDemonstrationHostBindings,
        writer: DemonstrationEpisodeWriter | None = None,
        fire_detection_threshold: float = _DEFAULT_FIRE_DETECTION_THRESHOLD,
    ) -> None:
        if not isinstance(map_catalog, MapCatalog):
            raise TypeError(f"map_catalog must be a MapCatalog, got {type(map_catalog).__name__}")
        if not isinstance(collection_plan, DemonstrationCollectionPlan):
            raise TypeError(
                f"collection_plan must be a DemonstrationCollectionPlan, got "
                f"{type(collection_plan).__name__}"
            )
        if not isinstance(host, HumanDemonstrationHostBindings):
            raise TypeError(
                f"host must be a HumanDemonstrationHostBindings, got {type(host).__name__}"
            )

        self._map_catalog = map_catalog
        self._collection_plan = collection_plan
        self._sim_directory = Path(sim_directory)
        self._output_root = Path(output_root)
        self._host = host
        self._writer = writer or DemonstrationEpisodeWriter()
        self._fire_detection_threshold = float(fire_detection_threshold)

        self._setup = DemonstrationCollectionSetup(collection_plan=collection_plan)
        self._state = HumanDemonstrationRuntimeState.SETUP

        self._active_identity: DemonstrationEpisodeIdentity | None = None
        self._started_at_utc: datetime | None = None
        self._decisions: list[DemonstrationDecisionRecord] = []
        self._fires_loaded_count = 0
        self._step_allocator: EpisodeDecisionStepAllocator | None = None
        self._active_session: ManualDemonstrationSelectionSession | None = None
        self._pending_result: CoordinationResult | None = None

    # -- read-only state -----------------------------------------------

    @property
    def state(self) -> HumanDemonstrationRuntimeState:
        return self._state

    @property
    def setup(self) -> DemonstrationCollectionSetup:
        return self._setup

    @property
    def has_active_episode(self) -> bool:
        return self._active_identity is not None

    @property
    def active_identity(self) -> DemonstrationEpisodeIdentity | None:
        return self._active_identity

    @property
    def fires_loaded_count(self) -> int:
        return self._fires_loaded_count

    @property
    def active_session(self) -> ManualDemonstrationSelectionSession | None:
        return self._active_session

    @property
    def decision_count(self) -> int:
        return len(self._decisions)

    # -- collector / map / episode selection ----------------------------

    def _ensure_no_active_episode(self) -> None:
        if self._active_identity is not None:
            raise HumanDemonstrationRuntimeStateError(
                "cannot change collector/map/episode while an episode is active; "
                "Finish or Abort it first"
            )

    def select_collector(self, collector_id: str) -> None:
        self._ensure_no_active_episode()
        self._setup = self._setup.select_collector(collector_id)

    def select_map(self, map_id: str) -> None:
        self._ensure_no_active_episode()
        self._setup = self._setup.select_map(map_id)

    def select_episode(self, episode_number: int) -> None:
        self._ensure_no_active_episode()
        self._setup = self._setup.select_episode(episode_number)

    def select_next_unrecorded_episode(self) -> None:
        self._ensure_no_active_episode()
        self._setup = self._setup.select_next_unrecorded_episode(self.recorded_episode_keys())

    def previous_episode(self) -> None:
        self._ensure_no_active_episode()
        if self._setup.selected_map_id is None or self._setup.selected_episode_number is None:
            raise HumanDemonstrationRuntimeStateError("no episode is currently selected")
        new_number = self._setup.selected_episode_number - 1
        if new_number < 1:
            raise HumanDemonstrationRuntimeStateError("already at the first planned episode")
        self._setup = self._setup.select_episode(new_number)

    def next_episode(self) -> None:
        self._ensure_no_active_episode()
        if self._setup.selected_map_id is None or self._setup.selected_episode_number is None:
            raise HumanDemonstrationRuntimeStateError("no episode is currently selected")
        total = self._collection_plan.total_episodes_for_map(self._setup.selected_map_id)
        new_number = self._setup.selected_episode_number + 1
        if new_number > total:
            raise HumanDemonstrationRuntimeStateError("already at the last planned episode")
        self._setup = self._setup.select_episode(new_number)

    # -- recorded / accepted tracking (explicit filesystem scan) --------

    def recorded_episode_keys(self) -> frozenset[EpisodeKey]:
        keys: set[EpisodeKey] = set()
        for storage_dirname in ("pending_review", "accepted"):
            for _, metadata in _scan_metadata_field(self._output_root / storage_dirname, "map_id"):
                try:
                    keys.add((str(metadata["map_id"]), int(metadata["episode_number"])))
                except (KeyError, ValueError, TypeError):
                    continue
        return frozenset(keys)

    def accepted_episode_keys(self) -> frozenset[EpisodeKey]:
        keys: set[EpisodeKey] = set()
        for _, metadata in _scan_metadata_field(self._output_root / "accepted", "map_id"):
            try:
                keys.add((str(metadata["map_id"]), int(metadata["episode_number"])))
            except (KeyError, ValueError, TypeError):
                continue
        return frozenset(keys)

    def _next_attempt_number(self, map_id: str, episode_number: int) -> int:
        max_attempt = 0
        for storage_dirname in ("pending_review", "accepted", "rejected"):
            for _, metadata in _scan_metadata_field(self._output_root / storage_dirname, "map_id"):
                try:
                    if str(metadata["map_id"]) == map_id and int(metadata["episode_number"]) == episode_number:
                        max_attempt = max(max_attempt, int(metadata["attempt_number"]))
                except (KeyError, ValueError, TypeError):
                    continue
        return max_attempt + 1

    # -- display text -----------------------------------------------------

    def episode_position_text(self) -> str:
        return self._setup.current_episode_position_text

    def recorded_progress_text(self) -> str:
        return self._setup.recorded_progress_text(self.recorded_episode_keys())

    def accepted_progress_text(self) -> str:
        return self._setup.accepted_progress_text(self.accepted_episode_keys())

    def map_complete_text(self) -> str | None:
        if self._setup.selected_map_id is None:
            return None
        map_id = self._setup.selected_map_id
        recorded = self.recorded_episode_keys()
        total = self._collection_plan.total_episodes_for_map(map_id)
        done = sum(
            1
            for episode in self._collection_plan.episodes_for_map(map_id)
            if (map_id, episode.episode_number) in recorded
        )
        if done >= total:
            return f"Map complete: Recorded {done} of {total}"
        return None

    # -- Load Episode -----------------------------------------------------

    def load_episode(self) -> DemonstrationEpisodeIdentity:
        if self._active_identity is not None:
            raise HumanDemonstrationRuntimeStateError(
                "an episode is already active; Finish or Abort it first"
            )
        if self._setup.selected_map_id is None or self._setup.selected_episode_number is None:
            raise HumanDemonstrationRuntimeStateError(
                "collector, map, and episode must all be selected before Load Episode"
            )

        map_id = self._setup.selected_map_id
        map_entry = self._map_catalog.get_map(map_id)
        planned_episode = self._collection_plan.episode_for_map(
            map_id, self._setup.selected_episode_number
        )
        scenario = self._map_catalog.get_scenario(map_id, planned_episode.scenario_id)

        # 1-2. Resolve + load the .sim file via the real simulator API.
        self._host.load_sim_file(self._sim_directory / map_entry.filename)

        # 3-4. Clear the previous episode's fires, then inject exactly this
        # scenario's ground-truth fire positions via the real add_fire API
        # -- never the obstacles collection.
        self._host.clear_fires()
        for x, y in scenario.fire_positions:
            self._host.add_fire(x, y)

        # 6. Reset belief/hazard discovery state via the real path so
        # nothing from a previous episode leaks in as already-discovered.
        self._host.reset_hazard_belief()

        attempt_number = self._next_attempt_number(map_id, planned_episode.episode_number)
        identity = DemonstrationEpisodeIdentity(
            episode_id=str(uuid.uuid4()),
            plan_id=self._collection_plan.plan_id,
            episode_number=planned_episode.episode_number,
            attempt_number=attempt_number,
            collector_id=self._setup.collector_id,
            corpus_id=self._collection_plan.corpus_id,
            map_id=map_id,
            scenario_id=planned_episode.scenario_id,
            seed=planned_episode.seed,
            created_at_utc=datetime.now(timezone.utc),
        )

        self._step_allocator = EpisodeDecisionStepAllocator()
        self._step_allocator.start_episode(0)
        self._active_identity = identity
        self._started_at_utc = identity.created_at_utc
        self._decisions = []
        self._fires_loaded_count = len(scenario.fire_positions)
        self._active_session = None
        self._pending_result = None
        self._state = HumanDemonstrationRuntimeState.RECORDING
        return identity

    # -- request_executor seam -------------------------------------------

    def request_executor(self, request: CoordinationRequest) -> CoordinationResult:
        """Installed as the host's ``request_executor`` seam
        (Callable[[CoordinationRequest], CoordinationResult]) while Human
        Demonstration mode is active. Never calls ``plugin.assign()``."""

        if self._active_identity is None:
            raise HumanDemonstrationRuntimeStateError(
                "request_executor invoked with no active human-demonstration episode"
            )
        if self._state is HumanDemonstrationRuntimeState.WAITING_FOR_SELECTION:
            raise HumanDemonstrationRuntimeStateError(
                "request_executor invoked while a manual selection round is already open"
            )

        robot_ids = tuple(int(r) for r in request.robots_to_assign)
        if not robot_ids:
            return CoordinationResult(strategy="manual_demonstration")

        candidate_pool = capture_candidate_pool(request, robot_ids)
        decision_steps_by_robot = dict(self._step_allocator.allocate_many(robot_ids))

        session = ManualDemonstrationSelectionSession(
            identity=self._active_identity,
            simulation_time_s=self._host.get_simulation_time_s(),
            candidate_pool=candidate_pool,
            robot_ids_pending=robot_ids,
            decision_steps_by_robot=decision_steps_by_robot,
        )
        self._active_session = session
        self._pending_result = None
        self._state = HumanDemonstrationRuntimeState.WAITING_FOR_SELECTION

        self._host.request_pause()
        self._host.wait_for_human_resume()  # blocks until resume()/abort_episode() runs

        result = self._pending_result
        self._pending_result = None
        if result is None:
            raise HumanDemonstrationRuntimeStateError(
                "manual selection round ended without a result (this should be unreachable "
                "unless the host's wait_for_human_resume() returned without calling resume() "
                "or abort_episode())"
            )
        return result

    # -- selection passthrough (called from GUI click handlers) ---------

    def _require_active_session(self) -> ManualDemonstrationSelectionSession:
        if self._state is not HumanDemonstrationRuntimeState.WAITING_FOR_SELECTION or self._active_session is None:
            raise HumanDemonstrationRuntimeStateError("no manual selection round is currently open")
        return self._active_session

    def select_robot(self, robot_id: int) -> None:
        self._require_active_session().select_robot(robot_id)

    def select_candidate(
        self,
        *,
        robot_id: int,
        candidate_index: int,
        candidate_id: str,
        human_response_time_s: float | None = None,
    ) -> None:
        self._require_active_session().select_candidate(
            robot_id=robot_id,
            candidate_index=candidate_index,
            candidate_id=candidate_id,
            human_response_time_s=human_response_time_s,
        )

    def candidates_for_robot(self, robot_id: int) -> tuple:
        return self._require_active_session().candidates_for_robot(robot_id)

    def resume(self) -> None:
        """Called by the host in direct response to an explicit user
        'Resume' action. Builds the real CoordinationResult exactly once
        and stores it for request_executor() to return. Does not itself
        unblock wait_for_human_resume() -- the host does that (e.g. by
        quitting its nested event loop) immediately after this returns."""

        session = self._require_active_session()
        if not session.ready_to_apply:
            raise HumanDemonstrationRuntimeStateError(
                "resume() requires every pending robot to have a selection"
            )
        result = session.build_manual_coordination_result()
        self._decisions.extend(session.decisions())
        self._pending_result = result
        self._active_session = None
        self._state = HumanDemonstrationRuntimeState.RECORDING

    # -- Abort / Finish ----------------------------------------------------

    def abort_episode(self) -> None:
        if self._active_identity is None:
            raise HumanDemonstrationRuntimeStateError("abort_episode() called with no active episode")

        if self._active_session is not None:
            pending_robot_ids = self._active_session.robot_ids_pending
            self._active_session.abort()
            self._pending_result = CoordinationResult(
                targets=tuple(None for _ in pending_robot_ids),
                reasons=tuple("human_demonstration_episode_aborted" for _ in pending_robot_ids),
                strategy="manual_demonstration",
                assignments=tuple(
                    CoordinationAssignment(
                        robot_id=rid,
                        status="HOLD",
                        target=None,
                        reason="human_demonstration_episode_aborted",
                    )
                    for rid in pending_robot_ids
                ),
                commands=tuple(
                    RobotCommand(robot_id=rid, status="HOLD", reason="human_demonstration_episode_aborted")
                    for rid in pending_robot_ids
                ),
            )
            self._active_session = None

        if self._step_allocator is not None and self._step_allocator.is_active:
            self._step_allocator.abort_episode()

        self._active_identity = None
        self._started_at_utc = None
        self._decisions = []
        self._fires_loaded_count = 0
        self._step_allocator = None
        self._state = HumanDemonstrationRuntimeState.SETUP

    def _validate_record(self, record: DemonstrationEpisodeRecord) -> DemonstrationIntegrityReport:
        errors: list[str] = []
        if not record.decisions:
            errors.append("episode has zero decisions")
        seen_steps: set[int] = set()
        for decision in record.decisions:
            if decision.decision_step in seen_steps:
                errors.append(f"duplicate decision_step {decision.decision_step}")
            seen_steps.add(decision.decision_step)
            if decision.episode_id != record.identity.episode_id:
                errors.append(f"decision {decision.decision_step} belongs to a different episode_id")
        return DemonstrationIntegrityReport(
            valid=not errors,
            errors=tuple(errors),
            warnings=(),
            validator_version="human_demonstration_runtime_v0",
        )

    def finish_episode(self) -> DemonstrationEpisodeLayout:
        if self._active_identity is None:
            raise HumanDemonstrationRuntimeStateError("finish_episode() called with no active episode")
        if self._state is HumanDemonstrationRuntimeState.WAITING_FOR_SELECTION:
            raise HumanDemonstrationRuntimeStateError(
                "cannot finish while a manual selection round is open; use Abort Episode first"
            )
        if not self._decisions:
            raise HumanDemonstrationRuntimeStateError(
                "cannot finish an episode with zero recorded decisions"
            )

        finished_at = datetime.now(timezone.utc)
        metrics = {str(k): float(v) for k, v in dict(self._host.get_final_metrics()).items()}

        record = DemonstrationEpisodeRecord(
            identity=self._active_identity,
            started_at_utc=self._started_at_utc,
            finished_at_utc=finished_at,
            termination_reason="human_finished",
            completed=True,
            decisions=tuple(self._decisions),
            final_metrics=metrics,
            fire_detection_threshold=self._fire_detection_threshold,
            schema_version=1,
        )
        integrity_report = self._validate_record(record)
        layout = self._writer.write_pending_episode(
            record, output_root=self._output_root, integrity_report=integrity_report
        )

        finished_map_id = self._active_identity.map_id
        self._active_identity = None
        self._started_at_utc = None
        self._decisions = []
        self._fires_loaded_count = 0
        if self._step_allocator is not None and self._step_allocator.is_active:
            self._step_allocator.finish_episode()
        self._step_allocator = None
        self._state = HumanDemonstrationRuntimeState.SETUP

        self._setup = self._setup.select_map(finished_map_id)
        try:
            self._setup = self._setup.select_next_unrecorded_episode(self.recorded_episode_keys())
        except DemonstrationCollectionSetupError:
            pass  # map complete; caller should check map_complete_text()

        return layout


def build_default_human_demonstration_runtime(
    host: HumanDemonstrationHostBindings, *, repo_root: Path | None = None
) -> HumanDemonstrationRuntime:
    """Convenience factory loading the standard smoke_v0 manifest + plan
    (experiments/maps/smoke_v0/manifest.json,
    experiments/collection_plans/human_demo_smoke_v0.json) and writing to
    experiments/datasets/human_demonstrations_v0/."""

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    manifest_path = root / DEFAULT_MANIFEST_PATH
    plan_path = root / DEFAULT_PLAN_PATH
    output_root = root / DEFAULT_OUTPUT_ROOT

    map_catalog = load_map_catalog(manifest_path)
    collection_plan = load_demonstration_collection_plan(plan_path, map_catalog=map_catalog)

    return HumanDemonstrationRuntime(
        map_catalog=map_catalog,
        collection_plan=collection_plan,
        sim_directory=manifest_path.parent,
        output_root=output_root,
        host=host,
    )
