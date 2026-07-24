"""Host-side action catalog: translate action_index -> the real candidate.

A policy emits an ``action_index``.  ``ActionCatalogAssembler`` builds, for
each robot and decision step, an executable, immutable catalog that maps
that index back to the real ``ExplorationCandidate`` data needed to act on
it (target, optional heading, kind, source, enabled/reachable) -- without
ever storing the ``ExplorationCandidate``/``CandidateCaptureInput`` objects
themselves or reading ``candidate.metadata``.

v0 candidate/action semantics (see also
``robotics_sim.learning.capture_inputs.RuntimeActorFrame`` and
``robotics_interfaces.learning.actions.LearningAction``):

- one ``ExplorationCandidate`` is one selectable action;
- a candidate carries at most one optional heading; representing the same
  viewpoint with different headings requires the candidate generator to
  emit separate candidates -- this module never creates or expands
  headings;
- the v0 action space is one-dimensional per candidate: ``action_index``
  always equals ``candidate_index``;
- ``heading_index`` on the resulting ``LearningAction`` is not a second
  action dimension, only a record of whether the candidate had an explicit
  heading.

Allowed dependency direction: robotics_sim.learning ->
robotics_interfaces.learning.  No Qt, numpy, torch, pandas, robotics_sim.app
or engine imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robotics_interfaces.learning import CandidateKind, LearningAction
from robotics_sim.learning.capture_inputs import RuntimeActorFrame
from robotics_sim.learning.observation_batch import ActorObservationBatch, build_candidate_id


@dataclass(frozen=True)
class ActionOption:
    """One executable action: the real candidate data behind one
    ``action_index``.

    Stores no ``ExplorationCandidate``, no ``CandidateCaptureInput``, no
    ``candidate.metadata``, no mutable references and no ground truth --
    only plain, immutable, policy-safe values.
    """

    robot_id: int
    candidate_id: str
    candidate_index: int
    action_index: int
    target_xy: tuple[float, float]
    heading_rad: float | None
    kind: CandidateKind
    source: str
    enabled: bool
    reachable: bool

    def __post_init__(self) -> None:
        for name in ("robot_id", "candidate_index", "action_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise ValueError(f"candidate_id must be a non-empty string, got {self.candidate_id!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(f"source must be a non-empty string, got {self.source!r}")

        xy = tuple(self.target_xy)
        if len(xy) != 2:
            raise ValueError(f"target_xy must be an (x, y) pair, got {xy!r}")
        for v in xy:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"target_xy must contain real numbers, got {xy!r}")
            if not math.isfinite(v):
                raise ValueError(f"target_xy must be finite, got {xy!r}")
        object.__setattr__(self, "target_xy", (float(xy[0]), float(xy[1])))

        if self.heading_rad is not None:
            if isinstance(self.heading_rad, bool) or not isinstance(self.heading_rad, (int, float)):
                raise TypeError(
                    f"heading_rad must be a real number or None, got "
                    f"{type(self.heading_rad).__name__}"
                )
            if not math.isfinite(self.heading_rad):
                raise ValueError(f"heading_rad must be finite, got {self.heading_rad!r}")
            object.__setattr__(self, "heading_rad", float(self.heading_rad))

        if not isinstance(self.kind, CandidateKind):
            raise TypeError(f"kind must be a CandidateKind, got {type(self.kind).__name__}")
        if self.kind is CandidateKind.HOLD:
            raise ValueError(
                "ActionOption kind must not be HOLD: HOLD v0 is a host-side fallback, never "
                "a policy-selectable action"
            )

        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be bool, got {type(self.enabled).__name__}")
        if not isinstance(self.reachable, bool):
            raise TypeError(f"reachable must be bool, got {type(self.reachable).__name__}")
        if self.enabled and not self.reachable:
            raise ValueError("ActionOption must not be enabled while unreachable")

        if self.action_index != self.candidate_index:
            raise ValueError(
                f"v0 requires action_index == candidate_index, got "
                f"action_index={self.action_index}, candidate_index={self.candidate_index}"
            )

    def to_learning_action(self, issued_at_step: int) -> LearningAction:
        """Convert this option into an executable ``LearningAction``.

        Raises if this option is disabled or unreachable: only an
        enabled, reachable option can be issued as an action.
        """

        if isinstance(issued_at_step, bool) or not isinstance(issued_at_step, int):
            raise TypeError(
                f"issued_at_step must be an int, got {type(issued_at_step).__name__}"
            )
        if issued_at_step < 0:
            raise ValueError(f"issued_at_step must be non-negative, got {issued_at_step}")
        # Checked in this order (reachable before enabled) so both branches
        # are reachable via real ActionOption construction: enabled=True
        # always implies reachable=True (see __post_init__), so an option
        # with enabled=False can still have reachable=True (disabled-only)
        # or reachable=False (unreachable, and therefore also disabled).
        if not self.reachable:
            raise ValueError(
                f"ActionOption {self.candidate_id!r} is unreachable and cannot be issued as "
                f"an action"
            )
        if not self.enabled:
            raise ValueError(
                f"ActionOption {self.candidate_id!r} is disabled and cannot be issued as an "
                f"action"
            )

        heading_index = 0 if self.heading_rad is not None else None
        return LearningAction(
            robot_id=self.robot_id,
            candidate_id=self.candidate_id,
            candidate_index=self.candidate_index,
            heading_index=heading_index,
            action_index=self.action_index,
            issued_at_step=issued_at_step,
        )


@dataclass(frozen=True)
class RobotActionCatalog:
    """All executable actions for one robot at one decision step, in the
    exact order they were built."""

    robot_id: int
    decision_step: int
    options: tuple[ActionOption, ...]

    def __post_init__(self) -> None:
        if isinstance(self.robot_id, bool) or not isinstance(self.robot_id, int):
            raise TypeError(f"robot_id must be an int, got {type(self.robot_id).__name__}")
        if self.robot_id < 0:
            raise ValueError(f"robot_id must be non-negative, got {self.robot_id}")
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")

        options = tuple(self.options)
        seen_action_index: set[int] = set()
        seen_candidate_index: set[int] = set()
        seen_candidate_id: set[str] = set()
        for i, option in enumerate(options):
            if not isinstance(option, ActionOption):
                raise TypeError(
                    f"options[{i}] must be an ActionOption, got {type(option).__name__}"
                )
            if option.robot_id != self.robot_id:
                raise ValueError(
                    f"options[{i}].robot_id={option.robot_id} does not match catalog "
                    f"robot_id={self.robot_id}"
                )
            if option.action_index in seen_action_index:
                raise ValueError(f"options contains duplicate action_index {option.action_index}")
            seen_action_index.add(option.action_index)
            if option.candidate_index in seen_candidate_index:
                raise ValueError(
                    f"options contains duplicate candidate_index {option.candidate_index}"
                )
            seen_candidate_index.add(option.candidate_index)
            if option.candidate_id in seen_candidate_id:
                raise ValueError(f"options contains duplicate candidate_id {option.candidate_id!r}")
            seen_candidate_id.add(option.candidate_id)
        object.__setattr__(self, "options", options)

    def get_by_action_index(self, action_index: int) -> ActionOption:
        for option in self.options:
            if option.action_index == action_index:
                return option
        raise KeyError(f"no ActionOption with action_index {action_index}")

    def get_by_candidate_id(self, candidate_id: str) -> ActionOption:
        for option in self.options:
            if option.candidate_id == candidate_id:
                return option
        raise KeyError(f"no ActionOption with candidate_id {candidate_id!r}")

    def enabled_options(self) -> tuple[ActionOption, ...]:
        return tuple(option for option in self.options if option.enabled)


@dataclass(frozen=True)
class ActionCatalogBatch:
    """One RobotActionCatalog per robot for one decision step."""

    episode_id: str
    decision_step: int
    time_s: float
    catalogs: tuple[RobotActionCatalog, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError(f"episode_id must be a non-empty string, got {self.episode_id!r}")
        if self.decision_step < 0:
            raise ValueError(f"decision_step must be non-negative, got {self.decision_step}")

        catalogs = tuple(self.catalogs)
        seen_robot_ids: set[int] = set()
        for i, catalog in enumerate(catalogs):
            if not isinstance(catalog, RobotActionCatalog):
                raise TypeError(
                    f"catalogs[{i}] must be a RobotActionCatalog, got {type(catalog).__name__}"
                )
            if catalog.decision_step != self.decision_step:
                raise ValueError(
                    f"catalogs[{i}].decision_step={catalog.decision_step} does not match batch "
                    f"decision_step={self.decision_step}"
                )
            if catalog.robot_id in seen_robot_ids:
                raise ValueError(f"catalogs contains duplicate robot_id {catalog.robot_id}")
            seen_robot_ids.add(catalog.robot_id)
        object.__setattr__(self, "catalogs", catalogs)

    def get_for_robot(self, robot_id: int) -> RobotActionCatalog:
        for catalog in self.catalogs:
            if catalog.robot_id == robot_id:
                return catalog
        raise KeyError(f"no RobotActionCatalog for robot_id {robot_id}")


class ActionCatalogAssembler:
    """Pure host-side assembler: (RuntimeActorFrame, ActorObservationBatch)
    -> ActionCatalogBatch.

    Never re-sorts or truncates candidates, never reads
    ``candidate.metadata``, and never reconstructs ``target_xy`` from
    normalized features -- it reads the real ``ExplorationCandidate``
    carried by ``RuntimeActorFrame`` directly.
    """

    def build(
        self, frame: RuntimeActorFrame, actor_batch: ActorObservationBatch
    ) -> ActionCatalogBatch:
        if not isinstance(frame, RuntimeActorFrame):
            raise TypeError(f"frame must be a RuntimeActorFrame, got {type(frame).__name__}")
        if not isinstance(actor_batch, ActorObservationBatch):
            raise TypeError(
                f"actor_batch must be an ActorObservationBatch, got "
                f"{type(actor_batch).__name__}"
            )
        if frame.episode_id != actor_batch.episode_id:
            raise ValueError(
                f"frame.episode_id={frame.episode_id!r} does not match "
                f"actor_batch.episode_id={actor_batch.episode_id!r}"
            )
        if frame.decision_step != actor_batch.decision_step:
            raise ValueError(
                f"frame.decision_step={frame.decision_step} does not match "
                f"actor_batch.decision_step={actor_batch.decision_step}"
            )
        if frame.time_s != actor_batch.time_s:
            raise ValueError(
                f"frame.time_s={frame.time_s} does not match actor_batch.time_s="
                f"{actor_batch.time_s}"
            )

        frame_robot_ids = tuple(robot_capture.robot.robot_id for robot_capture in frame.robots)
        batch_robot_ids = tuple(observation.robot_id for observation in actor_batch.observations)
        if frame_robot_ids != batch_robot_ids:
            raise ValueError(
                f"frame robot order {frame_robot_ids} does not match actor_batch robot order "
                f"{batch_robot_ids}"
            )

        catalogs = []
        for robot_capture, observation in zip(frame.robots, actor_batch.observations):
            robot_id = robot_capture.robot.robot_id
            if len(robot_capture.candidates) != len(observation.candidate_ids):
                raise ValueError(
                    f"robot {robot_id}: {len(robot_capture.candidates)} captured candidates "
                    f"but observation has {len(observation.candidate_ids)} candidate_ids"
                )

            options = []
            for index, candidate_capture in enumerate(robot_capture.candidates):
                candidate_id = build_candidate_id(robot_id, frame.decision_step, index)
                if candidate_id != observation.candidate_ids[index]:
                    raise ValueError(
                        f"robot {robot_id}: candidate_id mismatch at position {index}: "
                        f"expected {candidate_id!r}, observation has "
                        f"{observation.candidate_ids[index]!r}"
                    )
                candidate = candidate_capture.candidate
                options.append(
                    ActionOption(
                        robot_id=robot_id,
                        candidate_id=candidate_id,
                        candidate_index=index,
                        action_index=index,
                        target_xy=candidate.target,
                        heading_rad=candidate.heading_rad,
                        kind=candidate_capture.kind,
                        source=candidate.source,
                        enabled=candidate_capture.enabled,
                        reachable=candidate_capture.reachable,
                    )
                )

            catalogs.append(
                RobotActionCatalog(
                    robot_id=robot_id, decision_step=frame.decision_step, options=tuple(options)
                )
            )

        return ActionCatalogBatch(
            episode_id=frame.episode_id,
            decision_step=frame.decision_step,
            time_s=frame.time_s,
            catalogs=tuple(catalogs),
        )
