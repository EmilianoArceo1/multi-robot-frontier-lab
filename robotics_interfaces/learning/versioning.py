"""Explicit contract versions and a combined, process-stable bundle hash.

The bundle hash is a SHA-256 over a canonical JSON serialization of the
contract manifest (versions plus structural descriptors -- the field names
of every frozen contract).  It is deterministic across processes (unlike
Python's ``hash()``), independent of mapping insertion order, and changes
whenever any contract version or contract shape changes.

No robotics_sim, Qt, numpy, torch or pandas imports are allowed here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Mapping

from robotics_interfaces.learning.primitives import to_primitive
from robotics_interfaces.learning import (
    actions as _actions,
    candidates as _candidates,
    export as _export,
    observations as _observations,
    reservations as _reservations,
    rewards as _rewards,
    termination as _termination,
    transitions as _transitions,
)

OBSERVATION_SPEC_VERSION = "0.1.0"
CANDIDATE_SPEC_VERSION = "0.1.0"
ACTION_SPEC_VERSION = "0.1.0"
REWARD_SPEC_VERSION = "0.1.0"
TRANSITION_SPEC_VERSION = "0.1.0"
TERMINATION_SPEC_VERSION = "0.1.0"
RESERVATION_SPEC_VERSION = "0.1.0"
TRAJECTORY_EXPORT_SPEC_VERSION = "0.1.0"

CONTRACT_VERSIONS: Mapping[str, str] = {
    "ObservationSpec": OBSERVATION_SPEC_VERSION,
    "CandidateSpec": CANDIDATE_SPEC_VERSION,
    "ActionSpec": ACTION_SPEC_VERSION,
    "RewardSpec": REWARD_SPEC_VERSION,
    "TransitionSpec": TRANSITION_SPEC_VERSION,
    "TerminationSpec": TERMINATION_SPEC_VERSION,
    "ReservationSpec": RESERVATION_SPEC_VERSION,
    "TrajectoryExportSpec": TRAJECTORY_EXPORT_SPEC_VERSION,
}

# Dataclasses whose field layout is part of each contract's descriptor.
_CONTRACT_DATACLASSES: Mapping[str, tuple[type, ...]] = {
    "ObservationSpec": (
        _observations.ActorObservation,
        _observations.CriticState,
        _observations.GroundTruthSnapshot,
    ),
    "CandidateSpec": (
        _candidates.CandidateObservation,
        _candidates.CandidateSetSpec,
        _candidates.HoldPolicy,
    ),
    "ActionSpec": (_actions.LearningAction,),
    "RewardSpec": (
        _rewards.LinearWeightWarmup,
        _rewards.RewardTermSpec,
        _rewards.RewardSpec,
    ),
    "TransitionSpec": (_transitions.RewardComponent, _transitions.LearningTransition),
    "TerminationSpec": (_termination.TerminationSpec,),
    "ReservationSpec": (_reservations.RouteReservation, _reservations.ReservationSpec),
    "TrajectoryExportSpec": (
        _export.EpisodeFireMetrics,
        _export.EpisodeMetadata,
        _export.TrajectoryExportSpec,
    ),
}


def build_contract_manifest() -> dict[str, dict[str, object]]:
    """Build the mapping of contract name -> version + structural descriptor.

    The descriptor lists each contract dataclass's field names, so the
    bundle hash changes when a contract's shape changes even if its version
    string was not bumped.
    """

    manifest: dict[str, dict[str, object]] = {}
    for contract_name, version in CONTRACT_VERSIONS.items():
        descriptors = {
            cls.__name__: [f.name for f in dataclasses.fields(cls)]
            for cls in _CONTRACT_DATACLASSES[contract_name]
        }
        manifest[contract_name] = {"version": version, "dataclasses": descriptors}
    return manifest


def compute_contract_bundle_hash(manifest: Mapping[str, object]) -> str:
    """SHA-256 hex digest of the canonical JSON form of ``manifest``.

    Keys are sorted, so the digest is independent of insertion order and
    stable across processes.  Python's built-in ``hash()`` is deliberately
    not used because it is salted per process.
    """

    canonical = json.dumps(
        to_primitive(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
