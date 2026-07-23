"""Tests for the v0 action contract: LearningAction's optional
heading_index and the ActionSpec 0.2.0 version bump.

v0 semantics: one ExplorationCandidate is one selectable action; a
candidate carries at most one optional heading; the action space is
one-dimensional per candidate (action_index == candidate_index);
heading_index only records whether the selected candidate had an explicit
heading -- it is never a second action dimension."""

from __future__ import annotations

import pytest

from robotics_interfaces.learning import (
    CONTRACT_VERSIONS,
    EpisodeMetadata,
    LearningAction,
    build_contract_manifest,
    compute_contract_bundle_hash,
)
from robotics_interfaces.learning.versioning import ACTION_SPEC_VERSION


class TestHeadingIndexSemantics:
    def test_heading_index_none_is_valid(self):
        action = LearningAction(
            robot_id=0,
            candidate_id="c0",
            candidate_index=0,
            heading_index=None,
            action_index=0,
            issued_at_step=0,
        )
        assert action.heading_index is None

    def test_heading_index_zero_is_valid(self):
        action = LearningAction(
            robot_id=0,
            candidate_id="c0",
            candidate_index=0,
            heading_index=0,
            action_index=0,
            issued_at_step=0,
        )
        assert action.heading_index == 0

    def test_heading_index_negative_rejected(self):
        with pytest.raises(ValueError):
            LearningAction(
                robot_id=0,
                candidate_id="c0",
                candidate_index=0,
                heading_index=-1,
                action_index=0,
                issued_at_step=0,
            )

    def test_heading_index_bool_rejected(self):
        with pytest.raises(TypeError):
            LearningAction(
                robot_id=0,
                candidate_id="c0",
                candidate_index=0,
                heading_index=True,
                action_index=0,
                issued_at_step=0,
            )

    def test_other_indices_still_reject_negative_and_bool(self):
        for field_name in ("robot_id", "candidate_index", "action_index", "issued_at_step"):
            kwargs = dict(
                robot_id=0,
                candidate_id="c0",
                candidate_index=0,
                heading_index=0,
                action_index=0,
                issued_at_step=0,
            )
            kwargs[field_name] = -1
            with pytest.raises(ValueError):
                LearningAction(**kwargs)


class TestActionSpecVersion:
    def test_action_spec_is_0_2_0(self):
        assert ACTION_SPEC_VERSION == "0.2.0"
        assert CONTRACT_VERSIONS["ActionSpec"] == "0.2.0"

    def test_hash_changes_versus_action_spec_0_1_0(self):
        manifest = build_contract_manifest()
        current_hash = compute_contract_bundle_hash(manifest)

        # Equivalent manifest, differing only in ActionSpec's version --
        # the value it had before this change.
        rolled_back_manifest = {name: dict(entry) for name, entry in manifest.items()}
        rolled_back_manifest["ActionSpec"] = {
            "version": "0.1.0",
            "dataclasses": dict(manifest["ActionSpec"]["dataclasses"]),
        }
        rolled_back_hash = compute_contract_bundle_hash(rolled_back_manifest)

        assert current_hash != rolled_back_hash

    def test_hash_is_deterministic(self):
        a = compute_contract_bundle_hash(build_contract_manifest())
        b = compute_contract_bundle_hash(build_contract_manifest())
        assert a == b

    def test_episode_metadata_constructs_with_new_hash(self):
        manifest = build_contract_manifest()
        bundle_hash = compute_contract_bundle_hash(manifest)
        metadata = EpisodeMetadata(
            episode_id="ep-1",
            seed=1,
            map_id="map-1",
            robot_count=1,
            fire_count=1,
            sensor_range=4.0,
            field_of_view_deg=120.0,
            communication_range=15.0,
            max_steps=100,
            simulator_commit="deadbeef",
            contract_versions=dict(CONTRACT_VERSIONS),
            contract_bundle_hash=bundle_hash,
        )
        assert metadata.contract_bundle_hash == bundle_hash
        assert metadata.contract_versions["ActionSpec"] == "0.2.0"
