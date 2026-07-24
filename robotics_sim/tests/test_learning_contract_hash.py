"""Tests for contract versioning and the combined SHA-256 bundle hash."""

from __future__ import annotations

import re

from robotics_interfaces.learning import (
    CONTRACT_VERSIONS,
    build_contract_manifest,
    compute_contract_bundle_hash,
)

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_CONTRACTS = {
    "ObservationSpec",
    "CandidateSpec",
    "ActionSpec",
    "RewardSpec",
    "TransitionSpec",
    "TerminationSpec",
    "ReservationSpec",
    "TrajectoryExportSpec",
}


class TestManifest:
    def test_all_eight_contracts_have_semantic_versions(self):
        assert set(CONTRACT_VERSIONS) == EXPECTED_CONTRACTS
        for name, version in CONTRACT_VERSIONS.items():
            assert SEMVER.match(version), f"{name} version {version!r} is not semantic"

    def test_manifest_contains_versions_and_descriptors(self):
        manifest = build_contract_manifest()
        assert set(manifest) == EXPECTED_CONTRACTS
        for name, entry in manifest.items():
            assert entry["version"] == CONTRACT_VERSIONS[name]
            assert entry["dataclasses"], f"{name} has no structural descriptor"


class TestBundleHash:
    def test_hash_is_sha256_hex(self):
        digest = compute_contract_bundle_hash(build_contract_manifest())
        assert SHA256_HEX.match(digest)

    def test_hash_is_deterministic(self):
        a = compute_contract_bundle_hash(build_contract_manifest())
        b = compute_contract_bundle_hash(build_contract_manifest())
        assert a == b

    def test_hash_is_independent_of_insertion_order(self):
        manifest = build_contract_manifest()
        reversed_manifest = dict(reversed(list(manifest.items())))
        assert list(manifest) != list(reversed_manifest)
        assert compute_contract_bundle_hash(manifest) == compute_contract_bundle_hash(
            reversed_manifest
        )

    def test_hash_changes_when_a_version_changes(self):
        manifest = build_contract_manifest()
        original = compute_contract_bundle_hash(manifest)
        modified = {
            name: dict(entry) for name, entry in manifest.items()
        }
        modified["RewardSpec"]["version"] = "0.2.0"
        assert compute_contract_bundle_hash(modified) != original

    def test_hash_changes_when_a_descriptor_changes(self):
        manifest = build_contract_manifest()
        original = compute_contract_bundle_hash(manifest)
        modified = {name: dict(entry) for name, entry in manifest.items()}
        descriptors = dict(modified["ActionSpec"]["dataclasses"])
        descriptors["LearningAction"] = list(descriptors["LearningAction"]) + ["new_field"]
        modified["ActionSpec"] = {
            "version": modified["ActionSpec"]["version"],
            "dataclasses": descriptors,
        }
        assert compute_contract_bundle_hash(modified) != original
