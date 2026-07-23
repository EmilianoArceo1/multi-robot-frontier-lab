"""Tests for inspect_learning_compatibility and
LearningCoordinatorCompatibilityError."""

from __future__ import annotations

import pytest

from algorithms.fuel_frontier_baseline.plugin import (
    create_plugin as create_fuel_frontier_baseline,
)
from algorithms.global_noic_legacy.plugin import create_plugin as create_global_noic_legacy
from algorithms.independent_baseline.plugin import (
    INDEPENDENT_BASELINE_COORDINATOR,
    create_plugin as create_independent_baseline,
)
from robotics_interfaces.plugins import CandidateInputMode, PluginMetadata
from robotics_sim.learning.coordination_decision_source import (
    LearningCoordinationDecisionSource,
    LearningCoordinatorCompatibilityError,
    inspect_learning_compatibility,
)


class TestIndependentBaselineSupported:
    def test_compatibility_supported(self):
        compat = inspect_learning_compatibility(create_independent_baseline())
        assert compat.supported is True
        assert compat.candidate_input_mode is CandidateInputMode.HOST_CANDIDATES
        assert compat.plugin_name == INDEPENDENT_BASELINE_COORDINATOR

    def test_source_constructs_without_raising(self):
        source = LearningCoordinationDecisionSource(create_independent_baseline())
        assert source.compatibility.supported is True


class TestGlobalNoicLegacyRejected:
    def test_compatibility_rejected(self):
        compat = inspect_learning_compatibility(create_global_noic_legacy())
        assert compat.supported is False
        assert compat.candidate_input_mode is CandidateInputMode.LEGACY_INTEGRATED

    def test_source_raises(self):
        with pytest.raises(LearningCoordinatorCompatibilityError):
            LearningCoordinationDecisionSource(create_global_noic_legacy())


class TestHybridRejected:
    def test_compatibility_rejected(self):
        compat = inspect_learning_compatibility(create_fuel_frontier_baseline())
        assert compat.supported is False
        assert compat.candidate_input_mode is CandidateInputMode.HYBRID

    def test_source_raises(self):
        with pytest.raises(LearningCoordinatorCompatibilityError):
            LearningCoordinationDecisionSource(create_fuel_frontier_baseline())


class _NoMetadataPlugin:
    def assign(self, request):  # pragma: no cover - must never be called
        raise AssertionError("assign() must not be called on an incompatible plugin")


class _NoAssignPlugin:
    metadata = PluginMetadata(
        name="no-assign-plugin",
        version="0.0.0",
        description="",
        capabilities=(),
        candidate_input_mode=CandidateInputMode.HOST_CANDIDATES,
    )


class TestFakePluginsRejected:
    def test_no_metadata_rejected(self):
        compat = inspect_learning_compatibility(_NoMetadataPlugin())
        assert compat.supported is False
        assert compat.candidate_input_mode is None
        assert "metadata" in compat.reason.lower()

    def test_no_metadata_source_raises(self):
        with pytest.raises(LearningCoordinatorCompatibilityError):
            LearningCoordinationDecisionSource(_NoMetadataPlugin())

    def test_no_assign_rejected(self):
        compat = inspect_learning_compatibility(_NoAssignPlugin())
        assert compat.supported is False
        assert "assign" in compat.reason.lower()

    def test_no_assign_source_raises(self):
        with pytest.raises(LearningCoordinatorCompatibilityError):
            LearningCoordinationDecisionSource(_NoAssignPlugin())


class TestErrorMessage:
    def test_error_includes_plugin_and_reason(self):
        plugin = create_global_noic_legacy()
        expected_name = plugin.metadata.name
        with pytest.raises(LearningCoordinatorCompatibilityError) as exc_info:
            LearningCoordinationDecisionSource(plugin)
        error = exc_info.value
        assert error.plugin_name == expected_name
        assert error.reason
        assert expected_name in str(error)
        assert error.reason in str(error)


class TestNoFallbackByName:
    def test_plugin_named_like_independent_but_wrong_mode_is_rejected(self):
        # A plugin whose *name* matches the known-good coordinator must
        # still be judged solely on metadata.candidate_input_mode, never on
        # the name string.
        class _ImpostorPlugin:
            metadata = PluginMetadata(
                name=INDEPENDENT_BASELINE_COORDINATOR,
                version="0.0.0",
                description="",
                capabilities=(),
                candidate_input_mode=CandidateInputMode.PLUGIN_INTERNAL,
            )

            def assign(self, request):  # pragma: no cover - must never be called
                raise AssertionError("assign() must not be called on an incompatible plugin")

        compat = inspect_learning_compatibility(_ImpostorPlugin())
        assert compat.supported is False
        assert compat.candidate_input_mode is CandidateInputMode.PLUGIN_INTERNAL
        assert compat.plugin_name == INDEPENDENT_BASELINE_COORDINATOR

        with pytest.raises(LearningCoordinatorCompatibilityError):
            LearningCoordinationDecisionSource(_ImpostorPlugin())

    def test_plugin_internal_rejected(self):
        class _InternalPlugin:
            metadata = PluginMetadata(
                name="internal-plugin",
                version="0.0.0",
                description="",
                capabilities=(),
                candidate_input_mode=CandidateInputMode.PLUGIN_INTERNAL,
            )

            def assign(self, request):  # pragma: no cover
                raise AssertionError("must not be called")

        compat = inspect_learning_compatibility(_InternalPlugin())
        assert compat.supported is False

    def test_legacy_integrated_rejected(self):
        class _LegacyPlugin:
            metadata = PluginMetadata(
                name="legacy-plugin",
                version="0.0.0",
                description="",
                capabilities=(),
                candidate_input_mode=CandidateInputMode.LEGACY_INTEGRATED,
            )

            def assign(self, request):  # pragma: no cover
                raise AssertionError("must not be called")

        compat = inspect_learning_compatibility(_LegacyPlugin())
        assert compat.supported is False

    def test_undeclared_mode_rejected(self):
        class _UndeclaredPlugin:
            metadata = PluginMetadata(
                name="undeclared-plugin",
                version="0.0.0",
                description="",
                capabilities=(),
                candidate_input_mode=None,
            )

            def assign(self, request):  # pragma: no cover
                raise AssertionError("must not be called")

        compat = inspect_learning_compatibility(_UndeclaredPlugin())
        assert compat.supported is False
        assert compat.candidate_input_mode is None
