"""Host-side adapters that build the neutral learning contracts.

Dependency direction: robotics_sim.learning -> robotics_interfaces.learning
only.  No runtime integration, no file output, no Qt/numpy/torch/pandas.
"""

from __future__ import annotations

from robotics_sim.learning.source_models import (
    ActorObservationBuildInput,
    CandidateFeatureSource,
    CriticStateBuildInput,
    FeatureSchema,
    GroundTruthBuildInput,
    TeammateFeatureSource,
)
from robotics_sim.learning.capture_inputs import (
    CandidateCaptureInput,
    RobotActorCaptureInput,
    RuntimeActorFrame,
)
from robotics_sim.learning.observation_batch import (
    ActorObservationBatch,
    ActorObservationBatchAssembler,
    build_candidate_id,
)
from robotics_sim.learning.action_catalog import (
    ActionCatalogAssembler,
    ActionCatalogBatch,
    ActionOption,
    RobotActionCatalog,
)
from robotics_sim.learning.decision_batch import (
    DecisionCaptureAssembler,
    DecisionCaptureBatch,
)
from robotics_sim.learning.builders import (
    ActorObservationBuilder,
    BuilderError,
    CriticStateBuilder,
    DuplicateCandidateIdError,
    FeatureSchemaMismatchError,
    GroundTruthSnapshotBuilder,
    InvalidFeatureValueError,
)
from robotics_sim.learning.feature_schema_v0 import (
    CANDIDATE_FEATURE_NAMES_V0,
    ROBOT_FEATURE_NAMES_V0,
    TEAMMATE_FEATURE_NAMES_V0,
    build_feature_schema_v0,
)
from robotics_sim.learning.feature_inputs import (
    CandidateFeatureExtractionInput,
    FeatureNormalizationConfig,
    RobotFeatureExtractionInput,
    TeammateFeatureExtractionInput,
    normalize_by_scale,
    require_number,
)
from robotics_sim.learning.feature_extractors import (
    CandidateFeatureExtractor,
    RobotFeatureExtractor,
    TeammateFeatureExtractor,
)
from robotics_sim.learning.recorder import (
    ContractBundleHashMismatchError,
    EpisodeIdMismatchError,
    EpisodeRecord,
    InMemoryTrajectoryRecorder,
    NonMonotonicDecisionStepError,
    RecorderError,
    RecorderStateError,
)

__all__ = [
    "ActionCatalogAssembler",
    "ActionCatalogBatch",
    "ActionOption",
    "ActorObservationBatch",
    "ActorObservationBatchAssembler",
    "ActorObservationBuildInput",
    "ActorObservationBuilder",
    "BuilderError",
    "CANDIDATE_FEATURE_NAMES_V0",
    "CandidateCaptureInput",
    "CandidateFeatureExtractionInput",
    "CandidateFeatureExtractor",
    "CandidateFeatureSource",
    "ContractBundleHashMismatchError",
    "CriticStateBuildInput",
    "CriticStateBuilder",
    "DecisionCaptureAssembler",
    "DecisionCaptureBatch",
    "DuplicateCandidateIdError",
    "EpisodeIdMismatchError",
    "EpisodeRecord",
    "FeatureNormalizationConfig",
    "FeatureSchema",
    "FeatureSchemaMismatchError",
    "GroundTruthBuildInput",
    "GroundTruthSnapshotBuilder",
    "InMemoryTrajectoryRecorder",
    "InvalidFeatureValueError",
    "NonMonotonicDecisionStepError",
    "RecorderError",
    "RecorderStateError",
    "ROBOT_FEATURE_NAMES_V0",
    "RobotActionCatalog",
    "RobotActorCaptureInput",
    "RobotFeatureExtractionInput",
    "RobotFeatureExtractor",
    "RuntimeActorFrame",
    "TEAMMATE_FEATURE_NAMES_V0",
    "TeammateFeatureExtractionInput",
    "TeammateFeatureExtractor",
    "TeammateFeatureSource",
    "build_candidate_id",
    "build_feature_schema_v0",
    "normalize_by_scale",
    "require_number",
]
