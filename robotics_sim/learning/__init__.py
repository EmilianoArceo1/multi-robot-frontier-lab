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
    "ActorObservationBuildInput",
    "ActorObservationBuilder",
    "BuilderError",
    "CANDIDATE_FEATURE_NAMES_V0",
    "CandidateFeatureExtractionInput",
    "CandidateFeatureExtractor",
    "CandidateFeatureSource",
    "ContractBundleHashMismatchError",
    "CriticStateBuildInput",
    "CriticStateBuilder",
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
    "RobotFeatureExtractionInput",
    "RobotFeatureExtractor",
    "TEAMMATE_FEATURE_NAMES_V0",
    "TeammateFeatureExtractionInput",
    "TeammateFeatureExtractor",
    "TeammateFeatureSource",
    "build_feature_schema_v0",
    "normalize_by_scale",
    "require_number",
]
