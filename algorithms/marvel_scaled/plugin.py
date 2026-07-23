"""Scale-normalized MARVEL coordination plugin.

This adapter uses the authors' unmodified PolicyNet and checkpoint.  It scales
all physical graph lengths together from the selected sensor range so compact
host environments preserve the dimensionless geometry seen during training.
The original paper-scale plugin remains available separately.
"""

from __future__ import annotations

from robotics_interfaces.plugins import (
    CandidateInputMode,
    CoordinationPlugin,
    PluginCapability,
    PluginMetadata,
)

from algorithms.marvel.backend import (
    MARVEL_SCALED_COORDINATOR,
    MARVEL_SOURCE,
    SCALED_SPATIAL_MODE,
    MarvelInferenceBackend,
)
from algorithms.marvel.plugin import MarvelPlugin


class MarvelScaledPlugin(MarvelPlugin):
    metadata = PluginMetadata(
        name=MARVEL_SCALED_COORDINATOR,
        version="0.1.0",
        description=(
            "MARVEL PolicyNet with a scale-normalized viewpoint graph for "
            "compact environments. Spatial ratios match the published "
            "10 m sensor and 4 m node configuration."
        ),
        capabilities=(
            PluginCapability.COORDINATION,
            PluginCapability.TASK_GENERATION,
            PluginCapability.TASK_ALLOCATION,
        ),
        source=MARVEL_SOURCE,
        candidate_input_mode=CandidateInputMode.PLUGIN_INTERNAL,
    )

    def __init__(self) -> None:
        super().__init__(
            observation_backend=MarvelInferenceBackend(
                strategy_name=MARVEL_SCALED_COORDINATOR,
                spatial_mode=SCALED_SPATIAL_MODE,
            )
        )


def create_plugin() -> CoordinationPlugin:
    return MarvelScaledPlugin()
