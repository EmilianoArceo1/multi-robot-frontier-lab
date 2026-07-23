"""Weight and backend boundary for the MARVEL inference integration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol


MARVEL_WEIGHTS_ENV = "MARVEL_WEIGHTS_PATH"
DEFAULT_CHECKPOINT = Path(__file__).with_name("weights") / "checkpoint.pth"


class MarvelObservationBackend(Protocol):
    """Interchangeable bridge from simulator snapshots to MARVEL tensors."""

    def assign(self, request, policy) -> object:
        ...


@dataclass(frozen=True)
class MarvelRuntimeConfiguration:
    checkpoint_path: Path

    @classmethod
    def from_environment(cls) -> "MarvelRuntimeConfiguration":
        configured = os.getenv(MARVEL_WEIGHTS_ENV, "").strip()
        if not configured:
            configured = _local_env_value(MARVEL_WEIGHTS_ENV)
        return cls(Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT)

    def readiness_error(self) -> str | None:
        if not self.checkpoint_path.is_file():
            return (
                f"MARVEL checkpoint not found at {self.checkpoint_path}. "
                f"Place checkpoint.pth in algorithms/marvel/weights or set "
                f"{MARVEL_WEIGHTS_ENV}."
            )
        return None

    def load_policy(self):
        """Load the authors' PolicyNet lazily so plugin discovery needs no Torch."""
        import torch

        from algorithms.marvel.model import PolicyNet

        policy = PolicyNet(node_dim=6, embedding_dim=128, num_angles_bin=36)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        state_dict = checkpoint.get("policy_model", checkpoint)
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy


def _local_env_value(key: str) -> str:
    """Read one local .env value without adding a dotenv dependency."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    prefix = f"{key}="
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip().strip("\"'")
    return ""
