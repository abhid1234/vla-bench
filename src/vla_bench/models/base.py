from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class VLAModel(ABC):
    """Minimal interface every VLA model must implement.

    Real models (OpenVLA, Pi0.5, GR00T-N1) wrap this around their
    HuggingFace / native loaders. Keep the surface small so swapping
    in a new model is a 30-line file.
    """

    name: str

    @abstractmethod
    def predict(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        """Map (observation, natural-language instruction) -> action vector.

        observation: env-specific dict (typically {"image": HxWx3 uint8, "state": vec})
        instruction: e.g. "pick up the red block"
        returns: 1-D action vector (env-specific shape, typically 7-DoF end-effector)
        """
        ...

    def reset(self) -> None:
        """Optional hook called at the start of each rollout."""
        return None
