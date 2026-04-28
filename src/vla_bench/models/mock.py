from __future__ import annotations

from typing import Any

import numpy as np

from vla_bench.models.base import VLAModel


class MockVLA(VLAModel):
    """Returns deterministic-pseudo-random actions. No GPU, no deps beyond numpy.

    Used to validate the harness end-to-end before paying for real GPU.
    Success rate on MockEnv is tunable via `bias` so the metrics pipeline
    has non-trivial output to exercise.
    """

    name = "mock"

    def __init__(self, action_dim: int = 7, seed: int = 0, bias: float = 0.4) -> None:
        self.action_dim = action_dim
        self.bias = bias
        self._rng = np.random.default_rng(seed)

    def predict(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        # Bias the action distribution toward "success" so the mock
        # produces non-zero success rates and exercises the metrics.
        action = self._rng.normal(loc=self.bias, scale=1.0, size=self.action_dim)
        return action.astype(np.float32)

    def reset(self) -> None:
        # Re-seed per rollout so rollouts are independent.
        self._rng = np.random.default_rng(self._rng.integers(0, 2**31 - 1))
