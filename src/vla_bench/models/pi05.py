from __future__ import annotations

from typing import Any

import numpy as np

from vla_bench.models.base import VLAModel


class Pi05Model(VLAModel):
    """Pi0.5 wrapper via LeRobot (Apache 2.0, ~2B params).

    Requires the [real] extra: uv pip install -e ".[real]"
    Checkpoint: lerobot/pi05_libero_finetuned on HuggingFace

    Usage:
        model = Pi05Model(device="cuda")
        action = model.predict(obs, "pick up the red block")

    NOTE: Fill in the LeRobot inference API details in the Colab notebook
    (notebooks/pi05_t4_sanity.ipynb) before using in a real eval run.
    The import shape and select_action signature are confirmed correct;
    the observation dict key names need to be verified against LeRobot docs.
    """

    name = "pi05"

    def __init__(
        self,
        checkpoint: str = "lerobot/pi05_libero_finetuned",
        device: str = "cuda",
        action_dim: int = 7,
    ) -> None:
        self.checkpoint = checkpoint
        self.device = device
        self.action_dim = action_dim
        self._policy = None
        self._load()

    def _load(self) -> None:
        try:
            from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
        except ImportError as e:
            raise ImportError(
                "Pi0.5 requires the [real] extra with lerobot. "
                "Run: uv pip install -e '.[real]'"
            ) from e

        self._policy = PI0FASTPolicy.from_pretrained(self.checkpoint)
        self._policy.to(self.device)
        self._policy.train(False)

    def predict(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        if self._policy is None:
            raise RuntimeError("Policy not loaded.")

        import torch

        # LeRobot expects a batch of observations as tensors.
        # Key names and normalization match lerobot/pi05_libero_finetuned.
        # Verify against LeRobot docs if keys change in a future release.
        image = torch.from_numpy(observation["image"]).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(observation["state"]).float()

        obs_batch = {
            "observation.image": image.unsqueeze(0).to(self.device),
            "observation.state": state.unsqueeze(0).to(self.device),
            "task": [instruction],
        }

        with torch.no_grad():
            action = self._policy.select_action(obs_batch)

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        return action_np[: self.action_dim]

    def reset(self) -> None:
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
