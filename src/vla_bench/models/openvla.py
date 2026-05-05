from __future__ import annotations

from typing import Any, Literal

import numpy as np

from vla_bench.models.base import VLAModel

TaskSuite = Literal["spatial", "goal"]

_CHECKPOINTS: dict[TaskSuite, str] = {
    "spatial": "moojink/openvla-oft-libero-spatial",
    "goal": "moojink/openvla-oft-libero-goal",
}

_UNNORM_KEYS: dict[TaskSuite, str] = {
    "spatial": "libero_spatial",
    "goal": "libero_goal",
}


class OpenVLAOFTModel(VLAModel):
    """OpenVLA-OFT wrapper (Feb 2025, 97.1% avg on LIBERO).

    Requires the [real] extra: uv pip install -e ".[real]"
    Checkpoints: moojink/openvla-oft-libero-{spatial,goal} on HuggingFace

    Usage:
        model = OpenVLAOFTModel(task_suite="spatial", device="cuda")
        action = model.predict(obs, "pick up the red block")
    """

    name = "openvla"

    def __init__(
        self,
        task_suite: TaskSuite = "spatial",
        device: str = "cuda",
        action_dim: int = 7,
    ) -> None:
        if task_suite not in _CHECKPOINTS:
            raise ValueError(f"task_suite must be 'spatial' or 'goal', got {task_suite!r}")
        self.task_suite = task_suite
        self.checkpoint = _CHECKPOINTS[task_suite]
        self.unnorm_key = _UNNORM_KEYS[task_suite]
        self.device = device
        self.action_dim = action_dim
        self._processor = None
        self._model = None
        self._load()

    def _load(self) -> None:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "OpenVLA-OFT requires the [real] extra. "
                "Run: uv pip install -e '.[real]'"
            ) from e

        import torch

        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self._model.train(False)

    def predict(self, observation: dict[str, Any], instruction: str) -> np.ndarray:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded.")

        import torch
        from PIL import Image

        image = Image.fromarray(observation["image"])
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self._processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            action = self._model.predict_action(**inputs, unnorm_key=self.unnorm_key)

        return np.asarray(action, dtype=np.float32).flatten()[: self.action_dim]

    def reset(self) -> None:
        pass
