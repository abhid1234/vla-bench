from __future__ import annotations

from typing import Any

import numpy as np

from vla_bench.envs.base import Env


# Hand-picked instructions echoing LIBERO Spatial / Goal task framing.
_TASK_INSTRUCTIONS: dict[str, str] = {
    "task_0": "pick up the red block",
    "task_1": "place the bowl on the plate",
    "task_2": "open the top drawer",
    "task_3": "put the cup in the sink",
    "task_4": "stack the blue block on the red block",
    "task_5": "press the button next to the lamp",
    "task_6": "move the apple to the basket",
    "task_7": "close the cabinet door",
    "task_8": "pour from the kettle into the mug",
    "task_9": "hand the spoon to the robot on the right",
}


class MockLIBEROEnv(Env):
    """Pure-Python deterministic env that mimics LIBERO's API surface.

    Success in each step is determined by whether mean(action) exceeds a
    per-task threshold. With MockVLA bias=0.4 and action_dim=7, this gives
    a spread of ~60% → ~10% success rates across tasks, exercising the
    metrics pipeline without a GPU. Zero external dependencies.
    """

    name = "mock-libero"

    # Calibrated for MockVLA(bias=0.4, action_dim=7): mean(action) ~ N(0.4, 0.378).
    # Each threshold maps to a per-step success probability, which compounds
    # over max_steps=50 to give the target per-task overall success rates
    # (~60%, 53%, 46%, 39%, 32%, 26%, 20%, 15%, 11%, 7% across tasks 0-9).
    _TASK_THRESHOLDS = [1.19, 1.22, 1.25, 1.28, 1.31, 1.34, 1.37, 1.40, 1.43, 1.47]

    def __init__(self, max_steps: int = 50, num_tasks: int | None = None) -> None:
        all_tasks = list(_TASK_INSTRUCTIONS.keys())
        self._tasks = all_tasks if num_tasks is None else all_tasks[:num_tasks]
        self._max_steps = max_steps
        self._current_task: str | None = None
        self._step: int = 0
        self._action_threshold: float = 0.0
        self._rng: np.random.Generator | None = None

    def list_tasks(self) -> list[str]:
        return list(self._tasks)

    def task_instruction(self, task_id: str) -> str:
        if task_id not in _TASK_INSTRUCTIONS:
            raise ValueError(f"Unknown task {task_id!r}")
        return _TASK_INSTRUCTIONS[task_id]

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id!r} not in this env's task set")
        self._current_task = task_id
        self._step = 0
        self._rng = np.random.default_rng(seed)
        task_idx = self._tasks.index(task_id)
        self._action_threshold = self._TASK_THRESHOLDS[task_idx]
        return self._observation()

    def step(self, action) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._current_task is None or self._rng is None:
            raise RuntimeError("Call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        self._step += 1
        # Success if action mean exceeds task threshold — ties model bias to outcome.
        success = bool(float(np.mean(action)) > self._action_threshold)
        done = success or self._step >= self._max_steps
        return (
            self._observation(),
            1.0 if success else 0.0,
            done,
            {"success": success, "action_mean": float(np.mean(action)), "threshold": self._action_threshold},
        )

    def _observation(self) -> dict[str, Any]:
        # 64x64x3 fake "image" + 8-vector "state" — same shape contract as real LIBERO,
        # so swapping to the real env later is mostly an import change.
        rng = self._rng if self._rng is not None else np.random.default_rng(0)
        return {
            "image": rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8),
            "state": rng.normal(size=8).astype(np.float32),
        }
