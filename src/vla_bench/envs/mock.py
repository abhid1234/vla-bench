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

    A rollout "succeeds" when the cumulative L2 norm of issued actions
    crosses a threshold within max_steps, with per-task threshold variance
    so success rates spread across tasks and exercise the metrics pipeline.
    Zero external dependencies — runs anywhere.
    """

    name = "mock-libero"

    def __init__(self, max_steps: int = 50, num_tasks: int | None = None) -> None:
        all_tasks = list(_TASK_INSTRUCTIONS.keys())
        self._tasks = all_tasks if num_tasks is None else all_tasks[:num_tasks]
        self._max_steps = max_steps
        self._current_task: str | None = None
        self._step: int = 0
        self._cumulative: float = 0.0
        self._threshold: float = 0.0
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
        self._cumulative = 0.0
        self._rng = np.random.default_rng(seed)
        # Per-task difficulty: harder tasks (higher index) need larger cumulative action norm.
        task_idx = self._tasks.index(task_id)
        self._threshold = 12.0 + task_idx * 1.5
        return self._observation()

    def step(self, action) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._current_task is None or self._rng is None:
            raise RuntimeError("Call reset() before step().")
        action = np.asarray(action, dtype=np.float32)
        self._cumulative += float(np.linalg.norm(action))
        self._step += 1
        done = self._step >= self._max_steps or self._cumulative >= self._threshold
        success = self._cumulative >= self._threshold
        return (
            self._observation(),
            1.0 if success else 0.0,
            done,
            {"success": success, "cumulative": self._cumulative, "threshold": self._threshold},
        )

    def _observation(self) -> dict[str, Any]:
        # 64x64x3 fake "image" + 8-vector "state" — same shape contract as real LIBERO,
        # so swapping to the real env later is mostly an import change.
        rng = self._rng if self._rng is not None else np.random.default_rng(0)
        return {
            "image": rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8),
            "state": rng.normal(size=8).astype(np.float32),
        }
