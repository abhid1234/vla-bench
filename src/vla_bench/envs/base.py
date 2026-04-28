from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutResult:
    success: bool
    steps: int
    total_inference_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Env(ABC):
    """Minimal interface every sim env must implement."""

    name: str

    @abstractmethod
    def list_tasks(self) -> list[str]:
        """Return the list of task IDs the env supports."""
        ...

    @abstractmethod
    def task_instruction(self, task_id: str) -> str:
        """Natural-language instruction for the task (passed to the VLA)."""
        ...

    @abstractmethod
    def reset(self, task_id: str, seed: int) -> dict[str, Any]:
        """Reset to the start of `task_id` with given seed; return initial observation."""
        ...

    @abstractmethod
    def step(self, action) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply action; return (obs, reward, done, info). info["success"] is the eval criterion."""
        ...

    @property
    @abstractmethod
    def max_steps(self) -> int: ...
