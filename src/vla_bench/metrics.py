from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

from vla_bench.envs.base import RolloutResult


@dataclass
class TaskMetrics:
    task_id: str
    instruction: str
    rollouts: int
    successes: int
    success_rate: float
    mean_steps: float
    mean_inference_ms: float
    cost_usd: float | None = None  # GPU cost for this task's rollouts; None on free tier
    gpu_type: str | None = None    # e.g. "T4", "A100", "CPU"

    def to_dict(self) -> dict:
        d: dict = {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "rollouts": self.rollouts,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "mean_steps": round(self.mean_steps, 2),
            "mean_inference_ms": round(self.mean_inference_ms, 2),
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else None,
            "gpu_type": self.gpu_type,
        }
        return d


def compute_task_metrics(
    task_id: str, instruction: str, rollouts: list[RolloutResult]
) -> TaskMetrics:
    n = len(rollouts)
    if n == 0:
        return TaskMetrics(task_id, instruction, 0, 0, 0.0, 0.0, 0.0)
    successes = sum(1 for r in rollouts if r.success)
    return TaskMetrics(
        task_id=task_id,
        instruction=instruction,
        rollouts=n,
        successes=successes,
        success_rate=successes / n,
        mean_steps=mean(r.steps for r in rollouts),
        mean_inference_ms=mean(r.total_inference_ms for r in rollouts),
    )


def overall_success_rate(task_metrics: list[TaskMetrics]) -> float:
    total_rollouts = sum(t.rollouts for t in task_metrics)
    if total_rollouts == 0:
        return 0.0
    total_successes = sum(t.successes for t in task_metrics)
    return total_successes / total_rollouts


def task_rate_stdev(task_metrics: list[TaskMetrics]) -> float:
    rates = [t.success_rate for t in task_metrics if t.rollouts > 0]
    return stdev(rates) if len(rates) >= 2 else 0.0
