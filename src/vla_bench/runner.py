from __future__ import annotations

import time
from datetime import datetime, timezone

from vla_bench.envs.base import Env, RolloutResult
from vla_bench.metrics import TaskMetrics, compute_task_metrics
from vla_bench.models.base import VLAModel


def run_rollout(model: VLAModel, env: Env, task_id: str, seed: int) -> RolloutResult:
    instruction = env.task_instruction(task_id)
    obs = env.reset(task_id, seed=seed)
    model.reset()

    steps = 0
    inference_ms = 0.0
    success = False

    for _ in range(env.max_steps):
        t0 = time.perf_counter()
        action = model.predict(obs, instruction)
        inference_ms += (time.perf_counter() - t0) * 1000.0

        obs, _reward, done, info = env.step(action)
        steps += 1
        if done:
            success = bool(info.get("success", False))
            break

    return RolloutResult(success=success, steps=steps, total_inference_ms=inference_ms)


def run_task(
    model: VLAModel, env: Env, task_id: str, num_rollouts: int, base_seed: int
) -> TaskMetrics:
    instruction = env.task_instruction(task_id)
    rollouts = [
        run_rollout(model, env, task_id, seed=base_seed + i) for i in range(num_rollouts)
    ]
    return compute_task_metrics(task_id, instruction, rollouts)


def run_benchmark(
    *,
    model: VLAModel,
    env: Env,
    num_tasks: int | None = None,
    rollouts_per_task: int = 20,
    base_seed: int = 42,
) -> tuple[list[TaskMetrics], datetime, datetime]:
    started_at = datetime.now(timezone.utc)
    task_ids = env.list_tasks()
    if num_tasks is not None:
        task_ids = task_ids[:num_tasks]

    metrics: list[TaskMetrics] = []
    for task_id in task_ids:
        m = run_task(model, env, task_id, rollouts_per_task, base_seed)
        metrics.append(m)
        print(
            f"  {task_id:<8s} {m.success_rate:5.1%}  "
            f"({m.successes}/{m.rollouts})  "
            f"steps~{m.mean_steps:5.1f}  inf~{m.mean_inference_ms:6.2f}ms"
        )

    completed_at = datetime.now(timezone.utc)
    return metrics, started_at, completed_at
