from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from vla_bench.metrics import TaskMetrics, overall_success_rate, task_rate_stdev


SCHEMA_VERSION = "0.1"


def build_results_payload(
    *,
    model_name: str,
    model_config: dict,
    env_name: str,
    env_config: dict,
    started_at: datetime,
    completed_at: datetime,
    task_metrics: list[TaskMetrics],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {"name": model_name, "config": model_config},
        "env": {"name": env_name, "config": env_config},
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_s": (completed_at - started_at).total_seconds(),
        "tasks": [t.to_dict() for t in task_metrics],
        "summary": {
            "num_tasks": len(task_metrics),
            "total_rollouts": sum(t.rollouts for t in task_metrics),
            "overall_success_rate": round(overall_success_rate(task_metrics), 4),
            "task_success_rate_stdev": round(task_rate_stdev(task_metrics), 4),
        },
    }


def write_results(payload: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"{ts}-{payload['model']['name']}-{payload['env']['name']}.json"
    path = results_dir / fname
    path.write_text(json.dumps(payload, indent=2))
    return path
