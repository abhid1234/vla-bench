from __future__ import annotations

import json

from vla_bench.cli import main
from vla_bench.envs import load_env
from vla_bench.models import load_model
from vla_bench.results import SCHEMA_VERSION, build_results_payload
from vla_bench.runner import run_benchmark


def test_smoke_end_to_end(tmp_path):
    """Mock model + mock env should run, write a JSON, and produce non-trivial metrics."""
    rc = main(
        [
            "eval",
            "--model", "mock",
            "--env", "mock-libero",
            "--tasks", "3",
            "--rollouts", "5",
            "--results-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["model"]["name"] == "mock"
    assert payload["env"]["name"] == "mock-libero"
    assert payload["summary"]["num_tasks"] == 3
    assert payload["summary"]["total_rollouts"] == 15
    assert 0.0 <= payload["summary"]["overall_success_rate"] <= 1.0
    assert all(t["rollouts"] == 5 for t in payload["tasks"])


def test_metrics_payload_shape():
    model = load_model("mock")
    env = load_env("mock-libero")
    metrics, started_at, completed_at = run_benchmark(
        model=model, env=env, num_tasks=2, rollouts_per_task=3
    )
    payload = build_results_payload(
        model_name="mock",
        model_config={},
        env_name="mock-libero",
        env_config={},
        started_at=started_at,
        completed_at=completed_at,
        task_metrics=metrics,
    )
    assert {"schema_version", "model", "env", "tasks", "summary"} <= set(payload.keys())
    assert payload["summary"]["num_tasks"] == 2
    for task in payload["tasks"]:
        assert {"task_id", "instruction", "rollouts", "successes", "success_rate"} <= set(task)


def test_unknown_model_rejected():
    try:
        load_model("does-not-exist")
    except ValueError as e:
        assert "Unknown model" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_unknown_env_rejected():
    try:
        load_env("does-not-exist")
    except ValueError as e:
        assert "Unknown env" in str(e)
    else:
        raise AssertionError("expected ValueError")
