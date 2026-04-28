from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vla_bench import __version__
from vla_bench.envs import load_env
from vla_bench.models import load_model
from vla_bench.results import build_results_payload, write_results
from vla_bench.runner import run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vla-bench", description="VLA benchmark runner")
    parser.add_argument("--version", action="version", version=f"vla-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("eval", help="Run an evaluation pass")
    p_run.add_argument("--model", default="mock", help="Model name (default: mock)")
    p_run.add_argument("--env", default="mock-libero", help="Env name (default: mock-libero)")
    p_run.add_argument("--tasks", type=int, default=None, help="Number of tasks to run (default: all)")
    p_run.add_argument("--rollouts", type=int, default=20, help="Rollouts per task (default: 20)")
    p_run.add_argument("--seed", type=int, default=42, help="Base seed (default: 42)")
    p_run.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "results",
        help="Where to write results JSON",
    )

    args = parser.parse_args(argv)
    if args.command != "eval":
        parser.print_help()
        return 2

    print(f"vla-bench {__version__}")
    print(f"  model: {args.model}")
    print(f"  env:   {args.env}")
    print(f"  tasks: {args.tasks if args.tasks is not None else 'all'}, rollouts: {args.rollouts}")
    print()

    model = load_model(args.model)
    env = load_env(args.env)

    metrics, started_at, completed_at = run_benchmark(
        model=model,
        env=env,
        num_tasks=args.tasks,
        rollouts_per_task=args.rollouts,
        base_seed=args.seed,
    )

    payload = build_results_payload(
        model_name=args.model,
        model_config={},
        env_name=args.env,
        env_config={},
        started_at=started_at,
        completed_at=completed_at,
        task_metrics=metrics,
    )

    out_path = write_results(payload, args.results_dir)
    summary = payload["summary"]
    print()
    print(
        f"Overall: {summary['overall_success_rate']:.1%}  "
        f"({summary['total_rollouts']} rollouts across {summary['num_tasks']} tasks)  "
        f"in {payload['duration_s']:.2f}s"
    )
    print(f"Results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
