# CLAUDE.md — vla-bench project context

## What this is

`vla-bench` is a reproducible, cost-disclosed benchmark for open Vision-Language-Action (VLA) models on standardized simulation tasks. Public companion artifact to the open-world-evaluations thesis from [Kapoor et al. 2026 (CRUX paper)](https://cruxevals.com/open-world-evaluations.pdf), scoped to physical-AI sim tasks.

**Goals (in priority order):**
1. Learn — Abhi is code-fluent but rusty; this is a deliberate Java-to-Python-with-AI ramp.
2. Ship a public artifact — leaderboard at `bench.ondeviceml.space/vla` (extends his existing benchmark site).
3. Substack post (technical, HN/r/MachineLearning audience) + LinkedIn post (strategic, partnerships peers).

**This is not:**
- A real-robot benchmark. Sim only.
- An attempt to beat published OpenVLA paper numbers. Different protocol.
- A 130-task full LIBERO sweep. v1 covers ~30 tasks.

## Hard constraints — DO NOT VIOLATE

1. **$50 total budget.** Live ledger at `docs/COSTS.md`. Discipline rules:
   - Never spin up a paid GPU without a 1-hour timer
   - If not working at 30 min, kill it and go back to free Colab
   - Update `docs/COSTS.md` same day a charge hits
   - At $40 cumulative spend, stop until the project ships
2. **Free tier first, always.** Stack: Colab Pro ($10/mo) + Kaggle (30 free GPU hrs/wk) + Lightning.ai free credits + HuggingFace Spaces. Spend zero on iteration cycles.
3. **Mock harness must work without GPU.** `make eval-mock` is the smoke test. If you ever break that, the project lost its dev-without-spend property.
4. **Results JSON is versioned.** `schema_version` field in every output. Bumping it requires a migration note.

## Architecture (current)

Two clean abstractions, designed so adding a real model is a single new file.

```
src/vla_bench/
├── models/
│   ├── base.py    VLAModel ABC: predict(obs, instruction) -> action
│   └── mock.py    MockVLA — numpy random, bias=0.4 for non-trivial success
└── envs/
    ├── base.py    Env ABC: reset/step/list_tasks/task_instruction
    └── mock.py    MockLIBEROEnv — 10 LIBERO-shaped tasks, deterministic, CPU-only
```

`runner.py` loops rollouts, `metrics.py` computes per-task + overall success/latency, `results.py` writes versioned JSON, `cli.py` is the entry point.

**Adding a real model = ONE new file** (`models/openvla.py` etc.) + ONE line in `models/__init__.py` REGISTRY.

## v1 model picks (locked)

- **OpenVLA** (Stanford, 7B, Apache 2.0, 4-bit quantized) — most-cited baseline
- **Pi0.5** (Physical Intelligence, ~2B open weights) — newest small open VLA

GR00T-N1 (NVIDIA) deferred to a v2 sequel post.

## v1 task picks (locked)

- LIBERO-Spatial (~10 tasks)
- LIBERO-Goal (~10 tasks)

LIBERO-Object and LIBERO-Long deferred to v2.

## Eval protocol (locked)

- 20 rollouts/task/model
- Same instructions/observations/seeds across models
- Env-reported success criterion (not text-judged)
- Logged: success/failure, step count, total inference latency, action sequence
- Stats: per-task success rate, overall success rate, stdev across tasks, mean latency
- Optional: 95% bootstrapped CI on overall success rate (1000 resamples)

## Common commands

```bash
make eval-mock          # smoke test the harness, no GPU
make test               # pytest, no GPU
make fmt                # ruff format
make lint               # ruff check
vla-bench eval --model mock --env mock-libero --tasks 5 --rollouts 5
```

## Industry context (for the blog/LinkedIn pegs)

Confirmed 2026-04-28 via `/last30days "Vision Language Action models"`:
- **Sereact raised $110M Series B** for VLA-driven robotics adaptability (Apr 27)
- **Jensen Huang named VLAs as a foundational concept** at NVIDIA (The Drum, Apr 21)
- **arXiv: "Characterizing VLA Models across XPUs"** (Zhou et al., Apr 28) — the closest existing academic work; it's a characterization paper, not a public benchmark. **The gap this project fills.**
- HuggingPapers VLA Safety Survey + RedVLA red-teaming + EmbodiedMidtrain (Apr 27-28) — field is mature enough to survey, mature enough to benchmark
- r/deeplearning thread "Understanding VLA Models — comments needed" — audience demand signal

## How to work with Abhi

Stable across sessions; carries from `~/Core/Workspace/ClaudeCode/CLAUDE.md`:
- Be concise and direct. Lead with the answer.
- Don't over-explain things he already knows. He knows GCP, partnerships, deal mechanics. Newer to ML internals.
- Default to sub-agents for research, multi-file reads, transcript analysis, anything where only the result matters.
- Visual outputs (slides, charts, blog drafts): iterate at least twice against the skill rules before showing.
- Never commit social/launch/marketing files (per `feedback_social_posts_never_in_git.md`). Drafts live outside git.
- Memory rules from `~/Core/Workspace/ClaudeCode/CLAUDE.md` apply here.

## Source-paper anchors (already in his Learning wiki)

- CRUX paper — `Learning/wiki/concepts/open-world-evaluations.md` and `Learning/wiki/sources/2026-04-22-crux-open-world-evaluations.md`
- a16z continual learning — `Learning/wiki/concepts/continual-learning.md` and `Learning/wiki/sources/2026-04-22-a16z-continual-learning.md`
- Anthropic Mythos benchmark-saturation admission — `Learning/wiki/domains/foundation-models/claude-mythos.md`
- Karpathy AutoResearch (canonical open-world eval per CRUX) — referenced across multiple wiki pages
