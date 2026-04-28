# Methodology

## Why this exists

VLA models (OpenVLA, Pi0, GR00T, etc.) are the new model class for robotics — single transformers that map (pixels + language instruction) → action tokens. The field has lots of papers but no public, reproducible benchmark you can submit to.

This benchmark is a small, public, cost-disclosed answer.

## CRUX alignment

Built in the spirit of [Kapoor et al. (2026), "Open-world evaluations for measuring frontier AI capabilities"](https://cruxevals.com/open-world-evaluations.pdf). Specifically:

1. **Small samples by design.** 20 rollouts/task, not 200. Statistical signal is sufficient at this depth; the cost saving funds reproducibility.
2. **Qualitative log analysis matters more than aggregate numbers.** Every rollout's full step-by-step is logged; the blog will surface failure modes, not just success rates.
3. **Reproducible Docker image + open results JSON.** Anyone can clone, run, and submit additional models.
4. **Cost disclosed.** See [COSTS.md](COSTS.md). Every dollar tracked.
5. **Single-domain focus (manipulation in sim).** Not trying to boil the ocean. Future iterations expand.

## Model selection

Two open VLAs in v1, chosen for accessibility:

- **OpenVLA** (Stanford, 7B, Apache 2.0) — most-cited baseline. 4-bit quantized for cost.
- **Pi0.5** (Physical Intelligence, ~2B open weights) — newest small open VLA.

GR00T-N1 (NVIDIA) deferred to a v2 sequel post.

## Task selection

Subset of [LIBERO](https://libero-project.github.io/) — the de facto standard for VLA evaluation. v1 covers:

- **LIBERO-Spatial** (10 tasks): spatial reasoning, "put the X near the Y"
- **LIBERO-Goal** (10 tasks): goal-conditioned long horizon

Other LIBERO suites (Object, Long, 10) deferred to v2 to stay in budget.

## Eval protocol

- 20 rollouts per task per model
- Same task instructions, observations, and seeds across models
- Hard wall-clock cap per rollout (env max-steps)
- Success criterion: env-reported, not output-text-judged
- Logged: success/failure, step count, total inference latency, action sequence

## Stat reporting

- Per-task success rate
- Overall success rate (rollout-weighted)
- Standard deviation of per-task success rates (within-model variance)
- Mean inference latency per step
- 95% confidence intervals on overall success rate (bootstrapped, 1000 resamples)

## What this isn't

- Not a benchmark to "beat OpenVLA's paper number." Different protocol, different sample size.
- Not a real-robot benchmark. Sim only. Sim-to-real gap is a known limitation; addressed in the writeup.
- Not exhaustive. 20 of LIBERO's 130 tasks. Selection criterion is documented; not a full leaderboard.

## What's next

- v2: add GR00T-N1, expand to all 4 LIBERO suites, run on Phase 1 hardware budget allowing.
- v3: real-robot run if cost allows or via a hardware partner.
