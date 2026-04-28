# Cost ledger

Every dollar spent on this project, logged. Hard ceiling: **$50.**

## Running total: $0.00

| Date | Item | Hours | Rate | Subtotal | Running |
|------|------|------:|-----:|---------:|--------:|
|      | (no spend yet) |   |   |   | $0.00 |

## Free-tier resources used

These don't count against the $50 ceiling but are tracked for honesty:

| Date | Resource | Hours used | Notes |
|------|----------|-----------:|-------|
|      | (none yet) |   |   |

## Budget plan

| Phase | Item | Estimated cost |
|-------|------|---------------:|
| 1 | Local + Colab T4 dev (mock + first model load) | $0 |
| 2 | Vast.ai 3090 spot for end-to-end debug | $4 |
| 3 | Vast.ai A100 spot for first smoke test | $5 |
| 4 | Vast.ai A100 spot for canonical run (2 models × 20 tasks × 20 rollouts) | $25 |
| 5 | Buffer for re-runs and mistakes | $10 |
| | **Total ceiling** | **$44 of $50** |

## Discipline rules

1. Never spin up a paid GPU without a 1-hour timer.
2. If it's not working at 30 minutes, kill the instance and go back to free Colab.
3. Update this file the same day a charge hits.
4. If the running total crosses $40, stop spending until the project is shipped.
