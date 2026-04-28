# vla-bench

A reproducible benchmark for open Vision-Language-Action (VLA) models on standardized simulation tasks.

CRUX-aligned (small samples, qualitative log analysis, cost-disclosed). Built as the public benchmark companion to the open-world-evaluations thesis from [Kapoor et al. (2026)](https://cruxevals.com/open-world-evaluations.pdf), scoped to physical-AI sim tasks.

## Why this exists

VLA models (OpenVLA, Pi0, GR00T) are the new model class for robotics — single-transformer pixels-plus-instruction → action-tokens. Lots of papers, no public reproducible benchmark you can submit to.

This is that benchmark, v1. Two open models on a curated subset of LIBERO. Open results JSON. Cost-disclosed.

## Status

**Phase 0 — scaffolding.** Mock VLA + mock LIBERO env runs end-to-end. No real models yet.

## Quickstart

```bash
# Install (editable, no GPU required for mock)
pip install -e .

# Run the mock harness — should print task results and write a JSON
python -m vla_bench.cli eval --model mock --env mock-libero --tasks 5 --rollouts 10

# Or via Make
make eval-mock
```

A successful run writes `results/<timestamp>-mock-libero.json`.

## Phase 1 plan

| Step | What | Cost ceiling |
|------|------|--------------|
| 1 | Mock harness (this) | $0 |
| 2 | OpenVLA (4-bit) on Colab T4, single LIBERO Spatial task | $0 (free Colab) |
| 3 | Pi0.5 on Colab T4 | $0 |
| 4 | Both models, LIBERO Spatial + Goal subset (~30 tasks), 20 rollouts each, on Vast.ai A100 spot | ~$25 |
| 5 | Results page on `bench.ondeviceml.space/vla` | $0 |
| 6 | Substack post + LinkedIn post | $0 |

**Hard budget ceiling: $50.** See [docs/COSTS.md](docs/COSTS.md) for live ledger.

## License

MIT. See [LICENSE](LICENSE).
