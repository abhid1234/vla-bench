# vla-bench status

Where the project is right now and what to do next. Update this file at the end of every working session.

## Last updated

2026-04-28 — initial scaffold complete

## Phase status

- ✅ **Phase 0: Scaffold** — Mock VLA + mock LIBERO env + runner + metrics + results JSON + tests + Docker + Makefile + budget ledger. Syntax-clean, git-committed.
- ⏳ **Phase 1: First real model in free tier** — Get OpenVLA-7B (4-bit) running on Colab T4 against a single LIBERO Spatial task. **Cost so far: $0.**
- ⏸️ **Phase 2: Both models, full v1 task set** — Pi0.5 + OpenVLA on Spatial + Goal subsets, 20 rollouts each, on Vast.ai A100. Budgeted $25.
- ⏸️ **Phase 3: Public leaderboard** — `bench.ondeviceml.space/vla` page.
- ⏸️ **Phase 4: Substack + LinkedIn launch.**

## Resume here next session

**The exact next step:**

> Validate the scaffold actually runs in Abhi's home dev environment.
>
> ```bash
> cd ~/Core/Workspace/ClaudeCode/vla-bench
> python -m venv .venv && source .venv/bin/activate
> pip install -e ".[dev]"
> make eval-mock
> make test
> ```
>
> Expected: per-task success rates printed (~30-50% range from MockVLA's bias=0.4), then a JSON file dropped in `results/`. Both `make` commands exit 0.
>
> If that works → start Phase 1 by drafting `src/vla_bench/models/openvla.py` as a stub with the right import shape, then move to a Colab notebook to fill in the implementation against a 4-bit checkpoint.
>
> If `make eval-mock` fails → debug the mock harness first. Cost ceiling for this debug: $0 (it's all CPU).

## Open questions to resolve before Phase 1 paid runs

1. Which OpenVLA 4-bit checkpoint to use — `openvla/openvla-7b-bnb-4bit` on HuggingFace? Verify it loads on a T4 (16 GB).
2. Pi0.5 release format — is there an official open-weight checkpoint, or do we use a community port? Check Physical Intelligence's HF page.
3. LIBERO installation — is it Python-native or does it need a heavy MuJoCo env? (Affects whether Colab T4 is sufficient or we need Lambda/Vast right away.)
4. Confirm 20-task subset (10 Spatial + 10 Goal) — pick by which tasks the OpenVLA paper reports baselines for, so our numbers can be put next to theirs.

## Decisions log (don't relitigate without reason)

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-28 | $50 hard budget | User-specified |
| 2026-04-28 | 2 models in v1 (OpenVLA + Pi0.5), defer GR00T-N1 to sequel | Budget cut from original 3-model plan |
| 2026-04-28 | LIBERO Spatial + Goal only (~30 tasks), defer Object + Long | Budget |
| 2026-04-28 | 20 rollouts/task (vs paper's 50) | Statistical signal still defensible at 20; cost halved |
| 2026-04-28 | Mock-first scaffold; real models additive via REGISTRY | Lets every iteration cycle stay free |
| 2026-04-28 | Both Substack (technical) and LinkedIn (strategic) posts at launch | Two audiences, same project |
| 2026-04-28 | Sibling repo to `web-ai-bench`, not subdir | Open-sourceable on its own |
| 2026-04-28 | MIT license | Matches `web-ai-bench` and other Abhi public projects |

## Things I'd flag to a new session

- Don't propose "let's add GR00T-N1 to v1" — already explicitly deferred for budget reasons. v2 sequel post.
- Don't propose changing the rollouts count without re-checking the budget math.
- The corp Chromebook this scaffold was built on has Corp Airlock blocking pip; smoke test was deferred to Abhi's home env. Don't re-attempt installs from a corp-blocked environment.
- "Mock harness works without GPU" is load-bearing for the dev-without-spend property. Don't introduce any required dep that needs CUDA in the mock paths.
