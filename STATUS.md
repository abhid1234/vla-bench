# vla-bench status

Where the project is right now and what to do next. Update this file at the end of every working session.

## Last updated

2026-04-28 — scaffold validated in home dev env; mock harness smoke-tested

## Phase status

- ✅ **Phase 0: Scaffold** — Mock VLA + mock LIBERO env + runner + metrics + results JSON + tests + Docker + Makefile + budget ledger. Syntax-clean, git-committed.
- ✅ **Phase 0.5: Home dev env validation** — `make eval-mock` and `make test` both exit 0. Fixed mock env bug (thresholds were too low → always 100% success); now gives realistic 20–55% spread. Makefile switched from `python`/`pip` to `uv run`/`uv pip` to bypass Corp Airlock.
- ⏳ **Phase 1: First real model in free tier** — OpenVLA-OFT stub + Pi0.5 stub + Colab notebook written. Schema bumped to v0.2 with cost fields. Next: run the notebook on a free T4 to confirm VRAM fit and latency. **Cost so far: $0.**
- ⏸️ **Phase 2: Both models, full v1 task set** — Pi0.5 + OpenVLA on Spatial + Goal subsets, 20 rollouts each, on Vast.ai A100. Budgeted $25.
- ⏸️ **Phase 3: Public leaderboard** — `bench.ondeviceml.space/vla` page.
- ⏸️ **Phase 4: Substack + LinkedIn launch.**

## Resume here next session

**The exact next step: Run `notebooks/openvla_t4_sanity.ipynb` on Colab free T4**

1. Push this branch (or upload the repo as a zip) to GitHub so Colab can clone it.
2. Open `notebooks/openvla_t4_sanity.ipynb` in Colab, switch runtime to **T4 GPU** (free).
3. Run all cells top-to-bottom. Cell 7 prints a summary — copy those numbers into this file.

**Key questions the notebook answers:**
- Does `openvla/openvla-7b-bnb-4bit` fit in 16 GB VRAM? (expect ~8-10 GB)
- What is inference latency per step? (determines whether T4 is fast enough for 20-rollout runs)
- If latency × 20 rollouts × 30 tasks × 20 steps > ~4 hrs → need Kaggle/Vast, not free Colab.

**Note:** Cell 2 clones `https://github.com/abhidaas/vla-bench.git` — update URL if the repo name differs.

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
| 2026-04-29 | Switch OpenVLA → OpenVLA-OFT (`moojink/openvla-oft-libero-*`) | Base checkpoint doesn't have official 4-bit quant; OFT is the 2026 SOTA baseline (97.1% LIBERO) |
| 2026-04-29 | Pi0.5 via LeRobot (`lerobot/pi05_libero_finetuned`) | Confirmed Apache 2.0, open-weight, officially supported in LeRobot |
| 2026-04-29 | Add cost_usd + gpu_type to results JSON (schema v0.2) | Core differentiator vs Allen AI vla-eval; no other public VLA benchmark discloses GPU cost per task |
| 2026-04-29 | Stay LIBERO-only for v1, defer ManiSkill3 to v2 | $50 budget; Allen AI vla-eval already covers ManiSkill3; narrative focus wins over breadth |
| 2026-04-28 | LIBERO Spatial + Goal only (~30 tasks), defer Object + Long | Budget |
| 2026-04-28 | 20 rollouts/task (vs paper's 50) | Statistical signal still defensible at 20; cost halved |
| 2026-04-28 | Mock-first scaffold; real models additive via REGISTRY | Lets every iteration cycle stay free |
| 2026-04-28 | Both Substack (technical) and LinkedIn (strategic) posts at launch | Two audiences, same project |
| 2026-04-28 | Sibling repo to `web-ai-bench`, not subdir | Open-sourceable on its own |
| 2026-04-28 | MIT license | Matches `web-ai-bench` and other Abhi public projects |

## Things I'd flag to a new session

- Don't propose "let's add GR00T-N1 to v1" — commercially licensed, not open-weight. v2 only.
- Don't propose going back to base OpenVLA — superseded by OFT, checkpoint issue documented in decisions log.
- Don't propose ManiSkill3 for v1 — deferred. Allen AI vla-eval already covers it.
- Don't propose changing the rollouts count without re-checking the budget math.
- The corp Chromebook this scaffold was built on has Corp Airlock blocking pip; smoke test was deferred to Abhi's home env. Don't re-attempt installs from a corp-blocked environment.
- "Mock harness works without GPU" is load-bearing for the dev-without-spend property. Don't introduce any required dep that needs CUDA in the mock paths.
