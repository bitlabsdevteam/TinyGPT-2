# 10: Pretraining notebook

**What to build:** A Colab notebook that mounts Drive, clones the repo, installs dependencies with pip, and runs the GPT-2 small Pretraining run on an A100, resuming from the latest Checkpoint. The human runs it on the A100 and reports the Profile report.

**Blocked by:** 08, 09

**Status:** ready-for-human

- [x] Notebook is written and runs top to bottom on a tiny config
- [ ] Human run on an A100: GPT-2 small run completes or resumes after a dropped session
- [ ] Profile report from the A100 run is saved to Drive

## Hand-off to the human

Waiting on you (the agent cannot run an A100):
1. Create the GitHub repo and push (not done yet; no remote exists), then fill in `REPO_URL` in `notebooks/pretrain.ipynb`.
2. On Colab, pick the A100 runtime, run once with `SMOKE_TEST = True` (about 2 minutes), then with `SMOKE_TEST = False`.
3. Interrupt a run on purpose and *Run all* again to confirm it resumes.
4. Report back the `profile_report.json` summary (tokens/sec, peak memory, MFU). If CUDA runs out of memory, halve `BATCH_SIZE` and double `GRAD_ACCUM_STEPS`.

Spec: spec.md
