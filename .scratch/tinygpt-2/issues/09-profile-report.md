# 09: Profile report

**What to build:** Every run emits a Profile report: train and validation loss curves, perplexity, tokens/sec, step time, peak GPU memory, MFU estimate, a torch.profiler trace (wait/warmup/active) for TensorBoard and Chrome, and sample generations at Checkpoints.

**Blocked by:** 04, 07

**Status:** ready-for-agent

- [ ] All required fields are present after a short run
- [ ] Profiler trace file is written
- [ ] Peak memory and MFU degrade gracefully on CPU
- [ ] Loss curves are saved as images

Spec: spec.md
