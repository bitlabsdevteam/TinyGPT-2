# 08: Checkpoint and resume

**What to build:** Save and load Checkpoints (model, optimizer, scaler, step, RNG state) to a configurable location, and resume from the latest one as if uninterrupted.

**Blocked by:** 07

**Status:** ready-for-agent

- [ ] A run resumed from a Checkpoint reproduces the losses of an uninterrupted run
- [ ] Latest Checkpoint is found automatically
- [ ] Checkpoint location is configurable (Drive in Colab)

Spec: spec.md
