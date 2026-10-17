# 13: QLoRA Fine-tuning run

**What to build:** The Fine-tuning run entry point: load our pretrained Checkpoint as the Base model, quantize its weights to 4-bit NF4 with bitsandbytes, attach Adapters, train on TinyStoriesInstruct with response-only loss, and emit a Profile report.

**Blocked by:** 08, 09, 11, 12

**Status:** ready-for-agent

- [ ] Only Adapter parameters train and the Base model stays frozen
- [ ] NF4 path works on a GPU (test is GPU-only)
- [ ] Run resumes from a Checkpoint
- [ ] Profile report is produced
- [ ] Adapter can be merged for inference

Spec: spec.md
