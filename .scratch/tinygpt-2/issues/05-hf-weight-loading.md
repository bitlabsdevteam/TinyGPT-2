# 05: HF weight loading and logit match

**What to build:** Load Hugging Face GPT-2 weights into our model and prove correctness: logits match transformers GPT-2 within tolerance.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Conv1D weights are transposed correctly; mask buffers are skipped (checked against the installed transformers version)
- [ ] Logits match within a documented tolerance on sample inputs
- [ ] Test is skipped cleanly when transformers or network is unavailable

Spec: spec.md
