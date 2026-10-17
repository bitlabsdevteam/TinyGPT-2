# 02: Attention Section (QKV)

**What to build:** A heavily commented attention Section: Q/K/V projections, head split, causal mask, scaled dot-product, head merge, output projection. A reader can follow every tensor shape.

**Blocked by:** 01

**Status:** resolved

- [x] Output shape equals input shape for both presets
- [x] Changing a future token leaves earlier positions' outputs unchanged
- [x] Comments cover what/why, shapes at each step, and the softmax(QK^T/sqrt(d_k)+mask)V math
- [x] Projections are nn.Linear and named to match Hugging Face GPT-2

Spec: spec.md
