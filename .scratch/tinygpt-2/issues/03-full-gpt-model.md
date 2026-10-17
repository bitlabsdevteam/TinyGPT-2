# 03: Full GPT model

**What to build:** The complete model: embeddings (token + learned position + dropout), MLP, LayerNorm, pre-norm Block, final LayerNorm and LM head tied to the token embedding, GPT-2 init with scaled residual projections, and forward with optional loss.

**Blocked by:** 02

**Status:** resolved

- [x] Each Section has a shape test
- [x] LM head shares storage with the token embedding
- [x] A tiny batch overfits to ~0 loss
- [x] Parameter names match Hugging Face GPT-2
- [x] Every Section is its own module and heavily commented

Spec: spec.md
