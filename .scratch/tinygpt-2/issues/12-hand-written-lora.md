# 12: Hand-written LoRA

**What to build:** A hand-written LoRA layer (frozen base Linear plus trainable B.A scaled by alpha/r, r=16, alpha=32, dropout 0.05) that can be attached to every Block Linear, plus a merge function. Heavily commented; no peft.

**Blocked by:** 03

**Status:** ready-for-agent

- [ ] Only Adapter parameters receive gradients (CPU, unquantized base)
- [ ] Merged logits equal unmerged logits within tolerance
- [ ] Embeddings, tied LM head and LayerNorms stay frozen
- [ ] At initialisation the adapted model equals the base model

Spec: spec.md
