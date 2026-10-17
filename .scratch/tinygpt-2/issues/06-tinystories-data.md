# 06: TinyStories data pipeline

**What to build:** Download TinyStories train and validation, tokenize with GPT-2 BPE with <|endoftext|> between Stories, write token shards, and provide a loader that serves batches.

**Blocked by:** 01

**Status:** ready-for-agent

- [ ] Shards written and re-readable (location configurable, e.g. Drive)
- [ ] Story boundaries are marked with <|endoftext|>
- [ ] Loader yields (input, target) batches of the requested shape and is seed-deterministic
- [ ] Tokenization is skipped when shards already exist

Spec: spec.md
