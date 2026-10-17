# 11: TinyStoriesInstruct data and loss mask

**What to build:** Prepare TinyStoriesInstruct: split each Instruction story into prompt (Words, Features, Summary or Random sentence) and response (Story), and apply response-only loss by default with a switch to whole-sequence loss.

**Blocked by:** 06

**Status:** ready-for-agent

- [ ] Prompt tokens are excluded from the loss by default
- [ ] The switch includes prompt tokens in the loss
- [ ] Batches have the expected shapes and masks
- [ ] Examples longer than the context are handled predictably

Spec: spec.md
