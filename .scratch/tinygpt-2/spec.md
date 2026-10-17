Status: ready-for-agent

# TinyGPT-2: GPT-2 from scratch, pretrained on TinyStories, fine-tuned with QLoRA

## Problem Statement

I want to understand a GPT-2 decoder-only transformer by building every part of it myself in PyTorch, from the token embedding to the output layer, instead of calling a library. I then want to see it actually learn: pretrain it on TinyStories, fine-tune it to follow story instructions, and measure how the training behaves. All of this must run on a single Colab A100, survive Colab session drops, and be readable enough to teach from. Today there is no code at all.

## Solution

A small, heavily commented PyTorch project in which each architectural Section (embeddings, attention, MLP, LayerNorm, Block, output head, assembled model) is its own module, and LoRA is a further hand-written module. Two Colab notebooks drive a Pretraining run and a Fine-tuning run. Every run produces a Profile report. The model can load Hugging Face GPT-2 weights, and its logits match `transformers` GPT-2, which proves the implementation is correct. The code is delivered through a private GitHub repo that Colab clones, with Checkpoints and data shards stored on Google Drive.

## User Stories

1. As a learner, I want each Section of GPT-2 in its own module, so that I can read one layer at a time.
2. As a learner, I want comments on every component explaining what it is and why GPT-2 uses it, so that the code teaches the architecture.
3. As a learner, I want the tensor shape annotated at every step, so that I can follow data through the model.
4. As a learner, I want the math written next to the code that implements it, so that I can connect formula to implementation.
5. As a learner, I want non-obvious choices (weight tying, pre-norm, causal masking, init scaling) explained where they occur, so that I understand why they exist.
6. As a learner, I want the Token embedding and learned positional embedding in an embeddings Section, so that I see how token ids become vectors.
7. As a learner, I want attention split into Q, K, V projections, head split, causal mask, scaled dot-product, head merge and output projection, so that each step of attention is visible.
8. As a learner, I want the feed-forward Section to expand to 4x width, apply GELU and project back, so that I see the MLP of a Block.
9. As a learner, I want pre-norm placement shown in the Block, so that I understand the residual stream.
10. As a learner, I want the final LayerNorm and LM head with weights tied to the token embedding, so that I understand the output layer.
11. As a learner, I want GPT-2 initialisation with scaled residual projections, so that deep models train stably.
12. As a model author, I want a single config object holding vocab size, context length, layers, heads, embedding width, dropout and bias, so that model size is changed in one place.
13. As a model author, I want a GPT-2 small (124M, context 1024) config, so that I can pretrain the real target.
14. As a model author, I want a small (~20M) config, so that tests and debugging run fast on any hardware.
15. As a model author, I want parameter names that match Hugging Face GPT-2, so that pretrained weights load directly.
16. As a model author, I want a `generate` method with temperature and top-k sampling, so that I can read stories out of the model.
17. As a verifier, I want to load HF GPT-2 weights (transposing the Conv1D weights) and see matching logits within tolerance, so that I know my implementation is correct.
18. As a verifier, I want a test showing that changing a future token never changes earlier logits, so that I know the causal mask works.
19. As a verifier, I want a tiny-batch overfit test that drives loss to ~0, so that I know the model and optimiser can learn.
20. As a verifier, I want a shape test for every Section, so that regressions in one layer are caught locally.
21. As a data preparer, I want TinyStories train and validation downloaded and tokenized with GPT-2 BPE, so that the model trains on the right data.
22. As a data preparer, I want Stories separated by `<|endoftext|>` in the token stream, so that the model learns story boundaries.
23. As a data preparer, I want tokens stored as shards on Drive, so that a dropped Colab session does not repeat tokenization.
24. As a trainer, I want AdamW with weight decay on 2-D parameters only, so that norms and biases are not decayed.
25. As a trainer, I want linear warmup followed by cosine decay, so that training is stable and ends gently.
26. As a trainer, I want gradient clipping, so that rare spikes do not derail a run.
27. As a trainer, I want gradient accumulation to a target effective batch size, so that batch size is independent of GPU memory.
28. As a trainer, I want bf16 autocast on the A100 (and an fp16 + GradScaler fallback on other GPUs), so that hardware is used efficiently and the run works elsewhere.
29. As a trainer, I want validation at a fixed interval, so that I can see generalisation during the run.
30. As a trainer, I want a token-cap flag, so that a Pretraining run fits in one Colab session.
31. As a trainer, I want everything seeded and the full config logged with each run, so that runs are reproducible and comparable.
32. As a Colab user, I want Checkpoints (model, optimizer, scaler, step, RNG state) written to Drive, so that a dropped session loses little.
33. As a Colab user, I want any run to resume from the latest Checkpoint and continue as if uninterrupted, so that session drops are harmless.
34. As a Colab user, I want a notebook per run type that mounts Drive, clones the repo and installs dependencies with pip, so that setup is one click.
35. As a profiler, I want train and validation loss curves and perplexity, so that I can judge learning.
36. As a profiler, I want throughput in tokens/sec and step time, so that I can judge speed.
37. As a profiler, I want peak GPU memory, so that I know how close a run is to the limit.
38. As a profiler, I want an MFU estimate, so that I know how well the GPU is used.
39. As a profiler, I want a `torch.profiler` trace over a few steps (wait/warmup/active), exported for TensorBoard and Chrome, so that I can inspect kernels and bottlenecks.
40. As a profiler, I want sample generations at Checkpoints, so that I can judge quality qualitatively.
41. As a fine-tuner, I want TinyStoriesInstruct split into a prompt part (Words, Features, Summary or Random sentence) and a response part (the Story), so that the model learns to follow instructions.
42. As a fine-tuner, I want loss counted on response tokens only by default, with a flag for whole-sequence loss, so that I can compare both.
43. As a fine-tuner, I want a hand-written LoRA layer (frozen base Linear plus trainable low-rank `B·A` scaled by `alpha/r`), so that I learn how LoRA works.
44. As a fine-tuner, I want the Base model's weights quantized to 4-bit NF4 with bitsandbytes, so that this is true QLoRA.
45. As a fine-tuner, I want Adapters on every Block Linear (`c_attn`, attention `c_proj`, `c_fc`, MLP `c_proj`) with r=16, alpha=32, dropout 0.05, so that the setup is fixed and documented.
46. As a fine-tuner, I want embeddings, the tied LM head and LayerNorms frozen, so that only the Adapter trains.
47. As a fine-tuner, I want to start from my own pretrained Checkpoint, so that the whole pipeline is mine end to end.
48. As a fine-tuner, I want a merge function that folds the Adapter into the base weights, so that inference has no extra cost.
49. As a fine-tuner, I want the Fine-tuning run to produce the same Profile report as pretraining, so that runs are comparable.
50. As the repo owner, I want the project in a local git repo with a private GitHub remote (`TinyGPT-2`) created only when I say so, so that nothing is published without my approval.
51. As the repo owner, I want a `GLOSSARY.md` of the domain terms, so that docs, issues and code use the same words.

## Implementation Decisions

- **Scope**: single GPU only. No DDP or FSDP code paths. Device is `cuda` if available, else `cpu`.
- **Sections as modules**: config, embeddings, attention, MLP, LayerNorm, Block, output head, and an assembling model module. LoRA is a further module. The model module only assembles the Sections.
- **Config**: one dataclass holding vocab size (50257), context length, layer count, head count, embedding width, dropout and bias. Two presets: GPT-2 small (124M, context 1024) and a small ~20M config (about 6 layers, 6 heads, 384 width, context 512) for tests.
- **Naming**: parameter names match Hugging Face GPT-2 (`wte`, `wpe`, `h.{i}.attn.c_attn`, `c_proj`, `ln_1`, `ln_2`, `mlp.c_fc`, `ln_f`).
- **Linear layers**: all projections are `nn.Linear`, not `Conv1D`, so they can be quantized. The HF weight loader transposes Conv1D weights and skips the mask buffers.
- **Architecture**: pre-norm Blocks (`x + attn(ln_1(x))`, then `x + mlp(ln_2(x))`), GELU MLP at 4x width, a final LayerNorm, and an LM head tied to the token embedding. GPT-2 init scales residual projections by `1/sqrt(2·n_layer)`.
- **Model interface**: forward takes token ids (and optional targets) and returns logits (and loss). `generate` supports temperature and top-k.
- **Data**: pretraining from `roneneldan/TinyStories` (train and validation); fine-tuning from `roneneldan/TinyStoriesInstruct`. Tokenizer is tiktoken `gpt2`; Stories are separated by `<|endoftext|>`. Tokens are stored as shards on Drive.
- **Fine-tuning data format**: prompt = instruction fields (Words, Features, Summary or Random sentence); response = Story. Loss on response tokens only by default, with a switch to whole-sequence loss.
- **Pretraining**: full-parameter, about one epoch of TinyStories (~470M tokens) on GPT-2 small, with a token-cap flag.
- **Training loop**: AdamW (decay on 2-D params only), warmup plus cosine schedule, gradient clipping, gradient accumulation, periodic validation, seeded everything, full config logged per run.
- **Precision**: A100 baseline uses bf16 autocast with no GradScaler. If bf16 is unsupported, fall back to fp16 with GradScaler.
- **Checkpoints**: model, optimizer, scaler, step and RNG state saved to Drive. Runs resume from the latest Checkpoint.
- **Profiling (a Profile report per run)**: loss curves, perplexity, tokens/sec, step time, peak memory, MFU, a `torch.profiler` trace with wait/warmup/active schedule exported for TensorBoard and Chrome, and sample generations at Checkpoints.
- **QLoRA**: the Base model is frozen and quantized to NF4 with bitsandbytes; the LoRA layer is hand-written (frozen base plus `B·A` scaled by `alpha/r`, no peft). Adapters on every Block Linear, r=16, alpha=32, dropout 0.05. Embeddings, tied LM head and LayerNorms stay frozen. A merge function folds the Adapter into the base weights.
- **Standard**: the "Anthropic production standard" is implemented as the training and profiling practices above. No external specification is cited.
- **Delivery**: notebooks mount Drive (Checkpoints and shards only), clone the repo from the private GitHub remote and install dependencies with pip (`torch`, `datasets`, `tiktoken`, `matplotlib`, `tensorboard`, `bitsandbytes`; `transformers` for verification only).
- **Comments**: intentionally heavy, covering what and why, tensor shapes, the math and non-obvious choices. They are kept up to date during refactors.

## Testing Decisions

- **What makes a good test**: it exercises external behaviour through a seam, not internal structure, and survives refactors that keep behaviour. It is deterministic (seeded) and runs on the small config unless it needs the real one.
- **Seam 1, the model (`forward` and `generate`)**:
  - Logits match `transformers` GPT-2 when HF weights are loaded.
  - Changing a future token leaves earlier logits unchanged (causal mask).
  - A tiny batch overfits to ~0 loss.
  - Output shapes are correct and the LM head is tied to the token embedding.
- **Seam 2, the Pretraining run entry point (small config, CPU or any GPU)**:
  - A short run writes a Checkpoint.
  - Resuming from it reproduces the losses of an uninterrupted run.
  - The run emits a Profile report containing every required field.
- **Seam 3, the Fine-tuning run entry point**:
  - Only Adapter parameters receive gradients; the Base model stays frozen.
  - Merging the Adapter gives the same logits as the unmerged model.
  - Prompt tokens contribute nothing to the loss by default.
  - LoRA math is tested on CPU with an unquantized base. The NF4 path needs CUDA, so its test runs only on a GPU.
- **One exception to few seams**: a cheap shape test per Section, as required by the project rules.
- **Not tested directly**: tokenization and sharding (covered through the run seam), and profiler-trace contents (only that the trace file exists).
- **Prior art**: none. The repository is greenfield.

## Out of Scope

- Multi-GPU, DDP, FSDP or any distributed training.
- Parameter-efficient methods other than QLoRA, and the `peft` library.
- Fine-tuning tasks other than TinyStoriesInstruct (such as classification).
- Model sizes other than the two presets; other tokenizers.
- Serving, an inference API or a UI.
- Citing or implementing any published "Anthropic production standard".
- Creating the GitHub remote or pushing. That happens only on the owner's go-ahead.

## Further Notes

- bf16 is assumed because the target is a Colab A100. A rough estimate puts one epoch of GPT-2 small at under an hour at ~150k tokens/sec, but real throughput will come from the Profile report.
- QLoRA gives little memory benefit on a 124M model; it is chosen for learning value.
- Vocabulary follows `GLOSSARY.md`: Pretraining run, Fine-tuning run, Adapter, Base model, Checkpoint, Section, Block, Profile report.
