# CLAUDE.md

Project brief: `IDEA.MD`. It is the source of truth for scope. This file turns it into working rules.

## Goal

Build a **GPT-2 decoder-only transformer from scratch in PyTorch**, covering every stage from the token embedding to the final output layer (LM head). Pretrain it on TinyStories and fine-tune it, both inside **Google Colab** on a **single GPU**, and profile the training runs.

Status: greenfield. No code exists yet, so the layout and commands below are *planned*. Update this file as they become real.

## Architecture: one section per layer

Split the code precisely by layer. Each architectural component gets its own clearly separated section, which means its own module/file:

| Section | Contents |
|---|---|
| Config | `GPTConfig` dataclass: `vocab_size=50257`, `n_ctx/block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`, `bias` |
| Embeddings | token embedding (`wte`) + learned positional embedding (`wpe`) + dropout |
| Attention (QKV) | Q, K, V projections, head split, causal mask, scaled dot-product, head merge, output projection |
| MLP / feed-forward | `n_embd → 4·n_embd`, GELU, `→ n_embd`, dropout |
| LayerNorm | pre-norm placement (GPT-2 style) |
| Transformer block | `x + attn(ln_1(x))`, then `x + mlp(ln_2(x))` |
| Output | final LayerNorm (`ln_f`) + LM head with weights tied to `wte` |
| Model | assembles the sections, applies GPT-2 init (residual-projection scaling `1/sqrt(2·n_layer)`), runs `generate()` |

Planned layout (subject to change):

```text
model/        config.py, embeddings.py, attention.py, mlp.py, layernorm.py, block.py, output_head.py, gpt2.py
data/         TinyStories download + GPT-2 BPE tokenization (tiktoken "gpt2") → token shards
train/        pretrain.py, finetune.py, profiling utilities
notebooks/    Colab entry points: pretrain.ipynb, finetune.ipynb
```

Colab imports: each notebook mounts Drive (checkpoints and data shards only), `git clone`s the project repo, and adds the repo root to `sys.path`.

Weight names should match Hugging Face GPT-2 (`wte`, `wpe`, `h.{i}.attn.c_attn`, `c_proj`, `ln_1`, `ln_2`, `mlp.c_fc`, `ln_f`) so pretrained GPT-2 weights can be loaded for verification and comparison.

## Commenting rule: intentionally heavy

This project is **educational**. The heavy commenting overrides any default toward minimal comments. Put detailed comments in every code section to explain every component and what it does:

- what the component is and why GPT-2 uses it
- the **tensor shape** at each step, e.g. `# (B, T, C) -> (B, n_head, T, head_dim)`
- the math being implemented (e.g. `softmax(QKᵀ/√d_k + mask)·V`)
- non-obvious choices: weight tying, pre-norm, causal masking, init scaling

Keep these comments when refactoring. Update them whenever the code changes.

## Data

- Pretraining data: Hugging Face `roneneldan/TinyStories` (verified ID), with the `train` and `validation` splits.
- Tokenizer: GPT-2 BPE. Use `<|endoftext|>` to separate stories.
- Fine-tuning data: `roneneldan/TinyStoriesInstruct` (user-confirmed), instruction → story. Prompt = instruction fields (Words, Features, Summary/Random sentence), response = Story, with response-only loss by default (flag for whole-sequence loss).

## Fine-tuning: QLoRA (user-confirmed)

Pretraining is full-parameter on GPT-2 small (124M, ctx 1024), about 1 epoch of TinyStories with a token-cap flag. A small ~20M config exists for tests and the overfit check. Fine-tuning starts from our own pretrained checkpoint and uses QLoRA: base weights frozen and quantized to 4-bit NF4, with trainable low-rank adapters. HF GPT-2 weights are used only for the logit-match verification. Because the model must be quantizable, use `nn.Linear` (not `Conv1D`) in our own modules.

- LoRA is hand-written in `model/lora.py` (the user wants to learn it), with the same heavy comments as other sections: frozen base `Linear` plus trainable `B·A` scaled by `alpha/r`. No `peft`. `bitsandbytes` supplies only the NF4 quantization.
- Adapters go on every `nn.Linear` inside the Blocks (`c_attn`, attention `c_proj`, `c_fc`, MLP `c_proj`): r=16, alpha=32, dropout 0.05. Embeddings, the tied LM head and LayerNorms stay frozen.
- Provide a function that merges the adapter into the base weights for inference.
- GitHub repo: `TinyGPT-2`, private, created only on the user's go-ahead.

## Training

These rules follow from the requirements: single GPU, Colab, and production-grade practice.

- **Single GPU only.** Write no DDP or FSDP code paths. Select the device with `cuda` if available, else `cpu`.
- **Target hardware: Colab A100** (user-confirmed). Use bf16 autocast, so no `GradScaler` is needed. Keep a fp16 + `GradScaler` fallback only if a non-bf16 GPU is detected.
- **Training loop:** AdamW with weight decay applied only to 2-D params, a warmup + cosine LR schedule, gradient clipping, and gradient accumulation to reach the target effective batch size. Run validation on a fixed interval.
- **Colab sessions drop**, so checkpoint model, optimizer, scaler, step, and RNG state to Google Drive, and make training resumable from the latest checkpoint.
- **Reproducibility:** seed everything and log the full config with every run.
- **Fit Colab GPU memory:** default configs must fit an A100 (40 GB). Provide a small TinyStories-scale config (for tests and debugging) as well as a GPT-2 small (124M) config.
- **Dependencies:** everything must be installable with `pip` in a notebook cell (`torch`, `datasets`, `tiktoken`, `matplotlib`, `tensorboard`; `transformers` for weight loading and verification only; `bitsandbytes` for 4-bit quantization in QLoRA).

## Profiling the training results

Profiling is a required deliverable. Each run must report:

- train and validation loss curves, plus perplexity
- throughput in tokens/sec and step time
- peak GPU memory (`torch.cuda.max_memory_allocated`)
- MFU estimate (model FLOPs utilization)
- a `torch.profiler` trace over a few steps, with schedule wait/warmup/active, exported for TensorBoard/Chrome trace viewing
- sample generations at checkpoints, as a qualitative check

## Verification

- Shape tests for every section, plus a check that the causal mask blocks attention to future tokens.
- A tiny-batch overfit test: loss should approach ~0.
- Load HF GPT-2 weights into the model and check that its logits match `transformers` GPT-2 within tolerance. Gotcha (check it against the installed `transformers` version): HF GPT-2 builds `c_attn`, `c_proj`, and `c_fc` with `Conv1D`, which stores weights transposed relative to `nn.Linear`, so transpose them when loading. Also skip the mask buffers in the state dict (`attn.bias`, `attn.masked_bias`).

## Open questions (do not invent answers; ask the user)

Settled: the user says to follow the "Anthropic production standard". No published document by that name has been identified, so it is implemented as the Training and Profiling sections above. Do not cite it as an external spec. Fine-tuning is TinyStoriesInstruct. Code is split one file per layer. Git is initialized (branch `main`); Colab reaches code via a git remote.

Still open:

None. The design is settled. Do not create the GitHub remote or push until the user says so.

Normalized from IDEA.MD: "QKB" → QKV; "Trin results" → training results; "tiny story dataset" → TinyStories.

## Agent skills

### Issue tracker

Issues and specs are local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
