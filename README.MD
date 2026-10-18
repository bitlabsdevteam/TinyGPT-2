# TinyGPT-2

A GPT-2 decoder-only transformer written from scratch in PyTorch, from the token embedding to the LM head. It is pretrained on TinyStories and fine-tuned with QLoRA, both on a single Google Colab GPU (A100 target). The code is heavily commented on purpose, with tensor shapes, the math and the reasons for each design choice, so you can read it as a tutorial.

## Status

| Stage | State |
|---|---|
| Model (one file per layer), `generate()` | Done |
| Hugging Face GPT-2 weight loading and logit-match check | Done |
| TinyStories download and GPT-2 BPE sharding | Done |
| Pretraining run (AdamW, warmup + cosine, bf16, grad accumulation) | Done |
| Checkpointing and exact resume | Done |
| Profile report (loss curves, perplexity, tokens/sec, MFU, peak memory, profiler trace, samples) | Done |
| Colab pretraining notebook | Done; the A100 run is pending |
| TinyStoriesInstruct data and loss mask | Planned |
| Hand-written LoRA and QLoRA fine-tuning | Planned |
| Colab fine-tuning notebook | Planned |

Issue tracking lives in `.scratch/tinygpt-2/issues/`, and the spec is in `.scratch/tinygpt-2/spec.md`.

## Layout

```text
model/       config.py, embeddings.py, attention.py, mlp.py, layernorm.py,
             block.py, output_head.py, gpt2.py, hf_weights.py
data/        prepare.py (download + tokenize into shards), loader.py
train/       pretrain.py, checkpoint.py, profiling.py
notebooks/   pretrain.ipynb (Colab entry point)
tests/       pytest suite
```

Each architectural component has its own module:

| Section | File |
|---|---|
| Config (`GPTConfig`) | `model/config.py` |
| Token + positional embeddings | `model/embeddings.py` |
| Causal self-attention (QKV) | `model/attention.py` |
| MLP / feed-forward (GELU) | `model/mlp.py` |
| LayerNorm (pre-norm) | `model/layernorm.py` |
| Transformer block | `model/block.py` |
| Final LayerNorm + tied LM head | `model/output_head.py` |
| Full model, GPT-2 init, `generate()` | `model/gpt2.py` |

Parameter names match Hugging Face GPT-2 (`wte`, `wpe`, `h.{i}.attn.c_attn`, `ln_1`, `mlp.c_fc`, `ln_f`, ...), so pretrained GPT-2 weights can be loaded for verification.

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # test dependencies
```

Requirements: `torch`, `datasets`, `tiktoken`, `matplotlib`, `tensorboard`, `bitsandbytes` (QLoRA, GPU only) and `transformers` (weight loading and verification only).

## Usage

Tokenize TinyStories into shards (writing to a Drive folder works well in Colab):

```bash
python -m data.prepare --out_dir /path/to/shards
```

Pretrain. `--preset small` is a ~20M model for tests and debugging, and `--preset gpt2_small` is the 124M model:

```bash
python -m train.pretrain --preset gpt2_small \
    --data_dir /path/to/shards --out_dir runs/gpt2_small \
    --checkpoint_dir /path/to/drive/checkpoints
```

Useful flags: `--max_tokens` (token cap), `--batch_size`, `--grad_accum_steps`, `--learning_rate`, `--seed`, `--checkpoint_interval`, `--stop_after_step`, `--no_resume`. If `--checkpoint_dir` already holds checkpoints, training resumes from the latest one, which matters because Colab sessions drop.

In Colab, open `notebooks/pretrain.ipynb`. It mounts Drive (for checkpoints and shards only), clones this repo and runs the steps above.

## Training and profiling

- Single GPU only, with no DDP or FSDP code paths.
- bf16 autocast on A100, with an fp16 + `GradScaler` fallback on other GPUs.
- AdamW with weight decay on 2-D parameters only, warmup + cosine LR, gradient clipping and gradient accumulation.
- Everything is seeded, and the full config is logged with each run.

Each run writes a profile report containing train and validation loss curves, perplexity, tokens/sec, step time, peak GPU memory, an MFU estimate, a `torch.profiler` trace (viewable in TensorBoard or the Chrome trace viewer) and sample generations at checkpoints.

## Tests

```bash
pytest
```

The tests cover shapes for every section, the causal mask, checkpoint and resume, the data pipeline, profiling, and logit parity with Hugging Face GPT-2.

## Planned: QLoRA fine-tuning

Fine-tuning will start from the pretrained checkpoint, using `roneneldan/TinyStoriesInstruct` with the loss on the response only by default. LoRA will be hand-written in `model/lora.py` (r=16, alpha=32, dropout 0.05) on every `nn.Linear` inside the blocks. The base weights will be frozen and quantized to 4-bit NF4 with `bitsandbytes`, and `peft` will not be used. A merge function will fold the adapters back into the base weights for inference.

## Project docs

- `IDEA.MD`: the original brief
- `CLAUDE.md`: working rules for the project
- `GLOSSARY.md`: domain terms
