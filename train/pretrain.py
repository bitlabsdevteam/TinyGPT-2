"""Pretraining run: teach a randomly initialised GPT-2 to predict the next token.

The loop, once per optimizer step
    1. set the learning rate for this step (warmup, then cosine decay)
    2. run `grad_accum_steps` micro-batches, adding up their gradients
    3. clip the gradient norm, take one AdamW step
    and every `eval_interval` steps measure the loss on the held-out validation split.

Everything a run needs to be reproducible and comparable is written next to the
results: the full model and training config go to `config.json`, and the seed fixes
model initialisation, data order and dropout.

Single GPU only: there is no distributed code path. The device is `cuda` if available,
else `cpu`.
"""
import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from data.loader import TokenLoader
from model.config import GPTConfig, gpt2_small, small
from model.gpt2 import GPT


@dataclass
class TrainConfig:
    data_dir: str  # folder holding the train / validation shards (see data/prepare.py)
    out_dir: str  # where config.json (and later checkpoints) are written

    # Batch size. The *effective* batch is batch_size * grad_accum_steps sequences per
    # optimizer step. A big effective batch gives smooth gradients, but does not fit in
    # GPU memory at once, so it is processed as `grad_accum_steps` smaller micro-batches
    # whose gradients are added up before the optimizer steps.
    batch_size: int = 16  # micro-batch: sequences per forward/backward pass
    grad_accum_steps: int = 32  # micro-batches per optimizer step (16*1024*32 = 0.5M tokens)

    # Token cap: the run stops after this many training tokens. None = one pass over
    # the whole train split (about one epoch).
    max_tokens: int | None = None

    # Learning-rate schedule: linear warmup to `learning_rate`, then cosine decay to `min_lr`.
    learning_rate: float = 6e-4  # peak LR; GPT-2 small is usually trained near 6e-4
    min_lr: float = 6e-5  # LR at the end of the run (10% of the peak)
    warmup_steps: int = 100  # optimizer steps of linear warmup

    weight_decay: float = 0.1  # AdamW decay strength, applied to matrices only
    beta1: float = 0.9  # AdamW momentum
    beta2: float = 0.95  # AdamW second-moment decay; 0.95 (not 0.999) is the usual LLM choice
    grad_clip: float = 1.0  # maximum gradient norm

    eval_interval: int = 100  # validate every this many optimizer steps
    eval_batches: int = 20  # micro-batches averaged per validation
    log_interval: int = 10  # print a progress line every this many steps

    seed: int = 1337
    device: str | None = None  # None = cuda if available, else cpu
    dtype: str = "auto"  # "auto", "bf16", "fp16" or "fp32"; see resolve_precision


@dataclass
class PretrainResult:
    model: GPT
    train_log: list  # one dict per optimizer step: step, loss, lr, grad_norm, tokens_seen
    eval_log: list  # one dict per validation: step, val_loss
    tokens_per_step: int
    tokens_seen: int
    config_path: str


def seed_everything(seed: int) -> None:
    """Seed every random number generator the run touches, for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op without a GPU


def resolve_precision(device_type: str, dtype: str = "auto", bf16_supported: bool | None = None):
    """Choose the mixed-precision mode. Returns (autocast_dtype, use_grad_scaler).

    Mixed precision runs the matrix multiplications in 16 bits (faster, half the
    memory) while keeping the weights and the optimizer in 32 bits.
      * bf16 has the same exponent range as fp32, so gradients cannot underflow and no
        loss scaling is needed. This is what an A100 uses.
      * fp16 has a much narrower range: small gradients round to zero. A GradScaler
        multiplies the loss up before backward() (and divides the gradients back down)
        to prevent that. It is the fallback for GPUs without bf16 (e.g. a T4).
      * None means plain fp32 (the CPU default).
    """
    if dtype == "fp32" or device_type != "cuda" and dtype == "auto":
        return None, False
    if dtype == "auto":
        if bf16_supported is None:
            bf16_supported = torch.cuda.is_bf16_supported()
        dtype = "bf16" if bf16_supported else "fp16"
    if dtype == "bf16":
        return torch.bfloat16, False
    if dtype == "fp16":
        return torch.float16, True
    raise ValueError(f"unknown dtype {dtype!r}")


def configure_optimizer(model, learning_rate, weight_decay, betas, device_type):
    """AdamW with weight decay on weight matrices only.

    Weight decay shrinks weights a little every step, which regularises the model. It
    makes sense for the big matrices (Linear weights, embeddings: 2-D tensors) but not
    for biases and LayerNorm gains (1-D), where shrinking towards 0 only hurts. So the
    parameters are split into two groups by dimensionality.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    decayed = [p for p in params if p.dim() >= 2]
    plain = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decayed, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]
    # The fused kernel is faster on a GPU; it is not available on the CPU.
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=(device_type == "cuda"))


def get_lr(step: int, cfg: TrainConfig, num_steps: int) -> float:
    """Learning rate for optimizer step `step` (0-based): linear warmup, then cosine decay.

    Warmup: Adam's statistics are unreliable in the first steps and a full-size LR
    can blow training up, so the LR ramps up linearly to the peak.
    Cosine decay: afterwards the LR glides from the peak down to min_lr along half a
    cosine wave, 0.5 * (1 + cos(pi * progress)), which falls slowly at first, fastest
    in the middle and slowly again at the end.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, num_steps - cfg.warmup_steps)  # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))  # 1 -> 0
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def evaluate(model, cfg: TrainConfig, block_size, device, device_type, amp_dtype) -> float:
    """Mean loss over `eval_batches` validation micro-batches.

    A fresh loader with a fixed seed is used each time, so every validation looks at
    the SAME windows and successive validation losses are directly comparable.
    """
    loader = TokenLoader(cfg.data_dir, "validation", cfg.batch_size, block_size, seed=cfg.seed + 1)
    model.eval()  # turn dropout off
    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y = (t.to(device) for t in loader.get_batch())
        with torch.autocast(device_type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / cfg.eval_batches


def pretrain(model_config: GPTConfig, cfg: TrainConfig) -> PretrainResult:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_dtype, use_scaler = resolve_precision(device_type, cfg.dtype)

    seed_everything(cfg.seed)
    model = GPT(model_config).to(device)
    model.train()

    train_loader = TokenLoader(cfg.data_dir, "train", cfg.batch_size, model_config.block_size, cfg.seed)

    # Token budget. One optimizer step consumes batch_size * grad_accum_steps sequences
    # of block_size tokens. The cap fixes the number of steps, which the LR schedule needs.
    tokens_per_step = cfg.batch_size * cfg.grad_accum_steps * model_config.block_size
    max_tokens = cfg.max_tokens
    if max_tokens is None:
        max_tokens = sum(len(s) for s in train_loader.shards)  # one pass over the train split
    num_steps = max_tokens // tokens_per_step
    if num_steps < 1:
        raise ValueError(
            f"token cap {max_tokens} is smaller than one optimizer step ({tokens_per_step} tokens)")

    # Log the complete configuration with the run, so any result can be traced to its settings.
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps({
        "model": asdict(model_config),
        "train": asdict(cfg),
        "device": device,
        "autocast_dtype": str(amp_dtype),
        "grad_scaler": use_scaler,
        "tokens_per_step": tokens_per_step,
        "max_tokens": max_tokens,
        "num_steps": num_steps,
        "torch_version": torch.__version__,
    }, indent=2))

    optimizer = configure_optimizer(
        model, cfg.learning_rate, cfg.weight_decay, (cfg.beta1, cfg.beta2), device_type)
    # With enabled=False the scaler is a transparent pass-through (bf16 / fp32 runs).
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)

    train_log, eval_log = [], []
    for step in range(num_steps):
        # 1. Learning rate for this step.
        lr = get_lr(step, cfg, num_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. Accumulate gradients over several micro-batches.
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = (t.to(device) for t in train_loader.get_batch())  # each (batch_size, T)
            with torch.autocast(device_type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss = model(x, y)
            # backward() SUMS gradients into .grad. To end up with the gradient of the
            # AVERAGE loss over all micro-batches (what one big batch would give), each
            # micro-batch loss is divided by the number of micro-batches first.
            loss = loss / cfg.grad_accum_steps
            step_loss += loss.item()
            scaler.scale(loss).backward()

        # 3. Clip, then step. Gradient clipping rescales the whole gradient when its
        #    norm exceeds grad_clip, so one freak batch cannot throw the weights far off.
        #    With fp16 the gradients must be un-scaled before their norm means anything.
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        scaler.step(optimizer)
        scaler.update()

        done = step + 1
        train_log.append({
            "step": done, "loss": step_loss, "lr": lr, "grad_norm": grad_norm,  # norm BEFORE clipping
            "tokens_seen": done * tokens_per_step,
        })
        if done % cfg.log_interval == 0 or done == num_steps:
            print(f"step {done}/{num_steps} | loss {step_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.2f}")

        # 4. Periodic validation, and one at the very end.
        if done % cfg.eval_interval == 0 or done == num_steps:
            val_loss = evaluate(model, cfg, model_config.block_size, device, device_type, amp_dtype)
            eval_log.append({"step": done, "val_loss": val_loss})
            print(f"step {done}/{num_steps} | val_loss {val_loss:.4f} | val_ppl {math.exp(val_loss):.2f}")

    return PretrainResult(
        model=model, train_log=train_log, eval_log=eval_log, tokens_per_step=tokens_per_step,
        tokens_seen=num_steps * tokens_per_step, config_path=str(config_path))


PRESETS = {"gpt2_small": gpt2_small, "small": small}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain GPT-2 on prepared TinyStories shards.")
    parser.add_argument("--preset", choices=PRESETS, default="small")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=TrainConfig.grad_accum_steps)
    parser.add_argument("--learning_rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()
    pretrain(PRESETS[args.preset](), TrainConfig(
        data_dir=args.data_dir, out_dir=args.out_dir, max_tokens=args.max_tokens,
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate, seed=args.seed))
