"""Profiling: measure how a run behaves, and write the Profile report.

A Profile report answers two questions about a training run:
    * Is the model learning?   loss curves, perplexity, sample generations
    * Is the hardware used well?   tokens/sec, step time, peak memory, MFU, a profiler trace

The report is plain files in the run folder, so it survives Colab sessions and can be
opened later:
    profile_report.json    every number below, plus the samples and notes
    loss_curves.png        train/validation loss and validation perplexity
    profile/*.json         torch.profiler trace (Chrome tracing and TensorBoard)

The same functions serve the pretraining run and, later, the fine-tuning run.
"""
import json
import math
import re
import statistics
from pathlib import Path

import tiktoken
import torch
from matplotlib.figure import Figure
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from train.checkpoint import capture_rng_state, restore_rng_state


# ---------------------------------------------------------------------------
# MFU: Model FLOPs Utilization
# ---------------------------------------------------------------------------
# MFU = (FLOPs the model needs per second, counted in theory) / (the GPU's peak FLOPs/s).
# 100% would mean every cycle of the GPU does useful matrix maths. Well-tuned LLM training
# reaches ~40-60% on an A100; a simple PyTorch loop typically lands lower, so the number
# tells you how much speed is left on the table.

def estimate_flops_per_token(model) -> int:
    """Theoretical FLOPs to train on ONE token (forward + backward).

        6 * N  +  12 * n_layer * n_embd * block_size

    * 6 * N: every parameter takes part in one multiply-add (2 FLOPs) per token in the
      forward pass, and about twice that in the backward pass: 2 + 4 = 6 FLOPs.
      N counts the weights that are used in matrix multiplications. The position table
      is only a lookup, so it is left out; the token embedding stays in because it is
      tied to the output layer, which does multiply by it.
    * 12 * n_layer * n_embd * block_size: the attention scores (Q Kᵀ) and the weighted
      sum over V cost FLOPs that grow with the sequence length but are not parameters.
    (The PaLM paper's accounting, as also used in nanoGPT.)
    """
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    n_params -= model.embeddings.wpe.weight.numel()
    return 6 * n_params + 12 * cfg.n_layer * cfg.n_embd * cfg.block_size


# Peak dense 16-bit tensor-core throughput (FLOPs/s) of the GPUs Colab hands out, from the
# vendors' datasheets (without the "with sparsity" doubling). Word boundaries keep "L4" from
# matching an "L40". For any GPU not listed the peak is unknown and MFU is reported as
# null; pass TrainConfig.peak_tflops to supply it.
_PEAK_FLOPS = [
    (r"H100", 989e12),
    (r"A100", 312e12),
    (r"\bL4\b", 121e12),
    (r"\bT4\b", 65e12),
]


def peak_flops_for_gpu_name(name: str) -> float | None:
    for pattern, flops in _PEAK_FLOPS:
        if re.search(pattern, name):
            return flops
    return None


def peak_flops(device_type: str, amp_dtype, override_tflops: float | None = None) -> float | None:
    """Peak FLOPs/s of the device in use, or None if unknown."""
    if override_tflops is not None:
        return override_tflops * 1e12
    # The table holds 16-bit figures, which only apply when the run uses bf16/fp16.
    if device_type != "cuda" or amp_dtype is None:
        return None
    return peak_flops_for_gpu_name(torch.cuda.get_device_name())


def mfu(tokens_per_sec: float, flops_per_token: float, peak: float | None) -> float | None:
    """Achieved FLOPs/s divided by the hardware peak; None when the peak is unknown."""
    if peak is None:
        return None
    return tokens_per_sec * flops_per_token / peak


def peak_memory_bytes(device_type: str) -> int | None:
    """Highest GPU memory held by tensors during the run, or None without a GPU."""
    if device_type != "cuda":
        return None
    return torch.cuda.max_memory_allocated()


# ---------------------------------------------------------------------------
# torch.profiler trace
# ---------------------------------------------------------------------------

class _NoProfiler:
    """Stands in for torch.profiler when profiling is switched off."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def step(self):
        pass


def make_profiler(enabled: bool, trace_dir, wait: int, warmup: int, active: int, device_type: str):
    """A torch.profiler that records a few steps and writes a trace file.

    Profiling every step would slow training and produce enormous files, so a schedule
    picks a short window; call .step() once per optimizer step:
        wait    steps ignored (the first steps are unrepresentative: allocator warm-up, ...)
        warmup  steps profiled but thrown away (the profiler's own start-up cost)
        active  steps recorded and saved
    The trace is Chrome-trace JSON: open it in chrome://tracing or Perfetto
    (ui.perfetto.dev), or in TensorBoard with the torch-tb-profiler plugin.
    """
    if not enabled:
        return _NoProfiler()
    activities = [ProfilerActivity.CPU]
    if device_type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    return profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
    )


# ---------------------------------------------------------------------------
# Sample generations
# ---------------------------------------------------------------------------

def generate_samples(model, prompts, step: int, max_new_tokens: int, temperature: float,
                     top_k: int | None, seed: int, device) -> list[dict]:
    """Complete each prompt with the current model: a qualitative check on top of the loss.

    Loss says how well the model predicts; reading what it writes shows whether it has
    learned grammar, names, story structure.

    Sampling draws random numbers, which would shift the stream that drives dropout and
    data order and make a run differ from one that skipped sampling (and break exact
    resume). So the global random state is saved first, a fixed seed makes the sample
    reproducible, and the state is restored afterwards.
    """
    enc = tiktoken.get_encoding("gpt2")
    saved = capture_rng_state()
    samples = []
    try:
        for prompt in prompts:
            torch.manual_seed(seed)
            ids = torch.tensor([enc.encode_ordinary(prompt)], device=device)  # (1, T_prompt)
            out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k)
            samples.append({"step": step, "prompt": prompt, "text": enc.decode(out[0].tolist())})
    finally:
        restore_rng_state(saved)
    return samples


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def plot_loss_curves(train_log, eval_log, path) -> None:
    """Left: training loss per step and validation loss. Right: validation perplexity.

    Perplexity = exp(loss): the number of equally likely tokens the model is, on average,
    choosing between. A perfect model has perplexity 1; one guessing uniformly over the
    50,257-token vocabulary has 50,257. It is the same information as the loss on a scale
    that is easier to picture, so it gets its own (logarithmic) axis.
    """
    fig = Figure(figsize=(11, 4))
    loss_ax, ppl_ax = fig.subplots(1, 2)

    loss_ax.plot([r["step"] for r in train_log], [r["loss"] for r in train_log],
                 label="train", alpha=0.7)
    if eval_log:
        loss_ax.plot([e["step"] for e in eval_log], [e["val_loss"] for e in eval_log],
                     "o-", label="validation")
        ppl_ax.plot([e["step"] for e in eval_log], [math.exp(e["val_loss"]) for e in eval_log], "o-")
    loss_ax.set(xlabel="optimizer step", ylabel="cross-entropy loss", title="Loss")
    loss_ax.legend()
    loss_ax.grid(alpha=0.3)
    ppl_ax.set(xlabel="optimizer step", ylabel="perplexity", title="Validation perplexity",
               yscale="log")
    ppl_ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=110)


def write_profile_report(out_dir, *, train_log, eval_log, samples, tokens_per_step,
                         flops_per_token, peak, peak_memory, device, dtype_name, trace_dir) -> dict:
    """Assemble, save and return the Profile report. See the module docstring for the files."""
    out_dir = Path(out_dir)
    notes = []

    # Throughput. The first step is left out: it includes one-off start-up costs (kernel
    # selection, memory allocation) that say nothing about steady-state speed.
    times = [r["step_time_s"] for r in train_log]
    steady = times[1:] if len(times) > 1 else times
    tokens_per_sec = tokens_per_step * len(steady) / sum(steady) if steady else None
    achieved = tokens_per_sec * flops_per_token if tokens_per_sec else None  # achieved FLOPs/s
    run_mfu = mfu(tokens_per_sec, flops_per_token, peak) if tokens_per_sec else None

    # Losses. The training loss of one step is noisy, so the last (up to) 10 are averaged.
    recent = [r["loss"] for r in train_log[-10:]]
    train_loss = sum(recent) / len(recent) if recent else None
    val_loss = eval_log[-1]["val_loss"] if eval_log else None

    if peak_memory is None:
        notes.append("peak memory: not measured because no CUDA device is in use")
    if run_mfu is None:
        notes.append(f"MFU: peak FLOPs of this device ({device}, {dtype_name}) are unknown; "
                     "set TrainConfig.peak_tflops to enable it")

    summary = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_perplexity": math.exp(train_loss) if train_loss is not None else None,
        "val_perplexity": math.exp(val_loss) if val_loss is not None else None,
        "tokens_per_sec": tokens_per_sec,
        "step_time_s": statistics.median(steady) if steady else None,
        "peak_memory_bytes": peak_memory,
        "peak_memory_gb": peak_memory / 1e9 if peak_memory is not None else None,
        "flops_per_token": flops_per_token,
        "achieved_tflops": achieved / 1e12 if achieved is not None else None,
        "mfu": run_mfu,
        "num_steps": len(train_log),
        "tokens_seen": train_log[-1]["tokens_seen"] if train_log else 0,
        "device": device,
        "dtype": dtype_name,
    }

    curves = out_dir / "loss_curves.png"
    plot_loss_curves(train_log, eval_log, curves)
    report = {
        "summary": summary,
        "samples": samples,
        "notes": notes,
        "files": {
            "loss_curves": str(curves),
            "trace_dir": str(trace_dir) if Path(trace_dir).exists() else None,
        },
    }
    (out_dir / "profile_report.json").write_text(json.dumps(report, indent=2))
    return report
