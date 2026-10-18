"""Seam: the Profile report emitted by a pretraining run, and its building blocks (ticket 09)."""
import json
import math

import numpy as np
import pytest

from data.prepare import write_shards
from model.config import GPTConfig, gpt2_small
from model.gpt2 import GPT
from train.pretrain import TrainConfig, pretrain
from train.profiling import estimate_flops_per_token, mfu, peak_flops_for_gpu_name

# Vocabulary of 50257 (tiny otherwise) so that real GPT-2 BPE prompts can be fed to it.
MODEL = GPTConfig(vocab_size=50257, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.1)
PROMPT = "Once upon a time"


@pytest.fixture
def data(tmp_path):
    tokens = np.tile(np.arange(10), 400)
    write_shards(iter([tokens]), tmp_path / "data", "train", shard_size=10_000)
    write_shards(iter([tokens]), tmp_path / "data", "validation", shard_size=10_000)
    return tmp_path


def run(root, steps=12, **overrides):
    cfg = dict(
        data_dir=str(root / "data"), out_dir=str(root / "run"), batch_size=2, grad_accum_steps=1,
        max_tokens=steps * 2 * MODEL.block_size, learning_rate=3e-3, min_lr=3e-4, warmup_steps=3,
        eval_interval=6, eval_batches=2, log_interval=1000, checkpoint_interval=6, seed=0,
        device="cpu", sample_prompts=(PROMPT,), sample_tokens=5,
        profile_wait=1, profile_warmup=1, profile_active=3,
    )
    cfg.update(overrides)
    return pretrain(MODEL, TrainConfig(**cfg))


def report_of(root):
    return json.loads((root / "run" / "profile_report.json").read_text())


# ---- the report ----

def test_report_has_every_required_field_after_a_short_run(data):
    run(data)
    summary = report_of(data)["summary"]
    for field in ["train_loss", "val_loss", "train_perplexity", "val_perplexity", "tokens_per_sec",
                  "step_time_s", "peak_memory_bytes", "mfu", "flops_per_token", "num_steps",
                  "tokens_seen", "device"]:
        assert field in summary, field
    assert math.isfinite(summary["train_loss"]) and math.isfinite(summary["val_loss"])
    assert summary["tokens_per_sec"] > 0 and summary["step_time_s"] > 0
    assert summary["num_steps"] == 12 and summary["tokens_seen"] == 12 * 2 * MODEL.block_size
    # Perplexity is exp(loss): how many equally likely tokens the model is "choosing between".
    assert summary["train_perplexity"] == pytest.approx(math.exp(summary["train_loss"]))
    assert summary["val_perplexity"] == pytest.approx(math.exp(summary["val_loss"]))


def test_peak_memory_and_mfu_degrade_gracefully_on_cpu(data):
    run(data)
    report = report_of(data)
    assert report["summary"]["peak_memory_bytes"] is None  # no CUDA device
    assert report["summary"]["mfu"] is None  # peak FLOPs of a CPU is unknown
    notes = " ".join(report["notes"])
    assert "peak memory" in notes and "MFU" in notes  # and the report says why


def test_profiler_trace_is_written_in_chrome_trace_format(data):
    run(data)
    traces = list((data / "run" / "profile").glob("*.json"))
    assert traces
    assert len(json.loads(traces[0].read_text())["traceEvents"]) > 0


def test_profiling_can_be_switched_off(data):
    run(data, profile=False)
    assert not (data / "run" / "profile").exists()
    assert "summary" in report_of(data)  # the rest of the report is still produced


def test_loss_curves_are_saved_as_an_image(data):
    run(data)
    image = data / "run" / "loss_curves.png"
    assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert image.stat().st_size > 2_000


def test_sample_generations_are_recorded_at_every_checkpoint(data):
    run(data)  # checkpoint interval 6 over 12 steps
    samples = report_of(data)["samples"]
    assert [s["step"] for s in samples] == [6, 12]
    assert all(s["prompt"] == PROMPT and s["text"].startswith(PROMPT) for s in samples)


# ---- sampling must not disturb training ----

def test_generating_samples_does_not_change_training(data):
    with_samples = run(data, out_dir=str(data / "a"))
    without = run(data, out_dir=str(data / "b"), sample_prompts=())
    assert [r["loss"] for r in with_samples.train_log] == [r["loss"] for r in without.train_log]


def test_report_and_samples_of_a_resumed_run_match_an_uninterrupted_run(data):
    run(data, out_dir=str(data / "full"))
    run(data, out_dir=str(data / "res"), stop_after_step=6)
    run(data, out_dir=str(data / "res"))
    full = json.loads((data / "full" / "profile_report.json").read_text())
    resumed = json.loads((data / "res" / "profile_report.json").read_text())
    assert resumed["samples"] == full["samples"]  # saved in the checkpoint, deterministic
    assert resumed["summary"]["num_steps"] == full["summary"]["num_steps"] == 12
    assert resumed["summary"]["train_loss"] == full["summary"]["train_loss"]


# ---- the numbers behind the report (independent, hand-computed expectations) ----

def test_flops_per_token_of_gpt2_small():
    # 6 * N + 12 * n_layer * n_embd * block_size, with N = 124,439,808 parameters minus the
    # position table (786,432): that table is a lookup, not a matrix multiplication.
    n = 124_439_808 - 1024 * 768
    expected = 6 * n + 12 * 12 * 768 * 1024
    assert expected == 855_166_464
    assert estimate_flops_per_token(GPT(gpt2_small())) == expected


def test_mfu_is_achieved_flops_over_the_hardware_peak():
    # 1000 tokens/s * 855,166,464 FLOPs/token = 0.855 TFLOP/s achieved, of 312 TFLOP/s peak.
    assert mfu(1000, 855_166_464, 312e12) == pytest.approx(0.855166464e12 / 312e12)
    assert mfu(1000, 855_166_464, None) is None


@pytest.mark.parametrize("name,peak", [
    ("NVIDIA A100-SXM4-40GB", 312e12),
    ("NVIDIA A100-SXM4-80GB", 312e12),
    ("Tesla T4", 65e12),
    ("NVIDIA L4", 121e12),
    ("NVIDIA H100 80GB HBM3", 989e12),
    ("Some Future GPU", None),
])
def test_peak_flops_of_known_colab_gpus(name, peak):
    assert peak_flops_for_gpu_name(name) == peak
