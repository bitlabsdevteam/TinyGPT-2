"""Seam: the pretraining run entry point, on a tiny config on CPU (ticket 07)."""
import json
import math

import numpy as np
import pytest
import torch

from data.prepare import write_shards
from model.config import GPTConfig
from model.gpt2 import GPT
from train.pretrain import (
    TrainConfig, configure_optimizer, pretrain, resolve_precision, seed_everything,
)

MODEL = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=64)


def make_data(tmp_path, tokens=None):
    """Train and validation shards. Default: the pattern 0,1,...,9,0,1,... (learnable)."""
    if tokens is None:
        tokens = np.tile(np.arange(10), 400)
    write_shards(iter([tokens]), tmp_path, "train", shard_size=10_000)
    write_shards(iter([tokens]), tmp_path, "validation", shard_size=10_000)
    return tmp_path


def run(tmp_path, steps=20, **overrides):
    """Run `steps` optimizer steps (token cap = steps x tokens per step)."""
    cfg = dict(
        data_dir=str(tmp_path), out_dir=str(tmp_path / "run"), batch_size=2, grad_accum_steps=2,
        learning_rate=3e-3, min_lr=3e-4, warmup_steps=5, eval_interval=10, eval_batches=2,
        log_interval=1000, seed=0, device="cpu",
        sample_prompts=(), profile=False,  # covered by tests/test_profile.py
    )
    cfg.update(overrides)
    cfg.setdefault("max_tokens", steps * cfg["batch_size"] * cfg["grad_accum_steps"] * MODEL.block_size)
    return pretrain(MODEL, TrainConfig(**cfg))


def losses(result):
    return [r["loss"] for r in result.train_log]


def test_loss_decreases_over_a_short_run(tmp_path):
    make_data(tmp_path)
    result = run(tmp_path, steps=60)
    first, last = np.mean(losses(result)[:5]), np.mean(losses(result)[-5:])
    assert last < 0.5 * first


def test_same_seed_gives_the_same_losses_and_a_different_seed_does_not(tmp_path):
    make_data(tmp_path)
    a = run(tmp_path, steps=10, out_dir=str(tmp_path / "a"))
    b = run(tmp_path, steps=10, out_dir=str(tmp_path / "b"))
    c = run(tmp_path, steps=10, out_dir=str(tmp_path / "c"), seed=1)
    assert losses(a) == losses(b)
    assert a.eval_log == b.eval_log
    assert losses(a) != losses(c)


def test_accumulation_gives_the_same_gradient_as_one_big_batch(tmp_path):
    # Every window of a constant token stream is identical, so ANY way of splitting the
    # same number of rows must give the same gradient, as long as the loss is averaged.
    make_data(tmp_path, tokens=np.full(5000, 7))
    big = run(tmp_path, steps=3, batch_size=8, grad_accum_steps=1, out_dir=str(tmp_path / "big"))
    acc = run(tmp_path, steps=3, batch_size=2, grad_accum_steps=4, out_dir=str(tmp_path / "acc"))
    assert [r["grad_norm"] for r in acc.train_log] == pytest.approx(
        [r["grad_norm"] for r in big.train_log], rel=1e-4)
    assert losses(acc) == pytest.approx(losses(big), rel=1e-4)


def test_effective_batch_is_micro_batch_times_accumulation(tmp_path):
    make_data(tmp_path)
    result = run(tmp_path, steps=4, batch_size=3, grad_accum_steps=5)
    assert result.tokens_per_step == 3 * 5 * MODEL.block_size


def test_token_cap_stops_the_run(tmp_path):
    make_data(tmp_path)
    per_step = 2 * 2 * MODEL.block_size  # 64 tokens
    result = run(tmp_path, max_tokens=per_step * 7 + 10)  # room for 7 full steps, not 8
    assert len(result.train_log) == 7
    assert result.tokens_seen == 7 * per_step <= per_step * 7 + 10


def test_a_token_cap_smaller_than_one_step_is_rejected(tmp_path):
    make_data(tmp_path)
    with pytest.raises(ValueError):
        run(tmp_path, max_tokens=10)


def test_without_a_token_cap_the_run_is_one_pass_over_the_train_split(tmp_path):
    make_data(tmp_path, tokens=np.tile(np.arange(10), 64))  # 640 tokens
    result = pretrain(MODEL, TrainConfig(
        data_dir=str(tmp_path), out_dir=str(tmp_path / "run"), batch_size=2, grad_accum_steps=1,
        eval_interval=1000, log_interval=1000, device="cpu", sample_prompts=(), profile=False))
    assert len(result.train_log) == 640 // (2 * MODEL.block_size)


def test_full_config_is_logged_with_the_run(tmp_path):
    make_data(tmp_path)
    result = run(tmp_path, steps=3, seed=123, learning_rate=0.003)
    logged = json.loads(open(result.config_path).read())
    assert logged["model"]["n_layer"] == MODEL.n_layer
    assert logged["model"]["n_embd"] == MODEL.n_embd
    assert logged["train"]["seed"] == 123
    assert logged["train"]["learning_rate"] == 0.003
    assert logged["tokens_per_step"] == 2 * 2 * MODEL.block_size
    assert logged["num_steps"] == 3
    assert logged["device"] == "cpu"


def test_weight_decay_applies_only_to_matrices(tmp_path):
    model = GPT(MODEL)
    opt = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.1, betas=(0.9, 0.95),
                              device_type="cpu")
    decayed = [g for g in opt.param_groups if g["weight_decay"] == 0.1]
    plain = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    assert len(decayed) == 1 and len(plain) == 1
    assert all(p.dim() >= 2 for p in decayed[0]["params"])  # weights, embeddings
    assert all(p.dim() < 2 for p in plain[0]["params"])  # biases, LayerNorm gains
    # Every parameter is in exactly one group.
    n_grouped = sum(len(g["params"]) for g in opt.param_groups)
    assert n_grouped == len(list(model.parameters()))


def test_learning_rate_warms_up_then_follows_a_cosine_to_the_minimum(tmp_path):
    make_data(tmp_path)
    result = run(tmp_path, steps=40, warmup_steps=5, learning_rate=3e-3, min_lr=3e-4)
    lr = [r["lr"] for r in result.train_log]
    assert lr[0] == pytest.approx(3e-3 / 5)  # first warmup step: 1/5 of the peak
    assert lr[4] == pytest.approx(3e-3)  # end of warmup: the peak
    assert lr[5] == pytest.approx(3e-3)  # cosine starts at the peak
    assert all(a >= b for a, b in zip(lr[5:], lr[6:]))  # then only decreases
    assert lr[-1] == pytest.approx(3e-4, rel=0.02)  # and ends at the minimum
    midpoint = lr[5 + (40 - 5) // 2]
    assert midpoint == pytest.approx((3e-3 + 3e-4) / 2, rel=0.05)  # half-way = mean of peak and min


def test_validation_runs_at_the_interval_and_at_the_end(tmp_path):
    make_data(tmp_path)
    result = run(tmp_path, steps=25, eval_interval=10)
    assert [e["step"] for e in result.eval_log] == [10, 20, 25]
    assert all(math.isfinite(e["val_loss"]) for e in result.eval_log)
    assert result.eval_log[-1]["val_loss"] < result.eval_log[0]["val_loss"]


def test_gradient_clipping_limits_how_far_the_weights_move(tmp_path):
    make_data(tmp_path)
    seed_everything(0)
    start = {k: v.clone() for k, v in GPT(MODEL).state_dict().items()}

    def moved(grad_clip):
        res = run(tmp_path, steps=3, grad_clip=grad_clip, out_dir=str(tmp_path / f"c{grad_clip}"))
        return sum((res.model.state_dict()[k] - start[k]).pow(2).sum() for k in start).sqrt().item()

    # A clip threshold so tiny that gradients fall far below Adam's epsilon makes steps ~0.
    assert moved(1e-9) < 0.01 * moved(1.0)


def test_logged_grad_norm_is_the_norm_before_clipping(tmp_path):
    make_data(tmp_path)
    result = run(tmp_path, steps=2, grad_clip=1e-9)
    assert all(r["grad_norm"] > 1e-6 for r in result.train_log)


def test_precision_choice_follows_the_hardware():
    assert resolve_precision("cuda", "auto", bf16_supported=True) == (torch.bfloat16, False)
    assert resolve_precision("cuda", "auto", bf16_supported=False) == (torch.float16, True)
    assert resolve_precision("cpu", "auto") == (None, False)
    assert resolve_precision("cuda", "fp32", bf16_supported=True) == (None, False)
