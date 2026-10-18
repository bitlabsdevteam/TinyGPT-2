"""Seam: checkpointing and resuming through the pretraining entry point (ticket 08)."""
import numpy as np
import pytest
import torch

from data.prepare import write_shards
from model.config import GPTConfig
from train.checkpoint import latest_checkpoint
from train.pretrain import TrainConfig, pretrain

# Dropout is ON: a resumed run can only match an uninterrupted one if the random
# number generators that drive dropout are restored exactly.
MODEL = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=64, dropout=0.1)


@pytest.fixture
def data(tmp_path):
    tokens = np.tile(np.arange(10), 400)
    write_shards(iter([tokens]), tmp_path / "data", "train", shard_size=10_000)
    write_shards(iter([tokens]), tmp_path / "data", "validation", shard_size=10_000)
    return tmp_path


def run(root, model=MODEL, **overrides):
    cfg = dict(
        data_dir=str(root / "data"), out_dir=str(root / "run"), batch_size=2, grad_accum_steps=2,
        max_tokens=20 * 2 * 2 * MODEL.block_size,  # 20 optimizer steps
        learning_rate=3e-3, min_lr=3e-4, warmup_steps=3, eval_interval=6, eval_batches=2,
        log_interval=1000, checkpoint_interval=5, seed=0, device="cpu",
    )
    cfg.update(overrides)
    return pretrain(model, TrainConfig(**cfg))


def losses(result):
    return [r["loss"] for r in result.train_log]


def same_weights(a, b):
    sa, sb = a.model.state_dict(), b.model.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_a_resumed_run_reproduces_an_uninterrupted_run_exactly(data):
    full = run(data, out_dir=str(data / "full"))

    paused = run(data, out_dir=str(data / "resumed"), stop_after_step=8)
    assert paused.start_step == 0 and len(paused.train_log) == 8

    resumed = run(data, out_dir=str(data / "resumed"))  # same call, nothing else
    assert resumed.start_step == 8  # it continued, it did not start over
    assert losses(resumed) == losses(full) and len(resumed.train_log) == 20
    assert resumed.eval_log == full.eval_log
    assert [r["lr"] for r in resumed.train_log] == [r["lr"] for r in full.train_log]
    assert same_weights(resumed, full)


def test_resuming_after_a_crash_restarts_from_the_last_interval_checkpoint(data):
    full = run(data, out_dir=str(data / "full"))
    run(data, out_dir=str(data / "crashed"), stop_after_step=8)  # writes checkpoints 5 and 8
    ckpt_dir = data / "crashed" / "checkpoints"
    (ckpt_dir / "ckpt_00000008.pt").unlink()  # the session died before step 8 was saved

    resumed = run(data, out_dir=str(data / "crashed"))
    assert resumed.start_step == 5
    assert losses(resumed) == losses(full)
    assert same_weights(resumed, full)


def test_a_finished_run_resumes_as_a_no_op_that_returns_the_trained_model(data):
    first = run(data)
    again = run(data)
    assert again.start_step == 20
    assert losses(again) == losses(first)
    assert same_weights(again, first)


def test_latest_checkpoint_is_found_by_step_number_and_ignores_partial_files(tmp_path):
    assert latest_checkpoint(tmp_path / "missing") is None
    tmp_path.mkdir(exist_ok=True)
    assert latest_checkpoint(tmp_path) is None
    for name in ["ckpt_00000005.pt", "ckpt_00000020.pt", "ckpt_00000100.pt",
                 "ckpt_00000150.pt.tmp", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    assert latest_checkpoint(tmp_path).name == "ckpt_00000100.pt"


def test_checkpoints_go_to_the_configured_location(data):
    drive = data / "drive" / "my_checkpoints"
    run(data, checkpoint_dir=str(drive))
    assert latest_checkpoint(drive) is not None
    assert not (data / "run" / "checkpoints").exists()


def test_default_checkpoint_location_is_inside_the_run_folder(data):
    run(data)
    assert latest_checkpoint(data / "run" / "checkpoints") is not None


def test_only_the_most_recent_checkpoints_are_kept(data):
    run(data, keep_last=2)  # interval 5 over 20 steps: 5, 10, 15, 20
    names = sorted(p.name for p in (data / "run" / "checkpoints").glob("ckpt_*.pt"))
    assert names == ["ckpt_00000015.pt", "ckpt_00000020.pt"]


def test_resuming_with_a_different_model_is_refused(data):
    run(data, stop_after_step=5)
    other = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.1)
    with pytest.raises(ValueError):
        run(data, model=other)


def test_resume_can_be_switched_off(data):
    run(data, stop_after_step=8)
    fresh = run(data, resume=False)
    assert fresh.start_step == 0 and len(fresh.train_log) == 20


def test_rerunning_a_paused_run_with_the_same_pause_point_does_nothing(data):
    first = run(data, stop_after_step=8)
    again = run(data, stop_after_step=8)  # e.g. the same Colab cell executed twice
    assert again.start_step == 8
    assert len(again.train_log) == 8  # no extra training happened
    assert again.tokens_seen == first.tokens_seen
    assert same_weights(again, first)
    # Raising the pause point continues from where it stopped.
    further = run(data, stop_after_step=12)
    assert further.start_step == 8 and len(further.train_log) == 12
