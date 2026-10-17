"""Seams: shard writing, TinyStories preparation, and the batch loader (ticket 06)."""
import numpy as np
import pytest
import torch

from data.loader import TokenLoader
from data.prepare import prepare_split, read_shards, write_shards

EOT = 50256  # GPT-2's <|endoftext|> token id
HELLO_WORLD = [31373, 995]  # GPT-2 BPE ids for "hello world" (published values)


def stories_that_must_not_be_read():
    raise AssertionError("stories were read although shards already exist")


# ---- shards ----

def test_story_tokens_are_gpt2_bpe_with_endoftext_after_each_story(tmp_path):
    prepare_split("train", tmp_path, lambda: ["hello world", "hello world"])
    tokens = read_shards(tmp_path, "train")
    assert tokens.tolist() == HELLO_WORLD + [EOT] + HELLO_WORLD + [EOT]
    assert tokens.dtype == np.uint16


def test_shards_are_fixed_size_except_the_last_and_reassemble_in_order(tmp_path):
    stream = [np.arange(0, 7), np.arange(7, 10), np.arange(10, 25)]  # 25 tokens in 3 chunks
    paths = write_shards(iter(stream), tmp_path, "train", shard_size=10)
    sizes = [len(np.load(p)) for p in paths]
    assert sizes == [10, 10, 5]
    assert read_shards(tmp_path, "train").tolist() == list(range(25))


def test_shards_are_re_readable_from_a_fresh_process_view(tmp_path):
    prepare_split("validation", tmp_path, lambda: ["hello world"])
    again = read_shards(tmp_path, "validation")  # only the files on disk are used
    assert again.tolist() == HELLO_WORLD + [EOT]


def test_preparation_is_skipped_when_shards_already_exist(tmp_path):
    first = prepare_split("train", tmp_path, lambda: ["hello world"])
    second = prepare_split("train", tmp_path, stories_that_must_not_be_read)
    assert first == second


def test_unfinished_shards_are_rebuilt_and_stale_ones_removed(tmp_path):
    # A crashed earlier run left shards but no completion marker.
    np.save(tmp_path / "train_00000.npy", np.zeros(3, dtype=np.uint16))
    np.save(tmp_path / "train_00009.npy", np.zeros(3, dtype=np.uint16))
    prepare_split("train", tmp_path, lambda: ["hello world"])
    assert read_shards(tmp_path, "train").tolist() == HELLO_WORLD + [EOT]


def test_splits_are_kept_apart(tmp_path):
    prepare_split("train", tmp_path, lambda: ["hello world"])
    prepare_split("validation", tmp_path, lambda: ["hello world", "hello world"])
    assert len(read_shards(tmp_path, "train")) == 3
    assert len(read_shards(tmp_path, "validation")) == 6


def test_reading_a_split_that_was_never_prepared_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_shards(tmp_path, "train")


# ---- loader ----

def make_counting_shards(tmp_path, n_tokens=1000, shard_size=300):
    write_shards(iter([np.arange(n_tokens)]), tmp_path, "train", shard_size=shard_size)


def test_batches_have_the_requested_shape_and_targets_are_inputs_shifted_by_one(tmp_path):
    make_counting_shards(tmp_path)
    x, y = TokenLoader(tmp_path, "train", batch_size=4, block_size=16, seed=0).get_batch()
    assert x.shape == y.shape == (4, 16)
    assert x.dtype == y.dtype == torch.long
    # The data is 0,1,2,...: the next token is always the current one plus one.
    assert torch.equal(y, x + 1)


def test_same_seed_gives_identical_batches_and_different_seed_does_not(tmp_path):
    make_counting_shards(tmp_path)
    a = TokenLoader(tmp_path, "train", batch_size=8, block_size=16, seed=1)
    b = TokenLoader(tmp_path, "train", batch_size=8, block_size=16, seed=1)
    c = TokenLoader(tmp_path, "train", batch_size=8, block_size=16, seed=2)
    xa, xb, xc = a.get_batch()[0], b.get_batch()[0], c.get_batch()[0]
    assert torch.equal(xa, xb)
    assert not torch.equal(xa, xc)
    assert not torch.equal(a.get_batch()[0], xa)  # successive batches differ


def test_loader_works_when_data_is_exactly_one_window_long(tmp_path):
    write_shards(iter([np.arange(17)]), tmp_path, "train", shard_size=100)
    x, y = TokenLoader(tmp_path, "train", batch_size=2, block_size=16, seed=0).get_batch()
    assert x[0].tolist() == list(range(16)) and y[0].tolist() == list(range(1, 17))


def test_loader_rejects_data_shorter_than_one_window(tmp_path):
    write_shards(iter([np.arange(10)]), tmp_path, "train", shard_size=100)
    with pytest.raises(ValueError):
        TokenLoader(tmp_path, "train", batch_size=2, block_size=16, seed=0)
