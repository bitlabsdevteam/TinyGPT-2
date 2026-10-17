"""Serve (input, target) batches of token windows from prepared shards.

Next-token prediction
    The model reads tokens t0..t(T-1) and must predict t1..tT. So from a window of
    T+1 consecutive tokens we form

        x = window[:-1]    (T tokens in)
        y = window[1:]     (the same tokens shifted left by one: what comes next)

    and the loss at position i compares the model's output for x[i] with y[i].

Sampling
    Each batch picks batch_size random windows from random places in the data
    (uniform over all valid positions, with replacement). Over a run that sees about
    as many tokens as the dataset holds, that is roughly one epoch.
    Randomness comes from a private torch.Generator seeded by `seed`, so the same
    seed always yields the same sequence of batches and nothing else (model init,
    dropout) disturbs it.

Shards are memory-mapped: only the windows actually used are read from disk.
"""
import numpy as np
import torch

from data.prepare import shard_paths


class TokenLoader:
    def __init__(self, data_dir, split: str, batch_size: int, block_size: int, seed: int):
        self.batch_size = batch_size
        self.block_size = block_size
        self.shards = [np.load(p, mmap_mode="r") for p in shard_paths(data_dir, split)]

        # A window needs block_size + 1 tokens, so a shard of n tokens offers
        # n - block_size possible start positions (0 if it is too short).
        self.n_starts = [max(0, len(s) - block_size) for s in self.shards]
        if sum(self.n_starts) == 0:
            raise ValueError(f"no shard holds block_size + 1 = {block_size + 1} tokens")

        self.generator = torch.Generator().manual_seed(seed)

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y), both (batch_size, block_size) int64; y is x shifted by one token."""
        B, T = self.batch_size, self.block_size
        # Choose a shard for each row in proportion to how many windows it holds,
        # so every window position in the dataset is equally likely.
        weights = torch.tensor(self.n_starts, dtype=torch.float)
        shard_ids = torch.multinomial(weights, B, replacement=True, generator=self.generator)

        xs, ys = [], []
        for sid in shard_ids.tolist():
            start = torch.randint(0, self.n_starts[sid], (1,), generator=self.generator).item()
            window = np.asarray(self.shards[sid][start:start + T + 1]).astype(np.int64)  # (T+1,)
            xs.append(window[:-1])  # (T,)
            ys.append(window[1:])  # (T,)
        return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))  # each (B, T)
