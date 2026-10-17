"""Embeddings Section: turn token ids into the vectors the Blocks work on.

What it is
    Two lookup tables, added together:

        wte (token embedding):      token id  -> vector   (vocab_size, C)
        wpe (positional embedding): position  -> vector   (block_size, C)

        x = dropout( wte[token_ids] + wpe[positions] )

Why GPT-2 uses it
    Neural networks need vectors, not integer ids. wte gives each of the 50,257
    BPE tokens its own learned vector. But attention by itself ignores order
    (shuffle the tokens and each token still sees the same set), so the model also
    needs to be told WHERE each token sits. GPT-2 does this with a *learned*
    positional table: one trainable vector per position 0..block_size-1. (The
    original Transformer used fixed sine waves instead.) Adding the two vectors
    puts "what" and "where" into one signal.

    wte is also reused as the output layer (weight tying), see output_head.py.
"""
import torch
from torch import nn

from model.config import GPTConfig


class Embeddings(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Names `wte` and `wpe` match Hugging Face GPT-2.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # (vocab_size, C)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # (block_size, C)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()  # idx: (B, T) integer token ids
        # Positions 0..T-1, the same for every sequence in the batch. Created on
        # the same device as the ids.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        tok_emb = self.wte(idx)  # (B, T) -> (B, T, C)
        pos_emb = self.wpe(pos)  # (T,) -> (T, C)
        # (B, T, C) + (T, C): pos_emb is broadcast across the batch.
        return self.drop(tok_emb + pos_emb)  # (B, T, C)
