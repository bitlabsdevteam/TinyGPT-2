"""MLP Section: the position-wise feed-forward network inside every Block.

What it is
    Two Linear layers with a GELU in between, applied to every token separately:

        MLP(x) = dropout( W2 · GELU(W1 · x + b1) + b2 )

    The hidden layer is 4x wider than the embedding: C -> 4C -> C.

Why GPT-2 uses it
    Attention moves information BETWEEN tokens but is mostly a weighted average,
    i.e. linear in the values. The MLP supplies the non-linear computation done
    WITHIN each token, and holds most of the model's parameters (about two thirds
    of each Block), which is where much of the model's factual "memory" lives.

GELU
    A smooth version of ReLU: GELU(x) = x * Phi(x), with Phi the normal CDF.
    GPT-2 uses the tanh approximation of it ("gelu_new" in Hugging Face), so we
    use approximate="tanh" to match its numbers exactly.
"""
import torch
from torch import nn

from model.config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Expand: C -> 4C. Named `c_fc` ("fully connected") like Hugging Face GPT-2.
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # tanh-approximated GELU, to match GPT-2.
        self.gelu = nn.GELU(approximate="tanh")
        # Project back: 4C -> C. Named `c_proj`, like the attention output projection.
        # Its weights get the smaller residual init (see GPT._init_weights).
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)  # (B, T, C) -> (B, T, 4C)
        x = self.gelu(x)  # (B, T, 4C) -> (B, T, 4C)
        x = self.c_proj(x)  # (B, T, 4C) -> (B, T, C)
        return self.dropout(x)  # (B, T, C)
