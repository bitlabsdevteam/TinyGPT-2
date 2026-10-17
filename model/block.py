"""Transformer Block: one attention step followed by one MLP step, with residuals.

What it is

        x = x + attn( ln_1(x) )      # tokens exchange information
        x = x + mlp ( ln_2(x) )      # each token is processed on its own

Why GPT-2 uses it
    * Residual connections ("x + ..."): each sub-layer only has to learn a small
      correction to x instead of a whole new representation, and gradients flow
      straight back through the "+" to early layers. This is what lets many
      Blocks be stacked.
    * Pre-norm placement: LayerNorm sits inside the residual branch, before
      attention/MLP, so the residual stream x itself is never normalised. (The
      original Transformer put the norm after the "+", "post-norm", which is
      harder to train when deep.)

A GPT-2 model is `n_layer` of these Blocks in a row. Their attribute names
(ln_1, attn, ln_2, mlp) match Hugging Face GPT-2.
"""
import torch
from torch import nn

from model.attention import CausalSelfAttention
from model.config import GPTConfig
from model.layernorm import LayerNorm
from model.mlp import MLP


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))  # (B, T, C) -> (B, T, C)
        x = x + self.mlp(self.ln_2(x))  # (B, T, C) -> (B, T, C)
        return x
