"""Output Section: final LayerNorm and the language-model head.

What it is

        logits = lm_head( ln_f(x) )         (B, T, C) -> (B, T, vocab_size)

    ln_f is one last LayerNorm (needed because pre-norm Blocks leave the
    residual stream un-normalised). lm_head is a Linear layer with NO bias that
    maps each token's C-dimensional vector to one score ("logit") per vocabulary
    entry. Softmax over those scores gives the predicted next-token distribution.

Weight tying
    lm_head.weight has shape (vocab_size, C), exactly the shape of the token
    embedding wte. GPT-2 makes them the SAME tensor. Intuition: "which token
    does this vector look like?" is the inverse of "what vector is this token?".
    Tying saves vocab_size * C parameters (~38M of GPT-2 small's 124M) and
    usually improves quality. The tie itself is done in GPT.__init__, where both
    modules are visible; this class only creates the head.
"""
import torch
from torch import nn

from model.config import GPTConfig
from model.layernorm import LayerNorm


class OutputHead(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_f(x)  # (B, T, C) -> (B, T, C)
        return self.lm_head(x)  # (B, T, C) -> (B, T, vocab_size)
