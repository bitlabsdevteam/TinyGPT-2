"""LayerNorm Section: normalise each token's vector to zero mean and unit variance.

What it is
    For one token's vector x of width C, LayerNorm computes

        y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

    mean and var are taken over the C features of that single token (not over the
    batch), so it behaves identically in training and generation.
    gamma ("weight") and beta ("bias") are learned: they let the model undo the
    normalisation wherever that helps.

Why GPT-2 uses it, and where
    Activations in a deep residual network can drift in scale from layer to layer,
    which makes training unstable. Normalising keeps them in a healthy range.
    GPT-2 uses *pre-norm*: LayerNorm is applied to the INPUT of each attention
    and MLP sub-layer (ln_1, ln_2), plus once more at the very end (ln_f). The
    residual stream itself is left un-normalised, giving gradients a clean
    path straight through the whole network, which is what makes 12+ layers
    train reliably.

Written out by hand (instead of nn.LayerNorm) so the math is visible. The tests
check it against torch.nn.LayerNorm.
"""
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # keeps the division stable when the variance is ~0
        # gamma: per-feature scale, starts at 1 so the layer begins as pure normalisation.
        # Named `weight`/`bias` to match Hugging Face GPT-2.
        self.weight = nn.Parameter(torch.ones(ndim))  # (C,)
        # beta: per-feature shift, starts at 0. Optional, as in the config's `bias` flag.
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None  # (C,) or None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Statistics are per token, over the last (feature) axis.
        mean = x.mean(dim=-1, keepdim=True)  # (B, T, C) -> (B, T, 1)
        # Biased variance (divide by C, not C-1), matching nn.LayerNorm.
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (B, T, C) -> (B, T, 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)  # (B, T, C): zero mean, unit variance
        y = x_hat * self.weight  # (B, T, C) * (C,) broadcasts over B and T
        if self.bias is not None:
            y = y + self.bias
        return y  # (B, T, C)
