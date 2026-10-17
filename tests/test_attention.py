"""Seam: the attention Section's forward pass (ticket 02)."""
import pytest
import torch
from torch import nn

from model.attention import CausalSelfAttention
from model.config import gpt2_small, small


@pytest.mark.parametrize("make_config", [gpt2_small, small])
def test_output_shape_equals_input_shape(make_config):
    cfg = make_config()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 16, cfg.n_embd)  # (B, T, C)
    assert attn(x).shape == x.shape


def test_changing_a_future_position_leaves_earlier_outputs_unchanged():
    torch.manual_seed(0)
    cfg = small()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(1, 12, cfg.n_embd)
    changed_from = 7
    x2 = x.clone()
    x2[:, changed_from:] = torch.randn(1, 12 - changed_from, cfg.n_embd)

    y, y2 = attn(x), attn(x2)

    # Positions before the change cannot see it: outputs identical.
    assert torch.allclose(y[:, :changed_from], y2[:, :changed_from], atol=1e-6)
    # Guard against a vacuous test: the changed positions do differ.
    assert not torch.allclose(y[:, changed_from:], y2[:, changed_from:], atol=1e-3)


def test_projections_are_linear_layers_named_like_hugging_face_gpt2():
    cfg = small()
    attn = CausalSelfAttention(cfg)
    assert isinstance(attn.c_attn, nn.Linear)
    assert isinstance(attn.c_proj, nn.Linear)
    # c_attn produces Q, K and V in one matmul, so it is 3x as wide.
    assert attn.c_attn.weight.shape == (3 * cfg.n_embd, cfg.n_embd)
    assert attn.c_proj.weight.shape == (cfg.n_embd, cfg.n_embd)
