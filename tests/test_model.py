"""Seams: the model's forward pass, plus one shape test per Section (ticket 03)."""
import math

import pytest
import torch
from torch import nn

from model.block import Block
from model.config import GPTConfig, gpt2_small, small
from model.embeddings import Embeddings
from model.gpt2 import GPT, to_hf_name
from model.layernorm import LayerNorm
from model.mlp import MLP
from model.output_head import OutputHead

TINY = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)


# ---- Section shape tests (the agreed exception to "few seams") ----

def test_embeddings_turns_token_ids_into_vectors():
    cfg = small()
    out = Embeddings(cfg)(torch.randint(0, cfg.vocab_size, (2, 10)))
    assert out.shape == (2, 10, cfg.n_embd)


def test_layernorm_matches_torch_layernorm():
    torch.manual_seed(0)
    ours, ref = LayerNorm(64, bias=True), nn.LayerNorm(64)
    x = torch.randn(2, 5, 64) * 3 + 1
    assert ours(x).shape == x.shape
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def test_layernorm_without_bias_has_no_bias_parameter():
    assert LayerNorm(64, bias=False).bias is None


def test_mlp_keeps_shape_and_expands_four_times_inside():
    cfg = small()
    mlp = MLP(cfg)
    x = torch.randn(2, 7, cfg.n_embd)
    assert mlp(x).shape == x.shape
    assert mlp.c_fc.weight.shape == (4 * cfg.n_embd, cfg.n_embd)


def test_block_keeps_shape():
    cfg = small()
    x = torch.randn(2, 7, cfg.n_embd)
    assert Block(cfg)(x).shape == x.shape


def test_output_head_maps_vectors_to_vocab_logits():
    cfg = small()
    out = OutputHead(cfg)(torch.randn(2, 7, cfg.n_embd))
    assert out.shape == (2, 7, cfg.vocab_size)


# ---- Model seam ----

def test_forward_returns_logits_for_every_position_and_no_loss_without_targets():
    model = GPT(TINY)
    logits, loss = model(torch.randint(0, 100, (3, 10)))
    assert logits.shape == (3, 10, 100)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss():
    model = GPT(TINY)
    idx = torch.randint(0, 100, (3, 10))
    _, loss = model(idx, idx)
    assert loss.shape == ()


def test_targets_equal_to_ignore_index_do_not_contribute_to_loss():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    idx = torch.randint(0, 100, (2, 8))
    targets = idx.clone()
    masked = targets.clone()
    masked[:, :4] = -100  # first half ignored
    _, loss_full = model(idx, targets)
    _, loss_masked = model(idx, masked)
    logits, _ = model(idx)
    expected = nn.functional.cross_entropy(
        logits[:, 4:].reshape(-1, 100), targets[:, 4:].reshape(-1)
    )
    assert torch.allclose(loss_masked, expected, atol=1e-6)
    assert not torch.allclose(loss_full, loss_masked)


def test_sequence_longer_than_block_size_is_rejected():
    model = GPT(TINY)
    with pytest.raises(ValueError):
        model(torch.randint(0, 100, (1, TINY.block_size + 1)))


def test_lm_head_shares_storage_with_token_embedding():
    model = GPT(TINY)
    head_w = model.output_head.lm_head.weight
    wte_w = model.embeddings.wte.weight
    assert head_w is wte_w or head_w.data_ptr() == wte_w.data_ptr()


def test_gpt2_small_has_124m_parameters():
    # 124,439,808 is the published GPT-2 small count (head tied, so counted once).
    assert sum(p.numel() for p in GPT(gpt2_small()).parameters()) == 124_439_808


def test_untrained_model_predicts_roughly_uniformly():
    # With GPT-2 init the logits are near zero, so loss ~ ln(vocab_size).
    torch.manual_seed(0)
    cfg = small()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    # Independent targets: scoring against the input tokens themselves would be
    # biased, because tying lm_head to wte boosts each token's own logit.
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.5


def test_residual_projections_are_initialised_smaller_than_other_weights():
    torch.manual_seed(0)
    cfg = small()
    model = GPT(cfg)
    expected_resid_std = 0.02 / math.sqrt(2 * cfg.n_layer)
    resid = model.h[0].attn.c_proj.weight.std().item()
    other = model.h[0].attn.c_attn.weight.std().item()
    assert resid == pytest.approx(expected_resid_std, rel=0.1)
    assert other == pytest.approx(0.02, rel=0.1)


def test_parameter_names_map_onto_hugging_face_gpt2_names():
    cfg = TINY
    # Independent source of truth: the Hugging Face GPT-2 naming scheme, written by hand.
    expected = {"transformer.wte.weight", "transformer.wpe.weight",
                "transformer.ln_f.weight", "transformer.ln_f.bias", "lm_head.weight"}
    for i in range(cfg.n_layer):
        p = f"transformer.h.{i}."
        for ln in ("ln_1", "ln_2"):
            expected |= {f"{p}{ln}.weight", f"{p}{ln}.bias"}
        for lin in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            expected |= {f"{p}{lin}.weight", f"{p}{lin}.bias"}
    ours = {to_hf_name(k) for k in GPT(cfg).state_dict()}
    assert ours == expected


def test_tiny_batch_overfits_to_near_zero_loss():
    torch.manual_seed(0)
    model = GPT(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, 100, (4, 16))
    targets = torch.randint(0, 100, (4, 16))
    for _ in range(300):
        _, loss = model(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
