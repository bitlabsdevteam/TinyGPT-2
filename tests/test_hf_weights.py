"""Seam: loading Hugging Face GPT-2 weights into our model (ticket 05).

Logit tolerance: atol=1e-4 for the small random models (float32 round-off between two
independent implementations is ~1e-5), atol=1e-3 for the real 124M GPT-2, whose logits are
~tens in magnitude so absolute round-off grows.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from model.config import GPTConfig  # noqa: E402
from model.gpt2 import GPT  # noqa: E402
from model.hf_weights import load_hf_state_dict, load_pretrained_gpt2  # noqa: E402

CFG = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)


def make_hf_model(seed=0):
    """A small Hugging Face GPT-2 with weights large enough that logits are O(1).

    Default GPT-2 init (std 0.02) gives near-zero logits, which would let a broken
    implementation pass an allclose check. LayerNorm gains and all biases are
    randomised too, so every parameter matters.
    """
    torch.manual_seed(seed)
    hf = GPT2LMHeadModel(GPT2Config(
        vocab_size=CFG.vocab_size, n_positions=CFG.block_size, n_ctx=CFG.block_size,
        n_embd=CFG.n_embd, n_layer=CFG.n_layer, n_head=CFG.n_head,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    ))
    with torch.no_grad():
        for name, p in hf.named_parameters():
            p.normal_(mean=1.0 if "ln" in name and name.endswith("weight") else 0.0, std=0.3)
    return hf.eval()


def test_logits_match_hugging_face_after_loading_its_weights():
    hf = make_hf_model()
    ours = GPT(CFG).eval()
    load_hf_state_dict(ours, hf.state_dict())
    idx = torch.randint(0, CFG.vocab_size, (3, 20))
    with torch.no_grad():
        ours_logits, _ = ours(idx)
        hf_logits = hf(idx).logits
    assert hf_logits.abs().max() > 1.0  # the comparison is not vacuous
    assert torch.allclose(ours_logits, hf_logits, atol=1e-4)


def test_conv1d_weights_are_transposed_when_loaded():
    hf = make_hf_model()
    ours = GPT(CFG)
    load_hf_state_dict(ours, hf.state_dict())
    hf_w = hf.state_dict()["transformer.h.1.attn.c_attn.weight"]  # Conv1D: (in, out)
    assert hf_w.shape == (CFG.n_embd, 3 * CFG.n_embd)
    assert torch.equal(ours.h[1].attn.c_attn.weight, hf_w.T)  # nn.Linear: (out, in)
    # Non-Conv1D tensors are copied as they are.
    assert torch.equal(ours.embeddings.wte.weight, hf.state_dict()["transformer.wte.weight"])


def test_mask_buffers_of_older_transformers_versions_are_skipped():
    hf_state = dict(make_hf_model().state_dict())
    for i in range(CFG.n_layer):
        hf_state[f"transformer.h.{i}.attn.bias"] = torch.tril(torch.ones(1, 1, 32, 32))
        hf_state[f"transformer.h.{i}.attn.masked_bias"] = torch.tensor(-1e4)
    load_hf_state_dict(GPT(CFG), hf_state)  # must not raise


def test_a_missing_weight_is_an_error():
    hf_state = dict(make_hf_model().state_dict())
    del hf_state["transformer.ln_f.bias"]
    with pytest.raises(KeyError):
        load_hf_state_dict(GPT(CFG), hf_state)


def test_an_unexpected_weight_is_an_error():
    hf_state = dict(make_hf_model().state_dict())
    hf_state["transformer.h.0.mystery.weight"] = torch.zeros(1)
    with pytest.raises(KeyError):
        load_hf_state_dict(GPT(CFG), hf_state)


def test_real_gpt2_small_logits_match_hugging_face():
    try:
        hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    except Exception as exc:  # no network, no cache, hub error...
        pytest.skip(f"pretrained GPT-2 unavailable: {exc}")
    ours = load_pretrained_gpt2("gpt2").eval()
    assert sum(p.numel() for p in ours.parameters()) == 124_439_808
    # Token ids of "Hello, my name is" and a longer arbitrary run of ids.
    idx = torch.tensor([[15496, 11, 616, 1438, 318], [464, 2068, 7586, 21831, 18045]])
    with torch.no_grad():
        ours_logits, _ = ours(idx)
        hf_logits = hf(idx).logits
    assert torch.allclose(ours_logits, hf_logits, atol=1e-3)
    assert torch.equal(ours_logits.argmax(-1), hf_logits.argmax(-1))
