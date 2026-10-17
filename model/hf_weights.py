"""Load Hugging Face GPT-2 weights into our model.

Why
    Our model is written from scratch, so the only way to PROVE it is correct is to
    give it the weights of a model known to be correct and check that both produce
    the same logits. If every Section (embeddings, attention, MLP, LayerNorm, tying,
    init-independent maths) is right, the outputs agree to floating-point round-off.
    A single mistake anywhere makes them disagree.

Two differences between the weight layouts, both handled here
    1. Conv1D vs Linear. Hugging Face builds c_attn, c_proj and c_fc with its own
       `Conv1D` layer, which stores the weight as (in_features, out_features) and
       computes x @ W. nn.Linear stores (out_features, in_features) and computes
       x @ W.T. Same maths, transposed storage, so those weights are transposed
       while loading. (Checked against transformers 5.20.1.)
    2. Names. Our Sections sit one level deeper; `to_hf_name` maps our names to
       theirs.

Mask buffers
    Older transformers versions also stored the causal mask in the state dict
    (`attn.bias`, a (1, 1, 1024, 1024) tensor, and `attn.masked_bias`). They are not
    learned weights, and our mask is a non-persistent buffer built from the config,
    so they are skipped.
"""
from typing import Mapping

import torch

from model.config import GPTConfig
from model.gpt2 import GPT, to_hf_name

# HF names ending with these hold Conv1D weights and need transposing.
_CONV1D_SUFFIXES = ("c_attn.weight", "c_proj.weight", "c_fc.weight")
# HF names ending with these are mask buffers (older transformers versions): skipped.
_MASK_SUFFIXES = (".attn.bias", ".attn.masked_bias")


def load_hf_state_dict(model: GPT, hf_state: Mapping[str, torch.Tensor]) -> None:
    """Copy a Hugging Face GPT-2 state dict into `model`, in place.

    Strict in both directions: an unknown Hugging Face key, or one of our parameters
    with no Hugging Face counterpart, raises KeyError, and nothing is modified.
    Shape mismatches (e.g. a different model size) raise RuntimeError from the copy.
    """
    our_state = model.state_dict()
    ours_by_hf_name = {to_hf_name(name): name for name in our_state}

    # Validate everything first so a bad checkpoint cannot leave a half-loaded model.
    usable = {k: v for k, v in hf_state.items() if not k.endswith(_MASK_SUFFIXES)}
    unexpected = sorted(set(usable) - set(ours_by_hf_name))
    missing = sorted(set(ours_by_hf_name) - set(usable))
    if unexpected or missing:
        raise KeyError(f"unexpected Hugging Face keys: {unexpected}; missing keys: {missing}")

    with torch.no_grad():
        for hf_name, tensor in usable.items():
            if hf_name.endswith(_CONV1D_SUFFIXES):
                tensor = tensor.T  # (in, out) -> (out, in)
            our_state[ours_by_hf_name[hf_name]].copy_(tensor)


def load_pretrained_gpt2(model_name: str = "gpt2") -> GPT:
    """Build our GPT with the sizes of a Hugging Face GPT-2 and load its weights.

    model_name "gpt2" is GPT-2 small (124M). Downloads from the Hugging Face hub on
    first use. `transformers` is imported here only: the model itself never needs it.
    """
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(model_name)
    c = hf.config
    config = GPTConfig(
        vocab_size=c.vocab_size,
        block_size=c.n_positions,
        n_layer=c.n_layer,
        n_head=c.n_head,
        n_embd=c.n_embd,
        dropout=0.0,
        bias=True,
    )
    model = GPT(config)
    load_hf_state_dict(model, hf.state_dict())
    return model
