"""Seam: GPT.generate (ticket 04)."""
import pytest
import torch

from model.config import GPTConfig
from model.gpt2 import GPT

CFG = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=64)


def make_model(**overrides):
    torch.manual_seed(0)
    cfg = GPTConfig(**{**CFG.__dict__, **overrides})
    return GPT(cfg)


def test_same_seed_gives_same_generation():
    model = make_model()
    prompt = torch.randint(0, 100, (2, 5))
    torch.manual_seed(123)
    a = model.generate(prompt, max_new_tokens=10)
    torch.manual_seed(123)
    b = model.generate(prompt, max_new_tokens=10)
    assert torch.equal(a, b)


def test_generation_ignores_dropout_even_if_model_is_in_train_mode():
    model = make_model(dropout=0.5)
    model.train()
    prompt = torch.randint(0, 100, (1, 5))
    torch.manual_seed(7)
    a = model.generate(prompt, max_new_tokens=8)
    torch.manual_seed(7)
    b = model.generate(prompt, max_new_tokens=8)
    assert torch.equal(a, b)
    assert model.training  # the caller's mode is restored


def test_output_keeps_prompt_and_appends_exactly_max_new_tokens():
    model = make_model()
    prompt = torch.randint(0, 100, (3, 4))
    out = model.generate(prompt, max_new_tokens=6)
    assert out.shape == (3, 10)
    assert torch.equal(out[:, :4], prompt)
    assert out.min() >= 0 and out.max() < 100


def test_generation_can_run_past_the_context_length():
    model = make_model()
    prompt = torch.randint(0, 100, (1, 10))
    out = model.generate(prompt, max_new_tokens=CFG.block_size * 2)  # far beyond block_size
    assert out.shape == (1, 10 + CFG.block_size * 2)


def test_only_the_last_block_size_tokens_are_used_as_context():
    model = make_model().eval()
    long_prompt = torch.randint(0, 100, (1, CFG.block_size + 5))
    # Greedy (top_k=1) makes the next token deterministic: the argmax of the logits
    # computed from the most recent block_size tokens.
    out = model.generate(long_prompt, max_new_tokens=1, top_k=1)
    logits, _ = model(long_prompt[:, -CFG.block_size:])
    assert out[0, -1].item() == logits[0, -1].argmax().item()


def test_top_k_one_is_greedy_whatever_the_seed():
    model = make_model()
    prompt = torch.randint(0, 100, (1, 5))
    torch.manual_seed(1)
    a = model.generate(prompt, max_new_tokens=8, top_k=1)
    torch.manual_seed(2)
    b = model.generate(prompt, max_new_tokens=8, top_k=1)
    assert torch.equal(a, b)


def test_top_k_only_samples_from_the_k_most_likely_tokens():
    model = make_model().eval()
    prompt = torch.randint(0, 100, (1, 5)).repeat(500, 1)  # 500 draws from one distribution
    logits, _ = model(prompt[:1])
    allowed = set(logits[0, -1].topk(3).indices.tolist())
    torch.manual_seed(0)
    sampled = model.generate(prompt, max_new_tokens=1, top_k=3)[:, -1].tolist()
    assert set(sampled) <= allowed
    assert len(set(sampled)) > 1  # genuinely sampling, not collapsing to greedy


def test_temperature_sharpens_or_flattens_the_distribution():
    model = make_model().eval()
    prompt = torch.randint(0, 100, (1, 5)).repeat(2000, 1)
    torch.manual_seed(0)
    cold = model.generate(prompt, max_new_tokens=1, temperature=0.001)[:, -1].tolist()
    hot = model.generate(prompt, max_new_tokens=1, temperature=1000.0)[:, -1].tolist()
    assert len(set(cold)) == 1  # near-zero temperature -> always the argmax
    assert len(set(hot)) > 50  # huge temperature -> nearly uniform over the 100 tokens


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_temperature_is_rejected(bad):
    with pytest.raises(ValueError):
        make_model().generate(torch.randint(0, 100, (1, 3)), max_new_tokens=1, temperature=bad)
