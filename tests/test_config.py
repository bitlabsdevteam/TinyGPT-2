"""Seam: the public GPTConfig and its two presets (ticket 01)."""
from model.config import GPTConfig, gpt2_small, small


def test_gpt2_small_preset_matches_gpt2_124m():
    cfg = gpt2_small()
    # Literal values from the published GPT-2 small architecture.
    assert (cfg.vocab_size, cfg.block_size) == (50257, 1024)
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd) == (12, 12, 768)


def test_small_preset_is_the_20m_test_config():
    cfg = small()
    assert (cfg.vocab_size, cfg.block_size) == (50257, 512)
    assert (cfg.n_layer, cfg.n_head, cfg.n_embd) == (6, 6, 384)


def test_every_config_carries_dropout_and_bias_settings():
    for cfg in (gpt2_small(), small()):
        assert isinstance(cfg, GPTConfig)
        assert 0.0 <= cfg.dropout < 1.0
        assert isinstance(cfg.bias, bool)
