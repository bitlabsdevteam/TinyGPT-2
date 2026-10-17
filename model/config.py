"""Config Section: every size and switch of the model lives in one place.

A transformer's shape is fully described by a handful of integers. Keeping them
in a single dataclass means each Section (embeddings, attention, MLP, ...) reads
its sizes from the same object, so changing the model size is a one-line change
and nothing can drift out of sync.
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Number of distinct token ids. GPT-2's BPE vocabulary has 50,257 entries
    # (50,256 byte-pair merges + the <|endoftext|> token). Sets the rows of the
    # token embedding `wte` (vocab_size, n_embd) and the width of the output
    # logits (..., vocab_size).
    vocab_size: int = 50257

    # Maximum sequence length T the model can attend over ("context window").
    # Sets the rows of the learned positional embedding `wpe` (block_size, n_embd)
    # and the size of the causal mask (block_size, block_size).
    block_size: int = 1024

    # Number of stacked transformer Blocks (each = attention + MLP).
    n_layer: int = 12

    # Number of attention heads. Each head attends over a slice of the embedding
    # of size head_dim = n_embd // n_head, so n_embd must be divisible by n_head.
    n_head: int = 12

    # Embedding width C: the size of the vector that flows through the residual
    # stream between Blocks. The MLP hidden layer is 4 * n_embd wide.
    n_embd: int = 768

    # Probability of zeroing an activation during training (regularisation).
    # Applied after embeddings, attention weights/output and the MLP. 0.0 disables it.
    dropout: float = 0.0

    # Whether Linear layers and LayerNorms carry a bias vector. Original GPT-2
    # uses biases (True); keeping True makes HF GPT-2 weights load one-to-one.
    bias: bool = True


def gpt2_small() -> GPTConfig:
    """GPT-2 small (~124M parameters): the real pretraining target on an A100.

    These are the published GPT-2 small hyperparameters, which is also why
    Hugging Face GPT-2 weights can be loaded into a model built from this config.
    """
    return GPTConfig(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
    )


def small() -> GPTConfig:
    """Small (~20M parameter) config for tests, debugging and the overfit check.

    Same architecture and same vocabulary as GPT-2 small, only narrower and
    shallower, so it runs in seconds on a CPU. Context is 512 because TinyStories
    are short.
    """
    return GPTConfig(
        vocab_size=50257,
        block_size=512,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.0,
        bias=True,
    )
