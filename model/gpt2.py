"""Model: assembles every Section into GPT-2.

    token ids (B, T)
      -> Embeddings            wte + wpe + dropout          (B, T, C)
      -> n_layer x Block       attention + MLP, residual    (B, T, C)
      -> OutputHead            ln_f + lm_head (tied)        (B, T, vocab_size)
      -> logits  (and, if targets are given, the loss)

This file contains no layer maths of its own: the Sections do that. It only wires
them together, applies the GPT-2 weight initialisation, and ties the output head
to the token embedding.
"""
import torch
import torch.nn.functional as F
from torch import nn

from model.block import Block
from model.config import GPTConfig
from model.embeddings import Embeddings
from model.output_head import OutputHead


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.output_head = OutputHead(config)

        # Weight tying: the output projection and the token embedding become the
        # same Parameter. Gradients from both uses accumulate into it.
        self.output_head.lm_head.weight = self.embeddings.wte.weight

        # GPT-2 initialisation, applied to every submodule.
        self.apply(self._init_weights)
        # Residual-projection scaling (see below), applied on top.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 init: weights ~ N(0, 0.02), biases 0 (LayerNorm keeps 1 / 0).

        Small weights keep the initial logits near 0, so an untrained model
        predicts roughly uniformly and the first loss is about ln(vocab_size).
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Why the extra scaling on c_proj weights (attention and MLP output projections):
    # every Block ADDS two terms (attention and MLP) into the residual stream, so after
    # n_layer Blocks the stream is a sum of 2 * n_layer terms. Summing N independent
    # terms multiplies the standard deviation by sqrt(N); shrinking each term's weights
    # by 1/sqrt(2 * n_layer) keeps the stream's scale roughly constant with depth.

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. targets: optional (B, T) next-token ids.

        Returns (logits, loss). logits is (B, T, vocab_size) for EVERY position;
        loss is None when no targets are given. Target value -100 is ignored by the
        loss (PyTorch's default ignore_index), which fine-tuning uses to mask prompt
        tokens.
        """
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")

        x = self.embeddings(idx)  # (B, T) -> (B, T, C)
        for block in self.h:
            x = block(x)  # (B, T, C) -> (B, T, C)
        logits = self.output_head(x)  # (B, T, C) -> (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Next-token prediction: position t's logits are scored against targets[t]
            # with cross-entropy, -log softmax(logits)[target]. Flatten batch and
            # time into one axis, as cross_entropy expects (N, classes) and (N,).
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B, T, V) -> (B*T, V)
                targets.view(-1),  # (B, T) -> (B*T,)
            )
        return logits, loss


def to_hf_name(name: str) -> str:
    """Map one of our parameter names to its Hugging Face GPT-2 name.

    Our Sections are separate modules, so their parameters sit one level deeper
    than in Hugging Face, where everything below `transformer.` is flat. Only the
    prefix differs; every name below it is identical.

        embeddings.wte.weight      -> transformer.wte.weight
        embeddings.wpe.weight      -> transformer.wpe.weight
        h.0.attn.c_attn.weight     -> transformer.h.0.attn.c_attn.weight
        output_head.ln_f.weight    -> transformer.ln_f.weight
        output_head.lm_head.weight -> lm_head.weight
    """
    if name.startswith("embeddings."):
        return "transformer." + name[len("embeddings."):]
    if name.startswith("h."):
        return "transformer." + name
    if name.startswith("output_head.ln_f."):
        return "transformer." + name[len("output_head."):]
    if name.startswith("output_head.lm_head."):
        return name[len("output_head."):]
    raise KeyError(f"no Hugging Face name for {name!r}")
