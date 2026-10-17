"""Attention Section: causal multi-head self-attention (the "QKV" layer).

What it is
    Self-attention lets every token look at the tokens before it and pull in
    the information it finds relevant. Each token is turned into three vectors:

        Q (query): "what am I looking for?"
        K (key):   "what do I contain?"
        V (value): "what do I hand over if someone attends to me?"

    A token's output is a weighted average of the V vectors of the tokens it
    attends to, where the weights come from how well its Q matches their K:

        Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + mask) · V

Why GPT-2 uses it
    It is the only place in the model where information moves *between*
    positions. Everything else (embeddings, MLP, LayerNorm) works on each
    position independently.

Why "causal"
    GPT-2 is trained to predict the next token, so position t must not see
    positions > t, or it could simply copy the answer. The mask adds -inf to
    the scores of future positions so softmax gives them weight exactly 0.

Why "multi-head"
    Instead of one big attention, the embedding is split into n_head slices
    ("heads"), each attending independently with its own Q/K/V. Different heads
    can learn different relations (e.g. previous word, matching bracket).

Shape legend used in comments
    B = batch size, T = sequence length, C = n_embd (embedding width),
    nh = n_head, hs = head size = C // nh.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from model.config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Each head gets an equal slice of the embedding, so C must divide evenly.
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Q, K and V projections fused into ONE Linear: C -> 3C.
        # One big matmul is faster than three small ones; we split the result
        # into Q, K, V afterwards. The name `c_attn` matches Hugging Face GPT-2
        # so pretrained weights can be loaded by name. It is an nn.Linear (HF
        # uses Conv1D, which stores the weight transposed) so it can be quantized
        # for QLoRA later.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output projection: mixes the concatenated heads back into the residual
        # stream. C -> C. Named `c_proj` to match Hugging Face GPT-2.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout on the attention weights, and on the output before it is added
        # back to the residual stream.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of ones, (1, 1, block_size, block_size).
        # Entry [i, j] = 1 means "position i may attend to position j" (j <= i).
        # A buffer, not a parameter: it is never trained, but it moves with the
        # module to the GPU. persistent=False keeps it out of the state dict, so
        # it never collides with Hugging Face's mask buffers when loading weights.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # x: (B, T, C)

        # 1. Project to Q, K, V in one matmul, then split.
        qkv = self.c_attn(x)  # (B, T, C) -> (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C)

        # 2. Head split: carve C into nh heads of size hs, and move the head axis
        #    next to the batch axis so each head is processed independently.
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, T, C) -> (B, nh, T, hs)

        # 3. Scaled dot-product scores: how well each query matches each key.
        #    Dividing by √hs keeps the scores' variance near 1; without it large
        #    dot products push softmax into saturated regions with tiny gradients.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hs)  # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)

        # 4. Causal mask: set scores of future positions (j > i) to -inf so that
        #    softmax assigns them probability 0. Crop the mask to the current T.
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))  # (B, nh, T, T)

        # 5. softmax over the key axis turns each row of scores into weights that
        #    are >= 0 and sum to 1: "how much of each earlier token do I take?"
        weights = F.softmax(scores, dim=-1)  # (B, nh, T, T)
        weights = self.attn_dropout(weights)

        # 6. Weighted average of the value vectors.
        out = weights @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        # 7. Head merge: undo step 2, concatenating all heads side by side.
        #    transpose makes the tensor non-contiguous, so .contiguous() is
        #    required before .view().
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, nh, T, hs) -> (B, T, C)

        # 8. Output projection, then dropout, ready to be added to the residual stream.
        return self.resid_dropout(self.c_proj(out))  # (B, T, C) -> (B, T, C)
