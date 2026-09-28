"""Mixtral-style decoder with sparse MoE FFN."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from faeyon import A, F, materialize, X, faek
from faeyon.models.implementations.qwen import QKTransform
from faeyon.nn import MoE, MultiHeadAttention, RotaryEmbedding


def mixtral_block(
    hidden: int,
    heads: int,
    group_size: int,
    intermediate: int,
    n_experts: int,
    top_k: int,
    eps: float,
):
    head_dim = hidden // heads
    rope = RotaryEmbedding(embed_dim=head_dim)
    attn = MultiHeadAttention(
        dm=hidden,
        num_heads=heads,
        group_size=group_size,
        bias=False,
        fq=QKTransform(rope, head_dim, eps),
        fk=QKTransform(rope, head_dim, eps),
    )
    with faek:
        return (
            X
            + (nn.RMSNorm(hidden, eps=eps) >> attn(X, X, X, is_causal=True)) % "attn"
            >> X
            + (
                nn.RMSNorm(hidden, eps=eps)
                >> MoE(hidden, intermediate, n_experts, top_k=top_k)
            )
            % "moe"
        )


def build_mixtral(
    vocab_size: int,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    intermediate_size: int,
    n_experts: int = 8,
    top_k: int = 2,
    group_size: int = 1,
    eps: float = 1e-6,
    padding_idx: int = 0,
) -> nn.Module:
    with faek:
        embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)
        block = mixtral_block(
            hidden_size,
            num_heads,
            group_size,
            intermediate_size,
            n_experts,
            top_k,
            eps,
        )
        return materialize(
            embedding(A["ids"])
            >> (block % "layer" >> num_layers)
            >> nn.RMSNorm(hidden_size, eps=eps)
            >> F(nn.functional.linear, X, embedding.weight) % "lm_head"
        )


class Mixtral(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = build_mixtral(**kwargs)

    def forward(self, ids: torch.LongTensor, attn_mask: Optional[torch.Tensor] = None):
        from faeyon import Input

        return self.model(Input(ids=ids, mask=attn_mask))
