"""Qwen2-style decoder stack using ``>> num_layers`` cloning."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from faeyon import A, F, materialize, X, faek
from faeyon.nn import MultiHeadAttention, RotaryEmbedding, SwiGLU


class QKTransform(nn.Module):
    def __init__(self, rotary_embedding: RotaryEmbedding, head_dim: int, eps: float) -> None:
        super().__init__()
        self.rotary_embedding = rotary_embedding
        self.norm = nn.RMSNorm(head_dim, eps=eps)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.rotary_embedding(self.norm(x), mask=mask)


def decoder_block(
    hidden: int,
    heads: int,
    group_size: int,
    intermediate: int,
    eps: float,
    dropout: float = 0.0,
):
    head_dim = hidden // heads
    rope = RotaryEmbedding(embed_dim=head_dim)
    attn = MultiHeadAttention(
        dm=hidden,
        num_heads=heads,
        group_size=group_size,
        dropout=dropout,
        bias=False,
        fq=QKTransform(rope, head_dim, eps),
        fk=QKTransform(rope, head_dim, eps),
    )
    with faek:
        return (
            X
            + (
                nn.RMSNorm(hidden, eps=eps)
                >> attn(X, X, X, attn_mask=A["mask"], is_causal=True)
            )
            % "attn"
            >> X + (nn.RMSNorm(hidden, eps=eps) >> SwiGLU(hidden, intermediate)) % "mlp"
        )


def build_qwen(
    vocab_size: int,
    hidden_size: int,
    num_heads: int,
    num_layers: int,
    intermediate_size: int,
    group_size: int = 1,
    dropout: float = 0.0,
    eps: float = 1e-6,
    padding_idx: int = 0,
) -> nn.Module:
    with faek:
        embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)
        block = decoder_block(
            hidden_size, num_heads, group_size, intermediate_size, eps, dropout
        )
        return materialize(
            embedding(A["ids"])
            >> (block % "layer" >> num_layers)
            >> nn.RMSNorm(hidden_size, eps=eps)
            >> F(nn.functional.linear, X, embedding.weight) % "lm_head"
        )


class Qwen(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        padding_idx: int,
        intermediate_size: int,
        group_size: int = 1,
        dropout: float = 0.1,
        bias: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.model = build_qwen(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            group_size=group_size,
            dropout=dropout,
            eps=eps,
            padding_idx=padding_idx,
        )

    def forward(
        self,
        ids: torch.LongTensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        from faeyon import Input

        return self.model(Input(ids=ids, mask=attn_mask))
