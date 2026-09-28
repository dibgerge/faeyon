"""DeepSeek-V2/V3 style MLA + MoE (simplified Faeyon-shaped)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from faeyon import A, F, materialize, X, faek
from faeyon.nn import MoE, RotaryEmbedding


class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (compressed KV).

    Q is projected to full heads; K/V go through a low-rank latent then decompress.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_proj = nn.Linear(hidden, num_heads * qk_rope_head_dim, bias=bias)
        self.kv_a_proj = nn.Linear(hidden, kv_lora_rank, bias=bias)
        self.kv_a_norm = nn.RMSNorm(kv_lora_rank)
        self.k_b_proj = nn.Linear(kv_lora_rank, num_heads * qk_rope_head_dim, bias=bias)
        self.v_b_proj = nn.Linear(kv_lora_rank, num_heads * v_head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden, bias=bias)
        self.rope = RotaryEmbedding(embed_dim=qk_rope_head_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.qk_rope_head_dim)
        kv = self.kv_a_norm(self.kv_a_proj(x))
        k = self.k_b_proj(kv).view(b, t, self.num_heads, self.qk_rope_head_dim)
        v = self.v_b_proj(kv).view(b, t, self.num_heads, self.v_head_dim)

        # Apply RoPE on q/k in head dim (batch of heads via reshape).
        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)


def deepseek_block(
    hidden: int,
    num_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    intermediate: int,
    n_experts: int,
    top_k: int,
    eps: float,
):
    attn = MLAAttention(
        hidden=hidden,
        num_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
    )
    with faek:
        return (
            X + (nn.RMSNorm(hidden, eps=eps) >> attn(X)) % "attn"
            >> X
            + (
                nn.RMSNorm(hidden, eps=eps)
                >> MoE(hidden, intermediate, n_experts, top_k=top_k)
            )
            % "moe"
        )


def build_deepseek(
    vocab_size: int,
    hidden_size: int = 512,
    num_heads: int = 8,
    num_layers: int = 2,
    kv_lora_rank: int = 64,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 64,
    intermediate_size: int = 1024,
    n_experts: int = 4,
    top_k: int = 2,
    eps: float = 1e-6,
    padding_idx: int = 0,
) -> nn.Module:
    with faek:
        embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)
        block = deepseek_block(
            hidden_size,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            v_head_dim,
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


class DeepSeek(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = build_deepseek(**kwargs)

    def forward(self, ids: torch.LongTensor):
        from faeyon import Input

        return self.model(Input(ids=ids))
