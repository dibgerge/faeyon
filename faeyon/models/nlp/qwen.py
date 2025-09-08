import torch

from typing import Optional
from torch import nn
from faeyon.nn import RotaryEmbedding, FaeBlock, MultiHeadAttention
from faeyon import Op, X


class QKTransform(nn.Module):
    def __init__(self, rotary_embedding: RotaryEmbedding, head_dim: int, eps: float) -> None:
        super().__init__()

        self.rotary_embedding = rotary_embedding
        self.norm = nn.RMSNorm(head_dim, eps=eps)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return x >> self.norm >> self.rotary_embedding(X, mask=mask)


class Qwen(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        hidden_size: int, 
        num_heads: int,
        num_layers: int,
        padding_idx: int,
        intermediate_size: int,
        group_size: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)

        head_dim = hidden_size // num_heads
        rotary_embedding = RotaryEmbedding(embed_dim=head_dim)

        attention = [
            MultiHeadAttention(
                dm=hidden_size,
                num_heads=num_heads,
                group_size=group_size,
                dropout=dropout,
                bias=bias,
                fq=QKTransform(rotary_embedding, head_dim, eps=eps),
                fk=QKTransform(rotary_embedding, head_dim, eps=eps),
            ) 
            for _ in range(num_layers)
        ]
            
        self.decoder = FaeBlock({
            "attention": attention,
            "norm_out": nn.RMSNorm(hidden_size, eps=eps),
            "norm_in": nn.RMSNorm(hidden_size, eps=eps),
            "gate_proj": nn.Linear(hidden_size, intermediate_size, bias=bias),
            "up_proj": nn.Linear(hidden_size, intermediate_size, bias=bias),
            "activation": nn.SiLU(),
            "down_proj": nn.Linear(intermediate_size, hidden_size, bias=bias),
        })

        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)

    def forward(
        self, 
        x: torch.LongTensor, 
        attn_mask: Optional[torch.Tensor] = None,
        is_causal = True
    ) -> torch.Tensor:
         return (
            x 
            >> self.embedding
            >> (
                Op(X) + (
                    self.decoder.norm_in
                    << self.decoder.attention(X, X, X, attn_mask=attn_mask, is_causal=is_causal)
                )
                << Op(X) + (
                    self.decoder.norm_out
                    << self.decoder.up_proj * (self.decoder.gate_proj << self.decoder.activation)
                    << self.decoder.down_proj
                )
            )
            >> self.norm
            >> self.lm_head
         )
