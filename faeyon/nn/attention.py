import enum
import torch
import torch.nn.functional as F

from typing import Callable, Optional
from faeyon import A, Op
from torch import nn
from torch.nn.attention.flex_attention import flex_attention


class Enum(enum.StrEnum):
    RoPe = "rope"
   

class Attention(nn.Module):
    """
    Base class from which all attention modules inherit.
    """
    def __init__(
        self,
        embed_size: int,
        key_size: Optional[int] = None,
        query_size: Optional[int] = None,
        value_size: Optional[int] = None
    ) -> None:

        super().__init__()
        if key_size is None:
            key_size = embed_size
        if query_size is None:
            query_size = embed_size
        if value_size is None:
            value_size = embed_size

        if any([embed_size <= 0, key_size <= 0, query_size <= 0, value_size <= 0]):
            raise ValueError(
                "`embed_size`, `key_size`, `query_size` and `value_size` must be positive, "
                "if specified."
            )

        self.embed_size = embed_size
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size


class AdditiveAttention(Attention):
    def __init__(
        self,
        embed_size: int,
        key_size: Optional[int] = None,
        query_size: Optional[int] = None,
        value_size: Optional[int] = None
    ) -> None:
        super().__init__(
            embed_size=embed_size,
            key_size=key_size,
            query_size=query_size,
            value_size=value_size
        )
        self.key_proj = nn.Linear(self.key_size, self.embed_size, bias=False)
        self.query_proj = nn.Linear(self.query_size, self.embed_size, bias=False)
        self.score_proj = nn.Linear(self.embed_size, 1, bias=True)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        `query`, `key` and `value` tensors must have the same number of dimensions.

        Let:
            $B$ be the batch size
            $T_k$, $T_q$ be the sequence length for key and query respectively
            $E_k$, $E_q$, $E_v$ be the embedding size for key, query and value, respectively
            $E_c$ be the context embedding size

        Parameters
        ----------
        key : torch.Tensor
            Key tensor with shape $(B?, T_k, E_k)$. This is the hidden state of the encoder.

        query : torch.Tensor
            Query tensor with shape $(B?, T_q, E_q)$. This is the hidden state of the decoder.

        value : torch.Tensor
            Value tensor with shape $(B?, T_k, E_v)$. This is the hidden state of the encoder, so
            usually it will be the same as the key.

        Returns
        -------
        output : torch.Tensor
            This is the context vector of shape $(B?, T_q, E_c)$ where $E_c$ is the
            context embedding size.
        """

        # Key shape: (?, 1, Tk, E_c)
        key = self.key_proj(key).unsqueeze(-3)

        # Query shape: (?, Tq, 1, E_c)
        query = self.query_proj(query).unsqueeze(-2)

        # Score shape: (?, Tq, Tk, 1)
        score = self.score_proj(torch.tanh(key + query))
        alpha = torch.softmax(score, dim=-1)

        c = torch.sum(alpha * value.unsqueeze(-3), dim=-2)
        return c


class MultiHeadAttention(nn.Module):
    """
    softmax( fa( fq(xq Wq + bq) fk(xk Wk + bk)^T ) / scale ) fv(xv Wv)
    If fa is a string, a preset is used, and we should not specify fq, fk, fv.
    """
    def __init__(
        self,
        embed_size: int,
        num_heads: int = 1,
        key_size: Optional[int] = None,
        query_size: Optional[int] = None,
        value_size: Optional[int] = None,
        scale: Optional[float] = None,
        dropout: Optional[float] = 0.0,
        bias: Optional[bool] = False,
        fa: Optional[Callable | str] = None,
        fq: Optional[Callable] = None,
        fk: Optional[Callable] = None,
        fv: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            embed_size=embed_size,
            key_size=key_size,
            query_size=query_size,
            value_size=value_size
        )
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.key_proj = nn.Linear(self.key_size, self.embed_size, bias=bias)
        self.query_proj = nn.Linear(self.query_size, self.embed_size, bias=bias)
        self.value_proj = nn.Linear(self.value_size, self.embed_size, bias=bias)
        self.out_proj = nn.Linear(self.embed_size, self.embed_size, bias=bias)

        if isinstance(fa, str):
            if fk is not None or fv is not None or fq is not None:
                raise ValueError("`fk`, `fv` and `fq` must be None if `fa` is a string.")

            self.fa, self.fqk, self.fv = fa()
        else:        
            self.fa = fa
            self.fq = Op(fq)
            self.fk = Op(fk)
            self.fv = Op(fv)
        self.scale = scale
        self.dropout = dropout
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------

        query : torch.Tensor    
            Query tensor with shape (B, T_q, Eq). 

        key : torch.Tensor
            Key tensor with shape (B, T_kv, Ek). 

        value : torch.Tensor
            Value tensor with shape (B, T_kv, Ev).

        attn_mask : torch.Tensor, optional
            Attention mask with shape (B, T_q, T_kv), **Default**: None
        """
        q = query >> self.query_proj >> A(X, attn_mask=self.attn_mask) >> self.fq
        k = key >> self.key_proj >> A(X, attn_mask=self.attn_mask) >> self.fk
        v = value >> self.value_proj >> A(X, attn_mask=self.attn_mask) >> self.fv
        
        if self.fa is not None:
            # TODO: Adding dropout?
            attention = flex_attention(   
                q, k, v,
                score_mod=self.fa,
                block_mask=None,
                scale=self.scale,
            )
        else:
            attention = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout,
                scale=self.scale,
                is_causal=True,
            )

        return attention >> self.out_proj


class RotaryEmbedding(nn.Module):
    def __init__(
        self, 
        max_position_embeddings: int, 
        embed_dim: int, 
        num_heads: int
    ):
        super().__init__()
        base = 10000.0
        self.inv_freq = 1.0 / base ** (torch.arange(0, num_heads, 2) / embedding_dim)
        self.embed_dim = embed_dim

        if self.embed_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even for rotary embedding.")
    
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q : 
            A tensor of shape (B, T, D), where B is batch size and T is the sequence length, 
            and D is the embedding dimension.

        k : 
            A tensor of shape (B, T, D), where B is batch size and T is the sequence length, 
            and D is the embedding dimension.
        """
        t, d = x.shape[-2:]
        if d != self.embed_dim:
            raise ValueError(
                "q and k must have the same embedding dimension and the rotaty."
            )
            
        pos = torch.where(attn_mask == 0, torch.arange(t, device=x.device), 1)
        f = torch.outer(pos, self.inv_freq)
        cos, sin = f.cos(), f.sin()
        d = x.shape[-1] // 2
        out1 =  x[..., :d] * cos - x[..., d:] * sin
        out2 =  x[..., d:] * cos + x[..., :d] * sin
        return torch.cat([out1, out2], dim=-1)
