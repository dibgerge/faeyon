import enum
import torch
import torch.nn.functional as F

from typing import Callable, Optional
from faeyon import A, Op, X
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

    fv callable which transforms the value tensor
    fk, fq callable which transforms the key and query tensors. The also get the attention mask and is_causal as arguments.
    """
    def __init__(
        self,
        embed_size: int,
        num_heads: int = 1,
        num_heads_kv: Optional[int] = None,
        kdim: Optional[int] = None,
        qdim: Optional[int] = None,
        vdim: Optional[int] = None,
        scale: Optional[float] = None,
        bias: Optional[bool] = False,
        dropout: Optional[float] = 0.0,
        fa: Optional[Callable | str] = None,
        fq: Optional[Callable] = None,
        fk: Optional[Callable] = None,
        fv: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads.")

        if num_heads_kv is None:
            num_heads_kv = num_heads
        else:
            if num_heads % num_heads_kv != 0:
                raise ValueError("num_heads must be divisible by num_heads_kv.")
        
        self.num_heads_kv = num_heads_kv
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        if kdim is None:
            kdim = embed_size
        if qdim is None:
            qdim = embed_size
        if vdim is None:
            vdim = embed_size
        
        self.query_proj = nn.Linear(qdim, embed_size, bias=bias)
        self.key_proj = nn.Linear(kdim, self.num_heads_kv * self.head_dim, bias=bias)
        self.value_proj = nn.Linear(vdim, self.num_heads_kv * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_size, embed_size, bias=bias)

        if isinstance(fa, str):
            if fk is not None or fv is not None or fq is not None:
                raise ValueError("`fk`, `fv` and `fq` must be None if `fa` is a string.")

            self.fa, self.fqk, self.fv = fa()
        else:        
            self.fa = fa
            self.fq = fq
            self.fk = fk
            self.fv = fv
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
        # TODO: consider the causal mask combined with attn_mask.
        b, t = query.shape[:2]
        reshape = X.view(b, t, -1, self.head_dim).transpose(1, 2)

        q = query >> self.query_proj >> reshape >> Op(self.fq, X, mask=attn_mask)
        k = key >> self.key_proj >> reshape >> Op(self.fk, X, mask=attn_mask)
        v = value >> self.value_proj >> reshape >> Op(self.fv)
        
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
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
                is_causal=True,
                enable_gqa=self.num_heads != self.num_heads_kv,
            )

        return attention >> X.transpose(1, 2).reshape(b, t, -1) >> self.out_proj

