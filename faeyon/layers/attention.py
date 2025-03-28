import torch

from typing import Optional
from torch import nn

nn.MultiheadAttention


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
