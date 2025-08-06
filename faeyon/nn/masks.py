import torch
from torch import nn
from typing import Optional


class TokenizedMask(nn.Module):
    """
    Given an input, this applies the following mask:
    x * (1.0 - mask) + tokens * mask
    """
    tokens: Optional[nn.Parameter]

    def __init__(self, embedding_dim: int, enabled: bool = True) -> None:
        super().__init__()
        if enabled:
            self.tokens = nn.Parameter(torch.randn(embedding_dim))
        else:
            self.tokens = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:        
        """
        Parameters
        ----------
        x: 
            Tensor of shape (B, E, *size)

        mask: 
            Indicates which locations are masked (True) and which aren't (False).
            Should be of shape `(B, *size)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, E, *size)
        """
        if self.tokens is None and mask is not None:
            raise ValueError(
                "Mask token is not initialized. Please set `enabled=True` "
                "when initializing {self.__class__.__name__}."

            )
        if mask is None or self.tokens is None:
            return x

        batch_size, _, *size = x.shape
        tokens = self.tokens.view(1, -1, *[1]*len(size)).expand(batch_size, -1, *size)
        float_mask = mask.unsqueeze(1).type_as(x)
        return x * (1.0 - float_mask) + tokens * float_mask
