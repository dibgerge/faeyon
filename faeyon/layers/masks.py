import torch
from torch import nn
from typing import Optional


class TokenizedMask(nn.Module):
    mask_token: Optional[nn.Parameter] = None

    def __init__(self, embedding_dim: int, enabled: bool = True) -> None:
        if enabled:
            self.mask_token = nn.Parameter(torch.randn(embedding_dim))
        else:
            self.mask_token = None
        super().__init__()

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:        
        """
        x: 
            Tensor of shape (B, *size, E)

        mask: 
            Indicates which locations are masked (True) and which aren't (False).
            Should be of shape `(B, *size)`.
        """
        if not self.mask_token and mask is not None:
            raise ValueError(
                "Mask token is not initialized. Please set `enabled=True` "
                "when initializing {self.__class__.__name__}."

            )
        if mask is None or self.mask_token is None:
            return x

        batch_size, embed_dim, *size = x.shape
        tokens = self.mask_token[(None,) * (1 + len(size))].expand(batch_size, *size, -1)
        float_mask = mask.unsqueeze(-1).type_as(x)
        return x * (1.0 - float_mask) + tokens * float_mask
