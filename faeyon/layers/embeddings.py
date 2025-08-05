import math
import torch
from torch import nn
from typing import Optional


class InterpEmbedding(nn.Module):
    """
    Interpolates positional embeddings to match the input size. Also supports 
    non-positional embeddings. If this is specified, the outputs will be 
    flattened.
    """
    non_positional: Optional[nn.Parameter] = None

    def __init__(
        self,
        size: int | tuple[int, ...],
        embedding_dim: int,
        non_positional: Optional[int] = None,
        interpolate: Optional[str] = "nearest",
        align_corners: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(size, int):
            size = (size,)

        self.size = size
        self.embeddings = nn.Parameter(torch.randn(1, embedding_dim, *size))

        if non_positional is not None:
            self.non_positional = nn.Parameter(torch.randn(1, embedding_dim, non_positional))
        else:
            self.non_positional = None
        
        self.interpolate = interpolate
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input tensor expected to be of shape `(B, E, *size)`.
        Output tensor will be of shape `(B, E, *size)` if `non_positional` is not specified.
        Otherwise, the output will be of shape `(B, E, prod(size) + non_positional)`.
        """
        if x.ndim != len(self.size) + 2:
            raise ValueError(
                f"Input dimensions {x.ndim}. Expected {len(self.size) + 2}."
            )

        size = x.shape[2:]
        if self.interpolate is None:
            if size != self.size:
                raise ValueError(
                    f"Input size {size} doesn't match embedding size {self.size}. Set "
                    f"`interpolate` to interpolate positional embeddings."
                )
            return self.embeddings

        out = nn.functional.interpolate(
            self.embeddings,
            size=size,
            mode=self.interpolate,
            align_corners=self.align_corners,
        )

        if self.non_positional is not None:
            return torch.cat((self.non_positional, out.flatten(-2)), dim=2)
        return out
