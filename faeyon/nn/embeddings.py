import math
import torch
from torch import nn
from typing import Optional


class PosInterpEmbedding(nn.Module):
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
        align_corners: Optional[bool] = None,
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

    def forward(self, input_size: tuple[int, ...] | list[int]) -> torch.Tensor:
        """
        Input tensor expected to be of shape `(B, E, *size)`. This tells the embedding what the 
        batch size should be and what is the size of the input. Actual inputs values are not used.

        Output tensor will be of shape `(B, E, *size)` if `non_positional` is not specified.
        Otherwise, the output will be of shape `(B, E, prod(size) + non_positional)`.
        """
        if len(input_size) != len(self.size):
            raise ValueError(
                f"Input has {len(input_size)} dimensions. Expected {len(self.size)}."
            )

        if self.interpolate is None:
            if input_size != self.size:
                raise ValueError(
                    f"Input size {input_size} doesn't match embedding size {self.size}. Set "
                    f"`interpolate` to interpolate positional embeddings."
                )
            return self.embeddings

        out = nn.functional.interpolate(
            self.embeddings,
            size=input_size,
            mode=self.interpolate,
            align_corners=self.align_corners,
        )

        if self.non_positional is not None:
            return torch.cat((self.non_positional, out.flatten(-2)), dim=2)
        return out
