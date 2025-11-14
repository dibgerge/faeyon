import torch
from collections.abc import Sequence
from torch import nn
from typing import Optional
from functools import lru_cache


class PosInterpEmbedding(nn.Module):
    """
    Interpolates positional embeddings to match the input size. Also supports 
    non-positional embeddings. If this is specified, the outputs will be 
    flattened.
    """
    non_positional: Optional[nn.Parameter]

    def __init__(
        self,
        size: int | tuple[int, ...],
        embedding_dim: int,
        interpolate: Optional[str] = "nearest",
        align_corners: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if isinstance(size, int):
            size = (size,)

        self.size = size
        self.embeddings = nn.Parameter(torch.randn(1, embedding_dim, *size))       
        self.interpolate = interpolate
        self.align_corners = align_corners

    def forward(self, input_size: tuple[int, ...] | list[int]) -> torch.Tensor:
        """
        Input tensor expected to be of shape `(B, E, *size)`. This tells the embedding what the 
        batch size should be and what is the size of the input. Actual inputs values are not used.

        Output tensor is of shape (B, E, *size).
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

        return out


class RotaryEmbedding(nn.Module):
    """
    RoPe embeddings. Must call this on keys and queries individually.
    """
    def __init__(
        self, 
        embed_dim: int, 
        base: float = 1000000.0
    ) -> None:
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / base ** (torch.arange(0, embed_dim, 2) / embed_dim))
        self.embed_dim = embed_dim

        if self.embed_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even for rotary embedding.")
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
        
        pos = torch.arange(t, device=x.device)
        if mask is not None:
            pos = torch.where(mask == 0, pos, 1)

        f = torch.outer(pos, self.inv_freq)
        cos, sin = f.cos(), f.sin()
        d = x.shape[-1] // 2
        out1 =  x[..., :d] * cos - x[..., d:] * sin
        out2 =  x[..., d:] * cos + x[..., :d] * sin
        return torch.cat([out1, out2], dim=-1)


class SinCosPosEmbedding(nn.Module):
    """
    Implements the sine and cosine positional embeddings as first proposed in the original Attention Is All You Need paper.
    """
    def __init__(self, embed_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.base = base

        # Register a dummy buffer to track device
        self.register_buffer("_device_tracker", torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self._device_tracker.device

    @device.setter
    def device(self, value: torch.device) -> None:
        raise NotImplementedError("Use `to` method to move the module to a different device.")

    @lru_cache(maxsize=1000)
    def calculate(self, input_size: tuple[int, ...]) -> torch.Tensor:
        if self.embed_dim % len(input_size) != 0:
            raise ValueError(
                "Embedding dimension must be divisible by the number of input dimensions for "
                "sine and cosine embeddings."
            )
        
        embed_dim = self.embed_dim // len(input_size)
        
        axes = [torch.arange(sz, device=self.device) for sz in input_size]
        components = []
        for grid in torch.meshgrid(*axes, indexing="xy"):
            d =  2 * torch.arange(embed_dim // 2, device=self.device) / embed_dim
            omega = 1.0 / self.base ** d

            x = torch.outer(grid.ravel(), omega)
            components.append(torch.cat([x.sin(), x.cos()], dim=-1))
        return torch.cat(components, dim=-1)

    def forward(self, input_size: Sequence[int]) -> torch.Tensor:
        """
        Input size represents the spatial/temporal size of the input, so it should 
        be of shape (H, W) for images, (T) for sequences, (T, H, W) for videos, etc.
        """
        # Get device from module's parameters or buffers
        return self.calculate(tuple(input_size))
