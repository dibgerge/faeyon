import torch
from collections.abc import Sequence
from torch import nn
from typing import Optional
from functools import lru_cache


class InterpolatedPosEmbedding(nn.Module):
    """
    Interpolates positional embeddings to match the input size. 

    Parameters
    ----------
    size : int | tuple[int, ...]
        The size of the positional embeddings during training. 

    embedding_dim : int
        The dimension of the embedding.

    interpolate : str | None
        The interpolation mode. If not specified, no interpolation is performed.

    align_corners : bool | None
        Argument for the interpolation function.
    """
    def __init__(
        self,
        size: int | tuple[int, ...],
        embeddings: int | torch.Tensor,
        interpolate: Optional[str] = "nearest",
        align_corners: Optional[bool] = None,
    ) -> None:
        super().__init__()

        if isinstance(size, int):
            size = (size,)

        self.size = size

        if isinstance(embeddings, int):
            self.embeddings = nn.Parameter(torch.randn(1, embeddings, *size))
        else:
            self.register_buffer("embeddings", embeddings)

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
    ) -> torch.Tensor:
        """
        x : 
            A tensor of shape (B, T, D), where B is batch size and T is the sequence length, 
            and D is the embedding dimension. This is usually the key or query tensor, since RoPE is applied to both keys and queries.

        mask : Optional[torch.Tensor]
            A mask of shape (B, T), where B is batch size and T is the sequence length. This is used to mask positions that are not valid.
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
    def calculate(self, input_size: tuple[int, ...], device: torch.device) -> torch.Tensor:
        if self.embed_dim % len(input_size) != 0:
            raise ValueError(
                "Embedding dimension must be divisible by the number of input dimensions for "
                "sine and cosine embeddings."
            )
        
        embed_dim = self.embed_dim // len(input_size)
        
        axes = [torch.arange(sz, device=device) for sz in input_size]
        components = []
        for grid in torch.meshgrid(*axes, indexing="xy"):
            d =  2 * torch.arange(embed_dim // 2, device=device) / embed_dim
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
        return self.calculate(tuple(input_size), self.device)
