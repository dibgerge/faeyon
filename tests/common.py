import torch
from torch import nn


class ConstantLayer(nn.Module):
    """
    A layer that multiplies the input by a constant value. Allows for predictable outputs during testing.
    """
    def __init__(
        self, 
        size: int | tuple[int, ...], 
        value: int | float,
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        if isinstance(size, int):
            size = (size,)
        self.value = value
        self.weight = nn.Parameter(torch.ones(*size, dtype=dtype) * value, requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight
