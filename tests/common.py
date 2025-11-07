import torch
from torch import nn
from typing import Optional
from faeyon import X, Op


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


class BasicModel(nn.Module):
    def __init__(
        self, 
        num_inputs: int = 5,
        embedding: Optional[nn.Embedding] = None,
        num_hidden: int = 20, 
        num_outputs: int = 3
    ) -> None:
        super().__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = Op(X)

        self.layer1 = nn.Linear(num_inputs, num_hidden)
        self.layer2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        return x >> self.embedding >> self.layer1 >> self.layer2
