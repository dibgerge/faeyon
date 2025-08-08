import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.concat(args, dim=dim)

