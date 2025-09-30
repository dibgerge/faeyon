import torch
from typing import Protocol


class Metric(Protocol):
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None: ...

    def compute(self) -> torch.Tensor: ...

    def reset(self) -> None: ...
