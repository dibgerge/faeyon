from __future__ import annotations
import torch
from typing import Optional
from .base import Metric


class MeanMetric(Metric):
    value: torch.Tensor
    count: torch.Tensor

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name)
        self.reset()

    def update(self, value: torch.Tensor, count: Optional[int | torch.Tensor] = None) -> None:        
        value_size = value.numel()
        if isinstance(count, torch.Tensor):
            if count.ndim != 0:
                raise ValueError("`count` argument for `MeanMetric` must be a scalar.")
        elif count is not None:
            if value_size > 1:
                if count != value_size:
                    raise ValueError("`count` and `value` length mismatch for non-scalar `value`.")
        else:
            count = value_size

        self.value += value.sum()
        self.count += count

    def compute(self) -> torch.Tensor:
        if self.count == 0:
            return torch.tensor(0)

        return self.value / self.count

    def reset(self) -> None:
        self.value = torch.tensor(0.0)
        self.count = torch.tensor(0.0)
