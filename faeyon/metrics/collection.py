from __future__ import annotations
import torch
from typing import Optional
from .base import Metric


class MetricCollection:
    def __init__(self, metrics: Optional[list[Metric]] = None, name: Optional[str] = None) -> None:
        if name is None:
            self.name = "metrics"
        else:
            self.name = name

        if metrics is not None:
            self.metrics = {metric.name: metric for metric in metrics}

            if len(self.metrics) != len(metrics):
                raise ValueError(
                    f"Metrics must have unique names. Found duplicate names: {self.metrics.keys()}"
                )
        else:
            self.metrics = {}

    def __getitem__(self, key: str) -> Metric:
        return self.metrics[key]

    def clone(self, name: Optional[str] = None) -> MetricCollection:
        return MetricCollection(
            name=name or self.name,
            metrics=[metric.clone() for metric in self.metrics.values()]
        )

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        for metric in self.metrics.values():
            metric.update(predictions, targets)
    
    def compute(self) -> dict[str, torch.Tensor]:
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def __len__(self) -> int:
        return len(self.metrics)
