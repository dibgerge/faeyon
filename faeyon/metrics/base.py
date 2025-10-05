import abc
import torch
from typing import Optional, overload


class Metric(abc.ABC):
    name: str

    def __init__(self, name: Optional[str] = None):
        if name is None:
            self.name = self.__class__.__name__.lower()
        else:
            self.name = name

    @abc.abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None: ...

    @overload
    @abc.abstractmethod
    def compute(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def reset(self) -> None: ...


class MetricCollection(Metric):
    def __init__(self, name: Optional[str] = None, metrics: Optional[list[Metric]] = None):
        super().__init__(name)

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
