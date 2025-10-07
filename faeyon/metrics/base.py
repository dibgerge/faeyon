from __future__ import annotations
import abc
import inspect
import torch
from typing import Optional
from collections import OrderedDict


class Metric(abc.ABC):
    name: str
    _arguments: inspect.BoundArguments

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        sig = inspect.signature(instance.__init__)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        super(cls, instance).__setattr__("_arguments", bound)
        return instance

    def __init__(self, name: Optional[str] = None):
        if name is None:
            self.name = self.__class__.__name__.lower()
        else:
            self.name = name
        
    @abc.abstractmethod
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None: ...

    @abc.abstractmethod
    def compute(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def reset(self) -> None: ...

    def clone[T: Metric](self: T, *args, **kwargs) -> T:
        cls = self.__class__
        sig = inspect.signature(cls.__init__)
        bound = sig.bind_partial(*args, **kwargs)
        cur_arguments = OrderedDict(self._arguments.arguments)
        cur_arguments.update(bound.arguments)
        new_bound = inspect.BoundArguments(sig, cur_arguments)
        return cls(*new_bound.args, **new_bound.kwargs)
