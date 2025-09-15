import abc
import fnmatch
import importlib
import re
import torch

from typing import Optional, Iterable
from torch import nn, optim


class FaeOptimizer:
    """
    Wraps regular torch optimizers, but instead of having to specify parameters directly, use a 
    regexp to specify their names and we can do late bindings.

    name : str | Optimzier
        The name of the optimizer object. If given a string, it should be an existing
        optimizer in the `torch.optim` module, or the full path of the optimizer e.g.
        `foo.bar.MyOptimizer`.

    regex : bool
        Whether to use regular expressions to match parameter names. If True, the `patterns`
        should be a list of regular expressions. If False, the `patterns` should be a list of
        strings. If `None`, it will be inferred from the type of the `patterns` argument. A string
        will be treated as a glob pattern.
    """
    def __init__(
        self,
        name: str,
        patterns: Optional[str | re.Pattern | list[str | re.Pattern]] = None,
        regex: Optional[bool] = None,
        **kwargs
    ):
        module, _, obj = name.rpartition(".")
        if not module:
            module = "optim"

        if patterns is not None:
            if not isinstance(patterns, list):
                patterns = [patterns]

            if regex is False:
                if any(isinstance(pat, re.Pattern) for pat in patterns):
                    raise ValueError(
                        "`regex` is False, but a `re.Pattern` object given to `params`."
                    )
            elif regex is True:
                patterns = [
                    pattern if isinstance(pattern, re.Pattern) else re.compile(pattern) 
                    for pattern in patterns
                ]

        module_obj = importlib.import_module(module)
        self.optimizer = getattr(module_obj, obj)
        self.patterns = patterns
        self.kwargs = kwargs
            
    def __call__(self, model: nn.Module) -> optim.Optimizer:
        matches: Iterable[str]
        valid_names: list[str] = []
        
        if self.patterns is None:
            return self.optimizer(model.parameters(), **self.kwargs)

        parameters = dict(model.named_parameters())
        names = list(parameters.keys())

        for pattern in self.patterns:            
            if isinstance(pattern, re.Pattern):
                matches = filter(pattern.fullmatch, names)
            else:
                matches = fnmatch.filter(names, pattern)
            valid_names.extend(matches)

        return self.optimizer([parameters[k] for k in valid_names], **self.kwargs)


class Recipe:
    """ The base class for training recipes """
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer,
        callbacks: list
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        
    @abc.abstractmethod
    def train_step(self) -> torch.Tensor:
        pass
    
    @abc.abstractmethod
    def val_step(self):
        pass

    @abc.abstractmethod
    def test_step(self):
        pass

    def train(
        self, 
        train_data, 
        *, 
        min_period, 
        max_period, 
        val_period, 
        val_data = None
    ):
        for batch_idx, batch in enumerate(train_data):
            loss = self.training_step(batch, batch_idx)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def validate(self):
        pass
    

class ClassifyRecipe(Recipe):
    
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: FaeOptimizer,
        loss: Optional[nn.Module] = None
    ) -> None:
        if loss is None:
            loss = nn.CrossEntropyLoss()
        super().__init__(model=model, loss=loss, optimizer=FaeOptimizer)


    def train_step(self, batch, batch_idx) -> torch.Tensor:
        """
        batch contains the model arguments  
        """
        pred = self.model(*args, **kwargs)
        loss = self.loss(pred, target)
        return loss

    def val_step(self):
        pass

    def test_step(self):
        pass

