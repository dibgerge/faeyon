from __future__ import annotations
from torch import nn

from typing import Optional, Any
from faeyon import F, A
from faeyon.io import load
from .tasks import Task


class Pipeline(nn.Module):
    """
    A baseclass for model pipelines, which combines a sequence of transforms, a model, 
    a pooling operation, and a task.

    This is modeled as follows:

        Input -> Transforms -> Model -> Pooling -> Task -> Output

    The transforms should be encapsulated in an nn.Module object in order to facilitate 
    saving and loading (serialization of Delayables and X is not supported yet).

    The same goes for the other components of the pipeline.
    """
    def __init__(
        self, 
        model: nn.Module,
        transforms: Optional[nn.Module] = None,
        pooling: Optional[nn.Module] = None,
        task: Optional[Task] = None,
    ) -> None:
        super().__init__()
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = F(None)
        
        self.model = model

        if pooling is not None:
            self.pooling = pooling
        else:
            self.pooling = F(None)

        if task is not None:
            self.task = task
        else:
            self.task = F(None)

    @classmethod
    def from_file(
        cls, 
        name: str, 
        load_state: bool = True, 
        load_transforms: bool = True,
        load_pooling: bool = True,
        load_task: bool = True,
        cache: bool = True, 
        trust_code: bool = False, 
        **kwargs: Any
    ) -> Pipeline | nn.Module:
        overrides : dict[str, Any] = {}
        if not load_transforms:
            overrides["transforms"] = None
        if not load_pooling:
            overrides["pooling"] = None
        if not load_task:
            overrides["task"] = None

        pipeline = load(
            name, 
            load_state, 
            cls, 
            cache=cache, 
            trust_code=trust_code, 
            overrides=overrides, 
            **kwargs
        )
        if not (load_transforms or load_pooling or load_task):
            return pipeline.model
        return pipeline

    def forward(self, x: Any) -> Any:
        """ This is the basic linear pipeline forward pass. """
        return x >> self.transforms >> self.model >> self.pooling >> self.task

