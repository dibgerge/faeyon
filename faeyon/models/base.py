from __future__ import annotations
from torch import nn

from faeyon import X, Op
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
            self.transforms = Op(X)
        
        self.model = model

        if pooling is not None:
            self.pooling = pooling
        else:
            self.pooling = Op(X)

        if task is not None:
            self.task = task
        else:
            self.task = Op(X)

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
    ) -> Pipeline:
        from faeyon.io import load 
        pipeline = load(name, load_state, cls, cache=cache, trust_code=trust_code, **kwargs)


    def forward(self, *args, **kwargs) -> Any:
        """ This is the basic linear pipeline forward pass. """
        return self.transforms(*args, **kwargs) >> self.model >> self.pooling >> self.task

