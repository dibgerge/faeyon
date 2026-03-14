from torch import nn

from typing import Any
from .spells import Delayable
from faeyon import Input, Substitute


class FaeModule(nn.Module):
    def __init__(self, chain: Delayable) -> None:
        self._chain = chain

    def forward(self, *args, **kwargs) -> Any:
        return Substitute(A=Input(*args, **kwargs)) | self._chain
