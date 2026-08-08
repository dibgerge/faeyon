from torch import nn

from typing import Any
from .spells import Delayable, F, Input


class FaeModule(nn.Module):
    def __new__(cls, chain: Delayable) -> "FaeModule":
        # Bypass faek's nn.Module.__new__ patch: it would return a DelayedModule
        # because `chain` is Delayable.  We always want a real FaeModule here.
        return object.__new__(cls)

    def __init__(self, chain: Delayable) -> None:
        super().__init__()
        self._chain = chain
        self._extract_modules()

    def _extract_modules(self) -> None:
        from .faek import faek

        seen: set[int] = set()
        counter = [0]

        def visit(node: Delayable) -> None:
            if not isinstance(node, F):
                return None
            if node.fae.op is not faek.module__call__:
                return None
            module = node.fae.args[0]
            if not isinstance(module, nn.Module) or id(module) in seen:
                return None
            seen.add(id(module))
            name = node.fae.name or f"_{counter[0]}"
            counter[0] += 1
            self.add_module(name, module)
            return None

        self._chain.fae.find(F, visit)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Bypass faek's nn.Module.__call__ patch (which returns an F node).
        # FaeModule is always invoked eagerly; use the original __call__ so
        # that forward hooks and gradient hooks still fire correctly.
        from .faek import faek
        return faek.module__call__(self, *args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs:
            return self._chain._resolve(args[0])
        return self._chain._resolve(Input(*args, **kwargs))
