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
        print("extracting modules")
        print(self._chain.fae)
        self._chain.fae_find(F, visit)

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


class DelayedModule(F):
    """
    This is for modules whose constructor arguments are delayables.
    """
    # def __init__(self, module: type[nn.Module], /, *args, **kwargs) -> None:
    #     super().__init__(module, *args, **kwargs)
        
    def _resolve(self, _default: Any, /, **kwargs: Any) -> Any:
        raise ValueError("`DelayedModule` cannot be resolved directly.")

    def __rshift__(self, other: Any) -> Any:
        if isinstance(other, Delayable):
            return self(X) >> other
        else:
            return super().__rshift__(other)

    def __rrshift__(self, other: Any) -> Any:
        if isinstance(other, Delayable):
            return other >> self(X)
        else:
            return super().__rrshift__(other)

    def _generate(self, I: int, P: Optional[Sequence[Any]] = None) -> F:
        """
        If  
        """
        if P is not None:
            return super()._resolve(_NoValue, P=P, I=I)

        return super()._resolve(_NoValue, I=I)
        # from .faek import _resolved_call

        # resolve_kwargs = {"I": I}
        # if P is not None:
        #     resolve_kwargs["P"] = P

        # module_cls = self.fae.arguments.arguments["module"]
        # raw_args = self.fae.arguments.arguments.get("args", ())
        # raw_kwargs = self.fae.arguments.arguments.get("kwargs", {})

        # resolved_args = [
        #     v._resolve(_NoValue, **resolve_kwargs) if isinstance(v, Delayable) else v
        #     for v in raw_args
        # ]
        # resolved_kwargs = {
        #     k: v._resolve(_NoValue, **resolve_kwargs) if isinstance(v, Delayable) else v
        #     for k, v in raw_kwargs.items()
        # }

        # module = module_cls(*resolved_args, **resolved_kwargs)
        # return F(_resolved_call, module, X)

