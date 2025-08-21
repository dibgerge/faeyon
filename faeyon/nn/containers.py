import inspect
import torch
from torch import nn
from typing import Any, Optional, overload, Iterator
from collections import OrderedDict

from faeyon.magic.spells import ContainerBase, Wire, X, A, Op, Wiring


class FaeSequential(nn.Module):
    fae_wire: Optional[Wire]
    reports: list[ContainerBase]

    @overload
    def __init__(self, *args: nn.Module) -> None:
        ...

    @overload
    def __init__(self, *args: OrderedDict[str, nn.Module]) -> None:
        ...

    def __init__(self, *args) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        expected_type = None
        for module in self:
            if expected_type is None:
                expected_type = type(module)
            elif not isinstance(module, expected_type):
                raise ValueError("All modules must be of the same type.")

        self.sig = inspect.signature(args[0].forward)
        self.reset()
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.fae_wire is None:
            self.wire(X)

        out = None
        for name, module in self._modules.items():  # type: ignore
            if out is None:
                fae_args = self.fae_wire.init(self.sig, *args, **kwargs)  # type: ignore
            else:
                fae_args = out >> self.fae_wire

            out = fae_args >> module
            for report in self.reports:
                if report.is_appendable:
                    try:
                        out >> report[name]  # type: ignore
                    except (KeyError, TypeError):
                        out >> report

        for report in self.reports:
            if not report.is_appendable:
                out >> report

        return out

    def wire[T: FaeSequential](self: T, *args: Any, **kwargs: Any) -> T:
        if self.fae_wire is not None:
            raise ValueError("Cannot wire multiple times. Call `reset` first.")
        self.fae_wire = Wire(*args, **kwargs)
        return self

    def report[T: FaeSequential](self: T, *variables: ContainerBase) -> T:
        self.reports = list(variables)
        return self

    def reset[T: FaeSequential](self: T) -> T:
        self.fae_wire = None
        self.reports = []
        return self

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())  # type: ignore

    def __mod__[T: FaeSequential](self: T, other: ContainerBase | A) -> T:
        if isinstance(other, ContainerBase):
            return self.report(other)
        elif isinstance(other, list | tuple):
            if not all(isinstance(x, ContainerBase) for x in other):
                raise TypeError(
                    f"All elements in list must be instances of {ContainerBase.__name__}."
                )
            return self.report(*other)
        elif isinstance(other, A):
            return self.wire(*other.args, **other.kwargs)
        else:
            raise TypeError(
                f"Unsupported type for pipe operator with {self.__class__.__name__} | {type(other)}"
            )


class OpCollection(Op):
    pass


class FaeModuleList(nn.Module):
    def __init__(self, module: nn.Module, repeats: int) -> None:
        super().__init__()
        self.mlist = nn.ModuleList(module * repeats)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Op:
        ops = []
        for i, layer in enumerate(self.mlist):
            layer_args = [arg[i] if isinstance(arg, Wiring) else arg for arg in args]
            layer_kwargs = {
                k: arg[i] if isinstance(arg, Wiring) else arg 
                for k, arg in kwargs.items()
            }
            ops.append(layer(*layer_args, **layer_kwargs))
        return Op(ops)


class FaeBlock(nn.Module):
    """ 
    TODO: Handling non-module components will require ways to re-initialize them. Maybe for 
    `Parameter`/`Buffer` type, give the generator method instead of the tensor itself...
    """
    def __init__(self, components: dict[str, nn.Module], repeats: int) -> None:
        if repeats < 1:
            raise ValueError("Repeats must be at least 1.")

        super().__init__()
        for key, component in components.items():
            if isinstance(component, nn.Module):
                self.add_module(key, FaeModuleList(component, repeats))
            else:
                # TODO: Handle non-module components
                raise NotImplementedError("Non-module components are not supported yet.")
