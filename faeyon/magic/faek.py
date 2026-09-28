import sys
import inspect
import itertools
from torch import nn
from typing import Any, overload
from collections.abc import Callable
from ._opinfo import get_opinfo, OperatorType, OpInfo
from .base import _NoValue, _new_instance
from .spells import (
    F,
    X,
    Delayable,
    DelayedModule,
)

from faeyon.utils import Singleton


def __new__(cls, *args, **kwargs):
    """
    Allow `nn.Module` to save constructor arguments passed to it, so that the could be used
    later for cloning modules.

    When any of the arguments is of type `FList` or `FDict`, special handing is applied to
    generate clones.
    """
    for arg in itertools.chain(args, kwargs.values()):
        if isinstance(arg, Delayable):
            return DelayedModule(cls, *args, **kwargs)

    out = _new_instance(cls, *args, **kwargs)
    return out


def __default_new__(cls, *args, **kwargs):
    """
    Once we override __new__ in `nn.Module`, we cannot restore the old one, since nn.Module
    (as of PyTorch 2.7) does not implement `__new__`, and hence expect it to have no arguments.
    The custom __new__ method we implemented above does not match this signature, and hence
    we cannot restore the old one. As a workaround, we define a default __new__ method that
    matches the signature of the default __new__ method in `nn.Module`, but calls the parent object
    without any arguments.
    See: https://stackoverflow.com/questions/79716674/why-does-monkey-patching-a-classs-new-not-always-work/79717493#79717493
    """
    return object.__new__(cls)


@overload
def __mul__[T: nn.Module](self: T, other: int) -> list[T]: ...


@overload
def __mul__[T: nn.Module](self: T, other: nn.Module | Delayable) -> F: ...


def __mul__[T: nn.Module](self: T, other: int | nn.Module | Delayable) -> list[T] | F:
    """Clone ``other`` times, or build a delayed ``self(X) * other`` expression."""
    if isinstance(other, nn.Module):
        return self(X) * other(X)
    if isinstance(other, Delayable):
        return self(X) * other
    if not isinstance(other, int):
        return NotImplemented
    if other < 1:
        raise ValueError("Number of modules must be greater than 0.")
    return [self.clone() for _ in range(other)]


@overload
def __rmul__[T: nn.Module](self: T, other: int) -> list[T]: ...


@overload
def __rmul__[T: nn.Module](self: T, other: nn.Module | Delayable) -> F: ...


def __rmul__[T: nn.Module](self: T, other: int | nn.Module | Delayable) -> list[T] | F:
    """Multiplication is commutative for modules."""
    return self.__mul__(other)  # type: ignore[misc]


def __rshift__[T: nn.Module](self: T, other: Any) -> Any:
    """``module >> delayable|module`` builds a Chain (needed when both sides share a type)."""
    if isinstance(other, Delayable):
        return self(X) >> other
    if isinstance(other, nn.Module):
        return self(X) >> other(X)
    return NotImplemented


def __rrshift__[T: nn.Module](self: T, other: Any) -> Any:
    """
    ``delayable >> module`` / ``module >> module`` build chains; ``data >> module`` evaluates.
    """
    if isinstance(other, Delayable):
        return other >> self(X)
    if isinstance(other, nn.Module):
        return other(X) >> self(X)
    return faek.module__call__(self, other)


def clone[T: nn.Module](self: T, *args: Any, **kwargs: Any) -> T:
    """
    Create a new instance of the same module with the same arguments. This method should be used
    carefully, since it is does not do any deep copying on all types of module arguments.

    The module is cloned based on the arguments passed to its constructor during its creation.
    If any of the arguments were changed after the module was created, the changes will not be
    reflected in the cloned module unless the changes were made on the argument itself inplace
    causing its mutation. E.g. passing a list to the current module and then mutating that same list
    outside the module...

    If you need to clone a module with a argument which should be a new object rather than a shared
    object, you can pass a copy of the argument to the clone method with the new object to use.
    """
    cls = self.__class__
    sig = inspect.signature(self.__init__)  # type: ignore
    bound = sig.bind_partial(*args, **kwargs)
    cur_arguments = dict(self._arguments.arguments)
    cur_arguments.update(bound.arguments)
    new_bound = inspect.BoundArguments(sig, cur_arguments)  # type: ignore[arg-type]
    return cls(*new_bound.args, **new_bound.kwargs)


def __call__(self, *args, **kwargs):
    return F(faek.module__call__, self, *args, **kwargs)


def delayed_method[T: nn.Module](op_info: OpInfo) -> Callable[..., F]:
    """
    Arithmetic on modules: ``module1 + module2``, ``module + delayable``, unary ``-module``, etc.
    Left forms only; right forms fall through to the other operand when needed.
    """
    if op_info.type == OperatorType.UNARY:

        def unary(self: T) -> F:
            return op_info.operator(self(X))

        return unary

    if op_info.type == OperatorType.RBINARY:

        def rbinary(self: T, other: nn.Module | Delayable) -> F:
            if isinstance(other, nn.Module):
                return op_info.operator(other(X), self(X))
            return op_info.operator(other, self(X))

        return rbinary

    if op_info.type == OperatorType.BINARY:

        def binary(self: T, other: nn.Module | Delayable) -> F:
            if isinstance(other, nn.Module):
                return op_info.operator(self(X), other(X))
            return op_info.operator(self(X), other)

        return binary

    raise ValueError(f"Unsupported operator type: {op_info.type}.")


def from_file(
    cls,
    name: str,
    load_state: bool | str = True,
    cache: bool = True,
    trust_code: bool = False,
    **kwargs: Any,
) -> nn.Module:
    from faeyon.io import load

    return load(name, load_state, cls, cache=cache, trust_code=trust_code, **kwargs)


def load(
    self,
    load_state: str,
    cache: bool = True,
    trust_code: bool = False,
    **kwargs: Any,
) -> nn.Module:
    from faeyon.io import load as load_model

    return load_model(self, load_state, cache=cache, trust_code=trust_code, **kwargs)


class Faek(metaclass=Singleton):
    """
    This is a singleton class intended to be used as a context manager or as a general tool
    to enable the `ModuleMixin` functionality by Monkey patching the `nn.Module` in PyTorch.

    The patch is opt-in (importing faeyon does not enable it). Either enable it
    process-wide with `faek.on()` / `faek.off()`, or scope it to a block:

        with faek:
            expr = nn.Linear(10, 5) >> nn.ReLU()

    The context manager is reentrant and restores the previous state on exit, so
    nesting `with faek:` blocks or combining them with an explicit `faek.on()` is safe.
    Only *building* expressions requires the patch; evaluating or materializing an
    existing tree (`data | expr`, `materialize`) works with the patch off.
    """

    def __init__(self):
        self._is_on = False
        self._entered: list[bool] = []
        self.module__call__ = nn.Module.__call__

    @property
    def is_on(self) -> bool:
        return self._is_on

    def __enter__(self):
        self._entered.append(self._is_on)
        self.on()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        was_on = self._entered.pop()
        if not was_on:
            self.off()

    # Operators that must not be patched onto nn.Module: pipeline / pipe collide
    # with bitwise shift and bitwise-or OpInfos of the same names.
    _MODULE_SKIP_OPS = frozenset({
        "mul", "rmul",  # specialised int-clone + arithmetic below
        "rshift", "rrshift", "lshift", "rlshift",
        "or", "ror",
    })

    def on(self):
        if self._is_on:
            return

        from faeyon.io import save

        nn.Module.__new__ = staticmethod(__new__)
        nn.Module.__call__ = __call__
        nn.Module.clone = clone
        nn.Module.save = save
        nn.Module.from_file = classmethod(from_file)
        nn.Module.load = load
        nn.Module.__rrshift__ = __rrshift__
        nn.Module.__rshift__ = __rshift__

        for opinfo in get_opinfo(type=OperatorType.ARITHMETIC):
            if opinfo.name in self._MODULE_SKIP_OPS:
                continue
            setattr(nn.Module, opinfo.attr_name, delayed_method(opinfo))

        nn.Module.__mul__ = __mul__
        nn.Module.__rmul__ = __rmul__

        self._is_on = True

    def off(self):
        if not self._is_on:
            return

        for opinfo in get_opinfo(type=OperatorType.ARITHMETIC):
            if opinfo.name in self._MODULE_SKIP_OPS:
                continue
            if hasattr(nn.Module, opinfo.attr_name):
                delattr(nn.Module, opinfo.attr_name)

        for name in (
            "__mul__", "__rmul__", "__rrshift__", "__rshift__",
            "clone", "save", "from_file", "load",
        ):
            if hasattr(nn.Module, name):
                delattr(nn.Module, name)

        nn.Module.__new__ = staticmethod(__default_new__)
        nn.Module.__call__ = self.module__call__
        self._is_on = False


class ModuleCall(F):
    """
    Wrapper around a delayed module call. Currently unused; kept for experiment hooks.
    """

    def __init__(self, module, /, *args, **kwargs):
        Delayable.__init__(self, module, *args, **kwargs)
        self._call = F(faek.module__call__, module, *args, **kwargs)

    def _resolve(self, /, **kwargs: Any) -> Any:
        return self._call._resolve(**kwargs)


faek = Faek()
