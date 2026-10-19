import inspect
import itertools
from torch import nn
from typing import Any
from collections.abc import Callable
from ._opinfo import get_opinfo, OperatorType, OpInfo
from .base import _new_instance
from .spells import (
    F,
    X,
    Delayable,
    DelayedModule,
)

from faeyon.io import save
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
            # Recipe only — ``>>`` / ``recipe(...)`` promote to ``module(X)``.
            out = DelayedModule(cls, *args, **kwargs)
            out.fae_name = cls.__name__
            return out

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
    if OperatorType.UNARY in op_info.type:

        def unary(self: T) -> F:
            return op_info.operator(self(X))

        return unary

    if OperatorType.RBINARY in op_info.type:

        def rbinary(self: T, other: nn.Module | Delayable) -> F:
            if isinstance(other, nn.Module):
                # Unreachable since will be handled by operator of the other module.
                return op_info.operator(other(X), self(X))
            return op_info.operator(other, self(X))

        return rbinary

    if OperatorType.BINARY in op_info.type:

        def binary(self: T, other: nn.Module | Delayable) -> F:
            if isinstance(other, nn.Module):
                return op_info.operator(self(X), other(X))
            return op_info.operator(self(X), other)

        return binary

    raise ValueError(f"Unsupported operator type: {op_info.type}.")


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

    def on(self):
        if self._is_on:
            return

        nn.Module.__new__ = staticmethod(__new__)
        nn.Module.__call__ = __call__
        nn.Module.clone = clone
        nn.Module.save = save

        for opinfo in get_opinfo(type=OperatorType.ARITHMETIC | OperatorType.RESERVED):
            setattr(nn.Module, opinfo.attr_name, delayed_method(opinfo))

        self._is_on = True

    def off(self):
        if not self._is_on:
            return

        for opinfo in get_opinfo(type=OperatorType.ARITHMETIC | OperatorType.RESERVED):
            if opinfo.attr_name in nn.Module.__dict__:
                delattr(nn.Module, opinfo.attr_name)

        delattr(nn.Module, "clone")
        delattr(nn.Module, "save")
        nn.Module.__new__ = staticmethod(__default_new__)
        nn.Module.__call__ = self.module__call__
        self._is_on = False


faek = Faek()
