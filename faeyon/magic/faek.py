import sys
import inspect
import torch

from torch import nn
from typing import Any


def __new__(cls, *args, **kwargs):
    instance = object.__new__(cls)
    sig = inspect.signature(cls.__init__)
    bound = sig.bind(instance, *args, **kwargs)
    bound.apply_defaults()
    del bound.arguments["self"]
    instance._arguments = bound.arguments
    return instance


def __mul__(self: nn.Module, other: int) -> list[nn.Module]:
    """
    Creates a ModuleList of `other` clones of this module.
    """
    if not isinstance(other, int):
        raise TypeError(
            f"Cannot multiply {self} with {type(other)}. Only multiplication by "
            f"int is supported."
        )
    if other < 1:
        raise ValueError("Number of modules must be greater than 0.")

    module_list = []
    for _ in range(other):
        try:
            module_list.append(self.clone())
        except TypeError as e:
            raise TypeError(
                f"Cannot multiply {self} to create a ModuleList of {other} clones due to\n"
                "unsupported argument types. Try generating the ModuleList manually."
            ) from e

    return module_list


def __rmul__(self: nn.Module, other: int) -> list[nn.Module]:
    """ Multiplication is commutative!"""
    return self.__mul__(other)


def __rrshift__[T: nn.Module](self: T, other: torch.Any) -> torch.Any:
    if isinstance(other, nn.Module):
        if not isinstance(other, nn.ModuleList):
            left = [other]
        else:
            left = other

        if not isinstance(self, nn.ModuleList):
            right = [self]
        else:
            right = self

        return nn.ModuleList(left + right)

    elif isinstance(other, torch.Tensor):
        return self(other)
    else:
        raise ValueError(
            f"Unsupported types: {type(self)} and {type(other)} for operation __rshift__ (>>)."
        )


def clone(self: nn.Module) -> nn.Module:
    """
    Create a new instance of the same module with the same arguments. This function only works
    on a strict subset of types, so that to avoid any possible side effects. If any of the 
    Module arguments are not supported, a TypeError will be raised.

    Here is a list of supported types and behavior:
    - nn.Module: The module is cloned recursively.
    - Numbers and strings: Passed as is to the cloned instance
    - torch.Tensor / np.ndarray : Passed by reference to the cloned instance.
    - list/tuple/dict: Iterate recursively and use the above rules for each element. 
        Return a new list/tuple/dict.
    """

    def _get_argument_clone(arg):
        if isinstance(arg, nn.Module):
            return arg.clone()

        # if isinstance(arg, (Number, str, torch.Tensor, np.ndarray)):                
        #     return arg

        if isinstance(arg, (list, tuple)):
            return type(arg)(_get_argument_clone(item) for item in arg)

        if isinstance(arg, dict):
            return {k: _get_argument_clone(v) for k, v in arg.items()}

        # raise TypeError(
        #     f"Unsupported type: {type(arg)} ({arg}) for operation _get_argument_clone."
        # )
        return arg

    cls = self.__class__
    params_clone = {
        name: _get_argument_clone(param)
        for name, param in self._arguments.items()
    }
    return cls(**params_clone)


class Singleton(type):
    """
    This is a singleton metaclass intended to be used as a metaclass for the `FaeMagic` class, 
    so that we cannot create multiple instances of `FaeMagic`.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
        return self.__instance


class Faek(metaclass=Singleton):
    """
    This is a singleton class intended to be used as a context manager or as a general tool
    to enable the `ModuleMixin` functionality by Monkey patching the `nn.Module` in PyTorch.
    """

    methods = [
        "__new__",
        "__mul__",
        "__rmul__",
        "__rrshift__",
        "clone"
    ]

    def __init__(self):
        self._is_on = False

    def __enter__(self):
        self.on()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.off()

    def on(self):
        current_module = sys.modules[__name__]
        for method in self.methods:
            setattr(nn.Module, method, getattr(current_module, method))
        self._is_on = True

    def off(self):
        for method in self.methods:
            if method != "__new__":
                delattr(nn.Module, method)
        self._is_on = False


faek = Faek()
