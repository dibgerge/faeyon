import sys
import inspect

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


def __default_new__(cls, *args, **kwargs):
    """ Once we override __new__ in `nn.Module`, we cannot restore the old one, since nn.Module 
    (as of PyTorch 2.7) does not implement `__new__`, and hence expect it to have no arguments. 
    The custom __new__ method we implemented above does not match this signature, and hence 
    we cannot restore the old one. As a workaround, we define a default __new__ method that 
    matches the signature of the default __new__ method in `nn.Module`, but calls the parent object 
    without any arguments.
    """
    return object.__new__(cls)


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

    return [self.clone() for _ in range(other)]


def __rmul__(self: nn.Module, other: int) -> list[nn.Module]:
    """ Multiplication is commutative!"""
    return self.__mul__(other)


def __rrshift__[T: nn.Module](self: T, other: Any) -> Any:
    """
    This is an alias for `__call__`. The limitation here is that it only works for 
    single inputs. If you need to pass multiple inputs, use the `FaeArgs` class.
    """
    return self(other)


def clone(self: nn.Module) -> nn.Module:
    """
    Create a new instance of the same module with the same arguments. This method should be used 
    carefully, since it is does not do any deep copying on all types of module arguments. 

    The module is cloned based on the arguments passed to the constructor. If any of the arguments
    were changed after the module was created, the cloned module will have the same arguments as the
    original module (unless the argument itself was mutated).
    
    Here is the behavior on how the arguments are handled during cloning based on their type:
    - `nn.Module`: The module is cloned recursively.
    - list/tuple/dict: Iterate recursively and use the above rules for each element. 
        Return a new list/tuple/dict.
    - Anything else is passed as is to the new instance.
    """
    def _get_argument_clone(arg):
        if isinstance(arg, nn.Module):
            return arg.clone()

        if isinstance(arg, (list, tuple)):
            return type(arg)(_get_argument_clone(item) for item in arg)

        if isinstance(arg, dict):
            return {k: _get_argument_clone(v) for k, v in arg.items()}

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

        nn.Module.__new__ = staticmethod(__new__)
        self._is_on = True

    def off(self):
        for method in self.methods:
            delattr(nn.Module, method)

        nn.Module.__new__ = staticmethod(__default_new__)
        self._is_on = False


faek = Faek()
