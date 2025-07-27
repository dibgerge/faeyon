from collections.abc import Callable
from typing import Any


# Define methods supported by the _MetaX metaclass
_methods = [
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__",
    "__hash__",
    "__bool__",
    "__getattr__",


    # Emulating callables
    "__call__",
    
    # Containers
    "__len__",
    "__getitem__",
    "__iter__",
    "__reversed__",
    "__contains__"

    # Numeric types
    "__add__",
    "__sub__",
    "__mul__",
    "__matmul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__or__",
    "__xor__",

    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rmatmul__",
    "__rtruediv__",
    "__rfloordiv__",
    "__rmod__",
    "__rdivmod__",
    "__rpow__",
    "__rlshift__",
    "__rrshift__",
    "__rand__",
    "__ror__",
    "__rxor__",

    # Unary operators
    "__neg__",
    "__pos__",
    "__abs__",
    "__invert__",

    # built-in number types
    "__complex__",
    "__int__",
    "__float__",
    "__round__",
    "__trunc__",
    "__floor__",
    "__ceil__"
]


def _meta_method[T](name: str) -> Callable[..., T]:
    def method(cls: type[T], *args, **kwargs) -> T:
        obj = getattr(super(_MetaX, cls), name)
        return getattr(obj, name)(*args, **kwargs)
    return method


def _x_method[T](name: str) -> Callable[..., T]:
    def method(self, *args, **kwargs):
        self._buffer.append((name, args, kwargs))
        return self
    return method


_MetaX = type("_MetaX", (type,), {k: _meta_method(k) for k in _methods})


class X(metaclass=_MetaX):  # type: ignore
    """ 
    A buffer for lazy access of operations on any object. This dummy variable is used to hold 
    generally read-only operatons.
    """
    def __init__(self) -> None:
        self._buffer: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []


for method in _methods:
    setattr(X, method, _x_method(method))


class FaeArgs:
    """
    A placeholder for mapping next operation's arguments to the left side of the >> operator.

    The arguments are stored in `args` and `kwargs` attributes will be used to specify the operations of the next layer. Their values can be used to access the previous layer using the `X` placeholder, or just provide static arguments if that's all you need.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
