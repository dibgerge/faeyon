from __future__ import annotations
import sys
from collections.abc import Callable
from typing import Any
from ..utils import is_ipython


# Define methods supported by the _MetaX metaclass
_methods = {
    "__lt__": "{X} < {}",
    "__le__": "{X} <= {}",
    "__eq__": "{X} == {}",
    "__ne__": "{X} != {}",
    "__gt__": "{X} > {}",
    "__ge__": "{X} >= {}",
    "__getattr__": "{X}.{}",

    # Emulating callables
    "__call__": "{X}({}, {})",
    
    # Containers
    "__getitem__": "{X}[{}]",
    "__reversed__": "reversed({X})",

    # Numeric types
    "__add__": "{X} + {}",
    "__sub__": "{X} - {}",
    "__mul__": "{X} * {}",
    "__matmul__": "{X} @ {}",
    "__truediv__": "{X} / {}",
    "__floordiv__": "{X} // {}",
    "__mod__": "{X} % {}",
    "__divmod__": "divmod({X}, {})",
    "__pow__": "{X} ** {}",
    "__lshift__": "{X} << {}",
    "__rshift__": "{X} >> {}",
    "__and__": "{X} & {}",
    "__or__": "{X} | {}",
    "__xor__": "{X} ^ {}",

    "__radd__": "{} + {X}",
    "__rsub__": "{} - {X}",
    "__rmul__": "{} * {X}",
    "__rmatmul__": "{} @ {X}",
    "__rtruediv__": "{} / {X}",
    "__rfloordiv__": "{} // {X}",
    "__rmod__": "{} % {X}",
    "__rdivmod__": "divmod({}, {X})",
    "__rpow__": "{} ** {X}",
    "__rlshift__": "{} << {X}",
    "__rrshift__": "{} >> {X}",
    "__rand__": "{} & {X}",
    "__ror__": "{} | {X}",
    "__rxor__": "{} ^ {X}",

    # Unary operators
    "__neg__": "-{X}",
    "__pos__": "+{X}",
    "__abs__": "abs({X})",
    "__invert__": "~{X}",

   # built-in number types
    "__round__": "round({X})",
    "__trunc__": "trunc({X})",
    "__floor__": "floor({X})",
    "__ceil__": "ceil({X})",
}


def __rshift__(self, other):
    from .spells import Op
    return Op(self) >> other


def __rrshift__(self, other):
    from .spells import Op
    return other >> Op(self)


def __lshift__(self, other):
    from .spells import Op
    return Op(self) << other


def __rlshift__(self, other):
    from .spells import Op
    return other << Op(self)


def _meta_method[T](name: str) -> Callable[..., T]:
    def method(cls: type[T], *args, **kwargs) -> T:
        # Call the constructor method to return an instance of the class
        # This removes the requirement to explicity initialize the class with `X()`, since 
        # the latter should be interpreted as a function call
        obj = super(_MetaX, cls).__call__()  # type: ignore
        return getattr(obj, name)(*args, **kwargs)
    return method


def _x_method[T](name: str) -> Callable[..., T]:
    def method(self, *args, **kwargs) -> Any:
        self._buffer.append((name, args, kwargs))
        return self

    def _getattr_(self, key: str) -> Any:
        # Bypass IPython's internal check for the _ipython_canary_method_should_not_exist_ attribute
        # This is required to make IPython display the object's contents
        if is_ipython() and key == "_ipython_canary_method_should_not_exist_":
            return self
        if key == "__torch_function__":
            raise AttributeError
            
        return method(self, key)

    if name == "__getattr__":
        return _getattr_

    # For "special" operators in Faeyon used as pipe operators rather than bit shift
    cur_module = sys.modules[__name__]
    if name in ["__rshift__", "__rrshift__", "__lshift__", "__rlshift__"]:
        return getattr(cur_module, name)

    return method


def _meta_hash(cls) -> int:
    return super(_MetaX, cls).__hash__()  # type: ignore


def _meta_len(cls) -> int:
    return 0


def _meta_iter(cls):
    return iter([])


def _meta_repr(cls):
    return "X"


def _meta_instancecheck(cls, instance):
    return (
        super(_MetaX, cls).__instancecheck__(instance) 
        or (isinstance(instance, type) and issubclass(instance, cls))
    )


_MetaX = type("_MetaX", (type,), {k: _meta_method(k) for k in _methods})
_MetaX.__hash__ = _meta_hash  # type: ignore
_MetaX.__len__ = _meta_len  # type: ignore
_MetaX.__iter__ = _meta_iter  # type: ignore
_MetaX.__repr__ = _meta_repr  # type: ignore
_MetaX.__instancecheck__ = _meta_instancecheck  # type: ignore


class X(metaclass=_MetaX):  # type: ignore
    """ 
    A buffer for lazy access of operations on any object. This dummy variable is used to hold 
    generally read-only operatons.
    """
    def __init__(self) -> None:
        self._buffer: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __repr__(self) -> str:
        output = "X"
        for name, args, kwargs in self:

            # Don't show the parentheses for __getattr__
            if name == "__getattr__":
                args_f = str(args[0])
            else:
                args_f = ", ".join(map(repr, args))
            
            kwargs_f = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            output = _methods[name].format(args_f, kwargs_f, X=output)        
        return output

    def __iter__(self):
        return iter(self._buffer)

    def __len__(self):
        return len(self._buffer)


for method in _methods:
    setattr(X, method, _x_method(method))
