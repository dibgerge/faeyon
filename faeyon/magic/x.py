from __future__ import annotations
import sys
from collections.abc import Callable
from typing import Any
from ..utils import is_ipython
from .spells import Delayable


def _meta_method[T](name: str) -> Callable[..., T]:
    def method(cls: type[T], *args, **kwargs) -> T:
        # Call the constructor method to return an instance of the class
        # This removes the requirement to explicity initialize the class with `X()`, since 
        # the latter should be interpreted as a function call
        obj = super(_XMeta, cls).__call__()  # type: ignore
        return getattr(obj, name)(*args, **kwargs)
    return method


def _x_method[T](name: str) -> Callable[..., T]:

    # To check if you are inside PyTorch FX tracing, you can use:
    # import inspect
    # def _is_fx_tracing():
    #     # Avoid direct import at top-level to minimize overhead
    #     try:
    #         import torch.fx
    #     except ImportError:
    #         return False
    #     # Check for a Tracer object in the call stack, which is present during FX tracing
    #     for frame_info in inspect.stack():
    #         self_in_frame = frame_info.frame.f_locals.get("self", None)
    #         if self_in_frame is not None and self_in_frame.__class__.__name__ == "Tracer":
    #             return True
    #     return False
    
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


# def _meta_hash(cls) -> int:
#     return super(_MetaX, cls).__hash__()  # type: ignore


# def _meta_len(cls) -> int:
#     return 0


# def _meta_iter(cls):
#     return iter([])


# def _meta_repr(cls):
#     return "X"


# def _meta_instancecheck(cls, instance):
#     return (
#         super(_MetaX, cls).__instancecheck__(instance) 
#         or (isinstance(instance, type) and issubclass(instance, cls))
#     )


# _MetaX = type("_MetaX", (type,), {k: _meta_method(k) for k in _methods})
# _MetaX.__hash__ = _meta_hash  # type: ignore
# _MetaX.__len__ = _meta_len  # type: ignore
# _MetaX.__iter__ = _meta_iter  # type: ignore
# _MetaX.__repr__ = _meta_repr  # type: ignore
# _MetaX.__instancecheck__ = _meta_instancecheck  # type: ignore





class X(metaclass=_MetaOps):  # type: ignore
    """ 
    A buffer for lazy access of operations on any object. This dummy variable is used to hold 
    generally read-only operatons.
    """
    def __init__(self) -> None:
        self._buffer: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._fn = None

    def __repr__(self) -> str:
        output = "X"
        for name, args, kwargs in self:

            # Don't show the parentheses for __getattr__
            if name == "__getattr__":
                args_f = str(args[0])
            else:
                args_f = ", ".join(map(repr, args))
            
            kwargs_f = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())

            if kwargs_f:
                args_f = args_f + ", " + kwargs_f

            output = _methods[name].format(args_f, X=output)        
        return output

    def __iter__(self):
        return iter(self._buffer)

    def __len__(self):
        return len(self._buffer)

    def compile(self):
        # TODO: check if compilable, which no args/kwargs are instances of X
        if self._fn is not None:
            return self._fn
        
        def fn(data):
            for name, args, kwargs in self:
                if name == "__getitem__":
                    data = data[args[0]]
                elif name == "__getattr__":
                    data = getattr(data, args[0])
                else:  
                    data = getattr(data, name)(*args, **kwargs)
            return data
        
        self._fn = fn
        return fn

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

    def __ror__(self, data):
        """
        This is the pipe `|` operator. It is used to evaluate an expression (see conjure). This 
        only works if data is on the left and X is on the right.
        """
        from .spells import Op
        return Op(self) | 
    
    def __or__(self, other):
        raise NotImplementedError(
            "The pipe `|` operator is reserved, and only works if expression is on the right side."
        )


for method in _methods:
    if method not in ["__rshift__", "__rrshift__", "__lshift__", "__rlshift__"]:
        setattr(X, method, _x_method(method))
