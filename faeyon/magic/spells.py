from __future__ import annotations
import torch
import abc
import sys
import inspect
import enum
import itertools
from collections.abc import Callable, Iterator, Iterable
from typing import Any, Optional, overload
from types import NoneType
from abc import ABC, abstractmethod
from collections import defaultdict

from torch import nn
from ._opinfo import get_opinfo, OpInfo


# TODO: This will be expanded to include other modifier types, like optimizer, etc...
modifierType = str


class DelayableMeta(abc.ABCMeta):
    def __instancecheck__(cls, instance):
        return (
            super().__instancecheck__(instance) 
            or (isinstance(instance, type) and issubclass(instance, cls))
        )


class Delayable(abc.ABC, metaclass=DelayableMeta):
    """
    Delayable is the base class for all delayable objects. It provides the base functionality for
    conditional evaluation, chaining, and resolving with data.
    """
    _condition: Optional[bool | Delayable]
    _else_: Optional[Delayable]

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._condition = None
        obj._else_ = None
        obj._name = None
        return obj

    def copy(self):
        """Perform a shallow copy of the object."""
        out = self.__new__(self.__class__)
        for k, v in self.__dict__.items():
            setattr(out, k, v)
        return out

    @abstractmethod
    def _resolve(self, data: Any) -> Any:
        """Uses data to resolve the delayable. Must be implemented by subclasses."""

    def _using(self, data: Any) -> Any:
        if self._condition is not None:
            condition = conjure(self._condition, data)
            if condition:
                return self._resolve(data)

            if self._else_ is not None:
                return conjure(self._else_, data)
            else:
                return data

        return self._resolve(data)

    def if_[T: Delayable](
        self: T, condition: bool | Delayable, else_: Optional[Delayable] = None
    ) -> T:
        out = self.copy()
        out._condition = condition
        out._else_ = else_
        return out

    def __or__(self, other: Any) -> Any:
        """ The case of `X | data` is not defined."""
        if isinstance(other, torch.Tensor):
            # Prevent __torch_function__ from being called for `X | tensor`.
            # Because torch_function __ror__ uses bitwise_or instead, which we don't want to 
            # handle there since it might be called by the function name, and not the operator 
            # magic.
            raise TypeError("Cannot pipe a tensor to a Delayable.")
        return NotImplemented

    def __ror__(self, other: Any) -> Any:
        """ `data | X` results in evaluating the delayed operations. """
        if isinstance(other, Delayable):
            return NotImplemented
        return self._using(other)

    def __mod__[T: Delayable](self: T, other: modifierType) -> T:
        """
        The modulate operator `%` is used to name the operation. It can also be used to modify Delayables, for example, set Optimizer to parameters in delayable modules, etc...
        (TODO: How to handle general modifiers, e.g. optimizer.)
        """
        if isinstance(other, modifierType):
            self._name = other
            return self
        return NotImplemented
        
    def __rmod__[T: Delayable](self: T, other: str) -> T:
        """
        Modifiers should always be to the right of the Delayable.
        """
        if isinstance(other, modifierType):
            raise TypeError(f"Modifier should be to the right of the Delayable, not the left.")
        return NotImplemented
        
    def __rshift__(self, other: Delayable) -> Chain:
        if not isinstance(other, Delayable):
            return NotImplemented
        return Chain(self, other, reduce=False)

    def __rrshift__(self, other: Any) -> Any:
        """
        In this case `other` cannot be of type `Delayable` (__rshift__ is called instead).
        """
        return NotImplemented

    def __lshift__(self, other: Delayable) -> Chain:
        """
        a << b. Both `a` and `b` are Delayable objects. This is not defined for any other type.

        TODO: Set direction of chain to be "gather".
        """
        if not isinstance(other, Delayable):
            return NotImplemented
        return Chain(self, other, reduce=True)

    def __rlshift__(self, other: Any) -> Any:
        """
        In this case `other` cannot be of type `Delayable` (__lshift__ is called instead).
        """
        return NotImplemented

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError("X is not iterable.")


class _MetaOpAction[T: type](DelayableMeta, abc.ABCMeta):
    """
    Use as metaclass for types which require implementing operators, etc, as symbolic operations
    without having to initialize an instance of the class first. 

    Example: 
        X + 1 
    
    where `X` is a some type will `_MetaOps` metaclass, thus it supports the add operator on the 
    class itself.

    Another important distinction is when using usual object initialization like `X(..., args)`, 
    the class `__init__` will not be called, but rather the `__call__` method of the class instance.
    """
    # --- Binary arithmetic operators ---
    def _make_operands(cls, other: Any) -> tuple[T, Any]:
        obj = super().__call__()
        return obj, other

    def __add__(cls, other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj + other

    def __radd__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other + obj

    def __sub__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj - other

    def __rsub__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other - obj

    def __mul__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj * other

    def __rmul__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other * obj

    def __matmul__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj @ other

    def __rmatmul__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other @ obj

    def __truediv__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj / other

    def __rtruediv__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other / obj
    
    def __floordiv__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj // other

    def __rfloordiv__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other // obj
    
    def __mod__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj % other

    def __rmod__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other % obj
    
    def __divmod__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return divmod(obj, other)

    def __rdivmod__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return divmod(other, obj)
    
    def __pow__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj ** other

    def __rpow__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other ** obj
    
    def __lshift__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj << other
    
    def __rlshift__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other << obj
    
    def __rshift__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj >> other

    def __rrshift__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other >> obj

    def __and__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj & other

    def __rand__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other & obj

    def __or__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj | other
    
    def __ror__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other | obj

    def __xor__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return obj ^ other

    def __rxor__[T](cls: type[T], other: Any) -> T:
        obj, other = cls._make_operands(other)
        return other ^ obj

    # --- Unary arithmetic operators ---
    def __neg__[T](cls: type[T]) -> T:
        return -super().__call__()

    def __pos__[T](cls: type[T]) -> T:
        return +super().__call__()

    def __abs__[T](cls: type[T]) -> T:
        return abs(super().__call__())

    def __invert__[T](cls: type[T]) -> T:
        return ~super().__call__()

    def __round__[T](cls: type[T]) -> T:
        return round(super().__call__())

    # --- Comparison operators ---
    def __lt__[T](cls: type[T], other: Any) -> T:
        return super().__call__() < other

    def __le__[T](cls: type[T], other: Any) -> T:
        return super().__call__() <= other

    def __eq__[T](cls: type[T], other: Any) -> T:
        return super().__call__() == other
    
    def __ne__[T](cls: type[T], other: Any) -> T:
        return super().__call__() != other

    def __gt__[T](cls: type[T], other: Any) -> T:
        return super().__call__() > other

    def __ge__[T](cls: type[T], other: Any) -> T:
        return super().__call__() >= other
    
    # --- Other operators ---
    def __getattr__[T](cls: type[T], name: str) -> T:
        return getattr(super().__call__(), name)
    
    def __getitem__[T](cls: type[T], key: Any) -> T:
        return super().__call__()[key]
    
    def __call__[T](cls: type[T], *args: Any, **kwargs: Any) -> T:
        return super().__call__()(*args, **kwargs)

    def __reversed__[T](cls: type[T]) -> T:
        return reversed(super().__call__())

    def __hash__(self) -> int:
        return super().__hash__()

    def __len__(cls) -> int:
        return len(super().__call__())

    def __iter__(cls):
        # TODO: Need to support *X instead of this....
        # return iter([])
        raise NotImplementedError("X is not iterable.")

    def __repr__(cls):
        return cls.__name__

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Note: For operators like `+`, `-`, `*`, etc., the `__torch_function__` is called only if tensor is the left operand, otherwise the operand must be handled by the right hand side Delayable.
        """
        if kwargs is None:
            kwargs = {}
        
        try:
            # Special/reserved operators must be handled by right hand side operator.
            if func.__name__ in {
                "__lshift__", 
                "__rlshift__", 
                "__rshift__", 
                "__rrshift__", 
                "__or__", 
            }:
                return NotImplemented
        except AttributeError:
            # raised if functon has no __name__ attribute
            pass
        return F(func, *args, **kwargs)


class _DelayableOpAction(Delayable):
    """
    Base class for delayables which support (arithmetic) operations.
    TODO: Update return types to generics instead of X.
    """
    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> F:
        """
        Specify what actions to takes for a given op attribute name and its corresponding arguments.
        """
        return F(get_opinfo(attr_name=name), self, *args, **kwargs)
        
    # --- Binary arithmetic operators ---
    def __add__(self, other: Any) -> X:
        return self._op_action("__add__", other)

    def __radd__(self, other: Any) -> X:
        return self._op_action("__radd__", other)

    def __sub__(self, other: Any) -> X:
        return self._op_action("__sub__", other)

    def __rsub__(self, other: Any) -> X:
        return self._op_action("__rsub__", other)

    def __mul__(self, other: Any) -> X:
        return self._op_action("__mul__", other)

    def __rmul__(self, other: Any) -> X:
        return self._op_action("__rmul__", other)

    def __matmul__(self, other: Any) -> X:
        return self._op_action("__matmul__", other)

    def __rmatmul__(self, other: Any) -> X:
        return self._op_action("__rmatmul__", other)

    def __truediv__(self, other: Any) -> X:
        return self._op_action("__truediv__", other)

    def __rtruediv__(self, other: Any) -> X:
        return self._op_action("__rtruediv__", other)

    def __floordiv__(self, other: Any) -> X:
        return self._op_action("__floordiv__", other)

    def __rfloordiv__(self, other: Any) -> X:
        return self._op_action("__rfloordiv__", other)

    def __mod__(self, other: Any) -> X:
        """
        If `other` qualifies as a Faeyon modifier, use the parent class implementation, otherwise, 
        the modulus % operator is treated as a normal arithmetic operation.
        """
        out =  super().__mod__(other)
        if out is NotImplemented:
            return self._op_action("__mod__", other)
        return out
    
    def __rmod__(self, other: Any) -> X:
        out = super().__rmod__(other)
        if out is NotImplemented:
            return self._op_action("__rmod__", other)
        return out

    def __divmod__(self, other: Any) -> X:
        return self._op_action("__divmod__", other)

    def __rdivmod__(self, other: Any) -> X:
        return self._op_action("__rdivmod__", other)

    def __pow__(self, other: Any) -> X:
        return self._op_action("__pow__", other)

    def __rpow__(self, other: Any) -> X:
        return self._op_action("__rpow__", other)

    def __and__(self, other: Any) -> X:
        return self._op_action("__and__", other)

    def __rand__(self, other: Any) -> X:
        return self._op_action("__rand__", other)

    def __xor__(self, other: Any) -> X:
        return self._op_action("__xor__", other)

    def __rxor__(self, other: Any) -> X:
        return self._op_action("__rxor__", other)

    # --- Unary arithmetic operators ---
    def __neg__(self) -> X:
        return self._op_action("__neg__")

    def __pos__(self) -> X:
        return self._op_action("__pos__")

    def __abs__(self) -> X:
        return self._op_action("__abs__")

    def __invert__(self) -> X:
        return self._op_action("__invert__")

    def __round__(self) -> X:
        return self._op_action("__round__")

    # --- Comparison operators ---
    def __lt__(self, other: Any) -> X:
        return self._op_action("__lt__", other)

    def __le__(self, other: Any) -> X:
        return self._op_action("__le__", other)

    def __eq__(self, other: Any) -> X:
        return self._op_action("__eq__", other)

    def __ne__(self, other: Any) -> X:
        return self._op_action("__ne__", other)

    def __gt__(self, other: Any) -> X:
        return self._op_action("__gt__", other)

    def __ge__(self, other: Any) -> X:
        return self._op_action("__ge__", other)

    #--- Other operators ---
    def __getattr__(self, name: str) -> X:
        if name == "__torch_function__":
            return type(self).__torch_function__
        
        return self._op_action("__getattr__", name)

    def __getitem__(self, key: Any) -> X:
        return self._op_action("__getitem__", key)

    def __call__(self, *args: Any, **kwargs: Any) -> X:
        return self._op_action("__call__", *args, **kwargs)

    def __reversed__(self) -> X:
        return self._op_action("__reversed__")


class X(_DelayableOpAction, metaclass=_MetaOpAction):
    def _resolve(self, data: Any) -> Any:
        """ Only X without operations on it will be required to be resolved here. """
        return data

    # def __iter__(self):
    #     # TODO: Need to support *X instead of this....
    #     return iter([])

    def __repr__(self) -> str:
        return "X"
        # output = self.__class__.__name__
        # for name, args, kwargs in self._fbuffer:
        #     # Don't show the parentheses for __getattr__
        #     if name == "__getattr__":
        #         args_f = str(args[0])
        #     else:
        #         args_f = ", ".join(map(repr, args))

        #     kwargs_f = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())

        #     if kwargs_f:
        #         args_f = args_f + ", " + kwargs_f

        #     output = get_opinfo(attr_name=name).to_string(args_f, X=output)
        # return output


class A(X):
    """
    A placeholder for providing arguments to resolve delayables. Examples:

        Input(data, bias=bar) | X[0] >> 2 * X + A["bias"]

    A and X are usually interchangeable, but in some case they are distinct, for example 
    when used in chain nodes.
    """
    pass


# class _OpBase(Delayable):
#     def _unary_op(self, oper):
#         return F(oper, self)

#     def _binary_op(self, oper, other):
#         if isinstance(other, Parallels):
#             return Parallels([self], other, func=oper)
#         elif isinstance(other, nn.Module):
#             return F(oper, self, other(X))
#         else:
#             return F(oper, self, other)

#     def _rbinary_op(self, oper, other):
#         return F(oper, other, self)


# def op_unary_method[T: _OpBase](oper: Callable[[T], _OpBase]) -> Callable[[T], _OpBase]:
#     def func(self: T) -> _OpBase:
#         return self._unary_op(oper)

#     return func


# def op_binary_method[T: _OpBase](
#     oper: Callable[[T, T], T], is_right: bool
# ) -> Callable[[T, Any], F]:
#     def func(self: T, other: Any) -> F:
#         if is_right:
#             return self._rbinary_op(oper, other)
#         else:
#             return self._binary_op(oper, other)

#     return func


# for bin_op, (method, rmethod) in binary_operators.items():
#     if method in ("__rshift__", "__lshift__"):
#         continue
#     setattr(_OpBase, method, op_binary_method(bin_op, False))
#     setattr(_OpBase, rmethod, op_binary_method(bin_op, True))

# for uni_op, method in unary_operators.items():
#     setattr(_OpBase, method, op_unary_method(uni_op))


class _Strategy(ABC):
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class _NoopStrategy(_Strategy):
    def __call__(self, data: Any) -> Any:
        return data

    def __repr__(self):
        return "<noop>"


class _XStrategy(_Strategy):
    """
    F Strategy when `op` is an instance of `X`. E.g.:

    ```python
    data >> F(X[0])
    ```
    """

    def __init__(self, op: X) -> None:
        self._op = op

    def __call__(self, data: Any) -> Any:
        return conjure(self._op, data)

    def __repr__(self):
        return f"{self._op!r}"


class _CallableStrategy(_Strategy):
    """
    F Strategy when `op` is a Callable.out._condition = self._condition
        out._else_ = self._else_
        return out E.g.:

    ```python
    data >> F(torch.cat, [X[0], X[1]], dim=1)
    ```
    """

    def __init__(self, op: Callable[..., Any], *args, **kwargs) -> None:
        self.op = op
        # self.args = A(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data: Any) -> Any:
        args = [conjure(arg, data) for arg in self.args]
        kwargs = {k: conjure(v, data) for k, v in self.kwargs.items()}
        return self.op(*args, **kwargs)
        # return data >> self.args >> self.op

    def __repr__(self):
        try:
            name = self.op.__name__
        except AttributeError:
            name = f"{self.op!r}"

        if name == "Module.__call__" and len(self.args.args) > 0:
            name, *args = self.args  # .args
        else:
            args = self.args  # .args

        args = ", ".join(map(repr, args))
        # kwargs = ", ".join(f"{k}={v!r}" for k, v in self.args.kwargs.items())

        # if kwargs:
        #     args = args + ", " + kwargs

        return f"{name}({args})"


class F(_DelayableOpAction):
    def __init__(self, op: Callable[..., Any], *args, **kwargs) -> None:
        self.op = op
        self.args = args
        self.kwargs = kwargs

    # def copy(self):
    #     out = F.from_strategy(self.strategy)
    #     out._condition = self._condition
    #     out._else_ = self._else_
    #     return out

    def _resolve(self, data: Any) -> Any:
        args = tuple(
            data | arg if isinstance(arg, (Delayable, _MetaOpAction)) else arg
            for arg in self.args
        )
        kwargs = {
            k: data | v if isinstance(v, (Delayable, _MetaOpAction)) else v
            for k, v in self.kwargs.items()
        }
        return self.op(*args, **kwargs)

    def __repr__(self):
        # return f"F({self.strategy!r})"
        if isinstance(self.op, OpInfo):
            return self.op.to_string(*self.args, **self.kwargs)
        else:
            try:
                name = self.op.__name__
            except AttributeError:
                name = f"{self.op!r}"

            if name == "Module.__call__" and len(self.args.args) > 0:
                name, *args = self.args  # .args
            else:
                args = self.args  # .args

            args = ", ".join(map(repr, args))
            # kwargs = ", ".join(f"{k}={v!r}" for k, v in self.args.kwargs.items())

            # if kwargs:
            #     args = args + ", " + kwargs

            return f"{name}({args})"


class Input:
    """
    A placeholder for providing arguments to resolve delayables. Examples:

        A(data, bias=bar) | X[0] >> 2 * X + A["bias"]
    
    This makes expressions act like functions, where the expression can resolve position arguments
    by their index, e.g. A[0] will use the first argument in the provided `A` input to the 
    expression. Similar, A["bias"] will use the value of the `bias` key    if isinstance(out, list) and len(out) == 1:
        return out[0]
    else:
        return out

    Some rules for using `A` to resolve delayables:
    * `A` arguments should not be delayables themselves, only static data values. 
    
    * Calling e.g. like `A(data, bias=bar)` will create an instance intended to be used by
      expression resolution by the pipe operator `|`. On the other hand, indexing `A` (e.g. `A[0]`)
      is used inside expresssions so they can received outside inputs anywhere in the expression.
    
    * `A` cannot be used by itself inside an expression. For example, the following is invalid:
      `2 * X >> A["bias"]`.

    * The first item in an expression chain can use `A` or `X` interchangeably.

    The difference between `A` and `X`: 
    * Each node in a chain has two sources of inputs: 
    1. From the previous node in the chain.
    2. From the `A` instance.

    Since first node does not have data from previous node, we make `X` equivalent to `A`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs
        self._items = (
            tuple(zip(itertools.repeat(None), args))
            + tuple(kwargs.items())
        )
        
    def __len__(self) -> int:
        return len(self._items)
    
    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def nargs(self) -> int:
        return len(self._args)
    
    @property
    def nkwargs(self) -> int:
        return len(self._kwargs)
    
    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, int):
            return self._items[key][1]
        elif isinstance(key, str):
            return self._kwargs[key]
        else:
            raise TypeError(f"Key must be an integer or string. Got {type(key)}.")

    def __repr__(self) -> str:
        arguments = [
            f"{val!r}" if key is None else f"{key}={val!r}" 
            for key, val in self._items
        ]
        return f"A({', '.join(arguments)})"


# Alias for Input
I = Input


def conjure(x: Any, data: Any, fast: bool = False) -> Any:
    """
    Evaluate the operations stored in the `X` buffer. If the input is not an instance of `X`,
    return it as is.
    """

    if not fast:
        if isinstance(x, Delayable):
            return x._using(data)

        if not isinstance(x, X):
            return x

    inputs = data
    for name, args, kwargs in x:
        # Recursively evaluate the arguments.
        args = tuple(
            conjure(arg, inputs, fast=True) if isinstance(arg, Delayable) else arg
            for arg in args
        )
        kwargs = {
            k: conjure(v, inputs, fast=True) if isinstance(v, Delayable) else v
            for k, v in kwargs.items()
        }

        if name == "__getattr__":
            data = getattr(data, args[0])
        elif name == "__getitem__":
            data = data[args[0]]
        elif name == "__call__":
            data = data(*args, **kwargs)
        else:
            data = getattr(data, name)(*args, **kwargs)
    return data


def _new_instance(cls, *args, **kwargs):
    instance = object.__new__(cls)
    sig = inspect.signature(cls.__init__)

    # Bypass Dynamo's GraphModule, which overrides __new__, but does not pass arguments to super...
    # TODO: File a bug report/PR to PyTorch
    try:
        bound = sig.bind(instance, *args, **kwargs)
        bound.apply_defaults()
        del bound.arguments["self"]
    except TypeError:
        bound = None

    super(cls, instance).__setattr__("_arguments", bound)
    return instance


class Exportable:
    """
    TODO: Update code to incorporate this to model and other objects IO.
    Right now, load and save methods are implemented in the `faeyon.io` module, and only work
    for `nn.Module` objects, I need to generalize this to other Faeyon objects.
    """

    _arguments: Optional[inspect.BoundArguments] = None

    def __new__(cls, *args, **kwargs) -> Any:
        return _new_instance(cls, *args, **kwargs)

    def export(self, trust_code: bool = False) -> dict[str, Any]:
        target = f"{self.__class__.__module__}.{self.__class__.__name__}"

        if self._arguments is None:
            raise ValueError(f"Cannot export `{self.__class__.__name__}` with no arguments.")

        args = []
        for arg in self._arguments.args:
            if isinstance(arg, nn.Module):
                args.append(arg.save(save_state=False, trust_code=trust_code)["_config_"])
            else:
                args.append(arg)

        kwargs = {}
        for k, v in self._arguments.kwargs.items():
            if isinstance(v, nn.Module):
                kwargs[k] = v.save(save_state=False, trust_code=trust_code)["_config_"]
            else:
                kwargs[k] = v

        config = {"_target_": target, "_args_": args, "_kwargs_": kwargs, "_meta_": {}}
        return config


class _NoValue:
    """A unique sentinel to represent empty."""

    @property
    def value(self):
        return self

    def __repr__(self):
        return "<NO_VALUE>"

    def __str__(self):
        return "<NO_VALUE>"


class _Variable:
    """
    A wrappert to hold a container value so that it can be passed by reference across different
    Container selections.
    """
    def __init__(self, *args) -> None:
        if len(args) == 1:
            self.value = args[0]
        elif len(args) > 1:
            raise ValueError("`_Variable` can only be initialized with one or no arguments.")
        else:
            self.value = _NoValue()

    def has_value(self) -> bool:
        return not isinstance(self.value, _NoValue)

    def __repr__(self) -> str:
        return f"{self.value!r}"


class ContainerBase(Delayable, ABC):
    def __init__(self, *args) -> None:
        self._expression: Optional[X | F] = None
        self._value = _Variable(*args)
        # parent is used when morphing to higher type container, we need to make sure parent is
        # morphed too. Current setup has only two level tree (fvar -> fdict/flist, )
        self._parents: list[ContainerBase] = []

    def morph(self, tobj: type[ContainerBase]) -> None:
        self.__class__ = tobj
        for parent in self._parents:
            parent.morph(tobj)

    @property
    def value(self) -> Any:
        return self._value.value

    @value.setter
    def value(self, val: Any) -> None:
        self._value.value = val

    def select[T: ContainerBase](self: T, expression: X | F) -> T:
        if self._expression is not None:
            raise ValueError(
                f"Cannot reassign expression to {self.__class__.__name__}, "
                "since expression has not been used."
            )

        if not isinstance(expression, (X, F)):
            raise ValueError(
                f"Cannot assign expression to {self.__class__.__name__}, "
                "since expression is not an instance of `X` or `F`."
            )

        out = self.copy()
        out._expression = expression
        return out

    def __matmul__[T: ContainerBase](self: T, expression: X | F) -> T:
        return self.select(expression)

    @overload
    def copy[T: ContainerBase](self: T, target: None = None) -> T: ...

    @overload
    def copy(self, target: ContainerBase) -> ContainerBase: ...

    def copy[T: ContainerBase](
        self: T, target: Optional[ContainerBase] = None
    ) -> T | ContainerBase:
        if target is None:
            target = type(self)()

        for k, v in target.__dict__.items():
            if k == "_parents":
                target._parents = list(self._parents)
            else:
                setattr(target, k, getattr(self, k, None))

        return target

    def if_(
        self, condition: bool | Delayable, else_: Optional[Delayable] = None
    ) -> ContainerBase:
        out = super().if_(condition, else_)
        out._parents.append(self)
        return out

    @abstractmethod
    def _set(self, data: Any) -> None:
        pass

    def _resolve(self, data: Any) -> Any:
        new_data = data
        if self._expression is not None:
            new_data = conjure(self._expression, data)
        self._set(new_data)
        return data

    def __rrshift__(self, data: Any) -> Any:
        return self._using(data)

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @property
    def is_selected(self) -> bool:
        return self._expression is not None

    @property
    @abstractmethod
    def is_appendable(self) -> bool:
        pass

    @property
    def sheddable(self) -> bool:
        return not isinstance(self.value, _NoValue) and not self.is_selected

    def _shedder(self) -> Any:
        """Overridable method to do shallow copy of value based on subclass type."""
        return self.value

    def shed(self) -> Any:
        if not self.sheddable:
            raise ValueError(
                f"Cannot shed value from {self.__class__.__name__} with no value or a "
                f"pending select."
            )
        return self._shedder()

    def __pos__(self) -> Any:
        return self.shed()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value!r})"


class FVar(ContainerBase):
    """
    `FVar` holds a single value. If it is morphable, then it can be converted to a `FList`
    or `FDict` if requesting a key when it is empty, or adding a new value if another already
    exists.
    """

    def __init__(self, morphable: bool = True) -> None:
        super().__init__()
        self.morphable = morphable

    def __getitem__(self, key: str):
        if not self.is_empty:
            raise ValueError("Cannot promote FVar to FDict from non-empty FVar.")
        self.value = {}
        self._key = None
        self.morph(FDict)
        # self.__class__ = FDict  # type: ignore[assignment]
        return self[key]

    def _set(self, data: Any) -> None:
        if self.morphable and not self.is_empty:
            self.value = [self.value]
            self.morph(FList)
            self._set(data)
        else:
            self.value = data

    @property
    def is_empty(self) -> bool:
        return not self._value.has_value()

    @property
    def is_appendable(self) -> bool:
        return False


class FList(ContainerBase):
    def __init__(self, *args) -> None:
        super().__init__(list(args))

    def _set(self, data: Any) -> None:
        self.value.append(data)

    def __len__(self):
        return len(self.value)

    def _shedder(self) -> Any:
        return list(self.value)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    @property
    def is_appendable(self) -> bool:
        return True


class KeyedContainer(ContainerBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self._key = None

    def __getitem__(self, key: str):
        if self._key is not None:
            raise KeyError(
                f"Key has already been assigned to {self.__class__.__name__} and no data used yet."
            )
        out = self.copy()
        out._key = key
        out._parents.append(self)
        return out

    @abstractmethod
    def _set_item(self, data: Any) -> None:
        pass

    def _set(self, data: Any) -> None:
        if self._key is None:
            raise KeyError(f"No key has been provided {self.__class__.__name__}, cannot set value.")

        self._set_item(data)

    def __len__(self):
        return len(self.value)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0


class FDict(KeyedContainer):
    def __init__(self, morphable: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.morphable = morphable

    def _set_item(self, data: Any) -> None:
        if self._key in self.value and self.morphable:
            mmap = defaultdict(list)
            for k, v in self.value.items():
                mmap[k].append(v)
            self.value = mmap
            self.morph(FMMap)
            # self.__class__ = FMMap  # type: ignore[assignment]
            # self._parent.__class__ = FMMap
            self._set_item(data)
        else:
            self.value[self._key] = data

    def _shedder(self) -> Any:
        return dict(self.value)

    @property
    def is_appendable(self) -> bool:
        return self._key is None


class FMMap(KeyedContainer):
    def __init__(self, **kwargs) -> None:
        for value in kwargs.values():
            if not isinstance(value, list):
                raise ValueError("All values in FMMap must be lists.")

        super().__init__(**kwargs)
        self.value = defaultdict(list, self.value)

    def _set_item(self, data: Any) -> None:
        self.value[self._key].append(data)

    def _shedder(self) -> Any:
        out = {}
        for k, v in self.value.items():
            out[k] = list(v)
        return out

    @property
    def is_appendable(self) -> bool:
        return True


class Chain(_DelayableOpAction):
    """
    A Chain is a sequence of operations: `op0 >> op1 << op2 >> ... >> opn`.
    """
    def __init__(
        self, 
        *ops: Delayable | nn.Module | list[Delayable | nn.Module] | dict[str, Delayable | nn.Module],
        reduce: Optional[list[bool], bool] = None
    ) -> None:
        self._ops = []
        for op in ops:
            if isinstance(op, nn.Module):
                self._ops.append(op(X))
            elif isinstance(op, Delayable):
                self._ops.append(op)
            elif isinstance(op, list):
                if not all(isinstance(item, Delayable | nn.Module) for item in op):
                    raise ValueError(
                        "All items in a list must be of subtype `Delayable` or `nn.Module`."
                    )
                self._ops.append(op)
            elif isinstance(op, dict):
                if not all(isinstance(item, Delayable | nn.Module) for item in op.values()):
                    raise ValueError(
                        "All values in a dictionary must be of subtype `Delayable` or `nn.Module`."
                    )
                self._ops.append(op)
            else:
                raise ValueError("All arguments must be of subtype `Delayable` or `nn.Module`.")
        self._ops = tuple(self._ops)

        if reduce is None:
            self._reduce = [False] * (len(self._ops) - 1)
        else:
            if isinstance(reduce, bool):
                self._reduce = [reduce] * (len(self._ops) - 1)
            elif isinstance(reduce, list):
                if len(reduce) != len(self._ops) - 1:
                    raise ValueError(
                        "`gather` must be the same length as the number of operations minus one."
                    )
                self._reduce = reduce
            else:
                raise ValueError(
                    "`gather` must be a boolean or list of booleans of the same length as "
                    "the number of operations minus one."
                )

    def __getitem__(self, idx: int | str) -> Delayable:
        # TODO: Handle string indexing, which will try to resolve the operation by name, if any 
        # operation has a name (include regex support).
        return self._ops[idx]

    def copy(self):
        out = Chain(*self.ops)
        out._condition = self._condition
        out._else_ = self._else_
        return out

    def _resolve(self, data: Any) -> Any:
        if not self._ops:
            return data

        data = self._ops[0]._using(data)
        for op, reduce in zip(self._ops[1:], self._reduce):
            if reduce:
                data = op._using(data)
            else:
                data = op._using(data)
        return data

    def __len__(self) -> int:
        return len(self._ops)


# class Parallels(_OpBase):
#     ops: tuple[Iterable[Delayable] | Parallels, ...]
#     _else_: Optional[Parallels]

#     def __init__(
#         self,
#         *ops: list[Delayable | X | nn.Module] | Delayable | X | nn.Module,
#         func: Optional[Callable[[Any], Any]] = None,
#     ) -> None:
#         lengths = []
#         error_msg = (
#             "All arguments must be of subtype `Delayable | X | nn.Module` or lists of that with "
#             "broadcastable lengths."
#         )

#         for op in ops:
#             if isinstance(op, list):
#                 if not all(isinstance(item, Delayable | X | nn.Module) for item in op):
#                     raise ValueError(error_msg)
#                 lengths.append(len(op))
#             elif isinstance(op, Parallels):
#                 lengths.append(len(op))
#             elif not isinstance(op, Delayable | X | nn.Module):
#                 raise ValueError(error_msg)
#             else:
#                 lengths.append(1)

#         unique_lengths = set(lengths)
#         self.length = max(unique_lengths)

#         unique_lengths.discard(1)
#         if len(unique_lengths) > 1:
#             raise ValueError(error_msg)

#         new_ops: list[list[Delayable] | Parallels] = []
#         for op in ops:
#             if isinstance(op, list):
#                 new_op = []
#                 for item in op:
#                     if isinstance(item, nn.Module):
#                         new_op.append(item(X))
#                     elif isinstance(item, X):
#                         new_op.append(F(item))
#                     else:
#                         new_op.append(item)

#                 if len(new_op) != self.length:
#                     new_op = op * self.length
#                 new_ops.append(new_op)
#             elif isinstance(op, Parallels):
#                 if len(op) != self.length:
#                     new_ops.append([op] * self.length)
#                 else:
#                     new_ops.append(op)
#             elif isinstance(op, Delayable):
#                 new_ops.append([op] * self.length)
#             elif isinstance(op, X):
#                 new_ops.append([F(op)] * self.length)
#             elif isinstance(op, nn.Module):
#                 new_ops.append([op(X)] * self.length)
#             else:
#                 raise ValueError(error_msg)

#         self.ops = tuple(new_ops)
#         self.func = func
#         self.items = []
#         # for i in range(self.length):
#         #     args = [op[i] for op in self.ops]
#         #     if self.func is not None:
#         #         self.items.append(F(self.func, *args))
#         #     else:
#         #         self.items.append(Serials(*args))

#     def __getitem__(self, idx: int) -> Delayable:
#         # out = self.items[idx]
#         args = [op[idx] for op in self.ops]

#         out: Delayable
#         if self.func is not None:
#             out = F(self.func, *args)
#         else:
#             out = Serials(*args)

#         if self._condition is not None:
#             if self._else_ is not None:
#                 if len(self._else_) == 1:
#                     else_ = self._else_[0]
#                 else:
#                     else_ = self._else_[idx]
#             else:
#                 else_ = None
#             out = out.if_(self._condition, else_)

#         return out

#     def if_(self, condition: bool | Delayable, else_: Optional[Parallels] = None) -> Parallels:
#         if else_ is not None:
#             if not isinstance(else_, Parallels):
#                 raise ValueError("`else_` must be of type `Parallels`.")
#             if len(else_) != self.length:
#                 raise ValueError("`else_` must have length equal to `self.length`.")

#         return super().if_(condition, else_)

#     def __len__(self) -> int:
#         return self.length

#     def copy(self):
#         out = Parallels(*self.ops)
#         out._condition = self._condition
#         out._else_ = self._else_
#         return out

#     def _resolve(self, data: Any) -> Any:
#         for i in range(self.length):
#             data = data >> self[i]
#         return data

#     def _unary_op(self, oper):
#         return Parallels(self, func=oper)

#     def _binary_op(self, oper, other):
#         if isinstance(other, _OpBase):
#             return Parallels(self, other, func=oper)
#         return NotImplemented


class W(enum.Enum):
    Fanout = "Fanout"
    Pass = "Pass"
    Mux = "Mux"

    def __call__(self, *args, **kwargs) -> Any:
        cls = getattr(sys.modules[__name__], f"_{self.name}")
        return cls(*args, **kwargs)


class Wiring(ABC):
    @abstractmethod
    def __getitem__(self, key: int) -> Any:
        pass


class _Fanout(Wiring):
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def __getitem__(self, key: int) -> Any:
        """Basic usage of Fanout only needs to support integer keys, and no slices."""
        if isinstance(self.obj, Delayable):
            return self.obj >> X[key]

        return self.obj[key]


class _Pass(Wiring):
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def __getitem__(self, key: int) -> Any:
        return self.obj


class _Mux(Wiring):
    def __init__(self, s0: Any, s1: Any) -> None:
        self.s0 = s0
        self.s1 = s1

    def __getitem__(self, key: int) -> Any:
        if key == 0:
            return self.s0

        return self.s1


class Wire:
    _fanout: dict[str, Iterator]
    _current_wire: Optional[inspect.BoundArguments]

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.reset()

    def reset(self) -> None:
        self._fanout = {}
        self._current_wire = None

    def init(self, sig: inspect.Signature, *args, **kwargs) -> A:
        self.reset()
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        given_wire = sig.bind_partial(*self.args, **self.kwargs)

        # Replace fanout arguments with their first value in bound
        # update given_wire with bound arguments for passthru arguments only, or if argument does
        # not exist in given_wire, add it.
        for k, v in bound.arguments.items():
            if k not in given_wire.arguments:
                given_wire.arguments[k] = v
            elif not isinstance(given_wire.arguments[k], W):
                continue

            match given_wire.arguments[k]:
                case W.Fanout:
                    self._fanout[k] = iter(v)
                    bound.arguments[k] = next(self._fanout[k])
                case W.Pass:
                    given_wire.arguments[k] = v

        self._current_wire = given_wire
        return A(*bound.args, **bound.kwargs)

    def step(self, data: Any) -> A:
        if self._current_wire is None:
            raise ValueError("No wire has been initialized.")

        for k, v in self._fanout.items():
            try:
                v = next(v)
            except StopIteration:
                raise ValueError(f"Fanout argument {k} has no more values.")
            else:
                self._current_wire.arguments[k] = v

        return data >> A(*self._current_wire.args, **self._current_wire.kwargs)

    def __rrshift__(self, data: Any) -> A:
        return self.step(data)
