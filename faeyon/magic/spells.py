from __future__ import annotations
from sympy.core import N
import torch
import abc
import sys
import inspect
import enum
import itertools
from collections.abc import Callable, Iterator, Iterable, Sequence
from typing import Any, Optional, overload
from types import NoneType
from abc import ABC, abstractmethod
from collections import defaultdict

from torch import nn
from torch import optim
from ._opinfo import get_opinfo, OpInfo


# TODO: This will be expanded to include other modifier types, like optimizer, etc...
modifierType = str


class _MappingKey(str):
    """ 
    This is a sentinel type to be used with Delayable objects to indicate a map packing.
    """
    pass


class Delayable:
    """
    Delayable is the base class for all delayable objects. It provides the base functionality for
    conditional evaluation, chaining, and resolving with data.
    # TODO: Delayable should be an abstract base class.
    """
    @abstractmethod
    def _resolve(self, data: Any, symbols: Optional[Sequence[Symbol]] = None) -> Any:
        """Uses data to resolve the delayable. Must be implemented by subclasses."""

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
        return self._resolve(other)

    def __mod__[T: Delayable](self: T, other: modifierType) -> T:
        """
        The modulate operator `%` is used to name the operation. It can also be used to modify Delayables, for example, set Optimizer to parameters in delayable modules, etc...
        (TODO: How to handle general modifiers, e.g. optimizer.)
        """
        if isinstance(other, str):
            return Modifiers(self, name=other)
        return NotImplemented
        
    def __rmod__[T: Delayable](self: T, other: str) -> T:
        """
        Modifiers should always be to the right of the Delayable.
        """
        if isinstance(other, modifierType):
            raise TypeError(f"Modifier should be to the right of the Delayable, not the left.")
        return NotImplemented
        
    def __rshift__(self, other: Delayable) -> Chain:
        if isinstance(other, int):
            # TODO: Do cloning of Module types...
            pass
        
        if not isinstance(other, Delayable):
            return NotImplemented
        return Chain(self, other, reduce=False)

    def __rrshift__(self, other: Any) -> Any:
        """
        In this case `other` cannot be of type `Delayable` (__rshift__ is called instead).
        """
        return NotImplemented

    # def __lshift__(self, other: Delayable) -> Chain:
    #     """
    #     """
    #     if not isinstance(other, Delayable):
    #         return NotImplemented
    #     return Chain(self, other)

    # def __rlshift__(self, other: Any) -> Any:
    #     """
    #     In this case `other` cannot be of type `Delayable` (__lshift__ is called instead).
    #     """
    #     return NotImplemented


class _OpActionMixin:
    """
    Base class for delayables which support (arithmetic) operations.
    TODO: Update return types to generics instead of X.
    """
    def keys(self) -> Iterator[_MappingKey]:
        return [_MappingKey(self),]
    
    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> F:
        """
        Specify what actions to takes for a given op attribute name and its corresponding arguments.
        """
        opinfo = get_opinfo(attr_name=name)
        if any(
            isinstance(arg, (FList, FDict)) 
            for arg in itertools.chain(args, kwargs.values())
        ):
            return NotImplemented
        return F(opinfo, self, *args, **kwargs)
        
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
        if isinstance(key, _MappingKey):
            return _Unpack(self, is_map=True)
        return self._op_action("__getitem__", key)

    def __call__(self, *args: Any, **kwargs: Any) -> X:
        return self._op_action("__call__", *args, **kwargs)

    def __reversed__(self) -> X:
        return self._op_action("__reversed__")

    def __iter__(self):
        return iter([_Unpack(self)])

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


class _SymbolMeta(_OpActionMixin, Delayable, abc.ABCMeta):
    _registry: dict[str, type[Symbol]] = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        if any(isinstance(base, _SymbolMeta) for base in bases):
            mcs._registry[name] = cls
        
        return cls

    def _resolve(self, data: Any, symbols: Optional[Sequence[Symbol]] = None) -> Any:
        """ Only X without operations on it will be required to be resolved here. """

        if symbols is not None:
            if not isinstance(self, tuple(symbols)):
                return self

        return data

    def __instancecheck__(cls, instance):
        return (
            super().__instancecheck__(instance) 
            or (isinstance(instance, type) and issubclass(instance, cls))
        )

    def __hash__(cls) -> int:
        """
        Need to define hash since __eq__ is overridden which sets hash to None, and this breaks __instancecheck__.
        """
        return hash(id(cls))

    def __repr__(cls) -> str:
        return cls.__name__


class _SymMeta(type):
    def __getattr__(self, name):
        if name in _SymbolMeta._registry:
            return _SymbolMeta._registry[name]

        return type(name, (Symbol,), {})

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Cannot call Sym")


class Sym(metaclass=_SymMeta):
    """
    dynamically create a symbol class, for example Sym.Y will create a new `Symbol` class called Y, and it will be addeed to the symbol registry.
    """
    pass
   

class Symbol(metaclass=_SymbolMeta):
    pass


class X(Symbol):
    pass


class A(Symbol):
    """
    A placeholder for providing arguments to resolve delayables. Examples:

        Input(data, bias=bar) | X[0] >> 2 * X + A["bias"]

    A and X are usually interchangeable, but in some case they are distinct, for example 
    when used in chain nodes.
    """
    pass


class _Unpack(Delayable):
    """
    Represents an unpacking operation (*X). When resolved, unpacks the data as *args.
    """
    def __init__(self, target, is_map: bool = False) -> None:
        if not isinstance(target, Delayable):
            raise ValueError(f"Target of Unpack must be a Delayable, got {type(target)}.")
        self.target = target
        self.is_map = is_map
    
    def _resolve(self, data: Any, symbols: Optional[Sequence[Symbol]] = None) -> Iterator[Any]:
        return self.target._resolve(data, symbols)
    
    def __repr__(self) -> str:
        if self.is_map:
            prefix = "**"
        else:
            prefix = "*"
        return f"{prefix}{self.target!r}"


class F(_OpActionMixin, Delayable):
    def __init__(self, op: Callable[..., Any], *args, **kwargs) -> None:
        self._fae_op = op
        self._fae_args = args
        self._fae_kwargs = kwargs

    def _resolve(self, data: Any, symbols: Optional[Sequence[Symbol]] = None) -> Any:
        resolved_args = []
        for arg in self._fae_args:
            if isinstance(arg, Delayable):
                resolved = arg._resolve(data, symbols)
            else:
                resolved = arg
            
            if isinstance(arg, _Unpack):
                resolved_args.extend(resolved)
            else:
                resolved_args.append(resolved)

        resolved_kwargs = {}
        for k, v in self._fae_kwargs.items():
            if isinstance(v, Delayable):
                resolved = v._resolve(data, symbols)
            else:
                resolved = v
        
            if not isinstance(v, _Unpack):
                resolved = {k: resolved}
            
            # Need to do this because of the unpacking operation might have same key multiple times.
            for k in resolved:
                if k in resolved_kwargs:
                    raise TypeError(f"{self._fae_op} got multiple values for argument '{k}'.")
            resolved_kwargs.update(resolved)

        if any(
            isinstance(a, Delayable) 
            for a in itertools.chain(resolved_args, resolved_kwargs.values())
        ):
            return F(self._fae_op, *resolved_args, **resolved_kwargs)

        return self._fae_op(*resolved_args, **resolved_kwargs)

    def __str__(self):
        if isinstance(self._fae_op, OpInfo):
            return self._fae_op.to_string(*self._fae_args, **self._fae_kwargs)
        else:
            try:
                name = self._fae_op.__name__
            except AttributeError:
                name = f"{self._fae_op!r}"

            # TODO: Might need special handling of module.__call__
            # if name == "Module.__call__" and len(self.args.args) > 0:
            #     name, *args = self.args  # .args
            # else:
            #     args = self.args  # .args

            args = list(map(repr, self._fae_args))
            args.extend(f"{k}={v!r}" for k, v in self._fae_kwargs.items())
            args = ", ".join(args)
            return f"{name}({args})"

    def __repr__(self) -> str:
        return str(self)


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


class Substitute:
    """
    Performs substitution of specific symbols with their values.

    Examples:
        Substitute(X=10, Y=20) | X + Y => 30

        Substitute(X=10) | X + Y  => 10 + Y (Delayable object is returned)

    """
    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def __or__(self, other: Delayable) -> Any:
        return other._resolve(self._kwargs, symbols=[X])



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


class IF(_OpActionMixin, Delayable):
    def __init__(
        self, 
        condition: bool | Symbol | F,
        then_: Delayable,
        else_: Optional[Delayable] = None
    ) -> None:
        self._condition = condition
        self._then_ = then_
        self._else_ = else_

    def _resolve(self, data: Any) -> Any:
        if isinstance(self._condition, bool):
            condition = self._condition
        else:
            condition = data | self.condition

        if condition:
            return data | self._then_
        else:
            if self._else_ is not None:
                return data | self._else_
            else:
                return data


class Modifiers(_OpActionMixin, Delayable):
    def __init__(
        self, 
        expr: Delayable, 
        name: Optional[str] = None,
        optimizer: Optional[optim.Optimizer] = None,
        
    ) -> None:
        if all(mod is None for mod in (name, optimizer)):
            raise ValueError(
                "At least one of `name` or `optimizer` must be provided."
            )
        self._name = name
        self._optimizer = optimizer

        if isinstance(expr, Modifiers):
            if name is not None and expr.fae_has_name:
                raise ValueError(f"Name already set on given modifier expression {expr}.")
            if optimizer is not None and expr.fae_has_optimizer:
                raise ValueError(f"Optimizer already set on given modifier expression {expr}.")
        self.expr = expr

    @property
    def fae_has_name(self) -> bool:
        return self._name is not None

    @property
    def fae_has_optimizer(self) -> bool:
        return self._optimizer is not None
            

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


class FList(_OpActionMixin, Delayable):
    """
    TODO: Make FList generic e.g. Flist[Delayable, etc..]
    """
    def __init__(self, expressions: list[Delayable]) -> None:
        self._fae_expr = expressions

    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> FList:
        opinfo = get_opinfo(attr_name=name)

        raveled = []
        n = 0
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, FList):
                n += 1
                raveled.append(arg._fae_expr)
            elif isinstance(arg, FDict):
                raise ValueError("Cannot mix `FList` and `FDict` arguments. Choose one.")
            else:
                raveled.append(itertools.repeat(arg))

        if n == 0:
            return FList([F(opinfo, item, *args, **kwargs) for item in self._fae_expr])

        raveled = zip(*raveled)
        out = []
        for item, arg in zip(self._fae_expr, raveled):
            items_args = arg[:len(args)]
            items_kwargs = dict(zip(kwargs.keys(), arg[len(args):]))
            out.append(F(opinfo, item, *items_args, **items_kwargs))
        return FList(out)

    def _resolve(self, data: Any) -> Any:
        return [item._resolve(data) for item in self._fae_expr]

    def __lshift__(self, other: Delayable) -> FList:
        out = []
        for expr in self._fae_expr:
            out.append(expr >> other)
        return FList(out)
        
    def __str__(self) -> str:
        return str(self._fae_expr)

    def __repr__(self) -> str:
        return str(self)


class FDict(_OpActionMixin, Delayable):
    def __init__(self, expressions: dict[str, Delayable]) -> None:
        self._fae_expr = expressions

    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> FDict:
        opinfo = get_opinfo(attr_name=name)

        raveled = defaultdict(list)
        n = 0
        keys = set(self._fae_expr)
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, FDict):
                n += 1
                if keys != set(arg._fae_expr):
                    raise ValueError("All arguments of type `FDict` must have the same keys.")

                for key, item in arg._fae_expr.items():
                    raveled[key].append(item)
            elif isinstance(arg, FList):
                raise ValueError("Cannot mix `FList` and `FDict` arguments. Choose one.")
            else:
                for key in keys:
                    raveled[key].append(arg)

        if n == 0:
            return FDict(
                {key: F(opinfo, item, *args, **kwargs) 
                for key, item in self._fae_expr.items()}
            )

        out = {}
        nargs = len(args)
        for key, value in self._fae_expr.items():
            items_args = raveled[key][:nargs]
            items_kwargs = dict(zip(kwargs, raveled[key][nargs:]))
            out[key] = F(opinfo, value, *items_args, **items_kwargs)
        return FDict(out)

    def _resolve(self, data: Any) -> Any:
        return {key: item._resolve(data) for key, item in self._fae_expr.items()}

    def __lshift__(self, other: Delayable) -> FDict:
        out = {}
        for key, item in self._fae_expr.items():
            out[key] = item >> other
        return FDict(out)

    def __str__(self) -> str:
        return str(self._fae_expr)

    def __repr__(self) -> str:
        return str(self)


class Chain(_OpActionMixin, Delayable):
    """
    A Chain is a sequence of operations: `op0 >> op1 << op2 >> ... >> opn`.
    """
    def __init__(self, *ops: Delayable,) -> None:
        self._fae_ops = []
        for op in ops:
            if isinstance(op, Delayable):
                self._fae_ops.append(op)
            else:
                raise ValueError("All arguments must be of subtype `Delayable` or `nn.Module`.")
        self._fae_ops = tuple(self._fae_ops)

    def _resolve(self, inputs: Any) -> Any:
        """
        TODO: Need to differentiate between X and A here.
        """
        if not self._ops:
            return inputs

        data = self._ops[0]._resolve(inputs)

        for op, reduce in zip(self._ops[1:], self._reduce):
            if reduce:
                data = op._using(inputs)
            else:
                data = op._using(inputs)
        return data

    def __lshift__(self, other: Delayable) -> Chain:
        return Chain(*self._fae_ops[:-1], self._fae_ops[-1] << other)

    def __len__(self) -> int:
        return len(self._ops)


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
