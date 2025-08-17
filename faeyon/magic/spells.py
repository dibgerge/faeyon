from __future__ import annotations
import inspect
import enum
import itertools
import operator
from collections.abc import Callable, Iterator
from typing import Any, Optional, overload
from abc import ABC, abstractmethod
from collections import defaultdict

from torch import nn
from .x import X


binary_operators = {
    operator.rshift: ("__rshift__", "__rrshift__"),
    operator.lshift: ("__lshift__", "__rlshift__"),
    operator.add: ("__add__", "__radd__"),
    operator.sub: ("__sub__", "__rsub__"),
    operator.mul: ("__mul__", "__rmul__"),
    operator.truediv: ("__truediv__", "__rtruediv__"),
    operator.floordiv: ("__floordiv__", "__rfloordiv__"),
    operator.mod: ("__mod__", "__rmod__"),
    operator.pow: ("__pow__", "__rpow__"),
    operator.matmul: ("__matmul__", "__rmatmul__"),
    operator.and_: ("__and__", "__rand__"),
    operator.or_: ("__or__", "__ror__"),
    operator.xor: ("__xor__", "__rxor__"),
}

unary_operators = {
    operator.abs: "__abs__",
    operator.invert: "__invert__",
    operator.neg: "__neg__",
    operator.pos: "__pos__",
}


def conjure(x: Any, data: Any) -> Any:
    """ 
    Evaluate the operations stored in the `X` buffer. If the input is not an instance of `X`, 
    return it as is.
    """
    if isinstance(x, Op):
        return x.using(data)
    
    if not isinstance(x, X):
        return x

    for name, args, kwargs in x:
        # Recursively evaluate the arguments.
        args = tuple(conjure(arg, data) for arg in args)
        kwargs = {k: conjure(v, data) for k, v in kwargs.items()}

        if name == "__getattr__":
            data = getattr(data, args[0])
        else:
            data = getattr(data, name)(*args, **kwargs)
    return data


class A:
    """
    A placeholder for mapping next operation's arguments to the left side of the >> operator.

    The arguments are stored in `args` and `kwargs` attributes will be used to specify the 
    operations of the next layer. Their values can be used to access the previous layer using the 
    `X` placeholder, or just provide static arguments if that's all you need.
    """
    def __init__(self, *args, **kwargs) -> None:
        self._is_resolved = not any(
            isinstance(item, (X, Op)) for item in itertools.chain(args, kwargs.values())
        )

        self.args = args
        self.kwargs = kwargs

    def call(self, func: Callable[..., Any]) -> Any:
        if not self.is_resolved:
            return Op(func, *self.args, **self.kwargs)
        return func(*self.args, **self.kwargs)

    def __rshift__(self, func: Callable[..., Any]) -> Any:
        return self.call(func)

    @property
    def is_resolved(self) -> bool:
        return self._is_resolved
         
    def using(self, data: Any) -> A:
        if self.is_resolved:
            return self
        
        resolved_args = tuple(conjure(arg, data) for arg in self.args)
        resolved_kwargs = {k: conjure(v, data) for k, v in self.kwargs.items()}
        return A(*resolved_args, **resolved_kwargs)

    def __rrshift__(self, data: Any) -> A:
        return self.using(data)

    def __repr__(self) -> str:
        args, kwargs = "", ""
        if self.args:
            args = ", ".join(map(repr, self.args))
            args += ", "

        if self.kwargs:
            kwargs = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"A({args}{kwargs})"


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


class ContainerBase(ABC):
    _condition: Optional[bool | X | Op]
    
    def __init__(self, *args) -> None:
        self._expression: Optional[X | Op] = None
        self._condition = None
        self._value = _Variable(*args)

    @property
    def value(self) -> Any:
        return self._value.value
    
    @value.setter
    def value(self, val: Any) -> None:
        self._value.value = val
    
    def select[T: ContainerBase](self: T, expression: X | Op) -> T:
        if self._expression is not None:
            raise ValueError(
                f"Cannot reassign expression to {self.__class__.__name__}, "
                "since expression has not been used."
            )

        if not isinstance(expression, (X, Op)):
            raise ValueError(
                f"Cannot assign expression to {self.__class__.__name__}, "
                "since expression is not an instance of `X` or `Op`."
            )

        out = self._copy()
        out._expression = expression
        return out

    def __matmul__[T: ContainerBase](self: T, expression: X | Op) -> T:
        return self.select(expression)

    @overload
    def _copy[T: ContainerBase](self: T, target: None = None) -> T: ...

    @overload
    def _copy(self, target: ContainerBase) -> ContainerBase: ...

    def _copy[T: ContainerBase](
        self: T, 
        target: Optional[ContainerBase] = None
    ) -> T | ContainerBase:
        if target is None:
            target = type(self)()
        
        for k, v in target.__dict__.items():
            setattr(target, k, getattr(self, k, None))
        return target

    def if_[T: ContainerBase](self: T, condition: Any) -> T:
        out = self._copy()
        out._condition = condition
        return out
        
    @abstractmethod
    def _set(self, data: Any) -> None:
        pass
        
    def using(self, data: Any) -> Any:
        if self._condition is not None:
            condition = conjure(self._condition, data)
            if not condition:
                return data

        new_data = data
        if self._expression is not None:
            new_data = conjure(self._expression, data)
        self._set(new_data)
        return data
    
    def __rrshift__(self, data: Any) -> Any:
        return self.using(data)

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
        """ Overridable method to do shallow copy of value based on subclass type."""
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
    `FVar` holds a single value. If it is morphable, then it can be converted to a `FList` or `FDict` if requesting a key when it is empty, or adding a new value if another already exists.
    """
    def __init__(self, morphable: bool = True) -> None:
        super().__init__()
        self.morphable = morphable

    def __getitem__(self, key: str):
        if not self.is_empty:
            raise ValueError("Cannot promote FVar to FDict from non-empty FVar.")
        self.value = {}
        self._key = None
        self.__class__ = FDict  # type: ignore[assignment]
        return self[key]
        
    def _set(self, data: Any) -> None:
        if self.morphable and not self.is_empty:
            self.value = [self.value]
            self.__class__ = FList  # type: ignore[assignment]
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
        # parent is used when morphing a FDict to an FMMap, we need to make sure parent is 
        # morphed too.
        self._parent = None

    def __getitem__(self, key: str):
        if self._key is not None:
            raise KeyError(
                f"Key has already been assigned to {self.__class__.__name__} and no data used yet."
            )

        out = self._copy()
        out._key = key
        out._parent = self
        return out

    @abstractmethod
    def _set_item(self, data: Any) -> None:
        pass
    
    def _set(self, data: Any) -> None:
        if self._key is None:
            raise KeyError(
                f"No key has been provided {self.__class__.__name__}, cannot set value."
            )

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
            self.__class__ = FMMap  # type: ignore[assignment]
            self._parent.__class__ = FMMap
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


class Op:
    strategy: _OpStrategy
    # _condition: Optional[X | Op | bool]
    # _else_: Optional[X | Op]

    def __init__(self, op: X | Callable[..., Any] | Op | list[Op], *args,  **kwargs) -> None:
        # Note: X is callable, but not vice versa.
        if isinstance(op, X):
            if len(args) > 0 or len(kwargs) > 0:
                raise ValueError(
                    "`op` cannot be an instance of `X` if `args` or `kwargs` are provided.")
            self.strategy = _OpX(op)
        elif isinstance(op, Op):
            if len(kwargs) > 0:
                raise ValueError("Cannot have keyword arguments if given instances of `Op`.")
            self.strategy = _OpSerial(op, *args)
        elif isinstance(op, list):
            self.strategy = _OpParallel(op, *args)
        elif isinstance(op, Callable):  # type: ignore[arg-type]
            self.strategy = _OpCallable(op, *args, **kwargs)
        else:
            raise ValueError("Arguments should be of type `X`, `Op`, or Callable.")

        # This indicates that the op is ready to be executed when it is resolvable and data is fed.
        # Sometimes we want to delay the execution of the op until it is needed, for example when  
        # defining parallel ops. This is is useful so we can feed in data without the need for 
        # parentheses. 
        self.final = False

    @classmethod
    def from_strategy(cls, strategy: _OpStrategy) -> Op:
        out = cls.__new__(cls)
        out.strategy = strategy
        # out._condition = None
        # out._else_ = None
        return out
            
    def using(self, data: Any) -> Any:
        # if self._condition is not None:
        #     condition = conjure(self._condition, data)
        #     if condition:
        #         return self.strategy(data)
            
        #     if self._else_ is not None:
        #         return conjure(self._else_, data)
        #     else:
        #         return data

        return self.strategy(data)

    def if_(self, condition: bool | X | Op, else_: Optional[X | Op] = None) -> Op:
        """ 
        Add a condition for executing the op. 
        If condition is False, optionally execute else_ instead, if given. This creates a new copy
        of the op with the condition added. Since condition can be an immediately available bool, 
        or a delayed Op expected to return a bool, we will delay the evaluation of the condition
        until it is needed, so we don't have to split the logic here based on the type of condition.
        """
        strategy = self.strategy.if_(condition, else_)
        out = self.from_strategy(strategy)
        out.final = self.final
        # out._condition = condition
        # out._else_ = else_
        return out
        
    def __rshift__(self, data: Op | nn.Module) -> Op:
        if isinstance(data, Op):
            return Op(self, data)
        elif isinstance(data, nn.Module):
            return Op(self, data(X))
        
        return NotImplemented
        
    def __rrshift__(self, data: Any) -> Any:
        if isinstance(data, nn.Module):
            return Op(data(X), self)

        return self.using(data)

    # def __add__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)        
    #     return Op(operator.add, self, other)

    # def __radd__(self, other: Any) -> Op:
    #     return self.__add__(other)
    
    # def __sub__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.sub, self, other)
    
    # def __rsub__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.sub, other, self)
    
    # def __mul__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.mul, self, other)

    # def __rmul__(self, other: Any) -> Op:
    #     return self.__mul__(other)

    # def __truediv__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.truediv, self, other)
    
    # def __rtruediv__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.truediv, other, self)
    
    # def __floordiv__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.floordiv, self, other)
    
    # def __rfloordiv__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.floordiv, other, self)
    
    # def __mod__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.mod, self, other)
    
    # def __rmod__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.mod, other, self)

    # def __matmul__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.matmul, self, other)
    
    # def __rmatmul__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.matmul, other, self)
    
    # def __pow__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.pow, self, other)
    
    # def __rpow__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.pow, other, self)
    
    # def __and__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.and_, self, other)
    
    # def __rand__(self, other: Any) -> Op:
    #     return self.__and__(other)
    
    # def __or__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.or_, self, other)
    
    # def __ror__(self, other: Any) -> Op:
    #     return self.__or__(other)
    
    # def __xor__(self, other: Any) -> Op:
    #     if isinstance(other, nn.Module):
    #         other = other(X)
    #     return Op(operator.xor, self, other)
    
    # def __rxor__(self, other: Any) -> Op:
    #     return self.__xor__(other)
    
    # def __pos__(self) -> Op:
    #     return Op(operator.pos, self)
    
    # def __neg__(self) -> Op:
    #     return Op(operator.neg, self)

    # def __abs__(self) -> Op:
    #     return Op(operator.abs, self)

    # def __invert__(self) -> Op:
    #     return Op(operator.invert, self)

    def __repr__(self):
        return f"Op({self.strategy!r})"


def op_unary_method[T: Op](oper: Callable[[T], T]) -> Callable[[T], Op]:
    def func(self: T) -> Op:
        return Op(oper, self)
    return func


def op_binary_method[T: Op](oper: Callable[[T, T], T], is_right: bool) -> Callable[[T, Any], Op]:
    
    def func(self: T, other: Any) -> Op:
        if isinstance(other, nn.Module):
            other = other(X)
        return Op(oper, self, other)

    def rfunc(self: T, other: Any) -> Op:
        if isinstance(other, nn.Module):
            other = other(X)
        return Op(oper, other, self)

    return rfunc if is_right else func


for oper, (method, rmethod) in binary_operators.items():
    if method in ("__rshift__", "__lshift__"):
        continue

    setattr(Op, method, op_binary_method(oper, False))
    setattr(Op, rmethod, op_binary_method(oper, True))

for oper, method in unary_operators.items():    
    setattr(Op, method, op_unary_method(oper))


class _OpStrategy(ABC):

    _condition: Optional[X | Op | bool]
    _else_: Optional[X | Op]

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._condition = None
        obj._else_ = None
        return obj
   
    def __call__(self, data: Any) -> Any:
        if self._condition is not None:
            condition = conjure(self._condition, data)
            if condition:
                return self.using(data)
            
            if self._else_ is not None:
                return conjure(self._else_, data)
            else:
                return data
        return self.using(data)

    def if_(self, condition: bool | X | Op, else_: Optional[X | Op] = None):
        out = self._copy()
        out._condition = condition
        out._else_ = else_
        return out

    @abstractmethod
    def _copy(self):
        pass

    @abstractmethod
    def using(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def __repr__(self):
        pass


class _OpX(_OpStrategy):
    """ 
    Op Strategy when `op` is an instance of `X`. E.g.:
    
    ```python
    data >> Op(X[0])
    ```
    """
    def __init__(self, op: X) -> None:
        self.op = op

    def _copy(self):
        return type(self)(self.op)

    def using(self, data: Any) -> Any:
        return conjure(self.op, data)

    def __repr__(self):
        return f"{self.op!r}"


class _OpCallable(_OpStrategy):
    """ 
    Op Strategy when `op` is a Callable. E.g.:
    
    ```python
    data >> Op(torch.cat, [X[0], X[1]], dim=1)
    ```
    """
    def __init__(self, op: Callable[..., Any], *args, **kwargs) -> None:
        self.op = op
        self.args = A(*args, **kwargs)

    def _copy(self):
        return type(self)(self.op, *self.args.args, **self.args.kwargs)
    
    def using(self, data: Any) -> Any:
        resolved = data >> self.args
        return self.op(*resolved.args, **resolved.kwargs)

    def __repr__(self):
        try:
            name = self.op.__name__
        except AttributeError:
            name = f"{self.op!r}"
        
        if name == "Module.__call__" and len(self.args.args) > 0:
            name, *args = self.args.args
        else:
            args = self.args.args

        args = ", ".join(map(repr, args))
        kwargs = ", ".join(f"{k}={v!r}" for k, v in self.args.kwargs.items())

        if kwargs:
            args = args + ", " + kwargs

        return f"{name}({args})"


class _OpSerial(_OpStrategy):
    """ 
    Op Strategy when `op` is a list of `Op` Instances. E.g.:
    
    ```python
    linear1 >> linear2
    ```

    which is the same as `Op(linear1, linear2)`. This calls each op in sequence.
    """
    def __init__(self, *args: Op) -> None:
        for op in args:
            if not isinstance(op, Op):
                raise ValueError("All arguments must be of type `Op`.")
        self.ops = args

    def _copy(self):
        return type(self)(*self.ops)
    
    def using(self, data: Any) -> Any:
        for op in self.ops:
            data = op.using(data)
        return data

    def __rshift__(self, data: Op) -> _OpSerial:
        return _OpSerial(*self.ops, data)

    def __repr__(self):
        return f"\n  {'\n  >> '.join(map(repr, self.ops))}\n"


class _OpParallel(_OpStrategy):
    """ 
    Apply consecutive ops in parallel. When an op is added to the list, 

    ```python
    linear1 >> linear2
    ```

    which is the same as `Op(linear1, linear2)`. This calls each op in parallel.
    """
    def __init__(self, *args: list[Op]) -> None:
        for arg in args:
            if not isinstance(arg, Op) or not all(isinstance(item, Op) for item in arg):
                raise ValueError("All arguments must be of a list of type `Op`.")

        lengths = {len(arg) for arg in args}
        lengths.discard(1)
        if len(lengths) != 1:
            raise ValueError(
                "All arguments to `Op/_OpParallel` strategy must have the same length."
            )
            
        self.ops = args

    def _copy(self):
        pass

    def using(self, data: Any) -> Any:
        if self.wire is None:
            raise ValueError("`wire` must be provided to `Op/_OpParallel` strategy.")
        
        for serial_ops in zip(*self.ops):
            for op in serial_ops:
                data = op.using(data)
        
            data >> self.wire
            data = op.using(data)
        return data

    def __repr__(self):
        pass


class Wiring(enum.Enum):
    Fanout = "Fanout"
    Passthru = "Passthru"
    InOut = "InOut"

    def __getitem__(self, key: str) -> Any:
        pass
    
    def __call__(self, *args, **kwargs) -> Any:
        pass
        

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
            elif not isinstance(given_wire.arguments[k], Wiring):
                continue
                
            match(given_wire.arguments[k]):
                case Wiring.Fanout:
                    self._fanout[k] = iter(v)
                    bound.arguments[k] = next(self._fanout[k])
                case Wiring.Passthru:
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
