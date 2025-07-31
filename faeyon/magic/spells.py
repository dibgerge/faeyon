import inspect
import enum
import itertools
from collections.abc import Callable, Iterator
from typing import Any, Optional
from faeyon.utils import is_ipython
from abc import ABC, abstractmethod
from collections import defaultdict


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


def _meta_method[T](name: str) -> Callable[..., T]:
    def method(cls: type[T], *args, **kwargs) -> T:
        # Call the constructor method to return an instance of the class
        # This removes the requirement to explicity initialize the class with `X()`, since 
        # the latter should be interpreted as a function call
        obj = super(_MetaX, cls).__call__()  # type: ignore
        return getattr(obj, name)(*args, **kwargs)
    return method


def _x_method[T](name: str) -> Callable[..., T]:
    def method(self, *args, **kwargs):
        self._buffer.append((name, args, kwargs))
        return self

    def __getattr__(self, key: str):
        # Bypass IPython's internal check for the _ipython_canary_method_should_not_exist_ attribute
        # This is required to make IPython display the object's contents
        if is_ipython() and key == "_ipython_canary_method_should_not_exist_":
            return self
        return method(self, key)

    if name == "__getattr__":
        return __getattr__

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


class FaeArgs:
    """
    A placeholder for mapping next operation's arguments to the left side of the >> operator.

    The arguments are stored in `args` and `kwargs` attributes will be used to specify the 
    operations of the next layer. Their values can be used to access the previous layer using the 
    `X` placeholder, or just provide static arguments if that's all you need.
    """
    def __init__(self, *args, **kwargs) -> None:
        self._is_resolved = not any(
            isinstance(item, X) for item in itertools.chain(args, kwargs.values())
        )

        self.args = args
        self.kwargs = kwargs

    def call(self, func: Callable[..., Any]) -> Any:
        if not self.is_resolved:
            raise ValueError(
                f"Cannot call {func} with unresolved arguments. Feed data `FaeArgs` to resolve it."
            )
        return func(*self.args, **self.kwargs)

    def __rshift__(self, func: Callable[..., Any]) -> Any:
        return self.call(func)

    @property
    def is_resolved(self) -> bool:
        return self._is_resolved

    @staticmethod
    def _resolve_item(item: Any, data: Any) -> Any:
        if not isinstance(item, X):
            return item
        
        for name, args, kwargs in item:
            data = getattr(data, name)(*args, **kwargs)
        
        return data
         
    def bind(self, data: Any) -> "FaeArgs":
        if self.is_resolved:
            return self
        
        resolved_args = tuple(self._resolve_item(arg, data) for arg in self.args)
        resolved_kwargs = {k: self._resolve_item(v, data) for k, v in self.kwargs.items()}
        return FaeArgs(*resolved_args, **resolved_kwargs)

    def __rrshift__(self, data: Any) -> "FaeArgs":
        return self.bind(data)


class _Variable:
    """
    A wrappert to hold a container value so that it can be passed by reference across different
    Container selections.
    """
    _value: Any
    
    def __init__(self, *args) -> None:
        self.empty = False

        if len(args) == 1:
            self._value = args[0]
        elif len(args) > 1:
            raise ValueError("`_Variable` can only be initialized with one or no arguments.")
        else:
            self.empty = True
    
    @property
    def value(self) -> Any:
        if self.empty:
            raise ValueError("No value.")
        return self._value

    @value.setter
    def value(self, val: Any) -> None:
        self._value = val
        self.empty = False

    def __repr__(self):
        return f"_Variable({self.value!r})"



class ContainerBase(ABC):
    
    def __init__(self, *args) -> None:
        self._value: Any = _Variable(*args)
        self._expression: Optional[X] = None
    
    def select[T: ContainerBase](self: T, expression: X) -> T:
        if self._expression is not None:
            raise ValueError(
                f"Cannot reassign expression to {self.__class__.__name__}, "
                "since expression has not been used."
            )

        if not isinstance(expression, X):
            raise ValueError(
                f"Cannot assign expression to {self.__class__.__name__}, "
                "since expression is not an instance of `X`."
            )

        out = self._copy()
        out._expression = expression
        return out

    def __matmul__[T: ContainerBase](self: T, expression: X) -> T:
        return self.select(expression)

    def __rmatmul__[T: ContainerBase](self: T, expression: X) -> T:
        return self.select(expression)

    def _copy[T: ContainerBase](self: T) -> T:
        out = type(self)()
        for k, v in self.__dict__.items():
            setattr(out, k, v)
        return out
        
    @abstractmethod
    def _set(self, data: Any) -> None:
        pass
        
    def using(self, data: Any) -> Any:
        resolved_data = data
        if self._expression is not None:
            for name, args, kwargs in self._expression:
                resolved_data = getattr(resolved_data, name)(*args, **kwargs)
        
        self._set(resolved_data)
        return data
    
    def __rrshift__(self, data: Any) -> Any:
        return self.using(data)

    @property
    def is_selected(self) -> bool:
        return self._expression is not None

    @property
    @abstractmethod
    def is_appendable(self) -> bool:
        pass

    @property
    def sheddable(self) -> bool:
        return not self._value.empty and not self.is_selected

    def shed(self) -> Any:
        if not self.sheddable:
            raise ValueError(
                "Cannot shed value from {self.__class__.__name__} with no value or a "
                "pending select."
            )
        return self._value.value

    def __pos__(self) -> Any:
        return self.shed()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._value.value})"
    

class FaeList(ContainerBase):
    def __init__(self, *args) -> None:
        super().__init__(list(args))
        
    def _set(self, data: Any) -> None:
        self._value.value.append(data)

    def __len__(self):
        return len(self._value.value)

    @property
    def is_appendable(self) -> bool:
        return True
    

class FaeVar(ContainerBase):
    def __init__(self, strict: bool = True) -> None:
        super().__init__()
        self.strict = strict
        
    def _set(self, data: Any) -> None:
        if not self._value.empty:
            if self.strict:
                raise ValueError(
                    "Cannot bind a value to a strict FaeVar with existing value."
                )
        
        self._value.value = data

    @property
    def is_appendable(self) -> bool:
        return False


class KeyedContainer(ContainerBase):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self._key = None

    def __getitem__(self, key: str):
        if self._key is not None:
            raise KeyError(
                "Key has already been assigned to {self.__class__.__name__} and no data used yet."
            )

        out = self._copy()
        out._key = key
        return out

    @abstractmethod
    def _set_item(self, data: Any) -> None:
        pass
    
    def _set(self, data: Any) -> None:
        if self._key is None:
            raise KeyError(
                "No key has been provided {self.__class__.__name__}, cannot set value."
            )

        self._set_item(data)

    def __len__(self):
        return len(self._value.value)


class FaeDict(KeyedContainer):
    def __init__(self, strict: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.strict = strict

    def _set_item(self, data: Any) -> None:
        if self._key in self._value.value:
            if self.strict:
                raise ValueError(
                    "Cannot bind a value to a strict FaeDict with existing key."
                )
        self._value.value[self._key] = data

    @property
    def is_appendable(self) -> bool:
        return self._key is None


class FaeMultiMap(KeyedContainer):
    def __init__(self, **kwargs) -> None:
        for value in kwargs.values():
            if not isinstance(value, list):
                raise ValueError(
                    "All values in FaeMultiMap must be lists."
                )

        super().__init__(**kwargs)
        self._value.value = defaultdict(list, self._value.value)

    def _set_item(self, data: Any) -> None:
        self._value.value[self._key].append(data)

    def shed(self) -> Any:
        return dict(super().shed())

    @property
    def is_appendable(self) -> bool:
        return True


class Op:
    def __init__(self, op: X, /) -> None:
        if not isinstance(op, X):
            raise ValueError("Operations should use the `X` buffer.")
        self.op = op
            
    def using(self, data: Any) -> Any:
        for name, args, kwargs in self.op:
            data = getattr(data, name)(*args, **kwargs)
        return data

    def __rrshift__(self, data: Any) -> Any:
        return self.using(data)

    
    def __repr__(self):
        return f"Op({self.op!r})"


class Wiring(enum.StrEnum):
    Fanout = "Fanout"
    Passthru = "Passthru"


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
    
    def init(self, sig: inspect.Signature, *args, **kwargs) -> FaeArgs:
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
        return FaeArgs(*bound.args, **bound.kwargs)

    def step(self, data: Any) -> FaeArgs:
        if self._current_wire is None:
            raise ValueError("No wire has been initialized.")
        
        for k, v in self._fanout.items():
            try:
                v = next(v)
            except StopIteration:
                raise ValueError(f"Fanout argument {k} has no more values.")
            else:
                self._current_wire.arguments[k] = v
                
        return data >> FaeArgs(*self._current_wire.args, **self._current_wire.kwargs)

    def __rrshift__(self, data: Any) -> FaeArgs:
        return self.step(data)
