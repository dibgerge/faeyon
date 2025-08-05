from __future__ import annotations

import sys
import inspect
import itertools
import enum
import math
import operator

from typing import Any, overload, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator

from faeyon.utils import is_ipython
from torch import nn


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
    def method(self, *args, **kwargs) -> Any:
        self._buffer.append((name, args, kwargs))
        return self

    def _getattr_(self, key: str) -> Any:
        # Bypass IPython's internal check for the _ipython_canary_method_should_not_exist_ attribute
        # This is required to make IPython display the object's contents
        if is_ipython() and key == "_ipython_canary_method_should_not_exist_":
            return self
        return method(self, key)

    if name == "__getattr__":
        return _getattr_

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
        data = getattr(data, name)(*args, **kwargs)
    return data
        

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
         
    def bind(self, data: Any) -> "FaeArgs":
        if self.is_resolved:
            return self
        
        resolved_args = tuple(conjure(arg, data) for arg in self.args)
        resolved_kwargs = {k: conjure(v, data) for k, v in self.kwargs.items()}
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
        if self._expression is not None:
            data = conjure(self._expression, data)
        self._set(data)
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


class _OpStrategy(ABC):
    @abstractmethod
    def __call__(self, data: Any) -> Any:
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

    def __call__(self, data: Any) -> Any:
        return conjure(self.op, data)


class _OpCallable(_OpStrategy):
    """ 
    Op Strategy when `op` is a Callable. E.g.:
    
    ```python
    data >> Op(torch.cat, [X[0], X[1]], dim=1)
    ```
    """
    def __init__(self, op: Callable[..., Any], *args, **kwargs) -> None:
        self.op = op
        self.args = FaeArgs(*args, **kwargs)

    def __call__(self, data: Any) -> Any:
        resolved = data >> self.args
        return self.op(*resolved.args, **resolved.kwargs)


class _OpOp(_OpStrategy):
    """ 
    Op Strategy when `op` is a list of `Op` Instances. E.g.:
    
    ```python
    linear1 >> linear2
    ```

    which is the same as `Op(linear1, linear2)`. This calls each op in sequence.
    """
    def __init__(self, *args: "Op") -> None:
        for op in args:
            if not isinstance(op, Op):
                raise ValueError("All arguments must be of type `Op`.")
        self.ops = args

    def __call__(self, data: Any) -> Any:
        for op in self.ops:
            data = op.using(data)
        return data


class Op:
    strategy: _OpStrategy

    def __init__(self, op: X | Callable[..., Any] | "Op", *args,  **kwargs) -> None:
      
        # Note: X is callable, but not vice versa.
        if isinstance(op, X):
            if len(args) > 0 or len(kwargs) > 0:
                raise ValueError(
                    "`op` cannot be an instance of `X` if `args` or `kwargs` are provided.")
            self.strategy = _OpX(op)
        elif isinstance(op, Op):
            if len(kwargs) > 0:
                raise ValueError("Cannot have keyword arguments if given instances of `Op`.")
            self.strategy = _OpOp(op, *args)
        elif isinstance(op, Callable):  # type: ignore[arg-type]
            self.strategy = _OpCallable(op, *args, **kwargs)
        else:
            raise ValueError("Arguments should be of type `X`, `Op`, or Callable.")
            
    def using(self, data: Any) -> Any:
        return self.strategy(data)
        
    def __rrshift__(self, data: Any) -> Any:
        return self.using(data)

    def __add__(self, other: Any) -> Op:
        return Op(operator.add, self, other)
    
    def __sub__(self, other: Any) -> Op:
        return Op(operator.sub  , self, other)
    
    def __mul__(self, other: Any) -> Op:
        return Op(operator.mul, self, other)

    def __truediv__(self, other: Any) -> Op:
        return Op(operator.truediv, self, other)
    
    def __rtruediv__(self, other: Any) -> Op:
        return Op(operator.truediv, other, self)
    
    def __floordiv__(self, other: Any) -> Op:
        return Op(operator.floordiv, self, other)
    
    def __rfloordiv__(self, other: Any) -> Op:
        return Op(operator.floordiv, other, self)
    
    def __mod__(self, other: Any) -> Op:
        return Op(operator.mod, self, other)
    
    def __rmod__(self, other: Any) -> Op:
        return Op(operator.mod, other, self)
    
    def __pos__(self) -> Op:
        return Op(operator.pos, self)
    
    def __neg__(self) -> Op:
        return Op(operator.neg, self)

    def __abs__(self) -> Op:
        return Op(operator.abs, self)

    def __invert__(self) -> Op:
        return Op(operator.invert, self)

    def __round__(self, ndigits: int = 0) -> Op:
        return Op(round, self, ndigits)

    def __trunc__(self) -> Op:
        return Op(math.trunc, self)

    def __floor__(self) -> Op:
        return Op(math.floor, self)

    def __ceil__(self) -> Op:
        return Op(math.ceil, self)

    def __repr__(self):
        return f"Op({self.op!r})"


class Wiring(enum.Enum):
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


class FaeTree:
    def __init__(self) -> None:
        pass

    def __rrshift__(self, data: Any) -> Any:
        """ This will evaluate the tree using the data provided as the starting point. """
        if not root:
            return 0
        
        stack = [(root, 1)]
        ans = 0
        
        while stack:
            node, depth = stack.pop()
            ans = max(ans, depth)
            if node.left:
                stack.append((node.left, depth + 1))
            if node.right:
                stack.append((node.right, depth + 1))
        
        return ans


class FaeNode:
    def __init__(self, caller, op_name, other: Optional[nn.Module] = None) -> None:
        self.caller = caller
        self.op_name = op_name
        self.other = other
        
    def __rrshift__(self, data: Any) -> Any:
        pass


def _new_instance(cls, *args, **kwargs):
    instance = object.__new__(cls)
    sig = inspect.signature(cls.__init__)
    bound = sig.bind(instance, *args, **kwargs)
    bound.apply_defaults()
    del bound.arguments["self"]
    instance._arguments = bound
    return instance


def __new__(cls, *args, **kwargs):
    """
    Allow `nn.Module` to save constructor arguments passed to it, so that the could be used 
    later for cloning modules.

    When any of the arguments is of type `FaeList` or `FaeDict`, special handing is applied to 
    generate clones.
    """
    try:
        kwargs_keys, kwargs_values = zip(*kwargs.items())
    except ValueError:
        kwargs_keys, kwargs_values = [], []
    
    raveled_args = []
    num_faelist = 0
    num_faedict = 0
    fae_keys = None
    fae_len = None
    for arg in itertools.chain(args, kwargs_values):
        if isinstance(arg, FaeList):
            num_faelist += 1
            arg_value = arg.shed()

            if fae_len is None:
                fae_len = len(arg_value)
            elif fae_len != len(arg_value):
                raise ValueError("All arguments of type `FaeList` must have the same length.")

            raveled_args.append(arg_value)
        elif isinstance(arg, FaeDict):
            num_faedict += 1
            arg_value = arg.shed()

            if fae_keys is None:
                fae_keys = list(arg_value.keys())
                fae_len = len(fae_keys)
            else:
                if set(fae_keys) != set(arg_value.keys()):
                    raise ValueError("All arguments of type `FaeDict` must have the same keys.")

            raveled_args.append([arg_value[k] for k in fae_keys])
        else:
            raveled_args.append(itertools.repeat(arg))

    if num_faelist > 0 and num_faedict > 0:
        raise ValueError("Cannot mix `FaeList` and `FaeDict` arguments. Choose one.")

    num_fae = max(num_faelist, num_faedict)

    # No argument parametrization, return a regular nn.Module instance
    if num_fae == 0:
        return _new_instance(cls, *args, **kwargs)

    nkeys = len(kwargs_keys)
    nargs = len(raveled_args)
    args = [val[:nargs-nkeys] for val in zip(*raveled_args)]
    kwargs = [dict(zip(kwargs_keys, val[nargs-nkeys:])) for val in zip(*raveled_args)]

    out = []
    for c_args, c_kwargs in zip(args, kwargs):
        inst = _new_instance(cls, *c_args, **c_kwargs)
        # Call __init__ on each instance since it will not be called when different class type is 
        # returned in __new__
        cls.__init__(inst, *c_args, **c_kwargs)
        out.append(inst)
    
    if fae_keys is not None:
        out = dict(zip(fae_keys, out))

    return out


def __default_new__(cls, *args, **kwargs):
    """ Once we override __new__ in `nn.Module`, we cannot restore the old one, since nn.Module 
    (as of PyTorch 2.7) does not implement `__new__`, and hence expect it to have no arguments. 
    The custom __new__ method we implemented above does not match this signature, and hence 
    we cannot restore the old one. As a workaround, we define a default __new__ method that 
    matches the signature of the default __new__ method in `nn.Module`, but calls the parent object 
    without any arguments.
    """
    return object.__new__(cls)


@overload
def __mul__[T: nn.Module](self: T, other: int) -> list[T]: ...

@overload
def __mul__[T: nn.Module](self: T, other: nn.Module) -> Op: ...


def __mul__[T: nn.Module](self: T, other: int | nn.Module) -> list[T] | Op:
    """
    Creates a ModuleList of `other` clones of this module.
    """
    if isinstance(other, nn.Module):
        left = self(X)
        right = other(X)
        return Op(left, getattr(X, "__mul__")(right))

    if not isinstance(other, int):
        raise TypeError(
            f"Cannot multiply {self} with {type(other)}. Only multiplication by "
            f"int is supported."
        )
    if other < 1:
        raise ValueError("Number of modules must be greater than 0.")

    return [self.clone() for _ in range(other)]


@overload
def __rmul__[T: nn.Module](self: T, other: int) -> list[T]: ...


@overload
def __rmul__[T: nn.Module](self: T, other: nn.Module) -> Op: ...


def __rmul__[T: nn.Module](self: T, other: int | nn.Module) -> list[T] | Op:
    """ Multiplication is commutative!"""
    return self.__mul__(other)  # type: ignore


def __rrshift__[T: nn.Module](self: T, other: Any) -> Any:
    """
    This is an alias for `__call__`. The limitation here is that it only works for 
    single inputs. If you need to pass multiple inputs, use the `FaeArgs` class.
    """
    return self(other)


def clone[T: nn.Module](self: T, *args: Any, **kwargs: Any) -> T:
    """
    Create a new instance of the same module with the same arguments. This method should be used 
    carefully, since it is does not do any deep copying on all types of module arguments. 

    The module is cloned based on the arguments passed to its constructor during its creation. 
    If any of the arguments were changed after the module was created, the changes will not be 
    reflected in the cloned module unless the changes were made on the arguments itslef inplace 
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


def __call__[T: nn.Module](self: T, *args: Any, **kwargs: Any) -> Any:
    fae_args = FaeArgs(*args, **kwargs)
    if fae_args.is_resolved:
        return faek.module__call__(self, *args, **kwargs)
    
    return Op(faek.module__call__, self, *args, **kwargs)

    
def delayed_unary_method[T: nn.Module](op_name: str) -> Callable[[T], Op]:
    def func(self: T) -> Op:
        return getattr(self(X), op_name)()
    return func


def delayed_binary_method[T: nn.Module](op_name: str) -> Callable[[T, nn.Module], Op]:
    def func_binary(self: T, other: nn.Module) -> Op:
        return getattr(self(X), op_name)(other(X))
    return func_binary
    

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
        "__call__",
        "__mul__",
        "__rmul__",
        "__rrshift__",
        "clone",
    ]

    # Note: `__mul__` is a special case, since it can operate on integers and module types.
    delayed_binary_methods = [
        "__add__",
        "__sub__",
        "__matmul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__divmod__",
        "__pow__",
        "__and__",
        "__or__",
        "__xor__",
    ]
    delayed_unary_methods = [
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__",
        "__round__",
        "__trunc__",
        "__floor__",
        "__ceil__",
    ]

    def __init__(self):
        self._is_on = False
        self.module__call__ = nn.Module.__call__

    def __enter__(self):
        self.on()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.off()

    def on(self):
        if self._is_on:
            return

        current_module = sys.modules[__name__]
        for method in self.methods:
            setattr(nn.Module, method, getattr(current_module, method))

        nn.Module.__new__ = staticmethod(__new__)
        # for method in Faek.delayed_binary_methods:
        #     setattr(nn.Module, method, delayed_binary_method(method))
        # for method in Faek.delayed_unary_methods:
        #     setattr(nn.Module, method, delayed_unary_method(method))

        self._is_on = True

    def off(self):
        if not self._is_on:
            return
        
        for method in self.methods:
            delattr(nn.Module, method)

        nn.Module.__new__ = staticmethod(__default_new__)
        nn.Module.__call__ = self.module__call__
        self._is_on = False


faek = Faek()
