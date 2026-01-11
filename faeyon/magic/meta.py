import operator

from abc import ABCMeta
from collections.abc import Callable
from typing import Any


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

unary_operators: dict[Callable[[Any], Any], str] = {
    operator.abs: "__abs__",
    operator.invert: "__invert__",
    operator.neg: "__neg__",
    operator.pos: "__pos__",
}


# Define methods supported by the _MetaX metaclass
_meta_methods = {
    "__lt__": "{X} < {}",
    "__le__": "{X} <= {}",
    "__eq__": "{X} == {}",
    "__ne__": "{X} != {}",
    "__gt__": "{X} > {}",
    "__ge__": "{X} >= {}",
    "__getattr__": "{X}.{}",

    # Emulating callables
    "__call__": "{X}({})",
    
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


class _MetaOps[T: type](ABCMeta):
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
    def __add__(cls, other: Any) -> T:
        return super().__call__() + other

    def __radd__[T](cls: type[T], other: Any) -> T:
        return other + super().__call__()

    def __sub__[T](cls: type[T], other: Any) -> T:
        return super().__call__() - other

    def __rsub__[T](cls: type[T], other: Any) -> T:
        return other - super().__call__()

    def __mul__[T](cls: type[T], other: Any) -> T:
        return  super().__call__() * other

    def __rmul__[T](cls: type[T], other: Any) -> T:
        return other * super().__call__()

    def __matmul__[T](cls: type[T], other: Any) -> T:
        return super().__call__() @ other

    def __rmatmul__[T](cls: type[T], other: Any) -> T:
        return other @ super().__call__()

    def __truediv__[T](cls: type[T], other: Any) -> T:
        return super().__call__() / other

    def __rtruediv__[T](cls: type[T], other: Any) -> T:
        return other / super().__call__()
    
    def __floordiv__[T](cls: type[T], other: Any) -> T:
        return super().__call__() // other

    def __rfloordiv__[T](cls: type[T], other: Any) -> T:
        return other // super().__call__()
    
    def __mod__[T](cls: type[T], other: Any) -> T:
        return super().__call__() % other

    def __rmod__[T](cls: type[T], other: Any) -> T:
        return other % super().__call__()
    
    def __divmod__[T](cls: type[T], other: Any) -> T:
        return divmod(super().__call__(), other)

    def __rdivmod__[T](cls: type[T], other: Any) -> T:
        return divmod(other, super().__call__())
    
    def __pow__[T](cls: type[T], other: Any) -> T:
        return super().__call__() ** other

    def __rpow__[T](cls: type[T], other: Any) -> T:
        return other ** super().__call__()
    
    def __lshift__[T](cls: type[T], other: Any) -> T:
        return super().__call__() << other
    
    def __rlshift__[T](cls: type[T], other: Any) -> T:
        return other << super().__call__()
    
    def __rshift__[T](cls: type[T], other: Any) -> T:
        return super().__call__() >> other

    def __rrshift__[T](cls: type[T], other: Any) -> T:
        return other >> super().__call__()

    def __and__[T](cls: type[T], other: Any) -> T:
        return super().__call__() & other

    def __rand__[T](cls: type[T], other: Any) -> T:
        return other & super().__call__()

    def __or__[T](cls: type[T], other: Any) -> T:
        return super().__call__() | other
    
    def __ror__[T](cls: type[T], other: Any) -> T:
        return other | super().__call__()

    def __xor__[T](cls: type[T], other: Any) -> T:
        return super().__call__() ^ other

    def __rxor__[T](cls: type[T], other: Any) -> T:
        return other ^ super().__call__()

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

    def __trunc__[T](cls: type[T]) -> T:
        return trunc(super().__call__())

    def __floor__[T](cls: type[T]) -> T:
        return floor(super().__call__())

    def __ceil__[T](cls: type[T]) -> T:
        return ceil(super().__call__())

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
        return 0

    def __iter__(cls):
        return iter([])

    def __repr__(cls):
        return cls.__name__

    def __instancecheck__(cls, instance):
        return (
            super().__instancecheck__(instance) 
            or (isinstance(instance, type) and issubclass(instance, cls))
        )
