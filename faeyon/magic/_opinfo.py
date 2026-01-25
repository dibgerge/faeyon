import dataclasses
import enum
import operator

from collections.abc import Callable
from typing import Any, Optional, overload
from functools import lru_cache


class OperatorType(enum.Flag):
    BINARY = enum.auto()
    RBINARY = enum.auto()
    COMPARISON = enum.auto()
    UNARY = enum.auto()
    UTILITY = enum.auto()
    ARITHMETIC = BINARY | RBINARY | UNARY


def _call(data: Any, *args: Any, **kwargs: Any) -> Any:
    return data(*args, **kwargs)


def _getattr(data: Any, name: str) -> Any:
    return getattr(data, name)


def _getitem(data: Any, key: Any) -> Any:
    return data[key]


def get_precedence(item) -> Optional[int]:
    """
    Gets the precedence of the item. Item can be an OpInfo or F object. 
    """
    from .spells import F
    if isinstance(item, F):
        return item._fae_op.precedence
    elif isinstance(item, OpInfo):
        return item.precedence
    return None


@dataclasses.dataclass
class OpInfo:
    name: str
    type: OperatorType
    fmt: str
    operator: Callable[[..., Any], Any]
    precedence: int = 0

    @property
    def is_right(self) -> bool:
        return self.type == OperatorType.RBINARY

    @property
    def is_binary(self) -> bool:
        return self.type in {OperatorType.BINARY, OperatorType.RBINARY}

    @property
    def attr_name(self) -> str:
        return f"__{self.name}__"

    def _add_parens(self, item) -> str:
        precedence = get_precedence(item)
        if precedence and precedence > self.precedence:
            return f"({item})"
        return item

    def to_string(self, X: Any, *args: Any, **kwargs: Any) -> str:
        """
        Each item in args and kwargs will be formatted individually based on the 
        format_spec/conversion in the "arg" field of the format string.
        Then all arguments will be joined together with a comma.
        """
        out_args = []
        for arg in args:
            out_args.append(self._add_parens(arg))

        # If one argument is given, just use it directly
        if len(out_args) == 1 and not kwargs:
            arg = out_args[0]

            # Special case to support X('foo') instead of X(foo)
            if self.name == "call" and isinstance(arg, str):
                arg = repr(arg)
            return self.fmt.format(X=self._add_parens(X), arg=arg)
        
        out_args = list(map(repr, out_args))

        for k, v in kwargs.items():
            out_args.append(f"{k}={self._add_parens(v)!r}")

        return self.fmt.format(X=self._add_parens(X), arg=", ".join(out_args))

    def __call__(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        if self.type == OperatorType.UNARY:
            return self.operator(data)
        elif self.type == OperatorType.BINARY:
            return self.operator(data, args[0])
        elif self.type == OperatorType.RBINARY:
            return self.operator(args[0], data)
        elif self.type == OperatorType.COMPARISON:
            return self.operator(data, args[0])
        else:
            return self.operator(data, *args, **kwargs)


ops = [
    OpInfo(
        name="pow", 
        type=OperatorType.BINARY, 
        operator=operator.pow, 
        fmt="{X} ** {arg}",
        precedence=10
    ),
    OpInfo(
        name="rpow", 
        type=OperatorType.RBINARY, 
        operator=operator.pow, 
        fmt="{arg} ** {X}",
        precedence=10
    ),
    OpInfo(
        name="invert", 
        type=OperatorType.UNARY, 
        operator=operator.invert, 
        fmt="~{X}",
        precedence=20
    ),
    OpInfo(
        name="neg", 
        type=OperatorType.UNARY, 
        operator=operator.neg, 
        fmt="-{X}",
        precedence=20
    ),
    OpInfo(
        name="pos", 
        type=OperatorType.UNARY, 
        operator=operator.pos, 
        fmt="+{X}",
        precedence=20
    ),
    OpInfo(
        name="mul", 
        type=OperatorType.BINARY, 
        operator=operator.mul, 
        fmt="{X} * {arg}",
        precedence=30
    ),
    OpInfo(
        name="rmul", 
        type=OperatorType.RBINARY, 
        operator=operator.mul, 
        fmt="{arg} * {X}",
        precedence=30
    ),
    OpInfo(
        name="truediv", 
        type=OperatorType.BINARY, 
        operator=operator.truediv, 
        fmt="{X} / {arg}",
        precedence=30
    ),
    OpInfo(
        name="rtruediv", 
        type=OperatorType.RBINARY, 
        operator=operator.truediv, 
        fmt="{arg} / {X}",
        precedence=30
    ),
    OpInfo(
        name="floordiv", 
        type=OperatorType.BINARY, 
        operator=operator.floordiv, 
        fmt="{X} // {arg}",
        precedence=30
    ),
    OpInfo(
        name="rfloordiv", 
        type=OperatorType.RBINARY, 
        operator=operator.floordiv, 
        fmt="{arg} // {X}",
        precedence=30
    ),
    OpInfo(
        name="mod", 
        type=OperatorType.BINARY, 
        operator=operator.mod, 
        fmt="{X} % {arg}",
        precedence=30
    ),
    OpInfo(
        name="rmod", 
        type=OperatorType.RBINARY, 
        operator=operator.mod, 
        fmt="{arg} % {X}",
        precedence=30
    ),
    OpInfo(
        name="matmul", 
        type=OperatorType.BINARY, 
        operator=operator.matmul, 
        fmt="{X} @ {arg}",
        precedence=30
    ),
    OpInfo(
        name="rmatmul", 
        type=OperatorType.RBINARY, 
        operator=operator.matmul, 
        fmt="{arg} @ {X}",
        precedence=30
    ),
    OpInfo(
        name="add", 
        type=OperatorType.BINARY, 
        operator=operator.add, 
        fmt="{X} + {arg}",
        precedence=40
    ),
    OpInfo(
        name="radd", 
        type=OperatorType.RBINARY, 
        operator=operator.add, 
        fmt="{arg} + {X}",
        precedence=40
    ),
    OpInfo(
        name="sub", 
        type=OperatorType.BINARY, 
        operator=operator.sub, 
        fmt="{X} - {arg}",
        precedence=40
    ),
    OpInfo(
        name="rsub", 
        type=OperatorType.RBINARY, 
        operator=operator.sub, 
        fmt="{arg} - {X}",
        precedence=40
    ),

    OpInfo(
        name="rshift", 
        type=OperatorType.BINARY, 
        operator=operator.rshift, 
        fmt="{X} >> {arg}",
        precedence=50
    ),
    OpInfo(
        name="rrshift", 
        type=OperatorType.RBINARY, 
        operator=operator.rshift, 
        fmt="{arg} >> {X}",
        precedence=50
    ),
    OpInfo(
        name="lshift", 
        type=OperatorType.BINARY, 
        operator=operator.lshift, 
        fmt="{X} << {arg}",
        precedence=50
    ),
    OpInfo(
        name="rlshift", 
        type=OperatorType.RBINARY, 
        operator=operator.lshift, 
        fmt="{arg} << {X}",
        precedence=50
    ),
    OpInfo(
        name="and", 
        type=OperatorType.BINARY, 
        operator=operator.and_, 
        fmt="{X} & {arg}",
        precedence=60
    ),
    OpInfo(
        name="rand", 
        type=OperatorType.RBINARY, 
        operator=operator.and_, 
        fmt="{arg} & {X}",
        precedence=60
    ),

    OpInfo(
        name="xor", 
        type=OperatorType.BINARY, 
        operator=operator.xor, 
        fmt="{X} ^ {arg}",
        precedence=70
    ),
    OpInfo(
        name="rxor", 
        type=OperatorType.RBINARY, 
        operator=operator.xor, 
        fmt="{arg} ^ {X}",
        precedence=70
    ),

    OpInfo(
        name="or", 
        type=OperatorType.BINARY, 
        operator=operator.or_, 
        fmt="{X} | {arg}",
        precedence=80
    ),
    OpInfo(
        name="ror", 
        type=OperatorType.RBINARY, 
        operator=operator.or_, 
        fmt="{arg} | {X}",
        precedence=80
    ),
    OpInfo(
        name="lt", 
        type=OperatorType.COMPARISON, 
        operator=operator.lt, 
        fmt="{X} < {arg}",
        precedence=90
    ),
    OpInfo(
        name="le", 
        type=OperatorType.COMPARISON, 
        operator=operator.le, 
        fmt="{X} <= {arg}",
        precedence=90
    ),
    OpInfo(
        name="eq", 
        type=OperatorType.COMPARISON, 
        operator=operator.eq, 
        fmt="{X} == {arg}",
        precedence=90
    ),
    OpInfo(
        name="ne", 
        type=OperatorType.COMPARISON, 
        operator=operator.ne, 
        fmt="{X} != {arg}",
        precedence=90
    ),
    OpInfo(
        name="gt", 
        type=OperatorType.COMPARISON, 
        operator=operator.gt, 
        fmt="{X} > {arg}",
        precedence=90
    ),
    OpInfo(
        name="ge", 
        type=OperatorType.COMPARISON, 
        operator=operator.ge, 
        fmt="{X} >= {arg!r}",
        precedence=90
    ),
    OpInfo(
        name="divmod", 
        type=OperatorType.BINARY, 
        operator=divmod,
        fmt="divmod({X}, {arg})",
        precedence=0
    ),
    OpInfo(
        name="rdivmod", 
        type=OperatorType.RBINARY, 
        operator=divmod,
        fmt="divmod({arg}, {X})",
        precedence=0
    ),
    OpInfo(
        name="abs", 
        type=OperatorType.UTILITY, 
        operator=operator.abs, 
        fmt="abs({X})",
        precedence=0
    ),
    OpInfo(
        name="round", 
        type=OperatorType.UTILITY, 
        operator=round,
        fmt="round({X})",
        precedence=0
    ),
    OpInfo(
        name="reversed", 
        type=OperatorType.UTILITY, 
        operator=reversed,
        fmt="reversed({X})",
        precedence=0
    ),
    OpInfo(
        name="call", 
        type=OperatorType.UTILITY, 
        operator=_call,
        fmt="{X}({arg})",
        precedence=0
    ),
    OpInfo(
        name="getitem", 
        type=OperatorType.UTILITY, 
        operator=_getitem,
        fmt="{X}[{arg}]",
        precedence=0
    ),
    OpInfo(
        name="getattr", 
        type=OperatorType.UTILITY, 
        operator=_getattr,
        fmt="{X}.{arg}",
        precedence=0
    ),
    OpInfo(
        name="len", 
        type=OperatorType.UTILITY, 
        operator=len,
        fmt="len({X})",
        precedence=0
    ),
]

_attr_to_info = {op.attr_name: op for op in ops}


@overload
def get_opinfo(
    attr_name: str,
    type: None = None,
    operator: None = None,
    name: None = None,
) -> OpInfo: ...


@overload
def get_opinfo(
    attr_name: None = None,
    type: None = None,
    operator: None = None,
    name: str = None,
) -> OpInfo: ...


@overload
def get_opinfo(
    attr_name: None = None,
    type: OperatorType = None,
    operator: None = None,
    name: None = None,
) -> list[OpInfo]: ...


@overload
def get_opinfo(
    attr_name: None = None,
    type: None = None,
    operator: Callable[..., Any] = None,
    name: None = None,
) -> list[OpInfo]: ...


@lru_cache(maxsize=100)
def get_opinfo(
    attr_name: Optional[str] = None, 
    type: Optional[OperatorType] = None, 
    operator: Optional[Callable[..., Any]] = None, 
    name: Optional[str] = None
) -> OpInfo | list[OpInfo]:
    """
    Get the OpInfo for a given attribute name, type, operator, or name. Only one of the arguments should be provided.
    """
    if sum(x is not None for x in [attr_name, type, operator, name]) != 1:
        raise ValueError("Exactly one of the arguments should be provided.")

    if attr_name is not None:
        try:
            return _attr_to_info[attr_name]
        except KeyError:
            raise ValueError(f"No OpInfo found for attribute name {attr_name}.")
    elif type is not None:
        out = [op for op in ops if op.type in type]
        if len(out) == 0:
            raise ValueError(f"No OpInfo found for type {type}.")
        return out
    elif operator is not None:
        out = [op for op in ops if op.operator is operator]
        if len(out) == 0:
            raise ValueError(f"No OpInfo found for operator {operator}.")
        return out
    elif name is not None:
        out = [op for op in ops if op.name == name]
        if len(out) == 1:
            return out[0]
        elif len(out) == 0:
            raise ValueError(f"No OpInfo found for name {name}.")
        else:
            raise ValueError(
                f"Multiple OpInfos found for name {name}."
            )
    else:
        raise ValueError("Exactly one of the arguments should be provided.")
