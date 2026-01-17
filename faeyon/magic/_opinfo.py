import dataclasses
import enum
import operator

from collections.abc import Callable
from typing import Any, Optional, overload


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


@dataclasses.dataclass
class OpInfo:
    name: str
    type: OperatorType
    fmt: str
    operator: Callable[[..., Any], Any]

    @property
    def is_right(self) -> bool:
        return self.type == OperatorType.RBINARY

    @property
    def attr_name(self) -> str:
        return f"__{self.name}__"

    def to_string(self, X: str, *args, **kwargs: Any) -> str:
        return self.fmt.format(*args, X=X, **kwargs)

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
    OpInfo(name="rshift", type=OperatorType.BINARY, operator=operator.rshift, fmt="{X} >> {}"),
    OpInfo(name="rrshift", type=OperatorType.RBINARY, operator=operator.rshift, fmt="{} >> {X}"),
    OpInfo(name="lshift", type=OperatorType.BINARY, operator=operator.lshift, fmt="{X} << {}"),
    OpInfo(name="rlshift", type=OperatorType.RBINARY, operator=operator.lshift, fmt="{} << {X}"),
    OpInfo(name="add", type=OperatorType.BINARY, operator=operator.add, fmt="{X} + {}"),
    OpInfo(name="radd", type=OperatorType.RBINARY, operator=operator.add, fmt="{} + {X}"),
    OpInfo(name="sub", type=OperatorType.BINARY, operator=operator.sub, fmt="{X} - {}"),
    OpInfo(name="rsub", type=OperatorType.RBINARY, operator=operator.sub, fmt="{} - {X}"),
    OpInfo(name="mul", type=OperatorType.BINARY, operator=operator.mul, fmt="{X} * {}"),
    OpInfo(name="rmul", type=OperatorType.RBINARY, operator=operator.mul, fmt="{} * {X}"),
    OpInfo(name="truediv", type=OperatorType.BINARY, operator=operator.truediv, fmt="{X} / {}"),
    OpInfo(name="rtruediv", type=OperatorType.RBINARY, operator=operator.truediv, fmt="{} / {X}"),
    OpInfo(name="floordiv", type=OperatorType.BINARY, operator=operator.floordiv, fmt="{X} // {}"),
    OpInfo(name="rfloordiv", type=OperatorType.RBINARY, operator=operator.floordiv, fmt="{} // {X}"),
    OpInfo(name="mod", type=OperatorType.BINARY, operator=operator.mod, fmt="{X} % {}"),
    OpInfo(name="rmod", type=OperatorType.RBINARY, operator=operator.mod, fmt="{} % {X}"),
    OpInfo(name="pow", type=OperatorType.BINARY, operator=operator.pow, fmt="{X} ** {}"),
    OpInfo(name="rpow", type=OperatorType.RBINARY, operator=operator.pow, fmt="{} ** {X}"),
    OpInfo(name="matmul", type=OperatorType.BINARY, operator=operator.matmul, fmt="{X} @ {}"),
    OpInfo(name="rmatmul", type=OperatorType.RBINARY, operator=operator.matmul, fmt="{} @ {X}"),
    OpInfo(name="floordiv", type=OperatorType.BINARY, operator=operator.floordiv, fmt="{X} // {}"),
    OpInfo(name="rfloordiv", type=OperatorType.RBINARY, operator=operator.floordiv, fmt="{} // {X}"),
    OpInfo(name="mod", type=OperatorType.BINARY, operator=operator.mod, fmt="{X} % {}"),
    OpInfo(name="rmod", type=OperatorType.RBINARY, operator=operator.mod, fmt="{} % {X}"),
    OpInfo(name="and", type=OperatorType.BINARY, operator=operator.and_, fmt="{X} & {}"),
    OpInfo(name="rand", type=OperatorType.RBINARY, operator=operator.and_, fmt="{} & {X}"),
    OpInfo(name="or", type=OperatorType.BINARY, operator=operator.or_, fmt="{X} | {}"),
    OpInfo(name="ror", type=OperatorType.RBINARY, operator=operator.or_, fmt="{} | {X}"),
    OpInfo(name="xor", type=OperatorType.BINARY, operator=operator.xor, fmt="{X} ^ {}"),
    OpInfo(name="rxor", type=OperatorType.RBINARY, operator=operator.xor, fmt="{} ^ {X}"),
    OpInfo(name="abs", type=OperatorType.UNARY, operator=operator.abs, fmt="abs({X})"),
    OpInfo(name="invert", type=OperatorType.UNARY, operator=operator.invert, fmt="~{X}"),
    OpInfo(name="neg", type=OperatorType.UNARY, operator=operator.neg, fmt="-{X}"),
    OpInfo(name="pos", type=OperatorType.UNARY, operator=operator.pos, fmt="+{X}"),
    OpInfo(name="round", type=OperatorType.UNARY, operator=round, fmt="round({X})"),
    OpInfo(name="reversed", type=OperatorType.UNARY, operator=reversed, fmt="reversed({X})"),
    OpInfo(name="lt", type=OperatorType.COMPARISON, operator=operator.lt, fmt="{X} < {}"),
    OpInfo(name="le", type=OperatorType.COMPARISON, operator=operator.le, fmt="{X} <= {}"),
    OpInfo(name="eq", type=OperatorType.COMPARISON, operator=operator.eq, fmt="{X} == {}"),
    OpInfo(name="ne", type=OperatorType.COMPARISON, operator=operator.ne, fmt="{X} != {}"),
    OpInfo(name="gt", type=OperatorType.COMPARISON, operator=operator.gt, fmt="{X} > {}"),
    OpInfo(name="ge", type=OperatorType.COMPARISON, operator=operator.ge, fmt="{X} >= {}"),
    OpInfo(name="call", type=OperatorType.UTILITY, operator=_call, fmt="{X}({})"),
    OpInfo(name="getitem", type=OperatorType.UTILITY, operator=_getitem, fmt="{X}[{}]"),
    OpInfo(name="getattr", type=OperatorType.UTILITY, operator=_getattr, fmt="{X}.{}"),
    OpInfo(name="len", type=OperatorType.UTILITY, operator=len, fmt="len({X})"),
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
