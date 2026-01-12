import dataclasses
import enum
import operator

from collections.abc import Callable
from typing import Any


class OperatorType(enum.Enum):
    BINARY = enum.auto()
    RBINARY = enum.auto()
    COMPARISON = enum.auto()
    UNARY = enum.auto()
    UTILITY = enum.auto()
    

@dataclasses.dataclass
class OpInfo:
    name: str
    type: OperatorType
    fmt: str
    operator: Callable[[Any, Any], Any]  = None
    
    @property
    def is_right(self) -> bool:
        return self.type == OperatorType.RBINARY

    @property
    def attr_name(self) -> str:
        return f"__{self.name}__"

    def to_string(self, *args, X: str, **kwargs: Any) -> str:
        return self.fmt.format(*args, X=X, **kwargs)


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
    OpInfo(name="call", type=OperatorType.UTILITY, fmt="{X}({})"),
    OpInfo(name="getitem", type=OperatorType.UTILITY, fmt="{X}[{}]"),
    OpInfo(name="getattr", type=OperatorType.UTILITY, fmt="{X}.{}"),
]

attr_to_info = {op.attr_name: op for op in ops}


def get_opinfo(attr_name: str) -> OpInfo:
    return attr_to_info[attr_name]


# binary_operators = {
#     operator.rshift: ("__rshift__", "__rrshift__"),
#     operator.lshift: ("__lshift__", "__rlshift__"),
#     operator.add: ("__add__", "__radd__"),
#     operator.sub: ("__sub__", "__rsub__"),
#     operator.mul: ("__mul__", "__rmul__"),
#     operator.truediv: ("__truediv__", "__rtruediv__"),
#     operator.floordiv: ("__floordiv__", "__rfloordiv__"),
#     operator.mod: ("__mod__", "__rmod__"),
#     operator.pow: ("__pow__", "__rpow__"),
#     operator.matmul: ("__matmul__", "__rmatmul__"),
#     operator.and_: ("__and__", "__rand__"),
#     operator.or_: ("__or__", "__ror__"),
#     operator.xor: ("__xor__", "__rxor__"),
# }

# unary_operators: dict[Callable[[Any], Any], str] = {
#     operator.abs: "__abs__",
#     operator.invert: "__invert__",
#     operator.neg: "__neg__",
#     operator.pos: "__pos__",
# }


# # Define methods supported by the _MetaX metaclass
# _meta_methods = {
#     "__lt__": "{X} < {}",
#     "__le__": "{X} <= {}",
#     "__eq__": "{X} == {}",
#     "__ne__": "{X} != {}",
#     "__gt__": "{X} > {}",
#     "__ge__": "{X} >= {}",
#     "__getattr__": "{X}.{}",

#     # Emulating callables
#     "__call__": "{X}({})",
    
#     # Containers
#     "__getitem__": "{X}[{}]",
#     "__reversed__": "reversed({X})",

#     # Numeric types
#     "__add__": "{X} + {}",
#     "__sub__": "{X} - {}",
#     "__mul__": "{X} * {}",
#     "__matmul__": "{X} @ {}",
#     "__truediv__": "{X} / {}",
#     "__floordiv__": "{X} // {}",
#     "__mod__": "{X} % {}",
#     "__divmod__": "divmod({X}, {})",
#     "__pow__": "{X} ** {}",
#     "__lshift__": "{X} << {}",
#     "__rshift__": "{X} >> {}",
#     "__and__": "{X} & {}",
#     "__or__": "{X} | {}",
#     "__xor__": "{X} ^ {}",

#     "__radd__": "{} + {X}", 
#     "__rpow__": "{} ** {X}",
#     "__rlshift__": "{} << {X}",
#     "__rrshift__": "{} >> {X}",
#     "__rand__": "{} & {X}",
#     "__ror__": "{} | {X}",
#     "__rxor__": "{} ^ {X}",

#     # Unary operators
#     "__neg__": "-{X}",
#     "__pos__": "+{X}",
#     "__abs__": "abs({X})",
#     "__invert__": "~{X}",

#    # built-in number types
#     "__round__": "round({X})",
#     "__trunc__": "trunc({X})",
#     "__floor__": "floor({X})",
#     "__ceil__": "ceil({X})",
# }
