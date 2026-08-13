from __future__ import annotations
import torch
import abc
import dataclasses
import sys
import inspect
import enum
import itertools
import re

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Optional, overload

from torch import nn
from ._opinfo import get_opinfo, OpInfo


modifierType = str


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


class _NoValue:
    """A unique sentinel to represent empty."""
    __slots__ = ()

    @property
    def value(self):
        return self

    def __repr__(self):
        return "<NO_VALUE>"

    def __str__(self):
        return "<NO_VALUE>"


class _MappingKey(str):
    """ 
    This is a sentinel type to be used with Delayable objects to indicate a map packing.
    """
    pass


class _Frame:
    """One suspended `_fae_exchange` coroutine in an iterative traversal."""
    __slots__ = ("node", "exchange", "path", "sent", "changed")
    def __init__(self, node: Delayable, path: str) -> None:
        self.node = node
        self.exchange = node._fae_exchange()
        self.path = path
        self.sent: Any = None       # value to resume the coroutine with
        self.changed = False        # did any item come back different?


class _RecallTable(dict):
    """
    Per-evaluation storage of named-node outputs, read back by the `R` symbol.

    A fresh table is seeded by the outermost `Chain` of every evaluation (and by the
    forward emitted by `lower()`), so recalled values never leak across calls.
    """
    def __missing__(self, key):
        raise KeyError(
            f"R[{key!r}] was resolved before any node named {key!r} produced a value. "
            "The named node must execute before the recall site (i.e. appear earlier "
            "in the same chain)."
        )


# The key under which the recall table travels through resolution kwargs. It must equal
# the class name of the `R` symbol so that symbol resolution finds the table by name.
_RECALL_KEY = "R"


class Delayable:
    """
    Delayable is the base class for all delayable objects. It provides the base functionality for
    conditional evaluation, chaining, and resolving with data.
    # TODO: Delayable should be an abstract base class.
    """
    def __init__(self, *args, **kwargs) -> None:
        sig = inspect.signature(self.__init__)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        self.fae_name = None
        self._fae_arguments = bound

    @abstractmethod
    def _resolve(self, _default: Any, /, **kwargs: Any) -> Any:
        """
        Uses data to resolve the delayable. Must be implemented by subclasses.
        """

    def _fae_exchange(self) -> Delayable:
        """ 
        This is a coroutine that yields the direct children of the delayable. This should be 
        implemented by subclasses, where the subclasses yield immediate children of the delayable,
        expect altered children, and return a new delayable instance.
        """
        yield from ()
        return self

    def fae_children(self, items: bool = False) -> Iterator[Any]:
        """Yield this node's direct children, echoing every item back unchanged."""
        exchange = self._fae_exchange()
        sent = None
        while True:
            try:
                item = exchange.send(sent)
            except StopIteration:
                return
            sent = item
            if items or isinstance(item, Delayable):
                yield item

    def _fae_traverse(self, visit, *, always_copy=False, visit_items=False) -> Any:
        root_path = self.fae_name or "_"
        replaced = visit(self, root_path)
        if replaced is not None:
            return replaced
        stack = [_Frame(self, root_path)]
        
        while True:
            frame = stack[-1]
            try:
                item = frame.exchange.send(frame.sent)
            except StopIteration as stop:
                node = stop.value if frame.changed or always_copy else frame.node
                stack.pop()
                if not stack:
                    return node
                parent = stack[-1]
                parent.sent = node
                parent.changed = parent.changed or node is not frame.node
                continue
            
            is_node = isinstance(item, Delayable)
            path = f"{frame.path}.{item.fae_name or '_'}" if is_node else frame.path
            if is_node or visit_items:
                new_item = visit(item, path)
                if new_item is not None:
                    frame.sent = new_item      # replaced: prune, never open its coroutine
                    frame.changed = True
                    continue
            frame.sent = item
            if is_node:
                stack.append(_Frame(item, path))

    def _fae_apply(self, fn=None, *, always_copy: bool = False) -> Delayable:
        """Exchange this node's own items; children are not traversed."""
        exchange = self._fae_exchange()
        changed, sent = False, None
        while True:
            try:
                item = exchange.send(sent)
            except StopIteration as stop:
                return stop.value if changed or always_copy else self
            new_item = fn(item) if fn is not None else None
            changed = changed or (new_item is not None and new_item is not item)
            sent = item if new_item is None else new_item

    def fae_walk(self, *, breadth_first: bool = False, items: bool = False) -> Iterator[Delayable]:
        """Yield this node and every descendant. Nothing is rebuilt."""
        pending = deque([self])
        while pending:
            node = pending.popleft() if breadth_first else pending.pop()
            yield node
            children = list(node.fae_children(items=items)) if isinstance(node, Delayable) else []
            pending.extend(children if breadth_first else reversed(children))

    def fae_clone(
        self,
        recurse: bool = False,
        clone_modules: bool = True
    ) -> Delayable:
        if not recurse:
            return self._fae_apply(always_copy=True)

        def visit(item: Any, path: str) -> Any:            
            if clone_modules and isinstance(item, nn.Module):
                return item.clone()
            return None

        return self._fae_traverse(visit, always_copy=True, visit_items=True)

    def fae_find(
        self,
        pattern: str | type[Delayable],
        callback: Optional[Callable[[Delayable], Delayable]] = None,
    ) -> Delayable:
        if isinstance(pattern, type):
            def matches(node: Delayable, path: str) -> bool:
                return isinstance(node, pattern)
        else:
            def matches(node: Delayable, path: str) -> bool:
                return re.fullmatch(pattern, path) is not None
        
        def visit(node: Delayable, path: str) -> Optional[Delayable]:
            if callback is None or not matches(node, path):
                return None
            return callback(node)
        
        return self._fae_traverse(visit)

    def _record(self, result: Any, kwargs: dict[str, Any]) -> None:
        """
        Store `result` in the current evaluation's recall table (if one is active) under
        this node's name, so `R["name"]` can read it later in the same evaluation.

        No-op when the node is unnamed, no table is active, or the result is still
        delayed (partial evaluation). Symbols (which are classes and whose `fae.name`
        is the class name) are never recorded.
        """
        if self.fae.name is None or isinstance(self, type) or isinstance(result, Delayable):
            return
        table = kwargs.get(_RECALL_KEY)
        if table is not None:
            table[self.fae.name] = result
    
    def __or__(self, other: Any) -> Any:
        """ 
        The case of `Delayable | Any` is not defined.
        """
        if isinstance(other, torch.Tensor):
            # Prevent __torch_function__ from being called for `X | tensor`.
            # Because torch_function __ror__ uses bitwise_or instead, which we don't want to 
            # handle there since it might be called by the function name, and not the operator 
            # magic.
            raise TypeError("Cannot pipe a tensor to a Delayable.")
        return NotImplemented

    def __ror__(self, other: Any) -> Any:
        """ 
        `data | Delayable` results in evaluating the delayed operations.
        """
        if isinstance(other, Delayable):
            return NotImplemented
        return self._resolve(other)

    def __mod__[T: Delayable](self: T, modifier: modifierType) -> T:
        """
        The modulate operator `%` is used to name the operation. It can also be used to modify 
        Delayables, for example, set Optimizer to parameters in delayable modules, etc...
        (TODO: How to handle general modifiers, e.g. optimizer.)
        """
        if isinstance(modifier, str):
            if "." in modifier:
                raise ValueError("Modifier cannot contain a period.")
            out = self.fae.clone()
            out.fae = dataclasses.replace(out.fae, name=modifier)
            return out
        else:
            return modifier.__rmod__(self) 
        
    def __rmod__[T: Delayable](self: T, other: str) -> T:
        """
        Modifiers should always be to the right of the Delayable.
        """
        if isinstance(other, modifierType):
            raise TypeError(f"Modifier should be to the right of the Delayable, not the left.")
        return NotImplemented
        
    def __rshift__(self, other: Delayable | int | Sequence[Any]) -> Chain:
        """
        The right shift operator (>>) is used to chain Delayables together, like the layers in 
        a neural network.

        There are three possible cases:
        1. `Delayable >> Delayable` -> Chain(Delayable, Delayable)
        2. `Delayable >> int` -> Chain(Delayable, Delayable, ..., Delayable) 
        3. `Delayable >> Sequence[Any]` 
        """
        if isinstance(other, Delayable):
            return Chain(self, other)
        elif isinstance(other, int):
            out = None
            for i in range(other):
                cloned = self.fae.clone(recurse=True, clone_modules=i)
                if out is None: 
                    out = cloned
                else:
                    out = out >> cloned
            return out
        elif isinstance(other, Sequence):
            out = None
            for i in range(len(other)):
                cloned = self.fae.clone(recurse=True, clone_modules=i, data=other)
                if out is None: 
                    out = cloned
                else:
                    out = out >> cloned
            return out

        return NotImplemented

    def __rrshift__(self, other: Any) -> Any:
        """
        The rshift operator (>>) is only supported when both sides are Delayables. 
        In this case `other` cannot be of type `Delayable` (`__rshift__` is called instead).
        """
        return NotImplemented


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
            # TODO: do i need this check?
            return NotImplemented
        return F(opinfo, self, *args, **kwargs)
        
    # --- Binary arithmetic operators ---
    def __add__(self, other: Any) -> Delayable:
        return self._op_action("__add__", other)

    def __radd__(self, other: Any) -> Delayable:
        return self._op_action("__radd__", other)

    def __sub__(self, other: Any) -> Delayable:
        return self._op_action("__sub__", other)

    def __rsub__(self, other: Any) -> Delayable:
        return self._op_action("__rsub__", other)

    def __mul__(self, other: Any) -> Delayable:
        return self._op_action("__mul__", other)

    def __rmul__(self, other: Any) -> Delayable:
        return self._op_action("__rmul__", other)

    def __matmul__(self, other: Any) -> Delayable:
        return self._op_action("__matmul__", other)

    def __rmatmul__(self, other: Any) -> Delayable:
        return self._op_action("__rmatmul__", other)

    def __truediv__(self, other: Any) -> Delayable:
        return self._op_action("__truediv__", other)

    def __rtruediv__(self, other: Any) -> Delayable:
        return self._op_action("__rtruediv__", other)

    def __floordiv__(self, other: Any) -> Delayable:
        return self._op_action("__floordiv__", other)

    def __rfloordiv__(self, other: Any) -> Delayable:
        return self._op_action("__rfloordiv__", other)

    def __mod__(self, other: Any) -> Delayable:
        """
        If `other` qualifies as a Faeyon modifier, use the parent class implementation, otherwise, 
        the modulus % operator is treated as a normal arithmetic operation.
        """
        out =  super().__mod__(other)
        if out is NotImplemented:
            return self._op_action("__mod__", other)
        return out
    
    def __rmod__(self, other: Any) -> Delayable:
        out = super().__rmod__(other)
        if out is NotImplemented:
            return self._op_action("__rmod__", other)
        return out

    def __divmod__(self, other: Any) -> Delayable:
        return self._op_action("__divmod__", other)

    def __rdivmod__(self, other: Any) -> Delayable:
        return self._op_action("__rdivmod__", other)

    def __pow__(self, other: Any) -> Delayable:
        return self._op_action("__pow__", other)

    def __rpow__(self, other: Any) -> Delayable:
        return self._op_action("__rpow__", other)

    def __and__(self, other: Any) -> Delayable:
        return self._op_action("__and__", other)

    def __rand__(self, other: Any) -> Delayable:
        return self._op_action("__rand__", other)

    def __xor__(self, other: Any) -> Delayable:
        return self._op_action("__xor__", other)

    def __rxor__(self, other: Any) -> Delayable:
        return self._op_action("__rxor__", other)

    # --- Unary arithmetic operators ---
    def __neg__(self) -> Delayable:
        return self._op_action("__neg__")

    def __pos__(self) -> Delayable:
        return self._op_action("__pos__")

    def __abs__(self) -> Delayable:
        return self._op_action("__abs__")

    def __invert__(self) -> Delayable:
        return self._op_action("__invert__")

    def __round__(self) -> Delayable:
        return self._op_action("__round__")

    # --- Comparison operators ---
    def __lt__(self, other: Any) -> Delayable:
        return self._op_action("__lt__", other)

    def __le__(self, other: Any) -> Delayable:
        return self._op_action("__le__", other)

    def __eq__(self, other: Any) -> Delayable:
        return self._op_action("__eq__", other)

    def __ne__(self, other: Any) -> Delayable:
        return self._op_action("__ne__", other)

    def __gt__(self, other: Any) -> Delayable:
        return self._op_action("__gt__", other)

    def __ge__(self, other: Any) -> Delayable:
        return self._op_action("__ge__", other)

    #--- Other operators ---
    def __getattr__(self, name: str) -> Delayable:
        if name == "__torch_function__":
            return type(self).__torch_function__
        
        return self._op_action("__getattr__", name)

    def __getitem__(self, key: Any) -> Delayable:
        if isinstance(key, _MappingKey):
            return _Unpack(self, is_map=True)
        return self._op_action("__getitem__", key)

    def __call__(self, *args: Any, **kwargs: Any) -> Delayable:
        return self._op_action("__call__", *args, **kwargs)

    def __reversed__(self) -> X:
        return self._op_action("__reversed__")

    def __iter__(self):
        return iter([_Unpack(self)])

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Note: For operators like `+`, `-`, `*`, etc., the `__torch_function__` is called only if 
        tensor is the left operand, otherwise the operand must be handled by the right hand side 
        Delayable.
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
    """
    TODO: I should disable applying modifiers or names to symbols.... because they apply globally
    on class instances.
    """
    _registry: dict[str, type[Symbol]] = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        if any(isinstance(base, _SymbolMeta) for base in bases):
            mcs._registry[name] = cls
    
        return cls

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, name=cls.__name__, **kwargs)

    def _resolve(self, _default: Any = _NoValue, /, **kwargs: Any) -> Any:
        """ 
        If symbol is in kwargs, it will be replaced with the value of the key.
        If symbol is not in kwargs, the default value will be used, if specified. 
        If no default value is specified, and no value is provided in kwargs, the symbol will be returned as is.
        """
        if self.fae.name in kwargs:
            return kwargs[self.fae.name]
        if _default is not _NoValue:
            return _default
        return self

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
    Dynamically create a symbol class, for example Sym.Y will create a new `Symbol` class called Y, 
    and it will be addeed to the symbol registry.
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


# class _IndexMeta(_SymbolMeta):
#     def __getitem__(self, data: Sequence[Any]) -> int:
#         return F(_getitem, data, I)


class I(Symbol):
    """ A special symbol that represents an index."""
    pass


class P(Symbol):
    """ A special symbol that represents a placeholder for a parameter."""
    pass


class R(Symbol):
    """
    The recall symbol: `R["name"]` resolves to the output that the node named `% "name"`
    produced earlier in the *current* evaluation. This turns names into long-range skip
    connections (U-Net, FPN) without threading values through the pipeline by hand:

        unet = (
            enc_block(1, 64) % "e1" >> down(64)
            >> bottleneck(64, 128)
            >> up(128, 64) >> F(torch.cat, FList([X, R["e1"]]), dim=1) >> dec_block(...)
        )

    Semantics:
    * Recorded outputs live in a per-evaluation table seeded by the outermost `Chain`
      (or by `lower()`'s emitted forward); nothing leaks across calls.
    * The named node must execute before the recall site — recalling a name that has
      not produced a value yet raises a `KeyError` at evaluation time.
    * Names are matched by their plain node name (the string given to `%`), not by
      dotted path; recalled names should therefore be unique within one model.
    """
    @classmethod
    def _resolve(cls, _default: Any = _NoValue, /, **kwargs: Any) -> Any:
        # Unlike other symbols, R never falls back to `_default`: its only meaning is
        # the recall table of the current evaluation. With no active table it stays
        # unresolved (partial evaluation) instead of silently capturing the data.
        return kwargs.get(_RECALL_KEY, cls)
    

class _Unpack(Delayable):
    """
    Represents an unpacking operation (*X). When resolved, unpacks the data as *args.
    """
    def __init__(self, target, is_map: bool = False) -> None:
        super().__init__(target=target, is_map=is_map)
        self._fae_target = target
        self._fae_is_map = is_map
    
    def _resolve(self, _default: Any, /, **kwargs: Any) -> Iterator[Any]:
        return self.fae.target._resolve(_default, **kwargs)
    
    def __repr__(self) -> str:
        if self._fae_is_map:
            prefix = "**"
        else:
            prefix = "*"
        return f"{prefix}{self._fae_target!r}"


class F(_OpActionMixin, Delayable):
    def __init__(self, op: Callable[..., Any], /, *args, **kwargs) -> None:
        super().__init__(op, *args, **kwargs)
        self._fae_op = op
        self._fae_args = args
        self._fae_kwargs = kwargs

    def _resolve(self, _default: Any, /, **kwargs: Any) -> Any:
        resolved_args = []
        for arg in self.fae.args:
            if isinstance(arg, Delayable):
                resolved = arg._resolve(_default, **kwargs)
            else:
                resolved = arg
            
            if isinstance(arg, _Unpack):
                resolved_args.extend(resolved)
            else:
                resolved_args.append(resolved)

        resolved_kwargs = {}
        for k, v in self.fae.kwargs.items():
            if isinstance(v, Delayable):
                resolved = v._resolve(_default, **kwargs)
            else:
                resolved = v
        
            if not isinstance(v, _Unpack):
                resolved = {k: resolved}
            
            # Need to do this because the unpacking operation might have same key multiple times.
            for k in resolved:
                if k in resolved_kwargs:
                    raise TypeError(f"{self.fae.op} got multiple values for argument '{k}'.")
            resolved_kwargs.update(resolved)

        if any(
            isinstance(a, Delayable) 
            for a in itertools.chain(resolved_args, resolved_kwargs.values())
        ):
            return F(self.fae.op, *resolved_args, **resolved_kwargs)

        result = self.fae.op(*resolved_args, **resolved_kwargs)
        self._record(result, kwargs)
        return result

    def _fae_exchange(self) -> F:
        new_args = []
        for arg in self._fae_args:
            new_arg = yield arg
            new_args.append(new_arg)

        new_kwargs = {}
        for key, kwarg in self._fae_kwargs.items():
            new_kwarg = yield kwarg
            new_kwargs[key] = new_kwarg

        return self.__class__(self._fae_op, *new_args, **new_kwargs)
    
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


class Chain(_OpActionMixin, Delayable):
    """
    A Chain is a sequence of operations: `op0 >> op1 << op2 >> ... >> opn`.
    """
    def __init__(self, *ops: Delayable) -> None:
        if not ops:
            raise ValueError("Chain must have at least one operation.")
        
        self._fae_ops = []
        for op in ops:
            if isinstance(op, Delayable):
                self._fae_ops.append(op)
            else:
                raise ValueError("All arguments must be of subtype `Delayable` or `nn.Module`.")
        super().__init__(*ops)

    def _resolve(self, _default: Any = _NoValue, /, **kwargs: Any) -> Any:
        """
        data | chain. 

        - The first item in chain will be resolved like any F, based on the data provided.
        - X is a special symbol that represents the output of the previous item in chain.
          So if X is given as input, it will be replaced with the output of the 
          previous item in chain. If you have arguments needed downstream, use another symbol.
        """       
        # Seed the recall table for `R` at the outermost chain of this evaluation;
        # nested chains find the caller's table in kwargs and share it.
        kwargs.setdefault(_RECALL_KEY, _RecallTable())
        x = self._fae_ops[0]._resolve(_default, **kwargs)
        kwargs.pop("X", None)
        for op in self._fae_ops[1:]:
            x = op._resolve(_default, X=x, **kwargs)
        self._record(x, kwargs)
        return x

    def _fae_exchange(self) -> Chain:
        new_ops = []
        for op in self._fae_ops:
            new_op = yield op
            new_ops.append(new_op)
        return self.__class__(*new_ops)

    def __lshift__(self, other: Delayable) -> Chain:
        return Chain(*self._fae_ops[:-1], self._fae_ops[-1] << other)

    def __rshift__(self, other: Any) -> Any:
        if self.fae_name is not None:
            return super().__rshift__(other)

        if isinstance(other, Chain):
            return Chain(*self._fae_ops, *other._fae_ops)
        elif isinstance(other, Delayable):
            return Chain(*self._fae_ops, other)
        else:
            return super().__rshift__(other)

    def __len__(self) -> int:
        return len(self._fae_ops)

    def __repr__(self) -> str:
        out = []
        for item in self._fae_ops:
            out.append(repr(item))  
        return " >> ".join(out)


class DelayedModule(F):
    """
    This is for modules whose constructor arguments are delayables.
    """
    # def __init__(self, module: type[nn.Module], /, *args, **kwargs) -> None:
    #     super().__init__(module, *args, **kwargs)
        
    def _resolve(self, _default: Any, /, **kwargs: Any) -> Any:
        raise ValueError("`DelayedModule` cannot be resolved directly.")

    def __rshift__(self, other: Any) -> Any:
        if isinstance(other, Delayable):
            return self(X) >> other
        else:
            return super().__rshift__(other)

    def __rrshift__(self, other: Any) -> Any:
        if isinstance(other, Delayable):
            return other >> self(X)
        else:
            return super().__rrshift__(other)

    def _generate(self, I: int, P: Optional[Sequence[Any]] = None) -> F:
        """
        If  
        """
        if P is not None:
            return super()._resolve(_NoValue, P=P, I=I)

        return super()._resolve(_NoValue, I=I)
        # from .faek import _resolved_call

        # resolve_kwargs = {"I": I}
        # if P is not None:
        #     resolve_kwargs["P"] = P

        # module_cls = self.fae.arguments.arguments["module"]
        # raw_args = self.fae.arguments.arguments.get("args", ())
        # raw_kwargs = self.fae.arguments.arguments.get("kwargs", {})

        # resolved_args = [
        #     v._resolve(_NoValue, **resolve_kwargs) if isinstance(v, Delayable) else v
        #     for v in raw_args
        # ]
        # resolved_kwargs = {
        #     k: v._resolve(_NoValue, **resolve_kwargs) if isinstance(v, Delayable) else v
        #     for k, v in raw_kwargs.items()
        # }

        # module = module_cls(*resolved_args, **resolved_kwargs)
        # return F(_resolved_call, module, X)


class Input:
    """
    A placeholder for providing arguments to resolve delayables. Examples:

        Substitute(A=Input(data, bias=bar)) | A[0] >> 2 * X + A["bias"]
    
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
        return f"Input({', '.join(arguments)})"


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
        super().__init__(expressions=expressions)

    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> FList:
        opinfo = get_opinfo(attr_name=name)

        raveled = []
        n = 0
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, FList):
                n += 1
                raveled.append(arg.fae.expressions)
            elif isinstance(arg, FDict):
                raise ValueError("Cannot mix `FList` and `FDict` arguments. Choose one.")
            else:
                raveled.append(itertools.repeat(arg))

        if n == 0:
            return FList([F(opinfo, item, *args, **kwargs) for item in self.fae.expressions])

        raveled = zip(*raveled)
        out = []
        for item, arg in zip(self.fae.expressions, raveled):
            items_args = arg[:len(args)]
            items_kwargs = dict(zip(kwargs.keys(), arg[len(args):]))
            out.append(F(opinfo, item, *items_args, **items_kwargs))
        return FList(out)

    def _resolve(self, _default: Any, /, **kwargs: Any) -> Any:
        result = [item._resolve(_default, **kwargs) for item in self.fae.expressions]
        self._record(result, kwargs)
        return result

    def __lshift__(self, other: Delayable) -> FList:       
        if isinstance(other, FList):
            out = []
            if len(other) == len(self):
                out = [left >> right for left, right in zip(self.fae.expressions, other.fae.expressions)]
            elif len(other) == 1:
                right = other.fae.expressions[0]
                out = [left >> right for left in self.fae.expressions]
            elif len(self) == 1:
                left = self.fae.expressions[0]
                out = [left >> right for right in other.fae.expressions]  
            else:
                return NotImplemented

            return FList(out)
        elif isinstance(other, (Symbol, F)):
            return FList([expr >> other for expr in self.fae.expressions])
        else:
            return NotImplemented
        
    def __str__(self) -> str:
        return str(self.fae.expressions)

    def __len__(self) -> int:
        return len(self.fae.expressions)

    def __repr__(self) -> str:
        return str(self)
    

class FDict(_OpActionMixin, Delayable):
    def __init__(self, expressions: dict[str, Delayable]) -> None:
        super().__init__(expressions=expressions)

    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> FDict:
        opinfo = get_opinfo(attr_name=name)

        raveled = defaultdict(list)
        n = 0
        keys = set(self.fae.expressions)
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, FDict):
                n += 1
                if keys != set(arg.fae.expressions):
                    raise ValueError("All arguments of type `FDict` must have the same keys.")

                for key, item in arg.fae.expressions.items():
                    raveled[key].append(item)
            elif isinstance(arg, FList):
                raise ValueError("Cannot mix `FList` and `FDict` arguments. Choose one.")
            else:
                for key in keys:
                    raveled[key].append(arg)

        if n == 0:
            return FDict(
                {key: F(opinfo, item, *args, **kwargs) 
                for key, item in self.fae.expressions.items()}
            )

        out = {}
        nargs = len(args)
        for key, value in self.fae.expressions.items():
            items_args = raveled[key][:nargs]
            items_kwargs = dict(zip(kwargs, raveled[key][nargs:]))
            out[key] = F(opinfo, value, *items_args, **items_kwargs)
        return FDict(out)

    def _resolve(self, _default: Any, /, **kwargs: Any) -> Any:
        result = {
            key: item._resolve(_default, **kwargs) 
            for key, item in self.fae.expressions.items()
        }
        self._record(result, kwargs)
        return result

    def __lshift__(self, other: Delayable) -> FDict:
        if isinstance(other, FDict):
            out = {}
            other = other.fae.expressions

            if set(self.fae.expressions) != set(other):
                return NotImplemented

            for key, item in self.fae.expressions.items():
                out[key] = item >> other[key]
            return FDict(out)
        elif isinstance(other, (Symbol, F)):
            return FDict({key: item >> other for key, item in self.fae.expressions.items()})
        else:
            return NotImplemented

    def __str__(self) -> str:
        return str(self.fae.expressions)

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.fae.expressions)
