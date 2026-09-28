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
from .base import _NoValue, _MappingKey


modifierType = str


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
    def _resolve(self, /, **kwargs: Any) -> Any:
        """
        Resolve this delayable using keyword-bound symbols (at least ``X`` for
        the pipeline value). Must be implemented by subclasses.
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
            literal = re.escape(pattern) == pattern

            def matches(node: Delayable, path: str) -> bool:
                if re.fullmatch(pattern, path) is not None:
                    return True
                name = node.fae_name
                if name is None:
                    return False
                if re.fullmatch(pattern, name) is not None:
                    return True
                # Bare name "layer" also matches auto-suffixed "layer.0", "layer.1", …
                if literal and name.startswith(pattern + "."):
                    return True
                return False
        
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
        if self.fae_name is None or isinstance(self, type) or isinstance(result, Delayable):
            return

        table = kwargs.get(_RECALL_KEY)
        if table is not None:
            table[self.fae_name] = result
    
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
        return self._resolve(X=other)

    def __mod__[T: Delayable](self: T, modifier: modifierType) -> T:
        """
        The modulate operator `%` is used to name the operation. It can also be used to modify 
        Delayables, for example, set Optimizer to parameters in delayable modules, etc...
        (TODO: How to handle general modifiers, e.g. optimizer.)
        """
        if isinstance(modifier, str):
            if "." in modifier:
                raise ValueError("Name cannot contain a period.")

            self.fae_name = modifier
            return self
        else:
            return modifier.__rmod__(self) 
        
    def __rmod__[T: Delayable](self: T, other: str) -> T:
        """
        Modifiers should always be to the right of the Delayable.
        """
        if isinstance(other, modifierType):
            raise TypeError(f"Modifier should be to the right of the Delayable, not the left.")

        return NotImplemented
        
    def fae_bind(self, **symbols: Any) -> Any:
        """
        Substitute bound symbols structurally without requiring a pipeline ``X``.
        Used during ``>> Sequence`` / clone expansion. When ``I`` is provided,
        ``DelayedModule`` recipes in the subtree are materialized first.
        """
        node: Delayable = self
        if "I" in symbols:
            i_val = symbols["I"]
            p_val = symbols.get("P")

            def visit(item: Any, path: str) -> Any:
                if isinstance(item, DelayedModule):
                    return item._generate(I=i_val, P=p_val)
                return None

            node = node._fae_traverse(visit)

        return node._resolve(**symbols)

    def __rshift__(self, other: Delayable | int | Sequence[Any]) -> Chain:
        """
        The right shift operator (>>) is used to chain Delayables together, like the layers in 
        a neural network.

        There are three possible cases:
        1. `Delayable >> Delayable` -> Chain(Delayable, Delayable)
        2. `Delayable >> int` -> Chain of N clones (named clones get `.0`, `.1`, … suffixes)
        3. `Delayable >> Sequence` -> one clone per row, binding ``I`` (and ``P`` to the row)
        """
        if isinstance(other, Delayable):
            return Chain(self, other)
        elif isinstance(other, int):
            return self._fae_expand(range(other), rows=None)
        elif isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return self._fae_expand(range(len(other)), rows=other)

        return NotImplemented

    def _fae_expand(
        self,
        indices: Sequence[int],
        rows: Optional[Sequence[Any]],
    ) -> Delayable:
        """Clone this node once per index; bind I/P and suffix names when present."""
        base_name = self.fae_name
        out: Optional[Delayable] = None
        for i in indices:
            row = rows[i] if rows is not None else None
            cloned = self.fae_clone(recurse=True, clone_modules=True)
            if row is not None:
                cloned = cloned.fae_bind(I=row, P=row)
            else:
                cloned = cloned.fae_bind(I=i)

            if base_name is not None and isinstance(cloned, Delayable):
                cloned.fae_name = f"{base_name}.{i}"

            if not isinstance(cloned, Delayable):
                raise TypeError(f"Expansion produced non-Delayable: {type(cloned)}")
            out = cloned if out is None else (out >> cloned)
        if out is None:
            raise ValueError("Expansion produced an empty chain.")
        return out

    def __rrshift__(self, other: Any) -> Any:
        """
        The rshift operator (>>) is only supported when both sides are Delayables. 
        In this case `other` cannot be of type `Delayable` (`__rshift__` is called instead).
        """
        return NotImplemented


class _OpActionMixin[T: Delayable]:
    """
    Base class for delayables which support (arithmetic) operations.

    ``T`` is the concrete type produced by ``_op_action`` (e.g. ``F``, ``FaeList``,
    ``FaeDict``). Subclasses parameterize the mixin accordingly.
    """
    def keys(self) -> Iterator[_MappingKey]:
        return [_MappingKey(self),]
    
    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> T:
        """
        Specify what actions to takes for a given op attribute name and its corresponding arguments.
        """
        opinfo = get_opinfo(attr_name=name)
        if any(
            isinstance(arg, (FaeList, FaeDict))
            for arg in itertools.chain(args, kwargs.values())
        ):
            # TODO: do i need this check?
            return NotImplemented

        def _wrap(arg: Any) -> Any:
            # Embed modules as delayed calls so ``X + module`` resolves like ``module + X``.
            if isinstance(arg, nn.Module):
                from .faek import faek

                return F(faek.module__call__, arg, X)
            return arg

        args = tuple(_wrap(a) for a in args)
        kwargs = {k: _wrap(v) for k, v in kwargs.items()}
        return F(opinfo, self, *args, **kwargs)
        
    # --- Binary arithmetic operators ---
    def __add__(self, other: Any) -> T:
        return self._op_action("__add__", other)

    def __radd__(self, other: Any) -> T:
        return self._op_action("__radd__", other)

    def __sub__(self, other: Any) -> T:
        return self._op_action("__sub__", other)

    def __rsub__(self, other: Any) -> T:
        return self._op_action("__rsub__", other)

    def __mul__(self, other: Any) -> T:
        return self._op_action("__mul__", other)

    def __rmul__(self, other: Any) -> T:
        return self._op_action("__rmul__", other)

    def __matmul__(self, other: Any) -> T:
        return self._op_action("__matmul__", other)

    def __rmatmul__(self, other: Any) -> T:
        return self._op_action("__rmatmul__", other)

    def __truediv__(self, other: Any) -> T:
        return self._op_action("__truediv__", other)

    def __rtruediv__(self, other: Any) -> T:
        return self._op_action("__rtruediv__", other)

    def __floordiv__(self, other: Any) -> T:
        return self._op_action("__floordiv__", other)

    def __rfloordiv__(self, other: Any) -> T:
        return self._op_action("__rfloordiv__", other)

    def __mod__(self, other: Any) -> T:
        """
        If `other` qualifies as a Faeyon modifier, use the parent class implementation, otherwise, 
        the modulus % operator is treated as a normal arithmetic operation.
        """
        out =  super().__mod__(other)
        if out is NotImplemented:
            return self._op_action("__mod__", other)
        return out
    
    def __rmod__(self, other: Any) -> T:
        out = super().__rmod__(other)
        if out is NotImplemented:
            return self._op_action("__rmod__", other)
        return out

    def __divmod__(self, other: Any) -> T:
        return self._op_action("__divmod__", other)

    def __rdivmod__(self, other: Any) -> T:
        return self._op_action("__rdivmod__", other)

    def __pow__(self, other: Any) -> T:
        return self._op_action("__pow__", other)

    def __rpow__(self, other: Any) -> T:
        return self._op_action("__rpow__", other)

    def __and__(self, other: Any) -> T:
        return self._op_action("__and__", other)

    def __rand__(self, other: Any) -> T:
        return self._op_action("__rand__", other)

    def __xor__(self, other: Any) -> T:
        return self._op_action("__xor__", other)

    def __rxor__(self, other: Any) -> T:
        return self._op_action("__rxor__", other)

    # --- Unary arithmetic operators ---
    def __neg__(self) -> T:
        return self._op_action("__neg__")

    def __pos__(self) -> T:
        return self._op_action("__pos__")

    def __abs__(self) -> T:
        return self._op_action("__abs__")

    def __invert__(self) -> T:
        return self._op_action("__invert__")

    def __round__(self) -> T:
        return self._op_action("__round__")

    # --- Comparison operators ---
    def __lt__(self, other: Any) -> T:
        return self._op_action("__lt__", other)

    def __le__(self, other: Any) -> T:
        return self._op_action("__le__", other)

    def __eq__(self, other: Any) -> T:
        return self._op_action("__eq__", other)

    def __ne__(self, other: Any) -> T:
        return self._op_action("__ne__", other)

    def __gt__(self, other: Any) -> T:
        return self._op_action("__gt__", other)

    def __ge__(self, other: Any) -> T:
        return self._op_action("__ge__", other)

    #--- Other operators ---
    def __getattr__(self, name: str) -> T:
        if name == "__torch_function__":
            return type(self).__torch_function__
        
        return self._op_action("__getattr__", name)

    def __getitem__(self, key: Any) -> T:
        if isinstance(key, _MappingKey):
            return _Unpack(self, is_map=True)
        return self._op_action("__getitem__", key)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        return self._op_action("__call__", *args, **kwargs)

    def __reversed__(self) -> T:
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


class _SymbolMeta(_OpActionMixin["F"], Delayable, abc.ABCMeta):
    """
    TODO: I should disable applying modifiers or names to symbols.... because they apply globally
    on class instances.
    """
    _registry: dict[str, type[Symbol]] = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs) -> type[Symbol]:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        if any(isinstance(base, _SymbolMeta) for base in bases):
            mcs._registry[name] = cls
    
        return cls

    def __init__(cls, name, bases, namespace, **kwargs) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        # Delayable.__init__ (via metaclass MRO) leaves fae_name=None; symbols
        # must resolve by class name so `X=...` / `A=...` kwargs match.
        cls.fae_name = name

    def _resolve(self, /, **kwargs: Any) -> Any:
        """
        If symbol is in kwargs, replace with that value; otherwise leave unresolved.
        """
        if self.fae_name in kwargs:
            return kwargs[self.fae_name]
        return self

    def __instancecheck__(cls, instance):
        return (
            super().__instancecheck__(instance) 
            or (isinstance(instance, type) and issubclass(instance, cls))
        )

    def __hash__(cls) -> int:
        """
        Need to define hash since __eq__ is overridden which sets hash to None, and this breaks 
        __instancecheck__.
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
            enc_block(1, 64) % "e1"
            >> down(64)
            >> bottleneck(64, 128)
            >> up(128, 64) 
            >> torch.cat(FaeList([X, R["e1"]]), dim=1)
            >> dec_block(...)
        )

    Semantics:
    * Recorded outputs live in a per-evaluation table seeded by the outermost `Chain`
    (or by `materialize`'s emitted / bound forward); nothing leaks across calls.
    * The named node must execute before the recall site — recalling a name that has
    not produced a value yet raises a `KeyError` at evaluation time.
    * Names are matched by their plain node name (the string given to `%`), not by
    dotted path; recalled names should therefore be unique within one model.
    """
    @classmethod
    def _resolve(cls, /, **kwargs: Any) -> Any:
        # R only means the recall table for the current evaluation.
        return kwargs.get(_RECALL_KEY, cls)
    

class _Unpack(Delayable):
    """
    Represents an unpacking operation (*X). When resolved, unpacks the data as *args.
    """
    def __init__(self, target: Delayable, is_map: bool = False) -> None:
        super().__init__(target=target, is_map=is_map)
        self._fae_target = target
        self._fae_is_map = is_map
    
    def _resolve(self, /, **kwargs: Any) -> Iterator[Any]:
        return self._fae_target._resolve(**kwargs)
    
    def __repr__(self) -> str:
        if self._fae_is_map:
            prefix = "**"
        else:
            prefix = "*"
        return f"{prefix}{self._fae_target!r}"


class F(_OpActionMixin["F"], Delayable):
    def __init__(self, op: Callable[..., Any], /, *args, **kwargs) -> None:
        super().__init__(op, *args, **kwargs)
        self._fae_op = op
        self._fae_args = args
        self._fae_kwargs = kwargs

    def _resolve(self, /, **kwargs: Any) -> Any:
        resolved_args = []
        for arg in self._fae_args:
            if isinstance(arg, Delayable):
                resolved = arg._resolve(**kwargs)
            else:
                resolved = arg
            
            if isinstance(arg, _Unpack):
                resolved_args.extend(resolved)
            else:
                resolved_args.append(resolved)

        resolved_kwargs = {}
        for k, v in self._fae_kwargs.items():
            if isinstance(v, Delayable):
                resolved = v._resolve(**kwargs)
            else:
                resolved = v
        
            if not isinstance(v, _Unpack):
                resolved = {k: resolved}
            
            # Need to do this because the unpacking operation might have same key multiple times.
            for k in resolved:
                if k in resolved_kwargs:
                    raise TypeError(f"{self._fae_op} got multiple values for argument '{k}'.")
            resolved_kwargs.update(resolved)

        if any(
            isinstance(a, Delayable) 
            for a in itertools.chain(resolved_args, resolved_kwargs.values())
        ):
            return F(self._fae_op, *resolved_args, **resolved_kwargs)

        result = self._fae_op(*resolved_args, **resolved_kwargs)
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
    
    def __str__(self) -> str:
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


class DelayedModule(F):
    """
    Module constructor whose arguments still contain delayables (I/P holes).
    Materialized via `_generate` during `>> Sequence` / clone expansion.
    """

    def _resolve(self, /, **kwargs: Any) -> Any:
        raise ValueError("`DelayedModule` cannot be resolved directly.")

    def __rshift__(self, other: Any) -> Any:
        # Keep recipes as Chain nodes (do not wrap in OpInfo ``call``); materialize later.
        from .faek import faek

        if isinstance(other, nn.Module):
            other = F(faek.module__call__, other, X)
        if isinstance(other, Delayable):
            return Chain(self, other)
        return super().__rshift__(other)

    def __rrshift__(self, other: Any) -> Any:
        from .faek import faek

        if isinstance(other, nn.Module):
            other = F(faek.module__call__, other, X)
        if isinstance(other, Delayable):
            return Chain(other, self)
        return super().__rrshift__(other)

    def _generate(self, I: Any, P: Optional[Sequence[Any]] = None) -> F:
        """Resolve ctor holes with clone index/row, then wrap as `module(X)`."""
        from .faek import faek

        resolve_kwargs: dict[str, Any] = {"I": I}
        if P is not None:
            resolve_kwargs["P"] = P

        module_cls = self._fae_op
        resolved_args = [
            v._resolve(**resolve_kwargs) if isinstance(v, Delayable) else v
            for v in self._fae_args
        ]
        resolved_kwargs = {
            k: v._resolve(**resolve_kwargs) if isinstance(v, Delayable) else v
            for k, v in self._fae_kwargs.items()
        }
        module = module_cls(*resolved_args, **resolved_kwargs)
        return F(faek.module__call__, module, X)


class Chain(_OpActionMixin[F], Delayable):
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

    def _resolve(self, /, **kwargs: Any) -> Any:
        """
        data | chain. 

        - The first item is resolved with the caller's ``X`` (and other symbols).
        - Downstream items see ``X`` as the previous item's result. Use another
          symbol (e.g. ``A``) for arguments that must not change along the chain.
        """       
        # Seed the recall table for `R` at the outermost chain of this evaluation;
        # nested chains find the caller's table in kwargs and share it.
        kwargs = dict(kwargs)
        kwargs.setdefault(_RECALL_KEY, _RecallTable())
        x = self._fae_ops[0]._resolve(**kwargs)
        for op in self._fae_ops[1:]:
            kwargs["X"] = x
            x = op._resolve(**kwargs)
        self._record(x, kwargs)
        return x

    def fae_bind(self, **symbols: Any) -> Chain:
        # Materialize DelayedModules on the whole chain first, then bind each op.
        node: Delayable = self
        if "I" in symbols:
            i_val = symbols["I"]
            p_val = symbols.get("P")

            def visit(item: Any, path: str) -> Any:
                if isinstance(item, DelayedModule):
                    return item._generate(I=i_val, P=p_val)
                return None

            node = node._fae_traverse(visit)

        if not isinstance(node, Chain):
            bound = node.fae_bind(**symbols) if isinstance(node, Delayable) else node
            if isinstance(bound, Chain):
                bound.fae_name = self.fae_name
                return bound
            out = Chain(bound) if isinstance(bound, Delayable) else node
            if isinstance(out, Chain):
                out.fae_name = self.fae_name
            return out  # type: ignore[return-value]

        bound = Chain(*(
            op.fae_bind(**symbols) if isinstance(op, Delayable) else op
            for op in node._fae_ops
        ))
        bound.fae_name = self.fae_name
        return bound

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
            # Named chains on the RHS stay opaque so their name is preserved.
            if other.fae_name is not None:
                return Chain(*self._fae_ops, other)
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


class Input:
    """
    A placeholder for providing arguments to resolve delayables. Examples:

        Substitute(A=Input(data, bias=bar)) | A[0] >> 2 * X + A["bias"]

    This makes expressions act like functions, where the expression can resolve position arguments
    by their index, e.g. A[0] will use the first argument in the provided `A` input to the
    expression. Similar, A["bias"] will use the value of the `bias` key.

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
        return other._resolve(**self._kwargs)


class FaeList(_OpActionMixin["FaeList"], Delayable):
    """
    TODO: Make FList generic e.g. Flist[Delayable, etc..]
    """
    def __init__(self, expressions: list[Delayable]) -> None:
        super().__init__(expressions=expressions)
        self._fae_expressions = list(expressions)

    def _fae_exchange(self) -> FaeList:
        new_exprs = []
        for expr in self._fae_expressions:
            new_exprs.append((yield expr))
        return self.__class__(new_exprs)

    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> FaeList:
        opinfo = get_opinfo(attr_name=name)

        raveled = []
        n = 0
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, FaeList):
                n += 1
                raveled.append(arg._fae_expressions)
            elif isinstance(arg, FaeDict):
                raise ValueError("Cannot mix `FaeList` and `FaeDict` arguments. Choose one.")
            else:
                raveled.append(itertools.repeat(arg))

        if n == 0:
            return FaeList([F(opinfo, item, *args, **kwargs) for item in self._fae_expressions])

        raveled = zip(*raveled)
        out = []
        for item, arg in zip(self._fae_expressions, raveled):
            items_args = arg[:len(args)]
            items_kwargs = dict(zip(kwargs.keys(), arg[len(args):]))
            out.append(F(opinfo, item, *items_args, **items_kwargs))
        return FaeList(out)

    def _resolve(self, /, **kwargs: Any) -> Any:
        result = [item._resolve(**kwargs) for item in self._fae_expressions]
        self._record(result, kwargs)
        return result

    def __lshift__(self, other: Delayable) -> FaeList:
        if isinstance(other, FaeList):
            if len(other) == len(self):
                out = [
                    left >> right
                    for left, right in zip(self._fae_expressions, other._fae_expressions)
                ]
            elif len(other) == 1:
                right = other._fae_expressions[0]
                out = [left >> right for left in self._fae_expressions]
            elif len(self) == 1:
                left = self._fae_expressions[0]
                out = [left >> right for right in other._fae_expressions]
            else:
                return NotImplemented

            return FaeList(out)
        elif isinstance(other, (Symbol, F)):
            return FaeList([expr >> other for expr in self._fae_expressions])
        else:
            return NotImplemented

    def __str__(self) -> str:
        return str(self._fae_expressions)

    def __len__(self) -> int:
        return len(self._fae_expressions)

    def __repr__(self) -> str:
        return str(self)


class FaeDict(_OpActionMixin["FaeDict"], Delayable):
    def __init__(self, expressions: dict[str, Delayable]) -> None:
        super().__init__(expressions=expressions)
        self._fae_expressions = dict(expressions)

    def _fae_exchange(self) -> FaeDict:
        new_exprs = {}
        for key, expr in self._fae_expressions.items():
            new_exprs[key] = yield expr
        return self.__class__(new_exprs)

    def _op_action(self, name: str, *args: Any, **kwargs: Any) -> FaeDict:
        opinfo = get_opinfo(attr_name=name)

        raveled = defaultdict(list)
        n = 0
        keys = set(self._fae_expressions)
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, FaeDict):
                n += 1
                if keys != set(arg._fae_expressions):
                    raise ValueError("All arguments of type `FaeDict` must have the same keys.")

                for key, item in arg._fae_expressions.items():
                    raveled[key].append(item)
            elif isinstance(arg, FaeList):
                raise ValueError("Cannot mix `FaeList` and `FaeDict` arguments. Choose one.")
            else:
                for key in keys:
                    raveled[key].append(arg)

        if n == 0:
            return FaeDict(
                {key: F(opinfo, item, *args, **kwargs)
                for key, item in self._fae_expressions.items()}
            )

        out = {}
        nargs = len(args)
        for key, value in self._fae_expressions.items():
            items_args = raveled[key][:nargs]
            items_kwargs = dict(zip(kwargs, raveled[key][nargs:]))
            out[key] = F(opinfo, value, *items_args, **items_kwargs)
        return FaeDict(out)

    def _resolve(self, /, **kwargs: Any) -> Any:
        result = {
            key: item._resolve(**kwargs)
            for key, item in self._fae_expressions.items()
        }
        self._record(result, kwargs)
        return result

    def __lshift__(self, other: Delayable) -> FaeDict:
        if isinstance(other, FaeDict):
            other_exprs = other._fae_expressions
            if set(self._fae_expressions) != set(other_exprs):
                return NotImplemented
            return FaeDict({
                key: item >> other_exprs[key]
                for key, item in self._fae_expressions.items()
            })
        elif isinstance(other, (Symbol, F)):
            return FaeDict({key: item >> other for key, item in self._fae_expressions.items()})
        else:
            return NotImplemented

    def __str__(self) -> str:
        return str(self._fae_expressions)

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self._fae_expressions)
