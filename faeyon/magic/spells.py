from __future__ import annotations
import torch
import abc
import inspect
import itertools
import re
import weakref

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Optional, overload

from torch import nn
from ._opinfo import get_opinfo, OpInfo
from .base import _MappingKey


class _Frame:
    """One suspended `fae_exchange` coroutine in an iterative traversal."""
    __slots__ = ("node", "exchange", "path", "sent", "changed")
    def __init__(self, node: Delayable, path: str, *, force_copy: bool = False) -> None:
        self.node = node
        self.exchange = node.fae_exchange(force_copy=force_copy)
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


class _DelayableBase:
    """
    Shared Delayable behavior used by both ``Delayable`` instances and the symbol
    metaclass. Kept separate from ``ABC`` so ``_SymbolMeta`` can subclass ``type``
    without also subclassing an ``ABCMeta``-created ABC (which breaks isinstance).
    """
    def __init__(self, *args, **kwargs) -> None:
        sig = inspect.signature(self.__init__)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        self._fae_arguments = bound

        self._fae_name = None
        self._fae_parent: Optional[weakref.ref[Delayable]] = None
        # name -> direct children with that ``fae_name`` (Symbols omitted).
        self._fae_child_names: dict[str, list[Delayable]] = defaultdict(list)
        # Subclasses set child storage before ``super().__init__``; then we
        # auto-name and uniquify child names once here.
        self.fae_init()

        parent_ref = weakref.ref(self)
        for c in self.fae_children():
            if not isinstance(c, Delayable) or isinstance(c, type):
                continue
            c._fae_parent = parent_ref
            name = c.fae_name
            if name is not None:
                self._fae_child_names[name].append(c)
        self._dedup_child_names()

    @property
    def fae_name(self) -> Optional[str]:
        return self._fae_name

    @fae_name.setter
    def fae_name(self, value: Optional[str]) -> None:
        old = self._fae_name
        if old == value:
            return
        self._fae_name = value

        parent = self._fae_parent() if self._fae_parent is not None else None
        if parent is None:
            return

        # Move this node between the parent's name buckets, then re-dedup.
        names = parent._fae_child_names
        if old is not None:
            bucket = names.get(old)
            if bucket is not None:
                bucket = [c for c in bucket if c is not self]
                if bucket:
                    names[old] = bucket
                else:
                    del names[old]
        if value is not None:
            bucket = names[value]
            # Identity check: ``Delayable.__eq__`` builds a delayed comparison.
            if all(c is not self for c in bucket):
                bucket.append(self)
        parent._dedup_child_names()

    def fae_init(self) -> None:
        """Subclass hook to set an auto ``fae_name`` after base fields exist."""

    def _dedup_child_names(self) -> None:
        """
        Among direct children, suffix duplicate ``fae_name`` values:
        ``add``, ``add`` → ``add[0]``, ``add[1]``.

        Uses ``_fae_child_names`` so repeat calls are cheap when nothing is
        duplicated. Applies to auto-names and ``@`` names; Symbols are never
        renamed.
        """
        dups = {
            name: nodes
            for name, nodes in self._fae_child_names.items()
            if len(nodes) > 1
        }
        if not dups:
            return

        for name, nodes in dups.items():
            del self._fae_child_names[name]
            for i, c in enumerate(nodes):
                # Bypass the setter: it would re-enter this dedup mid-iteration.
                new_name = f"{name}[{i}]"
                c._fae_name = new_name
                self._fae_child_names[new_name].append(c)

    def fae_exchange(self, *, force_copy: bool = False) -> Delayable:
        """
        Coroutine that yields direct children, expects replacements, and returns
        a (possibly new) delayable. When children are unchanged, returns ``self``
        unless ``force_copy`` (used by ``fae_clone``).
        """
        yield from ()
        return self

    def fae_children(self, items: bool = False) -> Iterator[Any]:
        """Yield this node's direct children, echoing every item back unchanged."""
        exchange = self.fae_exchange()
        sent = None
        while True:
            try:
                item = exchange.send(sent)
            except StopIteration:
                return
            
            sent = item
            if items or isinstance(item, Delayable):
                yield item

    def fae_uses(self, symbol: type) -> bool:
        """
        True if resolving this node depends on ``symbol``.

        Walks Delayable children. ``Chain`` overrides this for ``X``: only the
        first op sees the incoming pipeline value, so a nested chain uses ``X``
        only when its first op does.
        """
        return any(
            isinstance(child, Delayable) and child.fae_uses(symbol)
            for child in self.fae_children()
        )

    def fae_traverse(
        self,
        pre_fn: Optional[Callable[[Delayable, str], Any]] = None,
        *,
        always_copy: bool = False,
        pre_items: bool = False,
        post_fn: Optional[Callable[[Delayable, str], Any]] = None,
    ) -> Delayable:
        """
        Depth-first rebuild of the tree rooted at ``self``, without recursion.

        Each node is identified by a dotted ``path`` of ``fae_name`` values
        (``"_"`` for unnamed nodes), e.g. ``"Chain.Linear"``.

        Parameters
        ----------
        pre_fn:
            Pre-order hook, ``pre_fn(node, path)``, called before a node's children
            are opened. Return ``None`` to descend into the node, or a replacement
            to splice in its place; a replaced node is not descended into and
            ``post_fn`` is not called on it. ``None`` descends into every node.
        always_copy:
            Rebuild every container even when none of its children changed
            (used by ``fae_clone``). Otherwise unchanged subtrees are reused as is.
        pre_items:
            Also pass non-``Delayable`` items (modules, constants) to ``pre_fn``.
            They are never descended into, and never passed to ``post_fn``.
        post_fn:
            Post-order hook, ``post_fn(node, path)``, called once all of a node's
            children have been processed and the node has been rebuilt from them.
            Return the node itself to keep it, or anything else to replace it in
            its parent (the replacement may be a non-``Delayable``; the parent's
            ``fae_exchange`` decides whether that is allowed). ``fae_bind`` uses
            it to resolve each node whose children are all concrete.

        Returns
        -------
        The rebuilt root, or ``self`` when nothing changed and ``always_copy``
        is false.
        """
        root_path = self.fae_name or "_"
        replaced = pre_fn(self, root_path) if pre_fn is not None else None
        
        if replaced is not None:
            return replaced
        
        stack = [_Frame(self, root_path, force_copy=always_copy)]
        
        while True:
            frame = stack[-1]
            try:
                item = frame.exchange.send(frame.sent)
            except StopIteration as stop:
                node = stop.value if frame.changed or always_copy else frame.node
                # Exchange may collapse a node to a non-Delayable (e.g. IF → arm);
                # the parent decides whether that is allowed. Only Delayables have
                # post_fn / fae_resolve hooks.
                if post_fn is not None and isinstance(node, Delayable):
                    updated = post_fn(node, frame.path)
                    if updated is not node:
                        frame.changed = True
                        node = updated
                stack.pop()
                
                if not stack:
                    return node
                
                parent = stack[-1]
                parent.sent = node
                parent.changed = parent.changed or node is not frame.node
                continue
            
            is_node = isinstance(item, Delayable)
            path = f"{frame.path}.{item.fae_name or '_'}" if is_node else frame.path
            
            if pre_fn is not None and (is_node or pre_items):
                new_item = pre_fn(item, path)
                if new_item is not None:
                    frame.sent = new_item      # replaced: prune, never open its coroutine
                    frame.changed = True
                    continue
            
            frame.sent = item
            if is_node:
                stack.append(_Frame(item, path, force_copy=always_copy))

    def fae_apply(self, fn=None, *, always_copy: bool = False) -> Delayable:
        """Exchange this node's own items; children are not traversed."""
        exchange = self.fae_exchange(force_copy=always_copy)
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

    def fae_clone(self, recurse: bool = False, clone_modules: bool = True) -> Delayable:
        if not recurse:
            return self.fae_apply(always_copy=True)

        def visit(item: Any, path: str) -> Any:            
            if clone_modules and isinstance(item, nn.Module):
                return item.clone()
            return None

        return self.fae_traverse(visit, always_copy=True, pre_items=True)

    def fae_match(
        self,
        pattern: str | type[Delayable],
    ) -> Callable[[Delayable, str], bool]:
        if isinstance(pattern, type):
            def matches(node: Delayable, path: str) -> bool:
                return isinstance(node, pattern)

            return matches

        literal = re.escape(pattern) == pattern

        def matches(node: Delayable, path: str) -> bool:
            if re.fullmatch(pattern, path) is not None:
                return True
            name = node.fae_name
            if name is None:
                return False
            if re.fullmatch(pattern, name) is not None:
                return True
            # Bare name "layer" also matches auto-suffixed "layer[0]", "layer[1]", …
            if literal and name.startswith(pattern + "["):
                return True
            return False

        return matches

    @overload
    def fae_find(
        self,
        pattern: str | type[Delayable],
        callback: None = None,
    ) -> Iterator[Delayable]: ...

    @overload
    def fae_find(
        self,
        pattern: str | type[Delayable],
        callback: Callable[[Delayable], Delayable],
    ) -> Delayable: ...

    def fae_find(
        self,
        pattern: str | type[Delayable],
        callback: Optional[Callable[[Delayable], Delayable]] = None,
    ) -> Iterator[Delayable] | Delayable:
        """
        Find nodes matching ``pattern`` (name / path regex, or Delayable type).

        Without ``callback``, yields matching nodes. With ``callback``, rebuilds the
        tree replacing each match with ``callback(node)``.
        """
        matches = self.fae_match(pattern)

        if callback is None:
            found: list[Delayable] = []

            def collect(node: Delayable, path: str) -> None:
                if matches(node, path):
                    found.append(node)
                return None

            self.fae_traverse(collect)
            return iter(found)

        def visit(node: Delayable, path: str) -> Optional[Delayable]:
            if not matches(node, path):
                return None
            return callback(node)

        return self.fae_traverse(visit)

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
        ``|`` is reserved for resolve (``data | expr`` via ``__ror__``), not
        bitwise or. Reject left-hand Delayable uses so they cannot be confused
        with piping; use ``torch.bitwise_or`` for bitwise OR.
        """
        if isinstance(other, torch.Tensor):
            # Prevent ``X | tensor`` from becoming torch bitwise via __torch_function__.
            raise TypeError(
                "Cannot use `|` with a tensor on the right; use `tensor | expr` to resolve."
            )
        raise TypeError(
            "`|` is reserved for resolve (`data | expr`). For bitwise or, use `torch.bitwise_or`."
        )

    def __ror__(self, other: Any) -> Any:
        """``data | Delayable`` evaluates the delayed expression."""
        if isinstance(other, Delayable):
            return NotImplemented
        return self.fae_resolve(X=other)

    def __matmul__[T: Delayable](self: T, other: Any) -> T:
        """
        ``expr @ "name"`` sets ``fae_name`` (overrides auto-names). Matmul with a
        non-string RHS is handled by ``_OpActionMixin``.
        """
        if isinstance(other, str):
            if isinstance(self, type):
                raise TypeError("Cannot rename a Symbol with `@`.")
            
            if "." in other:
                raise ValueError("Name cannot contain a period.")
            
            self.fae_name = other
            return self
        return NotImplemented

    def __rmatmul__[T: Delayable](self: T, other: Any) -> T:
        if isinstance(other, str):
            raise TypeError('Name must be on the right: expr @ "name".')
        return NotImplemented

    def __mod__[T: Delayable](self: T, modifier: Any) -> T:
        """
        `%` applies a modifier (`Modify` / objects with `__rmod__`).
        Arithmetic `%` is handled by `_OpActionMixin` when this returns
        `NotImplemented`. Naming uses `@`, not `%`.
        """
        rmod = getattr(modifier, "__rmod__", None)
        if callable(rmod):
            out = rmod(self)
            if out is not NotImplemented:
                return out
        return NotImplemented
        
    def __rmod__[T: Delayable](self: T, other: Any) -> T:
        """Modifiers should always be to the right of the Delayable."""
        return NotImplemented
        
    def fae_bind(self, value: Any) -> Any:
        """
        Substitute `N` with `value` through one `fae_traverse` without running
        a pipeline.

        Each node is rebuilt from its bound children; a node with no `Delayable`
        children left is then resolved (`N` becomes `value`, an `F` over
        constants is computed). `_Unpack` is kept so its parent still splats it.
        Containers reject a child that is no longer a `Delayable` in their
        `fae_exchange`.
        """
        def post_fn(node: Delayable, path: str) -> Any:
            if isinstance(node, _Unpack) or any(
                isinstance(child, Delayable) for child in node.fae_children()
            ):
                return node
            return node.fae_resolve(N=value)

        return self.fae_traverse(post_fn=post_fn)

    def __rshift__(self, other: Delayable | int | Sequence[Any]) -> Chain:
        """
        The right shift operator (>>) is used to chain Delayables together, like the layers in 
        a neural network.

        There are three possible cases:
        1. `Delayable >> Delayable` -> Chain(Delayable, Delayable)
        2. `Delayable >> int` -> expand over ``range(n)``, binding ``N`` to each index
        3. `Delayable >> Sequence` -> one clone per element, binding ``N`` to that element

        ``DelayedModule`` on either side is promoted to ``module(X)`` (override /
        ``Chain``); use ``recipe(other)`` for a non-``X`` call arg.
        """
        if isinstance(other, DelayedModule):
            other = other(X)
        if isinstance(other, Delayable):
            return Chain(self, other)
        elif isinstance(other, int):
            return self.fae_expand(range(other))
        elif isinstance(other, Sequence) and not isinstance(other, (str, bytes)):
            return self.fae_expand(other)

        return NotImplemented

    def fae_expand(self, rows: Sequence[Any]) -> Delayable:
        """
        Clone once per element of `rows`, binding `N` to that element. 
        `>> n` passes `range(n)`; `>> seq` passes `seq`.
        """
        base_name = self.fae_name
        out: Optional[Delayable] = None

        for i, row in enumerate(rows):
            cloned = self.fae_clone(recurse=True, clone_modules=True)
            cloned = cloned.fae_bind(row)

            if base_name is not None and isinstance(cloned, Delayable):
                cloned.fae_name = f"{base_name}[{i}]"

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


class Delayable(_DelayableBase, ABC):
    """
    Delayable is the base class for all delayable objects. It provides the base functionality for
    conditional evaluation, chaining, and resolving with data.
    """

    @abstractmethod
    def fae_resolve(self, /, **kwargs: Any) -> Any:
        """
        Resolve this delayable using keyword-bound symbols (at least ``X`` for
        the pipeline value). Must be implemented by subclasses.
        """


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
            # Defer when the right argument is a FaeList or FaeDict.
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
        out = super().__matmul__(other)
        if out is not NotImplemented:
            return out
        return self._op_action("__matmul__", other)

    def __rmatmul__(self, other: Any) -> T:
        out = super().__rmatmul__(other)
        if out is not NotImplemented:
            return out
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


class _SymbolMeta(_OpActionMixin["F"], _DelayableBase, abc.ABCMeta):
    """
    TODO: I should disable applying modifiers or names to symbols.... because they apply globally
    on class instances.

    Inherits ``_DelayableBase`` (not ``Delayable``) so this metaclass can subclass
    ``ABCMeta`` / ``type``. Virtual subclass registration keeps ``isinstance(X, Delayable)``.
    """
    _registry: dict[str, type[Symbol]] = {}
    
    def __new__(mcs, name, bases, namespace, **kwargs) -> type[Symbol]:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        
        if any(isinstance(base, _SymbolMeta) for base in bases):
            mcs._registry[name] = cls
    
        return cls

    def __init__(cls, name, bases, namespace, **kwargs) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        # _DelayableBase.__init__ (via metaclass MRO) leaves fae_name=None; symbols
        # must resolve by class name so `X=...` / `A=...` kwargs match.
        cls.fae_name = name

    def fae_resolve(self, /, **kwargs: Any) -> Any:
        """
        If symbol is in kwargs, replace with that value; otherwise leave unresolved.
        """
        if self.fae_name in kwargs:
            return kwargs[self.fae_name]
        return self

    def fae_uses(self, symbol: type) -> bool:
        return self is symbol

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


Delayable.register(_SymbolMeta)


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


class N(Symbol):
    """
    Forward-looking symbol, bound by ``>>`` to each item on its right:
    ``expr >> 3`` binds ``N`` to 0, 1, 2 and ``expr >> [a, b]`` to ``a``, ``b``.
    ``X`` instead looks back at the previous result.
    """
    pass


class R(Symbol):
    """
    The recall symbol: `R["name"]` resolves to the output that the node named
    ``@ "name"`` produced earlier in the *current* evaluation. This turns names
    into long-range skip connections (U-Net, FPN) without threading values through
    the pipeline by hand:

        unet = (
            enc_block(1, 64) @ "e1"
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
    * Names are matched by their plain node name (the string given to ``@``), not by
    dotted path; recalled names should therefore be unique within one model.
    """
    @classmethod
    def fae_resolve(cls, /, **kwargs: Any) -> Any:
        # R only means the recall table for the current evaluation.
        return kwargs.get(_RECALL_KEY, cls)
    

class _Unpack(Delayable):
    """
    Represents an unpacking operation (*X). When resolved, unpacks the data as *args.
    """
    def __init__(self, target: Any, is_map: bool = False) -> None:
        self._fae_target = target
        self._fae_is_map = is_map
        super().__init__(target=target, is_map=is_map)
    
    def fae_exchange(self, *, force_copy: bool = False) -> "_Unpack":
        new_target = yield self._fae_target
        if new_target is self._fae_target and not force_copy:
            return self
        return _Unpack(new_target, is_map=self._fae_is_map)

    def fae_resolve(self, /, **kwargs: Any) -> Any:
        target = self._fae_target
        if isinstance(target, Delayable):
            return target.fae_resolve(**kwargs)
        return target
    
    def __repr__(self) -> str:
        if self._fae_is_map:
            prefix = "**"
        else:
            prefix = "*"
        return f"{prefix}{self._fae_target!r}"


class F(_OpActionMixin["F"], Delayable):
    def __init__(self, op: Callable[..., Any], /, *args, **kwargs) -> None:
        self._fae_op = op
        self._fae_args = args
        self._fae_kwargs = kwargs
        super().__init__(op, *args, **kwargs)

    def fae_init(self) -> None:
        if self.fae_name is not None:
            return

        op, args = self._fae_op, self._fae_args
        if isinstance(op, OpInfo):
            self.fae_name = op.name
            return

        from .faek import faek

        if args and isinstance(args[0], nn.Module):
            op_name = getattr(op, "__name__", "")
            if op is faek.module__call__ or op_name in {
                "_wrapped_call_impl", "__call__", "_call_impl",
            }:
                self.fae_name = type(args[0]).__name__
                return

        if isinstance(op, type):
            self.fae_name = op.__name__
            return

        name = getattr(op, "__name__", None)
        if isinstance(name, str) and name and not name.startswith("<"):
            self.fae_name = name

    def fae_resolve(self, /, **kwargs: Any) -> Any:
        resolved_args = []
        for arg in self._fae_args:
            if isinstance(arg, Delayable):
                resolved = arg.fae_resolve(**kwargs)
            else:
                resolved = arg
            
            if isinstance(arg, _Unpack):
                resolved_args.extend(resolved)
            else:
                resolved_args.append(resolved)

        resolved_kwargs = {}
        for k, v in self._fae_kwargs.items():
            if isinstance(v, Delayable):
                resolved = v.fae_resolve(**kwargs)
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
            out = self.__class__(self._fae_op, *resolved_args, **resolved_kwargs)
            out.fae_name = self.fae_name
            return out

        result = self._fae_op(*resolved_args, **resolved_kwargs)
        self._record(result, kwargs)
        return result

    def fae_exchange(self, *, force_copy: bool = False) -> F:
        new_args = []
        changed = False
        for arg in self._fae_args:
            new_arg = yield arg
            changed = changed or new_arg is not arg
            new_args.append(new_arg)

        new_kwargs = {}
        for key, kwarg in self._fae_kwargs.items():
            new_kwarg = yield kwarg
            changed = changed or new_kwarg is not kwarg
            new_kwargs[key] = new_kwarg

        if not changed and not force_copy:
            return self
        out = self.__class__(self._fae_op, *new_args, **new_kwargs)
        out.fae_name = self.fae_name
        return out
    
    def __str__(self) -> str:
        if isinstance(self._fae_op, OpInfo):
            return self._fae_op.to_string(*self._fae_args, **self._fae_kwargs)

        # ``module(X)`` is stored as ``F(Module.__call__, module, X)``; PyTorch
        # aliases ``__call__`` to ``_wrapped_call_impl``, so pretty-print as
        # ``Linear(...)(X)`` instead of ``_wrapped_call_impl(Linear(...), X)``.
        from .faek import faek

        op = self._fae_op
        op_name = getattr(op, "__name__", "")
        if (
            self._fae_args
            and isinstance(self._fae_args[0], nn.Module)
            and (
                op is faek.module__call__ 
                or op_name in {"_wrapped_call_impl", "__call__", "_call_impl"}
            )
        ):
            module, *call_args = self._fae_args
            args = list(map(repr, call_args))
            args.extend(f"{k}={v!r}" for k, v in self._fae_kwargs.items())
            return f"{module!r}({', '.join(args)})"

        try:
            name = op.__name__
        except AttributeError:
            name = f"{op!r}"

        args = list(map(repr, self._fae_args))
        args.extend(f"{k}={v!r}" for k, v in self._fae_kwargs.items())
        return f"{name}({', '.join(args)})"

    def __repr__(self) -> str:
        return str(self)


class DelayedModule(F):
    """
    Module constructor recipe: ``F`` of an ``nn.Module`` class whose args still
    contain delayables (e.g. ``nn.Linear(N, N * 3)`` via faek ``__new__``).

    Resolution is the shared ``F.fae_resolve`` path. ``>>`` promotes to
    ``module(X)``; use ``recipe(arg)`` for an explicit call arg other than ``X``.
    """

    def __rshift__(self, other: Any) -> Any:
        return self(X) >> other

    def __call__(self, *args: Any, **kwargs: Any) -> F:
        from .faek import faek

        call = F(faek.module__call__, self, *args, **kwargs)
        call.fae_name = self.fae_name or (
            self._fae_op.__name__ if isinstance(self._fae_op, type) else None
        )
        return call


class Chain(_OpActionMixin[F], Delayable):
    """
    A Chain is a sequence of operations: `op0 >> op1 << op2 >> ... >> opn`.

    Each stage after the first must use ``X`` so it takes the previous result.
    The first stage may use ``A`` (or ``X``) for the pipeline / call arguments.
    """
    def __init__(self, *ops: Delayable) -> None:
        if not ops:
            raise ValueError("Chain must have at least one operation.")
        
        self._fae_ops = []
        for op in ops:
            if isinstance(op, DelayedModule):
                self._fae_ops.append(op(X))
            elif isinstance(op, Delayable):
                self._fae_ops.append(op)
            else:
                raise ValueError("All arguments must be of subtype `Delayable` or `nn.Module`.")
        self._check_ops(self._fae_ops)
        # Chains stay unnamed unless the user sets ``@ "name"``.
        super().__init__(*ops)

    @staticmethod
    def _op_label(op: Delayable) -> str:
        return getattr(op, "fae_name", None) or type(op).__name__

    @staticmethod
    def _check_ops(ops: Sequence[Delayable]) -> None:
        for i, op in enumerate(ops):
            if not isinstance(op, Delayable):
                raise TypeError(
                    f"Chain op {Chain._op_label(op)!r} is {type(op).__name__}, "
                    "which is not a Delayable."
                )
            # First stage may use A (or anything); later stages must take X.
            if i > 0 and not op.fae_uses(X):
                raise TypeError(
                    f"Chain op {Chain._op_label(op)!r} does not use X; "
                    "each stage after the first must take the previous result via X."
                )

    def fae_uses(self, symbol: type) -> bool:
        # Incoming X is only fed to the first op; later ops see the prior result as X.
        if symbol is X:
            return self._fae_ops[0].fae_uses(X)
        return super().fae_uses(symbol)

    def fae_resolve(self, /, **kwargs: Any) -> Any:
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
        x = self._fae_ops[0].fae_resolve(**kwargs)
        for op in self._fae_ops[1:]:
            kwargs["X"] = x
            x = op.fae_resolve(**kwargs)
        self._record(x, kwargs)
        return x

    def fae_exchange(self, *, force_copy: bool = False) -> Chain:
        new_ops = []
        changed = False
        for op in self._fae_ops:
            new_op = yield op
            changed = changed or new_op is not op
            if not isinstance(new_op, Delayable):
                raise TypeError(
                    f"Chain op {self._op_label(op)!r} bound to {type(new_op).__name__}, "
                    "which is not a Delayable."
                )
            new_ops.append(new_op)
        if changed:
            self._check_ops(new_ops)
        if not changed and not force_copy:
            return self
        out = self.__class__(*new_ops)
        out.fae_name = self.fae_name
        return out

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
        return other.fae_resolve(**self._kwargs)


class FaeList(_OpActionMixin["FaeList"], Delayable):
    """
    TODO: Make FList generic e.g. Flist[Delayable, etc..]
    """
    def __init__(self, expressions: list[Delayable]) -> None:
        self._fae_expressions = list(expressions)
        super().__init__(expressions=expressions)

    def fae_init(self) -> None:
        if self.fae_name is None:
            self.fae_name = "FaeList"

    def fae_exchange(self, *, force_copy: bool = False) -> FaeList:
        new_exprs = []
        changed = False
        for expr in self._fae_expressions:
            new_expr = yield expr
            changed = changed or new_expr is not expr
            if not isinstance(new_expr, Delayable):
                raise TypeError(
                    f"FaeList item bound to {type(new_expr).__name__}, "
                    "which is not a Delayable."
                )
            new_exprs.append(new_expr)
        if not changed and not force_copy:
            return self
        out = self.__class__(new_exprs)
        out.fae_name = self.fae_name
        return out

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

    def fae_resolve(self, /, **kwargs: Any) -> Any:
        result = [item.fae_resolve(**kwargs) for item in self._fae_expressions]
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
        self._fae_expressions = dict(expressions)
        super().__init__(expressions=expressions)

    def fae_init(self) -> None:
        if self.fae_name is None:
            self.fae_name = "FaeDict"

    def fae_exchange(self, *, force_copy: bool = False) -> FaeDict:
        new_exprs = {}
        changed = False
        for key, expr in self._fae_expressions.items():
            new_expr = yield expr
            changed = changed or new_expr is not expr
            if not isinstance(new_expr, Delayable):
                raise TypeError(
                    f"FaeDict item {key!r} bound to {type(new_expr).__name__}, "
                    "which is not a Delayable."
                )
            new_exprs[key] = new_expr
        if not changed and not force_copy:
            return self
        out = self.__class__(new_exprs)
        out.fae_name = self.fae_name
        return out

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

    def fae_resolve(self, /, **kwargs: Any) -> Any:
        result = {
            key: item.fae_resolve(**kwargs)
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
