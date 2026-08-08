"""
Level-2 compilation: lower a FaeModule to a pure nn.Module with no Delayable
objects at runtime.

The expression tree is walked **once** at compile time to build nested Python
closures.  At inference time only those closures execute — no Delayable._resolve
calls, no tree traversal.

Interface of every compiled node:  ``fn(data, x, r) -> result``

  data  — the original model input (for ``A["name"]`` subscript access)
  x     — the current pipeline value (for ``X`` symbol access)
  r     — the per-call recall table (for ``R["name"]`` access), or ``None``
          when the tree contains no recalls
"""
from __future__ import annotations

from typing import Any, Callable
from torch import nn

from ._opinfo import OpInfo
from .spells import (
    Delayable,
    F,
    Chain,
    FList,
    FDict,
    Symbol,
    _RecallTable,
    X as X_sym,
    A as A_sym,
    R as R_sym,
)


def _const(val: Any) -> Callable:
    """Return a closure that always yields a constant value."""
    return lambda data, x, r: val


def _recall_names(node: Any, names: set[str]) -> None:
    """Collect every name referenced via ``R["name"]`` anywhere in the tree."""
    if not isinstance(node, Delayable) or isinstance(node, Symbol):
        return
    if isinstance(node, F):
        op, args = node.fae.op, node.fae.args
        if (
            isinstance(op, OpInfo)
            and op.name == "getitem"
            and len(args) == 2
            and isinstance(args[0], R_sym)
            and isinstance(args[1], str)
        ):
            names.add(args[1])
    for child in node.fae:
        _recall_names(child, names)


def _compile(node: Any, recorded: set[str]) -> Callable[[Any, Any, Any], Any]:
    """
    Recursively compile a Delayable tree node into a ``(data, x, r) -> result``
    Python callable with no Delayable objects remaining.

    ``recorded`` is the set of node names that are recalled somewhere via
    ``R["name"]``; only those nodes get a store into the recall table.
    """
    fn = _compile_node(node, recorded)

    name = node.fae.name if isinstance(node, Delayable) and not isinstance(node, Symbol) else None
    if name is not None and name in recorded:

        def named_fn(data: Any, x: Any, r: Any, _fn=fn, _name=name) -> Any:
            result = _fn(data, x, r)
            if r is not None:
                r[_name] = result
            return result

        return named_fn

    return fn


def _compile_node(node: Any, recorded: set[str]) -> Callable[[Any, Any, Any], Any]:
    if not isinstance(node, Delayable):
        return _const(node)

    # --- Symbols ---
    if isinstance(node, X_sym):
        return lambda data, x, r: x

    if isinstance(node, A_sym):
        return lambda data, x, r: data

    if isinstance(node, R_sym):
        return lambda data, x, r: r

    if isinstance(node, Symbol):
        raise ValueError(
            f"Symbol '{node.fae.name}' is still unresolved; ensure all P[I] "
            "templates are fully instantiated before calling lower()."
        )

    # --- Sequential chain ---
    if isinstance(node, Chain):
        ops = node.fae.ops
        if not ops:
            return lambda data, x, r: x
        fns: list[Callable] = [_compile(op, recorded) for op in ops]

        def chain_fn(data: Any, x: Any, r: Any, _fns: list[Callable] = fns) -> Any:
            # First op sees the incoming pipeline value: `data` at the top level
            # (forward passes x=data), the previous step's output when nested.
            result = _fns[0](data, x, r)
            for fn in _fns[1:]:
                result = fn(data, result, r)
            return result

        return chain_fn

    # --- Generic delayed call ---
    if isinstance(node, F):
        op = node.fae.op
        compiled_args: list[Callable] = [_compile(a, recorded) for a in node.fae.args]
        compiled_kwargs: dict[str, Callable] = {
            k: _compile(v, recorded) for k, v in node.fae.kwargs.items()
        }

        def f_fn(data: Any, x: Any, r: Any, _op=op, _ca=compiled_args, _ck=compiled_kwargs) -> Any:
            args = tuple(f(data, x, r) for f in _ca)
            kwargs = {k: v(data, x, r) for k, v in _ck.items()}
            return _op(*args, **kwargs)

        return f_fn

    # --- Parallel list ---
    if isinstance(node, FList):
        fns = [_compile(expr, recorded) for expr in node.fae.expressions]
        return lambda data, x, r, _f=fns: [f(data, x, r) for f in _f]

    # --- Parallel dict ---
    if isinstance(node, FDict):
        keys = list(node.fae.expressions.keys())
        fns = [_compile(expr, recorded) for expr in node.fae.expressions.values()]
        return lambda data, x, r, _k=keys, _f=fns: {k: f(data, x, r) for k, f in zip(_k, _f)}

    # Fallback for unknown node types: delegate to _resolve (sharing the recall table).
    def fallback(data: Any, x: Any, r: Any, _node=node) -> Any:
        kwargs = {"X": x}
        if r is not None:
            kwargs["R"] = r
        return _node._resolve(data, **kwargs)

    return fallback


class LoweredFaeModule(nn.Module):
    """
    An ``nn.Module`` produced by ``lower()``.  All sub-modules are registered
    normally so that ``.parameters()``, ``.to(device)``, and ``torch.compile``
    all work without special handling.

    The ``forward`` method calls pre-built Python closures with zero
    ``Delayable._resolve`` overhead.
    """

    def __new__(
        cls,
        forward_fn: Callable,
        sub_modules: list[tuple[str, nn.Module]],
        uses_recall: bool = False,
    ) -> "LoweredFaeModule":
        return object.__new__(cls)

    def __init__(
        self,
        forward_fn: Callable,
        sub_modules: list[tuple[str, nn.Module]],
        uses_recall: bool = False,
    ) -> None:
        super().__init__()
        self._forward_fn = forward_fn
        self._uses_recall = uses_recall
        for name, module in sub_modules:
            self.add_module(name, module)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from .faek import faek
        return faek.module__call__(self, *args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        from .spells import Input

        if len(args) == 1 and not kwargs:
            data = args[0]
        else:
            data = Input(*args, **kwargs)
        recalls = _RecallTable() if self._uses_recall else None
        return self._forward_fn(data, data, recalls)


def lower(model: "FaeModule") -> LoweredFaeModule:  # noqa: F821
    """
    Lower a ``FaeModule`` to a ``LoweredFaeModule`` with no Delayable runtime
    objects.

    The expression tree is compiled to Python closures **once** at this call
    site.  Every subsequent call to ``forward`` skips all Delayable machinery.

    Example::

        model = FaeModule(nn.Linear(10, 5) >> nn.ReLU() >> nn.Linear(5, 2))
        fast = lower(model)          # compile once
        out = fast(torch.randn(3, 10))  # pure closure dispatch

    Parameters
    ----------
    model:
        A ``FaeModule`` whose expression tree contains no unresolved symbols
        (``P``, ``I``) — i.e., all parameterised block templates have already
        been instantiated.

    Returns
    -------
    LoweredFaeModule
        An ``nn.Module`` equivalent to ``model`` but with no Delayable overhead
        at inference time.
    """
    recorded: set[str] = set()
    _recall_names(model._chain, recorded)
    forward_fn = _compile(model._chain, recorded)
    sub_modules = list(model.named_children())
    return LoweredFaeModule(forward_fn, sub_modules, uses_recall=bool(recorded))
