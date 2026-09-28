"""Compile a Delayable expression tree into a real ``nn.Module``.

Prefers AST codegen (``ast`` → ``compile`` / ``exec``; debug dumps via
``ast.unparse``). MVP AST coverage: ``Chain``, module calls, ``X``, binary
``+`` / ``*``. Trees with other nodes fall back to a thin runtime binder that
``_resolve``s on each forward (same role the old ``FaeModule`` wrapper had).
"""

from __future__ import annotations

import ast
from typing import Any, Optional

import torch
from torch import nn

from ._opinfo import OpInfo
from .spells import Chain, Delayable, F, Input, Symbol

_BINOPS = {
    "add": ast.Add,
    "mul": ast.Mult,
    "sub": ast.Sub,
    "truediv": ast.Div,
}


def _name(id: str, ctx: type = ast.Load) -> ast.Name:
    return ast.Name(id=id, ctx=ctx())


def _const(value: Any) -> ast.Constant:
    return ast.Constant(value=value)


def _attr(value: ast.AST, attr: str, ctx: type = ast.Load) -> ast.Attribute:
    return ast.Attribute(value=value, attr=attr, ctx=ctx())


def _call(
    func: ast.AST,
    args: list[ast.AST] | None = None,
    keywords: list[ast.keyword] | None = None,
) -> ast.Call:
    return ast.Call(func=func, args=args or [], keywords=keywords or [])


def _assign(target: str, value: ast.AST) -> ast.Assign:
    return ast.Assign(targets=[_name(target, ast.Store)], value=value)


class _Emitter:
    def __init__(self) -> None:
        self.stmts: list[ast.stmt] = []
        self.modules: dict[str, nn.Module] = {}
        self.constants: dict[str, Any] = {}
        self.callables: dict[str, Any] = {}
        self._tmp = 0
        self._used_names: set[str] = set()

    def fresh(self, prefix: str = "t") -> str:
        self._tmp += 1
        return f"{prefix}{self._tmp}"

    def register(self, module: nn.Module, preferred: Optional[str] = None) -> str:
        for name, existing in self.modules.items():
            if existing is module:
                return name
        base = preferred if preferred else f"_{len(self.modules)}"
        name = base
        n = 0
        while name in self._used_names or name in self.modules:
            n += 1
            name = f"{base}_{n}"
        self._used_names.add(name)
        self.modules[name] = module
        return name

    def _record_name(self, name: str, value_id: str) -> None:
        self.stmts.append(
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=_name("_r"),
                        slice=_const(name),
                        ctx=ast.Store(),
                    )
                ],
                value=_name(value_id),
            )
        )

    def emit(self, node: Any, env: dict[str, str]) -> str:
        """Emit ``node`` under symbol→local env; return the local holding the result."""
        if isinstance(node, type) and issubclass(node, Symbol):
            key = node.fae_name or node.__name__
            if key not in env:
                raise ValueError(f"materialize: unbound symbol {key!r}")
            return env[key]

        if not isinstance(node, Delayable):
            lit = self.fresh("c")
            self.constants[lit] = node
            return lit

        if isinstance(node, Chain):
            cur = env["X"]
            for op in node._fae_ops:
                cur = self.emit(op, {**env, "X": cur})
                if op.fae_name is not None:
                    self._record_name(op.fae_name, cur)
            return cur

        if isinstance(node, F):
            return self._emit_f(node, env)

        raise TypeError(f"materialize: unsupported node type {type(node).__name__}")

    def _is_module_call(self, node: F) -> bool:
        from .faek import faek

        if not node._fae_args or not isinstance(node._fae_args[0], nn.Module):
            return False
        op = node._fae_op
        if op is faek.module__call__:
            return True
        name = getattr(op, "__name__", "")
        return name in {"_wrapped_call_impl", "__call__"}

    def _emit_f(self, node: F, env: dict[str, str]) -> str:
        if self._is_module_call(node):
            module = node._fae_args[0]
            name = self.register(module, node.fae_name)
            call_args = [_name(self.emit(a, env)) for a in node._fae_args[1:]]
            call_kwargs = [
                ast.keyword(arg=k, value=_name(self.emit(v, env)))
                for k, v in node._fae_kwargs.items()
            ]
            out = self.fresh()
            self.stmts.append(
                _assign(
                    out,
                    _call(
                        _name("_call"),
                        args=[_attr(_name("self"), name), *call_args],
                        keywords=call_kwargs,
                    ),
                )
            )
            if node.fae_name is not None:
                self._record_name(node.fae_name, out)
            return out

        op = node._fae_op
        if isinstance(op, OpInfo):
            if op.name not in _BINOPS:
                raise TypeError(f"materialize: unsupported OpInfo {op.name!r}")
            left = self.emit(node._fae_args[0], env)
            right = self.emit(node._fae_args[1], env)
            out = self.fresh()
            self.stmts.append(
                _assign(
                    out,
                    ast.BinOp(
                        left=_name(left),
                        op=_BINOPS[op.name](),
                        right=_name(right),
                    ),
                )
            )
            if node.fae_name is not None:
                self._record_name(node.fae_name, out)
            return out

        fn_name = self.fresh("fn")
        self.callables[fn_name] = op
        args = [_name(self.emit(a, env)) for a in node._fae_args]
        keywords = [
            ast.keyword(arg=k, value=_name(self.emit(v, env)))
            for k, v in node._fae_kwargs.items()
        ]
        out = self.fresh()
        self.stmts.append(
            _assign(out, _call(_name(fn_name), args=args, keywords=keywords))
        )
        return out


def _build_module_ast(
    class_name: str,
    emitter: _Emitter,
    result: str,
) -> ast.Module:
    init_fn = ast.FunctionDef(
        name="__init__",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="self"),
                ast.arg(arg="_modules"),
                ast.arg(arg="_constants"),
                ast.arg(arg="_callables"),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Expr(value=_call(_attr(_call(_name("super")), "__init__"))),
            ast.Assign(
                targets=[_attr(_name("self"), "_fae_constants", ast.Store)],
                value=_name("_constants"),
            ),
            ast.Assign(
                targets=[_attr(_name("self"), "_fae_callables", ast.Store)],
                value=_name("_callables"),
            ),
            ast.For(
                target=ast.Tuple(
                    elts=[_name("_name", ast.Store), _name("_mod", ast.Store)],
                    ctx=ast.Store(),
                ),
                iter=_call(_attr(_name("_modules"), "items")),
                body=[
                    ast.Expr(
                        value=_call(
                            _attr(_name("self"), "add_module"),
                            args=[_name("_name"), _name("_mod")],
                        )
                    )
                ],
                orelse=[],
            ),
        ],
        decorator_list=[],
        type_params=[],
    )

    call_fn = ast.FunctionDef(
        name="__call__",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self")],
            vararg=ast.arg(arg="args"),
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=ast.arg(arg="kwargs"),
            defaults=[],
        ),
        body=[
            ast.ImportFrom(
                module="faeyon.magic.faek",
                names=[ast.alias(name="faek", asname="_faek")],
                level=0,
            ),
            ast.Return(
                value=ast.Call(
                    func=_attr(_name("_faek"), "module__call__"),
                    args=[
                        _name("self"),
                        ast.Starred(value=_name("args"), ctx=ast.Load()),
                    ],
                    keywords=[ast.keyword(arg=None, value=_name("kwargs"))],
                )
            ),
        ],
        decorator_list=[],
        type_params=[],
    )

    forward_body: list[ast.stmt] = [
        ast.ImportFrom(
            module="faeyon.magic.faek",
            names=[ast.alias(name="faek", asname="_faek")],
            level=0,
        ),
        _assign("_call", _attr(_name("_faek"), "module__call__")),
        _assign("_r", ast.Dict(keys=[], values=[])),
    ]
    for name in emitter.constants:
        forward_body.append(
            _assign(
                name,
                ast.Subscript(
                    value=_attr(_name("self"), "_fae_constants"),
                    slice=_const(name),
                    ctx=ast.Load(),
                ),
            )
        )
    for name in emitter.callables:
        forward_body.append(
            _assign(
                name,
                ast.Subscript(
                    value=_attr(_name("self"), "_fae_callables"),
                    slice=_const(name),
                    ctx=ast.Load(),
                ),
            )
        )
    forward_body.extend(emitter.stmts)
    forward_body.append(ast.Return(value=_name(result)))

    forward_fn = ast.FunctionDef(
        name="forward",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="self"), ast.arg(arg="x")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=forward_body,
        decorator_list=[],
        type_params=[],
    )

    class_def = ast.ClassDef(
        name=class_name,
        bases=[_attr(_name("nn"), "Module")],
        keywords=[],
        body=[init_fn, call_fn, forward_fn],
        decorator_list=[],
        type_params=[],
    )
    return ast.Module(body=[class_def], type_ignores=[])


class _BoundModule(nn.Module):
    """Runtime binder: register leaf modules and ``_resolve`` on each forward."""

    def __new__(cls, chain: Delayable) -> "_BoundModule":
        # Bypass faek's nn.Module.__new__ patch (Delayable ctor arg → DelayedModule).
        return object.__new__(cls)

    def __init__(self, chain: Delayable) -> None:
        super().__init__()
        self._chain = chain
        self._extract_modules()

    def _extract_modules(self) -> None:
        from .faek import faek

        seen: set[int] = set()
        counter = [0]

        def visit(node: Delayable) -> Delayable | None:
            if not isinstance(node, F):
                return None
            if node._fae_op is not faek.module__call__:
                return None
            module = node._fae_args[0]
            if not isinstance(module, nn.Module) or id(module) in seen:
                return None
            seen.add(id(module))
            name = node.fae_name or f"_{counter[0]}"
            counter[0] += 1
            self.add_module(name, module)
            return None

        self._chain.fae_find(F, visit)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from .faek import faek

        return faek.module__call__(self, *args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs:
            data = args[0]
            if isinstance(data, Input):
                return self._chain._resolve(X=data, A=data)
            return self._chain._resolve(X=data)
        inp = Input(*args, **kwargs)
        return self._chain._resolve(X=inp, A=inp)


def _ast_supported(expr: Delayable) -> bool:
    """Heuristic: MVP AST only handles ``Chain`` / ``F`` / ``Symbol`` leaves."""
    ok = True

    def visit(node: Delayable) -> None:
        nonlocal ok
        if isinstance(node, (Chain, F)):
            return
        if isinstance(node, type) and issubclass(node, Symbol):
            return
        ok = False

    try:
        expr.fae_find(Delayable, visit)
    except Exception:
        return False
    return ok


def _ast_materialize(
    expr: Delayable,
    *,
    class_name: str = "Materialized",
    debug_source: bool = False,
) -> nn.Module:
    emitter = _Emitter()
    result = emitter.emit(expr, {"X": "x"})
    tree = _build_module_ast(class_name, emitter, result)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<faeyon:materialize>", "exec")

    ns: dict[str, Any] = {"nn": nn, "torch": torch}
    exec(code, ns)
    module = ns[class_name](emitter.modules, emitter.constants, emitter.callables)
    if debug_source:
        module._fae_source = ast.unparse(tree)  # type: ignore[attr-defined]
    return module


def materialize(
    expr: Delayable,
    *,
    class_name: str = "Materialized",
    debug_source: bool = False,
) -> nn.Module:
    """
    Turn a Delayable tree into a normal ``nn.Module``.

    Uses AST codegen when the tree is within MVP coverage; otherwise binds via
    runtime ``_resolve``.

    Parameters
    ----------
    expr :
        Fully bound expression (no unresolved ``I`` / ``DelayedModule`` recipes).
    class_name :
        Generated class name (AST path only).
    debug_source :
        If True (AST path), attach ``ast.unparse`` output on ``module._fae_source``.
    """
    if not isinstance(expr, Delayable):
        raise TypeError("materialize expects a Delayable expression")

    if _ast_supported(expr):
        try:
            return _ast_materialize(
                expr, class_name=class_name, debug_source=debug_source
            )
        except (TypeError, ValueError, KeyError):
            pass
    return _BoundModule(expr)


def to_module(expr: Delayable, **kwargs: Any) -> nn.Module:
    """Alias for :func:`materialize`."""
    return materialize(expr, **kwargs)
