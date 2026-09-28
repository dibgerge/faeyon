import abc
from typing import Any, Optional

from faeyon.magic.base import _NoValue
from faeyon.magic.spells import Delayable, DelayedModule, F, X


class Modifier(abc.ABC):
    """
    Base class for modifiers. A modifier is a callable that takes a node in the expression
    tree and returns a new node that replaces it.

    Usage:
        model % Modify("encoder.attn", QuantModifier())
    """
    @abc.abstractmethod
    def __call__(self, node: Delayable) -> Delayable:
        """Return a new Delayable that replaces `node`."""


class Modify:
    """
    Locates a node in the expression tree by path or type, then replaces it by calling
    `modifier(node)`. The modifier is responsible for returning the replacement node.

    Examples:
        model % Modify("encoder.layer", Quantizer())
        model % Modify(F, CacheModifier())
    """
    def __init__(self, lookup: str | type[Delayable], modifier: Modifier) -> None:
        self.lookup = lookup
        self.modifier = modifier

    def __rmod__(self, root: Delayable) -> Delayable:
        if not isinstance(root, Delayable):
            return NotImplemented

        def callback(node: Delayable) -> Delayable:
            return self.modifier(node)

        return root.fae_find(self.lookup, callback=callback)


class IF(Delayable, Modifier):
    """
    Conditional branch used both as an expression node and as a ``Modify`` target.

    Expression form (clone-time / resolve-time)::

        main + IF(cond, X, else_=proj)

    When ``cond`` is statically known after ``fae_bind`` (e.g. after ``I`` is bound),
    the unused arm is dropped. Otherwise a runtime select ``F`` is left in the tree.

    Modifier form::

        model % Modify(r".*dropout", IF(training, else_=X))
    """
    def __init__(
        self,
        condition: bool | Delayable,
        then: Any = _NoValue,
        else_: Optional[Any] = None,
    ) -> None:
        self.condition = condition
        self._modifier_mode = then is _NoValue
        self.then = None if self._modifier_mode else then
        self.else_ = X if else_ is None else else_
        super().__init__(condition, then, else_=else_)

    def __call__(self, node: Delayable) -> Delayable:
        if self._modifier_mode:
            return self._select(node)
        raise TypeError("Expression-form IF is not callable; use IF(cond, then, else_=...)")

    def _select(self, then_branch: Any) -> Any:
        cond = self.condition
        if isinstance(cond, Delayable):
            return F(
                lambda c, a, b: a if c else b,
                cond,
                then_branch,
                self.else_,
            )
        if isinstance(cond, bool):
            return then_branch if cond else self.else_
        raise ValueError(f"Invalid condition type: {type(cond)}")

    def _resolve(self, /, **kwargs: Any) -> Any:
        if self._modifier_mode:
            raise ValueError("Modifier-form IF cannot be resolved; apply it via Modify")
        cond = self.condition
        if isinstance(cond, Delayable):
            cond = cond._resolve(**kwargs)
        if isinstance(cond, Delayable):
            then = self.then._resolve(**kwargs) if isinstance(self.then, Delayable) else self.then
            else_ = self.else_._resolve(**kwargs) if isinstance(self.else_, Delayable) else self.else_
            return F(lambda c, a, b: a if c else b, cond, then, else_)
        chosen = self.then if cond else self.else_
        if isinstance(chosen, Delayable):
            return chosen._resolve(**kwargs)
        return chosen

    def fae_bind(self, **symbols: Any) -> Any:
        if self._modifier_mode:
            return self

        node: Delayable = self
        if "I" in symbols:
            i_val = symbols["I"]
            p_val = symbols.get("P")

            def visit(item: Any, path: str) -> Any:
                if isinstance(item, DelayedModule):
                    return item._generate(I=i_val, P=p_val)
                return None

            node = node._fae_traverse(visit)
            if not isinstance(node, IF):
                return node.fae_bind(**symbols) if isinstance(node, Delayable) else node

        cond = node.condition
        if isinstance(cond, Delayable):
            cond = cond.fae_bind(**symbols)
        then = node.then.fae_bind(**symbols) if isinstance(node.then, Delayable) else node.then
        else_ = node.else_.fae_bind(**symbols) if isinstance(node.else_, Delayable) else node.else_
        if isinstance(cond, Delayable):
            out = IF(cond, then, else_=else_)
            out.fae_name = node.fae_name
            return out
        chosen = then if cond else else_
        if isinstance(chosen, Delayable) and node.fae_name is not None:
            chosen.fae_name = node.fae_name
        return chosen

    def _fae_exchange(self) -> "IF":
        if self._modifier_mode:
            yield from ()
            return self
        cond = yield self.condition
        then = yield self.then
        else_ = yield self.else_
        out = IF(cond, then, else_=else_)
        out.fae_name = self.fae_name
        return out

    def __repr__(self) -> str:
        if self._modifier_mode:
            return f"IF({self.condition!r}, else_={self.else_!r})"
        return f"IF({self.condition!r}, {self.then!r}, else_={self.else_!r})"


class FVar:
    """Mutable cell for modifier state (activations, KV cache buffers)."""

    def __init__(self, value: Any = None) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"FVar({self.value!r})"


class Record(Modifier):
    """Append each matched node's resolved output into an ``FVar`` list (inspection)."""

    def __init__(self, sink: FVar) -> None:
        if sink.value is None:
            sink.value = []
        self.sink = sink

    def __call__(self, node: Delayable) -> Delayable:
        sink = self.sink

        def _tap(x: Any) -> Any:
            sink.value.append(x)
            return x

        return F(_tap, node) if not isinstance(node, type) else node


class LoRA(Modifier):
    """
    Inject a low-rank residual path beside a linear-like call node:

        node + (X >> A >> B) * scale
    """

    def __init__(self, rank: int = 8, alpha: float = 16.0) -> None:
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

    def __call__(self, node: Delayable) -> Delayable:
        from torch import nn
        from faeyon import faek

        # Best-effort: if node is module(X) with a Linear, attach LoRA beside it.
        module = None
        if isinstance(node, F) and len(node._fae_args) >= 1:
            cand = node._fae_args[0]
            if isinstance(cand, nn.Linear):
                module = cand

        if module is None:
            # Generic residual adapter on the pipeline value.
            with faek:
                adapter = (
                    nn.Linear(self.rank, self.rank, bias=False)  # placeholder; rewritten below
                )
            # Without known in/out features, wrap as additive identity no-op marker.
            return node

        in_f, out_f = module.in_features, module.out_features
        with faek:
            a = nn.Linear(in_f, self.rank, bias=False)
            b = nn.Linear(self.rank, out_f, bias=False)
            nn.init.kaiming_uniform_(a.weight)
            nn.init.zeros_(b.weight)
            for p in module.parameters():
                p.requires_grad = False
            lora_path = (a(X) >> b) * self.scale
            out = node + lora_path
            out.fae_name = node.fae_name
            return out


class Quantize(Modifier):
    """Fake-quant wrapper: round weights to ``bits`` on the matched Linear module."""

    def __init__(self, bits: int = 8) -> None:
        if bits < 2 or bits > 16:
            raise ValueError("bits must be in [2, 16]")
        self.bits = bits

    def __call__(self, node: Delayable) -> Delayable:
        from torch import nn
        import torch

        if not (isinstance(node, F) and node._fae_args and isinstance(node._fae_args[0], nn.Linear)):
            return node

        linear: nn.Linear = node._fae_args[0]
        qmin, qmax = 0, (1 << self.bits) - 1
        w = linear.weight.data
        w_min, w_max = w.min(), w.max()
        scale = (w_max - w_min).clamp_min(1e-8) / qmax
        q = torch.clamp(torch.round((w - w_min) / scale), qmin, qmax)
        linear.weight.data = q * scale + w_min
        linear.weight.requires_grad = False
        return node


class KVCache(Modifier):
    """
    Concatenate projected K/V with a growing cache stored in ``FVar``.

    Apply separately to k_proj and v_proj call sites.
    """

    def __init__(self, cache: FVar) -> None:
        self.cache = cache

    def __call__(self, node: Delayable) -> Delayable:
        cache = self.cache

        def _append(x: Any) -> Any:
            import torch

            if cache.value is None:
                cache.value = x
            else:
                cache.value = torch.cat([cache.value, x], dim=-2)
            return cache.value

        return F(_append, node)
