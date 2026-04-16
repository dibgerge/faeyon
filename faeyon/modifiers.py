import abc

from typing import Optional
from faeyon.magic.spells import Delayable, X


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

        return root.fae.find(self.lookup, callback=callback)


class IF(Modifier):
    """
    Conditionally includes or replaces a node.

    - If `condition` is True:  the node is kept as-is.
    - If `condition` is False: the node is replaced with `else_` (or X if not provided).
    - If `condition` is a Delayable: the replacement is deferred to resolve time via
      a conditional F expression.

    Example:
        model % Modify("encoder.dropout", IF(training, else_=X))
    """
    def __init__(
        self,
        condition: bool | Delayable,
        else_: Optional[Delayable] = None,
    ) -> None:
        self.condition = condition
        self.else_ = else_

    def __call__(self, node: Delayable) -> Delayable:
        from faeyon.magic.spells import F

        if isinstance(self.condition, bool):
            if self.condition:
                return node
            else:
                return self.else_ if self.else_ is not None else X
        elif isinstance(self.condition, Delayable):
            else_ = self.else_ if self.else_ is not None else X
            return F(lambda cond, a, b: a if cond else b, self.condition, node, else_)
        else:
            raise ValueError(f"Invalid condition type: {type(self.condition)}")
