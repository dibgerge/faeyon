import abc

from typing import Any, Optional
from faeyon.magic.spells import Delayable
from faeyon import FVar, X


class Modify:
    """
    Used in conjunction with modifiers to specify where at in the expression tree to 
    apply the modifiers.
    """
    def __init__(self, lookup: str | type[Delayable], *modifiers: Modifier) -> None:
        self.lookup = lookup
        self.modifiers = modifiers

    def __rmod__(self, root: Delayable) -> Delayable:        
        def callback(node: Delayable) -> Delayable:
            return node.fae.append(*self.modifiers)

        if not isinstance(root, Delayable):
            return NotImplemented
        return root.fae.find(self.lookup, callback=callback)


class Modifier(abc.ABC):
    """ 
    Base class for modifiers.

    # TODO: make this a protocol
    """
    @abc.abstractmethod
    def on_init(self, node: Delayable) -> None:
        pass

    @abc.abstractmethod
    def on_build(self) -> Any:
        """
        Logic used to specify the new expression to be used.
        """

    @abc.abstractmethod
    def on_resolve(self, result: Any) -> None:
        """
        Called at resolve time, which the modify can cache or use the result of the node.
        """


class Record:
    node: Delayable

    def __init__(self, output: FVar = None) -> None:
        self.output = output

    def on_init(self, node: Delayable) -> None:
        self.node = node

    def on_build(self) -> Any:
        return self.node

    def on_resolve(self, result: Any) -> None:
        if self.output is not None:
            # TODO: need to update FVar interface
            self.output.value = result
        
        self.node.fae.cache = result


class IF:
    # TODO: this does not work
    def __init__(
        self, 
        condition: bool | Delayable,
        else_: Optional[Delayable] = None
    ) -> None:
        self.condition = condition
        self.else_ = else_

    def on_init(self, node: Delayable) -> None:
        self.node = node

    def on_build(self) -> Any:
        if isinstance(self.condition, bool):
            if self.condition:
                return self.node
            else:
                if self.else_ is not None:  
                    return self.else_
                else:
                    return X
        elif isinstance(self.condition, Delayable):
            return 
        else:
            raise ValueError(f"Invalid condition type: {type(self.condition)}")

    def on_resolve(self, result: Any) -> None:
        pass


