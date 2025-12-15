from torch.fx import Proxy, Tracer
from typing import Any, Callable, Optional
from torch.fx.proxy import Argument
from .x import X
from torch.fx.graph import Node
from torch.fx.node import Target
from torch.fx._compatibility import compatibility
from .spells import Delayable, A, Op, binary_operators
import torch
import operator


class FaeTracer(Tracer):
    def create_arg(self, a: Any) -> Argument:
        if isinstance(a, X):
            return a
            
        elif isinstance(a, Delayable):
            return a
        elif isinstance(a, A):
            return a
        return super().create_arg(a)

    def call_module(
        self,
        m: torch.nn.Module,
        forward,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        fargs = A(*args, **kwargs)

        if fargs.is_resolved:
            return super().call_module(m, forward, args, kwargs)
        
        return Op(m, *args, **kwargs)

    @compatibility(is_backward_compatible=True)
    def create_node(
        self, 
        kind: str, 
        target: Target, 
        args: tuple[Argument, ...], 
        kwargs: dict[str, Argument], 
        name: Optional[str] = None, 
        type_expr: Optional[Any] = None
    ) -> Node:
        out =  super().create_node(kind, target, args, kwargs, name, type_expr)
        return out 

    def create_proxy(
        self, 
        kind: str, 
        target: Target, 
        args: tuple[Argument, ...], 
        kwargs: dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Callable[[Node], "Proxy"] = None,
    ) -> Proxy:

        if kind == "call_function":
            if target is operator.rshift:
                if isinstance(args[1], torch.nn.Module):
                    return args[1](args[0])            
                elif isinstance(args[1], A | Delayable):
                    return args[1].using(args[0])
            
            elif target in binary_operators:
                if isinstance(args[1], Delayable):
                    return getattr(args[1], binary_operators[target][1])(args[0])

        return super().create_proxy(kind, target, args, kwargs)
