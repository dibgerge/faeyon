import sys
import inspect
import itertools

from torch import nn
from typing import Any, overload
from collections.abc import Callable

from .spells import Op, FList, FDict, A, X


class FState:
    def __init__(self):
        self._states = {}
    
    def __getattr__(self, name):
        pass
        

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
    
    super(cls, instance).__setattr__("fstate", FState())
    super(cls, instance).__setattr__("_arguments", bound)
    return instance


def __new__(cls, *args, **kwargs):
    """
    Allow `nn.Module` to save constructor arguments passed to it, so that the could be used 
    later for cloning modules.

    When any of the arguments is of type `FList` or `FDict`, special handing is applied to 
    generate clones.
    """
    try:
        kwargs_keys, kwargs_values = zip(*kwargs.items())
    except ValueError:
        kwargs_keys, kwargs_values = [], []
    
    raveled_args = []
    num_faelist = 0
    num_faedict = 0
    fae_keys = None
    fae_len = None
    for arg in itertools.chain(args, kwargs_values):
        if isinstance(arg, FList):
            num_faelist += 1
            arg_value = arg.shed()

            if fae_len is None:
                fae_len = len(arg_value)
            elif fae_len != len(arg_value):
                raise ValueError("All arguments of type `FList` must have the same length.")

            raveled_args.append(arg_value)
        elif isinstance(arg, FDict):
            num_faedict += 1
            arg_value = arg.shed()

            if fae_keys is None:
                fae_keys = list(arg_value.keys())
                fae_len = len(fae_keys)
            else:
                if set(fae_keys) != set(arg_value.keys()):
                    raise ValueError("All arguments of type `FDict` must have the same keys.")

            raveled_args.append([arg_value[k] for k in fae_keys])
        else:
            raveled_args.append(itertools.repeat(arg))

    if num_faelist > 0 and num_faedict > 0:
        raise ValueError("Cannot mix `FList` and `FDict` arguments. Choose one.")

    num_fae = max(num_faelist, num_faedict)

    # No argument parametrization, return a regular nn.Module instance
    if num_fae == 0:
        return _new_instance(cls, *args, **kwargs)

    nkeys = len(kwargs_keys)
    nargs = len(raveled_args)
    args = [val[:nargs-nkeys] for val in zip(*raveled_args)]
    kwargs = [dict(zip(kwargs_keys, val[nargs-nkeys:])) for val in zip(*raveled_args)]

    out = []
    for c_args, c_kwargs in zip(args, kwargs):
        inst = _new_instance(cls, *c_args, **c_kwargs)
        # Call __init__ on each instance since it will not be called when different class type is 
        # returned in __new__
        cls.__init__(inst, *c_args, **c_kwargs)
        out.append(inst)
    
    if fae_keys is not None:
        out = dict(zip(fae_keys, out))

    return out


def __default_new__(cls, *args, **kwargs):
    """ Once we override __new__ in `nn.Module`, we cannot restore the old one, since nn.Module 
    (as of PyTorch 2.7) does not implement `__new__`, and hence expect it to have no arguments. 
    The custom __new__ method we implemented above does not match this signature, and hence 
    we cannot restore the old one. As a workaround, we define a default __new__ method that 
    matches the signature of the default __new__ method in `nn.Module`, but calls the parent object 
    without any arguments.
    See: https://stackoverflow.com/questions/79716674/why-does-monkey-patching-a-classs-new-not-always-work/79717493#79717493
    """
    return object.__new__(cls)


@overload
def __mul__[T: nn.Module](self: T, other: int) -> list[T]: ...

@overload
def __mul__[T: nn.Module](self: T, other: nn.Module) -> Op: ...


def __mul__[T: nn.Module](self: T, other: int | nn.Module) -> list[T] | Op:
    """
    Creates a ModuleList of `other` clones of this module.
    """
    if isinstance(other, nn.Module):
        return getattr(self(X), "__mul__")(other(X))

    if not isinstance(other, int):
        return NotImplemented
    
    if other < 1:
        raise ValueError("Number of modules must be greater than 0.")

    return [self.clone() for _ in range(other)]


@overload
def __rmul__[T: nn.Module](self: T, other: int) -> list[T]: ...


@overload
def __rmul__[T: nn.Module](self: T, other: nn.Module) -> Op: ...


def __rmul__[T: nn.Module](self: T, other: int | nn.Module) -> list[T] | Op:
    """ Multiplication is commutative!"""
    return self.__mul__(other)  # type: ignore


def __rrshift__[T: nn.Module](self: T, other: Any) -> Any:
    """
    This is an alias for `__call__`. The limitation here is that it only works for 
    single inputs. If you need to pass multiple inputs, use the `A` class.
    """
    return self(other)


def clone[T: nn.Module](self: T, *args: Any, **kwargs: Any) -> T:
    """
    Create a new instance of the same module with the same arguments. This method should be used 
    carefully, since it is does not do any deep copying on all types of module arguments. 

    The module is cloned based on the arguments passed to its constructor during its creation. 
    If any of the arguments were changed after the module was created, the changes will not be 
    reflected in the cloned module unless the changes were made on the arguments itslef inplace 
    causing its mutation. E.g. passing a list to the current module and then mutating that same list
    outside the module...

    If you need to clone a module with a argument which should be a new object rather than a shared
    object, you can pass a copy of the argument to the clone method with the new object to use.    
    """
    cls = self.__class__
    sig = inspect.signature(self.__init__)  # type: ignore
    bound = sig.bind_partial(*args, **kwargs)
    cur_arguments = dict(self._arguments.arguments)
    cur_arguments.update(bound.arguments)
    new_bound = inspect.BoundArguments(sig, cur_arguments)  # type: ignore[arg-type]
    return cls(*new_bound.args, **new_bound.kwargs)


def __call__(self, *args, **kwargs):
    fargs = A(*args, **kwargs)
    if fargs.is_resolved:
        return faek.module__call__(self, *args, **kwargs)
    
    return Op(faek.module__call__, self, *args, **kwargs)

    
def delayed_unary_method[T: nn.Module](op_name: str) -> Callable[[T], Op]:
    def func(self: T) -> Op:
        return getattr(self(X), op_name)()
    return func


def delayed_binary_method[T: nn.Module](op_name: str) -> Callable[[T, nn.Module], Op]:
    """
    This method only handles arithmetic on two modules, e.g. module1 + module2. Thus we expect
    to implement only the left versions of the operators. If that failed, will manually call the 
    left type 

    """
    def func(self: T, other: nn.Module) -> Op:
        if isinstance(other, nn.Module):
            return getattr(self(X), op_name)(other(X))
        
        return NotImplemented
    return func


class Singleton(type):
    """
    This is a singleton metaclass intended to be used as a metaclass for the `FaeMagic` class, 
    so that we cannot create multiple instances of `FaeMagic`.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
        return self.__instance


class Faek(metaclass=Singleton):
    """
    This is a singleton class intended to be used as a context manager or as a general tool
    to enable the `ModuleMixin` functionality by Monkey patching the `nn.Module` in PyTorch.
    """

    methods = [
        "__call__",
        "__mul__",
        "__rmul__",
        "__rrshift__",
        "clone",
    ]

    # Note: `__mul__` is a special case, since it can operate on integers and module types.
    delayed_binary_methods = [
        "__rshift__",
        "__add__",
        "__sub__",
        "__matmul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__and__",
        "__or__",
        "__xor__",
    ]
    delayed_unary_methods = [
        "__neg__",
        "__pos__",
        "__abs__",
        "__invert__"
    ]

    def __init__(self):
        self._is_on = False
        self.module__call__ = nn.Module.__call__

    def __enter__(self):
        self.on()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.off()

    def on(self):
        if self._is_on:
            return

        current_module = sys.modules[__name__]
        for method in self.methods:
            setattr(nn.Module, method, getattr(current_module, method))
        
        nn.Module.__new__ = staticmethod(__new__)
        for method in Faek.delayed_binary_methods:
            setattr(nn.Module, method, delayed_binary_method(method))
        for method in Faek.delayed_unary_methods:
            setattr(nn.Module, method, delayed_unary_method(method))

        self._is_on = True

    def off(self):
        if not self._is_on:
            return
        
        for method in self.methods:
            delattr(nn.Module, method)

        nn.Module.__new__ = staticmethod(__default_new__)
        nn.Module.__call__ = self.module__call__
        self._is_on = False


faek = Faek()
