import torch
from typing import Optional


def is_probability(x: torch.Tensor, dim: Optional[int] = None) -> bool:
    """
    Checks if the tensor contains probabilities. If `dim` is not provided, it just check that all
    values are in [0, 1]. If `dim` is provided, it checks that the sum of the values along 
    the specified dimension is 1.
    """
    if not x.is_floating_point():
        return False

    if x.min() < 0 or x.max() > 1:
        return False
    
    if dim is not None:
        if not x.sum(dim=dim).isclose(torch.tensor(1.0)).all():
            return False

    return True 


def is_binary(x: torch.Tensor) -> bool:
    """
    Checks if the tensor contains integers binary values (0 or 1).
    """
    if x.is_floating_point():
        return False
    
    if x.min() < 0 or x.max() > 1:
        return False
    
    return True


def is_onehot(x: torch.Tensor, dim: int = -1) -> bool:
    """
    Checks if the tensor is one-hot encoded along a specified dimension.
    A tensor is one-hot if, along the given dimension, exactly one element is 1 and the rest are 0.
    """
    if x.is_floating_point():
        return False
    
    if not is_binary(x):
        return False
    
    if (x.sum(dim=dim) != 1).any():
        return False

    return True


def is_inrange(x: torch.Tensor, min: Optional[float] = None, max: Optional[float] = None) -> bool:
    """ Inclusive of the boundaries."""
    if max is not None:
        if x.max() > max:
            return False
    
    if min is not None:
        if x.min() < min:
            return False
    
    return True

