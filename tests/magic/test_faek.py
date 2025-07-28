"""
Note: Currently faek is automatically enabled when faeyon is imported. 
"""
import pytest
import torch
from torch import nn
from faeyon import faek, FaeArgs


def _is_faek_on():
    model = nn.Linear(10, 10)
    return (
        hasattr(model, "clone")
        and hasattr(model, "__mul__")
        and hasattr(model, "__rmul__")
        and hasattr(model, "__rrshift__")
        and hasattr(model, "_arguments")
    )


def test_faek_on_off():
    """ Test that faek can be enabled and disabled"""
    assert not _is_faek_on()
    faek.on()
    assert _is_faek_on()
    faek.off()
    assert not _is_faek_on()


def test_faek_as_context_manager():
    assert not _is_faek_on()

    with faek:
        assert _is_faek_on()
    
    assert not _is_faek_on()


def test_clone():
    with faek:
        model = nn.Linear(in_features=10, out_features=2)
        cloned_model = model.clone()
        assert model is not cloned_model
        assert model.in_features == cloned_model.in_features
        assert model.out_features == cloned_model.out_features
        

def test_mul():
    with faek:
        model = nn.Linear(in_features=10, out_features=2)
        layers = model * 3

        assert len(layers) == 3
        for layer in layers:
            assert layer.in_features == model.in_features
            assert layer.out_features == model.out_features


def test_rmul():
    """ Multiplication should also work if model is on the right hand side of the * operator. """
    with faek:
        model = nn.Linear(in_features=10, out_features=2)
        layers = 3 * model

        assert len(layers) == 3
        for layer in layers:
            assert layer.in_features == model.in_features
            assert layer.out_features == model.out_features


def test_mul_error():
    """ Layer multiplication should raise error if the multiplier is not a positive integer. """
    with faek:
        model = nn.Linear(in_features=10, out_features=2)
        with pytest.raises(TypeError):
            layers = model * 1.1

        with pytest.raises(ValueError):
            layers = model * -1


def test_rrshift():
    with faek:
        model = nn.Linear(in_features=10, out_features=2)
        x = torch.randn(1, 10)
        y = x >> model
        assert y.shape == (1, 2)


def test_rrshift_with_faeargs():
    """ 
    This is added here for extra assurance that FaeArgs rshift will be called instead of the 
    faek nn.Module rrshift operator. 
    """
    model = nn.Linear(in_features=10, out_features=2)
    x = FaeArgs(torch.randn(1, 10))
    y = x >> model
    assert y.shape == (1, 2)