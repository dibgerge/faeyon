"""
Note: Currently faek is automatically enabled when faeyon is imported. 
"""
import pytest
import torch
from torch import nn
from faeyon import faek, FaeArgs, FaeList, FaeDict


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


def test_new_with_faelist():
    with faek:
        out_features = [1, 2, 3]

        models = [
            nn.Linear(in_features=10, out_features=FaeList(1, 2, 3)),
            nn.Linear(10, FaeList(1, 2, 3)),
            nn.Linear(10, out_features=FaeList(1, 2, 3)),
        ]

        for model in models:
            assert isinstance(model, list)
            assert len(model) == 3

            for i, layer in enumerate(model):
                assert layer.in_features == 10
                assert layer.out_features == out_features[i]


def test_new_with_faedict():
    with faek:
        out_features = {"a": 1, "b": 2, "c": 3}
        model = nn.Linear(in_features=10, out_features=FaeDict(**out_features))
        assert isinstance(model, dict)
        assert len(model) == 3
        assert set(model.keys()) == set(out_features.keys())

        for k, layer in model.items():
            assert layer.in_features == 10
            assert layer.out_features == out_features[k]


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


@pytest.mark.parametrize("args,kwargs,expected_in_features,expected_out_features", [
    ([], {}, 10, 2),
    ([], {"in_features": 20}, 20, 2),
    ([20], {}, 20, 2),
    ([20], {"out_features": 5}, 20, 5),
])
def test_clone(args, kwargs, expected_in_features, expected_out_features):
    with faek:
        model = nn.Linear(in_features=10, out_features=2)
        cloned_model = model.clone(*args, **kwargs)
        assert model is not cloned_model
        assert cloned_model.in_features == expected_in_features
        assert cloned_model.out_features == expected_out_features


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