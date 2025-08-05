"""
Note: Currently faek is automatically enabled when faeyon is imported. 
"""
import math
import pytest
import torch
from torch import nn
from faeyon import faek, FaeArgs, FaeList, FaeDict, Op, X
from tests.common import ConstantLayer


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


@pytest.mark.parametrize("op,expected", [
    ("add", torch.tensor([[2.0, 2.0]])), 
    ("sub", torch.tensor([[0.0, 0.0]])),
    ("truediv", torch.tensor([[1.0, 1.0]])),
    ("floordiv", torch.tensor([[1.0, 1.0]])),
    ("mod", torch.tensor([[0.0, 0.0]])),
    ("pow", torch.tensor([[1.0, 1.0]])),
])
def test_module_delayed_float_binary(op, expected):
    x = torch.ones(1, 2)
    with faek:
        layer1 = ConstantLayer((1, 2), value=1.0)
        layer2 = ConstantLayer((1, 2), value=1.0)
        delayed = getattr(layer1, f"__{op}__")(layer2)
        assert isinstance(delayed, Op)
        res = x >> delayed

    torch.testing.assert_close(res, expected)


def test_module_delayed_matmul():
    x = torch.ones(2, 1)
    with faek:
        layer1 = ConstantLayer((2, 2), value=1.0)
        layer2 = ConstantLayer((2, 1), value=1.0)
        delayed = layer1 @ layer2
        assert isinstance(delayed, Op)
        res = x >> delayed
    expected = 2.0 * torch.ones(2, 1)
    torch.testing.assert_close(res, expected)


@pytest.mark.parametrize("op,expected", [
    ("and", torch.tensor([[0, 0]])),
    ("or", torch.tensor([[3, 3]])),
    ("xor", torch.tensor([[3, 3]])),
])
def test_module_delayed_bitwise(op, expected):
    x = torch.ones(1, 2, dtype=torch.int64)
    with faek:
        layer1 = ConstantLayer((1, 2), value=2, dtype=torch.int64)
        layer2 = ConstantLayer((1, 2), value=1, dtype=torch.int64)
        delayed = getattr(layer1, f"__{op}__")(layer2)
        assert isinstance(delayed, Op)
        res = x >> delayed
    torch.testing.assert_close(res, expected)


@pytest.mark.parametrize("op,expected", [
    ("neg", torch.tensor([[-2.1, -2.1]])),
    ("pos", torch.tensor([[2.1, 2.1]])),
    ("abs", torch.tensor([[2.1, 2.1]]))
])
def test_module_delayed_unary(op, expected):
    x = torch.ones(1, 2)
    with faek:
        layer = ConstantLayer((1, 2), value=2.1)
        delayed = getattr(layer, f"__{op}__")()
        assert isinstance(delayed, Op)
        res = x >> delayed
    torch.testing.assert_close(res, expected)


def test_module_delayed_invert():
    x = torch.ones(1, 2, dtype=torch.int64)
    with faek:
        layer = ConstantLayer((1, 2), value=2, dtype=torch.int64)
        delayed = ~layer
        assert isinstance(delayed, Op)
        res = x >> delayed
    torch.testing.assert_close(res, torch.tensor([[-3, -3]]))
