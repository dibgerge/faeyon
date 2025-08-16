"""
Note: Currently faek is automatically enabled when faeyon is imported. 
"""
import math
import pytest
import torch
from torch import nn
from faeyon import faek, A, FList, FDict, Op, X
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
    assert _is_faek_on()
    faek.on()
    assert _is_faek_on()
    faek.off()
    assert not _is_faek_on()

    # I need to turn it on to remove the side effects to other tests
    faek.on()


def test_faek_as_context_manager():
    assert _is_faek_on()
    faek.off()
    assert not _is_faek_on()

    with faek:
        assert _is_faek_on()
    
    assert not _is_faek_on()

    # I need to turn it on to remove the side effects to other tests
    faek.on()


def test_new_with_flist():
    out_features = [1, 2, 3]
    in_features = [[10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 11, 12]]
    models = [
        nn.Linear(in_features=10, out_features=FList(1, 2, 3)),
        nn.Linear(10, FList(1, 2, 3)),
        nn.Linear(10, out_features=FList(1, 2, 3)),
        nn.Linear(FList(10, 11, 12), FList(1, 2, 3)),
    ]

    for expected_in_feats, model in zip(in_features, models):
        assert isinstance(model, list)
        assert len(model) == 3

        for i, (expected_in, layer) in enumerate(zip(expected_in_feats, model)):
            assert layer.in_features == expected_in
            assert layer.out_features == out_features[i]


def test_new_with_flist_error():
    """ Cannot have parametrized arguments with different lengths. """
    with pytest.raises(ValueError):
        model = nn.Linear(in_features=FList(10, 11), out_features=FList(1, 2, 3))


def test_new_with_fdict():
    out_features = {"a": 1, "b": 2, "c": 3}
    model = nn.Linear(in_features=10, out_features=FDict(**out_features))
    assert isinstance(model, dict)
    assert len(model) == 3
    assert set(model.keys()) == set(out_features.keys())

    for k, layer in model.items():
        assert layer.in_features == 10
        assert layer.out_features == out_features[k]


def test_new_with_fdict_error():
    """ Cannot have parametrized arguments with different keys. """
    with pytest.raises(ValueError):
        model = nn.Linear(
            in_features=FDict(a=10, b=11, d=12),
            out_features=FDict(a=1, b=2, c=3)
        )


def test_new_with_dict_list_error():
    """ Cannot have parametrized with mixed FDict/FList arguments. """
    with pytest.raises(ValueError):
        model = nn.Linear(
            in_features=FDict(a=10, b=11, d=12),
            out_features=FList(1, 2, 3)
        )


def test_mul():
    model = nn.Linear(in_features=10, out_features=2)
    layers = model * 3

    assert len(layers) == 3
    for layer in layers:
        assert layer.in_features == model.in_features
        assert layer.out_features == model.out_features


def test_rmul():
    """ Multiplication should also work if model is on the right hand side of the * operator. """
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
    model = nn.Linear(in_features=10, out_features=2)
    cloned_model = model.clone(*args, **kwargs)
    assert model is not cloned_model
    assert cloned_model.in_features == expected_in_features
    assert cloned_model.out_features == expected_out_features


def test_mul_error():
    """ Layer multiplication should raise error if the multiplier is not a positive integer. """
    model = nn.Linear(in_features=10, out_features=2)
    with pytest.raises(TypeError):
        layers = model * 1.1

    with pytest.raises(ValueError):
        layers = model * -1


def test_rrshift():
    model = nn.Linear(in_features=10, out_features=2)
    x = torch.randn(1, 10)
    y = x >> model
    assert y.shape == (1, 2)


def test_rrshift_with_faeargs():
    """ 
    This is added here for extra assurance that `A` rshift will be called instead of the 
    faek nn.Module rrshift operator. 
    """
    model = nn.Linear(in_features=10, out_features=2)
    x = A(torch.randn(1, 10))
    y = x >> model
    assert y.shape == (1, 2)


@pytest.mark.parametrize("op,expected", [
    ("add", [[2.0, 2.0]]), 
    ("sub", [[0.0, 0.0]]),
    ("mul", [[1.0, 1.0]]),
    ("truediv", [[1.0, 1.0]]),
    ("floordiv", [[1.0, 1.0]]),
    ("mod", [[0.0, 0.0]]),
    ("pow", [[1.0, 1.0]]),
])
def test_module_delayed_float_binary(op, expected):
    """
    layer1(x) = [1.0, 1.0]
    layer2(x) = [1.0, 1.0]
    layer1(x) + layer2(x) = [2.0, 2.0]
    layer1(x) - layer2(x) = [0.0, 0.0]
    layer1(x) * layer2(x) = [1.0, 1.0]
    layer1(x) / layer2(x) = [1.0, 1.0]
    layer1(x) // layer2(x) = [1.0, 1.0]
    layer1(x) % layer2(x) = [0.0, 0.0]
    layer1(x) ** layer2(x) = [1.0, 1.0]
    """
    x = torch.ones(1, 2)
    layer1 = ConstantLayer((1, 2), value=1.0)
    layer2 = ConstantLayer((1, 2), value=1.0)
    delayed = getattr(layer1, f"__{op}__")(layer2)
    assert isinstance(delayed, Op)
    res = x >> delayed

    torch.testing.assert_close(res, torch.tensor(expected))


def test_module_delayed_matmul():
    x = torch.ones(2, 1)
    layer1 = ConstantLayer((2, 2), value=1.0)
    layer2 = ConstantLayer((2, 1), value=1.0)
    delayed = layer1 @ layer2
    assert isinstance(delayed, Op)
    res = x >> delayed
    expected = 2.0 * torch.ones(2, 1)
    torch.testing.assert_close(res, expected)


@pytest.mark.parametrize("op,expected", [
    ("and", [[0, 0]]),
    ("or", [[3, 3]]),
    ("xor", [[3, 3]]),
])
def test_module_delayed_bitwise(op, expected):
    x = torch.ones(1, 2, dtype=torch.int64)
    layer1 = ConstantLayer((1, 2), value=2, dtype=torch.int64)
    layer2 = ConstantLayer((1, 2), value=1, dtype=torch.int64)
    delayed = getattr(layer1, f"__{op}__")(layer2)
    assert isinstance(delayed, Op)
    res = x >> delayed
    torch.testing.assert_close(res, torch.tensor(expected))


@pytest.mark.parametrize("op,expected", [
    ("neg", [[-2, -2]]),
    ("pos", [[2, 2]]),
    ("abs", [[2, 2]]),
    ("invert", [[-3, -3]]),
])
def test_module_delayed_unary(op, expected):
    x = torch.ones(1, 2, dtype=torch.int64)
    layer = ConstantLayer((1, 2), value=2, dtype=torch.int64)
    delayed = getattr(layer, f"__{op}__")()
    assert isinstance(delayed, Op)
    res = x >> delayed
    torch.testing.assert_close(res, torch.tensor(expected))


class ModelWithFstate(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=5, out_features=4)
        self.layer2 = nn.Linear(in_features=4, out_features=4)
        self.layer3 = nn.Linear(in_features=4, out_features=4)
        self.layer4 = nn.Linear(in_features=4, out_features=2)

    def forward(self, x):
        return (
            self.layer1(x) 
            >> self.fstate.Y 
            >> self.layer2 
            >> self.fstate.Y 
            >> self.layer3
            >> self.fstate.Z["foo"]
            >> self.layer4
        )


def test_fstate():
    model = ModelWithFstate()
    x = torch.randn(1, 5)
    y = model(x)
    assert y.shape == (1, 2)
    assert isinstance(model.fstate.Y, FList)
    assert isinstance(model.fstate.Z, FDict)


def test_fstate_collect():
    model = ModelWithFstate()    
    x = torch.randn(1, 5)
    y = model(x)

    fstates = model.fstate.collect()
    assert isinstance(fstates, dict)
    assert set(fstates.keys()) == {"Y", "Z"}


def test_fstate_reset():
    model = ModelWithFstate()    
    x = torch.randn(1, 5)
    y1 = model(x)
    fstates1 = model.fstate.collect()

    x = torch.randn(1, 5)
    y2 = model(x)
    fstates2 = model.fstate.collect()

    for item1, item2 in zip(fstates1["Y"], fstates2["Y"]):
        assert not torch.isclose(item1, item2).all()
    assert not torch.isclose(fstates2["Z"]["foo"], fstates1["Z"]["foo"]).all()
