"""
Note: faek is NOT enabled on import; the session fixture in tests/conftest.py turns it
on for the test suite. The on/off/context-manager tests below manage state themselves.
"""
import pytest
import torch

from pytest import param
from torch import nn
from faeyon import faek, F, X, Chain
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
    """Test that faek can be enabled and disabled."""
    assert _is_faek_on()
    faek.on()
    assert _is_faek_on()
    faek.off()
    assert not _is_faek_on()

    # Restore session fixture state for other tests.
    faek.on()


def test_faek_as_context_manager():
    assert _is_faek_on()
    faek.off()
    assert not _is_faek_on()

    with faek:
        assert _is_faek_on()

    assert not _is_faek_on()
    faek.on()


def test_faek_context_manager_reentrant():
    """Exiting a nested/redundant context must restore, not blindly disable."""
    assert _is_faek_on()

    with faek:
        assert _is_faek_on()
        with faek:
            assert _is_faek_on()
        assert _is_faek_on()
    assert _is_faek_on()

    faek.off()
    with faek:
        assert _is_faek_on()
        with faek:
            assert _is_faek_on()
        assert _is_faek_on()
    assert not _is_faek_on()

    faek.on()


def test_module_rshift():
    expr = X + 1 >> nn.Linear(in_features=10, out_features=2)
    assert isinstance(expr, Chain)
    assert len(expr) == 2


def test_module_rrshift():
    expr = nn.Linear(in_features=10, out_features=2) >> X + 1
    assert isinstance(expr, Chain)
    assert len(expr) == 2


def test_new_with_flist():
    pytest.skip("FaeList ctor expansion via nn.Module.__new__ not implemented yet")


def test_new_with_flist_error():
    pytest.skip("FaeList ctor expansion via nn.Module.__new__ not implemented yet")


def test_new_with_fdict():
    pytest.skip("FaeDict ctor expansion via nn.Module.__new__ not implemented yet")


def test_new_with_fdict_error():
    pytest.skip("FaeDict ctor expansion via nn.Module.__new__ not implemented yet")


def test_new_with_dict_list_error():
    pytest.skip("FaeDict/FaeList ctor expansion via nn.Module.__new__ not implemented yet")


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


class TestModuleOperators:
    """Module arithmetic / shift paths patched by faek (left and right forms)."""

    def test_mul_int(self):
        """Multiplication with int creates a list of clones; commutative."""
        model = nn.Linear(in_features=10, out_features=2)
        for layers in (3 * model, model * 3):
            assert len(layers) == 3
            for layer in layers:
                assert layer.in_features == model.in_features
                assert layer.out_features == model.out_features
                assert layer is not model

    def test_mul_int_error(self):
        model = nn.Linear(in_features=10, out_features=2)
        with pytest.raises(TypeError):
            _ = model * 1.1
        with pytest.raises(ValueError):
            _ = model * -1

    def test_rshift_mm(self):
        """module >> module → Chain of module calls."""
        delayed = (
            nn.Linear(in_features=10, out_features=2)
            >> nn.Linear(in_features=2, out_features=2)
        )
        assert isinstance(delayed, Chain)
        assert len(delayed) == 2
        y = torch.randn(1, 10) | delayed
        assert y.shape == (1, 2)

    def test_rshift_mo(self):
        """module >> F → Chain."""
        delayed = ConstantLayer(2, value=2.0) >> (2 * X)
        assert isinstance(delayed, Chain)
        assert len(delayed) == 2
        y = torch.tensor([1.0, 2.0]) | delayed
        torch.testing.assert_close(y, torch.tensor([4.0, 8.0]))

    def test_rrshift_data(self):
        """data >> module evaluates (via Module.__rrshift__)."""
        model = nn.Linear(in_features=10, out_features=2)
        x = torch.randn(1, 10)
        y = x >> model
        assert y.shape == (1, 2)

    def test_rrshift_om(self):
        """F >> module → Chain."""
        delayed = (2 * X) >> ConstantLayer(2, value=2.0)
        assert isinstance(delayed, Chain)
        assert len(delayed) == 2
        y = torch.tensor([1.0, 2.0]) | delayed
        torch.testing.assert_close(y, torch.tensor([4.0, 8.0]))

    @pytest.mark.parametrize("op, expected", [
        ("add", [[2.0, 2.0]]),
        ("sub", [[0.0, 0.0]]),
        ("mul", [[1.0, 1.0]]),
        ("truediv", [[1.0, 1.0]]),
        ("floordiv", [[1.0, 1.0]]),
        ("mod", [[0.0, 0.0]]),
        ("pow", [[1.0, 1.0]]),
    ])
    def test_float_operators_mm(self, op, expected):
        """module ○ module for float arithmetic."""
        x = torch.ones(1, 2)
        layer1 = ConstantLayer((1, 2), value=1.0)
        layer2 = ConstantLayer((1, 2), value=1.0)
        delayed = getattr(layer1, f"__{op}__")(layer2)
        assert isinstance(delayed, F)
        res = x | delayed
        torch.testing.assert_close(res, torch.tensor(expected))

    def test_matmul_mm(self):
        x = torch.ones(2, 1)
        layer1 = ConstantLayer((2, 2), value=1.0)
        layer2 = ConstantLayer((2, 1), value=1.0)
        delayed = layer1 @ layer2
        assert isinstance(delayed, F)
        res = x | delayed
        torch.testing.assert_close(res, 2.0 * torch.ones(2, 1))

    @pytest.mark.parametrize("op, expected", [
        ("and", [[0, 0]]),
        ("xor", [[3, 3]]),
    ])
    def test_bitwise_mm(self, op, expected):
        x = torch.ones(1, 2, dtype=torch.int64)
        layer1 = ConstantLayer((1, 2), value=2, dtype=torch.int64)
        layer2 = ConstantLayer((1, 2), value=1, dtype=torch.int64)
        delayed = getattr(layer1, f"__{op}__")(layer2)
        assert isinstance(delayed, F)
        res = x | delayed
        torch.testing.assert_close(res, torch.tensor(expected))

    @pytest.mark.parametrize("op, expected", [
        ("neg", [[-2, -2]]),
        ("pos", [[2, 2]]),
        ("abs", [[2, 2]]),
        ("invert", [[-3, -3]]),
    ])
    def test_unary_operators(self, op, expected):
        x = torch.ones(1, 2, dtype=torch.int64)
        layer = ConstantLayer((1, 2), value=2, dtype=torch.int64)
        delayed = getattr(layer, f"__{op}__")()
        assert isinstance(delayed, F)
        res = x | delayed
        torch.testing.assert_close(res, torch.tensor(expected))

    @pytest.mark.parametrize("op, expected, data", [
        ("add", [3.0, 6.0], [1.0, 2.0]),
        ("sub", [1.0, 2.0], [1.0, 2.0]),
        ("mul", [2.0, 8.0], [1.0, 2.0]),
        ("truediv", [2.0, 2.0], [1.0, 2.0]),
        ("floordiv", [2.0, 2.0], [1.0, 2.0]),
        ("mod", [0.0, 0.0], [1.0, 2.0]),
        ("pow", [2.0, 16.0], [1.0, 2.0]),
    ])
    def test_binary_operators_mo(self, op, expected, data):
        """``nn.Module`` on the left-hand side."""
        layer = ConstantLayer(2, value=2.0)
        delayed = getattr(layer, f"__{op}__")(X)
        assert isinstance(delayed, F)
        out = torch.tensor(data) | delayed
        torch.testing.assert_close(out, torch.tensor(expected))

    def test_binary_operators_mo_matmul(self):
        delayed = ConstantLayer(2, value=2.0) @ X[None].T
        assert isinstance(delayed, F)
        out = torch.tensor([1.0, 2.0]) | delayed
        torch.testing.assert_close(out, torch.tensor([10.0]))

    @pytest.mark.parametrize("op, expected, data", [
        ("and", [0, 0, 0], [1, 1, 1]),
        ("xor", [3, 3, 3], [1, 1, 1]),
    ])
    def test_binary_operators_mo_bitwise(self, op, expected, data):
        layer = ConstantLayer(3, value=2, dtype=torch.int64)
        delayed = getattr(layer, f"__{op}__")(X)
        assert isinstance(delayed, F)
        out = torch.tensor(data) | delayed
        torch.testing.assert_close(out, torch.tensor(expected))

    @pytest.mark.parametrize("op, expected, data", [
        ("add", [3.0, 6.0], [1.0, 2.0]),
        ("sub", [-1.0, -2.0], [1.0, 2.0]),
        ("mul", [2.0, 8.0], [1.0, 2.0]),
        ("truediv", [0.5, 0.5], [1.0, 2.0]),
        ("floordiv", [0.0, 0.0], [1.0, 2.0]),
        ("mod", [1.0, 2.0], [1.0, 2.0]),
        ("pow", [1.0, 16.0], [1.0, 2.0]),
    ])
    def test_binary_operators_om(self, op, expected, data):
        """``nn.Module`` on the right-hand side."""
        layer = ConstantLayer(2, value=2.0)
        delayed = getattr(X, f"__{op}__")(layer)
        assert isinstance(delayed, F)
        out = torch.tensor(data) | delayed
        torch.testing.assert_close(out, torch.tensor(expected))

    def test_binary_operators_om_matmul(self):
        delayed = X[None] @ ConstantLayer(2, value=2.0)
        assert isinstance(delayed, F)
        out = torch.tensor([1.0, 2.0]) | delayed
        torch.testing.assert_close(out, torch.tensor([10.0]))

    @pytest.mark.parametrize("op, expected, data", [
        ("and", [0, 0, 0], [1, 1, 1]),
        ("xor", [3, 3, 3], [1, 1, 1]),
    ])
    def test_binary_operators_om_bitwise(self, op, expected, data):
        layer = ConstantLayer(3, value=2, dtype=torch.int64)
        delayed = getattr(X, f"__{op}__")(layer)
        assert isinstance(delayed, F)
        out = torch.tensor(data) | delayed
        torch.testing.assert_close(out, torch.tensor(expected))
