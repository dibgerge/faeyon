"""
Note: faek is NOT enabled on import; the function-scoped autouse fixture in
tests/conftest.py turns it on before each test (and off after). Tests below may
toggle faek freely without restoring for siblings.
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


def test_faek_is_on():
    assert faek.is_on


def test_faek_off_off():
    faek.off()
    assert not faek.is_on
    faek.off()
    assert not faek.is_on


def test_faek_on_off():
    """Test that faek can be enabled and disabled."""
    assert _is_faek_on()
    faek.on()
    assert _is_faek_on()
    faek.off()
    assert not _is_faek_on()


def test_faek_as_context_manager():
    assert _is_faek_on()
    faek.off()
    assert not _is_faek_on()

    with faek:
        assert _is_faek_on()

    assert not _is_faek_on()


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


def test_module_rshift():
    expr = X + 1 >> nn.Linear(in_features=10, out_features=2)
    assert isinstance(expr, Chain)
    assert len(expr) == 2


def test_module_rshift_error():
    with pytest.raises(TypeError):
        nn.Linear(in_features=10, out_features=2) >> 1.0


def test_module_rrshift():
    expr = nn.Linear(in_features=10, out_features=2) >> X + 1
    assert isinstance(expr, Chain)
    assert len(expr) == 2


def test_module_rrshift_error():
    with pytest.raises(TypeError):
        1.0 >> nn.Linear(in_features=10, out_features=2)


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

    # def test_mul_int_is_arithmetic_not_clone(self):
    #     """``module * n`` scales ``module(X)``; cloning is ``module >> n``."""
    #     model = ConstantLayer(2, value=2.0)
    #     delayed = model * 3
    #     assert isinstance(delayed, F)
    #     y = torch.tensor([1.0, 2.0]) | delayed
    #     torch.testing.assert_close(y, torch.tensor([6.0, 12.0]))

    def test_rshift_int_clones(self):
        model = nn.Linear(in_features=4, out_features=4)
        chain = model >> 3
        assert isinstance(chain, Chain)
        assert len(chain) == 3

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

    def test_rrshift_om(self):
        """F >> module → Chain."""
        delayed = (2 * X) >> ConstantLayer(2, value=2.0)
        assert isinstance(delayed, Chain)
        assert len(delayed) == 2
        y = torch.tensor([1.0, 2.0]) | delayed
        torch.testing.assert_close(y, torch.tensor([4.0, 8.0]))

    def test_ror_data(self):
        """data | module evaluates (via Module.__ror__)."""
        model = nn.Linear(in_features=10, out_features=2)
        x = torch.randn(1, 10)
        y = x | model
        assert y.shape == (1, 2)

    def test_lshift_unsupported_like_f(self):
        """Modules lift to ``F``; structural ``<<`` is FaeList/FaeDict-only."""
        m1 = nn.Linear(in_features=10, out_features=2)
        m2 = nn.Linear(in_features=2, out_features=3)
        with pytest.raises(TypeError):
            _ = m1 << m2
        with pytest.raises(TypeError):
            _ = m1 << (2 * X)
        with pytest.raises(TypeError):
            _ = (2 * X) << m2

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
