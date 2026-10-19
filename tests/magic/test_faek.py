"""
Note: faek is NOT enabled on import; the function-scoped autouse fixture in
tests/conftest.py turns it on before each test (and off after). Tests below may
toggle faek freely without restoring for siblings.
"""
import pytest
import torch

from torch import nn
from faeyon import faek, F, X, Chain
from faeyon.magic.spells import DelayedModule, F
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
    def test_module_ctor_recipe(self):
        """Delayable ctor args → ``DelayedModule``; ``>>`` promotes to ``module(X)``."""
        from faeyon import faek

        expr = nn.Linear(in_features=X, out_features=2)
        assert isinstance(expr, DelayedModule) and expr._fae_op is nn.Linear
        assert expr.fae_name == "Linear"

        chained = expr >> nn.ReLU()
        assert isinstance(chained, Chain)
        left = chained._fae_ops[0]
        assert left._fae_op is faek.module__call__
        assert isinstance(left._fae_args[0], DelayedModule)

    def test_or_reserved_for_resolve(self):
        """``|`` is resolve-only; Delayable on the left is rejected."""
        with pytest.raises(TypeError, match="reserved for resolve"):
            _ = (X + 1) | (X + 2)
        with pytest.raises(TypeError, match="tensor on the right"):
            _ = X | torch.tensor([1, 2])

    def test_ror_data(self):
        """ Data | Module evaluates (via Module.__ror__). """
        result = torch.tensor([1.0, 2.0]) | ConstantLayer(2, value=2.0)
        torch.testing.assert_close(result, torch.tensor([2.0, 4.0]))

    def test_rshift_int_clones(self):
        model = nn.Linear(in_features=4, out_features=4)
        chain = model >> 3
        assert isinstance(chain, Chain)
        assert len(chain) == 3
        assert chain._fae_ops[0] is not chain._fae_ops[1]
        assert chain._fae_ops[1] is not chain._fae_ops[2]
        assert chain._fae_ops[0] is not model

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

    def test_float_operators_mm(self):
        """module ○ module for float arithmetic."""
        x = torch.ones(1, 2)
        layer1 = ConstantLayer((1, 2), value=1.0)
        layer2 = ConstantLayer((1, 2), value=1.0)

        cases = [
            (layer1 + layer2, [[2.0, 2.0]]),
            (layer1 - layer2, [[0.0, 0.0]]),
            (layer1 * layer2, [[1.0, 1.0]]),
            (layer1 / layer2, [[1.0, 1.0]]),
            (layer1 // layer2, [[1.0, 1.0]]),
            (layer1 % layer2, [[0.0, 0.0]]),
            (layer1 ** layer2, [[1.0, 1.0]]),
        ]
        for delayed, expected in cases:
            assert isinstance(delayed, F)
            torch.testing.assert_close(x | delayed, torch.tensor(expected))

    def test_matmul_mm(self):
        x = torch.ones(2, 1)
        layer1 = ConstantLayer((2, 2), value=1.0)
        layer2 = ConstantLayer((2, 1), value=1.0)
        delayed = layer1 @ layer2
        assert isinstance(delayed, F)
        res = x | delayed
        torch.testing.assert_close(res, 2.0 * torch.ones(2, 1))

    def test_bitwise_mm(self):
        x = torch.ones(1, 2, dtype=torch.int64)
        layer1 = ConstantLayer((1, 2), value=2, dtype=torch.int64)
        layer2 = ConstantLayer((1, 2), value=1, dtype=torch.int64)

        cases = [
            (layer1 & layer2, [[0, 0]]),
            (layer1 ^ layer2, [[3, 3]]),
            (torch.bitwise_or(layer1(X), layer2(X)), [[3, 3]]),
        ]
        for delayed, expected in cases:
            assert isinstance(delayed, F)
            torch.testing.assert_close(x | delayed, torch.tensor(expected))

    def test_unary_operators(self):
        x = torch.ones(1, 2, dtype=torch.int64)
        layer = ConstantLayer((1, 2), value=2, dtype=torch.int64)

        cases = [
            (-layer, [[-2, -2]]),
            (+layer, [[2, 2]]),
            (abs(layer), [[2, 2]]),
            (~layer, [[-3, -3]]),
        ]
        for delayed, expected in cases:
            assert isinstance(delayed, F)
            torch.testing.assert_close(x | delayed, torch.tensor(expected))

    def test_binary_operators_mo(self):
        """``nn.Module`` on the left-hand side."""
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])

        cases = [
            (layer + X, [3.0, 6.0]),
            (layer - X, [1.0, 2.0]),
            (layer * X, [2.0, 8.0]),
            (layer / X, [2.0, 2.0]),
            (layer // X, [2.0, 2.0]),
            (layer % X, [0.0, 0.0]),
            (layer ** X, [2.0, 16.0]),
        ]
        for delayed, expected in cases:
            assert isinstance(delayed, F)
            torch.testing.assert_close(data | delayed, torch.tensor(expected))

    def test_binary_operators_mo_matmul(self):
        delayed = ConstantLayer(2, value=2.0) @ X[None].T
        assert isinstance(delayed, F)
        out = torch.tensor([1.0, 2.0]) | delayed
        torch.testing.assert_close(out, torch.tensor([10.0]))

    def test_binary_operators_mo_bitwise(self):
        layer = ConstantLayer(3, value=2, dtype=torch.int64)
        data = torch.tensor([1, 1, 1])

        cases = [
            (layer & X, [0, 0, 0]),
            (layer ^ X, [3, 3, 3]),
            (torch.bitwise_or(layer(X), X), [3, 3, 3]),
        ]
        for delayed, expected in cases:
            assert isinstance(delayed, F)
            torch.testing.assert_close(data | delayed, torch.tensor(expected))

    def test_binary_operators_om(self):
        """``nn.Module`` on the right-hand side."""
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])

        cases = [
            (X + layer, [3.0, 6.0]),
            (X - layer, [-1.0, -2.0]),
            (X * layer, [2.0, 8.0]),
            (X / layer, [0.5, 0.5]),
            (X // layer, [0.0, 0.0]),
            (X % layer, [1.0, 2.0]),
            (X ** layer, [1.0, 16.0]),
        ]
        for delayed, expected in cases:
            assert isinstance(delayed, F)
            torch.testing.assert_close(data | delayed, torch.tensor(expected))

    def test_binary_operators_om_matmul(self):
        delayed = X[None] @ ConstantLayer(2, value=2.0)
        assert isinstance(delayed, F)
        out = torch.tensor([1.0, 2.0]) | delayed
        torch.testing.assert_close(out, torch.tensor([10.0]))

    def test_binary_operators_om_bitwise(self):
        layer = ConstantLayer(3, value=2, dtype=torch.int64)
        data = torch.tensor([1, 1, 1])

        cases = [
            (X & layer, [0, 0, 0]),
            (X ^ layer, [3, 3, 3]),
            (torch.bitwise_or(X, layer(X)), [3, 3, 3]),
        ]
        for delayed, expected in cases:
            assert isinstance(delayed, F)
            torch.testing.assert_close(data | delayed, torch.tensor(expected))
