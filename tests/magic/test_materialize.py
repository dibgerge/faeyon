"""Tests for faeyon.magic.materialize."""

import torch
from torch import nn

from faeyon import X, faek, materialize


class TestMaterialize:
    def test_chain_linear_relu(self):
        with faek:
            expr = nn.Linear(8, 4) >> nn.ReLU() >> nn.Linear(4, 2)
        model = materialize(expr, debug_source=True)
        x = torch.randn(3, 8)
        out = model(x)
        assert out.shape == (3, 2)
        assert isinstance(model, nn.Module)
        assert "def forward" in model._fae_source

    def test_residual_add(self):
        with faek:
            lin = nn.Linear(6, 6)
            expr = X + lin(X)
        model = materialize(expr)
        x = torch.randn(2, 6)
        # Eager linear call (bypass faek patch).
        from faeyon.magic.faek import faek as _faek
        expected = x + _faek.module__call__(lin, x)
        torch.testing.assert_close(model(x), expected)

    def test_named_module_attr(self):
        with faek:
            expr = (nn.Linear(4, 4) % "proj") >> nn.ReLU()
        model = materialize(expr)
        assert "proj" in dict(model.named_children())

    def test_matches_resolve(self):
        with faek:
            expr = nn.Linear(5, 5) >> nn.ReLU()
        model = materialize(expr)
        x = torch.randn(2, 5)
        y_mod = model(x)
        y_res = x | expr
        torch.testing.assert_close(y_mod, y_res)
