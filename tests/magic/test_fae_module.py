import sys
import torch
import pytest
from torch import nn
from faeyon import R, X, faek, lower
from faeyon.magic.fae_module import FaeModule

_compile_skip = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="torch.compile not supported on Python 3.14+",
)


class TestFaeModule:
    def test_forward_simple(self):
        chain = nn.Linear(10, 5) >> nn.ReLU() >> nn.Linear(5, 2)
        model = FaeModule(chain)
        x = torch.randn(3, 10)
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([3, 2])

    def test_sub_modules_registered(self):
        linear1, linear2 = nn.Linear(10, 5), nn.Linear(5, 2)
        chain = linear1 >> linear2
        model = FaeModule(chain)
        registered = dict(model.named_children())
        assert linear1 in registered.values()
        assert linear2 in registered.values()

    def test_parameters_tracked(self):
        chain = nn.Linear(10, 5) >> nn.Linear(5, 2)
        model = FaeModule(chain)
        expected = (10 * 5 + 5) + (5 * 2 + 2)
        actual = sum(p.numel() for p in model.parameters())
        assert actual == expected

    def test_forward_residual(self):
        linear = nn.Linear(10, 10)
        chain = X + linear(X)
        model = FaeModule(chain)
        x = torch.randn(3, 10)
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([3, 10])

    def test_forward_named_chain(self):
        chain = (nn.Linear(10, 5) >> nn.ReLU()) % "encoder"
        model = FaeModule(chain)
        x = torch.randn(3, 10)
        out = model(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([3, 5])

    def test_forward_recall_skip(self):
        """R["name"] recalls a named module's output as a long-range skip."""
        lin = nn.Linear(10, 10)
        chain = (lin % "skip") >> nn.ReLU() >> X + R["skip"]
        model = FaeModule(chain)
        x = torch.randn(3, 10)
        out = model(x)
        # Compute the expectation with plain functions (unaffected by faek's patch).
        h = torch.nn.functional.linear(x, lin.weight, lin.bias)
        torch.testing.assert_close(out, torch.relu(h) + h)

    @_compile_skip
    def test_torch_compile(self):
        chain = nn.Linear(10, 5) >> nn.ReLU() >> nn.Linear(5, 2)
        model = FaeModule(chain)
        compiled = torch.compile(model)
        x = torch.randn(3, 10)
        out = compiled(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([3, 2])

    @_compile_skip
    def test_torch_compile_residual(self):
        linear = nn.Linear(10, 10)
        chain = X + linear(X)
        model = FaeModule(chain)
        compiled = torch.compile(model)
        x = torch.randn(3, 10)
        out = compiled(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([3, 10])

    @_compile_skip
    def test_torch_compile_output_matches(self):
        """torch.compile output must match eager output."""
        torch.manual_seed(0)
        chain = nn.Linear(10, 5) >> nn.ReLU() >> nn.Linear(5, 2)
        model = FaeModule(chain)
        model.eval()
        compiled = torch.compile(model)

        x = torch.randn(3, 10)
        with torch.no_grad():
            eager_out = model(x)
            compiled_out = compiled(x)

        torch.testing.assert_close(eager_out, compiled_out)


class TestLower:
    def test_lower_simple(self):
        chain = nn.Linear(10, 5) >> nn.ReLU() >> nn.Linear(5, 2)
        model = FaeModule(chain)
        fast = lower(model)
        x = torch.randn(3, 10)
        torch.testing.assert_close(model(x), fast(x))

    def test_lower_sub_modules_registered(self):
        chain = nn.Linear(10, 5) >> nn.Linear(5, 2)
        model = FaeModule(chain)
        fast = lower(model)
        assert len(list(fast.parameters())) > 0

    def test_lower_residual(self):
        linear = nn.Linear(10, 10)
        chain = X + linear(X)
        model = FaeModule(chain)
        fast = lower(model)
        x = torch.randn(3, 10)
        torch.testing.assert_close(model(x), fast(x))

    def test_lower_named_chain(self):
        chain = (nn.Linear(10, 5) >> nn.ReLU()) % "encoder"
        model = FaeModule(chain)
        fast = lower(model)
        x = torch.randn(3, 10)
        torch.testing.assert_close(model(x), fast(x))

    def test_lower_recall_skip(self):
        chain = (nn.Linear(10, 10) % "skip") >> nn.ReLU() >> X + R["skip"]
        model = FaeModule(chain)
        fast = lower(model)
        x = torch.randn(3, 10)
        torch.testing.assert_close(model(x), fast(x))

    def test_lower_recall_table_is_per_call(self):
        chain = (nn.Linear(10, 10) % "skip") >> nn.ReLU() >> X + R["skip"]
        model = FaeModule(chain)
        fast = lower(model)
        a, b = torch.randn(3, 10), torch.randn(3, 10)
        torch.testing.assert_close(fast(a), model(a))
        torch.testing.assert_close(fast(b), model(b))
        torch.testing.assert_close(fast(a), model(a))

    def test_lower_nested_chain_uses_pipeline_value(self):
        """A chain nested inside an F node must seed from the pipeline value, not the model input."""
        inner = nn.Linear(10, 10) >> nn.ReLU()
        chain = nn.Linear(10, 10) >> X + inner
        model = FaeModule(chain)
        fast = lower(model)
        x = torch.randn(3, 10)
        torch.testing.assert_close(model(x), fast(x))

    @_compile_skip
    def test_torch_compile_lower_recall(self):
        chain = (nn.Linear(10, 10) % "skip") >> nn.ReLU() >> X + R["skip"]
        model = FaeModule(chain)
        model.eval()
        fast = lower(model)
        compiled = torch.compile(fast)
        x = torch.randn(3, 10)
        with torch.no_grad():
            torch.testing.assert_close(model(x), compiled(x))
