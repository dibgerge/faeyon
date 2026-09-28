"""Tests for faeyon.nn.moe."""

import torch

from faeyon import faek
from faeyon.nn import MoE, SwiGLU


def _call(module, *args, **kwargs):
    """Eager module call (bypass faek's delayed ``__call__``)."""
    return faek.module__call__(module, *args, **kwargs)


class TestSwiGLU:
    def test_forward_shape(self):
        layer = SwiGLU(hidden=8, intermediate=16)
        out = _call(layer, torch.randn(2, 5, 8))
        assert out.shape == (2, 5, 8)


class TestMoE:
    def test_forward_shape(self):
        moe = MoE(hidden=8, intermediate=16, n_experts=4, top_k=2)
        out = _call(moe, torch.randn(2, 5, 8))
        assert out.shape == (2, 5, 8)

    def test_forward_flat(self):
        moe = MoE(hidden=8, intermediate=16, n_experts=3, top_k=1)
        out = _call(moe, torch.randn(4, 8))
        assert out.shape == (4, 8)

    def test_topk_selects_expert(self):
        moe = MoE(hidden=4, intermediate=8, n_experts=3, top_k=1, bias=True)
        with torch.no_grad():
            moe.router.weight.zero_()
            moe.router.weight[1].fill_(1.0)
            for expert in moe.experts:
                for p in expert.parameters():
                    p.zero_()
            moe.experts[1].w3.bias.fill_(2.0)

        out = _call(moe, torch.ones(2, 4))
        torch.testing.assert_close(out, torch.full((2, 4), 2.0))

    def test_clone_independent_weights(self):
        with faek:
            moe = MoE(hidden=4, intermediate=8, n_experts=2, top_k=1)
            cloned = moe.clone()
        assert cloned is not moe
        assert cloned.router is not moe.router
        assert cloned.experts[0] is not moe.experts[0]
