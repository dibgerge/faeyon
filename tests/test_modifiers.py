"""Tests for faeyon.modifiers."""

import torch
from torch import nn

from faeyon import X, faek
from faeyon.modifiers import IF, Modify, LoRA, Quantize, Record, KVCache, FVar


class TestIF:
    def test_call_modifier_false(self):
        node = X + 1
        assert 3 | IF(False, else_=X)(node) == 3


class TestModify:
    def test_rmod_by_name(self):
        expr = (X + 1) % "n"
        out = expr % Modify("n", IF(False, else_=X))
        assert 4 | out == 4


class TestRecord:
    def test_call_appends(self):
        sink = FVar()
        expr = ((X + 1) % "n") % Modify("n", Record(sink))
        assert 2 | expr == 3
        assert sink.value == [3]


class TestQuantize:
    def test_call_fake_quant_linear(self):
        with faek:
            node = nn.Linear(4, 4)(X) % "lin"
        out = node % Modify("lin", Quantize(bits=4))
        y = torch.randn(2, 4) | out
        assert y.shape == (2, 4)


class TestLoRA:
    def test_call_adds_adapter(self):
        with faek:
            node = nn.Linear(8, 4)(X) % "lin"
        out = node % Modify("lin", LoRA(rank=2, alpha=4))
        y = torch.randn(2, 8) | out
        assert y.shape == (2, 4)


class TestKVCache:
    def test_call_grows_cache(self):
        cache = FVar()
        expr = (X % "k") % Modify("k", KVCache(cache))
        a = torch.randn(1, 2, 4)
        b = torch.randn(1, 3, 4)
        out1 = a | expr
        out2 = b | expr
        assert out1.shape[-2] == 2
        assert out2.shape[-2] == 5
