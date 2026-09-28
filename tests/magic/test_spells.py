"""Unit tests for faeyon.magic.spells."""

import pytest
import torch
from pytest import param
from torch import nn, tensor

from faeyon import A, R, X, FaeList, FaeDict, F, Chain, I, Substitute, Input, faek
from faeyon.magic.spells import Delayable, Symbol, Sym


def _assert_result(res, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(res, expected)
    else:
        assert res == expected


class TestDelayable:
    def test_isinstance_f(self):
        assert isinstance(X + 1, Delayable)
        assert isinstance(X + 1, F)

    def test_isinstance_symbol(self):
        assert isinstance(X, Delayable)
        assert isinstance(X, X)

    def test_rshift_int_suffixes_names(self):
        block = (X + 1) % "layer"
        chain = block >> 3
        assert isinstance(chain, Chain)
        assert len(chain) == 3
        assert [op.fae_name for op in chain._fae_ops] == ["layer.0", "layer.1", "layer.2"]

    def test_rshift_int_unnamed(self):
        chain = (X + 1) >> 2
        assert len(chain) == 2
        assert all(op.fae_name is None for op in chain._fae_ops)

    def test_rshift_sequence_binds_i(self):
        expr = (X + I) >> [10, 20]
        assert 1 | expr == 31  # (1+10)=11, then 11+20

    def test_mod_sets_name(self):
        expr = (X + 1) % "add"
        assert expr.fae_name == "add"

    def test_mod_rejects_dot(self):
        with pytest.raises(ValueError):
            (X + 1) % "a.b"

    def test_fae_find_stem_matches_suffixes(self):
        chain = ((X + 1) % "layer") >> 2
        names = []

        def capture(node):
            names.append(node.fae_name)
            return node

        chain.fae_find("layer", capture)
        assert names == ["layer.0", "layer.1"]


class TestF:
    def test_resolve_add(self):
        assert 11 | (X + 1) == 12

    def test_resolve_partial_leaves_symbol(self):
        out = (X + I)._resolve(I=3)
        assert isinstance(out, F)
        assert 10 | out == 13


class TestChain:
    def test_resolve_propagates_x(self):
        assert 11 | (X + 1 >> X / 2) == 6.0

    def test_resolve_three_ops(self):
        assert 11 | (X + 1 >> X / 2 >> 2 * X) == 12.0

    def test_rshift_flattens_unnamed(self):
        left = X + 1 >> X * 2
        right = X / 2 >> X - 1
        merged = left >> right
        assert isinstance(merged, Chain)
        assert len(merged) == 4

    def test_rshift_named_rhs_opaque(self):
        named = (X + 1 >> X * 2) % "block"
        outer = (X >> named)
        assert len(outer) == 2
        assert outer._fae_ops[-1].fae_name == "block"

    def test_len(self):
        assert len(X + 1 >> X * 2) == 2


class TestX:
    def test_resolve_from_kwargs(self):
        assert X._resolve(X=99) == 99

    def test_resolve_unbound(self):
        assert X._resolve() is X

    def test_fae_name(self):
        assert X.fae_name == "X"

    def test_rshift_builds_chain(self):
        expr = X + 1 >> X * 2
        assert isinstance(expr, Chain)
        assert len(expr) == 2

    @pytest.mark.parametrize("expr, expected", [
        param(X, "X", id="X"),
        param(X + 1, "X + 1", id="X + 1"),
        param(X[0] + 1, "X[0] + 1", id="X[0] + 1"),
        param(X(1, foo="bar"), "X(1, foo='bar')", id="X(1, foo='bar')"),
        param(X.a, "X.a", id="X.a"),
        param(X(), "X()", id="X()"),
        param(X + X * 2, "X + X * 2", id="X + X * 2"),
        param(X + 2 * X, "X + 2 * X", id="X + 2 * X"),
        param((X + 1) * (2 + X), "(X + 1) * (2 + X)", id="arithmetic_parens_1"),
        param((X + 1) * X, "(X + 1) * X", id="arithmetic_parens_2"),
        param(X * 2 / (X + 1), "X * 2 / (X + 1)", id="arithmetic_parens_3"),
    ])
    def test_repr(self, expr, expected):
        assert repr(expr) == expected

    def test_matmul(self):
        data = torch.tensor([1.0, 1.0])
        mat = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        torch.testing.assert_close(mat | (X @ data), torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(data | (X @ X), torch.tensor(2.0))
        torch.testing.assert_close(data | (mat @ X), torch.tensor([2.0, 2.0]))


class TestOpActionMixin:
    """
    Cover every ``_OpActionMixin`` arithmetic / comparison path on ``X`` / ``F``,
    including reflected (right-hand) forms for both meta (``X``) and instance (``X+1``).
    """

    @pytest.mark.parametrize("expr, expected", [
        # add
        param(X + (X + 1), 3, id="meta+instance"),
        param((X + 1) + X, 3, id="instance+meta"),
        param((X + 1) + (X + 1), 4, id="instance+instance"),
        param(X + 1, 2, id="meta+int"),
        param(X + 1 + 1, 3, id="instance+int"),
        param(X + X, 2, id="meta+meta"),
        param(X + tensor([1, 2, 3]), tensor([2, 3, 4]), id="meta+tensor"),
        param((X + 1) + tensor([1, 2, 3]), tensor([3, 4, 5]), id="instance+tensor"),
        # radd
        param(1 + X, 2, id="int+meta"),
        param(1 + (1 + X), 3, id="int+instance"),
        param(tensor([1, 2, 3]) + X, tensor([2, 3, 4]), id="tensor+meta"),
        param(tensor([1, 2, 3]) + (X + 1), tensor([3, 4, 5]), id="tensor+instance"),
        # sub
        param(X - (X - 1), 1, id="meta-instance"),
        param((X - 1) - X, -1, id="instance-meta"),
        param((X + 1) - (X + 1), 0, id="instance-instance"),
        param(X - 1, 0, id="meta-int"),
        param(X - 1 - 1, -1, id="instance-int"),
        param(X - X, 0, id="meta-meta"),
        param(X - tensor([1, 2, 3]), tensor([0, -1, -2]), id="meta-tensor"),
        param((X + 1) - tensor([1, 2, 3]), tensor([1, 0, -1]), id="instance-tensor"),
        # rsub
        param(1 - X, 0, id="int-meta"),
        param(1 - (1 + X), -1, id="int-instance"),
        param(tensor([1, 2, 3]) - X, tensor([0, 1, 2]), id="tensor-meta"),
        param(tensor([1, 2, 3]) - (X + 1), tensor([-1, 0, 1]), id="tensor-instance"),
        # mul
        param(X * (X * 1), 1, id="meta*instance"),
        param((X * 1) * X, 1, id="instance*meta"),
        param((X * 1) * (X * 1), 1, id="instance*instance"),
        param(X * 1, 1, id="meta*int"),
        param(X * 1 * 1, 1, id="instance*int"),
        param(X * X, 1, id="meta*meta"),
        param(X * tensor([1, 2, 3]), tensor([1, 2, 3]), id="meta*tensor"),
        param((X * 1) * tensor([1, 2, 3]), tensor([1, 2, 3]), id="instance*tensor"),
        # rmul
        param(1 * X, 1, id="int*meta"),
        param(1 * (1 * X), 1, id="int*instance"),
        param(tensor([1, 2, 3]) * X, tensor([1, 2, 3]), id="tensor*meta"),
        param(tensor([1, 2, 3]) * (X * 1), tensor([1, 2, 3]), id="tensor*instance"),
        # truediv
        param(X / (X / 1), 1.0, id="meta/instance"),
        param((X / 1) / X, 1.0, id="instance/meta"),
        param((X / 1) / (X / 1), 1.0, id="instance/instance"),
        param(X / 1, 1.0, id="meta/int"),
        param(X / 1 / 1, 1.0, id="instance/int"),
        param(X / X, 1.0, id="meta/meta"),
        param(X / tensor([1, 2, 3]), tensor([1.0, 0.5, 1.0 / 3]), id="meta/tensor"),
        param((X / 1) / tensor([1, 2, 3]), tensor([1.0, 0.5, 1.0 / 3]), id="instance/tensor"),
        # rtruediv
        param(1 / X, 1.0, id="int/meta"),
        param(1 / (1 / X), 1.0, id="int/instance"),
        param(tensor([1, 2, 3]) / X, tensor([1.0, 2.0, 3.0]), id="tensor/meta"),
        param(tensor([1, 2, 3]) / (X / 1), tensor([1.0, 2.0, 3.0]), id="tensor/instance"),
        # floordiv
        param(X // (X // 1), 1, id="meta//instance"),
        param((X // 1) // X, 1, id="instance//meta"),
        param((X // 1) // (X // 1), 1, id="instance//instance"),
        param(X // 1, 1, id="meta//int"),
        param(X // 1 // 1, 1, id="instance//int"),
        param(X // X, 1, id="meta//meta"),
        param(X // tensor([1, 2, 3]), tensor([1, 0, 0]), id="meta//tensor"),
        param((X // 1) // tensor([1, 2, 3]), tensor([1, 0, 0]), id="instance//tensor"),
        # rfloordiv
        param(1 // X, 1, id="int//meta"),
        param(1 // (1 // X), 1, id="int//instance"),
        param(tensor([1, 2, 3]) // X, tensor([1, 2, 3]), id="tensor//meta"),
        param(tensor([1, 2, 3]) // (X // 1), tensor([1, 2, 3]), id="tensor//instance"),
        # pow
        param(X ** (X ** 1), 1, id="meta**instance"),
        param((X ** 1) ** X, 1, id="instance**meta"),
        param((X ** 1) ** (X ** 1), 1, id="instance**instance"),
        param(X ** 1, 1, id="meta**int"),
        param(X ** 1 ** 1, 1, id="instance**int"),
        param(X ** X, 1, id="meta**meta"),
        param(X ** tensor([1, 2, 3]), tensor([1, 1, 1]), id="meta**tensor"),
        param((X ** 1) ** tensor([1, 2, 3]), tensor([1, 1, 1]), id="instance**tensor"),
        # rpow
        param(1 ** X, 1, id="int**meta"),
        param(1 ** (1 ** X), 1, id="int**instance"),
        param(tensor([1, 2, 3]) ** X, tensor([1, 2, 3]), id="tensor**meta"),
        param(tensor([1, 2, 3]) ** (X ** 1), tensor([1, 2, 3]), id="tensor**instance"),
        # bitwise and
        param(X & (X & 1), 1, id="meta&instance"),
        param((X & 1) & X, 1, id="instance&meta"),
        param((X & 1) & (X & 1), 1, id="instance&instance"),
        param(X & 1, 1, id="meta&int"),
        param(X & 1 & 1, 1, id="instance&int"),
        param(X & X, 1, id="meta&meta"),
        param(X & tensor([1, 2, 3]), tensor([1, 0, 1]), id="meta&tensor"),
        param((X & 1) & tensor([1, 2, 3]), tensor([1, 0, 1]), id="instance&tensor"),
        # rand
        param(1 & X, 1, id="int&meta"),
        param(1 & (1 & X), 1, id="int&instance"),
        param(tensor([1, 2, 3]) & X, tensor([1, 0, 1]), id="tensor&meta"),
        param(tensor([1, 2, 3]) & (X & 1), tensor([1, 0, 1]), id="tensor&instance"),
        # xor
        param(X ^ (X ^ 1), 1, id="meta^instance"),
        param((X ^ 1) ^ X, 1, id="instance^meta"),
        param((X ^ 1) ^ (X ^ 1), 0, id="instance^instance"),
        param(X ^ 1, 0, id="meta^int"),
        param(X ^ 1 ^ 1, 1, id="instance^int"),
        param(X ^ X, 0, id="meta^meta"),
        param(X ^ tensor([1, 2, 3]), tensor([0, 3, 2]), id="meta^tensor"),
        param((X ^ 1) ^ tensor([1, 2, 3]), tensor([1, 2, 3]), id="instance^tensor"),
        # rxor
        param(1 ^ X, 0, id="int^meta"),
        param(1 ^ (1 ^ X), 1, id="int^instance"),
        param(tensor([1, 2, 3]) ^ X, tensor([0, 3, 2]), id="tensor^meta"),
        param(tensor([1, 2, 3]) ^ (X ^ 1), tensor([1, 2, 3]), id="tensor^instance"),
        # gt
        param(X > (X + 1), False, id="meta>instance"),
        param((X + 1) > X, True, id="instance>meta"),
        param((X + 1) > (X + 1), False, id="instance>instance"),
        param(X > 1, False, id="meta>int"),
        param((X + 1) > 1, True, id="instance>int"),
        param(X > X, False, id="meta>meta"),
        param(X > tensor([1, 2, 3]), tensor([False, False, False]), id="meta>tensor"),
        param((X + 1) > tensor([1, 2, 3]), tensor([True, False, False]), id="instance>tensor"),
        # rgt
        param(1 > X, False, id="int>meta"),
        param(1 > (1 + X), False, id="int>instance"),
        param(tensor([1, 2, 3]) > X, tensor([False, True, True]), id="tensor>meta"),
        param(tensor([1, 2, 3]) > (X + 1), tensor([False, False, True]), id="tensor>instance"),
        # lt
        param(X < (X + 1), True, id="meta<instance"),
        param((X + 1) < X, False, id="instance<meta"),
        param((X + 1) < (X + 1), False, id="instance<instance"),
        param(X < 1, False, id="meta<int"),
        param((X + 1) < 1, False, id="instance<int"),
        param(X < X, False, id="meta<meta"),
        param(X < tensor([1, 2, 3]), tensor([False, True, True]), id="meta<tensor"),
        param((X + 1) < tensor([1, 2, 3]), tensor([False, False, True]), id="instance<tensor"),
        # rlt
        param(1 < X, False, id="int<meta"),
        param(1 < (1 + X), True, id="int<instance"),
        param(tensor([1, 2, 3]) < X, tensor([False, False, False]), id="tensor<meta"),
        param(tensor([1, 2, 3]) < (X + 1), tensor([True, False, False]), id="tensor<instance"),
        # ge
        param(X >= (X + 1), False, id="meta>=instance"),
        param((X + 1) >= X, True, id="instance>=meta"),
        param((X + 1) >= (X + 1), True, id="instance>=instance"),
        param(X >= 1, True, id="meta>=int"),
        param((X + 1) >= 1, True, id="instance>=int"),
        param(X >= X, True, id="meta>=meta"),
        param(X >= tensor([1, 2, 3]), tensor([True, False, False]), id="meta>=tensor"),
        param((X + 1) >= tensor([1, 2, 3]), tensor([True, True, False]), id="instance>=tensor"),
        # rge
        param(1 >= X, True, id="int>=meta"),
        param(1 >= (1 + X), False, id="int>=instance"),
        param(tensor([1, 2, 3]) >= X, tensor([True, True, True]), id="tensor>=meta"),
        param(tensor([1, 2, 3]) >= (X + 1), tensor([False, True, True]), id="tensor>=instance"),
        # le
        param(X <= (X + 1), True, id="meta<=instance"),
        param((X + 1) <= X, False, id="instance<=meta"),
        param((X + 1) <= (X + 1), True, id="instance<=instance"),
        param(X <= 1, True, id="meta<=int"),
        param((X + 1) <= 1, False, id="instance<=int"),
        param(X <= X, True, id="meta<=meta"),
        param(X <= tensor([1, 2, 3]), tensor([True, True, True]), id="meta<=tensor"),
        param((X + 1) <= tensor([1, 2, 3]), tensor([False, True, True]), id="instance<=tensor"),
        # rle
        param(1 <= X, True, id="int<=meta"),
        param(1 <= (1 + X), True, id="int<=instance"),
        param(tensor([1, 2, 3]) <= X, tensor([True, False, False]), id="tensor<=meta"),
        param(tensor([1, 2, 3]) <= (X + 1), tensor([True, True, False]), id="tensor<=instance"),
        # eq
        param(X == (X + 1), False, id="meta==instance"),
        param((X + 1) == X, False, id="instance==meta"),
        param((X + 1) == (X + 1), True, id="instance==instance"),
        param(X == 1, True, id="meta==int"),
        param((X + 1) == 1, False, id="instance==int"),
        param(X == X, True, id="meta==meta"),
        param(X == tensor([1, 2, 3]), tensor([True, False, False]), id="meta==tensor"),
        param((X + 1) == tensor([1, 2, 3]), tensor([False, True, False]), id="instance==tensor"),
        # ne
        param(X != (X + 1), True, id="meta!=instance"),
        param((X + 1) != X, True, id="instance!=meta"),
        param((X + 1) != (X + 1), False, id="instance!=instance"),
        param(X != 1, False, id="meta!=int"),
        param((X + 1) != 1, True, id="instance!=int"),
        param(X != X, False, id="meta!=meta"),
        param(X != tensor([1, 2, 3]), tensor([False, True, True]), id="meta!=tensor"),
        param((X + 1) != tensor([1, 2, 3]), tensor([True, False, True]), id="instance!=tensor"),
        # mod (int RHS only — string RHS is naming via Delayable.__mod__)
        param(X % 2, 1, id="meta%int"),
        param((X + 1) % 2, 0, id="instance%int"),
        param(X % X, 0, id="meta%meta"),
        param(5 % X, 0, id="int%meta"),
        param(5 % (X + 2), 2, id="int%instance"),
    ])
    @pytest.mark.parametrize("inputs", [
        param(1, id="input_int"),
        param(torch.tensor(1), id="input_tensor"),
    ])
    def test_binary_ops(self, expr, expected, inputs):
        assert isinstance(expr, F)
        _assert_result(inputs | expr, expected)

    @pytest.mark.parametrize("expr, expected", [
        param(-X, -1, id="neg-meta"),
        param(-(X + 1), -2, id="neg-instance"),
        param(+X, 1, id="pos-meta"),
        param(+(X + 1), 2, id="pos-instance"),
        param(abs(X), 1, id="abs-meta"),
        param(abs(X - 3), 2, id="abs-instance"),
        param(~X, ~1, id="invert-meta"),
        param(~(X + 1), ~2, id="invert-instance"),
        param(round(X + 0.6), 2, id="round-instance"),
    ])
    def test_unary_ops(self, expr, expected):
        assert isinstance(expr, F)
        assert 1 | expr == expected

    @pytest.mark.parametrize("expr, data, expected", [
        param(X[1], [10, 20, 30], 20, id="getitem"),
        param(X["a"], {"a": 7}, 7, id="getitem-str"),
        param(X.real, 1 + 2j, 1.0, id="getattr"),
        param(X(2), (lambda n: n + 1), 3, id="call"),
    ])
    def test_utility_ops(self, expr, data, expected):
        assert isinstance(expr, F)
        assert data | expr == expected


class TestA:
    def test_fae_name(self):
        assert A.fae_name == "A"

    def test_distinct_from_x_in_chain(self):
        expr = A + 1 >> X * 2
        assert Substitute(A=3, X=10) | expr == 8  # first uses A=3 -> 4; then X=4*2


class TestI:
    def test_bind_in_expr(self):
        assert (X + I)._resolve(X=1, I=5) == 6


class TestR:
    def test_resolve_recall(self):
        expr = (X + 1) % "a" >> X + R["a"]
        assert 3 | expr == 8  # (3+1)=4, then 4+4

    def test_resolve_missing_raises(self):
        expr = X >> X + R["missing"]
        with pytest.raises(KeyError):
            1 | expr


class TestFaeList:
    def test_resolve_list(self):
        assert 2 | FaeList([X + 1, X * 3]) == [3, 6]

    def test_len(self):
        assert len(FaeList([X, X + 1])) == 2

    @pytest.mark.parametrize("expr, expected", [
        param(FaeList([X, X + 1]) + 1, [2, 3], id="list+int"),
        param(1 + FaeList([X, X + 1]), [2, 3], id="int+list"),
        param(FaeList([X, X + 1]) * 2, [2, 4], id="list*int"),
        param(2 * FaeList([X, X + 1]), [2, 4], id="int*list"),
    ])
    def test_elementwise_ops(self, expr, expected):
        assert 1 | expr == expected


class TestFaeDict:
    def test_resolve_dict(self):
        assert 2 | FaeDict({"a": X + 1, "b": X * 3}) == {"a": 3, "b": 6}

    def test_len(self):
        assert len(FaeDict({"a": X})) == 1

    @pytest.mark.parametrize("expr, expected", [
        param(FaeDict({"a": X, "b": X + 1}) + 1, {"a": 2, "b": 3}, id="dict+int"),
        param(1 + FaeDict({"a": X, "b": X + 1}), {"a": 2, "b": 3}, id="int+dict"),
    ])
    def test_elementwise_ops(self, expr, expected):
        assert 1 | expr == expected


class TestSubstitute:
    def test_or_binds_symbols(self):
        assert (Substitute(X=10, Y=20) | (X + Sym.Y)) == 30

    def test_or_partial(self):
        out = Substitute(X=10) | (X + Sym.Y)
        assert isinstance(out, F)


class TestInput:
    def test_getitem_pos(self):
        inp = Input(1, 2, bias=3)
        assert inp[0] == 1
        assert inp["bias"] == 3


class TestSym:
    def test_dynamic_symbol(self):
        Y = Sym.Y
        assert Y.fae_name == "Y"
        assert Y._resolve(Y=7) == 7


class TestIF:
    def test_resolve_static_true(self):
        from faeyon.modifiers import IF
        assert 5 | IF(True, X, else_=0) == 5

    def test_resolve_static_false(self):
        from faeyon.modifiers import IF
        assert 5 | IF(False, X, else_=0) == 0

    def test_fae_bind_drops_arm(self):
        from faeyon.modifiers import IF
        bound = IF(I > 0, X + 1, else_=X * 10).fae_bind(I=2)
        assert isinstance(bound, F)
        assert 3 | bound == 4

    def test_modify_strips_when_false(self):
        from faeyon.modifiers import IF, Modify
        expr = (X + 1) % "n"
        out = expr % Modify("n", IF(False, else_=X))
        assert 9 | out == 9
