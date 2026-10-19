"""Unit tests for faeyon.magic.spells."""

import operator
import pytest
import torch
from pytest import param
from torch import nn

from faeyon import A, R, X, FaeList, FaeDict, F, Chain, N, Substitute, Input, faek
from faeyon.magic._opinfo import OpInfo, _divmod
from faeyon.magic.spells import Delayable, Symbol, Sym, _Unpack


class TestDelayable:
    def test_isinstance_f(self):
        expr = X + 1
        assert isinstance(expr, Delayable)
        assert isinstance(expr, F)
        assert expr.fae_name == "add"

    def test_isinstance_symbol(self):
        assert isinstance(X, Delayable)
        assert isinstance(X, X)

    def test_rshift_int_suffixes_names(self):
        block = (X + 1) @ "layer"
        chain = block >> 3
        assert isinstance(chain, Chain)
        assert len(chain) == 3
        assert [op.fae_name for op in chain._fae_ops] == ["layer[0]", "layer[1]", "layer[2]"]

    def test_rshift_int_auto_name_suffixes(self):
        chain = (X + 1) >> 2
        assert len(chain) == 2
        assert [op.fae_name for op in chain._fae_ops] == ["add[0]", "add[1]"]

    def test_rshift_sequence_binds_i(self):
        expr = (X + N) >> [10, 20]
        assert 1 | expr == 31  # (1+10)=11, then 11+20

    def test_rshift_rejects_fully_bound_root(self):
        with pytest.raises(TypeError, match="non-Delayable"):
            _ = (N + 1) >> 2

    def test_rshift_rejects_fully_bound_chain_op(self):
        with pytest.raises(TypeError, match="not a Delayable"):
            _ = (X >> (N + 1)) >> 2

    def test_at_sets_name(self):
        expr = (X + 1) @ "add"
        assert expr.fae_name == "add"

    def test_at_rejects_symbol(self):
        with pytest.raises(TypeError, match="Cannot rename a Symbol"):
            _ = X @ "input"

    def test_at_error_name_on_left(self):
        with pytest.raises(TypeError, match="on the right"):
            _ = "add" @ (X + 1)

    def test_at_rejects_dot(self):
        with pytest.raises(ValueError):
            (X + 1) @ "a.b"

    def test_auto_name_sibling_suffix(self):
        expr = (X + 1) + (X + 2)
        left, right = [c for c in expr.fae_children() if isinstance(c, F)]
        assert expr.fae_name == "add"
        assert {left.fae_name, right.fae_name} == {"add[0]", "add[1]"}

    def test_user_name_sibling_suffix(self):
        """Duplicate ``@`` names among siblings are suffixed like auto-names."""
        expr = ((X + 1) @ "layer") + ((X + 2) @ "layer")
        left, right = [c for c in expr.fae_children() if isinstance(c, F)]
        assert {left.fae_name, right.fae_name} == {"layer[0]", "layer[1]"}

    def test_chain_user_name_sibling_suffix(self):
        chain = ((X + 1) @ "block") >> ((X * 2) @ "block")
        assert [op.fae_name for op in chain._fae_ops] == ["block[0]", "block[1]"]

    def test_at_updates_parent_child_cache(self):
        """Late ``@`` renames maintain the parent's child-name cache and re-suffix."""
        expr = (X + 1) + (X + 2)
        left, right = [c for c in expr.fae_children() if isinstance(c, F)]
        left @ "layer"
        right @ "layer"
        assert {left.fae_name, right.fae_name} == {"layer[0]", "layer[1]"}
        assert set(expr._fae_child_names) == {"layer[0]", "layer[1]"}
        assert all(len(v) == 1 for v in expr._fae_child_names.values())

    def test_chain_stays_unnamed_until_at(self):
        chain = (X + 1) >> (X * 2)
        assert chain.fae_name is None
        assert (chain @ "block").fae_name == "block"

    def test_fae_children_delayables_only(self):
        expr = (X + 1) @ "add"
        children = list(expr.fae_children())
        assert children == [X]

    def test_fae_children_items_includes_non_delayables(self):
        expr = X + 1
        assert list(expr.fae_children(items=True)) == [X, 1]

    def test_fae_children_chain(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        assert [c.fae_name for c in chain.fae_children()] == ["a", "b"]

    def test_fae_walk_depth_first(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        assert [n.fae_name for n in chain.fae_walk()] == [None, "a", "X", "b", "X"]

    def test_fae_walk_breadth_first(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        assert [n.fae_name for n in chain.fae_walk(breadth_first=True)] == [
            None, "a", "b", "X", "X",
        ]

    def test_fae_find_stem_matches_suffixes(self):
        chain = ((X + 1) @ "layer") >> 2
        assert [n.fae_name for n in chain.fae_find("layer")] == ["layer[0]", "layer[1]"]

    def test_fae_find_exact_name(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        assert [n.fae_name for n in chain.fae_find("b")] == ["b"]

    def test_fae_find_by_type(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        assert [n.fae_name for n in chain.fae_find(F)] == ["a", "b"]

    def test_fae_find_path_regex(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        assert [n.fae_name for n in chain.fae_find(r".*\.a")] == ["a"]

    def test_fae_find_regex_suffixed_names(self):
        chain = ((X + 1) @ "layer") >> 2
        assert [n.fae_name for n in chain.fae_find(r"layer\[\d+\]")] == [
            "layer[0]", "layer[1]",
        ]

    def test_fae_find_callback_replaces_node(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        out = chain.fae_find("a", lambda n: (X * 9) @ "replaced")
        assert [op.fae_name for op in out._fae_ops] == ["replaced", "b"]
        assert 2 | out == 36  # (2 * 9) * 2

    def test_fae_find_callback_replaces_root_early(self):
        """Matching the root returns immediately from ``fae_traverse`` (no child walk)."""
        root = (((X + 1) @ "a") >> ((X * 2) @ "b")) @ "block"
        out = root.fae_find("block", lambda n: (X * 7) @ "replaced")
        assert out is not root
        assert not isinstance(out, Chain)
        assert out.fae_name == "replaced"
        assert 3 | out == 21

    def test_fae_clone_shallow(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        cloned = chain.fae_clone()
        assert cloned is not chain
        assert isinstance(cloned, Chain)
        assert [op.fae_name for op in cloned._fae_ops] == ["a", "b"]
        assert 3 | cloned == 8  # (3+1)*2

    def test_fae_clone_recurse(self):
        chain = ((X + 1) @ "a") >> ((X * 2) @ "b")
        cloned = chain.fae_clone(recurse=True)
        assert cloned is not chain
        assert cloned._fae_ops[0] is not chain._fae_ops[0]
        assert 3 | cloned == 8

    def test_fae_clone_recurse_clones_modules(self):
        layer = nn.Linear(2, 2)
        expr = layer(X)
        cloned = expr.fae_clone(recurse=True, clone_modules=True)
        orig_mod = next(m for m in expr.fae_children(items=True) if isinstance(m, nn.Module))
        cloned_mod = next(m for m in cloned.fae_children(items=True) if isinstance(m, nn.Module))
        assert cloned is not expr
        assert cloned_mod is not orig_mod
        assert isinstance(cloned_mod, nn.Linear)

    def test_fae_clone_recurse_shares_modules_when_disabled(self):
        layer = nn.Linear(2, 2)
        expr = layer(X)
        cloned = expr.fae_clone(recurse=True, clone_modules=False)
        orig_mod = next(m for m in expr.fae_children(items=True) if isinstance(m, nn.Module))
        cloned_mod = next(m for m in cloned.fae_children(items=True) if isinstance(m, nn.Module))
        assert cloned_mod is orig_mod


class TestF:
    def test_module_call_str_omits_wrapped_call_impl(self):
        layer = nn.Linear(2, 2)
        expr = layer(X)
        text = str(expr)
        assert "_wrapped_call_impl" not in text
        assert text.startswith("Linear(")
        assert text.endswith("(X)")

    def test_resolve_add(self):
        assert 11 | (X + 1) == 12

    def test_resolve_partial_leaves_symbol(self):
        out = (X + N).fae_resolve(N=3)
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
        named = (X + 1 >> X * 2) @ "block"
        outer = (X >> named)
        assert len(outer) == 2
        assert outer._fae_ops[-1].fae_name == "block"

    def test_len(self):
        assert len(X + 1 >> X * 2) == 2


class TestX:
    def test_resolve_from_kwargs(self):
        assert X.fae_resolve(X=99) == 99

    def test_resolve_unbound(self):
        assert X.fae_resolve() is X

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


class TestOpActionMixin:
    """
    Arithmetic / comparison / utility coverage for ``_OpActionMixin``.

    Delayable×Delayable pairs cover every combination of Symbol (S), F, and Chain (C).
    Scalar/tensor reflected forms and utility ops are covered separately.
    """

    _FLOAT_OPS = [
        param(operator.add, "+", id="add"),
        param(operator.sub, "-", id="sub"),
        param(operator.mul, "*", id="mul"),
        param(operator.truediv, "/", id="truediv"),
        param(operator.floordiv, "//", id="floordiv"),
        param(operator.mod, "%", id="mod"),
        param(operator.pow, "**", id="pow"),
    ]

    _BITWISE_OPS = [
        param(operator.and_, "&", id="and"),
        param(operator.xor, "^", id="xor"),
    ]

    _COMPARE_OPS = [
        param(operator.lt, "<", id="lt"),
        param(operator.le, "<=", id="le"),
        param(operator.eq, "==", id="eq"),
        param(operator.ne, "!=", id="ne"),
        param(operator.gt, ">", id="gt"),
        param(operator.ge, ">=", id="ge"),
    ]

    # Float + bitwise (&/^) + compare + divmod. `|` is resolve-only
    _BINARY_OPS = [
        *_FLOAT_OPS,
        *_BITWISE_OPS,
        *_COMPARE_OPS,
        param(divmod, "divmod", id="divmod"),
    ]

    # Ops whose ``int ○ Delayable`` form hits OpInfo ``__r*__`` (comparisons
    # use the swapped forward op instead).
    _REFLECTED_OPINFO_OPS = frozenset(
        {p.values[0] for p in _FLOAT_OPS}
        | {p.values[0] for p in _BITWISE_OPS}
        | {divmod}
    )

    # Delayable operand kinds for combinatorial arithmetic tests.
    # Resolved values at pipeline input ``x``: S→x, F→2x, C→2*(x+1).
    _KIND_IDS = ("S", "F", "C")

    @staticmethod
    def _assert_result(res, expected):
        if isinstance(expected, tuple):
            assert isinstance(res, tuple) and len(res) == len(expected)
            for r, e in zip(res, expected):
                TestOpActionMixin._assert_result(r, e)
        elif isinstance(expected, torch.Tensor):
            torch.testing.assert_close(res, expected)
        else:
            assert res == expected

    @staticmethod
    def _apply_op(op, a, b):
        if op is divmod:
            return _divmod(a, b)
        return op(a, b)

    @staticmethod
    def _operands():
        return {
            "S": X,
            "F": 2 * X,
            "C": X + 1 >> 2 * X,
        }

    @staticmethod
    def _resolved(kind: str, x):
        if kind == "S":
            return x
        if kind == "F":
            return 2 * x
        return 2 * (x + 1)

    @staticmethod
    def _identity_operands():
        """Operands that resolve to the pipeline input unchanged (list or dict)."""
        return {"S": X, "F": F(lambda x: x, X), "C": Chain(X)}

    @pytest.mark.parametrize("left", _KIND_IDS)
    @pytest.mark.parametrize("right", _KIND_IDS)
    @pytest.mark.parametrize("op, _sym", _BINARY_OPS)
    @pytest.mark.parametrize("inputs", [
        param(2, id="input_int"),
        param(torch.tensor(2), id="input_tensor"),
    ])
    def test_binary_pairs(self, left, right, op, _sym, inputs):
        ops = self._operands()
        expr = op(ops[left], ops[right])
        assert isinstance(expr, F)
        expected = self._apply_op(
            op, self._resolved(left, inputs), self._resolved(right, inputs)
        )
        self._assert_result(inputs | expr, expected)

    @pytest.mark.parametrize("left", _KIND_IDS)
    @pytest.mark.parametrize("right", _KIND_IDS)
    def test_matmul_pairs(self, left, right):
        """
        Gets its own test because it is non-commutative.
        `left[None] @ right[:, None]` for all Delayable kinds.
        """
        # Identity-valued operands so shapes match: all resolve to ``data``.
        ids = {"S": X, "F": X * 1, "C": Chain(X)}
        expr = ids[left][None] @ ids[right][:, None]
        assert isinstance(expr, F)
        data = torch.tensor([1.0, 2.0])
        torch.testing.assert_close(data | expr, torch.tensor([[5.0]]))

    @pytest.mark.parametrize("kind", _KIND_IDS)
    @pytest.mark.parametrize("op, expected_fn", [
        param(operator.neg, lambda v: -v, id="neg"),
        param(operator.pos, lambda v: +v, id="pos"),
        param(operator.abs, abs, id="abs"),
        param(operator.invert, operator.invert, id="invert"),
    ])
    def test_unary_pairs(self, kind, op, expected_fn):
        ops = self._operands()
        expr = op(ops[kind])
        assert isinstance(expr, F)
        data = 3
        assert data | expr == expected_fn(self._resolved(kind, data))

    def test_unary_round(self):
        assert 1 | round(X + 0.6) == 2
        assert 1 | round(Chain(X + 0.6)) == 2

    @pytest.mark.parametrize("kind", _KIND_IDS)
    @pytest.mark.parametrize("op, _sym", _BINARY_OPS)
    @pytest.mark.parametrize("other", [
        param(3, id="int"),
        param(torch.tensor(3), id="tensor"),
    ])
    def test_binary_delayable_plain(self, kind, op, _sym, other):
        """Delayable ○ scalar/tensor uses the forward op (``__add__``, …)."""
        left = self._operands()[kind]
        expr = op(left, other)
        assert isinstance(expr, F)
        if isinstance(expr._fae_op, OpInfo):
            assert not expr._fae_op.name.startswith("r"), expr._fae_op.name
        data = 2
        expected = self._apply_op(op, self._resolved(kind, data), other)
        self._assert_result(data | expr, expected)

    @pytest.mark.parametrize("kind", _KIND_IDS)
    @pytest.mark.parametrize("op, _sym", _BINARY_OPS)
    @pytest.mark.parametrize("lhs", [
        param(10, id="int"),
        param(torch.tensor(10), id="tensor"),
    ])
    def test_binary_reflected(self, kind, op, _sym, lhs):
        """
        ``plain ○ Delayable``: int left hits OpInfo ``__r*__`` for arithmetic /
        &/^ / divmod; tensor left usually goes through ``__torch_function__``.
        """
        right = self._operands()[kind]
        expr = op(lhs, right)
        assert isinstance(expr, F)
        if op in self._REFLECTED_OPINFO_OPS and not isinstance(lhs, torch.Tensor):
            assert isinstance(expr._fae_op, OpInfo)
            assert expr._fae_op.name.startswith("r"), expr._fae_op.name
        elif op is divmod:
            # ``divmod(tensor, Delayable)`` still lands on ``__rdivmod__``.
            assert isinstance(expr._fae_op, OpInfo)
            assert expr._fae_op.name == "rdivmod"
        data = 2
        expected = self._apply_op(op, lhs, self._resolved(kind, data))
        self._assert_result(data | expr, expected)

    @pytest.mark.parametrize("kind", _KIND_IDS)
    def test_rmatmul(self, kind):
        """
        OpInfo ``__rmatmul__`` (tensor @ Delayable usually uses ``__torch_function__``;
        call ``__rmatmul__`` directly to exercise the OpInfo path).
        """
        ids = {"S": X, "F": X * 1, "C": Chain(X)}
        mat = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        right = ids[kind]
        expr = right.__rmatmul__(mat)
        assert isinstance(expr, F)
        assert isinstance(expr._fae_op, OpInfo)
        assert expr._fae_op.name == "rmatmul"
        data = torch.tensor([1.0, 2.0])
        torch.testing.assert_close(data | expr, torch.tensor([3.0, 3.0]))

        # User-facing ``tensor @ Delayable`` still resolves (torch path).
        torch_expr = mat @ right
        assert isinstance(torch_expr, F)
        torch.testing.assert_close(data | torch_expr, torch.tensor([3.0, 3.0]))

    @pytest.mark.parametrize("kind", _KIND_IDS)
    def test_unpack_star(self, kind):
        """``*Delayable`` yields ``_Unpack`` and spreads into call args."""
        op = self._identity_operands()[kind]
        unpack = next(iter(op))
        assert isinstance(unpack, _Unpack)
        assert not unpack._fae_is_map
        assert repr(unpack).startswith("*")

        expr = F(lambda *a: a, *op)
        assert isinstance(expr, F)
        assert any(isinstance(a, _Unpack) for a in expr._fae_args)
        assert [1, 2, 3] | expr == (1, 2, 3)

    @pytest.mark.parametrize("kind", _KIND_IDS)
    def test_unpack_star_mixed_args(self, kind):
        """Positional args around ``*Delayable`` stay in order."""
        op = self._identity_operands()[kind]
        expr = F(lambda *a: a, 0, *op, 9)
        assert [1, 2] | expr == (0, 1, 2, 9)

    @pytest.mark.parametrize("kind", _KIND_IDS)
    def test_unpack_map(self, kind):
        """``**Delayable`` via ``keys``/``getitem`` spreads into call kwargs."""
        op = self._identity_operands()[kind]
        mapping_key = op.keys()[0]
        unpack = op[mapping_key]
        assert isinstance(unpack, _Unpack)
        assert unpack._fae_is_map
        assert repr(unpack).startswith("**")

        expr = F(lambda **kw: kw, **op)
        assert isinstance(expr, F)
        assert any(isinstance(v, _Unpack) for v in expr._fae_kwargs.values())
        assert {"a": 1, "b": 2} | expr == {"a": 1, "b": 2}

    def test_unpack_map_duplicate_raises(self):
        expr = F(lambda a=0, b=0: a + b, a=1, **X)
        with pytest.raises(TypeError, match="multiple values"):
            {"a": 2, "b": 3} | expr

    @pytest.mark.parametrize("expr, data, expected", [
        param(X[1], [10, 20, 30], 20, id="getitem"),
        param(X["a"], {"a": 7}, 7, id="getitem-str"),
        param(X.real, 1 + 2j, 1.0, id="getattr"),
        param(X(2), (lambda n: n + 1), 3, id="call"),
        param(Chain(X)[1], [10, 20, 30], 20, id="chain-getitem"),
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
        assert (X + N).fae_resolve(X=1, N=5) == 6


class TestR:
    def test_resolve_recall(self):
        expr = (X + 1) @ "a" >> X + R["a"]
        assert 3 | expr == 8  # (3+1)=4, then 4+4

    def test_resolve_missing_raises(self):
        expr = X >> X + R["missing"]
        with pytest.raises(KeyError):
            1 | expr


class TestFaeList:
    def test_bind_rejects_concrete_item(self):
        with pytest.raises(TypeError, match="FaeList item"):
            _ = FaeList([X, N]).fae_bind(1)

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
        assert Y.fae_resolve(Y=7) == 7


class TestIF:
    def test_resolve_static_true(self):
        from faeyon.modifiers import IF
        assert 5 | IF(True, X, else_=0) == 5

    def test_resolve_static_false(self):
        from faeyon.modifiers import IF
        assert 5 | IF(False, X, else_=0) == 0

    def test_fae_bind_drops_arm(self):
        from faeyon.modifiers import IF
        bound = IF(N > 0, X + 1, else_=X * 10).fae_bind(2)
        assert isinstance(bound, F)
        assert 3 | bound == 4

    def test_modify_strips_when_false(self):
        from faeyon.modifiers import IF, Modify
        expr = (X + 1) @ "n"
        out = expr % Modify("n", IF(False, else_=X))
        assert 9 | out == 9
