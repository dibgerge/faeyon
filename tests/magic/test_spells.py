import pytest
import inspect
import torch
from faeyon import A, X, FVar, FList, FDict, FMMap, F, Wire, W, Chain
from faeyon.magic.spells import conjure

from tests.common import ConstantLayer
from pytest import param
from torch import tensor


class TestX:
    def test_noop(self):
        x = X
        assert isinstance(x, X)
        assert 1 | X == 1


    def test_rshift(self):
        """ The right shift operator results in a Chain Object if both arguments are of type X."""
        x = X[1] >> X + 1
        assert isinstance(x, Chain)
        assert len(x) == 2

    def test_rshift_error(self):
        """ Shift operator not defined with non-X arguments. """
        with pytest.raises(TypeError):
            x = X[1] >> 1

    def test_rrshift(self):
        """ Shift operator not defined with non-X arguments (Use | instead). """
        with pytest.raises(TypeError):
            x = 1 >> X[1]

    def test_or_error(self):
        """ Pipe magic X expression to anything is not defined. """
        with pytest.raises(TypeError):
            x = X[1] | 1
        
        with pytest.raises(TypeError):
            x = X[1] | X[2]

    @pytest.mark.parametrize("expr, expected", [
        param(X, "X", id="X"),
        param(X + 1, "X + 1", id="X + 1"),
        param(X[0] + 1, "X[0] + 1", id="X[0] + 1"),
        param(X(1, foo="bar"), "X(1, foo='bar')", id="X(1, foo='bar')"),
        param(X.a, "X.a", id="X.a"),
        param(X(), "X()", id="X()"),
    ])
    def test_repr(self, expr, expected):
        """ 
        Test some different ways of representing X and make sure the repr is correct. 
        """
        assert repr(expr) == expected

    # --- Test arithmetic operators ---
    @pytest.mark.parametrize("expr, expected", [
        # test add
        param(X + (X + 1), 3, id="meta+instance"),
        param((X+1) + X, 3, id="instance+meta"),
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
        param(X - (X - 1), 1, id="meta-instance)"),
        param((X-1) - X, -1, id="instance-meta)"),
        param((X + 1) - (X + 1), 0, id="instance-instance)"),
        param(X - 1, 0, id="meta-int)"),
        param(X - 1 - 1, -1, id="instance-int)"),
        param(X - X, 0, id="meta-meta)"),
        param(X - tensor([1, 2, 3]), tensor([0, -1, -2]), id="meta-tensor)"),
        param((X + 1) - tensor([1, 2, 3]), tensor([1, 0, -1]), id="instance-tensor)"),

        # rsub
        param(1 - X, 0, id="int-meta"),
        param(1 - (1 + X), -1, id="int-instance"),
        param(tensor([1, 2, 3]) - X, tensor([0, 1, 2]), id="tensor-meta)"),
        param(tensor([1, 2, 3]) - (X + 1), tensor([-1, 0, 1]), id="tensor-instance)"),

        # mult
        param(X * (X * 1), 1, id="meta*instance)"),
        param((X*1) * X, 1, id="instance*meta)"),
        param((X * 1) * (X * 1), 1, id="instance*instance)"),
        param(X * 1, 1, id="meta*int)"),
        param(X * 1 * 1, 1, id="instance*int)"),
        param(X * X, 1, id="meta*meta)"),
        param(X * tensor([1, 2, 3]), tensor([1, 2, 3]), id="meta*tensor)"),
        param((X * 1) * tensor([1, 2, 3]), tensor([1, 2, 3]), id="instance*tensor)"),

        # rmult
        param(1 * X, 1, id="int*meta"),
        param(1 * (1 * X), 1, id="int*instance"),
        param(tensor([1, 2, 3]) * X, tensor([1, 2, 3]), id="tensor*meta)"),
        param(tensor([1, 2, 3]) * (X * 1), tensor([1, 2, 3]), id="tensor*instance)"),

        # matmul
        # param(X @ (X @ 1), 1, id="meta@instance)"),
        # param((X@1) @ X, 1, id="instance@meta)"),
        # param((X @ 1) @ (X @ 1), 1, id="instance@instance)"),
        # param(X @ 1, 1, id="meta@int)"),
        # param(X @ 1 @ 1, 1, id="instance@int)"),
        # param(X @ X, 1, id="meta@meta)"),
        # param(X @ tensor([1, 2, 3]), tensor([1, 2, 3]), id="meta@tensor)"),
        # param((X @ 1) @ tensor([1, 2, 3]), tensor([1, 2, 3]), id="instance@tensor)"),

        # # rmatmul
        # param(1 @ X, 1, id="int@meta"),
        # param(1 @ (1 @ X), 1, id="int@instance"),
        # param(tensor([1, 2, 3]) @ X, tensor([1, 2, 3]), id="tensor@meta)"),
        # param(tensor([1, 2, 3]) @ (X @ 1), tensor([1, 2, 3]), id="tensor@instance)"),

        # truediv
        param(X / (X / 1), 1.0, id="meta/instance)"),
        param((X/1) / X, 1.0, id="instance/meta)"),
        param((X / 1) / (X / 1), 1.0, id="instance/instance)"),
        param(X / 1, 1.0, id="meta/int)"),
        param(X / 1 / 1, 1.0, id="instance/int)"),
        param(X / X, 1.0, id="meta/meta)"),
        param(X / tensor([1, 2, 3]), tensor([1.0, 0.5, 0.3333333]), id="meta/tensor)"),
        param((X / 1) / tensor([1, 2, 3]), tensor([1.0, 0.5, 0.3333333]), id="instance/tensor)"),

        # rtruediv
        param(1 / X, 1.0, id="int/meta"),
        param(1 / (1 / X), 1.0, id="int/instance"),
        param(tensor([1, 2, 3]) / X, tensor([1.0, 2.0, 3.0]), id="tensor/meta)"),
        param(tensor([1, 2, 3]) / (X / 1), tensor([1.0, 2.0, 3.0]), id="tensor/instance)"),

        # floordiv
        param(X // (X // 1), 1, id="meta//instance)"),
        param((X//1) // X, 1, id="instance//meta)"),
        param((X // 1) // (X // 1), 1, id="instance//instance)"),
        param(X // 1, 1, id="meta//int)"),
        param(X // 1 // 1, 1, id="instance//int)"),
        param(X // X, 1, id="meta//meta)"),
        param(X // tensor([1, 2, 3]), tensor([1, 0, 0]), id="meta//tensor)"),
        param((X // 1) // tensor([1, 2, 3]), tensor([1, 0, 0]), id="instance//tensor)"),

        # rfloordiv
        param(1 // X, 1, id="int//meta"),
        param(1 // (1 // X), 1, id="int//instance"),
        param(tensor([1, 2, 3]) // X, tensor([1, 2, 3]), id="tensor//meta)"),
        param(tensor([1, 2, 3]) // (X // 1), tensor([1, 2, 3]), id="tensor//instance)"),

        # # mod
        # param(X % (X % 1), 1, id="meta%instance)"),
        # param((X%1) % X, 1, id="instance%meta)"),
        # param((X % 1) % (X % 1), 1, id="instance%instance)"),
        # param(X % 1, 1, id="meta%int)"),
        # param(X % 1 % 1, 1, id="instance%int)"),
        # param(X % X, 1, id="meta%meta)"),
        # param(X % tensor([1, 2, 3]), tensor([1, 1, 1]), id="meta%tensor)"),
        # param((X % 1) % tensor([1, 2, 3]), tensor([1, 1, 1]), id="instance%tensor)"),

        # # rmod
        # param(1 % X, 1, id="int%meta"),
        # param(1 % (1 % X), 1, id="int%instance"),
        # param(tensor([1, 2, 3]) % X, tensor([1, 1, 1]), id="tensor%meta)"),
        # param(tensor([1, 2, 3]) % (X % 1), tensor([1, 1, 1]), id="tensor%instance)"),

        # pow
        param(X ** (X ** 1), 1, id="meta**instance)"),
        param((X**1) ** X, 1, id="instance**meta)"),
        param((X ** 1) ** (X ** 1), 1, id="instance**instance)"),
        param(X ** 1, 1, id="meta**int)"),
        param(X ** 1 ** 1, 1, id="instance**int)"),
        param(X ** X, 1, id="meta**meta)"),
        param(X ** tensor([1, 2, 3]), tensor([1, 1, 1]), id="meta**tensor)"),
        param((X ** 1) ** tensor([1, 2, 3]), tensor([1, 1, 1]), id="instance**tensor)"),

        # rpow
        param(1 ** X, 1, id="int**meta"),
        param(1 ** (1 ** X), 1, id="int**instance"),
        param(tensor([1, 2, 3]) ** X, tensor([1, 2, 3]), id="tensor**meta)"),
        param(tensor([1, 2, 3]) ** (X ** 1), tensor([1, 2, 3]), id="tensor**instance)"),

        # bitwise and
        param(X & (X & 1), 1, id="meta&instance)"),
        param((X&1) & X, 1, id="instance&meta)"),   
        param((X & 1) & (X & 1), 1, id="instance&instance)"),
        param(X & 1, 1, id="meta&int)"),
        param(X & 1 & 1, 1, id="instance&int)"),
        param(X & X, 1, id="meta&meta)"),
        param(X & tensor([1, 2, 3]), tensor([1, 0, 1]), id="meta&tensor)"),
        param((X & 1) & tensor([1, 2, 3]), tensor([1, 0, 1]), id="instance&tensor)"),

        # rbitwise and
        param(1 & X, 1, id="int&meta"),
        param(1 & (1 & X), 1, id="int&instance"),
        param(tensor([1, 2, 3]) & X, tensor([1, 0, 1]), id="tensor&meta)"),
        param(tensor([1, 2, 3]) & (X & 1), tensor([1, 0, 1]), id="tensor&instance)"),

        # # bitwise or
        # param(X | (X | 1), 1, id="meta|instance)"),
        # param((X|1) | X, 1, id="instance|meta)"),
        # param((X | 1) | (X | 1), 1, id="instance|instance)"),
        # param(X | 1, 1, id="meta|int)"),
        # param(X | 1 | 1, 1, id="instance|int)"),
        # param(X | X, 1, id="meta|meta)"),
        # param(X | tensor([1, 2, 3]), tensor([1, 1, 1]), id="meta|tensor)"),
        # param((X | 1) | tensor([1, 2, 3]), tensor([1, 1, 1]), id="instance|tensor)"),

        # # rbitwise or
        # param(1 | X, 1, id="int|meta"),
        # param(1 | (1 | X), 1, id="int|instance"),
        # param(tensor([1, 2, 3]) | X, tensor([1, 1, 1]), id="tensor|meta)"),
        # param(tensor([1, 2, 3]) | (X | 1), tensor([1, 1, 1]), id="tensor|instance)"),

        # bitwise xor
        param(X ^ (X ^ 1), 1, id="meta^instance)"),
        param((X^1) ^ X, 1, id="instance^meta)"),
        param((X ^ 1) ^ (X ^ 1), 0, id="instance^instance)"),
        param(X ^ 1, 0, id="meta^int)"),
        param(X ^ 1 ^ 1, 1, id="instance^int)"),
        param(X ^ X, 0, id="meta^meta)"),
        param(X ^ tensor([1, 2, 3]), tensor([0, 3, 2]), id="meta^tensor)"),
        param((X ^ 1) ^ tensor([1, 2, 3]), tensor([1, 2, 3]), id="instance^tensor)"),

        # rbitwise xor
        param(1 ^ X, 0, id="int^meta"),
        param(1 ^ (1 ^ X), 1, id="int^instance"),
        param(tensor([1, 2, 3]) ^ X, tensor([0, 3, 2]), id="tensor^meta)"),
        param(tensor([1, 2, 3]) ^ (X ^ 1), tensor([1, 2, 3]), id="tensor^instance)"),

        # gt
        param(X > (X + 1), False, id="meta>instance)"),
        param((X + 1) > X, True, id="instance>meta)"),
        param((X + 1) > (X + 1), False, id="instance>instance)"),
        param(X > 1, False, id="meta>int)"),
        param((X + 1) > 1, True, id="instance>int)"),
        param(X > X, False, id="meta>meta)"),
        param(X > tensor([1, 2, 3]), tensor([False, False, False]), id="meta>tensor)"),
        param((X + 1) > tensor([1, 2, 3]), tensor([True, False, False]), id="instance>tensor)"),

        # rgt
        param(1 > X, False, id="int>meta"),
        param(1 > (1 + X), False, id="int>instance"),
        param(tensor([1, 2, 3]) > X, tensor([False, True, True]), id="tensor>meta)"),
        param(tensor([1, 2, 3]) > (X + 1), tensor([False, False, True]), id="tensor>instance)"),

        # lt
        param(X < (X + 1), True, id="meta<instance)"),
        param((X + 1) < X, False, id="instance<meta)"),
        param((X + 1) < (X + 1), False, id="instance<instance)"),
        param(X < 1, False, id="meta<int)"),
        param((X + 1) < 1, False, id="instance<int)"),
        param(X < X, False, id="meta<meta)"),
        param(X < tensor([1, 2, 3]), tensor([False, True, True]), id="meta<tensor)"),
        param((X + 1) < tensor([1, 2, 3]), tensor([False, False, True]), id="instance<tensor)"),

        # rlt
        param(1 < X, False, id="int<meta"),
        param(1 < (1 + X), True, id="int<instance"),
        param(tensor([1, 2, 3]) < X, tensor([False, False, False]), id="tensor<meta)"),
        param(tensor([1, 2, 3]) < (X + 1), tensor([True, False, False]), id="tensor<instance)"),

        # ge
        param(X >= (X + 1), False, id="meta>=instance)"),
        param((X + 1) >= X, True, id="instance>=meta)"),
        param((X + 1) >= (X + 1), True, id="instance>=instance)"),
        param(X >= 1, True, id="meta>=int)"),
        param((X + 1) >= 1, True, id="instance>=int)"),
        param(X >= X, True, id="meta>=meta)"),
        param(X >= tensor([1, 2, 3]), tensor([True, False, False]), id="meta>=tensor)"),
        param((X + 1) >= tensor([1, 2, 3]), tensor([True, True, False]), id="instance>=tensor)"),

        # rge
        param(1 >= X, True, id="int>=meta"),
        param(1 >= (1 + X), False, id="int>=instance"),
        param(tensor([1, 2, 3]) >= X, tensor([True, True, True]), id="tensor>=meta)"),
        param(tensor([1, 2, 3]) >= (X + 1), tensor([False, True, True]), id="tensor>=instance)"),

        # le
        param(X <= (X + 1), True, id="meta<=instance)"),
        param((X + 1) <= X, False, id="instance<=meta)"),
        param((X + 1) <= (X + 1), True, id="instance<=instance)"),
        param(X <= 1, True, id="meta<=int)"),
        param((X + 1) <= 1, False, id="instance<=int)"),
        param(X <= X, True, id="meta<=meta)"),
        param(X <= tensor([1, 2, 3]), tensor([True, True, True]), id="meta<=tensor)"),
        param((X + 1) <= tensor([1, 2, 3]), tensor([False, True, True]), id="instance<=tensor)"),

        # rle
        param(1 <= X, True, id="int<=meta"),
        param(1 <= (1 + X), True, id="int<=instance"),
        param(tensor([1, 2, 3]) <= X, tensor([True, False, False]), id="tensor<=meta)"),
        param(tensor([1, 2, 3]) <= (X + 1), tensor([True, True, False]), id="tensor<=instance)"),

        # eq
        param(X == (X + 1), False, id="meta==instance)"),
        param((X + 1) == X, False, id="instance==meta)"),
        param((X + 1) == (X + 1), True, id="instance==instance)"),
        param(X == 1, True, id="meta==int)"),
        param((X + 1) == 1, False, id="instance==int)"),
        param(X == X, True, id="meta==meta)"),
        param(X == tensor([1, 2, 3]), tensor([True, False, False]), id="meta==tensor)"),
        param((X + 1) == tensor([1, 2, 3]), tensor([False, True, False]), id="instance==tensor)"),

        # ne
        param(X != (X + 1), True, id="meta!=instance)"),
        param((X + 1) != X, True, id="instance!=meta)"),
        param((X + 1) != (X + 1), False, id="instance!=instance)"),
        param(X != 1, False, id="meta!=int)"),
        param((X + 1) != 1, True, id="instance!=int)"),
        param(X != X, False, id="meta!=meta)"),
        param(X != tensor([1, 2, 3]), tensor([False, True, True]), id="meta!=tensor)"),
        param((X + 1) != tensor([1, 2, 3]), tensor([True, False, True]), id="instance!=tensor)"),
    ])
    @pytest.mark.parametrize("inputs", [
        param(1, id="input_int"), 
        param(torch.tensor(1), id="input_tensor")]
    )
    def test_operators(self, expr, expected, inputs):
        assert isinstance(expr, F)
        res = inputs | expr

        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(res, expected)
        else:
            assert res == expected
    
    def test_matmul(self):
        data = torch.tensor([1.0, 1.0])
        mat = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        torch.testing.assert_close(mat | X @ data, torch.tensor([2.0, 2.0]))
        torch.testing.assert_close(data | X @ X, torch.tensor(2.0))

        # test rmatmul
        torch.testing.assert_close(data | mat @ X, torch.tensor([2.0, 2.0]))


class TestX2:

    def test_noop(self):
        """ No operation available in `X` """
        x = X
        assert issubclass(x, X)
        assert len(x) == 0

    def test_single_op(self):
        """ One operation available in `X` """

        # getitem
        x = X[0]
        assert isinstance(x, X)
        assert len(x) == 1

        # add
        x = X + 1
        assert isinstance(x, X)
        assert len(x) == 1

        # call
        x = X("foo", bar="baz")
        assert isinstance(x, X)
        assert len(x) == 1

        # getattr
        x = X.a
        assert isinstance(x, X)
        assert len(x) == 1

    def test_multiple_ops(self):
        """ Multiple operations available in `X` """
        x = X[0] + 1
        assert isinstance(x, X)
        assert len(x) == 2

        x = round(X[0] + 1)
        assert isinstance(x, X)
        assert len(x) == 3

    def test_iter_noop(self):
        all_ops = list(X)
        assert len(all_ops) == 0

    def test_iter_multiops(self):
        x = round(X[0] + 1)
        all_ops = [op[0] for op in x]
        assert all_ops == ["__getitem__", "__add__", "__round__"]

    def test_rshift(self):
        delayed = X >> X[1] + 1
        assert isinstance(delayed, Serials)
        assert len(delayed) == 2

    @pytest.mark.parametrize("data,expected", [
        ([1, 2, 3], 2),
        (torch.tensor([1, 2, 3]), torch.tensor(2)),
    ])
    def test_rrshift(self, data, expected):
        res = data >> X[1]
        assert res == expected

    def test_lshift(self):
        delayed = X << X[1] + 1
        assert isinstance(delayed, Parallels)
        assert len(delayed) == 1
        res = [1, 2, 3] >> delayed
        assert res == 3

    def test_rlshift(self):

        def func(a):
            return a + [10]

        delayed = Op(func, X) << X[1] + 1
        assert isinstance(delayed, Parallels)
        assert len(delayed) == 1
        res = [1, 2, 3] >> delayed
        assert res == 3

    def test_repr_noops(self):
        assert str(X) == "X"

    def test_repr_multiops(self):
        x = round(X("foo", k="bar")[0].a + 1)
        assert repr(x) == "round(X('foo', k='bar')[0].a + 1)"


# class TestA:

#     def func_simple(self, x):
#         return x + 1

#     def func_multi(self, x=1, y=0):
#         return x + y
    
#     def test_init(self):
#         fae = A(1, 2, 3)
#         assert isinstance(fae, A)
#         assert fae.args == (1, 2, 3)
#         assert fae.kwargs == {}

#     def test_call_raisesTypeError(self):
#         """ 
#         When `A` does not match the callable's required number of arguments.
#         """
#         fae_args = A(1, 2, 3)

#         with pytest.raises(TypeError):
#             fae_args.call(self.func_simple)
        
#         with pytest.raises(TypeError):
#             fae_args >> self.func_simple

#     def test_call_unresolved(self):
#         """ 
#         When `A` has unresolved arguments, an error is raised.
#         """
#         fae_args = A(X[0])
#         delayed = fae_args >> self.func_simple
#         assert isinstance(delayed, Op)
#         res = [1, 2, 3] >> delayed
#         assert res == 2

#     @pytest.mark.parametrize("args,kwargs", [
#         ((), {}),
#         ((2,), {}),
#         ((), {"y": 10}),
#         ((1,), {"y": 10}),
#     ])
#     def test_call(self, args, kwargs):
#         """ Tests the call (A >> callable) operator/method. """
#         fae_args = A(*args, **kwargs)
#         expected = self.func_multi(*args, **kwargs)
#         assert fae_args.call(self.func_multi) == expected
#         assert fae_args >> self.func_multi == expected

#     def test_using_resolved(self):
#         """ Tests the bind (Any >> A) operator when `A` is already resolved. """
#         fae_args = A(1, x="Bar")
#         data = "Foo"
#         out_args = fae_args.using(data)
#         assert out_args.args == (1,)
#         assert out_args.kwargs == {"x": "Bar"}
        
#         out_args = data >> fae_args
#         assert out_args.args == (1,)
#         assert out_args.kwargs == {"x": "Bar"}

#     def test_using_unresolved(self):
#         fea_args = A(X[0], x="Bar", y=X[1])
#         data = [10, 11, 12]

#         out_args = fea_args.using(data)
#         assert out_args.args == (10,)
#         assert out_args.kwargs == {"x": "Bar", "y": 11}

#         out_args = data >> fea_args
#         assert out_args.args == (10,)
#         assert out_args.kwargs == {"x": "Bar", "y": 11}


# class TestFVar:
    
#     def test_delayed(self):
#         fvar = FVar()
#         delayed = Op(X) >> fvar
#         assert isinstance(delayed, Serials)
    
#     def test_rrshift(self):
#         fvar = FVar()
#         2 >> fvar
#         assert +fvar == 2

#     def test_rrshift_overwrite(self):
#         """ 
#         Cannot bind a value to a strict FVar with existing value.
#         """
#         fvar = FVar(morphable=False)
#         2 >> fvar
#         3 >> fvar
#         assert +fvar == 3

#     def test_rrshift_morph(self):
#         fvar = FVar(morphable=True)
#         out = 2 >> fvar
#         assert +fvar == 2
#         assert out == 2

#         out = 3 >> fvar
#         assert isinstance(fvar, FList)
#         assert fvar.value == [2, 3]
#         assert out == 3

#     def test_lshift(self):
#         fvar = FVar(morphable=True)
#         out = 2 >> (X << fvar)
#         assert +fvar == 2
        
#     def test_lshift_parallels(self):
#         fvar = FVar(morphable=True)
#         out = 2 >> (Parallels([X + 1, X + 1]) << fvar)
#         assert isinstance(fvar, FList)
#         assert +fvar == [3, 4]

#     def test_lshift_parallels_strict(self):
#         fvar = FVar(morphable=False)
#         out = 2 >> (Parallels([X + 1, X + 1]) << fvar)
#         assert isinstance(fvar, FVar)
#         assert +fvar == 4

#     def test_using(self):
#         """ `using` method is same as >> operator. """
#         fvar = FVar(morphable=False)
#         out = fvar.using(2)
#         assert +fvar == 2
#         assert out == 2

#     def test_if_on_X(self):
#         fvar = FVar(morphable=False)
#         data = [1, 2, 3]
#         out = data >> fvar.if_(X[0] > 1) @ X[1]
#         assert data is out
#         assert fvar.is_empty

#     @pytest.mark.parametrize("condition, expected", [
#         (True, 2),
#         (False, None),
#         (Op(X[0] == 1), 2),
#         (Op(X[0] != 1), None),
#     ])
#     def test_if_general(self, condition, expected):
#         fvar = FVar(morphable=False)
#         data = [1, 2, 3]
#         out = data >> fvar.if_(condition) @ X[1]
#         assert data is out
#         if expected is None:
#             assert fvar.is_empty
#         else:
#             assert +fvar == expected
    
#     def test_if_consistent(self):
#         """ 
#         Using if_ returns a copy of the fvar, but keeps the same underlying data, so 
#         the parent data will change also. However, with morphable objects, the current 
#         object might be morphed, but not the parent one, which might sometimes give wrong results, e.g. append to list can result in list of lists. Thus, we keep track of all parents from if and morph them too.
#         """
#         fvar = FVar(morphable=True)
#         2 >> fvar.if_(True)
#         3 >> fvar.if_(True)
#         4 >> fvar.if_(True)
#         5 >> fvar.if_(False)
#         assert isinstance(fvar, FList)
#         assert +fvar == [2, 3, 4] 

#     def test_select_return_type(self):
#         fvar = FVar(morphable=False)
#         selectable = fvar.select(X[1])
#         assert isinstance(selectable, FVar)
#         assert selectable is not fvar

#         [10, 20] >> selectable
#         assert selectable.value == 20

#         [10, 20] >> fvar
#         assert fvar.value == [10, 20]
    
#     def test_select(self):
#         fvar = FVar(morphable=False)
#         data = [1, 2,  3]
#         out = data >> fvar.select(X[1])
#         assert data is out
#         assert fvar.value == 2

#     def test_matmul(self):
#         """ `matmul` (@) operator is same as select method. """
#         fvar = FVar(morphable=False)
#         data = [1, 2, 3]
#         out = data >> fvar @ X[1]
#         assert data is out
#         assert fvar.value == 2
    
#     def test_select_raisesValueError_multiple_calls(self):
#         """ 
#         Can only call once select on a non-morphable FVar. 
#         """
#         fvar = FVar(morphable=False)
#         with pytest.raises(ValueError):
#             fvar @ X[1] @ X[2]
            
#     def test_select_raisesValueError_wrong_type(self):
#         """ 
#         Cannot select an expression to a strict FVar with existing value.
#         """
#         fvar = FVar(morphable=False)
#         with pytest.raises(ValueError):
#             fvar @ "foo"

#     def test_shed(self):
#         fvar = FVar(morphable=False)
#         [1, 2, 3] >> fvar @ X[1:]
#         assert fvar.shed() == [2, 3]
#         assert +fvar == [2, 3]

#     def test_shed_raisesValueError(self):
#         fvar = FVar()
#         with pytest.raises(ValueError):
#             fvar.shed()

#         with pytest.raises(ValueError):
#             fvar @ X[1]
#             fvar.shed()

#     def test_morph_to_list(self):
#         fvar = FVar(morphable=True)
#         2 >> fvar
#         assert isinstance(fvar, FVar)
#         assert +fvar == 2

#         3 >> fvar
#         assert isinstance(fvar, FList)
#         assert +fvar == [2, 3]

#     def test_morph_to_dict(self):
#         fvar = FVar(morphable=True)
#         2 >> fvar["a"]
#         3 >> fvar["b"]
#         assert isinstance(fvar, FDict)
#         assert +fvar == {"a": 2, "b": 3}

#     def test_morph_to_dict_error(self):
#         """ Cannot morph to dict if value already exists """
#         fvar = FVar(morphable=True)
#         2 >> fvar
#         with pytest.raises(ValueError):
#             3 >> fvar["a"]

#     def test_repr(self):
#         fvar = FVar()
#         2 >> fvar
#         assert str(fvar) == "FVar(2)"

    
# class TestFList:
#     """ 
#     Some of the common methods and code paths with `FVar` are not tested here, especially that
#     the operator / method overloading are consistent, since this is implemented at the base class 
#     level.
#     """
#     def test_init_no_args(self):
#         flist = FList()
#         assert +flist == []

#     def test_init_with_args(self):
#         flist = FList(1, 2, 3)
#         assert +flist == [1, 2, 3]
    
#     def test_rrshift(self):
#         flist = FList()
#         2 >> flist
#         3 >> flist
#         assert +flist == [2, 3]

#     def test_general_usage(self):
#         """ X Selector on left hand side does not work, must wrap it in Op. """
#         flist = FList()
#         [10, 20, 30] >> flist @ X[1] >> flist @ Op(X[2])
#         assert +flist == [20, 30]

#     def test_len(self):
#         flist = FList()
#         assert len(flist) == 0
#         1 >> flist
#         assert len(flist) == 1


# class TestFDict:    
#     def test_init_no_args(self):
#         fdict = FDict()
#         assert +fdict == {}
    
#     def test_init_with_args(self):
#         init_data = {"a": 1, "b": 2, "c": 3}
#         fdict = FDict(**init_data)
#         assert +fdict == init_data
    
#     def test_rrshift(self):
#         fdict = FDict(b=10)
#         2 >> fdict["a"]
#         assert +fdict == {"a": 2, "b": 10}

#     def test_rrshift_overwrite(self):
#         """ Raise error when overwriting existing key in strict mode. """
#         fdict = FDict(morphable=False)
#         2 >> fdict["a"]
#         assert +fdict == {"a": 2}
#         3 >> fdict["a"]
#         assert +fdict == {"a": 3}
        
#     def test_rrshift_morph(self):
#         fdict = FDict(morphable=True)
#         2 >> fdict["a"]
#         assert +fdict == {"a": 2}
#         4 >> fdict["b"]
#         assert +fdict == {"a": 2, "b": 4}

#         3 >> fdict["a"]
#         assert +fdict == {"a": [2, 3], "b": [4]}
#         assert isinstance(fdict, FMMap)
    
#         10 >> fdict["b"]
#         11 >> fdict["c"]
#         assert +fdict == {"a": [2, 3], "b": [4, 10], "c": [11]}
#         assert isinstance(fdict, FMMap)
    
#     def test_raises_key_error(self):
#         fdict = FDict()
#         with pytest.raises(KeyError):
#             2 >> fdict

#         # KeyError when accessing non-existent key. Make sure there is no 
#         # side effects due to inplace operations.
#         with pytest.raises(KeyError):
#             fdict["a"]
#             2 >> fdict
    
#     def test_general_usage(self):
#         fdict = FDict()
#         [1, 2, 3] >> fdict["a"] @ X[1] >> fdict["b"]
#         assert +fdict == {"a": 2, "b": [1, 2, 3]}

#     def test_len(self):
#         fdict = FDict()
#         assert len(fdict) == 0
#         1 >> fdict["a"]
#         assert len(fdict) == 1


# class TestFMMap:
#     def test_init_no_args(self):
#         fmmap = FMMap()
#         assert +fmmap == {}
    
#     def test_init_with_args(self):
#         fmmap = FMMap(a=[1, 2, 3], b=[4, 5, 6])
#         assert +fmmap == {"a": [1, 2, 3], "b": [4, 5, 6]}
        
#     def test_init_value_error(self):
#         with pytest.raises(ValueError):
#             FMMap(a=[1, 2, 3], b=4)
    
#     def test_general_usage(self):
#         fmmap = FMMap()
#         out = (
#             [1, 2, 3] 
#             >> fmmap["a"] @ X[1] 
#             >> fmmap["a"] @ X[2] 
#             >> fmmap["b"] @ X[0]
#         )
#         assert +fmmap == {"a": [2, 3], "b": [1]}


# class Test_Variable:
#     def test_repr(self):
#         from faeyon.magic.spells import _Variable
#         var = _Variable(10)
#         assert str(var) == "10"


# class TestOp:
#     @staticmethod
#     def func(x: list) -> list:
#         return [i + 1 for i in x]

#     def test_init_error1(self):
#         """ Cannot initialize Op with more than one argument if first argument is X. """
#         with pytest.raises(ValueError):
#             Op(X, 1, 2)
        
#         with pytest.raises(ValueError):
#             Op(X, a=2)
    
#     def test_init_error4(self):
#         """ Unsupported argument type"""
#         with pytest.raises(ValueError):
#             Op(1)
    
#     def test_rshift(self):
#         delayed = Op(X[1:]) >> Op(X[1:])
#         out = torch.tensor([1, 2, 3]) >> delayed
#         assert isinstance(delayed, Serials)
#         torch.testing.assert_close(out, torch.tensor([3]))

#     def test_rrshift_x(self):
#         data = [1, 2, 3]
#         out = data >> Op(X[1:])
#         assert out == [2, 3]
    
#     def test_rrshift_callable(self):
#         expected = torch.tensor([1, 2, 3])
#         out = [1, 2, 3] >> Op(torch.tensor, X)
#         torch.testing.assert_close(out, expected)

#     def test_lshift_x_x(self):
#         data = [1, 2, 3]
#         delayed = Op(X[1:]) << Op(X[1:])
#         assert isinstance(delayed, Parallels)
#         out = data >> delayed
#         assert out == [3]

#     def test_lshift_x_callable(self):
#         out = [1, 2, 3] >> (Op(X[1:]) << Op(self.func, X))
#         assert out == [3, 4]

#     def test_lshift_x_serials(self):
#         delayed = Op(X[1:]) << (Op(self.func, X) >> Op(X + [10]))
#         out = [1, 2, 3] >> delayed
#         assert out == [3, 4, 10]

#     def test_lshift_opserial_opx(self):
#         delayed = Op(X[1:]) >> Op(self.func, X + [10]) << Op(X[1:])
#         out = [1, 2, 3] >> delayed
#         assert out == [4, 11]
    
#     def test_lshift_opserial_opserial(self):
#         delayed1 = Op(X[1:]) >> Op(self.func, X + [10])
#         delayed2 = Op(X[1:]) >> Op(self.func, X + [20])

#         out = [1, 2, 3] >> (delayed1 << delayed2)
#         assert out == [5, 12, 21]
    
#     def test_lshift_data_error(self):
#         """ No support for << with data input"""
#         with pytest.raises(TypeError):
#             [1, 2, 3] << Op(X)

#     @pytest.mark.parametrize("condition,else_,expected", [
#         (True, None, [2, 3]),
#         (False, None, [1, 2, 3]),
#         (True, Op(X[1:]), [2, 3]),
#         (False, Op(X[0]), 1),
#         (Op(X[0] > 1), None, [1, 2, 3]),
#         (Op(X[0] < 1), Op(X[0] - 1), 0),
#     ])
#     def test_if_(self, condition, else_, expected):
#         data = [1, 2, 3]
#         delayed = Op(X[1:]).if_(condition, else_=else_)
#         out = data >> delayed
#         assert out == expected

#     @pytest.mark.parametrize("delayed,expected,data", [
#         (Op(X[1:]) + Op(X[:-1]), [3, 5], [1, 2, 3]),
#         (Op(X[1:]) - Op(X[:-1]), [1, 1], [1, 2, 3]),
#         (Op(X[1:]) * Op(X[:-1]), [2, 6], [1, 2, 3]),
#         (Op(X[1:]) / Op(X[:-1]), [2.0, 1.5], [1, 2, 3]),
#         (Op(X[1:]) // Op(X[:-1]), [2, 1], [1, 2, 3]),
#         (Op(X) @ (ConstantLayer(2, value=2.0) >> Op(X[None].T)), [10.], [1.0, 2.0]),
#         (Op(X[1:]) % Op(X[:-1]), [0, 1], [1, 2, 3]),
#         (Op(X[1:]) ** Op(X[:-1]), [2, 9], [1, 2, 3]),
#         (Op(X[1:]) & Op(X[:-1]), [0, 2], [1, 2, 3]),
#         #(Op(X[1:]) | Op(X[:-1]), [3, 3], [1, 2, 3]),
#         (Op(X[1:]) ^ Op(X[:-1]), [3, 1], [1, 2, 3])
#     ])
#     def test_binary_operators_oo(self, delayed, expected, data):
#         """ math operator between two `Op` objects"""
#         assert isinstance(delayed, Op)
#         out = torch.tensor(data) >> delayed
#         torch.testing.assert_close(out, torch.tensor(expected))

#     @pytest.mark.parametrize("delayed,expected", [
#         (-Op(X), [-1, -2, 3]),
#         (+Op(X), [1, 2, -3]),
#         (abs(Op(X)), [1, 2, 3]),
#         (~Op(X), [-2, -3, 2]),
#     ])
#     def test_unary_operators_oo(self, delayed, expected):
#         data = torch.tensor([1, 2, -3], dtype=torch.int64)
#         out = data >> delayed
#         assert isinstance(delayed, Op)
#         torch.testing.assert_close(out, torch.tensor(expected))
    
#     def test_repr(self):
#         op = Op(X[1].a)
#         assert str(op) == "Op(X[1].a)"


# class TestParallels:
#     @pytest.mark.parametrize("delayed,expected", [
#         # simple input ok
#         (Parallels([X, X], X), [1, 2, 3]),
#         # same size lists of ops
#         (Parallels([Op(2 * X), Op(3 * X)], [Op(X[1:]), Op(X / 2)]), [6.0, 9.0]),
#         # broadcastable types
#         (Parallels([Op(2 * X), Op(3 * X)], [Op(X[1:])]), [18]),
#         (Parallels([Op(2 * X), Op(3 * X)], Op(X[1:])), [18]),
#         (Parallels([Op(2 * X), Op(3 * X)], X[1:]), [18]),
#         (Parallels([Op(2 * X), Op(3 * X)], ConstantLayer(3, value=2.0)), [24.0, 48.0, 72.0]),
#         # Mixed types in list
#         (Parallels([Op(2 * X), 3 * X], [Op(X[1:]), ConstantLayer(2, value=0.5)]), [6.0, 9.0]),
#         # Parallels
#         (Parallels(
#             Parallels([Op(2 * X), Op(3 * X)], [Op(X[1:]), Op(X / 2)]), X + 1), [8.5, 11.5]),
#         # Parallels with size 1 (broadcasted)
#         (Parallels(Parallels(Op(2 * X), Op(X[1:])), [X + 1, X + 2]), [16]),

#         # Parallels with custom function
#          (Parallels([Op(2 * X), Op(3 * X)], Op(X + 1), func=lambda a, b: a + b), [17, 29, 41]),
#     ])
#     def test_init(self, delayed, expected):
#         """ List of ops with same size. """
#         out = torch.tensor([1, 2, 3]) >> delayed
#         assert len(delayed) == 2
#         torch.testing.assert_close(out, torch.tensor(expected))
           
#     @pytest.mark.parametrize("args", [
#         # unbroadcastable sizes
#         ([Op(2 * X), Op(3 * X)], [Op(X), Op(X), X]),
#         # unknown types
#         ([Op(X)], [1, 2, 3]),
#         (10, X)
#     ])
#     def test_init_error(self, args):
#         """ """
#         with pytest.raises(ValueError):
#             Parallels(*args)

#     @pytest.mark.parametrize("delayed,expected", [
#         # X << Parallels
#         (X + 1 << Parallels([2 * X, 3 * X], [X + 1, X[1:]]), [24, 30]),
#         # OpX << Parallels
#         (Op(X + 1) << Parallels([2 * X, 3 * X], [X + 1, X[1:]]), [24, 30]),
#         # OpCallable << Parallels
#         (Op(lambda x: x + 1, X) << Parallels([2 * X, 3 * X], [X + 1, X[1:]]), [24, 30]),
#         # Serials << Parallels
#         (Op(X + 2) >> Op(X - 1) << Parallels([2 * X, 3 * X], [X + 1, X[1:]]), [24, 30]),
#         # Parallels << Parallels
#         (Parallels(Op(X + 2), Op(X - 1)) << Parallels([2 * X, 3 * X], [X + 1, X[1:]]), [24, 30]),
        
#         # Parallels << X
#         ( Parallels([2 * X, 3 * X], [X + 1, X[1:]]) << X + 1, [19, 25]),
#         # OpX << Parallels
#         (Parallels([2 * X, 3 * X], [X + 1, X[1:]]) << Op(X + 1), [19, 25]),
#         # # OpCallable << Parallels
#         (Parallels([2 * X, 3 * X], [X + 1, X[1:]]) << Op(lambda x: x + 1, X), [19, 25]),
#         # # Serials << Parallels
#         (Parallels([2 * X, 3 * X], [X + 1, X[1:]]) << (Op(X + 2) >> Op(X - 1)), [19, 25]),
#         # # Parallels << Parallels
#         (Parallels([2 * X, 3 * X], [X + 1, X[1:]]) << Parallels(Op(X + 2), Op(X - 1)), [19, 25]),
#     ])
#     def test_lshift(self, delayed, expected):
#         """
#         X << OP([X1, X2]) ---> X >> X1 >> X >> X2
#         """
#         out = torch.tensor([1, 2, 3]) >> delayed
#         assert isinstance(delayed, Parallels)
#         assert len(delayed) == 2
#         torch.testing.assert_close(out, torch.tensor(expected))

#     @pytest.mark.parametrize("delayed,expected", [
#         (-Parallels([2 * X, X - 1], X[1:]), [7]),
#         (+Parallels([2 * X, X - 1], X[1:]), [5]),
#         (abs(Parallels([2 * X, X - 1], X[1:])), [5]),
#         (~Parallels([2 * X, X - 1], X[1:]), [7]),
#     ])
#     def test_unary_operators(self, delayed, expected):
#         out = torch.tensor([1, 2, 3]) >> delayed
#         assert isinstance(delayed, Parallels)
#         assert len(delayed) == 2
#         torch.testing.assert_close(out, torch.tensor(expected))

#     @pytest.mark.parametrize("delayed,expected", [
#         # Parallels with Parallels
#         (Parallels(2 * X[1:]) + Parallels([2 * X, X - 1], X[1:]), [35]),
#         (Parallels(2 * X[1:]) - Parallels([2 * X, X - 1], X[1:]), [1]),
#         (Parallels(2 * X[1:]) * Parallels([2 * X, X - 1], X[1:]), [2520]),
#         (Parallels(2 * X[1:]) / Parallels([2 * X, X + 1], X[1:]), [1.]),
#         (Parallels(2 * X[1:]) // Parallels([2 * X, X + 1], X[1:]), [1]),
#         (Parallels(2 * X[1:]) % Parallels([2 * X, X - 1], X[1:]), [0]),
#         (Parallels(2 * X[1:]) ** Parallels([X, X - 216.0], X[1:]), [1.0]),
#         (Parallels(2.0 * X[None]) @ Parallels([2 * X, X - 1], X / 2.0), [756.0]),
#         # (Parallels(2 * X[1:]) | Parallels([2 * X, X - 1], X[1:]), [13]),
#         (Parallels(2 * X[1:]) ^ Parallels([2 * X, X - 1], X[1:]), [-1]),

#         # Ops with Parallels
#         (Op(2 * X[1:]) + Parallels([2 * X, X - 1], X[1:]), [35]),
#         (Op(2 * X[1:]) - Parallels([2 * X, X - 1], X[1:]), [1]),
#         (Op(2 * X[1:]) * Parallels([2 * X, X - 1], X[1:]), [2520]),
#         (Op(2 * X[1:]) / Parallels([2 * X, X + 1], X[1:]), [1.]),
#         (Op(2 * X[1:]) // Parallels([2 * X, X + 1], X[1:]), [1]),
#         (Op(2 * X[1:]) % Parallels([2 * X, X - 1], X[1:]), [0]),
#         (Op(2 * X[1:]) ** Parallels([X, X - 216.0], X[1:]), [1.0]),
#         (Op(2.0 * X[None]) @ Parallels([2 * X, X - 1], X / 2.0), [756.0]),
#         (Op(2 * X[1:]) & Parallels([2 * X, X - 1], X[1:]), [4]),
#         #(Op(2 * X[1:]) | Parallels([2 * X, X - 1], X[1:]), [13]),
#         (Op(2 * X[1:]) ^ Parallels([2 * X, X - 1], X[1:]), [-1]),

#         # parallels with Ops
#         (Parallels([2 * X, X - 1], X[1:]) + Op(2 * X[1:]), [35]),
#         (Parallels([2 * X, X - 1], X[1:]) - Op(2 * X[1:]) , [-1]),
#         (Parallels([2 * X, X - 1], X[1:]) * Op(2 * X[1:]) , [2520]),
#         (Parallels([2 * X, X + 1], X[1:]) / Op(2 * X[1:]) , [1.0]),
#         (Parallels([2 * X, X + 1], X[1:]) // Op(2 * X[1:]) , [1]),
#         (Parallels([2 * X, X - 1], X[1:]) % Op(2 * X[1:] - 1) , [0]),
#         (Parallels([X, X - 728.0], X[1:]) ** Op(2 * X[1:]) , [1.]),
#         (Parallels([2 * X, X - 1], X / 2.0) @ Op(2.0 * X[None].T), [756.]),
#         (Parallels([2 * X, X - 1], X[1:]) & Op(2 * X[1:]), [4]),
#         # (Parallels([2 * X, X - 1], X[1:]) | Op(2 * X[1:]), [13]),
#         (Parallels([2 * X, X - 1], X[1:]) ^ Op(2 * X[1:]), [-1]),
#     ])
#     def test_binary_operators(self, delayed, expected):
#         out = torch.tensor([1, 2, 3]) >> delayed
#         assert isinstance(delayed, Parallels)
#         assert len(delayed) == 2
#         torch.testing.assert_close(out, torch.tensor(expected))

#     @pytest.mark.parametrize("condition, else_, expected", [
#         (True, None, [8]),
#         (False, None, [1, 2, 3]),
#         (True, Parallels([X + 1, X]), [8]),
#         (False, Parallels([X + 1, X]), [2, 3, 4]),
#         (False, Parallels([Op(X + 1), Op(X + 2)]), [4, 5, 6]),
#     ])
#     def test_if(self, condition, else_, expected):
#         delayed = Parallels([Op(X + 1), Op(2 * X)], X[1:]).if_(condition, else_)
#         res = torch.tensor([1, 2, 3]) >> delayed
#         torch.testing.assert_close(res, torch.tensor(expected))

#     def test_if_nested(self):
#         delayed = Parallels(Parallels(Op(X + 1)).if_(False), Op(2 * X))
#         res = torch.tensor([1, 2, 3]) >> delayed
#         torch.testing.assert_close(res, torch.tensor([2, 4, 6]))

#     def test_if_nested_with_else(self):
#         delayed = Parallels(
#             Parallels([Op(X + 1), Op(X + 2)]).if_(False, Parallels([X + 1, X])), 
#             Op(2 * X)
#         )
#         res = torch.tensor([1, 2, 3]) >> delayed
#         torch.testing.assert_close(res, torch.tensor([8, 12, 16]))


# class TestWire:
#     @staticmethod
#     def example_func(x: int, y: int) -> int:
#         return x + y

#     @pytest.fixture(scope="class")
#     def sig(self):
#         return inspect.signature(self.example_func)

#     def test_init(self, sig):
#         wire = Wire(X)
#         out = wire.init(sig, 1, 2)
#         assert wire._fanout == {}
#         assert isinstance(out, A)
#         assert out.args == (1, 2)
#         assert out.kwargs == {}
    
#     def test_init_with_fanout(self, sig):
#         wire = Wire(x=X, y=W.Fanout)
#         out = wire.init(sig, 1, y=[10, 11])
#         assert set(wire._fanout.keys()) == {"y"}
#         assert out.args == (1, 10)
#         assert out.kwargs == {}

#     def test_init_with_passthru(self, sig):
#         wire = Wire(x=W.Pass, y=W.Pass)
#         out = wire.init(sig, 1, 2)
#         assert wire._fanout == {}
#         assert out.args == (1, 2)
#         assert out.kwargs == {}
        
#     def test_step(self, sig):
#         wire = Wire(X)
#         wire.init(sig, 1, 2)
#         out = wire.step(3)
#         assert out.args == (3, 2)
#         assert out.kwargs == {}
    
#         out = wire.step(4)
#         assert out.args == (4, 2)
#         assert out.kwargs == {}

#     def test_step_with_fanout(self, sig):
#         wire = Wire(x=X, y=W.Fanout)
#         wire.init(sig, 1, y=[10, 11])
#         out = wire.step(2)        
#         assert out.args == (2, 11)
#         assert out.kwargs == {}
    
#     def test_rrshift(self, sig):
#         wire = Wire(X)
#         wire.init(sig, 1, 2)
#         out = 3 >> wire
#         assert out.args == (3, 2)
#         assert out.kwargs == {}

#     def test_step_no_init(self):
#         wire = Wire(X)
#         with pytest.raises(ValueError):
#             wire.step(1)

#     def test_step_with_fanout_overflow(self, sig):
#         wire = Wire(x=X, y=W.Fanout)
#         wire.init(sig, 1, y=[10, 11])
#         with pytest.raises(ValueError):
#             wire.step(2)
#             wire.step(3)
#             wire.step(4)


# def test_conjure():
#     class Something:
#         def __init__(self, a):
#             self.a = a
    
#     data = [Something(1), Something(2)]
#     res = conjure(X[1].a, data)
#     assert res == 2


# def test_conjure_with_non_x():
#     """Test conjure with non-X input returns the input as is"""
#     test_data = [1, 2, 3]
#     result = conjure(test_data, None)
#     assert result is test_data


# def test_conjure_with_nested_x():
#     """Test conjure with nested X operations"""
#     class NestedData:
#         def __init__(self, value):
#             self.value = value
#             self.items = [self.value * i for i in range(1, 4)]
    
#     data = NestedData(10)
#     result = conjure(X.items[1], data)
#     assert result == 20


# class TestNoValue:
#     def test_value_property(self):
#         """Test that value property returns self"""
#         from faeyon.magic.spells import _NoValue
#         no_value = _NoValue()
#         assert no_value.value is no_value
    
#     def test_repr(self):
#         """Test string representation of _NoValue"""
#         from faeyon.magic.spells import _NoValue
#         assert repr(_NoValue()) == "<NO_VALUE>"
#         assert str(_NoValue()) == "<NO_VALUE>"


# class TestVariable:
#     def test_init_no_args(self):
#         """Test _Variable initialization with no arguments"""
#         from faeyon.magic.spells import _Variable, _NoValue
#         var = _Variable()
#         assert isinstance(var.value, _NoValue)
    
#     def test_init_one_arg(self):
#         """Test _Variable initialization with one argument"""
#         from faeyon.magic.spells import _Variable
#         test_value = "test_value"
#         var = _Variable(test_value)
#         assert var.value == test_value
    
#     def test_init_multiple_args_raises(self):
#         """Test _Variable initialization with multiple arguments raises ValueError"""
#         from faeyon.magic.spells import _Variable
#         with pytest.raises(ValueError, match="can only be initialized with one or no arguments"):
#             _Variable(1, 2, 3)
    
#     def test_has_value(self):
#         """Test has_value method of _Variable"""
#         from faeyon.magic.spells import _Variable, _NoValue
#         var1 = _Variable()
#         assert not var1.has_value()
        
#         var2 = _Variable(42)
#         assert var2.has_value()
    
#     def test_repr(self):
#         """Test string representation of _Variable"""
#         from faeyon.magic.spells import _Variable
#         var = _Variable("test")
#         assert repr(var) == "'test'"


# class TestWiring:
#     def test_wiring_abstract_base_class(self):
#         """Test that Wiring is an abstract base class"""
#         from faeyon.magic.spells import Wiring
#         with pytest.raises(TypeError, match="Can't instantiate abstract class"):
#             Wiring()  # type: ignore


# class TestFanout:
#     def test_fanout_getitem(self):
#         """Test _Fanout's __getitem__ method"""
#         from faeyon.magic.spells import _Fanout
#         test_list = [1, 2, 3, 4, 5]
#         fanout = _Fanout(test_list)
        
#         for i in range(len(test_list)):
#             assert fanout[i] == test_list[i]
        
#         # Test with negative indices
#         assert fanout[-1] == test_list[-1]


# class TestPass:
#     def test_pass_getitem(self):
#         """Test _Pass's __getitem__ always returns the same object"""
#         from faeyon.magic.spells import _Pass
#         test_obj = object()
#         passthrough = _Pass(test_obj)
        
#         for i in range(5):
#             assert passthrough[i] is test_obj


# class TestMux:
#     def test_mux_getitem(self):
#         """Test _Mux's __getitem__ returns s0 for key 0, s1 otherwise"""
#         from faeyon.magic.spells import _Mux
#         s0 = object()
#         s1 = object()
#         mux = _Mux(s0, s1)
        
#         assert mux[0] is s0
#         assert mux[1] is s1
#         assert mux[42] is s1
#         assert mux[-1] is s1


# class TestW:
#     def test_w_enum_values(self):
#         """Test W enum has expected values"""
#         from faeyon.magic.spells import W, _Fanout, _Pass, _Mux
        
#         # Test enum values
#         assert W.Fanout.value == "Fanout"
#         assert W.Pass.value == "Pass"
#         assert W.Mux.value == "Mux"
        
#         # Test __call__ returns correct class instances
#         assert isinstance(W.Fanout("test"), _Fanout)
#         assert isinstance(W.Pass("test"), _Pass)
        
#         # Test _Mux requires exactly 2 arguments
#         with pytest.raises(TypeError):
#             W.Mux("test")  # type: ignore
            
#         mux = W.Mux("s0", "s1")
#         assert isinstance(mux, _Mux)
#         assert mux.s0 == "s0"
#         assert mux.s1 == "s1"
