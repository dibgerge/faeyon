import inspect
import torch
from faeyon import A, X, FVar, FList, FDict, FMMap, Op, Wire, Wiring
from faeyon.magic.spells import conjure
import pytest
from tests.common import ConstantLayer
from faeyon.magic.spells import _OpParallel, _OpCallable


class TestX:

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

    def test_repr_noops(self):
        assert str(X) == "X"

    def test_repr_multiops(self):
        x = round(X("foo", k="bar")[0].a + 1)
        assert repr(x) == "round(X('foo', k='bar')[0].a + 1)"


class TestA:

    def func_simple(self, x):
        return x + 1

    def func_multi(self, x=1, y=0):
        return x + y
    
    def test_init(self):
        fae = A(1, 2, 3)
        assert isinstance(fae, A)
        assert fae.args == (1, 2, 3)
        assert fae.kwargs == {}

    def test_call_raisesTypeError(self):
        """ 
        When `A` does not match the callable's required number of arguments.
        """
        fae_args = A(1, 2, 3)

        with pytest.raises(TypeError):
            fae_args.call(self.func_simple)
        
        with pytest.raises(TypeError):
            fae_args >> self.func_simple

    def test_call_unresolved(self):
        """ 
        When `A` has unresolved arguments, an error is raised.
        """
        fae_args = A(X[0])
        delayed = fae_args >> self.func_simple
        assert isinstance(delayed, Op)
        res = [1, 2, 3] >> delayed
        assert res == 2

    @pytest.mark.parametrize("args,kwargs", [
        ((), {}),
        ((2,), {}),
        ((), {"y": 10}),
        ((1,), {"y": 10}),
    ])
    def test_call(self, args, kwargs):
        """ Tests the call (A >> callable) operator/method. """
        fae_args = A(*args, **kwargs)
        expected = self.func_multi(*args, **kwargs)
        assert fae_args.call(self.func_multi) == expected
        assert fae_args >> self.func_multi == expected

    def test_using_resolved(self):
        """ Tests the bind (Any >> A) operator when `A` is already resolved. """
        fae_args = A(1, x="Bar")
        data = "Foo"
        out_args = fae_args.using(data)
        assert out_args.args == (1,)
        assert out_args.kwargs == {"x": "Bar"}
        
        out_args = data >> fae_args
        assert out_args.args == (1,)
        assert out_args.kwargs == {"x": "Bar"}

    def test_using_unresolved(self):
        fea_args = A(X[0], x="Bar", y=X[1])
        data = [10, 11, 12]

        out_args = fea_args.using(data)
        assert out_args.args == (10,)
        assert out_args.kwargs == {"x": "Bar", "y": 11}

        out_args = data >> fea_args
        assert out_args.args == (10,)
        assert out_args.kwargs == {"x": "Bar", "y": 11}


class TestFVar:
        
    def test_rrshift(self):
        fvar = FVar()
        2 >> fvar
        assert +fvar == 2

    def test_rrshift_overwrite(self):
        """ 
        Cannot bind a value to a strict FVar with existing value.
        """
        fvar = FVar(morphable=False)
        2 >> fvar
        3 >> fvar
        assert +fvar == 3

    def test_rrshift_morph(self):
        fvar = FVar(morphable=True)
        out = 2 >> fvar
        assert +fvar == 2
        assert out == 2

        out = 3 >> fvar
        assert isinstance(fvar, FList)
        assert fvar.value == [2, 3]
        assert out == 3
        
    def test_using(self):
        """ `using` method is same as >> operator. """
        fvar = FVar(morphable=False)
        out = fvar.using(2)
        assert +fvar == 2
        assert out == 2

    def test_if_on_X(self):
        fvar = FVar(morphable=False)
        data = [1, 2, 3]
        out = data >> fvar.if_(X[0] > 1) @ X[1]
        assert data is out
        assert fvar.is_empty

    @pytest.mark.parametrize("condition, expected", [
        (True, 2),
        (False, None),
        (Op(X[0] == 1), 2),
        (Op(X[0] != 1), None),
    ])
    def test_if_general(self, condition, expected):
        fvar = FVar(morphable=False)
        data = [1, 2, 3]
        out = data >> fvar.if_(condition) @ X[1]
        assert data is out
        if expected is None:
            assert fvar.is_empty
        else:
            assert +fvar == expected

    def test_select_return_type(self):
        fvar = FVar(morphable=False)
        selectable = fvar.select(X[1])
        assert isinstance(selectable, FVar)
        assert selectable is not fvar

        [10, 20] >> selectable
        assert selectable.value == 20

        [10, 20] >> fvar
        assert fvar.value == [10, 20]
    
    def test_select(self):
        fvar = FVar(morphable=False)
        data = [1, 2,  3]
        out = data >> fvar.select(X[1])
        assert data is out
        assert fvar.value == 2

    def test_matmul(self):
        """ `matmul` (@) operator is same as select method. """
        fvar = FVar(morphable=False)
        data = [1, 2, 3]
        out = data >> fvar @ X[1]
        assert data is out
        assert fvar.value == 2
    
    def test_select_raisesValueError_multiple_calls(self):
        """ 
        Can only call once select on a strict FVar. 
        """
        fvar = FVar(morphable=False)
        with pytest.raises(ValueError):
            fvar @ X[1] @ X[2]
            
    def test_select_raisesValueError_wrong_type(self):
        """ 
        Cannot select an expression to a strict FVar with existing value.
        """
        fvar = FVar(morphable=False)
        with pytest.raises(ValueError):
            fvar @ "foo"

    def test_shed(self):
        fvar = FVar(morphable=False)
        [1, 2, 3] >> fvar @ X[1:]
        assert fvar.shed() == [2, 3]
        assert +fvar == [2, 3]

    def test_shed_raisesValueError(self):
        fvar = FVar()
        with pytest.raises(ValueError):
            fvar.shed()

        with pytest.raises(ValueError):
            fvar @ X[1]
            fvar.shed()

    def test_morph_to_list(self):
        fvar = FVar(morphable=True)
        2 >> fvar
        assert isinstance(fvar, FVar)
        assert +fvar == 2

        3 >> fvar
        assert isinstance(fvar, FList)
        assert +fvar == [2, 3]

    def test_morph_to_dict(self):
        fvar = FVar(morphable=True)
        2 >> fvar["a"]
        3 >> fvar["b"]
        assert isinstance(fvar, FDict)
        assert +fvar == {"a": 2, "b": 3}

    def test_morph_to_dict_error(self):
        """ Cannot morph to dict if value already exists """
        fvar = FVar(morphable=True)
        2 >> fvar
        with pytest.raises(ValueError):
            3 >> fvar["a"]

    def test_repr(self):
        fvar = FVar()
        2 >> fvar
        assert str(fvar) == "FVar(2)"

    
class TestFList:
    """ 
    Some of the common methods and code paths with `FVar` are not tested here, especially that
    the operator / method overloading are consistent, since this is implemented at the base class 
    level.
    """
    def test_init_no_args(self):
        flist = FList()
        assert +flist == []

    def test_init_with_args(self):
        flist = FList(1, 2, 3)
        assert +flist == [1, 2, 3]
    
    def test_rrshift(self):
        flist = FList()
        2 >> flist
        3 >> flist
        assert +flist == [2, 3]

    def test_general_usage(self):
        """ X Selector on left hand side does not work, must wrap it in Op. """
        flist = FList()
        [10, 20, 30] >> flist @ X[1] >> flist @ Op(X[2])
        assert +flist == [20, 30]

    def test_len(self):
        flist = FList()
        assert len(flist) == 0
        1 >> flist
        assert len(flist) == 1


class TestFDict:    
    def test_init_no_args(self):
        fdict = FDict()
        assert +fdict == {}
    
    def test_init_with_args(self):
        init_data = {"a": 1, "b": 2, "c": 3}
        fdict = FDict(**init_data)
        assert +fdict == init_data
    
    def test_rrshift(self):
        fdict = FDict(b=10)
        2 >> fdict["a"]
        assert +fdict == {"a": 2, "b": 10}

    def test_rrshift_overwrite(self):
        """ Raise error when overwriting existing key in strict mode. """
        fdict = FDict(morphable=False)
        2 >> fdict["a"]
        assert +fdict == {"a": 2}
        3 >> fdict["a"]
        assert +fdict == {"a": 3}
        
    def test_rrshift_morph(self):
        fdict = FDict(morphable=True)
        2 >> fdict["a"]
        assert +fdict == {"a": 2}
        4 >> fdict["b"]
        assert +fdict == {"a": 2, "b": 4}

        3 >> fdict["a"]
        assert +fdict == {"a": [2, 3], "b": [4]}
        assert isinstance(fdict, FMMap)
    
        10 >> fdict["b"]
        11 >> fdict["c"]
        assert +fdict == {"a": [2, 3], "b": [4, 10], "c": [11]}
        assert isinstance(fdict, FMMap)
    
    def test_raises_key_error(self):
        fdict = FDict()
        with pytest.raises(KeyError):
            2 >> fdict

        # KeyError when accessing non-existent key. Make sure there is no 
        # side effects due to inplace operations.
        with pytest.raises(KeyError):
            fdict["a"]
            2 >> fdict
    
    def test_general_usage(self):
        fdict = FDict()
        [1, 2, 3] >> fdict["a"] @ X[1] >> fdict["b"]
        assert +fdict == {"a": 2, "b": [1, 2, 3]}

    def test_len(self):
        fdict = FDict()
        assert len(fdict) == 0
        1 >> fdict["a"]
        assert len(fdict) == 1


class TestFMMap:
    def test_init_no_args(self):
        fmmap = FMMap()
        assert +fmmap == {}
    
    def test_init_with_args(self):
        fmmap = FMMap(a=[1, 2, 3], b=[4, 5, 6])
        assert +fmmap == {"a": [1, 2, 3], "b": [4, 5, 6]}
        
    def test_init_value_error(self):
        with pytest.raises(ValueError):
            FMMap(a=[1, 2, 3], b=4)
    
    def test_general_usage(self):
        fmmap = FMMap()
        out = (
            [1, 2, 3] 
            >> fmmap["a"] @ X[1] 
            >> fmmap["a"] @ X[2] 
            >> fmmap["b"] @ X[0]
        )
        assert +fmmap == {"a": [2, 3], "b": [1]}


class Test_Variable:
    def test_repr(self):
        from faeyon.magic.spells import _Variable
        var = _Variable(10)
        assert str(var) == "10"


class TestOp:
    @staticmethod
    def func(x: list) -> list:
        return [i + 1 for i in x]

    def test_init_error1(self):
        """ Cannot initialize Op with more than one argument if first argument is X. """
        with pytest.raises(ValueError):
            Op(X, 1, 2)
        
        with pytest.raises(ValueError):
            Op(X, a=2)
    
    def test_init_error2(self):
        """ If first argument is `Op`, all arguments must be `Op` as well. """
        with pytest.raises(ValueError):
            Op(Op(X), 1)
        
    def test_init_error3(self):
        """ Cannot have kwargs ig Op arugments are given """
        with pytest.raises(ValueError):
            Op(Op(X), Op(X), a=1)
    
    def test_init_error4(self):
        """ Unsupported argument type"""
        with pytest.raises(ValueError):
            Op(1)
    
    def test_usage(self):
        data = [1, 2, 3]
        out = data >> Op(X)
        assert out == data

    def test_usage2(self):
        data = [1, 2, 3]
        out = data >> Op(X[1:])
        assert out == [2, 3]

    def test_usage_callable(self):
        expected = torch.tensor([1, 2, 3])
        out = [1, 2, 3] >> Op(torch.tensor, X)
        torch.testing.assert_close(out, expected)

    def test_lshift_opx_opx(self):
        data = [1, 2, 3]
        delayed = Op(X[1:]) << Op(X[1:])
        assert isinstance(delayed, Op)
        assert isinstance(delayed.strategy, _OpParallel)
        out = data >> delayed
        assert out == [3]

    def test_lshift_opx_opcallable(self):
        out = [1, 2, 3] >> (Op(X[1:]) << Op(self.func, X))
        assert out == [3, 4]

    def test_lshift_opx_opserial(self):
        delayed = Op(X[1:]) << (Op(self.func, X) >> Op(X + [10]))
        out = [1, 2, 3] >> delayed
        assert out == [3, 4, 10]

    def test_lshift_opx_oparallel(self):
        """
        Op(X) << OP([X1, X2]) ---> X >> X1 >> X >> X2
        """
        delayed = (
            Op(X[1:]) 
            << Op([
                Op(self.func, X) >> Op(self.func, X),
                Op(self.func, X) >> Op(X + [10]),
            ])
        )
        out = [1, 2, 3] >> delayed
        assert out == [6, 10]

    def test_lshift_opserial_opx(self):
        delayed = Op(X[1:]) >> Op(self.func, X + [10]) << Op(X[1:])
        out = [1, 2, 3] >> delayed
        assert out == [4, 11]
    
    def test_lshift_opserial_opserial(self):
        delayed1 = Op(X[1:]) >> Op(self.func, X + [10])
        delayed2 = Op(X[1:]) >> Op(self.func, X + [20])

        out = [1, 2, 3] >> (delayed1 << delayed2)
        assert out == [5, 12, 21]
    
    def test_lshift_opserial_opparallel(self):
        delayed = (
            Op(X[1:]) 
            >> Op(self.func, X + [10])
            << Op([Op(self.func, X), Op(self.func, X)]))

        out = [1, 2, 3] >> delayed
        assert out == [6, 7, 14, 12]
    
    def test_lshift_opparallel_opx(self):
        delayed = Op([Op(self.func, X), Op(self.func, X)]) << Op(X[1:])
        out = [1, 2, 3] >> delayed
        assert out == [5]
        
    def test_lshift_opparallel_opserial(self):
        """f(x) >> X[1:] >> f(x) >> X[1:] >> f(x + [10])"""
        delayed = (
            Op([Op(self.func, X), Op(self.func, X)])
            << (
                Op(X[1:]) 
                >> Op(self.func, X + [10])
            )
        )
        out = [1, 2, 3] >> delayed
        assert out == [6, 11]

    def test_lshift_opparallel_opparallel(self):
        """f(x) >> X[1:] >> f(x) >> f(x + [10])"""
        delayed = (
            Op([Op(self.func, X), Op(self.func, X)])
            << Op([Op(X[1:]), Op(self.func, X + [10])])
        )
        out = [1, 2, 3] >> delayed
        assert out == [5, 6, 11] 

    def test_lshift_opparallel_opparallel_broadcast(self):
        """ f(x) >> X[1:] >> f(x) >> f(x + [10])"""
        delayed = (
            Op([Op(self.func, X)])
            << Op([Op(X[1:]), Op(self.func, X + [10])])
        )
        out = [1, 2, 3] >> delayed
        assert out == [5, 6, 11] 

    def test_lshift_opparallel_opparallel_err(self):
        """ Cannot have incompatible sizes """
        with pytest.raises(ValueError):
            delayed = (
                Op([Op(X[1:]), Op(X + [10])]) 
                << Op([Op(X[1:]), Op(self.func, X), Op(X[1:])])
            )

    def test_lshift_module_op(self):
        delayed = ConstantLayer(3, value=1.0) << Op(X[1:])
        out = torch.tensor([1, 2, 3]) >> delayed
        torch.testing.assert_close(out, torch.tensor([2., 3.]))

    def test_lshift_op_module(self):
        delayed = Op(X[1:]) << ConstantLayer(2, value=1.0)
        out = torch.tensor([1, 2, 3]) >> delayed
        torch.testing.assert_close(out, torch.tensor([2., 3.]))

    def test_lshift_data_error(self):
        """ No support for << with data input"""
        with pytest.raises(TypeError):
            [1, 2, 3] << Op(X)

    def test_lshift_if_error(self):
        """ Cannot have condition when using << on Serial Op"""
        with pytest.raises(ValueError):
            sop = (Op(X[1:]) >> Op(X[1:])).if_(Op(X[0] > 1))
            sop << Op(X[1:])

    def test_lshift_if_ok(self):
         delayed = Op([Op(self.func, X), Op(self.func, X)]) << Op(X[1:]).if_(False)
         out = [1, 2, 3] >> delayed
         assert out == [3, 4, 5]
         
    @pytest.mark.parametrize("condition,else_,expected", [
        (True, None, [2, 3]),
        (False, None, [1, 2, 3]),
        (True, Op(X[1:]), [2, 3]),
        (False, Op(X[0]), 1),
        (Op(X[0] > 1), None, [1, 2, 3]),
        (Op(X[0] < 1), Op(X[0] - 1), 0),
    ])
    def test_if_(self, condition, else_, expected):
        data = [1, 2, 3]
        delayed = Op(X[1:]).if_(condition, else_=else_)
        out = data >> delayed
        assert out == expected

    def test_delayed_op_op(self):
        data = [1, 2, 3]
        delayed = Op(X[1:]) >> Op(X[1:])
        assert isinstance(delayed, Op)
        out = data >> delayed
        assert out == [3]

    def test_delayed_module_op(self):
        data = torch.tensor([1.0, 2.0, 3.0])
        delayed = ConstantLayer(3, value=1.0) >> Op(X[1])
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor(2.0))

    def test_delayed_op_module(self):
        data = torch.tensor([1.0, 2.0, 3.0])
        delayed = Op(X[1:]) >> ConstantLayer(2, value=1.0)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor([2.0, 3.0]))

    @pytest.mark.parametrize("op,expected", [
        ("add", [3, 5]),
        ("sub", [1, 1]),
        ("mul", [2, 6]),
        ("truediv", [2.0, 1.5]),
        ("floordiv", [2, 1]),
        ("mod", [0, 1]),
        ("pow", [2, 9]),
    ])
    def test_delayed_op_op_arithmetic(self, op, expected):
        data = torch.tensor([1, 2, 3])
        delayed = getattr(Op(X[1:]), f"__{op}__")(Op(X[:-1]))
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor(expected))

    @pytest.mark.parametrize("op,expected", [
        ("and", [0, 2]),
        ("or", [3, 3]),
        ("xor", [3, 1]),
    ])
    def test_delayed_op_op_bitwise(self, op, expected):
        data = torch.tensor([1, 2, 3], dtype=torch.int64)
        delayed = getattr(Op(X[1:]), f"__{op}__")(Op(X[:-1]))
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor(expected))

    @pytest.mark.parametrize("op,expected", [
        ("neg", [-1, -2, 3]),
        ("pos", [1, 2, -3]),
        ("abs", [1, 2, 3]),
        ("invert", [-2, -3, 2]),
    ])
    def test_delayed_op_op_unary(self, op, expected):
        data = torch.tensor([1, 2, -3], dtype=torch.int64)
        delayed = getattr(Op(X), f"__{op}__")()
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor(expected))

    @pytest.mark.parametrize("op,expected", [
        ("add", [3.0, 6.0]),
        ("sub", [-1.0, -2.0]),
        ("mul", [2.0, 8.0]),
        ("truediv", [0.5, 0.5]),
        ("floordiv", [0., 0.]),
        ("mod", [1., 2.]),
        ("pow", [1.0, 16.0]),
    ])
    def test_delayed_op_module_arithmetic(self, op, expected):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        """
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = getattr(Op(X), f"__{op}__")(layer)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor(expected))

    def test_delayed_op_module_matmul(self):
        """
        As column vectors
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) @ [1, 2] = 10
        """
        layer = ConstantLayer((2, 1), value=2.0)
        data = torch.tensor([[1.0], [2.0]])
        delayed = Op(X.T) @ layer
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor([[10.0]]))

    def test_delayed_op_op_matmul(self):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) @ [1, 2] = 10
        """
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = Op(X) @ (layer >> Op(X[None].T))
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor([10.0]))

    @pytest.mark.parametrize("op,expected", [
        ("and", [0, 0, 0]),
        ("or", [3, 3, 3]),
        ("xor", [3, 3, 3]),
    ])
    def test_delayed_op_module_bitwise(self, op, expected):
        """
        layer output is [2, 2, 2] * [1, 1, 1] = [2, 2, 2]

            data         layer out
        [01, 01, 01] & [10, 10, 10] = [0, 0, 0]
        [01, 01, 01] | [10, 10, 10] = [11, 11, 11]
        [01, 01, 01] ^ [10, 10, 10] = [11, 11, 11]
        """
        layer = ConstantLayer(3, value=2, dtype=torch.int64)
        data = torch.tensor([1, 1, 1], dtype=torch.int64)
        delayed = getattr(Op(X), f"__{op}__")(layer)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor(expected, dtype=torch.int64))
    
    def test_delayed_module_op_add(self):
        """
        The layer comes on the right hand side of the operator. I cannot call layer.__add__, 
        because the python interpreter will not catch the NotImplemented return and call b.__radd__.
        I must use the `+` for this to happen.
        """
        expected = torch.tensor([3.0, 6.0])
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = layer + Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, expected)

    def test_delayed_module_op_sub(self):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) - [1, 2] = [2, 4] - [1, 2] = [1, 2]
        """
        expected = torch.tensor([1.0, 2.0])
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = layer - Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, expected)

    def test_delayed_module_op_mul(self):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) * [1, 2] = [2, 4] * [1, 2] = [2, 8]
        """
        expected = torch.tensor([2.0, 8.0])
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = layer * Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, expected)

    def test_delayed_module_op_truediv(self):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) / [1, 2] = [2, 4] / [1, 2] = [2, 2]
        """
        expected = torch.tensor([2.0, 2.0])
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = layer / Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, expected)

    def test_delayed_module_op_floordiv(self):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) // [1, 2] = [2, 4] // [1, 2] = [2, 2]
        """
        expected = torch.tensor([2.0, 2.0])
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = layer // Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, expected)

    def test_delayed_module_op_mod(self):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) % [1, 2] = [2, 4] % [1, 2] = [0, 0]
        """
        expected = torch.tensor([0.0, 0.0])
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = layer % Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, expected)

    def test_delayed_module_op_pow(self):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) ** [1, 2] = [2, 4] ** [1, 2] = [2, 16]
        """
        expected = torch.tensor([2.0, 16.0])
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = layer ** Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, expected)

    def test_delayed_module_op_matmul(self):
        """
        layer(x) = [2, 2] * [1, 2] = [2, 4]
        layer(x) @ [1, 2] = 10
        """
        layer = ConstantLayer(2, value=2.0)
        data = torch.tensor([1.0, 2.0])
        delayed = layer @ Op(X[None].T)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor([10.0]))

    def test_delayed_op_module_and(self):
        """
        layer(x) = [2, 2, 2]
        layer(x) & Op(x) = [10, 10, 10] & [01, 01, 01] = [0, 0, 0]
        """
        layer = ConstantLayer(3, value=2, dtype=torch.int64)
        data = torch.tensor([1, 1, 1], dtype=torch.int64)
        delayed = layer & Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor([0, 0, 0]))

    def test_delayed_op_module_or(self):
        """
        layer(x) = [2, 2, 2]
        layer(x) & Op(x) = [10, 10, 10] | [01, 01, 01] = [3, 3, 3]
        """
        layer = ConstantLayer(3, value=2, dtype=torch.int64)
        data = torch.tensor([1, 1, 1], dtype=torch.int64)
        delayed = layer | Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor([3, 3, 3]))

    def test_delayed_op_module_xor(self):
        """
        layer(x) = [2, 2, 2]
        layer(x) & Op(x) = [10, 10, 10] ^ [01, 01, 01] = [3, 3, 3]
        """
        layer = ConstantLayer(3, value=2, dtype=torch.int64)
        data = torch.tensor([1, 1, 1], dtype=torch.int64)
        delayed = layer ^ Op(X)
        assert isinstance(delayed, Op)
        out = data >> delayed
        torch.testing.assert_close(out, torch.tensor([3, 3, 3]))

    def test_repr(self):
        op = Op(X[1].a)
        assert str(op) == "Op(X[1].a)"


class Test_OpCallable:
    def test_repr(self):
        def func(x):
            return x
        op = _OpCallable(func, X[2])
        assert str(op) == "func(X[2])"

    def test_repr_module(self):
        op = _OpCallable(ConstantLayer(2, value=1.0), X[2])
        assert str(op) == "ConstantLayer()(X[2])"


class TestWire:
    @staticmethod
    def example_func(x: int, y: int) -> int:
        return x + y

    @pytest.fixture(scope="class")
    def sig(self):
        return inspect.signature(self.example_func)

    def test_init(self, sig):
        wire = Wire(X)
        out = wire.init(sig, 1, 2)
        assert wire._fanout == {}
        assert isinstance(out, A)
        assert out.args == (1, 2)
        assert out.kwargs == {}
    
    def test_init_with_fanout(self, sig):
        wire = Wire(x=X, y=Wiring.Fanout)
        out = wire.init(sig, 1, y=[10, 11])
        assert set(wire._fanout.keys()) == {"y"}
        assert out.args == (1, 10)
        assert out.kwargs == {}

    def test_init_with_passthru(self, sig):
        wire = Wire(x=Wiring.Passthru, y=Wiring.Passthru)
        out = wire.init(sig, 1, 2)
        assert wire._fanout == {}
        assert out.args == (1, 2)
        assert out.kwargs == {}
        
    def test_step(self, sig):
        wire = Wire(X)
        wire.init(sig, 1, 2)
        out = wire.step(3)
        assert out.args == (3, 2)
        assert out.kwargs == {}
    
        out = wire.step(4)
        assert out.args == (4, 2)
        assert out.kwargs == {}

    def test_step_with_fanout(self, sig):
        wire = Wire(x=X, y=Wiring.Fanout)
        wire.init(sig, 1, y=[10, 11])
        out = wire.step(2)        
        assert out.args == (2, 11)
        assert out.kwargs == {}
    
    def test_rrshift(self, sig):
        wire = Wire(X)
        wire.init(sig, 1, 2)
        out = 3 >> wire
        assert out.args == (3, 2)
        assert out.kwargs == {}

    def test_step_no_init(self):
        wire = Wire(X)
        with pytest.raises(ValueError):
            wire.step(1)

    def test_step_with_fanout_overflow(self, sig):
        wire = Wire(x=X, y=Wiring.Fanout)
        wire.init(sig, 1, y=[10, 11])
        with pytest.raises(ValueError):
            wire.step(2)
            wire.step(3)
            wire.step(4)


def test_conjure():
    class Something:
        def __init__(self, a):
            self.a = a
    
    data = [Something(1), Something(2)]
    res = conjure(X[1].a, data)
    assert res == 2

