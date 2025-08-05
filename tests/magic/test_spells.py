import inspect
from faeyon import X, FaeArgs, FaeVar, FaeList, FaeDict, FaeMultiMap, Op, Wire, Wiring
import pytest


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


class TestFaeArgs:

    def func_simple(self, x):
        return x + 1

    def func_multi(self, x=1, y=0):
        return x + y
    
    def test_init(self):
        faek = FaeArgs(1, 2, 3)
        assert isinstance(faek, FaeArgs)
        assert faek.args == (1, 2, 3)
        assert faek.kwargs == {}

    def test_call_raisesTypeError(self):
        """ 
        When `FaeArgs` does not match the callable's required number of arguments.
        """
        fae_args = FaeArgs(1, 2, 3)

        with pytest.raises(TypeError):
            fae_args.call(self.func_simple)
        
        with pytest.raises(TypeError):
            fae_args >> self.func_simple

    def test_call_raisesValueError(self):
        """ 
        When `FaeArgs` has unresolved arguments, an error is raised.
        """
        fae_args = FaeArgs(X[0])

        with pytest.raises(ValueError):
            fae_args.call(self.func_simple)

        with pytest.raises(ValueError):
            fae_args >> self.func_simple

    @pytest.mark.parametrize("args,kwargs", [
        ((), {}),
        ((2,), {}),
        ((), {"y": 10}),
        ((1,), {"y": 10}),
    ])
    def test_call(self, args, kwargs):
        """ Tests the call (FaeArgs >> callable) operator/method. """
        fae_args = FaeArgs(*args, **kwargs)
        expected = self.func_multi(*args, **kwargs)
        assert fae_args.call(self.func_multi) == expected
        assert fae_args >> self.func_multi == expected

    def test_using_resolved(self):
        """ Tests the bind (Any >> FaeArgs) operator when `FaeArgs` is already resolved. """
        fae_args = FaeArgs(1, x="Bar")
        data = "Foo"
        out_args = fae_args.using(data)
        assert out_args.args == (1,)
        assert out_args.kwargs == {"x": "Bar"}
        
        out_args = data >> fae_args
        assert out_args.args == (1,)
        assert out_args.kwargs == {"x": "Bar"}

    def test_using_unresolved(self):
        fea_args = FaeArgs(X[0], x="Bar", y=X[1])
        data = [10, 11, 12]

        out_args = fea_args.using(data)
        assert out_args.args == (10,)
        assert out_args.kwargs == {"x": "Bar", "y": 11}

        out_args = data >> fea_args
        assert out_args.args == (10,)
        assert out_args.kwargs == {"x": "Bar", "y": 11}


class TestFaeVar:
    def test_rrshift(self):
        fae_var = FaeVar(strict=True)
        2 >> fae_var
        assert fae_var._value.value == 2

    def test_rrshift_raisesValueError(self):
        """ 
        Cannot bind a value to a strict FaeVar with existing value.
        """
        fae_var = FaeVar(strict=True)
        with pytest.raises(ValueError):
            2 >> fae_var
            3 >> fae_var

    def test_rrshift_overwrite(self):
        fae_var = FaeVar(strict=False)
        out = 2 >> fae_var
        assert fae_var._value.value == 2
        assert out == 2

        out = 3 >> fae_var
        assert fae_var._value.value == 3
        assert out == 3

    def test_using(self):
        """ `using` method is same as >> operator. """
        fae_var = FaeVar(strict=True)
        out = fae_var.using(2)
        assert fae_var._value.value == 2
        assert out == 2

    def test_select_return_type(self):
        fae_var = FaeVar(strict=False)
        selectable = fae_var.select(X[1])
        assert isinstance(selectable, FaeVar)
        assert selectable is not fae_var

        [10, 20] >> selectable
        assert selectable._value.value == 20

        [10, 20] >> fae_var
        assert fae_var._value.value == [10, 20]
    
    def test_select(self):
        fae_var = FaeVar(strict=True)
        data = [1, 2,  3]
        out = data >> fae_var.select(X[1])
        assert data is out
        assert fae_var._value.value == 2

    def test_matmul(self):
        """ `matmul` (@) operator is same as select method. """
        fae_var = FaeVar(strict=True)
        data = [1, 2, 3]
        out = data >> fae_var @ X[1]
        assert data is out
        assert fae_var._value.value == 2
    
    def test_select_raisesValueError_multiple_calls(self):
        """ 
        Can only call once
        """
        fae_var = FaeVar(strict=True)
        with pytest.raises(ValueError):
            fae_var @ X[1] @ X[2]
            
    def test_select_raisesValueError_wrong_type(self):
        """ 
        Cannot select an expression to a strict FaeVar with existing value.
        """
        fae_var = FaeVar(strict=True)
        with pytest.raises(ValueError):
            fae_var @ "foo"

    def test_shed(self):
        fae_var = FaeVar()
        [1, 2, 3] >> fae_var @ X[1:]
        assert fae_var.shed() == [2, 3]
        assert +fae_var == [2, 3]

    def test_shed_raisesValueError(self):
        fae_var = FaeVar()
        with pytest.raises(ValueError):
            fae_var.shed()

        with pytest.raises(ValueError):
            fae_var @ X[1]
            fae_var.shed()

    def test_repr(self):
        fae_var = FaeVar()
        2 >> fae_var
        assert str(fae_var) == "FaeVar(2)"

    
class TestFaeList:
    """ 
    Some of the common methods and code paths with `FaeVar` are not tested here, especially that
    the operator / method overloading are consistent, since this is implemented at the base class 
    level.
    """
    def test_init_no_args(self):
        fae_list = FaeList()
        assert +fae_list == []

    def test_init_with_args(self):
        fae_list = FaeList(1, 2, 3)
        assert +fae_list == [1, 2, 3]
    
    def test_rrshift(self):
        fae_list = FaeList()
        2 >> fae_list
        3 >> fae_list
        assert +fae_list == [2, 3]

    def test_general_usage(self):
        fae_list = FaeList()
        [10, 20, 30] >> fae_list @ X[1] >> fae_list @ X[2]
        assert +fae_list == [20, 30]


class TestFaeDict:
    def test_init_no_args(self):
        fae_dict = FaeDict()
        assert +fae_dict == {}
    
    def test_init_with_args(self):
        init_data = {"a": 1, "b": 2, "c": 3}
        fae_dict = FaeDict(**init_data)
        assert +fae_dict == init_data
    
    def test_rrshift(self):
        fae_dict = FaeDict(b=10)
        2 >> fae_dict["a"]
        assert +fae_dict == {"a": 2, "b": 10}

    def test_rrshift_strict_raises(self):
        """ Raise error when overwriting existing key in strict mode. """
        fae_dict = FaeDict(strict=True)
        with pytest.raises(ValueError):
            2 >> fae_dict["a"]
            3 >> fae_dict["a"]
        
    def test_rrshift_nonstrict_overwrite(self):
        fae_dict = FaeDict(strict=False)
        2 >> fae_dict["a"]
        assert +fae_dict == {"a": 2}
        3 >> fae_dict["a"]
        assert +fae_dict == {"a": 3}
        4 >> fae_dict["b"]
        assert +fae_dict == {"a": 3, "b": 4}
    
    def test_raises_key_error(self):
        fae_dict = FaeDict()
        with pytest.raises(KeyError):
            2 >> fae_dict

        # KeyError when accessing non-existent key. Make sure there is no 
        # side effects due to inplace operations.
        with pytest.raises(KeyError):
            fae_dict["a"]
            2 >> fae_dict
    
    def test_general_usage(self):
        fae_dict = FaeDict()
        [1, 2, 3] >> fae_dict["a"] @ X[1] >> fae_dict["b"]
        assert +fae_dict == {"a": 2, "b": [1, 2, 3]}


class TestFaeMultiMap:
    def test_init_no_args(self):
        fae_multimap = FaeMultiMap()
        assert +fae_multimap == {}
    
    def test_init_with_args(self):
        fae_multimap = FaeMultiMap(a=[1, 2, 3], b=[4, 5, 6])
        assert +fae_multimap == {"a": [1, 2, 3], "b": [4, 5, 6]}
        
    def test_init_value_error(self):
        with pytest.raises(ValueError):
            FaeMultiMap(a=[1, 2, 3], b=4)
    
    def test_general_usage(self):
        fae_multimap = FaeMultiMap()
        out = (
            [1, 2, 3] 
            >> fae_multimap["a"] @ X[1] 
            >> fae_multimap["a"] @ X[2] 
            >> fae_multimap["b"] @ X[0]
        )
        assert +fae_multimap == {"a": [2, 3], "b": [1]}
        

class Test_Variable:
    def test_repr(self):
        from faeyon.magic.spells import _Variable
        var = _Variable(10)
        assert str(var) == "_Variable(10)"


class TestOp:
    def test_usage(self):
        data = [1, 2, 3]
        out = data >> Op(X)
        assert out == data

    def test_usage2(self):
        data = [1, 2, 3]
        out = data >> Op(X[1:])
        assert out == [2, 3]

    def test_repr(self):
        op = Op(X[1].a)
        assert str(op) == "Op(X[1].a)"


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
        assert isinstance(out, FaeArgs)
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
