from faeyon import X, FaeArgs
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
        x = round(X("foo", k="bar")[0] + 1)
        assert str(x) == "X('foo', k='bar') -> X[0] -> X + 1 -> round(X)"


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

    def test_bind_resolved(self):
        """ Tests the bind (Any >> FaeArgs) operator when `FaeArgs` is already resolved. """
        fae_args = FaeArgs(1, x="Bar")
        data = "Foo"
        out_args = fae_args.bind(data)
        assert out_args.args == (1,)
        assert out_args.kwargs == {"x": "Bar"}
        
        out_args = data >> fae_args
        assert out_args.args == (1,)
        assert out_args.kwargs == {"x": "Bar"}

    def test_bind_unresolved(self):
        fea_args = FaeArgs(X[0], x="Bar", y=X[1])
        data = [10, 11, 12]

        out_args = fea_args.bind(data)
        assert out_args.args == (10,)
        assert out_args.kwargs == {"x": "Bar", "y": 11}

        out_args = data >> fea_args
        assert out_args.args == (10,)
        assert out_args.kwargs == {"x": "Bar", "y": 11}