from faeyon.magic import X


def test_X_single_op():
    # getitem
    x = X[0]
    assert isinstance(x, X)
    assert len(x._buffer) == 1

    # add
    x = X + 1
    assert isinstance(x, X)
    assert len(x._buffer) == 1

    # call
    x = X("foo", bar="baz")
    assert isinstance(x, X)
    assert len(x._buffer) == 1

    # getattr
    x = X.a
    assert isinstance(x, X)
    assert len(x._buffer) == 1


def test_X_multiple_ops():
    x = X[0] + 1
    assert isinstance(x, X)
    assert len(x._buffer) == 2

    x = round(X[0] + 1)
    assert isinstance(x, X)
    assert len(x._buffer) == 3


