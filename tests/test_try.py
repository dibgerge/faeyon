from faeyon import F, X, A


# @pytest.mark.parametrize("expr", [X % "foo", "foo" % X])
def test_mod():
    """ Test mod operator for non-arithmetic operations. """
    # assert isinstance(expr, X)
    # assert expr._name == "foo"
    print("starting")
    expr = X % "BLA"
    print(expr, expr._name, expr._something)

    
