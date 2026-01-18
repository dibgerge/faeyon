from faeyon import F, X


# @pytest.mark.parametrize("expr", [X % "foo", "foo" % X])
def test_mod():
    """ Test mod operator for non-arithmetic operations. """
    # assert isinstance(expr, X)
    # assert expr._name == "foo"
    print("starting")
    expr = "foo %s" % X
    print(expr)

    
