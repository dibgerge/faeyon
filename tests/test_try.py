from faeyon import F, X, A


# @pytest.mark.parametrize("expr", [X % "foo", "foo" % X])
def test_mod():
    expr = X + 1 >> X + 2

    print(list(expr.fae))
    cloned = expr.fae.clone()
    print(list(cloned.fae))

    for original, cloned in zip(expr.fae, cloned.fae):
        print(original, cloned)
        res = original is cloned
        print(res)

    # for original, cloned in zip(expr.fae, cloned.fae):
    #     print(original, cloned)
    #     res = original is cloned
    #     print(res)

    
