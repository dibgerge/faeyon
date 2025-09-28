import pytest
from datetime import timedelta
from faeyon.training import Period


# class TestPeriod:

#     @pytest.mark.parametrize("expr, expected", [
#         ("100e", Period(epochs=100)),
#         ("100steps", Period(steps=100)),
#         ("100sec", Period(ts=timedelta(seconds=100))),
#         ("100m", Period(ts=timedelta(minutes=100))),
#         ("100h", Period(ts=timedelta(hours=100))),
#         ("100h|1e", Period(ts=timedelta(hours=100), epochs=1.0, condition="any")),
#         (
#             "100h|1e|10steps", 
#             Period(ts=timedelta(hours=100), epochs=1.0, steps=10, condition="any")
#         ),
#         ("10.5h&1.5e", Period(ts=timedelta(hours=10.5), epochs=1.5, condition="all")),
#         (
#             "10.5h & 1.5e & 20steps", 
#             Period(ts=timedelta(hours=10.5), epochs=1.5, steps=20, condition="all")
#         ),
#         (
#             "100h | 1e & 10steps",
#             Period(ts=timedelta(hours=100), epochs=1.0, steps=10, condition="ts")
#         ),
#         (
#             "100h & 1e | 10steps",
#             Period(ts=timedelta(hours=100), epochs=1.0, steps=10, condition="steps")
#         )
#     ])
#     def test_from_expr(self, expr, expected):
#         period = Period.from_expr(expr)
#         assert period == expected

#     def test_from_expr_invalid(self):
#         with pytest.raises(ValueError):
#             Period.from_expr("100h | 1e & 10steps & 100s")
#         # with pytest.raises(ValueError):
