import pytest
from datetime import timedelta
from faeyon.training import TrainPeriod


class TestTrainPeriod:

    @pytest.mark.parametrize("expr, expected", [
        ("100e", TrainPeriod(epochs=100)),
        ("100steps", TrainPeriod(steps=100)),
        ("100sec", TrainPeriod(ts=timedelta(seconds=100))),
        ("100m", TrainPeriod(ts=timedelta(minutes=100))),
        ("100h", TrainPeriod(ts=timedelta(hours=100))),
        ("100h|1e", TrainPeriod(ts=timedelta(hours=100), epochs=1.0, condition="any")),
        (
            "100h|1e|10steps", 
            TrainPeriod(ts=timedelta(hours=100), epochs=1.0, steps=10, condition="any")
        ),
        ("10.5h&1.5e", TrainPeriod(ts=timedelta(hours=10.5), epochs=1.5, condition="all")),
        (
            "10.5h & 1.5e & 20steps", 
            TrainPeriod(ts=timedelta(hours=10.5), epochs=1.5, steps=20, condition="all")
        ),
        (
            "100h | 1e & 10steps",
            TrainPeriod(ts=timedelta(hours=100), epochs=1.0, steps=10, condition="ts")
        ),
        (
            "100h & 1e | 10steps",
            TrainPeriod(ts=timedelta(hours=100), epochs=1.0, steps=10, condition="steps")
        )
    ])
    def test_from_expr(self, expr, expected):
        period = TrainPeriod.from_expr(expr)
        assert period == expected

    def test_from_expr_invalid(self):
        with pytest.raises(ValueError):
            TrainPeriod.from_expr("100h | 1e & 10steps & 100s")
        # with pytest.raises(ValueError):
