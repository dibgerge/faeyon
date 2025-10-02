import pytest
from faeyon.training import Period, FaeOptimizer, TrainState
from faeyon.enums import PeriodUnit


class TestPeriod:

    @pytest.mark.parametrize("expr, expected", [
        ("100e", Period(100, PeriodUnit.EPOCHS)),
        ("100steps", Period(100, PeriodUnit.STEPS)),
        ("100sec", Period(100, PeriodUnit.SECONDS)),
        ("100m", Period(100 * 60, PeriodUnit.SECONDS)),
        ("100h", Period(100 * 3600, PeriodUnit.SECONDS)),
        ("2d", Period(2 * 24 * 3600, PeriodUnit.SECONDS)),
    ])
    def test_from_expr(self, expr, expected):
        period = Period.from_expr(expr)
        assert period == expected

    def test_from_expr_invalid(self):
        with pytest.raises(ValueError):
            Period.from_expr("abc")
        
        with pytest.raises(ValueError):
            Period.from_expr("100e | 2d")

    @pytest.fixture
    def period(self):
        return Period(10, PeriodUnit.SECONDS)

    @pytest.mark.parametrize("op, expected", [
        (lambda x: x + 5, 15),
        (lambda x: 5 + x, 15),
        (lambda x: x - 5, 5),
        (lambda x: 15 - x, 5),
        (lambda x: x * 5, 50),
        (lambda x: 5 * x, 50),
        (lambda x: x / 4, 2.5),
        (lambda x: 4 / x, 0.4),
        (lambda x: x // 4, 2),
        (lambda x: 4 // x, 0),
        (lambda x: x % 4, 2),
        (lambda x: 4 % x, 4),
    ])
    def test_arithmetic(self, period, op, expected):
        period2 = op(period)
        assert period2.value == expected
        assert period2.unit == period.unit

    def test_iadd(self, period):
        period += 5
        assert period.value == 15
        assert period.unit == PeriodUnit.SECONDS
        
    def test_isub(self, period):
        period -= 5
        assert period.value == 5
        assert period.unit == PeriodUnit.SECONDS
        
    def test_imul(self, period):
        period *= 5
        assert period.value == 50
        assert period.unit == PeriodUnit.SECONDS

    def test_itruediv(self, period):
        period /= 4
        assert period.value == 2.5
        assert period.unit == PeriodUnit.SECONDS

    def test_ifloordiv(self, period):
        period //= 4
        assert period.value == 2
        assert period.unit == PeriodUnit.SECONDS

    def test_imod(self, period):
        period %= 4
        assert period.value == 2
        assert period.unit == PeriodUnit.SECONDS

    @pytest.mark.parametrize("other", [
        "10e",
        Period(10, PeriodUnit.EPOCHS),
    ])
    def test_arith_invalid(self, period, other):
        with pytest.raises(ValueError):
            period + other

    @pytest.mark.parametrize("op, expected", [
        (lambda x: x == 10, True),
        (lambda x: x == "10sec", True),
        (lambda x: x == "10e", False),
        (lambda x: x != "10e", True),
        (lambda x: x != "10sec", False),
        (lambda x: x < 15, True),
        (lambda x: x < "10e", None),
        (lambda x: x <= 15, True),
        (lambda x: x <= "10e", None),
        (lambda x: x > 15, False),
        (lambda x: x > "10e", None),
        (lambda x: x >= 15, False),
        (lambda x: x >= "10e", None),
    ])
    def test_comparison(self, period, op, expected):
        res = op(period)
        assert res == expected


class TestTrainState:
    def test_init(self):
        state = TrainState()
        assert state.epoch == 0
        assert state.step == 0

    def test_toc(self):
        state = TrainState()
        state.toc()
        assert state.epoch == 1
        assert state.step == 0
        assert state.epoch_step == 0
        assert state.epoch_start is not None 

    
    def test_toc_val(self):
        state = TrainState()
        state.toc(train=False)
        assert state.epoch == 0
        assert state.step == 0
        assert state.epoch_step == 0
        assert state.epoch_start is None

        
class TestFaeOptimizer:
    def test_init(self):
        optimizer = FaeOptimizer("Adam", lr=0.001)
        # assert optimizer.optimizer == torch.optim
        # assert optimizer.patterns == ["*"]
        assert optimizer.kwargs == {"lr": 0.001}
