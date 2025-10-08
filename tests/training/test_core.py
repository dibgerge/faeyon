import pytest
import torch
from faeyon.training import Period, FaeOptimizer, TrainState
from faeyon.enums import PeriodUnit
from faeyon.metrics import MetricCollection, Accuracy


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
    @pytest.fixture
    def state(self):
        metrics = MetricCollection([Accuracy(num_classes=1)])
        return TrainState(metrics=metrics)

    def assert_values(
        self, 
        state, 
        epoch=0,
        epoch_total_time=True,
        epoch_train_steps=0, 
        epoch_train_time=True, 
        epoch_val_steps=0, 
        epoch_val_time=True,
        total_time=True,
        total_train_steps=0, 
        total_train_time=True, 
        total_val_steps=0, 
        total_val_time=True
    ):
        """
        For the time parameters, if True, we assert that the value is zero.
        """
        assert state.epoch == epoch
        assert state.epoch_train_steps == epoch_train_steps
        assert state.epoch_val_steps == epoch_val_steps
        assert state.total_train_steps == total_train_steps       
        assert state.total_val_steps == total_val_steps
        
        assert ((state.epoch_total_time == 0) if epoch_total_time else (state.epoch_total_time > 0))
        assert ((state.epoch_train_time == 0) if epoch_train_time else (state.epoch_train_time > 0))
        assert ((state.epoch_val_time == 0) if epoch_val_time else (state.epoch_val_time > 0))
        assert ((state.total_train_time == 0) if total_train_time else (state.total_train_time > 0)  )
        assert ((state.total_time == 0) if total_time else (state.total_time > 0))
        assert ((state.total_val_time == 0) if total_val_time else (state.total_val_time > 0))

    def test_init(self, state):
        self.assert_values(state)

    def test_train_begin(self, state):
        state.on_train_begin()
        self.assert_values(state)

    def test_epoch_begin(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        self.assert_values(state, epoch=0, total_time=False, total_train_time=False)
        
    def test_train_step_begin(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        self.assert_values(
            state, 
            epoch=0, 
            epoch_train_steps=0, 
            total_train_steps=0, 
            epoch_total_time=False, 
            epoch_train_time=False,
            total_train_time=False, 
            total_time=False
        )
        
    def test_train_step_end(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=False
        )
        self.assert_values(
            state, 
            epoch=0, 
            epoch_train_steps=1, 
            epoch_total_time=False,
            epoch_train_time=False,
            total_time=False,
            total_train_time=False, 
            total_train_steps=1,
        )

    def test_val_begin(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=False
        )
        state.on_val_begin()
        self.assert_values(
            state, 
            epoch=0, 
            epoch_train_steps=1, 
            epoch_total_time=False,
            epoch_train_time=False,
            total_time=False,
            total_train_time=False, 
            total_train_steps=1,
        )

    def test_val_step_begin(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=False
        )
        state.on_val_begin()
        state.on_val_step_begin()
        self.assert_values(
            state, 
            epoch=0,
            epoch_train_steps=1, 
            epoch_total_time=False,
            epoch_train_time=False,
            epoch_val_steps=0,
            epoch_val_time=False,
            total_time=False,
            total_train_time=False, 
            total_train_steps=1,
            total_val_steps=0,
            total_val_time=False,
        )

    def test_val_step_end(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=False
        )
        state.on_val_begin()
        state.on_val_step_begin()
        state.on_val_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
        )
        self.assert_values(
            state, 
            epoch=0,
            epoch_train_steps=1, 
            epoch_total_time=False,
            epoch_train_time=False,
            epoch_val_steps=1,
            epoch_val_time=False,
            total_time=False,
            total_train_time=False, 
            total_train_steps=1,
            total_val_steps=1,
            total_val_time=False,
        )
    
    def test_val_end(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=False
        )
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=True
        )
        state.on_val_begin()
        state.on_val_step_begin()
        state.on_val_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
        )
        state.on_val_end()
        state.on_epoch_end()
        self.assert_values(
            state, 
            epoch=1,
            epoch_train_steps=2, 
            epoch_total_time=False,
            epoch_train_time=False,
            epoch_val_steps=1,
            epoch_val_time=False,
            total_time=False,
            total_train_time=False, 
            total_train_steps=2,
            total_val_steps=1,
            total_val_time=False,
        )

    def test_multiple_val_steps(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=True
        )
        state.on_val_begin()
        state.on_val_step_begin()
        state.on_val_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
        )
        state.on_val_step_begin()
        state.on_val_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
        )
        state.on_val_end()
        state.on_epoch_end()
        self.assert_values(
            state, 
            epoch=1,
            epoch_train_steps=1, 
            epoch_total_time=False,
            epoch_train_time=False,
            epoch_val_steps=2,
            epoch_val_time=False,
            total_time=False,
            total_train_time=False, 
            total_train_steps=1,
            total_val_steps=2,
            total_val_time=False,
        )

    def test_train_no_val(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=True
        )
        state.on_epoch_end()
        self.assert_values(
            state, 
            epoch=1,
            epoch_train_steps=1, 
            epoch_total_time=False,
            epoch_train_time=False,
            total_time=False,
            total_train_time=False, 
            total_train_steps=1,
        )

    def test_multiple_epochs(self, state):
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=True
        )
        state.on_epoch_end()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=True
        )
        state.on_epoch_end()
        state.on_train_end()
        self.assert_values(
            state, 
            epoch=2,
            epoch_train_steps=1, 
            epoch_total_time=False,
            epoch_train_time=False,
            total_time=False,
            total_train_time=False, 
            total_train_steps=2,
        )
    
    def test_incorrect_transition(self, state):
        with pytest.raises(RuntimeError):
            state.on_train_step_begin()
        
        with pytest.raises(RuntimeError):
            state.on_epoch_begin()
        
        with pytest.raises(RuntimeError):
            state.on_val_begin()
        
        with pytest.raises(RuntimeError):
            state.on_val_step_begin()

        with pytest.raises(RuntimeError):
            state.on_val_step_end()
        
        with pytest.raises(RuntimeError):
            state.on_val_end()
        
        with pytest.raises(RuntimeError):
            state.on_epoch_end()

    def test_incorrect_train_step_transition(self, state):
        """ Train step not ended """
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        with pytest.raises(RuntimeError):
            state.on_train_step_begin()

    def test_incorrect_val_step_transition(self, state):
        """ Train step not ended """
        state.on_train_begin()
        state.on_epoch_begin()
        state.on_train_step_begin()
        state.on_train_step_end(
            loss=torch.tensor(0.1),
            preds=torch.tensor([1, 0]), 
            targets=torch.tensor([1, 1]),
            is_last=True
        )
        state.on_val_begin()
        state.on_val_step_begin()
        with pytest.raises(RuntimeError):
            state.on_val_step_begin()        
        
        with pytest.raises(RuntimeError):
            state.on_val_end()

    def test_incorrect_step_end_transition(self, state):
        """ Train step not ended """
        state.on_train_begin()
        state.on_epoch_begin()
        with pytest.raises(RuntimeError):
            state.on_train_step_end(torch.tensor([0.5]), torch.tensor([0.5]))
        
        with pytest.raises(RuntimeError):
            state.on_val_step_end()
        
        with pytest.raises(RuntimeError):
            state.on_val_end()
        
        with pytest.raises(RuntimeError):
            state.on_epoch_end()
        
    def test_incorrect_train_end_transition(self, state):
        """ Train step not ended """
        state.on_train_begin()
        state.on_epoch_begin()
        with pytest.raises(RuntimeError):
            state.on_train_end()


class TestFaeOptimizer:
    def test_init(self):
        optimizer = FaeOptimizer("Adam", lr=0.001)
        # assert optimizer.optimizer == torch.optim
        # assert optimizer.patterns == ["*"]
        assert optimizer.kwargs == {"lr": 0.001}
