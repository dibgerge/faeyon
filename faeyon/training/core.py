from __future__ import annotations
from dataclasses import dataclass
from functools import wraps
import time
import fnmatch
import importlib
import re
import torch

from datetime import timedelta

from typing import Callable, Optional, Iterable, Union
from torch import nn, optim
from faeyon.enums import PeriodUnit
from faeyon.metrics import MeanMetric, MetricCollection, Metric


PeriodT = Union["Period", float, int, str]


class FaeOptimizer:
    """
    Wraps regular torch optimizers, but instead of having to specify parameters directly, use a 
    regexp to specify their names and we can do late bindings.

    name : str | Optimzier
        The name of the optimizer object. If given a string, it should be an existing
        optimizer in the `torch.optim` module, or the full path of the optimizer e.g.
        `foo.bar.MyOptimizer`.

    regex : bool
        Whether to use regular expressions to match parameter names. If True, the `patterns`
        should be a list of regular expressions. If False, the `patterns` should be a list of
        strings. If `None`, it will be inferred from the type of the `patterns` argument. A string
        will be treated as a glob pattern.
    """
    def __init__(
        self,
        name: str,
        patterns: Optional[str | re.Pattern | list[str | re.Pattern]] = None,
        regex: Optional[bool] = None,
        **kwargs
    ):
        module, _, obj = name.rpartition(".")
        if not module:
            module = "torch.optim"

        if patterns is not None:
            if not isinstance(patterns, list):
                patterns = [patterns]

            if regex is False:
                if any(isinstance(pat, re.Pattern) for pat in patterns):
                    raise ValueError(
                        "`regex` is False, but a `re.Pattern` object given to `params`."
                    )
            elif regex is True:
                patterns = [
                    pattern if isinstance(pattern, re.Pattern) else re.compile(pattern) 
                    for pattern in patterns
                ]

        module_obj = importlib.import_module(module)
        self.optimizer = getattr(module_obj, obj)
        self.patterns = patterns
        self.kwargs = kwargs
            
    def __call__(self, model: nn.Module) -> optim.Optimizer:
        matches: Iterable[str]
        valid_names: list[str] = []
        
        if self.patterns is None:
            return self.optimizer(model.parameters(), **self.kwargs)

        parameters = dict(model.named_parameters())
        names = list(parameters.keys())

        for pattern in self.patterns:            
            if isinstance(pattern, re.Pattern):
                matches = filter(pattern.fullmatch, names)
            else:
                matches = fnmatch.filter(names, pattern)
            valid_names.extend(matches)

        return self.optimizer([parameters[k] for k in valid_names], **self.kwargs)


class Period:
    def __init__(self, value: float | int, unit: PeriodUnit) -> None:
        self.value = value
        self.unit = unit

    @classmethod
    def from_expr(cls, expr: str) -> Period:
        pattern = r"""
        ([0-9]*\.?[0-9]+)       # Amount (Group 1)
        (?:
            (e(?:pochs?)?)  |   # Epochs (Group 2)
            (st(?:eps?)?)   |   # Steps (Group 3)
            (sec(?:onds?)?) |   # Seconds (Group 4)
            (m(?:inutes?)?) |   # Minutes (Group 5)
            (h(?:ours?)?)   |   # Hours (Group 6)
            (d(?:ays?)?)        # Days (Group 7)
        )
        """
        matcher = re.fullmatch(pattern, expr, re.IGNORECASE | re.VERBOSE)

        if not matcher:
            raise ValueError(f"Invalid period expression: {expr}")

        groups = matcher.groups()
        # amounts = {"epochs": None, "steps": None, "ts": None}
        group_map = {
            1: PeriodUnit.EPOCHS, 
            2: PeriodUnit.STEPS, 
            3: PeriodUnit.SECONDS, 
            4: PeriodUnit.SECONDS, 
            5: PeriodUnit.SECONDS, 
            6: PeriodUnit.SECONDS
        }
        time_map = {3: "seconds", 4: "minutes", 5: "hours", 6: "days"}

        value = float(groups[0])
            
        # find indices of non-None groups
        indices = [j for j in range(1, 7) if groups[j] is not None]

        if len(indices) != 1:
            raise ValueError(
                f"Invalid period expression: {expr}.")

        index = indices[0]
        unit = group_map[index]

        if unit == PeriodUnit.SECONDS:
            value = timedelta(**{time_map[index]: value}).total_seconds()
        
        return cls(value, unit)

    def _validate_period(self, other: PeriodT) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        elif isinstance(other, float | int):
            other = Period(other, self.unit)
        elif not isinstance(other, Period):
            raise ValueError(f"Invalid period: {other} used for arithmetic operation.")

        if other.unit != self.unit:
            raise ValueError(
                f"Given period has different units than the current period: "
                f"{self.unit} and {other.unit}."
            )
        return other

    def __iadd__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        self.value += other.value
        return self

    def __add__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value + other.value, unit=self.unit)

    __radd__ = __add__

    def __isub__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        self.value -= other.value
        return self
        
    def __sub__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value - other.value, unit=self.unit)

    def __rsub__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=other.value - self.value, unit=self.unit)

    def __imul__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        self.value *= other.value
        return self

    def __mul__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value * other.value, unit=self.unit)

    __rmul__ = __mul__

    def __itruediv__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        self.value /= other.value
        return self

    def __truediv__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value / other.value, unit=self.unit)

    def __rtruediv__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=other.value / self.value, unit=self.unit)

    def __ifloordiv__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        self.value //= other.value
        return self

    def __floordiv__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value // other.value, unit=self.unit)

    def __rfloordiv__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=other.value // self.value, unit=self.unit)

    def __imod__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        self.value %= other.value
        return self

    def __mod__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value % other.value, unit=self.unit)

    def __rmod__(self, other: PeriodT) -> Period:
        other = self._validate_period(other)
        return Period(value=other.value % self.value, unit=self.unit)

    def __eq__(self, other: object) -> bool:
        # TODO: Why MyPy complains if I use PeriodT here?
        if not isinstance(other, float | int | str | Period):
            return False

        try:
            other = self._validate_period(other)
        except ValueError:
            return False

        return self.value == other.value

    def __ne__(self, other: object) -> bool:
        return not self == other
    
    def __lt__(self, other: PeriodT) -> Optional[bool]:
        try:
            other = self._validate_period(other)
        except ValueError:
            return None
        return self.value < other.value

    def __le__(self, other: PeriodT) -> Optional[bool]:
        try:
            other = self._validate_period(other)
        except ValueError:
            return None
        return self.value <= other.value
    
    def __gt__(self, other: PeriodT) -> Optional[bool]:
        try:
            other = self._validate_period(other)
        except ValueError:
            return None
        return self.value > other.value
    
    def __ge__(self, other: PeriodT) -> Optional[bool]:
        try:
            other = self._validate_period(other)
        except ValueError:
            return None
        return self.value >= other.value

    def __repr__(self) -> str:
        return f"Period({self.value}, {self.unit})"


@dataclass
class TrainState:
    """
    A state machine for the training process, which assumes the following workflow:

    TrainBegin -> EpochBegin -> TrainStepBegin -> TrainStepEnd  --------
                    ↑                  ↑_____________|                 |
                    |                                                  ↓
    TrainEnd <- EpochEnd <- ValEnd <- ValStepEnd <- ValStepBegin <- ValBegin
                                            |_____________↑
    
    Can be compared to Periods, and arithmetic operations are supported as follows:
    - If the period is in epoch units, the comparison is done with `epoch`.
    - If the period is in step units, the comparison is done with `total_train_steps`.
    - If the period is in time units, the comparison is done with `total_time`.
    """
    train_metrics: MetricCollection
    train_loss: MeanMetric
    val_metrics: MetricCollection
    val_loss: MeanMetric

    epoch: int

    total_train_steps: int
    total_val_steps: int
    epoch_train_steps: int
    epoch_val_steps: int
    is_epoch_last_step: bool

    _start_time: Optional[float]
    _epoch_start_time: Optional[float]
    _epoch_val_start_time: Optional[float]
    _current_time: Optional[float]
    _total_val_time: float
    _next_flush: Period
    _expected_train_steps: Optional[int]

    def __init__(
        self, 
        metrics: Optional[list[Metric] | MetricCollection] = None,
        train_flush: str | Period = "1e",
    ) -> None:
        if metrics is None:
            self.train_metrics = MetricCollection()
        elif isinstance(metrics, list):
            self.train_metrics = MetricCollection(metrics=metrics)
        elif isinstance(metrics, MetricCollection):
            self.train_metrics = metrics
        else:
            raise ValueError(f"Invalid metrics: {metrics}.")
        self.val_metrics = self.train_metrics.clone()
        self.val_loss = MeanMetric()
        self.train_loss = MeanMetric()

        if isinstance(train_flush, str):
            self._train_flush = Period.from_expr(train_flush)
        else:
            self._train_flush = train_flush
        self.reset()

    @property
    def transitions(self) -> set[str]:
        return {"on_train_begin"}

    @staticmethod
    def valid_transition(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> None:
            if func.__name__ not in self.transitions:
                self.error(func.__name__)
            return func(self, *args, **kwargs)
        return wrapper

    def new_state(self, state: type[TrainState]) -> None:
        self.__class__ = state

    def reset(self) -> None:
        self.epoch = 0
        self._start_time = None
        self._current_time = None
        self._epoch_start_time = None
        self._epoch_val_start_time = None
        self._total_val_time = 0
        self.total_train_steps = 0
        self.total_val_steps = 0
        self.epoch_train_steps = 0
        self.epoch_val_steps = 0
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.val_loss.reset()
        self.train_loss.reset()
        self._next_flush = self._train_flush
        self._expected_train_steps = None
        self.is_epoch_last_step = False

    @valid_transition
    def on_train_begin(self, train_data_size: Optional[int] = None) -> None:
        """
        If the model was trained before, and epoch is > 0, we continue from where 
        we left off.
        """
        self.reset()
        self._start_time = time.time()
        self._current_time = self._start_time
        self._train_data_size = train_data_size
        self.new_state(TrainStateTrainBegin)

    @valid_transition
    def on_epoch_begin(self) -> None:
        self.new_state(TrainStateEpochBegin)
        self.epoch_train_steps = 0
        self.epoch_val_steps = 0
        self._epoch_start_time = time.time()
        self._epoch_val_start_time = None
        self._current_time = self._epoch_start_time
    
    @valid_transition
    def on_train_step_begin(self) -> None:
        self.new_state(TrainStateTrainStepBegin)
        self._current_time = time.time()
       
    @valid_transition
    def on_train_step_end(
        self, 
        loss: torch.Tensor, 
        preds: torch.Tensor, 
        targets: torch.Tensor,
        is_last: bool
    ) -> None:
        self.epoch_train_steps += 1
        self.total_train_steps += 1
        self._current_time = time.time()
        self.train_metrics.update(preds, targets)
        self.train_loss.update(loss, count=preds.numel())
        self.new_state(TrainStateTrainStepEnd)

        flush_count = self // self._next_flush
        if flush_count > 0:
            self._next_flush += flush_count * self._next_flush
            self.train_metrics.reset()
            self.train_loss.reset()

        if is_last:
            self.is_epoch_last_step = True
            if self._expected_train_steps is None:
                self._expected_train_steps = self.epoch_train_steps

    @valid_transition
    def on_val_begin(self) -> None:
        self.new_state(TrainStateValBegin)
        self._current_time = time.time()
        self._epoch_val_start_time = self._current_time

    @valid_transition
    def on_val_step_begin(self) -> None:
        self._current_time = time.time()
        self.new_state(TrainStateValStepBegin)

    @valid_transition
    def on_val_step_end(
        self, 
        loss: torch.Tensor, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> None:
        self.new_state(TrainStateValStepEnd)
        self.epoch_val_steps += 1
        self.total_val_steps += 1
        self._current_time = time.time()
        self._total_val_time += self.epoch_val_time
        self.val_metrics.update(preds, targets)
        self.val_loss.update(loss, count=preds.numel())

    @valid_transition
    def on_val_end(self) -> None:
        self._current_time = time.time()
        self.new_state(TrainStateValEnd)

    @valid_transition
    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._current_time = time.time()
        self.new_state(TrainStateEpochEnd)

    @valid_transition
    def on_train_end(self) -> None:
        self.new_state(TrainState)
        self._current_time = time.time()

    def error(self, method: str) -> None:
        name = self.__class__.__name__
        raise RuntimeError(f"Cannot transition to state {method}. Current state: {name}.")

    @property
    def total_time(self) -> float:
        if self._current_time is None or self._start_time is None:
            return 0
        return self._current_time - self._start_time

    @property
    def total_train_time(self) -> float:
        if self._current_time is None or self._start_time is None:
            return 0
        return self._current_time - self._start_time - self.total_val_time

    @property
    def total_val_time(self) -> float:
        return self._total_val_time + self.epoch_val_time

    @property
    def epoch_total_time(self) -> float:
        if self._current_time is None or self._epoch_start_time is None:
            return 0
        return self._current_time - self._epoch_start_time

    @property
    def epoch_val_time(self) -> float:
        if self._epoch_val_start_time is None or self._current_time is None:
            return 0
        return self._current_time - self._epoch_val_start_time

    @property
    def epoch_train_time(self) -> float:
        """ 
        How much time was spent in training at the current epoch so far (excluding 
        validation).
        """
        if self._current_time is None or self._epoch_start_time is None:
            return 0
        return self._current_time - self._epoch_start_time - self.epoch_val_time

    def _get_field(self, unit: PeriodUnit) -> Period:
        value: Optional[int | float]
        if unit == PeriodUnit.EPOCHS:
            value = self.epoch
            if self._expected_train_steps is not None:
                value += self.epoch_train_steps / self._expected_train_steps
        elif unit == PeriodUnit.STEPS:
            value = self.total_train_steps
        elif unit == PeriodUnit.SECONDS:
            value = self.total_time
        else:
            ValueError(f"Invalid unit: {unit}")

        if value is None:
            raise ValueError(f"Value is None for unit: {unit}.")

        return Period(value, unit=unit)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Period):
            return False
        elif isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) == other

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __lt__(self, other: Period | str) -> Optional[bool]:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) < other

    def __le__(self, other: Period | str) -> Optional[bool]:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) <= other

    def __gt__(self, other: Period | str) -> Optional[bool]:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) > other

    def __ge__(self, other: Period | str) -> Optional[bool]:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) >= other

    def __add__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) + other
    
    __radd__ = __add__
    
    def __sub__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) - other
    
    def __rsub__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return other - self._get_field(other.unit)

    def __mul__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) * other
    
    __rmul__ = __mul__
    
    def __truediv__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) / other

    def __rtruediv__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return other / self._get_field(other.unit)

    def __floordiv__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) // other
    
    def __rfloordiv__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return other // self._get_field(other.unit)

    def __mod__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return self._get_field(other.unit) % other
    
    def __rmod__(self, other: Period | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return other % self._get_field(other.unit)

    def __repr__(self) -> str:
        return (
            f"\nTrainState:\n"
            f"\tepoch={self.epoch}\n"
            f"\tepoch train steps={self.epoch_train_steps}\n"
            f"\tepoch val steps={self.epoch_val_steps}\n"
            f"\ttotal_train_steps={self.total_train_steps}\n"
            f"\ttotal time={self.total_time:.2f} seconds\n"
            f"\tis epoch last step={self.is_epoch_last_step}\n"
        )


class TrainStateTrainBegin(TrainState):
    @property
    def transitions(self) -> set[str]:
        return {"on_epoch_begin"}


class TrainStateEpochBegin(TrainState):
    @property
    def transitions(self) -> set[str]:
        return {"on_train_step_begin"}


class TrainStateTrainStepBegin(TrainState):
    @property
    def transitions(self) -> set[str]:
        return {"on_train_step_end"}


class TrainStateTrainStepEnd(TrainState):
    @property
    def transitions(self) -> set[str]:
        return {"on_val_begin", "on_epoch_end", "on_train_step_begin", "on_train_end"}


class TrainStateValBegin(TrainState):
    @property
    def transitions(self) -> set[str]:
        return {"on_val_step_begin"}


class TrainStateValStepBegin(TrainState):
    @property
    def transitions(self) -> set[str]:
        return {"on_val_step_end"}


class TrainStateValStepEnd(TrainState): 
    @property
    def transitions(self) -> set[str]:
        return {"on_val_step_begin", "on_val_end"}


class TrainStateValEnd(TrainState):
    @property
    def transitions(self) -> set[str]:
        return {"on_epoch_end"}


class TrainStateEpochEnd(TrainState):
    @property
    def transitions(self) -> set[str]:
        return {"on_epoch_begin", "on_train_end"}
