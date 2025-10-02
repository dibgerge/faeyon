from __future__ import annotations
from dataclasses import dataclass
import time
import fnmatch
import importlib
import re
from datetime import timedelta

from typing import Optional, Iterable, Literal, Union
from torch import nn, optim
from torch.utils.data import DataLoader
from faeyon.enums import PeriodUnit, TrainStateMode, TrainStage
from faeyon.metrics import MetricCollection


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
    metrics: MetricCollection
    epoch: int

    total_train_steps: int

    total_val_steps: int

    epoch_train_steps: int
    epoch_val_steps: int

    _start_time: float
    _epoch_start_time: float
    _epoch_val_start_time: float
    _current_time: float
    _total_val_time: float

    def __init__(self, metrics: MetricCollection) -> None:
        self.metrics = metrics
        self.reset()

    @classmethod
    def from_state(cls, state: TrainState) -> None:
        obj = object.__new__(cls)
        obj.metrics = state.metrics
        return obj

    def copy(self, other: TrainState) -> None:
        #TODO: 
        self.metrics = other.metrics
        self._started = other._started

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

    def on_train_begin(self) -> None:
        """
        If the model was trained before, and epoch is > 0, we continue from where 
        we left off.
        """
        self._start_time = time.time()
        self._current_time = self._start_time

    def on_epoch_begin(self) -> None:
        self.epoch += 1
        self.epoch_train_steps = 0
        self._epoch_start_time = time.time()
        self._current_time = self._epoch_start_time

    def on_train_step_begin(self) -> None:
        self.epoch_train_steps += 1
        self.total_train_steps += 1
        self._current_time = time.time()

    def on_val_begin(self) -> None:
        self.epoch_val_steps = 0
        self._current_time = time.time()
        self._epoch_val_start_time = self._current_time

    def on_val_step_begin(self) -> None:
        self.epoch_val_steps += 1
        self.total_val_steps += 1
        self._current_time = time.time()
    
    def on_val_step_end(self) -> None:
        self._current_time = time.time()
    
    def on_val_end(self) -> None:
        self._current_time = time.time()
        self._total_val_time += self.epoch_val_time
        self._epoch_val_start_time = None

    def on_epoch_end(self) -> None:
        self._current_time = time.time()                

    @property
    def total_time(self) -> float:
        return self._current_time - self._start_time

    @property
    def epoch_total_time(self) -> float:
        return self._current_time - self._epoch_start_time

    @property
    def epoch_val_time(self) -> float:
        if self._epoch_val_start_time is None:
            return 0
        return self._current_time - self._epoch_val_start_time

    @property
    def total_train_time(self) -> float:
        return self._current_time - self._start_time - self.total_val_time

    @property
    def total_val_time(self) -> float:
        return self._total_val_time + self.epoch_val_time

    @property
    def epoch_train_time(self) -> float:
        """ 
        How much time was spent in training at the current epoch so far (excluding 
        validation).
        """
        return self._current_time - self._epoch_start_time - self.epoch_val_time

    @property
    def started(self) -> bool:
        # TODO: review conditions
        if self._start_time is None or self.stage is None:
            return False
        return (self._current_time - self._start_time) > 0

    def _get_field(self, unit: PeriodUnit) -> Period:
        value: Optional[int | float]
        if unit == PeriodUnit.EPOCHS:
            value = self.epoch
        elif unit == PeriodUnit.STEPS:
            value = self.step
        elif unit == PeriodUnit.SECONDS:
            value = self.run_time
        else:
            ValueError(f"Invalid unit: {unit}")

        if value is None:
            raise ValueError(f"Value is None for unit: {unit}.")

        return Period(value, unit=unit)

    def __eq__(self, period: object) -> bool:
        if not isinstance(period, Period):
            return False
        return self._get_field(period.unit) == period

    def __ne__(self, period: object) -> bool:
        return not self == period

    def __lt__(self, period: Period) -> bool:
        return self._get_field(period.unit) < period

    def __le__(self, period: Period) -> bool:
        return self._get_field(period.unit) <= period

    def __gt__(self, period: Period) -> bool:
        return self._get_field(period.unit) > period

    def __ge__(self, period: Period) -> bool:
        return self._get_field(period.unit) >= period

    def __truediv__(self, other: Period | Number | str) -> float:
        return self._get_field(other.unit) / other.value

    def __floordiv__(self, other: Period | Number | str) -> int:
        return self._get_field(other.unit) // other.value

