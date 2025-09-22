from __future__ import annotations
import enum
from dataclasses import dataclass
from math import e
from torch.utils.data import DataLoader
import itertools
import time
import fnmatch
import importlib
import re
from .callbacks import Callback

from numbers import Number
from datetime import timedelta

from typing import Optional, Iterable, Literal
from torch import nn, optim


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
            module = "optim"

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


class PeriodUnit(enum.Enum):
    EPOCHS = "epochs"
    STEPS = "steps"
    SECONDS = "seconds"


class Period:
    """
    condition: str
        This can be "all", "any", "epochs", "steps", "ts". with the following meaning:
        - "all": epochs & steps & ts
        - "any": epochs | steps | ts
        - "epochs": epochs | steps & ts
        - "steps": steps | epochs & ts
        - "ts": ts | epochs & steps
    """
    epochs: Optional[float] = None
    steps: Optional[int] = None
    ts: Optional[timedelta] = None
    condition: Optional[Literal["all", "any", "epochs", "steps", "ts"]] = None

    def __init__(self, value: float | int, unit: PeriodUnit) -> None:
        self.value = value
        self.unit = unit

    @classmethod
    def from_expr(cls, expr: str):
        pattern = r"""
        ([0-9]*\.?[0-9]+)       # Amount (Group 1)
        (?:
            (e(?:pochs?)?)  |   # Epochs (Group 2)
            (steps?)        |   # Steps (Group 3)
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

    def _validate_period(self, other: Period | Number | str) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        elif isinstance(other, Number):
            other = Period(other, self.unit)

        if other.unit != self.unit:
            raise ValueError(
                f"Given period has different units than the current period: "
                f"{self.unit} and {other.unit}."
            )
        return other

    def __iadd__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        self.value += other.value
        return self

    def __add__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value + other.value, unit=self.unit)

    __radd__ = __add__

    def __isub__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        self.value -= other.value
        return self
        
    def __sub__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value - other.value, unit=self.unit)

    def __rsub__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        return Period(value=other.value - self.value, unit=self.unit)

    def __imul__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        self.value *= other.value
        return self

    def __mul__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        return Period(value=self.value * other.value, unit=self.unit)

    __rmul__ = __mul__

    def __itruediv__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        self.value /= other.value
        return self

    def __truediv__(self, other: Period | Number | str) -> float:
        other = self._validate_period(other)
        return self.value / other.value

    def __rtruediv__(self, other: Period | Number | str) -> float:
        other = self._validate_period(other)
        return other.value / self.value

    def __ifloordiv__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        self.value //= other.value
        return self

    def __floordiv__(self, other: Period | Number | str) -> int:
        other = self._validate_period(other)
        return self.value // other.value

    def __rfloordiv__(self, other: Period | Number | str) -> int:
        other = self._validate_period(other)
        return other.value // self.value

    def __imod__(self, other: Period | Number | str) -> Period:
        other = self._validate_period(other)
        self.value %= other.value
        return self

    def __mod__(self, other: Period | Number | str) -> float:
        other = self._validate_period(other)
        return self.value % other.value

    def __rmod__(self, other: Period | Number | str) -> float:
        other = self._validate_period(other)
        return other.value % self.value

    def __eq__(self, other: Period | Number | str) -> bool:
        return self.value == self._validate_period(other).value

    def __ne__(self, other: Period | Number | str) -> bool:
        return not self == other
    
    def __lt__(self, other: Period | Number | str) -> bool:
        return self.value < self._validate_period(other).value

    def __le__(self, other: Period | Number | str) -> bool:
        return self.value <= self._validate_period(other).value
    
    def __gt__(self, other: Period | Number | str) -> bool:
        return self.value > self._validate_period(other).value
    
    def __ge__(self, other: Period | Number | str) -> bool:
        return self.value >= self._validate_period(other).value

    def __repr__(self) -> str:
        return f"Period({self.value}, {self.unit})"


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    start_time: Optional[float] = None
    current_time: Optional[float] = None
    train_metrics: Optional[dict[str, float]] = None

    epoch_step: int = 0
    epoch_start: Optional[float] = None

    val_start: Optional[float] = None
    val_step: int = 0
    val_metrics: Optional[dict[str, float]] = None

    def reset(self) -> None:
        self.epoch = 0
        self.step = 0
        self.start_time = None
        self.current_time = None
        self.train_metrics = None

        self.epoch_step = 0
        self.epoch_start = None

        self.val_step = 0
        self.val_start = None
        self.val_metrics = None

    @property
    def run_time(self) -> float:
        if self.start_time is None or self.current_time is None:
            return 
        return self.current_time - self.start_time

    @property
    def epoch_time(self) -> float:
        if self.epoch_start is None or self.current_time is None:
            return 
        return self.current_time - self.epoch_start

    @property
    def val_time(self) -> float:
        if self.val_start is None or self.current_time is None:
            return 
        return self.current_time - self.val_start

    def toc(self, train: bool = True) -> None:
        if train:
            self.epoch += 1
            self.epoch_step = 0
            self.epoch_start = time.time()
        else:
            self.val_step = 0
            self.val_start = time.time()

        if self.start_time is None:
            self.start_time = self.epoch_start
        
    def tic(self, train: bool = True) -> None:
        self.current_time = time.time()
        if train:
            self.step += 1
            self.epoch_step += 1
        else:
            self.val_step += 1

    def is_epoch_begin(self) -> bool:
        return self.epoch_step == 1

    def _get_field(self, unit: PeriodUnit) -> float:
        if unit == PeriodUnit.EPOCHS:
            value = self.epoch
        elif unit == PeriodUnit.STEPS:
            value = self.step
        elif unit == PeriodUnit.SECONDS:
            value =self.run_time
        else:
            ValueError(f"Invalid unit: {unit}")
        return Period(value, unit=unit)

    def __eq__(self, period: Period) -> bool:
        return self._get_field(period.unit) == period

    def __lt__(self, period: Period) -> bool:
        state_period = Period(self._get_field(period.unit), unit=period.unit)
        return state_period < period

    def __le__(self, period: Period) -> bool:
        state_period = Period(self._get_field(period.unit), unit=period.unit)
        return state_period <= period

    def __gt__(self, period: Period) -> bool:
        state_period = Period(self._get_field(period.unit), unit=period.unit)
        return state_period > period

    def __ge__(self, period: Period) -> bool:
        state_period = Period(self._get_field(period.unit), unit=period.unit)
        return state_period >= period

    def __ne__(self, period: Period) -> bool:
        return not self == period

    def __truediv__(self, other: Period | Number | str) -> float:
        other = self._validate_period(other)
        return self.value / other.value

    def __floordiv__(self, other: Period | Number | str) -> int:
        other = self._validate_period(other)
        return self.value // other.value

