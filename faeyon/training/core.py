from __future__ import annotations
import enum
from dataclasses import dataclass
from torch.utils.data import DataLoader
import itertools
import time
import fnmatch
import importlib
import re
from .callbacks import Callback

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

    def __post_init__(self) -> None:
        nones = [self.epochs is None, self.steps is None, self.ts is None]

        if self.condition not in ["all", "any", "epochs", "steps", "ts", None]:
            raise ValueError(
                f"Invalid condition: {self.condition}. Must be one of: 'all', 'any', 'epochs', "
                f"'steps', 'ts'."
            )
                
        if self.condition in ["epochs", "steps", "ts"] and any(nones):
            raise ValueError(
                f"`condition` is {self.condition}, which requires all `epochs`, `steps`, and `ts` "
                "to be specified."
            )

        if sum(nones) == 1 and self.condition is not None:
            raise ValueError(
                f"Only one field is specified, `condition` is {self.condition}"
            )

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
        # Only three occurences of the pattern are allowed (epoch, steps, time - one of seconds, 
        # hours, days). Repeated occurences of one type is not allowed, will be checked in code.
        validator = rf"""
        \s*
        {pattern}
        (?:\s*([|&])\s*{pattern})?
        (?:\s*([|&])\s*{pattern})?
        \s*
        """
        matcher = re.fullmatch(validator, expr, re.IGNORECASE | re.VERBOSE)

        if not matcher:
            raise ValueError(
                f"Invalid period expression: {expr}. Make sure that only three occurences of the "
                f"pattern are allowed (epoch, steps, time - one of seconds, hours, days). Repeated "
                f"occurences of one type is not allowed."
            )

        groups = matcher.groups()
        amounts = {"epochs": None, "steps": None, "ts": None}
        group_map = {1: "epochs", 2: "steps", 3: "ts", 4: "ts", 5: "ts", 6: "ts"}
        time_map = {3: "seconds", 4: "minutes", 5: "hours", 6: "days"}
        keys, conditions = [], []

        # The pattern has 7 capturing groups + 1 condition group, and 3 repetitions of the pattern.
        for i in [0, 8, 16]:
            if groups[i] is None:
                continue

            value = float(groups[i])
            
            # find indices of non-None groups
            indices = [j for j in range(1, 7) if groups[i+j] is not None]

            if len(indices) != 1:
                raise ValueError(
                    f"Invalid period expression: {expr}.")

            index = indices[0]
            key = group_map[index]

            if amounts[key] is not None:
                raise ValueError(
                    f"Invalid period expression: {expr}. Repeated occurences of one type "
                    "is not allowed."
                )

            if i + 7 < len(groups) and groups[i + 7] is not None:
                conditions.append(groups[i + 7])
            keys.append(key)

            if key == "steps":
                if value.is_integer():
                    amounts["steps"] = int(value)
                else:
                    raise ValueError(f"Fractional steps are not supported {expr}.")
            elif key == "ts":
                value = timedelta(**{time_map[index]: value})

            amounts[key] = value

        if len(conditions) == 0:
            return cls(**amounts)

        conditions_set = set(conditions)
        if len(conditions_set) == 2:
            idx = conditions.index("&")
            operands  = {keys[idx], keys[idx + 1]}
            condition = set(amounts.keys()) - operands
            condition = condition.pop()
        elif conditions_set == {"&"}:
            condition = "all"
        elif conditions_set == {"|"}:
            condition = "any"
        else:
            raise ValueError(f"Invalid period expression: {expr}.")
        return cls(**amounts, condition=condition)

    def __eq__(self, other: Period) -> bool:
        return (
            self.epochs == other.epochs 
            and self.steps == other.steps 
            and self.ts == other.ts 
            and self.condition == other.condition
        )

    def __iadd__(self, other: str | Period) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)

        self.epochs += other.epochs
        self.steps += other.steps
        self.ts += other.ts
        return self

    def __add__(self, other: str | Period) -> Period:
        if isinstance(other, str):
            other = Period.from_expr(other)
        return Period(
            epochs=self.epochs + other.epochs,
            steps=self.steps + other.steps,
            ts=self.ts + other.ts, 
            condition=self.condition
        )

    def __ne__(self, other: Period) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return (
            f"Period("
                f"epochs={self.epochs}, "
                f"steps={self.steps}, "
                f"ts={self.ts}, "
                f"condition={self.condition}"
            f")"
        )


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    start_time: Optional[float] = None
    current_time: Optional[float] = None
    epoch_step: int = 0
    epoch_start: Optional[float] = None
    epoch_time: Optional[float] = None
    metrics: Optional[dict[str, float]] = None

    # def __init__(self) -> None:
    #     self.epoch = 
    #     self.epoch_step = None
    #     self.metrics = None

    def reset(self) -> None:
        self.period = Period()
        self.epoch_step = 0
        self.metrics = None

    def toc(self):
        self.epoch += 1
        self.epoch_step = 0
        self.epoch_ts = 0
        self.metrics = None

    def tic(self) -> None:
        self.step += 1
        self.epoch_step += 1
        self.epoch_ts += 1
        self.ts += 1
