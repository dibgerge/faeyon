from __future__ import annotations
import abc
import sys
import torch

from typing import Optional
from faeyon.enums import ClfTask
from torch.nn.functional import one_hot
from .base import Metric
from faeyon import utils


class Task(abc.ABC):
    _state: Optional[torch.Tensor]

    def __init__(self, metric: ClfMetricBase) -> None:
        self.metric = metric
        self.validate_metric()
        self._state = None

    @abc.abstractmethod
    def validate_metric(self) -> None:
        """ 
        Validates on the metric (without access to predictions/targets) and raises a 
        ValueError if invalid. 
        """
    
    @abc.abstractmethod
    def validate_inputs(self, preds: torch.Tensor, targets: torch.Tensor) -> bool:
        """
        preds and targets are expected to be of shape (B, C) when passed to this method.
        """
    
    @abc.abstractmethod
    def update(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]: ...

    @abc.abstractmethod
    def compute(self, state: torch.Tensor) -> dict[str, torch.Tensor]: ...

    @abc.abstractmethod
    def nclasses(self, preds: torch.Tensor, targets: torch.Tensor) -> Optional[int]: ...

    def validate(self, preds: torch.Tensor, targets: torch.Tensor) -> bool:
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)

        if preds.ndim != 2 or targets.ndim != 2:
            raise ValueError("Predictions/targets tensors must be of shape (B, C) or (B,).")
        return self.nclasses(preds, targets) is not None and self.validate_inputs(preds, targets)


class SparseTask(Task):
    def validate_metric(self) -> None:
        name = self.__class__.__name__
        if self.metric.multilabel:
            raise ValueError(f"{name} requires a metric with `multilabel=False`.")
        
        if self.metric.thresholds is not None:
            raise ValueError(f"{name} requires a metric with `thresholds=None`.")
                
        if self.metric.num_classes is not None and self.metric.num_classes == 1:
            raise ValueError(f"{name} requires a metric with `num_classes > 1`.")
        
    def validate_inputs(self, preds: torch.Tensor, targets: torch.Tensor) -> bool:
        tc, pc = targets.shape[-1], preds.shape[-1]
        if (nclasses := self.nclasses(preds, targets)) is None:
            return False

        conditions = [nclasses > 2]
        if tc == 1:
            conditions.append(utils.is_inrange(targets, min=0, max=nclasses-1))
        else:
            conditions.append(utils.is_onehot(targets))
     
        if pc == 1:
            conditions.append(self.metric.topk is None)
            conditions.append(utils.is_inrange(preds, min=0, max=nclasses-1))
            conditions.append(not preds.is_floating_point())
        else:
            conditions.append(preds.is_floating_point())

        return all(conditions)

    def nclasses(self, preds: torch.Tensor, targets: torch.Tensor) -> Optional[int]:
        nclasses: Optional[int] = None
        if targets.ndim == 1:
            tc = 1
        else:
            tc = targets.shape[-1]

        if preds.ndim == 1:
            pc = 1
        else:
            pc = preds.shape[-1]

        if tc > 1 and pc > 1:
            if tc != pc:
                return None
            else:
                nclasses = tc  
        elif tc > 1:
            nclasses = tc
        elif pc > 1:
            nclasses = pc
        else:
            nclasses = self.metric.num_classes

        if self.metric.num_classes is not None and nclasses != self.metric.num_classes:
            return None
        return nclasses

    def update(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        metric = self.metric
        if targets.ndim == 2 and targets.shape[-1] > 1:
            targets = targets.argmax(dim=-1)
        else:
            targets = targets.squeeze()

        if metric.topk is not None and metric.topk > 1:
            preds = preds.topk(metric.topk, dim=-1).indices
            pred_top = preds[:, 0]
            pred_tp = (targets.unsqueeze(-1) == preds).any(dim=-1)
            preds = torch.where(pred_tp, targets, pred_top)
        elif preds.ndim == 2 and preds.shape[-1] > 1:
            preds = preds.argmax(dim=-1)
        else:
            preds = preds.squeeze()

        items = torch.stack([preds, targets])

        size = (metric.num_classes, metric.num_classes)
        return items, size  # type: ignore[return-value]

    def compute(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        if state.ndim != 2:
            raise ValueError("State must be a 2D sparse tensor.")

        nclasses = state.shape[0]
        pred_idx, target_idx = state.indices()
        values = state.values()

        tp = torch.zeros(nclasses, device=values.device, dtype=values.dtype)
        fp = torch.zeros_like(tp)
        tn = torch.zeros_like(tp)
        fn = torch.zeros_like(tp)
        cardinality = torch.zeros_like(tp)

        tp_mask = pred_idx == target_idx
        itp_mask = ~tp_mask

        tp.scatter_add_(0, pred_idx[tp_mask], values[tp_mask])
        fp.scatter_add_(0, pred_idx[itp_mask], values[itp_mask])        
        fn.scatter_add_(0, target_idx[itp_mask], values[itp_mask])

        tn = values.sum() - (tp + fp + fn)
        cardinality.scatter_add_(0, target_idx, values)

        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "cardinality": cardinality}


class CategoricalTask(Task):
    def validate_metric(self) -> None:
        name = self.__class__.__name__
        if self.metric.multilabel:
            raise ValueError(f"{name} requires a metric with `multilabel=False`.")
        
        if self.metric.thresholds is None:
            raise ValueError(f"{name} requires a metric with `thresholds != None`.")
        
        if self.metric.topk is not None and self.metric.topk != 1:
            raise ValueError(f"{name} requires a metric with `topk=None` or `topk=1`.")

    def validate_inputs(self, preds: torch.Tensor, targets: torch.Tensor) -> bool:
        tc, pc = targets.shape[-1], preds.shape[-1]

        conditions = [
            pc >= 2,
            utils.is_probability(preds, dim=-1),
        ]
        if tc == 1:
            max_target = targets.max().item()
            min_target = targets.min().item()
            conditions.append((max_target < pc) and min_target >= 0)
        else:
            conditions.extend([
                utils.is_onehot(targets),
                tc == pc
            ])

        if self.metric.num_classes is not None:
            conditions.append(self.metric.num_classes != self.nclasses(preds, targets))

        return all(conditions)

    def nclasses(self, preds: torch.Tensor, targets: torch.Tensor) -> int:
        return preds.shape[-1]

    def update(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        n = preds.shape[0]
        metric = self.metric
        targets = targets.squeeze()
        if targets.ndim == 1:
            targets = one_hot(targets)
        
        nthresh = len(metric.thresholds)
        preds = (preds.unsqueeze(-1) >= metric.thresholds).to(torch.long)
        targets = targets.unsqueeze(-1).expand(n, -1, nthresh)
        classes = torch.arange(metric.num_classes).view(1, -1, 1).expand(n, -1, nthresh)
        thresholds = torch.arange(nthresh).view(1, 1, -1).expand(n, metric.num_classes, nthresh)
        items = torch.stack([classes, thresholds, preds, targets]).view(4, -1)
        size = (metric.num_classes, nthresh, 2, 2)
        return items, size

    def compute(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        state = state.to_dense()
        return {
            "tp": state[..., 1, 1],
            "fp": state[..., 1, 0],
            "tn": state[..., 0, 0],
            "fn": state[..., 0, 1],
            "cardinality": state.sum(dim=(-2, -1))
        }


class BinaryProbabilityTask(CategoricalTask):
    def validate_metric(self) -> None:
        name = self.__class__.__name__
        if self.metric.multilabel:
            raise ValueError(
                f"metric with `multilabel=True` is not supported for {name}.")

        if self.metric.thresholds is None:
            raise ValueError(f"{name} requires a metric with `thresholds`.")

    def validate_inputs(self, preds: torch.Tensor, targets: torch.Tensor) -> bool:
        tc, pc = targets.shape[-1], preds.shape[-1]
        conditions = [
            # Targets conditions
            tc in [1, 2],
            utils.is_binary(targets) if tc == 1 else utils.is_onehot(targets),

            # Preds conditions
            pc == 1,
            utils.is_probability(preds),
        ]
    
        if self.metric.num_classes is not None:
            conditions.append(self.metric.num_classes == self.nclasses(preds, targets))

        return all(conditions)

    def nclasses(self, preds: torch.Tensor, targets: torch.Tensor) -> Optional[int]:
        return 1

    def update(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        n = preds.shape[0]
        thresholds: torch.Tensor = self.metric.thresholds  # type: ignore

        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        
        if targets.shape[-1] == 2:
            targets = targets.argmax(dim=-1, keepdim=True)

        preds = (preds >= thresholds).to(torch.long)
        nthresh = len(thresholds)
        targets = targets.expand(n, nthresh)
        thresholds = torch.arange(nthresh).expand(n, -1)
        items = torch.stack([thresholds, preds, targets]).view(3, -1)
        size = (nthresh, 2, 2)
        return items, size


class BinaryTwoClassTask(SparseTask):
    def validate_metric(self) -> None:
        name = self.__class__.__name__
        if self.metric.multilabel:
            raise ValueError(
                f"metric with `multilabel=True` is not supported for {name}.")

        if self.metric.thresholds is not None:
            raise ValueError(f"{name} requires a metric with `thresholds=None`.")

        if self.metric.topk is not None and self.metric.topk > 1:
            raise ValueError(f"metric with `topk` must be `None` or `1` for {name}.")

    def validate_inputs(self, preds: torch.Tensor, targets: torch.Tensor) -> bool:        
        tc, pc = targets.shape[-1], preds.shape[-1]
        
        conditions = [
            # Targets conditions
            targets.ndim == 2,
            tc in [1, 2],

            # Preds conditions
            preds.ndim == 2,
            pc == 2,
            preds.is_floating_point(),
        ]

        if self.metric.num_classes is not None:
            conditions.append(self.metric.num_classes == self.nclasses(preds, targets))

        return all(conditions)

    def nclasses(self, preds: torch.Tensor, targets: torch.Tensor) -> int:
        return 1

    def update(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if preds.ndim == 2 and preds.shape[-1] == 2:
            preds = preds.argmax(dim=-1)
        else:
            preds = preds.squeeze()
        
        if targets.ndim == 2 and targets.shape[-1] == 2:
            targets = targets.argmax(dim=-1)
        else:
            targets = targets.squeeze()

        items = torch.stack([preds, targets])
        size = (2, 2)
        return items, size


class BinarySparseTask(BinaryTwoClassTask):
    def validate_metric(self) -> None:
        name = self.__class__.__name__
        conditions = {
            "multilabel": False,
            "thresholds": None,
            "topk": None
        }
        for k, value in conditions.items():
            if getattr(self.metric, k) != value:
                raise ValueError(f"{name} requires a metric with `{k}={value}`.")

    def validate_inputs(self, preds: torch.Tensor, targets: torch.Tensor) -> bool:
        tc, pc = targets.shape[-1], preds.shape[-1]
            
        conditions = [
            # Targets conditions
            tc in [1, 2],
            utils.is_binary(targets) if tc == 1 else utils.is_onehot(targets),

            # Preds conditions
            pc == 1,
            utils.is_binary(preds),
        ]
        if tc == 1:
            # Numclasses must be specified if 1-D targets
            conditions.append(self.metric.num_classes == 1)

        return all(conditions)


class MultilabelTask(CategoricalTask):
    def validate_metric(self) -> None:
        name = self.__class__.__name__
        if not self.metric.multilabel:
            raise ValueError(f"{name} requires a metric with `multilabel=True`.")
        
        if self.metric.thresholds is None:
            raise ValueError(f"{name} requires a metric with `thresholds != None`.")
        
        if self.metric.topk is not None:
            raise ValueError(f"{name} requires a metric with `topk=None`.")
    
    def validate_inputs(self, preds: torch.Tensor, targets: torch.Tensor) -> bool:
        tc, pc = targets.shape[-1], preds.shape[-1]
        return all([
            tc == pc,
            pc >= 2,
            utils.is_binary(targets),
            utils.is_probability(preds),
        ])
    

def select_task(
    metric: ClfMetricBase, 
    preds: torch.Tensor, 
    targets: torch.Tensor
) -> Task:
    # Common checks on inputs so we don't need to repeat them for each task class.
    if (
        preds.ndim not in [1, 2] 
        or targets.ndim not in [1, 2] 
        or preds.shape[0] != targets.shape[0]
    ):
        raise ValueError(
            f"Predictions/targets tensor must be of shape (B,) or (B, C). Got:\n"
            f"\tPredictions: {preds.shape} \n"
            f"\tTargets: {targets.shape}"
        ) 

    if targets.is_floating_point():
        raise ValueError(
            f"Targets tensor must be integer tensor with shape (B,) or (B, C). Got:\n"
            f"\tTargets dtype: {targets.dtype} and shape {targets.shape}"
        )

    classes = sys.modules[__name__].__dict__.values()
    tasks = []
    for cls in classes:
        if isinstance(cls, type) and issubclass(cls, Task) and cls is not Task:
            tasks.append(cls)
        
    valid_tasks = []
    for task_cls in tasks:
        try:
            task = task_cls(metric)
        except ValueError:
            continue
        else:
            if task.validate(preds, targets):
                valid_tasks.append(task)
    
    if len(valid_tasks) > 1:
        raise ValueError("Multiple valid tasks found, cannot determine which one to use.")
    
    if len(valid_tasks) == 0:
        raise ValueError(f"""
                Unsupported combination of predictions and targets.
                Predictions shape: {preds.shape}\tFloating point: {preds.is_floating_point()}
                Targets shape: {targets.shape} 
                Multilabel: {metric.multilabel}
                Number of classes: {metric.num_classes}
            """)
    
    return valid_tasks[0]


class ClfMetricBase(Metric):
    """
    num_classes: Optional[int]
        Will be inferred from the first targets/preds update if not provided. 
        This might not always be accurate, for example if targets/preds are not one-hot encoded.
    """
    thresholds: Optional[torch.Tensor]
    num_classes: Optional[int]
    _state: Optional[torch.Tensor]
    task: Optional[Task]

    def __init__(
        self, 
        num_classes: Optional[int] = None,
        thresholds: Optional[int | float | list[float] | torch.Tensor] = None,
        topk: Optional[int] = None,
        subset: Optional[list[int]] = None,
        average: Optional[str] = None,
        multilabel: bool = False,
    ):
        if thresholds is not None and topk is not None:
            raise ValueError("Cannot use both `thresholds` and `topk`.")

        if multilabel:        
            if thresholds is None:
                raise ValueError("`thresholds` is required for multilabel predictions.")
                    
        match num_classes:
            case x if x is not None and not isinstance(x, int):
                raise ValueError("`num_classes` must be an integer.")

            case x if x is not None and x < 1:
                raise ValueError("`num_classes` must be greater than 0.")

            case 1 | 2 if not multilabel:
                self.num_classes = 1
                
            case _:
                self.num_classes = num_classes

        match thresholds:
            case int():
                self.thresholds = torch.linspace(0, 1, thresholds)
            
            case float():
                self.thresholds = torch.tensor([thresholds])
                        
            case list() | tuple():
                self.thresholds = torch.tensor(thresholds)

            case torch.Tensor():
                self.thresholds = thresholds

            case None:
                self.thresholds = None

            case _:
                raise ValueError(
                    "`thresholds` must be an integer, float, or list/tensor of floats.")
        
        if self.thresholds is not None and (self.thresholds.max() > 1 or self.thresholds.min() < 0):
            raise ValueError("`thresholds` tensor values must all be in [0, 1].")

        self.task = None
        self.subset = subset
        self.average = average
        self.multilabel = multilabel
        self.topk = topk
        self.reset()

    def reset(self) -> None:
        self._state = None

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        predictions: (B, )
        targets: (B, ) or (B, C)

        where 
            - C: Number of classes
            - B: Batch size
        """
        if self.task is None:
            self.task = select_task(self, preds, targets)
        else:
            if not self.task.validate(preds, targets):
                raise ValueError(
                    "Given predictions/targets are not compatible with the current task.")
        
        if self.task is None:
            raise ValueError("Something went wrong, could not set Task.")

        nclasses = self.task.nclasses(preds, targets)
        if nclasses is None:
            raise ValueError("Something went wrong, could not infer number of classes.")
        elif self.num_classes is None:
            self.num_classes = nclasses

        indices, size = self.task.update(preds, targets)

        # sparse coo coalesce add the values of repeated indices
        values = torch.ones(indices.shape[-1], dtype=torch.long, device=indices.device)
        data = torch.sparse_coo_tensor(indices, values, dtype=torch.long, size=size).coalesce()

        if self._state is None:
            self._state = data
        else:
            self._state += data


class Accuracy(ClfMetricBase):
    def _compute_sparse(self, confusion: dict[str, torch.Tensor]) -> torch.Tensor:
        tp = confusion["tp"]
        cardinality = confusion["cardinality"]
        total = cardinality.sum()

        match self.average:
            case "macro":
                caccuracy = tp / cardinality
                return caccuracy.mean()
            case "micro":
                return tp.sum() / total
            case "weighted":
                caccuracy = tp / cardinality
                return (cardinality * caccuracy / total).sum()
            case "none" | None:
                return tp / cardinality
            case _:
                raise ValueError(f"Invalid average: {self.average}")
    
    def _compute_categorical(self, confusion: dict[str, torch.Tensor]) -> torch.Tensor:
        tp = confusion["tp"]
        tn = confusion["tn"]
        cardinality = confusion["cardinality"]
        return (tp + tn) / cardinality

    def compute(self) -> torch.Tensor:
        if self._state is None or self.task is None:
            raise ValueError("State is empty. Call update() before compute().")

        confusion = self.task.compute(self._state)
        if self._state.ndim == 2:
            return  self._compute_sparse(confusion)
        else:
            return self._compute_categorical(confusion)


class Precision(ClfMetricBase):
    def compute(self) -> float:
        return (self.predictions == self.targets).mean()


class Recall(ClfMetricBase):
    def compute(self) -> float:
        return (self.predictions == self.targets).mean()


class F1Score(ClfMetricBase):
    def compute(self) -> float:
        return (self.predictions == self.targets).mean()


class PRCurve(ClfMetricBase):
    def compute(self) -> float:
        return (self.predictions == self.targets).mean()


class ROC(ClfMetricBase):
    def compute(self) -> float:
        return (self.predictions == self.targets).mean()

class AUC(ClfMetricBase):
    def compute(self) -> float:
        return (self.predictions == self.targets).mean()


class ROCCurve(ClfMetricBase):
    def compute(self) -> float:
        return (self.predictions == self.targets).mean()

