import torch

from typing import Protocol, Optional, Literal
from faeyon.enums import ClfTask
from torch.nn.functional import one_hot


class Metric(Protocol):
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None: ...

    def compute(self) -> float: ...

    def reset(self) -> None: ...


class ClfMetricBase(Metric):
    """
    num_classes: Optional[int]
        Will be inferred from the first targets/preds update if not provided. 
        This might not always be accurate, for example if targets/preds are not one-hot encoded.
    """
    thresholds: Optional[torch.Tensor]
    task: Optional[ClfTask]

    def __init__(
        self, 
        num_classes: Optional[int] = None,
        thresholds: Optional[int | float | list[float] | torch.Tensor] = None,
        topk: Optional[int] = None,
        subset: Optional[list[int]] = None,
        average: Optional[str] = "weighted",
        multilabel: bool = False,
    ):
        if thresholds is not None and topk is not None:
            raise ValueError("Cannot use both `thresholds` and `topk`.")

        if multilabel:        
            if topk is not None:
                raise ValueError("`topk` is not supported for multilabel predictions.")
            if thresholds is None:
                raise ValueError("`thresholds` is required for multilabel predictions.")
            
            self.task = ClfTask.MULTILABEL
        
        match num_classes:
            case x if x is not None and not isinstance(x, int):
                raise ValueError("`num_classes` must be an integer.")

            case x if x is not None and x < 1:
                raise ValueError("`num_classes` must be greater than 0.")

            case 1 | 2 if not multilabel:
                self.task = ClfTask.BINARY
                self.num_classes = 2
                
            case _:
                self.task = None
                self.num_classes = num_classes

        match thresholds:
            case int():
                self.thresholds = torch.linspace(0, 1, thresholds)
            
            case float():
                self.thresholds = torch.tensor([thresholds])
            
            case torch.Tensor():
                if thresholds.max() > 1 or thresholds.min() < 0:
                    raise ValueError("`thresholds` tensor values must all be in [0, 1].")
            
            case list() | tuple():
                if max(thresholds) > 1 or min(thresholds) < 0:
                    raise ValueError("`thresholds` list/tuple values must all be in [0, 1].")
                self.thresholds = torch.tensor(thresholds)

            case None:
                self.thresholds = None

            case _:
                raise ValueError(
                    "`thresholds` must be an integer, float, or list/tensor of floats.")

        self.subset = subset
        self.average = average
        self.multilabel = multilabel
        self.topk = topk
        self.reset()

    def reset(self) -> None:
        pass

    def _infer_task(self, preds: torch.Tensor, targets: torch.Tensor) -> ClfTask:
        """
        Infer the task from the predictions and targets.
        """
        if preds.ndim not in [1, 2] or targets.ndim not in [1, 2]:
            raise ValueError(
                f"Predictions/targets tensor must be of shape (B,) or (B, C). Got:\n"
                f"\tPredictions: {preds.shape} \n"
                f"\tTargets: {targets.shape}"
            )

        max_pred = preds.max().item()
        min_pred = preds.min().item()

        if preds.ndim == 2:
            pclasses = preds.shape[-1]
        else:
            pclasses = None

        # Validate targets
        if targets.is_floating_point():
            raise ValueError(
                "Targets tensor must be an integer tensor for shape (B,) or (B, C).")

        max_target = targets.max().item()
        min_target = targets.min().item()
        if targets.ndim == 2:
            # 2-D targets must be one-hot encoded
            if max_target > 1 or min_target < 0:
                raise ValueError("Only one-hot encoded targets are supported for shape (B, C).")

            tclasses = targets.shape[-1]
        else:
            if min_target < 0:
                raise ValueError("Targets must be non-negative for shape (B,).")
            tclasses = None

        mismatch = False
        if pclasses is not None and tclasses is not None:
            mismatch = pclasses != tclasses

        if self.num_classes is not None:
            if pclasses is not None:
                mismatch = pclasses != self.num_classes
        
            if tclasses is not None:
                mismatch = tclasses != self.num_classes

            if max_target + 1 > self.num_classes:
                mismatch = True

            if not preds.is_floating_point() and max_pred + 1 > self.num_classes:
                mismatch = True

        if mismatch:
            raise ValueError(
                "Number of classes invalid for predictions or targets:\n"
                f"\tTarget: {tclasses}\n"
                f"\tPredictions: {pclasses}\n"
                f"\tNum classes: {self.num_classes}"
            )

        # Infer task
        if self.multilabel:
            if preds.ndim == 1:
                raise ValueError("Multilabel is not supported for preds of shape (B,).")
            else:
                if preds.max() > 1 or preds.min() < 0:
                    raise ValueError(
                        "Multilabel predictions of shape (B, C) must be probabilities.")
            
            if targets.ndim == 1:
                raise ValueError(
                    "Multilabel is not supported for targets of shape (B,). Targets must "
                    "be one-hot encoded for multilabel classification."
                )
            else:
                if targets.shape[-1] == 1:
                    raise ValueError("Multilabels are not supported for targets of shape (B, 1).")
            return ClfTask.MULTILABEL

        match(preds.ndim):
            # Binary task
            case 1 if preds.is_floating_point():
                
                if self.topk is not None:
                    raise ValueError("Topk is not supported for binary predictions.")

                if max_pred > 1 or min_pred < 0:
                    raise ValueError(
                        "Predictions as float tensors of shape (B,) must be probabilities."
                    )
                
                if self.thresholds is None:
                    raise ValueError("Binary predictions require `threshold`.")
                    
                if self.average is not None:
                    raise ValueError("`average` is not supported for binary predictions.")

                if max_target > 1:
                    raise ValueError("targets must be 0 or 1 for binary task.")

                if targets.ndim == 2:
                    if targets.shape[-1] > 2:
                        raise ValueError("2-D targets must be of shape (B, 2) or (B, 1).")

                if self.num_classes is None:
                    self.num_classes = 2

                return ClfTask.BINARY
            
            # Sparse task
            case 1: 
                if self.thresholds is not None:
                    raise ValueError("`threshold is not supported for sparse predictions.")
                
                if self.topk is not None:
                    raise ValueError("Topk is not supported for sparse predictions.")

                if self.num_classes is None:
                    if tclasses is None:
                        raise ValueError(
                            "Cannot infer number of classes from sparse inputs. `num_classes` "
                            "needs to be provided, or use categorical predictions and/or targets."
                        )
                    else:
                        self.num_classes = tclasses

                return ClfTask.SPARSE
            case 2:
                if not preds.is_floating_point():
                    raise ValueError(
                        "Predictions of shape (B, C) must be floating point representing "
                        "logits or probabilities."
                    )
                
                if self.num_classes is None:
                    self.num_classes = pclasses

                return ClfTask.CATEGORICAL

            case _:
                raise ValueError(
                    f"Predictions tensor must be of shape (B,) or (B, C). Got {preds.shape}")
     
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        predictions: (B, )
        targets: (B, ) or (B, C)

        where 
            - C: Number of classes
            - B: Batch size
        """
        task = self._infer_task(preds, targets)

        if self.task is None:
            self.task = task
        elif self.task != task:
            raise ValueError(
                "Mismatch between inferred and specified tasks:\n"
                f"\tSpecified: {self.task}\n"
                f"\tInferred: {task}"
            )

        n = preds.shape[0]
        c = self.num_classes

        match self.task:
            case ClfTask.BINARY:
                # One-hot encoded binary targets would be of shape (B, 2)
                if targets.ndim == 2:
                    if targets.shape[-1] == 2:
                        targets = targets.argmax(dim=-1, keepdim=True)
                else:
                    targets = targets.unsqueeze(-1)

                preds = (preds.unsqueeze(-1) >= self.thresholds).to(torch.long)
                nthresh = len(self.thresholds)
                targets = targets.expand(n, nthresh)
                thresholds = self.thresholds.expand(n, -1)
                items = torch.stack([thresholds, preds, targets], dim=-1).view(-1, 3)
    
            case ClfTask.SPARSE:                
                if targets.ndim == 2:
                    # Convert one-hot encoded targets to sparse
                    targets = targets.argmax(dim=-1)

                if self.topk is not None:
                    if self.topk > 1:
                        preds = preds.topk(self.topk, dim=-1).indices
                        pred_top = preds[:, 0]
                        pred_tp = (targets == preds).any(dim=-1)
                        preds = torch.where(pred_tp, targets, pred_top)
                    else:
                        # convert prediction probabilities to sparse class numbers
                        preds = preds.argmax(dim=-1)

                items = torch.column_stack([preds, targets])
                
            case ClfTask.CATEGORICAL | ClfTask.MULTILABEL:
                # This task means thresholds are given with categorical predictions
                if targets.ndim == 1:
                    targets = one_hot(targets)
                
                nthresh = len(self.thresholds)
                preds = (preds.unsqueeze(-1) >= self.thresholds).to(torch.long)
                targets = targets.unsqueeze(-1).expand(n, -1, nthresh)
                classes = torch.arange(self.num_classes).view(1, -1, 1).expand(n, -1, nthresh)
                thresholds = self.thresholds.view(1, 1, -1).expand(n, self.num_classes, 1)
                items = torch.stack([classes, thresholds, preds, targets], dim=-1).view(-1, 4)

        indices, counts = torch.unique(items, dim=0, return_counts=True)

        return indices, counts


class Accuracy(ClfMetricBase):
    def compute(self) -> float:
        return (self.predictions == self.targets).mean()


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

