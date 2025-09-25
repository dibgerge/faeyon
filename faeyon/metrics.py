import torch

from typing import Protocol, Optional, Literal
from faeyon.enums import ClfType


class Metric(Protocol):
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None: ...

    def compute(self) -> float: ...

    def reset(self) -> None: ...


class ClassificationMetricBase(Metric):
    """
    num_classes: Optional[int]
        Will be inferred from the first targets/preds update if not provided. 
        This might not always be accurate, for example if targets/preds are not one-hot encoded.
    """
    thresholds: Optional[torch.Tensor]

    def __init__(
        self, 
        num_classes: Optional[int] = None,
        thresholds: Optional[int | float | list[float]] = None,
        topk: Optional[int] = None,
        subset: Optional[list[int]] = None,
        average: Optional[str] = "weighted",
        multilabel: bool = False,
    ):
        if thresholds is not None and topk is not None:
            raise ValueError("Cannot use both `thresholds` and `topk`.")
        
        if topk is not None and multilabel:
            raise ValueError("`topk` is not supported for multilabel predictions.")

        if isinstance(thresholds, int):
            self.thresholds = torch.linspace(0, 1, thresholds)
        elif isinstance(thresholds, float):
            self.thresholds = torch.tensor([thresholds])
        elif isinstance(thresholds,  torch.Tensor):
            if thresholds.max() > 1 or thresholds.min() < 0:
                raise ValueError("`thresholds` tensor values must all be in [0, 1].")
        elif isinstance(thresholds, list | tuple):
            if max(thresholds) > 1 or min(thresholds) < 0:
                raise ValueError("`thresholds` list/tuple values must all be in [0, 1].")
            self.thresholds = torch.tensor(thresholds)
        elif thresholds is None:
            self.thresholds = None
        else:
            raise ValueError("`thresholds` must be an integer, float, or list/tensor of floats.")

        self.subset = subset
        self.average = average
        self.multilabel = multilabel
        self.topk = topk
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        pass

    def _validate_preds(self, preds: torch.Tensor) -> tuple[ClfType, int]:
        """
        `preds` tensor must have a shape of (B,) or (B, C)

        If shape (B,) is given:
        * If values in [0, 1], then it is a binary prediction.
            * `topk` is not supported for binary predictions.
            * `thresholds` is required for binary predictions.
            * `average` is not supported for binary predictions.
            * `multilabel` is not supported for binary predictions.

        * If values are integers, then it is a sparse prediction.
            * `topk` is not supported for sparse predictions.
            * `thresholds` is not supported for sparse predictions.
            * `multilabel` is not supported for sparse predictions.
        
        If shape (B, C) is given:
        * If `multilabel`, values must be probabilities encoded in [0, 1].
        """
        match(preds.ndim):
            case 1:
                if self.multilabel:
                    raise ValueError("Multilabel is not supported for preds of shape (B,).")
                
                if self.topk is not None:
                    raise ValueError("Topk is not supported for binary predictions.")

                if preds.is_floating_point():
                    if preds.max() > 1 or preds.min() < 0:
                        raise ValueError(
                            "Predictions as float tensors of shape (B,) must be probabilities."
                        )
                    if self.thresholds is None:
                        raise ValueError("Binary predictions require `threshold`.")
                    
                    if self.average is not None:
                        raise ValueError("`average` is not supported for binary predictions.")

                    pred_type = ClfType.BINARY
                    pred_classes = 1
                else:
                    if self.thresholds is not None:
                        raise ValueError("`threshold is not supported for sparse predictions.")
                    pred_type = ClfType.SPARSE
                    pred_classes = int(preds.max().item())
            case 2:
                if not preds.is_floating_point():
                    raise ValueError(
                        "Predictions of shape (B, C) must be floating point representing "
                        "logits or probabilities."
                    )
                
                if self.multilabel:
                    if preds.max() > 1 or preds.min() < 0:
                        raise ValueError(
                            "Multilabel predictions of shape (B, C) must be probabilities."
                        )
                    pred_type = ClfType.MULTILABEL
                else:
                    pred_type = ClfType.CATEGORICAL
                pred_classes = preds.size(-1)
            case _:
                raise ValueError(
                    f"Predictions tensor must be of shape (B,) or (B, C). Got {preds.shape}")

        return pred_type, pred_classes

    def _validate_targets(self, targets: torch.Tensor) -> tuple[ClfType, int]:
        """
        `targets` tensor must be of shape (B,) or (B, C) with integer values.

        If given shape (B,):
        * Invalid for `multilabel`.

        If given shape (B, C):
        * Values are one-hot encoded, and they must be either 0 or 1.
        """
        if targets.is_floating_point():
            raise ValueError(
                "Targets tensor must be an integer tensor for shape (B,) or (B, C).")

        match(targets.ndim):
            case 1:
                if self.multilabel:
                    raise ValueError(
                        "Multilabel is not supported for targets of shape (B,). Targets must "
                        "be one-hot encoded for multilabel classification."
                    )
                target_type = ClfType.SPARSE
                target_classes = int(targets.max().item())
            case 2:
                if not torch.isin(targets, torch.tensor([0, 1])).all():
                    raise ValueError("Only one-hot encoded targets are supported for shape (B, C).")
                target_classes = targets.size(-1)

                if target_classes == 1:
                    if self.multilabel:
                        raise ValueError(
                            "Multilabels are not supported for targets of shape (B, 1)."
                        )
                    target_type = ClfType.BINARY
                else:
                    if self.multilabel:
                        target_type = ClfType.MULTILABEL
                    else:
                        target_type = ClfType.CATEGORICAL
            case _:
                raise ValueError(
                    f"Targets tensor must be of shape (B,) or (B, C). Got {targets.shape}"
                )

        return target_type, target_classes

    def _validate_data(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """

        """
        pred_type, pred_classes = self._validate_preds(preds)
        target_type, target_classes = self._validate_targets(targets)

        if pred_type == ClfType.BINARY and target_type == ClfType.SPARSE:
            if targets.max() > 1 or targets.min() < 0:
                raise ValueError(
                    "Predictions are binary probabilities, but targets contain values other "
                    "than 0 or 1."
                )

        if ptype == "sparse" and ttype == "sparse":
            # Inferring from sparse inputs is not reliable since some classes 
            # might be missing in a batch.
            if self.num_classes is None:
                raise ValueError(
                    "Cannot infer number of classes from categorical inputs. `num_classes` needs "
                    "to be provided, or use sparse predictions and/or targets.")
            else:
                num_classes = self.num_classes
        else:
            if pred_classes is None or target_classes is None:
                num_classes = target_classes or pred_classes
            else:
                cond = pred_classes == target_classes
                if cond:
                    num_classes = pred_classes
                else:
                    num_classes = None
                
            if self.num_classes is not None and num_classes != self.num_classes:
                raise ValueError(
                    f"Number of classes mismatch:"
                    f"\tTarget: {target_classes}"
                    f"\tPredictions: {pred_classes}"
                    f"\tNum classes: {self.num_classes}"
                )
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        predictions: (B, )
        targets: (B, ) or (B, C)

        where 
            - C: Number of classes
            - B: Batch size

        """
        self._validate_data(preds, targets)

        if self.topk is not None:
            pred = preds.topk(self.topk, dim=-1).indices
            pred_top = pred[:, 0]
            pred_tp = (targets.unsqueeze(-1) == pred).any(dim=-1)
            preds = torch.where(pred_tp, targets, pred_top)
        
        if self.thresholds is not None:
            preds = preds >= self.thresholds

        if len(targets.shape) == 2:           
            # Not multilabel case
            targets = torch.argmax(targets, dim=-1)
        
        if len(preds.shape) == 2:            
            if self.topk > 1:
                preds = preds.topk(self.topk, dim=-1).indices
                # Prediction with highest probability
                pred_top = preds[:, 0]

                pred_tp = (targets.unsqueeze(-1) == preds).any(dim=-1)
                preds = torch.where(pred_tp, targets, pred_top)
            else:
                preds = preds.argmax(dim=-1, keepdim=True)

        else:
            if preds.is_floating_point():
                # Binary case
                preds = preds >= self.threshold

        predictions >= self.threshold
        self.targets = targets



class MultiClassAccuracy(Metric):
    def __init__(self, topk: int = 5):
        self.topk = topk

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.predictions = predictions
        self.targets = targets

    def compute(self) -> float:
        return (self.predictions == self.targets).mean()


class MultiLabelAccuracy(Metric):
    def __init__(self, topk: int = 5):
        self.topk = topk

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        self.predictions = predictions
        self.targets = targets

    def compute(self) -> float:
        return (self.predictions == self.targets).mean()
