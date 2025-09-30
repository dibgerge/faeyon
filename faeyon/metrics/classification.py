import torch

from typing import Optional
from faeyon.enums import ClfTask
from torch.nn.functional import one_hot
from .base import Metric


class BinarySparseTask:

    def condition(self, preds: torch.Tensor, targets: torch.Tensor) -> bool: ...
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None: ...


class ClfMetricBase(Metric):
    """
    num_classes: Optional[int]
        Will be inferred from the first targets/preds update if not provided. 
        This might not always be accurate, for example if targets/preds are not one-hot encoded.
    """
    thresholds: Optional[torch.Tensor]
    num_classes: Optional[int]
    _state: Optional[torch.Tensor]

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

        self.task_map = {
            ClfTask.BINARY: self._update_binary,
            ClfTask.SPARSE: self._update_sparse,
            ClfTask.CATEGORICAL: self._update_categorical,
            ClfTask.MULTILABEL: self._update_categorical,
        }
        self.subset = subset
        self.average = average
        self.multilabel = multilabel
        self.topk = topk
        self.reset()

    def reset(self) -> None:
        self._state = None
    
    def _infer_task(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[ClfTask, int]:
        """
        Infer the task from the predictions and targets.
        """
        if preds.ndim not in [1, 2] or targets.ndim not in [1, 2]:
            raise ValueError(
                f"Predictions/targets tensor must be of shape (B,) or (B, C). Got:\n"
                f"\tPredictions: {preds.shape} \n"
                f"\tTargets: {targets.shape}"
            )
        
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)

        max_pred = preds.max().item()
        min_pred = preds.min().item()
        max_target = targets.max().item()
        min_target = targets.min().item()

        # Validate targets
        if targets.is_floating_point() or min_target < 0:
            raise ValueError(
                "Targets tensor must be non-negative integer tensor for shape (B,) or (B, C).")

        is_pred_proba = preds.is_floating_point() and max_pred <= 1 and min_pred >= 0
        is_pred_binary = not preds.is_floating_point() and max_pred <= 1 and min_pred >= 0

        if targets.shape[-1] > 1:
            tsum = targets.sum(-1)
            is_target_multilabel = max_target <= 1 and min_target >= 0
            is_target_binary = (tsum == 1).all().item()
        else:
            is_target_multilabel = False
            is_target_binary = max_target <= 1 and min_target >= 0

        match self.multilabel, preds.shape, targets.shape:
            # Binary task cases
            case False, (pbatch, 1), (tbatch, tc) if preds.is_floating_point() and tc in [1, 2]:
                if not is_target_binary:
                    raise ValueError("Targets must be 0 or 1 for binary predictions.")

                if self.thresholds is None:
                    raise ValueError("Binary predictions require `threshold`.")

                if not is_pred_proba:
                    raise ValueError(
                        "Floating point predictions of shape (B,) or (B, 1) must be probabilities."
                    )

                task = ClfTask.BINARY
                nclasses = 1
            
            # Binary task cases, but with 2-D predictions
            case False, (pbatch, 2), (tbatch, tc) if (
                preds.is_floating_point() and pbatch == tbatch and tc in [1, 2]
            ):
                if not is_target_binary:
                    raise ValueError("Targets must be 0 or 1 for binary predictions.")

                if self.topk is not None and self.topk > 1:
                    raise ValueError("Topk can only be 1 for binary predictions of shape (B, 2).")

                if self.thresholds is not None:
                    task = ClfTask.CATEGORICAL
                    nclasses = 2
                else:
                    task = ClfTask.BINARY
                    nclasses = 1

            # Sparse predictions and targets for binary or sparse tasks
            case False, (pbatch, 1), (tbatch, 1) if (
                not preds.is_floating_point() and pbatch == tbatch
            ):
                if self.topk is not None or self.thresholds is not None:
                    raise ValueError("topk/thresholds are not supported for sparse predictions.")
                
                if self.num_classes is None:
                    raise ValueError("`num_classes` is not set/cannot be inferred.")
                elif self.num_classes == 1:
                    if not is_target_binary or not is_pred_binary:
                        raise ValueError(
                            "Integer Targets/Predictions must be 0 or 1 for binary predictions."
                        )
                    
                    task = ClfTask.BINARY
                    nclasses = 1
                else:
                    if (
                        max_pred >= self.num_classes 
                        or min_pred < 0 
                        or max_target >= self.num_classes
                    ):
                        raise ValueError(
                            "Sparse predictions/target values must be less than `num_classes`.")
                    nclasses = self.num_classes
                    task = ClfTask.SPARSE
            
            # Sparse
            case False, (pbatch, 1), (tbatch, tc) if (
                not preds.is_floating_point() and pbatch == tbatch and tc > 2
            ):
                if self.topk is not None or self.thresholds is not None:
                    raise ValueError("Topk/thresholds are not supported for sparse predictions.")
                
                if max_pred >= tc or min_pred < 0:
                    raise ValueError(
                        "Predictions values must be less than targets number of classes and "
                        "greater than 0."
                    )
                nclasses = tc
                task = ClfTask.SPARSE

            # Binary task cases for 2-D targets and sparse predictions
            case False, (pbatch, 1), (tbatch, 2) if (
                not preds.is_floating_point() 
                and (pbatch == tbatch)
            ):
                if not is_target_binary:
                    raise ValueError("Targets must be onehot encoded for binary predictions.")
                
                if self.topk is not None or self.thresholds is not None:
                    raise ValueError("Topk/thresholds are not supported for sparse predictions.")
                
                if not is_pred_binary:
                    raise ValueError("Integer Predictions must be 0 or 1 for binary predictions.")

                task = ClfTask.BINARY
                nclasses = 1
            
            # Categorical / sparse task - categorical targets
            case False, (pbatch, pc), (tbatch, tc) if (
                preds.is_floating_point() 
                and tc == pc and pbatch == tbatch and pc > 2
            ):
                if not is_target_binary:
                    raise ValueError("Targets must be onehot encoded for categorical predictions.")

                if self.thresholds is not None:
                    task = ClfTask.CATEGORICAL
                else:
                    task = ClfTask.SPARSE
                
                nclasses = pc

            # Categorical / sparse task - sparse targets
            case False, (pbatch, pc), (tbatch, 1) if (
                preds.is_floating_point() and pbatch == tbatch and pc > 2
            ):
                if max_target > pc:
                    raise ValueError(
                        "Target values must be less than predictions number of classes.")
                
                if self.thresholds is not None:
                    task = ClfTask.CATEGORICAL
                else:
                    task = ClfTask.SPARSE

                nclasses = pc

            # Multilabel task
            case True, (pbatch, pc), (tbatch, tc) if (
                is_pred_proba and pc == tc and pbatch == tbatch 
                and pc > 1 and is_target_multilabel
            ):
                task = ClfTask.MULTILABEL
                nclasses = pc
            case _:
                raise ValueError(f"""
                    Unsupported combination of predictions and targets.
                    Predictions shape: {preds.shape}\tFloating point: {preds.is_floating_point()}
                    Targets shape: {targets.shape} 
                    Multilabel: {self.multilabel}
                    Number of classes: {self.num_classes}
                """)
        return task, nclasses

    def _update_binary(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        size: tuple[int, ...]
        
        n = preds.shape[0]
        if self.thresholds is not None:
            if preds.ndim == 1:
                preds = preds.unsqueeze(-1)
            
            if targets.ndim == 1:
                targets = targets.unsqueeze(-1)
            
            if targets.shape[-1] == 2:
                targets = targets.argmax(dim=-1, keepdim=True)

            preds = (preds >= self.thresholds).to(torch.long)
            nthresh = len(self.thresholds)
            targets = targets.expand(n, nthresh)
            thresholds = torch.arange(nthresh).expand(n, -1)
            items = torch.stack([thresholds, preds, targets]).view(3, -1)
            size = (nthresh, 2, 2)
        else:
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

    def _update_sparse(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.num_classes is None:
            raise ValueError("`num_classes` is not set/cannot be inferred.")

        if targets.ndim == 2 and targets.shape[-1] > 1:
            targets = targets.argmax(dim=-1)
        else:
            targets = targets.squeeze()

        if self.topk is not None and self.topk > 1:
            preds = preds.topk(self.topk, dim=-1).indices
            pred_top = preds[:, 0]
            pred_tp = (targets.unsqueeze(-1) == preds).any(dim=-1)
            preds = torch.where(pred_tp, targets, pred_top)
        elif preds.ndim == 2 and preds.shape[-1] > 1:
            preds = preds.argmax(dim=-1)
        else:
            preds = preds.squeeze()

        items = torch.stack([preds, targets])
        size = (self.num_classes, self.num_classes)
        return items, size
     
    def _update_categorical(
        self, 
        preds: torch.Tensor, 
        targets: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        n = preds.shape[0]
        
        if self.thresholds is None:
            raise ValueError("Categorical/multilabel predictions require `thresholds`.")

        if self.num_classes is None:
            raise ValueError("`num_classes` is not set/cannot be inferred.")

        targets = targets.squeeze()
        if targets.ndim == 1:
            targets = one_hot(targets)
        
        nthresh = len(self.thresholds)
        preds = (preds.unsqueeze(-1) >= self.thresholds).to(torch.long)
        targets = targets.unsqueeze(-1).expand(n, -1, nthresh)
        classes = torch.arange(self.num_classes).view(1, -1, 1).expand(n, -1, nthresh)
        thresholds = torch.arange(nthresh).view(1, 1, -1).expand(n, self.num_classes, nthresh)
        items = torch.stack([classes, thresholds, preds, targets]).view(4, -1)
        size = (self.num_classes, nthresh, 2, 2)
        return items, size

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        predictions: (B, )
        targets: (B, ) or (B, C)

        where 
            - C: Number of classes
            - B: Batch size
        """
        task, nclasses = self._infer_task(preds, targets)

        if self.num_classes is None:
            self.num_classes = nclasses
        else:
            if nclasses != self.num_classes:
                raise ValueError(
                    f"Mismatch between inferred and provided number of classes. "
                    f"Inferred: {nclasses}\tProvided: {self.num_classes}"
                )

        indices, size = self.task_map[task](preds, targets)

        # sparse coo coalesce add the values of repeated indices
        values = torch.ones(indices.shape[-1], dtype=torch.long, device=indices.device)
        data = torch.sparse_coo_tensor(indices, values, dtype=torch.long, size=size).coalesce()

        if self._state is None:
            self._state = data
        else:
            self._state += data

    def _confusion_sparse(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise ValueError("State is empty. Call update() before compute().")

        if self._state.ndim != 2:
            raise ValueError("State must be a 2D sparse tensor.")

        nclasses = self._state.shape[0]
        pred_idx, target_idx = self._state.indices()
        values = self._state.values()

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

    def _confusion_categorical(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise ValueError("State is empty. Call update() before compute().")
        
        if self._state.ndim not in [3, 4]:
            raise ValueError("State must be a 3D or 4D sparse tensor.")

        state = self._state.to_dense()
        return {
            "tp": state[..., 1, 1],
            "fp": state[..., 1, 0],
            "tn": state[..., 0, 0],
            "fn": state[..., 0, 1],
            "cardinality": state.sum(dim=(-2, -1))
        }


class Accuracy(ClfMetricBase):
    def _compute_sparse(self) -> torch.Tensor:
        confusion = self._confusion_sparse()
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
    
    def _compute_categorical(self) -> torch.Tensor:
        confusion = self._confusion_categorical()
        tp = confusion["tp"]
        tn = confusion["tn"]
        cardinality = confusion["cardinality"]
        return (tp + tn) / cardinality

    def compute(self) -> torch.Tensor:
        if self._state is None:
            raise ValueError("State is empty. Call update() before compute().")

        if self._state.ndim == 2:
            return  self._compute_sparse()
        else:
            return self._compute_categorical()      


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

