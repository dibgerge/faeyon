import torch

from typing import Optional, Any
from torch import nn, optim
from torch.utils.data import DataLoader

from .distributed import cleanup_distributed
from .callbacks import Callback, CallbackCollection

from .core import Period, FaeOptimizer, TrainState
from faeyon.metrics.base import Metric
from faeyon import A
from faeyon.enums import PeriodUnit


class Recipe:
    """ 
    The base class for training recipes. 

    metrics : Optional[list[Metric]]
        A list of metrics which will be used for training and validation.
    """

    optimizer: optim.Optimizer

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer | FaeOptimizer,
        metrics: Optional[list[Metric]] = None,
        callbacks: Optional[list[Callback]] = None,
        train_flush: str | Period = "1e",
        val_period: str | Period = "1e"
    ):
        self.model = model
        self.loss = loss

        if callbacks is None:
            callbacks = []

        if not isinstance(callbacks, CallbackCollection):
            self.callbacks = CallbackCollection(callbacks)
        else:
            self.callbacks = callbacks

        self.state = TrainState(metrics, train_flush=train_flush)

        if isinstance(optimizer, FaeOptimizer):
            self.optimizer = optimizer(self.model)
        else:
            self.optimizer = optimizer

        if isinstance(val_period, str):
            self.val_period = Period.from_expr(val_period)
        else:
            self.val_period = val_period

    def train_step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch for training or validation. Must return loss, predictions, and targets.
        """
        self.optimizer.zero_grad()

        inputs = batch["inputs"]
        targets = batch["targets"]

        if isinstance(inputs, dict):
            inputs = A(**inputs)

        preds = inputs >> self.model
        loss = self.loss(preds, targets)
        loss.backward()
        self.optimizer.step()
        return loss, preds, targets

    def val_step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one validation step"""
        inputs = batch["inputs"]
        targets = batch["targets"]

        if isinstance(inputs, dict):
            inputs = A(**inputs)

        preds = inputs >> self.model
        loss = self.loss(preds, targets)
        return loss, preds, targets
           
    @torch.no_grad()
    def _validate_epoch(self, data: DataLoader) -> None:
        """Validate the model on validation data"""
        mode = self.model.training

        self.model.eval()
        self.state.on_val_begin()
        self.callbacks.on_val_begin(self.state)

        for batch in data:
            self.state.on_val_step_begin()
            self.callbacks.on_val_step_begin(self.state)

            loss, preds, targets = self.val_step(batch)

            self.state.on_val_step_end(loss, preds, targets)
            self.callbacks.on_val_step_end(self.state)
        
        self.state.on_val_end()
        self.callbacks.on_val_end(self.state)
        self.model.train(mode)

    def _data_iter(self, data: DataLoader):
        """
        Iterate over the data with lookahead to determine if the current batch is the last one.

        Returns:
            A tuple of (is_first, is_last, batch).
        """
        while True:
            data_iter = iter(data)
            try:
                batch = next(data_iter)
            except StopIteration:
                return True, True, None
            
            # For loop never runs if there's only one batch, can't use enumerate...
            count = 0
            for next_batch in data_iter:
                yield count == 0, False, batch
                batch = next_batch
                count += 1

            yield count == 0, True, batch

    def fit(
        self,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        min_period: Optional[str | Period] = None,
        max_period: Optional[str | Period] = None,
    ) -> None:
        """
        Train the model with flexible min/max periods.
        
        Args:
            train_data: Training data loader
            min_period: Minimum training period (e.g., "30s", "5e", "1000s")
            max_period: Maximum training period (e.g., "2m", "10e", "5000s")
            val_data: Validation data loader
            val_freq: Validation frequency (every N steps)
            
        Returns:
            Training history dictionary
        """
        self.model.train()

        if min_period is None:
            min_period = Period(0, PeriodUnit.SECONDS)
        elif isinstance(min_period, str):
            min_period = Period.from_expr(min_period)

        if max_period is None:
            max_period = Period(float("inf"), PeriodUnit.SECONDS)
        elif isinstance(max_period, str):
            max_period = Period.from_expr(max_period)

        next_val = self.val_period
        self.state.on_train_begin(len(train_data))
        self.callbacks.on_train_begin(self.state)
        stop_requested = False

        for is_first, is_last, batch in self._data_iter(train_data):
            stop: list[Optional[bool]] = []

            if is_first:
                self.state.on_epoch_begin()
                self.callbacks.on_epoch_begin(self.state)

            self.state.on_train_step_begin()
            self.callbacks.on_train_step_begin(self.state)
            
            loss, preds, targets = self.train_step(batch)
            
            self.state.on_train_step_end(loss, preds, targets, is_last)
            stop.append(self.callbacks.on_train_step_end(self.state))
            stop.append(self.callbacks.trigger(self.state))

            if val_data is not None:
                val_count = self.state // next_val
                if val_count > 0:
                    next_val += val_count * self.val_period
                    stop.append(self._validate_epoch(val_data))
            
            if is_last:                
                self.state.on_epoch_end()
                stop.append(self.callbacks.on_epoch_end(self.state))

            stop_requested = stop_requested or any(stop)

            if self.state > max_period or (stop_requested and self.state > min_period):
                break

        self.state.on_train_end()
        self.callbacks.on_train_end(self.state)
                
    def cleanup(self):
        """Clean up distributed training"""
        if self.use_distributed:
            cleanup_distributed()


class ClassifyRecipe(Recipe):
    """Recipe for classification tasks"""
    
    def train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform one training step for classification.
        
        Args:
            batch: Tuple of (inputs, targets) or dict with 'inputs' and 'targets' keys
            batch_idx: The index of the current batch
            
        Returns:
            Loss tensor                state.metrics = metrics
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        elif isinstance(batch, dict):
            inputs = batch['inputs']
            targets = batch['targets']
        else:
            raise ValueError("Batch must be a tuple of (inputs, targets) or dict with 'inputs' and 'targets' keys")
        
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        
        return loss
    
    def val_step(self, batch: Any, batch_idx: int) -> dict[str, float]:
        """
        Perform one validation step for classification.
        
        Args:
            batch: Tuple of (inputs, targets) or dict with 'inputs' and 'targets' keys
            batch_idx: The index of the current batch
            
        Returns:
            Dictionary of metrics for this batch
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        elif isinstance(batch, dict):
            inputs = batch['inputs']
            targets = batch['targets']
        else:
            raise ValueError("Batch must be a tuple of (inputs, targets) or dict with 'inputs' and 'targets' keys")
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
        
        # Compute metrics
        metrics = {'val_loss': loss.item()}
        
        # Add accuracy
        accuracy = MetricsComputer.accuracy(outputs, targets)
        metrics['val_accuracy'] = accuracy
        
        # Add precision, recall, F1 if requested by callbacks
        for callback in self.callbacks:
            if isinstance(callback, MetricsLogger) and callback.metrics:
                if any(m in callback.metrics for m in ['precision', 'recall', 'f1']):
                    prf_metrics = MetricsComputer.precision_recall_f1(outputs, targets)
                    metrics.update({f'val_{k}': v for k, v in prf_metrics.items()})
                    break
        
        return metrics
