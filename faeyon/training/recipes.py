import abc
import torch

from typing import Generator, Optional, Any, Union
from torch import nn, optim
from torch.utils.data import DataLoader

from .distributed import cleanup_distributed
from .callbacks import Callback

from .core import Period, FaeOptimizer, TrainState
from faeyon.metrics.base import Metric
from faeyon import A
from faeyon.enums import TrainStateMode


class Recipe:
    """ The base class for training recipes """

    optimizer: optim.Optimizer

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: optim.Optimizer | FaeOptimizer,
        metrics: Optional[list[dict[str, str | Period] | Metric]] = None,
        callbacks: Optional[list[Callback]] = None,
        train_flush: str | Period = "1e",
        val_period: str | Period = "1e"
    ):
        self.model = model
        self.loss = loss
        self.callbacks = callbacks or []
        self.train_metrics = metrics or []
        self.val_metrics = metrics or []
        # self.state = TrainState(train_metrics=self.train_metrics, val_metrics=self.val_metrics)

        if isinstance(train_flush, str):
            self.train_flush = Period.from_expr(train_flush)
        else:
            self.train_flush = train_flush

        if isinstance(optimizer, FaeOptimizer):
            self.optimizer = optimizer(self.model)
        else:
            self.optimizer = optimizer

        self.state = TrainState()
        self.val_period = val_period
        self._last_flush = 0
        self._last_val = 0

    @abc.abstractmethod
    def train_step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare a batch for training or validation. Must return loss, predictions, and targets.
        """
        pass
    
    @abc.abstractmethod
    def val_step(self, batch: Any) -> dict[str, float]:
        """Perform one validation step"""
        pass
           
    @torch.no_grad()
    def _val_epoch(self, data: DataLoader) -> None:
        """Validate the model on validation data"""
        self.model.eval()
        self.val_metrics.reset()
        self.callbacks.on_val_begin(self.state)
        for batch in data:
            self.callbacks.on_val_step_begin(self.state)
            loss, preds, targets = self.val_step(batch)
            self.state.val_metrics.update(preds, targets)
            self.callbacks.on_val_step_end(self.state)
        
        self.callbacks.on_val_end(self.state)
        self.model.train()
    
    def _train_iter(
        self, 
        data: DataLoader, 
        min_period: Period, 
        max_period: Period
    ) -> Generator[dict[str, Any], None, None]:
        data_iter = iter(data)
        stop_requested = self.callbacks.on_train_begin(self.state)
        
        self.state.toc()
        while self.state < max_period and not (stop_requested and self.state >= min_period):
            stop = []
            try:     
                batch = next(data_iter)
            except StopIteration:
                stop.append(self.callbacks.on_epoch_end(self.state))
                data_iter = iter(data)
                self.state.toc()
            else:
                self.state.tic()
                stop.append(self.callbacks.trigger(self.state))

                if self.state.is_epoch_begin():
                    stop.append(self.callbacks.on_epoch_begin(self.state))

                stop.append(self.callbacks.on_train_step_begin(self.state))
                yield batch
                stop.append(self.callbacks.on_train_step_end(self.state))
    
            stop_requested = stop_requested or any(stop)
            
        self.callbacks.on_train_end(self.state)

    def train(
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

        min_period = Period.from_expr(min_period)
        max_period = Period.from_expr(max_period)

        for batch in self._train_iter(train_data, min_period, max_period):
            self.optimizer.zero_grad()

            inputs = batch["inputs"]
            targets = batch["targets"]

            if isinstance(inputs, dict):
                inputs = A(**inputs)

            preds = inputs >> self.model
            loss = self.loss(preds, targets)
            loss.backward()
            self.optimizer.step()
            self.state.train_metrics.update(preds, targets)

            flush_count = self.state // self.train_flush
            if flush_count > self._last_flush:
                self._last_flush = flush_count
                self.state.train_metrics.reset()
             
            if val_data is not None:
                val_count = self.state // self.val_period
                if val_count > self._last_val:
                    self._last_val = val_count
                self._validate_epoch(val_data)
                
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
