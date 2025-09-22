import abc
import itertools
import torch
import time
from collections import defaultdict

from typing import Optional, Any, Union
from torch import nn, optim
from torch.utils.data import DataLoader

from .distributed import cleanup_distributed
from .callbacks import Callback
from .loggers import MetricsComputer, MetricsLogger
from .core import Period, FaeOptimizer, TrainState
from faeyon.metrics import Metric


class Recipe:
    """ The base class for training recipes """
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Union[optim.Optimizer, FaeOptimizer],
        metrics: Optional[list[dict[str, str | Period] | Metric]] = None,
        callbacks: Optional[list[Callback]] = None,
        train_metrics_flush: str | Period = "1e",
        val_metrics_flush: str | Period = "1e",
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.metrics = metrics or []

        if isinstance(optimizer, FaeOptimizer):
            self.optimizer = optimizer(self.model)
        else:
            self.optimizer = optimizer

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
    def _validate_epoch(self, val_data: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """Validate the model on validation data"""
        self.model.eval()
        val_losses = []
        val_metrics = defaultdict(list)
        
        for batch_idx, batch in enumerate(val_data):
            metrics = self.val_step(batch, batch_idx)
            val_losses.append(metrics.get('val_loss', 0.0))
            
            # Collect other metrics
            for key, value in metrics.items():
                if key != 'val_loss':
                    val_metrics[key].append(value)
        
        # Average metrics
        avg_metrics = {'val_loss': sum(val_losses) / len(val_losses)}
        for key, values in val_metrics.items():
            avg_metrics[key] = sum(values) / len(values)
        
        self.model.train()
        return avg_metrics
    
    def _train_iter(self, data: DataLoader, min_period: Period, max_period: Period) -> TrainState:
        state = TrainState()
        data_iter = iter(data)
        train_start = time.time()
        stop_requested = self.callbacks.on_train_begin(state)
        
        while state < max_period and not (stop_requested and state >= min_period):
            state.tic()

            if state.epoch_step == 1:
                stop_requested = self.callbacks.on_epoch_begin(state) or stop_requested

            stop_requested = self.callbacks.on_train_step_begin(state) or stop_requested

            try:     
                batch = next(data_iter)
            except StopIteration:
                stop_requested = self.callbacks.on_epoch_end(state) or stop_requested
                data_iter = iter(data)
                state.toc()
            else:
                metrics = yield batch
                state.metrics = metrics
                stop_requested = self.callbacks.on_train_step_end(state) or stop_requested
            
        self.callbacks.on_train_end(state)
        return state

            
    def train(
        self,
        train_data: DataLoader,
        max_period: Optional[str | Period] = None,
        min_period: Optional[str | Period] = None,
        val_data: Optional[DataLoader] = None,
        val_period: str | Period = "1e",
        flush_period: str | Period = "1step",
        gradient_accumulation_steps: int = 1,
        verbose: bool = True
    ) -> dict[str, Any]:
        """
        Train the model with flexible min/max periods.
        
        Args:
            train_data: Training data loader
            min_period: Minimum training period (e.g., "30s", "5e", "1000s")
            max_period: Maximum training period (e.g., "2m", "10e", "5000s")
            val_data: Validation data loader
            val_freq: Validation frequency (every N steps)
            gradient_accumulation_steps: Number of steps to accumulate gradients before updating
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        self.model.train()

        for state in self.train_iter(train_data, min_period, max_period):
            self.optimizer.zero_grad()
            loss, preds, targets = self.train_step(state.batch)
            loss.backward()
            self.optimizer.step()

            for metric in self.metrics:
                metric.update(preds, targets)
                # Check if time to flush train metrics
                if state.period >= flush_period:
                    metric.rest()
             
            if val_data is not None and state.period >= val_period:
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
            Loss tensor
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
