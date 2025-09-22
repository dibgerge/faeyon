"""
Callback system for Faeyon training.

This module provides a flexible callback system for training hooks,
including early stopping, model checkpointing, and learning rate scheduling.
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Callable
import torch
from .core import Period, TrainState, PeriodUnit


class Callback(ABC):
    """Base class for all callbacks"""
    
    def __init__(self, trigger: Optional[str | Period] = None):
        """
        Trigger is the interval at which the callback is triggered, for example "2e", "10steps", "100seconds", etc.
        """
        self._trigger = trigger
        self.trigger_count = 0

    def trigger(self, state: TrainState) -> bool:
        # For time triggers, sometimes things are not exactly on the trigger
        count = state // self._trigger
        if count > self.trigger_count:
            self.trigger_count = count
            return self.on_trigger(state)
    
    def on_train_begin(self, logs: dict[str, Any]) -> Optional[bool]:
        """Called at the beginning of training"""
        pass
    
    def on_train_end(self, logs: dict[str, Any]) -> Optional[bool]:
        """Called at the end of training"""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: dict[str, Any]) -> Optional[bool]:
        """Called at the beginning of each epoch"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> Optional[bool]:
        """Called at the end of each epoch"""
        pass
    
    def on_train_step_begin(self, batch: int, logs: dict[str, Any]) -> Optional[bool]:
        """Called at the beginning of each batch"""
        pass
    
    def on_train_step_end(self, batch: int, logs: dict[str, Any]) -> Optional[bool]:
        """Called at the end of each batch"""
        pass

    def on_trigger(self, state: TrainState, trigger: str) -> Optional[bool]:
        pass


class CallbackCollection(Callback):
    """Collection of callbacks"""
    
    def __init__(self, callbacks: list[Callback]):
        super().__init__()
        self.callbacks = callbacks
    
    def __getattr__(self, name: str) -> Callable:
        """Called at the beginning of training"""
        def proxy(state: TrainState) -> Optional[bool]:
            should_stop = False
            for callback in self.callbacks:
                func = getattr(callback, name)
                result = func(state)
                if result is not None:
                    should_stop |= result
            return should_stop
        
        return proxy


class EarlyStopping(Callback):
    """Early stopping callback to stop training when a metric stops improving"""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 10,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_weights = None
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - min_delta
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b + min_delta
            self.best_score = float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """Check if early stopping should be triggered"""
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.monitor_op(current, self.best_score):
            self.best_score = current
            self.wait = 0
            if self.restore_best_weights and hasattr(self, 'model') and self.model is not None:
                self.best_weights = self.model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.load_state_dict(self.best_weights)
                self.request_stop()


class ModelCheckpoint(Callback):
    """Model checkpointing callback to save model weights during training"""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = True,
        mode: str = 'min',
        save_freq: int = 1,
        verbose: bool = True
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.save_freq = save_freq
        self.verbose = verbose
        
        self.best_score = None
        self.model = None
        self.optimizer = None
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b
            self.best_score = float('-inf')
    
    def on_train_begin(self, logs: dict[str, Any]) -> None:
        """Initialize model and optimizer references"""
        self.model = logs.get('model')
        self.optimizer = logs.get('optimizer')
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """Save model checkpoint if conditions are met"""
        if epoch % self.save_freq != 0:
            return
        
        current = logs.get(self.monitor)
        if current is None:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        if self.save_best_only:
            if self.monitor_op(current, self.best_score):
                self.best_score = current
                self._save_checkpoint(epoch, logs)
        else:
            self._save_checkpoint(epoch, logs)
    
    def _save_checkpoint(self, epoch: int, logs: dict[str, Any]) -> None:
        """Save the actual checkpoint"""
        if self.model is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'logs': logs
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if self.save_weights_only:
            torch.save(self.model.state_dict(), self.filepath)
        else:
            torch.save(checkpoint, self.filepath)
        
        if self.verbose:
            print(f"Model checkpoint saved to {self.filepath}")


class LearningRateScheduler(Callback):
    """Learning rate scheduling callback"""
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: str = 'val_loss',
        mode: str = 'min',
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-7,
        verbose: bool = True
    ):
        super().__init__()
        self.scheduler = scheduler
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best_score = None
        self.wait = 0
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b
            self.best_score = float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """Update learning rate based on monitoring metric"""
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.monitor_op(current, self.best_score):
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Reduce learning rate
                for param_group in self.scheduler.optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    param_group['lr'] = new_lr
                    
                    if self.verbose:
                        print(f"Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                
                self.wait = 0
        
        # Step the scheduler
        self.scheduler.step()


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8,
        verbose: bool = True
    ):
        super().__init__()
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.wait = 0
        
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - threshold
            self.best = float('inf')
        else:
            self.monitor_op = lambda a, b: a > b + threshold
            self.best = float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """Check if learning rate should be reduced"""
        current = logs.get('val_loss')
        if current is None:
            return
        
        if self.monitor_op(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr(epoch)
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown
    
    def _reduce_lr(self, epoch: int) -> None:
        """Reduce learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if self.verbose:
                print(f"Epoch {epoch}: reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")


class History(Callback):
    """Callback to track training history"""
    
    def __init__(self):
        super().__init__()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """Record metrics in history"""
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def on_batch_end(self, batch: int, logs: dict[str, Any]) -> None:
        """Record batch-level metrics"""
        if 'loss' in logs:
            self.history['train_loss'].append(logs['loss'])


class LambdaCallback(Callback):
    """Callback that executes arbitrary functions at various training stages"""
    
    def __init__(
        self,
        on_train_begin: callable = None,
        on_train_end: callable = None,
        on_epoch_begin: callable = None,
        on_epoch_end: callable = None,
        on_batch_begin: callable = None,
        on_batch_end: callable = None
    ):
        super().__init__()
        self.on_train_begin_func = on_train_begin
        self.on_train_end_func = on_train_end
        self.on_epoch_begin_func = on_epoch_begin
        self.on_epoch_end_func = on_epoch_end
        self.on_batch_begin_func = on_batch_begin
        self.on_batch_end_func = on_batch_end
    
    def on_train_begin(self, logs: dict[str, Any]) -> None:
        if self.on_train_begin_func:
            self.on_train_begin_func(logs)
    
    def on_train_end(self, logs: dict[str, Any]) -> None:
        if self.on_train_end_func:
            self.on_train_end_func(logs)
    
    def on_epoch_begin(self, epoch: int, logs: dict[str, Any]) -> None:
        if self.on_epoch_begin_func:
            self.on_epoch_begin_func(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        if self.on_epoch_end_func:
            self.on_epoch_end_func(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: dict[str, Any]) -> None:
        if self.on_batch_begin_func:
            self.on_batch_begin_func(batch, logs)
    
    def on_batch_end(self, batch: int, logs: dict[str, Any]) -> None:
        if self.on_batch_end_func:
            self.on_batch_end_func(batch, logs)


class TerminateOnNaN(Callback):
    """Callback that terminates training when a NaN loss is encountered"""
    
    def __init__(self):
        super().__init__()
    
    def on_batch_end(self, batch: int, logs: dict[str, Any]) -> None:
        """Check for NaN loss and terminate if found"""
        # loss = state.metrics.get("train/loss")

        loss = logs.get('loss')
        if loss is not None and torch.isnan(torch.tensor(loss)):
            print("NaN loss detected, terminating training")
            self.request_stop()


class CSVLogger(Callback):
    """Callback that logs metrics to a CSV file"""
    
    def __init__(self, filename: str, separator: str = ',', append: bool = False):
        super().__init__()
        self.filename = filename
        self.separator = separator
        self.append = append
        self.keys = None
        self.file = None
    
    def on_train_begin(self, logs: dict[str, Any]) -> None:
        """Initialize CSV file"""
        if self.append:
            self.file = open(self.filename, 'a')
        else:
            self.file = open(self.filename, 'w')
    
    def on_epoch_end(self, epoch: int, logs: dict[str, Any]) -> None:
        """Write metrics to CSV file"""
        if self.keys is None:
            self.keys = sorted(logs.keys())
            self.file.write(self.separator.join(['epoch'] + self.keys) + '\n')
        
        values = [str(epoch)] + [str(logs.get(key, '')) for key in self.keys]
        self.file.write(self.separator.join(values) + '\n')
        self.file.flush()
    
    def on_train_end(self, logs: dict[str, Any]) -> None:
        """Close CSV file"""
        if self.file:
            self.file.close()
