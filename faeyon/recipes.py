import fnmatch
import importlib
import re
import torch
import time
from collections import defaultdict

from typing import Optional, Iterable, Dict, Any, Union, Generator, Tuple, List
from torch import nn, optim
from torch.utils.data import DataLoader

from .distributed import (
    setup_distributed, cleanup_distributed, is_master_process,
    setup_multi_gpu, setup_distributed_model
)
from .callbacks import Callback
from .loggers import MetricsComputer, MetricsLogger


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


def parse_training_criteria(criteria: str) -> Tuple[str, Union[int, float]]:
    """
    Parse training criteria string into mode and value.
    
    Examples:
        "200e" -> ("epochs", 200)
        "1000s" -> ("steps", 1000) 
        "10h" -> ("time", 36000)  # 10 hours in seconds
        "30m" -> ("time", 1800)   # 30 minutes in seconds
        "2.5h" -> ("time", 9000)  # 2.5 hours in seconds
    """
    criteria = criteria.strip().lower()
    
    # Time parsing
    if criteria.endswith('h'):
        hours = float(criteria[:-1])
        return ("time", hours * 3600)
    elif criteria.endswith('m'):
        minutes = float(criteria[:-1])
        return ("time", minutes * 60)
    elif criteria.endswith('s') and not criteria.endswith('steps'):
        seconds = float(criteria[:-1])
        return ("time", seconds)
    
    # Steps parsing
    elif criteria.endswith('steps') or criteria.endswith('step'):
        steps = int(criteria.replace('steps', '').replace('step', ''))
        return ("steps", steps)
    elif criteria.endswith('s') and criteria.endswith('steps'):
        steps = int(criteria[:-6])  # Remove 'steps'
        return ("steps", steps)
    
    # Epochs parsing
    elif criteria.endswith('e') or criteria.endswith('epochs') or criteria.endswith('epoch'):
        epochs = int(criteria.replace('epochs', '').replace('epoch', '').replace('e', ''))
        return ("epochs", epochs)
    
    # Default to epochs if no suffix
    else:
        try:
            epochs = int(criteria)
            return ("epochs", epochs)
        except ValueError:
            raise ValueError(f"Invalid training criteria: {criteria}. Use format like '200e', '1000s', '10h', etc.")


def create_training_generator(
    train_data: DataLoader,
    val_data: Optional[DataLoader] = None,
    val_freq: int = 1
) -> Generator[Tuple[Any, int, bool], None, None]:
    """Create a generator for training loop
    
    Args:
        train_data: Training data loader
        val_data: Validation data loader
        val_freq: Validation frequency (every N steps)
    
    Yields:
        Tuple of (batch, step, should_validate)
    """
    step = 0
    for batch in train_data:
        should_validate = (val_data is not None and step % val_freq == 0)
        
        yield batch, step, should_validate
        step += 1


class DataHandler:
    def __init__(
        self, 
        train_data: DataLoader, 
        start_epoch: int = 0, 
        start_step: int = 0,
        min_period: Period,
        max_period: Period,
        val_freq: int = 1
    ):
        self.train_data = train_data

    def __iter__(self):
        return self
    
    def __next__(self) -> Any:
        batch = next(self.train_data)
        self.step += 1

        # Check if epoch should be incremented, or if the data loader should be reset
        return batch


class Recipe:
    """ The base class for training recipes """
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Union[optim.Optimizer, FaeOptimizer],
        callbacks: Optional[List[Callback]] = None,
        device: Optional[Union[torch.device, List[torch.device]]] = None,
        device_ids: Optional[List[int]] = None,
        use_distributed: bool = False
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.device_ids = device_ids
        self.use_distributed = use_distributed
        
        # Setup distributed training if requested
        if use_distributed:
            self.rank, self.world_size, self.local_rank, self.node_rank = setup_distributed()
            self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        else:
            self.rank, self.world_size, self.local_rank, self.node_rank = 0, 1, 0, 0
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            elif isinstance(device, list):
                self.device = device[0]  # Primary device
                self.device_ids = [d.index for d in device if d.type == 'cuda']
            else:
                self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup multi-device training
        if use_distributed:
            self.model = setup_distributed_model(self.model, self.device)
        elif device_ids and len(device_ids) > 1:
            self.model = setup_multi_gpu(self.model, device_ids)
        
        # Initialize optimizer
        if isinstance(optimizer, FaeOptimizer):
            self.optimizer = optimizer(self.model)
        else:
            self.optimizer = optimizer
    
    def train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform one training step.
        
        Args:
            batch: Training batch
            batch_idx: The index of the current batch
            
        Returns:
            Loss tensor
        """
        raise NotImplementedError("Subclasses must implement train_step")
    
    def val_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
        """
        Perform one validation step.
        
        Args:
            batch: Validation batch
            batch_idx: The index of the current batch
            
        Returns:
            Dictionary of metrics for this batch
        """
        raise NotImplementedError("Subclasses must implement val_step")
    
    def _call_callbacks(self, method_name: str, logs: Dict[str, Any]) -> None:
        """Call a method on all callbacks"""
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(logs)
    
    def _check_early_stopping(self, history: Dict[str, Any]) -> bool:
        """Check if early stopping should be triggered"""
        for callback in self.callbacks:
            callback.on_epoch_end(0, history)  # Pass 0 as epoch since we're in step-based training
            if callback.should_stop():
                return True
        return False
    
    def _check_period_reached(self, mode: str, value: float, step: int, history: Dict[str, Any], start_time: float) -> bool:
        """Check if a specific period has been reached"""
        if mode == "epochs":
            # For epochs, we need to calculate based on steps per epoch
            # This is a simplified check - in practice you might want to track epochs more precisely
            return step >= value
        elif mode == "steps":
            return step >= value
        elif mode == "time":
            elapsed = time.time() - start_time
            return elapsed >= value
        else:
            return False
    
    def _validate_epoch(self, val_data: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """Validate the model on validation data"""
        self.model.eval()
        val_losses = []
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
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
    
    def train(
        self,
        train_data: DataLoader,
        min_period: str,
        max_period: str,
        val_data: Optional[DataLoader] = None,
        val_freq: int = 1,
        gradient_accumulation_steps: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
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
        # Parse training periods
        min_mode, min_value = parse_training_criteria(min_period)
        max_mode, max_value = parse_training_criteria(max_period)
        
        # Initialize training
        self.model.train()
        start_time = time.time()
        
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'total_time': 0.0,
            'total_steps': 0,
            'min_period_reached': False,
            'max_period_reached': False,
            'early_stopped': False
        }
        
        # Call train begin callbacks
        train_logs = {
            'model': self.model,
            'optimizer': self.optimizer,
            'loss': self.loss
        }
        self._call_callbacks('on_train_begin', train_logs)
        
        # Training loop
        step = 0
        accumulation_step = 0
        accumulated_loss = 0.0
        
        for batch, step, should_validate in create_training_generator(train_data, val_data, val_freq):
            # Call batch begin callbacks
            self._call_callbacks('on_batch_begin', {'step': step, 'batch': batch})
            
            # Training step with gradient accumulation
            if accumulation_step == 0:
                self.optimizer.zero_grad()
            
            loss = self.train_step(batch, step)
            
            # Scale loss by accumulation steps to maintain correct gradient magnitudes
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            accumulated_loss += loss.item()
            accumulation_step += 1
            
            # Update weights only after accumulating gradients
            if accumulation_step == gradient_accumulation_steps:
                self.optimizer.step()
                accumulation_step = 0
                
                # Track loss (only on master process for distributed training)
                if is_master_process():
                    history['train_loss'].append(accumulated_loss / gradient_accumulation_steps)
                
                accumulated_loss = 0.0
            
            # Check if minimum period has been reached
            if not history['min_period_reached']:
                if self._check_period_reached(min_mode, min_value, step, history, start_time):
                    history['min_period_reached'] = True
                if verbose and is_master_process():
                    print(f"Minimum period reached: {min_period}")
            
            # Call batch end callbacks
            self._call_callbacks('on_batch_end', {'step': step, 'loss': loss.item()})
            
            # Validation (only on master process for distributed training)
            if should_validate and val_data and is_master_process():
                val_metrics = self._validate_epoch(val_data, verbose)
                for key, value in val_metrics.items():
                    if key not in history:
                        history[key] = []
                    history[key].append(value)
            
            # Check for early stopping (only after minimum period is reached)
            if history['min_period_reached'] and self._check_early_stopping(history):
                history['early_stopped'] = True
                if verbose and is_master_process():
                    print("Early stopping triggered")
                break
            
            # Check if maximum period has been reached
            if self._check_period_reached(max_mode, max_value, step, history, start_time):
                history['max_period_reached'] = True
                if verbose and is_master_process():
                    print(f"Maximum period reached: {max_period}")
                break
            
            # Progress logging (only on master process for distributed training)
            if verbose and step % 10 == 0 and is_master_process():
                elapsed = time.time() - start_time
                current_loss = accumulated_loss / max(accumulation_step, 1) if accumulation_step > 0 else 0.0
                if max_mode == "time":
                    print(f"Step {step} - Loss: {current_loss:.4f} - Time: {elapsed:.1f}s/{max_value:.1f}s")
                elif max_mode == "steps":
                    print(f"Step {step} - Loss: {current_loss:.4f} - Steps: {step}/{max_value}")
                else:  # epochs
                    print(f"Step {step} - Loss: {current_loss:.4f} - Epochs: {step // len(train_data)}")
        
        # Handle remaining accumulated gradients if training ends mid-accumulation
        if accumulation_step > 0:
            self.optimizer.step()
            if is_master_process():
                history['train_loss'].append(accumulated_loss / accumulation_step)
        
        # Finalize training
        history['total_time'] = time.time() - start_time
        history['total_steps'] = step
        
        # Call train end callbacks
        self._call_callbacks('on_train_end', history)
        
        return history
    
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
    
    def val_step(self, batch: Any, batch_idx: int) -> Dict[str, float]:
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
