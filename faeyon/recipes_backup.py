import abc
import fnmatch
import importlib
import re
import torch
import time
import os
from collections import defaultdict

from typing import Optional, Iterable, Dict, Any, Union, Generator, Tuple, List
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .distributed import (
    setup_distributed, cleanup_distributed, get_node_info, is_master_node, 
    is_master_process, barrier, all_reduce_tensor, gather_tensors,
    setup_distributed_dataloader, get_available_devices, setup_multi_gpu,
    setup_distributed_model
)
from .callbacks import (
    Callback, EarlyStopping, ModelCheckpoint, LearningRateScheduler,
    ReduceLROnPlateau, History, LambdaCallback, TerminateOnNaN, CSVLogger
)
from .loggers import (
    MetricsComputer, MetricsLogger, TensorBoardLogger, MLFlowLogger, 
    WandBLogger, ConsoleLogger, FileLogger, JSONLogger, MetricsTracker
)


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
    mode: str, 
    value: Union[int, float], 
    val_freq: int = 1
) -> Generator[Tuple[Any, int, bool], None, None]:
    """
    Create a generator that yields (batch, step, should_validate) tuples.
    
    Args:
        train_data: Training data loader
        mode: Training mode ("epochs", "steps", "time")
        value: Training value (number of epochs/steps or time in seconds)
        val_freq: Validation frequency
        
    Yields:
        Tuple of (batch, step, should_validate)
    """
    train_iter = iter(train_data)
    step = 0
    epoch = 0
    start_time = time.time()
    
    while True:
        # Check stopping criteria
        if mode == "epochs" and epoch >= value:
            break
        elif mode == "steps" and step >= value:
            break
        elif mode == "time" and (time.time() - start_time) >= value:
            break
        
        # Get next batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_data)
            batch = next(train_iter)
            epoch += 1
        
        # Determine if we should validate
        should_validate = False
        if mode == "epochs":
            should_validate = (step + 1) % (len(train_data) * val_freq) == 0
        elif mode == "steps":
            should_validate = (step + 1) % val_freq == 0
        elif mode == "time":
            should_validate = (step + 1) % val_freq == 0
        
        yield batch, step, should_validate
        step += 1


def setup_distributed(backend: str = None, init_method: str = None) -> Tuple[int, int, int, int]:
    """Initialize distributed training for multi-node support
    
    Args:
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method ('env://', 'file://', 'tcp://')
    
    Returns:
        Tuple of (rank, world_size, local_rank, node_rank)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Auto-detect backend if not specified
        if backend is None:
            if torch.cuda.is_available():
                backend = 'nccl'
            else:
                backend = 'gloo'
        
        # Auto-detect init method if not specified
        if init_method is None:
            if 'MASTER_ADDR' in os.environ and 'MASTER_PORT' in os.environ:
                init_method = 'env://'
            else:
                init_method = 'env://'
        
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
        
        # Calculate node rank
        local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        node_rank = rank // local_world_size
        
        return rank, world_size, local_rank, node_rank
    return None, 1, 0, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def get_node_info() -> Dict[str, int]:
    """Get information about the current node in multi-node setup
    
    Returns:
        Dictionary with node information
    """
    if not torch.distributed.is_initialized():
        return {"node_rank": 0, "local_rank": 0, "local_world_size": 1, "world_size": 1}
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    node_rank = rank // local_world_size
    
    return {
        "node_rank": node_rank,
        "local_rank": local_rank,
        "local_world_size": local_world_size,
        "world_size": world_size,
        "rank": rank
    }


def is_master_node() -> bool:
    """Check if current process is on the master node (node_rank == 0)"""
    if not torch.distributed.is_initialized():
        return True
    
    node_info = get_node_info()
    return node_info["node_rank"] == 0


def is_master_process() -> bool:
    """Check if current process is the master process (rank == 0)"""
    if not torch.distributed.is_initialized():
        return True
    
    return torch.distributed.get_rank() == 0


def barrier() -> None:
    """Synchronize all processes across all nodes"""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def all_reduce_tensor(tensor: torch.Tensor, op: str = 'sum') -> torch.Tensor:
    """All-reduce a tensor across all processes in all nodes
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ('sum', 'mean', 'max', 'min')
    
    Returns:
        Reduced tensor
    """
    if not torch.distributed.is_initialized():
        return tensor
    
    # Convert string to torch.distributed.ReduceOp
    reduce_op_map = {
        'sum': torch.distributed.ReduceOp.SUM,
        'mean': torch.distributed.ReduceOp.SUM,  # Will divide by world_size after
        'max': torch.distributed.ReduceOp.MAX,
        'min': torch.distributed.ReduceOp.MIN
    }
    
    reduce_op = reduce_op_map.get(op, torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(tensor, op=reduce_op)
    
    if op == 'mean':
        tensor = tensor / torch.distributed.get_world_size()
    
    return tensor


def gather_tensors(tensor: torch.Tensor, dst: int = 0) -> List[torch.Tensor]:
    """Gather tensors from all processes to a destination process
    
    Args:
        tensor: Tensor to gather
        dst: Destination rank (default: 0)
    
    Returns:
        List of tensors (only non-empty on destination process)
    """
    if not torch.distributed.is_initialized():
        return [tensor]
    
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    
    # Create list to hold gathered tensors
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    
    # Gather all tensors
    torch.distributed.all_gather(gathered_tensors, tensor)
    
    if rank == dst:
        return gathered_tensors
    else:
        return []


def get_available_devices() -> List[torch.device]:
    """Get list of available devices"""
    devices = []
    
    # Add CPU
    devices.append(torch.device('cpu'))
    
    # Add available GPUs
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f'cuda:{i}'))
    
    return devices


def setup_multi_device_model(
    model: nn.Module, 
    device_ids: Optional[List[int]] = None,
    use_distributed: bool = False
) -> nn.Module:
    """
    Setup model for multi-device training.
    
    Args:
        model: The model to wrap
        device_ids: List of GPU device IDs to use (None for all available)
        use_distributed: Whether to use DistributedDataParallel
        
    Returns:
        Wrapped model for multi-device training
    """
    if use_distributed:
        if not torch.distributed.is_initialized():
            raise RuntimeError("Distributed training not initialized. Call setup_distributed() first.")
        return DistributedDataParallel(model)
    
    elif device_ids is not None and len(device_ids) > 1:
        return DataParallel(model, device_ids=device_ids)
    
    elif torch.cuda.device_count() > 1:
        return DataParallel(model)
    
    else:
        return model


def setup_distributed_dataloader(
    dataloader: DataLoader,
    shuffle: bool = True
) -> DataLoader:
    """
    Setup DataLoader for distributed training.
    
    Args:
        dataloader: Original DataLoader
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader with DistributedSampler
    """
    if not torch.distributed.is_initialized():
        return dataloader
    
    dataset = dataloader.dataset
    sampler: DistributedSampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=shuffle
    )
    
    return DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last
    )


class Callback:
    """Base callback class for training hooks"""
    def __init__(self):
        self._should_stop = False
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        pass
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        pass
    
    def on_batch_begin(self, batch: int, logs: Dict[str, Any]) -> None:
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        pass
    
    def should_stop(self) -> bool:
        """Check if this callback wants to stop training"""
        return self._should_stop
    
    def request_stop(self) -> None:
        """Request that training should stop"""
        self._should_stop = True
    
    def reset_stop(self) -> None:
        """Reset the stop flag"""
        self._should_stop = False


class MetricsComputer:
    """Computes various metrics for model evaluation"""
    
    @staticmethod
    def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute accuracy for classification tasks"""
        if predictions.dim() > 1 and predictions.size(1) > 1:
            # Multi-class classification
            predicted_classes = torch.argmax(predictions, dim=1)
        else:
            # Binary classification
            predicted_classes = (predictions > 0.5).long()
        
        correct = (predicted_classes == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor, 
                           num_classes: int = None) -> Dict[str, float]:
        """Compute precision, recall, and F1 score"""
        if predictions.dim() > 1 and predictions.size(1) > 1:
            predicted_classes = torch.argmax(predictions, dim=1)
        else:
            predicted_classes = (predictions > 0.5).long()
        
        if num_classes is None:
            num_classes = max(predicted_classes.max().item(), targets.max().item()) + 1
        
        # Convert to one-hot for multi-class metrics
        if num_classes > 2:
            pred_one_hot = torch.zeros(predicted_classes.size(0), num_classes)
            pred_one_hot.scatter_(1, predicted_classes.unsqueeze(1), 1)
            
            target_one_hot = torch.zeros(targets.size(0), num_classes)
            target_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        else:
            pred_one_hot = predicted_classes.float()
            target_one_hot = targets.float()
        
        # Compute metrics for each class
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for i in range(num_classes):
            if num_classes > 2:
                tp = (pred_one_hot[:, i] * target_one_hot[:, i]).sum().item()
                fp = (pred_one_hot[:, i] * (1 - target_one_hot[:, i])).sum().item()
                fn = ((1 - pred_one_hot[:, i]) * target_one_hot[:, i]).sum().item()
            else:
                tp = ((pred_one_hot == 1) & (target_one_hot == 1)).sum().item()
                fp = ((pred_one_hot == 1) & (target_one_hot == 0)).sum().item()
                fn = ((pred_one_hot == 0) & (target_one_hot == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        return {
            'precision': sum(precision_scores) / len(precision_scores),
            'recall': sum(recall_scores) / len(recall_scores),
            'f1': sum(f1_scores) / len(f1_scores)
        }
    
    @staticmethod
    def mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Absolute Error for regression tasks"""
        return torch.abs(predictions - targets).mean().item()
    
    @staticmethod
    def mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Mean Squared Error for regression tasks"""
        return ((predictions - targets) ** 2).mean().item()
    
    @staticmethod
    def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Root Mean Squared Error for regression tasks"""
        return torch.sqrt(((predictions - targets) ** 2).mean()).item()


class Recipe:
    """ The base class for training recipes """
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Union[optim.Optimizer, FaeOptimizer],
        callbacks: Optional[list[Callback]] = None,
        device: Optional[Union[torch.device, List[torch.device]]] = None,
        device_ids: Optional[List[int]] = None,
        use_distributed: bool = False
    ):
        self.model = model
        self.loss = loss
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
        
        # Move model to primary device
        self.model.to(self.device)
        
        # Setup multi-device model if needed
        if use_distributed or (device_ids and len(device_ids) > 1) or torch.cuda.device_count() > 1:
            self.model = setup_multi_device_model(
                self.model, 
                device_ids=device_ids,
                use_distributed=use_distributed
            )
        
        # Initialize optimizer
        if isinstance(optimizer, FaeOptimizer):
            self.optimizer = optimizer(self.model)
        else:
            self.optimizer = optimizer
        
    @abc.abstractmethod
    def train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform one training step on a batch.
        
        Args:
            batch: The training batch data
            batch_idx: The index of the current batch
            
        Returns:
            The loss tensor for this batch
        """
        pass
    
    @abc.abstractmethod
    def val_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform one validation step on a batch.
        
        Args:
            batch: The validation batch data
            batch_idx: The index of the current batch
            
        Returns:
            Dictionary of metrics for this batch
        """
        pass

    @abc.abstractmethod
    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform one test step on a batch.
        
        Args:
            batch: The test batch data
            batch_idx: The index of the current batch
            
        Returns:
            Dictionary of metrics for this batch
        """
        pass

    def train(
        self, 
        train_data: DataLoader, 
        min_period: str,
        max_period: str,
        val_data: Optional[DataLoader] = None,
        val_freq: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model with minimum and maximum period constraints.
        
        Args:
            train_data: Training data loader
            min_period: Minimum training period (e.g., "50e", "500s", "5h", "15m")
            max_period: Maximum training period (e.g., "200e", "1000s", "10h", "30m")
                       Can use different format than min_period (e.g., min="50s", max="2e")
            val_data: Optional validation data loader
            val_freq: Validation frequency (every N steps/epochs)
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history with min_period_reached, 
            max_period_reached, and early_stopped flags
        """
        # Parse min and max periods
        min_mode, min_value = parse_training_criteria(min_period)
        max_mode, max_value = parse_training_criteria(max_period)
        
        # Setup distributed data loaders if needed
        if self.use_distributed:
            train_data = setup_distributed_dataloader(train_data, shuffle=True)
            if val_data is not None:
                val_data = setup_distributed_dataloader(val_data, shuffle=False)
        
        # Initialize training state
        self.model.train()
        start_time = time.time()
        history: Dict[str, Any] = {
            'train_loss': [], 'val_loss': [], 'total_time': 0.0, 'total_steps': 0,
            'min_period_reached': False, 'max_period_reached': False, 'early_stopped': False
        }
        
        # Call training begin callbacks
        self._call_callbacks('on_train_begin', {
            'min_period': min_period, 'max_period': max_period, 
            'min_mode': min_mode, 'min_value': min_value,
            'max_mode': max_mode, 'max_value': max_value,
            'model': self.model, 'optimizer': self.optimizer,
            'rank': self.rank, 'world_size': self.world_size
        })
        
        # Create training generator with max period
        train_gen = create_training_generator(train_data, max_mode, max_value, val_freq)
        
        # Single training loop
        for batch, step, should_validate in train_gen:
            # Call batch begin callbacks
            self._call_callbacks('on_batch_begin', {'step': step})
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Training step
            self.optimizer.zero_grad()
            loss = self.train_step(batch, step)
            loss.backward()
            self.optimizer.step()
            
            # Track loss (only on master process for distributed training)
            if is_master_process():
                history['train_loss'].append(loss.item())
            
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
                if max_mode == "time":
                    print(f"Step {step} - Loss: {loss.item():.4f} - Time: {elapsed:.1f}s/{max_value:.1f}s")
                elif max_mode == "steps":
                    print(f"Step {step} - Loss: {loss.item():.4f} - Steps: {step}/{max_value}")
                elif max_mode == "epochs":
                    print(f"Step {step} - Loss: {loss.item():.4f} - Epochs: {step}/{max_value}")
                else:
                    print(f"Step {step} - Loss: {loss.item():.4f}")
        
        # Finalize history
        history['total_time'] = time.time() - start_time
        history['total_steps'] = step + 1
        
        # Call training end callbacks
        self._call_callbacks('on_train_end', {'history': history})
        
        return history

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
    
    def _check_early_stopping(self, history: Dict[str, Any]) -> bool:
        """Check if any callback wants to stop training"""
        for callback in self.callbacks:
            # Call the callback to update its internal state
            callback.on_epoch_end(0, history)  # We don't have epoch info here
            if callback.should_stop():
                return True
        return False

    def _validate_epoch(self, val_data: DataLoader, verbose: bool) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                batch_metrics = self.val_step(batch, batch_idx)
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    metrics[key].append(value.item() if isinstance(value, torch.Tensor) else value)
        
        # Average metrics
        avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        
        if verbose:
            metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
            print(f"  Validation - {metric_str}")
        
        return avg_metrics

    def validate(self, val_data: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """
        Run validation on the given data.
        
        Args:
            val_data: Validation data loader
            verbose: Whether to print validation results
            
        Returns:
            Dictionary of validation metrics
        """
        return self._validate_epoch(val_data, verbose)

    def test(self, test_data: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """
        Run testing on the given data.
        
        Args:
            test_data: Test data loader
            verbose: Whether to print test results
            
        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                batch_metrics = self.test_step(batch, batch_idx)
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    metrics[key].append(value.item() if isinstance(value, torch.Tensor) else value)
        
        # Average metrics
        avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        
        if verbose:
            metric_str = " - ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
            print(f"Test Results - {metric_str}")
        
        return avg_metrics

    @classmethod
    def create_multi_gpu_recipe(
        cls,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Union[optim.Optimizer, FaeOptimizer],
        device_ids: Optional[List[int]] = None,
        callbacks: Optional[list[Callback]] = None
    ) -> 'Recipe':
        """
        Create a recipe for multi-GPU training using DataParallel.
        
        Args:
            model: The model to train
            loss: Loss function
            optimizer: Optimizer
            device_ids: List of GPU device IDs to use (None for all available)
            callbacks: Optional callbacks
            
        Returns:
            Recipe configured for multi-GPU training
        """
        return cls(
            model=model,
            loss=loss,
            optimizer=optimizer,
            callbacks=callbacks,
            device_ids=device_ids
        )

    @classmethod
    def create_distributed_recipe(
        cls,
        model: nn.Module,
        loss: nn.Module,
        optimizer: Union[optim.Optimizer, FaeOptimizer],
        callbacks: Optional[list[Callback]] = None
    ) -> 'Recipe':
        """
        Create a recipe for distributed training using DistributedDataParallel.
        
        Args:
            model: The model to train
            loss: Loss function
            optimizer: Optimizer
            callbacks: Optional callbacks
            
        Returns:
            Recipe configured for distributed training
        """
        return cls(
            model=model,
            loss=loss,
            optimizer=optimizer,
            callbacks=callbacks,
            use_distributed=True
        )

    def cleanup(self):
        """Clean up distributed training resources"""
        if self.use_distributed:
            cleanup_distributed()

    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch data to the appropriate device"""
        if isinstance(batch, (list, tuple)):
            return [self._move_tensor_to_device(item) for item in batch]
        elif isinstance(batch, dict):
            return {key: self._move_tensor_to_device(value) for key, value in batch.items()}
        else:
            return self._move_tensor_to_device(batch)
    
    def _move_tensor_to_device(self, tensor: Any) -> Any:
        """Move a tensor to the appropriate device"""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return tensor

    def _call_callbacks(self, method_name: str, logs: Dict[str, Any]) -> None:
        """Call a specific method on all callbacks"""
        for callback in self.callbacks:
            if hasattr(callback, method_name):
                getattr(callback, method_name)(logs)
    

class ClassifyRecipe(Recipe):
    """Recipe for classification tasks"""
    
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: Union[optim.Optimizer, FaeOptimizer],
        loss: Optional[nn.Module] = None,
        callbacks: Optional[list[Callback]] = None,
        device: Optional[torch.device] = None
    ) -> None:
        if loss is None:
            loss = nn.CrossEntropyLoss()
        super().__init__(model=model, loss=loss, optimizer=optimizer, callbacks=callbacks, device=device)

    def train_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Perform one training step for classification.
        
        Args:
            batch: Tuple of (inputs, targets) or dict with 'inputs' and 'targets' keys
            batch_idx: The index of the current batch
            
        Returns:
            The loss tensor for this batch
        """
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        elif isinstance(batch, dict):
            inputs = batch['inputs']
            targets = batch['targets']
        else:
            raise ValueError("Batch must be a tuple/list of (inputs, targets) or dict with 'inputs' and 'targets' keys")
        
        # Forward pass
        predictions = self.model(inputs)
        loss = self.loss(predictions, targets)
        return loss

    def val_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
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
            raise ValueError("Batch must be a tuple/list of (inputs, targets) or dict with 'inputs' and 'targets' keys")
        
        # Forward pass
        predictions = self.model(inputs)
        loss = self.loss(predictions, targets)
        
        # Calculate accuracy
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == targets).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Perform one test step for classification.
        
        Args:
            batch: Tuple of (inputs, targets) or dict with 'inputs' and 'targets' keys
            batch_idx: The index of the current batch
            
        Returns:
            Dictionary of metrics for this batch
        """
        return self.val_step(batch, batch_idx)


class EarlyStopping(Callback):
    """Early stopping callback to stop training when validation loss stops improving"""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        self._check_early_stopping(logs, f"epoch {epoch+1}")
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        # For step-based training, check every validation
        if 'val_loss' in logs:
            self._check_early_stopping(logs, f"step {batch}")
    
    def _check_early_stopping(self, logs: Dict[str, Any], context: str) -> None:
        if self.monitor not in logs:
            return
            
        current_score = logs[self.monitor]
        
        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.request_stop()
                print(f"Early stopping triggered at {context}")
    
    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class ModelCheckpoint(Callback):
    """Callback to save model checkpoints"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss', save_best_only: bool = True, mode: str = 'min'):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_score = None
        self.model = None
        self.optimizer = None
        
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        # Get model and optimizer from the recipe
        if hasattr(logs, 'model'):
            self.model = logs['model']
        if hasattr(logs, 'optimizer'):
            self.optimizer = logs['optimizer']
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self.monitor not in logs or self.model is None or self.optimizer is None:
            return
            
        current_score = logs[self.monitor]
        
        if not self.save_best_only or self.best_score is None or self._is_improvement(current_score, self.best_score):
            self.best_score = current_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'logs': logs
            }, self.filepath)
            print(f"Model checkpoint saved to {self.filepath}")
    
    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            return current < best
        else:
            return current > best


class LearningRateScheduler(Callback):
    """Callback to adjust learning rate during training"""
    
    def __init__(self, scheduler: torch.optim.lr_scheduler._LRScheduler):
        self.scheduler = scheduler
        
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        self.scheduler.step()
        logs['lr'] = self.scheduler.get_last_lr()[0]


class MetricsLogger(Callback):
    """Base class for metrics logging callbacks"""
    
    def __init__(self, metrics: List[str] = None, log_freq: int = 1):
        super().__init__()
        self.metrics = metrics or ['accuracy']
        self.log_freq = log_freq
        self.step_count = 0
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute specified metrics"""
        results = {}
        
        for metric in self.metrics:
            if metric == 'accuracy':
                results[metric] = MetricsComputer.accuracy(predictions, targets)
            elif metric == 'precision':
                results.update(MetricsComputer.precision_recall_f1(predictions, targets))
            elif metric == 'recall':
                results.update(MetricsComputer.precision_recall_f1(predictions, targets))
            elif metric == 'f1':
                results.update(MetricsComputer.precision_recall_f1(predictions, targets))
            elif metric == 'mae':
                results[metric] = MetricsComputer.mae(predictions, targets)
            elif metric == 'mse':
                results[metric] = MetricsComputer.mse(predictions, targets)
            elif metric == 'rmse':
                results[metric] = MetricsComputer.rmse(predictions, targets)
        
        return results


class TensorBoardLogger(MetricsLogger):
    """TensorBoard logging callback"""
    
    def __init__(self, log_dir: str = "logs", metrics: List[str] = None, log_freq: int = 1):
        super().__init__(metrics, log_freq)
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.SummaryWriter = SummaryWriter
        except ImportError:
            self.SummaryWriter = None
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        if self.SummaryWriter is None:
            print("TensorBoard not available. Install with: pip install tensorboard")
            return
        
        self.writer = self.SummaryWriter(self.log_dir)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if self.writer is None or batch % self.log_freq != 0:
            return
        
        # Log loss
        if 'loss' in logs:
            self.writer.add_scalar('Loss/Train', logs['loss'], batch)
        
        # Log metrics if available
        for key, value in logs.items():
            if key.startswith('val_') and isinstance(value, (int, float)):
                self.writer.add_scalar(f'Metrics/{key}', value, batch)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.writer:
            self.writer.close()


class MLFlowLogger(MetricsLogger):
    """MLFlow logging callback"""
    
    def __init__(self, experiment_name: str = "faeyon_experiment", metrics: List[str] = None, log_freq: int = 1):
        super().__init__(metrics, log_freq)
        self.experiment_name = experiment_name
        self.run = None
        
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            self.mlflow = None
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        if self.mlflow is None:
            print("MLFlow not available. Install with: pip install mlflow")
            return
        
        # Set experiment
        self.mlflow.set_experiment(self.experiment_name)
        
        # Start run
        self.run = self.mlflow.start_run()
        
        # Log parameters
        if 'model' in logs:
            self.mlflow.log_param("model_type", type(logs['model']).__name__)
        if 'optimizer' in logs:
            self.mlflow.log_param("optimizer_type", type(logs['optimizer']).__name__)
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if self.mlflow is None or batch % self.log_freq != 0:
            return
        
        # Log metrics
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.mlflow.log_metric(key, value, step=batch)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.mlflow and self.run:
            self.mlflow.end_run()


class WandBLogger(MetricsLogger):
    """Weights & Biases logging callback"""
    
    def __init__(self, project: str = "faeyon-project", metrics: List[str] = None, log_freq: int = 1, **kwargs):
        super().__init__(metrics, log_freq)
        self.project = project
        self.wandb = None
        self.wandb_kwargs = kwargs
        
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            self.wandb = None
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        if self.wandb is None:
            print("Weights & Biases not available. Install with: pip install wandb")
            return
        
        # Initialize wandb
        self.wandb.init(project=self.project, **self.wandb_kwargs)
        
        # Log model architecture
        if 'model' in logs:
            self.wandb.watch(logs['model'])
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if self.wandb is None or batch % self.log_freq != 0:
            return
        
        # Log metrics
        self.wandb.log(logs, step=batch)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.wandb:
            self.wandb.finish()


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
