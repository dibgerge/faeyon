"""
Logging callbacks for Faeyon training.

This module provides logging callbacks for various platforms including
TensorBoard, MLFlow, Weights & Biases, and custom metrics logging.
"""

import os
from typing import Any, Optional
import torch

from .callbacks import Callback
from lightning import LightningModule


class MetricsLogger(Callback):
    """Base class for metrics logging callbacks"""
    
    def __init__(self, metrics: list[str] = None, log_freq: int = 1):
        super().__init__()
        self.metrics = metrics or ['accuracy']
        self.log_freq = log_freq
        self.step_count = 0
    
    def compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute specified metrics"""
        results = {}
            
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


class ConsoleLogger(Callback):
    """Simple console logging callback"""
    
    def __init__(self, log_freq: int = 10):
        super().__init__()
        self.log_freq = log_freq
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if batch % self.log_freq == 0:
            loss = logs.get('loss', 'N/A')
            print(f"Batch {batch}: Loss = {loss}")
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        print(f"Epoch {epoch} completed")
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")


class FileLogger(Callback):
    """File logging callback"""
    
    def __init__(self, filename: str, log_freq: int = 1):
        super().__init__()
        self.filename = filename
        self.log_freq = log_freq
        self.file = None
    
    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        self.file = open(self.filename, 'w')
        self.file.write("Epoch,Batch,Loss,Val_Loss\n")
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if self.file and batch % self.log_freq == 0:
            loss = logs.get('loss', 'N/A')
            val_loss = logs.get('val_loss', 'N/A')
            self.file.write(f"0,{batch},{loss},{val_loss}\n")
            self.file.flush()
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        if self.file:
            loss = logs.get('loss', 'N/A')
            val_loss = logs.get('val_loss', 'N/A')
            self.file.write(f"{epoch},0,{loss},{val_loss}\n")
            self.file.flush()
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        if self.file:
            self.file.close()


class JSONLogger(Callback):
    """JSON logging callback"""
    
    def __init__(self, filename: str, log_freq: int = 1):
        super().__init__()
        self.filename = filename
        self.log_freq = log_freq
        self.logs = []
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        if batch % self.log_freq == 0:
            log_entry = {
                'batch': batch,
                'timestamp': torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else 0,
                **logs
            }
            self.logs.append(log_entry)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        log_entry = {
            'epoch': epoch,
            'timestamp': torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)) if torch.cuda.is_available() else 0,
            **logs
        }
        self.logs.append(log_entry)
    
    def on_train_end(self, logs: Dict[str, Any]) -> None:
        import json
        with open(self.filename, 'w') as f:
            json.dump(self.logs, f, indent=2)


class MetricsTracker(Callback):
    """Callback to track and compute metrics during training"""
    
    def __init__(self, metrics: List[str] = None, compute_freq: int = 10):
        super().__init__()
        self.metrics = metrics or ['accuracy']
        self.compute_freq = compute_freq
        self.predictions = []
        self.targets = []
        self.metrics_history = []
    
    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        # Store predictions and targets for metric computation
        if 'predictions' in logs and 'targets' in logs:
            self.predictions.append(logs['predictions'])
            self.targets.append(logs['targets'])
        
        # Compute metrics periodically
        if batch % self.compute_freq == 0 and self.predictions and self.targets:
            self._compute_metrics(batch)
    
    def _compute_metrics(self, batch: int) -> None:
        """Compute metrics from stored predictions and targets"""
        if not self.predictions or not self.targets:
            return
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Compute metrics
        metrics = MetricsComputer.compute_metrics(all_predictions, all_targets, self.metrics)
        
        # Store in history
        self.metrics_history.append({
            'batch': batch,
            **metrics
        })
        
        # Clear stored data
        self.predictions = []
        self.targets = []
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the metrics history"""
        return self.metrics_history
