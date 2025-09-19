"""
Logging callbacks for Faeyon training.

This module provides logging callbacks for various platforms including
TensorBoard, MLFlow, Weights & Biases, and custom metrics logging.
"""

import os
from typing import Any, Dict, List, Optional
import torch

from .callbacks import Callback


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
