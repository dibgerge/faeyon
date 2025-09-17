#!/usr/bin/env python3
"""
Example demonstrating how to use the Recipe class for training PyTorch models
with simple string-based training criteria and multi-device support.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from faeyon.recipes import ClassifyRecipe, FaeOptimizer
from faeyon.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from faeyon.distributed import get_available_devices
from faeyon.models.tasks import ClassifyTask


def create_sample_data(num_samples=1000, input_dim=784, num_classes=10):
    """Create sample data for demonstration"""
    # Generate random data
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return (X_train, y_train), (X_val, y_val)


def train_with_epochs(recipe, train_loader, val_loader):
    """Example: Train for a specific number of epochs"""
    print("=== Training for 5 epochs ===")
    
    history = recipe.train(
        train_data=train_loader,
        min_period="2e",  # Minimum 2 epochs
        max_period="5e",  # Maximum 5 epochs
        val_data=val_loader,
        val_freq=1,
        gradient_accumulation_steps=1,  # No accumulation
        verbose=True
    )
    
    print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")
    return history


def train_with_steps(recipe, train_loader, val_loader):
    """Example: Train for a specific number of steps"""
    print("\n=== Training for 100 steps ===")
    
    history = recipe.train(
        train_data=train_loader,
        min_period="50s",  # Minimum 50 steps
        max_period="100s", # Maximum 100 steps
        val_data=val_loader,
        val_freq=10,  # Validate every 10 steps
        verbose=True
    )
    
    print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    return history


def train_with_time(recipe, train_loader, val_loader):
    """Example: Train for a specific amount of time"""
    print("\n=== Training for 30 seconds ===")
    
    history = recipe.train(
        train_data=train_loader,
        min_period="15s",  # Minimum 15 seconds
        max_period="30s",  # Maximum 30 seconds
        val_data=val_loader,
        val_freq=5,  # Validate every 5 steps
        verbose=True
    )
    
    print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    return history


def train_with_hours(recipe, train_loader, val_loader):
    """Example: Train for a specific number of hours"""
    print("\n=== Training for 0.1 hours (6 minutes) ===")
    
    history = recipe.train(
        train_data=train_loader,
        min_period="0.05h",  # Minimum 0.05 hours = 3 minutes
        max_period="0.1h",   # Maximum 0.1 hours = 6 minutes
        val_data=val_loader,
        val_freq=10,  # Validate every 10 steps
        verbose=True
    )
    
    print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    return history


def train_single_gpu():
    """Example: Single GPU training"""
    print("=== Single GPU Training ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create sample data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    
    # Create optimizer
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create recipe
    recipe = ClassifyRecipe(
        model=model,
        optimizer=optimizer,
        device=device
    )
    
    # Train
    history = recipe.train(train_loader, "25s", "50s", val_loader, val_freq=10, verbose=True)
    print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")


def train_multi_gpu():
    """Example: Multi-GPU training using DataParallel"""
    print("\n=== Multi-GPU Training (DataParallel) ===")
    
    # Check available devices
    devices = get_available_devices()
    gpu_devices = [d for d in devices if d.type == 'cuda']
    
    if len(gpu_devices) < 2:
        print("Not enough GPUs available for multi-GPU training. Skipping...")
        return
    
    print(f"Available GPUs: {len(gpu_devices)}")
    
    # Create sample data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    
    # Create optimizer
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create multi-GPU recipe
    recipe = ClassifyRecipe.create_multi_gpu_recipe(
        model=model,
        optimizer=optimizer,
        device_ids=list(range(len(gpu_devices)))
    )
    
    # Train
    history = recipe.train(train_loader, "25s", "50s", val_loader, val_freq=10, verbose=True)
    print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")


def train_distributed():
    """Example: Distributed training using DistributedDataParallel"""
    print("\n=== Distributed Training (DistributedDataParallel) ===")
    print("Note: This example shows how to set up distributed training.")
    print("To actually run distributed training, use torchrun or similar.")
    
    # Create sample data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    
    # Create optimizer
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create distributed recipe
    recipe = ClassifyRecipe.create_distributed_recipe(
        model=model,
        optimizer=optimizer
    )
    
    try:
        # Train
        history = recipe.train(train_loader, "50s", val_loader, val_freq=10, verbose=True)
        print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    except RuntimeError as e:
        print(f"Distributed training not available: {e}")
    finally:
        # Clean up
        recipe.cleanup()


def train_with_mixed_periods(recipe, train_loader, val_loader):
    """Example: Train with different modes for min and max periods"""
    print("\n=== Training with mixed periods (min: 30s, max: 5e) ===")
    
    history = recipe.train(
        train_data=train_loader,
        min_period="30s",  # Minimum 30 seconds
        max_period="5e",   # Maximum 5 epochs
        val_data=val_loader,
        val_freq=10,
        gradient_accumulation_steps=1,  # No accumulation
        verbose=True
    )
    
    print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")
    return history


def train_with_gradient_accumulation(recipe, train_loader, val_loader):
    """Example: Train with gradient accumulation"""
    print("\n=== Training with gradient accumulation ===")
    
    # Use smaller batch size to simulate memory constraints
    small_train_loader = DataLoader(train_loader.dataset, batch_size=4, shuffle=True)
    small_val_loader = DataLoader(val_loader.dataset, batch_size=4, shuffle=False)
    
    print(f"Batch size: 4")
    print(f"Gradient accumulation steps: 4")
    print(f"Effective batch size: 16")
    
    history = recipe.train(
        train_data=small_train_loader,
        min_period="30s",  # Minimum 30 seconds
        max_period="1m",   # Maximum 1 minute
        val_data=small_val_loader,
        val_freq=5,
        gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
        verbose=True
    )
    
    print(f"Completed {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Min period reached: {history['min_period_reached']}")
    print(f"Max period reached: {history['max_period_reached']}")
    print(f"Early stopped: {history['early_stopped']}")
    return history


def main():
    """Main training example demonstrating multi-device training"""
    print("Demonstrating multi-device training capabilities...")
    
    # Single GPU training
    train_single_gpu()
    
    # Multi-GPU training
    train_multi_gpu()
    
    # Distributed training
    train_distributed()
    
    # Mixed periods example
    print("\n=== Mixed Periods Example ===")
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model and optimizer
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    )
    
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    recipe = ClassifyRecipe(model=model, optimizer=optimizer)
    
    # Train with mixed periods
    train_with_mixed_periods(recipe, train_loader, val_loader)
    
    # Gradient accumulation example
    train_with_gradient_accumulation(recipe, train_loader, val_loader)
    
    print("\n=== Training Examples Complete ===")
    print("To run distributed training on multiple machines, use:")
    print("torchrun --nproc_per_node=2 example_training.py")


if __name__ == "__main__":
    main()
