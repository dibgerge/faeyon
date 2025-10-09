#!/usr/bin/env python3
"""
Example demonstrating gradient accumulation in Faeyon training.

Gradient accumulation allows training with larger effective batch sizes
when memory is limited by accumulating gradients over multiple mini-batches
before updating the model parameters.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from faeyon.recipes import ClassifyRecipe, FaeOptimizer
from faeyon.callbacks import EarlyStopping, ModelCheckpoint


def create_sample_data(num_samples=1000, input_dim=784, num_classes=10):
    """Create sample data for demonstration"""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y


def create_model(input_dim=784, num_classes=10):
    """Create a simple neural network model"""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )


def train_without_accumulation():
    """Example: Training without gradient accumulation (normal training)"""
    print("=== Training WITHOUT Gradient Accumulation ===")
    
    # Create data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Small batch size to simulate memory constraints
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model and optimizer
    model = create_model()
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5)
    ]
    
    # Create recipe
    recipe = ClassifyRecipe(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        callbacks=callbacks
    )
    
    print(f"Batch size: 8")
    print(f"Effective batch size: 8")
    print(f"Number of batches per epoch: {len(train_loader)}")
    
    # Train
    history = recipe.train(
        train_loader, 
        min_period="30s", 
        max_period="1m", 
        val_data=val_loader, 
        val_freq=5,
        gradient_accumulation_steps=1,  # No accumulation
        verbose=True
    )
    
    print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if 'val_loss' in history:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print()


def train_with_accumulation():
    """Example: Training with gradient accumulation"""
    print("=== Training WITH Gradient Accumulation ===")
    
    # Create data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Small batch size to simulate memory constraints
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model and optimizer
    model = create_model()
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5)
    ]
    
    # Create recipe
    recipe = ClassifyRecipe(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        callbacks=callbacks
    )
    
    # Gradient accumulation parameters
    accumulation_steps = 4
    effective_batch_size = 8 * accumulation_steps
    
    print(f"Batch size: 8")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Number of batches per epoch: {len(train_loader)}")
    print(f"Number of optimizer updates per epoch: {len(train_loader) // accumulation_steps}")
    
    # Train
    history = recipe.train(
        train_loader, 
        min_period="30s", 
        max_period="1m", 
        val_data=val_loader, 
        val_freq=5,
        gradient_accumulation_steps=accumulation_steps,
        verbose=True
    )
    
    print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if 'val_loss' in history:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print()


def train_large_model_simulation():
    """Example: Simulating training a large model with memory constraints"""
    print("=== Large Model Simulation with Gradient Accumulation ===")
    
    # Create data
    (X_train, y_train), (X_val, y_val) = create_sample_data()
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Very small batch size to simulate memory constraints
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Create a larger model to simulate memory usage
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )
    
    optimizer = FaeOptimizer(name="Adam", patterns=["*"], lr=0.001)
    
    # Create callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(
            filepath="checkpoints/large_model_checkpoint.pth",
            monitor='val_loss',
            save_best_only=True,
            verbose=True
        )
    ]
    
    # Create recipe
    recipe = ClassifyRecipe(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        callbacks=callbacks
    )
    
    # Large gradient accumulation to simulate large effective batch size
    accumulation_steps = 16
    effective_batch_size = 2 * accumulation_steps
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: 2 (limited by memory)")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Memory usage per forward pass: ~{2 * 784 * 4 / 1024 / 1024:.2f} MB")
    print(f"Total memory for accumulation: ~{effective_batch_size * 784 * 4 / 1024 / 1024:.2f} MB")
    
    # Train
    history = recipe.train(
        train_loader, 
        min_period="30s", 
        max_period="2m", 
        val_data=val_loader, 
        val_freq=10,
        gradient_accumulation_steps=accumulation_steps,
        verbose=True
    )
    
    print(f"Training completed: {history['total_steps']} steps in {history['total_time']:.2f}s")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if 'val_loss' in history:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print()


def compare_training_strategies():
    """Compare different training strategies"""
    print("=== Comparing Training Strategies ===")
    
    strategies = [
        {"name": "Small Batch", "batch_size": 8, "accumulation": 1, "effective": 8},
        {"name": "Medium Accumulation", "batch_size": 4, "accumulation": 4, "effective": 16},
        {"name": "Large Accumulation", "batch_size": 2, "accumulation": 8, "effective": 16},
        {"name": "Very Large Accumulation", "batch_size": 1, "accumulation": 16, "effective": 16},
    ]
    
    for strategy in strategies:
        print(f"\n--- {strategy['name']} ---")
        print(f"Batch size: {strategy['batch_size']}")
        print(f"Accumulation steps: {strategy['accumulation']}")
        print(f"Effective batch size: {strategy['effective']}")
        print(f"Memory per batch: ~{strategy['batch_size'] * 784 * 4 / 1024:.1f} KB")
        print(f"Total memory: ~{strategy['effective'] * 784 * 4 / 1024:.1f} KB")
        
        # This would be the actual training code
        print("Training would use this configuration...")


def main():
    """Main function demonstrating gradient accumulation"""
    print("=== Faeyon Gradient Accumulation Example ===")
    print("This example demonstrates how to use gradient accumulation")
    print("to train models with larger effective batch sizes when memory is limited.")
    print()
    
    try:
        # Run examples
        train_without_accumulation()
        train_with_accumulation()
        train_large_model_simulation()
        compare_training_strategies()
        
        print("=== Gradient Accumulation Benefits ===")
        print("1. Memory Efficiency: Use smaller batch sizes to fit in memory")
        print("2. Larger Effective Batch Size: Accumulate gradients for better training")
        print("3. Stable Training: More stable gradients with larger effective batches")
        print("4. Flexible Configuration: Adjust accumulation based on available memory")
        print()
        print("=== When to Use Gradient Accumulation ===")
        print("- Large models that don't fit in memory with desired batch size")
        print("- Limited GPU memory")
        print("- Need larger effective batch sizes for stable training")
        print("- Training on multiple GPUs with small per-GPU batch sizes")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
